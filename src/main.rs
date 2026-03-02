mod cache;
mod diff;
mod discover;
mod fatbin;
mod html;
mod image;
mod nvidia;
mod progress;
mod registry;
mod scan;
mod site;
mod store;
mod style;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "wiimi",
    about = "WTF Is In My Image - GPU image scanning toolkit"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "warn", global = true)]
    log_level: String,
}

#[derive(Subcommand)]
enum Command {
    /// Scan an OCI image for CUDA compute capabilities
    Scan {
        /// Image reference (e.g. ghcr.io/llm-d/llm-d-cuda:v0.5.0)
        image: String,

        /// Output as JSON instead of human-readable report
        #[arg(long, conflicts_with = "html")]
        json: bool,

        /// Output an interactive HTML dependency graph
        #[arg(long, conflicts_with = "json")]
        html: bool,

        /// Path to Docker config JSON for registry authentication
        #[arg(long, env = "DOCKER_CONFIG")]
        docker_config: Option<String>,

        /// Registry hostnames to access over HTTP instead of HTTPS
        #[arg(long)]
        insecure_registry: Vec<String>,

        /// Skip all caches (blob + parse) and pull everything fresh
        #[arg(long)]
        no_cache: bool,

        /// Re-parse all binaries (skip parse cache) but keep blob cache
        #[arg(long)]
        reparse: bool,

        /// Load JS libraries from CDN instead of inlining them (HTML only)
        #[arg(long)]
        cdn: bool,

        /// Open the output file after writing (HTML only)
        #[arg(long)]
        open: bool,
    },

    /// Discover new tags from registries and scan them
    Discover {
        /// Path to the site config TOML file
        #[arg(long, default_value = "wiimi-site.toml")]
        config: String,

        /// Path to the scan store database
        #[arg(long)]
        db: Option<String>,

        /// Re-scan all images even if already in the store
        #[arg(long)]
        force: bool,

        /// Path to Docker config JSON for registry authentication
        #[arg(long, env = "DOCKER_CONFIG")]
        docker_config: Option<String>,

        /// Registry hostnames to access over HTTP instead of HTTPS
        #[arg(long)]
        insecure_registry: Vec<String>,

        /// Disable blob and parse caches (stream-only, saves disk on CI runners)
        #[arg(long)]
        no_cache: bool,
    },

    /// Generate a static site from stored scan results
    Site {
        /// Path to the site config TOML file
        #[arg(long, default_value = "wiimi-site.toml")]
        config: String,

        /// Path to the scan store database
        #[arg(long)]
        db: Option<String>,

        /// Open the generated index.html in a browser
        #[arg(long)]
        open: bool,
    },

    /// Compare two scans and show what changed
    Diff {
        /// First scan (JSON file path or image reference)
        from: String,

        /// Second scan (JSON file path or image reference)
        to: String,

        /// Output as JSON instead of HTML
        #[arg(long, conflicts_with = "html")]
        json: bool,

        /// Output as HTML (default)
        #[arg(long, conflicts_with = "json")]
        html: bool,

        /// Path to Docker config JSON for registry authentication
        #[arg(long, env = "DOCKER_CONFIG")]
        docker_config: Option<String>,

        /// Registry hostnames to access over HTTP instead of HTTPS
        #[arg(long)]
        insecure_registry: Vec<String>,

        /// Skip all caches (blob + parse) and pull everything fresh
        #[arg(long)]
        no_cache: bool,

        /// Re-parse all binaries (skip parse cache) but keep blob cache
        #[arg(long)]
        reparse: bool,

        /// Open the output file after writing (HTML only)
        #[arg(long)]
        open: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .with_target(false)
        .init();

    match cli.command {
        Command::Scan {
            image,
            json,
            html: html_flag,
            docker_config,
            insecure_registry,
            no_cache,
            reparse,
            cdn,
            open,
        } => {
            let show_progress = !json;
            let dockerconfig_bytes = load_docker_config(docker_config.as_deref())?;
            let use_blob_cache = !no_cache;
            let use_parse_cache = !no_cache && !reparse;

            let result = scan::scan_image(
                &image,
                dockerconfig_bytes.as_deref(),
                insecure_registry,
                show_progress,
                use_blob_cache,
                use_parse_cache,
            )
            .await?;

            // Automatically persist to scan store (unless caching is disabled)
            if !no_cache {
                if let Err(e) = auto_upsert_scan(&result) {
                    tracing::warn!("failed to persist scan to store: {e:#}");
                }
            }

            if json {
                let output = serde_json::to_string_pretty(&result)
                    .context("failed to serialize scan result")?;
                println!("{output}");
            } else if html_flag {
                let bundle = if cdn {
                    style::BundleMode::Cdn
                } else {
                    style::BundleMode::SelfContained
                };
                let output = html::render_html(&result, bundle);
                let filename = format!("wiimi-{}.html", sanitize_filename(&image));
                std::fs::write(&filename, &output)
                    .with_context(|| format!("failed to write HTML to {filename}"))?;
                println!("HTML report written to: {filename}");
                if open {
                    open_file(&filename)?;
                }
            } else {
                print!("{}", scan::format_report(&result));
            }
        }

        Command::Discover {
            config,
            db,
            force,
            docker_config,
            insecure_registry,
            no_cache,
        } => {
            let site_config = site::SiteConfig::load(std::path::Path::new(&config))?;
            let store = open_store(db.as_deref())?;
            let dockerconfig_bytes = load_docker_config(docker_config.as_deref())?;
            let client = std::sync::Arc::new(discover::build_client(insecure_registry));

            for family in &site_config.families {
                println!("Discovering: {}", family.name);
                let summary = discover::discover_family(
                    family,
                    &store,
                    client.clone(),
                    dockerconfig_bytes.as_deref(),
                    force,
                    no_cache,
                )
                .await?;
                println!(
                    "  {}: {} matched, {} new, {} cached, {} ignored",
                    summary.family_name,
                    summary.matched,
                    summary.new_scanned,
                    summary.cached,
                    summary.ignored,
                );
            }
        }

        Command::Site { config, db, open } => {
            let site_config = site::SiteConfig::load(std::path::Path::new(&config))?;
            let store = open_store(db.as_deref())?;
            site::generate_site(&site_config, &store)?;
            let index_path = format!("{}/index.html", site_config.output_dir);
            println!("Site generated at: {}", site_config.output_dir);
            if open {
                open_file(&index_path)?;
            }
        }

        Command::Diff {
            from,
            to,
            json,
            html: _,
            docker_config,
            insecure_registry,
            no_cache,
            reparse,
            open,
        } => {
            let dockerconfig_bytes = load_docker_config(docker_config.as_deref())?;
            let use_blob_cache = !no_cache;
            let use_parse_cache = !no_cache && !reparse;
            let show_progress = !json;

            // Run sequentially so progress bars don't interleave
            let from_result = resolve_scan_result(
                &from,
                dockerconfig_bytes.as_deref(),
                &insecure_registry,
                use_blob_cache,
                use_parse_cache,
                show_progress,
            )
            .await?;
            let to_result = resolve_scan_result(
                &to,
                dockerconfig_bytes.as_deref(),
                &insecure_registry,
                use_blob_cache,
                use_parse_cache,
                show_progress,
            )
            .await?;

            let diff_result = diff::compute_diff(&from_result, &to_result);

            if json {
                let output = serde_json::to_string_pretty(&diff_result)
                    .context("failed to serialize diff result")?;
                println!("{output}");
            } else {
                let output = diff::render_diff_html(&diff_result);
                let safe_from = sanitize_filename(&from);
                let safe_to = sanitize_filename(&to);
                let filename = format!("wiimi-diff-{safe_from}-vs-{safe_to}.html");
                std::fs::write(&filename, &output)
                    .with_context(|| format!("failed to write HTML to {filename}"))?;
                println!("Diff report written to: {filename}");
                if open {
                    open_file(&filename)?;
                }
            }
        }
    }

    Ok(())
}

/// Load Docker config bytes from an explicit path, or fall back to ~/.docker/config.json.
fn load_docker_config(explicit_path: Option<&str>) -> Result<Option<Vec<u8>>> {
    match explicit_path {
        Some(path) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("failed to read docker config at {path}"))?;
            Ok(Some(bytes))
        }
        None => {
            let default_path = default_docker_config_path();
            Ok(std::fs::read(&default_path).ok())
        }
    }
}

/// Resolve a scan input: if it's a path to an existing JSON file, deserialize it.
/// Otherwise treat it as an image reference and scan it live.
async fn resolve_scan_result(
    input: &str,
    dockerconfig: Option<&[u8]>,
    insecure_registries: &[String],
    use_blob_cache: bool,
    use_parse_cache: bool,
    show_progress: bool,
) -> Result<scan::ScanResult> {
    let path = std::path::Path::new(input);
    if path.exists() {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read scan file: {input}"))?;
        let result: scan::ScanResult = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse scan JSON from: {input}"))?;
        Ok(result)
    } else {
        scan::scan_image(
            input,
            dockerconfig,
            insecure_registries.to_vec(),
            show_progress,
            use_blob_cache,
            use_parse_cache,
        )
        .await
    }
}

/// Sanitize a string for use in filenames (replace non-alphanumeric chars).
pub(crate) fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Open a file with the platform's default handler.
fn open_file(path: &str) -> Result<()> {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "start"
    } else {
        "xdg-open"
    };
    std::process::Command::new(cmd)
        .arg(path)
        .spawn()
        .with_context(|| format!("failed to open {path} with {cmd}"))?;
    Ok(())
}

/// Default Docker config path (~/.docker/config.json).
fn default_docker_config_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/root"))
        .join(".docker/config.json")
}

/// Silently persist a scan result to the default store.
fn auto_upsert_scan(result: &scan::ScanResult) -> Result<()> {
    let s = store::ScanStore::default_location()?;
    s.upsert(result)
}

/// Open a scan store from an explicit path or the default location.
fn open_store(db: Option<&str>) -> Result<store::ScanStore> {
    match db {
        Some(path) => store::ScanStore::open(std::path::Path::new(path)),
        None => store::ScanStore::default_location(),
    }
}
