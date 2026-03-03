use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

use crate::diff;
use crate::html;
use crate::og;
use crate::scan::ScanResult;
use crate::store::ScanStore;
use crate::style;

use crate::templates;

/// Top-level site configuration, parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct SiteConfig {
    pub title: String,
    pub output_dir: String,
    /// Base URL for the deployed site (e.g. "https://wiimi.wseaton.com").
    /// Used for absolute OG image URLs. No trailing slash.
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(rename = "family")]
    pub families: Vec<FamilyConfig>,
}

/// How to order tags when applying `last_n` truncation.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TagOrder {
    /// Sort by tag string (works well for vX.Y.Z semver tags).
    #[default]
    Lexicographic,
    /// Sort by the time each image was first scanned/stored.
    ScanTime,
}

/// A named group of images discovered from a registry by tag pattern.
#[derive(Debug, Clone, Deserialize)]
pub struct FamilyConfig {
    pub name: String,
    /// Registry hostname (e.g. "ghcr.io").
    pub registry: String,
    /// Repository path (e.g. "llm-d/llm-d-cuda").
    pub repository: String,
    /// Regex pattern to match tags (e.g. `^v\d+\.\d+\.\d+(-rc\.\d+)?$`).
    pub tag_pattern: String,
    /// Only keep the last N matching tags. Useful for local previews or
    /// keeping the catalog focused on recent releases.
    #[serde(default)]
    pub last_n: Option<usize>,
    /// How to order tags when applying `last_n`. Defaults to lexicographic.
    #[serde(default)]
    pub order_by: TagOrder,
}

impl FamilyConfig {
    /// Image reference prefix for store queries, e.g. "ghcr.io/llm-d/llm-d-cuda:".
    pub fn image_prefix(&self) -> String {
        format!("{}/{}:", self.registry, self.repository)
    }

    /// Compile the tag pattern into a regex.
    pub fn compiled_tag_pattern(&self) -> Result<Regex> {
        Regex::new(&self.tag_pattern).with_context(|| {
            format!(
                "invalid tag_pattern for family '{}': {}",
                self.name, self.tag_pattern
            )
        })
    }

    /// Build a full image reference from a tag.
    pub fn image_ref(&self, tag: &str) -> String {
        format!("{}/{}:{}", self.registry, self.repository, tag)
    }
}

impl SiteConfig {
    pub fn from_toml(s: &str) -> Result<Self> {
        toml::from_str(s).context("failed to parse site config")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        Self::from_toml(&content)
    }
}

/// JSON blob embedded in the index page so the dropdowns can navigate.
#[derive(serde::Serialize)]
struct SiteData {
    families: Vec<FamilyData>,
}

#[derive(serde::Serialize)]
struct FamilyData {
    name: String,
    /// Map of "from_image|to_image" -> relative diff URL.
    diff_urls: std::collections::HashMap<String, String>,
}

/// Data for a single scan row in the index page template.
#[derive(serde::Serialize)]
struct IndexScanRow {
    image: String,
    slug: String,
    short_name: String,
    cuda: String,
    cc_range: String,
    ptx_class: &'static str,
    ptx_label: &'static str,
    bin_count: usize,
}

/// Data for a single family section in the index page template.
#[derive(serde::Serialize)]
struct IndexFamily {
    idx: usize,
    name: String,
    scans: Vec<IndexScanRow>,
}

/// Build the nav link HTML for site-generated pages.
fn nav_link_html(index_url: &str) -> String {
    format!(
        r#"<a href="{index_url}" style="display:flex;align-items:center;padding:0 16px;color:var(--accent);text-decoration:none;font-size:0.85rem;border-right:1px solid var(--border);">&larr; Catalog</a>"#
    )
}

/// Build an `OgContext` from `OgMeta`, resolving the image URL with the base URL.
fn og_context(meta: &og::OgMeta, base_url: Option<&str>) -> diff::OgContext {
    let image_url = match base_url {
        Some(base) => format!("{}{}", base.trim_end_matches('/'), meta.image_path),
        None => meta.image_path.clone(),
    };
    diff::OgContext {
        title: meta.title.clone(),
        description: meta.description.clone(),
        image_url,
        og_type: meta.og_type.clone(),
    }
}

/// Resolve scans belonging to a family by querying the store by prefix,
/// filtering tags against the family's pattern, sorting by the configured
/// order, and truncating to `last_n` if set.
pub fn resolve_family_scans(family: &FamilyConfig, store: &ScanStore) -> Result<Vec<ScanResult>> {
    use crate::store::ScanWithTimestamp;

    let prefix = family.image_prefix();
    let pattern = family.compiled_tag_pattern()?;
    let all_scans = store.query_by_prefix(&prefix)?;

    let mut matched: Vec<ScanWithTimestamp> = all_scans
        .into_iter()
        .filter(|entry| {
            let tag = entry.scan.image.strip_prefix(&prefix).unwrap_or("");
            pattern.is_match(tag)
        })
        .collect();

    match family.order_by {
        TagOrder::Lexicographic => matched.sort_by(|a, b| a.scan.image.cmp(&b.scan.image)),
        TagOrder::ScanTime => matched.sort_by(|a, b| a.scanned_at.cmp(&b.scanned_at)),
    }

    if let Some(n) = family.last_n {
        let start = matched.len().saturating_sub(n);
        matched = matched.split_off(start);
    }

    Ok(matched.into_iter().map(|entry| entry.scan).collect())
}

/// Generate the full static site into `config.output_dir`.
pub fn generate_site(config: &SiteConfig, store: &ScanStore) -> Result<()> {
    let out = Path::new(&config.output_dir);
    std::fs::create_dir_all(out.join("scan")).context("failed to create scan output directory")?;
    std::fs::create_dir_all(out.join("diff")).context("failed to create diff output directory")?;

    // Write favicon
    std::fs::write(out.join("favicon.svg"), templates::FAVICON_SVG)
        .context("failed to write favicon.svg")?;

    let nav = nav_link_html("../index.html");

    let mut site_data = SiteData {
        families: Vec::new(),
    };
    let mut index_families: Vec<IndexFamily> = Vec::new();
    let mut og_data: Vec<og::FamilyOgData> = Vec::new();

    for (family_idx, family) in config.families.iter().enumerate() {
        let scans = resolve_family_scans(family, store)?;
        if scans.is_empty() {
            tracing::warn!(family = %family.name, "no matching scans found, skipping");
            continue;
        }

        // Generate individual scan pages with OG meta tags
        for scan in &scans {
            let meta = og::scan_og_meta(scan);
            let og_ctx = og_context(&meta, config.base_url.as_deref());
            let html = html::render_html_with_og(
                scan,
                style::BundleMode::Cdn,
                Some(&og_ctx),
                "../favicon.svg",
                &nav,
            );
            let slug = slug_for_image(&scan.image);
            let path = out.join("scan").join(format!("{slug}.html"));
            std::fs::write(&path, &html)
                .with_context(|| format!("failed to write scan page: {}", path.display()))?;
        }

        // Compute and generate all N-choose-2 diff pages
        let pairs = all_pairs(scans.len());
        let mut diff_urls = std::collections::HashMap::new();

        for (i, j) in &pairs {
            let from = &scans[*i];
            let to = &scans[*j];
            let diff_result = diff::compute_diff(from, to);
            let meta = og::diff_og_meta(&diff_result);
            let og_ctx = og_context(&meta, config.base_url.as_deref());
            let html =
                diff::render_diff_html_with_og(&diff_result, Some(&og_ctx), "../favicon.svg", &nav);

            let from_slug = slug_for_image(&from.image);
            let to_slug = slug_for_image(&to.image);
            let filename = format!("{from_slug}-vs-{to_slug}.html");
            let path = out.join("diff").join(&filename);
            std::fs::write(&path, &html)
                .with_context(|| format!("failed to write diff page: {}", path.display()))?;

            let url = format!("diff/{filename}");
            let key = format!("{}|{}", from.image, to.image);
            diff_urls.insert(key, url);
        }

        site_data.families.push(FamilyData {
            name: family.name.clone(),
            diff_urls,
        });

        // Build index family data for the template
        let scan_rows: Vec<IndexScanRow> = scans
            .iter()
            .map(|scan| {
                let cc_range = match (&scan.effective_cc_min, &scan.effective_cc_max) {
                    (Some(min), Some(max)) => format!("{min} - {max}"),
                    _ => "-".to_string(),
                };
                IndexScanRow {
                    image: scan.image.clone(),
                    slug: slug_for_image(&scan.image),
                    short_name: short_image_name(&scan.image),
                    cuda: scan
                        .metadata
                        .cuda_version
                        .as_deref()
                        .unwrap_or("-")
                        .to_string(),
                    cc_range,
                    ptx_class: if scan.has_ptx_forward_compat {
                        "ptx-yes"
                    } else {
                        "ptx-no"
                    },
                    ptx_label: if scan.has_ptx_forward_compat {
                        "Yes"
                    } else {
                        "No"
                    },
                    bin_count: scan.binaries.len(),
                }
            })
            .collect();

        index_families.push(IndexFamily {
            idx: family_idx,
            name: family.name.clone(),
            scans: scan_rows,
        });

        og_data.push((scans, pairs));
    }

    // Generate OG card images
    og::generate_og_images(config, &og_data, out)?;

    // Render index page with OG meta
    let site_json = serde_json::to_string(&site_data).context("failed to serialize site data")?;
    let index_meta = og::index_og_meta(config);
    let og_image_url = match config.base_url.as_deref() {
        Some(base) => format!("{}{}", base.trim_end_matches('/'), index_meta.image_path),
        None => index_meta.image_path.clone(),
    };

    let tmpl = templates::HTML_ENV
        .get_template("index")
        .expect("index template registered");
    let index_html = tmpl
        .render(minijinja::context! {
            base_css => templates::BASE_CSS,
            site_title => config.title,
            families => index_families,
            site_data => site_json,
            og_title => index_meta.title,
            og_description => index_meta.description,
            og_image => og_image_url,
            og_type => index_meta.og_type,
        })
        .context("failed to render index template")?;

    std::fs::write(out.join("index.html"), &index_html).context("failed to write index.html")?;

    Ok(())
}

/// All (i, j) pairs where i < j, for N-choose-2 diff computation.
fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Turn an image reference into a filesystem-safe slug.
pub fn slug_for_image(image: &str) -> String {
    crate::sanitize_filename(image)
}

/// Shorten an image reference to just the tag or last meaningful segment.
pub fn short_image_name(image: &str) -> String {
    // "ghcr.io/llm-d/llm-d-cuda:v0.5.0" -> "llm-d-cuda:v0.5.0"
    if let Some(path_and_tag) = image.split('/').next_back() {
        path_and_tag.to_string()
    } else {
        image.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::image::BinaryPriority;
    use crate::nvidia::ComputeCapability;
    use crate::scan::{BinaryScanResult, EnvironmentInfo, ImageMetadata, ScanResult};
    use crate::site::{
        all_pairs, resolve_family_scans, short_image_name, slug_for_image, FamilyConfig, SiteConfig,
    };
    use crate::store::ScanStore;

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
    }

    fn test_family(name: &str, registry: &str, repository: &str, pattern: &str) -> FamilyConfig {
        FamilyConfig {
            name: name.to_string(),
            registry: registry.to_string(),
            repository: repository.to_string(),
            tag_pattern: pattern.to_string(),
            last_n: None,
            order_by: crate::site::TagOrder::Lexicographic,
        }
    }

    fn test_scan(image: &str) -> ScanResult {
        ScanResult {
            image: image.to_string(),
            metadata: ImageMetadata {
                cuda_version: Some("12.4".to_string()),
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![BinaryScanResult {
                path: "/lib/test.so".to_string(),
                priority: BinaryPriority::LinkerLibrary,
                size: 4096,
                cubins: vec![cc(7, 0), cc(9, 0)],
                ptx: vec![cc(9, 0)],
                needed: vec![],
                soname: None,
                rpath: vec![],
                runpath: vec![],
                layer_index: None,
            }],
            effective_cc_min: Some(cc(7, 0)),
            effective_cc_max: Some(cc(9, 0)),
            has_ptx_forward_compat: true,
            warnings: vec![],
            dep_graph: None,
            reachable_count: 1,
            dormant_count: 0,
            environment: EnvironmentInfo {
                os: None,
                python_environments: vec![],
                system_packages: vec![],
                ld_conf_paths: vec![],
                symlinks: HashMap::new(),
                file_owners: HashMap::new(),
            },
            labels: HashMap::new(),
            env_vars: vec![],
            layer_history: vec![],
        }
    }

    #[test]
    fn parse_site_config() {
        let toml = r#"
title = "Test Catalog"
output_dir = "_site"

[[family]]
name = "Test Family"
registry = "ghcr.io"
repository = "test/image"
tag_pattern = '^v\d+$'
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        assert_eq!(config.title, "Test Catalog");
        assert_eq!(config.output_dir, "_site");
        assert_eq!(config.families.len(), 1);
        assert_eq!(config.families[0].name, "Test Family");
        assert_eq!(config.families[0].registry, "ghcr.io");
        assert_eq!(config.families[0].repository, "test/image");
    }

    #[test]
    fn parse_multi_family_config() {
        let toml = r#"
title = "Multi"
output_dir = "out"

[[family]]
name = "A"
registry = "ghcr.io"
repository = "org/a"
tag_pattern = '^v\d+$'

[[family]]
name = "B"
registry = "ghcr.io"
repository = "org/b"
tag_pattern = '^v\d+$'
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        assert_eq!(config.families.len(), 2);
        assert_eq!(config.families[0].repository, "org/a");
        assert_eq!(config.families[1].repository, "org/b");
    }

    #[test]
    fn parse_config_with_last_n_and_order() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "Recent Only"
registry = "ghcr.io"
repository = "org/repo"
tag_pattern = '^v\d+$'
last_n = 5
order_by = "scan_time"
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        let family = &config.families[0];
        assert_eq!(family.last_n, Some(5));
        assert_eq!(family.order_by, crate::site::TagOrder::ScanTime);
    }

    #[test]
    fn parse_config_defaults_order_and_last_n() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "All"
registry = "ghcr.io"
repository = "org/repo"
tag_pattern = '^v\d+$'
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        let family = &config.families[0];
        assert_eq!(family.last_n, None);
        assert_eq!(family.order_by, crate::site::TagOrder::Lexicographic);
    }

    #[test]
    fn family_config_helpers() {
        let family = test_family("Test", "ghcr.io", "llm-d/llm-d-cuda", r"^v\d+\.\d+\.\d+$");
        assert_eq!(family.image_prefix(), "ghcr.io/llm-d/llm-d-cuda:");
        assert_eq!(
            family.image_ref("v0.5.0"),
            "ghcr.io/llm-d/llm-d-cuda:v0.5.0"
        );
        let re = family.compiled_tag_pattern().unwrap();
        assert!(re.is_match("v0.5.0"));
        assert!(!re.is_match("latest"));
        assert!(!re.is_match("sha-abc123"));
    }

    #[test]
    fn resolve_family_scans_filters_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        // Store scans with various tags
        store.upsert(&test_scan("ghcr.io/org/repo:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v2.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:latest")).unwrap();
        store
            .upsert(&test_scan("ghcr.io/org/repo:sha-abc123"))
            .unwrap();
        store
            .upsert(&test_scan("ghcr.io/org/other:v1.0.0"))
            .unwrap();

        let family = test_family("Test", "ghcr.io", "org/repo", r"^v\d+\.\d+\.\d+$");
        let scans = resolve_family_scans(&family, &store).unwrap();

        assert_eq!(scans.len(), 2);
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v1.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v2.0.0");
    }

    #[test]
    fn resolve_family_scans_last_n_lexicographic() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/org/repo:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v2.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v3.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v4.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v5.0.0")).unwrap();

        let mut family = test_family("Test", "ghcr.io", "org/repo", r"^v\d+\.\d+\.\d+$");
        family.last_n = Some(3);

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 3);
        // Should keep the last 3 lexicographically (highest versions)
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v3.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v4.0.0");
        assert_eq!(scans[2].image, "ghcr.io/org/repo:v5.0.0");
    }

    #[test]
    fn resolve_family_scans_last_n_scan_time() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        // Insert in non-lexicographic order to verify scan_time ordering
        // scanned_at is set to "now" on each upsert, so insertion order = scan time order
        store.upsert(&test_scan("ghcr.io/org/repo:v3.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v5.0.0")).unwrap();

        let mut family = test_family("Test", "ghcr.io", "org/repo", r"^v\d+\.\d+\.\d+$");
        family.last_n = Some(2);
        family.order_by = crate::site::TagOrder::ScanTime;

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 2);
        // Last 2 by scan time: v1.0.0 then v5.0.0 (insertion order)
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v1.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v5.0.0");
    }

    #[test]
    fn resolve_family_scans_last_n_larger_than_total() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/org/repo:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v2.0.0")).unwrap();

        let mut family = test_family("Test", "ghcr.io", "org/repo", r"^v\d+\.\d+\.\d+$");
        family.last_n = Some(10);

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 2);
    }

    #[test]
    fn resolve_family_scans_empty_when_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/org/repo:latest")).unwrap();

        let family = test_family("Test", "ghcr.io", "org/repo", r"^v\d+\.\d+\.\d+$");
        let scans = resolve_family_scans(&family, &store).unwrap();
        assert!(scans.is_empty());
    }

    #[test]
    fn all_pairs_count() {
        assert_eq!(all_pairs(9).len(), 36);
        assert_eq!(all_pairs(3).len(), 3);
        assert_eq!(all_pairs(2).len(), 1);
        assert_eq!(all_pairs(1).len(), 0);
        assert_eq!(all_pairs(0).len(), 0);
    }

    #[test]
    fn all_pairs_values() {
        let pairs = all_pairs(3);
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn slug_generation() {
        assert_eq!(
            slug_for_image("ghcr.io/llm-d/llm-d-cuda:v0.5.0"),
            "ghcr.io_llm-d_llm-d-cuda_v0.5.0"
        );
    }

    #[test]
    fn short_name_extraction() {
        assert_eq!(
            short_image_name("ghcr.io/llm-d/llm-d-cuda:v0.5.0"),
            "llm-d-cuda:v0.5.0"
        );
        assert_eq!(short_image_name("simple:latest"), "simple:latest");
    }

    #[test]
    fn nav_link_html_contains_catalog() {
        let html = crate::site::nav_link_html("../index.html");
        assert!(html.contains("Catalog"));
        assert!(html.contains("../index.html"));
    }

    #[test]
    fn generate_site_to_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/test/a:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/test/a:v2.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/test/a:v3.0.0")).unwrap();

        let out_dir = dir.path().join("site_out");
        let config = SiteConfig {
            title: "Test".to_string(),
            output_dir: out_dir.to_string_lossy().to_string(),
            base_url: None,
            families: vec![test_family(
                "Test",
                "ghcr.io",
                "test/a",
                r"^v\d+\.\d+\.\d+$",
            )],
        };

        crate::site::generate_site(&config, &store).unwrap();

        assert!(out_dir.join("index.html").exists());

        assert!(out_dir.join("scan").is_dir());
        let scan_files: Vec<_> = std::fs::read_dir(out_dir.join("scan")).unwrap().collect();
        assert_eq!(scan_files.len(), 3);

        // Diff pages: 3-choose-2 = 3
        assert!(out_dir.join("diff").is_dir());
        let diff_files: Vec<_> = std::fs::read_dir(out_dir.join("diff")).unwrap().collect();
        assert_eq!(diff_files.len(), 3);

        let index = std::fs::read_to_string(out_dir.join("index.html")).unwrap();
        assert!(index.contains("Test"));
        assert!(index.contains("diff-from"));
        assert!(index.contains("diff-to"));
    }

    #[test]
    fn generate_site_skips_empty_families() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        let out_dir = dir.path().join("site_out");
        let config = SiteConfig {
            title: "Test".to_string(),
            output_dir: out_dir.to_string_lossy().to_string(),
            base_url: None,
            families: vec![test_family(
                "Empty",
                "ghcr.io",
                "org/nonexistent",
                r"^v\d+$",
            )],
        };

        // Should succeed, not error, when no scans match
        crate::site::generate_site(&config, &store).unwrap();
        assert!(out_dir.join("index.html").exists());
    }

    #[test]
    fn nav_injected_in_scan_pages() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/test/x:v1.0.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/test/x:v2.0.0")).unwrap();

        let out_dir = dir.path().join("site_out");
        let config = SiteConfig {
            title: "Nav Test".to_string(),
            output_dir: out_dir.to_string_lossy().to_string(),
            base_url: None,
            families: vec![test_family("X", "ghcr.io", "test/x", r"^v\d+\.\d+\.\d+$")],
        };

        crate::site::generate_site(&config, &store).unwrap();

        let scan_files: Vec<_> = std::fs::read_dir(out_dir.join("scan"))
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        let scan_html = std::fs::read_to_string(scan_files[0].path()).unwrap();
        assert!(scan_html.contains("Catalog"));

        let diff_files: Vec<_> = std::fs::read_dir(out_dir.join("diff"))
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        let diff_html = std::fs::read_to_string(diff_files[0].path()).unwrap();
        assert!(diff_html.contains("Catalog"));
    }
}
