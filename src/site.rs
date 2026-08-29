use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

use crate::diff;
use crate::html;
use crate::nvidia::ComputeCapability;
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
    /// Split on non-digit characters and compare the resulting integer segments.
    /// Handles schemes like `0.13.0_rhai12` where lexicographic sorting breaks.
    Numeric,
    /// Proper semver comparison. Strips a leading `v` prefix before parsing.
    Semver,
    /// Semver comparison with a build number tiebreaker. Requires
    /// `build_separator` on the family config to split the tag into
    /// `{semver}{separator}{build_number}`.
    SemverBuild,
}

/// Extract all contiguous runs of digits from a tag as a `Vec<u64>`.
/// e.g. `"0.13.0_rhai12"` becomes `[0, 13, 0, 12]`.
pub fn numeric_sort_key(tag: &str) -> Vec<u64> {
    tag.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// Parse a tag as semver, stripping a leading `v` if present.
pub fn parse_semver(tag: &str) -> Option<semver::Version> {
    let stripped = tag.strip_prefix('v').unwrap_or(tag);
    semver::Version::parse(stripped).ok()
}

/// Parse a tag as semver + build number, splitting on the given separator.
/// Returns `(version, build_number)` for comparison.
pub fn parse_semver_build(tag: &str, separator: &str) -> Option<(semver::Version, u64)> {
    let (ver_part, build_part) = tag.rsplit_once(separator)?;
    let version = semver::Version::parse(ver_part).ok()?;
    let build = build_part.parse::<u64>().ok()?;
    Some((version, build))
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
    /// Separator between the semver portion and the build number in tags.
    /// Required when `order_by = "semver_build"`. E.g. `"_rhai"` splits
    /// `0.13.0_rhai12` into semver `0.13.0` and build `12`.
    #[serde(default)]
    pub build_separator: Option<String>,
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
        let config: Self = toml::from_str(s).context("failed to parse site config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        Self::from_toml(&content)
    }

    fn validate(&self) -> Result<()> {
        for family in &self.families {
            if family.order_by == TagOrder::SemverBuild && family.build_separator.is_none() {
                anyhow::bail!(
                    "family '{}': order_by = \"semver_build\" requires a build_separator",
                    family.name
                );
            }
        }
        Ok(())
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
    /// Image ref + display name for the compare dropdowns.
    scans: Vec<FamilyScanEntry>,
    /// Map of "from_image|to_image" -> relative diff URL.
    diff_urls: std::collections::HashMap<String, String>,
}

#[derive(serde::Serialize)]
struct FamilyScanEntry {
    image: String,
    short_name: String,
}

/// Data for a single scan row in the index page template.
#[derive(serde::Serialize)]
struct IndexScanRow {
    image: String,
    slug: String,
    short_name: String,
    cuda: String,
    cc_range: String,
    cc_note: String,
    cc_bar: String,
    ptx_class: &'static str,
    ptx_label: &'static str,
    bin_count: usize,
}

/// Data for a single family section in the index page template.
#[derive(serde::Serialize)]
struct IndexFamily {
    idx: usize,
    name: String,
    subtitle: String,
    /// The tag regex, shown on hover rather than printed into the page.
    pattern: String,
    scans: Vec<IndexScanRow>,
}

/// `sm_NN` value for a compute capability, e.g. 9.0 -> 90.
fn sm_value(cc: &ComputeCapability) -> u32 {
    cc.major * 10 + cc.minor
}

/// Every distinct `sm` value that appears anywhere in the catalog, sorted.
///
/// The index draws one cell per value from this shared scale so rows line up
/// and can be compared down the column. A per-scan scale would put different
/// architectures in the same position on different rows.
fn catalog_sm_scale(all: &[&ScanResult]) -> Vec<u32> {
    let mut vals: Vec<u32> = all
        .iter()
        .flat_map(|s| s.binaries.iter())
        .flat_map(|b| b.cubins.iter().chain(b.ptx.iter()))
        .map(sm_value)
        .collect();
    vals.sort_unstable();
    vals.dedup();
    vals
}

fn sm_gen_class(sm: u32) -> &'static str {
    match sm {
        ..50 => "g-legacy",
        50..60 => "g-maxwell",
        60..70 => "g-pascal",
        70..80 => "g-volta",
        80..90 => "g-ampere",
        90..100 => "g-hopper",
        _ => "g-blackwell",
    }
}

/// Render the effective capability range for the index, with a note when it is empty.
///
/// `cc_min` is the highest per-binary floor and `cc_max` the lowest per-binary
/// ceiling, so together they describe the architectures *every* binary supports.
/// A floor above the ceiling means that intersection is empty. The image still
/// runs; there is just no single architecture covering all of its binaries, which
/// is ordinary for a large image mixing narrowly targeted libraries. Printing the
/// endpoints in either order would claim the opposite of what holds.
fn index_cc_range(
    cc_min: Option<&ComputeCapability>,
    cc_max: Option<&ComputeCapability>,
) -> (String, String) {
    match (cc_min, cc_max) {
        (Some(min), Some(max)) if sm_value(min) > sm_value(max) => (
            "none".to_string(),
            format!(
                "no architecture is supported by every binary: the highest floor \
                 ({min}) is above the lowest ceiling ({max})"
            ),
        ),
        (Some(min), Some(max)) => (format!("{min} - {max}"), String::new()),
        _ => ("-".to_string(), String::new()),
    }
}

/// Render the shared-scale capability strip for one scan.
fn cc_bar_html(scan: &ScanResult, scale: &[u32]) -> String {
    use std::collections::HashSet;
    let cubins: HashSet<u32> = scan
        .binaries
        .iter()
        .flat_map(|b| b.cubins.iter())
        .map(sm_value)
        .collect();
    let ptx: HashSet<u32> = scan
        .binaries
        .iter()
        .flat_map(|b| b.ptx.iter())
        .map(sm_value)
        .collect();

    let mut h = String::from("<span class=\"cc-bar\">");
    for sm in scale {
        if cubins.contains(sm) {
            h.push_str(&format!(
                "<span class=\"c sass {}\" title=\"sm_{sm} compiled\"></span>",
                sm_gen_class(*sm)
            ));
        } else if ptx.contains(sm) {
            h.push_str(&format!(
                "<span class=\"c ptx\" title=\"sm_{sm} PTX only\"></span>"
            ));
        } else {
            h.push_str(&format!(
                "<span class=\"c empty\" title=\"sm_{sm} absent\"></span>"
            ));
        }
    }
    h.push_str("</span>");
    h
}

/// Build a human-readable subtitle explaining the family's filter/sort config.
fn family_subtitle(family: &FamilyConfig) -> String {
    let count = match family.last_n {
        Some(n) => format!("{n} most recent"),
        None => "All".to_string(),
    };
    let order = match family.order_by {
        TagOrder::Lexicographic => "by tag name",
        TagOrder::ScanTime => "by scan time",
        TagOrder::Numeric => "by version number",
        TagOrder::Semver => "by version",
        TagOrder::SemverBuild => "by version and build",
    };
    format!("{count} tags, ordered {order}")
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
        TagOrder::Numeric => {
            let prefix = family.image_prefix();
            matched.sort_by(|a, b| {
                let tag_a = a.scan.image.strip_prefix(&prefix).unwrap_or("");
                let tag_b = b.scan.image.strip_prefix(&prefix).unwrap_or("");
                numeric_sort_key(tag_a).cmp(&numeric_sort_key(tag_b))
            });
        }
        TagOrder::Semver => {
            let prefix = family.image_prefix();
            matched.sort_by(|a, b| {
                let tag_a = a.scan.image.strip_prefix(&prefix).unwrap_or("");
                let tag_b = b.scan.image.strip_prefix(&prefix).unwrap_or("");
                let va = parse_semver(tag_a);
                let vb = parse_semver(tag_b);
                match (va, vb) {
                    (Some(a), Some(b)) => a.cmp(&b),
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (None, None) => tag_a.cmp(tag_b),
                }
            });
        }
        TagOrder::SemverBuild => {
            let prefix = family.image_prefix();
            let sep = family
                .build_separator
                .as_deref()
                .expect("semver_build requires build_separator (validated at config load)");
            matched.sort_by(|a, b| {
                let tag_a = a.scan.image.strip_prefix(&prefix).unwrap_or("");
                let tag_b = b.scan.image.strip_prefix(&prefix).unwrap_or("");
                let va = parse_semver_build(tag_a, sep);
                let vb = parse_semver_build(tag_b, sep);
                match (va, vb) {
                    (Some(a), Some(b)) => a.cmp(&b),
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (None, None) => tag_a.cmp(tag_b),
                }
            });
        }
    }

    if let Some(n) = family.last_n {
        let start = matched.len().saturating_sub(n);
        matched = matched.split_off(start);
    }

    // Latest first: reverse so the table shows newest versions at the top.
    matched.reverse();

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

    // Resolve every family before rendering any of them: the capability strip
    // draws on one scale shared across the whole catalog, which is not known
    // until all scans are in hand.
    let mut resolved = Vec::new();
    for (family_idx, family) in config.families.iter().enumerate() {
        let scans = resolve_family_scans(family, store)?;
        if scans.is_empty() {
            tracing::warn!(family = %family.name, "no matching scans found, skipping");
            continue;
        }
        resolved.push((family_idx, family, scans));
    }

    let sm_scale;
    let show_ptx;
    {
        let all: Vec<&ScanResult> = resolved.iter().flat_map(|(_, _, s)| s.iter()).collect();
        sm_scale = catalog_sm_scale(&all);
        // A column that says the same thing on every row is not telling the
        // reader anything.
        let mut vals = all.iter().map(|s| s.has_ptx_forward_compat);
        show_ptx = match vals.next() {
            Some(first) => vals.any(|v| v != first),
            None => false,
        };
    }

    for (family_idx, family, scans) in resolved {
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

        let scan_entries: Vec<FamilyScanEntry> = scans
            .iter()
            .map(|s| FamilyScanEntry {
                image: s.image.clone(),
                short_name: short_image_name(&s.image),
            })
            .collect();

        site_data.families.push(FamilyData {
            name: family.name.clone(),
            scans: scan_entries,
            diff_urls,
        });

        // Build index family data for the template
        let scan_rows: Vec<IndexScanRow> = scans
            .iter()
            .map(|scan| {
                let (cc_range, cc_note) = index_cc_range(
                    scan.effective_cc_min.as_ref(),
                    scan.effective_cc_max.as_ref(),
                );
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
                    cc_note,
                    cc_bar: cc_bar_html(scan, &sm_scale),
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
            subtitle: family_subtitle(family),
            pattern: family.tag_pattern.clone(),
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
            show_ptx => show_ptx,
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
        all_pairs, catalog_sm_scale, cc_bar_html, family_subtitle, index_cc_range,
        numeric_sort_key, resolve_family_scans, short_image_name, slug_for_image, sm_gen_class,
        sm_value, FamilyConfig, SiteConfig, TagOrder,
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
            build_separator: None,
        }
    }

    #[test]
    fn sm_value_packs_major_minor() {
        assert_eq!(sm_value(&cc(9, 0)), 90);
        assert_eq!(sm_value(&cc(12, 1)), 121);
        assert_eq!(sm_value(&cc(5, 2)), 52);
    }

    #[test]
    fn sm_gen_class_covers_every_boundary() {
        assert_eq!(sm_gen_class(35), "g-legacy");
        assert_eq!(sm_gen_class(50), "g-maxwell");
        assert_eq!(sm_gen_class(52), "g-maxwell");
        assert_eq!(sm_gen_class(60), "g-pascal");
        assert_eq!(sm_gen_class(70), "g-volta");
        assert_eq!(sm_gen_class(75), "g-volta");
        assert_eq!(sm_gen_class(80), "g-ampere");
        assert_eq!(sm_gen_class(89), "g-ampere");
        assert_eq!(sm_gen_class(90), "g-hopper");
        assert_eq!(sm_gen_class(100), "g-blackwell");
        assert_eq!(sm_gen_class(121), "g-blackwell");
    }

    #[test]
    fn catalog_scale_is_sorted_deduped_and_spans_every_scan() {
        let a = test_scan("img:a");
        let b = test_scan("img:b");
        // test_scan carries cubins 7.0 and 9.0 plus ptx 9.0
        assert_eq!(catalog_sm_scale(&[&a, &b]), vec![70, 90]);
    }

    #[test]
    fn cc_bar_marks_compiled_ptx_and_absent_against_the_shared_scale() {
        let scan = test_scan("img:a");
        // 80 appears in neither cubins nor ptx for this scan.
        let html = cc_bar_html(&scan, &[70, 80, 90]);
        let cells: Vec<&str> = html.split("<span class=\"c ").skip(1).collect();
        assert_eq!(cells.len(), 3, "one cell per scale entry");
        assert!(
            cells[0].starts_with("sass g-volta"),
            "70 is compiled: {}",
            cells[0]
        );
        assert!(cells[1].starts_with("empty"), "80 is absent: {}", cells[1]);
        assert!(
            cells[2].starts_with("sass g-hopper"),
            "90 is compiled: {}",
            cells[2]
        );
    }

    #[test]
    fn cc_bar_prefers_compiled_over_ptx_for_the_same_level() {
        // test_scan has both a cubin and PTX at 9.0; compiled code wins the cell.
        let scan = test_scan("img:a");
        let html = cc_bar_html(&scan, &[90]);
        assert!(html.contains("sass g-hopper"));
        assert!(!html.contains("c ptx"));
    }

    #[test]
    fn cc_bar_on_an_empty_scale_is_an_empty_strip() {
        let scan = test_scan("img:a");
        assert_eq!(cc_bar_html(&scan, &[]), "<span class=\"cc-bar\"></span>");
    }

    #[test]
    fn family_subtitle_does_not_leak_the_tag_regex() {
        let family = FamilyConfig {
            name: "Test".to_string(),
            registry: "ghcr.io".to_string(),
            repository: "acme/thing".to_string(),
            tag_pattern: r"^v\d+\.\d+\.\d+$".to_string(),
            last_n: Some(3),
            order_by: TagOrder::Semver,
            build_separator: None,
        };
        let subtitle = family_subtitle(&family);
        assert!(
            !subtitle.contains("\\d"),
            "regex must not reach the page: {subtitle}"
        );
        assert!(
            !subtitle.contains('^'),
            "regex must not reach the page: {subtitle}"
        );
        assert!(subtitle.contains("3 most recent"), "{subtitle}");
        assert!(subtitle.contains("by version"), "{subtitle}");
    }

    #[test]
    fn cc_range_reports_an_empty_intersection_as_none() {
        // cc_min is the highest per-binary floor, cc_max the lowest ceiling, so
        // floor > ceiling means no architecture is common to every binary.
        // Printing "5.2 - 12.0" would claim the widest possible support.
        let (range, note) = index_cc_range(Some(&cc(12, 0)), Some(&cc(5, 2)));
        assert_eq!(range, "none");
        assert!(note.contains("every binary"), "{note}");
        assert!(note.contains("12.0") && note.contains("5.2"), "{note}");
    }

    #[test]
    fn cc_range_prints_a_real_intersection() {
        let (range, note) = index_cc_range(Some(&cc(7, 0)), Some(&cc(9, 0)));
        assert_eq!(range, "7.0 - 9.0");
        assert!(note.is_empty());
    }

    #[test]
    fn cc_range_handles_a_single_point_intersection() {
        let (range, note) = index_cc_range(Some(&cc(9, 0)), Some(&cc(9, 0)));
        assert_eq!(range, "9.0 - 9.0");
        assert!(note.is_empty());
    }

    #[test]
    fn cc_range_without_endpoints_is_a_dash() {
        assert_eq!(index_cc_range(None, None).0, "-");
        assert_eq!(index_cc_range(Some(&cc(9, 0)), None).0, "-");
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
        // Reversed: latest first
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v2.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v1.0.0");
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
        // Should keep the last 3 lexicographically, reversed (latest first)
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v5.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v4.0.0");
        assert_eq!(scans[2].image, "ghcr.io/org/repo:v3.0.0");
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
        // Last 2 by scan time, reversed (latest first)
        assert_eq!(scans[0].image, "ghcr.io/org/repo:v5.0.0");
        assert_eq!(scans[1].image, "ghcr.io/org/repo:v1.0.0");
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

    #[test]
    fn numeric_sort_key_extracts_components() {
        assert_eq!(numeric_sort_key("0.13.0_rhai12"), vec![0, 13, 0, 12]);
        assert_eq!(numeric_sort_key("v1.2.3"), vec![1, 2, 3]);
        assert_eq!(numeric_sort_key("0.14.1_rhai0"), vec![0, 14, 1, 0]);
        assert_eq!(numeric_sort_key("latest"), Vec::<u64>::new());
        assert_eq!(numeric_sort_key("42"), vec![42]);
        assert_eq!(numeric_sort_key(""), Vec::<u64>::new());
    }

    #[test]
    fn numeric_sort_key_ordering() {
        // The whole point: _rhai2 should sort before _rhai12
        let mut tags = vec![
            "0.13.0_rhai12",
            "0.13.0_rhai2",
            "0.14.1_rhai0",
            "0.13.0_rhai1",
        ];
        tags.sort_by_key(|a| numeric_sort_key(a));
        assert_eq!(
            tags,
            vec![
                "0.13.0_rhai1",
                "0.13.0_rhai2",
                "0.13.0_rhai12",
                "0.14.1_rhai0",
            ]
        );
    }

    #[test]
    fn rhai_tag_pattern_matches() {
        let pattern = regex::Regex::new(r"^\d+\.\d+\.\d+_rhai\d+$").unwrap();
        assert!(pattern.is_match("0.13.0_rhai12"));
        assert!(pattern.is_match("0.14.1_rhai0"));
        assert!(pattern.is_match("1.0.0_rhai99"));
        assert!(!pattern.is_match("v0.13.0_rhai12"));
        assert!(!pattern.is_match("latest"));
        assert!(!pattern.is_match("0.13.0"));
        assert!(!pattern.is_match("0.13.0_rhai"));
    }

    #[test]
    fn parse_config_with_numeric_order() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "CUDA"
registry = "quay.io"
repository = "vllm/vllm-cuda"
tag_pattern = '^\d+\.\d+\.\d+_rhai\d+$'
last_n = 10
order_by = "numeric"
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        let family = &config.families[0];
        assert_eq!(family.last_n, Some(10));
        assert_eq!(family.order_by, crate::site::TagOrder::Numeric);
    }

    #[test]
    fn resolve_family_scans_numeric_ordering_with_last_n() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        // Insert tags that would sort wrong lexicographically
        store
            .upsert(&test_scan("quay.io/vllm/vllm-cuda:0.13.0_rhai2"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/vllm-cuda:0.13.0_rhai12"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/vllm-cuda:0.13.0_rhai1"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/vllm-cuda:0.14.1_rhai0"))
            .unwrap();

        let mut family = test_family(
            "CUDA",
            "quay.io",
            "vllm/vllm-cuda",
            r"^\d+\.\d+\.\d+_rhai\d+$",
        );
        family.order_by = crate::site::TagOrder::Numeric;
        family.last_n = Some(2);

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 2);
        // Last 2 numerically, reversed (latest first)
        assert_eq!(scans[0].image, "quay.io/vllm/vllm-cuda:0.14.1_rhai0");
        assert_eq!(scans[1].image, "quay.io/vllm/vllm-cuda:0.13.0_rhai12");
    }

    #[test]
    fn parse_semver_valid() {
        use crate::site::parse_semver;

        let v = parse_semver("v1.2.3").unwrap();
        assert_eq!(v, semver::Version::new(1, 2, 3));

        let v = parse_semver("1.2.3").unwrap();
        assert_eq!(v, semver::Version::new(1, 2, 3));

        let v = parse_semver("v0.5.0-rc.1").unwrap();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 5);
        assert_eq!(v.patch, 0);
        assert!(!v.pre.is_empty());
    }

    #[test]
    fn parse_semver_invalid() {
        use crate::site::parse_semver;

        assert!(parse_semver("latest").is_none());
        assert!(parse_semver("").is_none());
        assert!(parse_semver("not-a-version").is_none());
        assert!(parse_semver("v1").is_none());
        assert!(parse_semver("v1.2").is_none());
    }

    #[test]
    fn parse_semver_build_valid() {
        use crate::site::parse_semver_build;

        let (v, b) = parse_semver_build("0.13.0_rhai12", "_rhai").unwrap();
        assert_eq!(v, semver::Version::new(0, 13, 0));
        assert_eq!(b, 12);

        let (v, b) = parse_semver_build("0.14.1_rhai0", "_rhai").unwrap();
        assert_eq!(v, semver::Version::new(0, 14, 1));
        assert_eq!(b, 0);
    }

    #[test]
    fn parse_semver_build_invalid() {
        use crate::site::parse_semver_build;

        assert!(parse_semver_build("0.13.0", "_rhai").is_none());
        assert!(parse_semver_build("0.13.0_rhai", "_rhai").is_none());
        assert!(parse_semver_build("latest_rhai5", "_rhai").is_none());
        assert!(parse_semver_build("", "_rhai").is_none());
    }

    #[test]
    fn semver_ordering() {
        use crate::site::parse_semver;

        let mut versions = vec!["v0.15.1", "v0.5.0-rc.1", "v0.5.0", "v0.7.3"];
        versions.sort_by_key(|a| parse_semver(a));
        assert_eq!(versions, vec!["v0.5.0-rc.1", "v0.5.0", "v0.7.3", "v0.15.1"]);
    }

    #[test]
    fn semver_build_ordering() {
        use crate::site::parse_semver_build;

        let sep = "_rhai";
        let mut tags = vec![
            "0.13.0_rhai12",
            "0.13.0_rhai2",
            "0.14.1_rhai0",
            "0.13.0_rhai1",
        ];
        tags.sort_by_key(|a| parse_semver_build(a, sep));
        assert_eq!(
            tags,
            vec![
                "0.13.0_rhai1",
                "0.13.0_rhai2",
                "0.13.0_rhai12",
                "0.14.1_rhai0",
            ]
        );
    }

    #[test]
    fn resolve_family_scans_semver_ordering_with_last_n() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        // These sort wrong lexicographically: "v0.7.3" > "v0.15.1"
        store
            .upsert(&test_scan("docker.io/vllm/openai:v0.7.3"))
            .unwrap();
        store
            .upsert(&test_scan("docker.io/vllm/openai:v0.15.1"))
            .unwrap();
        store
            .upsert(&test_scan("docker.io/vllm/openai:v0.5.0"))
            .unwrap();
        store
            .upsert(&test_scan("docker.io/vllm/openai:v0.5.0-rc.1"))
            .unwrap();

        let mut family = test_family(
            "vLLM",
            "docker.io",
            "vllm/openai",
            r"^v\d+\.\d+\.\d+(-rc\.\d+)?$",
        );
        family.order_by = crate::site::TagOrder::Semver;
        family.last_n = Some(3);

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 3);
        // Last 3 by semver, reversed (latest first)
        assert_eq!(scans[0].image, "docker.io/vllm/openai:v0.15.1");
        assert_eq!(scans[1].image, "docker.io/vllm/openai:v0.7.3");
        assert_eq!(scans[2].image, "docker.io/vllm/openai:v0.5.0");
    }

    #[test]
    fn resolve_family_scans_semver_build_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store
            .upsert(&test_scan("quay.io/vllm/cuda:0.13.0_rhai2"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/cuda:0.13.0_rhai12"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/cuda:0.14.1_rhai0"))
            .unwrap();
        store
            .upsert(&test_scan("quay.io/vllm/cuda:0.13.0_rhai1"))
            .unwrap();

        let mut family = test_family("RHAI", "quay.io", "vllm/cuda", r"^\d+\.\d+\.\d+_rhai\d+$");
        family.order_by = crate::site::TagOrder::SemverBuild;
        family.build_separator = Some("_rhai".to_string());
        family.last_n = Some(3);

        let scans = resolve_family_scans(&family, &store).unwrap();
        assert_eq!(scans.len(), 3);
        // Last 3 by semver+build, reversed (latest first)
        assert_eq!(scans[0].image, "quay.io/vllm/cuda:0.14.1_rhai0");
        assert_eq!(scans[1].image, "quay.io/vllm/cuda:0.13.0_rhai12");
        assert_eq!(scans[2].image, "quay.io/vllm/cuda:0.13.0_rhai2");
    }

    #[test]
    fn parse_config_with_semver_order() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "vLLM"
registry = "docker.io"
repository = "vllm/vllm-openai"
tag_pattern = '^v\d+\.\d+\.\d+$'
last_n = 10
order_by = "semver"
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        let family = &config.families[0];
        assert_eq!(family.order_by, crate::site::TagOrder::Semver);
        assert!(family.build_separator.is_none());
    }

    #[test]
    fn parse_config_with_semver_build_order() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "RHAI"
registry = "quay.io"
repository = "vllm/vllm-cuda"
tag_pattern = '^\d+\.\d+\.\d+_rhai\d+$'
last_n = 10
order_by = "semver_build"
build_separator = "_rhai"
"#;
        let config = SiteConfig::from_toml(toml).unwrap();
        let family = &config.families[0];
        assert_eq!(family.order_by, crate::site::TagOrder::SemverBuild);
        assert_eq!(family.build_separator.as_deref(), Some("_rhai"));
    }

    #[test]
    fn config_validation_rejects_semver_build_without_separator() {
        let toml = r#"
title = "Test"
output_dir = "_site"

[[family]]
name = "Bad"
registry = "quay.io"
repository = "vllm/vllm-cuda"
tag_pattern = '^\d+\.\d+\.\d+_rhai\d+$'
order_by = "semver_build"
"#;
        let err = SiteConfig::from_toml(toml).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("build_separator"),
            "expected error about build_separator, got: {msg}"
        );
    }
}
