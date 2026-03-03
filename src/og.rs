use std::path::Path;

use anyhow::{Context, Result};

use crate::diff::DiffResult;
use crate::scan::ScanResult;
use crate::site::{short_image_name, slug_for_image, SiteConfig};
use crate::templates;

/// Render an SVG string to a PNG byte vector at 1200x630.
fn render_svg_to_png(svg: &str) -> Result<Vec<u8>> {
    let mut fontdb = resvg::usvg::fontdb::Database::new();
    fontdb.load_system_fonts();

    let opts = resvg::usvg::Options {
        fontdb: std::sync::Arc::new(fontdb),
        ..Default::default()
    };
    let tree =
        resvg::usvg::Tree::from_str(svg, &opts).context("failed to parse OG card SVG template")?;

    let mut pixmap = resvg::tiny_skia::Pixmap::new(1200, 630)
        .context("failed to create 1200x630 pixmap for OG card")?;

    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::default(),
        &mut pixmap.as_mut(),
    );

    pixmap.encode_png().context("failed to encode OG card PNG")
}

/// Generate an OG card PNG for a scan result.
pub fn render_scan_card(scan: &ScanResult) -> Result<Vec<u8>> {
    let cuda = scan.metadata.cuda_version.as_deref().unwrap_or("unknown");

    let sm_range = match (&scan.effective_cc_min, &scan.effective_cc_max) {
        (Some(min), Some(max)) => format!("SM {min} \u{2192} {max}"),
        _ => "no SM data".to_string(),
    };

    let ptx = if scan.has_ptx_forward_compat {
        "PTX \u{2713}"
    } else {
        ""
    };

    let cuda_part = format!("CUDA {cuda}");
    let detail_parts: Vec<&str> = [cuda_part.as_str(), sm_range.as_str(), ptx]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    let details = detail_parts.join("  \u{00b7}  ");

    let binaries = format!(
        "{} binaries  ({} reachable, {} dormant)",
        scan.binaries.len(),
        scan.reachable_count,
        scan.dormant_count
    );

    let tmpl = templates::SVG_ENV
        .get_template("scan")
        .context("missing scan SVG template")?;
    let svg = tmpl
        .render(minijinja::context! {
            image => scan.image,
            details => details,
            binaries => binaries,
        })
        .context("failed to render scan OG card SVG")?;

    render_svg_to_png(&svg)
}

/// Generate an OG card PNG for a diff result.
pub fn render_diff_card(diff: &DiffResult) -> Result<Vec<u8>> {
    let from_short = short_image_name(&diff.from);
    let to_short = short_image_name(&diff.to);
    let from_to = format!("{from_short}  \u{2192}  {to_short}");

    let s = &diff.summary;

    let mut bin_parts = Vec::new();
    if s.binaries_added > 0 {
        bin_parts.push(format!("+{} binaries", s.binaries_added));
    }
    if s.binaries_removed > 0 {
        bin_parts.push(format!("-{} binaries", s.binaries_removed));
    }
    if s.binaries_changed > 0 {
        bin_parts.push(format!("{} changed", s.binaries_changed));
    }
    let binary_changes = if bin_parts.is_empty() {
        "No binary changes".to_string()
    } else {
        bin_parts.join("  ")
    };

    let mut pkg_parts = Vec::new();
    let pkg_added = s.python_packages_added + s.system_packages_added;
    let pkg_removed = s.python_packages_removed + s.system_packages_removed;
    if pkg_added > 0 {
        pkg_parts.push(format!("+{pkg_added} packages"));
    }
    if pkg_removed > 0 {
        pkg_parts.push(format!("-{pkg_removed} packages"));
    }
    let package_changes = if pkg_parts.is_empty() {
        "No package changes".to_string()
    } else {
        pkg_parts.join("  ")
    };

    let tmpl = templates::SVG_ENV
        .get_template("diff")
        .context("missing diff SVG template")?;
    let svg = tmpl
        .render(minijinja::context! {
            from_to => from_to,
            binary_changes => binary_changes,
            package_changes => package_changes,
        })
        .context("failed to render diff OG card SVG")?;

    render_svg_to_png(&svg)
}

/// Generate an OG card PNG for the index page.
pub fn render_index_card(config: &SiteConfig) -> Result<Vec<u8>> {
    let description = format!(
        "GPU image catalog with {} image {}",
        config.families.len(),
        if config.families.len() == 1 {
            "family"
        } else {
            "families"
        }
    );

    let tmpl = templates::SVG_ENV
        .get_template("index")
        .context("missing index SVG template")?;
    let svg = tmpl
        .render(minijinja::context! {
            title => config.title,
            description => description,
        })
        .context("failed to render index OG card SVG")?;

    render_svg_to_png(&svg)
}

/// OG metadata for template injection.
pub struct OgMeta {
    pub title: String,
    pub description: String,
    pub image_path: String,
    pub og_type: String,
}

/// Build OG metadata for a scan page.
pub fn scan_og_meta(scan: &ScanResult) -> OgMeta {
    let cuda = scan.metadata.cuda_version.as_deref().unwrap_or("unknown");
    let sm_range = match (&scan.effective_cc_min, &scan.effective_cc_max) {
        (Some(min), Some(max)) => format!("SM {min}-{max}"),
        _ => "no SM data".to_string(),
    };
    let description = format!(
        "CUDA {cuda}, {sm_range}, {} binaries ({} reachable)",
        scan.binaries.len(),
        scan.reachable_count
    );
    let slug = slug_for_image(&scan.image);

    OgMeta {
        title: scan.image.clone(),
        description,
        image_path: format!("/og/scan/{slug}.png"),
        og_type: "website".to_string(),
    }
}

/// Build OG metadata for a diff page.
pub fn diff_og_meta(diff: &DiffResult) -> OgMeta {
    let from_short = short_image_name(&diff.from);
    let to_short = short_image_name(&diff.to);
    let s = &diff.summary;
    let description = format!(
        "{} binaries added, {} removed, {} changed",
        s.binaries_added, s.binaries_removed, s.binaries_changed
    );
    let from_slug = slug_for_image(&diff.from);
    let to_slug = slug_for_image(&diff.to);

    OgMeta {
        title: format!("{from_short} \u{2192} {to_short}"),
        description,
        image_path: format!("/og/diff/{from_slug}-vs-{to_slug}.png"),
        og_type: "website".to_string(),
    }
}

/// Build OG metadata for the index page.
pub fn index_og_meta(config: &SiteConfig) -> OgMeta {
    OgMeta {
        title: config.title.clone(),
        description: format!(
            "GPU image scanning catalog with {} image {}",
            config.families.len(),
            if config.families.len() == 1 {
                "family"
            } else {
                "families"
            }
        ),
        image_path: "/og/index.png".to_string(),
        og_type: "website".to_string(),
    }
}

/// Family scans paired with their N-choose-2 diff index pairs.
pub type FamilyOgData = (Vec<ScanResult>, Vec<(usize, usize)>);

/// Write OG card PNGs for all scan and diff pages, plus the index.
pub fn generate_og_images(
    config: &SiteConfig,
    scans: &[FamilyOgData],
    out_dir: &Path,
) -> Result<()> {
    let og_dir = out_dir.join("og");
    std::fs::create_dir_all(og_dir.join("scan")).context("failed to create og/scan directory")?;
    std::fs::create_dir_all(og_dir.join("diff")).context("failed to create og/diff directory")?;

    // Index card
    let index_png = render_index_card(config)?;
    std::fs::write(og_dir.join("index.png"), &index_png).context("failed to write og/index.png")?;

    for (family_scans, pairs) in scans {
        // Scan cards
        for scan in family_scans {
            let slug = slug_for_image(&scan.image);
            let png = render_scan_card(scan)?;
            std::fs::write(og_dir.join("scan").join(format!("{slug}.png")), &png)
                .with_context(|| format!("failed to write OG card for {}", scan.image))?;
        }

        // Diff cards
        for (i, j) in pairs {
            let from = &family_scans[*i];
            let to = &family_scans[*j];
            let diff_result = crate::diff::compute_diff(from, to);
            let from_slug = slug_for_image(&from.image);
            let to_slug = slug_for_image(&to.image);
            let png = render_diff_card(&diff_result)?;
            std::fs::write(
                og_dir
                    .join("diff")
                    .join(format!("{from_slug}-vs-{to_slug}.png")),
                &png,
            )
            .with_context(|| {
                format!(
                    "failed to write OG diff card for {} vs {}",
                    from.image, to.image
                )
            })?;
        }
    }

    tracing::info!(
        path = %og_dir.display(),
        "OG card images generated"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::image::BinaryPriority;
    use crate::nvidia::ComputeCapability;
    use crate::scan::{BinaryScanResult, EnvironmentInfo, ImageMetadata, ScanResult};

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
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
    fn render_scan_card_produces_png() {
        let scan = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.5.0");
        let png = crate::og::render_scan_card(&scan).unwrap();
        // PNG magic bytes
        assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert!(png.len() > 1000, "PNG should have meaningful size");
    }

    #[test]
    fn render_diff_card_produces_png() {
        let from = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.4.0");
        let to = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.5.0");
        let diff = crate::diff::compute_diff(&from, &to);
        let png = crate::og::render_diff_card(&diff).unwrap();
        assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn render_index_card_produces_png() {
        let config = crate::site::SiteConfig {
            title: "Test Catalog".to_string(),
            output_dir: "/tmp/test".to_string(),
            base_url: None,
            families: vec![],
        };
        let png = crate::og::render_index_card(&config).unwrap();
        assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn svg_escape_filter_handles_special_chars() {
        let tmpl = crate::templates::SVG_ENV.get_template("scan").unwrap();
        // Render with special chars and verify they're escaped in the output
        let svg = tmpl
            .render(minijinja::context! {
                image => "a & b",
                details => "x < y > z",
                binaries => "ok",
            })
            .unwrap();
        assert!(svg.contains("a &amp; b"));
        assert!(svg.contains("x &lt; y &gt; z"));
    }

    #[test]
    fn scan_og_meta_fields() {
        let scan = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.5.0");
        let meta = crate::og::scan_og_meta(&scan);
        assert_eq!(meta.title, "ghcr.io/llm-d/llm-d-cuda:v0.5.0");
        assert!(meta.description.contains("CUDA 12.4"));
        assert!(meta.image_path.starts_with("/og/scan/"));
        assert!(meta.image_path.ends_with(".png"));
    }

    #[test]
    fn diff_og_meta_fields() {
        let from = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.4.0");
        let to = test_scan("ghcr.io/llm-d/llm-d-cuda:v0.5.0");
        let diff = crate::diff::compute_diff(&from, &to);
        let meta = crate::og::diff_og_meta(&diff);
        assert!(meta.title.contains("\u{2192}"));
        assert!(meta.image_path.contains("-vs-"));
    }
}
