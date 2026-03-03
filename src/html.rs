use std::collections::HashMap;

use crate::diff::OgContext;
use crate::scan::ScanResult;
use crate::style::{self, BundleMode};

const TEMPLATE: &str = include_str!("templates/html_template.html");

/// Compact binary entry for the HTML template (no DT_NEEDED, that's in the graph).
#[derive(serde::Serialize)]
struct CompactBinary {
    path: String,
    priority: crate::image::BinaryPriority,
    size: u64,
    reachable: bool,
    cubins: Vec<crate::nvidia::ComputeCapability>,
    ptx: Vec<crate::nvidia::ComputeCapability>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    rpath: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    runpath: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    package: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer_index: Option<usize>,
}

/// Top-level data structure injected into the HTML template as JSON.
#[derive(serde::Serialize)]
struct HtmlData<'a> {
    image: &'a str,
    metadata: &'a crate::scan::ImageMetadata,
    effective_cc_min: Option<String>,
    effective_cc_max: Option<String>,
    has_ptx: bool,
    reachable_count: usize,
    dormant_count: usize,
    warnings: &'a [String],
    sm_min: u32,
    sm_max: u32,
    graph: &'a crate::scan::DepGraph,
    /// All scanned binaries with reachability status (no DT_NEEDED, that's in graph).
    binaries: Vec<CompactBinary>,
    /// Environment metadata (OS, packages, symlinks, ld.so.conf paths).
    environment: &'a crate::scan::EnvironmentInfo,
    /// OCI image labels.
    labels: &'a std::collections::HashMap<String, String>,
    /// All environment variables from the image config.
    env_vars: &'a [(String, String)],
    /// Build history: layer index to Dockerfile command mapping.
    layer_history: &'a [crate::scan::LayerInfo],
}

/// Render the scan result as an interactive HTML page.
///
/// `bundle` controls whether JS dependencies are inlined (`SelfContained`)
/// or loaded from a CDN (`Cdn`).
pub fn render_html(result: &ScanResult, bundle: BundleMode) -> String {
    render_html_with_og(result, bundle, None, "", "")
}

pub fn render_html_with_og(
    result: &ScanResult,
    bundle: BundleMode,
    og: Option<&OgContext>,
    favicon_href: &str,
    nav_html: &str,
) -> String {
    let reachable_paths: std::collections::HashSet<&str> = result
        .dep_graph
        .as_ref()
        .map(|g| g.nodes.values().map(|n| n.path.as_str()).collect())
        .unwrap_or_default();

    // Build compact binary list: dedup by path (last wins, like container layers),
    // strip DT_NEEDED (already in the graph), add reachability flag.
    let mut path_seen = HashMap::new();
    for (i, b) in result.binaries.iter().enumerate() {
        path_seen.insert(b.path.as_str(), i);
    }
    let mut binaries: Vec<CompactBinary> = path_seen
        .values()
        .map(|&i| {
            let b = &result.binaries[i];
            CompactBinary {
                path: b.path.clone(),
                priority: b.priority,
                size: b.size,
                reachable: reachable_paths.contains(b.path.as_str()),
                cubins: b.cubins.clone(),
                ptx: b.ptx.clone(),
                rpath: b.rpath.clone(),
                runpath: b.runpath.clone(),
                package: result.environment.file_owners.get(&b.path).cloned(),
                layer_index: b.layer_index,
            }
        })
        .collect();
    binaries.sort_by(|a, b| a.path.cmp(&b.path));

    // Compute global SM range across all binaries for bar alignment
    let (sm_min, sm_max) = compute_sm_range(&result.binaries);

    let empty_graph = crate::scan::DepGraph {
        roots: vec![],
        nodes: HashMap::new(),
    };

    let data = HtmlData {
        image: &result.image,
        metadata: &result.metadata,
        effective_cc_min: result
            .effective_cc_min
            .map(|cc| format!("{}", cc.major * 10 + cc.minor)),
        effective_cc_max: result
            .effective_cc_max
            .map(|cc| format!("{}", cc.major * 10 + cc.minor)),
        has_ptx: result.has_ptx_forward_compat,
        reachable_count: result.reachable_count,
        dormant_count: result.dormant_count,
        warnings: &result.warnings,
        sm_min,
        sm_max,
        graph: result.dep_graph.as_ref().unwrap_or(&empty_graph),
        binaries,
        environment: &result.environment,
        labels: &result.labels,
        env_vars: &result.env_vars,
        layer_history: &result.layer_history,
    };

    let json = serde_json::to_string(&data).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));

    let mut env = minijinja::Environment::new();
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::Html);
    env.add_template("html", TEMPLATE)
        .expect("HTML template is valid");
    let tmpl = env.get_template("html").expect("html template registered");
    tmpl.render(minijinja::context! {
        base_css => style::BASE_CSS,
        scripts => style::script_block(bundle),
        graph_data => json,
        favicon_href => if favicon_href.is_empty() { "" } else { favicon_href },
        og => og.map(|o| minijinja::context! {
            title => o.title.clone(),
            description => o.description.clone(),
            image_url => o.image_url.clone(),
            og_type => o.og_type.clone(),
        }),
        nav_html => nav_html,
    })
    .expect("html template renders")
}

/// Compute the global SM range (min, max) across all binaries for consistent bar rendering.
fn compute_sm_range(binaries: &[crate::scan::BinaryScanResult]) -> (u32, u32) {
    let mut min = u32::MAX;
    let mut max = 0u32;

    for b in binaries {
        for cc in b.cubins.iter().chain(b.ptx.iter()) {
            let sm = cc.major * 10 + cc.minor;
            if sm < min {
                min = sm;
            }
            if sm > max {
                max = sm;
            }
        }
    }

    if min > max {
        (0, 0)
    } else {
        // Snap to multiples of 5 for cleaner alignment
        let min_snapped = (min / 5) * 5;
        let max_snapped = max.div_ceil(5) * 5;
        (min_snapped, max_snapped)
    }
}

#[cfg(test)]
mod tests {
    use crate::nvidia::ComputeCapability;
    use crate::scan::{
        BinaryScanResult, DepGraph, DepNode, EnvironmentInfo, ImageMetadata, ScanResult,
    };
    use std::collections::HashMap;

    fn empty_env() -> EnvironmentInfo {
        EnvironmentInfo {
            os: None,
            python_environments: vec![],
            system_packages: vec![],
            ld_conf_paths: vec![],
            symlinks: HashMap::new(),
            file_owners: HashMap::new(),
        }
    }

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
    }

    fn test_result() -> ScanResult {
        let mut nodes = HashMap::new();
        nodes.insert(
            "libcuda.so".to_string(),
            DepNode {
                path: "/lib/libcuda.so".to_string(),
                soname: "libcuda.so".to_string(),
                priority: crate::image::BinaryPriority::LinkerLibrary,
                size: 1024,
                cubins: vec![cc(7, 0), cc(9, 0)],
                ptx: vec![],
                deps: vec![],
                unresolved: vec![],
                rpath: vec![],
                runpath: vec![],
                package: None,
                layer_index: None,
            },
        );
        ScanResult {
            image: "test:latest".to_string(),
            metadata: ImageMetadata {
                cuda_version: Some("12.9.1".to_string()),
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![BinaryScanResult {
                path: "/lib/libcuda.so".to_string(),
                priority: crate::image::BinaryPriority::LinkerLibrary,
                size: 1024,
                cubins: vec![cc(7, 0), cc(9, 0)],
                ptx: vec![],
                needed: vec![],
                soname: None,
                rpath: vec![],
                runpath: vec![],
                layer_index: None,
            }],
            effective_cc_min: Some(cc(7, 0)),
            effective_cc_max: Some(cc(9, 0)),
            has_ptx_forward_compat: false,
            warnings: vec![],
            dep_graph: Some(DepGraph {
                roots: vec!["libcuda.so".to_string()],
                nodes,
            }),
            reachable_count: 1,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
            layer_history: vec![],
        }
    }

    #[test]
    fn html_contains_doctype() {
        let html = super::render_html(&test_result(), crate::style::BundleMode::SelfContained);
        assert!(html.starts_with("<!DOCTYPE html>"));
    }

    #[test]
    fn html_replaces_placeholder() {
        let html = super::render_html(&test_result(), crate::style::BundleMode::SelfContained);
        assert!(
            !html.contains("/*GRAPH_DATA*/null"),
            "placeholder should be replaced with actual data"
        );
    }

    #[test]
    fn html_contains_graph_data() {
        let html = super::render_html(&test_result(), crate::style::BundleMode::SelfContained);
        assert!(html.contains("libcuda.so"));
        assert!(html.contains("test:latest"));
    }

    #[test]
    fn html_empty_result() {
        let result = ScanResult {
            image: "empty:latest".to_string(),
            metadata: ImageMetadata {
                cuda_version: None,
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![],
            effective_cc_min: None,
            effective_cc_max: None,
            has_ptx_forward_compat: false,
            warnings: vec![],
            dep_graph: None,
            reachable_count: 0,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
            layer_history: vec![],
        };
        let html = super::render_html(&result, crate::style::BundleMode::SelfContained);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("empty:latest"));
    }

    #[test]
    fn sm_range_computation() {
        let binaries = vec![
            BinaryScanResult {
                path: "/a.so".to_string(),
                priority: crate::image::BinaryPriority::LinkerLibrary,
                size: 0,
                cubins: vec![cc(7, 0), cc(9, 0)],
                ptx: vec![cc(10, 0)],
                needed: vec![],
                soname: None,
                rpath: vec![],
                runpath: vec![],
                layer_index: None,
            },
            BinaryScanResult {
                path: "/b.so".to_string(),
                priority: crate::image::BinaryPriority::LinkerLibrary,
                size: 0,
                cubins: vec![cc(5, 0)],
                ptx: vec![],
                needed: vec![],
                soname: None,
                rpath: vec![],
                runpath: vec![],
                layer_index: None,
            },
        ];
        let (min, max) = super::compute_sm_range(&binaries);
        assert_eq!(min, 50);
        assert_eq!(max, 100);
    }

    #[test]
    fn sm_range_empty() {
        let (min, max) = super::compute_sm_range(&[]);
        assert_eq!(min, 0);
        assert_eq!(max, 0);
    }

    #[test]
    fn html_contains_labels_and_env_vars() {
        let mut result = ScanResult {
            image: "labeled:latest".to_string(),
            metadata: ImageMetadata {
                cuda_version: None,
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![],
            effective_cc_min: None,
            effective_cc_max: None,
            has_ptx_forward_compat: false,
            warnings: vec![],
            dep_graph: None,
            reachable_count: 0,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
            layer_history: vec![],
        };
        result
            .labels
            .insert("maintainer".to_string(), "test-user".to_string());
        result.labels.insert(
            "org.opencontainers.image.version".to_string(),
            "1.0".to_string(),
        );
        result
            .env_vars
            .push(("CUDA_VERSION".to_string(), "12.9.1".to_string()));

        let html = super::render_html(&result, crate::style::BundleMode::SelfContained);

        assert!(
            html.contains("maintainer"),
            "HTML should contain the label key 'maintainer'"
        );
        assert!(
            html.contains("test-user"),
            "HTML should contain the label value 'test-user'"
        );
        assert!(
            html.contains("org.opencontainers.image.version"),
            "HTML should contain the OCI label key"
        );
        assert!(
            html.contains("CUDA_VERSION"),
            "HTML should contain the env var key"
        );
        assert!(
            html.contains("12.9.1"),
            "HTML should contain the env var value"
        );
    }

    #[test]
    fn self_contained_inlines_cytoscape() {
        let html = super::render_html(&test_result(), crate::style::BundleMode::SelfContained);
        assert!(
            html.contains("<script>"),
            "self-contained report should have inline <script> tags"
        );
        assert!(
            !html.contains("unpkg.com"),
            "self-contained report should not reference CDN"
        );
    }

    #[test]
    fn cdn_uses_script_src_tags() {
        let html = super::render_html(&test_result(), crate::style::BundleMode::Cdn);
        assert!(
            html.contains(r#"<script src="https://unpkg.com/cytoscape@3.30.4"#),
            "CDN report should contain script src tags"
        );
        assert!(
            !html.contains("<!--SCRIPTS-->"),
            "placeholder should be replaced"
        );
    }
}
