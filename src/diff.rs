use std::collections::HashMap;

use crate::nvidia::ComputeCapability;

use crate::scan::{BinaryScanResult, PackageVersion, PythonEnvironment, ScanResult};
use crate::templates;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    bound(deserialize = "T: serde::de::DeserializeOwned")
)]
pub enum ChangeStatus<T: serde::Serialize> {
    Added { value: T },
    Removed { value: T },
    Changed { before: T, after: T },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinarySmStatus {
    Added,
    Removed,
    Changed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BinarySmDiff {
    pub path: String,
    pub status: BinarySmStatus,
    pub cubins_before: Vec<ComputeCapability>,
    pub cubins_after: Vec<ComputeCapability>,
    pub ptx_before: Vec<ComputeCapability>,
    pub ptx_after: Vec<ComputeCapability>,
    /// True when this is half of an added/removed pair that share the same
    /// filename and identical SM profile (just a versioned directory rename).
    pub renamed: bool,
    /// System package that owns this file, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    /// Layer index of this binary in the "from" image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_before: Option<usize>,
    /// Layer index of this binary in the "to" image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_after: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackageDiff {
    pub name: String,
    pub change: ChangeStatus<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PythonEnvDiff {
    pub label: String,
    pub packages: Vec<PackageDiff>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffSummary {
    pub binaries_added: usize,
    pub binaries_removed: usize,
    pub binaries_changed: usize,
    pub binaries_renamed: usize,
    pub python_packages_added: usize,
    pub python_packages_removed: usize,
    pub python_packages_changed: usize,
    pub system_packages_added: usize,
    pub system_packages_removed: usize,
    pub system_packages_changed: usize,
    pub labels_added: usize,
    pub labels_removed: usize,
    pub labels_changed: usize,
    pub env_vars_added: usize,
    pub env_vars_removed: usize,
    pub env_vars_changed: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LabelDiff {
    pub key: String,
    pub change: ChangeStatus<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvVarDiff {
    pub key: String,
    pub change: ChangeStatus<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffResult {
    pub from: String,
    pub to: String,
    pub binary_diffs: Vec<BinarySmDiff>,
    pub python_diffs: Vec<PythonEnvDiff>,
    pub system_package_diffs: Vec<PackageDiff>,
    pub label_diffs: Vec<LabelDiff>,
    pub env_var_diffs: Vec<EnvVarDiff>,
    pub summary: DiffSummary,
}

// ---------------------------------------------------------------------------
// Diff computation
// ---------------------------------------------------------------------------

type BinaryIndex<'a> = (
    &'a [ComputeCapability],
    &'a [ComputeCapability],
    Option<usize>,
);

/// Dedup binaries by path (last wins, matching container layer semantics).
fn index_binaries_by_path(result: &ScanResult) -> HashMap<&str, BinaryIndex<'_>> {
    let mut map = HashMap::new();
    for b in &result.binaries {
        map.insert(
            b.path.as_str(),
            (b.cubins.as_slice(), b.ptx.as_slice(), b.layer_index),
        );
    }
    map
}

/// Diff two package lists, returning only entries that changed.
fn diff_packages(from: &[PackageVersion], to: &[PackageVersion]) -> Vec<PackageDiff> {
    let from_map: HashMap<&str, &str> = from
        .iter()
        .map(|p| (p.name.as_str(), p.version.as_str()))
        .collect();
    let to_map: HashMap<&str, &str> = to
        .iter()
        .map(|p| (p.name.as_str(), p.version.as_str()))
        .collect();

    let mut all_names: Vec<&str> = from_map.keys().chain(to_map.keys()).copied().collect();
    all_names.sort_unstable();
    all_names.dedup();

    let mut diffs = Vec::new();
    for name in all_names {
        match (from_map.get(name), to_map.get(name)) {
            (Some(_), None) => diffs.push(PackageDiff {
                name: name.to_string(),
                change: ChangeStatus::Removed {
                    value: from_map[name].to_string(),
                },
            }),
            (None, Some(_)) => diffs.push(PackageDiff {
                name: name.to_string(),
                change: ChangeStatus::Added {
                    value: to_map[name].to_string(),
                },
            }),
            (Some(v_from), Some(v_to)) => {
                if v_from != v_to {
                    diffs.push(PackageDiff {
                        name: name.to_string(),
                        change: ChangeStatus::Changed {
                            before: v_from.to_string(),
                            after: v_to.to_string(),
                        },
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }
    diffs
}

fn diff_python_envs(from: &[PythonEnvironment], to: &[PythonEnvironment]) -> Vec<PythonEnvDiff> {
    let from_map: HashMap<&str, &PythonEnvironment> =
        from.iter().map(|e| (e.label.as_str(), e)).collect();
    let to_map: HashMap<&str, &PythonEnvironment> =
        to.iter().map(|e| (e.label.as_str(), e)).collect();

    let mut all_labels: Vec<&str> = from_map.keys().chain(to_map.keys()).copied().collect();
    all_labels.sort_unstable();
    all_labels.dedup();

    let mut diffs = Vec::new();
    for label in all_labels {
        match (from_map.get(label), to_map.get(label)) {
            (Some(env_from), None) => {
                // Entire env removed: every package is "removed"
                let packages = env_from
                    .packages
                    .iter()
                    .map(|p| PackageDiff {
                        name: p.name.clone(),
                        change: ChangeStatus::Removed {
                            value: p.version.clone(),
                        },
                    })
                    .collect();
                diffs.push(PythonEnvDiff {
                    label: label.to_string(),
                    packages,
                });
            }
            (None, Some(env_to)) => {
                // Entire env added: every package is "added"
                let packages = env_to
                    .packages
                    .iter()
                    .map(|p| PackageDiff {
                        name: p.name.clone(),
                        change: ChangeStatus::Added {
                            value: p.version.clone(),
                        },
                    })
                    .collect();
                diffs.push(PythonEnvDiff {
                    label: label.to_string(),
                    packages,
                });
            }
            (Some(env_from), Some(env_to)) => {
                let packages = diff_packages(&env_from.packages, &env_to.packages);
                if !packages.is_empty() {
                    diffs.push(PythonEnvDiff {
                        label: label.to_string(),
                        packages,
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }
    diffs
}

/// Effective identity for a binary: DT_SONAME if set, otherwise the filename.
fn binary_identity(b: &BinaryScanResult) -> &str {
    b.soname
        .as_deref()
        .unwrap_or_else(|| b.path.rsplit('/').next().unwrap_or(&b.path))
}

/// Build a map from path to binary identity for rename detection.
fn build_identity_map(result: &ScanResult) -> HashMap<&str, &str> {
    let mut map = HashMap::new();
    for b in &result.binaries {
        map.insert(b.path.as_str(), binary_identity(b));
    }
    map
}

/// Match added/removed pairs that share the same DT_SONAME (or filename) and
/// identical SM profiles. These are versioned directory moves, not real changes.
fn detect_renames(
    diffs: &mut [BinarySmDiff],
    from_ids: &HashMap<&str, &str>,
    to_ids: &HashMap<&str, &str>,
) {
    type SmKey<'a> = (&'a str, &'a [ComputeCapability], &'a [ComputeCapability]);
    let mut removed_by_key: HashMap<SmKey<'_>, Vec<usize>> = HashMap::new();
    for (i, d) in diffs.iter().enumerate() {
        if d.status == BinarySmStatus::Removed {
            if let Some(&identity) = from_ids.get(d.path.as_str()) {
                removed_by_key
                    .entry((identity, &d.cubins_before, &d.ptx_before))
                    .or_default()
                    .push(i);
            }
        }
    }

    // For each added entry, try to find a matching removed entry
    let mut rename_pairs: Vec<(usize, usize)> = Vec::new();
    for (i, d) in diffs.iter().enumerate() {
        if d.status == BinarySmStatus::Added {
            if let Some(&identity) = to_ids.get(d.path.as_str()) {
                let key = (identity, d.cubins_after.as_slice(), d.ptx_after.as_slice());
                if let Some(removed_indices) = removed_by_key.get_mut(&key) {
                    if let Some(removed_idx) = removed_indices.pop() {
                        rename_pairs.push((removed_idx, i));
                    }
                }
            }
        }
    }

    for (removed_idx, added_idx) in rename_pairs {
        diffs[removed_idx].renamed = true;
        diffs[added_idx].renamed = true;
    }
}

/// Diff two string maps, returning entries that were added, removed, or changed.
fn diff_string_map(
    from: &HashMap<String, String>,
    to: &HashMap<String, String>,
) -> Vec<(String, ChangeStatus<String>)> {
    let mut all_keys: Vec<&str> = from.keys().chain(to.keys()).map(|s| s.as_str()).collect();
    all_keys.sort_unstable();
    all_keys.dedup();

    let mut diffs = Vec::new();
    for key in all_keys {
        match (from.get(key), to.get(key)) {
            (Some(v), None) => {
                diffs.push((key.to_string(), ChangeStatus::Removed { value: v.clone() }));
            }
            (None, Some(v)) => {
                diffs.push((key.to_string(), ChangeStatus::Added { value: v.clone() }));
            }
            (Some(v_from), Some(v_to)) => {
                if v_from != v_to {
                    diffs.push((
                        key.to_string(),
                        ChangeStatus::Changed {
                            before: v_from.clone(),
                            after: v_to.clone(),
                        },
                    ));
                }
            }
            (None, None) => unreachable!(),
        }
    }
    diffs
}

pub fn compute_diff(from: &ScanResult, to: &ScanResult) -> DiffResult {
    // Binary SM diffs
    let from_bins = index_binaries_by_path(from);
    let to_bins = index_binaries_by_path(to);

    let mut all_paths: Vec<&str> = from_bins.keys().chain(to_bins.keys()).copied().collect();
    all_paths.sort_unstable();
    all_paths.dedup();

    let mut binary_diffs = Vec::new();
    for path in &all_paths {
        match (from_bins.get(path), to_bins.get(path)) {
            (Some(_), None) => {
                let (cubins, ptx, layer_idx) = from_bins[path];
                if !cubins.is_empty() || !ptx.is_empty() {
                    binary_diffs.push(BinarySmDiff {
                        path: path.to_string(),
                        status: BinarySmStatus::Removed,
                        cubins_before: cubins.to_vec(),
                        cubins_after: vec![],
                        ptx_before: ptx.to_vec(),
                        ptx_after: vec![],
                        renamed: false,
                        package: from.environment.file_owners.get(*path).cloned(),
                        layer_before: layer_idx,
                        layer_after: None,
                    });
                }
            }
            (None, Some(_)) => {
                let (cubins, ptx, layer_idx) = to_bins[path];
                if !cubins.is_empty() || !ptx.is_empty() {
                    binary_diffs.push(BinarySmDiff {
                        path: path.to_string(),
                        status: BinarySmStatus::Added,
                        cubins_before: vec![],
                        cubins_after: cubins.to_vec(),
                        ptx_before: vec![],
                        ptx_after: ptx.to_vec(),
                        renamed: false,
                        package: to.environment.file_owners.get(*path).cloned(),
                        layer_before: None,
                        layer_after: layer_idx,
                    });
                }
            }
            (Some((fc, fp, fl)), Some((tc, tp, tl))) => {
                let cubins_changed = fc != tc;
                let ptx_changed = fp != tp;
                if cubins_changed || ptx_changed {
                    binary_diffs.push(BinarySmDiff {
                        path: path.to_string(),
                        status: BinarySmStatus::Changed,
                        cubins_before: fc.to_vec(),
                        cubins_after: tc.to_vec(),
                        ptx_before: fp.to_vec(),
                        ptx_after: tp.to_vec(),
                        renamed: false,
                        package: to.environment.file_owners.get(*path).cloned(),
                        layer_before: *fl,
                        layer_after: *tl,
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }

    // Detect renames: added/removed pairs with the same DT_SONAME (or filename)
    // and identical SM profiles. These are versioned directory moves, not real changes.
    let from_ids = build_identity_map(from);
    let to_ids = build_identity_map(to);
    detect_renames(&mut binary_diffs, &from_ids, &to_ids);

    // Package diffs
    let python_diffs = diff_python_envs(
        &from.environment.python_environments,
        &to.environment.python_environments,
    );
    let system_package_diffs = diff_packages(
        &from.environment.system_packages,
        &to.environment.system_packages,
    );

    // Summary counts (exclude renames from added/removed tallies)
    let binaries_added = binary_diffs
        .iter()
        .filter(|d| d.status == BinarySmStatus::Added && !d.renamed)
        .count();
    let binaries_removed = binary_diffs
        .iter()
        .filter(|d| d.status == BinarySmStatus::Removed && !d.renamed)
        .count();
    let binaries_changed = binary_diffs
        .iter()
        .filter(|d| d.status == BinarySmStatus::Changed)
        .count();
    let binaries_renamed = binary_diffs.iter().filter(|d| d.renamed).count();

    let (mut py_added, mut py_removed, mut py_changed) = (0, 0, 0);
    for env in &python_diffs {
        for pkg in &env.packages {
            match &pkg.change {
                ChangeStatus::Added { .. } => py_added += 1,
                ChangeStatus::Removed { .. } => py_removed += 1,
                ChangeStatus::Changed { .. } => py_changed += 1,
            }
        }
    }

    let (mut sys_added, mut sys_removed, mut sys_changed) = (0, 0, 0);
    for pkg in &system_package_diffs {
        match &pkg.change {
            ChangeStatus::Added { .. } => sys_added += 1,
            ChangeStatus::Removed { .. } => sys_removed += 1,
            ChangeStatus::Changed { .. } => sys_changed += 1,
        }
    }

    // Label and env var diffs
    let label_diff_pairs = diff_string_map(&from.labels, &to.labels);
    let label_diffs: Vec<LabelDiff> = label_diff_pairs
        .into_iter()
        .map(|(key, change)| LabelDiff { key, change })
        .collect();

    let from_env_map: HashMap<String, String> = from.env_vars.iter().cloned().collect();
    let to_env_map: HashMap<String, String> = to.env_vars.iter().cloned().collect();
    let env_var_diff_pairs = diff_string_map(&from_env_map, &to_env_map);
    let env_var_diffs: Vec<EnvVarDiff> = env_var_diff_pairs
        .into_iter()
        .map(|(key, change)| EnvVarDiff { key, change })
        .collect();

    let (mut lbl_added, mut lbl_removed, mut lbl_changed) = (0, 0, 0);
    for d in &label_diffs {
        match &d.change {
            ChangeStatus::Added { .. } => lbl_added += 1,
            ChangeStatus::Removed { .. } => lbl_removed += 1,
            ChangeStatus::Changed { .. } => lbl_changed += 1,
        }
    }
    let (mut env_added, mut env_removed, mut env_changed) = (0, 0, 0);
    for d in &env_var_diffs {
        match &d.change {
            ChangeStatus::Added { .. } => env_added += 1,
            ChangeStatus::Removed { .. } => env_removed += 1,
            ChangeStatus::Changed { .. } => env_changed += 1,
        }
    }

    DiffResult {
        from: from.image.clone(),
        to: to.image.clone(),
        binary_diffs,
        python_diffs,
        system_package_diffs,
        label_diffs,
        env_var_diffs,
        summary: DiffSummary {
            binaries_added,
            binaries_removed,
            binaries_changed,
            binaries_renamed,
            python_packages_added: py_added,
            python_packages_removed: py_removed,
            python_packages_changed: py_changed,
            system_packages_added: sys_added,
            system_packages_removed: sys_removed,
            system_packages_changed: sys_changed,
            labels_added: lbl_added,
            labels_removed: lbl_removed,
            labels_changed: lbl_changed,
            env_vars_added: env_added,
            env_vars_removed: env_removed,
            env_vars_changed: env_changed,
        },
    }
}

// ---------------------------------------------------------------------------
// HTML rendering
// ---------------------------------------------------------------------------

/// Context for OG meta tag injection.
pub struct OgContext {
    pub title: String,
    pub description: String,
    pub image_url: String,
    pub og_type: String,
}

pub fn render_diff_html(diff: &DiffResult) -> String {
    render_diff_html_with_og(diff, None, "", "")
}

pub fn render_diff_html_with_og(
    diff: &DiffResult,
    og: Option<&OgContext>,
    favicon_href: &str,
    nav_html: &str,
) -> String {
    let json = serde_json::to_string(diff).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
    let tmpl = templates::HTML_ENV
        .get_template("diff")
        .expect("diff template registered");
    tmpl.render(minijinja::context! {
        base_css => templates::BASE_CSS,
        diff_data => json,
        favicon_href => if favicon_href.is_empty() { "" } else { favicon_href },
        og => og.map(|o| minijinja::context! {
            title => o.title.clone(),
            description => o.description.clone(),
            image_url => o.image_url.clone(),
            og_type => o.og_type.clone(),
        }),
        nav_html => nav_html,
    })
    .expect("diff template renders")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::nvidia::ComputeCapability;

    use crate::image::BinaryPriority;
    use crate::scan::{
        BinaryScanResult, EnvironmentInfo, ImageMetadata, PackageVersion, PythonEnvironment,
        ScanResult,
    };

    use crate::diff::{compute_diff, render_diff_html, BinarySmStatus, ChangeStatus};

    fn cc(major: u32, minor: u32) -> ComputeCapability {
        ComputeCapability::new(major, minor)
    }

    fn empty_metadata() -> ImageMetadata {
        ImageMetadata {
            cuda_version: None,
            torch_arch_list: None,
            nvidia_require: None,
            nvshmem_architectures: None,
        }
    }

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

    fn minimal_scan(image: &str) -> ScanResult {
        ScanResult {
            image: image.to_string(),
            metadata: empty_metadata(),
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
        }
    }

    fn make_binary(
        path: &str,
        cubins: Vec<ComputeCapability>,
        ptx: Vec<ComputeCapability>,
    ) -> BinaryScanResult {
        BinaryScanResult {
            path: path.to_string(),
            priority: BinaryPriority::LooseGpuFile,
            size: 1024,
            cubins,
            ptx,
            needed: vec![],
            soname: None,
            rpath: vec![],
            runpath: vec![],
            layer_index: None,
            crypto: Default::default(),
        }
    }

    fn make_binary_with_soname(
        path: &str,
        soname: &str,
        cubins: Vec<ComputeCapability>,
        ptx: Vec<ComputeCapability>,
    ) -> BinaryScanResult {
        let mut b = make_binary(path, cubins, ptx);
        b.soname = Some(soname.to_string());
        b
    }

    fn make_package(name: &str, version: &str) -> PackageVersion {
        PackageVersion {
            name: name.to_string(),
            version: version.to_string(),
            source: None,
        }
    }

    #[test]
    fn empty_inputs_produce_empty_diff() {
        let from = minimal_scan("a:v1");
        let to = minimal_scan("b:v2");
        let diff = compute_diff(&from, &to);

        assert!(diff.binary_diffs.is_empty());
        assert!(diff.python_diffs.is_empty());
        assert!(diff.system_package_diffs.is_empty());
        assert_eq!(diff.summary.binaries_added, 0);
    }

    #[test]
    fn binary_added() {
        let from = minimal_scan("a:v1");
        let mut to = minimal_scan("b:v2");
        to.binaries
            .push(make_binary("/lib/new.so", vec![cc(7, 0), cc(9, 0)], vec![]));

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.binary_diffs.len(), 1);
        assert_eq!(diff.binary_diffs[0].status, BinarySmStatus::Added);
        assert_eq!(diff.binary_diffs[0].cubins_after, vec![cc(7, 0), cc(9, 0)]);
        assert!(diff.binary_diffs[0].cubins_before.is_empty());
        assert_eq!(diff.summary.binaries_added, 1);
    }

    #[test]
    fn binary_removed() {
        let mut from = minimal_scan("a:v1");
        from.binaries
            .push(make_binary("/lib/old.so", vec![cc(7, 0)], vec![cc(8, 0)]));
        let to = minimal_scan("b:v2");

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.binary_diffs.len(), 1);
        assert_eq!(diff.binary_diffs[0].status, BinarySmStatus::Removed);
        assert_eq!(diff.summary.binaries_removed, 1);
    }

    #[test]
    fn binary_sm_changed() {
        let mut from = minimal_scan("a:v1");
        from.binaries
            .push(make_binary("/lib/lib.so", vec![cc(7, 0)], vec![]));
        let mut to = minimal_scan("b:v2");
        to.binaries
            .push(make_binary("/lib/lib.so", vec![cc(7, 0), cc(9, 0)], vec![]));

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.binary_diffs.len(), 1);
        assert_eq!(diff.binary_diffs[0].status, BinarySmStatus::Changed);
        assert_eq!(diff.summary.binaries_changed, 1);
    }

    #[test]
    fn binary_unchanged_not_emitted() {
        let mut from = minimal_scan("a:v1");
        from.binaries
            .push(make_binary("/lib/same.so", vec![cc(7, 0)], vec![]));
        let mut to = minimal_scan("b:v2");
        to.binaries
            .push(make_binary("/lib/same.so", vec![cc(7, 0)], vec![]));

        let diff = compute_diff(&from, &to);
        assert!(diff.binary_diffs.is_empty());
    }

    #[test]
    fn binary_without_cuda_not_emitted_on_add_remove() {
        let mut from = minimal_scan("a:v1");
        from.binaries.push(make_binary("/bin/tool", vec![], vec![]));
        let to = minimal_scan("b:v2");

        let diff = compute_diff(&from, &to);
        // Binary with no CUDA content shouldn't appear in diff
        assert!(diff.binary_diffs.is_empty());
    }

    #[test]
    fn system_package_changes() {
        let mut from = minimal_scan("a:v1");
        from.environment.system_packages = vec![
            make_package("curl", "7.80.0"),
            make_package("removed-pkg", "1.0"),
            make_package("openssl", "1.1.1"),
        ];
        let mut to = minimal_scan("b:v2");
        to.environment.system_packages = vec![
            make_package("curl", "7.80.0"),   // unchanged
            make_package("added-pkg", "2.0"), // added
            make_package("openssl", "3.0.0"), // changed
        ];

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.system_package_diffs.len(), 3);
        assert_eq!(diff.summary.system_packages_added, 1);
        assert_eq!(diff.summary.system_packages_removed, 1);
        assert_eq!(diff.summary.system_packages_changed, 1);

        // Verify specific changes
        let added = diff
            .system_package_diffs
            .iter()
            .find(|p| p.name == "added-pkg");
        assert!(
            matches!(added.map(|p| &p.change), Some(ChangeStatus::Added { value }) if value == "2.0")
        );

        let removed = diff
            .system_package_diffs
            .iter()
            .find(|p| p.name == "removed-pkg");
        assert!(
            matches!(removed.map(|p| &p.change), Some(ChangeStatus::Removed { value }) if value == "1.0")
        );

        let changed = diff
            .system_package_diffs
            .iter()
            .find(|p| p.name == "openssl");
        assert!(
            matches!(changed.map(|p| &p.change), Some(ChangeStatus::Changed { before, after }) if before == "1.1.1" && after == "3.0.0")
        );
    }

    #[test]
    fn python_env_added() {
        let from = minimal_scan("a:v1");
        let mut to = minimal_scan("b:v2");
        to.environment.python_environments.push(PythonEnvironment {
            label: "/opt/vllm".to_string(),
            site_packages_dir: "/opt/vllm/lib/python3.12/site-packages".to_string(),
            packages: vec![make_package("torch", "2.5.0")],
        });

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.python_diffs.len(), 1);
        assert_eq!(diff.python_diffs[0].label, "/opt/vllm");
        assert_eq!(diff.python_diffs[0].packages.len(), 1);
        assert!(
            matches!(&diff.python_diffs[0].packages[0].change, ChangeStatus::Added { value } if value == "2.5.0")
        );
    }

    #[test]
    fn python_env_matched_with_changes() {
        let mut from = minimal_scan("a:v1");
        from.environment
            .python_environments
            .push(PythonEnvironment {
                label: "system".to_string(),
                site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
                packages: vec![make_package("numpy", "1.24.0"), make_package("pip", "23.0")],
            });
        let mut to = minimal_scan("b:v2");
        to.environment.python_environments.push(PythonEnvironment {
            label: "system".to_string(),
            site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
            packages: vec![make_package("numpy", "1.26.0"), make_package("pip", "23.0")],
        });

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.python_diffs.len(), 1);
        // Only numpy changed, pip stayed the same
        assert_eq!(diff.python_diffs[0].packages.len(), 1);
        assert_eq!(diff.python_diffs[0].packages[0].name, "numpy");
    }

    #[test]
    fn python_env_no_changes_not_emitted() {
        let mut from = minimal_scan("a:v1");
        from.environment
            .python_environments
            .push(PythonEnvironment {
                label: "system".to_string(),
                site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
                packages: vec![make_package("numpy", "1.24.0")],
            });
        let mut to = minimal_scan("b:v2");
        to.environment.python_environments.push(PythonEnvironment {
            label: "system".to_string(),
            site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
            packages: vec![make_package("numpy", "1.24.0")],
        });

        let diff = compute_diff(&from, &to);
        assert!(diff.python_diffs.is_empty());
    }

    #[test]
    fn round_trip_scan_result_json() {
        let mut scan = minimal_scan("test:v1");
        scan.binaries
            .push(make_binary("/lib/test.so", vec![cc(7, 0)], vec![cc(9, 0)]));
        scan.environment
            .system_packages
            .push(make_package("curl", "7.80.0"));
        scan.environment
            .python_environments
            .push(PythonEnvironment {
                label: "system".to_string(),
                site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
                packages: vec![make_package("numpy", "1.24.0")],
            });

        let json = serde_json::to_string_pretty(&scan).expect("serialize");
        let deserialized: ScanResult = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.image, scan.image);
        assert_eq!(deserialized.binaries.len(), scan.binaries.len());
        assert_eq!(deserialized.binaries[0].cubins, scan.binaries[0].cubins);
        assert_eq!(deserialized.binaries[0].ptx, scan.binaries[0].ptx);
        assert_eq!(
            deserialized.environment.system_packages.len(),
            scan.environment.system_packages.len()
        );
        assert_eq!(
            deserialized.environment.python_environments.len(),
            scan.environment.python_environments.len()
        );
    }

    #[test]
    fn html_render_replaces_placeholder() {
        let from = minimal_scan("a:v1");
        let to = minimal_scan("b:v2");
        let diff = compute_diff(&from, &to);
        let html = render_diff_html(&diff);

        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(
            !html.contains("/*DIFF_DATA*/null"),
            "placeholder should be replaced"
        );
        assert!(html.contains("a:v1"));
        assert!(html.contains("b:v2"));
    }

    #[test]
    fn rename_detected_by_soname() {
        let mut from = minimal_scan("a:v1");
        from.binaries.push(make_binary_with_soname(
            "/opt/pkg-1.0/lib/libfoo.so",
            "libfoo.so",
            vec![cc(7, 0), cc(9, 0)],
            vec![],
        ));
        let mut to = minimal_scan("b:v2");
        to.binaries.push(make_binary_with_soname(
            "/opt/pkg-2.0/lib/libfoo.so",
            "libfoo.so",
            vec![cc(7, 0), cc(9, 0)],
            vec![],
        ));

        let diff = compute_diff(&from, &to);
        // Both entries should exist but be flagged as renames
        assert_eq!(diff.binary_diffs.len(), 2);
        assert!(diff.binary_diffs.iter().all(|d| d.renamed));
        assert_eq!(diff.summary.binaries_renamed, 2);
        // Renames don't count as real adds/removes
        assert_eq!(diff.summary.binaries_added, 0);
        assert_eq!(diff.summary.binaries_removed, 0);
    }

    #[test]
    fn rename_detected_by_filename_when_no_soname() {
        let mut from = minimal_scan("a:v1");
        from.binaries.push(make_binary(
            "/opt/pkg-1.0/ext.cpython-312.so",
            vec![cc(8, 0)],
            vec![],
        ));
        let mut to = minimal_scan("b:v2");
        to.binaries.push(make_binary(
            "/opt/pkg-2.0/ext.cpython-312.so",
            vec![cc(8, 0)],
            vec![],
        ));

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.binary_diffs.len(), 2);
        assert!(diff.binary_diffs.iter().all(|d| d.renamed));
    }

    #[test]
    fn no_rename_when_sm_profiles_differ() {
        let mut from = minimal_scan("a:v1");
        from.binaries.push(make_binary_with_soname(
            "/opt/pkg-1.0/lib/libfoo.so",
            "libfoo.so",
            vec![cc(7, 0)],
            vec![],
        ));
        let mut to = minimal_scan("b:v2");
        to.binaries.push(make_binary_with_soname(
            "/opt/pkg-2.0/lib/libfoo.so",
            "libfoo.so",
            vec![cc(7, 0), cc(9, 0)],
            vec![],
        ));

        let diff = compute_diff(&from, &to);
        assert_eq!(diff.binary_diffs.len(), 2);
        assert!(diff.binary_diffs.iter().all(|d| !d.renamed));
        assert_eq!(diff.summary.binaries_added, 1);
        assert_eq!(diff.summary.binaries_removed, 1);
    }

    #[test]
    fn html_render_contains_diff_data() {
        let mut from = minimal_scan("image:v1");
        from.binaries
            .push(make_binary("/lib/test.so", vec![cc(7, 0)], vec![]));
        let mut to = minimal_scan("image:v2");
        to.binaries.push(make_binary(
            "/lib/test.so",
            vec![cc(7, 0), cc(9, 0)],
            vec![],
        ));

        let diff = compute_diff(&from, &to);
        let html = render_diff_html(&diff);

        assert!(html.contains("/lib/test.so"));
        assert!(html.contains("changed"));
    }

    // -- diff_string_map --

    #[test]
    fn diff_string_map_all_changes() {
        let mut from = HashMap::new();
        from.insert("removed_key".to_string(), "old_val".to_string());
        from.insert("changed_key".to_string(), "before".to_string());
        from.insert("same_key".to_string(), "unchanged".to_string());

        let mut to = HashMap::new();
        to.insert("added_key".to_string(), "new_val".to_string());
        to.insert("changed_key".to_string(), "after".to_string());
        to.insert("same_key".to_string(), "unchanged".to_string());

        let diffs = super::diff_string_map(&from, &to);

        // Unchanged keys should not appear
        assert!(
            !diffs.iter().any(|(k, _)| k == "same_key"),
            "unchanged keys should not be in the diff"
        );

        // Check removed
        let removed = diffs.iter().find(|(k, _)| k == "removed_key").unwrap();
        assert!(matches!(
            &removed.1,
            ChangeStatus::Removed { value } if value == "old_val"
        ));

        // Check added
        let added = diffs.iter().find(|(k, _)| k == "added_key").unwrap();
        assert!(matches!(
            &added.1,
            ChangeStatus::Added { value } if value == "new_val"
        ));

        // Check changed
        let changed = diffs.iter().find(|(k, _)| k == "changed_key").unwrap();
        assert!(matches!(
            &changed.1,
            ChangeStatus::Changed { before, after } if before == "before" && after == "after"
        ));
    }

    // -- compute_diff with labels and env vars --

    #[test]
    fn compute_diff_with_labels_and_env_vars() {
        let mut from = minimal_scan("a:v1");
        from.labels
            .insert("maintainer".to_string(), "alice".to_string());
        from.labels
            .insert("removed_label".to_string(), "gone".to_string());
        from.env_vars
            .push(("PATH".to_string(), "/usr/bin".to_string()));
        from.env_vars
            .push(("OLD_VAR".to_string(), "old".to_string()));

        let mut to = minimal_scan("b:v2");
        to.labels
            .insert("maintainer".to_string(), "bob".to_string());
        to.labels
            .insert("new_label".to_string(), "fresh".to_string());
        to.env_vars
            .push(("PATH".to_string(), "/usr/local/bin:/usr/bin".to_string()));
        to.env_vars.push(("NEW_VAR".to_string(), "new".to_string()));

        let diff = compute_diff(&from, &to);

        // Label diffs: maintainer changed, removed_label removed, new_label added
        assert_eq!(diff.label_diffs.len(), 3);
        assert_eq!(diff.summary.labels_added, 1);
        assert_eq!(diff.summary.labels_removed, 1);
        assert_eq!(diff.summary.labels_changed, 1);

        let maintainer = diff
            .label_diffs
            .iter()
            .find(|d| d.key == "maintainer")
            .unwrap();
        assert!(matches!(
            &maintainer.change,
            ChangeStatus::Changed { before, after } if before == "alice" && after == "bob"
        ));

        // Env var diffs: PATH changed, OLD_VAR removed, NEW_VAR added
        assert_eq!(diff.env_var_diffs.len(), 3);
        assert_eq!(diff.summary.env_vars_added, 1);
        assert_eq!(diff.summary.env_vars_removed, 1);
        assert_eq!(diff.summary.env_vars_changed, 1);
    }

    // -- diff HTML with label diffs --

    #[test]
    fn diff_html_contains_label_diffs() {
        let mut from = minimal_scan("a:v1");
        from.labels
            .insert("maintainer".to_string(), "alice".to_string());

        let mut to = minimal_scan("b:v2");
        to.labels
            .insert("maintainer".to_string(), "bob".to_string());
        to.labels
            .insert("new_label".to_string(), "fresh".to_string());

        let diff = compute_diff(&from, &to);
        let html = render_diff_html(&diff);

        assert!(
            html.contains("maintainer"),
            "HTML should contain the label key"
        );
        assert!(
            html.contains("new_label"),
            "HTML should contain the added label key"
        );
    }
}
