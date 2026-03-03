use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::nvidia::ComputeCapability;
use anyhow::{Context, Result};
use oci_client::{Client, Reference};
use tokio::task::JoinSet;

use sha2::{Digest, Sha256};

use crate::cache::{BlobCache, CachedElfResult, CachedMetadata, LayerCache, ParseCache};
use crate::fatbin;
use crate::image::{self, BinaryPriority, ImageConfig};
use crate::progress::ScanProgress;

/// Try to resolve all binaries in a layer manifest from the parse cache.
///
/// Returns `Some((binaries, metadata))` if every classified binary has a parse
/// cache hit. Returns `None` on the first miss, signaling the caller to fall
/// back to full extraction.
fn try_resolve_from_manifest(
    layer_manifest: &crate::cache::LayerManifest,
    config: &ImageConfig,
    parse_cache: &ParseCache,
) -> Option<(Vec<BinaryScanResult>, Vec<image::DiscoveredMetadata>)> {
    let mut binaries = Vec::new();

    for entry in &layer_manifest.binaries {
        let priority = match image::classify_path(&entry.path, config) {
            Some(p) => p,
            None => continue,
        };

        let cached = parse_cache.get(&entry.content_sha256)?;

        binaries.push(BinaryScanResult {
            path: entry.path.clone(),
            priority,
            size: entry.size,
            cubins: cached.cubins,
            ptx: cached.ptx,
            needed: cached.needed,
            soname: cached.soname,
            rpath: cached.rpath,
            runpath: cached.runpath,
        });
    }

    let metadata = layer_manifest
        .metadata
        .iter()
        .map(|m| match m {
            CachedMetadata::Symlink { path, target } => image::DiscoveredMetadata::Symlink {
                path: path.clone(),
                target: target.clone(),
            },
            CachedMetadata::OsRelease(content) => {
                image::DiscoveredMetadata::OsRelease(content.clone())
            }
            CachedMetadata::LdSoConf(content) => {
                image::DiscoveredMetadata::LdSoConf(content.clone())
            }
            CachedMetadata::PythonPackage {
                name,
                version,
                site_packages_dir,
            } => image::DiscoveredMetadata::PythonPackage {
                name: name.clone(),
                version: version.clone(),
                site_packages_dir: site_packages_dir.clone(),
            },
            CachedMetadata::DpkgStatus(content) => {
                image::DiscoveredMetadata::DpkgStatus(content.clone())
            }
            CachedMetadata::DpkgFileList {
                package_name,
                content,
            } => image::DiscoveredMetadata::DpkgFileList {
                package_name: package_name.clone(),
                content: content.clone(),
            },
        })
        .collect();

    Some((binaries, metadata))
}

/// Image-level metadata extracted from env vars and labels.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageMetadata {
    pub cuda_version: Option<String>,
    pub torch_arch_list: Option<String>,
    pub nvidia_require: Option<String>,
    pub nvshmem_architectures: Option<String>,
}

/// Result of scanning a single binary file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BinaryScanResult {
    pub path: String,
    pub priority: BinaryPriority,
    /// File size in bytes.
    pub size: u64,
    /// Compiled SASS architectures (cubin).
    pub cubins: Vec<ComputeCapability>,
    /// PTX (JIT-compatible) architectures.
    pub ptx: Vec<ComputeCapability>,
    /// DT_NEEDED sonames from this binary's ELF dynamic section.
    pub needed: Vec<String>,
    /// DT_SONAME declared by this binary (libraries only, executables usually None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soname: Option<String>,
    /// DT_RPATH entries (deprecated, prefer DT_RUNPATH).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rpath: Vec<String>,
    /// DT_RUNPATH entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runpath: Vec<String>,
}

/// A node in the ELF dependency graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DepNode {
    /// Full path of the binary in the image.
    pub path: String,
    /// Soname (filename) used for dependency resolution.
    pub soname: String,
    /// Priority classification.
    pub priority: BinaryPriority,
    /// File size in bytes.
    pub size: u64,
    /// Compiled SASS architectures found in this binary.
    pub cubins: Vec<ComputeCapability>,
    /// PTX architectures found in this binary.
    pub ptx: Vec<ComputeCapability>,
    /// Resolved DT_NEEDED dependencies (sonames that mapped to known paths).
    pub deps: Vec<String>,
    /// Unresolved DT_NEEDED entries (sonames we couldn't find in the image).
    pub unresolved: Vec<String>,
    /// DT_RPATH entries from this binary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rpath: Vec<String>,
    /// DT_RUNPATH entries from this binary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runpath: Vec<String>,
    /// System package that owns this file (e.g. "libcuda1 550.90.07-1").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
}

impl DepNode {
    pub fn has_cuda(&self) -> bool {
        !self.cubins.is_empty() || !self.ptx.is_empty()
    }
}

/// Flat dependency graph: nodes keyed by soname, with adjacency via `deps`.
///
/// Using a flat representation (not a recursive tree) to handle diamond dependencies
/// and cycles cleanly, and to make graphviz rendering straightforward.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DepGraph {
    /// Root sonames (entrypoints + python extensions).
    pub roots: Vec<String>,
    /// All nodes, keyed by soname.
    pub nodes: HashMap<String, DepNode>,
}

/// Compute the effective soname for a binary: DT_SONAME if present, else filename.
fn effective_soname(binary: &BinaryScanResult) -> String {
    binary.soname.clone().unwrap_or_else(|| {
        binary
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&binary.path)
            .to_string()
    })
}

/// Build both the soname-to-path map and the soname-to-paths index in a single pass.
///
/// The map uses the first binary for each soname (higher priority sorts first).
/// The index tracks all unique paths per soname for collision detection.
fn build_soname_maps(
    binaries: &[BinaryScanResult],
) -> (HashMap<String, String>, HashMap<String, Vec<String>>) {
    let mut map = HashMap::new();
    let mut index: HashMap<String, Vec<String>> = HashMap::new();
    for binary in binaries {
        let key = effective_soname(binary);
        map.entry(key.clone())
            .or_insert_with(|| binary.path.clone());
        let paths = index.entry(key).or_default();
        if !paths.contains(&binary.path) {
            paths.push(binary.path.clone());
        }
    }
    (map, index)
}

/// Build the dependency graph via BFS from the given root paths.
///
/// Walks DT_NEEDED edges, resolving sonames to paths via the soname map.
/// Handles cycles by tracking visited sonames.
fn build_dep_graph(
    root_paths: &[String],
    binaries: &[BinaryScanResult],
    soname_map: &HashMap<String, String>,
    file_owners: &HashMap<String, String>,
) -> DepGraph {
    let path_to_binary: HashMap<&str, &BinaryScanResult> =
        binaries.iter().map(|b| (b.path.as_str(), b)).collect();

    let mut nodes: HashMap<String, DepNode> = HashMap::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut roots: Vec<String> = Vec::new();

    // Seed the BFS with root paths
    for path in root_paths {
        let soname = path.rsplit('/').next().unwrap_or(path).to_string();
        if visited.insert(soname.clone()) {
            queue.push_back(soname.clone());
            roots.push(soname);
        }
    }

    while let Some(soname) = queue.pop_front() {
        // Find the binary for this soname
        let path = soname_map.get(&soname);
        let binary = path.and_then(|p| path_to_binary.get(p.as_str()));

        let (cubins, ptx, needed, priority, size, rpath, runpath) = match binary {
            Some(b) => (
                b.cubins.clone(),
                b.ptx.clone(),
                b.needed.clone(),
                b.priority,
                b.size,
                b.rpath.clone(),
                b.runpath.clone(),
            ),
            None => (
                vec![],
                vec![],
                vec![],
                BinaryPriority::LooseGpuFile,
                0,
                vec![],
                vec![],
            ),
        };

        let mut resolved_deps = Vec::new();
        let mut unresolved = Vec::new();

        for dep_soname in &needed {
            if soname_map.contains_key(dep_soname.as_str()) {
                resolved_deps.push(dep_soname.clone());
                if visited.insert(dep_soname.clone()) {
                    queue.push_back(dep_soname.clone());
                }
            } else {
                unresolved.push(dep_soname.clone());
            }
        }

        let node_path = path
            .cloned()
            .unwrap_or_else(|| format!("<unresolved:{soname}>"));
        let package = file_owners.get(&node_path).cloned();

        nodes.insert(
            soname.clone(),
            DepNode {
                path: node_path,
                soname: soname.clone(),
                priority,
                size,
                cubins,
                ptx,
                deps: resolved_deps,
                unresolved,
                rpath,
                runpath,
                package,
            },
        );
    }

    DepGraph { roots, nodes }
}

/// Detect soname collisions: cases where multiple files in *different* directories
/// claim the same DT_SONAME. Same-directory duplicates are harmless (symlinks, same
/// file scanned twice) and get filtered out.
fn detect_soname_collisions(soname_index: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut collisions: Vec<(&String, &Vec<String>)> = soname_index
        .iter()
        .filter(|(_, paths)| {
            if paths.len() < 2 {
                return false;
            }
            let dirs: HashSet<&str> = paths
                .iter()
                .filter_map(|p| p.rsplit_once('/').map(|(dir, _)| dir))
                .collect();
            dirs.len() > 1
        })
        .collect();
    collisions.sort_by_key(|(soname, _)| soname.as_str());

    for (soname, paths) in collisions {
        warnings.push(format!(
            "soname collision: {} claimed by {}",
            soname,
            paths.join(", ")
        ));
    }
    warnings
}

/// Collect all paths reachable from the graph roots.
fn collect_reachable_paths(graph: &DepGraph) -> HashSet<String> {
    graph.nodes.values().map(|n| n.path.clone()).collect()
}

/// OS identification from /etc/os-release.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OsInfo {
    pub id: String,
    pub version_id: String,
    pub pretty_name: String,
}

/// A system or Python package with name, version, and optional source metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PackageVersion {
    pub name: String,
    pub version: String,
    /// Source package or vendor (dpkg `Source:` field, RPM vendor tag).
    /// Absent for Python packages and packages where the field wasn't set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A Python environment (virtualenv or system) with its installed packages.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PythonEnvironment {
    /// Human-readable label (e.g. "/opt/vllm" for a virtualenv, "system" for system Python).
    pub label: String,
    /// The site-packages directory path this environment was discovered from.
    pub site_packages_dir: String,
    /// Packages installed in this environment.
    pub packages: Vec<PackageVersion>,
}

/// Environment metadata collected from the image filesystem.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<OsInfo>,
    pub python_environments: Vec<PythonEnvironment>,
    pub system_packages: Vec<PackageVersion>,
    pub ld_conf_paths: Vec<String>,
    pub symlinks: HashMap<String, String>,
    /// Reverse map: absolute file path -> "package-name version" for RPM/dpkg ownership.
    #[serde(default)]
    pub file_owners: HashMap<String, String>,
}

/// Full scan result for an image.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScanResult {
    pub image: String,
    pub metadata: ImageMetadata,
    pub binaries: Vec<BinaryScanResult>,
    pub effective_cc_min: Option<ComputeCapability>,
    pub effective_cc_max: Option<ComputeCapability>,
    pub has_ptx_forward_compat: bool,
    pub warnings: Vec<String>,
    /// ELF dependency graph (populated when dep analysis is available).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dep_graph: Option<DepGraph>,
    /// Number of binaries reachable from entrypoint + python extensions.
    pub reachable_count: usize,
    /// Number of binaries not reachable (dormant).
    pub dormant_count: usize,
    /// Environment metadata (OS, packages, symlinks, ld.so.conf).
    pub environment: EnvironmentInfo,
    /// OCI image labels from the image config.
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// All environment variables from the image config.
    #[serde(default)]
    pub env_vars: Vec<(String, String)>,
}

/// Extract metadata from image config env vars.
fn extract_metadata(config: &ImageConfig) -> ImageMetadata {
    ImageMetadata {
        cuda_version: config.get_env("CUDA_VERSION").map(|s| s.to_string()),
        torch_arch_list: config
            .get_env("TORCH_CUDA_ARCH_LIST")
            .map(|s| s.to_string()),
        nvidia_require: config.get_env("NVIDIA_REQUIRE_CUDA").map(|s| s.to_string()),
        nvshmem_architectures: config
            .get_env("NVSHMEM_CUDA_ARCHITECTURES")
            .map(|s| s.to_string()),
    }
}

/// Parse /etc/os-release content into an OsInfo struct.
fn parse_os_release(content: &str) -> Option<OsInfo> {
    let mut id = None;
    let mut version_id = None;
    let mut pretty_name = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            // Strip optional quotes
            let value = value.trim_matches('"');
            match key {
                "ID" => id = Some(value.to_string()),
                "VERSION_ID" => version_id = Some(value.to_string()),
                "PRETTY_NAME" => pretty_name = Some(value.to_string()),
                _ => {}
            }
        }
    }

    Some(OsInfo {
        id: id.unwrap_or_default(),
        version_id: version_id.unwrap_or_default(),
        pretty_name: pretty_name.unwrap_or_default(),
    })
}

/// Parse /var/lib/dpkg/status into package name+version pairs.
fn parse_dpkg_status(content: &str) -> Vec<PackageVersion> {
    let mut packages = Vec::new();

    for block in content.split("\n\n") {
        let mut name = None;
        let mut version = None;
        let mut source = None;
        for line in block.lines() {
            if let Some(val) = line.strip_prefix("Package: ") {
                name = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("Version: ") {
                version = Some(val.trim().to_string());
            } else if let Some(val) = line.strip_prefix("Source: ") {
                // Source field can have a trailing version like "cuda-toolkit-12-9 (12.9.1-1)"
                let src = val.trim();
                let src_name = src.split_once(' ').map_or(src, |(name, _)| name);
                source = Some(src_name.to_string());
            }
        }
        if let (Some(n), Some(v)) = (name, version) {
            packages.push(PackageVersion {
                name: n,
                version: v,
                source,
            });
        }
    }

    packages
}

/// RPM header magic bytes: 0x8e 0xad 0xe8 0x01
const RPM_HEADER_MAGIC: [u8; 4] = [0x8e, 0xad, 0xe8, 0x01];

/// RPM tag constants.
const RPMTAG_NAME: u32 = 1000;
const RPMTAG_VERSION: u32 = 1001;
const RPMTAG_RELEASE: u32 = 1002;
const RPMTAG_VENDOR: u32 = 1011;
const RPMTAG_DIRINDEXES: u32 = 1116;
const RPMTAG_BASENAMES: u32 = 1117;
const RPMTAG_DIRNAMES: u32 = 1118;

/// RPM data type constants (from rpm header format).
const RPM_TYPE_INT32: u32 = 4;
const RPM_TYPE_STRING_ARRAY: u32 = 8;

/// Package metadata plus owned file paths from an RPM header.
struct RpmPackageInfo {
    package: PackageVersion,
    files: Vec<String>,
}

/// Read `count` null-terminated UTF-8 strings starting at `offset` in `store`.
fn read_string_array(store: &[u8], offset: usize, count: usize) -> Vec<String> {
    let mut result = Vec::with_capacity(count);
    let mut pos = offset;
    for _ in 0..count {
        if pos >= store.len() {
            break;
        }
        let remaining = &store[pos..];
        let end = remaining
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(remaining.len());
        if let Ok(s) = std::str::from_utf8(&remaining[..end]) {
            result.push(s.to_string());
        }
        pos += end + 1; // skip past the null terminator
    }
    result
}

/// Read `count` big-endian u32 values starting at `offset` in `store`.
fn read_u32_array(store: &[u8], offset: usize, count: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * 4;
        if start + 4 > store.len() {
            break;
        }
        result.push(u32::from_be_bytes([
            store[start],
            store[start + 1],
            store[start + 2],
            store[start + 3],
        ]));
    }
    result
}

/// Parse a single RPM header blob from rpmdb.sqlite into package info with file ownership.
///
/// RPM header binary format:
///   [0..4]   magic: 0x8e 0xad 0xe8 0x01
///   [4..8]   reserved
///   [8..12]  nindex: u32 BE (number of index entries)
///   [12..16] hsize: u32 BE (data store size in bytes)
///   [16..]   nindex * 16 bytes of index entries
///   [after]  hsize bytes of data store
///
/// Each index entry is 16 bytes:
///   tag:    u32 BE
///   type:   u32 BE
///   offset: i32 BE (into data store)
///   count:  u32 BE
fn parse_rpm_header_blob(blob: &[u8]) -> Option<RpmPackageInfo> {
    if blob.len() < 8 {
        return None;
    }

    // rpmdb.sqlite stores blobs in two formats:
    //   1. Raw header: [nindex(4)] [hsize(4)] [index entries] [data store]
    //   2. With intro: [magic(4)] [reserved(4)] [nindex(4)] [hsize(4)] [index entries] [data store]
    // Detect which by checking for the magic prefix.
    let skip = if blob.len() >= 16 && blob[0..4] == RPM_HEADER_MAGIC {
        8 // skip magic(4) + reserved(4)
    } else {
        0
    };
    if blob.len() < skip + 8 {
        return None;
    }

    let nindex =
        u32::from_be_bytes([blob[skip], blob[skip + 1], blob[skip + 2], blob[skip + 3]]) as usize;
    let hsize = u32::from_be_bytes([
        blob[skip + 4],
        blob[skip + 5],
        blob[skip + 6],
        blob[skip + 7],
    ]) as usize;

    let index_start = skip + 8;
    let index_end = index_start + nindex * 16;
    let store_start = index_end;
    let store_end = store_start + hsize;

    if blob.len() < store_end {
        return None;
    }

    let store = &blob[store_start..store_end];

    let mut name: Option<&str> = None;
    let mut version: Option<&str> = None;
    let mut release: Option<&str> = None;
    let mut vendor: Option<&str> = None;

    // File ownership tags
    let mut basenames: Option<Vec<String>> = None;
    let mut dirnames: Option<Vec<String>> = None;
    let mut dirindexes: Option<Vec<u32>> = None;

    for i in 0..nindex {
        let entry_off = index_start + i * 16;
        let tag = u32::from_be_bytes([
            blob[entry_off],
            blob[entry_off + 1],
            blob[entry_off + 2],
            blob[entry_off + 3],
        ]);
        let data_type = u32::from_be_bytes([
            blob[entry_off + 4],
            blob[entry_off + 5],
            blob[entry_off + 6],
            blob[entry_off + 7],
        ]);
        let offset = u32::from_be_bytes([
            blob[entry_off + 8],
            blob[entry_off + 9],
            blob[entry_off + 10],
            blob[entry_off + 11],
        ]) as usize;
        let count = u32::from_be_bytes([
            blob[entry_off + 12],
            blob[entry_off + 13],
            blob[entry_off + 14],
            blob[entry_off + 15],
        ]) as usize;

        match tag {
            RPMTAG_NAME | RPMTAG_VERSION | RPMTAG_RELEASE | RPMTAG_VENDOR => {
                if offset < store.len() {
                    let remaining = &store[offset..];
                    let end = remaining
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(remaining.len());
                    if let Ok(s) = std::str::from_utf8(&remaining[..end]) {
                        match tag {
                            RPMTAG_NAME => name = Some(s),
                            RPMTAG_VERSION => version = Some(s),
                            RPMTAG_RELEASE => release = Some(s),
                            RPMTAG_VENDOR => vendor = Some(s),
                            _ => {}
                        }
                    }
                }
            }
            RPMTAG_BASENAMES if data_type == RPM_TYPE_STRING_ARRAY => {
                basenames = Some(read_string_array(store, offset, count));
            }
            RPMTAG_DIRNAMES if data_type == RPM_TYPE_STRING_ARRAY => {
                dirnames = Some(read_string_array(store, offset, count));
            }
            RPMTAG_DIRINDEXES if data_type == RPM_TYPE_INT32 => {
                dirindexes = Some(read_u32_array(store, offset, count));
            }
            _ => {}
        }
    }

    let name = name?;
    let version_str = version?;
    let full_version = match release {
        Some(rel) => format!("{version_str}-{rel}"),
        None => version_str.to_string(),
    };

    // Reconstruct full file paths: dirnames[dirindexes[i]] + basenames[i]
    let files = match (basenames, dirnames, dirindexes) {
        (Some(bases), Some(dirs), Some(indexes)) => bases
            .iter()
            .zip(indexes.iter())
            .filter_map(|(base, &dir_idx)| {
                let dir_idx = dir_idx as usize;
                dirs.get(dir_idx).map(|dir| format!("{dir}{base}"))
            })
            .collect(),
        _ => Vec::new(),
    };

    Some(RpmPackageInfo {
        package: PackageVersion {
            name: name.to_string(),
            version: full_version,
            source: vendor.map(|s| s.to_string()),
        },
        files,
    })
}

/// Parse an rpmdb.sqlite database (written to a temp file) into package info with file ownership.
///
/// The RPM database stores packages in a `Packages` table with columns
/// `hnum INTEGER PRIMARY KEY` and `blob BLOB NOT NULL`. Each blob is a raw
/// RPM header in binary format (without the lead/signature, just the header).
fn parse_rpm_database(db_bytes: &[u8]) -> Vec<RpmPackageInfo> {
    // rusqlite needs a file path, so write bytes to a temp file.
    let db_path = std::env::temp_dir().join(format!("wiimi-rpmdb-{}.sqlite", std::process::id()));
    let result = parse_rpm_database_at(db_bytes, &db_path);
    let _ = std::fs::remove_file(&db_path);
    result
}

fn parse_rpm_database_at(db_bytes: &[u8], db_path: &std::path::Path) -> Vec<RpmPackageInfo> {
    if let Err(e) = std::fs::write(db_path, db_bytes) {
        tracing::warn!(error = %e, "failed to write rpmdb.sqlite to temp file");
        return Vec::new();
    }

    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "failed to open rpmdb.sqlite");
            return Vec::new();
        }
    };

    let mut stmt = match conn.prepare("SELECT blob FROM Packages") {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to query rpmdb.sqlite Packages table");
            return Vec::new();
        }
    };

    stmt.query_map([], |row| {
        let blob: Vec<u8> = row.get(0)?;
        Ok(blob)
    })
    .ok()
    .into_iter()
    .flatten()
    .filter_map(|r| r.ok())
    .filter_map(|blob| parse_rpm_header_blob(&blob))
    .collect()
}

/// Parse /etc/ld.so.conf (and included files) into a list of library search paths.
fn parse_ld_so_conf(content: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Skip "include" directives (we collect included files separately)
        if line.starts_with("include ") {
            continue;
        }
        paths.push(line.to_string());
    }
    paths
}

/// Build an EnvironmentInfo struct from collected metadata items.
fn collect_environment(metadata: Vec<image::DiscoveredMetadata>) -> EnvironmentInfo {
    let mut os = None;
    let mut py_by_dir: HashMap<String, Vec<PackageVersion>> = HashMap::new();
    let mut system_packages = Vec::new();
    let mut ld_conf_paths = Vec::new();
    let mut symlinks = HashMap::new();
    let mut file_owners: HashMap<String, String> = HashMap::new();

    // Bare dpkg package names collected from .list files, upgraded to "name version" later
    let mut dpkg_file_lists: Vec<(String, Vec<String>)> = Vec::new();

    for meta in metadata {
        match meta {
            image::DiscoveredMetadata::Symlink { path, target } => {
                symlinks.insert(path, target);
            }
            image::DiscoveredMetadata::OsRelease(content) => {
                os = parse_os_release(&content);
            }
            image::DiscoveredMetadata::LdSoConf(content) => {
                ld_conf_paths.extend(parse_ld_so_conf(&content));
            }
            image::DiscoveredMetadata::PythonPackage {
                name,
                version,
                site_packages_dir,
            } => {
                py_by_dir
                    .entry(site_packages_dir)
                    .or_default()
                    .push(PackageVersion {
                        name,
                        version,
                        source: None,
                    });
            }
            image::DiscoveredMetadata::DpkgStatus(content) => {
                system_packages = parse_dpkg_status(&content);
            }
            image::DiscoveredMetadata::DpkgFileList {
                package_name,
                content,
            } => {
                let paths: Vec<String> = content
                    .lines()
                    .filter(|line| line.starts_with('/'))
                    .map(|line| line.to_string())
                    .collect();
                dpkg_file_lists.push((package_name, paths));
            }
            image::DiscoveredMetadata::RpmDatabase(db_bytes) => {
                let rpm_packages = parse_rpm_database(&db_bytes);
                for info in rpm_packages {
                    let owner_label = format!("{} {}", info.package.name, info.package.version);
                    for file_path in &info.files {
                        file_owners.insert(file_path.clone(), owner_label.clone());
                    }
                    system_packages.push(info.package);
                }
            }
        }
    }

    // Upgrade dpkg file_owners from bare names to "name version" using system_packages
    let dpkg_version_map: HashMap<&str, &str> = system_packages
        .iter()
        .map(|p| (p.name.as_str(), p.version.as_str()))
        .collect();
    for (pkg_name, paths) in &dpkg_file_lists {
        let owner_label = match dpkg_version_map.get(pkg_name.as_str()) {
            Some(version) => format!("{pkg_name} {version}"),
            None => pkg_name.clone(),
        };
        for file_path in paths {
            file_owners.insert(file_path.clone(), owner_label.clone());
        }
    }

    // Symlink resolution: if a symlink target is owned but the link path is not,
    // propagate ownership to the link path.
    for (link_path, raw_target) in &symlinks {
        if file_owners.contains_key(link_path) {
            continue;
        }
        // Resolve relative targets against the link's parent directory
        let resolved = if raw_target.starts_with('/') {
            raw_target.clone()
        } else {
            let parent = link_path.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
            format!("{parent}/{raw_target}")
        };
        if let Some(owner) = file_owners.get(&resolved) {
            file_owners.insert(link_path.clone(), owner.clone());
        }
    }

    // Deduplicate ld.so.conf paths
    ld_conf_paths.sort();
    ld_conf_paths.dedup();

    // Build sorted, deduplicated Python environments
    let mut python_environments: Vec<PythonEnvironment> = py_by_dir
        .into_iter()
        .map(|(site_packages_dir, mut packages)| {
            packages.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
            packages.dedup_by(|a, b| a.name == b.name && a.version == b.version);
            let label = python_env_label(&site_packages_dir);
            PythonEnvironment {
                label,
                site_packages_dir,
                packages,
            }
        })
        .collect();
    // Sort environments: virtualenvs first (non-"system"), then by label
    python_environments.sort_by(|a, b| {
        let a_system = a.label == "system";
        let b_system = b.label == "system";
        a_system.cmp(&b_system).then(a.label.cmp(&b.label))
    });

    system_packages.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
    system_packages.dedup_by(|a, b| a.name == b.name && a.version == b.version);

    // Strip uv cache symlinks; they are internal to the package manager and just noise.
    symlinks.retain(|path, _| !path.contains("/.cache/uv/"));

    EnvironmentInfo {
        os,
        python_environments,
        system_packages,
        ld_conf_paths,
        symlinks,
        file_owners,
    }
}

/// Derive a human-readable label for a Python environment from its site-packages path.
///
/// System Python paths like `/usr/lib/python3.12/site-packages` get labeled "system".
/// Virtualenv paths like `/opt/vllm/lib/python3.12/site-packages` get the prefix
/// before `/lib/python` (e.g. "/opt/vllm").
fn python_env_label(site_packages_dir: &str) -> String {
    // Try to extract virtualenv root: everything before /lib/pythonX.Y/site-packages
    if let Some(idx) = site_packages_dir.find("/lib/python") {
        let prefix = &site_packages_dir[..idx];
        // System paths start with /usr or /usr/local
        if prefix == "/usr" || prefix == "/usr/local" {
            return "system".to_string();
        }
        return prefix.to_string();
    }
    // Fallback: use the full path
    site_packages_dir.to_string()
}

/// Channel capacity for the binary discovery → scanning pipeline.
///
/// Bounded to limit memory: each DiscoveredBinary can be tens of MB,
/// so we keep at most this many queued between extraction and scanning.
/// The channel's backpressure naturally throttles layer extraction when
/// the scanner workers can't keep up.
const BINARY_CHANNEL_CAPACITY: usize = 64;

/// Scan an OCI image for CUDA compute capabilities.
///
/// Architecture: producers (layer extractors) and consumers (fatbin scanners) are
/// decoupled via a bounded channel. Layer download/decompress/extract runs on async
/// tasks bridged to blocking I/O, while fatbin scanning runs on a separate blocking
/// worker pool. Both sides run concurrently, fully overlapping I/O with CPU work.
pub async fn scan_image(
    image_str: &str,
    dockerconfig: Option<&[u8]>,
    insecure_registries: Vec<String>,
    show_progress: bool,
    use_blob_cache: bool,
    use_parse_cache: bool,
) -> Result<ScanResult> {
    let image_ref: Reference = image_str
        .parse()
        .with_context(|| format!("invalid image reference: {image_str}"))?;

    let auth = crate::registry::resolve_auth_for_reference(dockerconfig, &image_ref);

    let protocol = if insecure_registries.is_empty() {
        oci_client::client::ClientProtocol::Https
    } else {
        oci_client::client::ClientProtocol::HttpsExcept(insecure_registries)
    };

    let client_config = oci_client::client::ClientConfig {
        protocol,
        platform_resolver: Some(Box::new(oci_client::client::linux_amd64_resolver)),
        ..Default::default()
    };
    let client = Arc::new(Client::new(client_config));

    tracing::info!(image = %image_ref, "pulling manifest and config");

    let (manifest, _digest, config_json) = client
        .pull_manifest_and_config(&image_ref, &auth)
        .await
        .context("failed to pull manifest and config")?;

    let config = ImageConfig::parse(&config_json)?;
    let metadata = extract_metadata(&config);

    // Total compressed size across all layers, for the download progress bar.
    let total_compressed: u64 = manifest
        .layers
        .iter()
        .map(|l| u64::try_from(l.size).unwrap_or(0))
        .sum();

    let progress = ScanProgress::new(
        total_compressed,
        manifest.layers.len(),
        show_progress,
        image_str.to_string(),
    );

    let blob_cache = if use_blob_cache {
        match BlobCache::default_location() {
            Some(c) => {
                tracing::info!(root = %c.root().display(), "blob cache enabled");
                Some(c)
            }
            None => {
                tracing::warn!("could not determine cache directory, blob caching disabled");
                None
            }
        }
    } else {
        tracing::info!("blob cache disabled (--no-cache)");
        None
    };

    let parse_cache = if use_parse_cache {
        match ParseCache::default_location() {
            Some(c) => {
                tracing::info!(root = %c.root().display(), "parse cache enabled");
                Some(Arc::new(c))
            }
            None => {
                tracing::warn!("could not determine cache directory, parse caching disabled");
                None
            }
        }
    } else {
        tracing::info!("parse cache disabled");
        None
    };

    let layer_cache = if use_blob_cache {
        match LayerCache::default_location() {
            Some(c) => {
                tracing::info!(root = %c.root().display(), "layer manifest cache enabled");
                Some(c)
            }
            None => None,
        }
    } else {
        None
    };

    // Phase 1: Try to resolve layers entirely from manifest + parse cache.
    //
    // For each layer where we have both a blob cache hit (so we know the layer
    // content hasn't changed) and a layer manifest, attempt to reconstruct all
    // BinaryScanResults directly from the parse cache without decompression,
    // tar extraction, or SHA256 hashing.
    let mut pre_resolved_results: Vec<BinaryScanResult> = Vec::new();
    let mut all_metadata: Vec<image::DiscoveredMetadata> = Vec::new();
    let mut layers_to_extract: Vec<(usize, oci_client::manifest::OciDescriptor)> = Vec::new();

    for (idx, layer) in manifest.layers.iter().enumerate() {
        let resolved = 'resolve: {
            let blob_cache = match blob_cache {
                Some(ref c) => c,
                None => break 'resolve false,
            };
            if blob_cache.get(&layer.digest).is_none() {
                break 'resolve false;
            }
            let layer_cache = match layer_cache {
                Some(ref c) => c,
                None => break 'resolve false,
            };
            let layer_manifest = match layer_cache.get(&layer.digest) {
                Some(m) => m,
                None => break 'resolve false,
            };
            let parse_cache = match parse_cache {
                Some(ref c) => c,
                None => break 'resolve false,
            };

            match try_resolve_from_manifest(&layer_manifest, &config, parse_cache) {
                Some((binaries, metadata)) => {
                    let num_binaries = binaries.len() as u64;
                    let layer_size = u64::try_from(layer.size).unwrap_or(0);
                    tracing::info!(
                        layer = idx,
                        digest = %layer.digest,
                        binaries = num_binaries,
                        "layer fully resolved from manifest cache"
                    );
                    pre_resolved_results.extend(binaries);
                    all_metadata.extend(metadata);

                    if layer_manifest.has_rpm_database {
                        if let Some(rpm_data) = layer_cache.get_rpm_database(&layer.digest) {
                            all_metadata.push(image::DiscoveredMetadata::RpmDatabase(rpm_data));
                        }
                    }

                    progress.skip_layer(layer_size, num_binaries);
                    true
                }
                None => false,
            }
        };

        if !resolved {
            layers_to_extract.push((idx, layer.clone()));
        }
    }

    let pre_resolved_count = pre_resolved_results.len();
    let layers_skipped = manifest.layers.len() - layers_to_extract.len();
    if layers_skipped > 0 {
        tracing::info!(
            skipped = layers_skipped,
            total = manifest.layers.len(),
            binaries = pre_resolved_count,
            "layers resolved from manifest cache"
        );
    }

    // Phase 2: Run the extraction pipeline for remaining layers.
    let mut pipeline_results = Vec::new();
    if !layers_to_extract.is_empty() {
        let (tx, rx) =
            tokio::sync::mpsc::channel::<image::DiscoveredBinary>(BINARY_CHANNEL_CAPACITY);
        let (meta_tx, meta_rx) =
            tokio::sync::mpsc::unbounded_channel::<image::DiscoveredMetadata>();

        let producer = tokio::spawn(image::stream_binaries_to(
            client,
            image_ref,
            auth,
            layers_to_extract,
            config.clone(),
            tx,
            meta_tx,
            progress.clone(),
            blob_cache,
            layer_cache,
        ));

        let consumer = tokio::spawn(scan_from_channel(rx, progress.clone(), parse_cache));

        producer
            .await
            .context("producer task panicked")?
            .context("layer extraction failed")?;

        pipeline_results = consumer
            .await
            .context("consumer task panicked")?
            .context("binary scanning failed")?;

        // Drain metadata from the pipeline
        let mut meta_rx = meta_rx;
        while let Ok(meta) = meta_rx.try_recv() {
            all_metadata.push(meta);
        }
    }

    // Phase 3: Merge pre-resolved and pipeline results.
    let mut scan_results = pre_resolved_results;
    scan_results.extend(pipeline_results);

    let (cache_hits, cache_misses) = progress.cache_stats();
    progress.finish();

    if use_parse_cache {
        tracing::info!(
            hits = cache_hits,
            misses = cache_misses,
            "parse cache: {cache_hits} hits, {cache_misses} misses"
        );
    }

    let environment = collect_environment(all_metadata);

    // Build dependency graph with hybrid reachability:
    // roots = entrypoint/cmd paths + all PythonExtension binaries
    let mut root_paths: Vec<String> = config
        .entrypoint
        .iter()
        .chain(config.cmd.iter())
        .filter(|p| p.starts_with('/'))
        .cloned()
        .collect();

    for binary in &scan_results {
        if binary.priority == BinaryPriority::PythonExtension {
            root_paths.push(binary.path.clone());
        }
    }

    let (soname_map, soname_index) = build_soname_maps(&scan_results);
    let dep_graph = build_dep_graph(
        &root_paths,
        &scan_results,
        &soname_map,
        &environment.file_owners,
    );
    let reachable = collect_reachable_paths(&dep_graph);
    let collision_warnings = detect_soname_collisions(&soname_index);

    // Separate CUDA binaries into reachable vs dormant (single pass)
    let mut reachable_cuda = Vec::new();
    let mut total_cuda = 0usize;
    for binary in &scan_results {
        if binary.cubins.is_empty() && binary.ptx.is_empty() {
            continue;
        }
        total_cuda += 1;
        if reachable.contains(&binary.path) {
            reachable_cuda.push(binary.clone());
        }
    }

    let reachable_count = reachable_cuda.len();
    let dormant_count = total_cuda - reachable_count;

    // Compute effective range using only reachable CUDA binaries
    let (effective_cc_min, effective_cc_max, has_ptx, mut warnings) =
        compute_effective_range(&reachable_cuda);
    warnings.extend(collision_warnings);

    Ok(ScanResult {
        image: image_str.to_string(),
        metadata,
        binaries: scan_results,
        effective_cc_min,
        effective_cc_max,
        has_ptx_forward_compat: has_ptx,
        warnings,
        dep_graph: Some(dep_graph),
        reachable_count,
        dormant_count,
        environment,
        labels: config.labels.clone(),
        env_vars: config.env.clone(),
    })
}

/// Consumer loop: pulls binaries from the channel and fans them out to blocking
/// worker threads for fatbin scanning.
///
/// Returns all successfully parsed ELFs (with or without CUDA content) so we can
/// build a complete dependency graph. Limits inflight work to 2x CPU cores so we
/// don't queue unbounded binary data in memory while waiting for scan threads.
async fn scan_from_channel(
    mut rx: tokio::sync::mpsc::Receiver<image::DiscoveredBinary>,
    progress: ScanProgress,
    parse_cache: Option<Arc<ParseCache>>,
) -> Result<Vec<BinaryScanResult>> {
    // Parsing ELF headers + checking parse cache is near-instant, so we can
    // afford a deep pipeline. The channel capacity (64) provides the real
    // backpressure; this just prevents unbounded JoinSet growth.
    let max_inflight = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .max(16)
        * 16;

    let mut join_set = JoinSet::new();
    let mut results = Vec::new();
    while let Some(binary) = rx.recv().await {
        while join_set.len() >= max_inflight {
            if let Some(result) = join_set.join_next().await {
                if let Some(scan_result) = result.context("scan task panicked")? {
                    progress.inc_scanned();
                    results.push(scan_result);
                }
            }
        }

        let cache = parse_cache.clone();
        let prog = progress.clone();
        join_set.spawn_blocking(move || scan_single_binary(binary, cache.as_deref(), &prog));
    }

    while let Some(result) = join_set.join_next().await {
        if let Some(scan_result) = result.context("scan task panicked")? {
            progress.inc_scanned();
            results.push(scan_result);
        }
    }

    results.sort_by(|a, b| a.priority.cmp(&b.priority));

    Ok(results)
}

/// Scan a single binary for fatbin sections and DT_NEEDED deps.
///
/// Returns a result for every valid ELF (even those without CUDA content)
/// so we can build the full dependency graph. The `has_cuda` flag distinguishes
/// binaries with fatbin content from pure dependency-chain intermediaries.
fn scan_single_binary(
    binary: image::DiscoveredBinary,
    parse_cache: Option<&ParseCache>,
    progress: &ScanProgress,
) -> Option<BinaryScanResult> {
    let content_hash = binary
        .content_sha256
        .unwrap_or_else(|| hex::encode(Sha256::digest(&binary.data)));

    // Check parse cache
    if let Some(cache) = parse_cache {
        if let Some(cached) = cache.get(&content_hash) {
            progress.inc_cache_hit();
            return Some(BinaryScanResult {
                path: binary.path,
                priority: binary.priority,
                size: binary.data.len() as u64,
                cubins: cached.cubins,
                ptx: cached.ptx,
                needed: cached.needed,
                soname: cached.soname,
                rpath: cached.rpath,
                runpath: cached.runpath,
            });
        }
    }

    let elf_info = match fatbin::scan_elf_with_deps(&binary.data) {
        Ok(info) => info,
        Err(e) => {
            tracing::debug!(
                path = %binary.path,
                error = %e,
                "failed to scan binary"
            );
            return None;
        }
    };

    let mut cubins: Vec<ComputeCapability> = elf_info
        .fatbin_entries
        .iter()
        .filter(|e| e.is_cubin)
        .map(|e| e.compute_capability())
        .collect();
    cubins.sort();
    cubins.dedup();

    let mut ptx: Vec<ComputeCapability> = elf_info
        .fatbin_entries
        .iter()
        .filter(|e| !e.is_cubin)
        .map(|e| e.compute_capability())
        .collect();
    ptx.sort();
    ptx.dedup();

    // Store in parse cache
    if let Some(cache) = parse_cache {
        cache.put(
            &content_hash,
            &CachedElfResult {
                cubins: cubins.clone(),
                ptx: ptx.clone(),
                needed: elf_info.needed.clone(),
                soname: elf_info.soname.clone(),
                rpath: elf_info.rpath.clone(),
                runpath: elf_info.runpath.clone(),
            },
        );
    }

    Some(BinaryScanResult {
        path: binary.path,
        priority: binary.priority,
        size: binary.data.len() as u64,
        cubins,
        ptx,
        needed: elf_info.needed,
        soname: elf_info.soname,
        rpath: elf_info.rpath,
        runpath: elf_info.runpath,
    })
}

/// Compute the effective CC range across all scanned binaries.
///
/// `effective_cc_min` = max of all per-binary minimums (the most restrictive component).
/// `effective_cc_max` = min of all per-binary maximums, unless any binary has PTX.
pub fn compute_effective_range(
    binaries: &[BinaryScanResult],
) -> (
    Option<ComputeCapability>,
    Option<ComputeCapability>,
    bool,
    Vec<String>,
) {
    if binaries.is_empty() {
        return (
            None,
            None,
            false,
            vec!["no CUDA binaries found".to_string()],
        );
    }

    let mut warnings = Vec::new();
    let mut global_min: Option<ComputeCapability> = None;
    let mut global_max: Option<ComputeCapability> = None;
    let mut any_ptx = false;

    for binary in binaries {
        let mut archs = binary.cubins.iter().chain(binary.ptx.iter()).peekable();
        if archs.peek().is_none() {
            continue;
        }

        let Some(&bin_min) = archs.clone().min() else {
            continue;
        };
        let Some(&bin_max) = archs.max() else {
            continue;
        };
        let has_ptx = !binary.ptx.is_empty();

        if has_ptx {
            any_ptx = true;
        }

        global_min = Some(global_min.map_or(bin_min, |cur| cur.max(bin_min)));

        if !has_ptx {
            global_max = Some(global_max.map_or(bin_max, |cur| cur.min(bin_max)));
        }
    }

    // Warn if the range is inverted (some binary has a max cubin below another's min)
    if let (Some(min), Some(max)) = (global_min, global_max) {
        if min > max {
            warnings.push(format!(
                "inverted range: cc_min ({min}) > cc_max ({max}), some binary may have stale/stub cubins"
            ));
        }
    }

    (global_min, global_max, any_ptx, warnings)
}

/// Threshold at which we write detailed results to a file.
const FILE_OUTPUT_THRESHOLD: usize = 50;

/// Format the scan result as a human-readable report with dependency tree.
pub fn format_report(result: &ScanResult) -> String {
    let mut out = String::new();

    out.push_str(&format!("Image: {}\n\n", result.image));

    // Metadata section
    out.push_str("Image metadata:\n");
    if let Some(ref v) = result.metadata.cuda_version {
        out.push_str(&format!("  CUDA_VERSION                = {v}\n"));
    }
    if let Some(ref v) = result.metadata.torch_arch_list {
        out.push_str(&format!("  TORCH_CUDA_ARCH_LIST        = {v}\n"));
    }
    if let Some(ref v) = result.metadata.nvidia_require {
        out.push_str(&format!("  NVIDIA_REQUIRE_CUDA         = {v}\n"));
    }
    if let Some(ref v) = result.metadata.nvshmem_architectures {
        out.push_str(&format!("  NVSHMEM_CUDA_ARCHITECTURES  = {v}\n"));
    }
    if result.metadata.cuda_version.is_none()
        && result.metadata.torch_arch_list.is_none()
        && result.metadata.nvidia_require.is_none()
        && result.metadata.nvshmem_architectures.is_none()
    {
        out.push_str("  (none found)\n");
    }
    out.push('\n');

    // Environment section
    let env = &result.environment;
    let has_env =
        env.os.is_some() || !env.python_environments.is_empty() || !env.system_packages.is_empty();
    if has_env {
        out.push_str("Environment:\n");
        if let Some(ref os) = env.os {
            let os_label = if os.pretty_name.is_empty() {
                format!("{} {}", os.id, os.version_id)
            } else {
                os.pretty_name.clone()
            };
            out.push_str(&format!("  OS              = {os_label}\n"));
        }
        for py_env in &env.python_environments {
            let count = py_env.packages.len();
            out.push_str(&format!("  Python ({}) = {count} packages\n", py_env.label));
        }
        if !env.system_packages.is_empty() {
            out.push_str(&format!(
                "  System packages = {} packages\n",
                env.system_packages.len()
            ));
        }
        out.push('\n');
    }

    // Dependency tree (if available)
    if let Some(ref graph) = result.dep_graph {
        out.push_str(&render_dep_tree(graph));
        out.push('\n');

        // Resolution issues: flag deprecated DT_RPATH usage
        let rpath_issues: Vec<&DepNode> = graph
            .nodes
            .values()
            .filter(|n| !n.rpath.is_empty())
            .collect();
        if !rpath_issues.is_empty() {
            out.push_str("Resolution issues:\n");
            for node in &rpath_issues {
                out.push_str(&format!(
                    "  {} uses deprecated DT_RPATH (should be DT_RUNPATH)\n",
                    node.path
                ));
            }
            out.push('\n');
        }
    }

    // Dormant summary
    if result.dormant_count > 0 {
        out.push_str(&format!(
            "Dormant (not reachable from entrypoint): {} binaries\n",
            result.dormant_count,
        ));
        out.push_str("  (use --json or --html for full graph)\n\n");
    }

    // Write full results to a file when there are many CUDA binaries
    let cuda_count = result
        .binaries
        .iter()
        .filter(|b| !b.cubins.is_empty() || !b.ptx.is_empty())
        .count();
    if cuda_count > FILE_OUTPUT_THRESHOLD {
        let detail_path = write_detail_file(result);
        match detail_path {
            Ok(path) => {
                out.push_str(&format!("Full results written to: {path}\n\n"));
            }
            Err(e) => {
                out.push_str(&format!("(failed to write detail file: {e})\n\n"));
            }
        }
    }

    // Effective range
    out.push_str(&format!(
        "Effective range ({} reachable binaries):\n",
        result.reachable_count,
    ));
    match result.effective_cc_min {
        Some(cc) => out.push_str(&format!("  cc_min = {cc}\n")),
        None => out.push_str("  cc_min = <unknown>\n"),
    }
    match result.effective_cc_max {
        Some(cc) => out.push_str(&format!("  cc_max = {cc}\n")),
        None if result.has_ptx_forward_compat => {
            out.push_str("  cc_max = <none> (PTX forward compat)\n");
        }
        None => out.push_str("  cc_max = <unknown>\n"),
    }
    out.push('\n');

    out
}

/// Render the dependency tree as ASCII art.
///
/// Only shows nodes with CUDA fatbins or that are direct parents of CUDA nodes.
/// Non-CUDA intermediaries are collapsed. Unresolved deps shown as dimmed leaves.
fn render_dep_tree(graph: &DepGraph) -> String {
    let mut out = String::new();
    out.push_str("Dependency tree (from entrypoint + python extensions):\n");

    // Determine which nodes have CUDA content or have descendants with CUDA
    let cuda_relevant = find_cuda_relevant_nodes(graph);

    for root in &graph.roots {
        // Skip roots that have no CUDA-relevant descendants (avoids hundreds
        // of Cython/numpy/etc. modules cluttering the tree)
        if !cuda_relevant.contains(root) {
            continue;
        }
        render_tree_node(
            &mut out,
            graph,
            root,
            "",
            true,
            &cuda_relevant,
            &mut HashSet::new(),
        );
    }

    out
}

/// Find all nodes that either have CUDA content or are ancestors of CUDA nodes.
fn find_cuda_relevant_nodes(graph: &DepGraph) -> HashSet<String> {
    let mut relevant = HashSet::new();

    // Mark all nodes with CUDA content
    for (soname, node) in &graph.nodes {
        if node.has_cuda() {
            relevant.insert(soname.clone());
        }
    }

    // For each root, DFS and mark ancestors of CUDA nodes
    for root in &graph.roots {
        mark_cuda_ancestors(graph, root, &mut relevant, &mut HashSet::new());
    }

    relevant
}

/// DFS to mark nodes that are ancestors of CUDA nodes. Returns true if this
/// subtree contains any CUDA node.
fn mark_cuda_ancestors(
    graph: &DepGraph,
    soname: &str,
    relevant: &mut HashSet<String>,
    visited: &mut HashSet<String>,
) -> bool {
    if !visited.insert(soname.to_string()) {
        return relevant.contains(soname);
    }

    let node = match graph.nodes.get(soname) {
        Some(n) => n,
        None => return false,
    };

    let self_has_cuda = node.has_cuda();
    let mut child_has_cuda = false;

    for dep in &node.deps {
        if mark_cuda_ancestors(graph, dep, relevant, visited) {
            child_has_cuda = true;
        }
    }

    if self_has_cuda || child_has_cuda {
        relevant.insert(soname.to_string());
        true
    } else {
        false
    }
}

/// Render a single tree node with box-drawing characters.
fn render_tree_node(
    out: &mut String,
    graph: &DepGraph,
    soname: &str,
    prefix: &str,
    is_last: bool,
    cuda_relevant: &HashSet<String>,
    visited: &mut HashSet<String>,
) {
    let connector = if prefix.is_empty() {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    let node = graph.nodes.get(soname);

    // Format CC range annotation
    let cc_annotation = match node {
        Some(n) if n.has_cuda() => {
            let min_sm = n
                .cubins
                .iter()
                .chain(n.ptx.iter())
                .min()
                .map(|cc| format!("sm_{}", cc.major * 10 + cc.minor));
            let max_sm = n
                .cubins
                .iter()
                .max()
                .map(|cc| format!("sm_{}", cc.major * 10 + cc.minor));
            let ptx_suffix = if !n.ptx.is_empty() { " + PTX" } else { "" };
            match (min_sm, max_sm) {
                (Some(min), Some(max)) if min != max => {
                    format!("  {min}..{max}{ptx_suffix}")
                }
                (Some(min), _) => format!("  {min}{ptx_suffix}"),
                _ => String::new(),
            }
        }
        _ => String::new(),
    };

    // Format RPATH/RUNPATH annotation
    let rpath_annotation = match node {
        Some(n) => {
            let mut parts = Vec::new();
            if !n.rpath.is_empty() {
                parts.push(format!("RPATH={}", n.rpath.join(":")));
            }
            if !n.runpath.is_empty() {
                parts.push(format!("RUNPATH={}", n.runpath.join(":")));
            }
            if parts.is_empty() {
                String::new()
            } else {
                format!("  [{}]", parts.join(", "))
            }
        }
        None => String::new(),
    };

    let display_name = match node {
        Some(n) => n.path.as_str(),
        None => soname,
    };

    out.push_str(&format!(
        "  {prefix}{connector}{display_name}{cc_annotation}{rpath_annotation}\n"
    ));

    // Cycle detection
    if !visited.insert(soname.to_string()) {
        return;
    }

    let child_prefix = if prefix.is_empty() {
        "".to_string()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    if let Some(node) = node {
        // Filter children to only cuda-relevant ones
        let relevant_deps: Vec<&String> = node
            .deps
            .iter()
            .filter(|d| cuda_relevant.contains(d.as_str()))
            .collect();

        // Count non-CUDA deps that we're hiding
        let hidden_count = node.deps.len() - relevant_deps.len();

        for (i, dep) in relevant_deps.iter().enumerate() {
            let is_last_child =
                i == relevant_deps.len() - 1 && node.unresolved.is_empty() && hidden_count == 0;
            render_tree_node(
                out,
                graph,
                dep,
                &child_prefix,
                is_last_child,
                cuda_relevant,
                visited,
            );
        }

        if hidden_count > 0 {
            let connector = if node.unresolved.is_empty() {
                "└── "
            } else {
                "├── "
            };
            out.push_str(&format!(
                "  {child_prefix}{connector}({hidden_count} deps without CUDA fatbins)\n"
            ));
        }

        if !node.unresolved.is_empty() {
            let unresolved_list = node.unresolved.join(", ");
            out.push_str(&format!(
                "  {child_prefix}└── ({} unresolved: {unresolved_list})\n",
                node.unresolved.len(),
            ));
        }
    }

    // Remove from visited so the same node can appear under different parents
    // (diamond deps). The cycle check above prevents infinite recursion for
    // true cycles because we return early before recursing.
    visited.remove(soname);
}

/// Write the full scan result as JSON to a file alongside the working directory.
fn write_detail_file(result: &ScanResult) -> Result<String> {
    let filename = format!(
        "wiimi-scan-{}.json",
        crate::sanitize_filename(&result.image)
    );
    let json = serde_json::to_string_pretty(result).context("failed to serialize scan result")?;
    std::fs::write(&filename, &json)
        .with_context(|| format!("failed to write detail file {filename}"))?;
    Ok(filename)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::nvidia::ComputeCapability;

    use crate::image::{BinaryPriority, ImageConfig};
    use crate::scan::{
        build_dep_graph, build_soname_maps, collect_environment, collect_reachable_paths,
        compute_effective_range, extract_metadata, format_report, parse_dpkg_status,
        parse_ld_so_conf, parse_os_release, parse_rpm_database_at, parse_rpm_header_blob,
        python_env_label, read_string_array, read_u32_array, render_dep_tree, BinaryScanResult,
        DepGraph, EnvironmentInfo, ScanResult, RPMTAG_BASENAMES, RPMTAG_DIRINDEXES,
        RPMTAG_DIRNAMES, RPMTAG_NAME, RPMTAG_RELEASE, RPMTAG_VENDOR, RPMTAG_VERSION,
        RPM_HEADER_MAGIC, RPM_TYPE_INT32, RPM_TYPE_STRING_ARRAY,
    };

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

    fn binary(
        path: &str,
        priority: BinaryPriority,
        cubins: Vec<ComputeCapability>,
        ptx: Vec<ComputeCapability>,
        needed: Vec<&str>,
    ) -> BinaryScanResult {
        BinaryScanResult {
            path: path.to_string(),
            priority,
            size: 0,
            cubins,
            ptx,
            needed: needed.into_iter().map(|s| s.to_string()).collect(),
            soname: None,
            rpath: vec![],
            runpath: vec![],
        }
    }

    fn dep_node(
        path: &str,
        soname: &str,
        cubins: Vec<ComputeCapability>,
        ptx: Vec<ComputeCapability>,
        deps: Vec<&str>,
        unresolved: Vec<&str>,
    ) -> crate::scan::DepNode {
        crate::scan::DepNode {
            path: path.to_string(),
            soname: soname.to_string(),
            priority: BinaryPriority::LinkerLibrary,
            size: 0,
            cubins,
            ptx,
            deps: deps.into_iter().map(|s| s.to_string()).collect(),
            unresolved: unresolved.into_iter().map(|s| s.to_string()).collect(),
            rpath: vec![],
            runpath: vec![],
            package: None,
        }
    }

    // -- extract_metadata --

    #[test]
    fn metadata_from_env() {
        let config = ImageConfig {
            path_dirs: vec![],
            ld_library_dirs: vec![],
            virtual_env: None,
            entrypoint: vec![],
            cmd: vec![],
            env: vec![
                ("CUDA_VERSION".to_string(), "12.9.1".to_string()),
                (
                    "TORCH_CUDA_ARCH_LIST".to_string(),
                    "7.0;8.0;9.0".to_string(),
                ),
                ("NVIDIA_REQUIRE_CUDA".to_string(), "cuda>=12.0".to_string()),
                (
                    "NVSHMEM_CUDA_ARCHITECTURES".to_string(),
                    "90a;100".to_string(),
                ),
            ],
            labels: HashMap::new(),
        };
        let meta = extract_metadata(&config);
        assert_eq!(meta.cuda_version.as_deref(), Some("12.9.1"));
        assert_eq!(meta.torch_arch_list.as_deref(), Some("7.0;8.0;9.0"));
        assert_eq!(meta.nvidia_require.as_deref(), Some("cuda>=12.0"));
        assert_eq!(meta.nvshmem_architectures.as_deref(), Some("90a;100"));
    }

    #[test]
    fn metadata_empty_env() {
        let config = ImageConfig {
            path_dirs: vec![],
            ld_library_dirs: vec![],
            virtual_env: None,
            entrypoint: vec![],
            cmd: vec![],
            env: vec![],
            labels: HashMap::new(),
        };
        let meta = extract_metadata(&config);
        assert!(meta.cuda_version.is_none());
        assert!(meta.torch_arch_list.is_none());
    }

    // -- compute_effective_range --

    #[test]
    fn effective_range_single_binary() {
        let binaries = vec![binary(
            "/lib/torch_cuda.so",
            BinaryPriority::LinkerLibrary,
            vec![cc(7, 0), cc(8, 0), cc(9, 0)],
            vec![],
            vec![],
        )];
        let (min, max, has_ptx, _) = compute_effective_range(&binaries);
        assert_eq!(min, Some(cc(7, 0)));
        assert_eq!(max, Some(cc(9, 0)));
        assert!(!has_ptx);
    }

    #[test]
    fn effective_range_with_ptx() {
        let binaries = vec![binary(
            "/lib/torch_cuda.so",
            BinaryPriority::LinkerLibrary,
            vec![cc(7, 0), cc(8, 0), cc(9, 0)],
            vec![cc(12, 0)],
            vec![],
        )];
        let (min, max, has_ptx, _) = compute_effective_range(&binaries);
        assert_eq!(min, Some(cc(7, 0)));
        assert_eq!(max, None);
        assert!(has_ptx);
    }

    #[test]
    fn effective_range_restrictive_component() {
        let binaries = vec![
            binary(
                "/lib/torch_cuda.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0), cc(8, 0), cc(9, 0), cc(10, 0)],
                vec![cc(12, 0)],
                vec![],
            ),
            binary(
                "/lib/libnvshmem.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(9, 0), cc(10, 0)],
                vec![],
                vec![],
            ),
        ];
        let (min, max, has_ptx, _) = compute_effective_range(&binaries);
        assert_eq!(min, Some(cc(9, 0)));
        assert_eq!(max, Some(cc(10, 0)));
        assert!(has_ptx);
    }

    #[test]
    fn effective_range_empty() {
        let (min, max, has_ptx, warnings) = compute_effective_range(&[]);
        assert_eq!(min, None);
        assert_eq!(max, None);
        assert!(!has_ptx);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn effective_range_all_ptx() {
        let binaries = vec![binary(
            "/lib/something.so",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![cc(7, 0)],
            vec![],
        )];
        let (min, max, has_ptx, _) = compute_effective_range(&binaries);
        assert_eq!(min, Some(cc(7, 0)));
        assert_eq!(max, None);
        assert!(has_ptx);
    }

    // -- dependency graph building --

    #[test]
    fn graph_empty_roots() {
        let binaries: Vec<BinaryScanResult> = vec![];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(&[], &binaries, &soname_map, &HashMap::new());
        assert!(graph.roots.is_empty());
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn graph_single_root_no_deps() {
        let binaries = vec![binary(
            "/usr/bin/python",
            BinaryPriority::Entrypoint,
            vec![],
            vec![],
            vec![],
        )];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/usr/bin/python".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        assert_eq!(graph.roots.len(), 1);
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.nodes.contains_key("python"));
    }

    #[test]
    fn graph_single_dep_chain() {
        let binaries = vec![
            binary(
                "/usr/bin/app",
                BinaryPriority::Entrypoint,
                vec![],
                vec![],
                vec!["libfoo.so"],
            ),
            binary(
                "/usr/lib/libfoo.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(9, 0)],
                vec![],
                vec!["libbar.so"],
            ),
            binary(
                "/usr/lib/libbar.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0), cc(9, 0)],
                vec![],
                vec![],
            ),
        ];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/usr/bin/app".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        assert_eq!(graph.nodes.len(), 3);
        let app = &graph.nodes["app"];
        assert_eq!(app.deps, vec!["libfoo.so"]);
        let foo = &graph.nodes["libfoo.so"];
        assert_eq!(foo.deps, vec!["libbar.so"]);
    }

    #[test]
    fn graph_diamond_dependency() {
        //   app -> libA, libB
        //   libA -> libCommon
        //   libB -> libCommon
        let binaries = vec![
            binary(
                "/bin/app",
                BinaryPriority::Entrypoint,
                vec![],
                vec![],
                vec!["libA.so", "libB.so"],
            ),
            binary(
                "/lib/libA.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0)],
                vec![],
                vec!["libCommon.so"],
            ),
            binary(
                "/lib/libB.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(9, 0)],
                vec![],
                vec!["libCommon.so"],
            ),
            binary(
                "/lib/libCommon.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0), cc(9, 0)],
                vec![],
                vec![],
            ),
        ];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/bin/app".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        // libCommon should appear once, not duplicated
        assert_eq!(graph.nodes.len(), 4);
        assert!(graph.nodes.contains_key("libCommon.so"));
    }

    #[test]
    fn graph_cycle_handling() {
        // libA -> libB -> libA (cycle)
        let binaries = vec![
            binary(
                "/lib/libA.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(9, 0)],
                vec![],
                vec!["libB.so"],
            ),
            binary(
                "/lib/libB.so",
                BinaryPriority::LinkerLibrary,
                vec![],
                vec![],
                vec!["libA.so"],
            ),
        ];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/lib/libA.so".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        // Should not infinite loop; both nodes present
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn graph_unresolved_sonames() {
        let binaries = vec![binary(
            "/bin/app",
            BinaryPriority::Entrypoint,
            vec![],
            vec![],
            vec!["libdl.so.2", "libcuda.so"],
        )];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/bin/app".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        let app = &graph.nodes["app"];
        // libdl.so.2 and libcuda.so aren't in the image, so they're unresolved
        assert_eq!(app.unresolved.len(), 2);
        assert!(app.unresolved.contains(&"libdl.so.2".to_string()));
    }

    // -- reachable paths --

    #[test]
    fn reachable_paths_from_graph() {
        let binaries = vec![
            binary(
                "/bin/app",
                BinaryPriority::Entrypoint,
                vec![],
                vec![],
                vec!["libfoo.so"],
            ),
            binary(
                "/lib/libfoo.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(9, 0)],
                vec![],
                vec![],
            ),
            binary(
                "/lib/dormant.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0)],
                vec![],
                vec![],
            ),
        ];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/bin/app".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );
        let reachable = collect_reachable_paths(&graph);
        assert!(reachable.contains("/bin/app"));
        assert!(reachable.contains("/lib/libfoo.so"));
        // dormant.so is NOT reachable (not a root, not a dependency)
        assert!(!reachable.contains("/lib/dormant.so"));
    }

    // -- terminal tree rendering --

    #[test]
    fn tree_renders_cuda_nodes() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "app".to_string(),
            dep_node(
                "/bin/app",
                "app",
                vec![],
                vec![],
                vec!["libcuda.so"],
                vec![],
            ),
        );
        nodes.insert(
            "libcuda.so".to_string(),
            dep_node(
                "/lib/libcuda.so",
                "libcuda.so",
                vec![cc(7, 0), cc(9, 0)],
                vec![],
                vec![],
                vec![],
            ),
        );
        let graph = DepGraph {
            roots: vec!["app".to_string()],
            nodes,
        };
        let tree = render_dep_tree(&graph);
        assert!(tree.contains("Dependency tree"));
        assert!(tree.contains("/bin/app"));
        assert!(tree.contains("/lib/libcuda.so"));
        assert!(tree.contains("sm_70"));
    }

    #[test]
    fn tree_shows_unresolved_deps() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "app".to_string(),
            dep_node(
                "/bin/app",
                "app",
                vec![cc(9, 0)],
                vec![],
                vec![],
                vec!["libdl.so.2"],
            ),
        );
        let graph = DepGraph {
            roots: vec!["app".to_string()],
            nodes,
        };
        let tree = render_dep_tree(&graph);
        assert!(tree.contains("unresolved"));
        assert!(tree.contains("libdl.so.2"));
    }

    #[test]
    fn tree_collapses_non_cuda_deps() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "app".to_string(),
            dep_node(
                "/bin/app",
                "app",
                vec![cc(9, 0)],
                vec![],
                vec!["libc.so", "libm.so"],
                vec![],
            ),
        );
        nodes.insert(
            "libc.so".to_string(),
            dep_node("/lib/libc.so", "libc.so", vec![], vec![], vec![], vec![]),
        );
        nodes.insert(
            "libm.so".to_string(),
            dep_node("/lib/libm.so", "libm.so", vec![], vec![], vec![], vec![]),
        );
        let graph = DepGraph {
            roots: vec!["app".to_string()],
            nodes,
        };
        let tree = render_dep_tree(&graph);
        // Non-CUDA deps should be collapsed
        assert!(tree.contains("deps without CUDA fatbins"));
        assert!(!tree.contains("/lib/libc.so"));
    }

    // -- format_report --

    #[test]
    fn report_includes_key_sections() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "torch.so".to_string(),
            dep_node(
                "/lib/torch.so",
                "torch.so",
                vec![cc(7, 0), cc(9, 0)],
                vec![cc(12, 0)],
                vec![],
                vec![],
            ),
        );
        let result = ScanResult {
            image: "ghcr.io/test/image:v1".to_string(),
            metadata: crate::scan::ImageMetadata {
                cuda_version: Some("12.9.1".to_string()),
                torch_arch_list: Some("7.0;9.0".to_string()),
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![binary(
                "/lib/torch.so",
                BinaryPriority::LinkerLibrary,
                vec![cc(7, 0), cc(9, 0)],
                vec![cc(12, 0)],
                vec![],
            )],
            effective_cc_min: Some(cc(7, 0)),
            effective_cc_max: None,
            has_ptx_forward_compat: true,
            warnings: vec![],
            dep_graph: Some(DepGraph {
                roots: vec!["torch.so".to_string()],
                nodes,
            }),
            reachable_count: 1,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
        };
        let report = format_report(&result);
        assert!(report.contains("ghcr.io/test/image:v1"));
        assert!(report.contains("CUDA_VERSION"));
        assert!(report.contains("12.9.1"));
        assert!(report.contains("sm_70"));
        assert!(report.contains("cc_min = 7.0"));
        assert!(report.contains("PTX forward compat"));
        assert!(report.contains("1 reachable binaries"));
    }

    #[test]
    fn report_no_binaries() {
        let result = ScanResult {
            image: "test:latest".to_string(),
            metadata: crate::scan::ImageMetadata {
                cuda_version: None,
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![],
            effective_cc_min: None,
            effective_cc_max: None,
            has_ptx_forward_compat: false,
            warnings: vec!["no CUDA binaries found".to_string()],
            dep_graph: None,
            reachable_count: 0,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
        };
        let report = format_report(&result);
        assert!(report.contains("(none found)"));
        assert!(report.contains("0 reachable binaries"));
    }

    #[test]
    fn report_shows_dormant_count() {
        let result = ScanResult {
            image: "test:latest".to_string(),
            metadata: crate::scan::ImageMetadata {
                cuda_version: None,
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![],
            effective_cc_min: Some(cc(9, 0)),
            effective_cc_max: Some(cc(9, 0)),
            has_ptx_forward_compat: false,
            warnings: vec![],
            dep_graph: None,
            reachable_count: 5,
            dormant_count: 42,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
        };
        let report = format_report(&result);
        assert!(report.contains("Dormant (not reachable from entrypoint): 42 binaries"));
    }

    // -- soname-based resolution --

    #[test]
    fn soname_map_uses_real_soname() {
        let mut b = binary(
            "/usr/lib/x86_64-linux-gnu/libcudart.so.12.9.37",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        b.soname = Some("libcudart.so.12".to_string());

        let (map, _) = build_soname_maps(&[b]);
        // Should be keyed by DT_SONAME, not filename
        assert!(map.contains_key("libcudart.so.12"));
        assert!(!map.contains_key("libcudart.so.12.9.37"));
    }

    #[test]
    fn soname_map_falls_back_to_filename() {
        let b = binary(
            "/usr/lib/libfoo.so.1",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        // soname is None by default from the helper
        let (map, _) = build_soname_maps(&[b]);
        assert!(map.contains_key("libfoo.so.1"));
    }

    #[test]
    fn dep_graph_resolves_via_soname() {
        let mut lib = binary(
            "/usr/lib/libcudart.so.12.9.37",
            BinaryPriority::LinkerLibrary,
            vec![cc(9, 0)],
            vec![],
            vec![],
        );
        lib.soname = Some("libcudart.so.12".to_string());

        let app = binary(
            "/bin/app",
            BinaryPriority::Entrypoint,
            vec![],
            vec![],
            vec!["libcudart.so.12"],
        );

        let binaries = vec![app, lib];
        let (soname_map, _) = build_soname_maps(&binaries);
        let graph = build_dep_graph(
            &["/bin/app".to_string()],
            &binaries,
            &soname_map,
            &HashMap::new(),
        );

        let app_node = &graph.nodes["app"];
        // Should resolve via DT_SONAME
        assert_eq!(app_node.deps, vec!["libcudart.so.12"]);
        assert!(app_node.unresolved.is_empty());
        assert!(graph.nodes.contains_key("libcudart.so.12"));
    }

    // -- RPATH/RUNPATH in dep tree rendering --

    #[test]
    fn tree_shows_rpath_annotation() {
        let mut nodes = HashMap::new();
        nodes.insert("app".to_string(), {
            let mut n = dep_node("/bin/app", "app", vec![cc(9, 0)], vec![], vec![], vec![]);
            n.runpath = vec!["/usr/local/cuda/lib64".to_string()];
            n
        });
        let graph = DepGraph {
            roots: vec!["app".to_string()],
            nodes,
        };
        let tree = render_dep_tree(&graph);
        assert!(tree.contains("RUNPATH=/usr/local/cuda/lib64"));
    }

    #[test]
    fn tree_shows_deprecated_rpath() {
        let mut nodes = HashMap::new();
        nodes.insert("app".to_string(), {
            let mut n = dep_node("/bin/app", "app", vec![cc(9, 0)], vec![], vec![], vec![]);
            n.rpath = vec!["/opt/lib".to_string()];
            n
        });
        let graph = DepGraph {
            roots: vec!["app".to_string()],
            nodes,
        };
        let tree = render_dep_tree(&graph);
        assert!(tree.contains("RPATH=/opt/lib"));
    }

    #[test]
    fn report_flags_deprecated_rpath() {
        let mut nodes = HashMap::new();
        nodes.insert("torch.so".to_string(), {
            let mut n = dep_node(
                "/lib/torch.so",
                "torch.so",
                vec![cc(9, 0)],
                vec![],
                vec![],
                vec![],
            );
            n.rpath = vec!["/usr/local/cuda/lib64".to_string()];
            n
        });
        let result = ScanResult {
            image: "test:latest".to_string(),
            metadata: crate::scan::ImageMetadata {
                cuda_version: None,
                torch_arch_list: None,
                nvidia_require: None,
                nvshmem_architectures: None,
            },
            binaries: vec![],
            effective_cc_min: Some(cc(9, 0)),
            effective_cc_max: Some(cc(9, 0)),
            has_ptx_forward_compat: false,
            warnings: vec![],
            dep_graph: Some(DepGraph {
                roots: vec!["torch.so".to_string()],
                nodes,
            }),
            reachable_count: 1,
            dormant_count: 0,
            environment: empty_env(),
            labels: HashMap::new(),
            env_vars: vec![],
        };
        let report = format_report(&result);
        assert!(report.contains("Resolution issues:"));
        assert!(report.contains("deprecated DT_RPATH"));
    }

    // -- os-release parser --

    #[test]
    fn parse_os_release_ubuntu() {
        let content = r#"NAME="Ubuntu"
VERSION="22.04.4 LTS (Jammy Jellyfish)"
ID=ubuntu
VERSION_ID="22.04"
PRETTY_NAME="Ubuntu 22.04.4 LTS"
HOME_URL="https://www.ubuntu.com/"
"#;
        let info = parse_os_release(content).unwrap();
        assert_eq!(info.id, "ubuntu");
        assert_eq!(info.version_id, "22.04");
        assert_eq!(info.pretty_name, "Ubuntu 22.04.4 LTS");
    }

    #[test]
    fn parse_os_release_ubi() {
        let content = r#"NAME="Red Hat Enterprise Linux"
VERSION="9.4 (Plow)"
ID="rhel"
VERSION_ID="9.4"
PRETTY_NAME="Red Hat Enterprise Linux 9.4 (Plow)"
"#;
        let info = parse_os_release(content).unwrap();
        assert_eq!(info.id, "rhel");
        assert_eq!(info.version_id, "9.4");
    }

    #[test]
    fn parse_os_release_alpine_no_pretty_name() {
        let content = "ID=alpine\nVERSION_ID=3.19\n";
        let info = parse_os_release(content).unwrap();
        assert_eq!(info.id, "alpine");
        assert_eq!(info.version_id, "3.19");
        assert_eq!(info.pretty_name, "");
    }

    #[test]
    fn parse_os_release_empty() {
        let info = parse_os_release("").unwrap();
        assert_eq!(info.id, "");
    }

    #[test]
    fn parse_os_release_comments() {
        let content = "# comment\nID=test\n# another comment\n";
        let info = parse_os_release(content).unwrap();
        assert_eq!(info.id, "test");
    }

    // -- dpkg status parser --

    #[test]
    fn parse_dpkg_single_package() {
        let content = "Package: libcudart12\nVersion: 12.9.37-1\nStatus: install ok installed\n";
        let pkgs = parse_dpkg_status(content);
        assert_eq!(pkgs.len(), 1);
        assert_eq!(pkgs[0].name, "libcudart12");
        assert_eq!(pkgs[0].version, "12.9.37-1");
        assert_eq!(pkgs[0].source, None);
    }

    #[test]
    fn parse_dpkg_with_source() {
        let content = "\
Package: libcudart12
Version: 12.9.37-1
Source: cuda-toolkit-12-9 (12.9.37-1)
Status: install ok installed
";
        let pkgs = parse_dpkg_status(content);
        assert_eq!(pkgs.len(), 1);
        assert_eq!(pkgs[0].name, "libcudart12");
        assert_eq!(pkgs[0].source.as_deref(), Some("cuda-toolkit-12-9"));
    }

    #[test]
    fn parse_dpkg_source_without_version() {
        let content =
            "Package: libnccl2\nVersion: 2.25.1-1\nSource: nccl\nStatus: install ok installed\n";
        let pkgs = parse_dpkg_status(content);
        assert_eq!(pkgs.len(), 1);
        assert_eq!(pkgs[0].source.as_deref(), Some("nccl"));
    }

    #[test]
    fn parse_dpkg_multiple_packages() {
        let content = "\
Package: libcudnn9-cuda-12
Version: 9.7.1.1-1
Status: install ok installed

Package: libnccl2
Version: 2.25.1-1+cuda12.9
Status: install ok installed
";
        let pkgs = parse_dpkg_status(content);
        assert_eq!(pkgs.len(), 2);
        assert_eq!(pkgs[0].name, "libcudnn9-cuda-12");
        assert_eq!(pkgs[1].name, "libnccl2");
    }

    #[test]
    fn parse_dpkg_partial_block() {
        // Block with only Package but no Version should be skipped
        let content = "Package: broken\nStatus: install ok installed\n";
        let pkgs = parse_dpkg_status(content);
        assert!(pkgs.is_empty());
    }

    #[test]
    fn parse_dpkg_empty() {
        let pkgs = parse_dpkg_status("");
        assert!(pkgs.is_empty());
    }

    // -- ld.so.conf parser --

    #[test]
    fn parse_ld_conf_paths() {
        let content = "/usr/local/cuda/lib64\n/usr/lib/x86_64-linux-gnu\n";
        let paths = parse_ld_so_conf(content);
        assert_eq!(
            paths,
            vec!["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
        );
    }

    #[test]
    fn parse_ld_conf_with_include() {
        let content = "include /etc/ld.so.conf.d/*.conf\n/usr/lib\n";
        let paths = parse_ld_so_conf(content);
        assert_eq!(paths, vec!["/usr/lib"]);
    }

    #[test]
    fn parse_ld_conf_comments_and_empty() {
        let content = "# CUDA paths\n\n/opt/cuda/lib64\n# end\n";
        let paths = parse_ld_so_conf(content);
        assert_eq!(paths, vec!["/opt/cuda/lib64"]);
    }

    #[test]
    fn parse_ld_conf_empty() {
        let paths = parse_ld_so_conf("");
        assert!(paths.is_empty());
    }

    // -- environment in report --

    #[test]
    fn report_shows_environment() {
        let mut env = empty_env();
        env.os = Some(crate::scan::OsInfo {
            id: "ubuntu".to_string(),
            version_id: "22.04".to_string(),
            pretty_name: "Ubuntu 22.04.4 LTS".to_string(),
        });
        env.python_environments = vec![crate::scan::PythonEnvironment {
            label: "/opt/vllm".to_string(),
            site_packages_dir: "/opt/vllm/lib/python3.12/site-packages".to_string(),
            packages: vec![
                crate::scan::PackageVersion {
                    name: "torch".to_string(),
                    version: "2.5.1+cu124".to_string(),
                    source: None,
                },
                crate::scan::PackageVersion {
                    name: "numpy".to_string(),
                    version: "1.26.4".to_string(),
                    source: None,
                },
            ],
        }];

        let result = ScanResult {
            image: "test:latest".to_string(),
            metadata: crate::scan::ImageMetadata {
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
            environment: env,
            labels: HashMap::new(),
            env_vars: vec![],
        };
        let report = format_report(&result);
        assert!(report.contains("Environment:"));
        assert!(report.contains("Ubuntu 22.04.4 LTS"));
        assert!(report.contains("Python (/opt/vllm) = 2 packages"));
    }

    // -- python_env_label --

    #[test]
    fn env_label_virtualenv() {
        assert_eq!(
            python_env_label("/opt/vllm/lib/python3.12/site-packages"),
            "/opt/vllm"
        );
    }

    #[test]
    fn env_label_system_usr() {
        assert_eq!(
            python_env_label("/usr/lib/python3.12/site-packages"),
            "system"
        );
    }

    #[test]
    fn env_label_system_usr_local() {
        assert_eq!(
            python_env_label("/usr/local/lib/python3.12/site-packages"),
            "system"
        );
    }

    #[test]
    fn env_label_no_lib_python() {
        assert_eq!(
            python_env_label("/some/weird/site-packages"),
            "/some/weird/site-packages"
        );
    }

    // -- soname collision detection --

    #[test]
    fn soname_collision_detected() {
        let mut a = binary(
            "/usr/lib/libcudart.so.12.9.37",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        a.soname = Some("libcudart.so.12".to_string());

        let mut b = binary(
            "/usr/local/cuda/lib64/libcudart.so.12.8.0",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        b.soname = Some("libcudart.so.12".to_string());

        let (_, index) = crate::scan::build_soname_maps(&[a, b]);
        let warnings = crate::scan::detect_soname_collisions(&index);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("soname collision"));
        assert!(warnings[0].contains("libcudart.so.12"));
    }

    #[test]
    fn no_collision_for_unique_sonames() {
        let mut a = binary(
            "/usr/lib/libcudart.so.12",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        a.soname = Some("libcudart.so.12".to_string());

        let mut b = binary(
            "/usr/lib/libcudnn.so.9",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        b.soname = Some("libcudnn.so.9".to_string());

        let (_, index) = crate::scan::build_soname_maps(&[a, b]);
        let warnings = crate::scan::detect_soname_collisions(&index);
        assert!(warnings.is_empty());
    }

    #[test]
    fn no_collision_for_same_directory() {
        let mut a = binary(
            "/usr/lib64/libfoo.so.1.0",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        a.soname = Some("libfoo.so.1".to_string());

        let mut b = binary(
            "/usr/lib64/libfoo.so.1.1",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        b.soname = Some("libfoo.so.1".to_string());

        let (_, index) = crate::scan::build_soname_maps(&[a, b]);
        let warnings = crate::scan::detect_soname_collisions(&index);
        assert!(
            warnings.is_empty(),
            "same-directory collisions should be filtered out"
        );
    }

    #[test]
    fn no_collision_for_duplicate_paths() {
        let mut a = binary(
            "/usr/lib64/gconv/ANSI_X3.110.so",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        a.soname = None;

        let mut b = binary(
            "/usr/lib64/gconv/ANSI_X3.110.so",
            BinaryPriority::LinkerLibrary,
            vec![],
            vec![],
            vec![],
        );
        b.soname = None;

        let (_, index) = crate::scan::build_soname_maps(&[a, b]);
        // Duplicate paths should be deduplicated in the index
        assert_eq!(index.get("ANSI_X3.110.so").map(|v| v.len()), Some(1));
        let warnings = crate::scan::detect_soname_collisions(&index);
        assert!(warnings.is_empty());
    }

    // -- RPM header parser tests --

    /// Build a minimal RPM header blob with the given name, version, release.
    fn build_rpm_header_blob(name: &str, version: &str, release: &str) -> Vec<u8> {
        // 3 index entries (name, version, release), each 16 bytes
        let nindex: u32 = 3;
        // Data store: 3 null-terminated strings packed sequentially
        let name_bytes = name.as_bytes();
        let version_bytes = version.as_bytes();
        let release_bytes = release.as_bytes();
        let hsize: u32 =
            (name_bytes.len() + 1 + version_bytes.len() + 1 + release_bytes.len() + 1) as u32;

        let mut blob = Vec::new();

        // [0..4] magic, [4..8] reserved, [8..12] nindex, [12..16] hsize
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&[0u8; 4]); // reserved
        blob.extend_from_slice(&nindex.to_be_bytes());
        blob.extend_from_slice(&hsize.to_be_bytes());

        // Index entries: tag(4) + type(4) + offset(4) + count(4)
        let name_offset: u32 = 0;
        let version_offset: u32 = (name_bytes.len() + 1) as u32;
        let release_offset: u32 = version_offset + (version_bytes.len() + 1) as u32;

        // Name entry
        blob.extend_from_slice(&RPMTAG_NAME.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes()); // type STRING
        blob.extend_from_slice(&name_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes()); // count

        // Version entry
        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&version_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Release entry
        blob.extend_from_slice(&RPMTAG_RELEASE.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&release_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Data store: null-terminated strings
        blob.extend_from_slice(name_bytes);
        blob.push(0);
        blob.extend_from_slice(version_bytes);
        blob.push(0);
        blob.extend_from_slice(release_bytes);
        blob.push(0);

        blob
    }

    fn build_rpm_header_blob_with_vendor(
        name: &str,
        version: &str,
        release: &str,
        vendor: Option<&str>,
    ) -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let version_bytes = version.as_bytes();
        let release_bytes = release.as_bytes();
        let vendor_bytes = vendor.map(|v| v.as_bytes());

        let nindex: u32 = if vendor.is_some() { 4 } else { 3 };
        let mut store_size =
            name_bytes.len() + 1 + version_bytes.len() + 1 + release_bytes.len() + 1;
        if let Some(vb) = vendor_bytes {
            store_size += vb.len() + 1;
        }
        let hsize = store_size as u32;

        let mut blob = Vec::new();
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&[0u8; 4]);
        blob.extend_from_slice(&nindex.to_be_bytes());
        blob.extend_from_slice(&hsize.to_be_bytes());

        let name_offset: u32 = 0;
        let version_offset: u32 = (name_bytes.len() + 1) as u32;
        let release_offset: u32 = version_offset + (version_bytes.len() + 1) as u32;
        let vendor_offset: u32 = release_offset + (release_bytes.len() + 1) as u32;

        // Index entries
        blob.extend_from_slice(&RPMTAG_NAME.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&name_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&version_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        blob.extend_from_slice(&RPMTAG_RELEASE.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&release_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        if vendor.is_some() {
            blob.extend_from_slice(&RPMTAG_VENDOR.to_be_bytes());
            blob.extend_from_slice(&6u32.to_be_bytes());
            blob.extend_from_slice(&vendor_offset.to_be_bytes());
            blob.extend_from_slice(&1u32.to_be_bytes());
        }

        // Data store
        blob.extend_from_slice(name_bytes);
        blob.push(0);
        blob.extend_from_slice(version_bytes);
        blob.push(0);
        blob.extend_from_slice(release_bytes);
        blob.push(0);
        if let Some(vb) = vendor_bytes {
            blob.extend_from_slice(vb);
            blob.push(0);
        }

        blob
    }

    #[test]
    fn rpm_header_blob_valid() {
        let blob = build_rpm_header_blob("test-pkg", "1.0", "1.el9");
        let pkg = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(pkg.package.name, "test-pkg");
        assert_eq!(pkg.package.version, "1.0-1.el9");
        assert_eq!(pkg.package.source, None);
    }

    #[test]
    fn rpm_header_blob_with_vendor() {
        let blob = build_rpm_header_blob_with_vendor(
            "cuda-cudart",
            "12.9",
            "1.el9",
            Some("NVIDIA CORPORATION"),
        );
        let pkg = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(pkg.package.name, "cuda-cudart");
        assert_eq!(pkg.package.version, "12.9-1.el9");
        assert_eq!(pkg.package.source.as_deref(), Some("NVIDIA CORPORATION"));
    }

    #[test]
    fn rpm_header_blob_no_vendor() {
        let blob = build_rpm_header_blob_with_vendor("bash", "5.1", "1.el9", None);
        let pkg = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(pkg.package.name, "bash");
        assert_eq!(pkg.package.source, None);
    }

    #[test]
    fn rpm_header_blob_raw_no_magic() {
        // Real rpmdb.sqlite blobs omit the magic+reserved prefix.
        // Build a blob without it: [nindex(4)] [hsize(4)] [entries] [store]
        let mut blob = Vec::new();
        let nindex: u32 = 3;
        let name = b"bash";
        let ver = b"5.1.8";
        let rel = b"9.el9";
        let hsize: u32 = (name.len() + 1 + ver.len() + 1 + rel.len() + 1) as u32;

        blob.extend_from_slice(&nindex.to_be_bytes());
        blob.extend_from_slice(&hsize.to_be_bytes());

        let name_off: u32 = 0;
        let ver_off: u32 = (name.len() + 1) as u32;
        let rel_off: u32 = ver_off + (ver.len() + 1) as u32;

        // Name
        blob.extend_from_slice(&RPMTAG_NAME.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&name_off.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());
        // Version
        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&ver_off.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());
        // Release
        blob.extend_from_slice(&RPMTAG_RELEASE.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&rel_off.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Data store
        blob.extend_from_slice(name);
        blob.push(0);
        blob.extend_from_slice(ver);
        blob.push(0);
        blob.extend_from_slice(rel);
        blob.push(0);

        let pkg = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(pkg.package.name, "bash");
        assert_eq!(pkg.package.version, "5.1.8-9.el9");
    }

    #[test]
    fn rpm_header_blob_no_release() {
        // Build a blob with only name + version (no release tag)
        let mut blob = Vec::new();
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&[0u8; 4]); // reserved
        blob.extend_from_slice(&2u32.to_be_bytes()); // 2 entries
        let hsize: u32 = 4 + 1 + 3 + 1; // "test" + \0 + "2.0" + \0
        blob.extend_from_slice(&hsize.to_be_bytes());

        // Name at offset 0
        blob.extend_from_slice(&RPMTAG_NAME.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Version at offset 5
        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&5u32.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Data store
        blob.extend_from_slice(b"test\0");
        blob.extend_from_slice(b"2.0\0");

        let pkg = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(pkg.package.name, "test");
        assert_eq!(pkg.package.version, "2.0");
    }

    #[test]
    fn rpm_header_blob_truncated() {
        assert!(parse_rpm_header_blob(&[]).is_none());
        assert!(parse_rpm_header_blob(&[0x8e, 0xad, 0xe8]).is_none());
        // Valid magic but declares more data than available
        let mut blob = Vec::new();
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&100u32.to_be_bytes()); // 100 entries
        blob.extend_from_slice(&100u32.to_be_bytes()); // 100 bytes store
                                                       // Not enough data follows
        assert!(parse_rpm_header_blob(&blob).is_none());
    }

    #[test]
    fn rpm_header_blob_wrong_magic() {
        let mut blob = build_rpm_header_blob("pkg", "1.0", "1");
        blob[0] = 0x00; // corrupt magic
        assert!(parse_rpm_header_blob(&blob).is_none());
    }

    #[test]
    fn rpm_header_blob_missing_name() {
        // Only version + release, no name tag
        let mut blob = Vec::new();
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&[0u8; 4]); // reserved
        blob.extend_from_slice(&2u32.to_be_bytes());
        let hsize: u32 = 3 + 1 + 1 + 1; // "1.0" + \0 + "1" + \0
        blob.extend_from_slice(&hsize.to_be_bytes());

        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        blob.extend_from_slice(&RPMTAG_RELEASE.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&4u32.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        blob.extend_from_slice(b"1.0\0");
        blob.extend_from_slice(b"1\0");

        assert!(parse_rpm_header_blob(&blob).is_none());
    }

    #[test]
    fn rpm_database_roundtrip() {
        // Create a real SQLite DB with synthetic RPM header blobs
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("rpmdb.sqlite");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute(
                "CREATE TABLE Packages (hnum INTEGER PRIMARY KEY, blob BLOB NOT NULL)",
                [],
            )
            .unwrap();

            let blob1 = build_rpm_header_blob("glibc", "2.34", "60.el9");
            let blob2 = build_rpm_header_blob("cuda-toolkit", "12.4.1", "1.x86_64");
            let blob3 = build_rpm_header_blob("openssl", "3.0.7", "27.el9");

            conn.execute("INSERT INTO Packages (hnum, blob) VALUES (1, ?1)", [&blob1])
                .unwrap();
            conn.execute("INSERT INTO Packages (hnum, blob) VALUES (2, ?1)", [&blob2])
                .unwrap();
            conn.execute("INSERT INTO Packages (hnum, blob) VALUES (3, ?1)", [&blob3])
                .unwrap();
        }

        let db_bytes = std::fs::read(&db_path).unwrap();
        let out_path = tmp.path().join("rpmdb_test.sqlite");
        let packages = parse_rpm_database_at(&db_bytes, &out_path);

        assert_eq!(packages.len(), 3);

        let names: Vec<&str> = packages.iter().map(|p| p.package.name.as_str()).collect();
        assert!(names.contains(&"glibc"));
        assert!(names.contains(&"cuda-toolkit"));
        assert!(names.contains(&"openssl"));

        let glibc = packages.iter().find(|p| p.package.name == "glibc").unwrap();
        assert_eq!(glibc.package.version, "2.34-60.el9");
    }

    #[test]
    fn rpm_database_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("rpmdb.sqlite");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute(
                "CREATE TABLE Packages (hnum INTEGER PRIMARY KEY, blob BLOB NOT NULL)",
                [],
            )
            .unwrap();
        }

        let db_bytes = std::fs::read(&db_path).unwrap();
        let out_path = tmp.path().join("rpmdb_test.sqlite");
        let packages = parse_rpm_database_at(&db_bytes, &out_path);
        assert!(packages.is_empty());
    }

    #[test]
    fn rpm_database_invalid_blobs() {
        // DB with blobs that aren't valid RPM headers should be skipped gracefully
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("rpmdb.sqlite");

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute(
                "CREATE TABLE Packages (hnum INTEGER PRIMARY KEY, blob BLOB NOT NULL)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO Packages (hnum, blob) VALUES (1, X'deadbeef')",
                [],
            )
            .unwrap();
            // One valid blob among the garbage
            let valid = build_rpm_header_blob("valid-pkg", "1.0", "1");
            conn.execute("INSERT INTO Packages (hnum, blob) VALUES (2, ?1)", [&valid])
                .unwrap();
        }

        let db_bytes = std::fs::read(&db_path).unwrap();
        let out_path = tmp.path().join("rpmdb_test.sqlite");
        let packages = parse_rpm_database_at(&db_bytes, &out_path);
        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0].package.name, "valid-pkg");
    }

    // -- read_string_array / read_u32_array helpers --

    #[test]
    fn read_string_array_basic() {
        let data = b"hello\0world\0foo\0";
        let result = read_string_array(data, 0, 3);
        assert_eq!(result, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn read_u32_array_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_be_bytes());
        data.extend_from_slice(&100u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        let result = read_u32_array(&data, 0, 3);
        assert_eq!(result, vec![42, 100, 0]);
    }

    // -- RPM header blob with file ownership --

    /// Build an RPM header blob that includes BASENAMES, DIRNAMES, and DIRINDEXES tags.
    fn build_rpm_header_blob_with_files(
        name: &str,
        version: &str,
        release: &str,
        dirnames: &[&str],
        basenames: &[&str],
        dir_indexes: &[u32],
    ) -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let version_bytes = version.as_bytes();
        let release_bytes = release.as_bytes();

        // Build data store: name\0 version\0 release\0 basenames...\0 dirnames...\0 dirindexes(u32 BE)
        let mut store = Vec::new();

        let name_offset = store.len() as u32;
        store.extend_from_slice(name_bytes);
        store.push(0);

        let version_offset = store.len() as u32;
        store.extend_from_slice(version_bytes);
        store.push(0);

        let release_offset = store.len() as u32;
        store.extend_from_slice(release_bytes);
        store.push(0);

        let basenames_offset = store.len() as u32;
        for bn in basenames {
            store.extend_from_slice(bn.as_bytes());
            store.push(0);
        }

        let dirnames_offset = store.len() as u32;
        for dn in dirnames {
            store.extend_from_slice(dn.as_bytes());
            store.push(0);
        }

        let dirindexes_offset = store.len() as u32;
        for &idx in dir_indexes {
            store.extend_from_slice(&idx.to_be_bytes());
        }

        let nindex: u32 = 6; // name, version, release, basenames, dirnames, dirindexes
        let hsize = store.len() as u32;

        let mut blob = Vec::new();
        blob.extend_from_slice(&RPM_HEADER_MAGIC);
        blob.extend_from_slice(&[0u8; 4]); // reserved
        blob.extend_from_slice(&nindex.to_be_bytes());
        blob.extend_from_slice(&hsize.to_be_bytes());

        // Index entries (tag, type, offset, count)
        // Name (STRING type = 6)
        blob.extend_from_slice(&RPMTAG_NAME.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&name_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Version
        blob.extend_from_slice(&RPMTAG_VERSION.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&version_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Release
        blob.extend_from_slice(&RPMTAG_RELEASE.to_be_bytes());
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&release_offset.to_be_bytes());
        blob.extend_from_slice(&1u32.to_be_bytes());

        // Basenames (STRING_ARRAY type = 8)
        blob.extend_from_slice(&RPMTAG_BASENAMES.to_be_bytes());
        blob.extend_from_slice(&RPM_TYPE_STRING_ARRAY.to_be_bytes());
        blob.extend_from_slice(&basenames_offset.to_be_bytes());
        blob.extend_from_slice(&(basenames.len() as u32).to_be_bytes());

        // Dirnames (STRING_ARRAY type = 8)
        blob.extend_from_slice(&RPMTAG_DIRNAMES.to_be_bytes());
        blob.extend_from_slice(&RPM_TYPE_STRING_ARRAY.to_be_bytes());
        blob.extend_from_slice(&dirnames_offset.to_be_bytes());
        blob.extend_from_slice(&(dirnames.len() as u32).to_be_bytes());

        // Dirindexes (INT32 type = 4)
        blob.extend_from_slice(&RPMTAG_DIRINDEXES.to_be_bytes());
        blob.extend_from_slice(&RPM_TYPE_INT32.to_be_bytes());
        blob.extend_from_slice(&dirindexes_offset.to_be_bytes());
        blob.extend_from_slice(&(dir_indexes.len() as u32).to_be_bytes());

        // Data store
        blob.extend_from_slice(&store);

        blob
    }

    #[test]
    fn rpm_header_blob_with_files() {
        let blob = build_rpm_header_blob_with_files(
            "libfoo",
            "1.2",
            "3.el9",
            &["/usr/lib64/", "/usr/bin/"],
            &["libfoo.so.1", "foo-tool"],
            &[0, 1], // libfoo.so.1 -> /usr/lib64/, foo-tool -> /usr/bin/
        );
        let info = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(info.package.name, "libfoo");
        assert_eq!(info.package.version, "1.2-3.el9");
        assert_eq!(info.files.len(), 2);
        assert!(info.files.contains(&"/usr/lib64/libfoo.so.1".to_string()));
        assert!(info.files.contains(&"/usr/bin/foo-tool".to_string()));
    }

    #[test]
    fn rpm_header_blob_without_file_tags() {
        let blob = build_rpm_header_blob("nfiles-pkg", "2.0", "1.el9");
        let info = parse_rpm_header_blob(&blob).unwrap();
        assert_eq!(info.package.name, "nfiles-pkg");
        assert_eq!(info.package.version, "2.0-1.el9");
        assert!(info.files.is_empty());
    }

    // -- dpkg file ownership --

    #[test]
    fn dpkg_file_owners_resolved() {
        use crate::image::DiscoveredMetadata;

        let metadata = vec![
            DiscoveredMetadata::DpkgStatus(
                "Package: libfoo\nVersion: 1.0\n\nPackage: other\nVersion: 2.0\n".to_string(),
            ),
            DiscoveredMetadata::DpkgFileList {
                package_name: "libfoo".to_string(),
                content: "/usr/lib/libfoo.so\n/usr/lib/libfoo.so.1\n".to_string(),
            },
        ];

        let env = collect_environment(metadata);
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so"),
            Some(&"libfoo 1.0".to_string())
        );
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so.1"),
            Some(&"libfoo 1.0".to_string())
        );
    }

    #[test]
    fn dpkg_file_list_before_status() {
        use crate::image::DiscoveredMetadata;

        // DpkgFileList comes before DpkgStatus in the vec; version resolution
        // should still work because we defer the join until after all metadata
        // has been processed.
        let metadata = vec![
            DiscoveredMetadata::DpkgFileList {
                package_name: "libfoo".to_string(),
                content: "/usr/lib/libfoo.so\n/usr/lib/libfoo.so.1\n".to_string(),
            },
            DiscoveredMetadata::DpkgStatus("Package: libfoo\nVersion: 1.0\n".to_string()),
        ];

        let env = collect_environment(metadata);
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so"),
            Some(&"libfoo 1.0".to_string())
        );
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so.1"),
            Some(&"libfoo 1.0".to_string())
        );
    }

    // -- Symlink ownership propagation --

    #[test]
    fn symlink_ownership_propagated() {
        use crate::image::DiscoveredMetadata;

        let metadata = vec![
            DiscoveredMetadata::DpkgStatus("Package: libfoo\nVersion: 1.0\n".to_string()),
            DiscoveredMetadata::DpkgFileList {
                package_name: "libfoo".to_string(),
                content: "/usr/lib/libfoo.so.1.2.3\n".to_string(),
            },
            DiscoveredMetadata::Symlink {
                path: "/usr/lib/libfoo.so.1".to_string(),
                target: "libfoo.so.1.2.3".to_string(),
            },
        ];

        let env = collect_environment(metadata);
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so.1.2.3"),
            Some(&"libfoo 1.0".to_string())
        );
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so.1"),
            Some(&"libfoo 1.0".to_string()),
            "symlink should inherit ownership from its target"
        );
    }

    #[test]
    fn symlink_already_owned_not_overwritten() {
        use crate::image::DiscoveredMetadata;

        let metadata = vec![
            DiscoveredMetadata::DpkgStatus(
                "Package: libfoo\nVersion: 1.0\n\nPackage: libbar\nVersion: 2.0\n".to_string(),
            ),
            DiscoveredMetadata::DpkgFileList {
                package_name: "libfoo".to_string(),
                content: "/usr/lib/libfoo.so.1.2.3\n".to_string(),
            },
            DiscoveredMetadata::DpkgFileList {
                package_name: "libbar".to_string(),
                content: "/usr/lib/libfoo.so.1\n".to_string(),
            },
            DiscoveredMetadata::Symlink {
                path: "/usr/lib/libfoo.so.1".to_string(),
                target: "libfoo.so.1.2.3".to_string(),
            },
        ];

        let env = collect_environment(metadata);
        // The symlink path is already owned by libbar via its .list file,
        // so symlink resolution should NOT overwrite it.
        assert_eq!(
            env.file_owners.get("/usr/lib/libfoo.so.1"),
            Some(&"libbar 2.0".to_string()),
            "existing ownership should not be overwritten by symlink resolution"
        );
    }

    #[test]
    fn uv_cache_symlinks_excluded() {
        use crate::image::DiscoveredMetadata;

        let metadata = vec![
            DiscoveredMetadata::Symlink {
                path: "/usr/lib/libfoo.so.1".to_string(),
                target: "libfoo.so.1.2.3".to_string(),
            },
            DiscoveredMetadata::Symlink {
                path: "/root/.cache/uv/archive-v0/X40fsdviGxksd7EP64z_a/cuda/core/experimental/_utils/cuda_utils.py".to_string(),
                target: "opt/vllm/lib/python3.12/site-packages/cuda/core/experimental/_utils/cuda_utils.py".to_string(),
            },
            DiscoveredMetadata::Symlink {
                path: "/home/user/.cache/uv/some-other-thing".to_string(),
                target: "whatever".to_string(),
            },
        ];

        let env = collect_environment(metadata);
        assert_eq!(
            env.symlinks.len(),
            1,
            "only the non-uv symlink should survive"
        );
        assert!(env.symlinks.contains_key("/usr/lib/libfoo.so.1"));
    }

    #[test]
    fn uv_cache_symlinks_still_propagate_ownership() {
        use crate::image::DiscoveredMetadata;

        // A uv cache symlink should still participate in ownership propagation
        // even though it gets stripped from the final symlinks map.
        // The ownership propagation runs before the filter, so the link path
        // should pick up ownership from its target.
        let metadata = vec![
            DiscoveredMetadata::DpkgStatus("Package: cuda-core\nVersion: 12.0\n".to_string()),
            DiscoveredMetadata::DpkgFileList {
                package_name: "cuda-core".to_string(),
                content: "/opt/vllm/lib/python3.12/site-packages/cuda/core/utils.py\n".to_string(),
            },
            DiscoveredMetadata::Symlink {
                path: "/root/.cache/uv/archive-v0/abc123/cuda/core/utils.py".to_string(),
                target: "/opt/vllm/lib/python3.12/site-packages/cuda/core/utils.py".to_string(),
            },
        ];

        let env = collect_environment(metadata);
        // The uv symlink path should have inherited ownership from the target
        assert_eq!(
            env.file_owners
                .get("/root/.cache/uv/archive-v0/abc123/cuda/core/utils.py"),
            Some(&"cuda-core 12.0".to_string()),
            "uv cache symlink should still get ownership propagated"
        );
        // But it should NOT appear in the symlinks display map
        assert!(
            !env.symlinks
                .contains_key("/root/.cache/uv/archive-v0/abc123/cuda/core/utils.py"),
            "uv cache symlink should be filtered from the symlinks map"
        );
    }
}
