use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use oci_client::secrets::RegistryAuth;
use oci_client::{Client, Reference};
use tokio_util::io::SyncIoBridge;

use sha2::{Digest, Sha256};

use crate::cache::{BlobCache, CachedMetadata, LayerCache, LayerManifest, ManifestBinary};
use crate::progress::ScanProgress;

/// Max file size we'll extract from a layer (2 GiB). Anything larger gets skipped.
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Default linker library paths to scan when LD_LIBRARY_PATH is empty.
const DEFAULT_LIB_DIRS: &[&str] = &[
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib64",
    "/usr/local/lib",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
];

/// Default Python site-packages path prefix.
const PYTHON_SITE_PACKAGES: &str = "site-packages/";

/// Priority classification for discovered binaries.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum BinaryPriority {
    /// Image entrypoint or cmd binary.
    Entrypoint,
    /// On the image's $PATH.
    PathBinary,
    /// In LD_LIBRARY_PATH or standard linker directories.
    LinkerLibrary,
    /// Python extension module (*.so in site-packages).
    PythonExtension,
    /// Standalone GPU artifact (.cubin, .fatbin, .ptx).
    LooseGpuFile,
}

/// Parsed environment and entrypoint info from the OCI image config.
#[derive(Debug, Clone)]
pub struct ImageConfig {
    pub path_dirs: Vec<String>,
    pub ld_library_dirs: Vec<String>,
    pub virtual_env: Option<String>,
    pub entrypoint: Vec<String>,
    pub cmd: Vec<String>,
    /// Raw env vars for metadata extraction.
    pub env: Vec<(String, String)>,
    /// OCI labels from the image config.
    pub labels: HashMap<String, String>,
}

/// A file extracted from an image layer with its priority classification.
#[derive(Debug)]
pub struct DiscoveredBinary {
    pub path: String,
    pub data: Vec<u8>,
    pub priority: BinaryPriority,
    /// Pre-computed SHA256 hex digest of `data`, set during extraction to avoid
    /// re-hashing in the consumer.
    pub content_sha256: Option<String>,
}

/// Metadata discovered while iterating tar entries (symlinks, config files, packages).
#[derive(Debug, Clone)]
pub enum DiscoveredMetadata {
    /// A symlink found in the image filesystem.
    Symlink { path: String, target: String },
    /// Contents of /etc/os-release.
    OsRelease(String),
    /// Contents of /etc/ld.so.conf or /etc/ld.so.conf.d/*.conf.
    LdSoConf(String),
    /// A Python package found via site-packages/*/METADATA.
    PythonPackage {
        name: String,
        version: String,
        /// The site-packages directory prefix (e.g. "/opt/vllm/lib/python3.12/site-packages").
        site_packages_dir: String,
    },
    /// Contents of /var/lib/dpkg/status (Debian/Ubuntu package database).
    DpkgStatus(String),
    /// A dpkg file list from /var/lib/dpkg/info/<package>.list.
    DpkgFileList {
        package_name: String,
        content: String,
    },
    /// Raw bytes of /var/lib/rpm/rpmdb.sqlite (RPM package database).
    RpmDatabase(Vec<u8>),
}

impl ImageConfig {
    /// Parse the image config from the JSON string returned by `pull_manifest_and_config`.
    pub fn parse(config_json: &str) -> Result<Self> {
        let parsed: serde_json::Value =
            serde_json::from_str(config_json).context("invalid image config JSON")?;

        let config_obj = parsed.get("config").unwrap_or(&serde_json::Value::Null);

        let env_array = config_obj
            .get("Env")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut env = Vec::new();
        let mut path_dirs = Vec::new();
        let mut ld_library_dirs = Vec::new();
        let mut virtual_env = None;

        for item in &env_array {
            let s = match item.as_str() {
                Some(s) => s,
                None => continue,
            };
            if let Some((key, value)) = s.split_once('=') {
                env.push((key.to_string(), value.to_string()));
                match key {
                    "PATH" => {
                        path_dirs = value.split(':').map(|s| s.to_string()).collect();
                    }
                    "LD_LIBRARY_PATH" => {
                        ld_library_dirs = value.split(':').map(|s| s.to_string()).collect();
                    }
                    "VIRTUAL_ENV" => {
                        virtual_env = Some(value.to_string());
                    }
                    _ => {}
                }
            }
        }

        let entrypoint = config_obj
            .get("Entrypoint")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let cmd = config_obj
            .get("Cmd")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let labels = config_obj
            .get("Labels")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            path_dirs,
            ld_library_dirs,
            virtual_env,
            entrypoint,
            cmd,
            env,
            labels,
        })
    }

    /// Get an env var value by key.
    pub fn get_env(&self, key: &str) -> Option<&str> {
        self.env
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Classify a file path from a tar entry according to the image config.
///
/// Expects an already-normalized absolute path (leading `/`).
pub(crate) fn classify_path(path: &str, config: &ImageConfig) -> Option<BinaryPriority> {
    if path.ends_with(".cubin") || path.ends_with(".fatbin") || path.ends_with(".ptx") {
        return Some(BinaryPriority::LooseGpuFile);
    }

    let is_shared_lib = is_shared_library(path);
    if !is_shared_lib && path.contains('.') {
        return None;
    }

    let is_entrypoint = config
        .entrypoint
        .iter()
        .chain(config.cmd.iter())
        .any(|s| s == path);
    if is_entrypoint {
        return Some(BinaryPriority::Entrypoint);
    }

    for dir in &config.path_dirs {
        let dir = dir.trim_end_matches('/');
        if let Some(rest) = path.strip_prefix(dir) {
            if rest.starts_with('/') && !rest[1..].contains('/') {
                return Some(BinaryPriority::PathBinary);
            }
        }
    }

    if !is_shared_lib {
        return None;
    }

    // Checked before Python to avoid VIRTUAL_ENV false positives
    // on libs that happen to live under the venv prefix.
    let is_linker_lib = config
        .ld_library_dirs
        .iter()
        .map(|s| s.as_str())
        .chain(DEFAULT_LIB_DIRS.iter().copied())
        .any(|dir| {
            let dir = dir.trim_end_matches('/');
            path == dir || path.starts_with(&format!("{dir}/"))
        });
    if is_linker_lib {
        return Some(BinaryPriority::LinkerLibrary);
    }

    if path.contains(PYTHON_SITE_PACKAGES) {
        return Some(BinaryPriority::PythonExtension);
    }
    if let Some(ref venv) = config.virtual_env {
        if path.starts_with(venv.as_str()) && path.contains("/lib") {
            return Some(BinaryPriority::PythonExtension);
        }
    }

    None
}

/// Check if a path looks like a shared library (*.so or *.so.*).
fn is_shared_library(path: &str) -> bool {
    let filename = path.rsplit('/').next().unwrap_or(path);
    filename.ends_with(".so") || filename.contains(".so.")
}

/// Default number of layers to download and process concurrently.
const DEFAULT_LAYER_CONCURRENCY: usize = 16;

/// Stream binaries from image layers into a channel for concurrent processing.
///
/// Each layer is streamed (download → decompress → tar extract) in parallel.
/// Discovered binaries are sent over `tx` as they're found, allowing consumers
/// to start scanning immediately without waiting for all layers to finish.
///
/// When a `BlobCache` is provided, layers are read from disk if already cached,
/// and downloaded blobs are tee'd into the cache for next time.
///
/// When a `LayerCache` is provided, extraction manifests are stored after each
/// layer is processed, enabling future scans to skip decompression entirely.
///
/// The channel's bounded capacity provides natural backpressure: if consumers
/// can't keep up, producers slow down, capping memory usage.
#[allow(clippy::too_many_arguments)]
pub async fn stream_binaries_to(
    client: Arc<Client>,
    image_ref: Reference,
    auth: RegistryAuth,
    layers: Vec<(usize, oci_client::manifest::OciDescriptor)>,
    config: ImageConfig,
    tx: tokio::sync::mpsc::Sender<DiscoveredBinary>,
    meta_tx: tokio::sync::mpsc::UnboundedSender<DiscoveredMetadata>,
    progress: ScanProgress,
    blob_cache: Option<BlobCache>,
    layer_cache: Option<LayerCache>,
) -> Result<()> {
    use futures::StreamExt;

    // Authenticate for blob pulls
    client
        .auth(&image_ref, &auth, oci_client::RegistryOperation::Pull)
        .await
        .context("registry authentication failed")?;

    let blob_cache = blob_cache.map(Arc::new);
    let layer_cache = layer_cache.map(Arc::new);

    let results: Vec<Result<()>> = futures::stream::iter(layers.into_iter())
        .map(|(layer_idx, layer)| {
            let client = Arc::clone(&client);
            let image_ref = image_ref.clone();
            let config = config.clone();
            let tx = tx.clone();
            let meta_tx = meta_tx.clone();
            let progress = progress.clone();
            let blob_cache = blob_cache.clone();
            let layer_cache = layer_cache.clone();
            async move {
                let layer_size = u64::try_from(layer.size).unwrap_or(0);
                let layer_progress = progress.add_layer(layer_idx, layer_size);

                // Check blob cache first
                if let Some(ref cache) = blob_cache {
                    if let Some(cached_path) = cache.get(&layer.digest) {
                        tracing::debug!(
                            digest = %layer.digest,
                            "blob cache hit"
                        );
                        let layer_bar = layer_progress;
                        let digest = layer.digest.clone();
                        let lc = layer_cache.clone();
                        return tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&cached_path)
                                .with_context(|| format!("failed to open cached blob {}", cached_path.display()))?;
                            let counting_reader =
                                crate::progress::ProgressReader::new(file, layer_bar.clone());
                            let (manifest, rpm_data) = extract_and_send(counting_reader, &config, &tx, &meta_tx, &progress)?;
                            if let Some(ref lc) = lc {
                                lc.put(&digest, &manifest, rpm_data.as_deref());
                            }
                            layer_bar.clear();
                            Ok(())
                        })
                        .await
                        .context("layer extraction task panicked")?;
                    }
                }

                tracing::debug!(digest = %layer.digest, "cache miss, pulling from registry");

                let stream = client
                    .pull_blob_stream(&image_ref, &layer)
                    .await
                    .with_context(|| format!("failed to pull layer {}", layer.digest))?;

                // Stream registry bytes -> AsyncRead -> sync Read -> gzip -> tar.
                // ProgressReader drives both the per-layer bar and aggregate total.
                let async_reader = tokio_util::io::StreamReader::new(stream.stream);
                let sync_reader = SyncIoBridge::new(async_reader);
                let layer_bar = layer_progress;

                let digest = layer.digest.clone();
                let blob_cache_for_blocking = blob_cache.clone();
                let lc = layer_cache.clone();

                tokio::task::spawn_blocking(move || {
                    let extract_result = if let Some(ref cache) = blob_cache_for_blocking {
                        match cache.prepare(&digest) {
                            Ok(cache_path) => {
                                let caching_reader =
                                    crate::cache::CachingReader::new(sync_reader, cache_path);
                                let counting_reader =
                                    crate::progress::ProgressReader::new(caching_reader, layer_bar.clone());
                                extract_and_send(counting_reader, &config, &tx, &meta_tx, &progress)
                            }
                            Err(e) => {
                                tracing::debug!(error = %e, "cache prepare failed, streaming without cache");
                                let counting_reader =
                                    crate::progress::ProgressReader::new(sync_reader, layer_bar.clone());
                                extract_and_send(counting_reader, &config, &tx, &meta_tx, &progress)
                            }
                        }
                    } else {
                        let counting_reader =
                            crate::progress::ProgressReader::new(sync_reader, layer_bar.clone());
                        extract_and_send(counting_reader, &config, &tx, &meta_tx, &progress)
                    };

                    let (manifest, rpm_data) = extract_result?;
                    if let Some(ref lc) = lc {
                        lc.put(&digest, &manifest, rpm_data.as_deref());
                    }

                    layer_bar.clear();
                    Ok(())
                })
                .await
                .context("layer extraction task panicked")?
            }
        })
        .buffer_unordered(DEFAULT_LAYER_CONCURRENCY)
        .collect()
        .await;

    for result in results {
        result?;
    }

    Ok(())
}

/// Extract binaries from a gzipped tar stream and send each one over the channel.
///
/// Returns a `LayerManifest` recording every binary and metadata item found, plus
/// optional RPM database bytes. The caller stores these in the `LayerCache` so
/// future scans can skip decompression and tar extraction entirely.
///
/// Returns early if the receiver is dropped (channel closed), which means the
/// consumer side shut down and there's no point continuing extraction.
///
/// After tar iteration completes, the underlying reader is drained to EOF so
/// that any wrapping `CachingReader` sees the zero-byte read and finalizes its
/// cache entry. Without this drain, the gzip decoder stops pulling bytes once
/// it hits the gzip footer, the `CachingReader` never sees EOF, and the temp
/// file gets deleted on drop instead of being promoted to the cache.
fn extract_and_send<R: Read>(
    reader: R,
    config: &ImageConfig,
    tx: &tokio::sync::mpsc::Sender<DiscoveredBinary>,
    meta_tx: &tokio::sync::mpsc::UnboundedSender<DiscoveredMetadata>,
    progress: &ScanProgress,
) -> Result<(LayerManifest, Option<Vec<u8>>)> {
    let gz = GzDecoder::new(reader);
    let mut archive = tar::Archive::new(gz);

    let mut manifest = LayerManifest::new();
    let mut rpm_data: Option<Vec<u8>> = None;

    let entries = archive.entries().context("failed to read tar entries")?;

    for entry_result in entries {
        let mut entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!(error = %e, "skipping unreadable tar entry");
                continue;
            }
        };

        let entry_type = entry.header().entry_type();

        let raw_path = match entry.path() {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        // OCI tar entries omit the leading slash; normalize to absolute paths.
        let path = {
            let stripped = raw_path.strip_prefix("./").unwrap_or(&raw_path);
            if stripped.starts_with('/') {
                stripped.to_string()
            } else {
                format!("/{stripped}")
            }
        };

        // Collect symlinks as metadata (before any size/classification checks)
        if entry_type == tar::EntryType::Symlink || entry_type == tar::EntryType::Link {
            if let Ok(Some(link)) = entry.link_name() {
                let target = link.to_string_lossy().to_string();
                manifest.metadata.push(CachedMetadata::Symlink {
                    path: path.clone(),
                    target: target.clone(),
                });
                let _ = meta_tx.send(DiscoveredMetadata::Symlink { path, target });
            }
            continue;
        }

        // Collect config files and package metadata
        if let Some(metadata) = classify_metadata(&path) {
            let size = entry.size();
            // RPM database is binary (SQLite), handle separately from text metadata.
            // Cap at 100 MB to avoid blowing up memory on pathological images.
            if matches!(metadata, MetadataKind::RpmDatabase) {
                const RPM_DB_MAX: u64 = 100 * 1024 * 1024;
                if size <= RPM_DB_MAX {
                    let mut data = Vec::with_capacity(size as usize);
                    if entry.read_to_end(&mut data).is_ok() {
                        manifest.has_rpm_database = true;
                        rpm_data = Some(data.clone());
                        let _ = meta_tx.send(DiscoveredMetadata::RpmDatabase(data));
                    }
                } else {
                    tracing::warn!(path = %path, size, "skipping oversized rpmdb.sqlite");
                }
                continue;
            }
            if size <= MAX_FILE_SIZE {
                let mut content = String::new();
                if entry.read_to_string(&mut content).is_ok() {
                    match metadata {
                        MetadataKind::OsRelease => {
                            manifest
                                .metadata
                                .push(CachedMetadata::OsRelease(content.clone()));
                            let _ = meta_tx.send(DiscoveredMetadata::OsRelease(content));
                        }
                        MetadataKind::LdSoConf => {
                            manifest
                                .metadata
                                .push(CachedMetadata::LdSoConf(content.clone()));
                            let _ = meta_tx.send(DiscoveredMetadata::LdSoConf(content));
                        }
                        MetadataKind::PythonMetadata => {
                            if let Some((name, version)) = parse_python_metadata(&content) {
                                let site_packages_dir =
                                    extract_site_packages_dir(&path).to_string();
                                manifest.metadata.push(CachedMetadata::PythonPackage {
                                    name: name.clone(),
                                    version: version.clone(),
                                    site_packages_dir: site_packages_dir.clone(),
                                });
                                let _ = meta_tx.send(DiscoveredMetadata::PythonPackage {
                                    name,
                                    version,
                                    site_packages_dir,
                                });
                            }
                        }
                        MetadataKind::DpkgStatus => {
                            manifest
                                .metadata
                                .push(CachedMetadata::DpkgStatus(content.clone()));
                            let _ = meta_tx.send(DiscoveredMetadata::DpkgStatus(content));
                        }
                        MetadataKind::DpkgFileList => {
                            // Extract package name from path:
                            // /var/lib/dpkg/info/<package>.list
                            // Also handles arch-qualified names like <package>:<arch>.list
                            let package_name = path
                                .strip_prefix("/var/lib/dpkg/info/")
                                .and_then(|s| s.strip_suffix(".list"))
                                .map(|s| {
                                    // Strip :arch suffix (e.g. "libfoo:amd64" -> "libfoo")
                                    s.split_once(':').map_or(s, |(name, _)| name)
                                })
                                .unwrap_or("")
                                .to_string();
                            if !package_name.is_empty() {
                                manifest.metadata.push(CachedMetadata::DpkgFileList {
                                    package_name: package_name.clone(),
                                    content: content.clone(),
                                });
                                let _ = meta_tx.send(DiscoveredMetadata::DpkgFileList {
                                    package_name,
                                    content,
                                });
                            }
                        }
                        MetadataKind::RpmDatabase => unreachable!(),
                    }
                }
            }
            continue;
        }

        let size = entry.size();
        if size > MAX_FILE_SIZE {
            tracing::warn!(path = %path, size, "skipping file larger than 2 GiB");
            continue;
        }

        let priority = match classify_path(&path, config) {
            Some(p) => p,
            None => continue,
        };

        let mut data = Vec::with_capacity(size as usize);
        if let Err(e) = entry.read_to_end(&mut data) {
            tracing::debug!(path = %path, error = %e, "failed to read tar entry");
            continue;
        }

        if priority != BinaryPriority::LooseGpuFile && !is_elf(&data) {
            continue;
        }

        let content_sha256 = hex::encode(Sha256::digest(&data));

        manifest.binaries.push(ManifestBinary {
            path: path.clone(),
            content_sha256: content_sha256.clone(),
            size: data.len() as u64,
        });

        tracing::debug!(path = %path, priority = ?priority, size = data.len(), "discovered binary");

        // Send over channel. blocking_send is fine here because we're already
        // inside spawn_blocking. If the receiver is gone, stop extracting.
        if tx
            .blocking_send(DiscoveredBinary {
                path,
                data,
                priority,
                content_sha256: Some(content_sha256),
            })
            .is_err()
        {
            tracing::debug!("channel closed, stopping layer extraction");
            return Ok((manifest, rpm_data));
        }

        progress.inc_extracted();
    }

    // Drain the underlying reader to EOF. The tar iterator stops after the
    // end-of-archive marker and the GzDecoder stops after the gzip footer,
    // leaving the raw reader (potentially a CachingReader) mid-stream. Reading
    // it to completion triggers the CachingReader's EOF path which flushes and
    // renames the temp file into the blob cache.
    let gz = archive.into_inner();
    let mut raw = gz.into_inner();
    let _ = std::io::copy(&mut raw, &mut std::io::sink());

    Ok((manifest, rpm_data))
}

/// Classification of metadata files we want to extract from tar entries.
enum MetadataKind {
    OsRelease,
    LdSoConf,
    PythonMetadata,
    DpkgStatus,
    DpkgFileList,
    RpmDatabase,
}

/// Classify a path as a metadata file we want to extract, or None for uninteresting paths.
fn classify_metadata(path: &str) -> Option<MetadataKind> {
    if path == "/etc/os-release" || path == "/usr/lib/os-release" {
        return Some(MetadataKind::OsRelease);
    }
    if path == "/etc/ld.so.conf"
        || (path.starts_with("/etc/ld.so.conf.d/") && path.ends_with(".conf"))
    {
        return Some(MetadataKind::LdSoConf);
    }
    // Python package METADATA: */site-packages/*/METADATA
    if path.contains("site-packages/") && path.ends_with("/METADATA") {
        // Verify it's exactly one level deep inside a package dir
        if let Some(after) = path.rsplit("site-packages/").next() {
            if after.matches('/').count() == 1 {
                return Some(MetadataKind::PythonMetadata);
            }
        }
    }
    if path == "/var/lib/dpkg/status" {
        return Some(MetadataKind::DpkgStatus);
    }
    // dpkg file lists: /var/lib/dpkg/info/<package>.list
    if path.starts_with("/var/lib/dpkg/info/") && path.ends_with(".list") {
        return Some(MetadataKind::DpkgFileList);
    }
    if path == "/var/lib/rpm/rpmdb.sqlite" || path == "/usr/lib/sysimage/rpm/rpmdb.sqlite" {
        return Some(MetadataKind::RpmDatabase);
    }
    None
}

/// Extract the site-packages directory prefix from a METADATA path.
///
/// Given `/opt/vllm/lib/python3.12/site-packages/torch/METADATA`,
/// returns `/opt/vllm/lib/python3.12/site-packages`.
fn extract_site_packages_dir(path: &str) -> &str {
    const MARKER: &str = "site-packages";
    if let Some(idx) = path.find(MARKER) {
        &path[..idx + MARKER.len()]
    } else {
        path
    }
}

/// Parse Python METADATA file for Name and Version fields.
fn parse_python_metadata(content: &str) -> Option<(String, String)> {
    let mut name = None;
    let mut version = None;
    for line in content.lines() {
        if let Some(val) = line.strip_prefix("Name: ") {
            name = Some(val.trim().to_string());
        } else if let Some(val) = line.strip_prefix("Version: ") {
            version = Some(val.trim().to_string());
        }
        // Stop after the first blank line (end of headers)
        if line.is_empty() {
            break;
        }
    }
    match (name, version) {
        (Some(n), Some(v)) => Some((n, v)),
        _ => None,
    }
}

/// Quick check: does this look like an ELF binary?
fn is_elf(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == [0x7f, b'E', b'L', b'F']
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::image::{
        classify_metadata, classify_path, extract_site_packages_dir, is_elf, is_shared_library,
        parse_python_metadata, BinaryPriority, ImageConfig, MetadataKind,
    };

    fn test_config() -> ImageConfig {
        ImageConfig {
            path_dirs: vec![
                "/usr/local/bin".to_string(),
                "/usr/bin".to_string(),
                "/opt/vllm/bin".to_string(),
            ],
            ld_library_dirs: vec!["/opt/vllm/lib64".to_string()],
            virtual_env: Some("/opt/vllm".to_string()),
            entrypoint: vec!["/opt/vllm/bin/python".to_string()],
            cmd: vec![],
            env: vec![
                (
                    "PATH".to_string(),
                    "/usr/local/bin:/usr/bin:/opt/vllm/bin".to_string(),
                ),
                ("LD_LIBRARY_PATH".to_string(), "/opt/vllm/lib64".to_string()),
                ("VIRTUAL_ENV".to_string(), "/opt/vllm".to_string()),
                ("CUDA_VERSION".to_string(), "12.9.1".to_string()),
            ],
            labels: HashMap::new(),
        }
    }

    // -- ImageConfig parsing --

    #[test]
    fn parse_config_full() {
        let json = r#"{
            "config": {
                "Env": [
                    "PATH=/usr/local/bin:/usr/bin",
                    "LD_LIBRARY_PATH=/opt/lib",
                    "VIRTUAL_ENV=/opt/venv",
                    "CUDA_VERSION=12.9.1"
                ],
                "Entrypoint": ["/usr/bin/python"],
                "Cmd": ["-m", "vllm.entrypoints.openai.api_server"]
            }
        }"#;
        let config = ImageConfig::parse(json).unwrap();
        assert_eq!(config.path_dirs, vec!["/usr/local/bin", "/usr/bin"]);
        assert_eq!(config.ld_library_dirs, vec!["/opt/lib"]);
        assert_eq!(config.virtual_env, Some("/opt/venv".to_string()));
        assert_eq!(config.entrypoint, vec!["/usr/bin/python"]);
        assert_eq!(config.cmd, vec!["-m", "vllm.entrypoints.openai.api_server"]);
        assert_eq!(config.get_env("CUDA_VERSION"), Some("12.9.1"));
    }

    #[test]
    fn parse_config_minimal() {
        let json = r#"{"config": {}}"#;
        let config = ImageConfig::parse(json).unwrap();
        assert!(config.path_dirs.is_empty());
        assert!(config.ld_library_dirs.is_empty());
        assert!(config.virtual_env.is_none());
        assert!(config.entrypoint.is_empty());
        assert!(config.cmd.is_empty());
    }

    #[test]
    fn parse_config_no_config_key() {
        let json = r#"{}"#;
        let config = ImageConfig::parse(json).unwrap();
        assert!(config.path_dirs.is_empty());
    }

    #[test]
    fn parse_config_invalid_json() {
        assert!(ImageConfig::parse("not json").is_err());
    }

    #[test]
    fn get_env_existing() {
        let config = test_config();
        assert_eq!(config.get_env("CUDA_VERSION"), Some("12.9.1"));
    }

    #[test]
    fn get_env_missing() {
        let config = test_config();
        assert_eq!(config.get_env("NONEXISTENT"), None);
    }

    // -- Classification --

    #[test]
    fn classify_entrypoint() {
        let config = test_config();
        assert_eq!(
            classify_path("/opt/vllm/bin/python", &config),
            Some(BinaryPriority::Entrypoint)
        );
    }

    #[test]
    fn classify_path_binary() {
        let config = test_config();
        assert_eq!(
            classify_path("/usr/local/bin/nvidia-smi", &config),
            Some(BinaryPriority::PathBinary)
        );
    }

    #[test]
    fn classify_linker_library_ld_path() {
        let config = test_config();
        assert_eq!(
            classify_path("/opt/vllm/lib64/libtorch.so", &config),
            Some(BinaryPriority::LinkerLibrary)
        );
    }

    #[test]
    fn classify_linker_library_default_dir() {
        let config = test_config();
        assert_eq!(
            classify_path("/usr/local/cuda/lib64/libcublas.so.12", &config),
            Some(BinaryPriority::LinkerLibrary)
        );
    }

    #[test]
    fn classify_python_extension() {
        let config = test_config();
        assert_eq!(
            classify_path(
                "/opt/vllm/lib/python3.12/site-packages/torch/lib/torch_cuda.so",
                &config
            ),
            Some(BinaryPriority::PythonExtension)
        );
    }

    #[test]
    fn classify_loose_cubin() {
        let config = test_config();
        assert_eq!(
            classify_path("/some/random/path/kernel.cubin", &config),
            Some(BinaryPriority::LooseGpuFile)
        );
    }

    #[test]
    fn classify_loose_fatbin() {
        let config = test_config();
        assert_eq!(
            classify_path("./some/kernel.fatbin", &config),
            Some(BinaryPriority::LooseGpuFile)
        );
    }

    #[test]
    fn classify_loose_ptx() {
        let config = test_config();
        assert_eq!(
            classify_path("/tmp/something.ptx", &config),
            Some(BinaryPriority::LooseGpuFile)
        );
    }

    #[test]
    fn classify_irrelevant_file() {
        let config = test_config();
        assert_eq!(classify_path("/etc/passwd", &config), None);
    }

    #[test]
    fn classify_random_so_not_in_known_dir() {
        // .so files not in any known directory get no priority
        let config = test_config();
        assert_eq!(classify_path("/random/place/thing.so", &config), None);
    }

    #[test]
    fn classify_with_absolute_path() {
        // Paths are normalized to absolute before reaching classify_path
        let config = test_config();
        assert_eq!(
            classify_path("/usr/local/cuda/lib64/libcudart.so.12", &config),
            Some(BinaryPriority::LinkerLibrary)
        );
    }

    // -- is_shared_library --

    #[test]
    fn is_so_basic() {
        assert!(is_shared_library("libtorch.so"));
    }

    #[test]
    fn is_so_versioned() {
        assert!(is_shared_library("libcublas.so.12"));
    }

    #[test]
    fn is_so_with_path() {
        assert!(is_shared_library("/usr/lib64/libcuda.so.1"));
    }

    #[test]
    fn is_not_so() {
        assert!(!is_shared_library("/usr/bin/python"));
    }

    // -- is_elf --

    #[test]
    fn is_elf_valid() {
        assert!(is_elf(&[0x7f, b'E', b'L', b'F', 0, 0, 0, 0]));
    }

    #[test]
    fn is_elf_too_short() {
        assert!(!is_elf(&[0x7f, b'E', b'L']));
    }

    #[test]
    fn is_elf_wrong_magic() {
        assert!(!is_elf(&[0x00, 0x00, 0x00, 0x00]));
    }

    // -- Priority ordering --

    #[test]
    fn priority_ordering() {
        assert!(BinaryPriority::Entrypoint < BinaryPriority::PathBinary);
        assert!(BinaryPriority::PathBinary < BinaryPriority::LinkerLibrary);
        assert!(BinaryPriority::LinkerLibrary < BinaryPriority::PythonExtension);
        assert!(BinaryPriority::PythonExtension < BinaryPriority::LooseGpuFile);
    }

    // -- Metadata classification --

    #[test]
    fn classify_os_release() {
        assert!(matches!(
            classify_metadata("/etc/os-release"),
            Some(MetadataKind::OsRelease)
        ));
        assert!(matches!(
            classify_metadata("/usr/lib/os-release"),
            Some(MetadataKind::OsRelease)
        ));
    }

    #[test]
    fn classify_ld_so_conf() {
        assert!(matches!(
            classify_metadata("/etc/ld.so.conf"),
            Some(MetadataKind::LdSoConf)
        ));
        assert!(matches!(
            classify_metadata("/etc/ld.so.conf.d/cuda.conf"),
            Some(MetadataKind::LdSoConf)
        ));
    }

    #[test]
    fn classify_python_metadata() {
        assert!(matches!(
            classify_metadata("/usr/lib/python3.12/site-packages/torch/METADATA"),
            Some(MetadataKind::PythonMetadata)
        ));
        // Too deep (sub-sub-directory)
        assert!(
            classify_metadata("/usr/lib/python3.12/site-packages/torch/sub/METADATA").is_none()
        );
    }

    #[test]
    fn classify_dpkg_status() {
        assert!(matches!(
            classify_metadata("/var/lib/dpkg/status"),
            Some(MetadataKind::DpkgStatus)
        ));
    }

    #[test]
    fn classify_rpm_database() {
        assert!(matches!(
            classify_metadata("/var/lib/rpm/rpmdb.sqlite"),
            Some(MetadataKind::RpmDatabase)
        ));
        assert!(matches!(
            classify_metadata("/usr/lib/sysimage/rpm/rpmdb.sqlite"),
            Some(MetadataKind::RpmDatabase)
        ));
        // Non-matching RPM paths
        assert!(classify_metadata("/var/lib/rpm/other.db").is_none());
    }

    #[test]
    fn classify_metadata_irrelevant() {
        assert!(classify_metadata("/etc/passwd").is_none());
        assert!(classify_metadata("/usr/bin/python").is_none());
    }

    #[test]
    fn classify_metadata_does_not_fall_through_to_elf() {
        // Metadata files should be caught before the ELF classification path
        // This verifies they're recognized as metadata, not skipped
        assert!(classify_metadata("/etc/os-release").is_some());
        assert!(classify_metadata("/var/lib/dpkg/status").is_some());
    }

    // -- Python METADATA parser --

    #[test]
    fn python_metadata_basic() {
        let content =
            "Metadata-Version: 2.1\nName: torch\nVersion: 2.5.1+cu124\n\nDescription here";
        let (name, version) = parse_python_metadata(content).unwrap();
        assert_eq!(name, "torch");
        assert_eq!(version, "2.5.1+cu124");
    }

    #[test]
    fn python_metadata_missing_version() {
        let content = "Metadata-Version: 2.1\nName: torch\n";
        assert!(parse_python_metadata(content).is_none());
    }

    #[test]
    fn python_metadata_missing_name() {
        let content = "Metadata-Version: 2.1\nVersion: 1.0\n";
        assert!(parse_python_metadata(content).is_none());
    }

    #[test]
    fn python_metadata_empty() {
        assert!(parse_python_metadata("").is_none());
    }

    // -- extract_site_packages_dir --

    #[test]
    fn site_packages_dir_virtualenv() {
        assert_eq!(
            extract_site_packages_dir("/opt/vllm/lib/python3.12/site-packages/torch/METADATA"),
            "/opt/vllm/lib/python3.12/site-packages"
        );
    }

    #[test]
    fn site_packages_dir_system() {
        assert_eq!(
            extract_site_packages_dir("/usr/lib/python3.12/site-packages/numpy/METADATA"),
            "/usr/lib/python3.12/site-packages"
        );
    }

    #[test]
    fn site_packages_dir_no_marker() {
        assert_eq!(
            extract_site_packages_dir("/some/other/path/METADATA"),
            "/some/other/path/METADATA"
        );
    }

    // -- Labels parsing --

    #[test]
    fn parse_config_with_labels() {
        let json = r#"{
            "config": {
                "Env": ["PATH=/usr/bin"],
                "Labels": {
                    "maintainer": "test",
                    "version": "1.0"
                }
            }
        }"#;
        let config = ImageConfig::parse(json).unwrap();
        assert_eq!(config.labels.len(), 2);
        assert_eq!(config.labels.get("maintainer"), Some(&"test".to_string()));
        assert_eq!(config.labels.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn parse_config_labels_null() {
        let json = r#"{
            "config": {
                "Env": [],
                "Labels": null
            }
        }"#;
        let config = ImageConfig::parse(json).unwrap();
        assert!(config.labels.is_empty());
    }

    #[test]
    fn parse_config_labels_absent() {
        let json = r#"{
            "config": {
                "Env": []
            }
        }"#;
        let config = ImageConfig::parse(json).unwrap();
        assert!(config.labels.is_empty());
    }
}
