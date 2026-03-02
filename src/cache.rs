use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Resolve the wiimi cache root: `$XDG_CACHE_HOME/wiimi/<subdir>` or `~/.cache/wiimi/<subdir>`.
pub(crate) fn cache_root(subdir: &str) -> Option<PathBuf> {
    Some(dirs::cache_dir()?.join("wiimi").join(subdir))
}

/// On-disk blob cache for OCI layer data.
///
/// Stores compressed layer blobs keyed by their registry digest, so subsequent
/// scans of the same image skip the network entirely.
///
///   ~/.cache/wiimi/blobs/sha256/<hex>
///
/// The cache is content-addressed: if the digest matches, the blob is valid.
/// No expiry, no LRU. Users can `rm -rf ~/.cache/wiimi` to clear it.
#[derive(Debug, Clone)]
pub struct BlobCache {
    root: PathBuf,
}

impl BlobCache {
    /// Create a cache rooted at the platform's default cache directory.
    ///
    /// Returns `None` if we can't determine the cache dir (unlikely).
    pub fn default_location() -> Option<Self> {
        Some(Self {
            root: cache_root("blobs")?,
        })
    }

    /// Path for a given digest (e.g., "sha256:abc123..." → root/sha256/abc123...).
    fn blob_path(&self, digest: &str) -> Option<PathBuf> {
        let (algo, hex) = digest.split_once(':')?;
        Some(self.root.join(algo).join(hex))
    }

    /// Check if a blob is cached and return its path.
    pub fn get(&self, digest: &str) -> Option<PathBuf> {
        let path = self.blob_path(digest)?;
        if path.is_file() {
            Some(path)
        } else {
            None
        }
    }

    /// Prepare a path for writing a new blob. Creates parent dirs if needed.
    ///
    /// Returns the final path where the blob should end up. Callers should
    /// write to a temp file and rename for atomicity, or use `CachingReader`.
    pub fn prepare(&self, digest: &str) -> Result<PathBuf> {
        let path = self.blob_path(digest).context("invalid digest format")?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create cache dir {}", parent.display()))?;
        }

        Ok(path)
    }

    /// The root directory of the cache.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// A `Read` wrapper that tees all bytes to a cache file as they flow through.
///
/// On success (inner reader hits EOF cleanly), the temp file is atomically
/// renamed to its final cache path. On error or drop-without-EOF, the temp
/// file is cleaned up.
pub struct CachingReader<R> {
    inner: R,
    writer: Option<BufWriter<std::fs::File>>,
    temp_path: PathBuf,
    final_path: PathBuf,
    finished: bool,
}

impl<R> CachingReader<R> {
    /// Wrap a reader, tee-ing bytes to a temp file next to `final_path`.
    ///
    /// Returns the original reader unwrapped if temp file creation fails
    /// (cache write is best-effort, not fatal).
    pub fn new(inner: R, final_path: PathBuf) -> Self {
        let temp_path = final_path.with_extension("tmp");
        let writer = std::fs::File::create(&temp_path).map(BufWriter::new).ok();

        if writer.is_none() {
            tracing::debug!(
                path = %final_path.display(),
                "failed to create cache temp file, caching disabled for this blob"
            );
        }

        Self {
            inner,
            writer,
            temp_path,
            final_path,
            finished: false,
        }
    }

    /// Finalize the cache entry: flush and rename temp → final.
    fn finalize(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;

        if let Some(ref mut w) = self.writer {
            if w.flush().is_ok() {
                if let Err(e) = std::fs::rename(&self.temp_path, &self.final_path) {
                    tracing::debug!(
                        error = %e,
                        path = %self.final_path.display(),
                        "failed to finalize cache entry"
                    );
                    let _ = std::fs::remove_file(&self.temp_path);
                } else {
                    tracing::debug!(path = %self.final_path.display(), "cached blob");
                }
            } else {
                let _ = std::fs::remove_file(&self.temp_path);
            }
        }
    }
}

impl<R: Read> Read for CachingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            if let Some(ref mut w) = self.writer {
                // Best-effort: if cache write fails, disable it but keep reading
                if w.write_all(&buf[..n]).is_err() {
                    self.writer = None;
                    let _ = std::fs::remove_file(&self.temp_path);
                }
            }
        } else {
            // EOF: finalize the cache entry
            self.finalize();
        }
        Ok(n)
    }
}

impl<R> Drop for CachingReader<R> {
    fn drop(&mut self) {
        if !self.finished {
            // Reader dropped without hitting EOF (error path). Clean up temp file.
            self.writer = None;
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

/// On-disk cache for parsed ELF results, keyed by SHA256 of binary content.
///
/// Stores JSON-serialized parse results so repeat scans skip CPU-intensive
/// ELF parsing and fatbin scanning when the binary bytes haven't changed.
///
///   ~/.cache/wiimi/parse/<binary_salt>/sha256/<content_hex>.json
///
/// The `binary_salt` is derived from a SHA256 hash of the wiimi binary
/// itself. When the binary is rebuilt (new parsing logic, bug fixes, etc.),
/// the salt changes and all stale cache entries are automatically bypassed.
#[derive(Debug, Clone)]
pub struct ParseCache {
    root: PathBuf,
}

/// Content-derived fields from ELF parsing that are safe to cache.
///
/// Path, priority, and size are intentionally excluded: they depend on where
/// the binary appears in the image, not on its content.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedElfResult {
    pub cubins: Vec<crate::nvidia::ComputeCapability>,
    pub ptx: Vec<crate::nvidia::ComputeCapability>,
    pub needed: Vec<String>,
    pub soname: Option<String>,
    pub rpath: Vec<String>,
    pub runpath: Vec<String>,
}

/// Compute a truncated SHA256 hex digest of the currently running binary.
///
/// Used as a salt in the parse cache path so that any change to the binary
/// (new version, different parsing logic) automatically invalidates all
/// cached parse results.
fn binary_salt() -> Option<String> {
    use sha2::{Digest, Sha256};

    let exe_path = std::env::current_exe().ok()?;
    let exe_bytes = std::fs::read(exe_path).ok()?;
    let hash = Sha256::digest(&exe_bytes);
    Some(hex::encode(&hash[..8]))
}

/// Remove parse cache directories from previous binary builds.
///
/// Each binary build gets its own salt directory under `base`. When a new
/// build runs, stale directories from old builds are no longer useful.
/// This is best-effort: failures are silently ignored.
fn gc_stale_salts(base: &Path, current_salt: &str) {
    let entries = match std::fs::read_dir(base) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if name != current_salt {
                tracing::debug!(dir = %name, "removing stale parse cache salt");
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

impl ParseCache {
    /// Create a parse cache at the platform's default cache directory.
    ///
    /// Returns `None` if we can't determine the cache dir or hash the binary.
    pub fn default_location() -> Option<Self> {
        let salt = binary_salt()?;
        let base = cache_root("parse")?;
        gc_stale_salts(&base, &salt);
        Some(Self {
            root: base.join(salt),
        })
    }

    /// Path for a given content SHA256 hex digest.
    fn entry_path(&self, hex_digest: &str) -> PathBuf {
        self.root.join("sha256").join(format!("{hex_digest}.json"))
    }

    /// Look up a cached parse result by content digest.
    pub fn get(&self, hex_digest: &str) -> Option<CachedElfResult> {
        let path = self.entry_path(hex_digest);
        let data = std::fs::read(&path).ok()?;
        match serde_json::from_slice(&data) {
            Ok(result) => {
                tracing::debug!(digest = %hex_digest, "parse cache hit");
                Some(result)
            }
            Err(e) => {
                tracing::debug!(digest = %hex_digest, error = %e, "parse cache entry corrupt, ignoring");
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    /// Store a parse result, keyed by content digest. Best-effort, not fatal.
    pub fn put(&self, hex_digest: &str, result: &CachedElfResult) {
        let path = self.entry_path(hex_digest);
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }

        // Atomic write via temp file + rename
        let tmp_path = path.with_extension("tmp");
        let data = match serde_json::to_vec(result) {
            Ok(d) => d,
            Err(_) => return,
        };
        if std::fs::write(&tmp_path, &data).is_ok() {
            let _ = std::fs::rename(&tmp_path, &path);
        } else {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }

    /// The root directory of this cache (for logging).
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// On-disk cache for layer extraction manifests.
///
/// Stores the list of binaries and metadata discovered in each layer, keyed by
/// the layer's registry digest. When both the blob cache and layer manifest exist,
/// subsequent scans can skip gzip decompression, tar extraction, and SHA256 hashing
/// entirely, jumping straight to parse cache lookups.
///
///   ~/.cache/wiimi/layers/sha256/<digest_hex>.json
///   ~/.cache/wiimi/layers/sha256/<digest_hex>.rpmdb  (optional, raw bytes)
#[derive(Debug, Clone)]
pub struct LayerCache {
    root: PathBuf,
}

/// Current manifest format version. Bump when the schema changes.
const LAYER_MANIFEST_VERSION: u32 = 1;

/// Contents of a layer: every classified binary and metadata item found during extraction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerManifest {
    pub version: u32,
    pub binaries: Vec<ManifestBinary>,
    pub metadata: Vec<CachedMetadata>,
    pub has_rpm_database: bool,
}

impl LayerManifest {
    pub fn new() -> Self {
        Self {
            version: LAYER_MANIFEST_VERSION,
            binaries: Vec::new(),
            metadata: Vec::new(),
            has_rpm_database: false,
        }
    }
}

/// A binary discovered during layer extraction, with its pre-computed content hash.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestBinary {
    pub path: String,
    pub content_sha256: String,
    pub size: u64,
}

/// Serializable metadata from layer extraction.
///
/// Mirrors `DiscoveredMetadata` variants except `RpmDatabase`, which is stored
/// as a separate binary file alongside the manifest JSON.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CachedMetadata {
    Symlink {
        path: String,
        target: String,
    },
    OsRelease(String),
    LdSoConf(String),
    PythonPackage {
        name: String,
        version: String,
        site_packages_dir: String,
    },
    DpkgStatus(String),
    DpkgFileList {
        package_name: String,
        content: String,
    },
}

impl LayerCache {
    /// Create a layer cache at the platform's default cache directory.
    pub fn default_location() -> Option<Self> {
        Some(Self {
            root: cache_root("layers")?,
        })
    }

    fn manifest_path(&self, digest: &str) -> Option<PathBuf> {
        let (algo, hex) = digest.split_once(':')?;
        Some(self.root.join(algo).join(format!("{hex}.json")))
    }

    fn rpmdb_path(&self, digest: &str) -> Option<PathBuf> {
        let (algo, hex) = digest.split_once(':')?;
        Some(self.root.join(algo).join(format!("{hex}.rpmdb")))
    }

    /// Look up a cached layer manifest by digest.
    pub fn get(&self, digest: &str) -> Option<LayerManifest> {
        let path = self.manifest_path(digest)?;
        let data = std::fs::read(&path).ok()?;
        match serde_json::from_slice::<LayerManifest>(&data) {
            Ok(manifest) if manifest.version == LAYER_MANIFEST_VERSION => {
                tracing::debug!(
                    digest = %digest,
                    binaries = manifest.binaries.len(),
                    "layer manifest cache hit"
                );
                Some(manifest)
            }
            Ok(_) => {
                tracing::debug!(digest = %digest, "layer manifest version mismatch, ignoring");
                let _ = std::fs::remove_file(&path);
                None
            }
            Err(e) => {
                tracing::debug!(digest = %digest, error = %e, "layer manifest corrupt, ignoring");
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    /// Load cached RPM database bytes for a layer.
    pub fn get_rpm_database(&self, digest: &str) -> Option<Vec<u8>> {
        let path = self.rpmdb_path(digest)?;
        std::fs::read(&path).ok()
    }

    /// Store a layer manifest and optional RPM database. Best-effort, not fatal.
    pub fn put(&self, digest: &str, manifest: &LayerManifest, rpm_data: Option<&[u8]>) {
        let json_path = match self.manifest_path(digest) {
            Some(p) => p,
            None => return,
        };
        if let Some(parent) = json_path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }

        let tmp = json_path.with_extension("tmp");
        let data = match serde_json::to_vec(manifest) {
            Ok(d) => d,
            Err(_) => return,
        };
        if std::fs::write(&tmp, &data).is_ok() {
            let _ = std::fs::rename(&tmp, &json_path);
        } else {
            let _ = std::fs::remove_file(&tmp);
            return;
        }

        if let Some(rpm) = rpm_data {
            if let Some(rpmdb_path) = self.rpmdb_path(digest) {
                let tmp = rpmdb_path.with_extension("tmp");
                if std::fs::write(&tmp, rpm).is_ok() {
                    let _ = std::fs::rename(&tmp, &rpmdb_path);
                } else {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use crate::cache::{BlobCache, CachedElfResult, CachingReader, ParseCache};

    #[test]
    fn default_location_uses_home() {
        // Just verify it returns Some when HOME is set (which it always is in tests)
        if std::env::var_os("HOME").is_some() {
            assert!(BlobCache::default_location().is_some());
        }
    }

    #[test]
    fn blob_path_format() {
        let cache = BlobCache {
            root: "/tmp/test-cache/blobs".into(),
        };
        let path = cache.blob_path("sha256:abc123").unwrap();
        assert_eq!(
            path.to_string_lossy(),
            "/tmp/test-cache/blobs/sha256/abc123"
        );
    }

    #[test]
    fn blob_path_invalid_digest() {
        let cache = BlobCache {
            root: "/tmp/test-cache/blobs".into(),
        };
        assert!(cache.blob_path("no-colon").is_none());
    }

    #[test]
    fn get_nonexistent() {
        let cache = BlobCache {
            root: "/tmp/wiimi-test-nonexistent/blobs".into(),
        };
        assert!(cache.get("sha256:doesnotexist").is_none());
    }

    #[test]
    fn caching_reader_tees_and_finalizes() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("sha256").join("testblob");
        std::fs::create_dir_all(final_path.parent().unwrap()).unwrap();

        let data = b"hello world this is test data";
        let cursor = std::io::Cursor::new(data.to_vec());

        let mut reader = CachingReader::new(cursor, final_path.clone());
        let mut output = Vec::new();
        reader.read_to_end(&mut output).unwrap();

        assert_eq!(output, data);
        assert!(final_path.is_file());
        assert_eq!(std::fs::read(&final_path).unwrap(), data);
    }

    #[test]
    fn caching_reader_cleans_up_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("sha256").join("dropped");
        std::fs::create_dir_all(final_path.parent().unwrap()).unwrap();

        let data = b"partial data";
        let cursor = std::io::Cursor::new(data.to_vec());

        let mut reader = CachingReader::new(cursor, final_path.clone());
        let mut buf = [0u8; 4];
        // Read only part of the data, then drop
        let _ = reader.read(&mut buf).unwrap();
        drop(reader);

        // Neither final nor temp should exist
        assert!(!final_path.is_file());
    }

    // -- ParseCache tests --

    #[test]
    fn parse_cache_default_location_returns_some() {
        if std::env::var_os("HOME").is_some() {
            assert!(ParseCache::default_location().is_some());
        }
    }

    #[test]
    fn parse_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ParseCache {
            root: dir.path().to_path_buf(),
        };

        let result = CachedElfResult {
            cubins: vec![crate::nvidia::ComputeCapability::new(9, 0)],
            ptx: vec![crate::nvidia::ComputeCapability::new(8, 0)],
            needed: vec!["libcuda.so.1".to_string()],
            soname: Some("libfoo.so".to_string()),
            rpath: vec![],
            runpath: vec!["/usr/lib64".to_string()],
        };

        cache.put("deadbeef1234", &result);
        let got = cache.get("deadbeef1234").expect("cache miss after put");

        assert_eq!(got.cubins, result.cubins);
        assert_eq!(got.ptx, result.ptx);
        assert_eq!(got.needed, result.needed);
        assert_eq!(got.soname, result.soname);
        assert_eq!(got.rpath, result.rpath);
        assert_eq!(got.runpath, result.runpath);
    }

    #[test]
    fn parse_cache_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ParseCache {
            root: dir.path().to_path_buf(),
        };
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn parse_cache_corrupt_entry_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ParseCache {
            root: dir.path().to_path_buf(),
        };

        // Write garbage to the expected path
        let path = cache.entry_path("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not valid json").unwrap();

        assert!(cache.get("corrupt").is_none());
        // Corrupt file should be cleaned up
        assert!(!path.exists());
    }

    #[test]
    fn gc_stale_salts_removes_old_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();

        // Create current and stale salt directories
        let current = "aabbccdd11223344";
        let stale1 = "oldold0011223344";
        let stale2 = "deadbeef00000000";
        std::fs::create_dir_all(base.join(current).join("sha256")).unwrap();
        std::fs::create_dir_all(base.join(stale1).join("sha256")).unwrap();
        std::fs::create_dir_all(base.join(stale2).join("sha256")).unwrap();

        // Put a file in stale1 to ensure it removes non-empty dirs
        std::fs::write(base.join(stale1).join("sha256").join("test.json"), b"{}").unwrap();

        crate::cache::gc_stale_salts(base, current);

        assert!(base.join(current).exists(), "current salt should survive");
        assert!(!base.join(stale1).exists(), "stale1 should be removed");
        assert!(!base.join(stale2).exists(), "stale2 should be removed");
    }

    // -- LayerCache tests --

    use crate::cache::{CachedMetadata, LayerCache, LayerManifest, ManifestBinary};

    #[test]
    fn layer_cache_default_location_returns_some() {
        if std::env::var_os("HOME").is_some() {
            assert!(LayerCache::default_location().is_some());
        }
    }

    #[test]
    fn layer_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };

        let manifest = LayerManifest {
            version: 1,
            binaries: vec![ManifestBinary {
                path: "/usr/lib64/libcuda.so.1".to_string(),
                content_sha256: "abcdef1234567890".to_string(),
                size: 42000,
            }],
            metadata: vec![
                CachedMetadata::Symlink {
                    path: "/usr/lib/libcuda.so".to_string(),
                    target: "libcuda.so.1".to_string(),
                },
                CachedMetadata::OsRelease("NAME=Ubuntu\nVERSION=22.04\n".to_string()),
            ],
            has_rpm_database: false,
        };

        cache.put("sha256:deadbeef", &manifest, None);
        let got = cache.get("sha256:deadbeef").expect("cache miss after put");

        assert_eq!(got.binaries.len(), 1);
        assert_eq!(got.binaries[0].path, "/usr/lib64/libcuda.so.1");
        assert_eq!(got.binaries[0].content_sha256, "abcdef1234567890");
        assert_eq!(got.binaries[0].size, 42000);
        assert_eq!(got.metadata.len(), 2);
        assert!(!got.has_rpm_database);
    }

    #[test]
    fn layer_cache_with_rpm_database() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };

        let manifest = LayerManifest {
            version: 1,
            binaries: Vec::new(),
            metadata: Vec::new(),
            has_rpm_database: true,
        };

        let rpm_data = b"fake rpm database bytes";
        cache.put("sha256:rpmtest", &manifest, Some(rpm_data));

        let got = cache.get("sha256:rpmtest").expect("cache miss after put");
        assert!(got.has_rpm_database);

        let rpm = cache
            .get_rpm_database("sha256:rpmtest")
            .expect("rpm database missing");
        assert_eq!(rpm, rpm_data);
    }

    #[test]
    fn layer_cache_miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };
        assert!(cache.get("sha256:nonexistent").is_none());
    }

    #[test]
    fn layer_cache_corrupt_entry_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };

        let path = cache.manifest_path("sha256:corrupt").unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not valid json").unwrap();

        assert!(cache.get("sha256:corrupt").is_none());
        assert!(!path.exists());
    }

    #[test]
    fn layer_cache_version_mismatch_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };

        // Write a manifest with a future version
        let manifest = LayerManifest {
            version: 999,
            binaries: Vec::new(),
            metadata: Vec::new(),
            has_rpm_database: false,
        };
        cache.put("sha256:oldver", &manifest, None);

        // Reading it back should fail (version mismatch)
        // But wait, put writes whatever version is in the struct, and get checks LAYER_MANIFEST_VERSION.
        // Since 999 != 1, it should return None.
        assert!(cache.get("sha256:oldver").is_none());
    }

    #[test]
    fn layer_cache_invalid_digest_format() {
        let dir = tempfile::tempdir().unwrap();
        let cache = LayerCache {
            root: dir.path().to_path_buf(),
        };
        assert!(cache.get("no-colon").is_none());
        assert!(cache.get_rpm_database("no-colon").is_none());
    }
}
