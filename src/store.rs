use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::Connection;

use crate::scan::ScanResult;

/// A scan result paired with its store timestamp.
#[derive(Debug, Clone)]
pub struct ScanWithTimestamp {
    pub scan: ScanResult,
    pub scanned_at: String,
}

/// Metadata row from the scan store (no blob deserialization).
#[derive(Debug, Clone)]
pub struct ScanMeta {
    pub image: String,
    pub scanned_at: String,
    pub cuda_version: Option<String>,
    pub cc_min: Option<String>,
    pub cc_max: Option<String>,
    pub has_ptx: bool,
    pub binary_count: usize,
}

/// SQLite-backed scan result store.
///
/// Each scan is stored as a MessagePack blob alongside denormalized metadata
/// columns for quick listing without deserializing the full result.
pub struct ScanStore {
    conn: Connection,
}

impl ScanStore {
    /// Open (or create) a scan store at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create store directory: {}", parent.display())
            })?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open scan store: {}", path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS scans (
                image        TEXT NOT NULL,
                scanned_at   TEXT NOT NULL,
                cuda_version TEXT,
                cc_min       TEXT,
                cc_max       TEXT,
                has_ptx      INTEGER NOT NULL,
                binary_count INTEGER NOT NULL,
                data         BLOB NOT NULL,
                PRIMARY KEY (image)
            );",
        )
        .context("failed to create scans table")?;
        Ok(Self { conn })
    }

    /// Open a store at the default cache location (`~/.cache/wiimi/scans.db`).
    pub fn default_location() -> Result<Self> {
        let path =
            default_store_path().context("could not determine cache directory for scan store")?;
        Self::open(&path)
    }

    /// Insert or replace a scan result.
    pub fn upsert(&self, result: &ScanResult) -> Result<()> {
        let data =
            rmp_serde::to_vec_named(result).context("failed to serialize scan to msgpack")?;
        let now = Utc::now().to_rfc3339();
        let cuda_version = result.metadata.cuda_version.as_deref();
        let cc_min = result.effective_cc_min.as_ref().map(|c| c.to_string());
        let cc_max = result.effective_cc_max.as_ref().map(|c| c.to_string());
        let has_ptx = result.has_ptx_forward_compat;
        let binary_count = result.binaries.len();

        self.conn
            .execute(
                "INSERT OR REPLACE INTO scans
                    (image, scanned_at, cuda_version, cc_min, cc_max, has_ptx, binary_count, data)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    result.image,
                    now,
                    cuda_version,
                    cc_min,
                    cc_max,
                    has_ptx,
                    binary_count,
                    data,
                ],
            )
            .context("failed to upsert scan")?;
        Ok(())
    }

    /// Load a scan result by image reference.
    pub fn get(&self, image: &str) -> Result<Option<ScanResult>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM scans WHERE image = ?1")
            .context("failed to prepare get query")?;

        let result = stmt
            .query_row(rusqlite::params![image], |row| {
                let data: Vec<u8> = row.get(0)?;
                Ok(data)
            })
            .optional()
            .context("failed to query scan")?;

        match result {
            Some(data) => {
                let scan: ScanResult = rmp_serde::from_slice(&data)
                    .context("failed to deserialize scan from msgpack")?;
                Ok(Some(scan))
            }
            None => Ok(None),
        }
    }

    /// List metadata for all stored scans (no blob deserialization).
    pub fn list(&self) -> Result<Vec<ScanMeta>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT image, scanned_at, cuda_version, cc_min, cc_max, has_ptx, binary_count
                 FROM scans ORDER BY image",
            )
            .context("failed to prepare list query")?;

        let rows = stmt
            .query_map([], |row| {
                Ok(ScanMeta {
                    image: row.get(0)?,
                    scanned_at: row.get(1)?,
                    cuda_version: row.get(2)?,
                    cc_min: row.get(3)?,
                    cc_max: row.get(4)?,
                    has_ptx: row.get::<_, bool>(5)?,
                    binary_count: row.get::<_, usize>(6)?,
                })
            })
            .context("failed to list scans")?;

        rows.collect::<Result<Vec<_>, _>>()
            .context("failed to collect scan metadata")
    }

    /// Load all stored scan results (deserializes every blob).
    pub fn get_all(&self) -> Result<Vec<ScanResult>> {
        let mut stmt = self
            .conn
            .prepare("SELECT data FROM scans ORDER BY image")
            .context("failed to prepare get_all query")?;

        let rows = stmt
            .query_map([], |row| {
                let data: Vec<u8> = row.get(0)?;
                Ok(data)
            })
            .context("failed to query all scans")?;

        let mut results = Vec::new();
        for row in rows {
            let data = row.context("failed to read scan row")?;
            let scan: ScanResult =
                rmp_serde::from_slice(&data).context("failed to deserialize scan from msgpack")?;
            results.push(scan);
        }
        Ok(results)
    }

    /// Load all scans whose image reference starts with the given prefix,
    /// along with their store timestamps.
    pub fn query_by_prefix(&self, prefix: &str) -> Result<Vec<ScanWithTimestamp>> {
        let pattern = format!("{prefix}%");
        let mut stmt = self
            .conn
            .prepare("SELECT data, scanned_at FROM scans WHERE image LIKE ?1 ORDER BY image")
            .context("failed to prepare prefix query")?;

        let rows = stmt
            .query_map(rusqlite::params![pattern], |row| {
                let data: Vec<u8> = row.get(0)?;
                let scanned_at: String = row.get(1)?;
                Ok((data, scanned_at))
            })
            .context("failed to query scans by prefix")?;

        let mut results = Vec::new();
        for row in rows {
            let (data, scanned_at) = row.context("failed to read scan row")?;
            let scan: ScanResult =
                rmp_serde::from_slice(&data).context("failed to deserialize scan from msgpack")?;
            results.push(ScanWithTimestamp { scan, scanned_at });
        }
        Ok(results)
    }

    /// Check if a scan exists for the given image.
    pub fn contains(&self, image: &str) -> Result<bool> {
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM scans WHERE image = ?1")
            .context("failed to prepare contains query")?;
        let exists = stmt
            .query_row(rusqlite::params![image], |_| Ok(()))
            .optional()
            .context("failed to check scan existence")?;
        Ok(exists.is_some())
    }
}

/// Default scan store path: `~/.cache/wiimi/scans.db`.
pub fn default_store_path() -> Option<PathBuf> {
    crate::cache::cache_root("store").map(|p| p.join("scans.db"))
}

// We need the `optional()` method on `Result<T, rusqlite::Error>`.
use rusqlite::OptionalExtension;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::image::BinaryPriority;
    use crate::nvidia::ComputeCapability;
    use crate::scan::{
        BinaryScanResult, EnvironmentInfo, ImageMetadata, PackageVersion, PythonEnvironment,
        ScanResult,
    };
    use crate::store::ScanStore;

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
                needed: vec!["libc.so.6".to_string()],
                soname: Some("libtest.so".to_string()),
                rpath: vec![],
                runpath: vec![],
            }],
            effective_cc_min: Some(cc(7, 0)),
            effective_cc_max: Some(cc(9, 0)),
            has_ptx_forward_compat: true,
            warnings: vec!["test warning".to_string()],
            dep_graph: None,
            reachable_count: 1,
            dormant_count: 0,
            environment: EnvironmentInfo {
                os: None,
                python_environments: vec![PythonEnvironment {
                    label: "system".to_string(),
                    site_packages_dir: "/usr/lib/python3/dist-packages".to_string(),
                    packages: vec![PackageVersion {
                        name: "numpy".to_string(),
                        version: "1.26.0".to_string(),
                        source: None,
                    }],
                }],
                system_packages: vec![PackageVersion {
                    name: "curl".to_string(),
                    version: "8.0".to_string(),
                    source: None,
                }],
                ld_conf_paths: vec![],
                symlinks: HashMap::new(),
                file_owners: HashMap::new(),
            },
            labels: HashMap::from([("maintainer".to_string(), "test".to_string())]),
            env_vars: vec![("PATH".to_string(), "/usr/bin".to_string())],
        }
    }

    #[test]
    fn round_trip_upsert_get() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        let scan = test_scan("ghcr.io/test/image:v1");
        store.upsert(&scan).unwrap();

        let loaded = store.get("ghcr.io/test/image:v1").unwrap().unwrap();
        assert_eq!(loaded.image, scan.image);
        assert_eq!(loaded.binaries.len(), 1);
        assert_eq!(loaded.binaries[0].cubins, vec![cc(7, 0), cc(9, 0)]);
        assert_eq!(loaded.binaries[0].ptx, vec![cc(9, 0)]);
        assert_eq!(loaded.metadata.cuda_version, Some("12.4".to_string()));
        assert_eq!(loaded.effective_cc_min, Some(cc(7, 0)));
        assert!(loaded.has_ptx_forward_compat);
        assert_eq!(loaded.environment.system_packages.len(), 1);
        assert_eq!(loaded.environment.python_environments.len(), 1);
        assert_eq!(loaded.labels.get("maintainer").unwrap(), "test");
        assert_eq!(loaded.env_vars.len(), 1);
    }

    #[test]
    fn get_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        assert!(store.get("nonexistent:v1").unwrap().is_none());
    }

    #[test]
    fn upsert_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        let mut scan1 = test_scan("ghcr.io/test/image:v1");
        store.upsert(&scan1).unwrap();

        scan1.metadata.cuda_version = Some("12.6".to_string());
        scan1.has_ptx_forward_compat = false;
        store.upsert(&scan1).unwrap();

        let loaded = store.get("ghcr.io/test/image:v1").unwrap().unwrap();
        assert_eq!(loaded.metadata.cuda_version, Some("12.6".to_string()));
        assert!(!loaded.has_ptx_forward_compat);

        // Should still only have one row
        let metas = store.list().unwrap();
        assert_eq!(metas.len(), 1);
    }

    #[test]
    fn list_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/a:v1")).unwrap();
        store.upsert(&test_scan("ghcr.io/b:v2")).unwrap();

        let metas = store.list().unwrap();
        assert_eq!(metas.len(), 2);

        let first = &metas[0];
        assert_eq!(first.image, "ghcr.io/a:v1");
        assert_eq!(first.cuda_version.as_deref(), Some("12.4"));
        assert_eq!(first.cc_min.as_deref(), Some("7.0"));
        assert_eq!(first.cc_max.as_deref(), Some("9.0"));
        assert!(first.has_ptx);
        assert_eq!(first.binary_count, 1);
        assert!(!first.scanned_at.is_empty());
    }

    #[test]
    fn get_all_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/a:v1")).unwrap();
        store.upsert(&test_scan("ghcr.io/b:v2")).unwrap();

        let all = store.get_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn query_by_prefix_filters_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        store.upsert(&test_scan("ghcr.io/org/repo:v1.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/repo:v2.0")).unwrap();
        store.upsert(&test_scan("ghcr.io/org/other:v1.0")).unwrap();
        store
            .upsert(&test_scan("docker.io/lib/foo:latest"))
            .unwrap();

        let results = store.query_by_prefix("ghcr.io/org/repo:").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].scan.image, "ghcr.io/org/repo:v1.0");
        assert_eq!(results[1].scan.image, "ghcr.io/org/repo:v2.0");
        assert!(!results[0].scanned_at.is_empty());

        let results = store.query_by_prefix("ghcr.io/org/other:").unwrap();
        assert_eq!(results.len(), 1);

        let results = store.query_by_prefix("nonexistent:").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn contains_check() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = ScanStore::open(&db_path).unwrap();

        assert!(!store.contains("ghcr.io/test:v1").unwrap());
        store.upsert(&test_scan("ghcr.io/test:v1")).unwrap();
        assert!(store.contains("ghcr.io/test:v1").unwrap());
    }
}
