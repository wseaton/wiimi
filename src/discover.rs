use std::sync::Arc;

use anyhow::{Context, Result};
use oci_client::client::ClientConfig;
use oci_client::{Client, Reference};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::registry::resolve_auth_for_reference;
use crate::scan;
use crate::site::{FamilyConfig, TagOrder};
use crate::store::ScanStore;

/// Summary of a discovery run for one family.
pub struct DiscoverySummary {
    pub family_name: String,
    pub matched: usize,
    pub new_scanned: usize,
    pub cached: usize,
    pub ignored: usize,
}

/// List all tags for a repository, handling pagination.
async fn list_all_tags(
    client: &Client,
    image_ref: &Reference,
    auth: &oci_client::secrets::RegistryAuth,
) -> Result<Vec<String>> {
    let page_size = 100;
    let mut all_tags = Vec::new();
    let mut last: Option<String> = None;

    loop {
        let response = client
            .list_tags(image_ref, auth, Some(page_size), last.as_deref())
            .await
            .context("failed to list tags from registry")?;

        let count = response.tags.len();
        all_tags.extend(response.tags);

        if count < page_size {
            break;
        }
        last = all_tags.last().cloned();
    }

    Ok(all_tags)
}

/// Discover new images for a family: list tags from the registry, filter by
/// pattern, scan anything not already in the store. Images are scanned
/// concurrently up to `concurrency` at a time (a la Trivy's parallel scanning).
pub async fn discover_family(
    family: &FamilyConfig,
    store: &ScanStore,
    client: Arc<Client>,
    dockerconfig: Option<&[u8]>,
    force: bool,
    no_cache: bool,
    concurrency: usize,
) -> Result<DiscoverySummary> {
    // Build a reference for tag listing (tag is ignored, just need registry/repo)
    let ref_str = format!("{}/{}:latest", family.registry, family.repository);
    let image_ref: Reference = ref_str
        .parse()
        .with_context(|| format!("invalid reference: {ref_str}"))?;
    let auth = resolve_auth_for_reference(dockerconfig, &image_ref);

    let all_tags = list_all_tags(&client, &image_ref, &auth).await?;
    let total_tags = all_tags.len();

    let pattern = family.compiled_tag_pattern()?;
    let mut matched_tags: Vec<&str> = all_tags
        .iter()
        .filter(|t| pattern.is_match(t))
        .map(|t| t.as_str())
        .collect();

    // For lexicographic ordering, we can pre-filter to last_n before scanning.
    // For scan_time ordering, we need to scan everything and let site generation
    // handle truncation (we don't know timestamps until after scanning).
    if family.order_by == TagOrder::Lexicographic {
        matched_tags.sort();
        if let Some(n) = family.last_n {
            let start = matched_tags.len().saturating_sub(n);
            matched_tags = matched_tags.split_off(start);
        }
    }

    let matched = matched_tags.len();
    let ignored = total_tags - matched;

    // Partition into cached (already in store) and images that need scanning.
    let mut to_scan = Vec::new();
    let mut cached = 0;

    for tag in &matched_tags {
        let image = family.image_ref(tag);
        if !force && store.contains(&image)? {
            tracing::debug!(image = %image, "already in store, skipping");
            cached += 1;
        } else {
            to_scan.push(image);
        }
    }

    let total_to_scan = to_scan.len();
    if total_to_scan == 0 {
        return Ok(DiscoverySummary {
            family_name: family.name.clone(),
            matched,
            new_scanned: 0,
            cached,
            ignored,
        });
    }

    tracing::info!(
        family = %family.name,
        total = total_to_scan,
        concurrency = concurrency,
        "scanning images"
    );

    // Own the dockerconfig bytes so they can be shared across spawned tasks.
    let dockerconfig_owned: Arc<Option<Vec<u8>>> = Arc::new(dockerconfig.map(|d| d.to_vec()));
    let use_blob_cache = !no_cache;
    let use_parse_cache = !no_cache;
    let semaphore = Arc::new(Semaphore::new(concurrency));

    let mut join_set = JoinSet::new();
    for (idx, image) in to_scan.into_iter().enumerate() {
        let sem = semaphore.clone();
        let dc = dockerconfig_owned.clone();
        join_set.spawn(async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("semaphore closed: {e}"))?;
            tracing::info!(image = %image, progress = idx + 1, total = total_to_scan, "scanning");
            let result = scan::scan_image(
                &image,
                dc.as_deref(),
                vec![],
                true,
                use_blob_cache,
                use_parse_cache,
            )
            .await?;
            Ok::<_, anyhow::Error>(result)
        });
    }

    // Collect results and persist. ScanStore isn't Sync so we upsert as
    // results arrive on the main task rather than inside the spawned futures.
    let mut new_scanned = 0;
    while let Some(result) = join_set.join_next().await {
        let scan_result = result.context("scan task panicked")??;
        store.upsert(&scan_result)?;
        new_scanned += 1;
    }

    Ok(DiscoverySummary {
        family_name: family.name.clone(),
        matched,
        new_scanned,
        cached,
        ignored,
    })
}

/// Build an OCI client configured for discovery.
pub fn build_client(insecure_registries: Vec<String>) -> Client {
    let protocol = if insecure_registries.is_empty() {
        oci_client::client::ClientProtocol::Https
    } else {
        oci_client::client::ClientProtocol::HttpsExcept(insecure_registries)
    };
    let config = ClientConfig {
        protocol,
        platform_resolver: Some(Box::new(oci_client::client::linux_amd64_resolver)),
        ..Default::default()
    };
    Client::new(config)
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    #[test]
    fn tag_pattern_filtering() {
        let pattern = Regex::new(r"^v\d+\.\d+\.\d+(-rc\.\d+)?$").unwrap();

        let tags = vec![
            "v0.3.0",
            "v0.3.1-rc.4",
            "v0.3.1",
            "v0.5.0-rc.4",
            "latest",
            "sha-abc123",
            "main",
            "v0.5.0",
            "pr-42",
            "nightly-20240101",
        ];

        let matched: Vec<&&str> = tags.iter().filter(|t| pattern.is_match(t)).collect();
        assert_eq!(
            matched,
            vec![
                &"v0.3.0",
                &"v0.3.1-rc.4",
                &"v0.3.1",
                &"v0.5.0-rc.4",
                &"v0.5.0"
            ]
        );
    }

    #[test]
    fn tag_pattern_rejects_invalid() {
        let pattern = Regex::new(r"^v\d+\.\d+\.\d+(-rc\.\d+)?$").unwrap();
        assert!(!pattern.is_match(""));
        assert!(!pattern.is_match("v1"));
        assert!(!pattern.is_match("v1.0"));
        assert!(!pattern.is_match("1.0.0"));
        assert!(!pattern.is_match("v1.0.0-beta"));
    }

    #[test]
    fn tag_pattern_accepts_valid() {
        let pattern = Regex::new(r"^v\d+\.\d+\.\d+(-rc\.\d+)?$").unwrap();
        assert!(pattern.is_match("v0.0.0"));
        assert!(pattern.is_match("v1.2.3"));
        assert!(pattern.is_match("v10.20.30"));
        assert!(pattern.is_match("v1.0.0-rc.1"));
        assert!(pattern.is_match("v0.5.1-rc.42"));
    }
}
