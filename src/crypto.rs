//! Vendored crypto detection for FIPS auditing.
//!
//! Two complementary detection layers per ELF:
//!
//!   1. cargo-auditable metadata: Rust binaries built with `cargo auditable`
//!      embed their full crate graph as zlib-compressed JSON in a `.dep-v0`
//!      section. Exact crate names + versions, but only present when the
//!      builder opted in.
//!
//!   2. Byte signatures: symbol prefixes and rodata strings left behind by
//!      statically linked crypto implementations (ring, AWS-LC, BoringSSL,
//!      vendored OpenSSL). Works on binaries with no audit metadata, though
//!      symbol-based signatures can be lost to stripping.
//!
//! Both are cross-referenced with DT_NEEDED: crypto material inside a binary
//! that does NOT link libcrypto/libssl means the crypto is vendored and will
//! ignore the host's FIPS policy.

use serde::{Deserialize, Serialize};

/// Statically linked crypto implementation detected via byte signatures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VendoredCryptoKind {
    /// ring (BoringSSL-derived Rust crypto).
    Ring,
    /// AWS-LC via aws-lc-sys (default rustls provider since 0.23).
    AwsLc,
    /// BoringSSL via boring-sys.
    BoringSsl,
    /// Statically linked OpenSSL (openssl-src / vendored feature).
    OpenSslStatic,
    /// TLS cipher-suite name strings present without libssl linkage:
    /// an embedded TLS stack of unknown implementation.
    EmbeddedTls,
}

impl std::fmt::Display for VendoredCryptoKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Ring => "ring",
            Self::AwsLc => "aws-lc",
            Self::BoringSsl => "boringssl",
            Self::OpenSslStatic => "openssl-static",
            Self::EmbeddedTls => "embedded-tls",
        };
        f.write_str(s)
    }
}

/// One byte-signature hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VendoredCrypto {
    pub kind: VendoredCryptoKind,
    /// Version recovered from version-mangled symbol prefixes, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// What matched, for report readers.
    pub evidence: String,
}

/// Category of a deny-listed crate from cargo-auditable metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CryptoCrateCategory {
    /// Full crypto provider with its own primitives (ring, aws-lc-sys, ...).
    Provider,
    /// TLS stack that consumes a provider (rustls, s2n-tls).
    TlsStack,
    /// Pure-Rust primitive (RustCrypto sha2, rsa, hmac, ...).
    Primitive,
}

/// A crypto-relevant crate found in a binary's cargo-auditable metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoCrate {
    pub name: String,
    pub version: String,
    pub category: CryptoCrateCategory,
}

/// Per-binary crypto findings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoInfo {
    /// Byte-signature hits for statically linked crypto.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vendored: Vec<VendoredCrypto>,
    /// Crypto-relevant crates from cargo-auditable metadata (runtime deps only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audit_crates: Vec<CryptoCrate>,
    /// Binary links libcrypto.so via DT_NEEDED.
    #[serde(default)]
    pub links_libcrypto: bool,
    /// Binary links libssl.so via DT_NEEDED.
    #[serde(default)]
    pub links_libssl: bool,
}

impl CryptoInfo {
    pub fn is_empty(&self) -> bool {
        self.vendored.is_empty() && self.audit_crates.is_empty()
    }

    /// The binary performs crypto without going through a system libcrypto,
    /// so it ignores the host crypto policy (the FIPS finding).
    pub fn has_vendored_crypto(&self) -> bool {
        !self.vendored.is_empty()
            || self
                .audit_crates
                .iter()
                .any(|c| c.category != CryptoCrateCategory::TlsStack)
    }

    /// One-line summary for warnings and reports.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = self
            .vendored
            .iter()
            .map(|v| match &v.version {
                Some(ver) => format!("{} {ver}", v.kind),
                None => v.kind.to_string(),
            })
            .collect();
        for c in &self.audit_crates {
            let entry = format!("{} {} (crate)", c.name, c.version);
            if !parts.iter().any(|p| p.starts_with(&c.name)) {
                parts.push(entry);
            }
        }
        parts.join(", ")
    }
}

/// Sonames of system crypto implementation libraries. Signature hits inside
/// these are their own primitives, not vendored crypto: they are the
/// RPM-managed, crypto-policy-following implementations themselves.
const CRYPTO_IMPL_SONAMES: &[&str] = &[
    "libcrypto.so",
    "libssl.so",
    "libgnutls.so",
    "libnss3.so",
    "libssl3.so",
    "libnettle.so",
    "libgcrypt.so",
];

/// Whether a soname identifies a system crypto implementation library.
pub fn is_crypto_implementation(soname: &str) -> bool {
    CRYPTO_IMPL_SONAMES
        .iter()
        .any(|prefix| soname == prefix.trim_end_matches(".so") || soname.starts_with(prefix))
}

/// Crates that indicate crypto compiled into the binary. Bindings crates that
/// dynamically link the system library (openssl, openssl-sys) are deliberately
/// absent; their vendored builds show up as openssl-src instead.
const CRYPTO_CRATES: &[(&str, CryptoCrateCategory)] = &[
    ("ring", CryptoCrateCategory::Provider),
    ("aws-lc-rs", CryptoCrateCategory::Provider),
    ("aws-lc-sys", CryptoCrateCategory::Provider),
    ("aws-lc-fips-sys", CryptoCrateCategory::Provider),
    ("boring", CryptoCrateCategory::Provider),
    ("boring-sys", CryptoCrateCategory::Provider),
    ("openssl-src", CryptoCrateCategory::Provider),
    ("rustls", CryptoCrateCategory::TlsStack),
    ("s2n-tls", CryptoCrateCategory::TlsStack),
    ("sha1", CryptoCrateCategory::Primitive),
    ("sha2", CryptoCrateCategory::Primitive),
    ("sha3", CryptoCrateCategory::Primitive),
    ("md-5", CryptoCrateCategory::Primitive),
    ("md5", CryptoCrateCategory::Primitive),
    ("hmac", CryptoCrateCategory::Primitive),
    ("hkdf", CryptoCrateCategory::Primitive),
    ("pbkdf2", CryptoCrateCategory::Primitive),
    ("aes", CryptoCrateCategory::Primitive),
    ("aes-gcm", CryptoCrateCategory::Primitive),
    ("aes-gcm-siv", CryptoCrateCategory::Primitive),
    ("chacha20", CryptoCrateCategory::Primitive),
    ("chacha20poly1305", CryptoCrateCategory::Primitive),
    ("poly1305", CryptoCrateCategory::Primitive),
    ("rsa", CryptoCrateCategory::Primitive),
    ("dsa", CryptoCrateCategory::Primitive),
    ("ecdsa", CryptoCrateCategory::Primitive),
    ("ed25519-dalek", CryptoCrateCategory::Primitive),
    ("curve25519-dalek", CryptoCrateCategory::Primitive),
    ("x25519-dalek", CryptoCrateCategory::Primitive),
    ("p256", CryptoCrateCategory::Primitive),
    ("p384", CryptoCrateCategory::Primitive),
    ("p521", CryptoCrateCategory::Primitive),
    ("k256", CryptoCrateCategory::Primitive),
];

/// One package entry in the cargo-auditable `.dep-v0` JSON.
#[derive(Debug, Deserialize)]
struct AuditPackage {
    name: String,
    version: String,
    #[serde(default = "default_kind")]
    kind: String,
}

fn default_kind() -> String {
    "runtime".to_string()
}

#[derive(Debug, Deserialize)]
struct AuditData {
    packages: Vec<AuditPackage>,
}

/// Parse zlib-compressed cargo-auditable JSON from a `.dep-v0` section and
/// return the crypto-relevant runtime crates.
pub fn parse_cargo_audit(section_bytes: &[u8]) -> Option<Vec<CryptoCrate>> {
    use std::io::Read;

    let mut decoder = flate2::read::ZlibDecoder::new(section_bytes);
    // Audit data is small (tens of KB); cap decompression at 16 MB to avoid
    // a hostile section blowing up memory.
    let mut json = Vec::new();
    decoder
        .by_ref()
        .take(16 * 1024 * 1024)
        .read_to_end(&mut json)
        .ok()?;

    let data: AuditData = serde_json::from_slice(&json).ok()?;
    let mut crates: Vec<CryptoCrate> = data
        .packages
        .iter()
        .filter(|p| p.kind == "runtime")
        .filter_map(|p| {
            CRYPTO_CRATES
                .iter()
                .find(|(name, _)| *name == p.name)
                .map(|(_, category)| CryptoCrate {
                    name: p.name.clone(),
                    version: p.version.clone(),
                    category: *category,
                })
        })
        .collect();
    crates.sort_by(|a, b| a.name.cmp(&b.name));
    crates.dedup();
    Some(crates)
}

/// Extract a version from a version-mangled symbol prefix like
/// `ring_core_0_17_14_` -> "0.17.14". `bytes` starts right after the prefix.
fn mangled_version(bytes: &[u8]) -> Option<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    for &b in bytes.iter().take(24) {
        match b {
            b'0'..=b'9' => current.push(b as char),
            b'_' => {
                if current.is_empty() {
                    break;
                }
                segments.push(std::mem::take(&mut current));
                if segments.len() == 3 {
                    return Some(segments.join("."));
                }
            }
            _ => break,
        }
    }
    None
}

fn find_version_after(data: &[u8], prefix: &[u8]) -> Option<String> {
    let finder = memchr::memmem::Finder::new(prefix);
    finder
        .find_iter(data)
        .find_map(|pos| mangled_version(&data[pos + prefix.len()..]))
}

/// TLS 1.3 / 1.2 cipher-suite names as they appear in every TLS stack's
/// rodata. Present without libssl linkage, they mean an embedded TLS stack.
const CIPHER_SUITE_STRINGS: &[&[u8]] = &[
    b"TLS_AES_128_GCM_SHA256",
    b"TLS_AES_256_GCM_SHA384",
    b"TLS_CHACHA20_POLY1305_SHA256",
    b"ECDHE-RSA-AES128-GCM-SHA256",
];

/// Scan raw ELF bytes for vendored crypto signatures.
///
/// `links_libcrypto` / `links_libssl` come from DT_NEEDED and gate the
/// heuristics that only make sense for statically linked crypto.
pub fn scan_signatures(
    data: &[u8],
    links_libcrypto: bool,
    links_libssl: bool,
) -> Vec<VendoredCrypto> {
    use memchr::memmem;

    let mut found = Vec::new();

    if memmem::find(data, b"ring_core_").is_some() {
        found.push(VendoredCrypto {
            kind: VendoredCryptoKind::Ring,
            version: find_version_after(data, b"ring_core_"),
            evidence: "ring_core_ symbol prefix".to_string(),
        });
    }

    if let Some(version) = find_version_after(data, b"aws_lc_") {
        found.push(VendoredCrypto {
            kind: VendoredCryptoKind::AwsLc,
            version: Some(version),
            evidence: "aws_lc_ version-mangled symbol prefix".to_string(),
        });
    } else if memmem::find(data, b"AWS-LC").is_some() {
        found.push(VendoredCrypto {
            kind: VendoredCryptoKind::AwsLc,
            version: None,
            evidence: "AWS-LC version string".to_string(),
        });
    }

    if memmem::find(data, b"BoringSSL").is_some() {
        found.push(VendoredCrypto {
            kind: VendoredCryptoKind::BoringSsl,
            version: None,
            evidence: "BoringSSL version string".to_string(),
        });
    }

    // An OpenSSL version banner inside a binary that does not link libcrypto
    // means OpenSSL was compiled in. The banner alone is not enough: programs
    // that only report their build config (git's `scalar diagnose`, curl -V)
    // embed OPENSSL_VERSION_TEXT without containing any OpenSSL code, so
    // require rodata that only ships inside libcrypto itself.
    let libcrypto_rodata = [
        b"common libcrypto routines".as_slice(),
        b"OpenSSL default provider".as_slice(),
        b"OPENSSL_cleanse".as_slice(),
    ];
    if !links_libcrypto
        && libcrypto_rodata
            .iter()
            .any(|s| memmem::find(data, s).is_some())
    {
        if let Some(version) = find_openssl_banner(data) {
            found.push(VendoredCrypto {
                kind: VendoredCryptoKind::OpenSslStatic,
                version: Some(version),
                evidence: "OpenSSL version banner without libcrypto linkage".to_string(),
            });
        }
    }

    // Cipher-suite rodata without libssl linkage: embedded TLS stack.
    // Only report when nothing above already explains it.
    if found.is_empty() && !links_libssl {
        if let Some(suite) = CIPHER_SUITE_STRINGS
            .iter()
            .find(|s| memmem::find(data, s).is_some())
        {
            found.push(VendoredCrypto {
                kind: VendoredCryptoKind::EmbeddedTls,
                version: None,
                evidence: format!(
                    "cipher suite string {} without libssl linkage",
                    String::from_utf8_lossy(suite)
                ),
            });
        }
    }

    found
}

/// Find an `OpenSSL X.Y.Z` version banner and return the version.
fn find_openssl_banner(data: &[u8]) -> Option<String> {
    let prefix = b"OpenSSL ";
    let finder = memchr::memmem::Finder::new(prefix);
    for pos in finder.find_iter(data) {
        let rest = &data[pos + prefix.len()..];
        let version: String = rest
            .iter()
            .take(16)
            .take_while(|&&b| b.is_ascii_digit() || b == b'.')
            .map(|&b| b as char)
            .collect();
        // Require a full X.Y.Z so prose mentioning "OpenSSL" doesn't match.
        if version.matches('.').count() >= 2 && !version.ends_with('.') {
            return Some(version);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::crypto::*;
    use std::io::Write;

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn ring_signature_with_version() {
        let data = b"garbage ring_core_0_17_14_OPENSSL_memcpy more";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::Ring);
        assert_eq!(found[0].version.as_deref(), Some("0.17.14"));
    }

    #[test]
    fn aws_lc_mangled_symbol() {
        let data = b"xx aws_lc_0_39_0_EVP_aead_aes_128_gcm yy";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::AwsLc);
        assert_eq!(found[0].version.as_deref(), Some("0.39.0"));
    }

    #[test]
    fn aws_lc_banner_only() {
        let data = b"built with AWS-LC and love";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::AwsLc);
        assert_eq!(found[0].version, None);
    }

    #[test]
    fn openssl_banner_static() {
        let data = b"common libcrypto routines OpenSSL 3.0.7 1 Nov 2022";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::OpenSslStatic);
        assert_eq!(found[0].version.as_deref(), Some("3.0.7"));
    }

    #[test]
    fn openssl_banner_ignored_when_dynamic() {
        let data = b"common libcrypto routines OpenSSL 3.0.7 1 Nov 2022";
        assert!(scan_signatures(data, true, false).is_empty());
    }

    #[test]
    fn openssl_banner_alone_not_matched() {
        // git's scalar embeds OPENSSL_VERSION_TEXT for `scalar diagnose`
        // without containing any OpenSSL code.
        let data = b"libcurl: %s OpenSSL 3.2.2 4 Jun 2024 OpenSSL: %s";
        assert!(scan_signatures(data, false, false).is_empty());
    }

    #[test]
    fn openssl_prose_not_matched() {
        let data = b"see the OpenSSL docs for details";
        assert!(scan_signatures(data, false, false).is_empty());
    }

    #[test]
    fn embedded_tls_cipher_suite() {
        let data = b"handshake TLS_AES_128_GCM_SHA256 done";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::EmbeddedTls);
    }

    #[test]
    fn cipher_suite_ignored_when_libssl_linked() {
        let data = b"handshake TLS_AES_128_GCM_SHA256 done";
        assert!(scan_signatures(data, false, true).is_empty());
    }

    #[test]
    fn cipher_suite_not_double_reported() {
        let data = b"ring_core_0_17_14_x TLS_AES_128_GCM_SHA256";
        let found = scan_signatures(data, false, false);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::Ring);
    }

    #[test]
    fn clean_binary_no_findings() {
        let data = b"just a normal binary with strings in it";
        assert!(scan_signatures(data, false, false).is_empty());
    }

    #[test]
    fn audit_data_filters_and_categorizes() {
        let json = br#"{"packages":[
            {"name":"rustls","version":"0.23.40"},
            {"name":"ring","version":"0.17.14"},
            {"name":"sha2","version":"0.10.9"},
            {"name":"cc","version":"1.0.0","kind":"build"},
            {"name":"serde","version":"1.0.0"}
        ]}"#;
        let crates = parse_cargo_audit(&zlib(json)).unwrap();
        let names: Vec<&str> = crates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["ring", "rustls", "sha2"]);
        assert_eq!(crates[0].category, CryptoCrateCategory::Provider);
        assert_eq!(crates[1].category, CryptoCrateCategory::TlsStack);
        assert_eq!(crates[2].category, CryptoCrateCategory::Primitive);
    }

    #[test]
    fn audit_data_build_kind_excluded() {
        let json = br#"{"packages":[{"name":"ring","version":"0.17.14","kind":"build"}]}"#;
        let crates = parse_cargo_audit(&zlib(json)).unwrap();
        assert!(crates.is_empty());
    }

    #[test]
    fn audit_data_corrupt_returns_none() {
        assert!(parse_cargo_audit(b"not zlib at all").is_none());
        assert!(parse_cargo_audit(&zlib(b"not json")).is_none());
    }

    #[test]
    fn mangled_version_stops_at_non_digit() {
        assert_eq!(mangled_version(b"0_17_14_foo"), Some("0.17.14".to_string()));
        assert_eq!(mangled_version(b"0_17_x"), None);
        assert_eq!(mangled_version(b"abc"), None);
    }

    #[test]
    fn crypto_implementation_sonames() {
        assert!(is_crypto_implementation("libcrypto.so.3"));
        assert!(is_crypto_implementation("libssl.so.3.5.5"));
        assert!(is_crypto_implementation("libgnutls.so.30"));
        assert!(!is_crypto_implementation("libcurl.so.4"));
        assert!(!is_crypto_implementation("libcryptofoo.so.1"));
    }

    #[test]
    fn summary_formats() {
        let info = CryptoInfo {
            vendored: vec![VendoredCrypto {
                kind: VendoredCryptoKind::Ring,
                version: Some("0.17.14".to_string()),
                evidence: "x".to_string(),
            }],
            audit_crates: vec![
                CryptoCrate {
                    name: "ring".to_string(),
                    version: "0.17.14".to_string(),
                    category: CryptoCrateCategory::Provider,
                },
                CryptoCrate {
                    name: "sha2".to_string(),
                    version: "0.10.9".to_string(),
                    category: CryptoCrateCategory::Primitive,
                },
            ],
            links_libcrypto: false,
            links_libssl: false,
        };
        assert_eq!(info.summary(), "ring 0.17.14, sha2 0.10.9 (crate)");
        assert!(info.has_vendored_crypto());
    }

    #[test]
    fn tls_stack_alone_is_not_vendored_crypto() {
        let info = CryptoInfo {
            vendored: vec![],
            audit_crates: vec![CryptoCrate {
                name: "rustls".to_string(),
                version: "0.23.40".to_string(),
                category: CryptoCrateCategory::TlsStack,
            }],
            links_libcrypto: false,
            links_libssl: false,
        };
        assert!(!info.has_vendored_crypto());
        assert!(!info.is_empty());
    }
}
