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
    #[serde(rename = "boringssl")]
    BoringSsl,
    /// Statically linked OpenSSL (openssl-src / vendored feature).
    #[serde(rename = "openssl-static")]
    OpenSslStatic,
    /// Statically linked libsodium (libsodium-sys / sodiumoxide).
    Libsodium,
    /// Go stdlib crypto/tls compiled in (default pure-Go crypto).
    GoStdlibCrypto,
    /// Go BoringCrypto (goboring): statically linked BoringSSL module.
    #[serde(rename = "go-boringcrypto")]
    GoBoringCrypto,
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
            Self::Libsodium => "libsodium",
            Self::GoStdlibCrypto => "go-stdlib-crypto",
            Self::GoBoringCrypto => "go-boringcrypto",
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
    "libsodium.so",
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
    ("libsodium-sys", CryptoCrateCategory::Provider),
    ("sodiumoxide", CryptoCrateCategory::Provider),
];

/// Go modules that indicate crypto compiled into the binary, matched against
/// `dep` lines from the embedded module info.
const GO_CRYPTO_MODULES: &[(&str, CryptoCrateCategory)] =
    &[("golang.org/x/crypto", CryptoCrateCategory::Primitive)];

/// Go module providing the dlopen-based system OpenSSL backend used by
/// Red Hat's FIPS-patched toolchain. Its presence means the binary defers to
/// the host libcrypto at runtime, so stdlib crypto markers are not vendored.
const GO_FIPS_OPENSSL_MODULE: &str = "github.com/golang-fips/openssl";

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

/// Whether any DT_NEEDED soname starts with the given library prefix.
pub fn links(needed: &[String], prefix: &str) -> bool {
    needed.iter().any(|n| n.starts_with(prefix))
}

/// Result of scanning one ELF: signature hits plus crypto-relevant Go modules
/// recovered from the embedded module info.
pub struct SignatureScan {
    pub vendored: Vec<VendoredCrypto>,
    pub go_modules: Vec<CryptoCrate>,
}

/// Scan raw ELF bytes for vendored crypto signatures.
///
/// `needed` is the DT_NEEDED soname list; linkage against the system
/// libcrypto/libssl/libsodium gates the heuristics that only make sense for
/// statically linked crypto.
pub fn scan_signatures(data: &[u8], needed: &[String]) -> SignatureScan {
    use memchr::memmem;

    let links_libcrypto = links(needed, "libcrypto.so");
    let links_libssl = links(needed, "libssl.so");
    let mut found = Vec::new();
    let mut go_modules = Vec::new();

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

    // Statically linked libsodium: its exported symbol names in a binary
    // that does not link the system libsodium.
    if !links(needed, "libsodium.so")
        && (memmem::find(data, b"sodium_init").is_some()
            || memmem::find(data, b"crypto_pwhash_argon2id").is_some())
    {
        found.push(VendoredCrypto {
            kind: VendoredCryptoKind::Libsodium,
            version: None,
            evidence: "libsodium symbol names without libsodium linkage".to_string(),
        });
    }

    // Go binaries: the buildinfo magic marks them, the pclntab keeps package
    // paths even in stripped binaries, and the module info names dependencies.
    let is_go = memmem::find(data, b"\xff Go buildinf:").is_some();
    if is_go {
        go_modules = go_crypto_modules(data);
        let has_fips_backend = go_modules_contain(data, GO_FIPS_OPENSSL_MODULE);
        let has_boring = memmem::find(data, b"goboringcrypto").is_some();
        if has_boring {
            found.push(VendoredCrypto {
                kind: VendoredCryptoKind::GoBoringCrypto,
                version: go_version(data),
                evidence: "goboringcrypto symbols (static BoringSSL module)".to_string(),
            });
        } else if !has_fips_backend
            && !links_libcrypto
            && memmem::find(data, b"crypto/tls.").is_some()
        {
            found.push(VendoredCrypto {
                kind: VendoredCryptoKind::GoStdlibCrypto,
                version: go_version(data),
                evidence: "crypto/tls package without system crypto backend".to_string(),
            });
        }
    }

    // Cipher-suite rodata without libssl linkage: embedded TLS stack.
    // Only report when nothing above already explains it; Go's crypto/tls
    // carries these strings and is handled above.
    if found.is_empty() && !is_go && !links_libssl {
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

    SignatureScan {
        vendored: found,
        go_modules,
    }
}

/// Sentinels wrapping the module info string in Go binaries
/// (`runtime/debug.modinfo`).
const GO_INFO_START: &[u8] = &[
    0x30, 0x77, 0xaf, 0x0c, 0x92, 0x74, 0x08, 0x02, 0x41, 0xe1, 0xc1, 0x07, 0xe6, 0xd6, 0x18, 0xe6,
];
const GO_INFO_END: &[u8] = &[
    0xf9, 0x32, 0x43, 0x31, 0x86, 0x18, 0x20, 0x72, 0x00, 0x82, 0x42, 0x10, 0x41, 0x16, 0x38, 0x69,
];

/// Extract the Go module info text between its sentinels.
fn go_modinfo(data: &[u8]) -> Option<&str> {
    use memchr::memmem;
    let start = memmem::find(data, GO_INFO_START)? + GO_INFO_START.len();
    let len = memmem::find(&data[start..], GO_INFO_END)?;
    std::str::from_utf8(&data[start..start + len]).ok()
}

/// Whether the module info lists a dependency on the given module path.
fn go_modules_contain(data: &[u8], module: &str) -> bool {
    go_modinfo(data).is_some_and(|info| {
        info.lines().any(|line| {
            let mut fields = line.split('\t');
            matches!(fields.next(), Some("dep" | "mod")) && fields.next() == Some(module)
        })
    })
}

/// Crypto-relevant Go modules from the embedded module info.
fn go_crypto_modules(data: &[u8]) -> Vec<CryptoCrate> {
    let Some(info) = go_modinfo(data) else {
        return Vec::new();
    };
    let mut modules: Vec<CryptoCrate> = info
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            if !matches!(fields.next(), Some("dep")) {
                return None;
            }
            let path = fields.next()?;
            let version = fields.next().unwrap_or("unknown");
            GO_CRYPTO_MODULES
                .iter()
                .find(|(name, _)| *name == path)
                .map(|(_, category)| CryptoCrate {
                    name: path.to_string(),
                    version: version.to_string(),
                    category: *category,
                })
        })
        .collect();
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    modules.dedup();
    modules
}

/// Extract the Go toolchain version (e.g. "go1.24.5") from binary strings.
fn go_version(data: &[u8]) -> Option<String> {
    let finder = memchr::memmem::Finder::new(b"go1.");
    for pos in finder.find_iter(data) {
        let rest = &data[pos..];
        let token: String = rest
            .iter()
            .take(16)
            .take_while(|&&b| b.is_ascii_alphanumeric() || b == b'.')
            .map(|&b| b as char)
            .collect();
        // Require go1.X.Y with digits so module paths like "go1x" don't match.
        let numeric = token.trim_start_matches("go");
        if numeric.matches('.').count() >= 2
            && numeric.chars().all(|c| c.is_ascii_digit() || c == '.')
        {
            return Some(token);
        }
    }
    None
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

/// One accepted finding in a crypto baseline file.
///
/// `path` is matched against the binary path, with `*` matching any run of
/// characters (wheel paths embed content hashes). With no `kinds` / `crates`
/// constraints, every finding on a matching path is accepted; otherwise every
/// vendored kind must appear in `kinds` and every flagged crate in `crates`.
#[derive(Debug, Clone, Deserialize)]
pub struct BaselineEntry {
    pub path: String,
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
    #[serde(default)]
    pub crates: Option<Vec<String>>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Accepted findings loaded from a TOML baseline file. Findings covered by an
/// entry are reported but do not fail the scan; only new findings do.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Baseline {
    #[serde(default)]
    pub accept: Vec<BaselineEntry>,
}

impl Baseline {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read baseline file {}", path.display()))?;
        toml::from_str(&content)
            .with_context(|| format!("failed to parse baseline file {}", path.display()))
    }

    /// Index of the entry accepting this binary's findings, if any.
    pub fn accepts(&self, binary_path: &str, info: &CryptoInfo) -> Option<usize> {
        self.accept
            .iter()
            .position(|entry| entry.accepts(binary_path, info))
    }
}

impl BaselineEntry {
    fn accepts(&self, binary_path: &str, info: &CryptoInfo) -> bool {
        if !glob_match(&self.path, binary_path) {
            return false;
        }
        if let Some(ref kinds) = self.kinds {
            let all_kinds = info
                .vendored
                .iter()
                .all(|v| kinds.iter().any(|k| k == &v.kind.to_string()));
            if !all_kinds {
                return false;
            }
        }
        if let Some(ref crates) = self.crates {
            let all_crates = info
                .audit_crates
                .iter()
                .filter(|c| c.category != CryptoCrateCategory::TlsStack)
                .all(|c| crates.contains(&c.name));
            if !all_crates {
                return false;
            }
        }
        true
    }
}

/// Match `pattern` against `text` where `*` matches any run of characters.
fn glob_match(pattern: &str, text: &str) -> bool {
    fn inner(p: &[u8], t: &[u8]) -> bool {
        match p.split_first() {
            None => t.is_empty(),
            Some((b'*', rest)) => (0..=t.len()).any(|skip| inner(rest, &t[skip..])),
            Some((c, rest)) => t
                .split_first()
                .is_some_and(|(tc, tr)| tc == c && inner(rest, tr)),
        }
    }
    inner(pattern.as_bytes(), text.as_bytes())
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
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::Ring);
        assert_eq!(found[0].version.as_deref(), Some("0.17.14"));
    }

    #[test]
    fn aws_lc_mangled_symbol() {
        let data = b"xx aws_lc_0_39_0_EVP_aead_aes_128_gcm yy";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::AwsLc);
        assert_eq!(found[0].version.as_deref(), Some("0.39.0"));
    }

    #[test]
    fn aws_lc_banner_only() {
        let data = b"built with AWS-LC and love";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::AwsLc);
        assert_eq!(found[0].version, None);
    }

    #[test]
    fn openssl_banner_static() {
        let data = b"common libcrypto routines OpenSSL 3.0.7 1 Nov 2022";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::OpenSslStatic);
        assert_eq!(found[0].version.as_deref(), Some("3.0.7"));
    }

    #[test]
    fn openssl_banner_ignored_when_dynamic() {
        let data = b"common libcrypto routines OpenSSL 3.0.7 1 Nov 2022";
        assert!(scan_signatures(data, &["libcrypto.so.3".to_string()])
            .vendored
            .is_empty());
    }

    #[test]
    fn openssl_banner_alone_not_matched() {
        // git's scalar embeds OPENSSL_VERSION_TEXT for `scalar diagnose`
        // without containing any OpenSSL code.
        let data = b"libcurl: %s OpenSSL 3.2.2 4 Jun 2024 OpenSSL: %s";
        assert!(scan_signatures(data, &[]).vendored.is_empty());
    }

    #[test]
    fn openssl_prose_not_matched() {
        let data = b"see the OpenSSL docs for details";
        assert!(scan_signatures(data, &[]).vendored.is_empty());
    }

    #[test]
    fn embedded_tls_cipher_suite() {
        let data = b"handshake TLS_AES_128_GCM_SHA256 done";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::EmbeddedTls);
    }

    #[test]
    fn cipher_suite_ignored_when_libssl_linked() {
        let data = b"handshake TLS_AES_128_GCM_SHA256 done";
        assert!(scan_signatures(data, &["libssl.so.3".to_string()])
            .vendored
            .is_empty());
    }

    #[test]
    fn cipher_suite_not_double_reported() {
        let data = b"ring_core_0_17_14_x TLS_AES_128_GCM_SHA256";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::Ring);
    }

    #[test]
    fn clean_binary_no_findings() {
        let data = b"just a normal binary with strings in it";
        assert!(scan_signatures(data, &[]).vendored.is_empty());
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

    fn go_binary(modinfo: &str, extra: &[u8]) -> Vec<u8> {
        let mut data = b"\xff Go buildinf:\x08\x02 go1.24.5 ".to_vec();
        data.extend_from_slice(GO_INFO_START);
        data.extend_from_slice(modinfo.as_bytes());
        data.extend_from_slice(GO_INFO_END);
        data.extend_from_slice(extra);
        data
    }

    #[test]
    fn libsodium_static() {
        let data = b"xx sodium_init yy";
        let found = scan_signatures(data, &[]).vendored;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, VendoredCryptoKind::Libsodium);
    }

    #[test]
    fn libsodium_ignored_when_dynamic() {
        let data = b"xx sodium_init yy";
        assert!(scan_signatures(data, &["libsodium.so.23".to_string()])
            .vendored
            .is_empty());
    }

    #[test]
    fn go_stdlib_crypto_detected() {
        let data = go_binary(
            "path\texample.com/app\ndep\tgolang.org/x/crypto\tv0.31.0\th1:abc\n",
            b" crypto/tls.(*Conn).Handshake ",
        );
        let scan = scan_signatures(&data, &[]);
        assert_eq!(scan.vendored.len(), 1);
        assert_eq!(scan.vendored[0].kind, VendoredCryptoKind::GoStdlibCrypto);
        assert_eq!(scan.vendored[0].version.as_deref(), Some("go1.24.5"));
        assert_eq!(scan.go_modules.len(), 1);
        assert_eq!(scan.go_modules[0].name, "golang.org/x/crypto");
        assert_eq!(scan.go_modules[0].version, "v0.31.0");
    }

    #[test]
    fn go_boringcrypto_detected() {
        let data = go_binary("path\texample.com/app\n", b" goboringcrypto_AES ");
        let scan = scan_signatures(&data, &[]);
        assert_eq!(scan.vendored.len(), 1);
        assert_eq!(scan.vendored[0].kind, VendoredCryptoKind::GoBoringCrypto);
    }

    #[test]
    fn go_fips_backend_suppresses_stdlib_finding() {
        let data = go_binary(
            "path\texample.com/app\ndep\tgithub.com/golang-fips/openssl\tv2.0.0\th1:x\n",
            b" crypto/tls.(*Conn).Handshake ",
        );
        assert!(scan_signatures(&data, &[]).vendored.is_empty());
    }

    #[test]
    fn go_binary_skips_cipher_suite_heuristic() {
        let data = go_binary("path\texample.com/app\n", b" TLS_AES_128_GCM_SHA256 ");
        // Not linked against libcrypto, has cipher suites, but no crypto/tls
        // marker: a Go binary without TLS should not trip EmbeddedTls.
        assert!(scan_signatures(&data, &[]).vendored.is_empty());
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("/usr/bin/uv", "/usr/bin/uv"));
        assert!(glob_match("/opt/*/bin/uv*", "/opt/app-root/bin/uvx"));
        assert!(glob_match(
            "*libcrypto-*.so.1.1.1k",
            "/x/libcrypto-bdaed0ea.so.1.1.1k"
        ));
        assert!(!glob_match("/usr/bin/uv", "/usr/bin/uvx"));
        assert!(!glob_match("/opt/*/uv", "/usr/bin/uv"));
    }

    #[test]
    fn baseline_accepts_and_constrains() {
        let baseline: Baseline = toml::from_str(
            r#"
            [[accept]]
            path = "/opt/app-root/bin/uv*"
            kinds = ["aws-lc"]
            crates = ["ring", "rustls", "aws-lc-rs", "aws-lc-sys"]
            reason = "uv is a build tool"

            [[accept]]
            path = "/usr/bin/anything"
            "#,
        )
        .unwrap();

        let accepted = CryptoInfo {
            vendored: vec![VendoredCrypto {
                kind: VendoredCryptoKind::AwsLc,
                version: None,
                evidence: "x".to_string(),
            }],
            audit_crates: vec![CryptoCrate {
                name: "ring".to_string(),
                version: "0.17.14".to_string(),
                category: CryptoCrateCategory::Provider,
            }],
            links_libcrypto: false,
            links_libssl: false,
        };
        assert_eq!(baseline.accepts("/opt/app-root/bin/uv", &accepted), Some(0));
        assert_eq!(
            baseline.accepts("/opt/app-root/bin/uvx", &accepted),
            Some(0)
        );
        // Unconstrained entry accepts any findings on its path
        assert_eq!(baseline.accepts("/usr/bin/anything", &accepted), Some(1));

        // A kind outside the allowed set is not accepted
        let ring_static = CryptoInfo {
            vendored: vec![VendoredCrypto {
                kind: VendoredCryptoKind::Ring,
                version: None,
                evidence: "x".to_string(),
            }],
            ..accepted.clone()
        };
        assert_eq!(baseline.accepts("/opt/app-root/bin/uv", &ring_static), None);
        // Wrong path is not accepted
        assert_eq!(baseline.accepts("/usr/local/bin/other", &accepted), None);
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
