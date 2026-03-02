use std::io::Cursor;

use crate::nvidia::ComputeCapability;
use anyhow::{Context, Result};

/// Magic bytes for a CUDA fatbinary header (little-endian `0xBA55ED50`).
const FATBIN_MAGIC: [u8; 4] = [0x50, 0xED, 0x55, 0xBA];

/// A single entry extracted from a CUDA fatbin section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FatbinEntry {
    /// The SM architecture (e.g. 70, 86, 90, 100).
    pub sm_arch: u32,
    /// True if this is compiled SASS (cubin), false if PTX (JIT-compatible).
    pub is_cubin: bool,
}

impl FatbinEntry {
    /// Convert the raw `sm_arch` value to a `ComputeCapability`.
    ///
    /// sm_arch 90  -> CC 9.0
    /// sm_arch 86  -> CC 8.6
    /// sm_arch 100 -> CC 10.0
    pub fn compute_capability(&self) -> ComputeCapability {
        ComputeCapability::new(self.sm_arch / 10, self.sm_arch % 10)
    }
}

/// Scan a `.nv_fatbin` section for concatenated fatbin blobs and extract architecture entries.
///
/// The section may contain multiple fatbin blobs (one per translation unit). We scan for
/// the fatbin magic (`0xBA55ED50`) and attempt to parse each occurrence.
pub fn extract_architectures(section_bytes: &[u8]) -> Vec<FatbinEntry> {
    let mut entries = Vec::new();

    for offset in find_magic_offsets(section_bytes, &FATBIN_MAGIC) {
        let slice = &section_bytes[offset..];
        let cursor = Cursor::new(slice);

        match fatbinary::FatBinary::read(cursor) {
            Ok(fatbin) => {
                for entry in fatbin.entries() {
                    entries.push(FatbinEntry {
                        sm_arch: entry.get_sm_arch(),
                        is_cubin: entry.contains_elf(),
                    });
                }
            }
            Err(e) => {
                tracing::debug!(
                    offset,
                    error = %e,
                    "failed to parse fatbin at offset, skipping"
                );
            }
        }
    }

    entries
}

/// Combined result of scanning an ELF: fatbin entries + dynamic section metadata.
#[derive(Debug, Clone)]
pub struct ElfInfo {
    pub fatbin_entries: Vec<FatbinEntry>,
    /// Shared library names from DT_NEEDED entries (e.g. "libcublas.so.12").
    pub needed: Vec<String>,
    /// DT_SONAME: the canonical shared object name declared by this library.
    pub soname: Option<String>,
    /// DT_RPATH entries (deprecated, but still honored by the dynamic linker).
    pub rpath: Vec<String>,
    /// DT_RUNPATH entries (preferred over DT_RPATH by modern linkers).
    pub runpath: Vec<String>,
}

/// Scan an ELF binary for both CUDA fatbin entries and DT_NEEDED dependencies.
///
/// Parses the ELF once and extracts:
/// - All `.nv_fatbin` section entries (deduplicated)
/// - All DT_NEEDED sonames from the dynamic section
pub fn scan_elf_with_deps(elf_bytes: &[u8]) -> Result<ElfInfo> {
    let elf = goblin::elf::Elf::parse(elf_bytes).context("failed to parse ELF")?;

    // Extract dynamic section metadata
    let needed: Vec<String> = elf.libraries.iter().map(|s| s.to_string()).collect();
    let soname = elf.soname.map(|s| s.to_string());
    let rpath: Vec<String> = elf.rpaths.iter().map(|s| s.to_string()).collect();
    let runpath: Vec<String> = elf.runpaths.iter().map(|s| s.to_string()).collect();

    // Find .nv_fatbin sections
    let sections: Vec<(usize, usize)> = elf
        .section_headers
        .iter()
        .filter_map(|sh| {
            let name = elf.shdr_strtab.get_at(sh.sh_name)?;
            if name == ".nv_fatbin" {
                let offset = sh.sh_offset as usize;
                let size = sh.sh_size as usize;
                Some((offset, size))
            } else {
                None
            }
        })
        .collect();

    let mut all_entries = Vec::new();
    for (offset, size) in sections {
        let end = offset.saturating_add(size).min(elf_bytes.len());
        if offset >= elf_bytes.len() {
            continue;
        }
        let section_bytes = &elf_bytes[offset..end];
        all_entries.extend(extract_architectures(section_bytes));
    }

    all_entries.sort_by_key(|e| (e.sm_arch, e.is_cubin));
    all_entries.dedup();

    Ok(ElfInfo {
        fatbin_entries: all_entries,
        needed,
        soname,
        rpath,
        runpath,
    })
}

/// Find all offsets of a 4-byte magic pattern in a byte slice.
///
/// Uses `memchr::memmem` for SIMD-accelerated search instead of byte-by-byte scanning.
fn find_magic_offsets(data: &[u8], magic: &[u8; 4]) -> Vec<usize> {
    memchr::memmem::find_iter(data, magic).collect()
}

#[cfg(test)]
mod tests {
    use crate::fatbin::{
        extract_architectures, find_magic_offsets, scan_elf_with_deps, FatbinEntry, FATBIN_MAGIC,
    };
    use crate::nvidia::ComputeCapability;

    #[test]
    fn sm_arch_to_cc_basic() {
        let entry = FatbinEntry {
            sm_arch: 70,
            is_cubin: true,
        };
        assert_eq!(entry.compute_capability(), ComputeCapability::new(7, 0));
    }

    #[test]
    fn sm_arch_to_cc_with_minor() {
        let entry = FatbinEntry {
            sm_arch: 86,
            is_cubin: true,
        };
        assert_eq!(entry.compute_capability(), ComputeCapability::new(8, 6));
    }

    #[test]
    fn sm_arch_to_cc_high() {
        let entry = FatbinEntry {
            sm_arch: 100,
            is_cubin: true,
        };
        assert_eq!(entry.compute_capability(), ComputeCapability::new(10, 0));
    }

    #[test]
    fn sm_arch_to_cc_ptx_entry() {
        let entry = FatbinEntry {
            sm_arch: 120,
            is_cubin: false,
        };
        assert_eq!(entry.compute_capability(), ComputeCapability::new(12, 0));
    }

    #[test]
    fn find_magic_no_match() {
        let data = [0u8; 16];
        assert!(find_magic_offsets(&data, &FATBIN_MAGIC).is_empty());
    }

    #[test]
    fn find_magic_single_match() {
        let mut data = vec![0u8; 8];
        data[2..6].copy_from_slice(&FATBIN_MAGIC);
        let offsets = find_magic_offsets(&data, &FATBIN_MAGIC);
        assert_eq!(offsets, vec![2]);
    }

    #[test]
    fn find_magic_multiple_matches() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&FATBIN_MAGIC);
        data[10..14].copy_from_slice(&FATBIN_MAGIC);
        let offsets = find_magic_offsets(&data, &FATBIN_MAGIC);
        assert_eq!(offsets, vec![0, 10]);
    }

    #[test]
    fn find_magic_empty_data() {
        let data: &[u8] = &[];
        assert!(find_magic_offsets(data, &FATBIN_MAGIC).is_empty());
    }

    #[test]
    fn find_magic_too_short() {
        let data = [0x50, 0xED, 0x55]; // 3 bytes, needs 4
        assert!(find_magic_offsets(&data, &FATBIN_MAGIC).is_empty());
    }

    #[test]
    fn extract_architectures_no_magic() {
        let data = vec![0u8; 64];
        let entries = extract_architectures(&data);
        assert!(entries.is_empty());
    }

    #[test]
    fn extract_architectures_invalid_fatbin_after_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&FATBIN_MAGIC);
        let entries = extract_architectures(&data);
        assert!(entries.is_empty());
    }

    #[test]
    fn scan_elf_with_deps_non_elf() {
        let data = b"not an elf";
        let result = scan_elf_with_deps(data);
        assert!(result.is_err());
    }

    #[test]
    fn scan_elf_with_deps_empty_data() {
        let data: &[u8] = &[];
        let result = scan_elf_with_deps(data);
        assert!(result.is_err());
    }

    #[test]
    fn fatbin_entry_dedup() {
        let mut entries = vec![
            FatbinEntry {
                sm_arch: 90,
                is_cubin: true,
            },
            FatbinEntry {
                sm_arch: 90,
                is_cubin: true,
            },
            FatbinEntry {
                sm_arch: 90,
                is_cubin: false,
            },
        ];
        entries.sort_by_key(|e| (e.sm_arch, e.is_cubin));
        entries.dedup();
        assert_eq!(entries.len(), 2);
    }
}
