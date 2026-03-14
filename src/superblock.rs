use crate::checksum;
use crate::error::{Error, Result};
use crate::io::{Le, ReadAt};

/// HDF5 file signature: `\x89HDF\r\n\x1a\n`
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// The "undefined address" sentinel: all 0xFF bytes.
pub const UNDEF_ADDR: u64 = u64::MAX;

/// Parsed superblock (v2 or v3).
///
/// ## On-disk layout (superblock v2/v3)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
///   0       8   Signature (\x89HDF\r\n\x1a\n)
///   8       1   Superblock version (2 or 3)
///   9       1   Size of offsets (typically 8)
///  10       1   Size of lengths (typically 8)
///  11       1   File consistency flags
///  12       O   Base address (O = size of offsets)
///  12+O     O   Superblock extension address (UNDEF if absent)
///  12+2O    O   End of file address
///  12+3O    O   Root group object header address
///  12+4O    4   Superblock checksum (lookup3)
/// ```
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version: 2 or 3.
    pub version: u8,
    /// Size of file offsets in bytes (4 or 8).
    pub size_of_offsets: u8,
    /// Size of file lengths in bytes (4 or 8).
    pub size_of_lengths: u8,
    /// File consistency flags.
    ///
    /// Bit 0: file consistency flags are valid.
    /// Bit 1: SWMR write access is in progress (v3 only).
    /// Bit 2: SWMR write access was not closed properly (v3 only).
    pub file_consistency_flags: u8,
    /// Base address for file (usually 0).
    pub base_address: u64,
    /// Address of the superblock extension object header, or `UNDEF_ADDR`.
    pub superblock_extension_address: u64,
    /// End-of-file address.
    pub end_of_file_address: u64,
    /// Address of the root group's object header.
    pub root_group_object_header_address: u64,
    /// The stored checksum (for reference).
    pub checksum: u32,
}

impl Superblock {
    /// Total serialized size of the superblock (including signature).
    pub fn serialized_size(&self) -> u64 {
        // 8 (sig) + 1 (ver) + 1 (offsets) + 1 (lengths) + 1 (flags)
        // + 4 * size_of_offsets + 4 (checksum)
        12 + 4 * self.size_of_offsets as u64 + 4
    }

    /// Parse a superblock from the given reader starting at `offset`.
    ///
    /// Validates the signature, version (must be 2 or 3), and checksum.
    pub fn parse<R: ReadAt + ?Sized>(reader: &R, offset: u64) -> Result<Self> {
        // Read signature
        let mut sig = [0u8; 8];
        reader
            .read_exact_at(offset, &mut sig)
            .map_err(Error::Io)?;

        if sig != HDF5_SIGNATURE {
            return Err(Error::InvalidSignature { offset });
        }

        let version = Le::read_u8(reader, offset + 8).map_err(Error::Io)?;
        if version != 2 && version != 3 {
            return Err(Error::UnsupportedSuperblockVersion { version });
        }

        let size_of_offsets = Le::read_u8(reader, offset + 9).map_err(Error::Io)?;
        let size_of_lengths = Le::read_u8(reader, offset + 10).map_err(Error::Io)?;
        let file_consistency_flags = Le::read_u8(reader, offset + 11).map_err(Error::Io)?;

        let o = size_of_offsets;
        let mut pos = offset + 12;

        let base_address = Le::read_offset(reader, pos, o).map_err(Error::Io)?;
        pos += o as u64;

        let superblock_extension_address = Le::read_offset(reader, pos, o).map_err(Error::Io)?;
        pos += o as u64;

        let end_of_file_address = Le::read_offset(reader, pos, o).map_err(Error::Io)?;
        pos += o as u64;

        let root_group_object_header_address =
            Le::read_offset(reader, pos, o).map_err(Error::Io)?;
        pos += o as u64;

        let stored_checksum = Le::read_u32(reader, pos).map_err(Error::Io)?;

        // Validate checksum: computed over bytes from signature through the address before checksum
        let checksum_len = (pos - offset) as usize;
        let mut checksum_data = vec![0u8; checksum_len];
        reader
            .read_exact_at(offset, &mut checksum_data)
            .map_err(Error::Io)?;
        let computed = checksum::lookup3(&checksum_data);

        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        Ok(Superblock {
            version,
            size_of_offsets,
            size_of_lengths,
            file_consistency_flags,
            base_address,
            superblock_extension_address,
            end_of_file_address,
            root_group_object_header_address,
            checksum: stored_checksum,
        })
    }

    /// Returns true if this is a superblock v3 with SWMR write in progress.
    pub fn swmr_write_in_progress(&self) -> bool {
        self.version >= 3 && (self.file_consistency_flags & 0x02) != 0
    }

    /// Returns true if the superblock extension is present.
    pub fn has_extension(&self) -> bool {
        self.superblock_extension_address != UNDEF_ADDR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid superblock v2 in a byte buffer.
    fn make_superblock_v2() -> Vec<u8> {
        let mut buf = Vec::new();
        // Signature
        buf.extend_from_slice(&HDF5_SIGNATURE);
        // Version
        buf.push(2);
        // Size of offsets
        buf.push(8);
        // Size of lengths
        buf.push(8);
        // File consistency flags
        buf.push(0);
        // Base address (8 bytes LE)
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Superblock extension address (UNDEF)
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        // End of file address
        buf.extend_from_slice(&4096u64.to_le_bytes());
        // Root group object header address
        buf.extend_from_slice(&96u64.to_le_bytes());
        // Checksum (compute over everything before this point)
        let cksum = crate::checksum::lookup3(&buf);
        buf.extend_from_slice(&cksum.to_le_bytes());
        buf
    }

    #[test]
    fn parse_valid_v2() {
        let buf = make_superblock_v2();
        let sb = Superblock::parse(buf.as_slice(), 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.size_of_offsets, 8);
        assert_eq!(sb.size_of_lengths, 8);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.superblock_extension_address, UNDEF_ADDR);
        assert_eq!(sb.end_of_file_address, 4096);
        assert_eq!(sb.root_group_object_header_address, 96);
        assert!(!sb.has_extension());
        assert!(!sb.swmr_write_in_progress());
    }

    #[test]
    fn reject_bad_signature() {
        let mut buf = make_superblock_v2();
        buf[0] = 0x00; // corrupt signature
        let err = Superblock::parse(buf.as_slice(), 0).unwrap_err();
        assert!(matches!(err, Error::InvalidSignature { offset: 0 }));
    }

    #[test]
    fn reject_v0_superblock() {
        let mut buf = make_superblock_v2();
        buf[8] = 0; // set version to 0
        // Recompute checksum
        let cksum_offset = buf.len() - 4;
        let cksum = crate::checksum::lookup3(&buf[..cksum_offset]);
        buf[cksum_offset..].copy_from_slice(&cksum.to_le_bytes());
        let err = Superblock::parse(buf.as_slice(), 0).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedSuperblockVersion { version: 0 }
        ));
    }

    #[test]
    fn reject_bad_checksum() {
        let mut buf = make_superblock_v2();
        let last = buf.len() - 1;
        buf[last] ^= 0xFF; // corrupt checksum
        let err = Superblock::parse(buf.as_slice(), 0).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }
}
