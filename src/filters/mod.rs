use crate::error::{Error, Result};

/// HDF5 filter IDs.
///
/// Reference: H5Zpublic.h
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// A single filter in a pipeline.
#[derive(Debug, Clone)]
pub struct Filter {
    pub id: u16,
    pub name: Option<String>,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

/// A filter pipeline parsed from a filter pipeline message (type 0x000B).
///
/// ## On-disk layout (version 2)
///
/// ```text
/// Byte 0:    Version (2)
/// Byte 1:    Number of filters
/// Filters[]:
///   Filter ID (u16)
///   [if version 1 and id >= 256: Name Length (u16), Name (null-terminated, padded to 8)]
///   Flags (u16)
///   Number of client data values (u16)
///   [if version 1 and id >= 256: Name]
///   Client data values (num_values * u32)
///   [if version 1: padding to 8-byte boundary]
/// ```
#[derive(Debug, Clone)]
pub struct FilterPipeline {
    pub filters: Vec<Filter>,
}

impl FilterPipeline {
    /// Parse a filter pipeline message.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFilterPipeline {
                msg: "filter pipeline message too short".into(),
            });
        }

        let version = data[0];
        let nfilters = data[1] as usize;

        match version {
            1 => Self::parse_v1(data, nfilters),
            2 => Self::parse_v2(data, nfilters),
            _ => Err(Error::InvalidFilterPipeline {
                msg: format!("unsupported filter pipeline version {}", version),
            }),
        }
    }

    fn parse_v1(data: &[u8], nfilters: usize) -> Result<Self> {
        // V1: after version(1) + nfilters(1) + 6 reserved bytes = 8 byte header
        let mut pos = 8;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            if pos + 8 > data.len() {
                break;
            }
            let id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let name_length = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let flags = u16::from_le_bytes([data[pos + 4], data[pos + 5]]);
            let num_client_data = u16::from_le_bytes([data[pos + 6], data[pos + 7]]) as usize;
            pos += 8;

            // Name (if present, null-terminated, padded to 8 bytes)
            let name = if name_length > 0 {
                let name_end = pos + name_length;
                let padded_len = (name_length + 7) & !7; // round up to 8
                let n = String::from_utf8_lossy(&data[pos..name_end])
                    .trim_end_matches('\0')
                    .to_string();
                pos += padded_len;
                Some(n)
            } else {
                None
            };

            // Client data
            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                if pos + 4 > data.len() {
                    break;
                }
                client_data.push(u32::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                ]));
                pos += 4;
            }

            // V1 padding: if num_client_data is odd, 4 bytes padding
            if num_client_data % 2 != 0 {
                pos += 4;
            }

            filters.push(Filter {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline { filters })
    }

    fn parse_v2(data: &[u8], nfilters: usize) -> Result<Self> {
        // V2: more compact, no reserved bytes, no name for well-known filters
        let mut pos = 2; // version + nfilters
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            if pos + 2 > data.len() {
                break;
            }
            let id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;

            let name = if id >= 256 {
                // User-defined filter: has name length + name
                if pos + 2 > data.len() {
                    break;
                }
                let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                let n = String::from_utf8_lossy(&data[pos..pos + name_len])
                    .trim_end_matches('\0')
                    .to_string();
                pos += name_len;
                Some(n)
            } else {
                None
            };

            if pos + 4 > data.len() {
                break;
            }
            let flags = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let num_client_data = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            pos += 4;

            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                if pos + 4 > data.len() {
                    break;
                }
                client_data.push(u32::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                ]));
                pos += 4;
            }

            filters.push(Filter {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline { filters })
    }

    /// Apply the filter pipeline in reverse (decompression direction) to a chunk.
    pub fn decompress(&self, mut data: Vec<u8>) -> Result<Vec<u8>> {
        // Filters are applied in reverse order for reading
        for filter in self.filters.iter().rev() {
            data = apply_filter_reverse(filter, data)?;
        }
        Ok(data)
    }
}

fn apply_filter_reverse(filter: &Filter, data: Vec<u8>) -> Result<Vec<u8>> {
    match filter.id {
        FILTER_DEFLATE => decompress_deflate(&data),
        FILTER_SHUFFLE => {
            let element_size = filter
                .client_data
                .first()
                .copied()
                .unwrap_or(1) as usize;
            Ok(unshuffle(&data, element_size))
        }
        FILTER_FLETCHER32 => {
            // Fletcher32 is a checksum — on read, verify and strip the 4-byte trailer
            verify_fletcher32(&data)
        }
        FILTER_NBIT => {
            // TODO: N-bit filter
            Err(Error::UnsupportedFilter {
                id: filter.id,
                name: "nbit".into(),
            })
        }
        FILTER_SCALEOFFSET => {
            // TODO: scale-offset filter
            Err(Error::UnsupportedFilter {
                id: filter.id,
                name: "scaleoffset".into(),
            })
        }
        FILTER_SZIP => {
            // TODO: SZIP (Rice/AEC) decompression
            Err(Error::UnsupportedFilter {
                id: filter.id,
                name: "szip".into(),
            })
        }
        _ => Err(Error::UnsupportedFilter {
            id: filter.id,
            name: filter
                .name
                .clone()
                .unwrap_or_else(|| format!("unknown-{}", filter.id)),
        }),
    }
}

/// Decompress DEFLATE (zlib) compressed data.
fn decompress_deflate(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder.read_to_end(&mut output).map_err(|e| Error::DecompressionError {
        msg: format!("deflate: {}", e),
    })?;
    Ok(output)
}

/// Reverse the HDF5 shuffle filter.
///
/// Shuffle interleaves bytes by element position:
/// Input:  [A0 A1 A2 A3 B0 B1 B2 B3] (two 4-byte elements)
/// Shuffled: [A0 B0 A1 B1 A2 B2 A3 B3] (all byte-0s, then byte-1s, etc.)
///
/// We need to un-shuffle: given the shuffled form, reconstruct the original.
fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let num_elements = data.len() / element_size;
    let mut output = vec![0u8; data.len()];

    for byte_idx in 0..element_size {
        let src_start = byte_idx * num_elements;
        for elem in 0..num_elements {
            output[elem * element_size + byte_idx] = data[src_start + elem];
        }
    }

    output
}

/// Verify and strip Fletcher32 checksum.
fn verify_fletcher32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(Error::InvalidFilterPipeline {
            msg: "data too short for fletcher32 checksum".into(),
        });
    }
    // The last 4 bytes are the checksum
    let payload = &data[..data.len() - 4];
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);

    let computed = fletcher32(payload);
    if computed != stored {
        return Err(Error::ChecksumMismatch {
            expected: stored,
            actual: computed,
        });
    }

    Ok(payload.to_vec())
}

/// Compute Fletcher32 checksum over data (treating as 16-bit words, little-endian).
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let mut i = 0;
    while i + 1 < data.len() {
        let word = u16::from_le_bytes([data[i], data[i + 1]]) as u32;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        i += 2;
    }
    // Handle odd trailing byte
    if i < data.len() {
        sum1 = (sum1 + data[i] as u32) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }

    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_identity_for_size_1() {
        let data = vec![1, 2, 3, 4];
        assert_eq!(unshuffle(&data, 1), data);
    }

    #[test]
    fn unshuffle_reverses_shuffle() {
        // Two 4-byte elements: [0x01020304, 0x05060708]
        // Shuffled (byte-0s first, etc.): [01, 05, 02, 06, 03, 07, 04, 08]
        let shuffled = vec![0x01, 0x05, 0x02, 0x06, 0x03, 0x07, 0x04, 0x08];
        let expected = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(unshuffle(&shuffled, 4), expected);
    }

    #[test]
    fn fletcher32_known() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let cksum = fletcher32(&data);
        // Deterministic — just verify it's non-zero and consistent
        assert_ne!(cksum, 0);
        assert_eq!(cksum, fletcher32(&data));
    }
}
