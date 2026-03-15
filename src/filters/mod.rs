use crate::error::Error;
use crate::error::Result;

/// HDF5 filter IDs.
///
/// Reference: H5Zpublic.h
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// Third-party filter: LZF (registered by h5py).
pub const FILTER_LZF: u16 = 32000;

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
            let element_size = filter.client_data.first().copied().unwrap_or(1) as usize;
            Ok(unshuffle(&data, element_size))
        }
        FILTER_FLETCHER32 => {
            // Fletcher32 is a checksum — on read, verify and strip the 4-byte trailer
            verify_fletcher32(&data)
        }
        FILTER_NBIT => decompress_nbit(&data, &filter.client_data),
        FILTER_SCALEOFFSET => decompress_scaleoffset(&data, &filter.client_data),
        FILTER_LZF => {
            let out_size = filter.client_data.get(2).copied().unwrap_or(0) as usize;
            decompress_lzf(&data, out_size)
        }
        FILTER_SZIP => Err(Error::UnsupportedFilter {
            id: filter.id,
            name: "szip".into(),
        }),
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
    use std::io::Read;

    use flate2::read::ZlibDecoder;

    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| Error::DecompressionError {
            msg: format!("deflate: {}", e),
        })?;
    Ok(output)
}

/// Decompress LZF compressed data.
///
/// LZF is a simple LZ77 variant by Marc Lehmann, used by h5py (filter ID 32000).
/// Format: stream of literal runs and back-references.
///   - If byte >> 5 == 0: literal run, length = byte + 1, followed by that many bytes.
///   - Otherwise: back-reference. length = (byte >> 5) + 2.
///     If length == 9 (all 3 high bits set), read one more byte and add to length + 2.
///     Offset = ((byte & 0x1f) << 8) | next_byte, plus 1.
fn decompress_lzf(data: &[u8], out_size_hint: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(if out_size_hint > 0 {
        out_size_hint
    } else {
        data.len() * 2
    });
    let mut ip = 0; // input position

    while ip < data.len() {
        let ctrl = data[ip] as usize;
        ip += 1;

        if ctrl < (1 << 5) {
            // Literal run: ctrl + 1 bytes
            let len = ctrl + 1;
            if ip + len > data.len() {
                return Err(Error::DecompressionError {
                    msg: format!("lzf: literal overflows input at ip={}, len={}", ip, len),
                });
            }
            output.extend_from_slice(&data[ip..ip + len]);
            ip += len;
        } else {
            // Back-reference
            let mut len = (ctrl >> 5) + 2;
            if len == 9 {
                // Extended length
                if ip >= data.len() {
                    return Err(Error::DecompressionError {
                        msg: "lzf: truncated extended length".into(),
                    });
                }
                len += data[ip] as usize;
                ip += 1;
            }

            if ip >= data.len() {
                return Err(Error::DecompressionError {
                    msg: "lzf: truncated back-reference offset".into(),
                });
            }
            let offset = ((ctrl & 0x1f) << 8) | (data[ip] as usize);
            ip += 1;

            // offset is 0-based from current output position - 1
            let ref_pos = output.len().wrapping_sub(offset + 1);
            if ref_pos >= output.len() {
                return Err(Error::DecompressionError {
                    msg: format!(
                        "lzf: back-reference out of bounds: ref_pos={}, output_len={}",
                        ref_pos,
                        output.len()
                    ),
                });
            }

            // Copy byte-by-byte since source and dest may overlap
            for i in 0..len {
                let b = output[ref_pos + i];
                output.push(b);
            }
        }
    }

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
    // The last 4 bytes are the checksum, stored big-endian
    let payload = &data[..data.len() - 4];
    let stored = u32::from_be_bytes([
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

/// Compute Fletcher32 checksum over data (little-endian 16-bit words).
///
/// Matches the HDF5 library's `H5_checksum_fletcher32` on LE systems.
/// HDF5 returns (sum1 << 16) | sum2 where sum1 is the simple accumulator
/// and sum2 is the running cumulative sum.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let mut remaining = data.len() / 2;
    let mut i = 0;

    while remaining > 0 {
        let tlen = remaining.min(360);
        remaining -= tlen;
        for _ in 0..tlen {
            let word = u16::from_le_bytes([data[i], data[i + 1]]) as u32;
            sum1 += word;
            sum2 += sum1;
            i += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Handle odd trailing byte
    if data.len() % 2 != 0 {
        sum1 += (data[i] as u32) << 8;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 += sum1;
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Second reduction step
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum1 << 16) | sum2
}

// ========== N-bit filter ==========

const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
const NBIT_ORDER_LE: u32 = 0;

struct NbitAtomicParms {
    size: u32,
    order: u32,
    precision: u32,
    offset: u32,
}

struct NbitState<'a> {
    buffer: &'a [u8],
    j: usize,
    buf_len: usize, // bits remaining in current byte
}

impl<'a> NbitState<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        NbitState {
            buffer,
            j: 0,
            buf_len: 8,
        }
    }

    fn next_byte(&mut self) {
        self.j += 1;
        self.buf_len = 8;
    }

    fn cur(&self) -> u8 {
        if self.j < self.buffer.len() {
            self.buffer[self.j]
        } else {
            0
        }
    }
}

fn nbit_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: usize,
    begin_i: usize,
    end_i: usize,
    state: &mut NbitState,
    p: &NbitAtomicParms,
    datatype_len: u32,
) {
    let (dat_len, dat_offset);

    if begin_i != end_i {
        if k == begin_i {
            dat_len = 8 - ((datatype_len - p.precision - p.offset) % 8) as usize;
            dat_offset = 0;
        } else if k == end_i {
            dat_len = (8 - (p.offset % 8)) as usize;
            dat_offset = 8 - dat_len;
        } else {
            dat_len = 8;
            dat_offset = 0;
        }
    } else {
        dat_offset = (p.offset % 8) as usize;
        dat_len = p.precision as usize;
    }

    let val = state.cur();

    if state.buf_len > dat_len {
        data[data_offset + k] = (((val >> (state.buf_len - dat_len)) as u32
            & ((1u32 << dat_len) - 1))
            << dat_offset) as u8;
        state.buf_len -= dat_len;
    } else {
        data[data_offset + k] = ((((val as u32) & ((1u32 << state.buf_len) - 1))
            << (dat_len - state.buf_len))
            << dat_offset) as u8;
        let remaining = dat_len - state.buf_len;
        state.next_byte();
        if remaining == 0 {
            return;
        }
        let val2 = state.cur();
        data[data_offset + k] |= ((((val2 >> (state.buf_len - remaining)) as u32)
            & ((1u32 << remaining) - 1))
            << dat_offset) as u8;
        state.buf_len -= remaining;
    }
}

fn nbit_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    p: &NbitAtomicParms,
) {
    let datatype_len = p.size * 8;

    if p.order == NBIT_ORDER_LE {
        let begin_i = if (p.precision + p.offset) % 8 != 0 {
            ((p.precision + p.offset) / 8) as usize
        } else {
            ((p.precision + p.offset) / 8 - 1) as usize
        };
        let end_i = (p.offset / 8) as usize;

        let mut k = begin_i as isize;
        while k >= end_i as isize {
            nbit_decompress_one_byte(
                data,
                data_offset,
                k as usize,
                begin_i,
                end_i,
                state,
                p,
                datatype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = ((datatype_len - p.precision - p.offset) / 8) as usize;
        let end_i = if p.offset % 8 != 0 {
            ((datatype_len - p.offset) / 8) as usize
        } else {
            ((datatype_len - p.offset) / 8 - 1) as usize
        };

        for k in begin_i..=end_i {
            nbit_decompress_one_byte(data, data_offset, k, begin_i, end_i, state, p, datatype_len);
        }
    }
}

fn nbit_decompress_one_nooptype(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    size: usize,
) {
    for i in 0..size {
        let val = state.cur();
        data[data_offset + i] =
            (((val as u32) & ((1u32 << state.buf_len) - 1)) << (8 - state.buf_len)) as u8;
        let remaining = 8 - state.buf_len;
        state.next_byte();
        if remaining == 0 {
            continue;
        }
        let val2 = state.cur();
        data[data_offset + i] |=
            ((val2 >> (state.buf_len - remaining)) as u32 & ((1u32 << remaining) - 1)) as u8;
        state.buf_len -= remaining;
    }
}

fn nbit_decompress_one_array(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    parms: &[u32],
    parms_index: &mut usize,
) -> Result<()> {
    let total_size = parms[*parms_index] as usize;
    *parms_index += 1;
    let base_class = parms[*parms_index];
    *parms_index += 1;

    match base_class {
        NBIT_ATOMIC => {
            let p = NbitAtomicParms {
                size: parms[*parms_index],
                order: parms[*parms_index + 1],
                precision: parms[*parms_index + 2],
                offset: parms[*parms_index + 3],
            };
            *parms_index += 4;
            let n = total_size / p.size as usize;
            for i in 0..n {
                nbit_decompress_one_atomic(data, data_offset + i * p.size as usize, state, &p);
            }
        }
        NBIT_ARRAY => {
            let base_size = parms[*parms_index] as usize;
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n {
                nbit_decompress_one_array(
                    data,
                    data_offset + i * base_size,
                    state,
                    parms,
                    parms_index,
                )?;
                *parms_index = begin_index;
            }
        }
        NBIT_COMPOUND => {
            let base_size = parms[*parms_index] as usize;
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n {
                nbit_decompress_one_compound(
                    data,
                    data_offset + i * base_size,
                    state,
                    parms,
                    parms_index,
                )?;
                *parms_index = begin_index;
            }
        }
        NBIT_NOOPTYPE => {
            *parms_index += 1; // skip size
            nbit_decompress_one_nooptype(data, data_offset, state, total_size);
        }
        _ => {
            return Err(Error::DecompressionError {
                msg: format!("nbit: unknown class {}", base_class),
            });
        }
    }
    Ok(())
}

fn nbit_decompress_one_compound(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    parms: &[u32],
    parms_index: &mut usize,
) -> Result<()> {
    let _size = parms[*parms_index] as usize;
    *parms_index += 1;
    let nmembers = parms[*parms_index] as usize;
    *parms_index += 1;

    for _ in 0..nmembers {
        let member_offset = parms[*parms_index] as usize;
        *parms_index += 1;
        let member_class = parms[*parms_index];
        *parms_index += 1;

        match member_class {
            NBIT_ATOMIC => {
                let p = NbitAtomicParms {
                    size: parms[*parms_index],
                    order: parms[*parms_index + 1],
                    precision: parms[*parms_index + 2],
                    offset: parms[*parms_index + 3],
                };
                *parms_index += 4;
                nbit_decompress_one_atomic(data, data_offset + member_offset, state, &p);
            }
            NBIT_ARRAY => {
                nbit_decompress_one_array(
                    data,
                    data_offset + member_offset,
                    state,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_COMPOUND => {
                nbit_decompress_one_compound(
                    data,
                    data_offset + member_offset,
                    state,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_NOOPTYPE => {
                let member_size = parms[*parms_index] as usize;
                *parms_index += 1;
                nbit_decompress_one_nooptype(data, data_offset + member_offset, state, member_size);
            }
            _ => {
                return Err(Error::DecompressionError {
                    msg: format!("nbit: unknown member class {}", member_class),
                });
            }
        }
    }
    Ok(())
}

fn decompress_nbit(data: &[u8], cd_values: &[u32]) -> Result<Vec<u8>> {
    if cd_values.len() < 5 {
        return Err(Error::DecompressionError {
            msg: "nbit: cd_values too short".into(),
        });
    }

    // cd_values[1]: need_not_compress flag
    if cd_values[1] != 0 {
        return Ok(data.to_vec());
    }

    let d_nelmts = cd_values[2] as usize;
    let elem_size = cd_values[4] as usize;
    let size_out = d_nelmts * elem_size;

    let mut output = vec![0u8; size_out];
    let mut state = NbitState::new(data);

    // parms start at cd_values[3]
    let parms = &cd_values[3..];

    match parms[0] {
        NBIT_ATOMIC => {
            let p = NbitAtomicParms {
                size: parms[1],
                order: parms[2],
                precision: parms[3],
                offset: parms[4],
            };
            if p.precision > p.size * 8 || (p.precision + p.offset) > p.size * 8 {
                return Err(Error::DecompressionError {
                    msg: "nbit: invalid precision/offset".into(),
                });
            }
            for i in 0..d_nelmts {
                nbit_decompress_one_atomic(&mut output, i * p.size as usize, &mut state, &p);
            }
        }
        NBIT_ARRAY => {
            let size = parms[1] as usize;
            let mut parms_index: usize = 1; // relative to parms (which is cd_values[3..])
            for i in 0..d_nelmts {
                nbit_decompress_one_array(
                    &mut output,
                    i * size,
                    &mut state,
                    parms,
                    &mut parms_index,
                )?;
                parms_index = 1;
            }
        }
        NBIT_COMPOUND => {
            let size = parms[1] as usize;
            let mut parms_index: usize = 1;
            for i in 0..d_nelmts {
                nbit_decompress_one_compound(
                    &mut output,
                    i * size,
                    &mut state,
                    parms,
                    &mut parms_index,
                )?;
                parms_index = 1;
            }
        }
        _ => {
            return Err(Error::DecompressionError {
                msg: format!("nbit: unsupported top-level class {}", parms[0]),
            });
        }
    }

    Ok(output)
}

// ========== Scale-offset filter ==========

const SO_CLS_INTEGER: u32 = 0;
const SO_CLS_FLOAT: u32 = 1;
const SO_PARM_SCALETYPE: usize = 0;
const SO_PARM_SCALEFACTOR: usize = 1;
const SO_PARM_NELMTS: usize = 2;
const SO_PARM_CLASS: usize = 3;
const SO_PARM_SIZE: usize = 4;
const SO_PARM_SIGN: usize = 5;
const SO_PARM_ORDER: usize = 6;
const SO_PARM_FILAVAIL: usize = 7;

const SO_FLOAT_DSCALE: u32 = 0;
const SO_FILL_DEFINED: u32 = 1;

fn so_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: usize,
    begin_i: usize,
    buffer: &[u8],
    j: &mut usize,
    bits_to_fill: &mut usize,
    minbits: u32,
    dtype_len: u32,
) {
    if *j >= buffer.len() {
        return;
    }

    let val = buffer[*j];
    let bits_to_copy = if k == begin_i {
        8 - ((dtype_len - minbits) % 8) as usize
    } else {
        8
    };

    if *bits_to_fill > bits_to_copy {
        data[data_offset + k] =
            ((val >> (*bits_to_fill - bits_to_copy)) as u32 & ((1u32 << bits_to_copy) - 1)) as u8;
        *bits_to_fill -= bits_to_copy;
    } else {
        data[data_offset + k] = (((val as u32) & ((1u32 << *bits_to_fill) - 1))
            << (bits_to_copy - *bits_to_fill)) as u8;
        let remaining = bits_to_copy - *bits_to_fill;
        *j += 1;
        *bits_to_fill = 8;
        if remaining == 0 {
            return;
        }
        if *j >= buffer.len() {
            return;
        }
        let val2 = buffer[*j];
        data[data_offset + k] |=
            ((val2 >> (*bits_to_fill - remaining)) as u32 & ((1u32 << remaining) - 1)) as u8;
        *bits_to_fill -= remaining;
    }
}

fn so_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    j: &mut usize,
    bits_to_fill: &mut usize,
    size: u32,
    minbits: u32,
    mem_order_le: bool,
) {
    let dtype_len = size * 8;

    if mem_order_le {
        let begin_i = (size as usize) - 1 - ((dtype_len - minbits) / 8) as usize;
        let mut k = begin_i as isize;
        while k >= 0 {
            so_decompress_one_byte(
                data,
                data_offset,
                k as usize,
                begin_i,
                buffer,
                j,
                bits_to_fill,
                minbits,
                dtype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = ((dtype_len - minbits) / 8) as usize;
        for k in begin_i..size as usize {
            so_decompress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                buffer,
                j,
                bits_to_fill,
                minbits,
                dtype_len,
            );
        }
    }
}

fn decompress_scaleoffset(data: &[u8], cd_values: &[u32]) -> Result<Vec<u8>> {
    if cd_values.len() < 8 {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: cd_values too short".into(),
        });
    }

    let scale_type = cd_values[SO_PARM_SCALETYPE];
    let scale_factor = cd_values[SO_PARM_SCALEFACTOR] as i32;
    let d_nelmts = cd_values[SO_PARM_NELMTS] as usize;
    let dtype_class = cd_values[SO_PARM_CLASS];
    let dtype_size = cd_values[SO_PARM_SIZE] as usize;
    let dtype_sign = cd_values[SO_PARM_SIGN];
    let dtype_order = cd_values[SO_PARM_ORDER];
    let filavail = cd_values[SO_PARM_FILAVAIL];

    if data.len() < 5 {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: buffer too short".into(),
        });
    }

    // Read minbits (4 bytes LE)
    let minbits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // Read minval_size and minval
    let stored_minval_size = data[4] as usize;
    let minval_size = stored_minval_size.min(8);
    let mut minval: u64 = 0;
    if data.len() < 5 + minval_size {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: buffer too short for minval".into(),
        });
    }
    for i in 0..minval_size {
        minval |= (data[5 + i] as u64) << (i * 8);
    }

    let buf_offset: usize = 21;
    let size_out = d_nelmts * dtype_size;

    // Special case: minbits == full precision — raw copy
    if minbits == (dtype_size as u32) * 8 {
        if data.len() < buf_offset + size_out {
            return Err(Error::DecompressionError {
                msg: "scaleoffset: buffer too short for full copy".into(),
            });
        }
        let mut output = data[buf_offset..buf_offset + size_out].to_vec();

        // Convert byte order if needed (data is stored in dataset order, we want native LE)
        if dtype_order != 0 {
            // dtype_order != LE means data is BE, need to swap to native LE
            for chunk in output.chunks_exact_mut(dtype_size) {
                chunk.reverse();
            }
        }
        return Ok(output);
    }

    let mut output = vec![0u8; size_out];

    // Decompress packed data
    if minbits != 0 {
        let packed = if buf_offset < data.len() {
            &data[buf_offset..]
        } else {
            &[] as &[u8]
        };
        let mut j: usize = 0;
        let mut bits_to_fill: usize = 8;

        // Scale-offset always uses LE memory order for the bit unpacking
        // (it stores data in native order during compression)
        let mem_order_le = true;

        for i in 0..d_nelmts {
            so_decompress_one_atomic(
                &mut output,
                i * dtype_size,
                packed,
                &mut j,
                &mut bits_to_fill,
                dtype_size as u32,
                minbits,
                mem_order_le,
            );
        }
    }

    // Post-decompress: add minval back
    if dtype_class == SO_CLS_INTEGER {
        so_postdecompress_int(
            &mut output,
            d_nelmts,
            dtype_size,
            dtype_sign,
            filavail,
            cd_values,
            minbits,
            minval,
        );
    } else if dtype_class == SO_CLS_FLOAT && scale_type == SO_FLOAT_DSCALE {
        so_postdecompress_float(
            &mut output,
            d_nelmts,
            dtype_size,
            filavail,
            cd_values,
            minbits,
            minval,
            scale_factor as f64,
        );
    }

    // Convert byte order if dataset is BE and we're on LE
    if dtype_order != 0 {
        for chunk in output.chunks_exact_mut(dtype_size) {
            chunk.reverse();
        }
    }

    Ok(output)
}

fn so_postdecompress_int(
    data: &mut [u8],
    d_nelmts: usize,
    dtype_size: usize,
    dtype_sign: u32,
    filavail: u32,
    cd_values: &[u32],
    minbits: u32,
    minval: u64,
) {
    let is_signed = dtype_sign != 0;

    for i in 0..d_nelmts {
        let off = i * dtype_size;

        // Read element as u64 LE
        let mut elem: u64 = 0;
        for b in 0..dtype_size {
            elem |= (data[off + b] as u64) << (b * 8);
        }

        // Check for fill value sentinel
        if filavail == SO_FILL_DEFINED && minbits < 64 {
            let sentinel = (1u64 << minbits) - 1;
            if elem == sentinel {
                // Reconstruct fill value from cd_values[8..]
                let mut filval: u64 = 0;
                let filval_start = SO_PARM_FILAVAIL + 1; // cd_values[8]
                for b in 0..dtype_size.min(cd_values.len().saturating_sub(filval_start) * 4) {
                    let cd_idx = filval_start + b / 4;
                    if cd_idx >= cd_values.len() {
                        break;
                    }
                    let byte_in_cd = b % 4;
                    filval |= (((cd_values[cd_idx] >> (byte_in_cd * 8)) & 0xFF) as u64) << (b * 8);
                }
                for b in 0..dtype_size {
                    data[off + b] = (filval >> (b * 8)) as u8;
                }
                continue;
            }
        }

        // Add minval
        let result = if is_signed {
            let sminval = minval as i64;
            let selem = elem as i64;
            (selem.wrapping_add(sminval)) as u64
        } else {
            elem.wrapping_add(minval)
        };

        for b in 0..dtype_size {
            data[off + b] = (result >> (b * 8)) as u8;
        }
    }
}

fn so_postdecompress_float(
    data: &mut [u8],
    d_nelmts: usize,
    dtype_size: usize,
    filavail: u32,
    cd_values: &[u32],
    minbits: u32,
    minval: u64,
    d_val: f64,
) {
    // minval is the bit pattern of the float minimum, stored as u64 LE
    // For float: reinterpret as f32; for double: reinterpret as f64
    let min_float: f64 = if dtype_size == 4 {
        f32::from_le_bytes((minval as u32).to_le_bytes()) as f64
    } else {
        f64::from_le_bytes(minval.to_le_bytes())
    };

    let pow10 = 10.0f64.powf(d_val);

    for i in 0..d_nelmts {
        let off = i * dtype_size;

        // Read the integer value (unpacked differences are stored as integers)
        let mut ival: u64 = 0;
        for b in 0..dtype_size {
            ival |= (data[off + b] as u64) << (b * 8);
        }

        // Check fill value sentinel
        if filavail == SO_FILL_DEFINED && minbits < 64 {
            let sentinel = (1u64 << minbits) - 1;
            if ival == sentinel {
                let mut filval: u64 = 0;
                let filval_start = SO_PARM_FILAVAIL + 1;
                for b in 0..dtype_size.min(cd_values.len().saturating_sub(filval_start) * 4) {
                    let cd_idx = filval_start + b / 4;
                    if cd_idx >= cd_values.len() {
                        break;
                    }
                    let byte_in_cd = b % 4;
                    filval |= (((cd_values[cd_idx] >> (byte_in_cd * 8)) & 0xFF) as u64) << (b * 8);
                }
                for b in 0..dtype_size {
                    data[off + b] = (filval >> (b * 8)) as u8;
                }
                continue;
            }
        }

        // D-scale postprocess: float_val = (integer_val as float) / 10^D + min
        let result: f64 = if dtype_size == 4 {
            let int_as_signed = ival as i32;
            (int_as_signed as f64) / pow10 + min_float
        } else {
            let int_as_signed = ival as i64;
            (int_as_signed as f64) / pow10 + min_float
        };

        // Write back
        if dtype_size == 4 {
            let bytes = (result as f32).to_le_bytes();
            data[off..off + 4].copy_from_slice(&bytes);
        } else {
            let bytes = result.to_le_bytes();
            data[off..off + 8].copy_from_slice(&bytes);
        }
    }
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

    #[test]
    fn nbit_need_not_compress() {
        // cd_values[1] = 1 means no compression needed
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let cd = vec![8, 1, 2, 1, 2, 0, 16, 0]; // need_not_compress=1
        let result = decompress_nbit(&data, &cd).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn nbit_atomic_u8_4bit() {
        // 4 uint8 values with 4-bit precision, offset 0, LE
        // Values: 0x0A, 0x0B, 0x0C, 0x0D (only low 4 bits significant)
        // Packed: 0xAB, 0xCD (4 bits each, MSB-first in buffer)
        let packed = vec![0xAB, 0xCD];
        let cd = vec![8, 0, 4, 1, 1, 0, 4, 0]; // ATOMIC, size=1, order=LE, prec=4, off=0
        let result = decompress_nbit(&packed, &cd).unwrap();
        assert_eq!(result, vec![0x0A, 0x0B, 0x0C, 0x0D]);
    }

    #[test]
    fn scaleoffset_int_simple() {
        // Manually construct a scaleoffset buffer:
        // minbits=3 (values 0..7 fit in 3 bits), minval=1000 as u64
        // d_nelmts=4, dtype_size=4, dtype_class=INTEGER(0), sign=1(signed), order=0(LE)
        // cd_values: [scaletype=2, scalefactor=0, nelmts=4, class=0, size=4, sign=1, order=0, filavail=0]
        let cd = vec![2u32, 0, 4, 0, 4, 1, 0, 0];

        // Buffer: minbits(4) + sizeof_ull(1) + minval(8) + padding(8) + packed data
        let mut buf = vec![0u8; 21];
        // minbits = 3
        buf[0] = 3;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 0;
        // sizeof_ull = 8
        buf[4] = 8;
        // minval = 1000 = 0x3E8
        buf[5] = 0xE8;
        buf[6] = 0x03;
        buf[7] = 0;
        buf[8] = 0;
        buf[9] = 0;
        buf[10] = 0;
        buf[11] = 0;
        buf[12] = 0;
        // Packed data: values 0,1,2,3 in 3-bit fields, MSB-first
        // 000 001 010 011 = 0b00000101_0011xxxx = 0x05 0x30
        buf.push(0b00000101);
        buf.push(0b00110000);

        let result = decompress_scaleoffset(&buf, &cd).unwrap();
        assert_eq!(result.len(), 16); // 4 * 4 bytes
        let values: Vec<i32> = result
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values, vec![1000, 1001, 1002, 1003]);
    }
}
