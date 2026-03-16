use crate::checksum;
use crate::datatype::ByteOrder;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::StringPadding;
use crate::error::Error;
use crate::error::Result;
use crate::filters;
use crate::superblock::HDF5_SIGNATURE;
use crate::superblock::UNDEF_ADDR;
use crate::writer::file_writer::WriteOptions;
use crate::writer::types::AttrData;
use crate::writer::types::ChunkFilter;

pub(crate) const SIZE_OF_OFFSETS: u8 = 8;
pub(crate) const SIZE_OF_LENGTHS: u8 = 8;
pub(crate) const SUPERBLOCK_SIZE: usize = 48;

/// A single object header message: (type_id, body_bytes, message_flags).
pub(crate) type OhdrMsg = (u8, Vec<u8>, u8);

/// Total object header size given the sum of all message (header+body) bytes.
pub(crate) fn ohdr_overhead(total_msg_bytes: usize, opts: &WriteOptions) -> usize {
    let (prefix_size, _) = chunk_size_encoding(total_msg_bytes, opts);
    prefix_size + total_msg_bytes + 4
}

fn chunk_size_encoding(total_msg_bytes: usize, opts: &WriteOptions) -> (usize, u8) {
    let ts_extra = if opts.timestamps.is_some() { 16 } else { 0 };
    let base_flags: u8 = if opts.timestamps.is_some() { 0x20 } else { 0 };
    if total_msg_bytes <= 0xFF {
        (7 + ts_extra, base_flags)
    } else if total_msg_bytes <= 0xFFFF {
        (8 + ts_extra, base_flags | 0x01)
    } else {
        (10 + ts_extra, base_flags | 0x02)
    }
}

/// Encode an object header from messages.
///
/// If `target_chunk_size` is Some, pad with a NIL message to reach that chunk size.
pub(crate) fn encode_object_header(
    messages: &[OhdrMsg],
    opts: &WriteOptions,
    target_chunk_size: Option<usize>,
) -> Result<Vec<u8>> {
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();

    let total_msg_bytes = if let Some(target) = target_chunk_size {
        assert!(
            target >= real_msg_bytes,
            "target_chunk_size ({target}) < real messages ({real_msg_bytes})"
        );
        target
    } else {
        real_msg_bytes
    };

    let (prefix_size, flags) = chunk_size_encoding(total_msg_bytes, opts);

    let mut buf = Vec::with_capacity(prefix_size + total_msg_bytes + 4);

    buf.extend_from_slice(b"OHDR");
    buf.push(2);
    buf.push(flags);

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    match flags & 0x03 {
        0x00 => buf.push(total_msg_bytes as u8),
        0x01 => buf.extend_from_slice(&(total_msg_bytes as u16).to_le_bytes()),
        0x02 => buf.extend_from_slice(&(total_msg_bytes as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    for (type_id, body, msg_flags) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(*msg_flags);
        buf.extend_from_slice(body);
    }

    // Pad with NIL message or gap if needed
    let nil_bytes = total_msg_bytes - real_msg_bytes;
    if nil_bytes >= 4 {
        // Full NIL message: header(4) + body
        let nil_body_len = nil_bytes - 4;
        buf.push(0x00); // NIL type
        buf.extend_from_slice(&(nil_body_len as u16).to_le_bytes());
        buf.push(0x00); // NIL flags
        buf.resize(buf.len() + nil_body_len, 0);
    } else if nil_bytes > 0 {
        // Small gap (1-3 bytes): just zero-fill per v2 OHDR spec
        buf.resize(buf.len() + nil_bytes, 0);
    }

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    Ok(buf)
}

/// Encode an object header with optional non-default attribute phase change values.
///
/// When `phase_change` is Some, flag 0x10 is set and the (max_compact, min_dense)
/// values are inserted into the header prefix before the chunk size field.
pub(crate) fn encode_object_header_with_phase_change(
    messages: &[OhdrMsg],
    opts: &WriteOptions,
    target_chunk_size: Option<usize>,
    phase_change: Option<(u16, u16)>,
) -> Result<Vec<u8>> {
    if phase_change.is_none() {
        return encode_object_header(messages, opts, target_chunk_size);
    }
    let (max_compact, min_dense) = phase_change.unwrap();

    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
    let total_msg_bytes = if let Some(target) = target_chunk_size {
        assert!(target >= real_msg_bytes);
        target
    } else {
        real_msg_bytes
    };

    // Compute flags and prefix. Phase change adds 4 bytes to the prefix.
    let ts_extra = if opts.timestamps.is_some() { 16 } else { 0 };
    let base_flags: u8 = if opts.timestamps.is_some() { 0x20 } else { 0 };
    let flags = base_flags | 0x10 | // phase change flag
        if total_msg_bytes <= 0xFF { 0x00 }
        else if total_msg_bytes <= 0xFFFF { 0x01 }
        else { 0x02 };

    let cs_size = match flags & 0x03 {
        0x00 => 1,
        0x01 => 2,
        _ => 4,
    };
    let prefix_size = 6 + ts_extra + 4 + cs_size; // 4 extra for phase change

    let mut buf = Vec::with_capacity(prefix_size + total_msg_bytes + 4);

    buf.extend_from_slice(b"OHDR");
    buf.push(2);
    buf.push(flags);

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    // Phase change values: max_compact(2) + min_dense(2)
    buf.extend_from_slice(&max_compact.to_le_bytes());
    buf.extend_from_slice(&min_dense.to_le_bytes());

    match flags & 0x03 {
        0x00 => buf.push(total_msg_bytes as u8),
        0x01 => buf.extend_from_slice(&(total_msg_bytes as u16).to_le_bytes()),
        0x02 => buf.extend_from_slice(&(total_msg_bytes as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    for (type_id, body, msg_flags) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(*msg_flags);
        buf.extend_from_slice(body);
    }

    // Pad with NIL message or gap if needed.
    let nil_bytes = total_msg_bytes - real_msg_bytes;
    if nil_bytes >= 4 {
        let nil_body_len = nil_bytes - 4;
        buf.push(0x00);
        buf.extend_from_slice(&(nil_body_len as u16).to_le_bytes());
        buf.push(0x00);
        buf.resize(buf.len() + nil_body_len, 0);
    } else if nil_bytes > 0 {
        buf.resize(buf.len() + nil_bytes, 0);
    }

    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    Ok(buf)
}

pub(crate) fn encode_superblock(root_group_addr: u64, eof: u64) -> Vec<u8> {
    encode_superblock_versioned(root_group_addr, eof, 2)
}

pub(crate) fn encode_superblock_versioned(root_group_addr: u64, eof: u64, version: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SUPERBLOCK_SIZE);
    buf.extend_from_slice(&HDF5_SIGNATURE);
    buf.push(version);
    buf.push(SIZE_OF_OFFSETS);
    buf.push(SIZE_OF_LENGTHS);
    buf.push(0);
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf.extend_from_slice(&eof.to_le_bytes());
    buf.extend_from_slice(&root_group_addr.to_le_bytes());
    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    debug_assert_eq!(buf.len(), SUPERBLOCK_SIZE);
    buf
}

pub(crate) fn encode_datatype(dt: &Datatype) -> Result<Vec<u8>> {
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let mut buf = Vec::with_capacity(12);
            buf.push(0x10);
            let mut f0 = 0u8;
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            if *signed {
                f0 |= 0x08;
            }
            buf.push(f0);
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            Ok(buf)
        }
        Datatype::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } => {
            let mut buf = Vec::with_capacity(20);
            buf.push(0x11);
            let mut f0 = 0x20u8;
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            buf.push(f0);
            let sign_bit_pos = exponent_location + exponent_size;
            buf.push(sign_bit_pos);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            buf.push(*exponent_location);
            buf.push(*exponent_size);
            buf.push(*mantissa_location);
            buf.push(*mantissa_size);
            buf.extend_from_slice(&exponent_bias.to_le_bytes());
            Ok(buf)
        }
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            let mut buf = Vec::with_capacity(8);
            buf.push(0x13);
            let pad = match padding {
                StringPadding::NullTerminate => 0u8,
                StringPadding::NullPad => 1,
                StringPadding::SpacePad => 2,
            };
            let cs = match char_set {
                CharacterSet::Ascii => 0u8,
                CharacterSet::Utf8 => 1,
            };
            buf.push(pad | (cs << 4));
            buf.push(0);
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            Ok(buf)
        }
        Datatype::Compound { size, members } => {
            let nmembers = members.len() as u16;
            let class_version_byte = (6u8) | (3u8 << 4);
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            let off_size = limit_enc_size(*size);
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
                match off_size {
                    1 => buf.push(m.byte_offset as u8),
                    2 => buf.extend_from_slice(&(m.byte_offset as u16).to_le_bytes()),
                    3 => {
                        buf.push(m.byte_offset as u8);
                        buf.push((m.byte_offset >> 8) as u8);
                        buf.push((m.byte_offset >> 16) as u8);
                    }
                    _ => buf.extend_from_slice(&m.byte_offset.to_le_bytes()),
                }
                buf.extend_from_slice(&encode_datatype(&m.datatype)?);
            }
            Ok(buf)
        }
        Datatype::Enum { base, members } => {
            let nmembers = members.len() as u16;
            let class_version_byte = (8u8) | (3u8 << 4);
            let base_size = base.element_size();
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            buf.extend_from_slice(&base_size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(base)?);
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
            }
            for m in members {
                buf.extend_from_slice(&m.value);
            }
            Ok(buf)
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            let class_version_byte = (10u8) | (3u8 << 4);
            let ndims = dimensions.len() as u8;
            let total_elements: u32 = dimensions.iter().product();
            let elem_size = total_elements * element_type.element_size();
            let mut buf = vec![class_version_byte, 0, 0, 0];
            buf.extend_from_slice(&elem_size.to_le_bytes());
            buf.push(ndims);
            for d in dimensions {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        Datatype::Complex { size, base } => {
            let class_version_byte = (11u8) | (5u8 << 4);
            let mut buf = vec![class_version_byte, 0x01, 0, 0];
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(base)?);
            Ok(buf)
        }
        Datatype::VarLen {
            element_type,
            is_string,
            padding,
            char_set,
        } => {
            let class_version_byte = (9u8) | (1u8 << 4);
            let mut class_bits_lo = 0u8;
            if *is_string {
                class_bits_lo |= 0x01;
                if let Some(p) = padding {
                    let pad_val = match p {
                        StringPadding::NullTerminate => 0u8,
                        StringPadding::NullPad => 1,
                        StringPadding::SpacePad => 2,
                    };
                    class_bits_lo |= pad_val << 4;
                }
            }
            let mut class_bits_hi = 0u8;
            if *is_string {
                if let Some(cs) = char_set {
                    class_bits_hi = match cs {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    };
                }
            }
            let vlen_element_size: u32 = 4 + SIZE_OF_OFFSETS as u32 + 4;
            let mut buf = vec![class_version_byte, class_bits_lo, class_bits_hi, 0];
            buf.extend_from_slice(&vlen_element_size.to_le_bytes());
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        _ => Err(Error::Other {
            msg: format!(
                "encoding not yet supported for datatype: {:?}",
                std::mem::discriminant(dt)
            ),
        }),
    }
}

fn limit_enc_size(size: u32) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFF {
        3
    } else {
        4
    }
}

pub(crate) fn encode_dataspace(shape: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    if shape.is_empty() {
        return vec![2, 0, 0, 0];
    }
    let has_max = max_dims.is_some();
    let mut buf = Vec::with_capacity(4 + shape.len() * 8 * if has_max { 2 } else { 1 });
    buf.push(2);
    buf.push(shape.len() as u8);
    buf.push(if has_max { 0x01 } else { 0x00 });
    buf.push(1);
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    if let Some(md) = max_dims {
        for &dim in md {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
    }
    buf
}

pub(crate) fn encode_link_info() -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    buf
}

pub(crate) fn encode_group_info() -> Vec<u8> {
    vec![0, 0]
}

pub(crate) fn encode_link(name: &str, target_addr: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    let (name_enc_bits, name_enc_size) = if name_len <= 0xFF {
        (0u8, 1usize)
    } else if name_len <= 0xFFFF {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    let mut buf = Vec::with_capacity(2 + name_enc_size + name_len + 8);
    buf.push(1);
    buf.push(name_enc_bits);

    match name_enc_size {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&target_addr.to_le_bytes());

    buf
}

/// Encode a link message body with creation order for storage in a fractal heap.
pub(crate) fn encode_link_body_crt_order(
    name: &str,
    target_addr: u64,
    creation_order: u64,
) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();
    let (name_enc_bits, name_enc_size) = if name_len <= 0xFF {
        (0u8, 1usize)
    } else if name_len <= 0xFFFF {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };
    // flags: bit 2 = creation order present, bits 4-5 = name length size encoding
    let flags = 0x04 | (name_enc_bits << 4);
    let mut buf = Vec::with_capacity(2 + 8 + name_enc_size + name_len + 8);
    buf.push(1); // version
    buf.push(flags);
    buf.extend_from_slice(&creation_order.to_le_bytes());
    match name_enc_size {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&target_addr.to_le_bytes());
    buf
}

/// Encode LinkInfo message with dense addresses (no creation order).
pub(crate) fn encode_link_info_dense(frhp_addr: u64, bthd_name_addr: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0); // version
    buf.push(0x00); // flags: no creation order
    buf.extend_from_slice(&frhp_addr.to_le_bytes());
    buf.extend_from_slice(&bthd_name_addr.to_le_bytes());
    buf
}

/// Encode LinkInfo message with dense addresses and creation order.
pub(crate) fn encode_link_info_dense_crt(
    frhp_addr: u64,
    bthd_name_addr: u64,
    bthd_crt_addr: u64,
    max_creation_order: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(34);
    buf.push(0); // version
    buf.push(0x03); // flags: bit 0 = creation order tracked, bit 1 = creation order indexed
    buf.extend_from_slice(&max_creation_order.to_le_bytes());
    buf.extend_from_slice(&frhp_addr.to_le_bytes());
    buf.extend_from_slice(&bthd_name_addr.to_le_bytes());
    buf.extend_from_slice(&bthd_crt_addr.to_le_bytes());
    buf
}

/// Encode AttributeInfo message with dense addresses and creation order.
pub(crate) fn encode_attribute_info_dense_crt(
    frhp_addr: u64,
    bthd_name_addr: u64,
    bthd_crt_addr: u64,
    max_creation_order: u16,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);
    buf.push(0); // version
    buf.push(0x03); // flags: bit 0 = creation order tracked, bit 1 = creation order indexed
    buf.extend_from_slice(&max_creation_order.to_le_bytes());
    buf.extend_from_slice(&frhp_addr.to_le_bytes());
    buf.extend_from_slice(&bthd_name_addr.to_le_bytes());
    buf.extend_from_slice(&bthd_crt_addr.to_le_bytes());
    buf
}

/// Encode GroupInfo message with link phase change thresholds.
pub(crate) fn encode_group_info_with_link_phase(max_compact: u16, min_dense: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(6);
    buf.push(0); // version
    buf.push(0x01); // flags: bit 0 = link phase change stored
    buf.extend_from_slice(&max_compact.to_le_bytes());
    buf.extend_from_slice(&min_dense.to_le_bytes());
    buf
}

/// Encode an OHDR with creation order tracking.
///
/// Creation order adds bits 2+3 to flags and makes message headers 6 bytes
/// (type(1) + size(2) + flags(1) + creation_order(2)).
pub(crate) fn encode_object_header_creation_order(
    messages: &[OhdrMsg],
    opts: &WriteOptions,
    target_chunk_size: Option<usize>,
    attr_phase_change: Option<(u16, u16)>,
) -> Result<Vec<u8>> {
    // With creation order, each message header is 6 bytes instead of 4.
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 6 + b.len()).sum();
    let total_msg_bytes = if let Some(target) = target_chunk_size {
        assert!(
            target >= real_msg_bytes,
            "target_chunk_size ({target}) < real messages ({real_msg_bytes})"
        );
        target
    } else {
        real_msg_bytes
    };

    let ts_extra = if opts.timestamps.is_some() { 16 } else { 0 };
    let base_flags: u8 = if opts.timestamps.is_some() { 0x20 } else { 0 };
    // Add bits 2+3 for creation order tracked+indexed
    let mut flags = base_flags | 0x04 | 0x08;
    // Add bit 4 for attr phase change if present
    let phase_extra = if attr_phase_change.is_some() {
        flags |= 0x10;
        4
    } else {
        0
    };
    // Chunk size encoding
    flags |= if total_msg_bytes <= 0xFF {
        0x00
    } else if total_msg_bytes <= 0xFFFF {
        0x01
    } else {
        0x02
    };

    let cs_size = match flags & 0x03 {
        0x00 => 1,
        0x01 => 2,
        _ => 4,
    };
    let prefix_size = 6 + ts_extra + phase_extra + cs_size;

    let mut buf = Vec::with_capacity(prefix_size + total_msg_bytes + 4);
    buf.extend_from_slice(b"OHDR");
    buf.push(2);
    buf.push(flags);

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    if let Some((mc, md)) = attr_phase_change {
        buf.extend_from_slice(&mc.to_le_bytes());
        buf.extend_from_slice(&md.to_le_bytes());
    }

    match flags & 0x03 {
        0x00 => buf.push(total_msg_bytes as u8),
        0x01 => buf.extend_from_slice(&(total_msg_bytes as u16).to_le_bytes()),
        0x02 => buf.extend_from_slice(&(total_msg_bytes as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    // Write messages with 6-byte headers (includes 2-byte creation_order=0).
    for (type_id, body, msg_flags) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(*msg_flags);
        buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order = 0
        buf.extend_from_slice(body);
    }

    // Pad with NIL message or gap.
    let nil_bytes = total_msg_bytes - real_msg_bytes;
    if nil_bytes >= 6 {
        let nil_body_len = nil_bytes - 6;
        buf.push(0x00); // type NIL
        buf.extend_from_slice(&(nil_body_len as u16).to_le_bytes());
        buf.push(0x00); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order
        buf.resize(buf.len() + nil_body_len, 0);
    } else if nil_bytes > 0 {
        buf.resize(buf.len() + nil_bytes, 0); // gap
    }

    let cksum = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    Ok(buf)
}

/// Encode an OCHK (object header continuation chunk).
///
/// Messages use 6-byte headers when creation order is tracked.
pub(crate) fn encode_ochk_creation_order(messages: &[OhdrMsg]) -> Vec<u8> {
    let msg_bytes: usize = messages.iter().map(|(_, b, _)| 6 + b.len()).sum();
    let total = 4 + msg_bytes + 4; // sig + messages + cksum
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(b"OCHK");
    for (type_id, body, msg_flags) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(*msg_flags);
        buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order = 0
        buf.extend_from_slice(body);
    }
    let cksum = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    buf
}

/// Encode Fill Value message.
///
/// When `compat` is true, use late space allocation (matching the C library default for contiguous).
pub(crate) fn encode_fill_value_msg(fill_data: &Option<Vec<u8>>, compat: bool) -> Vec<u8> {
    use crate::writer::types::SpaceAllocTime;
    let alloc_time = if compat {
        SpaceAllocTime::Late
    } else {
        SpaceAllocTime::Early
    };
    encode_fill_value_msg_alloc(fill_data, alloc_time)
}

/// Encode Fill Value message with explicit space allocation time.
pub(crate) fn encode_fill_value_msg_alloc(
    fill_data: &Option<Vec<u8>>,
    alloc_time: crate::writer::types::SpaceAllocTime,
) -> Vec<u8> {
    let alloc_time = alloc_time as u8;
    // flags bits 0-1: space alloc time, bits 2-3: fill write time (2=if-set),
    // bit 5: fill value size+data present
    let base_flags = 0x08 | (alloc_time & 0x03);
    match fill_data {
        Some(data) => {
            let flags = base_flags | 0x20; // bit 5: fill value present
            let mut buf = Vec::with_capacity(6 + data.len());
            buf.push(3);
            buf.push(flags);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
            buf
        }
        None => {
            vec![3, base_flags]
        }
    }
}

/// Encode Attribute Info message (type 0x15) with UNDEF addresses.
pub(crate) fn encode_attribute_info() -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0); // version
    buf.push(0); // flags
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // fractal heap addr
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // btree name addr
    buf
}

/// Encode Attribute Info message (type 0x15) with real fractal heap and B-tree addresses.
pub(crate) fn encode_attribute_info_addrs(fheap_addr: u64, btree_name_addr: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0); // version
    buf.push(0); // flags
    buf.extend_from_slice(&fheap_addr.to_le_bytes());
    buf.extend_from_slice(&btree_name_addr.to_le_bytes());
    buf
}

pub(crate) fn encode_contiguous_layout(data_addr: u64, data_size: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(3);
    buf.push(1);
    buf.extend_from_slice(&data_addr.to_le_bytes());
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf
}

pub(crate) fn encode_compact_layout(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + data.len());
    buf.push(3);
    buf.push(0);
    buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
    buf.extend_from_slice(data);
    buf
}

/// Encode a layout v3 chunked message (B-tree v1 chunk indexing).
pub(crate) fn encode_chunked_layout_v3(chunk_dims_with_elem: &[u64], btree_addr: u64) -> Vec<u8> {
    let dimensionality = chunk_dims_with_elem.len() as u8;
    let mut buf = vec![3, 2, dimensionality];
    buf.extend_from_slice(&btree_addr.to_le_bytes());
    for &dim in chunk_dims_with_elem {
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    buf
}

pub(crate) fn encode_chunked_layout(
    chunk_dims: &[u64],
    index_type_id: u8,
    chunk_index_addr: u64,
    single_filtered_size: Option<u64>,
) -> Vec<u8> {
    let ndims = chunk_dims.len() as u8;

    let max_dim = chunk_dims.iter().copied().max().unwrap_or(1);
    let enc_bytes = if max_dim <= 0xFF {
        1u8
    } else if max_dim <= 0xFFFF {
        2
    } else if max_dim <= 0xFFFFFFFF {
        4
    } else {
        8
    };

    let mut flags = 0u8;
    if single_filtered_size.is_some() {
        flags |= 0x02;
    }

    let mut buf = vec![4, 2, flags, ndims, enc_bytes];

    for &dim in chunk_dims {
        match enc_bytes {
            1 => buf.push(dim as u8),
            2 => buf.extend_from_slice(&(dim as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(dim as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&dim.to_le_bytes()),
            _ => unreachable!(),
        }
    }

    buf.push(index_type_id);

    match index_type_id {
        1 => {
            if let Some(filtered_size) = single_filtered_size {
                buf.extend_from_slice(&filtered_size.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        2 => {}
        3 => {
            // FixedArray: max_dblk_page_nelmts_bits (C library default: 10)
            buf.push(10);
        }
        4 => {
            // ExtArray creation parameters (C library defaults):
            buf.push(32); // max_nelmts_bits
            buf.push(4); // idx_blk_elmts
            buf.push(4); // sup_blk_min_data_ptrs
            buf.push(16); // data_blk_min_elmts
            buf.push(10); // max_dblk_page_nelmts_bits
        }
        5 => {
            // BTreeV2 creation parameters:
            buf.extend_from_slice(&2048u32.to_le_bytes()); // node_size
            buf.push(100); // split_percent
            buf.push(40); // merge_percent
        }
        _ => {}
    }

    buf.extend_from_slice(&chunk_index_addr.to_le_bytes());

    buf
}

pub(crate) fn encode_filter_pipeline(chunk_filters: &[ChunkFilter], element_size: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(2);
    buf.push(chunk_filters.len() as u8);

    for filter in chunk_filters {
        match filter {
            ChunkFilter::Deflate(level) => {
                buf.extend_from_slice(&filters::FILTER_DEFLATE.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes()); // optional flag (C library default)
                buf.extend_from_slice(&1u16.to_le_bytes());
                buf.extend_from_slice(&level.to_le_bytes());
            }
            ChunkFilter::Shuffle => {
                buf.extend_from_slice(&filters::FILTER_SHUFFLE.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes()); // optional flag (C library default)
                buf.extend_from_slice(&1u16.to_le_bytes());
                buf.extend_from_slice(&element_size.to_le_bytes());
            }
            ChunkFilter::Fletcher32 => {
                buf.extend_from_slice(&filters::FILTER_FLETCHER32.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes()); // mandatory (C library default)
                buf.extend_from_slice(&0u16.to_le_bytes());
            }
            ChunkFilter::ScaleOffset(params) => {
                buf.extend_from_slice(&filters::FILTER_SCALEOFFSET.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes()); // optional flag
                buf.extend_from_slice(&20u16.to_le_bytes()); // 20 cd_values
                for &v in &params.cd_values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            ChunkFilter::Nbit(params) => {
                buf.extend_from_slice(&filters::FILTER_NBIT.to_le_bytes());
                buf.extend_from_slice(&1u16.to_le_bytes()); // optional flag
                buf.extend_from_slice(&(params.cd_values.len() as u16).to_le_bytes());
                for &v in &params.cd_values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            ChunkFilter::Lzf => {
                buf.extend_from_slice(&filters::FILTER_LZF.to_le_bytes());
                // Third-party filter (id >= 256): name_length + flags + ncv + name + cd_values
                buf.extend_from_slice(&4u16.to_le_bytes()); // name_length including NUL
                buf.extend_from_slice(&1u16.to_le_bytes()); // flags: 1 = optional
                buf.extend_from_slice(&0u16.to_le_bytes()); // ncv: 0 cd_values
                buf.extend_from_slice(b"lzf\0"); // name (no padding in v2)
            }
        }
    }

    buf
}

pub(crate) fn encode_attribute(attr: &AttrData) -> Result<Vec<u8>> {
    let dt_enc = encode_datatype(&attr.datatype)?;
    let ds_enc = encode_dataspace(&attr.shape, None);
    let name_with_nul = attr.name.len() + 1;

    let mut buf = Vec::new();
    buf.push(3);
    buf.push(0);
    buf.extend_from_slice(&(name_with_nul as u16).to_le_bytes());
    buf.extend_from_slice(&(dt_enc.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(ds_enc.len() as u16).to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(attr.name.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&dt_enc);
    buf.extend_from_slice(&ds_enc);
    buf.extend_from_slice(&attr.value);

    Ok(buf)
}

/// Encode an attribute whose datatype is a shared reference to a committed type.
pub(crate) fn encode_attribute_shared(
    attr: &AttrData,
    committed_type_addr: u64,
) -> Result<Vec<u8>> {
    // Shared type reference: version(1)=2 + type(1)=2 + addr(8) = 10 bytes
    let mut dt_enc = Vec::with_capacity(10);
    dt_enc.push(2);
    dt_enc.push(2);
    dt_enc.extend_from_slice(&committed_type_addr.to_le_bytes());

    let ds_enc = encode_dataspace(&attr.shape, None);
    let name_with_nul = attr.name.len() + 1;

    let mut buf = Vec::new();
    buf.push(3);
    buf.push(0x01); // flags: bit 0 = datatype is shared
    buf.extend_from_slice(&(name_with_nul as u16).to_le_bytes());
    buf.extend_from_slice(&(dt_enc.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(ds_enc.len() as u16).to_le_bytes());
    buf.push(0); // encoding = ASCII
    buf.extend_from_slice(attr.name.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&dt_enc);
    buf.extend_from_slice(&ds_enc);
    buf.extend_from_slice(&attr.value);

    Ok(buf)
}

pub(crate) fn limit_enc_size_u64(size: u64) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFFFF {
        4
    } else {
        8
    }
}

/// Compute the group OHDR target chunk size as the C library would.
///
/// The C library estimates space for `est_num_entries` links of `est_name_len` chars,
/// plus LinkInfo and GroupInfo messages.
pub(crate) fn compat_group_chunk_size(est_num_entries: usize, est_name_len: usize) -> usize {
    let link_info_msg = 4 + 18; // header + body
    let group_info_msg = 4 + 2;
    let per_link = 4 + 1 + 1 + 1 + est_name_len + 8; // header(4) + ver(1) + flags(1) + name_len_field(1) + name + addr(8)
    link_info_msg + group_info_msg + est_num_entries * per_link
}

/// Compute the dataset OHDR target chunk size as the C library would.
///
/// Returns max(256, actual_messages), which matches the C library's minimum OHDR allocation.
pub(crate) fn compat_dataset_chunk_size(actual_msg_bytes: usize) -> usize {
    actual_msg_bytes.max(256)
}
