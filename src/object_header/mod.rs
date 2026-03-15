pub mod messages;

use messages::Message;
use messages::MessageType;

use crate::checksum;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

/// Magic bytes for object header v2: `OHDR`
pub const OHDR_MAGIC: [u8; 4] = *b"OHDR";
/// Magic bytes for object header continuation chunk: `OCHK`
pub const OCHK_MAGIC: [u8; 4] = *b"OCHK";

/// A parsed object header (version 2 only — v1 not supported since we only target superblock v2/v3).
///
/// ## On-disk layout (object header v2 prefix)
///
/// ```text
/// Byte 0-3:  Signature ("OHDR")
/// Byte 4:    Version (2)
/// Byte 5:    Flags
///            Bit 0-1: size of chunk0 field (0=1byte, 1=2byte, 2=4byte, 3=8byte)
///            Bit 2:   attribute creation order tracked
///            Bit 3:   attribute creation order indexed
///            Bit 4:   non-default attribute storage phase change values stored
///            Bit 5:   access/modification/change/birth times stored
/// Byte 6+:   [if flags bit 5] access time (4 bytes), mod time (4), change time (4), birth time (4)
///            [if flags bit 4] max compact attrs (2), min dense attrs (2)
///            Chunk 0 data size (1/2/4/8 bytes per flags bits 0-1)
///            Messages...
///            Gap (0 bytes) or padding
///            Checksum (4 bytes)
/// ```
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    pub flags: u8,
    pub access_time: Option<u32>,
    pub modification_time: Option<u32>,
    pub change_time: Option<u32>,
    pub birth_time: Option<u32>,
    pub max_compact_attributes: Option<u16>,
    pub min_dense_attributes: Option<u16>,
    pub messages: Vec<Message>,
}

impl ObjectHeader {
    /// Parse an object header v2 starting at `addr`.
    pub fn parse<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != OHDR_MAGIC {
            return Err(Error::InvalidObjectHeader {
                msg: format!("expected OHDR magic at {:#x}, got {:?}", addr, magic),
            });
        }

        let version = Le::read_u8(reader, addr + 4).map_err(Error::Io)?;
        if version != 2 {
            return Err(Error::InvalidObjectHeader {
                msg: format!("expected object header version 2, got {}", version),
            });
        }

        let flags = Le::read_u8(reader, addr + 5).map_err(Error::Io)?;
        let mut pos = addr + 6;

        // Optional timestamps
        let (access_time, modification_time, change_time, birth_time) = if (flags & 0x20) != 0 {
            let at = Le::read_u32(reader, pos).map_err(Error::Io)?;
            let mt = Le::read_u32(reader, pos + 4).map_err(Error::Io)?;
            let ct = Le::read_u32(reader, pos + 8).map_err(Error::Io)?;
            let bt = Le::read_u32(reader, pos + 12).map_err(Error::Io)?;
            pos += 16;
            (Some(at), Some(mt), Some(ct), Some(bt))
        } else {
            (None, None, None, None)
        };

        // Optional attribute phase change values
        let (max_compact_attributes, min_dense_attributes) = if (flags & 0x10) != 0 {
            let mc = Le::read_u16(reader, pos).map_err(Error::Io)?;
            let md = Le::read_u16(reader, pos + 2).map_err(Error::Io)?;
            pos += 4;
            (Some(mc), Some(md))
        } else {
            (None, None)
        };

        // Chunk 0 data size
        let chunk_size_enc = flags & 0x03;
        let chunk0_size = match chunk_size_enc {
            0 => {
                let v = Le::read_u8(reader, pos).map_err(Error::Io)?;
                pos += 1;
                v as u64
            }
            1 => {
                let v = Le::read_u16(reader, pos).map_err(Error::Io)?;
                pos += 2;
                v as u64
            }
            2 => {
                let v = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                v as u64
            }
            3 => {
                let v = Le::read_u64(reader, pos).map_err(Error::Io)?;
                pos += 8;
                v
            }
            _ => unreachable!(),
        };

        // The chunk0 data region starts at `pos` and is `chunk0_size` bytes,
        // followed by a 4-byte checksum.
        let chunk0_data_start = pos;
        let chunk0_data_end = chunk0_data_start + chunk0_size;

        // Read all bytes from header start through end of chunk0 data for checksum
        let checksum_region_len = (chunk0_data_end - addr) as usize;
        let mut checksum_region = vec![0u8; checksum_region_len];
        reader
            .read_exact_at(addr, &mut checksum_region)
            .map_err(Error::Io)?;
        let stored_checksum = Le::read_u32(reader, chunk0_data_end).map_err(Error::Io)?;
        let computed = checksum::lookup3(&checksum_region);
        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        // Parse messages from chunk0
        let mut messages = Vec::new();
        Self::parse_messages(
            reader,
            chunk0_data_start,
            chunk0_size,
            flags,
            size_of_offsets,
            size_of_lengths,
            &mut messages,
        )?;

        // Follow continuation messages
        let mut continuations: Vec<(u64, u64)> = messages
            .iter()
            .filter_map(|m| {
                if m.msg_type == MessageType::ObjectHeaderContinuation {
                    parse_continuation_payload(&m.data, size_of_offsets, size_of_lengths)
                } else {
                    None
                }
            })
            .collect();

        while let Some((cont_addr, cont_size)) = continuations.pop() {
            Self::parse_continuation_chunk(
                reader,
                cont_addr,
                cont_size,
                flags,
                size_of_offsets,
                size_of_lengths,
                &mut messages,
                &mut continuations,
            )?;
        }

        // Remove continuation messages from the final list (internal bookkeeping)
        messages.retain(|m| m.msg_type != MessageType::ObjectHeaderContinuation);

        Ok(ObjectHeader {
            flags,
            access_time,
            modification_time,
            change_time,
            birth_time,
            max_compact_attributes,
            min_dense_attributes,
            messages,
        })
    }

    /// Parse messages within a chunk of `chunk_size` bytes starting at `chunk_start`.
    fn parse_messages<R: ReadAt + ?Sized>(
        reader: &R,
        chunk_start: u64,
        chunk_size: u64,
        header_flags: u8,
        _size_of_offsets: u8,
        _size_of_lengths: u8,
        messages: &mut Vec<Message>,
    ) -> Result<()> {
        let mut pos = chunk_start;
        let chunk_end = chunk_start + chunk_size;

        while pos < chunk_end {
            // Check for gap: if remaining bytes are fewer than a message header,
            // they are gap padding (H5Ocache.c:1246 — loop reads until eom_ptr).
            let min_msg_header: u64 = if (header_flags & 0x04) != 0 { 6 } else { 4 };
            if chunk_end - pos < min_msg_header {
                break;
            }

            // Message header: type (u8), size (u16), flags (u8)
            //   [+ creation_order (u16) if tracked]
            // Type 0 is the Nil message — a valid message that acts as padding
            // with a proper header and size field (H5Ocache.c:1309).
            let msg_type_raw = Le::read_u8(reader, pos).map_err(Error::Io)?;
            let msg_size = Le::read_u16(reader, pos + 1).map_err(Error::Io)?;
            let msg_flags = Le::read_u8(reader, pos + 3).map_err(Error::Io)?;

            let mut msg_header_size: u64 = 4;

            // If creation order is tracked (header flags bit 2), 2 more bytes
            let creation_order = if (header_flags & 0x04) != 0 {
                let co = Le::read_u16(reader, pos + 4).map_err(Error::Io)?;
                msg_header_size += 2;
                Some(co)
            } else {
                None
            };

            // Read message data
            let msg_data_start = pos + msg_header_size;
            let mut msg_data = vec![0u8; msg_size as usize];
            if msg_size > 0 {
                reader
                    .read_exact_at(msg_data_start, &mut msg_data)
                    .map_err(Error::Io)?;
            }

            let msg_type = MessageType::from_u8(msg_type_raw);

            messages.push(Message {
                msg_type,
                flags: msg_flags,
                creation_order,
                data: msg_data,
            });

            pos = msg_data_start + msg_size as u64;
        }

        Ok(())
    }

    /// Parse a continuation chunk (OCHK magic, messages, checksum).
    fn parse_continuation_chunk<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size: u64,
        header_flags: u8,
        size_of_offsets: u8,
        size_of_lengths: u8,
        messages: &mut Vec<Message>,
        continuations: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        // Verify OCHK magic
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != OCHK_MAGIC {
            return Err(Error::InvalidObjectHeader {
                msg: format!("expected OCHK magic at {:#x}, got {:?}", addr, magic),
            });
        }

        // The chunk is: OCHK(4) + messages(size - 8) + checksum(4)
        // So message region is from addr+4 to addr+size-4
        let msg_region_start = addr + 4;
        let msg_region_size = size - 8; // minus OCHK(4) and checksum(4)

        // Verify checksum over everything except the checksum itself
        let check_len = (size - 4) as usize;
        let mut check_data = vec![0u8; check_len];
        reader
            .read_exact_at(addr, &mut check_data)
            .map_err(Error::Io)?;
        let stored = Le::read_u32(reader, addr + size - 4).map_err(Error::Io)?;
        let computed = checksum::lookup3(&check_data);
        if computed != stored {
            return Err(Error::ChecksumMismatch {
                expected: stored,
                actual: computed,
            });
        }

        // Parse messages
        let before_count = messages.len();
        Self::parse_messages(
            reader,
            msg_region_start,
            msg_region_size,
            header_flags,
            size_of_offsets,
            size_of_lengths,
            messages,
        )?;

        // Check for new continuation messages
        for msg in &messages[before_count..] {
            if msg.msg_type == MessageType::ObjectHeaderContinuation {
                if let Some(cont) =
                    parse_continuation_payload(&msg.data, size_of_offsets, size_of_lengths)
                {
                    continuations.push(cont);
                }
            }
        }

        Ok(())
    }
}

/// Parse the payload of a continuation message: (offset, length).
fn parse_continuation_payload(
    data: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Option<(u64, u64)> {
    let o = size_of_offsets as usize;
    let l = size_of_lengths as usize;
    if data.len() < o + l {
        return None;
    }
    let addr = match size_of_offsets {
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
        8 => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
        _ => return None,
    };
    let size = match size_of_lengths {
        4 => u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as u64,
        8 => u64::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
            data[o + 4],
            data[o + 5],
            data[o + 6],
            data[o + 7],
        ]),
        _ => return None,
    };
    Some((addr, size))
}
