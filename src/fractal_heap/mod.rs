use crate::checksum;
use crate::error::{Error, Result};
use crate::io::{Le, ReadAt};

/// Fractal heap header magic: `FRHP`
pub const FRHP_MAGIC: [u8; 4] = *b"FRHP";
/// Fractal heap direct block magic: `FHDB`
pub const FHDB_MAGIC: [u8; 4] = *b"FHDB";
/// Fractal heap indirect block magic: `FHIB`
pub const FHIB_MAGIC: [u8; 4] = *b"FHIB";

/// Parsed fractal heap header.
///
/// ## On-disk layout
///
/// ```text
/// Byte 0-3:   Signature ("FRHP")
/// Byte 4:     Version (0)
/// Byte 5-6:   Heap ID length (u16)
/// Byte 7-8:   I/O filter pipeline encoded length (u16)
/// Byte 9:     Flags
/// Byte 10-13: Max managed object size (u32)
/// Byte 14+L:  Next huge object ID (size_of_lengths)
/// +O:         B-tree v2 address for huge objects (size_of_offsets)
/// +L:         Free space in managed objects (size_of_lengths)
/// +O:         Free space manager address (size_of_offsets)
/// +L:         Managed space total (size_of_lengths)
/// +L:         Managed space allocated (size_of_lengths)
/// +L:         Managed alloc iterator offset (size_of_lengths)
/// +L:         Managed objects count (size_of_lengths)
/// +L:         Huge objects total size (size_of_lengths)
/// +L:         Huge objects count (size_of_lengths)
/// +L:         Tiny objects total size (size_of_lengths)
/// +L:         Tiny objects count (size_of_lengths)
/// +2:         Table width (u16)
/// +L:         Starting block size (size_of_lengths)
/// +L:         Max direct block size (size_of_lengths)
/// +2:         Max heap size (u16, in bits — e.g. 32 means max offset = 2^32)
/// +2:         Starting # of rows in root indirect block (u16)
/// +O:         Root block address (size_of_offsets)
/// +2:         Current # of rows in root indirect block (u16)
/// [optional I/O filter info if encoded length > 0]
/// +4:         Checksum
/// ```
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_id_length: u16,
    pub io_filter_encoded_length: u16,
    pub flags: u8,
    pub max_managed_object_size: u32,
    pub next_huge_object_id: u64,
    pub huge_bt2_address: u64,
    pub free_space_in_managed: u64,
    pub free_space_manager_address: u64,
    pub managed_space_total: u64,
    pub managed_space_allocated: u64,
    pub managed_alloc_iterator_offset: u64,
    pub managed_objects_count: u64,
    pub huge_objects_total_size: u64,
    pub huge_objects_count: u64,
    pub tiny_objects_total_size: u64,
    pub tiny_objects_count: u64,
    pub table_width: u16,
    pub starting_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size_bits: u16,
    pub starting_root_rows: u16,
    pub root_block_address: u64,
    pub current_root_rows: u16,
    /// If I/O filters are present, the filtered root direct block size.
    pub filtered_root_direct_block_size: Option<u64>,
    /// If I/O filters are present, the I/O filter mask.
    pub io_filter_mask: Option<u32>,
}

impl FractalHeapHeader {
    pub fn parse<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != FRHP_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected FRHP magic at {:#x}, got {:?}", addr, magic),
            });
        }

        let version = Le::read_u8(reader, addr + 4).map_err(Error::Io)?;
        if version != 0 {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected fractal heap version 0, got {}", version),
            });
        }

        let heap_id_length = Le::read_u16(reader, addr + 5).map_err(Error::Io)?;
        let io_filter_encoded_length = Le::read_u16(reader, addr + 7).map_err(Error::Io)?;
        let flags = Le::read_u8(reader, addr + 9).map_err(Error::Io)?;
        let max_managed_object_size = Le::read_u32(reader, addr + 10).map_err(Error::Io)?;

        let o = size_of_offsets as u64;
        let l = size_of_lengths as u64;
        let mut pos = addr + 14;

        let next_huge_object_id = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_bt2_address = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let free_space_in_managed = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let free_space_manager_address = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let managed_space_total = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_space_allocated = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_alloc_iterator_offset = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_objects_count = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_total_size = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_count = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_total_size = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_count = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;

        let table_width = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_block_size = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_direct_block_size = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_heap_size_bits = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let root_block_address = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let current_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;

        // Optional I/O filter info
        let (filtered_root_direct_block_size, io_filter_mask) =
            if io_filter_encoded_length > 0 {
                let fsize = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
                pos += l;
                let mask = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                // Skip the actual encoded filter pipeline bytes
                pos += io_filter_encoded_length as u64;
                (Some(fsize), Some(mask))
            } else {
                (None, None)
            };

        // Checksum
        let stored_checksum = Le::read_u32(reader, pos).map_err(Error::Io)?;
        let check_len = (pos - addr) as usize;
        let mut check_data = vec![0u8; check_len];
        reader
            .read_exact_at(addr, &mut check_data)
            .map_err(Error::Io)?;
        let computed = checksum::lookup3(&check_data);
        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        Ok(FractalHeapHeader {
            heap_id_length,
            io_filter_encoded_length,
            flags,
            max_managed_object_size,
            next_huge_object_id,
            huge_bt2_address,
            free_space_in_managed,
            free_space_manager_address,
            managed_space_total,
            managed_space_allocated,
            managed_alloc_iterator_offset,
            managed_objects_count,
            huge_objects_total_size,
            huge_objects_count,
            tiny_objects_total_size,
            tiny_objects_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size_bits,
            starting_root_rows,
            root_block_address,
            current_root_rows,
            filtered_root_direct_block_size,
            io_filter_mask,
        })
    }

    /// Compute the block size for a given row number.
    ///
    /// Row 0 and row 1 both have `starting_block_size`.
    /// Row 2 = 2 * starting_block_size, row 3 = 4 * starting_block_size, etc.
    ///
    /// Formula: row_block_size[0] = starting_block_size
    ///          row_block_size[u] = starting_block_size * 2^(u-1) for u >= 1
    pub fn block_size_for_row(&self, row: u32) -> u64 {
        if row <= 1 {
            self.starting_block_size
        } else {
            let size = self.starting_block_size * (1u64 << (row - 1));
            size.min(self.max_direct_block_size)
        }
    }

    /// Number of bytes needed to encode a block offset within the heap.
    /// This is ceil(max_heap_size_bits / 8).
    pub fn block_offset_byte_size(&self) -> usize {
        ((self.max_heap_size_bits as usize) + 7) / 8
    }

    /// True if the root block is a direct block (current_root_rows == 0).
    pub fn root_is_direct(&self) -> bool {
        self.current_root_rows == 0
    }
}

/// Read a managed object from the fractal heap given a heap ID.
///
/// For "managed" objects (the common case), the heap ID encodes:
/// - Version/type (bits 6-7 of byte 0): 0 = managed, 1 = tiny, 2 = huge
/// - Offset into the heap (variable bits)
/// - Length of the object (variable bits)
///
/// This function handles the "managed" type by locating the direct block
/// and reading the object bytes.
pub fn read_managed_object<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    if heap_id.is_empty() {
        return Err(Error::InvalidFractalHeap {
            msg: "empty heap ID".into(),
        });
    }

    let id_type = (heap_id[0] >> 4) & 0x03;
    match id_type {
        0 => read_managed_object_inner(reader, header, heap_id, size_of_offsets, size_of_lengths),
        1 => read_tiny_object(header, heap_id),
        2 => {
            // Huge object — stored separately, pointed to by B-tree v2
            Err(Error::InvalidFractalHeap {
                msg: "huge object reading not yet implemented".into(),
            })
        }
        _ => Err(Error::InvalidFractalHeap {
            msg: format!("unknown heap ID type {}", id_type),
        }),
    }
}

fn read_managed_object_inner<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    _size_of_lengths: u8,
) -> Result<Vec<u8>> {
    // Managed heap ID layout (after the type/version nibble):
    //   offset: variable bits (max_heap_size_bits wide)
    //   length: remaining bits
    // The offset is the byte position within the managed heap space.
    // We need to decode it from the heap ID bytes.

    let offset_bits = header.max_heap_size_bits as usize;
    // The length field uses the remaining bytes of the heap ID after the offset
    let id_len = header.heap_id_length as usize;

    // Read the offset and length from the heap ID (skip first nibble of byte 0)
    // The encoding packs offset and length into the heap_id bytes starting at bit 4 of byte 0.
    let (heap_offset, obj_length) = decode_managed_heap_id(heap_id, offset_bits, id_len)?;

    // Now we need to find which direct block contains this offset.
    // Walk the fractal heap structure to find the right direct block.
    let root_addr = header.root_block_address;
    if root_addr == u64::MAX {
        return Err(Error::InvalidFractalHeap {
            msg: "root block address is undefined".into(),
        });
    }

    if header.root_is_direct() {
        // Root is a direct block — read from it directly
        read_from_direct_block(reader, header, root_addr, heap_offset, obj_length, size_of_offsets)
    } else {
        // Root is an indirect block — need to traverse
        read_from_indirect_block(
            reader,
            header,
            root_addr,
            header.current_root_rows as u32,
            heap_offset,
            obj_length,
            size_of_offsets,
        )
    }
}

fn read_tiny_object(_header: &FractalHeapHeader, heap_id: &[u8]) -> Result<Vec<u8>> {
    // Tiny objects are stored directly in the heap ID.
    // The lower 4 bits of byte 0 encode the length (minus 1).
    let len = ((heap_id[0] & 0x0F) + 1) as usize;
    if heap_id.len() < 1 + len {
        return Err(Error::InvalidFractalHeap {
            msg: "tiny object extends past heap ID".into(),
        });
    }
    Ok(heap_id[1..1 + len].to_vec())
}

/// Decode a managed heap ID into (offset, length).
fn decode_managed_heap_id(
    heap_id: &[u8],
    offset_bits: usize,
    id_len: usize,
) -> Result<(u64, u64)> {
    // The heap ID starts with a flags byte. Bits 4-5 encode the type (0 = managed).
    // Bits 0-3 are reserved. The actual offset and length are packed into
    // bytes 1..id_len.
    //
    // For managed objects:
    //   Bytes 1..: offset (ceil(offset_bits/8) bytes, LE) followed by length (remaining bytes, LE)

    let offset_bytes = (offset_bits + 7) / 8;
    let length_bytes = id_len - 1 - offset_bytes;

    if heap_id.len() < 1 + offset_bytes + length_bytes {
        return Err(Error::InvalidFractalHeap {
            msg: "heap ID too short for managed object".into(),
        });
    }

    let offset = read_var_le(&heap_id[1..], offset_bytes);
    let length = read_var_le(&heap_id[1 + offset_bytes..], length_bytes);

    Ok((offset, length))
}

/// Read variable-length little-endian unsigned integer.
fn read_var_le(data: &[u8], len: usize) -> u64 {
    let mut result = 0u64;
    for i in 0..len.min(8) {
        result |= (data[i] as u64) << (i * 8);
    }
    result
}

/// Read an object from a direct block at the given heap offset.
fn read_from_direct_block<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    block_addr: u64,
    heap_offset: u64,
    obj_length: u64,
    _size_of_offsets: u8,
) -> Result<Vec<u8>> {
    // Direct block layout:
    //   Signature "FHDB" (4)
    //   Version (1)
    //   Heap header address (size_of_offsets)
    //   Block offset within heap (block_offset_byte_size)
    //   [optional checksum if heap flags indicate it]
    //   Data...
    //
    // The object is at (block_data_start + heap_offset_within_block).

    let mut magic = [0u8; 4];
    reader
        .read_exact_at(block_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != FHDB_MAGIC {
        return Err(Error::InvalidFractalHeap {
            msg: format!("expected FHDB magic at {:#x}", block_addr),
        });
    }

    // Skip: magic(4) + version(1) + heap_addr(size_of_offsets) + block_offset(block_offset_byte_size)
    let block_offset_size = header.block_offset_byte_size();
    let overhead = 5 + _size_of_offsets as u64 + block_offset_size as u64;
    let has_checksum = (header.flags & 0x02) != 0;
    let _checksum_size = if has_checksum { 4u64 } else { 0 };

    // The data starts after the overhead
    let data_start = block_addr + overhead;

    // The heap_offset is relative to the start of managed heap space.
    // For a root direct block, the block's heap offset is 0, so the object
    // is at data_start + heap_offset.
    // For non-root direct blocks, we'd need to subtract the block's heap offset.
    // For now, assume this is the correct direct block and use relative addressing.

    let mut obj_data = vec![0u8; obj_length as usize];
    reader
        .read_exact_at(data_start + heap_offset, &mut obj_data)
        .map_err(Error::Io)?;
    Ok(obj_data)
}

/// Walk an indirect block to find the direct block containing `heap_offset`.
fn read_from_indirect_block<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    iblock_addr: u64,
    nrows: u32,
    heap_offset: u64,
    obj_length: u64,
    size_of_offsets: u8,
) -> Result<Vec<u8>> {
    // Indirect block layout:
    //   Signature "FHIB" (4)
    //   Version (1)
    //   Heap header address (size_of_offsets)
    //   Block offset within heap (block_offset_byte_size bytes)
    //   Child block entries:
    //     For direct block rows: address (size_of_offsets) [+ filtered size + filter mask if filtered]
    //     For indirect block rows: address (size_of_offsets)
    //   Checksum (4)

    let mut magic = [0u8; 4];
    reader
        .read_exact_at(iblock_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != FHIB_MAGIC {
        return Err(Error::InvalidFractalHeap {
            msg: format!("expected FHIB magic at {:#x}", iblock_addr),
        });
    }

    let block_offset_size = header.block_offset_byte_size();
    let overhead = 5 + size_of_offsets as u64 + block_offset_size as u64;
    let mut pos = iblock_addr + overhead;

    let width = header.table_width as u64;

    // Determine which rows are direct vs indirect.
    // Rows 0..max_direct_rows are direct blocks.
    // max_direct_rows = log2(max_direct_block_size / starting_block_size) + 1
    let max_direct_rows = if header.max_direct_block_size > 0 && header.starting_block_size > 0 {
        (header.max_direct_block_size / header.starting_block_size)
            .trailing_zeros() as u32
            + 1
    } else {
        nrows
    };

    // Walk through entries to find the one containing our heap_offset
    let mut current_heap_offset = 0u64;

    for row in 0..nrows {
        let block_size = header.block_size_for_row(row);
        let is_direct = row < max_direct_rows;

        for _col in 0..width {
            let child_addr =
                Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
            pos += size_of_offsets as u64;

            // If filtered direct blocks, skip filtered_size and filter_mask
            if is_direct && header.io_filter_encoded_length > 0 {
                // filtered_size (size_of_lengths) + filter_mask (4)
                // For simplicity, skip
                pos += 8 + 4; // TODO: use actual size_of_lengths
            }

            let next_offset = current_heap_offset + block_size;

            if heap_offset >= current_heap_offset && heap_offset < next_offset {
                // Found the block
                if child_addr == u64::MAX {
                    return Err(Error::InvalidFractalHeap {
                        msg: "child block address is undefined".into(),
                    });
                }
                let offset_in_block = heap_offset - current_heap_offset;
                if is_direct {
                    return read_from_direct_block(
                        reader,
                        header,
                        child_addr,
                        offset_in_block,
                        obj_length,
                        size_of_offsets,
                    );
                } else {
                    // Recurse into indirect block
                    let child_nrows = row - max_direct_rows + 1; // simplified
                    return read_from_indirect_block(
                        reader,
                        header,
                        child_addr,
                        child_nrows,
                        offset_in_block,
                        obj_length,
                        size_of_offsets,
                    );
                }
            }

            current_heap_offset = next_offset;
        }
    }

    Err(Error::InvalidFractalHeap {
        msg: format!(
            "heap offset {} not found in indirect block at {:#x}",
            heap_offset, iblock_addr
        ),
    })
}
