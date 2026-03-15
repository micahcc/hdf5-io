use crate::checksum;
use crate::error::Error;
use crate::error::Result;
use crate::filters::FilterPipeline;
use crate::io::Le;
use crate::io::ReadAt;

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
    /// If I/O filters are present, the parsed filter pipeline.
    pub filter_pipeline: Option<FilterPipeline>,
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

        let next_huge_object_id =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_bt2_address = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let free_space_in_managed =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let free_space_manager_address =
            Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let managed_space_total =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_space_allocated =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_alloc_iterator_offset =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let managed_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_total_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let huge_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_total_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let tiny_objects_count =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;

        let table_width = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_block_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_direct_block_size =
            Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
        pos += l;
        let max_heap_size_bits = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let starting_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;
        let root_block_address =
            Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
        pos += o;
        let current_root_rows = Le::read_u16(reader, pos).map_err(Error::Io)?;
        pos += 2;

        // Optional I/O filter info
        let (filtered_root_direct_block_size, io_filter_mask, filter_pipeline) =
            if io_filter_encoded_length > 0 {
                let fsize = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
                pos += l;
                let mask = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                // Read and parse the encoded filter pipeline message
                let fpl = io_filter_encoded_length as usize;
                let mut filter_data = vec![0u8; fpl];
                reader
                    .read_exact_at(pos, &mut filter_data)
                    .map_err(Error::Io)?;
                let pipeline = FilterPipeline::parse(&filter_data)?;
                pos += fpl as u64;
                (Some(fsize), Some(mask), Some(pipeline))
            } else {
                (None, None, None)
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
            filter_pipeline,
        })
    }

    /// Compute the block size for a given row number.
    ///
    /// Row 0 and row 1 both have `starting_block_size`.
    /// Row 2 = 2 * starting_block_size, row 3 = 4 * starting_block_size, etc.
    ///
    /// Formula: row_block_size[0] = starting_block_size
    ///          row_block_size[u] = starting_block_size * 2^(u-1) for u >= 1
    /// Compute the block size for a given row number.
    ///
    /// Row 0 and row 1 both have `starting_block_size`.
    /// Row 2 = 2 * starting_block_size, row 3 = 4 * starting_block_size, etc.
    /// Does NOT cap at max_direct_block_size — indirect rows continue doubling
    /// to represent the total managed space of their sub-trees (H5HFdtable.c:110-119).
    pub fn block_size_for_row(&self, row: u32) -> u64 {
        if row <= 1 {
            self.starting_block_size
        } else {
            self.starting_block_size * (1u64 << (row - 1))
        }
    }

    /// Number of rows that contain direct blocks.
    ///
    /// C: `max_dblock_rows = (max_direct_bits - start_bits) + 2` (H5HFdtable.c:94)
    pub fn max_direct_rows(&self) -> u32 {
        if self.max_direct_block_size == 0 || self.starting_block_size == 0 {
            return 0;
        }
        (self.max_direct_block_size / self.starting_block_size).trailing_zeros() as u32 + 2
    }

    /// Number of rows in a child indirect block at a given row.
    ///
    /// C: `nrows = log2(row_block_size) - first_row_bits + 1` (H5HFdtable.c:237-251)
    pub fn child_indirect_nrows(&self, row: u32) -> u32 {
        let first_row_bits = (self.starting_block_size * self.table_width as u64).trailing_zeros();
        let log2_bs = self.block_size_for_row(row).trailing_zeros();
        log2_bs - first_row_bits + 1
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
        2 => read_huge_object(reader, header, heap_id, size_of_offsets, size_of_lengths),
        _ => Err(Error::InvalidFractalHeap {
            msg: format!("unknown heap ID type {}", id_type),
        }),
    }
}

fn read_huge_object<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    let o = size_of_offsets as usize;
    let l = size_of_lengths as usize;

    // `huge_ids_direct` is NOT a flag bit — it's derived from whether the
    // heap ID is large enough to hold address + length inline.
    // H5HFhuge.c:166-219: sizeof_addr + sizeof_size [+ 4 + sizeof_size if filtered]
    let has_filters = header.io_filter_encoded_length > 0;
    let needed = if has_filters { o + l + 4 + l } else { o + l };
    let directly_accessed = needed <= (header.heap_id_length as usize).saturating_sub(1);

    if directly_accessed {
        if has_filters {
            // Direct access with filters: address + filtered_length + filter_mask + mem_length
            if heap_id.len() < 1 + o + l + 4 + l {
                return Err(Error::InvalidFractalHeap {
                    msg: "huge object heap ID too short for filtered direct access".into(),
                });
            }
            let address = read_var_le(&heap_id[1..], o);
            let filtered_length = read_var_le(&heap_id[1 + o..], l);
            let filter_mask = u32::from_le_bytes([
                heap_id[1 + o + l],
                heap_id[1 + o + l + 1],
                heap_id[1 + o + l + 2],
                heap_id[1 + o + l + 3],
            ]);
            let _mem_length = read_var_le(&heap_id[1 + o + l + 4..], l);

            let mut compressed = vec![0u8; filtered_length as usize];
            reader
                .read_exact_at(address, &mut compressed)
                .map_err(Error::Io)?;

            if let Some(pipeline) = &header.filter_pipeline {
                if filter_mask == 0 {
                    compressed = pipeline.decompress(compressed)?;
                }
            }
            Ok(compressed)
        } else {
            // Direct access without filters: address + length
            if heap_id.len() < 1 + o + l {
                return Err(Error::InvalidFractalHeap {
                    msg: "huge object heap ID too short for direct access".into(),
                });
            }
            let address = read_var_le(&heap_id[1..], o);
            let length = read_var_le(&heap_id[1 + o..], l);

            let mut data = vec![0u8; length as usize];
            reader
                .read_exact_at(address, &mut data)
                .map_err(Error::Io)?;
            Ok(data)
        }
    } else {
        // Indirectly accessed: heap ID contains a unique ID, look up in B-tree v2
        let id_bytes = (header.heap_id_length as usize).saturating_sub(1);
        if heap_id.len() < 1 + id_bytes {
            return Err(Error::InvalidFractalHeap {
                msg: "huge object heap ID too short for indirect access".into(),
            });
        }
        let obj_id = read_var_le(&heap_id[1..], id_bytes);

        if header.huge_bt2_address == u64::MAX {
            return Err(Error::InvalidFractalHeap {
                msg: "huge B-tree v2 address is undefined".into(),
            });
        }

        let bt2 = crate::btree2::BTree2Header::parse(
            reader,
            header.huge_bt2_address,
            size_of_offsets,
            size_of_lengths,
        )?;

        // Search B-tree for matching ID
        // Type 2 record: address(O) + length(L) + id(L)
        let mut found: Option<(u64, u64)> = None;
        crate::btree2::iterate_records(reader, &bt2, size_of_offsets, |record| {
            if record.data.len() >= o + l + l {
                let addr = read_var_le(&record.data, o);
                let len = read_var_le(&record.data[o..], l);
                let id = read_var_le(&record.data[o + l..], l);
                if id == obj_id {
                    found = Some((addr, len));
                }
            }
            Ok(())
        })?;

        match found {
            Some((address, length)) => {
                let mut data = vec![0u8; length as usize];
                reader
                    .read_exact_at(address, &mut data)
                    .map_err(Error::Io)?;
                Ok(data)
            }
            None => Err(Error::InvalidFractalHeap {
                msg: format!("huge object ID {} not found in B-tree", obj_id),
            }),
        }
    }
}

fn read_managed_object_inner<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    heap_id: &[u8],
    size_of_offsets: u8,
    size_of_lengths: u8,
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
        // Root is a direct block — may be filtered
        let filtered_size = header.filtered_root_direct_block_size;
        let filter_mask = header.io_filter_mask.unwrap_or(0);
        let block_size = header.starting_block_size;
        read_from_direct_block(
            reader,
            header,
            root_addr,
            heap_offset,
            obj_length,
            size_of_offsets,
            filtered_size,
            filter_mask,
            block_size,
        )
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
            size_of_lengths,
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
fn decode_managed_heap_id(heap_id: &[u8], offset_bits: usize, id_len: usize) -> Result<(u64, u64)> {
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
///
/// If `filtered_size` is `Some`, the block is compressed: read `filtered_size` bytes,
/// decompress through the heap's filter pipeline, then extract the object.
fn read_from_direct_block<R: ReadAt + ?Sized>(
    reader: &R,
    header: &FractalHeapHeader,
    block_addr: u64,
    heap_offset: u64,
    obj_length: u64,
    _size_of_offsets: u8,
    filtered_size: Option<u64>,
    _filter_mask: u32,
    block_size: u64,
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

    if let (Some(fsize), Some(pipeline)) = (filtered_size, &header.filter_pipeline) {
        // Filtered direct block: read the on-disk (compressed) data and decompress
        let read_size = fsize as usize;
        let mut compressed = vec![0u8; read_size];
        reader
            .read_exact_at(block_addr, &mut compressed)
            .map_err(Error::Io)?;

        // Decompress through the filter pipeline
        let decompressed = pipeline.decompress(compressed)?;

        // Verify we got the expected block size
        if decompressed.len() != block_size as usize {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "filtered direct block decompressed to {} bytes, expected {}",
                    decompressed.len(),
                    block_size
                ),
            });
        }

        // Verify FHDB magic in decompressed data
        if decompressed.len() < 4 || decompressed[..4] != FHDB_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "expected FHDB magic in decompressed block at {:#x}",
                    block_addr
                ),
            });
        }

        // Extract the object from the decompressed block
        let off = heap_offset as usize;
        let end = off + obj_length as usize;
        if end > decompressed.len() {
            return Err(Error::InvalidFractalHeap {
                msg: format!(
                    "object at offset {} length {} exceeds decompressed block size {}",
                    off,
                    obj_length,
                    decompressed.len()
                ),
            });
        }
        Ok(decompressed[off..end].to_vec())
    } else {
        // Unfiltered direct block: read directly from file
        let mut magic = [0u8; 4];
        reader
            .read_exact_at(block_addr, &mut magic)
            .map_err(Error::Io)?;
        if magic != FHDB_MAGIC {
            return Err(Error::InvalidFractalHeap {
                msg: format!("expected FHDB magic at {:#x}", block_addr),
            });
        }

        // The heap offset in the managed object ID includes the direct block overhead.
        // Objects are addressed relative to the start of the direct block.
        let mut obj_data = vec![0u8; obj_length as usize];
        reader
            .read_exact_at(block_addr + heap_offset, &mut obj_data)
            .map_err(Error::Io)?;
        Ok(obj_data)
    }
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
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    // Indirect block layout:
    //   Signature "FHIB" (4)
    //   Version (1)
    //   Heap header address (size_of_offsets)
    //   Block offset within heap (block_offset_byte_size bytes)
    //   Child block entries:
    //     For direct block rows: address (O) [+ filtered_size (L) + filter_mask (4) if filtered]
    //     For indirect block rows: address (O)
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
    let has_filters = header.io_filter_encoded_length > 0;

    // Determine which rows are direct vs indirect.
    let max_direct_rows = header.max_direct_rows();

    // Walk through entries to find the one containing our heap_offset
    let mut current_heap_offset = 0u64;

    for row in 0..nrows {
        let block_size = header.block_size_for_row(row);
        let is_direct = row < max_direct_rows;

        for _col in 0..width {
            let child_addr = Le::read_offset(reader, pos, size_of_offsets).map_err(Error::Io)?;
            pos += size_of_offsets as u64;

            // For filtered direct blocks, read filtered_size and filter_mask
            let (entry_filtered_size, entry_filter_mask) = if is_direct && has_filters {
                let fsize = Le::read_length(reader, pos, size_of_lengths).map_err(Error::Io)?;
                pos += size_of_lengths as u64;
                let fmask = Le::read_u32(reader, pos).map_err(Error::Io)?;
                pos += 4;
                (Some(fsize), fmask)
            } else {
                (None, 0)
            };

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
                        entry_filtered_size,
                        entry_filter_mask,
                        block_size,
                    );
                } else {
                    // Recurse into child indirect block
                    let child_nrows = header.child_indirect_nrows(row);
                    return read_from_indirect_block(
                        reader,
                        header,
                        child_addr,
                        child_nrows,
                        offset_in_block,
                        obj_length,
                        size_of_offsets,
                        size_of_lengths,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a FractalHeapHeader with just the fields needed for doubling-table math.
    fn make_header(
        starting_block_size: u64,
        max_direct_block_size: u64,
        table_width: u16,
    ) -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 0,
            flags: 0,
            max_managed_object_size: 0,
            next_huge_object_id: 0,
            huge_bt2_address: u64::MAX,
            free_space_in_managed: 0,
            free_space_manager_address: u64::MAX,
            managed_space_total: 0,
            managed_space_allocated: 0,
            managed_alloc_iterator_offset: 0,
            managed_objects_count: 0,
            huge_objects_total_size: 0,
            huge_objects_count: 0,
            tiny_objects_total_size: 0,
            tiny_objects_count: 0,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size_bits: 32,
            starting_root_rows: 0,
            root_block_address: u64::MAX,
            current_root_rows: 0,
            filtered_root_direct_block_size: None,
            io_filter_mask: None,
            filter_pipeline: None,
        }
    }

    #[test]
    fn test_block_size_for_row() {
        let h = make_header(512, 65536, 4);
        assert_eq!(h.block_size_for_row(0), 512);
        assert_eq!(h.block_size_for_row(1), 512);
        assert_eq!(h.block_size_for_row(2), 1024);
        assert_eq!(h.block_size_for_row(3), 2048);
        assert_eq!(h.block_size_for_row(8), 65536);
        // Indirect rows continue doubling past max_direct_block_size
        assert_eq!(h.block_size_for_row(9), 131072);
        assert_eq!(h.block_size_for_row(10), 262144);
    }

    #[test]
    fn test_max_direct_rows() {
        // start=512, max_direct=65536: (16-9)+2 = 9
        let h = make_header(512, 65536, 4);
        assert_eq!(h.max_direct_rows(), 9);

        // start=1024, max_direct=65536: (16-10)+2 = 8
        let h2 = make_header(1024, 65536, 4);
        assert_eq!(h2.max_direct_rows(), 8);

        // start=256, max_direct=65536: (16-8)+2 = 10
        let h3 = make_header(256, 65536, 4);
        assert_eq!(h3.max_direct_rows(), 10);

        // start=512, max_direct=512: (9-9)+2 = 2
        let h4 = make_header(512, 512, 4);
        assert_eq!(h4.max_direct_rows(), 2);
    }

    #[test]
    fn test_child_indirect_nrows() {
        // start=512, width=4, max_direct=65536
        // first_row_bits = log2(512*4) = log2(2048) = 11
        let h = make_header(512, 65536, 4);

        // Row 9 (first indirect row): block_size = 131072, log2 = 17
        // nrows = 17 - 11 + 1 = 7
        assert_eq!(h.child_indirect_nrows(9), 7);

        // Row 10: block_size = 262144, log2 = 18
        // nrows = 18 - 11 + 1 = 8
        assert_eq!(h.child_indirect_nrows(10), 8);
    }

    #[test]
    fn huge_ids_direct_computed_not_flag() {
        // huge_ids_direct is derived from id_len, NOT from flags byte.
        // flags=0x02 is "checksum direct blocks", not "direct huge IDs".
        // H5HFhuge.c: direct if sizeof_addr + sizeof_size <= id_len - 1
        let sizeof_addr: u8 = 8;
        let sizeof_size: u8 = 8;

        // id_len=17 → 17-1=16 >= 8+8=16 → direct
        let h1 = FractalHeapHeader {
            heap_id_length: 17,
            io_filter_encoded_length: 0,
            flags: 0, // no flag bits set, yet should still be "direct"
            ..make_header(512, 65536, 4)
        };
        let needed = sizeof_addr as usize + sizeof_size as usize;
        assert!(needed <= (h1.heap_id_length as usize) - 1);

        // id_len=8 → 8-1=7 < 16 → indirect (must use B-tree)
        let h2 = FractalHeapHeader {
            heap_id_length: 8,
            io_filter_encoded_length: 0,
            flags: 0x02, // checksum flag set, but should NOT mean "direct"
            ..make_header(512, 65536, 4)
        };
        assert!(needed > (h2.heap_id_length as usize) - 1);

        // With filters: need sizeof_addr + sizeof_size + 4 + sizeof_size
        let needed_filtered =
            sizeof_addr as usize + sizeof_size as usize + 4 + sizeof_size as usize;
        // id_len=25 → 25-1=24 < 28 → indirect
        let h3 = FractalHeapHeader {
            heap_id_length: 25,
            io_filter_encoded_length: 12,
            flags: 0,
            ..make_header(512, 65536, 4)
        };
        assert!(needed_filtered > (h3.heap_id_length as usize) - 1);

        // id_len=29 → 29-1=28 >= 28 → direct
        let h4 = FractalHeapHeader {
            heap_id_length: 29,
            io_filter_encoded_length: 12,
            flags: 0,
            ..make_header(512, 65536, 4)
        };
        assert!(needed_filtered <= (h4.heap_id_length as usize) - 1);
    }
}
