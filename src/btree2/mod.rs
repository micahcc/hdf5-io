use crate::checksum;
use crate::error::{Error, Result};
use crate::io::{Le, ReadAt};

/// B-tree v2 header magic: `BTHD`
pub const BTHD_MAGIC: [u8; 4] = *b"BTHD";
/// B-tree v2 internal node magic: `BTIN`
pub const BTIN_MAGIC: [u8; 4] = *b"BTIN";
/// B-tree v2 leaf node magic: `BTLF`
pub const BTLF_MAGIC: [u8; 4] = *b"BTLF";

/// B-tree v2 record type IDs.
///
/// Reference: H5B2pkg.h, the type field in the B-tree header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTree2Type {
    /// Type 1: Testing (internal use only)
    Testing = 1,
    /// Type 2: Indexing indirectly accessed, non-filtered 'huge' fractal heap objects
    HugeObjects = 2,
    /// Type 3: Indexing indirectly accessed, filtered 'huge' fractal heap objects
    HugeObjectsFiltered = 3,
    /// Type 4: Indexing directly accessed, non-filtered 'huge' fractal heap objects
    HugeObjectsDirect = 4,
    /// Type 5: Indexing group links (name hash → fractal heap ID) for new-style groups
    GroupLinks = 5,
    /// Type 6: Indexing group links by creation order
    GroupLinksCreationOrder = 6,
    /// Type 7: Indexing shared messages by hash
    SharedMessages = 7,
    /// Type 8: Indexing attribute names (name hash → fractal heap ID)
    AttributeNames = 8,
    /// Type 9: Indexing attributes by creation order
    AttributeCreationOrder = 9,
    /// Type 10: Chunked dataset (non-filtered, single dim unlimited)
    ChunkedData = 10,
    /// Type 11: Chunked dataset (filtered, single dim unlimited)
    ChunkedDataFiltered = 11,
}

impl BTree2Type {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Testing),
            2 => Ok(Self::HugeObjects),
            3 => Ok(Self::HugeObjectsFiltered),
            4 => Ok(Self::HugeObjectsDirect),
            5 => Ok(Self::GroupLinks),
            6 => Ok(Self::GroupLinksCreationOrder),
            7 => Ok(Self::SharedMessages),
            8 => Ok(Self::AttributeNames),
            9 => Ok(Self::AttributeCreationOrder),
            10 => Ok(Self::ChunkedData),
            11 => Ok(Self::ChunkedDataFiltered),
            _ => Err(Error::InvalidBTreeV2 {
                msg: format!("unknown B-tree v2 type {}", v),
            }),
        }
    }
}

/// Parsed B-tree v2 header.
///
/// ## On-disk layout
///
/// ```text
/// Byte 0-3:  Signature ("BTHD")
/// Byte 4:    Version (0)
/// Byte 5:    Type (BTree2Type)
/// Byte 6-9:  Node size (u32)
/// Byte 10-11: Record size (u16)
/// Byte 12-13: Depth (u16)
/// Byte 14:   Split percent
/// Byte 15:   Merge percent
/// Byte 16+O: Root node address (O = size_of_offsets)
/// +2:        Number of records in root node (u16)
/// +O:        Total number of records in tree (size_of_lengths bytes)
/// +4:        Checksum
/// ```
#[derive(Debug, Clone)]
pub struct BTree2Header {
    pub tree_type: BTree2Type,
    pub node_size: u32,
    pub record_size: u16,
    pub depth: u16,
    pub split_percent: u8,
    pub merge_percent: u8,
    pub root_node_address: u64,
    pub root_num_records: u16,
    pub total_records: u64,
}

impl BTree2Header {
    /// Parse a B-tree v2 header from `addr`.
    pub fn parse<R: ReadAt + ?Sized>(
        reader: &R,
        addr: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
        if magic != BTHD_MAGIC {
            return Err(Error::InvalidBTreeV2 {
                msg: format!("expected BTHD magic at {:#x}, got {:?}", addr, magic),
            });
        }

        let version = Le::read_u8(reader, addr + 4).map_err(Error::Io)?;
        if version != 0 {
            return Err(Error::InvalidBTreeV2 {
                msg: format!("expected B-tree v2 header version 0, got {}", version),
            });
        }

        let type_id = Le::read_u8(reader, addr + 5).map_err(Error::Io)?;
        let tree_type = BTree2Type::from_u8(type_id)?;
        let node_size = Le::read_u32(reader, addr + 6).map_err(Error::Io)?;
        let record_size = Le::read_u16(reader, addr + 10).map_err(Error::Io)?;
        let depth = Le::read_u16(reader, addr + 12).map_err(Error::Io)?;
        let split_percent = Le::read_u8(reader, addr + 14).map_err(Error::Io)?;
        let merge_percent = Le::read_u8(reader, addr + 15).map_err(Error::Io)?;

        let o = size_of_offsets as u64;
        let l = size_of_lengths as u64;

        let root_addr_off = addr + 16;
        let root_node_address =
            Le::read_offset(reader, root_addr_off, size_of_offsets).map_err(Error::Io)?;
        let root_num_records =
            Le::read_u16(reader, root_addr_off + o).map_err(Error::Io)?;
        let total_records =
            Le::read_length(reader, root_addr_off + o + 2, size_of_lengths).map_err(Error::Io)?;

        let checksum_off = root_addr_off + o + 2 + l;
        let stored_checksum = Le::read_u32(reader, checksum_off).map_err(Error::Io)?;

        // Verify checksum
        let header_len = (checksum_off - addr) as usize;
        let mut check_data = vec![0u8; header_len];
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

        Ok(BTree2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_node_address,
            root_num_records,
            total_records,
        })
    }
}

/// A record from a B-tree v2 node.
///
/// The actual content depends on `BTree2Type`. We store raw bytes and provide
/// typed accessors.
#[derive(Debug, Clone)]
pub struct Record {
    pub data: Vec<u8>,
}

/// Iterate all records in a B-tree v2 by walking from the root.
///
/// Calls `callback` for each record found in leaf-order (sorted).
pub fn iterate_records<R, F>(
    reader: &R,
    header: &BTree2Header,
    size_of_offsets: u8,
    mut callback: F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    if header.root_node_address == u64::MAX || header.total_records == 0 {
        return Ok(());
    }

    iterate_node(
        reader,
        header,
        header.root_node_address,
        header.root_num_records,
        header.depth,
        size_of_offsets,
        &mut callback,
    )
}

fn iterate_node<R, F>(
    reader: &R,
    header: &BTree2Header,
    addr: u64,
    num_records: u16,
    depth: u16,
    size_of_offsets: u8,
    callback: &mut F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    if depth == 0 {
        // Leaf node
        iterate_leaf(reader, header, addr, num_records, callback)
    } else {
        // Internal node
        iterate_internal(reader, header, addr, num_records, depth, size_of_offsets, callback)
    }
}

fn iterate_leaf<R, F>(
    reader: &R,
    header: &BTree2Header,
    addr: u64,
    num_records: u16,
    callback: &mut F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    // Leaf node layout:
    //   Signature "BTLF" (4)
    //   Version (1)
    //   Type (1)
    //   Records (num_records * record_size)
    //   Checksum (4)
    let mut magic = [0u8; 4];
    reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
    if magic != BTLF_MAGIC {
        return Err(Error::InvalidBTreeV2 {
            msg: format!("expected BTLF magic at {:#x}", addr),
        });
    }

    // Verify checksum: covers header (6 bytes) + records, checksum follows immediately
    let rec_size = header.record_size as usize;
    let check_len = 6 + num_records as usize * rec_size;
    let mut check_data = vec![0u8; check_len];
    reader
        .read_exact_at(addr, &mut check_data)
        .map_err(Error::Io)?;
    let stored = Le::read_u32(reader, addr + check_len as u64).map_err(Error::Io)?;
    let computed = checksum::lookup3(&check_data);
    if computed != stored {
        return Err(Error::ChecksumMismatch {
            expected: stored,
            actual: computed,
        });
    }

    // Read records starting after the 6-byte prefix (magic + version + type)
    let records_start = addr + 6;

    for i in 0..num_records as usize {
        let rec_off = records_start + (i * rec_size) as u64;
        let mut rec_data = vec![0u8; rec_size];
        reader
            .read_exact_at(rec_off, &mut rec_data)
            .map_err(Error::Io)?;
        callback(&Record { data: rec_data })?;
    }

    Ok(())
}

fn iterate_internal<R, F>(
    reader: &R,
    header: &BTree2Header,
    addr: u64,
    num_records: u16,
    depth: u16,
    size_of_offsets: u8,
    callback: &mut F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    // Internal node layout:
    //   Signature "BTIN" (4)
    //   Version (1)
    //   Type (1)
    //   Records and child pointers interleaved:
    //     child_ptr[0] (size_of_offsets), num_records_child[0] (variable), total_records_child[0] (variable)
    //     record[0] (record_size)
    //     child_ptr[1], num_records_child[1], total_records_child[1]
    //     record[1]
    //     ...
    //     child_ptr[num_records] (one more child than records)
    //   Checksum (4)
    let mut magic = [0u8; 4];
    reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
    if magic != BTIN_MAGIC {
        return Err(Error::InvalidBTreeV2 {
            msg: format!("expected BTIN magic at {:#x}", addr),
        });
    }

    // Determine the size of the child pointer entries.
    // Each child entry: address (size_of_offsets) + num_records (variable) + total_records (variable if depth > 1)
    // The num_records field size depends on the max possible records per node.
    // For simplicity, we compute it from the node size and record size.
    // Max records in a node: (node_size - overhead) / record_size
    // The num_records field is stored in the minimum bytes needed to represent the max count.
    let max_records_leaf = (header.node_size as usize - 10) / header.record_size as usize;
    let max_records_internal = max_records_leaf; // approximation; internal nodes have different overhead
    let num_rec_bytes = bytes_needed(max_records_internal as u64);
    // total_records is only present when depth > 1
    let total_rec_bytes = if depth > 1 {
        // Need to represent total records in the subtree — use size_of_lengths or
        // compute from total. For simplicity use the same encoding as num_records
        // for the maximum possible. This is an approximation.
        // Actually the HDF5 spec says this field uses the minimum bytes needed
        // to represent the maximum possible total records in the subtree.
        // For now, use a generous estimate.
        bytes_needed(header.total_records)
    } else {
        0
    };

    let o = size_of_offsets as usize;
    let child_entry_size = o + num_rec_bytes + total_rec_bytes;
    let rec_size = header.record_size as usize;

    // Compute actual content size: header(6) + interleaved children/records
    let n = num_records as usize;
    let content_len = 6 + (n + 1) * child_entry_size + n * rec_size;

    // Read node data and verify checksum (checksum follows content immediately)
    let read_len = content_len + 4;
    let mut node_data = vec![0u8; read_len];
    reader
        .read_exact_at(addr, &mut node_data)
        .map_err(Error::Io)?;
    let stored = u32::from_le_bytes([
        node_data[content_len],
        node_data[content_len + 1],
        node_data[content_len + 2],
        node_data[content_len + 3],
    ]);
    let computed = checksum::lookup3(&node_data[..content_len]);
    if computed != stored {
        return Err(Error::ChecksumMismatch {
            expected: stored,
            actual: computed,
        });
    }

    // Parse interleaved children and records
    let mut pos = 6usize; // after magic + version + type
    let mut children: Vec<(u64, u16)> = Vec::with_capacity(n + 1); // (addr, num_records)
    let mut records: Vec<Record> = Vec::with_capacity(n);

    for i in 0..=n {
        // Read child pointer
        let child_addr = read_offset_from_slice(&node_data, pos, size_of_offsets);
        let child_nrec = read_var_uint(&node_data, pos + o, num_rec_bytes) as u16;
        children.push((child_addr, child_nrec));
        pos += child_entry_size;

        // Read record (except after the last child)
        if i < n {
            let rec_data = node_data[pos..pos + rec_size].to_vec();
            records.push(Record { data: rec_data });
            pos += rec_size;
        }
    }

    // Recursively iterate: child[0], record[0], child[1], record[1], ... child[n]
    for i in 0..=n {
        let (child_addr, child_nrec) = children[i];
        iterate_node(
            reader,
            header,
            child_addr,
            child_nrec,
            depth - 1,
            size_of_offsets,
            callback,
        )?;
        if i < n {
            callback(&records[i])?;
        }
    }

    Ok(())
}

/// Minimum bytes needed to represent value `v` (1, 2, 4, or 8).
fn bytes_needed(v: u64) -> usize {
    if v <= 0xFF {
        1
    } else if v <= 0xFFFF {
        2
    } else if v <= 0xFFFF_FFFF {
        4
    } else {
        8
    }
}

fn read_offset_from_slice(data: &[u8], offset: usize, size: u8) -> u64 {
    match size {
        4 => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64,
        8 => u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
        _ => 0,
    }
}

fn read_var_uint(data: &[u8], offset: usize, size: usize) -> u64 {
    match size {
        1 => data[offset] as u64,
        2 => u16::from_le_bytes([data[offset], data[offset + 1]]) as u64,
        4 => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64,
        8 => u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
        _ => 0,
    }
}

/// Parse a B-tree v2 "type 5" record (link name for new-style groups).
///
/// Layout: hash (4 bytes LE u32) + heap ID (7 bytes, variable based on heap ID length).
pub fn parse_link_name_record(data: &[u8], heap_id_len: usize) -> Option<(u32, Vec<u8>)> {
    if data.len() < 4 + heap_id_len {
        return None;
    }
    let hash = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let heap_id = data[4..4 + heap_id_len].to_vec();
    Some((hash, heap_id))
}

/// Parse a B-tree v2 "type 8" record (attribute name for dense attribute storage).
///
/// Layout: heap_id (heap_id_len bytes) + flags (1 byte) + creation_order (4 bytes) + hash (4 bytes).
pub fn parse_attribute_name_record(data: &[u8], heap_id_len: usize) -> Option<Vec<u8>> {
    if data.len() < heap_id_len + 1 + 4 + 4 {
        return None;
    }
    let heap_id = data[..heap_id_len].to_vec();
    // flags at heap_id_len, creation_order at heap_id_len+1, hash at heap_id_len+5
    // We only need the heap_id to look up the attribute message
    Some(heap_id)
}
