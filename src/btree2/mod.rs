use crate::checksum;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

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
        let root_num_records = Le::read_u16(reader, root_addr_off + o).map_err(Error::Io)?;
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

/// Per-depth-level node capacity information, matching H5B2_node_info_t in the C library.
struct NodeInfo {
    /// Maximum records in a node at this depth level.
    max_nrec: usize,
    /// Maximum total records in a subtree rooted at this depth level.
    cum_max_nrec: u64,
    /// Bytes needed to encode cum_max_nrec (0 for leaf level).
    cum_max_nrec_size: usize,
}

/// Compute the node_info array for all depth levels, plus the global max_nrec_size.
///
/// Matches H5B2_hdr_init() in H5B2hdr.c.
fn compute_node_info(
    node_size: u32,
    record_size: u16,
    sizeof_addr: u8,
    depth: u16,
) -> (usize, Vec<NodeInfo>) {
    let mut info = Vec::with_capacity(depth as usize + 1);

    // Level 0: leaf nodes
    // max_nrec = (node_size - H5B2_METADATA_PREFIX_SIZE) / record_size
    // H5B2_METADATA_PREFIX_SIZE = signature(4) + version(1) + type(1) + checksum(4) = 10
    let leaf_max_nrec = (node_size as usize - 10) / record_size as usize;
    info.push(NodeInfo {
        max_nrec: leaf_max_nrec,
        cum_max_nrec: leaf_max_nrec as u64,
        cum_max_nrec_size: 0,
    });

    // max_nrec_size: bytes to encode the max record count in any node (leaf has the most)
    let max_nrec_size = limit_enc_size(leaf_max_nrec as u64);

    // Levels 1..=depth: internal nodes
    for u in 1..=depth as usize {
        let ptr_size = sizeof_addr as usize + max_nrec_size + info[u - 1].cum_max_nrec_size;
        // max_nrec[u] = (node_size - 10 - ptr_size) / (record_size + ptr_size)
        // The extra ptr_size in the numerator accounts for the (n+1)th child pointer.
        let int_max_nrec = (node_size as usize - 10 - ptr_size) / (record_size as usize + ptr_size);
        let cum = (int_max_nrec as u64 + 1) * info[u - 1].cum_max_nrec + int_max_nrec as u64;
        let cum_size = limit_enc_size(cum);
        info.push(NodeInfo {
            max_nrec: int_max_nrec,
            cum_max_nrec: cum,
            cum_max_nrec_size: cum_size,
        });
    }

    (max_nrec_size, info)
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

    let (max_nrec_size, node_info) = compute_node_info(
        header.node_size,
        header.record_size,
        size_of_offsets,
        header.depth,
    );

    iterate_node(
        reader,
        header,
        header.root_node_address,
        header.root_num_records,
        header.depth,
        size_of_offsets,
        max_nrec_size,
        &node_info,
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
    max_nrec_size: usize,
    node_info: &[NodeInfo],
    callback: &mut F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    if depth == 0 {
        iterate_leaf(reader, header, addr, num_records, callback)
    } else {
        iterate_internal(
            reader,
            header,
            addr,
            num_records,
            depth,
            size_of_offsets,
            max_nrec_size,
            node_info,
            callback,
        )
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
    max_nrec_size: usize,
    node_info: &[NodeInfo],
    callback: &mut F,
) -> Result<()>
where
    R: ReadAt + ?Sized,
    F: FnMut(&Record) -> Result<()>,
{
    // Internal node on-disk layout (H5B2cache.c):
    //   Signature "BTIN" (4)
    //   Version (1)
    //   Type (1)
    //   Records[0..nrec-1]  (nrec * record_size)          — all records first
    //   ChildPtrs[0..nrec]  ((nrec+1) * child_entry_size) — then all child pointers
    //   Checksum (4)
    //
    // Each child pointer entry:
    //   address        (size_of_offsets bytes)
    //   node_nrec      (max_nrec_size bytes)
    //   all_nrec       (cum_max_nrec_size bytes, only when depth > 1)
    let mut magic = [0u8; 4];
    reader.read_exact_at(addr, &mut magic).map_err(Error::Io)?;
    if magic != BTIN_MAGIC {
        return Err(Error::InvalidBTreeV2 {
            msg: format!("expected BTIN magic at {:#x}", addr),
        });
    }

    let o = size_of_offsets as usize;
    let rec_size = header.record_size as usize;
    let n = num_records as usize;

    // Child pointer entry size: addr + nrec + all_nrec (all_nrec only at depth > 1)
    let cum_size = node_info[depth as usize - 1].cum_max_nrec_size;
    let child_entry_size = o + max_nrec_size + cum_size;

    // Content = header(6) + all records + all child pointers
    let content_len = 6 + n * rec_size + (n + 1) * child_entry_size;

    // Read node data and verify checksum
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

    // Parse all records (contiguous block starting at offset 6)
    let mut pos = 6usize;
    let mut records: Vec<Record> = Vec::with_capacity(n);
    for _ in 0..n {
        let rec_data = node_data[pos..pos + rec_size].to_vec();
        records.push(Record { data: rec_data });
        pos += rec_size;
    }

    // Parse all child pointers (contiguous block after records)
    let mut children: Vec<(u64, u16)> = Vec::with_capacity(n + 1);
    for _ in 0..=n {
        let child_addr = read_offset_from_slice(&node_data, pos, size_of_offsets);
        let child_nrec = read_var_uint(&node_data, pos + o, max_nrec_size) as u16;
        children.push((child_addr, child_nrec));
        pos += child_entry_size;
    }

    // Recursively iterate in sorted order: child[0], record[0], child[1], ..., child[n]
    for i in 0..=n {
        let (child_addr, child_nrec) = children[i];
        iterate_node(
            reader,
            header,
            child_addr,
            child_nrec,
            depth - 1,
            size_of_offsets,
            max_nrec_size,
            node_info,
            callback,
        )?;
        if i < n {
            callback(&records[i])?;
        }
    }

    Ok(())
}

/// Minimum bytes needed to encode value `v`, matching H5VM_limit_enc_size.
///
/// Returns floor(log2(v)) / 8 + 1 for v > 0, or 1 for v == 0.
/// Unlike a power-of-two rounding, this gives continuous values: 1, 2, 3, 4, ...
fn limit_enc_size(v: u64) -> usize {
    if v == 0 {
        return 1;
    }
    let log2 = 63 - v.leading_zeros() as usize;
    (log2 / 8) + 1
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
    let mut result = 0u64;
    for i in 0..size.min(8) {
        result |= (data[offset + i] as u64) << (i * 8);
    }
    result
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

/// Parse a B-tree v2 "type 6" record (link creation order for new-style groups).
///
/// Layout: creation_order (8 bytes LE u64) + heap ID (heap_id_len bytes).
pub fn parse_link_creation_order_record(data: &[u8], heap_id_len: usize) -> Option<(u64, Vec<u8>)> {
    if data.len() < 8 + heap_id_len {
        return None;
    }
    let creation_order = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]);
    let heap_id = data[8..8 + heap_id_len].to_vec();
    Some((creation_order, heap_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_enc_size() {
        assert_eq!(limit_enc_size(0), 1);
        assert_eq!(limit_enc_size(1), 1);
        assert_eq!(limit_enc_size(255), 1);
        assert_eq!(limit_enc_size(256), 2);
        assert_eq!(limit_enc_size(65535), 2);
        assert_eq!(limit_enc_size(65536), 3);
        assert_eq!(limit_enc_size((1 << 24) - 1), 3);
        assert_eq!(limit_enc_size(1 << 24), 4);
    }

    #[test]
    fn test_compute_node_info() {
        // node_size=4096, record_size=24 (non-filtered 2D BT2 chunk record),
        // sizeof_addr=8, depth=1
        let (max_nrec_size, info) = compute_node_info(4096, 24, 8, 1);

        // Leaf: max_nrec = (4096 - 10) / 24 = 170
        assert_eq!(info[0].max_nrec, 170);
        assert_eq!(info[0].cum_max_nrec, 170);
        assert_eq!(info[0].cum_max_nrec_size, 0);

        // max_nrec_size = limit_enc_size(170) = 1
        assert_eq!(max_nrec_size, 1);

        // Internal (depth=1): ptr_size = 8 + 1 + 0 = 9
        // max_nrec = (4096 - 10 - 9) / (24 + 9) = 4077 / 33 = 123
        assert_eq!(info[1].max_nrec, 123);
        // cum_max_nrec = (123 + 1) * 170 + 123 = 21203
        assert_eq!(info[1].cum_max_nrec, 21203);
    }
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

/// Parse a B-tree v2 "type 9" record (attribute creation order).
///
/// Layout: heap_id (heap_id_len bytes) + flags (1 byte) + creation_order (4 bytes).
pub fn parse_attribute_creation_order_record(
    data: &[u8],
    heap_id_len: usize,
) -> Option<(u32, Vec<u8>)> {
    if data.len() < heap_id_len + 1 + 4 {
        return None;
    }
    let heap_id = data[..heap_id_len].to_vec();
    // flags at heap_id_len
    let co_off = heap_id_len + 1;
    let creation_order = u32::from_le_bytes([
        data[co_off],
        data[co_off + 1],
        data[co_off + 2],
        data[co_off + 3],
    ]);
    Some((creation_order, heap_id))
}
