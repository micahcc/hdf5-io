use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

const GCOL_MAGIC: &[u8; 4] = b"GCOL";

/// A global heap collection.
///
/// HDF5 stores variable-length data in "global heap collections" scattered
/// throughout the file. Each collection contains one or more objects indexed
/// by a 16-bit object index.
///
/// ## On-disk layout
///
/// ```text
/// Magic "GCOL" (4 bytes)
/// Version (1 byte, must be 1)
/// Reserved (3 bytes)
/// Collection size (size_of_lengths bytes)
/// Objects[]:
///   Object index (2 bytes)
///   Reference count (2 bytes)
///   Reserved (4 bytes)
///   Object size (size_of_lengths bytes)
///   Object data (size bytes, padded to 8-byte boundary)
/// ```
///
/// An object with index 0 marks free space.

/// Read a single object from a global heap collection.
///
/// `collection_addr` is the file offset of the "GCOL" signature.
/// `object_index` is the 1-based index of the desired object.
///
/// Returns the raw object data bytes.
pub fn read_global_heap_object<R: ReadAt + ?Sized>(
    reader: &R,
    collection_addr: u64,
    object_index: u32,
    size_of_lengths: u8,
) -> Result<Vec<u8>> {
    let sl = size_of_lengths as u64;

    // Read and verify magic
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(collection_addr, &mut magic)
        .map_err(Error::Io)?;
    if &magic != GCOL_MAGIC {
        return Err(Error::Other {
            msg: format!(
                "invalid global heap magic at offset {:#x}: {:?}",
                collection_addr, magic
            ),
        });
    }

    // Version
    let version = Le::read_u8(reader, collection_addr + 4).map_err(Error::Io)?;
    if version != 1 {
        return Err(Error::Other {
            msg: format!("unsupported global heap version {}", version),
        });
    }

    // Reserved (3 bytes), then collection size
    let collection_size =
        Le::read_length(reader, collection_addr + 8, size_of_lengths).map_err(Error::Io)?;

    // Walk objects starting after the header (4 + 1 + 3 + sl bytes)
    let header_size = 8 + sl;
    let collection_end = collection_addr + collection_size;
    let mut pos = collection_addr + header_size;

    // Object header: index(2) + refcount(2) + reserved(4) + size(sl) = 8 + sl
    let obj_header_size = 8 + sl;

    while pos + obj_header_size <= collection_end {
        let idx = Le::read_u16(reader, pos).map_err(Error::Io)? as u32;
        let obj_size = Le::read_length(reader, pos + 8, size_of_lengths).map_err(Error::Io)?;
        let data_start = pos + obj_header_size;

        if idx == 0 {
            // Free space entry — skip over it and continue (H5HGcache.c:352-358).
            // For idx 0, the size field includes the entry header; advance from
            // the start of this entry by obj_size bytes.
            if obj_size == 0 {
                break;
            }
            pos += obj_size;
            continue;
        }

        if idx == object_index {
            let mut data = vec![0u8; obj_size as usize];
            reader
                .read_exact_at(data_start, &mut data)
                .map_err(Error::Io)?;
            return Ok(data);
        }

        // Advance to next object (data padded to 8-byte boundary)
        let padded_size = (obj_size + 7) & !7;
        pos = data_start + padded_size;
    }

    Err(Error::Other {
        msg: format!(
            "global heap object index {} not found in collection at {:#x}",
            object_index, collection_addr
        ),
    })
}

/// Parse a variable-length dataset's global heap IDs and resolve them.
///
/// Each element in a vlen dataset is stored on disk as:
/// - sequence_length: u32 (4 bytes)
/// - collection_address: size_of_offsets bytes
/// - object_index: u32 (4 bytes)
///
/// Returns a Vec of raw byte vectors, one per element.
pub fn resolve_vlen_elements<R: ReadAt + ?Sized>(
    reader: &R,
    raw_data: &[u8],
    num_elements: usize,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<Vec<u8>>> {
    let heap_id_size = 4 + size_of_offsets as usize + 4; // seq_len + addr + index
    if raw_data.len() < num_elements * heap_id_size {
        return Err(Error::Other {
            msg: format!(
                "vlen raw data too short: {} bytes for {} elements (need {} each)",
                raw_data.len(),
                num_elements,
                heap_id_size
            ),
        });
    }

    let mut results = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let base = i * heap_id_size;
        let seq_len = u32::from_le_bytes([
            raw_data[base],
            raw_data[base + 1],
            raw_data[base + 2],
            raw_data[base + 3],
        ]);

        let addr_pos = base + 4;
        let collection_addr = match size_of_offsets {
            4 => u32::from_le_bytes([
                raw_data[addr_pos],
                raw_data[addr_pos + 1],
                raw_data[addr_pos + 2],
                raw_data[addr_pos + 3],
            ]) as u64,
            8 => u64::from_le_bytes([
                raw_data[addr_pos],
                raw_data[addr_pos + 1],
                raw_data[addr_pos + 2],
                raw_data[addr_pos + 3],
                raw_data[addr_pos + 4],
                raw_data[addr_pos + 5],
                raw_data[addr_pos + 6],
                raw_data[addr_pos + 7],
            ]),
            _ => {
                return Err(Error::Other {
                    msg: format!("unsupported size_of_offsets: {}", size_of_offsets),
                });
            }
        };

        let idx_pos = addr_pos + size_of_offsets as usize;
        let object_index = u32::from_le_bytes([
            raw_data[idx_pos],
            raw_data[idx_pos + 1],
            raw_data[idx_pos + 2],
            raw_data[idx_pos + 3],
        ]);

        if seq_len == 0 || collection_addr == u64::MAX {
            // Null / empty vlen element
            results.push(Vec::new());
        } else {
            let obj_data =
                read_global_heap_object(reader, collection_addr, object_index, size_of_lengths)?;
            results.push(obj_data);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory global heap collection with one object.
    fn build_collection(object_index: u16, object_data: &[u8]) -> Vec<u8> {
        let sl = 8usize; // size_of_lengths = 8
        let obj_header_size = 8 + sl; // index(2) + refcount(2) + reserved(4) + size(sl)
        let padded_data = (object_data.len() + 7) & !7;
        // Free space marker object header
        let free_header_size = obj_header_size;
        let collection_size = 8 + sl + obj_header_size + padded_data + free_header_size;

        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(b"GCOL");
        // Version
        buf.push(1);
        // Reserved
        buf.extend_from_slice(&[0, 0, 0]);
        // Collection size (8 bytes LE)
        buf.extend_from_slice(&(collection_size as u64).to_le_bytes());

        // Object entry
        buf.extend_from_slice(&object_index.to_le_bytes()); // index
        buf.extend_from_slice(&1u16.to_le_bytes()); // refcount
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(object_data.len() as u64).to_le_bytes()); // size
        buf.extend_from_slice(object_data);
        // Pad to 8 bytes
        let padding = padded_data - object_data.len();
        buf.extend_from_slice(&vec![0u8; padding]);

        // Free space marker (index = 0)
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&0u64.to_le_bytes());

        buf
    }

    #[test]
    fn read_single_object() {
        let data = b"hello world";
        let collection = build_collection(1, data);
        let result = read_global_heap_object(collection.as_slice(), 0, 1, 8).unwrap();
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn object_not_found() {
        let collection = build_collection(1, b"data");
        let result = read_global_heap_object(collection.as_slice(), 0, 99, 8);
        assert!(result.is_err());
    }

    #[test]
    fn bad_magic() {
        let mut collection = build_collection(1, b"data");
        collection[0] = b'X';
        let result = read_global_heap_object(collection.as_slice(), 0, 1, 8);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_vlen_two_elements() {
        // Build two collections, each with one object
        let col1 = build_collection(1, b"hello");
        let col2 = build_collection(1, b"world!");

        // Concatenate into a "file" with col1 at offset 0, col2 right after
        let col2_offset = col1.len() as u64;
        let mut file_data = col1.clone();
        file_data.extend_from_slice(&col2);

        // Build raw vlen heap IDs: seq_len(4) + addr(8) + index(4) = 16 bytes each
        let mut raw = Vec::new();
        // Element 0: seq_len=5, addr=0, index=1
        raw.extend_from_slice(&5u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes());
        // Element 1: seq_len=6, addr=col2_offset, index=1
        raw.extend_from_slice(&6u32.to_le_bytes());
        raw.extend_from_slice(&col2_offset.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes());

        let results = resolve_vlen_elements(file_data.as_slice(), &raw, 2, 8, 8).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], b"hello");
        assert_eq!(results[1], b"world!");
    }

    /// Build a collection with a free-space entry (index 0) before a real object.
    #[test]
    fn skip_free_space_entry() {
        let sl = 8usize;
        let obj_header_size = 8 + sl;

        // Object: index=2, data = "found"
        let obj_data = b"found!!!"; // 8 bytes, already aligned
        let obj_padded = 8usize;

        // Free space entry: index=0, size = total size of the free space region
        // (includes the free space entry header itself)
        let free_total_size = obj_header_size + 16; // header + 16 bytes of dead space

        let collection_size = (8 + sl) + free_total_size + obj_header_size + obj_padded;

        let mut buf = Vec::new();
        // Collection header
        buf.extend_from_slice(b"GCOL");
        buf.push(1);
        buf.extend_from_slice(&[0, 0, 0]);
        buf.extend_from_slice(&(collection_size as u64).to_le_bytes());

        // Free space entry (index 0)
        buf.extend_from_slice(&0u16.to_le_bytes()); // index = 0
        buf.extend_from_slice(&0u16.to_le_bytes()); // refcount
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(free_total_size as u64).to_le_bytes()); // size (includes header)
        // Dead space within the free entry
        buf.extend_from_slice(&[0u8; 16]);

        // Real object (index 2)
        buf.extend_from_slice(&2u16.to_le_bytes()); // index = 2
        buf.extend_from_slice(&1u16.to_le_bytes()); // refcount
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(obj_data.len() as u64).to_le_bytes()); // size
        buf.extend_from_slice(obj_data);

        let result = read_global_heap_object(buf.as_slice(), 0, 2, 8).unwrap();
        assert_eq!(result, b"found!!!");
    }

    #[test]
    fn resolve_empty_vlen_element() {
        let mut raw = Vec::new();
        // seq_len=0, addr=0, index=0
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());

        let file_data: Vec<u8> = Vec::new();
        let results = resolve_vlen_elements(file_data.as_slice(), &raw, 1, 8, 8).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }
}
