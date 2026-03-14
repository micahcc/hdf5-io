use crate::checksum;
use crate::error::{Error, Result};
use crate::filters::FilterPipeline;
use crate::io::{Le, ReadAt};
use crate::layout::{ChunkIndexType, DataLayout};

/// A single chunk's location and metadata.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// File address of the (possibly filtered) chunk data.
    pub address: u64,
    /// On-disk size in bytes (after filtering/compression).
    pub filtered_size: u64,
    /// Filter mask — bit N set means filter N was NOT applied.
    pub filter_mask: u32,
    /// Scaled chunk coordinates (chunk_offset / chunk_dim for each axis).
    pub scaled: Vec<u64>,
}

/// Read a chunked dataset, assembling all chunks into a contiguous buffer.
///
/// `chunk_dims`: the chunk dimensions (element counts per axis).
/// `dataset_dims`: the dataset dimensions.
/// `element_size`: size of one element in bytes.
/// `layout`: the parsed DataLayout::Chunked.
/// `filters`: optional filter pipeline.
/// `max_dims`: optional max dimensions from dataspace (needed for B-tree v2 chunk index).
pub fn read_chunked<R: ReadAt + ?Sized>(
    reader: &R,
    layout: &DataLayout,
    dataset_dims: &[u64],
    element_size: u32,
    filters: &Option<FilterPipeline>,
    size_of_offsets: u8,
    size_of_lengths: u8,
    max_dims: Option<&[u64]>,
) -> Result<Vec<u8>> {
    let (chunk_dims, address, chunk_index_type, _chunk_flags, layout_version,
         single_filtered_size, single_filter_mask) = match layout {
        DataLayout::Chunked {
            chunk_dims,
            address,
            chunk_index_type,
            chunk_flags,
            layout_version,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
            ..
        } => (
            chunk_dims,
            *address,
            chunk_index_type,
            *chunk_flags,
            *layout_version,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(Error::InvalidLayout {
                msg: "expected chunked layout".into(),
            })
        }
    };

    let ndims = dataset_dims.len();

    // Both layout v3 and v4 store dimensionality = rank + 1 for chunked
    // datasets, with the last dimension being the element size in bytes.
    // Strip the extra dimension so we work with just the dataset-rank dimensions.
    let actual_chunk_dims: Vec<u32>;
    let chunk_dims = if chunk_dims.len() == ndims + 1 {
        actual_chunk_dims = chunk_dims[..ndims].to_vec();
        &actual_chunk_dims
    } else {
        chunk_dims
    };

    // Compute the uncompressed chunk size in bytes
    let chunk_elems: u64 = chunk_dims.iter().map(|&d| d as u64).product();
    let chunk_byte_size = chunk_elems * element_size as u64;

    // Compute total output size
    let total_elems: u64 = dataset_dims.iter().product();
    let total_size = total_elems * element_size as u64;
    let mut output = vec![0u8; total_size as usize];

    // If no data has been allocated, return the zeroed output buffer
    if address == u64::MAX {
        return Ok(output);
    }

    // Determine chunk index type and collect entries
    let entries = if layout_version == 3 {
        // Layout v3 always uses B-tree v1
        let v3_ndims = ndims + 1; // dimensionality in v3 = rank + 1
        read_btree_v1_entries(
            reader,
            address,
            dataset_dims,
            chunk_dims,
            v3_ndims,
            size_of_offsets,
        )?
    } else {
        let idx_type = chunk_index_type.unwrap_or(ChunkIndexType::SingleChunk);
        match idx_type {
        ChunkIndexType::SingleChunk => {
            read_single_chunk_entries(
                address,
                chunk_byte_size,
                single_filtered_size,
                single_filter_mask,
                ndims,
            )?
        }
        ChunkIndexType::Implicit => {
            read_implicit_chunk_entries(
                address,
                dataset_dims,
                chunk_dims,
                chunk_byte_size,
            )?
        }
        ChunkIndexType::FixedArray => {
            read_fixed_array_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
            )?
        }
        ChunkIndexType::ExtensibleArray => {
            read_extensible_array_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
            )?
        }
        ChunkIndexType::BTreeV2 => {
            read_btree_v2_chunk_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                element_size,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
                max_dims,
            )?
        }
    }
    };

    // Read each chunk and place it into the output buffer
    for entry in &entries {
        if entry.address == u64::MAX {
            // Chunk not allocated — leave as zeros (fill value)
            continue;
        }

        // Read the raw (possibly compressed) chunk data.
        // For unfiltered entries, filtered_size may be 0 — use chunk_byte_size instead.
        let read_size = if entry.filtered_size > 0 {
            entry.filtered_size as usize
        } else {
            chunk_byte_size as usize
        };
        let mut chunk_data = vec![0u8; read_size];
        reader
            .read_exact_at(entry.address, &mut chunk_data)
            .map_err(Error::Io)?;

        // Apply filter pipeline in reverse (decompress)
        if let Some(pipeline) = filters {
            if entry.filter_mask == 0 {
                // All filters applied
                chunk_data = pipeline.decompress(chunk_data)?;
            }
            // TODO: handle partial filter masks
        }

        // Place chunk data into the output buffer at the correct position
        copy_chunk_to_output(
            &chunk_data,
            &entry.scaled,
            chunk_dims,
            dataset_dims,
            element_size as usize,
            &mut output,
        );
    }

    Ok(output)
}

/// Read a hyperslab from a chunked dataset, only reading overlapping chunks.
pub fn read_chunked_slice<R: ReadAt + ?Sized>(
    reader: &R,
    layout: &DataLayout,
    dataset_dims: &[u64],
    element_size: u32,
    filters: &Option<FilterPipeline>,
    size_of_offsets: u8,
    size_of_lengths: u8,
    max_dims: Option<&[u64]>,
    start: &[u64],
    count: &[u64],
) -> Result<Vec<u8>> {
    let (chunk_dims, address, chunk_index_type, _chunk_flags, layout_version,
         single_filtered_size, single_filter_mask) = match layout {
        DataLayout::Chunked {
            chunk_dims, address, chunk_index_type, chunk_flags,
            layout_version, single_chunk_filtered_size, single_chunk_filter_mask, ..
        } => (chunk_dims, *address, chunk_index_type, *chunk_flags, *layout_version,
              *single_chunk_filtered_size, *single_chunk_filter_mask),
        _ => return Err(Error::InvalidLayout { msg: "expected chunked layout".into() }),
    };

    let ndims = dataset_dims.len();
    let actual_chunk_dims: Vec<u32>;
    let chunk_dims = if chunk_dims.len() == ndims + 1 {
        actual_chunk_dims = chunk_dims[..ndims].to_vec();
        &actual_chunk_dims
    } else {
        chunk_dims
    };

    let chunk_elems: u64 = chunk_dims.iter().map(|&d| d as u64).product();
    let chunk_byte_size = chunk_elems * element_size as u64;
    let elem = element_size as usize;

    // Output buffer for the selection
    let out_elems: u64 = count.iter().product();
    let mut output = vec![0u8; (out_elems * element_size as u64) as usize];

    // Compute output strides (row-major, in bytes)
    let mut out_strides = vec![elem; ndims];
    for i in (0..ndims - 1).rev() {
        out_strides[i] = out_strides[i + 1] * count[i + 1] as usize;
    }

    // If no data has been allocated, return the zeroed output buffer
    if address == u64::MAX {
        return Ok(output);
    }

    // Collect chunk entries (reuse the same logic as read_chunked)
    let entries = if layout_version == 3 {
        let v3_ndims = ndims + 1;
        read_btree_v1_entries(reader, address, dataset_dims, chunk_dims, v3_ndims, size_of_offsets)?
    } else {
        let idx_type = chunk_index_type.unwrap_or(ChunkIndexType::SingleChunk);
        match idx_type {
            ChunkIndexType::SingleChunk => read_single_chunk_entries(address, chunk_byte_size, single_filtered_size, single_filter_mask, ndims)?,
            ChunkIndexType::Implicit => read_implicit_chunk_entries(address, dataset_dims, chunk_dims, chunk_byte_size)?,
            ChunkIndexType::FixedArray => read_fixed_array_entries(reader, address, dataset_dims, chunk_dims, filters.is_some(), size_of_offsets, size_of_lengths)?,
            ChunkIndexType::ExtensibleArray => read_extensible_array_entries(reader, address, dataset_dims, chunk_dims, filters.is_some(), size_of_offsets, size_of_lengths)?,
            ChunkIndexType::BTreeV2 => read_btree_v2_chunk_entries(reader, address, dataset_dims, chunk_dims, element_size, filters.is_some(), size_of_offsets, size_of_lengths, max_dims)?,
        }
    };

    // For each chunk, check if it overlaps the selection
    for entry in &entries {
        if entry.address == u64::MAX {
            continue;
        }

        // Chunk covers [chunk_start[i], chunk_end[i]) in each dimension
        let mut overlaps = true;
        for i in 0..ndims {
            let cs = entry.scaled[i] * chunk_dims[i] as u64;
            let ce = (cs + chunk_dims[i] as u64).min(dataset_dims[i]);
            let ss = start[i];
            let se = start[i] + count[i];
            if cs >= se || ce <= ss {
                overlaps = false;
                break;
            }
        }
        if !overlaps {
            continue;
        }

        // Read and decompress chunk
        let read_size = if entry.filtered_size > 0 { entry.filtered_size as usize } else { chunk_byte_size as usize };
        let mut chunk_data = vec![0u8; read_size];
        reader.read_exact_at(entry.address, &mut chunk_data).map_err(Error::Io)?;
        if let Some(pipeline) = filters {
            if entry.filter_mask == 0 {
                chunk_data = pipeline.decompress(chunk_data)?;
            }
        }

        // Compute chunk strides
        let mut ch_strides = vec![elem; ndims];
        for i in (0..ndims - 1).rev() {
            ch_strides[i] = ch_strides[i + 1] * chunk_dims[i + 1] as usize;
        }

        // Compute overlap region in dataset coordinates
        let mut ov_start = vec![0u64; ndims];
        let mut ov_count = vec![0u64; ndims];
        for i in 0..ndims {
            let cs = entry.scaled[i] * chunk_dims[i] as u64;
            let ce = (cs + chunk_dims[i] as u64).min(dataset_dims[i]);
            ov_start[i] = start[i].max(cs);
            let ov_end = (start[i] + count[i]).min(ce);
            ov_count[i] = ov_end - ov_start[i];
        }

        // Copy the overlap region from chunk to output, row by row
        let inner_count = ov_count[ndims - 1] as usize * elem;
        let nrows: usize = ov_count[..ndims - 1].iter().map(|&c| c as usize).product::<usize>().max(1);

        for row in 0..nrows {
            let mut remaining = row;
            let mut src_off = 0usize;
            let mut dst_off = 0usize;

            for i in 0..ndims - 1 {
                let rows_below: usize = ov_count[i + 1..ndims - 1]
                    .iter().map(|&c| c as usize).product::<usize>().max(1);
                let idx = remaining / rows_below;
                remaining %= rows_below;
                // Source offset within chunk
                src_off += (ov_start[i] - entry.scaled[i] * chunk_dims[i] as u64) as usize * ch_strides[i]
                    + idx * ch_strides[i];
                // Dest offset within output
                dst_off += (ov_start[i] - start[i]) as usize * out_strides[i]
                    + idx * out_strides[i];
            }
            src_off += (ov_start[ndims - 1] - entry.scaled[ndims - 1] * chunk_dims[ndims - 1] as u64) as usize * elem;
            dst_off += (ov_start[ndims - 1] - start[ndims - 1]) as usize * elem;

            if src_off + inner_count <= chunk_data.len() && dst_off + inner_count <= output.len() {
                output[dst_off..dst_off + inner_count]
                    .copy_from_slice(&chunk_data[src_off..src_off + inner_count]);
            }
        }
    }

    Ok(output)
}

/// Single chunk: there's exactly one chunk covering the entire dataset.
fn read_single_chunk_entries(
    address: u64,
    chunk_byte_size: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    ndims: usize,
) -> Result<Vec<ChunkEntry>> {
    Ok(vec![ChunkEntry {
        address,
        filtered_size: filtered_size.unwrap_or(chunk_byte_size),
        filter_mask: filter_mask.unwrap_or(0),
        scaled: vec![0; ndims],
    }])
}

/// Implicit index: chunks are stored contiguously, no index structure.
/// Chunk at scaled coordinates is at: address + linear_index * chunk_byte_size.
fn read_implicit_chunk_entries(
    address: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    chunk_byte_size: u64,
) -> Result<Vec<ChunkEntry>> {
    let ndims = dataset_dims.len();
    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| (dataset_dims[i] + chunk_dims[i] as u64 - 1) / chunk_dims[i] as u64)
        .collect();

    let total_chunks: u64 = chunks_per_dim.iter().product();
    let mut entries = Vec::with_capacity(total_chunks as usize);

    for linear in 0..total_chunks {
        let scaled = linear_to_scaled(linear, &chunks_per_dim);
        entries.push(ChunkEntry {
            address: address + linear * chunk_byte_size,
            filtered_size: chunk_byte_size,
            filter_mask: 0,
            scaled,
        });
    }

    Ok(entries)
}

/// Fixed Array header magic: `FAHD`
const FAHD_MAGIC: [u8; 4] = *b"FAHD";
/// Fixed Array data block magic: `FADB`
const FADB_MAGIC: [u8; 4] = *b"FADB";

/// Read chunk entries from a Fixed Array index.
///
/// Fixed Array layout:
/// ```text
/// Header ("FAHD"):
///   magic(4) + version(1) + client_id(1) + entry_size(1) + max_dblk_page_nelmts_bits(1)
///   + nelmts(sizeof_lengths) + data_block_addr(sizeof_offsets) + checksum(4)
///
/// Data Block ("FADB"):
///   magic(4) + version(1) + client_id(1) + header_addr(sizeof_offsets)
///   + [optional page info if paged]
///   + entries(nelmts * entry_size)
///   + checksum(4)
/// ```
fn read_fixed_array_entries<R: ReadAt + ?Sized>(
    reader: &R,
    header_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    has_filters: bool,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<ChunkEntry>> {
    let o = size_of_offsets as u64;
    let l = size_of_lengths as u64;

    // Parse Fixed Array header
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(header_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != FAHD_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected FAHD magic at {:#x}, got {:?}",
                header_addr, magic
            ),
        });
    }

    let _version = Le::read_u8(reader, header_addr + 4).map_err(Error::Io)?;
    let _client_id = Le::read_u8(reader, header_addr + 5).map_err(Error::Io)?;
    let entry_size = Le::read_u8(reader, header_addr + 6).map_err(Error::Io)?;
    let _max_dblk_page_bits = Le::read_u8(reader, header_addr + 7).map_err(Error::Io)?;
    let nelmts =
        Le::read_length(reader, header_addr + 8, size_of_lengths).map_err(Error::Io)?;
    let dblk_addr =
        Le::read_offset(reader, header_addr + 8 + l, size_of_offsets).map_err(Error::Io)?;

    // Verify header checksum
    let hdr_size = (8 + l + o) as usize;
    let mut hdr_data = vec![0u8; hdr_size];
    reader
        .read_exact_at(header_addr, &mut hdr_data)
        .map_err(Error::Io)?;
    let stored_cksum =
        Le::read_u32(reader, header_addr + hdr_size as u64).map_err(Error::Io)?;
    let computed = checksum::lookup3(&hdr_data);
    if computed != stored_cksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_cksum,
            actual: computed,
        });
    }

    if dblk_addr == u64::MAX {
        // No data block — all chunks are unallocated
        return Ok(Vec::new());
    }

    // Parse Fixed Array data block
    let mut db_magic = [0u8; 4];
    reader
        .read_exact_at(dblk_addr, &mut db_magic)
        .map_err(Error::Io)?;
    if db_magic != FADB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected FADB magic at {:#x}, got {:?}",
                dblk_addr, db_magic
            ),
        });
    }

    // Data block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O)
    let db_prefix = 4 + 1 + 1 + o as usize;

    // TODO: handle paged data blocks (when max_dblk_page_bits > 0 and nelmts is large)
    // For now, assume non-paged (all entries in one contiguous block)

    // Read all entries
    let entries_start = dblk_addr + db_prefix as u64;
    let es = entry_size as usize;
    let ndims = dataset_dims.len();

    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| (dataset_dims[i] + chunk_dims[i] as u64 - 1) / chunk_dims[i] as u64)
        .collect();

    let mut entries = Vec::with_capacity(nelmts as usize);

    for i in 0..nelmts {
        let entry_off = entries_start + i * es as u64;

        if has_filters {
            // Filtered entry: address(O) + chunk_nbytes(entry_size - O - 4) + filter_mask(4)
            let addr =
                Le::read_offset(reader, entry_off, size_of_offsets).map_err(Error::Io)?;
            let nbytes_size = es - size_of_offsets as usize - 4;
            let nbytes = read_var_le(reader, entry_off + o, nbytes_size)?;
            let fmask = Le::read_u32(reader, entry_off + o + nbytes_size as u64)
                .map_err(Error::Io)?;

            let scaled = linear_to_scaled(i, &chunks_per_dim);
            entries.push(ChunkEntry {
                address: addr,
                filtered_size: nbytes,
                filter_mask: fmask,
                scaled,
            });
        } else {
            // Non-filtered entry: just address(O)
            let addr =
                Le::read_offset(reader, entry_off, size_of_offsets).map_err(Error::Io)?;
            let _chunk_byte_size: u64 =
                chunk_dims.iter().map(|&d| d as u64).product::<u64>()
                    * (es as u64 / size_of_offsets as u64).max(1); // approximate

            let scaled = linear_to_scaled(i, &chunks_per_dim);
            // For non-filtered, the on-disk size equals the uncompressed chunk size.
            // We don't know element_size here, but the caller will figure it out
            // from the entry_size vs offset_size.
            entries.push(ChunkEntry {
                address: addr,
                filtered_size: 0, // will be set by caller
                filter_mask: 0,
                scaled,
            });
        }
    }

    Ok(entries)
}

/// Extensible Array header magic: `EAHD`
const EAHD_MAGIC: [u8; 4] = *b"EAHD";
/// Extensible Array index block magic: `EAIB`
const EAIB_MAGIC: [u8; 4] = *b"EAIB";
/// Extensible Array super block magic: `EASB`
const EASB_MAGIC: [u8; 4] = *b"EASB";
/// Extensible Array data block magic: `EADB`
const EADB_MAGIC: [u8; 4] = *b"EADB";

/// Read chunk entries from an Extensible Array index.
///
/// The extensible array stores chunk addresses in a tiered structure:
/// 1. First `idx_blk_elmts` elements are stored directly in the index block
/// 2. Additional elements in data blocks addressed from the index block
/// 3. Further elements in data blocks addressed via super blocks
fn read_extensible_array_entries<R: ReadAt + ?Sized>(
    reader: &R,
    header_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    has_filters: bool,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<ChunkEntry>> {
    let o = size_of_offsets as u64;
    let l = size_of_lengths as u64;

    // Parse EA header
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(header_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != EAHD_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected EAHD magic at {:#x}, got {:?}",
                header_addr, magic
            ),
        });
    }

    let _version = Le::read_u8(reader, header_addr + 4).map_err(Error::Io)?;
    let _client_id = Le::read_u8(reader, header_addr + 5).map_err(Error::Io)?;
    let elmt_size = Le::read_u8(reader, header_addr + 6).map_err(Error::Io)? as usize;
    let max_nelmts_bits = Le::read_u8(reader, header_addr + 7).map_err(Error::Io)?;
    let idx_blk_elmts = Le::read_u8(reader, header_addr + 8).map_err(Error::Io)? as u64;
    let data_blk_min_elmts = Le::read_u8(reader, header_addr + 9).map_err(Error::Io)? as u64;
    let sup_blk_min_data_ptrs = Le::read_u8(reader, header_addr + 10).map_err(Error::Io)? as u64;
    let max_dblk_page_nelmts_bits = Le::read_u8(reader, header_addr + 11).map_err(Error::Io)?;

    // 6 stats (each sizeof_lengths)
    let stats_start = header_addr + 12;
    let _nsuper_blks =
        Le::read_length(reader, stats_start, size_of_lengths).map_err(Error::Io)?;
    let _super_blk_size =
        Le::read_length(reader, stats_start + l, size_of_lengths).map_err(Error::Io)?;
    let _ndata_blks =
        Le::read_length(reader, stats_start + 2 * l, size_of_lengths).map_err(Error::Io)?;
    let _data_blk_size =
        Le::read_length(reader, stats_start + 3 * l, size_of_lengths).map_err(Error::Io)?;
    let max_idx_set =
        Le::read_length(reader, stats_start + 4 * l, size_of_lengths).map_err(Error::Io)?;

    let idx_blk_addr =
        Le::read_offset(reader, stats_start + 6 * l, size_of_offsets).map_err(Error::Io)?;

    if idx_blk_addr == u64::MAX || max_idx_set == 0 {
        return Ok(Vec::new());
    }

    // Parse index block
    let mut ib_magic = [0u8; 4];
    reader
        .read_exact_at(idx_blk_addr, &mut ib_magic)
        .map_err(Error::Io)?;
    if ib_magic != EAIB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected EAIB magic at {:#x}, got {:?}",
                idx_blk_addr, ib_magic
            ),
        });
    }

    // Index block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O)
    let ib_prefix = 4 + 1 + 1 + o as usize;
    let elmts_start = idx_blk_addr + ib_prefix as u64;

    let ndims = dataset_dims.len();
    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| (dataset_dims[i] + chunk_dims[i] as u64 - 1) / chunk_dims[i] as u64)
        .collect();

    let mut entries = Vec::new();
    let es = elmt_size;

    // Read elements directly stored in the index block
    let direct_count = idx_blk_elmts.min(max_idx_set);
    for i in 0..direct_count {
        let entry = read_ea_element(
            reader,
            elmts_start + i * es as u64,
            i,
            has_filters,
            size_of_offsets,
            es,
            &chunks_per_dim,
        )?;
        entries.push(entry);
    }

    if max_idx_set <= idx_blk_elmts {
        return Ok(entries);
    }

    // Data block addresses follow the direct elements in the index block
    let dblk_addrs_start = elmts_start + idx_blk_elmts * es as u64;
    let ndblk_addrs = 2 * (sup_blk_min_data_ptrs - 1);

    // Data blocks directly addressed from index block each have data_blk_min_elmts elements
    let mut global_idx = idx_blk_elmts;
    let mut dblk_nelmts = data_blk_min_elmts;

    // The first sup_blk_min_data_ptrs data blocks have data_blk_min_elmts elements each,
    // the next sup_blk_min_data_ptrs have 2*data_blk_min_elmts, etc.
    // But only ndblk_addrs total are in the index block.
    for d in 0..ndblk_addrs {
        if global_idx >= max_idx_set {
            break;
        }
        // Data block size doubles every sup_blk_min_data_ptrs blocks
        if d > 0 && d == sup_blk_min_data_ptrs {
            dblk_nelmts *= 2;
        }

        let dblk_addr_off = dblk_addrs_start + d * o;
        let dblk_addr =
            Le::read_offset(reader, dblk_addr_off, size_of_offsets).map_err(Error::Io)?;

        if dblk_addr == u64::MAX {
            // Data block not allocated — skip these entries
            global_idx += dblk_nelmts;
            continue;
        }

        read_ea_data_block_entries(
            reader,
            dblk_addr,
            dblk_nelmts,
            global_idx,
            max_idx_set,
            has_filters,
            size_of_offsets,
            es,
            &chunks_per_dim,
            &mut entries,
            max_nelmts_bits,
            max_dblk_page_nelmts_bits,
        )?;
        global_idx += dblk_nelmts;
    }

    if global_idx >= max_idx_set {
        return Ok(entries);
    }

    // Super block addresses follow the data block addresses
    let sblk_addrs_start = dblk_addrs_start + ndblk_addrs * o;
    let mut sblk_idx = 0u64;

    while global_idx < max_idx_set {
        let sblk_addr_off = sblk_addrs_start + sblk_idx * o;
        let sblk_addr =
            Le::read_offset(reader, sblk_addr_off, size_of_offsets).map_err(Error::Io)?;

        if sblk_addr == u64::MAX {
            break;
        }

        // Parse super block
        let mut sb_magic = [0u8; 4];
        reader
            .read_exact_at(sblk_addr, &mut sb_magic)
            .map_err(Error::Io)?;
        if sb_magic != EASB_MAGIC {
            return Err(Error::InvalidLayout {
                msg: format!(
                    "expected EASB magic at {:#x}, got {:?}",
                    sblk_addr, sb_magic
                ),
            });
        }

        // Super block: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_off(varies)
        let arr_off_size = ((max_nelmts_bits as u64 + 7) / 8).max(1);
        let sb_prefix = 4 + 1 + 1 + o + arr_off_size;

        // Number of data blocks in this super block: doubles with each super block pair
        let sblk_ndblks = sup_blk_min_data_ptrs * (1u64 << (sblk_idx / 2));
        // Elements per data block in this super block
        let sblk_dblk_nelmts = data_blk_min_elmts * (1u64 << ((sblk_idx + 1) / 2));

        let sb_dblk_addrs_start = sblk_addr + sb_prefix;

        for d in 0..sblk_ndblks {
            if global_idx >= max_idx_set {
                break;
            }
            let db_addr = Le::read_offset(
                reader,
                sb_dblk_addrs_start + d * o,
                size_of_offsets,
            )
            .map_err(Error::Io)?;

            if db_addr == u64::MAX {
                global_idx += sblk_dblk_nelmts;
                continue;
            }

            read_ea_data_block_entries(
                reader,
                db_addr,
                sblk_dblk_nelmts,
                global_idx,
                max_idx_set,
                has_filters,
                size_of_offsets,
                es,
                &chunks_per_dim,
                &mut entries,
                max_nelmts_bits,
                max_dblk_page_nelmts_bits,
            )?;
            global_idx += sblk_dblk_nelmts;
        }

        sblk_idx += 1;
    }

    Ok(entries)
}

/// Read a single element record from an extensible/fixed array.
fn read_ea_element<R: ReadAt + ?Sized>(
    reader: &R,
    offset: u64,
    idx: u64,
    has_filters: bool,
    size_of_offsets: u8,
    elmt_size: usize,
    chunks_per_dim: &[u64],
) -> Result<ChunkEntry> {
    let o = size_of_offsets as u64;
    let addr = Le::read_offset(reader, offset, size_of_offsets).map_err(Error::Io)?;

    let (filtered_size, filter_mask) = if has_filters {
        let nbytes_size = elmt_size - size_of_offsets as usize - 4;
        let nbytes = read_var_le(reader, offset + o, nbytes_size)?;
        let fmask =
            Le::read_u32(reader, offset + o + nbytes_size as u64).map_err(Error::Io)?;
        (nbytes, fmask)
    } else {
        (0, 0) // unfiltered: size determined by caller
    };

    let scaled = linear_to_scaled(idx, chunks_per_dim);
    Ok(ChunkEntry {
        address: addr,
        filtered_size,
        filter_mask,
        scaled,
    })
}

/// Read entries from an EA data block.
fn read_ea_data_block_entries<R: ReadAt + ?Sized>(
    reader: &R,
    dblk_addr: u64,
    dblk_nelmts: u64,
    global_start_idx: u64,
    max_idx_set: u64,
    has_filters: bool,
    size_of_offsets: u8,
    elmt_size: usize,
    chunks_per_dim: &[u64],
    entries: &mut Vec<ChunkEntry>,
    max_nelmts_bits: u8,
    max_dblk_page_nelmts_bits: u8,
) -> Result<()> {
    let o = size_of_offsets as u64;

    let mut db_magic = [0u8; 4];
    reader
        .read_exact_at(dblk_addr, &mut db_magic)
        .map_err(Error::Io)?;
    if db_magic != EADB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected EADB magic at {:#x}, got {:?}",
                dblk_addr, db_magic
            ),
        });
    }

    // Data block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_off(varies)
    let arr_off_size = ((max_nelmts_bits as u64 + 7) / 8).max(1);
    let prefix_size = 4 + 1 + 1 + o + arr_off_size;
    let elmts_start = dblk_addr + prefix_size;

    // Paging: when max_dblk_page_nelmts_bits > 0 and the data block has more
    // elements than one page, the block is divided into pages with a 4-byte
    // checksum after each page's elements.
    let page_nelmts = if max_dblk_page_nelmts_bits > 0
        && dblk_nelmts > (1u64 << max_dblk_page_nelmts_bits)
    {
        1u64 << max_dblk_page_nelmts_bits
    } else {
        0 // no paging
    };

    let count = dblk_nelmts.min(max_idx_set.saturating_sub(global_start_idx));
    for i in 0..count {
        let offset = if page_nelmts > 0 {
            let page_idx = i / page_nelmts;
            let idx_in_page = i % page_nelmts;
            let page_data_bytes = page_nelmts * elmt_size as u64;
            // Each page: elements + 4-byte checksum
            elmts_start + page_idx * (page_data_bytes + 4) + idx_in_page * elmt_size as u64
        } else {
            elmts_start + i * elmt_size as u64
        };

        let entry = read_ea_element(
            reader,
            offset,
            global_start_idx + i,
            has_filters,
            size_of_offsets,
            elmt_size,
            chunks_per_dim,
        )?;
        entries.push(entry);
    }

    Ok(())
}

/// B-tree v1 signature: `TREE`
const TREE_MAGIC: [u8; 4] = *b"TREE";

/// Read chunk entries from a B-tree v1 index (layout version 3).
///
/// B-tree v1 node layout:
/// ```text
/// Signature: "TREE" (4 bytes)
/// Node type: 1 byte (1 = chunked raw data)
/// Node level: 1 byte (0 = leaf, >0 = internal)
/// Entries used: 2 bytes (u16 LE)
/// Left sibling: sizeof_offsets bytes
/// Right sibling: sizeof_offsets bytes
/// Then interleaved: Key[0] Child[0] Key[1] Child[1] ... Key[N-1] Child[N-1] Key[N]
///
/// Each key (type 1):
///   chunk_size: 4 bytes (u32) — on-disk (filtered) size
///   filter_mask: 4 bytes (u32)
///   offset[v3_ndims]: v3_ndims * sizeof_offsets bytes — element-coordinate offsets
/// Each child pointer: sizeof_offsets bytes
/// ```
fn read_btree_v1_entries<R: ReadAt + ?Sized>(
    reader: &R,
    node_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    v3_ndims: usize,
    size_of_offsets: u8,
) -> Result<Vec<ChunkEntry>> {
    let o = size_of_offsets as usize;

    // Read and validate signature
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(node_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != TREE_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected TREE magic at {:#x}, got {:?}",
                node_addr, magic
            ),
        });
    }

    let node_type = Le::read_u8(reader, node_addr + 4).map_err(Error::Io)?;
    if node_type != 1 {
        return Err(Error::InvalidLayout {
            msg: format!("expected B-tree v1 node type 1 (chunked), got {}", node_type),
        });
    }
    let node_level = Le::read_u8(reader, node_addr + 5).map_err(Error::Io)?;
    let entries_used = Le::read_u16(reader, node_addr + 6).map_err(Error::Io)? as usize;

    // Skip sibling pointers
    let keys_start = node_addr + 8 + 2 * o as u64;

    // Key size: chunk_size(4) + filter_mask(4) + offsets(v3_ndims * o)
    let key_size = 4 + 4 + v3_ndims * o;
    // Stride between consecutive keys: key_size + child_pointer(o)
    let entry_stride = key_size + o;

    let rank = dataset_dims.len();

    if node_level == 0 {
        // Leaf node: child pointers are chunk data addresses
        let mut entries = Vec::with_capacity(entries_used);
        for i in 0..entries_used {
            let key_off = keys_start + (i as u64) * entry_stride as u64;
            let child_off = key_off + key_size as u64;

            let chunk_size = Le::read_u32(reader, key_off).map_err(Error::Io)?;
            let filter_mask = Le::read_u32(reader, key_off + 4).map_err(Error::Io)?;

            // Read the chunk offsets (element coordinates)
            let mut offsets = Vec::with_capacity(rank);
            for d in 0..rank {
                let off = Le::read_offset(reader, key_off + 8 + (d * o) as u64, size_of_offsets)
                    .map_err(Error::Io)?;
                offsets.push(off);
            }

            let chunk_addr =
                Le::read_offset(reader, child_off, size_of_offsets).map_err(Error::Io)?;

            // Convert element-coordinate offsets to scaled chunk coordinates
            let scaled: Vec<u64> = (0..rank)
                .map(|d| offsets[d] / chunk_dims[d] as u64)
                .collect();

            entries.push(ChunkEntry {
                address: chunk_addr,
                filtered_size: chunk_size as u64,
                filter_mask,
                scaled,
            });
        }
        Ok(entries)
    } else {
        // Internal node: child pointers are addresses of child B-tree nodes
        let mut entries = Vec::new();
        for i in 0..entries_used {
            let key_off = keys_start + (i as u64) * entry_stride as u64;
            let child_off = key_off + key_size as u64;

            let child_addr =
                Le::read_offset(reader, child_off, size_of_offsets).map_err(Error::Io)?;

            if child_addr != u64::MAX {
                let child_entries = read_btree_v1_entries(
                    reader,
                    child_addr,
                    dataset_dims,
                    chunk_dims,
                    v3_ndims,
                    size_of_offsets,
                )?;
                entries.extend(child_entries);
            }
        }
        Ok(entries)
    }
}

/// Read chunk entries from a B-tree v2 chunk index (layout v4, index type 5).
///
/// B-tree v2 type 10 records (non-filtered): scaled_offsets[ndims] + address
/// B-tree v2 type 11 records (filtered): scaled_offsets[ndims] + address + nbytes + filter_mask
///
/// The scaled offsets use variable-width encoding: for each dimension, the minimum
/// bytes needed to represent the maximum number of chunks in that dimension.
/// Unlimited dimensions use 8 bytes.
fn read_btree_v2_chunk_entries<R: ReadAt + ?Sized>(
    reader: &R,
    bt2_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    element_size: u32,
    has_filters: bool,
    size_of_offsets: u8,
    size_of_lengths: u8,
    max_dims: Option<&[u64]>,
) -> Result<Vec<ChunkEntry>> {
    use crate::btree2::{self, BTree2Header};

    let ndims = dataset_dims.len();

    // Compute the encoding size for each dimension's scaled offset
    let dim_encode_sizes: Vec<usize> = (0..ndims)
        .map(|i| {
            let max_dim = max_dims
                .and_then(|md| md.get(i).copied())
                .unwrap_or(dataset_dims[i]);
            if max_dim == u64::MAX {
                // Unlimited dimension
                8
            } else {
                let max_chunks = (max_dim + chunk_dims[i] as u64 - 1) / chunk_dims[i] as u64;
                bytes_needed_for(max_chunks)
            }
        })
        .collect();

    // Compute chunk_size_len for filtered records
    let chunk_byte_size: u64 = chunk_dims.iter().map(|&d| d as u64).product::<u64>()
        * element_size as u64;
    let chunk_size_len = if has_filters {
        bytes_needed_for(chunk_byte_size)
    } else {
        0
    };

    // Parse the B-tree v2 header
    let bt2 = BTree2Header::parse(reader, bt2_addr, size_of_offsets, size_of_lengths)?;

    let o = size_of_offsets as usize;
    let mut entries = Vec::new();

    btree2::iterate_records(reader, &bt2, size_of_offsets, |record| {
        let data = &record.data;
        let mut pos = 0;

        // Read chunk address first (HDF5 stores address before scaled offsets)
        let address = read_var_uint_slice(data, pos, o);
        pos += o;

        // Read scaled offsets
        let mut scaled = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let enc_size = dim_encode_sizes[i];
            let val = read_var_uint_slice(data, pos, enc_size);
            scaled.push(val);
            pos += enc_size;
        }

        let (filtered_size, filter_mask) = if has_filters {
            let nbytes = read_var_uint_slice(data, pos, chunk_size_len);
            pos += chunk_size_len;
            let fmask = if pos + 4 <= data.len() {
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
            } else {
                0
            };
            (nbytes, fmask)
        } else {
            (0, 0)
        };

        entries.push(ChunkEntry {
            address,
            filtered_size,
            filter_mask,
            scaled,
        });
        Ok(())
    })?;

    Ok(entries)
}

/// Minimum bytes needed to represent value `v` (1, 2, 4, or 8).
fn bytes_needed_for(v: u64) -> usize {
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

fn read_var_uint_slice(data: &[u8], offset: usize, size: usize) -> u64 {
    let mut result = 0u64;
    for i in 0..size.min(8) {
        if offset + i < data.len() {
            result |= (data[offset + i] as u64) << (i * 8);
        }
    }
    result
}

/// Convert a linear chunk index to scaled (per-dimension) chunk coordinates.
fn linear_to_scaled(mut linear: u64, chunks_per_dim: &[u64]) -> Vec<u64> {
    let ndims = chunks_per_dim.len();
    let mut scaled = vec![0u64; ndims];
    // Row-major order: last dimension varies fastest
    for i in (0..ndims).rev() {
        scaled[i] = linear % chunks_per_dim[i];
        linear /= chunks_per_dim[i];
    }
    scaled
}

/// Copy a chunk's decompressed data into the correct position in the output buffer.
///
/// Handles edge chunks that may be smaller than a full chunk
/// (when dataset dims aren't evenly divisible by chunk dims).
fn copy_chunk_to_output(
    chunk_data: &[u8],
    scaled: &[u64],
    chunk_dims: &[u32],
    dataset_dims: &[u64],
    element_size: usize,
    output: &mut [u8],
) {
    let ndims = dataset_dims.len();

    if ndims == 0 {
        // Scalar — shouldn't happen for chunked, but handle gracefully
        let len = chunk_data.len().min(output.len());
        output[..len].copy_from_slice(&chunk_data[..len]);
        return;
    }

    // Compute the actual element counts for this chunk (handle edge chunks)
    let mut actual_dims = vec![0u64; ndims];
    for i in 0..ndims {
        let start = scaled[i] * chunk_dims[i] as u64;
        let end = (start + chunk_dims[i] as u64).min(dataset_dims[i]);
        actual_dims[i] = end - start;
    }

    // For 1D: simple memcpy
    if ndims == 1 {
        let dst_start = scaled[0] as usize * chunk_dims[0] as usize * element_size;
        let copy_len = actual_dims[0] as usize * element_size;
        let src_len = copy_len.min(chunk_data.len());
        if dst_start + src_len <= output.len() {
            output[dst_start..dst_start + src_len]
                .copy_from_slice(&chunk_data[..src_len]);
        }
        return;
    }

    // For N-D: copy row by row (innermost dimension is contiguous)
    // We iterate over all rows in the chunk and copy each one.
    let inner_len = actual_dims[ndims - 1] as usize * element_size;
    let _chunk_inner_stride = chunk_dims[ndims - 1] as usize * element_size;

    // Compute strides for the dataset (row-major)
    let mut ds_strides = vec![element_size; ndims];
    for i in (0..ndims - 1).rev() {
        ds_strides[i] = ds_strides[i + 1] * dataset_dims[i + 1] as usize;
    }
    // Compute strides for the chunk
    let mut ch_strides = vec![element_size; ndims];
    for i in (0..ndims - 1).rev() {
        ch_strides[i] = ch_strides[i + 1] * chunk_dims[i + 1] as usize;
    }

    // Number of "rows" to copy (product of all dims except the innermost)
    let nrows: usize = actual_dims[..ndims - 1].iter().map(|&d| d as usize).product();

    for row in 0..nrows {
        // Convert row index to per-dimension indices (excluding innermost)
        let mut remaining = row;
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;

        for i in 0..ndims - 1 {
            let _dim_count = actual_dims[i] as usize;
            let rows_below: usize = actual_dims[i + 1..ndims - 1]
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let idx = remaining / rows_below;
            remaining %= rows_below;

            src_offset += idx * ch_strides[i];
            dst_offset += (scaled[i] as usize * chunk_dims[i] as usize + idx) * ds_strides[i];
        }
        // Add the innermost dimension's base offset
        dst_offset += scaled[ndims - 1] as usize * chunk_dims[ndims - 1] as usize * element_size;

        if src_offset + inner_len <= chunk_data.len()
            && dst_offset + inner_len <= output.len()
        {
            output[dst_offset..dst_offset + inner_len]
                .copy_from_slice(&chunk_data[src_offset..src_offset + inner_len]);
        }
    }
}

fn read_var_le<R: ReadAt + ?Sized>(reader: &R, offset: u64, size: usize) -> Result<u64> {
    let mut buf = [0u8; 8];
    let n = size.min(8);
    reader
        .read_exact_at(offset, &mut buf[..n])
        .map_err(Error::Io)?;
    let mut result = 0u64;
    for i in 0..n {
        result |= (buf[i] as u64) << (i * 8);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── linear_to_scaled ──

    #[test]
    fn linear_to_scaled_1d() {
        // 10 chunks in one dimension
        let cpd = vec![10];
        assert_eq!(linear_to_scaled(0, &cpd), vec![0]);
        assert_eq!(linear_to_scaled(5, &cpd), vec![5]);
        assert_eq!(linear_to_scaled(9, &cpd), vec![9]);
    }

    #[test]
    fn linear_to_scaled_2d() {
        // 3x4 grid of chunks (row-major: last dim varies fastest)
        let cpd = vec![3, 4];
        assert_eq!(linear_to_scaled(0, &cpd), vec![0, 0]);
        assert_eq!(linear_to_scaled(1, &cpd), vec![0, 1]);
        assert_eq!(linear_to_scaled(3, &cpd), vec![0, 3]);
        assert_eq!(linear_to_scaled(4, &cpd), vec![1, 0]);
        assert_eq!(linear_to_scaled(11, &cpd), vec![2, 3]);
    }

    #[test]
    fn linear_to_scaled_3d() {
        let cpd = vec![2, 3, 4];
        assert_eq!(linear_to_scaled(0, &cpd), vec![0, 0, 0]);
        assert_eq!(linear_to_scaled(1, &cpd), vec![0, 0, 1]);
        assert_eq!(linear_to_scaled(4, &cpd), vec![0, 1, 0]);
        assert_eq!(linear_to_scaled(12, &cpd), vec![1, 0, 0]);
        assert_eq!(linear_to_scaled(23, &cpd), vec![1, 2, 3]);
    }

    // ── copy_chunk_to_output ──

    #[test]
    fn copy_chunk_1d_full() {
        // 1D dataset [8], chunk size 4, element size 2 (i16)
        let chunk_dims = [4u32];
        let dataset_dims = [8u64];
        let elem = 2;
        let mut output = vec![0u8; 16];

        // Chunk at scaled=[0]: copy 8 bytes to offset 0
        let chunk0 = vec![1, 0, 2, 0, 3, 0, 4, 0];
        copy_chunk_to_output(&chunk0, &[0], &chunk_dims, &dataset_dims, elem, &mut output);

        // Chunk at scaled=[1]: copy 8 bytes to offset 8
        let chunk1 = vec![5, 0, 6, 0, 7, 0, 8, 0];
        copy_chunk_to_output(&chunk1, &[1], &chunk_dims, &dataset_dims, elem, &mut output);

        assert_eq!(output, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);
    }

    #[test]
    fn copy_chunk_1d_edge() {
        // 1D dataset [6], chunk size 4, element size 1 → 2 chunks, second is edge
        let chunk_dims = [4u32];
        let dataset_dims = [6u64];
        let elem = 1;
        let mut output = vec![0u8; 6];

        // Chunk 0: full, 4 bytes
        copy_chunk_to_output(&[10, 20, 30, 40], &[0], &chunk_dims, &dataset_dims, elem, &mut output);
        // Chunk 1: edge, only 2 of 4 elements matter, but chunk data is still 4 bytes
        copy_chunk_to_output(&[50, 60, 0, 0], &[1], &chunk_dims, &dataset_dims, elem, &mut output);

        assert_eq!(output, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn copy_chunk_2d_full() {
        // 2D dataset [4,6], chunks [2,3], element size 1
        // 2x2 grid of chunks, each 2x3 = 6 bytes
        //
        // Desired output (row-major):
        //   Row 0: [ 1, 2, 3, 4, 5, 6]
        //   Row 1: [ 7, 8, 9,10,11,12]
        //   Row 2: [13,14,15,16,17,18]
        //   Row 3: [19,20,21,22,23,24]
        //
        // Each chunk stores its local region in chunk row-major order.
        let chunk_dims = [2u32, 3];
        let dataset_dims = [4u64, 6];
        let elem = 1;
        let mut output = vec![0u8; 24]; // 4*6

        // Chunk (0,0): rows 0-1, cols 0-2 → [1,2,3, 7,8,9]
        copy_chunk_to_output(&[1, 2, 3, 7, 8, 9], &[0, 0], &chunk_dims, &dataset_dims, elem, &mut output);

        // Chunk (0,1): rows 0-1, cols 3-5 → [4,5,6, 10,11,12]
        copy_chunk_to_output(&[4, 5, 6, 10, 11, 12], &[0, 1], &chunk_dims, &dataset_dims, elem, &mut output);

        // Chunk (1,0): rows 2-3, cols 0-2 → [13,14,15, 19,20,21]
        copy_chunk_to_output(&[13, 14, 15, 19, 20, 21], &[1, 0], &chunk_dims, &dataset_dims, elem, &mut output);

        // Chunk (1,1): rows 2-3, cols 3-5 → [16,17,18, 22,23,24]
        copy_chunk_to_output(&[16, 17, 18, 22, 23, 24], &[1, 1], &chunk_dims, &dataset_dims, elem, &mut output);

        // Expected: row-major [1..24]
        let expected: Vec<u8> = (1..=24).collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn copy_chunk_2d_edge() {
        // 2D dataset [3,5], chunks [2,3], element size 1
        // Grid: 2x2 chunks, but edges are partial
        //
        // Desired output (row-major):
        //   Row 0: [ 1, 2, 3, 4, 5]
        //   Row 1: [ 6, 7, 8, 9,10]
        //   Row 2: [11,12,13,14,15]
        //
        // Chunk (0,0): 2x3 full. Rows 0-1, cols 0-2.
        //   Chunk data: [1,2,3, 6,7,8]
        // Chunk (0,1): 2x2 edge (3 cols in chunk, only 2 used). Rows 0-1, cols 3-4.
        //   Chunk data: [4,5,0, 9,10,0]
        // Chunk (1,0): 1x3 edge (2 rows in chunk, only 1 used). Row 2, cols 0-2.
        //   Chunk data: [11,12,13, 0,0,0]
        // Chunk (1,1): 1x2 edge. Row 2, cols 3-4.
        //   Chunk data: [14,15,0, 0,0,0]
        let chunk_dims = [2u32, 3];
        let dataset_dims = [3u64, 5];
        let elem = 1;
        let mut output = vec![0u8; 15]; // 3*5

        copy_chunk_to_output(&[1, 2, 3, 6, 7, 8], &[0, 0], &chunk_dims, &dataset_dims, elem, &mut output);
        copy_chunk_to_output(&[4, 5, 0, 9, 10, 0], &[0, 1], &chunk_dims, &dataset_dims, elem, &mut output);
        copy_chunk_to_output(&[11, 12, 13, 0, 0, 0], &[1, 0], &chunk_dims, &dataset_dims, elem, &mut output);
        copy_chunk_to_output(&[14, 15, 0, 0, 0, 0], &[1, 1], &chunk_dims, &dataset_dims, elem, &mut output);

        let expected: Vec<u8> = (1..=15).collect();
        assert_eq!(output, expected);
    }

    // ── read_implicit_chunk_entries ──

    #[test]
    fn implicit_entries_1d() {
        let entries = read_implicit_chunk_entries(
            1000, // base address
            &[12], // dataset dims
            &[4],  // chunk dims
            16,    // chunk byte size (4 elements * 4 bytes)
        ).unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].address, 1000);
        assert_eq!(entries[0].scaled, vec![0]);
        assert_eq!(entries[1].address, 1016);
        assert_eq!(entries[1].scaled, vec![1]);
        assert_eq!(entries[2].address, 1032);
        assert_eq!(entries[2].scaled, vec![2]);
    }

    #[test]
    fn implicit_entries_2d() {
        let entries = read_implicit_chunk_entries(
            0, // base address
            &[10, 6], // dataset dims
            &[5, 3],  // chunk dims
            60,       // 5*3*4 bytes
        ).unwrap();

        assert_eq!(entries.len(), 4); // 2x2 chunks
        assert_eq!(entries[0].scaled, vec![0, 0]);
        assert_eq!(entries[0].address, 0);
        assert_eq!(entries[1].scaled, vec![0, 1]);
        assert_eq!(entries[1].address, 60);
        assert_eq!(entries[2].scaled, vec![1, 0]);
        assert_eq!(entries[2].address, 120);
        assert_eq!(entries[3].scaled, vec![1, 1]);
        assert_eq!(entries[3].address, 180);
    }

    #[test]
    fn implicit_entries_edge_chunks() {
        // Dataset [7], chunk [4] → 2 chunks (one is partial)
        let entries = read_implicit_chunk_entries(
            100,
            &[7],
            &[4],
            16,
        ).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].scaled, vec![0]);
        assert_eq!(entries[1].scaled, vec![1]);
    }

    // ── single chunk entries ──

    #[test]
    fn single_chunk_unfiltered() {
        let entries = read_single_chunk_entries(0x100, 80, None, None, 2).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].address, 0x100);
        assert_eq!(entries[0].filtered_size, 80);
        assert_eq!(entries[0].filter_mask, 0);
        assert_eq!(entries[0].scaled, vec![0, 0]);
    }

    #[test]
    fn single_chunk_filtered() {
        let entries = read_single_chunk_entries(0x200, 80, Some(42), Some(0x01), 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].filtered_size, 42);
        assert_eq!(entries[0].filter_mask, 0x01);
    }
}
