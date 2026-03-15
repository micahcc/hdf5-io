use crate::error::Result;
use crate::object_header::messages::MessageType;
use crate::superblock::UNDEF_ADDR;
use crate::writer::chunk_util::compute_chunk_count;
use crate::writer::chunk_util::enumerate_chunks;
use crate::writer::chunk_util::extract_chunk_data;
use crate::writer::dataset_node::DatasetNode;
use crate::writer::encode::OhdrMsg;
use crate::writer::encode::SUPERBLOCK_SIZE;
use crate::writer::encode::compat_dataset_chunk_size;
use crate::writer::encode::compat_group_chunk_size;
use crate::writer::encode::encode_attribute;
use crate::writer::encode::encode_attribute_info;
use crate::writer::encode::encode_chunked_layout;
use crate::writer::encode::encode_chunked_layout_v3;
use crate::writer::encode::encode_compact_layout;
use crate::writer::encode::encode_contiguous_layout;
use crate::writer::encode::encode_dataspace;
use crate::writer::encode::encode_datatype;
use crate::writer::encode::encode_fill_value_msg;
use crate::writer::encode::encode_fill_value_msg_alloc;
use crate::writer::encode::encode_filter_pipeline;
use crate::writer::encode::encode_group_info;
use crate::writer::encode::encode_link;
use crate::writer::encode::encode_link_info;
use crate::writer::encode::encode_object_header;
use crate::writer::encode::encode_superblock_versioned;
use crate::writer::encode::ohdr_overhead;
use crate::writer::file_writer::WriteOptions;
use crate::writer::group_node::GroupNode;
use crate::writer::types::ChunkFilter;
use crate::writer::types::ChildNode;
use crate::writer::types::StorageLayout;
use crate::writer::write_filters::apply_filters_forward;

/// Two-pass compat serialization: parent-first ordering + metadata block alignment.
///
/// Layout: [superblock] [root_group_ohdr] [child_ohdrs...] [padding] [data_blocks...]
pub(crate) fn write_tree_compat(
    root: &GroupNode,
    opts: &WriteOptions,
    meta_block_size: usize,
) -> Result<Vec<u8>> {
    // Phase 1: Flatten tree into objects, compute sizes with dummy addresses.
    let mut objects: Vec<ObjectInfo> = Vec::new();
    flatten_tree(root, opts, &mut objects)?;

    // Phase 2: Assign metadata addresses (parent-first, starting after superblock).
    // Index structures (FixedArray, etc.) are placed right after their OHDR.
    let mut meta_pos = SUPERBLOCK_SIZE;
    for obj in &mut objects {
        obj.meta_addr = meta_pos as u64;
        meta_pos += obj.ohdr_size + obj.index_meta_size;
    }

    // Phase 3: Compute data addresses (after metadata block).
    let has_external_data = objects.iter().any(|o| !o.data.is_empty());
    let data_start = if has_external_data {
        meta_block_size.max(meta_pos)
    } else {
        meta_pos
    };
    let mut data_pos = data_start;
    for obj in &mut objects {
        if !obj.data.is_empty() {
            obj.data_addr = data_pos as u64;
            data_pos += obj.data.len();
        }
    }

    let eof = data_pos;

    // Phase 4: Re-encode with real addresses.
    // We need to update:
    // - Group OHDRs: link messages point to child meta_addrs
    // - Dataset OHDRs: layout message points to data_addr
    let mut buf = vec![0u8; eof];

    // Write superblock (root group is at objects[0].meta_addr).
    let root_addr = objects[0].meta_addr;
    let sb_version = opts.superblock_version.unwrap_or(2);
    let sb = encode_superblock_versioned(root_addr, eof as u64, sb_version);
    buf[..SUPERBLOCK_SIZE].copy_from_slice(&sb);

    // Re-encode each object with correct addresses.
    for i in 0..objects.len() {
        let meta_addr = objects[i].meta_addr;
        let data_addr = objects[i].data_addr;
        let data = objects[i].data.clone();

        let ohdr_bytes = encode_object_final(&objects, i, opts)?;
        let start = meta_addr as usize;
        buf[start..start + ohdr_bytes.len()].copy_from_slice(&ohdr_bytes);

        // Write chunk index structure (FixedArray, etc.) if present.
        if objects[i].index_meta_size > 0 {
            if let ObjectKind::Dataset(ref d) = objects[i].kind {
                if let Some(ref ci) = d.chunked_info {
                    let index_addr = meta_addr + objects[i].ohdr_size as u64;
                    let index_bytes =
                        encode_chunk_index(ci, index_addr, data_addr)?;
                    let istart = index_addr as usize;
                    buf[istart..istart + index_bytes.len()]
                        .copy_from_slice(&index_bytes);
                }
            }
        }

        // Write data block.
        if !data.is_empty() {
            let ds = data_addr as usize;
            buf[ds..ds + data.len()].copy_from_slice(&data);
        }
    }

    // Truncate if data_pos < buf.len() (shouldn't happen, but be safe).
    buf.truncate(eof);
    Ok(buf)
}

struct ObjectInfo {
    kind: ObjectKind,
    /// Byte size of the encoded OHDR.
    ohdr_size: usize,
    /// Address assigned in metadata region.
    meta_addr: u64,
    /// Raw data bytes (for contiguous datasets).
    data: Vec<u8>,
    /// Address assigned for data (after metadata block).
    data_addr: u64,
    /// Child indices in the objects vec (for groups).
    child_indices: Vec<(String, usize)>,
    /// Target OHDR chunk size for NIL padding.
    target_chunk_size: Option<usize>,
    /// Size of chunk index metadata (FixedArray/ExtArray header+dblk) placed after OHDR.
    index_meta_size: usize,
}

enum ObjectKind {
    Group(GroupRef),
    Dataset(DatasetRef),
}

struct GroupRef {
    attributes: Vec<crate::writer::types::AttrData>,
}

struct DatasetRef {
    datatype: crate::datatype::Datatype,
    shape: Vec<u64>,
    max_dims: Option<Vec<u64>>,
    fill_value: Option<Vec<u8>>,
    attributes: Vec<crate::writer::types::AttrData>,
    vlen_elements: Option<Vec<Vec<u8>>>,
    /// Compact data (if layout is compact, data is embedded in OHDR).
    compact_data: Option<Vec<u8>>,
    /// Chunked layout info (if layout is chunked).
    chunked_info: Option<ChunkedInfo>,
}

struct ChunkedInfo {
    /// Chunk dimensions including element size as last dim (C library convention).
    chunk_dims_with_elem: Vec<u64>,
    filters: Vec<ChunkFilter>,
    index_type_id: u8,
    /// Individual chunk data blobs (after filtering).
    chunk_data_parts: Vec<Vec<u8>>,
    element_size: u32,
    /// For single-chunk with filters: the filtered chunk size.
    single_filtered_size: Option<u64>,
    /// Early allocation (affects fill value and index type).
    early_alloc: bool,
    /// Layout message version (3 = BTree v1, 4 = v4 index types).
    layout_version: u8,
    /// Chunk coordinates for each chunk (for BTree v1 keys).
    chunk_coords: Vec<Vec<u64>>,
    /// Dataset shape (for BTree v1 sentinel key).
    dataset_shape: Vec<u64>,
}

fn flatten_tree(
    group: &GroupNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let my_index = objects.len();

    // Reserve slot for this group.
    objects.push(ObjectInfo {
        kind: ObjectKind::Group(GroupRef {
            attributes: group.attributes.iter().map(clone_attr).collect(),
        }),
        ohdr_size: 0,
        meta_addr: 0,
        data: vec![],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
        index_meta_size: 0,
    });

    // Recursively flatten children.
    let mut child_indices = Vec::new();
    for (name, child) in &group.children {
        let child_idx = match child {
            ChildNode::Group(g) => flatten_tree(g, opts, objects)?,
            ChildNode::Dataset(d) => flatten_dataset(d, opts, objects)?,
        };
        child_indices.push((name.clone(), child_idx));
    }

    // Compute group OHDR size using dummy child addresses.
    let messages = build_group_messages_dummy(&child_indices, &group.attributes, opts)?;
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
    let target = compat_group_chunk_size(4, 8); // C library defaults: est_num_entries=4, est_name_len=8
    let target_chunk = if target > real_msg_bytes {
        Some(target)
    } else {
        None
    };
    let chunk_for_overhead = target_chunk.unwrap_or(real_msg_bytes);
    let ohdr_size = ohdr_overhead(chunk_for_overhead, opts);

    objects[my_index].child_indices = child_indices;
    objects[my_index].ohdr_size = ohdr_size;
    objects[my_index].target_chunk_size = target_chunk;

    Ok(my_index)
}

fn flatten_dataset(
    ds: &DatasetNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let idx = objects.len();

    let is_compact = matches!(ds.layout, StorageLayout::Compact);
    let is_chunked = matches!(ds.layout, StorageLayout::Chunked { .. });

    // Compute raw data (external data block) and optional chunked info.
    let (raw_data, chunked_info) = if is_chunked {
        flatten_chunked_data(ds, opts)?
    } else if is_compact {
        (vec![], None)
    } else if ds.vlen_elements.is_some() {
        (vec![], None)
    } else {
        (ds.data.clone(), None)
    };

    // Compute dataset OHDR size using dummy data address.
    let (target_chunk, ohdr_size) = if is_compact {
        let messages = build_dataset_messages_dummy_compact(ds, opts)?;
        let base_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
        let base_target = compat_dataset_chunk_size(base_msg_bytes);
        let target = base_target + ds.data.len();
        let tc = Some(target);
        (tc, ohdr_overhead(target, opts))
    } else if is_chunked {
        let messages =
            build_dataset_messages_dummy_chunked(ds, opts, chunked_info.as_ref().unwrap())?;
        let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
        let target = compat_dataset_chunk_size(real_msg_bytes);
        let tc = if target > real_msg_bytes {
            Some(target)
        } else {
            None
        };
        let chunk_for_overhead = tc.unwrap_or(real_msg_bytes);
        (tc, ohdr_overhead(chunk_for_overhead, opts))
    } else {
        let messages = build_dataset_messages_dummy(ds, opts, 0xDEAD_BEEF_0000_0000)?;
        let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
        let target = compat_dataset_chunk_size(real_msg_bytes);
        let tc = if target > real_msg_bytes {
            Some(target)
        } else {
            None
        };
        let chunk_for_overhead = tc.unwrap_or(real_msg_bytes);
        (tc, ohdr_overhead(chunk_for_overhead, opts))
    };

    // Compute index metadata size for chunk indices that need separate structures.
    let index_meta_size = chunked_info
        .as_ref()
        .map(|ci| compute_index_meta_size(ci))
        .unwrap_or(0);

    objects.push(ObjectInfo {
        kind: ObjectKind::Dataset(DatasetRef {
            datatype: ds.datatype.clone(),
            shape: ds.shape.clone(),
            max_dims: ds.max_dims.clone(),
            fill_value: ds.fill_value.clone(),
            attributes: ds.attributes.iter().map(clone_attr).collect(),
            vlen_elements: ds.vlen_elements.clone(),
            compact_data: if is_compact {
                Some(ds.data.clone())
            } else {
                None
            },
            chunked_info,
        }),
        ohdr_size,
        meta_addr: 0,
        data: raw_data,
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: target_chunk,
        index_meta_size,
    });

    Ok(idx)
}

/// Compute chunk data and metadata for a chunked dataset.
///
/// Returns (concatenated_chunk_data, ChunkedInfo).
fn flatten_chunked_data(
    ds: &DatasetNode,
    _opts: &WriteOptions,
) -> Result<(Vec<u8>, Option<ChunkedInfo>)> {
    let (chunk_dims, filters) = match &ds.layout {
        StorageLayout::Chunked {
            chunk_dims,
            filters,
        } => (chunk_dims, filters),
        _ => unreachable!(),
    };

    let element_size = ds.datatype.element_size();
    let has_filters = !filters.is_empty();

    // C library convention: chunk dims include element size as last dim.
    let mut chunk_dims_with_elem: Vec<u64> = chunk_dims.clone();
    chunk_dims_with_elem.push(element_size as u64);

    // Determine index type.
    let n_unlimited = ds
        .max_dims
        .as_ref()
        .map_or(0, |md| md.iter().filter(|&&d| d == u64::MAX).count());
    let total_chunks = compute_chunk_count(&ds.shape, chunk_dims);

    let index_type_id = if n_unlimited >= 2 {
        5u8 // BTreeV2
    } else if total_chunks == 1 && n_unlimited == 0 {
        1u8 // SingleChunk
    } else if n_unlimited == 1 {
        4u8 // ExtArray
    } else if ds.early_alloc && !has_filters {
        2u8 // Implicit (early alloc, no filters, fixed dims)
    } else {
        3u8 // FixedArray
    };

    // For datasets with no data written yet (e.g., empty_chunked), skip chunk computation.
    if ds.data.is_empty() {
        return Ok((
            vec![],
            Some(ChunkedInfo {
                chunk_dims_with_elem,
                filters: filters.clone(),
                index_type_id,
                chunk_data_parts: vec![],
                element_size,
                single_filtered_size: None,
                early_alloc: ds.early_alloc,
                layout_version: ds.layout_version,
                chunk_coords: vec![],
                dataset_shape: ds.shape.clone(),
            }),
        ));
    }

    // Compute chunk data parts (with filtering).
    let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);
    let mut chunk_data_parts: Vec<Vec<u8>> = Vec::new();
    for coords in &chunk_coords_list {
        let mut chunk = extract_chunk_data(
            &ds.data,
            &ds.shape,
            chunk_dims,
            coords,
            element_size as usize,
        );
        if has_filters {
            chunk = apply_filters_forward(filters, chunk, element_size)?;
        }
        chunk_data_parts.push(chunk);
    }

    let single_filtered_size = if index_type_id == 1 && has_filters {
        Some(chunk_data_parts[0].len() as u64)
    } else {
        None
    };

    // Concatenate all chunk data into one blob.
    let mut raw_data = Vec::new();
    for part in &chunk_data_parts {
        raw_data.extend_from_slice(part);
    }

    Ok((
        raw_data,
        Some(ChunkedInfo {
            chunk_dims_with_elem,
            filters: filters.clone(),
            index_type_id,
            chunk_data_parts,
            element_size,
            single_filtered_size,
            early_alloc: ds.early_alloc,
            layout_version: ds.layout_version,
            chunk_coords: chunk_coords_list,
            dataset_shape: ds.shape.clone(),
        }),
    ))
}

fn encode_object_final(
    objects: &[ObjectInfo],
    index: usize,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    let obj = &objects[index];
    let target_chunk = obj.target_chunk_size;

    match &obj.kind {
        ObjectKind::Group(g) => {
            let mut messages: Vec<OhdrMsg> = Vec::new();
            messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
            messages.push((
                MessageType::GroupInfo.as_u8(),
                encode_group_info(),
                0x01, // constant flag
            ));
            for (name, child_idx) in &obj.child_indices {
                let child_addr = objects[*child_idx].meta_addr;
                messages.push((MessageType::Link.as_u8(), encode_link(name, child_addr), 0));
            }
            for attr in &g.attributes {
                messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
            }

            encode_object_header(&messages, opts, target_chunk)
        }
        ObjectKind::Dataset(d) => {
            use crate::writer::types::SpaceAllocTime;

            let dt_body = encode_datatype(&d.datatype)?;

            // Always include max_dims in compat mode.
            let effective_max = Some(d.max_dims.clone().unwrap_or_else(|| d.shape.clone()));
            let ds_body = encode_dataspace(&d.shape, effective_max.as_deref());

            // Fill value allocation time depends on layout type.
            let alloc_time = if d.compact_data.is_some() {
                SpaceAllocTime::Early
            } else if let Some(ref ci) = d.chunked_info {
                if ci.early_alloc {
                    SpaceAllocTime::Early
                } else {
                    SpaceAllocTime::Incremental
                }
            } else {
                SpaceAllocTime::Late
            };
            let fv_body = encode_fill_value_msg_alloc(&d.fill_value, alloc_time);

            // Build layout message and optional filter pipeline.
            let (layout_body, filter_body) = if let Some(ref ci) = d.chunked_info {
                let chunk_index_addr = if obj.data.is_empty() {
                    UNDEF_ADDR
                } else if obj.index_meta_size > 0 {
                    // FixedArray/ExtArray/BTree v1: address points to index structure.
                    obj.meta_addr + obj.ohdr_size as u64
                } else {
                    // SingleChunk and Implicit: address points to chunk data.
                    obj.data_addr
                };
                let layout = if ci.layout_version == 3 {
                    encode_chunked_layout_v3(
                        &ci.chunk_dims_with_elem,
                        chunk_index_addr,
                    )
                } else {
                    encode_chunked_layout(
                        &ci.chunk_dims_with_elem,
                        ci.index_type_id,
                        chunk_index_addr,
                        ci.single_filtered_size,
                    )
                };
                let filter = if !ci.filters.is_empty() {
                    Some(encode_filter_pipeline(&ci.filters, ci.element_size))
                } else {
                    None
                };
                (layout, filter)
            } else if let Some(ref compact_data) = d.compact_data {
                (encode_compact_layout(compact_data), None)
            } else {
                let data_addr = if obj.data.is_empty() && d.vlen_elements.is_none() {
                    UNDEF_ADDR
                } else if d.vlen_elements.is_some() {
                    UNDEF_ADDR
                } else {
                    obj.data_addr
                };
                (
                    encode_contiguous_layout(data_addr, obj.data.len() as u64),
                    None,
                )
            };

            let attr_bodies: Vec<Vec<u8>> = d
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;

            let mut messages: Vec<OhdrMsg> = vec![
                (MessageType::Dataspace.as_u8(), ds_body, 0),
                (MessageType::Datatype.as_u8(), dt_body, 0x01),
                (MessageType::FillValue.as_u8(), fv_body, 0x01),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb, 0x01));
            }
            let layout_msg_flags =
                if d.chunked_info.as_ref().is_some_and(|ci| ci.early_alloc) {
                    0x01 // constant
                } else {
                    0x00
                };
            messages.push((MessageType::DataLayout.as_u8(), layout_body, layout_msg_flags));
            if !attr_bodies.is_empty() {
                messages.push((
                    MessageType::AttributeInfo.as_u8(),
                    encode_attribute_info(),
                    0x04,
                ));
            }
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body, 0));
            }

            encode_object_header(&messages, opts, target_chunk)
        }
    }
}

fn build_group_messages_dummy(
    child_indices: &[(String, usize)],
    attributes: &[crate::writer::types::AttrData],
    _opts: &WriteOptions,
) -> Result<Vec<OhdrMsg>> {
    let mut messages: Vec<OhdrMsg> = Vec::new();
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
    messages.push((MessageType::GroupInfo.as_u8(), encode_group_info(), 0x01));
    for (name, _) in child_indices {
        messages.push((MessageType::Link.as_u8(), encode_link(name, 0xDEAD_0000), 0));
    }
    for attr in attributes {
        messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
    }
    Ok(messages)
}

/// Build dummy messages for a compact dataset with empty data (for size estimation).
/// The C library allocates the OHDR before compact data is written, so the initial
/// layout message has no embedded data. The compact data size is added to the target later.
fn build_dataset_messages_dummy_compact(
    ds: &DatasetNode,
    _opts: &WriteOptions,
) -> Result<Vec<OhdrMsg>> {
    let dt_body = encode_datatype(&ds.datatype)?;
    let effective_max = Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()));
    let ds_body = encode_dataspace(&ds.shape, effective_max.as_deref());
    // Compact datasets use early allocation (compat=false).
    let fv_body = encode_fill_value_msg(&ds.fill_value, false);
    // Empty compact layout: version(1) + class(1) + size(2) = 4 bytes, no data
    let layout_body = encode_compact_layout(&[]);

    let attr_bodies: Vec<Vec<u8>> = ds
        .attributes
        .iter()
        .map(encode_attribute)
        .collect::<Result<Vec<_>>>()?;

    let mut messages: Vec<OhdrMsg> = vec![
        (MessageType::Dataspace.as_u8(), ds_body, 0),
        (MessageType::Datatype.as_u8(), dt_body, 0x01),
        (MessageType::FillValue.as_u8(), fv_body, 0x01),
        (MessageType::DataLayout.as_u8(), layout_body, 0),
    ];
    if !attr_bodies.is_empty() {
        messages.push((
            MessageType::AttributeInfo.as_u8(),
            encode_attribute_info(),
            0x04,
        ));
    }
    for body in attr_bodies {
        messages.push((MessageType::Attribute.as_u8(), body, 0));
    }
    Ok(messages)
}

/// Build dummy messages for a chunked dataset (for OHDR size estimation).
fn build_dataset_messages_dummy_chunked(
    ds: &DatasetNode,
    _opts: &WriteOptions,
    ci: &ChunkedInfo,
) -> Result<Vec<OhdrMsg>> {
    use crate::writer::types::SpaceAllocTime;

    let dt_body = encode_datatype(&ds.datatype)?;
    let effective_max = Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()));
    let ds_body = encode_dataspace(&ds.shape, effective_max.as_deref());
    let alloc_time = if ci.early_alloc {
        SpaceAllocTime::Early
    } else {
        SpaceAllocTime::Incremental
    };
    let fv_body = encode_fill_value_msg_alloc(&ds.fill_value, alloc_time);

    let layout_body = if ci.layout_version == 3 {
        encode_chunked_layout_v3(
            &ci.chunk_dims_with_elem,
            0xDEAD_BEEF_0000_0000,
        )
    } else {
        encode_chunked_layout(
            &ci.chunk_dims_with_elem,
            ci.index_type_id,
            0xDEAD_BEEF_0000_0000,
            ci.single_filtered_size,
        )
    };

    let filter_body = if !ci.filters.is_empty() {
        Some(encode_filter_pipeline(&ci.filters, ci.element_size))
    } else {
        None
    };

    let attr_bodies: Vec<Vec<u8>> = ds
        .attributes
        .iter()
        .map(encode_attribute)
        .collect::<Result<Vec<_>>>()?;

    let mut messages: Vec<OhdrMsg> = vec![
        (MessageType::Dataspace.as_u8(), ds_body, 0),
        (MessageType::Datatype.as_u8(), dt_body, 0x01),
        (MessageType::FillValue.as_u8(), fv_body, 0x01),
    ];
    if let Some(fb) = filter_body {
        messages.push((MessageType::FilterPipeline.as_u8(), fb, 0x01));
    }
    let layout_msg_flags = if ci.early_alloc { 0x01 } else { 0x00 };
    messages.push((MessageType::DataLayout.as_u8(), layout_body, layout_msg_flags));
    if !attr_bodies.is_empty() {
        messages.push((
            MessageType::AttributeInfo.as_u8(),
            encode_attribute_info(),
            0x04,
        ));
    }
    for body in attr_bodies {
        messages.push((MessageType::Attribute.as_u8(), body, 0));
    }
    Ok(messages)
}

fn build_dataset_messages_dummy(
    ds: &DatasetNode,
    _opts: &WriteOptions,
    dummy_data_addr: u64,
) -> Result<Vec<OhdrMsg>> {
    let dt_body = encode_datatype(&ds.datatype)?;

    // Always include max_dims in compat mode.
    let effective_max = Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()));
    let ds_body = encode_dataspace(&ds.shape, effective_max.as_deref());
    let fv_body = encode_fill_value_msg(&ds.fill_value, true);

    let data_size = ds.data.len() as u64;
    let layout_body = encode_contiguous_layout(dummy_data_addr, data_size);

    let attr_bodies: Vec<Vec<u8>> = ds
        .attributes
        .iter()
        .map(encode_attribute)
        .collect::<Result<Vec<_>>>()?;

    let mut messages: Vec<OhdrMsg> = vec![
        (MessageType::Dataspace.as_u8(), ds_body, 0),
        (MessageType::Datatype.as_u8(), dt_body, 0x01),
        (MessageType::FillValue.as_u8(), fv_body, 0x01),
        (MessageType::DataLayout.as_u8(), layout_body, 0),
    ];
    if !attr_bodies.is_empty() {
        messages.push((
            MessageType::AttributeInfo.as_u8(),
            encode_attribute_info(),
            0x04,
        ));
    }
    for body in attr_bodies {
        messages.push((MessageType::Attribute.as_u8(), body, 0));
    }
    Ok(messages)
}

/// Encode a chunk index structure, computing per-chunk addresses from `data_addr`.
fn encode_chunk_index(
    ci: &ChunkedInfo,
    index_addr: u64,
    data_addr: u64,
) -> Result<Vec<u8>> {
    // Compute per-chunk addresses within the data blob.
    let mut chunk_addrs = Vec::with_capacity(ci.chunk_data_parts.len());
    let mut offset = 0u64;
    let mut filtered_sizes = Vec::new();
    for part in &ci.chunk_data_parts {
        chunk_addrs.push(data_addr + offset);
        filtered_sizes.push(part.len() as u64);
        offset += part.len() as u64;
    }

    // BTree v1 (layout version 3)
    if ci.layout_version == 3 {
        return Ok(encode_btree_v1_chunk_node(ci, &chunk_addrs, &filtered_sizes));
    }

    match ci.index_type_id {
        3 => Ok(encode_fixed_array_index(
            index_addr,
            &chunk_addrs,
            !ci.filters.is_empty(),
            &filtered_sizes,
        )),
        4 => Ok(encode_ext_array_index(
            index_addr,
            &chunk_addrs,
            chunk_addrs.len(),
            !ci.filters.is_empty(),
            &filtered_sizes,
        )),
        _ => Ok(vec![]), // TODO: BTreeV2
    }
}

/// Compute how many bytes a chunk index structure needs in the metadata region.
/// Returns 0 for SingleChunk/Implicit (no separate index) or empty datasets.
fn compute_index_meta_size(ci: &ChunkedInfo) -> usize {
    let n_chunks = ci.chunk_data_parts.len();
    if n_chunks == 0 {
        return 0;
    }

    // BTree v1 (layout version 3): fixed-size node for 2K=64 entries.
    if ci.layout_version == 3 {
        let k: usize = 32; // default B-tree K for chunk data
        let dimensionality = ci.chunk_dims_with_elem.len();
        let key_size = 8 + dimensionality * 8; // nbytes(4) + mask(4) + dims*8
        let node_size = 24 + (2 * k + 1) * key_size + 2 * k * 8;
        return node_size;
    }

    match ci.index_type_id {
        1 | 2 => 0, // SingleChunk, Implicit: no separate structure
        3 => {
            // FixedArray: header + data block
            // Entry size: 8 (address only, no filters) or 8+4+2=14 (with filters)
            let entry_size: usize = if ci.filters.is_empty() { 8 } else { 8 + 4 + 2 };
            // Header: sig(4) + ver(1) + client(1) + entry_size(1) + page_bits(1) + nelmts(8) + dblk_addr(8) + checksum(4) = 28
            let hdr_size = 28;
            // Data block: sig(4) + ver(1) + client(1) + hdr_addr(8) + entries(n*entry_size) + checksum(4)
            let dblk_size = 4 + 1 + 1 + 8 + n_chunks * entry_size + 4;
            hdr_size + dblk_size
        }
        4 => {
            // ExtArray: header + index block + data blocks
            let (hdr_size, iblk_size, dblk_size) = ext_array_sizes(ci);
            hdr_size + iblk_size + dblk_size
        }
        _ => 0, // TODO: BTreeV2
    }
}

/// Encode a FixedArray index structure (header + data block).
/// `index_addr`: the metadata address where this structure starts.
/// `chunk_addrs`: addresses of individual chunk data blocks.
fn encode_fixed_array_index(
    index_addr: u64,
    chunk_addrs: &[u64],
    has_filters: bool,
    filtered_sizes: &[u64],
) -> Vec<u8> {
    let n_chunks = chunk_addrs.len();
    // Entry size: 8 (unfiltered: addr only) or 14 (filtered: addr + size_u32 + mask_u16)
    let entry_size: u8 = if has_filters { 14 } else { 8 };
    let client_id: u8 = if has_filters { 1 } else { 0 };

    // Header
    let hdr_size = 28usize;
    let dblk_addr = index_addr + hdr_size as u64;

    let mut buf = Vec::new();
    // FAHD signature
    buf.extend_from_slice(b"FAHD");
    buf.push(0); // version
    buf.push(client_id);
    buf.push(entry_size);
    buf.push(10); // page_bits (C library default)
    buf.extend_from_slice(&(n_chunks as u64).to_le_bytes()); // nelmts
    buf.extend_from_slice(&dblk_addr.to_le_bytes()); // dblk_addr
    let hdr_checksum = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&hdr_checksum.to_le_bytes());

    // Data block (FADB)
    let mut dblk = Vec::new();
    dblk.extend_from_slice(b"FADB");
    dblk.push(0); // version
    dblk.push(client_id);
    dblk.extend_from_slice(&index_addr.to_le_bytes()); // hdr_addr
    for (i, &addr) in chunk_addrs.iter().enumerate() {
        if has_filters {
            dblk.extend_from_slice(&addr.to_le_bytes());
            let fsize = filtered_sizes.get(i).copied().unwrap_or(0) as u32;
            dblk.extend_from_slice(&fsize.to_le_bytes()); // filtered size (u32)
            dblk.extend_from_slice(&0u16.to_le_bytes()); // filter mask (u16)
        } else {
            dblk.extend_from_slice(&addr.to_le_bytes());
        }
    }
    let dblk_checksum = crate::checksum::lookup3(&dblk);
    dblk.extend_from_slice(&dblk_checksum.to_le_bytes());

    buf.extend_from_slice(&dblk);
    buf
}

/// Spec for a single ExtArray data block.
struct EaDblkSpec {
    nelmts: u64,
    block_off: u64,
}

/// Compute which EA data blocks are needed for chunks beyond the index block.
///
/// The extensible array uses even-indexed super blocks for element mapping:
///   sblk_idx = 2 * floor(log2(loc_idx / min + 1))
/// The sblk info table (all indices) determines nelmts and block_off.
fn compute_ea_dblk_specs(n_chunks: usize, idx_blk_elmts: u32, data_blk_min_elmts: u32) -> Vec<EaDblkSpec> {
    if n_chunks as u32 <= idx_blk_elmts {
        return vec![];
    }

    let remaining_total = n_chunks as i64 - idx_blk_elmts as i64;
    let mut remaining = remaining_total;
    let mut dblks = Vec::new();

    // Build sblk info table and iterate through even-indexed sblks.
    // sblk s: ndblks = 2^(s/2), dblk_nelmts = min * 2^((s+1)/2)
    // start_idx = cumulative capacity of ALL prior sblks (including odd ones).
    let mut sblk_start = 0u64;
    let mut s = 0u32;
    while remaining > 0 {
        let ndblks = 1u64 << (s / 2);
        let dblk_nelmts = data_blk_min_elmts as u64 * (1u64 << ((s + 1) / 2));
        let capacity = ndblks * dblk_nelmts;

        if s % 2 == 0 {
            // Only even-indexed super blocks are used for element mapping.
            for d in 0..ndblks {
                let block_off = sblk_start + d * dblk_nelmts;
                dblks.push(EaDblkSpec { nelmts: dblk_nelmts, block_off });
                remaining -= dblk_nelmts as i64;
                if remaining <= 0 {
                    break;
                }
            }
        }

        sblk_start += capacity;
        s += 1;
    }

    dblks
}

/// Compute the sizes of ExtArray header, index block, and data blocks.
/// Returns (hdr_size, iblk_size, dblk_total_size).
fn ext_array_sizes(ci: &ChunkedInfo) -> (usize, usize, usize) {
    let idx_blk_elmts: u32 = 4;
    let sup_blk_min_data_ptrs: u32 = 4;
    let max_nelmts_bits: u32 = 32;
    let data_blk_min_elmts: u32 = 16;
    let entry_size: usize = if ci.filters.is_empty() { 8 } else { 8 + 4 + 2 };

    let ndblk_addrs = 2 * log2_of2(sup_blk_min_data_ptrs) as usize;
    let nsblks = 1 + (max_nelmts_bits - log2_of2(idx_blk_elmts * sup_blk_min_data_ptrs));
    let adj = ndblk_addrs as u32 + sup_blk_min_data_ptrs;
    let dblk_sblk_idx = 2 * (log2_gen(adj) - log2_of2(sup_blk_min_data_ptrs));
    let nsblk_addrs = (nsblks - dblk_sblk_idx) as usize;

    // EAHD: 72 bytes
    let hdr_size = 72;
    // EAIB: sig(4) + ver(1) + client(1) + hdr_addr(8) + elements + dblk_ptrs + sblk_ptrs + checksum(4)
    let iblk_size = 14 + idx_blk_elmts as usize * entry_size + (ndblk_addrs + nsblk_addrs) * 8 + 4;

    // Data blocks (EADBs) for chunks beyond the index block.
    let n_chunks = ci.chunk_data_parts.len();
    let block_off_size = (max_nelmts_bits / 8) as usize; // 4 bytes
    let dblk_specs = compute_ea_dblk_specs(n_chunks, idx_blk_elmts, data_blk_min_elmts);
    let dblk_total_size: usize = dblk_specs
        .iter()
        .map(|d| {
            // EADB: sig(4) + ver(1) + client(1) + hdr_addr(8) + block_off + entries + checksum(4)
            14 + block_off_size + d.nelmts as usize * entry_size + 4
        })
        .sum();

    (hdr_size, iblk_size, dblk_total_size)
}

/// Integer log2 for powers of 2.
fn log2_of2(n: u32) -> u32 {
    debug_assert!(n.is_power_of_two());
    n.trailing_zeros()
}

/// General integer log2 (floor).
fn log2_gen(n: u32) -> u32 {
    debug_assert!(n > 0);
    31 - n.leading_zeros()
}

/// Encode an Extensible Array index structure (header + index block + data blocks).
fn encode_ext_array_index(
    index_addr: u64,
    chunk_addrs: &[u64],
    n_stored: usize,
    has_filters: bool,
    filtered_sizes: &[u64],
) -> Vec<u8> {
    let entry_size: u8 = if has_filters { 14 } else { 8 };
    let idx_blk_elmts: u32 = 4;
    let sup_blk_min_data_ptrs: u32 = 4;
    let data_blk_min_elmts: u32 = 16;
    let max_nelmts_bits: u32 = 32;
    let max_dblk_page_nelmts_bits: u8 = 10;
    let block_off_size: usize = (max_nelmts_bits / 8) as usize; // 4

    let ndblk_addrs = 2 * log2_of2(sup_blk_min_data_ptrs) as usize;
    let nsblks = 1 + (max_nelmts_bits - log2_of2(idx_blk_elmts * sup_blk_min_data_ptrs));
    let adj = ndblk_addrs as u32 + sup_blk_min_data_ptrs;
    let dblk_sblk_idx = 2 * (log2_gen(adj) - log2_of2(sup_blk_min_data_ptrs));
    let nsblk_addrs = (nsblks - dblk_sblk_idx) as usize;

    let n_chunks = chunk_addrs.len();
    let client_id: u8 = if has_filters { 1 } else { 0 };

    // Compute data block specs for chunks beyond the index block.
    let dblk_specs = compute_ea_dblk_specs(n_chunks, idx_blk_elmts, data_blk_min_elmts);
    let n_dblks = dblk_specs.len();

    // Compute EAHD stats.
    let total_dblk_nelmts: u64 = dblk_specs.iter().map(|d| d.nelmts).sum();
    let nelmts = idx_blk_elmts as u64 + total_dblk_nelmts;

    // Compute sizes for address calculation.
    let hdr_size = 72usize;
    let iblk_size = 14 + idx_blk_elmts as usize * entry_size as usize
        + (ndblk_addrs + nsblk_addrs) * 8
        + 4;

    // Compute EADB sizes and addresses.
    let mut dblk_addrs = Vec::with_capacity(n_dblks);
    let mut dblk_offset = (hdr_size + iblk_size) as u64;
    let mut total_dblk_disk_size: u64 = 0;
    for spec in &dblk_specs {
        dblk_addrs.push(index_addr + dblk_offset);
        let dblk_disk_size =
            14 + block_off_size + spec.nelmts as usize * entry_size as usize + 4;
        dblk_offset += dblk_disk_size as u64;
        total_dblk_disk_size += dblk_disk_size as u64;
    }

    // --- EAHD header (72 bytes) ---
    let iblk_addr = index_addr + hdr_size as u64;

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"EAHD");
    hdr.push(0); // version
    hdr.push(client_id);
    hdr.push(entry_size);
    hdr.push(max_nelmts_bits as u8);
    hdr.push(idx_blk_elmts as u8);
    hdr.push(data_blk_min_elmts as u8);
    hdr.push(sup_blk_min_data_ptrs as u8);
    hdr.push(max_dblk_page_nelmts_bits);
    // Statistics
    hdr.extend_from_slice(&0u64.to_le_bytes()); // nsuper_blks
    hdr.extend_from_slice(&0u64.to_le_bytes()); // super_blk_size
    hdr.extend_from_slice(&(n_dblks as u64).to_le_bytes()); // ndata_blks
    hdr.extend_from_slice(&total_dblk_disk_size.to_le_bytes()); // data_blk_size
    hdr.extend_from_slice(&(n_stored as u64).to_le_bytes()); // max_idx_set
    hdr.extend_from_slice(&nelmts.to_le_bytes()); // nelmts
    hdr.extend_from_slice(&iblk_addr.to_le_bytes()); // idx_blk_addr
    let hdr_cs = crate::checksum::lookup3(&hdr);
    hdr.extend_from_slice(&hdr_cs.to_le_bytes());

    // --- EAIB index block ---
    let mut iblk = Vec::new();
    iblk.extend_from_slice(b"EAIB");
    iblk.push(0); // version
    iblk.push(client_id);
    iblk.extend_from_slice(&index_addr.to_le_bytes()); // hdr_addr

    // Direct elements (idx_blk_elmts entries)
    for i in 0..idx_blk_elmts as usize {
        if i < n_chunks {
            encode_ea_element(&mut iblk, chunk_addrs[i], has_filters, filtered_sizes[i]);
        } else {
            encode_ea_element_undef(&mut iblk, has_filters);
        }
    }

    // Data block pointers
    for i in 0..ndblk_addrs {
        if i < n_dblks {
            iblk.extend_from_slice(&dblk_addrs[i].to_le_bytes());
        } else {
            iblk.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
    }

    // Super block pointers (all UNDEF — no EASB structures needed)
    for _ in 0..nsblk_addrs {
        iblk.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    }

    let iblk_cs = crate::checksum::lookup3(&iblk);
    iblk.extend_from_slice(&iblk_cs.to_le_bytes());

    // --- EADB data blocks ---
    // Assign chunks to data blocks: chunks idx_blk_elmts..n_chunks go sequentially
    // into data blocks, filling each one up to its nelmts capacity.
    let mut chunk_cursor = idx_blk_elmts as usize; // next chunk to place
    let mut dblk_bufs: Vec<Vec<u8>> = Vec::with_capacity(n_dblks);

    for spec in &dblk_specs {
        let mut dblk = Vec::new();
        dblk.extend_from_slice(b"EADB");
        dblk.push(0); // version
        dblk.push(client_id);
        dblk.extend_from_slice(&index_addr.to_le_bytes()); // hdr_addr
        // block_off (4 bytes for max_nelmts_bits=32)
        dblk.extend_from_slice(&(spec.block_off as u32).to_le_bytes());

        // Elements
        let slots_to_fill = spec.nelmts as usize;
        for slot in 0..slots_to_fill {
            let _ = slot;
            if chunk_cursor < n_chunks {
                encode_ea_element(
                    &mut dblk,
                    chunk_addrs[chunk_cursor],
                    has_filters,
                    filtered_sizes[chunk_cursor],
                );
                chunk_cursor += 1;
            } else {
                encode_ea_element_undef(&mut dblk, has_filters);
            }
        }

        let dblk_cs = crate::checksum::lookup3(&dblk);
        dblk.extend_from_slice(&dblk_cs.to_le_bytes());
        dblk_bufs.push(dblk);
    }

    // Assemble final buffer: EAHD + EAIB + EADBs
    let mut buf = hdr;
    buf.extend_from_slice(&iblk);
    for dblk in dblk_bufs {
        buf.extend_from_slice(&dblk);
    }
    buf
}

fn encode_ea_element(buf: &mut Vec<u8>, addr: u64, has_filters: bool, fsize: u64) {
    if has_filters {
        buf.extend_from_slice(&addr.to_le_bytes());
        buf.extend_from_slice(&(fsize as u32).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
    } else {
        buf.extend_from_slice(&addr.to_le_bytes());
    }
}

fn encode_ea_element_undef(buf: &mut Vec<u8>, has_filters: bool) {
    if has_filters {
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
    } else {
        buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    }
}

/// Encode a B-tree v1 chunk node (leaf, single level).
///
/// The node has a fixed size based on K=32 (2K=64 max entries).
/// Format: "TREE" sig + type(1) + level(1) + entries_used(2) + left_sib(8) + right_sib(8)
/// then (2K+1) keys interleaved with 2K child pointers.
/// Key: nbytes(4) + filter_mask(4) + offsets(dimensionality × 8).
/// Unused entries are zero-filled.
fn encode_btree_v1_chunk_node(
    ci: &ChunkedInfo,
    chunk_addrs: &[u64],
    filtered_sizes: &[u64],
) -> Vec<u8> {
    let k: usize = 32;
    let dimensionality = ci.chunk_dims_with_elem.len();
    let key_size = 8 + dimensionality * 8;
    let node_size = 24 + (2 * k + 1) * key_size + 2 * k * 8;
    let n_chunks = chunk_addrs.len();

    let mut buf = Vec::with_capacity(node_size);

    // Header
    buf.extend_from_slice(b"TREE");
    buf.push(1); // node_type = 1 (data chunks)
    buf.push(0); // node_level = 0 (leaf)
    buf.extend_from_slice(&(n_chunks as u16).to_le_bytes());
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // left_sibling
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // right_sibling

    // Keys and child pointers: K0 C0 K1 C1 ... K(n-1) C(n-1) Kn [padding]
    for i in 0..n_chunks {
        // Key i
        let nbytes = filtered_sizes[i] as u32;
        buf.extend_from_slice(&nbytes.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask

        // Dimension offsets (element indices, not chunk indices)
        for d in 0..dimensionality - 1 {
            let off = ci.chunk_coords[i][d] * ci.chunk_dims_with_elem[d];
            buf.extend_from_slice(&off.to_le_bytes());
        }
        // Last dimension = 0 (element size dimension offset)
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Child pointer i
        buf.extend_from_slice(&chunk_addrs[i].to_le_bytes());
    }

    // Sentinel key (one past last chunk)
    // nbytes = 0, filter_mask = 0
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    // Dimension offsets: shape values for spatial dims, element_size for last dim
    for d in 0..dimensionality - 1 {
        buf.extend_from_slice(&ci.dataset_shape[d].to_le_bytes());
    }
    buf.extend_from_slice(&(ci.element_size as u64).to_le_bytes());

    // Zero-fill remaining entries to reach the fixed node size
    buf.resize(node_size, 0);
    buf
}

fn clone_attr(attr: &crate::writer::types::AttrData) -> crate::writer::types::AttrData {
    crate::writer::types::AttrData {
        name: attr.name.clone(),
        datatype: attr.datatype.clone(),
        shape: attr.shape.clone(),
        value: attr.value.clone(),
    }
}
