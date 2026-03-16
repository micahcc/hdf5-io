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
use crate::writer::encode::encode_attribute_info_addrs;
use crate::writer::encode::encode_attribute_shared;
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
use crate::writer::encode::encode_object_header_with_phase_change;
use crate::writer::encode::encode_superblock_versioned;
use crate::writer::encode::ohdr_overhead;
use crate::writer::file_writer::WriteOptions;
use crate::writer::group_node::GroupNode;
use crate::writer::types::ChildNode;
use crate::writer::types::ChunkFilter;
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
        // BTreeV2: place nodes after chunk data, aligned to meta_block_size.
        if let ObjectKind::Dataset(ref d) = obj.kind {
            if let Some(ref ci) = d.chunked_info {
                if ci.index_type_id == 5 && !ci.chunk_data_parts.is_empty() {
                    let nodes_start = align_up(data_pos, meta_block_size);
                    let node_size = 2048usize;
                    let nchunks = ci.chunk_data_parts.len();
                    let unfilt_chunk_bytes: u64 = ci.chunk_dims_with_elem.iter().product();
                    let has_filters = !ci.filters.is_empty();
                    let (_bt, rec_size, _csl) =
                        btree_v2_record_info(ci.ndims, has_filters, unfilt_chunk_bytes);
                    let layout = compute_btree_v2_layout(nchunks, rec_size as usize, node_size);
                    let num_nodes = layout.num_nodes();
                    obj.btree_v2_nodes_start = nodes_start as u64;
                    obj.btree_v2_num_nodes = num_nodes;
                    data_pos = nodes_start + num_nodes * node_size;
                }
            }
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

        // Write index/meta structures placed right after the OHDR.
        if objects[i].index_meta_size > 0 {
            let index_addr = meta_addr + objects[i].ohdr_size as u64;
            if let ObjectKind::Group(ref g) = objects[i].kind {
                if g.link_creation_order {
                    // Creation-order group: encode full interleaved meta blob.
                    let meta_bytes = encode_creation_order_index_meta(&objects[i], g, opts)?;
                    let istart = index_addr as usize;
                    buf[istart..istart + meta_bytes.len()].copy_from_slice(&meta_bytes);
                } else if !g.crt_order_child_names.is_empty() {
                    // Dense-link group (no creation order): interleaved child OHDRs + heap meta.
                    let meta_bytes = encode_dense_link_index_meta(&objects[i], g, opts)?;
                    let istart = index_addr as usize;
                    buf[istart..istart + meta_bytes.len()].copy_from_slice(&meta_bytes);
                } else if let Some(ref storage) = g.dense_attr_storage {
                    let dense_bytes = crate::writer::dense_attrs::encode_dense_attr_structures(
                        storage, index_addr,
                    )?;
                    let istart = index_addr as usize;
                    buf[istart..istart + dense_bytes.len()].copy_from_slice(&dense_bytes);
                }
            }
            if let ObjectKind::Dataset(ref d) = objects[i].kind {
                if let Some(ref storage) = d.dense_attr_storage {
                    // FHDB is in data region at data_addr + original_data_len.
                    let fhdb_addr = data_addr + d.original_data_len as u64;
                    let dense_bytes =
                        crate::writer::dense_attrs::encode_dense_attr_structures_split(
                            storage, index_addr, fhdb_addr,
                        )?;
                    let istart = index_addr as usize;
                    buf[istart..istart + dense_bytes.len()].copy_from_slice(&dense_bytes);
                }
                if let Some(ref ci) = d.chunked_info {
                    // Offset chunk index addr past any dense attr metadata.
                    let dense_meta = d
                        .dense_attr_storage
                        .as_ref()
                        .map(|s| {
                            let sizes = crate::writer::dense_attrs::compute_dense_attr_sizes(s);
                            sizes.frhp_size
                                + sizes.bthd_size
                                + sizes.fshd_size
                                + sizes.btlf_size
                                + sizes.fsse_size
                        })
                        .unwrap_or(0);
                    let ci_addr = index_addr + dense_meta as u64;
                    let index_bytes = encode_chunk_index(
                        ci,
                        ci_addr,
                        data_addr,
                        objects[i].btree_v2_nodes_start,
                    )?;
                    let istart = ci_addr as usize;
                    buf[istart..istart + index_bytes.len()].copy_from_slice(&index_bytes);
                }
            }
        }

        // Write data block (with vlen heap ID fixup if needed).
        if !data.is_empty() {
            let ds = data_addr as usize;
            if let ObjectKind::Group(ref g) = objects[i].kind {
                if g.link_creation_order {
                    // Creation-order group: encode FHDBs in data region.
                    let fhdbs = encode_creation_order_fhdbs(&objects[i], g, opts)?;
                    buf[ds..ds + fhdbs.len()].copy_from_slice(&fhdbs);
                } else if !g.crt_order_child_names.is_empty() {
                    // Dense-link group: encode FHDBs + FHIB in data region.
                    let data_bytes = encode_dense_link_data(&objects[i], g, opts)?;
                    buf[ds..ds + data_bytes.len()].copy_from_slice(&data_bytes);
                }
            } else if let ObjectKind::Dataset(ref d) = objects[i].kind {
                if let Some(ref storage) = d.dense_attr_storage {
                    // Write raw data first.
                    buf[ds..ds + d.original_data_len].copy_from_slice(&data[..d.original_data_len]);
                    // Encode and write FHDB at data_addr + original_data_len.
                    let frhp_addr = meta_addr + objects[i].ohdr_size as u64;
                    let fhdb =
                        crate::writer::dense_attrs::encode_fhdb_standalone(storage, frhp_addr);
                    let fhdb_start = ds + d.original_data_len;
                    buf[fhdb_start..fhdb_start + fhdb.len()].copy_from_slice(&fhdb);
                } else if let Some(ref vlen_elems) = d.vlen_elements {
                    // Rewrite the heap IDs with correct GCOL address.
                    let heap_id_bytes = vlen_elems.len() * 16;
                    let gcol_addr = data_addr + heap_id_bytes as u64;
                    let is_string = matches!(
                        d.datatype,
                        crate::datatype::Datatype::VarLen {
                            is_string: true,
                            ..
                        }
                    );
                    let gcol_order = if is_string {
                        vlen_string_gcol_order(vlen_elems.len())
                    } else {
                        (0..vlen_elems.len()).collect()
                    };
                    let base_elem_size = match &d.datatype {
                        crate::datatype::Datatype::VarLen { element_type, .. } => {
                            element_type.element_size() as usize
                        }
                        _ => 1,
                    };
                    let heap_ids = build_vlen_heap_ids_compat(
                        vlen_elems,
                        gcol_addr,
                        &gcol_order,
                        base_elem_size,
                    );
                    buf[ds..ds + heap_ids.len()].copy_from_slice(&heap_ids);
                    // GCOL bytes are already in data[heap_id_bytes..]
                    buf[ds + heap_id_bytes..ds + data.len()]
                        .copy_from_slice(&data[heap_id_bytes..]);
                } else {
                    buf[ds..ds + data.len()].copy_from_slice(&data);
                }
            } else {
                buf[ds..ds + data.len()].copy_from_slice(&data);
            }
        }

        // Write BTreeV2 nodes if applicable.
        if objects[i].btree_v2_nodes_start != UNDEF_ADDR {
            if let ObjectKind::Dataset(ref d) = objects[i].kind {
                if let Some(ref ci) = d.chunked_info {
                    if ci.index_type_id == 5 {
                        // Compute per-chunk addresses.
                        let mut chunk_addrs = Vec::new();
                        let mut offset = 0u64;
                        for part in &ci.chunk_data_parts {
                            chunk_addrs.push(data_addr + offset);
                            offset += part.len() as u64;
                        }
                        let unfilt_chunk_bytes: u64 = ci.chunk_dims_with_elem.iter().product();
                        let has_filters = !ci.filters.is_empty();
                        let (btree_type, rec_size, chunk_size_len) =
                            btree_v2_record_info(ci.ndims, has_filters, unfilt_chunk_bytes);
                        let mut filt_sizes: Vec<u64> =
                            ci.chunk_data_parts.iter().map(|p| p.len() as u64).collect();
                        if filt_sizes.is_empty() {
                            filt_sizes = vec![0; chunk_addrs.len()];
                        }
                        let node_size = 2048usize;
                        let nodes_start = objects[i].btree_v2_nodes_start as usize;
                        let layout = compute_btree_v2_layout(
                            chunk_addrs.len(),
                            rec_size as usize,
                            node_size,
                        );
                        write_btree_v2_nodes(
                            &mut buf,
                            nodes_start,
                            node_size,
                            &layout,
                            &chunk_addrs,
                            &ci.chunk_coords,
                            &filt_sizes,
                            ci.ndims,
                            btree_type,
                            chunk_size_len,
                        );
                    }
                }
            }
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
    /// For BTreeV2: start address of the node region (placed after chunk data).
    btree_v2_nodes_start: u64,
    /// For BTreeV2: number of node_size nodes allocated.
    btree_v2_num_nodes: usize,
}

enum ObjectKind {
    Group(GroupRef),
    Dataset(DatasetRef),
    CommittedDatatype(CommittedTypeRef),
    Continuation(ContinuationRef),
}

struct GroupRef {
    attributes: Vec<crate::writer::types::AttrData>,
    /// Non-default attribute phase change thresholds.
    attr_phase_change: Option<(u16, u16)>,
    /// Dense attribute storage info (if attributes exceed compact threshold).
    dense_attr_storage: Option<crate::writer::dense_attrs::DenseAttrStorage>,
    /// Total byte size of the dense attribute structures placed after this OHDR.
    #[allow(dead_code)]
    dense_attr_meta_size: usize,
    /// Non-default link phase change thresholds.
    link_phase_change: Option<(u16, u16)>,
    /// Track and index link creation order.
    link_creation_order: bool,
    /// Track and index attribute creation order.
    attr_creation_order: bool,
    /// Dense link storage (if links exceed compact threshold or phase_change=(0,0)).
    dense_link_storage: Option<crate::writer::dense_attrs::DenseLinkStorage>,
    /// For creation-order groups: child names in insertion order.
    /// Children are encoded inline, not as separate objects.
    crt_order_child_names: Vec<String>,
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
    /// If set, this dataset references a committed type at this object index.
    committed_type_index: Option<usize>,
    /// Map of committed type names to object indices for attribute shared type refs.
    attr_committed_types: std::collections::HashMap<String, usize>,
    /// Non-default attribute phase change thresholds.
    attr_phase_change: Option<(u16, u16)>,
    /// Dense attribute storage info (if attributes exceed compact threshold).
    dense_attr_storage: Option<crate::writer::dense_attrs::DenseAttrStorage>,
    /// Length of original dataset data (before FHDB was appended).
    original_data_len: usize,
}

struct CommittedTypeRef {
    datatype: crate::datatype::Datatype,
    /// Object index of the OCHK continuation block.
    ochk_index: Option<usize>,
}

struct ContinuationRef {
    datatype: crate::datatype::Datatype,
    refcount: u32,
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
    /// Number of spatial dimensions (for BTree v2 records).
    ndims: usize,
}

fn flatten_tree(
    group: &GroupNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let my_index = objects.len();

    // Creation-order groups have a special interleaved metadata layout.
    if group.link_creation_order {
        return flatten_creation_order_group(group, opts, objects);
    }

    // Dense link storage without creation order (fractal heap indirect block).
    // All children are encoded inline in the index_meta region.
    if group.link_phase_change == Some((0, 0))
        && !group.link_creation_order
        && !group.children.is_empty()
    {
        return flatten_dense_link_group(group, opts, objects);
    }

    // Determine if dense attribute storage is needed.
    let use_dense = if let Some((max_compact, _)) = group.attr_phase_change {
        group.attributes.len() > max_compact as usize
    } else {
        false
    };

    let (dense_storage, dense_meta_size) = if use_dense {
        let cloned_attrs: Vec<_> = group.attributes.iter().map(clone_attr).collect();
        let storage = crate::writer::dense_attrs::compute_dense_attr_storage(&cloned_attrs)?;
        let sizes = crate::writer::dense_attrs::compute_dense_attr_sizes(&storage);
        let meta_size = sizes.total();
        (Some(storage), meta_size)
    } else {
        (None, 0)
    };

    // Reserve slot for this group.
    objects.push(ObjectInfo {
        kind: ObjectKind::Group(GroupRef {
            attributes: group.attributes.iter().map(clone_attr).collect(),
            attr_phase_change: group.attr_phase_change,
            dense_attr_storage: dense_storage,
            dense_attr_meta_size: dense_meta_size,
            link_phase_change: group.link_phase_change,
            link_creation_order: group.link_creation_order,
            attr_creation_order: group.attr_creation_order,
            dense_link_storage: None,
            crt_order_child_names: vec![],
        }),
        ohdr_size: 0,
        meta_addr: 0,
        data: vec![],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
        index_meta_size: 0,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
    });

    // Pre-count committed type references for refcount message.
    // Base refcount starts at 1 (for the commit itself). Each dataset or attribute
    // that references the committed type adds 1.
    let mut ct_refcounts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (_, child) in &group.children {
        if let ChildNode::Dataset(d) = child {
            if let Some(ref ct_name) = d.committed_type_name {
                *ct_refcounts.entry(ct_name.clone()).or_insert(1) += 1;
            }
            // Count attributes that reference committed types.
            for attr in &d.attributes {
                if let Some(ref ct_name) = attr.committed_type_name {
                    *ct_refcounts.entry(ct_name.clone()).or_insert(1) += 1;
                }
            }
        }
    }

    // Recursively flatten children.
    let mut child_indices = Vec::new();
    let mut committed_types: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut ochk_placed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, child) in &group.children {
        let child_idx = match child {
            ChildNode::Group(g) => flatten_tree(g, opts, objects)?,
            ChildNode::Dataset(d) => {
                let ct_index = d
                    .committed_type_name
                    .as_ref()
                    .and_then(|n| committed_types.get(n).copied());
                let ds_idx = flatten_dataset(d, opts, objects, ct_index, &committed_types)?;
                // Place OCHK after the first dataset referencing a committed type.
                if let Some(ref ct_name) = d.committed_type_name {
                    if !ochk_placed.contains(ct_name) {
                        if let Some(&ct_obj_idx) = committed_types.get(ct_name) {
                            let refcount = ct_refcounts.get(ct_name).copied().unwrap_or(1);
                            let dt = match &objects[ct_obj_idx].kind {
                                ObjectKind::CommittedDatatype(ct) => ct.datatype.clone(),
                                _ => unreachable!(),
                            };
                            let ochk_idx = flatten_ochk(&dt, refcount, objects)?;
                            if let ObjectKind::CommittedDatatype(ref mut ct) =
                                objects[ct_obj_idx].kind
                            {
                                ct.ochk_index = Some(ochk_idx);
                            }
                            ochk_placed.insert(ct_name.clone());
                        }
                    }
                }
                ds_idx
            }
            ChildNode::CommittedDatatype(dt) => {
                let ct_idx = flatten_committed_type(dt, opts, objects)?;
                committed_types.insert(name.clone(), ct_idx);
                ct_idx
            }
        };
        child_indices.push((name.clone(), child_idx));
    }

    // Compute group OHDR size using dummy child addresses.
    let dense_attrs = if let ObjectKind::Group(ref g) = objects[my_index].kind {
        g.dense_attr_storage.is_some()
    } else {
        false
    };
    let attrs_for_dummy = if dense_attrs {
        &[] as &[crate::writer::types::AttrData]
    } else {
        &group.attributes[..]
    };
    let messages = build_group_messages_dummy(&child_indices, attrs_for_dummy, opts, dense_attrs)?;
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
    let target = compat_group_chunk_size(4, 8); // C library defaults: est_num_entries=4, est_name_len=8

    // For dense attribute groups, the C library's OHDR allocation reflects the history:
    // 1. Initial chunk = max(base_chunk, base_msgs + max_compact inline attrs)
    // 2. Dense conversion extends chunk by AttrInfo message size (22)
    // We replicate this to match the C library's final chunk_size.
    let chunk_for_overhead = if dense_attrs {
        let max_compact = group
            .attr_phase_change
            .map(|(mc, _)| mc as usize)
            .unwrap_or(0);
        let base_msgs: usize = messages
            .iter()
            .filter(|(t, _, _)| *t != MessageType::AttributeInfo.as_u8())
            .map(|(_, b, _)| 4 + b.len())
            .sum();
        let inline_attr_bytes: usize = group
            .attributes
            .iter()
            .take(max_compact)
            .map(|a| 4 + encode_attribute(a).map(|b| b.len()).unwrap_or(0))
            .sum();
        let attr_info_size = messages
            .iter()
            .find(|(t, _, _)| *t == MessageType::AttributeInfo.as_u8())
            .map(|(_, b, _)| 4 + b.len())
            .unwrap_or(22);
        target.max(base_msgs + inline_attr_bytes) + attr_info_size
    } else {
        let target_chunk = if target > real_msg_bytes {
            Some(target)
        } else {
            None
        };
        target_chunk.unwrap_or(real_msg_bytes)
    };

    // For groups with phase change, the OHDR prefix is 4 bytes larger.
    let phase_change_extra = if group.attr_phase_change.is_some() {
        4
    } else {
        0
    };
    let ohdr_size = ohdr_overhead(chunk_for_overhead, opts) + phase_change_extra;

    let target_chunk = if chunk_for_overhead > real_msg_bytes {
        Some(chunk_for_overhead)
    } else {
        None
    };

    objects[my_index].child_indices = child_indices;
    objects[my_index].ohdr_size = ohdr_size;
    objects[my_index].target_chunk_size = target_chunk;
    objects[my_index].index_meta_size = dense_meta_size;

    Ok(my_index)
}

/// Flatten a creation-order group into a single ObjectInfo.
///
/// Creation-order groups have an interleaved metadata layout where child OHDRs
/// are interleaved with dense link/attr structures. Children are encoded inline
/// rather than as separate objects.
fn flatten_creation_order_group(
    group: &GroupNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    use crate::writer::dense_attrs::compute_dense_attr_crt_sizes;
    use crate::writer::dense_attrs::compute_dense_attr_storage;
    use crate::writer::dense_attrs::compute_dense_link_sizes;
    use crate::writer::dense_attrs::compute_dense_link_storage;

    let my_index = objects.len();
    let child_names: Vec<String> = group.children.iter().map(|(n, _)| n.clone()).collect();
    let n_children = child_names.len();

    // Child groups are standard empty groups. Compute their OHDR size.
    let child_target_chunk = compat_group_chunk_size(4, 8); // 120
    let child_ohdr_size = ohdr_overhead(child_target_chunk, opts); // 147 with timestamps

    // Dense link storage (placeholder addresses, recomputed in phase 4).
    let placeholder_addrs: Vec<u64> = vec![0xDEAD_0000; n_children];
    let link_storage = compute_dense_link_storage(&child_names, &placeholder_addrs);
    let link_sizes = compute_dense_link_sizes(&link_storage);

    // Dense attr storage.
    let cloned_attrs: Vec<_> = group.attributes.iter().map(clone_attr).collect();
    let attr_storage = compute_dense_attr_storage(&cloned_attrs)?;
    let attr_crt_sizes = compute_dense_attr_crt_sizes(&attr_storage);

    // Interleaved layout offsets (from start of index_meta, after ordered OHDR):
    //   first_child_ohdr
    //   link dense meta (FRHP + BTHD_name + BTHD_crt + FSHD + BTLF_name + BTLF_crt)
    //   remaining child OHDRs
    //   attr dense part1 (FRHP + BTHD_name + BTHD_crt)
    //   OCHK (LinkInfo + AttrInfo)
    //   attr dense part2 (FSHD + BTLF_name + BTLF_crt)
    //   FSSE links
    //   FSSE attrs
    // Link dense meta excludes FSSE (it goes at the end separately).
    let first_child = child_ohdr_size;
    let link_meta_no_fsse = link_sizes.frhp_size
        + link_sizes.bthd_name_size
        + link_sizes.bthd_crt_size
        + link_sizes.fshd_size
        + link_sizes.btlf_name_size
        + link_sizes.btlf_crt_size;
    let remaining_children = (n_children - 1) * child_ohdr_size;
    let attr_part1 =
        attr_crt_sizes.frhp_size + attr_crt_sizes.bthd_name_size + attr_crt_sizes.bthd_crt_size;
    let ochk_size = 4 + (6 + 34) + (6 + 28) + 4; // OCHK sig + LinkInfo + AttrInfo + cksum = 82
    let attr_part2 =
        attr_crt_sizes.fshd_size + attr_crt_sizes.btlf_name_size + attr_crt_sizes.btlf_crt_size;
    let fsse_link = link_sizes.fsse_size;
    let fsse_attr = attr_crt_sizes.fsse_size;

    let index_meta_size = first_child
        + link_meta_no_fsse
        + remaining_children
        + attr_part1
        + ochk_size
        + attr_part2
        + fsse_link
        + fsse_attr;

    // Ordered group OHDR: creation-order flags, chunk_size=52.
    // Messages (6-byte headers): CONT(22) + NIL(18) + GroupInfo(12) = 52
    let ohdr_chunk_size = 52;
    // Prefix: "OHDR"(4) + ver(1) + flags(1) + timestamps(16) + phase_change(4) + chunk_size(1) = 27
    let ohdr_size = 27 + ohdr_chunk_size + 4; // 83

    // Data region: FHDB attrs + FHDB links
    let data_size = attr_crt_sizes.fhdb_size + link_sizes.fhdb_size;
    let data = vec![0u8; data_size];

    objects.push(ObjectInfo {
        kind: ObjectKind::Group(GroupRef {
            attributes: group.attributes.iter().map(clone_attr).collect(),
            attr_phase_change: group.attr_phase_change,
            dense_attr_storage: Some(attr_storage),
            dense_attr_meta_size: 0,
            link_phase_change: group.link_phase_change,
            link_creation_order: true,
            attr_creation_order: group.attr_creation_order,
            dense_link_storage: Some(link_storage),
            crt_order_child_names: child_names,
        }),
        ohdr_size,
        meta_addr: 0,
        data,
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
        index_meta_size,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
    });

    Ok(my_index)
}

/// Flatten a dense-link group (link_phase_change(0,0), no creation order).
///
/// Child OHDRs and dense link structures are inlined into the index_meta region.
/// FHDBs + FHIB go in the data region.
///
/// File layout:
///   group_ohdr | child_0_ohdr | FRHP+BTHD+FSHD+BTLF | children_1..N | FSSE | [data: FHDBs+FHIB]
fn flatten_dense_link_group(
    group: &GroupNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let my_index = objects.len();
    let child_names: Vec<String> = group.children.iter().map(|(n, _)| n.clone()).collect();
    let n_children = child_names.len();

    // All children are empty groups. Compute child OHDR size.
    let child_target_chunk = compat_group_chunk_size(4, 8); // 120
    let child_ohdr_size = ohdr_overhead(child_target_chunk, opts); // 147 with timestamps

    // Dense link storage (placeholder addresses to compute sizes).
    let placeholder_addrs: Vec<u64> = vec![0xDEAD_0000; n_children];
    let link_bodies: Vec<Vec<u8>> = child_names
        .iter()
        .zip(placeholder_addrs.iter())
        .map(|(name, addr)| crate::writer::dense_attrs::encode_link_body_no_crt(name, *addr))
        .collect();
    let total_link_bytes: usize = link_bodies.iter().map(|b| b.len()).sum();

    // Compute doubling table to determine block count and sizes.
    let start_block_size = 512usize;
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8); // 4
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 21
    let heap_id_len: u16 = 7;
    let node_size = 512usize;
    let rec_size: u16 = 4 + heap_id_len; // 11

    // Use the doubling table computation from dense_attrs.
    let dtable = crate::writer::dense_attrs::compute_doubling_table_pub(
        total_link_bytes,
        start_block_size,
        dblk_header_size,
    );
    let n_blocks = dtable.block_sizes.len();
    let nrows = dtable.nrows;

    // Distribute link bodies across blocks to compute free space.
    let mut block_used: Vec<usize> = vec![dblk_header_size; n_blocks];
    let mut block_idx = 0;
    for body in &link_bodies {
        let bsz = dtable.block_sizes[block_idx];
        if block_used[block_idx] + body.len() > bsz && block_idx + 1 < n_blocks {
            block_idx += 1;
        }
        block_used[block_idx] += body.len();
    }

    // Compute free sections.
    let mut block_free: Vec<(usize, usize)> = Vec::new(); // (heap_addr_of_free_start, free_size)
    for bi in 0..n_blocks {
        let bsz = dtable.block_sizes[bi];
        let blk_heap_off = dtable.block_offsets[bi] as usize;
        let free = bsz - block_used[bi];
        if free > 0 {
            block_free.push((blk_heap_off + block_used[bi], free));
        }
    }
    let total_free: u64 = block_free.iter().map(|(_, f)| *f as u64).sum();
    let n_sections = block_free.len();

    // Compute FSSE size.
    let sect_off_size = 4usize; // (max_heap_size_bits + 7) / 8
    let sect_len_size = crate::writer::dense_attrs::limit_enc_size_u64_pub(65536).max(1); // max_sect_size from FSHD
    let sect_cnt_size =
        crate::writer::dense_attrs::limit_enc_size_u64_pub(n_sections as u64).max(1);

    // Group sections by size.
    let mut size_groups: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for &(addr, size) in &block_free {
        size_groups.entry(size).or_default().push(addr);
    }

    let mut fsse_data_len = 0usize;
    for (_, addrs) in &size_groups {
        fsse_data_len += sect_cnt_size;
        fsse_data_len += sect_len_size;
        fsse_data_len += addrs.len() * (sect_off_size + 1);
    }
    let fsse_size = 4 + 1 + 8 + fsse_data_len + 4; // sig+ver+fshd_addr+data+cksum

    // Dense link metadata sizes.
    let frhp_size = 146usize;
    let bthd_size = 38usize;
    let fshd_size = 82usize;
    let btlf_size = node_size;

    // Index meta layout:
    //   child_0_ohdr + FRHP + BTHD + FSHD + BTLF + children_1..N + FSSE
    let index_meta_size = child_ohdr_size
        + frhp_size
        + bthd_size
        + fshd_size
        + btlf_size
        + (n_children - 1) * child_ohdr_size
        + fsse_size;

    // Data region: FHDBs (reverse block_offset order) + FHIB.
    let fhib_size = 4 + 1 + 8 + block_offset_bytes + nrows * dtable.width * 8 + 4;
    let data_size: usize = dtable.block_sizes.iter().sum::<usize>() + fhib_size;

    // Group OHDR: link info (no crt order) + group info (with phase change 0,0).
    // Messages: LinkInfo(4+18=22) + GroupInfo(4+6=10) = 32 chunk.
    // OHDR = sig(4)+ver(1)+flags(1)+timestamps(16)+chunk_size(1)+32+cksum(4) = 59
    let ohdr_chunk_size = 32;
    let ohdr_size = 4 + 1 + 1 + 16 + 1 + ohdr_chunk_size + 4; // 59

    objects.push(ObjectInfo {
        kind: ObjectKind::Group(GroupRef {
            attributes: vec![],
            attr_phase_change: group.attr_phase_change,
            dense_attr_storage: None,
            dense_attr_meta_size: 0,
            link_phase_change: group.link_phase_change,
            link_creation_order: false,
            attr_creation_order: false,
            dense_link_storage: None,
            crt_order_child_names: child_names,
        }),
        ohdr_size,
        meta_addr: 0,
        data: vec![0u8; data_size],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
        index_meta_size,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
    });

    Ok(my_index)
}

fn flatten_dataset(
    ds: &DatasetNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
    committed_type_index: Option<usize>,
    committed_types: &std::collections::HashMap<String, usize>,
) -> Result<usize> {
    let idx = objects.len();

    let is_compact = matches!(ds.layout, StorageLayout::Compact);
    let is_chunked = matches!(ds.layout, StorageLayout::Chunked { .. });

    // Compute raw data (external data block) and optional chunked info.
    let (mut raw_data, chunked_info) = if is_chunked {
        flatten_chunked_data(ds, opts)?
    } else if is_compact {
        (vec![], None)
    } else if let Some(ref vlen_elems) = ds.vlen_elements {
        // Determine GCOL ordering based on vlen type.
        let is_string = matches!(
            ds.datatype,
            crate::datatype::Datatype::VarLen {
                is_string: true,
                ..
            }
        );
        let gcol_order = if is_string {
            vlen_string_gcol_order(vlen_elems.len())
        } else {
            (0..vlen_elems.len()).collect()
        };
        let ordered_refs: Vec<&Vec<u8>> = gcol_order.iter().map(|&i| &vlen_elems[i]).collect();
        let gcol_bytes = build_gcol_compat(&ordered_refs);
        let heap_id_bytes = vlen_elems.len() * 16;
        // Placeholder: heap_ids (will be rewritten in phase 4) + gcol
        let mut blob = vec![0u8; heap_id_bytes];
        blob.extend_from_slice(&gcol_bytes);
        (blob, None)
    } else {
        (ds.data.clone(), None)
    };

    // Determine if dense attribute storage is needed for this dataset.
    let use_dense_ds = if let Some((max_compact, _)) = ds.attr_phase_change {
        ds.attributes.len() > max_compact as usize
    } else {
        false
    };

    let (dense_storage, dense_meta_size) = if use_dense_ds {
        let cloned_attrs: Vec<_> = ds.attributes.iter().map(|a| clone_attr(a)).collect();
        let storage = crate::writer::dense_attrs::compute_dense_attr_storage(&cloned_attrs)?;
        let sizes = crate::writer::dense_attrs::compute_dense_attr_sizes(&storage);
        // Meta = FRHP+BTHD+FSHD+BTLF+FSSE (not FHDB — that goes in data region)
        let meta_size =
            sizes.frhp_size + sizes.bthd_size + sizes.fshd_size + sizes.btlf_size + sizes.fsse_size;
        (Some(storage), meta_size)
    } else {
        (None, 0)
    };

    // For dense attrs, append FHDB placeholder to raw_data.
    let original_data_len = raw_data.len();
    if let Some(ref storage) = dense_storage {
        raw_data.resize(raw_data.len() + storage.dblk_size, 0);
    }

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
    } else if use_dense_ds {
        // Dense attrs on dataset: mimic C library's incremental OHDR growth.
        // Build messages WITHOUT inline attrs (only base + AINFO).
        let messages = build_dataset_messages_dummy(
            ds,
            opts,
            0xDEAD_BEEF_0000_0000,
            committed_type_index.is_some(),
        )?;
        // base_msgs = all messages EXCEPT AttributeInfo and inline attrs
        let base_msgs: usize = messages
            .iter()
            .filter(|(t, _, _)| {
                *t != MessageType::AttributeInfo.as_u8() && *t != MessageType::Attribute.as_u8()
            })
            .map(|(_, b, _)| 4 + b.len())
            .sum();
        let max_compact = ds.attr_phase_change.map(|(mc, _)| mc as usize).unwrap_or(8);
        let inline_attr_bytes: usize = ds
            .attributes
            .iter()
            .take(max_compact)
            .map(|a| 4 + encode_attribute(a).map(|b| b.len()).unwrap_or(0))
            .sum();
        let attr_info_size = messages
            .iter()
            .find(|(t, _, _)| *t == MessageType::AttributeInfo.as_u8())
            .map(|(_, b, _)| 4 + b.len())
            .unwrap_or(22);
        let target = 256usize; // H5D_MINHDR_SIZE
        let chunk_for_overhead = target.max(base_msgs + inline_attr_bytes) + attr_info_size;
        let tc = Some(chunk_for_overhead);
        // Dense datasets with default phase change (8,6) don't add phase_change to OHDR prefix
        let is_default_phase = ds.attr_phase_change == Some((8, 6));
        let phase_extra = if !is_default_phase && ds.attr_phase_change.is_some() {
            4
        } else {
            0
        };
        (tc, ohdr_overhead(chunk_for_overhead, opts) + phase_extra)
    } else {
        let messages = build_dataset_messages_dummy(
            ds,
            opts,
            0xDEAD_BEEF_0000_0000,
            committed_type_index.is_some(),
        )?;
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
    let mut index_meta_size = chunked_info
        .as_ref()
        .map(|ci| compute_index_meta_size(ci))
        .unwrap_or(0);
    // Dense attr metadata (FRHP+BTHD+FSHD+BTLF+FSSE) placed after OHDR.
    index_meta_size += dense_meta_size;

    // Build map of committed type names → object indices for shared-type attributes.
    let mut attr_ct_map = std::collections::HashMap::new();
    for attr in &ds.attributes {
        if let Some(ref ct_name) = attr.committed_type_name {
            if let Some(&ct_idx) = committed_types.get(ct_name) {
                attr_ct_map.insert(ct_name.clone(), ct_idx);
            }
        }
    }

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
            committed_type_index,
            attr_committed_types: attr_ct_map,
            attr_phase_change: ds.attr_phase_change,
            dense_attr_storage: dense_storage,
            original_data_len,
        }),
        ohdr_size,
        meta_addr: 0,
        data: raw_data,
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: target_chunk,
        index_meta_size,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
    });

    Ok(idx)
}

fn flatten_committed_type(
    dt: &crate::datatype::Datatype,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let idx = objects.len();

    // Committed type OHDR: continuation message (20 bytes) + gap in a small chunk.
    // The C library allocates a chunk sized for the datatype msg + 6 bytes extra.
    let dt_bytes = encode_datatype(dt)?;
    let committed_chunk_size = (4 + dt_bytes.len()) + 6;
    let ohdr_size = ohdr_overhead(committed_chunk_size, opts);

    objects.push(ObjectInfo {
        kind: ObjectKind::CommittedDatatype(CommittedTypeRef {
            datatype: dt.clone(),
            ochk_index: None,
        }),
        ohdr_size,
        meta_addr: 0,
        data: vec![],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: Some(committed_chunk_size),
        index_meta_size: 0,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
    });

    Ok(idx)
}

fn flatten_ochk(
    dt: &crate::datatype::Datatype,
    refcount: u32,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let idx = objects.len();

    let dt_bytes = encode_datatype(dt)?;
    // OCHK: sig(4) + dt_msg(4+dt_size) + refcount_msg(4+5) + checksum(4)
    let ochk_size = 4 + (4 + dt_bytes.len()) + (4 + 5) + 4;

    objects.push(ObjectInfo {
        kind: ObjectKind::Continuation(ContinuationRef {
            datatype: dt.clone(),
            refcount,
        }),
        ohdr_size: ochk_size,
        meta_addr: 0,
        data: vec![],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
        index_meta_size: 0,
        btree_v2_nodes_start: UNDEF_ADDR,
        btree_v2_num_nodes: 0,
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
                ndims: ds.shape.len(),
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
            ndims: ds.shape.len(),
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
        ObjectKind::Group(g) if g.link_creation_order => {
            // Creation-order group OHDR: CONT + NIL + GroupInfo, 6-byte headers.
            encode_creation_order_ohdr(obj, g, opts)
        }
        ObjectKind::Group(g) if !g.link_creation_order && !g.crt_order_child_names.is_empty() => {
            // Dense-link group without creation order: LinkInfo + GroupInfo only.
            encode_dense_link_ohdr(obj, g, opts)
        }
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
            if g.dense_attr_storage.is_some() {
                // Dense attribute storage: AttrInfo pointing to FRHP + BTHD
                // which are placed right after this OHDR.
                let frhp_addr = obj.meta_addr + obj.ohdr_size as u64;
                let sizes = crate::writer::dense_attrs::compute_dense_attr_sizes(
                    g.dense_attr_storage.as_ref().unwrap(),
                );
                let bthd_addr = frhp_addr + sizes.frhp_size as u64;
                let attr_info_body = encode_attribute_info_addrs(frhp_addr, bthd_addr);
                messages.push((
                    MessageType::AttributeInfo.as_u8(),
                    attr_info_body,
                    0x04, // not-shareable
                ));
            } else {
                for attr in &g.attributes {
                    messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
                }
            }

            encode_object_header_with_phase_change(
                &messages,
                opts,
                target_chunk,
                g.attr_phase_change,
            )
        }
        ObjectKind::Dataset(d) => {
            use crate::writer::types::SpaceAllocTime;

            // Datatype: shared reference if committed, else inline.
            let (dt_body, dt_flags) = if let Some(ct_idx) = d.committed_type_index {
                let ct_addr = objects[ct_idx].meta_addr;
                let mut body = Vec::with_capacity(10);
                body.push(2); // shared message version
                body.push(2); // type = object in file by address
                body.extend_from_slice(&ct_addr.to_le_bytes());
                (body, 0x03u8) // constant + shared
            } else {
                (encode_datatype(&d.datatype)?, 0x01u8)
            };

            // Always include max_dims in compat mode.
            let effective_max = Some(d.max_dims.clone().unwrap_or_else(|| d.shape.clone()));
            let ds_body = encode_dataspace(&d.shape, effective_max.as_deref());

            // Fill value allocation time depends on layout type.
            let fv_body = if d.vlen_elements.is_some() {
                // Vlen datasets: alloc_time=late, fill_write_time=on_alloc (flags=0x02)
                vec![3, 0x02]
            } else {
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
                encode_fill_value_msg_alloc(&d.fill_value, alloc_time)
            };

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
                    encode_chunked_layout_v3(&ci.chunk_dims_with_elem, chunk_index_addr)
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
                let (data_addr, data_size) = if let Some(ref vlen_elems) = d.vlen_elements {
                    if obj.data.is_empty() {
                        (UNDEF_ADDR, 0u64)
                    } else {
                        let heap_id_bytes = (vlen_elems.len() * 16) as u64;
                        (obj.data_addr, heap_id_bytes)
                    }
                } else if obj.data.is_empty() {
                    (UNDEF_ADDR, 0u64)
                } else {
                    // Use original_data_len to exclude FHDB from contiguous layout size.
                    (obj.data_addr, d.original_data_len as u64)
                };
                (encode_contiguous_layout(data_addr, data_size), None)
            };

            let mut messages: Vec<OhdrMsg> = vec![
                (MessageType::Dataspace.as_u8(), ds_body, 0),
                (MessageType::Datatype.as_u8(), dt_body, dt_flags),
                (MessageType::FillValue.as_u8(), fv_body, 0x01),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb, 0x01));
            }
            let layout_msg_flags = if d.chunked_info.as_ref().is_some_and(|ci| ci.early_alloc) {
                0x01 // constant
            } else {
                0x00
            };
            messages.push((
                MessageType::DataLayout.as_u8(),
                layout_body,
                layout_msg_flags,
            ));

            if d.dense_attr_storage.is_some() {
                // Dense attribute storage: AttrInfo pointing to FRHP + BTHD
                // placed right after this OHDR.
                let frhp_addr = obj.meta_addr + obj.ohdr_size as u64;
                let storage = d.dense_attr_storage.as_ref().unwrap();
                let sizes = crate::writer::dense_attrs::compute_dense_attr_sizes(storage);
                let bthd_addr = frhp_addr + sizes.frhp_size as u64;
                let attr_info_body = encode_attribute_info_addrs(frhp_addr, bthd_addr);
                messages.push((
                    MessageType::AttributeInfo.as_u8(),
                    attr_info_body,
                    0x04, // not-shareable
                ));
            } else {
                let mut attr_bodies: Vec<Vec<u8>> = Vec::new();
                for attr in &d.attributes {
                    if let Some(ref ct_name) = attr.committed_type_name {
                        if let Some(&ct_idx) = d.attr_committed_types.get(ct_name) {
                            let ct_addr = objects[ct_idx].meta_addr;
                            attr_bodies.push(encode_attribute_shared(attr, ct_addr)?);
                        } else {
                            attr_bodies.push(encode_attribute(attr)?);
                        }
                    } else {
                        attr_bodies.push(encode_attribute(attr)?);
                    }
                }
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
            }

            // For dense datasets with non-default phase change, use phase change header.
            let is_default_phase = d.attr_phase_change == Some((8, 6));
            if d.attr_phase_change.is_some() && !is_default_phase {
                encode_object_header_with_phase_change(
                    &messages,
                    opts,
                    target_chunk,
                    d.attr_phase_change,
                )
            } else {
                encode_object_header(&messages, opts, target_chunk)
            }
        }
        ObjectKind::CommittedDatatype(ct) => {
            // OHDR with a continuation message pointing to the OCHK.
            let ochk_addr = ct
                .ochk_index
                .map(|i| objects[i].meta_addr)
                .unwrap_or(UNDEF_ADDR);
            let ochk_size = ct
                .ochk_index
                .map(|i| objects[i].ohdr_size as u64)
                .unwrap_or(0);

            let mut cont_body = Vec::with_capacity(16);
            cont_body.extend_from_slice(&ochk_addr.to_le_bytes());
            cont_body.extend_from_slice(&ochk_size.to_le_bytes());

            let messages: Vec<OhdrMsg> = vec![(
                MessageType::ObjectHeaderContinuation.as_u8(),
                cont_body,
                0x00,
            )];

            encode_object_header(&messages, opts, target_chunk)
        }
        ObjectKind::Continuation(cont) => {
            // OCHK: sig(4) + datatype_msg + refcount_msg + checksum(4)
            let dt_bytes = encode_datatype(&cont.datatype)?;

            let mut buf = Vec::new();
            buf.extend_from_slice(b"OCHK");

            // Datatype message: type=0x03, flags=0x05 (constant + not-shareable)
            buf.push(MessageType::Datatype.as_u8());
            buf.extend_from_slice(&(dt_bytes.len() as u16).to_le_bytes());
            buf.push(0x05);
            buf.extend_from_slice(&dt_bytes);

            // Object Reference Count message: type=0x16, flags=0x04 (not-shareable)
            buf.push(0x16);
            buf.extend_from_slice(&5u16.to_le_bytes());
            buf.push(0x04);
            buf.push(0x00); // version
            buf.extend_from_slice(&cont.refcount.to_le_bytes());

            let cs = crate::checksum::lookup3(&buf);
            buf.extend_from_slice(&cs.to_le_bytes());

            Ok(buf)
        }
    }
}

/// Encode the OHDR for a creation-order group.
///
/// The C library's allocation history produces: CONT + NIL + GroupInfo
/// (LinkInfo was replaced with CONT+NIL; GroupInfo stays at its original position).
/// chunk_size = 52, flags = 0x3C.
/// Encode OHDR for a dense-link group without creation order.
///
/// Messages: LinkInfo(18 body) + GroupInfo(6 body, with link phase change).
/// 4-byte message headers. chunk0 = 32.
fn encode_dense_link_ohdr(obj: &ObjectInfo, g: &GroupRef, opts: &WriteOptions) -> Result<Vec<u8>> {
    use crate::writer::encode::encode_group_info_with_link_phase;
    use crate::writer::encode::encode_link_info_dense;

    let base_addr = obj.meta_addr + obj.ohdr_size as u64;
    let child_target = compat_group_chunk_size(4, 8);
    let child_ohdr_size = ohdr_overhead(child_target, opts);

    // FRHP is after child_0 OHDR.
    let frhp_addr = base_addr + child_ohdr_size as u64;
    let bthd_addr = frhp_addr + 146; // FRHP size

    let link_info_body = encode_link_info_dense(frhp_addr, bthd_addr);
    let (max_compact, min_dense) = g.link_phase_change.unwrap_or((0, 0));
    let group_info_body = encode_group_info_with_link_phase(max_compact, min_dense);

    let chunk_size: usize = 4 + link_info_body.len() + 4 + group_info_body.len(); // 32

    let mut buf = Vec::with_capacity(59);

    // Prefix: OHDR + ver + flags + timestamps + chunk_size
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0x20); // flags: bit 5 = timestamps

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    buf.push(chunk_size as u8); // 1-byte chunk size (< 256)

    // Message 1: LinkInfo (4-byte header + 18-byte body)
    buf.push(MessageType::LinkInfo.as_u8());
    buf.extend_from_slice(&(link_info_body.len() as u16).to_le_bytes());
    buf.push(0x00); // flags
    buf.extend_from_slice(&link_info_body);

    // Message 2: GroupInfo (4-byte header + 6-byte body)
    buf.push(MessageType::GroupInfo.as_u8());
    buf.extend_from_slice(&(group_info_body.len() as u16).to_le_bytes());
    buf.push(0x01); // constant flag
    buf.extend_from_slice(&group_info_body);

    // Checksum
    let cksum = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    debug_assert_eq!(buf.len(), 59);
    Ok(buf)
}

fn encode_creation_order_ohdr(
    obj: &ObjectInfo,
    g: &GroupRef,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    use crate::writer::dense_attrs::compute_dense_attr_crt_sizes;
    use crate::writer::dense_attrs::compute_dense_link_sizes;
    use crate::writer::encode::encode_group_info_with_link_phase;

    let n_children = g.crt_order_child_names.len();
    let child_ohdr_size = {
        let child_target = compat_group_chunk_size(4, 8);
        ohdr_overhead(child_target, opts)
    };

    let link_sizes = compute_dense_link_sizes(g.dense_link_storage.as_ref().unwrap());
    let attr_crt_sizes = compute_dense_attr_crt_sizes(g.dense_attr_storage.as_ref().unwrap());

    // Compute OCHK address within the interleaved layout (link meta excludes FSSE).
    let first_child = child_ohdr_size;
    let link_meta_no_fsse = link_sizes.frhp_size
        + link_sizes.bthd_name_size
        + link_sizes.bthd_crt_size
        + link_sizes.fshd_size
        + link_sizes.btlf_name_size
        + link_sizes.btlf_crt_size;
    let remaining_children = (n_children - 1) * child_ohdr_size;
    let attr_part1 =
        attr_crt_sizes.frhp_size + attr_crt_sizes.bthd_name_size + attr_crt_sizes.bthd_crt_size;
    let ochk_offset =
        obj.ohdr_size + first_child + link_meta_no_fsse + remaining_children + attr_part1;
    let ochk_addr = obj.meta_addr + ochk_offset as u64;
    let ochk_size = 82u64;

    // Build the OHDR manually with the exact C library message layout:
    // CONT(6+16=22) + NIL(6+12=18) + GroupInfo(6+6=12) = 52 bytes of messages.
    let chunk_size: usize = 52;
    let (max_compact, min_dense) = g.attr_phase_change.unwrap_or((0, 0));

    let mut buf = Vec::with_capacity(83);

    // Prefix: OHDR + ver + flags + timestamps + phase_change + chunk_size
    buf.extend_from_slice(b"OHDR");
    buf.push(2); // version
    buf.push(0x3C); // flags: bits 2,3,4,5

    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    buf.extend_from_slice(&max_compact.to_le_bytes());
    buf.extend_from_slice(&min_dense.to_le_bytes());
    buf.push(chunk_size as u8); // chunk_size < 256

    // Message 1: CONT (6-byte header + 16-byte body)
    buf.push(MessageType::ObjectHeaderContinuation.as_u8());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.push(0x00); // flags
    buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order
    buf.extend_from_slice(&ochk_addr.to_le_bytes());
    buf.extend_from_slice(&ochk_size.to_le_bytes());

    // Message 2: NIL (6-byte header + 12-byte body)
    buf.push(0x00); // NIL type
    buf.extend_from_slice(&12u16.to_le_bytes());
    buf.push(0x00);
    buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order
    buf.resize(buf.len() + 12, 0); // 12 zero bytes

    // Message 3: GroupInfo (6-byte header + 6-byte body)
    let group_info_body = encode_group_info_with_link_phase(
        g.link_phase_change.map(|(mc, _)| mc).unwrap_or(0),
        g.link_phase_change.map(|(_, md)| md).unwrap_or(0),
    );
    buf.push(MessageType::GroupInfo.as_u8());
    buf.extend_from_slice(&(group_info_body.len() as u16).to_le_bytes());
    buf.push(0x01); // constant flag
    buf.extend_from_slice(&0u16.to_le_bytes()); // creation_order
    buf.extend_from_slice(&group_info_body);

    // Checksum
    let cksum = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    debug_assert_eq!(buf.len(), 83);
    Ok(buf)
}

/// Encode the full interleaved index metadata for a creation-order group.
///
/// Layout:
///   child0_ohdr | link_dense_meta | child1..N_ohdrs | attr_part1 | ochk | attr_part2 | fsse_link | fsse_attr
fn encode_creation_order_index_meta(
    obj: &ObjectInfo,
    g: &GroupRef,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    use crate::writer::dense_attrs::compute_dense_attr_crt_sizes;
    use crate::writer::dense_attrs::compute_dense_link_sizes;
    use crate::writer::dense_attrs::compute_dense_link_storage;
    use crate::writer::dense_attrs::encode_attr_crt_fsse;
    use crate::writer::dense_attrs::encode_dense_attr_crt_part1;
    use crate::writer::dense_attrs::encode_dense_attr_crt_part2;
    use crate::writer::dense_attrs::encode_dense_link_meta;
    use crate::writer::dense_attrs::encode_link_fsse;
    use crate::writer::encode::encode_attribute_info_dense_crt;
    use crate::writer::encode::encode_link_info_dense_crt;
    use crate::writer::encode::encode_ochk_creation_order;

    let n_children = g.crt_order_child_names.len();
    let child_ohdr_size = {
        let child_target = compat_group_chunk_size(4, 8);
        ohdr_overhead(child_target, opts)
    };

    let attr_storage = g.dense_attr_storage.as_ref().unwrap();
    let link_sizes = compute_dense_link_sizes(g.dense_link_storage.as_ref().unwrap());
    let attr_crt_sizes = compute_dense_attr_crt_sizes(attr_storage);

    // Compute section offsets (from start of index_meta, i.e. after ordered OHDR).
    let base_addr = obj.meta_addr + obj.ohdr_size as u64;

    let link_meta_no_fsse = link_sizes.frhp_size
        + link_sizes.bthd_name_size
        + link_sizes.bthd_crt_size
        + link_sizes.fshd_size
        + link_sizes.btlf_name_size
        + link_sizes.btlf_crt_size;

    let first_child_offset = 0usize;
    let link_meta_offset = first_child_offset + child_ohdr_size;
    let remaining_children_offset = link_meta_offset + link_meta_no_fsse;
    let remaining_children_size = (n_children - 1) * child_ohdr_size;
    let attr_part1_offset = remaining_children_offset + remaining_children_size;
    let attr_part1_size =
        attr_crt_sizes.frhp_size + attr_crt_sizes.bthd_name_size + attr_crt_sizes.bthd_crt_size;
    let ochk_offset = attr_part1_offset + attr_part1_size;
    let ochk_size = 82usize;
    let attr_part2_offset = ochk_offset + ochk_size;
    let attr_part2_size =
        attr_crt_sizes.fshd_size + attr_crt_sizes.btlf_name_size + attr_crt_sizes.btlf_crt_size;
    let fsse_link_offset = attr_part2_offset + attr_part2_size;
    let fsse_attr_offset = fsse_link_offset + link_sizes.fsse_size;

    // Compute addresses.
    let child_addrs: Vec<u64> = (0..n_children)
        .map(|i| {
            if i == 0 {
                base_addr + first_child_offset as u64
            } else {
                base_addr + remaining_children_offset as u64 + ((i - 1) * child_ohdr_size) as u64
            }
        })
        .collect();

    let link_meta_addr = base_addr + link_meta_offset as u64;
    let attr_frhp_addr = base_addr + attr_part1_offset as u64;
    let attr_fshd_addr = base_addr + attr_part2_offset as u64;
    let fsse_link_addr = base_addr + fsse_link_offset as u64;
    let fsse_attr_addr = base_addr + fsse_attr_offset as u64;

    // FHDB addresses (in data region).
    let fhdb_attr_addr = obj.data_addr;
    let fhdb_link_addr = obj.data_addr + attr_crt_sizes.fhdb_size as u64;

    // Recompute dense link storage with real child addresses.
    let link_storage = compute_dense_link_storage(&g.crt_order_child_names, &child_addrs);

    // Link FSHD address within link_meta.
    let link_fshd_addr = link_meta_addr
        + link_sizes.frhp_size as u64
        + link_sizes.bthd_name_size as u64
        + link_sizes.bthd_crt_size as u64;

    // --- Build the output buffer ---
    let total_size = obj.index_meta_size;
    let mut buf = vec![0u8; total_size];

    // 1. First child OHDR
    let child0_ohdr = encode_empty_group_ohdr(opts)?;
    buf[first_child_offset..first_child_offset + child_ohdr_size].copy_from_slice(&child0_ohdr);

    // 2. Link dense meta (FRHP + BTHD_name + BTHD_crt + FSHD + BTLF_name + BTLF_crt)
    let link_meta_bytes = encode_dense_link_meta(
        &link_storage,
        link_meta_addr,
        fhdb_link_addr,
        fsse_link_addr,
    )?;
    buf[link_meta_offset..link_meta_offset + link_meta_bytes.len()]
        .copy_from_slice(&link_meta_bytes);

    // 3. Remaining child OHDRs
    for i in 1..n_children {
        let child_ohdr = encode_empty_group_ohdr(opts)?;
        let off = remaining_children_offset + (i - 1) * child_ohdr_size;
        buf[off..off + child_ohdr_size].copy_from_slice(&child_ohdr);
    }

    // 4. Attr dense part1 (FRHP + BTHD_name + BTHD_crt)
    let attr_part1_bytes = encode_dense_attr_crt_part1(
        attr_storage,
        attr_frhp_addr,
        fhdb_attr_addr,
        attr_fshd_addr,
        fsse_attr_addr,
    )?;
    buf[attr_part1_offset..attr_part1_offset + attr_part1_bytes.len()]
        .copy_from_slice(&attr_part1_bytes);

    // 5. OCHK (LinkInfo + AttrInfo)
    // LinkInfo addresses: link FRHP, link BTHD_name, link BTHD_crt
    let link_frhp_addr = link_meta_addr;
    let link_bthd_name_addr = link_frhp_addr + link_sizes.frhp_size as u64;
    let link_bthd_crt_addr = link_bthd_name_addr + link_sizes.bthd_name_size as u64;
    let link_info_body = encode_link_info_dense_crt(
        link_frhp_addr,
        link_bthd_name_addr,
        link_bthd_crt_addr,
        n_children as u64,
    );
    // AttrInfo addresses: attr FRHP, attr BTHD_name, attr BTHD_crt
    let attr_bthd_name_addr = attr_frhp_addr + attr_crt_sizes.frhp_size as u64;
    let attr_bthd_crt_addr = attr_bthd_name_addr + attr_crt_sizes.bthd_name_size as u64;
    let attr_info_body = encode_attribute_info_dense_crt(
        attr_frhp_addr,
        attr_bthd_name_addr,
        attr_bthd_crt_addr,
        g.attributes.len() as u16,
    );
    let ochk_messages: Vec<OhdrMsg> = vec![
        (MessageType::LinkInfo.as_u8(), link_info_body, 0x00),
        (
            MessageType::AttributeInfo.as_u8(),
            attr_info_body,
            0x04, // not-shareable
        ),
    ];
    let ochk_bytes = encode_ochk_creation_order(&ochk_messages);
    buf[ochk_offset..ochk_offset + ochk_bytes.len()].copy_from_slice(&ochk_bytes);

    // 6. Attr dense part2 (FSHD + BTLF_name + BTLF_crt)
    let attr_part2_bytes =
        encode_dense_attr_crt_part2(attr_storage, attr_fshd_addr, fsse_attr_addr)?;
    buf[attr_part2_offset..attr_part2_offset + attr_part2_bytes.len()]
        .copy_from_slice(&attr_part2_bytes);

    // 7. FSSE links
    let fsse_link_bytes = encode_link_fsse(&link_storage, link_fshd_addr);
    buf[fsse_link_offset..fsse_link_offset + fsse_link_bytes.len()]
        .copy_from_slice(&fsse_link_bytes);

    // 8. FSSE attrs
    let fsse_attr_bytes = encode_attr_crt_fsse(attr_storage, attr_fshd_addr);
    buf[fsse_attr_offset..fsse_attr_offset + fsse_attr_bytes.len()]
        .copy_from_slice(&fsse_attr_bytes);

    Ok(buf)
}

/// Encode a standard empty group OHDR (with timestamps, for creation-order children).
fn encode_empty_group_ohdr(opts: &WriteOptions) -> Result<Vec<u8>> {
    let messages: Vec<OhdrMsg> = vec![
        (MessageType::LinkInfo.as_u8(), encode_link_info(), 0),
        (MessageType::GroupInfo.as_u8(), encode_group_info(), 0x01),
    ];
    let target = compat_group_chunk_size(4, 8);
    encode_object_header(&messages, opts, Some(target))
}

/// Compute child OHDR addresses for a creation-order group.
fn compute_crt_order_child_addrs(obj: &ObjectInfo, g: &GroupRef, opts: &WriteOptions) -> Vec<u64> {
    let child_ohdr_size = {
        let child_target = compat_group_chunk_size(4, 8);
        ohdr_overhead(child_target, opts)
    };
    let link_sizes = crate::writer::dense_attrs::compute_dense_link_sizes(
        g.dense_link_storage.as_ref().unwrap(),
    );
    let n = g.crt_order_child_names.len();
    let base_addr = obj.meta_addr + obj.ohdr_size as u64;

    let link_meta_no_fsse = (link_sizes.frhp_size
        + link_sizes.bthd_name_size
        + link_sizes.bthd_crt_size
        + link_sizes.fshd_size
        + link_sizes.btlf_name_size
        + link_sizes.btlf_crt_size) as u64;
    let first_child_offset = 0u64;
    let link_meta_offset = child_ohdr_size as u64;
    let remaining_children_offset = link_meta_offset + link_meta_no_fsse;

    (0..n)
        .map(|i| {
            if i == 0 {
                base_addr + first_child_offset
            } else {
                base_addr + remaining_children_offset + ((i - 1) * child_ohdr_size) as u64
            }
        })
        .collect()
}

/// Encode FHDBs (fractal heap direct blocks) for the data region of a creation-order group.
/// Returns FHDB_attrs + FHDB_links concatenated.
fn encode_creation_order_fhdbs(
    obj: &ObjectInfo,
    g: &GroupRef,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    use crate::writer::dense_attrs::compute_dense_attr_crt_sizes;
    use crate::writer::dense_attrs::compute_dense_link_sizes;
    use crate::writer::dense_attrs::compute_dense_link_storage;
    use crate::writer::dense_attrs::encode_fhdb_standalone;
    use crate::writer::dense_attrs::encode_link_fhdb;

    let attr_storage = g.dense_attr_storage.as_ref().unwrap();
    let attr_crt_sizes = compute_dense_attr_crt_sizes(attr_storage);
    let link_sizes = compute_dense_link_sizes(g.dense_link_storage.as_ref().unwrap());
    let child_addrs = compute_crt_order_child_addrs(obj, g, opts);
    let child_ohdr_size = {
        let child_target = compat_group_chunk_size(4, 8);
        ohdr_overhead(child_target, opts)
    };
    let n_children = g.crt_order_child_names.len();

    // Attr FRHP addr: base + first_child + link_meta(no fsse) + remaining_children
    let link_meta_no_fsse = (link_sizes.frhp_size
        + link_sizes.bthd_name_size
        + link_sizes.bthd_crt_size
        + link_sizes.fshd_size
        + link_sizes.btlf_name_size
        + link_sizes.btlf_crt_size) as u64;
    let base_addr = obj.meta_addr + obj.ohdr_size as u64;
    let attr_frhp_addr = base_addr
        + child_ohdr_size as u64
        + link_meta_no_fsse
        + ((n_children - 1) * child_ohdr_size) as u64;
    let fhdb_attr = encode_fhdb_standalone(attr_storage, attr_frhp_addr);

    // Link FRHP addr: base + first_child
    let link_frhp_addr = base_addr + child_ohdr_size as u64;
    let link_storage = compute_dense_link_storage(&g.crt_order_child_names, &child_addrs);
    let fhdb_link = encode_link_fhdb(&link_storage, link_frhp_addr);

    let mut buf = Vec::with_capacity(fhdb_attr.len() + fhdb_link.len());
    buf.extend_from_slice(&fhdb_attr);
    buf.extend_from_slice(&fhdb_link);
    Ok(buf)
}

/// Encode the index_meta region for a dense-link group (no creation order).
///
/// Layout: child_0 OHDR | FRHP | BTHD | FSHD | BTLF | children 1..N OHDRs | FSSE
fn encode_dense_link_index_meta(
    obj: &ObjectInfo,
    g: &GroupRef,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    use crate::writer::dense_attrs::encode_link_body_no_crt;
    use crate::writer::dense_attrs::limit_enc_size_u64_pub;

    let n_children = g.crt_order_child_names.len();
    let child_target = compat_group_chunk_size(4, 8);
    let child_ohdr_size = ohdr_overhead(child_target, opts);

    let base_addr = obj.meta_addr + obj.ohdr_size as u64;
    let start_block_size = 512usize;
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8); // 4
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4; // 21
    let heap_id_len: u16 = 7;
    let node_size = 512usize;
    let rec_size: u16 = 4 + heap_id_len; // 11
    let frhp_size = 146usize;
    let bthd_size = 38usize;
    let fshd_size = 82usize;

    // Compute child addresses.
    let child_addrs: Vec<u64> = (0..n_children)
        .map(|i| {
            if i == 0 {
                base_addr // child_0 is first in index_meta
            } else {
                // children 1..N are after BTLF
                let offset_in_meta = child_ohdr_size // child_0
                    + frhp_size + bthd_size + fshd_size + node_size // FRHP+BTHD+FSHD+BTLF
                    + (i - 1) * child_ohdr_size;
                base_addr + offset_in_meta as u64
            }
        })
        .collect();

    // Build link bodies and hashes.
    let link_bodies: Vec<Vec<u8>> = g
        .crt_order_child_names
        .iter()
        .zip(child_addrs.iter())
        .map(|(name, addr)| encode_link_body_no_crt(name, *addr))
        .collect();
    let total_link_bytes: usize = link_bodies.iter().map(|b| b.len()).sum();
    let name_hashes: Vec<u32> = g
        .crt_order_child_names
        .iter()
        .map(|n| crate::checksum::lookup3(n.as_bytes()))
        .collect();

    // Compute doubling table.
    let dtable = crate::writer::dense_attrs::compute_doubling_table_pub(
        total_link_bytes,
        start_block_size,
        dblk_header_size,
    );
    let n_blocks = dtable.block_sizes.len();
    let nrows = dtable.nrows;

    // Distribute links across blocks.
    let mut block_bodies: Vec<Vec<usize>> = vec![Vec::new(); n_blocks]; // indices into link_bodies
    let mut block_used: Vec<usize> = vec![dblk_header_size; n_blocks];
    let mut block_idx = 0;
    for (li, body) in link_bodies.iter().enumerate() {
        let bsz = dtable.block_sizes[block_idx];
        if block_used[block_idx] + body.len() > bsz && block_idx + 1 < n_blocks {
            block_idx += 1;
        }
        block_bodies[block_idx].push(li);
        block_used[block_idx] += body.len();
    }

    // Compute heap offsets for each link.
    let mut heap_offsets: Vec<usize> = vec![0; n_children];
    for bi in 0..n_blocks {
        let blk_heap_off = dtable.block_offsets[bi] as usize;
        let mut off_in_blk = dblk_header_size;
        for &li in &block_bodies[bi] {
            heap_offsets[li] = blk_heap_off + off_in_blk;
            off_in_blk += link_bodies[li].len();
        }
    }

    // Build heap IDs (7 bytes each).
    let mut heap_ids: Vec<[u8; 7]> = Vec::with_capacity(n_children);
    for i in 0..n_children {
        let off = heap_offsets[i] as u64;
        let len = link_bodies[i].len() as u64;
        let mut hid = [0u8; 7];
        hid[0] = 0x00;
        let packed = off | (len << 32);
        hid[1..].copy_from_slice(&packed.to_le_bytes()[..6]);
        heap_ids.push(hid);
    }

    // Sort records by hash for B-tree.
    let mut name_records: Vec<(u32, usize)> = name_hashes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, h)| (h, i))
        .collect();
    name_records.sort_by_key(|&(hash, _)| hash);

    // Compute free space per block.
    let mut block_free: Vec<(usize, usize)> = Vec::new();
    for bi in 0..n_blocks {
        let bsz = dtable.block_sizes[bi];
        let free = bsz - block_used[bi];
        if free > 0 {
            block_free.push((dtable.block_offsets[bi] as usize + block_used[bi], free));
        }
    }
    let total_free: u64 = block_free.iter().map(|(_, f)| *f as u64).sum();
    let n_sections = block_free.len();
    // FSSE sizing.
    let sect_off_size = 4usize;
    let sect_len_size = limit_enc_size_u64_pub(65536u64).max(1); // max_sect_size from FSHD
    let sect_cnt_size = limit_enc_size_u64_pub(n_sections as u64).max(1);
    let mut size_groups: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for &(addr, size) in &block_free {
        size_groups.entry(size).or_default().push(addr);
    }
    let mut fsse_data_len = 0usize;
    for (_, addrs) in &size_groups {
        fsse_data_len += sect_cnt_size + sect_len_size + addrs.len() * (sect_off_size + 1);
    }
    let fsse_size = 4 + 1 + 8 + fsse_data_len + 4;

    // Addresses in the file.
    let frhp_addr = base_addr + child_ohdr_size as u64;
    let bthd_addr = frhp_addr + frhp_size as u64;
    let fshd_addr = bthd_addr + bthd_size as u64;
    let btlf_addr = fshd_addr + fshd_size as u64;
    let remaining_start = btlf_addr + node_size as u64;
    let fsse_addr = remaining_start + ((n_children - 1) * child_ohdr_size) as u64;

    let fhib_data_offset = dtable.block_sizes.iter().sum::<usize>();
    let fhib_file_addr = obj.data_addr + fhib_data_offset as u64;

    // Build FHDB file addresses (reverse block_offset order).
    let mut fhdb_file_order: Vec<usize> = (0..n_blocks).collect();
    fhdb_file_order.reverse();
    let mut fhdb_file_addrs: Vec<u64> = vec![0; n_blocks];
    let mut file_off = obj.data_addr;
    for &bi in &fhdb_file_order {
        fhdb_file_addrs[bi] = file_off;
        file_off += dtable.block_sizes[bi] as u64;
    }

    // === Encode FRHP ===
    // FRHP free_managed counts free across ALL managed space (including unallocated slots).
    let all_headers = (nrows * dtable.width * dblk_header_size) as u64;
    let frhp_free_managed = dtable.total_managed_space - total_link_bytes as u64 - all_headers;
    let frhp = crate::writer::dense_attrs::encode_frhp_indirect_pub(
        heap_id_len,
        max_heap_size_bits,
        n_children as u64,
        frhp_free_managed,
        fshd_addr,
        dtable.total_managed_space,
        dtable.total_alloc,
        dtable.total_alloc, // iter_offset
        start_block_size as u64,
        fhib_file_addr,
        nrows as u16,
    );

    // === Encode BTHD ===
    let bthd = crate::writer::dense_attrs::encode_bthd_pub(
        btlf_addr,
        n_children as u16,
        n_children as u64,
        rec_size,
        node_size as u32,
        5, // type 5 = name hash
        0, // depth 0
    );

    // === Encode FSHD ===
    let fshd = crate::writer::dense_attrs::encode_fshd_multi_pub(
        total_free,
        n_sections as u64,
        fsse_addr,
        fsse_size as u64,
        max_heap_size_bits,
    );

    // === Encode BTLF ===
    let btlf =
        crate::writer::dense_attrs::encode_btlf_type5_pub(&name_records, &heap_ids, node_size);

    // === Encode child OHDRs ===
    let child_ohdr_bytes = encode_empty_group_ohdr(opts)?;

    // === Encode FSSE ===
    let mut fsse_buf = vec![0u8; fsse_size];
    fsse_buf[0..4].copy_from_slice(b"FSSE");
    fsse_buf[4] = 0;
    fsse_buf[5..13].copy_from_slice(&fshd_addr.to_le_bytes());
    let mut fsse_off = 13;
    for (&size, addrs) in &size_groups {
        fsse_buf[fsse_off..fsse_off + sect_cnt_size]
            .copy_from_slice(&(addrs.len() as u64).to_le_bytes()[..sect_cnt_size]);
        fsse_off += sect_cnt_size;
        fsse_buf[fsse_off..fsse_off + sect_len_size]
            .copy_from_slice(&(size as u64).to_le_bytes()[..sect_len_size]);
        fsse_off += sect_len_size;
        let mut sorted_addrs = addrs.clone();
        sorted_addrs.sort();
        for &addr in &sorted_addrs {
            fsse_buf[fsse_off..fsse_off + sect_off_size]
                .copy_from_slice(&(addr as u64).to_le_bytes()[..sect_off_size]);
            fsse_off += sect_off_size;
            fsse_buf[fsse_off] = 0; // type SINGLE
            fsse_off += 1;
        }
    }
    let fsse_cksum = crate::checksum::lookup3(&fsse_buf[..fsse_size - 4]);
    fsse_buf[fsse_size - 4..].copy_from_slice(&fsse_cksum.to_le_bytes());

    // === Assemble index_meta ===
    let total_size = obj.index_meta_size;
    let mut buf = Vec::with_capacity(total_size);

    // child_0 OHDR
    buf.extend_from_slice(&child_ohdr_bytes);
    // FRHP + BTHD + FSHD + BTLF
    buf.extend_from_slice(&frhp);
    buf.extend_from_slice(&bthd);
    buf.extend_from_slice(&fshd);
    buf.extend_from_slice(&btlf);
    // children 1..N OHDRs
    for _ in 1..n_children {
        buf.extend_from_slice(&child_ohdr_bytes);
    }
    // FSSE
    buf.extend_from_slice(&fsse_buf);

    debug_assert_eq!(buf.len(), total_size);
    Ok(buf)
}

/// Encode the data region for a dense-link group: FHDBs (reverse order) + FHIB.
fn encode_dense_link_data(obj: &ObjectInfo, g: &GroupRef, opts: &WriteOptions) -> Result<Vec<u8>> {
    use crate::writer::dense_attrs::encode_link_body_no_crt;

    let n_children = g.crt_order_child_names.len();
    let child_target = compat_group_chunk_size(4, 8);
    let child_ohdr_size = ohdr_overhead(child_target, opts);

    let base_addr = obj.meta_addr + obj.ohdr_size as u64;
    let start_block_size = 512usize;
    let max_heap_size_bits: u16 = 32;
    let block_offset_bytes = (max_heap_size_bits as usize).div_ceil(8);
    let dblk_header_size = 4 + 1 + 8 + block_offset_bytes + 4;
    let frhp_size = 146usize;
    let bthd_size = 38usize;
    let fshd_size = 82usize;
    let node_size = 512usize;

    // Compute child addresses (same as index_meta).
    let child_addrs: Vec<u64> = (0..n_children)
        .map(|i| {
            if i == 0 {
                base_addr
            } else {
                let offset_in_meta = child_ohdr_size
                    + frhp_size
                    + bthd_size
                    + fshd_size
                    + node_size
                    + (i - 1) * child_ohdr_size;
                base_addr + offset_in_meta as u64
            }
        })
        .collect();

    let link_bodies: Vec<Vec<u8>> = g
        .crt_order_child_names
        .iter()
        .zip(child_addrs.iter())
        .map(|(name, addr)| encode_link_body_no_crt(name, *addr))
        .collect();
    let total_link_bytes: usize = link_bodies.iter().map(|b| b.len()).sum();

    let dtable = crate::writer::dense_attrs::compute_doubling_table_pub(
        total_link_bytes,
        start_block_size,
        dblk_header_size,
    );
    let n_blocks = dtable.block_sizes.len();
    let nrows = dtable.nrows;

    // Distribute links across blocks.
    let mut block_bodies: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    let mut block_used: Vec<usize> = vec![dblk_header_size; n_blocks];
    let mut block_idx = 0;
    for (li, body) in link_bodies.iter().enumerate() {
        let bsz = dtable.block_sizes[block_idx];
        if block_used[block_idx] + body.len() > bsz && block_idx + 1 < n_blocks {
            block_idx += 1;
        }
        block_bodies[block_idx].push(li);
        block_used[block_idx] += body.len();
    }

    let frhp_addr = base_addr + child_ohdr_size as u64;

    // FHDB file addresses (reverse block_offset order).
    let mut fhdb_file_order: Vec<usize> = (0..n_blocks).collect();
    fhdb_file_order.reverse();
    let mut fhdb_file_addrs: Vec<u64> = vec![0; n_blocks];
    let mut file_off = obj.data_addr;
    for &bi in &fhdb_file_order {
        fhdb_file_addrs[bi] = file_off;
        file_off += dtable.block_sizes[bi] as u64;
    }

    // Encode FHDBs in reverse block_offset order.
    let mut data_buf = Vec::new();
    for &bi in &fhdb_file_order {
        let bsz = dtable.block_sizes[bi];
        let mut blk = vec![0u8; bsz];
        blk[0..4].copy_from_slice(b"FHDB");
        blk[4] = 0; // version
        blk[5..13].copy_from_slice(&frhp_addr.to_le_bytes());
        let bo = dtable.block_offsets[bi];
        blk[13..13 + block_offset_bytes].copy_from_slice(&bo.to_le_bytes()[..block_offset_bytes]);

        // Place link bodies.
        let mut off_in_blk = dblk_header_size;
        for &li in &block_bodies[bi] {
            blk[off_in_blk..off_in_blk + link_bodies[li].len()].copy_from_slice(&link_bodies[li]);
            off_in_blk += link_bodies[li].len();
        }

        // Checksum over entire block.
        let cksum_off = 4 + 1 + 8 + block_offset_bytes;
        let cksum = crate::checksum::lookup3(&blk);
        blk[cksum_off..cksum_off + 4].copy_from_slice(&cksum.to_le_bytes());

        data_buf.extend_from_slice(&blk);
    }

    // Encode FHIB.
    let n_entries = nrows * dtable.width;
    let fhib_size = 4 + 1 + 8 + block_offset_bytes + n_entries * 8 + 4;
    let mut fhib = vec![0u8; fhib_size];
    fhib[0..4].copy_from_slice(b"FHIB");
    fhib[4] = 0; // version
    fhib[5..13].copy_from_slice(&frhp_addr.to_le_bytes());
    // block_offset = 0 (already zero)

    let mut off = 4 + 1 + 8 + block_offset_bytes;
    for entry_idx in 0..n_entries {
        if entry_idx < n_blocks {
            fhib[off..off + 8].copy_from_slice(&fhdb_file_addrs[entry_idx].to_le_bytes());
        } else {
            fhib[off..off + 8].copy_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
        off += 8;
    }
    let cksum = crate::checksum::lookup3(&fhib[..off]);
    fhib[off..off + 4].copy_from_slice(&cksum.to_le_bytes());

    data_buf.extend_from_slice(&fhib);
    Ok(data_buf)
}

fn build_group_messages_dummy(
    child_indices: &[(String, usize)],
    attributes: &[crate::writer::types::AttrData],
    _opts: &WriteOptions,
    dense_attrs: bool,
) -> Result<Vec<OhdrMsg>> {
    let mut messages: Vec<OhdrMsg> = Vec::new();
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
    messages.push((MessageType::GroupInfo.as_u8(), encode_group_info(), 0x01));
    for (name, _) in child_indices {
        messages.push((MessageType::Link.as_u8(), encode_link(name, 0xDEAD_0000), 0));
    }
    if dense_attrs {
        // Dense attributes: include AttrInfo message with dummy addresses.
        messages.push((
            MessageType::AttributeInfo.as_u8(),
            encode_attribute_info(),
            0x04,
        ));
    } else {
        for attr in attributes {
            messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
        }
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
        encode_chunked_layout_v3(&ci.chunk_dims_with_elem, 0xDEAD_BEEF_0000_0000)
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
    messages.push((
        MessageType::DataLayout.as_u8(),
        layout_body,
        layout_msg_flags,
    ));
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
    shared_type: bool,
) -> Result<Vec<OhdrMsg>> {
    let (dt_body, dt_flags) = if shared_type {
        // Shared type reference: version(1) + type(1) + addr(8) = 10 bytes
        (vec![0u8; 10], 0x03u8)
    } else {
        (encode_datatype(&ds.datatype)?, 0x01u8)
    };

    // Always include max_dims in compat mode.
    let effective_max = Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()));
    let ds_body = encode_dataspace(&ds.shape, effective_max.as_deref());
    let fv_body = encode_fill_value_msg(&ds.fill_value, true);

    let data_size = ds.data.len() as u64;
    let layout_body = encode_contiguous_layout(dummy_data_addr, data_size);

    // Encode attributes, using shared references for committed-type attrs.
    let mut attr_bodies: Vec<Vec<u8>> = Vec::new();
    for attr in &ds.attributes {
        if attr.committed_type_name.is_some() {
            // Use a dummy address; body size is the same regardless of address.
            attr_bodies.push(encode_attribute_shared(attr, 0xDEAD_0000)?);
        } else {
            attr_bodies.push(encode_attribute(attr)?);
        }
    }

    let mut messages: Vec<OhdrMsg> = vec![
        (MessageType::Dataspace.as_u8(), ds_body, 0),
        (MessageType::Datatype.as_u8(), dt_body, dt_flags),
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
    btree_v2_nodes_start: u64,
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
        return Ok(encode_btree_v1_chunk_node(
            ci,
            &chunk_addrs,
            &filtered_sizes,
        ));
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
        5 => {
            let unfilt_chunk_bytes: u64 = ci.chunk_dims_with_elem.iter().product();
            let has_filters = !ci.filters.is_empty();
            let (btree_type, rec_size, _csl) =
                btree_v2_record_info(ci.ndims, has_filters, unfilt_chunk_bytes);
            let nrecords = chunk_addrs.len();
            let node_size = 2048usize;
            let layout = compute_btree_v2_layout(nrecords, rec_size as usize, node_size);
            // Root node address: for depth 0, first node (leaf); for depth 1, second node (internal)
            let root_addr = btree_v2_nodes_start + (layout.root_node_index() * node_size) as u64;
            Ok(encode_btree_v2_header(
                root_addr,
                btree_type,
                rec_size,
                layout.depth,
                layout.root_nrec() as u16,
                nrecords as u64,
            ))
        }
        _ => Ok(vec![]),
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
        5 => {
            // BTreeV2: only the BTHD goes in the metadata region.
            // The leaf node goes after chunk data (handled separately).
            // BTHD: sig(4) + ver(1) + type(1) + node_size(4) + rec_size(2) + depth(2)
            //   + split(1) + merge(1) + root_addr(8) + nrecords(2) + total_records(8) + checksum(4)
            38
        }
        _ => 0,
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
fn compute_ea_dblk_specs(
    n_chunks: usize,
    idx_blk_elmts: u32,
    data_blk_min_elmts: u32,
) -> Vec<EaDblkSpec> {
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
                dblks.push(EaDblkSpec {
                    nelmts: dblk_nelmts,
                    block_off,
                });
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
    let iblk_size =
        14 + idx_blk_elmts as usize * entry_size as usize + (ndblk_addrs + nsblk_addrs) * 8 + 4;

    // Compute EADB sizes and addresses.
    let mut dblk_addrs = Vec::with_capacity(n_dblks);
    let mut dblk_offset = (hdr_size + iblk_size) as u64;
    let mut total_dblk_disk_size: u64 = 0;
    for spec in &dblk_specs {
        dblk_addrs.push(index_addr + dblk_offset);
        let dblk_disk_size = 14 + block_off_size + spec.nelmts as usize * entry_size as usize + 4;
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

fn align_up(pos: usize, alignment: usize) -> usize {
    (pos + alignment - 1) / alignment * alignment
}

/// Compute the BTree v2 record type and size for chunk indexing.
/// Returns (btree_type, rec_size, chunk_size_len).
/// Non-filtered: type 10, rec = addr(8) + ndims*scaled(8).
/// Filtered: type 11, rec = addr(8) + chunk_nbytes(chunk_size_len) + filter_mask(4) + ndims*scaled(8).
fn btree_v2_record_info(
    ndims: usize,
    has_filters: bool,
    unfilt_chunk_bytes: u64,
) -> (u8, u16, usize) {
    let scaled_enc: usize = 8; // unlimited dims → 8 bytes per scaled offset
    if has_filters {
        // chunk_size_len: empirically matches C library with formula
        // (floor(log2(size)) + 15) / 8, minimum 2.
        let log2 = if unfilt_chunk_bytes > 0 {
            63 - unfilt_chunk_bytes.leading_zeros() as usize
        } else {
            0
        };
        let chunk_size_len = ((log2 + 15) / 8).max(2);
        let rec_size = 8 + chunk_size_len + 4 + ndims * scaled_enc;
        (11, rec_size as u16, chunk_size_len)
    } else {
        let rec_size = 8 + ndims * scaled_enc;
        (10, rec_size as u16, 0)
    }
}

/// Encode BTree v2 header (BTHD) for chunk index type 5.
fn encode_btree_v2_header(
    root_addr: u64,
    btree_type: u8,
    rec_size: u16,
    depth: u16,
    root_nrec: u16,
    total_records: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(38);
    buf.extend_from_slice(b"BTHD");
    buf.push(0); // version
    buf.push(btree_type);
    buf.extend_from_slice(&2048u32.to_le_bytes()); // node_size
    buf.extend_from_slice(&rec_size.to_le_bytes());
    buf.extend_from_slice(&depth.to_le_bytes());
    buf.push(100); // split_percent
    buf.push(40); // merge_percent
    buf.extend_from_slice(&root_addr.to_le_bytes());
    buf.extend_from_slice(&root_nrec.to_le_bytes());
    buf.extend_from_slice(&total_records.to_le_bytes());
    let cs = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cs.to_le_bytes());
    debug_assert_eq!(buf.len(), 38);
    buf
}

/// Encode BTree v2 leaf node (BTLF) for chunk index type 5.
fn encode_btree_v2_leaf(
    chunk_addrs: &[u64],
    chunk_coords: &[Vec<u64>],
    filtered_sizes: &[u64],
    ndims: usize,
    btree_type: u8,
    chunk_size_len: usize,
    node_size: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(node_size);
    buf.extend_from_slice(b"BTLF");
    buf.push(0); // version
    buf.push(btree_type);

    for (i, &addr) in chunk_addrs.iter().enumerate() {
        buf.extend_from_slice(&addr.to_le_bytes());
        if btree_type == 11 {
            // Filtered record: chunk_nbytes(chunk_size_len) + filter_mask(4)
            let nbytes = filtered_sizes[i];
            match chunk_size_len {
                1 => buf.push(nbytes as u8),
                2 => buf.extend_from_slice(&(nbytes as u16).to_le_bytes()),
                4 => buf.extend_from_slice(&(nbytes as u32).to_le_bytes()),
                8 => buf.extend_from_slice(&nbytes.to_le_bytes()),
                _ => {
                    // Variable-width encoding (3, 5, 6, 7 bytes)
                    for b in 0..chunk_size_len {
                        buf.push((nbytes >> (b * 8)) as u8);
                    }
                }
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
        }
        // Scaled offsets (chunk_coords are already in grid/scaled space)
        for d in 0..ndims {
            buf.extend_from_slice(&chunk_coords[i][d].to_le_bytes());
        }
    }

    // Checksum immediately after records, then zero-pad to node_size.
    let cs = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cs.to_le_bytes());
    buf.resize(node_size, 0);
    debug_assert_eq!(buf.len(), node_size);
    buf
}

/// Layout of a BTree v2 tree produced by the C library's insertion algorithm.
/// For depth 0: single leaf node.
/// For depth 1: one internal node + multiple leaf nodes.
struct BTreeV2Layout {
    depth: u16,
    /// Number of records in each leaf, in allocation order.
    /// For depth 0: [nrecords].
    /// For depth 1: [leaf0_nrec, leaf1_nrec, ...].
    leaf_nrecs: Vec<usize>,
    /// Records that live in the internal node (indices into the global record list).
    /// Empty for depth 0.
    internal_records: Vec<usize>,
}

impl BTreeV2Layout {
    /// Total number of node_size nodes needed (leaves + internal).
    fn num_nodes(&self) -> usize {
        if self.depth == 0 {
            1
        } else {
            // Allocation order: leaf0, internal, leaf1, leaf2, ...
            1 + self.leaf_nrecs.len()
        }
    }

    /// Index of the root node in the allocation sequence.
    fn root_node_index(&self) -> usize {
        if self.depth == 0 { 0 } else { 1 } // internal is the 2nd node allocated
    }

    /// Number of records in the root node.
    fn root_nrec(&self) -> usize {
        if self.depth == 0 {
            self.leaf_nrecs[0]
        } else {
            self.internal_records.len()
        }
    }
}

/// Simulate the C library's BTree v2 insertion to determine the final tree layout.
fn compute_btree_v2_layout(nrecords: usize, rec_size: usize, node_size: usize) -> BTreeV2Layout {
    let max_leaf_nrec = (node_size - 10) / rec_size; // 10 = sig(4)+ver(1)+type(1)+checksum(4)
    let split_nrec = (max_leaf_nrec * 100) / 100; // split_percent=100

    if nrecords <= max_leaf_nrec {
        return BTreeV2Layout {
            depth: 0,
            leaf_nrecs: vec![nrecords],
            internal_records: vec![],
        };
    }

    // Simulate insertion with split/redistribute.
    // leaves[i] = number of records in leaf i.
    // internal_recs = list of global record indices stored in the internal node.
    // leaf_contents[i] = list of global record indices in leaf i.
    let mut leaf_contents: Vec<Vec<usize>> = vec![vec![]];
    let mut internal_recs: Vec<usize> = vec![];

    for rec in 0..nrecords {
        if leaf_contents.len() == 1 && internal_recs.is_empty() {
            // Depth 0: single leaf
            if leaf_contents[0].len() == split_nrec {
                // Split root
                let old = &leaf_contents[0];
                let mid = old.len() / 2;
                let left = old[..mid].to_vec();
                let right = old[mid + 1..].to_vec();
                internal_recs.push(old[mid]);
                leaf_contents = vec![left, right];
                // Now insert the current record
                insert_into_tree(rec, &mut leaf_contents, &mut internal_recs, split_nrec);
            } else {
                leaf_contents[0].push(rec);
            }
        } else {
            insert_into_tree(rec, &mut leaf_contents, &mut internal_recs, split_nrec);
        }
    }

    BTreeV2Layout {
        depth: 1,
        leaf_nrecs: leaf_contents.iter().map(|l| l.len()).collect(),
        internal_records: internal_recs,
    }
}

/// Insert a record into a depth-1 BTree v2 (internal node + leaf children).
fn insert_into_tree(
    rec: usize,
    leaves: &mut Vec<Vec<usize>>,
    internal_recs: &mut Vec<usize>,
    split_nrec: usize,
) {
    // Find which child to insert into (binary search on internal records).
    let mut idx = internal_recs.partition_point(|&r| r < rec);
    if idx < internal_recs.len() && internal_recs[idx] == rec {
        idx += 1; // shouldn't happen, but be safe
    }

    let mut retries = 2u32;
    while leaves[idx].len() == split_nrec {
        let nchildren = leaves.len();
        if idx == 0 {
            // Left-most child
            if retries > 0 && leaves[idx + 1].len() < split_nrec {
                redistribute2(leaves, internal_recs, idx);
            } else {
                split1(leaves, internal_recs, idx);
            }
        } else if idx == nchildren - 1 {
            // Right-most child
            if retries > 0 && leaves[idx - 1].len() < split_nrec {
                redistribute2(leaves, internal_recs, idx - 1);
            } else {
                split1(leaves, internal_recs, idx);
            }
        } else {
            // Middle child
            if retries > 0
                && (leaves[idx + 1].len() < split_nrec || leaves[idx - 1].len() < split_nrec)
            {
                redistribute2(leaves, internal_recs, idx - 1);
            } else {
                split1(leaves, internal_recs, idx);
            }
        }
        // Re-locate
        idx = internal_recs.partition_point(|&r| r < rec);
        if idx < internal_recs.len() && internal_recs[idx] == rec {
            idx += 1;
        }
        retries = retries.saturating_sub(1);
    }

    leaves[idx].push(rec);
}

/// Redistribute records between leaves[idx] and leaves[idx+1].
fn redistribute2(leaves: &mut Vec<Vec<usize>>, internal_recs: &mut Vec<usize>, idx: usize) {
    let left_n = leaves[idx].len();
    let right_n = leaves[idx + 1].len();

    if left_n < right_n {
        // Move from right to left
        let new_right_n = (left_n + right_n) / 2;
        let move_n = right_n - new_right_n;
        let moved: Vec<usize> = leaves[idx + 1][..move_n].to_vec();
        // Parent record goes down into left
        leaves[idx].push(internal_recs[idx]);
        // move_n-1 records from start of right go to left
        leaves[idx].extend_from_slice(&moved[..move_n - 1]);
        // Record at move_n-1 in right becomes new parent
        internal_recs[idx] = moved[move_n - 1];
        // Right keeps from move_n onwards
        leaves[idx + 1] = leaves[idx + 1][move_n..].to_vec();
    } else {
        // Move from left to right
        let new_left_n = (left_n + right_n) / 2;
        let move_n = left_n - new_left_n;
        // Build new right: records from end of left + parent + old right
        let mut new_right = leaves[idx][left_n - move_n + 1..].to_vec();
        new_right.push(internal_recs[idx]);
        new_right.extend_from_slice(&leaves[idx + 1]);
        leaves[idx + 1] = new_right;
        // New parent record
        internal_recs[idx] = leaves[idx][left_n - move_n];
        leaves[idx].truncate(left_n - move_n);
    }
}

/// Split leaves[idx] into two, promoting middle record to internal.
fn split1(leaves: &mut Vec<Vec<usize>>, internal_recs: &mut Vec<usize>, idx: usize) {
    let old = &leaves[idx];
    let mid = old.len() / 2;
    let new_leaf = old[mid + 1..].to_vec();
    let mid_rec = old[mid];
    leaves[idx] = old[..mid].to_vec();
    internal_recs.insert(idx, mid_rec);
    leaves.insert(idx + 1, new_leaf);
}

/// Write all BTree v2 nodes into the buffer.
fn write_btree_v2_nodes(
    buf: &mut [u8],
    nodes_start: usize,
    node_size: usize,
    layout: &BTreeV2Layout,
    chunk_addrs: &[u64],
    chunk_coords: &[Vec<u64>],
    filtered_sizes: &[u64],
    ndims: usize,
    btree_type: u8,
    chunk_size_len: usize,
) {
    if layout.depth == 0 {
        // Single leaf at nodes_start
        let leaf = encode_btree_v2_leaf(
            chunk_addrs,
            chunk_coords,
            filtered_sizes,
            ndims,
            btree_type,
            chunk_size_len,
            node_size,
        );
        buf[nodes_start..nodes_start + node_size].copy_from_slice(&leaf);
        return;
    }

    // Depth 1: allocation order is leaf0, internal, leaf1, leaf2, ...
    // Compute node addresses
    let leaf0_addr = nodes_start;
    let internal_addr = nodes_start + node_size;
    let mut leaf_addrs: Vec<usize> = vec![leaf0_addr];
    for i in 1..layout.leaf_nrecs.len() {
        leaf_addrs.push(nodes_start + (1 + i) * node_size);
    }

    // Build the record-index-to-global mapping. We need to figure out which
    // global records are in each leaf and which are in the internal node.
    // The global record order is 0..nrecords-1 (sorted), and the layout tells
    // us which indices are in internal vs leaves.

    // Reconstruct which global records are in each leaf.
    // All records are 0..total-1. Internal records are known. Leaf records are
    // the rest, distributed according to leaf_nrecs.
    let total = chunk_addrs.len();
    let mut all_sorted: Vec<usize> = (0..total).collect();
    // Remove internal records
    for &ir in &layout.internal_records {
        all_sorted.retain(|&x| x != ir);
    }
    // Split into leaf groups by count
    let mut leaf_records: Vec<Vec<usize>> = Vec::new();
    let mut pos = 0;
    for &nrec in &layout.leaf_nrecs {
        leaf_records.push(all_sorted[pos..pos + nrec].to_vec());
        pos += nrec;
    }

    // Write leaf nodes
    for (li, leaf_recs) in leaf_records.iter().enumerate() {
        let addrs: Vec<u64> = leaf_recs.iter().map(|&r| chunk_addrs[r]).collect();
        let coords: Vec<Vec<u64>> = leaf_recs.iter().map(|&r| chunk_coords[r].clone()).collect();
        let fsizes: Vec<u64> = leaf_recs.iter().map(|&r| filtered_sizes[r]).collect();
        let leaf = encode_btree_v2_leaf(
            &addrs,
            &coords,
            &fsizes,
            ndims,
            btree_type,
            chunk_size_len,
            node_size,
        );
        let addr = leaf_addrs[li];
        buf[addr..addr + node_size].copy_from_slice(&leaf);
    }

    // Write internal node (BTIN)
    let rec_size = if btree_type == 11 {
        8 + chunk_size_len + 4 + ndims * 8
    } else {
        8 + ndims * 8
    };
    let max_leaf_nrec = (node_size - 10) / rec_size;
    let nrec_width: usize = if max_leaf_nrec <= 255 { 1 } else { 2 };

    let internal = encode_btree_v2_internal(
        &layout.internal_records,
        chunk_addrs,
        chunk_coords,
        filtered_sizes,
        ndims,
        btree_type,
        chunk_size_len,
        &leaf_addrs,
        &layout.leaf_nrecs,
        nrec_width,
        node_size,
    );
    buf[internal_addr..internal_addr + node_size].copy_from_slice(&internal);
}

/// Encode a BTree v2 internal node (BTIN).
fn encode_btree_v2_internal(
    internal_records: &[usize],
    chunk_addrs: &[u64],
    chunk_coords: &[Vec<u64>],
    filtered_sizes: &[u64],
    ndims: usize,
    btree_type: u8,
    chunk_size_len: usize,
    child_addrs: &[usize],
    child_nrecs: &[usize],
    nrec_width: usize,
    node_size: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(node_size);
    buf.extend_from_slice(b"BTIN");
    buf.push(0); // version
    buf.push(btree_type);

    // Records
    for &rec_idx in internal_records {
        buf.extend_from_slice(&chunk_addrs[rec_idx].to_le_bytes());
        if btree_type == 11 {
            let nbytes = filtered_sizes[rec_idx];
            match chunk_size_len {
                1 => buf.push(nbytes as u8),
                2 => buf.extend_from_slice(&(nbytes as u16).to_le_bytes()),
                4 => buf.extend_from_slice(&(nbytes as u32).to_le_bytes()),
                8 => buf.extend_from_slice(&nbytes.to_le_bytes()),
                _ => {
                    for b in 0..chunk_size_len {
                        buf.push((nbytes >> (b * 8)) as u8);
                    }
                }
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
        }
        for d in 0..ndims {
            buf.extend_from_slice(&chunk_coords[rec_idx][d].to_le_bytes());
        }
    }

    // Child pointers: addr(8) + nrec(nrec_width) for each child
    for (ci, &addr) in child_addrs.iter().enumerate() {
        buf.extend_from_slice(&(addr as u64).to_le_bytes());
        match nrec_width {
            1 => buf.push(child_nrecs[ci] as u8),
            2 => buf.extend_from_slice(&(child_nrecs[ci] as u16).to_le_bytes()),
            _ => unreachable!(),
        }
    }

    // Checksum immediately after, then zero-pad
    let cs = crate::checksum::lookup3(&buf);
    buf.extend_from_slice(&cs.to_le_bytes());
    buf.resize(node_size, 0);
    buf
}

/// Compute the C library's vlen element insertion order.
///
/// The C library converts vlen data in-place. When dest element size (16 bytes
/// for heap IDs) > source element size (8 bytes for pointers), it processes
/// elements in a specific order to avoid buffer overwrites:
///   1. Compute `safe = nelmts - ceil(nelmts * src_stride / dst_stride)` safe
///      elements at the end that don't overlap with source.
///   2. Process those forward, then process remaining backwards.
///
/// For vlen strings (src=8, dst=16), this produces a specific ordering.
/// For vlen sequences (src=16, dst=16), forward order is used.
/// Compute vlen GCOL insertion order for string types (src_stride=8, dst_stride=16).
fn vlen_string_gcol_order(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let s_stride: usize = 8;
    let d_stride: usize = 16;

    let mut result = Vec::with_capacity(n);
    let mut remaining = n;

    while remaining > 0 {
        if d_stride > s_stride {
            let safe = remaining - ((remaining * s_stride + d_stride - 1) / d_stride);
            if safe < 2 {
                // Reverse the remaining elements (always at the start of the array)
                for i in (0..remaining).rev() {
                    result.push(i);
                }
                break;
            } else {
                // Process `safe` elements forward from position (remaining - safe)
                let start = n - remaining + (remaining - safe);
                for i in 0..safe {
                    result.push(start + i);
                }
                remaining -= safe;
            }
        } else {
            // Forward
            let start = n - remaining;
            for i in 0..remaining {
                result.push(start + i);
            }
            break;
        }
    }
    result
}

/// Build a compat GCOL (Global Heap Collection) for vlen elements.
///
/// The ordered_elements slice gives the elements in insertion order (as the C library
/// would produce). The GCOL is padded to 4096 bytes (H5HG_MINSIZE).
fn build_gcol_compat(ordered_elements: &[&Vec<u8>]) -> Vec<u8> {
    let sl = crate::writer::encode::SIZE_OF_LENGTHS as usize;
    let obj_hdr = 8 + sl; // index(2) + refcount(2) + reserved(4) + size(L)
    let header_size = 8 + sl; // sig(4) + ver(1) + reserved(3) + collection_size(L)

    // Compute objects size.
    let mut objects_size = 0usize;
    for elem in ordered_elements {
        let padded = (elem.len() + 7) & !7;
        objects_size += obj_hdr + padded;
    }

    // Free space marker.
    let used = header_size + objects_size;
    let min_size = 4096; // H5HG_MINSIZE
    let collection_size = used.max(min_size);
    // The free space marker needs at least obj_hdr bytes.
    let collection_size = collection_size.max(used + obj_hdr);

    let mut buf = vec![0u8; collection_size];

    // Header
    buf[0..4].copy_from_slice(b"GCOL");
    buf[4] = 1; // version
    // reserved[5..8] = 0
    buf[8..16].copy_from_slice(&(collection_size as u64).to_le_bytes());

    let mut pos = header_size;
    for (i, elem) in ordered_elements.iter().enumerate() {
        let idx = (i + 1) as u16;
        buf[pos..pos + 2].copy_from_slice(&idx.to_le_bytes());
        // ref_count = 0 (C library default for fresh objects)
        buf[pos + 2..pos + 4].copy_from_slice(&0u16.to_le_bytes());
        // reserved[4..8] = 0
        buf[pos + 8..pos + 16].copy_from_slice(&(elem.len() as u64).to_le_bytes());
        buf[pos + 16..pos + 16 + elem.len()].copy_from_slice(elem);
        let padded = (elem.len() + 7) & !7;
        pos += obj_hdr + padded;
    }

    // Free space marker: index=0, ref_count=0, reserved=0, size=remaining
    let free_space = collection_size - pos;
    // index(2) = 0 already, ref_count(2) = 0 already, reserved(4) = 0 already
    buf[pos + 8..pos + 16].copy_from_slice(&(free_space as u64).to_le_bytes());

    buf
}

/// Build vlen heap IDs with compat GCOL ordering.
///
/// Returns the heap ID data (4+8+4 bytes per element) where each element
/// references the correct GCOL object.
fn build_vlen_heap_ids_compat(
    elements: &[Vec<u8>],
    gcol_addr: u64,
    gcol_order: &[usize],
    base_elem_size: usize,
) -> Vec<u8> {
    // Build a mapping: original_element_index → gcol_object_index (1-based)
    let mut elem_to_gcol_idx = vec![0u32; elements.len()];
    for (gcol_pos, &orig_idx) in gcol_order.iter().enumerate() {
        elem_to_gcol_idx[orig_idx] = (gcol_pos + 1) as u32;
    }

    let heap_id_size = 4 + 8 + 4; // seq_len + addr + idx
    let mut buf = Vec::with_capacity(elements.len() * heap_id_size);
    for (i, elem) in elements.iter().enumerate() {
        if elem.is_empty() {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            // seq_len is the element count (bytes / base_element_size)
            let seq_len = (elem.len() / base_elem_size) as u32;
            buf.extend_from_slice(&seq_len.to_le_bytes());
            buf.extend_from_slice(&gcol_addr.to_le_bytes());
            buf.extend_from_slice(&elem_to_gcol_idx[i].to_le_bytes());
        }
    }
    buf
}

fn clone_attr(attr: &crate::writer::types::AttrData) -> crate::writer::types::AttrData {
    crate::writer::types::AttrData {
        name: attr.name.clone(),
        datatype: attr.datatype.clone(),
        shape: attr.shape.clone(),
        value: attr.value.clone(),
        committed_type_name: attr.committed_type_name.clone(),
    }
}
