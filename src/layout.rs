use crate::error::Error;
use crate::error::Result;

/// Chunk indexing method (HDF5 1.10+ / layout message version 4).
///
/// Reference: H5Dpkg.h `H5D_chunk_index_t`, H5Olayout.c.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndexType {
    /// Single chunk (no index needed, direct address).
    SingleChunk,
    /// Implicit indexing (chunks stored contiguously in file order).
    Implicit,
    /// Fixed array (non-filtered, non-extendable).
    FixedArray,
    /// Extensible array (one unlimited dimension).
    ExtensibleArray,
    /// B-tree v2 (multiple unlimited dimensions or filtered).
    BTreeV2,
}

/// Decoded data layout from a layout message (type 0x0008).
///
/// ## On-disk layout message (version 3/4)
///
/// Version 3 (HDF5 1.8):
/// ```text
/// Byte 0:    Version (3)
/// Byte 1:    Layout class (0=compact, 1=contiguous, 2=chunked)
/// Byte 2+:   Class-specific data
/// ```
///
/// Version 4 (HDF5 1.10+, used with superblock v2/v3):
/// ```text
/// Byte 0:    Version (4)
/// Byte 1:    Layout class (0=compact, 1=contiguous, 2=chunked, 3=virtual)
/// Byte 2+:   Class-specific data
/// ```
#[derive(Debug, Clone)]
pub enum DataLayout {
    /// Compact: data stored directly in the object header.
    Compact { data: Vec<u8> },
    /// Contiguous: data stored at a single file offset.
    Contiguous { address: u64, size: u64 },
    /// Chunked: data stored in fixed-size chunks, indexed.
    Chunked {
        /// Number of dimensions.
        /// Both v3 and v4 store rank + 1, with the last dimension being
        /// the element size in bytes.
        dimensionality: u8,
        /// Chunk dimensions (element counts per dimension).
        chunk_dims: Vec<u32>,
        /// For layout v3: address of the chunk B-tree v1.
        /// For layout v4: address of the chunk index structure.
        address: u64,
        /// Layout message version (3 or 4).
        layout_version: u8,
        /// Chunk index type (v4 only).
        chunk_index_type: Option<ChunkIndexType>,
        /// Chunk flags byte (v4 only).
        /// Bit 0: don't filter partial edge chunks.
        /// Bit 1: single index with filter info present.
        chunk_flags: u8,
        /// For single-chunk with filter: the filtered (on-disk) size.
        single_chunk_filtered_size: Option<u64>,
        /// For single-chunk with filter: the filter mask.
        single_chunk_filter_mask: Option<u32>,
    },
    /// Virtual: maps regions to other datasets (HDF5 1.10+).
    Virtual {
        // TODO: virtual dataset mappings
    },
}

impl DataLayout {
    /// Parse a layout message from its raw body bytes.
    pub fn parse(data: &[u8], size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidLayout {
                msg: "layout message too short".into(),
            });
        }

        let version = data[0];
        let layout_class = data[1];

        match version {
            3 => Self::parse_v3(data, layout_class, size_of_offsets, size_of_lengths),
            // v5 uses the same on-disk format as v4; the version bump only changes
            // which filtered-chunk size encoding the library *writes* (size_of_lengths
            // width), but the decode path is identical.
            4 | 5 => Self::parse_v4(data, layout_class, size_of_offsets, size_of_lengths),
            _ => Err(Error::InvalidLayout {
                msg: format!("unsupported layout message version {}", version),
            }),
        }
    }

    fn parse_v3(
        data: &[u8],
        layout_class: u8,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let o = size_of_offsets as usize;
        let l = size_of_lengths as usize;

        match layout_class {
            // Compact
            0 => {
                if data.len() < 4 {
                    return Err(Error::InvalidLayout {
                        msg: "compact layout v3 too short".into(),
                    });
                }
                let compact_size = u16::from_le_bytes([data[2], data[3]]) as usize;
                if data.len() < 4 + compact_size {
                    return Err(Error::InvalidLayout {
                        msg: "compact layout data truncated".into(),
                    });
                }
                Ok(DataLayout::Compact {
                    data: data[4..4 + compact_size].to_vec(),
                })
            }
            // Contiguous
            1 => {
                if data.len() < 2 + o + l {
                    return Err(Error::InvalidLayout {
                        msg: "contiguous layout v3 too short".into(),
                    });
                }
                let address = read_offset(&data[2..], size_of_offsets);
                let size = read_length(&data[2 + o..], size_of_lengths);
                Ok(DataLayout::Contiguous { address, size })
            }
            // Chunked
            2 => {
                if data.len() < 3 {
                    return Err(Error::InvalidLayout {
                        msg: "chunked layout v3 too short".into(),
                    });
                }
                // Layout v3 chunked:
                //   Byte 2: dimensionality (rank + 1, the extra dim is the element size)
                //   Byte 3..3+O: address of chunk B-tree v1
                //   Byte 3+O..: dim sizes (dimensionality * 4 bytes each)
                let dimensionality = data[2];
                let ndims = dimensionality as usize;
                if data.len() < 3 + o + ndims * 4 {
                    return Err(Error::InvalidLayout {
                        msg: "chunked layout v3 truncated".into(),
                    });
                }
                let address = read_offset(&data[3..], size_of_offsets);
                let mut chunk_dims = Vec::with_capacity(ndims);
                let dims_start = 3 + o;
                for i in 0..ndims {
                    let off = dims_start + i * 4;
                    chunk_dims.push(u32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]));
                }
                Ok(DataLayout::Chunked {
                    dimensionality,
                    chunk_dims,
                    address,
                    layout_version: 3,
                    chunk_index_type: None,
                    chunk_flags: 0,
                    single_chunk_filtered_size: None,
                    single_chunk_filter_mask: None,
                })
            }
            _ => Err(Error::InvalidLayout {
                msg: format!("unknown layout class {} in v3", layout_class),
            }),
        }
    }

    fn parse_v4(
        data: &[u8],
        layout_class: u8,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let o = size_of_offsets as usize;
        let l = size_of_lengths as usize;

        match layout_class {
            // Compact
            0 => {
                if data.len() < 4 {
                    return Err(Error::InvalidLayout {
                        msg: "compact layout v4 too short".into(),
                    });
                }
                let compact_size = u16::from_le_bytes([data[2], data[3]]) as usize;
                if data.len() < 4 + compact_size {
                    return Err(Error::InvalidLayout {
                        msg: "compact layout data truncated".into(),
                    });
                }
                Ok(DataLayout::Compact {
                    data: data[4..4 + compact_size].to_vec(),
                })
            }
            // Contiguous
            1 => {
                if data.len() < 2 + o + l {
                    return Err(Error::InvalidLayout {
                        msg: "contiguous layout v4 too short".into(),
                    });
                }
                let address = read_offset(&data[2..], size_of_offsets);
                let size = read_length(&data[2 + o..], size_of_lengths);
                Ok(DataLayout::Contiguous { address, size })
            }
            // Chunked
            2 => {
                // Layout v4 chunked (from H5Olayout.c):
                //   Byte 2: flags
                //     bit 0: H5O_LAYOUT_CHUNK_DONT_FILTER_PARTIAL_BOUND_CHUNKS
                //     bit 1: H5O_LAYOUT_CHUNK_SINGLE_INDEX_WITH_FILTER
                //   Byte 3: ndims (dataset rank, no +1 trick unlike v3)
                //   Byte 4: enc_bytes_per_dim (1-8)
                //   Byte 5..: chunk dims (ndims * enc_bytes_per_dim)
                //   Then: 1 byte chunk index type
                //   Then: index-specific creation parameters
                //   Then: sizeof_addr for chunk index address
                if data.len() < 5 {
                    return Err(Error::InvalidLayout {
                        msg: "chunked layout v4 too short".into(),
                    });
                }
                let flags = data[2];
                let dimensionality = data[3];
                let ndims = dimensionality as usize;
                let dim_enc_size = data[4] as usize;

                let dims_start = 5;
                if data.len() < dims_start + ndims * dim_enc_size + 1 {
                    return Err(Error::InvalidLayout {
                        msg: "chunked layout v4 dims truncated".into(),
                    });
                }

                let mut chunk_dims = Vec::with_capacity(ndims);
                for i in 0..ndims {
                    let off = dims_start + i * dim_enc_size;
                    let dim = read_var_uint(&data[off..], dim_enc_size);
                    chunk_dims.push(dim as u32);
                }

                // Chunk index type byte comes AFTER dimensions
                let mut pos = dims_start + ndims * dim_enc_size;
                let chunk_index_type_id = data[pos];
                pos += 1;

                let chunk_index_type = match chunk_index_type_id {
                    1 => ChunkIndexType::SingleChunk,
                    2 => ChunkIndexType::Implicit,
                    3 => ChunkIndexType::FixedArray,
                    4 => ChunkIndexType::ExtensibleArray,
                    5 => ChunkIndexType::BTreeV2,
                    _ => {
                        return Err(Error::InvalidLayout {
                            msg: format!("unknown chunk index type {}", chunk_index_type_id),
                        });
                    }
                };

                // Parse index-type-specific creation parameters
                let mut single_chunk_filtered_size = None;
                let mut single_chunk_filter_mask = None;

                match chunk_index_type {
                    ChunkIndexType::SingleChunk => {
                        if (flags & 0x02) != 0 {
                            // SINGLE_INDEX_WITH_FILTER: filtered_size + filter_mask
                            single_chunk_filtered_size = Some(read_var_uint(&data[pos..], l));
                            pos += l;
                            single_chunk_filter_mask = Some(u32::from_le_bytes([
                                data[pos],
                                data[pos + 1],
                                data[pos + 2],
                                data[pos + 3],
                            ]));
                            pos += 4;
                        }
                    }
                    ChunkIndexType::Implicit => {} // no params
                    ChunkIndexType::FixedArray => {
                        pos += 1; // max_dblk_page_nelmts_bits
                    }
                    ChunkIndexType::ExtensibleArray => {
                        pos += 5; // 5 creation parameters
                    }
                    ChunkIndexType::BTreeV2 => {
                        pos += 6; // node_size(4) + split_percent(1) + merge_percent(1)
                    }
                }

                // Index address
                let address = if pos + o <= data.len() {
                    read_offset(&data[pos..], size_of_offsets)
                } else {
                    u64::MAX
                };

                Ok(DataLayout::Chunked {
                    dimensionality,
                    chunk_dims,
                    address,
                    layout_version: 4,
                    chunk_index_type: Some(chunk_index_type),
                    chunk_flags: flags,
                    single_chunk_filtered_size,
                    single_chunk_filter_mask,
                })
            }
            // Virtual
            3 => {
                // TODO: parse virtual dataset mappings
                Ok(DataLayout::Virtual {})
            }
            _ => Err(Error::InvalidLayout {
                msg: format!("unknown layout class {} in v4", layout_class),
            }),
        }
    }
}

fn read_var_uint(data: &[u8], size: usize) -> u64 {
    let mut result = 0u64;
    for i in 0..size.min(8) {
        result |= (data[i] as u64) << (i * 8);
    }
    result
}

fn read_offset(data: &[u8], size: u8) -> u64 {
    match size {
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
        8 => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
        _ => 0,
    }
}

fn read_length(data: &[u8], size: u8) -> u64 {
    read_offset(data, size)
}
