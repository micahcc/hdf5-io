use crate::datatype::Datatype;
use crate::writer::types::AttrData;
use crate::writer::types::ChunkFilter;
use crate::writer::types::StorageLayout;

/// A dataset node in the file builder tree.
pub struct DatasetNode {
    pub(crate) datatype: Datatype,
    pub(crate) shape: Vec<u64>,
    pub(crate) max_dims: Option<Vec<u64>>,
    pub(crate) data: Vec<u8>,
    pub(crate) attributes: Vec<AttrData>,
    pub(crate) layout: StorageLayout,
    pub(crate) fill_value: Option<Vec<u8>>,
    /// Force layout message version (3 or 4). Default is 4.
    pub(crate) layout_version: u8,
    /// Variable-length data elements (if set, serialized into GCOL + heap IDs).
    pub(crate) vlen_elements: Option<Vec<Vec<u8>>>,
    /// When true, use early space allocation (affects fill value flags and chunk index type).
    pub(crate) early_alloc: bool,
}

impl DatasetNode {
    pub(crate) fn new(datatype: Datatype, shape: &[u64], data: Vec<u8>) -> Self {
        DatasetNode {
            datatype,
            shape: shape.to_vec(),
            max_dims: None,
            data,
            attributes: vec![],
            layout: StorageLayout::default(),
            fill_value: None,
            layout_version: 4,
            vlen_elements: None,
            early_alloc: false,
        }
    }

    pub(crate) fn new_vlen(datatype: Datatype, shape: &[u64], elements: Vec<Vec<u8>>) -> Self {
        DatasetNode {
            datatype,
            shape: shape.to_vec(),
            max_dims: None,
            data: vec![],
            attributes: vec![],
            layout: StorageLayout::default(),
            fill_value: None,
            layout_version: 4,
            vlen_elements: Some(elements),
            early_alloc: false,
        }
    }

    /// Add an attribute to this dataset.
    pub fn add_attribute(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        value: Vec<u8>,
    ) -> &mut Self {
        self.attributes.push(AttrData {
            name: name.to_string(),
            datatype,
            shape: shape.to_vec(),
            value,
        });
        self
    }

    /// Set the storage layout for this dataset.
    pub fn set_layout(&mut self, layout: StorageLayout) -> &mut Self {
        self.layout = layout;
        self
    }

    /// Set chunked storage with the given chunk dimensions and filters.
    pub fn set_chunked(&mut self, chunk_dims: &[u64], filters: Vec<ChunkFilter>) -> &mut Self {
        self.layout = StorageLayout::Chunked {
            chunk_dims: chunk_dims.to_vec(),
            filters,
        };
        self
    }

    /// Force a specific layout message version (3 or 4).
    /// Version 3 uses B-tree v1 chunk indexing. Default is 4.
    pub fn set_layout_version(&mut self, version: u8) -> &mut Self {
        self.layout_version = version;
        self
    }

    /// Set maximum dimensions for the dataspace.
    /// Use `u64::MAX` for an unlimited dimension.
    pub fn set_max_dims(&mut self, max_dims: &[u64]) -> &mut Self {
        self.max_dims = Some(max_dims.to_vec());
        self
    }

    /// Set an explicit fill value for this dataset.
    pub fn set_fill_value(&mut self, value: Vec<u8>) -> &mut Self {
        self.fill_value = Some(value);
        self
    }

    /// Clear raw data (for creating empty datasets, e.g. empty chunked).
    pub fn clear_data(&mut self) -> &mut Self {
        self.data.clear();
        self
    }

    /// Use early space allocation (affects fill value flags and chunk index type).
    ///
    /// For chunked datasets, early allocation with no filters and fixed dimensions
    /// selects the Implicit chunk index (no separate index structure).
    pub fn set_early_alloc(&mut self) -> &mut Self {
        self.early_alloc = true;
        self
    }
}
