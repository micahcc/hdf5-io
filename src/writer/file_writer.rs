use crate::error::Error;
use crate::error::Result;
use crate::writer::GroupNode;
use crate::writer::encode::SUPERBLOCK_SIZE;
use crate::writer::encode::encode_superblock;
use crate::writer::serialize::write_group;
use crate::writer::serialize_compat::write_tree_compat;

/// Options controlling how the HDF5 file is written.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// If set, store these timestamps on every object header.
    /// Tuple: (access_time, modification_time, change_time, birth_time) as Unix seconds.
    pub timestamps: Option<(u32, u32, u32, u32)>,

    /// When true, produce output byte-compatible with the HDF5 C library.
    ///
    /// This affects: message ordering, per-message flags, OHDR chunk padding,
    /// fill-value defaults, attribute-info messages, metadata block alignment,
    /// and parent-first object ordering.
    pub hdf5lib_compat: bool,

    /// Metadata block size (used in compat mode). Default is 2048.
    pub meta_block_size: Option<usize>,

    /// Superblock version to write. Default is 2.
    /// Use 3 for V110+ (HDF5 1.10) compatibility.
    pub superblock_version: Option<u8>,
}

/// Builds an HDF5 file in memory and writes it out.
///
/// # Example
/// ```
/// use hdf5_io::writer::FileWriter;
/// use hdf5_io::Datatype;
///
/// let mut w = FileWriter::new();
/// let data: Vec<u8> = (0..4i32).flat_map(|x| x.to_le_bytes()).collect();
/// w.root_mut().add_dataset("numbers", Datatype::native_i32(), &[4], data);
/// let bytes = w.to_bytes().unwrap();
/// ```
pub struct FileWriter {
    root: GroupNode,
    options: WriteOptions,
}

impl FileWriter {
    pub fn new() -> Self {
        FileWriter {
            root: GroupNode::new(),
            options: WriteOptions::default(),
        }
    }

    /// Create a writer with custom options.
    pub fn with_options(options: WriteOptions) -> Self {
        FileWriter {
            root: GroupNode::new(),
            options,
        }
    }

    pub fn root_mut(&mut self) -> &mut GroupNode {
        &mut self.root
    }

    /// Serialize the entire file to a byte vector.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if self.options.hdf5lib_compat {
            return self.to_bytes_compat();
        }

        let mut buf = vec![0u8; SUPERBLOCK_SIZE];
        let root_addr = write_group(&self.root, &mut buf, &self.options)?;
        let eof = buf.len() as u64;
        let sb = encode_superblock(root_addr, eof);
        buf[..SUPERBLOCK_SIZE].copy_from_slice(&sb);
        Ok(buf)
    }

    /// Compat-mode serialization: parent-first ordering, metadata block alignment.
    fn to_bytes_compat(&self) -> Result<Vec<u8>> {
        let meta_block_size = self.options.meta_block_size.unwrap_or(2048);
        let buf = write_tree_compat(&self.root, &self.options, meta_block_size)?;
        Ok(buf)
    }

    /// Serialize and write to a file on disk.
    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(Error::Io)
    }
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}
