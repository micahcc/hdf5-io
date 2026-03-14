use std::io;

/// All errors that can occur while reading an HDF5 file.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid HDF5 signature at offset {offset}")]
    InvalidSignature { offset: u64 },

    #[error("unsupported superblock version {version} (only v2 and v3 are supported)")]
    UnsupportedSuperblockVersion { version: u8 },

    #[error("invalid superblock: {msg}")]
    InvalidSuperblock { msg: String },

    #[error("checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    #[error("invalid object header: {msg}")]
    InvalidObjectHeader { msg: String },

    #[error("unknown object header message type {type_id:#06x}")]
    UnknownMessageType { type_id: u16 },

    #[error("invalid datatype: {msg}")]
    InvalidDatatype { msg: String },

    #[error("unsupported datatype class {class}")]
    UnsupportedDatatypeClass { class: u8 },

    #[error("invalid dataspace: {msg}")]
    InvalidDataspace { msg: String },

    #[error("invalid data layout: {msg}")]
    InvalidLayout { msg: String },

    #[error("invalid B-tree v2: {msg}")]
    InvalidBTreeV2 { msg: String },

    #[error("invalid fractal heap: {msg}")]
    InvalidFractalHeap { msg: String },

    #[error("invalid filter pipeline: {msg}")]
    InvalidFilterPipeline { msg: String },

    #[error("unsupported filter {id}: {name}")]
    UnsupportedFilter { id: u16, name: String },

    #[error("decompression error: {msg}")]
    DecompressionError { msg: String },

    #[error("path not found: {path}")]
    PathNotFound { path: String },

    #[error("not a group: {path}")]
    NotAGroup { path: String },

    #[error("not a dataset: {path}")]
    NotADataset { path: String },

    #[error("address is undefined")]
    UndefinedAddress,

    #[error("{msg}")]
    Other { msg: String },
}

pub type Result<T> = std::result::Result<T, Error>;
