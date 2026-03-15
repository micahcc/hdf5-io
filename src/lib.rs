//! Pure Rust HDF5 file reader.
//!
//! Supports superblock version 2 and 3 (HDF5 1.8+ / 1.10+ files).
//! Designed for WASM compatibility — no C dependencies.
//!
//! # Quick Start
//!
//! ```no_run
//! use hdf5_pure::File;
//!
//! // From a file on disk
//! let file = File::open("data.h5").unwrap();
//! let root = file.root_group().unwrap();
//! let members = root.members().unwrap();
//! println!("Root group contains: {:?}", members);
//!
//! // Read a dataset
//! let ds = root.dataset("my_dataset").unwrap();
//! let shape = ds.shape().unwrap();
//! let raw_data = ds.read_raw().unwrap();
//! ```

pub mod btree2;
pub mod checksum;
pub mod chunk;
pub mod dataspace;
pub mod datatype;
pub mod error;
pub mod file;
pub mod filters;
pub mod fractal_heap;
pub mod global_heap;
pub mod io;
pub mod layout;
pub mod link;
pub mod object_header;
pub mod superblock;

// Re-export the main public types at crate root.
pub use dataspace::Dataspace;
pub use datatype::Datatype;
pub use error::Error;
pub use error::Result;
pub use file::Attribute;
pub use file::Dataset;
pub use file::File;
pub use file::FillValue;
pub use file::Group;
pub use file::Node;
pub use io::ReadAt;
pub use layout::DataLayout;
pub use superblock::Superblock;
