//! Pure Rust HDF5 file reader.
//!
//! Supports superblock version 2 and 3 (HDF5 1.8+ / 1.10+ files).
//! Designed for WASM compatibility — no C dependencies.
//!
//! # Quick Start
//!
//! ```no_run
//! use hdf5_reader::File;
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
pub mod dataspace;
pub mod datatype;
pub mod error;
pub mod file;
pub mod filters;
pub mod fractal_heap;
pub mod io;
pub mod layout;
pub mod link;
pub mod object_header;
pub mod superblock;

// Re-export the main public types at crate root.
pub use error::{Error, Result};
pub use file::{Attribute, Dataset, File, Group, Node};
pub use dataspace::Dataspace;
pub use datatype::Datatype;
pub use io::ReadAt;
pub use layout::DataLayout;
pub use superblock::Superblock;
