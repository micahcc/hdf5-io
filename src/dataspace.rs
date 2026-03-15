use crate::error::Error;
use crate::error::Result;

/// A decoded HDF5 dataspace message.
///
/// ## On-disk layout (dataspace message, version 2)
///
/// ```text
/// Byte 0:    Version (2)
/// Byte 1:    Dimensionality (rank, 0-32)
/// Byte 2:    Flags (bit 0: max dims present, bit 1: permutation indices present [never used])
/// Byte 3:    Type (0=scalar, 1=simple, 2=null)
/// Byte 4+:   Dimensions (rank * 8 bytes each, u64 LE)
///            Max dimensions (rank * 8 bytes, only if flag bit 0 set; UNDEF_SIZE = unlimited)
/// ```
#[derive(Debug, Clone)]
pub enum Dataspace {
    /// Scalar dataspace (single element, rank 0).
    Scalar,
    /// Null dataspace (no data).
    Null,
    /// Simple (regular N-dimensional array).
    Simple {
        dimensions: Vec<u64>,
        max_dimensions: Option<Vec<u64>>,
    },
}

/// The sentinel value for "unlimited" in max dimensions.
pub const UNLIMITED: u64 = u64::MAX;

impl Dataspace {
    /// Total number of elements in this dataspace.
    pub fn num_elements(&self) -> u64 {
        match self {
            Dataspace::Scalar => 1,
            Dataspace::Null => 0,
            Dataspace::Simple { dimensions, .. } => dimensions.iter().product(),
        }
    }

    /// The rank (number of dimensions).
    pub fn rank(&self) -> usize {
        match self {
            Dataspace::Scalar | Dataspace::Null => 0,
            Dataspace::Simple { dimensions, .. } => dimensions.len(),
        }
    }

    /// The shape (dimension sizes), or empty for scalar/null.
    pub fn shape(&self) -> &[u64] {
        match self {
            Dataspace::Scalar | Dataspace::Null => &[],
            Dataspace::Simple { dimensions, .. } => dimensions,
        }
    }

    /// The maximum dimensions, if present.
    pub fn max_dimensions(&self) -> Option<&[u64]> {
        match self {
            Dataspace::Simple {
                max_dimensions: Some(md),
                ..
            } => Some(md),
            _ => None,
        }
    }

    /// Parse a dataspace message from raw bytes (the message body, not including
    /// the object header message prefix).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidDataspace {
                msg: "empty dataspace message".into(),
            });
        }

        let version = data[0];
        match version {
            1 => Self::parse_v1(data),
            2 => Self::parse_v2(data),
            _ => Err(Error::InvalidDataspace {
                msg: format!("unsupported dataspace version {}", version),
            }),
        }
    }

    fn parse_v1(data: &[u8]) -> Result<Self> {
        // Version 1 layout:
        //   0: version (1)
        //   1: rank
        //   2: flags (bit 0 = max dims present)
        //   3: reserved (was type in early versions, but always 1 for simple)
        //   4-7: reserved
        //   8+: dimensions (rank * 8), then optionally max dims (rank * 8)
        if data.len() < 8 {
            return Err(Error::InvalidDataspace {
                msg: "v1 dataspace message too short".into(),
            });
        }
        let rank = data[1] as usize;
        let flags = data[2];
        let has_max = (flags & 0x01) != 0;

        if rank == 0 {
            return Ok(Dataspace::Scalar);
        }

        let dim_start = 8;
        let needed = dim_start + rank * 8 + if has_max { rank * 8 } else { 0 };
        if data.len() < needed {
            return Err(Error::InvalidDataspace {
                msg: "v1 dataspace message truncated".into(),
            });
        }

        let dimensions = Self::read_dims(data, dim_start, rank);
        let max_dimensions = if has_max {
            Some(Self::read_dims(data, dim_start + rank * 8, rank))
        } else {
            None
        };

        Ok(Dataspace::Simple {
            dimensions,
            max_dimensions,
        })
    }

    fn parse_v2(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidDataspace {
                msg: "v2 dataspace message too short".into(),
            });
        }

        let rank = data[1] as usize;
        let flags = data[2];
        let ds_type = data[3];

        match ds_type {
            0 => return Ok(Dataspace::Scalar),
            2 => return Ok(Dataspace::Null),
            1 => {} // simple, continue parsing
            _ => {
                return Err(Error::InvalidDataspace {
                    msg: format!("unknown dataspace type {}", ds_type),
                });
            }
        }

        let has_max = (flags & 0x01) != 0;
        let dim_start = 4;
        let needed = dim_start + rank * 8 + if has_max { rank * 8 } else { 0 };
        if data.len() < needed {
            return Err(Error::InvalidDataspace {
                msg: "v2 dataspace message truncated".into(),
            });
        }

        let dimensions = Self::read_dims(data, dim_start, rank);
        let max_dimensions = if has_max {
            Some(Self::read_dims(data, dim_start + rank * 8, rank))
        } else {
            None
        };

        Ok(Dataspace::Simple {
            dimensions,
            max_dimensions,
        })
    }

    fn read_dims(data: &[u8], start: usize, count: usize) -> Vec<u64> {
        (0..count)
            .map(|i| {
                let off = start + i * 8;
                u64::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ])
            })
            .collect()
    }
}
