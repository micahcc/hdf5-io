use crate::error::{Error, Result};

/// HDF5 datatype class IDs (from the on-disk encoding).
///
/// Reference: H5Tpublic.h `H5T_class_t`, and HDF5 File Format Spec section III.D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DatatypeClass {
    FixedPoint = 0,
    FloatingPoint = 1,
    Time = 2,
    String = 3,
    BitField = 4,
    Opaque = 5,
    Compound = 6,
    Reference = 7,
    Enum = 8,
    VarLen = 9,
    Array = 10,
}

impl DatatypeClass {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::FixedPoint),
            1 => Ok(Self::FloatingPoint),
            2 => Ok(Self::Time),
            3 => Ok(Self::String),
            4 => Ok(Self::BitField),
            5 => Ok(Self::Opaque),
            6 => Ok(Self::Compound),
            7 => Ok(Self::Reference),
            8 => Ok(Self::Enum),
            9 => Ok(Self::VarLen),
            10 => Ok(Self::Array),
            _ => Err(Error::UnsupportedDatatypeClass { class: v }),
        }
    }
}

/// Byte order for fixed-point and floating-point types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    /// VAX mixed-endian (rare, floating-point only).
    Vax,
}

/// String padding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPadding {
    NullTerminate,
    NullPad,
    SpacePad,
}

/// String character set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    Ascii,
    Utf8,
}

/// Reference type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceType {
    Object,
    DatasetRegion,
}

/// A decoded HDF5 datatype message.
///
/// ## On-disk layout (datatype message in object header)
///
/// ```text
/// Byte 0-3: class_and_version (4 bits class, 4 bits version, 24 bits class-specific bitfield)
/// Byte 4-7: size (4 bytes LE, total size of one element in bytes)
/// Byte 8+:  class-specific properties
/// ```
#[derive(Debug, Clone)]
pub enum Datatype {
    /// Fixed-point (integer) type.
    FixedPoint {
        size: u32,
        byte_order: ByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// IEEE floating-point type.
    FloatingPoint {
        size: u32,
        byte_order: ByteOrder,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Fixed-length string.
    String {
        size: u32,
        padding: StringPadding,
        char_set: CharacterSet,
    },
    /// Compound type (struct-like).
    Compound {
        size: u32,
        members: Vec<CompoundMember>,
    },
    /// Enumeration type.
    Enum {
        base: Box<Datatype>,
        members: Vec<EnumMember>,
    },
    /// Array type.
    Array {
        element_type: Box<Datatype>,
        dimensions: Vec<u32>,
    },
    /// Variable-length type.
    VarLen {
        element_type: Box<Datatype>,
    },
    /// Opaque type.
    Opaque {
        size: u32,
        tag: String,
    },
    /// Bitfield type.
    BitField {
        size: u32,
        byte_order: ByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Reference type.
    Reference {
        ref_type: ReferenceType,
    },
    /// Time type (rarely used).
    Time {
        size: u32,
        bit_precision: u16,
    },
}

/// A member of a compound datatype.
#[derive(Debug, Clone)]
pub struct CompoundMember {
    pub name: String,
    pub byte_offset: u32,
    pub datatype: Datatype,
}

/// A member of an enumeration datatype.
#[derive(Debug, Clone)]
pub struct EnumMember {
    pub name: String,
    pub value: Vec<u8>,
}

impl Datatype {
    /// The size of one element of this type in bytes.
    pub fn element_size(&self) -> u32 {
        match self {
            Self::FixedPoint { size, .. } => *size,
            Self::FloatingPoint { size, .. } => *size,
            Self::String { size, .. } => *size,
            Self::Compound { size, .. } => *size,
            Self::Enum { base, .. } => base.element_size(),
            Self::Array {
                element_type,
                dimensions,
            } => {
                let count: u32 = dimensions.iter().product();
                element_type.element_size() * count
            }
            Self::VarLen { .. } => {
                // HDF5 vlen is stored as a (size, pointer) pair in memory,
                // but on disk it's in the global heap. The "size" field in the
                // datatype message is typically 4+offset_size.
                // We'll return the on-disk element size from the message.
                // This should be overridden by the actual message size field.
                16
            }
            Self::Opaque { size, .. } => *size,
            Self::BitField { size, .. } => *size,
            Self::Reference { ref_type } => match ref_type {
                ReferenceType::Object => 8,
                ReferenceType::DatasetRegion => 12,
            },
            Self::Time { size, .. } => *size,
        }
    }

    /// Parse a datatype message from raw bytes.
    ///
    /// `data` should contain the full datatype message starting at the
    /// class+version word.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "datatype message too short".into(),
            });
        }

        let class_and_version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let class_id = (class_and_version & 0x0F) as u8;
        let version = ((class_and_version >> 4) & 0x0F) as u8;
        let class_bits = class_and_version >> 8; // 24-bit class-specific bitfield
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        let props = &data[8..];
        let class = DatatypeClass::from_u8(class_id)?;

        match class {
            DatatypeClass::FixedPoint => Self::parse_fixed_point(size, class_bits, props),
            DatatypeClass::FloatingPoint => Self::parse_floating_point(size, class_bits, props),
            DatatypeClass::String => Self::parse_string(size, class_bits),
            DatatypeClass::Compound => Self::parse_compound(size, version, props),
            DatatypeClass::Enum => Self::parse_enum(size, class_bits, props),
            DatatypeClass::Array => Self::parse_array(version, props),
            DatatypeClass::VarLen => Self::parse_varlen(class_bits, props),
            DatatypeClass::Opaque => Self::parse_opaque(size, class_bits, props),
            DatatypeClass::BitField => Self::parse_bitfield(size, class_bits, props),
            DatatypeClass::Reference => Self::parse_reference(class_bits),
            DatatypeClass::Time => Self::parse_time(size, props),
        }
    }

    fn parse_fixed_point(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 4 {
            return Err(Error::InvalidDatatype {
                msg: "fixed-point properties too short".into(),
            });
        }

        let byte_order = if (class_bits & 0x01) == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let signed = (class_bits & 0x08) != 0;

        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);

        Ok(Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_floating_point(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 12 {
            return Err(Error::InvalidDatatype {
                msg: "floating-point properties too short".into(),
            });
        }

        let byte_order_lo = class_bits & 0x01;
        let byte_order_hi = (class_bits >> 6) & 0x01;
        let byte_order = match (byte_order_hi, byte_order_lo) {
            (0, 0) => ByteOrder::LittleEndian,
            (0, 1) => ByteOrder::BigEndian,
            (1, 0) => ByteOrder::Vax,
            _ => {
                return Err(Error::InvalidDatatype {
                    msg: "invalid floating-point byte order".into(),
                })
            }
        };

        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);
        let exponent_location = props[4];
        let exponent_size = props[5];
        let mantissa_location = props[6];
        let mantissa_size = props[7];
        let exponent_bias = u32::from_le_bytes([props[8], props[9], props[10], props[11]]);

        Ok(Datatype::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        })
    }

    fn parse_string(size: u32, class_bits: u32) -> Result<Self> {
        let padding = match class_bits & 0x0F {
            0 => StringPadding::NullTerminate,
            1 => StringPadding::NullPad,
            2 => StringPadding::SpacePad,
            p => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown string padding type {}", p),
                })
            }
        };
        let char_set = match (class_bits >> 4) & 0x0F {
            0 => CharacterSet::Ascii,
            1 => CharacterSet::Utf8,
            c => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown string charset {}", c),
                })
            }
        };

        Ok(Datatype::String {
            size,
            padding,
            char_set,
        })
    }

    fn parse_compound(size: u32, version: u8, props: &[u8]) -> Result<Self> {
        // Compound type version 3 (used with datatype message version 3):
        //   Each member: name (null-terminated), byte offset (variable), datatype message
        // The number of members is encoded in the class bits (bits 0-15 of the 24-bit field),
        // but we received class_bits separately. For compound, the member count is
        // the 16-bit value in class_bits[0..15].
        // Actually, the member count is passed via the class_bits in the caller.
        // Let's re-parse from the full data for correctness.
        // For now, stub this out.
        let _ = (version, props);
        Ok(Datatype::Compound {
            size,
            members: Vec::new(), // TODO: parse members
        })
    }

    fn parse_enum(_size: u32, _class_bits: u32, _props: &[u8]) -> Result<Self> {
        // TODO: parse base type + member names + values
        Err(Error::InvalidDatatype {
            msg: "enum datatype parsing not yet implemented".into(),
        })
    }

    fn parse_array(_version: u8, _props: &[u8]) -> Result<Self> {
        // TODO: parse dimensions + element type
        Err(Error::InvalidDatatype {
            msg: "array datatype parsing not yet implemented".into(),
        })
    }

    fn parse_varlen(_class_bits: u32, _props: &[u8]) -> Result<Self> {
        // TODO: parse element type
        Err(Error::InvalidDatatype {
            msg: "varlen datatype parsing not yet implemented".into(),
        })
    }

    fn parse_opaque(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        // class_bits contains the tag length (lower 8 bits)
        let tag_len = (class_bits & 0xFF) as usize;
        if props.len() < tag_len {
            return Err(Error::InvalidDatatype {
                msg: "opaque tag extends past properties".into(),
            });
        }
        let tag = String::from_utf8_lossy(&props[..tag_len])
            .trim_end_matches('\0')
            .to_string();
        Ok(Datatype::Opaque { size, tag })
    }

    fn parse_bitfield(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 4 {
            return Err(Error::InvalidDatatype {
                msg: "bitfield properties too short".into(),
            });
        }
        let byte_order = if (class_bits & 0x01) == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);
        Ok(Datatype::BitField {
            size,
            byte_order,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_reference(class_bits: u32) -> Result<Self> {
        let ref_type = match class_bits & 0x0F {
            0 => ReferenceType::Object,
            1 => ReferenceType::DatasetRegion,
            r => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown reference type {}", r),
                })
            }
        };
        Ok(Datatype::Reference { ref_type })
    }

    fn parse_time(size: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 2 {
            return Err(Error::InvalidDatatype {
                msg: "time properties too short".into(),
            });
        }
        let bit_precision = u16::from_le_bytes([props[0], props[1]]);
        Ok(Datatype::Time {
            size,
            bit_precision,
        })
    }
}
