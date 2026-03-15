use crate::error::Error;
use crate::error::Result;

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
    Complex = 11,
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
            11 => Ok(Self::Complex),
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
        /// True if this is a vlen string (class_bits type=1), false for sequence.
        is_string: bool,
        /// String padding (only for vlen strings).
        padding: Option<StringPadding>,
        /// Character set (only for vlen strings).
        char_set: Option<CharacterSet>,
    },
    /// Opaque type.
    Opaque { size: u32, tag: String },
    /// Bitfield type.
    BitField {
        size: u32,
        byte_order: ByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Reference type.
    Reference { ref_type: ReferenceType },
    /// Time type (rarely used).
    Time { size: u32, bit_precision: u16 },
    /// Complex number type (HDF5 2.0+).
    ///
    /// On-disk: two consecutive values of the base floating-point type
    /// (real part, then imaginary part). `size` = 2 * base element size.
    Complex { size: u32, base: Box<Datatype> },
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
            Self::Complex { size, .. } => *size,
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
            DatatypeClass::Compound => {
                let nmembers = (class_bits & 0xFFFF) as usize;
                Self::parse_compound(size, version, nmembers, props)
            }
            DatatypeClass::Enum => Self::parse_enum(size, version, class_bits, props),
            DatatypeClass::Array => Self::parse_array(version, props),
            DatatypeClass::VarLen => Self::parse_varlen(class_bits, props),
            DatatypeClass::Opaque => Self::parse_opaque(size, class_bits, props),
            DatatypeClass::BitField => Self::parse_bitfield(size, class_bits, props),
            DatatypeClass::Reference => Self::parse_reference(class_bits),
            DatatypeClass::Time => Self::parse_time(size, props),
            DatatypeClass::Complex => Self::parse_complex(size, props),
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
            (1, 1) => ByteOrder::Vax, // VAX requires both bits set (H5Odtype.c:189-201)
            _ => {
                return Err(Error::InvalidDatatype {
                    msg: "invalid floating-point byte order".into(),
                });
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
                });
            }
        };
        let char_set = match (class_bits >> 4) & 0x0F {
            0 => CharacterSet::Ascii,
            1 => CharacterSet::Utf8,
            c => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown string charset {}", c),
                });
            }
        };

        Ok(Datatype::String {
            size,
            padding,
            char_set,
        })
    }

    fn parse_compound(size: u32, version: u8, nmembers: usize, props: &[u8]) -> Result<Self> {
        // Compound type layout depends on version:
        //   v1: name (null-term, padded to 8), byte_offset(4), dimensionality(1),
        //       padding(3), dim_perm(4), padding(4), dim_sizes(4*4), datatype(8+)
        //   v2: name (null-term, padded to 8), byte_offset(4), datatype(8+)
        //   v3: name (null-term, NOT padded), byte_offset(1/2/4 based on type size),
        //       datatype(8+)
        let mut members = Vec::with_capacity(nmembers);
        let mut pos = 0;

        for _ in 0..nmembers {
            if pos >= props.len() {
                break;
            }

            // Read null-terminated name
            let name_start = pos;
            while pos < props.len() && props[pos] != 0 {
                pos += 1;
            }
            let name = String::from_utf8_lossy(&props[name_start..pos]).to_string();
            let name_len = pos - name_start;
            pos += 1; // skip null terminator

            // V1/v2: pad name (incl. null terminator) to 8-byte multiple from name start.
            // C formula: *pp += ((strlen + 8) / 8) * 8  (H5Odtype.c:427)
            if version <= 2 {
                pos = name_start + ((name_len + 8) / 8) * 8;
            }

            // Byte offset of this member within the compound type
            let byte_offset = match version {
                1 | 2 => {
                    if pos + 4 > props.len() {
                        return Err(Error::InvalidDatatype {
                            msg: "compound member offset truncated".into(),
                        });
                    }
                    let off = u32::from_le_bytes([
                        props[pos],
                        props[pos + 1],
                        props[pos + 2],
                        props[pos + 3],
                    ]);
                    pos += 4;
                    off
                }
                3..=5 => {
                    // Offset width: H5VM_limit_enc_size(size) = (log2(size)/8) + 1
                    let off_size = if size <= 0xFF {
                        1
                    } else if size <= 0xFFFF {
                        2
                    } else if size <= 0xFFFFFF {
                        3
                    } else {
                        4
                    };
                    if pos + off_size > props.len() {
                        return Err(Error::InvalidDatatype {
                            msg: "compound member offset truncated".into(),
                        });
                    }
                    let off = match off_size {
                        1 => props[pos] as u32,
                        2 => u16::from_le_bytes([props[pos], props[pos + 1]]) as u32,
                        3 => {
                            (props[pos] as u32)
                                | ((props[pos + 1] as u32) << 8)
                                | ((props[pos + 2] as u32) << 16)
                        }
                        _ => u32::from_le_bytes([
                            props[pos],
                            props[pos + 1],
                            props[pos + 2],
                            props[pos + 3],
                        ]),
                    };
                    pos += off_size;
                    off
                }
                _ => {
                    return Err(Error::InvalidDatatype {
                        msg: format!("unsupported compound datatype version {}", version),
                    });
                }
            };

            // V1 has extra fields: dimensionality(1) + padding(3) + dim_perm(4)
            //                       + padding(4) + dim_sizes(4*4 = 16)
            if version == 1 {
                pos += 1 + 3 + 4 + 4 + 16; // 28 bytes of v1-specific fields
            }

            // Parse the member's datatype (recursive)
            if pos + 8 > props.len() {
                return Err(Error::InvalidDatatype {
                    msg: "compound member datatype truncated".into(),
                });
            }
            let member_dt = Datatype::parse(&props[pos..])?;
            // The datatype message size: 8 (header) + properties
            let dt_total_size = Self::encoded_size(&props[pos..])?;
            pos += dt_total_size;

            members.push(CompoundMember {
                name,
                byte_offset,
                datatype: member_dt,
            });
        }

        Ok(Datatype::Compound { size, members })
    }

    /// Compute the encoded size of a datatype message starting at `data`.
    fn encoded_size(data: &[u8]) -> Result<usize> {
        if data.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "datatype too short to determine size".into(),
            });
        }
        let class_and_version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let class_id = (class_and_version & 0x0F) as u8;
        let version = ((class_and_version >> 4) & 0x0F) as u8;
        let class_bits = class_and_version >> 8;
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        let props_size = match class_id {
            0 => 4,  // FixedPoint: bit_offset(2) + bit_precision(2)
            1 => 12, // FloatingPoint
            2 => 2,  // Time
            3 => 0,  // String (all info in class_bits)
            4 => 4,  // BitField
            5 => {
                // Opaque: tag (class_bits & 0xFF) padded to 8
                let tag_len = (class_bits & 0xFF) as usize;
                (tag_len + 7) & !7
            }
            6 => {
                // Compound: need to sum up all member sizes (recursive)
                let nmembers = (class_bits & 0xFFFF) as usize;
                let mut pos = 0;
                let props = &data[8..];
                for _ in 0..nmembers {
                    // name (with padding from name start for v1/v2)
                    let name_start = pos;
                    while pos < props.len() && props[pos] != 0 {
                        pos += 1;
                    }
                    let name_len = pos - name_start;
                    pos += 1;
                    if version < 3 {
                        pos = name_start + ((name_len + 8) / 8) * 8;
                    }
                    // byte offset
                    if version >= 3 {
                        let off_size = if size <= 0xFF {
                            1
                        } else if size <= 0xFFFF {
                            2
                        } else if size <= 0xFFFFFF {
                            3
                        } else {
                            4
                        };
                        pos += off_size;
                    } else {
                        pos += 4;
                    }
                    // v1 extra fields
                    if version == 1 {
                        pos += 28;
                    }
                    // recursive datatype
                    if pos + 8 <= props.len() {
                        let sub_size = Self::encoded_size(&props[pos..])?;
                        pos += sub_size;
                    }
                }
                pos
            }
            7 => 0, // Reference
            8 => {
                // Enum: base_type + names + values (H5Odtype.c:1715-1730)
                let nmembers = (class_bits & 0xFFFF) as usize;
                let props = &data[8..];
                // Base type
                let base_encoded = Self::encoded_size(props)?;
                let base_elem_size =
                    u32::from_le_bytes([props[4], props[5], props[6], props[7]]) as usize;
                let mut pos = base_encoded;
                // Member names
                for _ in 0..nmembers {
                    let name_start = pos;
                    while pos < props.len() && props[pos] != 0 {
                        pos += 1;
                    }
                    let name_len = pos - name_start;
                    pos += 1;
                    if version < 3 {
                        pos = name_start + ((name_len + 8) / 8) * 8;
                    }
                }
                // Member values (packed)
                pos += nmembers * base_elem_size;
                pos
            }
            9 => {
                // VarLen: just the base type (H5Odtype.c:1732-1734)
                let props = &data[8..];
                if props.len() >= 8 {
                    Self::encoded_size(props)?
                } else {
                    0
                }
            }
            10 => {
                // Array (H5Odtype.c:1736-1744)
                let props = &data[8..];
                if props.is_empty() {
                    return Err(Error::InvalidDatatype {
                        msg: "array encoded_size: no properties".into(),
                    });
                }
                let ndims = props[0] as usize;
                let mut pos = 1;
                if version < 3 {
                    pos += 3; // 3 reserved bytes
                }
                pos += 4 * ndims; // dimension sizes
                if version < 3 {
                    pos += 4 * ndims; // permutation indices
                }
                // Base type
                if pos + 8 <= props.len() {
                    pos += Self::encoded_size(&props[pos..])?;
                }
                pos
            }
            11 => {
                // Complex: base floating-point type
                if data.len() > 8 {
                    Self::encoded_size(&data[8..])?
                } else {
                    0
                }
            }
            _ => 0,
        };

        Ok(8 + props_size)
    }

    fn parse_enum(_size: u32, version: u8, class_bits: u32, props: &[u8]) -> Result<Self> {
        let nmembers = (class_bits & 0xFFFF) as usize;

        // First: the base datatype message (at least 8 bytes)
        if props.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "enum base type truncated".into(),
            });
        }
        let base = Box::new(Datatype::parse(props)?);
        let base_size = Self::encoded_size(props)?;
        let mut pos = base_size;

        // Then: member names (null-terminated, padded to 8 in v1/v2, unpadded in v3)
        let mut members = Vec::with_capacity(nmembers);
        for _ in 0..nmembers {
            let name_start = pos;
            while pos < props.len() && props[pos] != 0 {
                pos += 1;
            }
            let name = String::from_utf8_lossy(&props[name_start..pos]).to_string();
            pos += 1; // skip null
            if version < 3 {
                pos = (pos + 7) & !7;
            }
            members.push(EnumMember {
                name,
                value: Vec::new(),
            });
        }

        // Then: member values (each is base_type.element_size() bytes)
        let val_size = base.element_size() as usize;
        for member in &mut members {
            if pos + val_size <= props.len() {
                member.value = props[pos..pos + val_size].to_vec();
                pos += val_size;
            }
        }

        Ok(Datatype::Enum { base, members })
    }

    fn parse_array(version: u8, props: &[u8]) -> Result<Self> {
        if props.len() < 1 {
            return Err(Error::InvalidDatatype {
                msg: "array properties too short".into(),
            });
        }

        let ndims = props[0] as usize;
        let mut pos = 1;

        // Version 1 has 3 bytes of padding after ndims
        if version < 2 {
            pos += 3;
        }

        // Read dimension sizes (ndims * 4 bytes)
        if pos + ndims * 4 > props.len() {
            return Err(Error::InvalidDatatype {
                msg: "array dimension sizes truncated".into(),
            });
        }
        let mut dimensions = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let dim =
                u32::from_le_bytes([props[pos], props[pos + 1], props[pos + 2], props[pos + 3]]);
            dimensions.push(dim);
            pos += 4;
        }

        // Version 1 has ndims * 4 bytes of permutation indices (unused, must be zeros)
        if version < 2 {
            pos += ndims * 4;
        }

        // Parse the element type (recursive)
        if pos + 8 > props.len() {
            return Err(Error::InvalidDatatype {
                msg: "array element type truncated".into(),
            });
        }
        let element_type = Box::new(Datatype::parse(&props[pos..])?);

        Ok(Datatype::Array {
            element_type,
            dimensions,
        })
    }

    fn parse_varlen(class_bits: u32, props: &[u8]) -> Result<Self> {
        // class_bits: bits 0-3 = type (0=sequence, 1=string)
        //             bits 4-7 = padding type (for strings)
        //             bits 8-11 = character set (for strings)
        let vlen_type = class_bits & 0x0F;

        if vlen_type == 1 {
            // Variable-length string: element_type is the character type but
            // we represent it as a VarLen with the string's charset info embedded.
            let padding = match (class_bits >> 4) & 0x0F {
                0 => StringPadding::NullTerminate,
                1 => StringPadding::NullPad,
                2 => StringPadding::SpacePad,
                _ => StringPadding::NullTerminate,
            };
            let char_set = match (class_bits >> 8) & 0x0F {
                0 => CharacterSet::Ascii,
                1 => CharacterSet::Utf8,
                _ => CharacterSet::Ascii,
            };
            // The base type in props describes the character type but for vlen strings
            // we wrap it as String info
            let element_type = if props.len() >= 8 {
                Box::new(Datatype::parse(props)?)
            } else {
                // Fallback: assume u8 characters
                Box::new(Datatype::FixedPoint {
                    size: 1,
                    byte_order: ByteOrder::LittleEndian,
                    signed: false,
                    bit_offset: 0,
                    bit_precision: 8,
                })
            };
            Ok(Datatype::VarLen {
                element_type,
                is_string: true,
                padding: Some(padding),
                char_set: Some(char_set),
            })
        } else {
            // Variable-length sequence
            if props.len() < 8 {
                return Err(Error::InvalidDatatype {
                    msg: "vlen element type truncated".into(),
                });
            }
            let element_type = Box::new(Datatype::parse(props)?);
            Ok(Datatype::VarLen {
                element_type,
                is_string: false,
                padding: None,
                char_set: None,
            })
        }
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
                });
            }
        };
        Ok(Datatype::Reference { ref_type })
    }

    fn parse_complex(size: u32, props: &[u8]) -> Result<Self> {
        // Properties contain the base floating-point datatype message.
        if props.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "complex base type truncated".into(),
            });
        }
        let base = Box::new(Datatype::parse(props)?);
        Ok(Datatype::Complex { size, base })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a raw datatype message from class, version, class_bits, size, and properties.
    fn build_dt_msg(class: u8, version: u8, class_bits: u32, size: u32, props: &[u8]) -> Vec<u8> {
        let cv = (class as u32) | ((version as u32) << 4) | (class_bits << 8);
        let mut buf = Vec::new();
        buf.extend_from_slice(&cv.to_le_bytes());
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(props);
        buf
    }

    #[test]
    fn parse_fixed_point_i32_le() {
        // class=0 (FixedPoint), version=1, signed (bit 3 of class_bits), LE (bit 0 = 0)
        let props = [0, 0, 32, 0]; // bit_offset=0, bit_precision=32
        let msg = build_dt_msg(0, 1, 0x08, 4, &props); // 0x08 = signed
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                assert_eq!(size, 4);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert!(signed);
                assert_eq!(bit_offset, 0);
                assert_eq!(bit_precision, 32);
            }
            other => panic!("expected FixedPoint, got {:?}", other),
        }
    }

    #[test]
    fn parse_fixed_point_u8_be() {
        // class=0, version=1, unsigned (no bit 3), BE (bit 0 = 1)
        let props = [0, 0, 8, 0]; // bit_offset=0, bit_precision=8
        let msg = build_dt_msg(0, 1, 0x01, 1, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::FixedPoint {
                size,
                byte_order,
                signed,
                ..
            } => {
                assert_eq!(size, 1);
                assert_eq!(byte_order, ByteOrder::BigEndian);
                assert!(!signed);
            }
            other => panic!("expected FixedPoint, got {:?}", other),
        }
    }

    #[test]
    fn parse_float_f64_le() {
        // class=1, version=1, LE (byte_order_lo=0, byte_order_hi=0)
        let mut props = vec![0u8; 12];
        props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision=64
        props[4] = 52; // exponent_location
        props[5] = 11; // exponent_size
        props[6] = 0; // mantissa_location
        props[7] = 52; // mantissa_size
        props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exponent_bias
        let msg = build_dt_msg(1, 1, 0, 8, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::FloatingPoint {
                size,
                byte_order,
                exponent_bias,
                mantissa_size,
                ..
            } => {
                assert_eq!(size, 8);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert_eq!(exponent_bias, 1023);
                assert_eq!(mantissa_size, 52);
            }
            other => panic!("expected FloatingPoint, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_ascii_nullterm() {
        // class=3, version=1, padding=NullTerminate(0), charset=Ascii(0)
        let msg = build_dt_msg(3, 1, 0x00, 10, &[]);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::String {
                size,
                padding,
                char_set,
            } => {
                assert_eq!(size, 10);
                assert_eq!(padding, StringPadding::NullTerminate);
                assert_eq!(char_set, CharacterSet::Ascii);
            }
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn parse_string_utf8_nullpad() {
        // padding=NullPad(1), charset=Utf8(1)
        // class_bits = (charset << 4) | padding = 0x11
        let msg = build_dt_msg(3, 1, 0x11, 32, &[]);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::String {
                padding, char_set, ..
            } => {
                assert_eq!(padding, StringPadding::NullPad);
                assert_eq!(char_set, CharacterSet::Utf8);
            }
            other => panic!("expected String, got {:?}", other),
        }
    }

    #[test]
    fn parse_compound_v3_two_members() {
        // Compound v3: unpadded names, variable-width offsets
        // Compound size=8 → offset width=1 byte
        // Member 1: "a" at offset 0, type = u32 LE (FixedPoint, 4 bytes, class_bits=0x08)
        // Member 2: "b" at offset 4, type = u32 LE
        let mut props = Vec::new();
        // Member "a": name "a\0", offset=0, then u32 LE datatype
        props.extend_from_slice(b"a\0");
        props.push(0); // offset = 0
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
        // Member "b": name "b\0", offset=4, then u32 LE datatype
        props.extend_from_slice(b"b\0");
        props.push(4); // offset = 4
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        // class=6 (Compound), version=3, class_bits lower 16 = nmembers=2
        let msg = build_dt_msg(6, 3, 2, 8, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 8);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "a");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "b");
                assert_eq!(members[1].byte_offset, 4);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn parse_enum_v3_i8() {
        // Enum v3: base type = i8 LE, 2 members
        let mut props = Vec::new();
        // Base type: i8 LE signed (class=0, version=3, class_bits=0x08, size=1)
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 1, &[0, 0, 8, 0]));
        // V3 names: unpadded, null-terminated
        props.extend_from_slice(b"OFF\0");
        props.extend_from_slice(b"ON\0");
        // Values: 0, 1
        props.push(0);
        props.push(1);

        // class=8 (Enum), version=3, nmembers=2
        let msg = build_dt_msg(8, 3, 2, 1, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Enum { base, members } => {
                assert_eq!(base.element_size(), 1);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "OFF");
                assert_eq!(members[0].value, vec![0]);
                assert_eq!(members[1].name, "ON");
                assert_eq!(members[1].value, vec![1]);
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn parse_array_v2_1d() {
        // Array v2: ndims=1, dim[0]=5, element type = f32 LE
        let mut props = Vec::new();
        props.push(1); // ndims
        // v2: no padding after ndims
        props.extend_from_slice(&5u32.to_le_bytes()); // dimension[0] = 5
        // Element type: f32 LE
        let mut f32_props = vec![0u8; 12];
        f32_props[2..4].copy_from_slice(&32u16.to_le_bytes()); // bit_precision=32
        f32_props[4] = 23; // exponent_location
        f32_props[5] = 8; // exponent_size
        f32_props[6] = 0; // mantissa_location
        f32_props[7] = 23; // mantissa_size
        f32_props[8..12].copy_from_slice(&127u32.to_le_bytes()); // exponent_bias
        props.extend_from_slice(&build_dt_msg(1, 1, 0, 4, &f32_props));

        // class=10 (Array), version=2, size doesn't matter (computed from dims * element)
        let msg = build_dt_msg(10, 2, 0, 20, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Array {
                ref element_type,
                ref dimensions,
            } => {
                assert_eq!(dimensions, &vec![5]);
                assert_eq!(element_type.element_size(), 4);
                assert_eq!(dt.element_size(), 20); // 5 * 4
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    #[test]
    fn parse_array_v1_with_padding() {
        // Array v1: ndims=1, then 3 bytes padding, dim[0]=3, then 4 bytes perm, element type
        let mut props = Vec::new();
        props.push(1); // ndims
        props.extend_from_slice(&[0, 0, 0]); // v1 padding
        props.extend_from_slice(&3u32.to_le_bytes()); // dimension[0] = 3
        props.extend_from_slice(&0u32.to_le_bytes()); // v1 permutation[0] = 0
        // Element type: i16 LE signed
        props.extend_from_slice(&build_dt_msg(0, 1, 0x08, 2, &[0, 0, 16, 0]));

        let msg = build_dt_msg(10, 1, 0, 6, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Array {
                ref element_type,
                ref dimensions,
            } => {
                assert_eq!(dimensions, &vec![3]);
                assert_eq!(element_type.element_size(), 2);
                assert_eq!(dt.element_size(), 6); // 3 * 2
            }
            other => panic!("expected Array, got {:?}", other),
        }
    }

    #[test]
    fn parse_opaque() {
        // class=5 (Opaque), tag_len in class_bits lower 8 = 4
        let props = b"test\0\0\0\0"; // tag "test", padded to 8
        let msg = build_dt_msg(5, 1, 4, 16, props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Opaque { size, tag } => {
                assert_eq!(size, 16);
                assert_eq!(tag, "test");
            }
            other => panic!("expected Opaque, got {:?}", other),
        }
    }

    #[test]
    fn element_size_compound() {
        let dt = Datatype::Compound {
            size: 24,
            members: vec![],
        };
        assert_eq!(dt.element_size(), 24);
    }

    #[test]
    fn element_size_enum() {
        let dt = Datatype::Enum {
            base: Box::new(Datatype::FixedPoint {
                size: 4,
                byte_order: ByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            }),
            members: vec![],
        };
        assert_eq!(dt.element_size(), 4);
    }

    #[test]
    fn element_size_array_2d() {
        let dt = Datatype::Array {
            element_type: Box::new(Datatype::FixedPoint {
                size: 8,
                byte_order: ByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 64,
            }),
            dimensions: vec![3, 4],
        };
        assert_eq!(dt.element_size(), 96); // 3 * 4 * 8
    }

    #[test]
    fn reject_too_short_message() {
        let data = [0u8; 4];
        assert!(Datatype::parse(&data).is_err());
    }

    #[test]
    fn parse_complex_f64() {
        // Complex of f64 LE: class=11, size=16, props = base f64 datatype
        let mut base_props = vec![0u8; 12];
        base_props[2..4].copy_from_slice(&64u16.to_le_bytes()); // bit_precision=64
        base_props[4] = 52; // exponent_location
        base_props[5] = 11; // exponent_size
        base_props[6] = 0; // mantissa_location
        base_props[7] = 52; // mantissa_size
        base_props[8..12].copy_from_slice(&1023u32.to_le_bytes()); // exponent_bias
        let base_msg = build_dt_msg(1, 1, 0, 8, &base_props);

        let msg = build_dt_msg(11, 1, 0, 16, &base_msg);
        let dt = Datatype::parse(&msg).unwrap();
        match &dt {
            Datatype::Complex { size, base } => {
                assert_eq!(*size, 16);
                assert_eq!(base.element_size(), 8);
                match base.as_ref() {
                    Datatype::FloatingPoint {
                        byte_order,
                        mantissa_size,
                        ..
                    } => {
                        assert_eq!(*byte_order, ByteOrder::LittleEndian);
                        assert_eq!(*mantissa_size, 52);
                    }
                    other => panic!("expected FloatingPoint base, got {:?}", other),
                }
            }
            other => panic!("expected Complex, got {:?}", other),
        }
        assert_eq!(dt.element_size(), 16);
    }

    #[test]
    fn reject_unknown_class() {
        let msg = build_dt_msg(15, 1, 0, 4, &[]);
        assert!(Datatype::parse(&msg).is_err());
    }

    #[test]
    fn compound_v2_name_padding_from_name_start() {
        // Compound v2: member 1 has String type (encoded_size=8), so member 2's
        // name starts at props offset 20 (8+4+8), which is NOT 8-aligned globally.
        // The v1/v2 padding must be relative to name_start, not absolute.
        let mut props = Vec::new();

        // Member "s": name padded to 8, offset(4), String type(8) = 20 bytes
        let mut name1 = vec![0u8; 8]; // ((1+8)/8)*8 = 8
        name1[0] = b's';
        props.extend_from_slice(&name1);
        props.extend_from_slice(&0u32.to_le_bytes()); // offset = 0
        props.extend_from_slice(&build_dt_msg(3, 2, 0x00, 4, &[])); // String ASCII

        // Member "n" starts at props offset 20. Name "n" padded to 8 from name_start.
        let mut name2 = vec![0u8; 8]; // ((1+8)/8)*8 = 8
        name2[0] = b'n';
        props.extend_from_slice(&name2);
        props.extend_from_slice(&4u32.to_le_bytes()); // offset = 4
        props.extend_from_slice(&build_dt_msg(0, 2, 0x08, 8, &[0, 0, 64, 0])); // i64 LE

        let msg = build_dt_msg(6, 2, 2, 12, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 12);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "s");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "n");
                assert_eq!(members[1].byte_offset, 4);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn compound_v3_three_byte_offset() {
        // Compound v3 with size > 65535 needs 3-byte member offset encoding.
        let compound_size: u32 = 70000;
        let mut props = Vec::new();

        // Member "a": name "a\0" (unpadded v3), 3-byte offset=0, i32 type
        props.extend_from_slice(b"a\0");
        props.extend_from_slice(&[0, 0, 0]); // 3 bytes LE = 0
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        // Member "b": name "b\0", 3-byte offset=69996, i32 type
        props.extend_from_slice(b"b\0");
        let off = 69996u32.to_le_bytes();
        props.extend_from_slice(&off[..3]); // lower 3 bytes
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        let msg = build_dt_msg(6, 3, 2, compound_size, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, compound_size);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "a");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "b");
                assert_eq!(members[1].byte_offset, 69996);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn compound_with_enum_member_encoded_size() {
        // Compound v3 with an enum member followed by i32. Tests that
        // encoded_size properly computes enum size (not data.len()).
        let mut props = Vec::new();

        // Member "e" at offset 0: Enum(i32, 2 members)
        props.extend_from_slice(b"e\0");
        props.push(0); // offset = 0 (1 byte, size=8)
        let mut enum_props = Vec::new();
        enum_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
        enum_props.extend_from_slice(b"A\0");
        enum_props.extend_from_slice(b"B\0");
        enum_props.extend_from_slice(&0i32.to_le_bytes());
        enum_props.extend_from_slice(&1i32.to_le_bytes());
        props.extend_from_slice(&build_dt_msg(8, 3, 2, 4, &enum_props));

        // Member "n" at offset 4: i32
        props.extend_from_slice(b"n\0");
        props.push(4); // offset = 4
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        let msg = build_dt_msg(6, 3, 2, 8, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 8);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "e");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "n");
                assert_eq!(members[1].byte_offset, 4);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn compound_with_array_member_encoded_size() {
        // Compound v3 with an array member followed by i32.
        let mut props = Vec::new();

        // Member "arr" at offset 0: Array(i32, dim=[3])
        props.extend_from_slice(b"arr\0");
        props.push(0); // offset = 0
        let mut arr_props = Vec::new();
        arr_props.push(1); // ndims = 1
        arr_props.extend_from_slice(&3u32.to_le_bytes()); // dim[0] = 3
        arr_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
        props.extend_from_slice(&build_dt_msg(10, 3, 0, 12, &arr_props));

        // Member "n" at offset 12: i32
        props.extend_from_slice(b"n\0");
        props.push(12); // offset = 12
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        let msg = build_dt_msg(6, 3, 2, 16, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 16);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "arr");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "n");
                assert_eq!(members[1].byte_offset, 12);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn compound_with_vlen_member_encoded_size() {
        // Compound v3 with a vlen member followed by i32.
        let mut props = Vec::new();

        // Member "v" at offset 0: VarLen(i32), sequence type
        props.extend_from_slice(b"v\0");
        props.push(0); // offset = 0
        let base = build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]);
        props.extend_from_slice(&build_dt_msg(9, 3, 0, 16, &base));

        // Member "n" at offset 16: i32
        props.extend_from_slice(b"n\0");
        props.push(16); // offset = 16
        props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));

        let msg = build_dt_msg(6, 3, 2, 20, &props);
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 20);
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "v");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "n");
                assert_eq!(members[1].byte_offset, 16);
            }
            other => panic!("expected Compound, got {:?}", other),
        }
    }

    #[test]
    fn encoded_size_vlen() {
        // VarLen(i32): 8 header + 12 base type = 20
        let base = build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]);
        let msg = build_dt_msg(9, 3, 0, 16, &base);
        assert_eq!(Datatype::encoded_size(&msg).unwrap(), 20);
    }

    #[test]
    fn encoded_size_enum() {
        // Enum v3(i32, 2 members "A","B"): 8 + 12(base) + 2("A\0") + 2("B\0") + 8(2*4 vals) = 32
        let mut enum_props = Vec::new();
        enum_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
        enum_props.extend_from_slice(b"A\0");
        enum_props.extend_from_slice(b"B\0");
        enum_props.extend_from_slice(&0i32.to_le_bytes());
        enum_props.extend_from_slice(&1i32.to_le_bytes());
        let msg = build_dt_msg(8, 3, 2, 4, &enum_props);
        assert_eq!(Datatype::encoded_size(&msg).unwrap(), 32);
    }

    #[test]
    fn parse_float_vax_byte_order() {
        // VAX byte order requires BOTH bit 0 and bit 6 set: class_bits = 0x41
        // (1,1) = VAX — not (1,0) which was the old (wrong) mapping.
        let mut props = vec![0u8; 12];
        props[2..4].copy_from_slice(&32u16.to_le_bytes());
        props[4] = 23;
        props[5] = 8;
        props[6] = 0;
        props[7] = 23;
        props[8..12].copy_from_slice(&127u32.to_le_bytes());
        let msg = build_dt_msg(1, 3, 0x41, 4, &props); // 0x41 = bit0 + bit6
        let dt = Datatype::parse(&msg).unwrap();
        match dt {
            Datatype::FloatingPoint { byte_order, .. } => {
                assert_eq!(byte_order, ByteOrder::Vax);
            }
            other => panic!("expected FloatingPoint, got {:?}", other),
        }
    }

    #[test]
    fn parse_float_invalid_byte_order_hi1_lo0() {
        // (byte_order_hi=1, byte_order_lo=0) = 0x40 → must be an error
        let mut props = vec![0u8; 12];
        props[2..4].copy_from_slice(&32u16.to_le_bytes());
        props[4] = 23;
        props[5] = 8;
        props[6] = 0;
        props[7] = 23;
        props[8..12].copy_from_slice(&127u32.to_le_bytes());
        let msg = build_dt_msg(1, 3, 0x40, 4, &props); // bit6 only, no bit0
        assert!(Datatype::parse(&msg).is_err());
    }

    #[test]
    fn encoded_size_array() {
        // Array v3: 8 + 1(ndims) + 4(dim) + 12(i32 type) = 25
        let mut arr_props = Vec::new();
        arr_props.push(1);
        arr_props.extend_from_slice(&3u32.to_le_bytes());
        arr_props.extend_from_slice(&build_dt_msg(0, 3, 0x08, 4, &[0, 0, 32, 0]));
        let msg = build_dt_msg(10, 3, 0, 12, &arr_props);
        assert_eq!(Datatype::encoded_size(&msg).unwrap(), 25);
    }
}
