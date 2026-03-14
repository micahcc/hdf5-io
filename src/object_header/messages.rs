/// Object header message type IDs.
///
/// Reference: H5Opkg.h, the `H5O_msg_class_t` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// 0x0000 - NIL (padding/deleted)
    Nil,
    /// 0x0001 - Dataspace
    Dataspace,
    /// 0x0002 - Link Info (new-style group)
    LinkInfo,
    /// 0x0003 - Datatype
    Datatype,
    /// 0x0004 - Fill Value (old, deprecated)
    FillValueOld,
    /// 0x0005 - Fill Value
    FillValue,
    /// 0x0006 - Link
    Link,
    /// 0x0007 - External Data Files
    ExternalDataFiles,
    /// 0x0008 - Data Layout
    DataLayout,
    /// 0x0009 - Bogus (testing only)
    Bogus,
    /// 0x000A - Group Info
    GroupInfo,
    /// 0x000B - Filter Pipeline
    FilterPipeline,
    /// 0x000C - Attribute
    Attribute,
    /// 0x000D - Object Comment
    ObjectComment,
    /// 0x000E - Object Modification Time (old)
    ObjectModTimeOld,
    /// 0x000F - Shared Message Table
    SharedMessageTable,
    /// 0x0010 - Object Header Continuation
    ObjectHeaderContinuation,
    /// 0x0011 - Symbol Table (v1 groups, shouldn't appear in v2 superblock files)
    SymbolTable,
    /// 0x0012 - Object Modification Time
    ObjectModTime,
    /// 0x0013 - B-tree 'K' values
    BTreeKValues,
    /// 0x0014 - Driver Info
    DriverInfo,
    /// 0x0015 - Attribute Info
    AttributeInfo,
    /// 0x0016 - Object Reference Count
    ObjectReferenceCount,
    /// 0x0017 - File Space Info (HDF5 1.10+)
    FileSpaceInfo,
    /// Unknown message type.
    Unknown(u8),
}

impl MessageType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0x00 => Self::Nil,
            0x01 => Self::Dataspace,
            0x02 => Self::LinkInfo,
            0x03 => Self::Datatype,
            0x04 => Self::FillValueOld,
            0x05 => Self::FillValue,
            0x06 => Self::Link,
            0x07 => Self::ExternalDataFiles,
            0x08 => Self::DataLayout,
            0x09 => Self::Bogus,
            0x0A => Self::GroupInfo,
            0x0B => Self::FilterPipeline,
            0x0C => Self::Attribute,
            0x0D => Self::ObjectComment,
            0x0E => Self::ObjectModTimeOld,
            0x0F => Self::SharedMessageTable,
            0x10 => Self::ObjectHeaderContinuation,
            0x11 => Self::SymbolTable,
            0x12 => Self::ObjectModTime,
            0x13 => Self::BTreeKValues,
            0x14 => Self::DriverInfo,
            0x15 => Self::AttributeInfo,
            0x16 => Self::ObjectReferenceCount,
            0x17 => Self::FileSpaceInfo,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Nil => 0x00,
            Self::Dataspace => 0x01,
            Self::LinkInfo => 0x02,
            Self::Datatype => 0x03,
            Self::FillValueOld => 0x04,
            Self::FillValue => 0x05,
            Self::Link => 0x06,
            Self::ExternalDataFiles => 0x07,
            Self::DataLayout => 0x08,
            Self::Bogus => 0x09,
            Self::GroupInfo => 0x0A,
            Self::FilterPipeline => 0x0B,
            Self::Attribute => 0x0C,
            Self::ObjectComment => 0x0D,
            Self::ObjectModTimeOld => 0x0E,
            Self::SharedMessageTable => 0x0F,
            Self::ObjectHeaderContinuation => 0x10,
            Self::SymbolTable => 0x11,
            Self::ObjectModTime => 0x12,
            Self::BTreeKValues => 0x13,
            Self::DriverInfo => 0x14,
            Self::AttributeInfo => 0x15,
            Self::ObjectReferenceCount => 0x16,
            Self::FileSpaceInfo => 0x17,
            Self::Unknown(v) => *v,
        }
    }
}

/// A decoded object header message (raw — the data is not yet interpreted).
///
/// After collecting all messages from the object header, callers use the
/// `msg_type` to dispatch to type-specific parsers (Datatype::parse,
/// Dataspace::parse, DataLayout::parse, etc.).
#[derive(Debug, Clone)]
pub struct Message {
    pub msg_type: MessageType,
    /// Message flags:
    /// - Bit 0: constant (message data is constant for the life of the object header)
    /// - Bit 1: shared (message is stored in the shared message heap)
    /// - Bit 2: message should not be shared
    /// - Bit 3: fail if unknown and file is opened for write
    /// - Bit 4: set bit 5 if unknown
    /// - Bit 5: was unknown and modified (set by library)
    /// - Bit 6: shareable
    /// - Bit 7: fail if unknown always
    pub flags: u8,
    /// Creation order index (present if object header tracks attribute creation order).
    pub creation_order: Option<u16>,
    /// Raw message body bytes.
    pub data: Vec<u8>,
}

impl Message {
    /// Returns true if this message is marked as shared.
    pub fn is_shared(&self) -> bool {
        (self.flags & 0x02) != 0
    }
}
