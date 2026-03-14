use crate::error::{Error, Result};

/// A decoded link message (type 0x0006).
///
/// ## On-disk layout (link message, version 1)
///
/// ```text
/// Byte 0:    Version (1)
/// Byte 1:    Flags
///            Bit 0-1: size of "size of link name" field (0=1, 1=2, 2=4, 3=8 bytes)
///            Bit 2:   creation order present
///            Bit 3:   link type present
///            Bit 4:   character set present
/// [if flags bit 3]: Link type (1 byte: 0=hard, 1=soft, 64=external)
/// [if flags bit 2]: Creation order (8 bytes)
/// [if flags bit 4]: Character set (1 byte: 0=ASCII, 1=UTF-8)
/// Size of link name (1/2/4/8 bytes per flags bits 0-1)
/// Link name (N bytes)
/// Link target (depends on link type):
///   Hard link:     object header address (size_of_offsets bytes)
///   Soft link:     2-byte length + string
///   External link: 2-byte length + filename + object name (null-separated)
/// ```
#[derive(Debug, Clone)]
pub struct Link {
    pub name: String,
    pub creation_order: Option<u64>,
    pub char_set: CharSet,
    pub target: LinkTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharSet {
    Ascii,
    Utf8,
}

#[derive(Debug, Clone)]
pub enum LinkTarget {
    /// Hard link: address of the target object header.
    Hard { address: u64 },
    /// Soft link: path string.
    Soft { path: String },
    /// External link: file name and object path.
    External { filename: String, object_path: String },
}

impl Link {
    /// Parse a link message from raw bytes.
    pub fn parse(data: &[u8], size_of_offsets: u8) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidObjectHeader {
                msg: "link message too short".into(),
            });
        }

        let version = data[0];
        if version != 1 {
            return Err(Error::InvalidObjectHeader {
                msg: format!("unsupported link message version {}", version),
            });
        }

        let flags = data[1];
        let name_size_enc = flags & 0x03;
        let has_creation_order = (flags & 0x04) != 0;
        let has_link_type = (flags & 0x08) != 0;
        let has_charset = (flags & 0x10) != 0;

        let mut pos = 2;

        // Link type
        let link_type = if has_link_type {
            let lt = data[pos];
            pos += 1;
            lt
        } else {
            0 // default: hard link
        };

        // Creation order
        let creation_order = if has_creation_order {
            if pos + 8 > data.len() {
                return Err(Error::InvalidObjectHeader {
                    msg: "link message truncated at creation order".into(),
                });
            }
            let co = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;
            Some(co)
        } else {
            None
        };

        // Character set
        let char_set = if has_charset {
            let cs = data[pos];
            pos += 1;
            match cs {
                0 => CharSet::Ascii,
                1 => CharSet::Utf8,
                _ => CharSet::Utf8, // default to UTF-8
            }
        } else {
            CharSet::Ascii
        };

        // Link name length
        let name_len = match name_size_enc {
            0 => {
                let v = data[pos] as usize;
                pos += 1;
                v
            }
            1 => {
                let v = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                v
            }
            2 => {
                let v = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
                pos += 4;
                v
            }
            3 => {
                let v = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                pos += 8;
                v
            }
            _ => unreachable!(),
        };

        // Link name
        if pos + name_len > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link message truncated at name".into(),
            });
        }
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Link target
        let target = match link_type {
            0 => {
                // Hard link: object header address
                let o = size_of_offsets as usize;
                if pos + o > data.len() {
                    return Err(Error::InvalidObjectHeader {
                        msg: "link message truncated at hard link address".into(),
                    });
                }
                let address = match size_of_offsets {
                    4 => u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as u64,
                    8 => u64::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                        data[pos + 4],
                        data[pos + 5],
                        data[pos + 6],
                        data[pos + 7],
                    ]),
                    _ => {
                        return Err(Error::InvalidObjectHeader {
                            msg: format!("unsupported size_of_offsets {}", size_of_offsets),
                        })
                    }
                };
                LinkTarget::Hard { address }
            }
            1 => {
                // Soft link: length (2 bytes) + path
                if pos + 2 > data.len() {
                    return Err(Error::InvalidObjectHeader {
                        msg: "link message truncated at soft link length".into(),
                    });
                }
                let path_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                if pos + path_len > data.len() {
                    return Err(Error::InvalidObjectHeader {
                        msg: "link message truncated at soft link path".into(),
                    });
                }
                let path = String::from_utf8_lossy(&data[pos..pos + path_len]).to_string();
                LinkTarget::Soft { path }
            }
            64 => {
                // External link: length (2 bytes) + filename\0objectname
                if pos + 2 > data.len() {
                    return Err(Error::InvalidObjectHeader {
                        msg: "link message truncated at external link length".into(),
                    });
                }
                let info_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                if pos + info_len > data.len() {
                    return Err(Error::InvalidObjectHeader {
                        msg: "link message truncated at external link info".into(),
                    });
                }
                // Skip version/flags byte
                let info = &data[pos + 1..pos + info_len];
                // Split on null byte
                let null_pos = info.iter().position(|&b| b == 0).unwrap_or(info.len());
                let filename = String::from_utf8_lossy(&info[..null_pos]).to_string();
                let object_path = if null_pos + 1 < info.len() {
                    String::from_utf8_lossy(&info[null_pos + 1..]).to_string()
                } else {
                    String::new()
                };
                LinkTarget::External {
                    filename,
                    object_path,
                }
            }
            _ => {
                return Err(Error::InvalidObjectHeader {
                    msg: format!("unknown link type {}", link_type),
                })
            }
        };

        Ok(Link {
            name,
            creation_order,
            char_set,
            target,
        })
    }
}
