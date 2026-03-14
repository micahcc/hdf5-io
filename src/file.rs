use crate::btree2::{self, BTree2Header};
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, Result};
use crate::filters::FilterPipeline;
use crate::fractal_heap::{self, FractalHeapHeader};
use crate::io::ReadAt;
use crate::layout::DataLayout;
use crate::link::{Link, LinkTarget};
use crate::object_header::messages::MessageType;
use crate::object_header::ObjectHeader;
use crate::superblock::Superblock;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 files. It holds a reference
/// to the underlying reader and the parsed superblock.
pub struct File<R: ReadAt + ?Sized> {
    reader: Box<R>,
    superblock: Superblock,
}

impl File<[u8]> {
    /// Open an HDF5 file from an in-memory byte buffer.
    pub fn from_bytes(data: Box<[u8]>) -> Result<File<[u8]>> {
        let superblock = Superblock::parse(&*data, 0)?;
        Ok(File {
            reader: data,
            superblock,
        })
    }
}

impl File<std::fs::File> {
    /// Open an HDF5 file from a filesystem path.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<File<std::fs::File>> {
        let f = std::fs::File::open(path).map_err(Error::Io)?;
        let superblock = Superblock::parse(&f, 0)?;
        Ok(File {
            reader: Box::new(f),
            superblock,
        })
    }
}

impl<R: ReadAt + ?Sized> File<R> {
    /// The parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the root group.
    pub fn root_group(&self) -> Result<Group<'_, R>> {
        let addr = self.superblock.root_group_object_header_address;
        let header = ObjectHeader::parse(
            &*self.reader,
            addr,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )?;
        Ok(Group {
            file: self,
            address: addr,
            header,
        })
    }

    /// Open a path like `"/group1/subgroup/dataset"`.
    ///
    /// Returns a `Node` which can be either a `Group` or a `Dataset`.
    pub fn open_path(&self, path: &str) -> Result<Node<'_, R>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let root = self.root_group()?;

        if parts.is_empty() {
            return Ok(Node::Group(root));
        }

        let mut current_group = root;

        for (i, part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;
            let link = current_group.find_link(part)?;

            match link.target {
                LinkTarget::Hard { address } => {
                    let header = ObjectHeader::parse(
                        &*self.reader,
                        address,
                        self.superblock.size_of_offsets,
                        self.superblock.size_of_lengths,
                    )?;

                    if is_last {
                        // Determine if this is a group or dataset
                        let has_layout = header
                            .messages
                            .iter()
                            .any(|m| m.msg_type == MessageType::DataLayout);
                        if has_layout {
                            return Ok(Node::Dataset(Dataset {
                                file: self,
                                address,
                                header,
                            }));
                        } else {
                            return Ok(Node::Group(Group {
                                file: self,
                                address,
                                header,
                            }));
                        }
                    } else {
                        current_group = Group {
                            file: self,
                            address,
                            header,
                        };
                    }
                }
                LinkTarget::Soft { ref path } => {
                    // Resolve soft link by re-opening from root
                    let resolved = self.open_path(path)?;
                    if is_last {
                        return Ok(resolved);
                    }
                    match resolved {
                        Node::Group(g) => current_group = g,
                        Node::Dataset(_) => {
                            return Err(Error::NotAGroup {
                                path: part.to_string(),
                            })
                        }
                    }
                }
                LinkTarget::External { .. } => {
                    return Err(Error::Other {
                        msg: "external links not supported".into(),
                    });
                }
            }
        }

        Ok(Node::Group(current_group))
    }

    fn size_of_offsets(&self) -> u8 {
        self.superblock.size_of_offsets
    }

    fn size_of_lengths(&self) -> u8 {
        self.superblock.size_of_lengths
    }
}

/// A node in the HDF5 hierarchy — either a group or a dataset.
pub enum Node<'a, R: ReadAt + ?Sized> {
    Group(Group<'a, R>),
    Dataset(Dataset<'a, R>),
}

/// A group (directory-like container) in the HDF5 file.
pub struct Group<'a, R: ReadAt + ?Sized> {
    file: &'a File<R>,
    address: u64,
    header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Group<'a, R> {
    /// List all child link names.
    pub fn members(&self) -> Result<Vec<String>> {
        let links = self.read_links()?;
        Ok(links.into_iter().map(|l| l.name).collect())
    }

    /// Get a specific child by name, returning the link.
    pub fn find_link(&self, name: &str) -> Result<Link> {
        let links = self.read_links()?;
        links
            .into_iter()
            .find(|l| l.name == name)
            .ok_or_else(|| Error::PathNotFound {
                path: name.to_string(),
            })
    }

    /// Open a child group by name.
    pub fn group(&self, name: &str) -> Result<Group<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Group {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotAGroup {
                path: name.to_string(),
            }),
        }
    }

    /// Open a child dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset<'a, R>> {
        let link = self.find_link(name)?;
        match link.target {
            LinkTarget::Hard { address } => {
                let header = ObjectHeader::parse(
                    &*self.file.reader,
                    address,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                )?;
                Ok(Dataset {
                    file: self.file,
                    address,
                    header,
                })
            }
            _ => Err(Error::NotADataset {
                path: name.to_string(),
            }),
        }
    }

    /// Read all attributes on this group.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read all links from this group's object header.
    ///
    /// Links can be stored in two ways:
    /// 1. Directly as Link messages (0x0006) in the object header (compact storage).
    /// 2. In a fractal heap + B-tree v2, referenced by a Link Info message (0x0002).
    fn read_links(&self) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();

        // First check for direct Link messages
        let mut links: Vec<Link> = Vec::new();
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Link {
                links.push(Link::parse(&msg.data, so)?);
            }
        }
        if !links.is_empty() {
            return Ok(links);
        }

        // Otherwise, look for Link Info message → fractal heap + B-tree v2
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::LinkInfo {
                return self.read_links_from_link_info(&msg.data);
            }
        }

        Ok(links) // empty — no links
    }

    /// Parse a Link Info message and read links from fractal heap.
    fn read_links_from_link_info(&self, data: &[u8]) -> Result<Vec<Link>> {
        let so = self.file.size_of_offsets();
        let sl = self.file.size_of_lengths();
        let o = so as usize;

        if data.len() < 2 {
            return Err(Error::InvalidObjectHeader {
                msg: "link info message too short".into(),
            });
        }

        let _version = data[0];
        let flags = data[1];
        let mut pos = 2;

        // Optional: max creation order (8 bytes, if flags bit 0 set)
        if (flags & 0x01) != 0 {
            pos += 8;
        }

        // Fractal heap address
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at fractal heap address".into(),
            });
        }
        let fheap_addr = read_offset_from_slice(data, pos, so);
        pos += o;

        // B-tree v2 address (name index)
        if pos + o > data.len() {
            return Err(Error::InvalidObjectHeader {
                msg: "link info: truncated at B-tree address".into(),
            });
        }
        let bt2_addr = read_offset_from_slice(data, pos, so);

        if fheap_addr == u64::MAX || bt2_addr == u64::MAX {
            return Ok(Vec::new()); // no links stored yet
        }

        // Parse the fractal heap header
        let fheap = FractalHeapHeader::parse(&*self.file.reader, fheap_addr, so, sl)?;

        // Parse the B-tree v2 header
        let bt2 = BTree2Header::parse(&*self.file.reader, bt2_addr, so, sl)?;

        // Iterate B-tree records → each contains a heap ID → look up in fractal heap
        let mut links = Vec::new();
        let heap_id_len = fheap.heap_id_length as usize;

        btree2::iterate_records(&*self.file.reader, &bt2, so, |record| {
            // Type 5 record: hash (4 bytes) + heap_id (heap_id_len bytes)
            if let Some((_hash, heap_id)) =
                btree2::parse_link_name_record(&record.data, heap_id_len)
            {
                // Read the link message from the fractal heap
                let link_data = fractal_heap::read_managed_object(
                    &*self.file.reader,
                    &fheap,
                    &heap_id,
                    so,
                    sl,
                )?;
                let link = Link::parse(&link_data, so)?;
                links.push(link);
            }
            Ok(())
        })?;

        Ok(links)
    }
}

/// A dataset in the HDF5 file.
pub struct Dataset<'a, R: ReadAt + ?Sized> {
    file: &'a File<R>,
    address: u64,
    header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Dataset<'a, R> {
    /// The dataset's datatype.
    pub fn datatype(&self) -> Result<Datatype> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Datatype {
                return Datatype::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no datatype message".into(),
        })
    }

    /// The dataset's dataspace (shape information).
    pub fn dataspace(&self) -> Result<Dataspace> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Dataspace {
                return Dataspace::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no dataspace message".into(),
        })
    }

    /// The dataset's shape (convenience wrapper).
    pub fn shape(&self) -> Result<Vec<u64>> {
        Ok(self.dataspace()?.shape().to_vec())
    }

    /// The data layout (contiguous, chunked, compact).
    pub fn layout(&self) -> Result<DataLayout> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::DataLayout {
                return DataLayout::parse(
                    &msg.data,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                );
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no layout message".into(),
        })
    }

    /// The filter pipeline, if any.
    pub fn filters(&self) -> Result<Option<FilterPipeline>> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FilterPipeline {
                return Ok(Some(FilterPipeline::parse(&msg.data)?));
            }
        }
        Ok(None)
    }

    /// Read all attributes on this dataset.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read the entire dataset as raw bytes.
    ///
    /// Returns the uncompressed, un-filtered data. The caller is responsible
    /// for interpreting the bytes according to `datatype()`.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        let layout = self.layout()?;
        let filters = self.filters()?;

        match layout {
            DataLayout::Compact { data } => {
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Contiguous { address, size } => {
                if address == u64::MAX {
                    // No data allocated — return fill value or zeros
                    let dtype = self.datatype()?;
                    let dspace = self.dataspace()?;
                    let total_size = dspace.num_elements() * dtype.element_size() as u64;
                    return Ok(vec![0u8; total_size as usize]);
                }
                let mut data = vec![0u8; size as usize];
                self.file
                    .reader
                    .read_exact_at(address, &mut data)
                    .map_err(Error::Io)?;
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Chunked { .. } => {
                // TODO: implement chunked reading
                //  1. Determine chunk index type
                //  2. Look up all chunks via B-tree v1/v2 or other index
                //  3. Read each chunk, apply filter pipeline
                //  4. Assemble into contiguous output buffer
                Err(Error::Other {
                    msg: "chunked dataset reading not yet implemented".into(),
                })
            }
            DataLayout::Virtual { .. } => Err(Error::Other {
                msg: "virtual dataset reading not supported".into(),
            }),
        }
    }
}

/// An attribute (name + value) on a group or dataset.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub datatype: Datatype,
    pub dataspace: Dataspace,
    pub raw_value: Vec<u8>,
}

/// Parse attribute messages from an object header.
fn parse_attributes<R: ReadAt + ?Sized>(
    header: &ObjectHeader,
    _file: &File<R>,
) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();

    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            if let Ok(attr) = parse_attribute_message(&msg.data) {
                attrs.push(attr);
            }
        }
    }

    // TODO: also handle AttributeInfo message → fractal heap for dense attribute storage

    Ok(attrs)
}

/// Parse an attribute message body.
///
/// Attribute message layout (version 3):
/// ```text
/// Byte 0:    Version (1, 2, or 3)
/// Byte 1:    Flags (bit 0: datatype shared, bit 1: dataspace shared)
/// Byte 2-3:  Name size (u16)
/// Byte 4-5:  Datatype size (u16)
/// Byte 6-7:  Dataspace size (u16)
/// Byte 8:    Character set (version 3 only: 0=ASCII, 1=UTF-8)
/// Name (null-terminated, NOT padded in version 3)
/// Datatype message
/// Dataspace message
/// Value
/// ```
fn parse_attribute_message(data: &[u8]) -> Result<Attribute> {
    if data.len() < 6 {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute message too short".into(),
        });
    }

    let version = data[0];
    let _flags = data[1];
    let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let dt_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let ds_size = u16::from_le_bytes([data[6], data[7]]) as usize;

    let mut pos = match version {
        1 | 2 => 8,
        3 => 9, // extra charset byte
        _ => {
            return Err(Error::InvalidObjectHeader {
                msg: format!("unsupported attribute message version {}", version),
            })
        }
    };

    // Name
    if pos + name_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute name truncated".into(),
        });
    }
    let name = String::from_utf8_lossy(&data[pos..pos + name_size])
        .trim_end_matches('\0')
        .to_string();
    pos += name_size;

    // Version 1 pads name, datatype, dataspace to 8-byte boundaries
    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Datatype
    if pos + dt_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute datatype truncated".into(),
        });
    }
    let datatype = Datatype::parse(&data[pos..pos + dt_size])?;
    pos += dt_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Dataspace
    if pos + ds_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute dataspace truncated".into(),
        });
    }
    let dataspace = Dataspace::parse(&data[pos..pos + ds_size])?;
    pos += ds_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Value (remaining bytes)
    let raw_value = data[pos..].to_vec();

    Ok(Attribute {
        name,
        datatype,
        dataspace,
        raw_value,
    })
}

fn read_offset_from_slice(data: &[u8], offset: usize, size: u8) -> u64 {
    match size {
        4 => u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64,
        8 => u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]),
        _ => 0,
    }
}
