use std::env;
use std::process;

use hdf5_pure::Attribute;
use hdf5_pure::DataLayout;
use hdf5_pure::Dataset;
use hdf5_pure::Dataspace;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::Group;
use hdf5_pure::Node;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: h5info <file.h5> [--read-dataset <path>]");
        process::exit(1);
    }

    let path = &args[1];
    let read_dataset = if args.len() >= 4 && args[2] == "--read-dataset" {
        Some(args[3].as_str())
    } else {
        None
    };
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening {}: {}", path, e);
            process::exit(1);
        }
    };

    let sb = file.superblock();
    println!("HDF5 \"{}\" {{", path);
    println!("SUPERBLOCK {{");
    println!("  version: {}", sb.version);
    println!("  size_of_offsets: {}", sb.size_of_offsets);
    println!("  size_of_lengths: {}", sb.size_of_lengths);
    println!(
        "  root_group_addr: {:#x}",
        sb.root_group_object_header_address
    );
    println!("}}");

    if let Some(ds_path) = read_dataset {
        match file.open_path(ds_path) {
            Ok(Node::Dataset(ds)) => {
                print_dataset_data(&ds, ds_path);
            }
            Ok(Node::Group(_)) => eprintln!("{} is a group, not a dataset", ds_path),
            Err(e) => eprintln!("Error opening {}: {}", ds_path, e),
        }
        return;
    }

    match file.root_group() {
        Ok(root) => print_group(&file, &root, "/", 0),
        Err(e) => eprintln!("Error reading root group: {}", e),
    }

    println!("}}");
}

fn indent(level: usize) -> String {
    "  ".repeat(level)
}

fn print_group<R: hdf5_pure::ReadAt + ?Sized>(
    file: &File<R>,
    group: &Group<'_, R>,
    name: &str,
    level: usize,
) {
    let pre = indent(level);
    println!("{}GROUP \"{}\" {{", pre, name);

    // Attributes
    match group.attributes() {
        Ok(attrs) => {
            for attr in &attrs {
                print_attribute(attr, level + 1);
            }
        }
        Err(e) => println!("{}  (error reading attributes: {})", pre, e),
    }

    // Members
    match group.members() {
        Ok(members) => {
            for member_name in &members {
                match file.open_path(&format_path(name, member_name)) {
                    Ok(Node::Group(g)) => print_group(file, &g, member_name, level + 1),
                    Ok(Node::Dataset(ds)) => print_dataset(&ds, member_name, level + 1),
                    Err(e) => println!("{}  {} (error: {})", pre, member_name, e),
                }
            }
        }
        Err(e) => println!("{}  (error reading members: {})", pre, e),
    }

    println!("{}}}", pre);
}

fn format_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", parent, child)
    }
}

fn print_dataset<R: hdf5_pure::ReadAt + ?Sized>(ds: &Dataset<'_, R>, name: &str, level: usize) {
    let pre = indent(level);
    println!("{}DATASET \"{}\" {{", pre, name);

    // Datatype
    match ds.datatype() {
        Ok(dt) => println!("{}  DATATYPE {}", pre, format_datatype(&dt)),
        Err(e) => println!("{}  DATATYPE (error: {})", pre, e),
    }

    // Dataspace
    match ds.dataspace() {
        Ok(dspace) => println!("{}  DATASPACE {}", pre, format_dataspace(&dspace)),
        Err(e) => println!("{}  DATASPACE (error: {})", pre, e),
    }

    // Layout
    match ds.layout() {
        Ok(layout) => println!("{}  LAYOUT {}", pre, format_layout(&layout)),
        Err(e) => println!("{}  LAYOUT (error: {})", pre, e),
    }

    // Filters
    match ds.filters() {
        Ok(Some(pipeline)) => {
            println!("{}  FILTERS {{", pre);
            for f in &pipeline.filters {
                let name = f.name.as_deref().unwrap_or("unnamed");
                println!("{}    {} (id={})", pre, name, f.id);
            }
            println!("{}  }}", pre);
        }
        Ok(None) => {}
        Err(e) => println!("{}  FILTERS (error: {})", pre, e),
    }

    // Attributes
    match ds.attributes() {
        Ok(attrs) => {
            for attr in &attrs {
                print_attribute(attr, level + 1);
            }
        }
        Err(e) => println!("{}  (error reading attributes: {})", pre, e),
    }

    println!("{}}}", pre);
}

fn print_attribute(attr: &Attribute, level: usize) {
    let pre = indent(level);
    println!("{}ATTRIBUTE \"{}\" {{", pre, attr.name);
    println!("{}  DATATYPE {}", pre, format_datatype(&attr.datatype));
    println!("{}  DATASPACE {}", pre, format_dataspace(&attr.dataspace));
    println!("{}  VALUE {}", pre, format_attr_value(attr));
    println!("{}}}", pre);
}

fn format_attr_value(attr: &Attribute) -> String {
    let raw = &attr.raw_value;
    if raw.is_empty() {
        return "(empty)".to_string();
    }

    match &attr.datatype {
        Datatype::String { size, .. } => {
            let s = std::str::from_utf8(&raw[..(*size as usize).min(raw.len())])
                .unwrap_or("<non-utf8>")
                .trim_end_matches('\0');
            format!("\"{}\"", s)
        }
        Datatype::FixedPoint { size, signed, .. } => match (*size, *signed) {
            (1, false) => format!("{}", raw[0]),
            (1, true) => format!("{}", raw[0] as i8),
            (2, false) => format!(
                "{}",
                u16::from_le_bytes(raw[..2].try_into().unwrap_or([0; 2]))
            ),
            (2, true) => format!(
                "{}",
                i16::from_le_bytes(raw[..2].try_into().unwrap_or([0; 2]))
            ),
            (4, false) => format!(
                "{}",
                u32::from_le_bytes(raw[..4].try_into().unwrap_or([0; 4]))
            ),
            (4, true) => format!(
                "{}",
                i32::from_le_bytes(raw[..4].try_into().unwrap_or([0; 4]))
            ),
            (8, false) => format!(
                "{}",
                u64::from_le_bytes(raw[..8].try_into().unwrap_or([0; 8]))
            ),
            (8, true) => format!(
                "{}",
                i64::from_le_bytes(raw[..8].try_into().unwrap_or([0; 8]))
            ),
            _ => format!("{:?}", &raw[..(*size as usize).min(raw.len())]),
        },
        Datatype::FloatingPoint { size, .. } => match *size {
            4 => format!(
                "{}",
                f32::from_le_bytes(raw[..4].try_into().unwrap_or([0; 4]))
            ),
            8 => format!(
                "{}",
                f64::from_le_bytes(raw[..8].try_into().unwrap_or([0; 8]))
            ),
            _ => format!("{:?}", &raw[..(*size as usize).min(raw.len())]),
        },
        _ => {
            if raw.len() <= 32 {
                format!("{:02x?}", raw)
            } else {
                format!("{:02x?}... ({} bytes)", &raw[..32], raw.len())
            }
        }
    }
}

fn format_datatype(dt: &Datatype) -> String {
    match dt {
        Datatype::FixedPoint {
            size,
            signed,
            byte_order,
            ..
        } => {
            let sign = if *signed { "I" } else { "U" };
            let order = match byte_order {
                hdf5_pure::datatype::ByteOrder::LittleEndian => "LE",
                hdf5_pure::datatype::ByteOrder::BigEndian => "BE",
                hdf5_pure::datatype::ByteOrder::Vax => "VAX",
            };
            format!("H5T_STD_{}{}{}", sign, size * 8, order)
        }
        Datatype::FloatingPoint {
            size, byte_order, ..
        } => {
            let order = match byte_order {
                hdf5_pure::datatype::ByteOrder::LittleEndian => "LE",
                hdf5_pure::datatype::ByteOrder::BigEndian => "BE",
                hdf5_pure::datatype::ByteOrder::Vax => "VAX",
            };
            format!("H5T_IEEE_F{}{}", size * 8, order)
        }
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            let pad = match padding {
                hdf5_pure::datatype::StringPadding::NullTerminate => "NULLTERM",
                hdf5_pure::datatype::StringPadding::NullPad => "NULLPAD",
                hdf5_pure::datatype::StringPadding::SpacePad => "SPACEPAD",
            };
            let cs = match char_set {
                hdf5_pure::datatype::CharacterSet::Ascii => "ASCII",
                hdf5_pure::datatype::CharacterSet::Utf8 => "UTF8",
            };
            format!("H5T_STRING {{ size={}; pad={}; cset={} }}", size, pad, cs)
        }
        Datatype::Compound { size, members } => {
            let mut s = format!("H5T_COMPOUND {{ size={};", size);
            for m in members {
                s.push_str(&format!(
                    " {} +{} {};",
                    format_datatype(&m.datatype),
                    m.byte_offset,
                    m.name
                ));
            }
            s.push_str(" }");
            s
        }
        Datatype::Enum { base, members } => {
            let mut s = format!("H5T_ENUM {{ base={}; ", format_datatype(base));
            for (i, m) in members.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}={:?}", m.name, m.value));
            }
            s.push_str(" }");
            s
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            format!(
                "H5T_ARRAY {{ dims={:?}; base={} }}",
                dimensions,
                format_datatype(element_type)
            )
        }
        Datatype::VarLen {
            element_type,
            is_string,
            ..
        } => {
            if *is_string {
                "H5T_VLEN_STRING".to_string()
            } else {
                format!("H5T_VLEN {{ base={} }}", format_datatype(element_type))
            }
        }
        Datatype::Opaque { size, tag } => {
            format!("H5T_OPAQUE {{ size={}; tag=\"{}\" }}", size, tag)
        }
        Datatype::BitField { size, .. } => format!("H5T_BITFIELD {{ size={} }}", size),
        Datatype::Reference { ref_type } => match ref_type {
            hdf5_pure::datatype::ReferenceType::Object => "H5T_REFERENCE(OBJECT)".to_string(),
            hdf5_pure::datatype::ReferenceType::DatasetRegion => {
                "H5T_REFERENCE(REGION)".to_string()
            }
        },
        Datatype::Time { size, .. } => format!("H5T_TIME {{ size={} }}", size),
        Datatype::Complex { base, .. } => {
            format!("H5T_COMPLEX {{ base={} }}", format_datatype(base))
        }
    }
}

fn format_dataspace(ds: &Dataspace) -> String {
    match ds {
        Dataspace::Scalar => "SCALAR".to_string(),
        Dataspace::Null => "NULL".to_string(),
        Dataspace::Simple {
            dimensions,
            max_dimensions,
        } => {
            let dims: Vec<String> = dimensions.iter().map(|d| d.to_string()).collect();
            let s = format!("SIMPLE {{ {} }}", dims.join(" x "));
            if let Some(maxd) = max_dimensions {
                let maxs: Vec<String> = maxd
                    .iter()
                    .map(|d| {
                        if *d == u64::MAX {
                            "UNLIMITED".to_string()
                        } else {
                            d.to_string()
                        }
                    })
                    .collect();
                format!("{} / {{ {} }}", s, maxs.join(" x "))
            } else {
                s
            }
        }
    }
}

fn format_layout(layout: &DataLayout) -> String {
    match layout {
        DataLayout::Compact { data } => format!("COMPACT {{ size={} }}", data.len()),
        DataLayout::Contiguous { address, size } => {
            format!("CONTIGUOUS {{ addr={:#x}; size={} }}", address, size)
        }
        DataLayout::Chunked {
            chunk_dims,
            address,
            chunk_index_type,
            ..
        } => {
            let dims: Vec<String> = chunk_dims.iter().map(|d| d.to_string()).collect();
            let idx = chunk_index_type
                .as_ref()
                .map(|t| format!("{:?}", t))
                .unwrap_or_else(|| "BTreeV1".to_string());
            format!(
                "CHUNKED {{ dims={}; index={}; addr={:#x} }}",
                dims.join("x"),
                idx,
                address
            )
        }
        DataLayout::Virtual { .. } => "VIRTUAL".to_string(),
    }
}

fn print_dataset_data<R: hdf5_pure::ReadAt + ?Sized>(ds: &Dataset<'_, R>, path: &str) {
    println!("DATASET \"{}\"", path);

    let dt = match ds.datatype() {
        Ok(dt) => {
            println!("  DATATYPE {}", format_datatype(&dt));
            dt
        }
        Err(e) => {
            println!("  DATATYPE (error: {})", e);
            return;
        }
    };

    let dspace = match ds.dataspace() {
        Ok(ds) => {
            println!("  DATASPACE {}", format_dataspace(&ds));
            ds
        }
        Err(e) => {
            println!("  DATASPACE (error: {})", e);
            return;
        }
    };

    let num_elements = dspace.num_elements();
    let elem_size = dt.element_size() as usize;
    println!("  elements={}, element_size={}", num_elements, elem_size);

    // Read first few elements
    let count = num_elements.min(5);
    let raw = match ds.read_slice(&[0], &[count]) {
        Ok(data) => data,
        Err(e) => {
            // Fall back to read_raw if slice fails (e.g. scalar)
            println!("  (read_slice failed: {}, trying read_raw)", e);
            match ds.read_raw() {
                Ok(data) => data,
                Err(e2) => {
                    println!("  DATA (error: {})", e2);
                    return;
                }
            }
        }
    };

    println!("  DATA (first {} elements, {} bytes):", count, raw.len());
    for i in 0..count as usize {
        let start = i * elem_size;
        let end = (start + elem_size).min(raw.len());
        if start >= raw.len() {
            break;
        }
        let elem = &raw[start..end];
        print!("    [{}]: ", i);
        print_element(&dt, elem);
        println!();
    }
}

fn print_element(dt: &Datatype, data: &[u8]) {
    match dt {
        Datatype::FixedPoint { size, signed, .. } => match (*size, *signed) {
            (1, false) => print!("{}", data[0]),
            (1, true) => print!("{}", data[0] as i8),
            (2, false) if data.len() >= 2 => {
                print!("{}", u16::from_le_bytes([data[0], data[1]]))
            }
            (2, true) if data.len() >= 2 => {
                print!("{}", i16::from_le_bytes([data[0], data[1]]))
            }
            (4, false) if data.len() >= 4 => {
                print!("{}", u32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            (4, true) if data.len() >= 4 => {
                print!("{}", i32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            (8, false) if data.len() >= 8 => {
                print!("{}", u64::from_le_bytes(data[..8].try_into().unwrap()))
            }
            (8, true) if data.len() >= 8 => {
                print!("{}", i64::from_le_bytes(data[..8].try_into().unwrap()))
            }
            _ => print!("{:02x?}", data),
        },
        Datatype::FloatingPoint { size, .. } => match *size {
            4 if data.len() >= 4 => {
                print!("{}", f32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            8 if data.len() >= 8 => {
                print!("{}", f64::from_le_bytes(data[..8].try_into().unwrap()))
            }
            _ => print!("{:02x?}", data),
        },
        Datatype::String { size, .. } => {
            let s = std::str::from_utf8(&data[..(*size as usize).min(data.len())])
                .unwrap_or("<non-utf8>")
                .trim_end_matches('\0');
            print!("\"{}\"", s);
        }
        Datatype::Compound { members, .. } => {
            print!("{{ ");
            for (i, m) in members.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                let off = m.byte_offset as usize;
                let msz = m.datatype.element_size() as usize;
                if off + msz <= data.len() {
                    print!("{}: ", m.name);
                    print_element(&m.datatype, &data[off..off + msz]);
                }
            }
            print!(" }}");
        }
        _ => print!("{:02x?}", &data[..data.len().min(32)]),
    }
}
