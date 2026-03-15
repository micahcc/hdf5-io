use std::env;
use std::io::Write;
use std::io::{self};
use std::process;

use hdf5_pure::Attribute;
use hdf5_pure::DataLayout;
use hdf5_pure::Dataset;
use hdf5_pure::Dataspace;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::Group;
use hdf5_pure::Node;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: h5json <file.h5> [-d /path/to/dataset] [--header]");
        process::exit(1);
    }

    let path = &args[1];
    let mut dataset_path: Option<&str> = None;
    let mut header_only = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--dataset" => {
                i += 1;
                if i < args.len() {
                    dataset_path = Some(&args[i]);
                }
            }
            "-H" | "--header" => header_only = true,
            _ => {}
        }
        i += 1;
    }

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening {}: {}", path, e);
            process::exit(1);
        }
    };

    let result = if let Some(ds_path) = dataset_path {
        match file.open_path(ds_path) {
            Ok(Node::Dataset(ds)) => dataset_to_json(&ds, ds_path, header_only),
            Ok(Node::Group(g)) => group_to_json(&file, &g, ds_path, header_only),
            Err(e) => json!({"error": e.to_string()}),
        }
    } else {
        let sb = file.superblock();
        let mut root_obj = json!({
            "file": path,
            "superblock": {
                "version": sb.version,
                "size_of_offsets": sb.size_of_offsets,
                "size_of_lengths": sb.size_of_lengths,
            }
        });
        match file.root_group() {
            Ok(root) => {
                root_obj["root"] = group_to_json(&file, &root, "/", header_only);
            }
            Err(e) => {
                root_obj["error"] = json!(e.to_string());
            }
        }
        root_obj
    };

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    serde_json::to_writer_pretty(&mut out, &result).unwrap();
    out.write_all(b"\n").unwrap();
}

fn group_to_json<R: hdf5_pure::ReadAt + ?Sized>(
    file: &File<R>,
    group: &Group<'_, R>,
    path: &str,
    header_only: bool,
) -> Value {
    let mut obj = Map::new();
    obj.insert("type".into(), json!("group"));

    // Attributes
    match group.attributes() {
        Ok(attrs) if !attrs.is_empty() => {
            let attr_obj: Map<String, Value> = attrs
                .iter()
                .map(|a| (a.name.clone(), attr_to_json(a)))
                .collect();
            obj.insert("attributes".into(), Value::Object(attr_obj));
        }
        Err(e) => {
            obj.insert("attributes_error".into(), json!(e.to_string()));
        }
        _ => {}
    }

    // Members
    match group.members() {
        Ok(members) => {
            let mut children = Map::new();
            for name in &members {
                let child_path = format_path(path, name);
                match file.open_path(&child_path) {
                    Ok(Node::Group(g)) => {
                        children.insert(
                            name.clone(),
                            group_to_json(file, &g, &child_path, header_only),
                        );
                    }
                    Ok(Node::Dataset(ds)) => {
                        children
                            .insert(name.clone(), dataset_to_json(&ds, &child_path, header_only));
                    }
                    Err(e) => {
                        children.insert(name.clone(), json!({"error": e.to_string()}));
                    }
                }
            }
            if !children.is_empty() {
                obj.insert("members".into(), Value::Object(children));
            }
        }
        Err(e) => {
            obj.insert("members_error".into(), json!(e.to_string()));
        }
    }

    Value::Object(obj)
}

fn dataset_to_json<R: hdf5_pure::ReadAt + ?Sized>(
    ds: &Dataset<'_, R>,
    _path: &str,
    header_only: bool,
) -> Value {
    let mut obj = Map::new();
    obj.insert("type".into(), json!("dataset"));

    let dt = match ds.datatype() {
        Ok(dt) => {
            obj.insert("datatype".into(), datatype_to_json(&dt));
            Some(dt)
        }
        Err(e) => {
            obj.insert("datatype_error".into(), json!(e.to_string()));
            None
        }
    };

    let dspace = match ds.dataspace() {
        Ok(ds) => {
            obj.insert("dataspace".into(), dataspace_to_json(&ds));
            Some(ds)
        }
        Err(e) => {
            obj.insert("dataspace_error".into(), json!(e.to_string()));
            None
        }
    };

    match ds.layout() {
        Ok(layout) => obj.insert("layout".into(), layout_to_json(&layout)),
        Err(e) => obj.insert("layout_error".into(), json!(e.to_string())),
    };

    match ds.filters() {
        Ok(Some(pipeline)) => {
            let filters: Vec<Value> = pipeline
                .filters
                .iter()
                .map(|f| {
                    json!({
                        "id": f.id,
                        "name": f.name,
                    })
                })
                .collect();
            obj.insert("filters".into(), json!(filters));
        }
        Err(e) => {
            obj.insert("filters_error".into(), json!(e.to_string()));
        }
        _ => {}
    }

    // Attributes
    match ds.attributes() {
        Ok(attrs) if !attrs.is_empty() => {
            let attr_obj: Map<String, Value> = attrs
                .iter()
                .map(|a| (a.name.clone(), attr_to_json(a)))
                .collect();
            obj.insert("attributes".into(), Value::Object(attr_obj));
        }
        Err(e) => {
            obj.insert("attributes_error".into(), json!(e.to_string()));
        }
        _ => {}
    }

    // Data
    if !header_only {
        if let (Some(dt), Some(dspace)) = (&dt, &dspace) {
            let num_elements = dspace.num_elements();
            let elem_size = dt.element_size() as usize;
            if num_elements > 0 && elem_size > 0 {
                match ds.read_raw() {
                    Ok(raw) => {
                        let data = decode_data(dt, &raw, num_elements as usize, elem_size);
                        obj.insert("data".into(), data);
                    }
                    Err(e) => {
                        obj.insert("data_error".into(), json!(e.to_string()));
                    }
                }
            }
        }
    }

    Value::Object(obj)
}

fn decode_data(dt: &Datatype, raw: &[u8], num_elements: usize, elem_size: usize) -> Value {
    let mut values = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * elem_size;
        let end = (start + elem_size).min(raw.len());
        if start >= raw.len() {
            break;
        }
        values.push(element_to_json(dt, &raw[start..end]));
    }
    Value::Array(values)
}

fn element_to_json(dt: &Datatype, data: &[u8]) -> Value {
    match dt {
        Datatype::FixedPoint { size, signed, .. } => match (*size, *signed) {
            (1, false) => json!(data[0]),
            (1, true) => json!(data[0] as i8),
            (2, false) if data.len() >= 2 => {
                json!(u16::from_le_bytes([data[0], data[1]]))
            }
            (2, true) if data.len() >= 2 => {
                json!(i16::from_le_bytes([data[0], data[1]]))
            }
            (4, false) if data.len() >= 4 => {
                json!(u32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            (4, true) if data.len() >= 4 => {
                json!(i32::from_le_bytes(data[..4].try_into().unwrap()))
            }
            (8, false) if data.len() >= 8 => {
                // JSON numbers are f64, so large u64 values may lose precision.
                // Use a number if it fits, otherwise a string.
                let v = u64::from_le_bytes(data[..8].try_into().unwrap());
                if v <= (1u64 << 53) {
                    json!(v)
                } else {
                    json!(v.to_string())
                }
            }
            (8, true) if data.len() >= 8 => {
                let v = i64::from_le_bytes(data[..8].try_into().unwrap());
                json!(v)
            }
            _ => json!(format!("0x{}", hex_str(data, *size as usize))),
        },
        Datatype::FloatingPoint { size, .. } => match *size {
            4 if data.len() >= 4 => {
                let v = f32::from_le_bytes(data[..4].try_into().unwrap());
                float_to_json(v as f64)
            }
            8 if data.len() >= 8 => {
                let v = f64::from_le_bytes(data[..8].try_into().unwrap());
                float_to_json(v)
            }
            _ => json!(format!("0x{}", hex_str(data, *size as usize))),
        },
        Datatype::String { size, .. } => {
            let end = (*size as usize).min(data.len());
            let s = std::str::from_utf8(&data[..end])
                .unwrap_or("<non-utf8>")
                .trim_end_matches('\0');
            json!(s)
        }
        Datatype::Compound { members, .. } => {
            let mut obj = Map::new();
            for m in members {
                let off = m.byte_offset as usize;
                let msz = m.datatype.element_size() as usize;
                if off + msz <= data.len() {
                    obj.insert(
                        m.name.clone(),
                        element_to_json(&m.datatype, &data[off..off + msz]),
                    );
                }
            }
            Value::Object(obj)
        }
        Datatype::Enum { base, members } => {
            let raw_val = element_to_json(base, data);
            // Try to find matching enum name
            let base_size = base.element_size() as usize;
            if base_size <= data.len() {
                let val_bytes = &data[..base_size];
                for m in members {
                    if m.value == val_bytes {
                        return json!(m.name);
                    }
                }
            }
            raw_val
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            let elem_size = element_type.element_size() as usize;
            let total: usize = dimensions.iter().map(|d| *d as usize).product();
            let mut arr = Vec::with_capacity(total);
            for i in 0..total {
                let start = i * elem_size;
                let end = (start + elem_size).min(data.len());
                if start >= data.len() {
                    break;
                }
                arr.push(element_to_json(element_type, &data[start..end]));
            }
            Value::Array(arr)
        }
        _ => json!(format!("0x{}", hex_str(data, data.len()))),
    }
}

fn float_to_json(v: f64) -> Value {
    if v.is_nan() {
        json!("NaN")
    } else if v.is_infinite() {
        if v > 0.0 {
            json!("Infinity")
        } else {
            json!("-Infinity")
        }
    } else {
        json!(v)
    }
}

fn hex_str(data: &[u8], len: usize) -> String {
    data[..len.min(data.len())]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn attr_to_json(attr: &Attribute) -> Value {
    let mut obj = Map::new();
    obj.insert("datatype".into(), datatype_to_json(&attr.datatype));
    obj.insert("dataspace".into(), dataspace_to_json(&attr.dataspace));

    let raw = &attr.raw_value;
    if !raw.is_empty() {
        let num_elements = attr.dataspace.num_elements() as usize;
        let elem_size = attr.datatype.element_size() as usize;
        if num_elements == 1 {
            obj.insert("value".into(), element_to_json(&attr.datatype, raw));
        } else if num_elements > 1 && elem_size > 0 {
            obj.insert(
                "value".into(),
                decode_data(&attr.datatype, raw, num_elements, elem_size),
            );
        }
    }

    Value::Object(obj)
}

fn datatype_to_json(dt: &Datatype) -> Value {
    match dt {
        Datatype::FixedPoint {
            size,
            signed,
            byte_order,
            ..
        } => {
            json!({
                "class": "integer",
                "size": size,
                "signed": signed,
                "order": order_str(byte_order),
            })
        }
        Datatype::FloatingPoint {
            size, byte_order, ..
        } => {
            json!({
                "class": "float",
                "size": size,
                "order": order_str(byte_order),
            })
        }
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            json!({
                "class": "string",
                "size": size,
                "padding": match padding {
                    hdf5_pure::datatype::StringPadding::NullTerminate => "nullterm",
                    hdf5_pure::datatype::StringPadding::NullPad => "nullpad",
                    hdf5_pure::datatype::StringPadding::SpacePad => "spacepad",
                },
                "charset": match char_set {
                    hdf5_pure::datatype::CharacterSet::Ascii => "ascii",
                    hdf5_pure::datatype::CharacterSet::Utf8 => "utf-8",
                },
            })
        }
        Datatype::Compound { size, members } => {
            let fields: Vec<Value> = members
                .iter()
                .map(|m| {
                    json!({
                        "name": m.name,
                        "offset": m.byte_offset,
                        "datatype": datatype_to_json(&m.datatype),
                    })
                })
                .collect();
            json!({
                "class": "compound",
                "size": size,
                "fields": fields,
            })
        }
        Datatype::Enum { base, members } => {
            let mapping: Map<String, Value> = members
                .iter()
                .map(|m| {
                    let val = element_to_json(base, &m.value);
                    (m.name.clone(), val)
                })
                .collect();
            json!({
                "class": "enum",
                "base": datatype_to_json(base),
                "members": Value::Object(mapping),
            })
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            json!({
                "class": "array",
                "dims": dimensions,
                "base": datatype_to_json(element_type),
            })
        }
        Datatype::VarLen {
            element_type,
            is_string,
            ..
        } => {
            json!({
                "class": if *is_string { "vlen_string" } else { "vlen" },
                "base": datatype_to_json(element_type),
            })
        }
        Datatype::Opaque { size, tag } => {
            json!({"class": "opaque", "size": size, "tag": tag})
        }
        Datatype::BitField { size, .. } => {
            json!({"class": "bitfield", "size": size})
        }
        Datatype::Reference { ref_type } => {
            json!({
                "class": "reference",
                "ref_type": match ref_type {
                    hdf5_pure::datatype::ReferenceType::Object => "object",
                    hdf5_pure::datatype::ReferenceType::DatasetRegion => "region",
                },
            })
        }
        Datatype::Time { size, .. } => {
            json!({"class": "time", "size": size})
        }
        Datatype::Complex { base, .. } => {
            json!({"class": "complex", "base": datatype_to_json(base)})
        }
    }
}

fn dataspace_to_json(ds: &Dataspace) -> Value {
    match ds {
        Dataspace::Scalar => json!({"class": "scalar"}),
        Dataspace::Null => json!({"class": "null"}),
        Dataspace::Simple {
            dimensions,
            max_dimensions,
        } => {
            let mut obj = json!({
                "class": "simple",
                "dims": dimensions,
            });
            if let Some(maxd) = max_dimensions {
                let maxvals: Vec<Value> = maxd
                    .iter()
                    .map(|d| {
                        if *d == u64::MAX {
                            json!("unlimited")
                        } else {
                            json!(d)
                        }
                    })
                    .collect();
                obj["maxdims"] = Value::Array(maxvals);
            }
            obj
        }
    }
}

fn layout_to_json(layout: &DataLayout) -> Value {
    match layout {
        DataLayout::Compact { data } => {
            json!({"class": "compact", "size": data.len()})
        }
        DataLayout::Contiguous { address, size } => {
            json!({"class": "contiguous", "address": format!("{:#x}", address), "size": size})
        }
        DataLayout::Chunked {
            chunk_dims,
            address,
            chunk_index_type,
            ..
        } => {
            json!({
                "class": "chunked",
                "dims": chunk_dims,
                "index": chunk_index_type.as_ref().map(|t| format!("{:?}", t)).unwrap_or("BTreeV1".into()),
                "address": format!("{:#x}", address),
            })
        }
        DataLayout::Virtual { .. } => json!({"class": "virtual"}),
    }
}

fn order_str(order: &hdf5_pure::datatype::ByteOrder) -> &'static str {
    match order {
        hdf5_pure::datatype::ByteOrder::LittleEndian => "LE",
        hdf5_pure::datatype::ByteOrder::BigEndian => "BE",
        hdf5_pure::datatype::ByteOrder::Vax => "VAX",
    }
}

fn format_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", parent, child)
    }
}
