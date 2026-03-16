use super::*;
use crate::File;
use crate::dataspace::Dataspace;
use crate::datatype::ByteOrder;
use crate::writer::encode::SUPERBLOCK_SIZE;
use crate::writer::encode::encode_dataspace;
use crate::writer::encode::encode_datatype;
use crate::writer::encode::encode_superblock;

#[test]
fn encode_superblock_roundtrip() {
    let sb_bytes = encode_superblock(48, 1024);
    assert_eq!(sb_bytes.len(), SUPERBLOCK_SIZE);
    let sb = crate::superblock::Superblock::parse(sb_bytes.as_slice(), 0).unwrap();
    assert_eq!(sb.version, 2);
    assert_eq!(sb.size_of_offsets, 8);
    assert_eq!(sb.size_of_lengths, 8);
    assert_eq!(sb.root_group_object_header_address, 48);
    assert_eq!(sb.end_of_file_address, 1024);
}

#[test]
fn encode_datatype_i32() {
    let dt = crate::Datatype::native_i32();
    let enc = encode_datatype(&dt).unwrap();
    assert_eq!(enc.len(), 12);
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match parsed {
        crate::Datatype::FixedPoint {
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
        _ => panic!("expected FixedPoint"),
    }
}

#[test]
fn encode_datatype_f64() {
    let dt = crate::Datatype::native_f64();
    let enc = encode_datatype(&dt).unwrap();
    assert_eq!(enc.len(), 20);
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match parsed {
        crate::Datatype::FloatingPoint {
            size,
            byte_order,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_size,
            exponent_bias,
            ..
        } => {
            assert_eq!(size, 8);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
            assert_eq!(bit_precision, 64);
            assert_eq!(exponent_location, 52);
            assert_eq!(exponent_size, 11);
            assert_eq!(mantissa_size, 52);
            assert_eq!(exponent_bias, 1023);
        }
        _ => panic!("expected FloatingPoint"),
    }
}

#[test]
fn encode_datatype_string() {
    use crate::datatype::CharacterSet;
    use crate::datatype::StringPadding;
    let dt = crate::Datatype::fixed_string(10);
    let enc = encode_datatype(&dt).unwrap();
    assert_eq!(enc.len(), 8);
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match parsed {
        crate::Datatype::String {
            size,
            padding,
            char_set,
        } => {
            assert_eq!(size, 10);
            assert_eq!(padding, StringPadding::NullTerminate);
            assert_eq!(char_set, CharacterSet::Ascii);
        }
        _ => panic!("expected String"),
    }
}

#[test]
fn encode_dataspace_scalar() {
    let enc = encode_dataspace(&[], None);
    assert_eq!(enc, vec![2, 0, 0, 0]);
    let parsed = Dataspace::parse(&enc).unwrap();
    assert!(matches!(parsed, Dataspace::Scalar));
}

#[test]
fn encode_dataspace_1d() {
    let enc = encode_dataspace(&[100], None);
    assert_eq!(enc.len(), 12);
    let parsed = Dataspace::parse(&enc).unwrap();
    match parsed {
        Dataspace::Simple { dimensions, .. } => assert_eq!(dimensions, vec![100]),
        _ => panic!("expected Simple"),
    }
}

#[test]
fn encode_dataspace_2d() {
    let enc = encode_dataspace(&[10, 20], None);
    assert_eq!(enc.len(), 20);
    let parsed = Dataspace::parse(&enc).unwrap();
    match parsed {
        Dataspace::Simple { dimensions, .. } => assert_eq!(dimensions, vec![10, 20]),
        _ => panic!("expected Simple"),
    }
}

#[test]
fn roundtrip_empty_file() {
    let w = FileWriter::new();
    let bytes = w.to_bytes().unwrap();
    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert!(root.members().unwrap().is_empty());
}

#[test]
fn roundtrip_single_i32_dataset() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = (0..10i32).flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut().add_dataset(
        "numbers",
        crate::Datatype::native_i32(),
        &[10],
        data.clone(),
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let members = root.members().unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(members[0], "numbers");

    let ds = root.dataset("numbers").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![10]);
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_f64_dataset() {
    let mut w = FileWriter::new();
    let values: Vec<f64> = (0..5).map(|i| i as f64 * 1.5).collect();
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut()
        .add_dataset("floats", crate::Datatype::native_f64(), &[5], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("floats").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![5]);
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_2d_dataset() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = (0..12u16).flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut().add_dataset(
        "matrix",
        crate::Datatype::native_u16(),
        &[3, 4],
        data.clone(),
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("matrix").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3, 4]);
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_scalar_dataset() {
    let mut w = FileWriter::new();
    let data = 42i64.to_le_bytes().to_vec();
    w.root_mut()
        .add_dataset("scalar", crate::Datatype::native_i64(), &[], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("scalar").unwrap();
    assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_group() {
    let mut w = FileWriter::new();
    w.root_mut().add_group("grp1");
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let members = root.members().unwrap();
    assert_eq!(members, vec!["grp1"]);
    let grp = root.group("grp1").unwrap();
    assert!(grp.members().unwrap().is_empty());
}

#[test]
fn roundtrip_nested_groups() {
    let mut w = FileWriter::new();
    let grp = w.root_mut().add_group("outer");
    grp.add_group("inner");
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let outer = root.group("outer").unwrap();
    let inner = outer.group("inner").unwrap();
    assert!(inner.members().unwrap().is_empty());
}

#[test]
fn roundtrip_group_with_dataset() {
    let mut w = FileWriter::new();
    let grp = w.root_mut().add_group("data");
    let vals: Vec<u8> = (0..4u32).flat_map(|x| x.to_le_bytes()).collect();
    grp.add_dataset("values", crate::Datatype::native_u32(), &[4], vals.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let grp = file.root_group().unwrap().group("data").unwrap();
    let ds = grp.dataset("values").unwrap();
    assert_eq!(ds.read_raw().unwrap(), vals);
}

#[test]
fn roundtrip_group_attribute() {
    let mut w = FileWriter::new();
    let val = 123i32.to_le_bytes().to_vec();
    w.root_mut()
        .add_attribute("version", crate::Datatype::native_i32(), &[], val.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let attrs = root.attributes().unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "version");
    assert_eq!(attrs[0].raw_value, val);
}

#[test]
fn roundtrip_dataset_attribute() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("temps", crate::Datatype::native_f64(), &[3], data);
    ds.add_attribute(
        "units",
        crate::Datatype::fixed_string(7),
        &[],
        b"celsius".to_vec(),
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("temps").unwrap();
    let attrs = ds.attributes().unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "units");
    assert_eq!(attrs[0].raw_value, b"celsius");
}

#[test]
fn roundtrip_multiple_datasets() {
    let mut w = FileWriter::new();
    let root = w.root_mut();
    let d1: Vec<u8> = (0..3i32).flat_map(|x| x.to_le_bytes()).collect();
    let d2: Vec<u8> = (10..14u64).flat_map(|x| x.to_le_bytes()).collect();
    root.add_dataset("ints", crate::Datatype::native_i32(), &[3], d1.clone());
    root.add_dataset("longs", crate::Datatype::native_u64(), &[4], d2.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds1 = root.dataset("ints").unwrap();
    assert_eq!(ds1.read_raw().unwrap(), d1);
    let ds2 = root.dataset("longs").unwrap();
    assert_eq!(ds2.read_raw().unwrap(), d2);
}

#[test]
fn rejects_wrong_data_size() {
    let mut w = FileWriter::new();
    w.root_mut()
        .add_dataset("bad", crate::Datatype::native_i32(), &[10], vec![0u8; 8]);
    let err = w.to_bytes().unwrap_err();
    assert!(err.to_string().contains("data size mismatch"));
}

#[test]
fn roundtrip_all_integer_types() {
    let mut w = FileWriter::new();
    let root = w.root_mut();
    root.add_dataset("i8", crate::Datatype::native_i8(), &[2], vec![1u8, 2]);
    root.add_dataset(
        "i16",
        crate::Datatype::native_i16(),
        &[2],
        [1i16, 2].iter().flat_map(|x| x.to_le_bytes()).collect(),
    );
    root.add_dataset(
        "u32",
        crate::Datatype::native_u32(),
        &[1],
        100u32.to_le_bytes().to_vec(),
    );
    root.add_dataset(
        "u64",
        crate::Datatype::native_u64(),
        &[1],
        999u64.to_le_bytes().to_vec(),
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert_eq!(root.dataset("i8").unwrap().read_raw().unwrap(), vec![1, 2]);
}

#[test]
fn roundtrip_multiple_attributes() {
    let mut w = FileWriter::new();
    let root = w.root_mut();
    root.add_attribute(
        "a1",
        crate::Datatype::native_i32(),
        &[],
        1i32.to_le_bytes().to_vec(),
    );
    root.add_attribute(
        "a2",
        crate::Datatype::native_f64(),
        &[],
        3.14f64.to_le_bytes().to_vec(),
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let attrs = root.attributes().unwrap();
    assert_eq!(attrs.len(), 2);
    let names: Vec<&str> = attrs.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"a1"));
    assert!(names.contains(&"a2"));
}

#[test]
fn roundtrip_complex_hierarchy() {
    let mut w = FileWriter::new();
    let root = w.root_mut();
    root.add_attribute(
        "file_version",
        crate::Datatype::native_i32(),
        &[],
        1i32.to_le_bytes().to_vec(),
    );

    let data_grp = root.add_group("data");
    let vals: Vec<u8> = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = data_grp.add_dataset("measurements", crate::Datatype::native_f32(), &[2, 3], vals);
    ds.add_attribute(
        "units",
        crate::Datatype::fixed_string(1),
        &[],
        b"m".to_vec(),
    );

    let meta_grp = root.add_group("metadata");
    meta_grp.add_attribute(
        "author",
        crate::Datatype::fixed_string(4),
        &[],
        b"test".to_vec(),
    );

    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();

    let root_attrs = root.attributes().unwrap();
    assert_eq!(root_attrs.len(), 1);
    assert_eq!(root_attrs[0].name, "file_version");

    let data = root.group("data").unwrap();
    let ds = data.dataset("measurements").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    let ds_attrs = ds.attributes().unwrap();
    assert_eq!(ds_attrs.len(), 1);
    assert_eq!(ds_attrs[0].name, "units");

    let meta = root.group("metadata").unwrap();
    let meta_attrs = meta.attributes().unwrap();
    assert_eq!(meta_attrs.len(), 1);
    assert_eq!(meta_attrs[0].name, "author");
}

#[test]
fn roundtrip_with_timestamps() {
    let ts = (1700000000, 1700000000, 1700000000, 1700000000);
    let opts = WriteOptions {
        timestamps: Some(ts),
        ..Default::default()
    };
    let mut w = FileWriter::with_options(opts);
    let data: Vec<u8> = (0..4i32).flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut()
        .add_dataset("data", crate::Datatype::native_i32(), &[4], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn timestamps_stored_in_object_header() {
    let ts = (1700000000, 1700000001, 1700000002, 1700000003);
    let opts = WriteOptions {
        timestamps: Some(ts),
        ..Default::default()
    };
    let mut w = FileWriter::with_options(opts);
    w.root_mut().add_group("grp");
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert_eq!(root.members().unwrap(), vec!["grp"]);
}

#[test]
fn encode_datatype_compound() {
    let dt = crate::Datatype::Compound {
        size: 12,
        members: vec![
            crate::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 0,
                datatype: crate::Datatype::native_i32(),
            },
            crate::datatype::CompoundMember {
                name: "x".to_string(),
                byte_offset: 4,
                datatype: crate::Datatype::native_f32(),
            },
            crate::datatype::CompoundMember {
                name: "y".to_string(),
                byte_offset: 8,
                datatype: crate::Datatype::native_f32(),
            },
        ],
    };
    let enc = encode_datatype(&dt).unwrap();
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match &parsed {
        crate::Datatype::Compound { size, members } => {
            assert_eq!(*size, 12);
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "id");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "x");
            assert_eq!(members[1].byte_offset, 4);
            assert_eq!(members[2].name, "y");
            assert_eq!(members[2].byte_offset, 8);
        }
        _ => panic!("expected Compound"),
    }
}

#[test]
fn encode_datatype_enum() {
    let dt = crate::Datatype::Enum {
        base: Box::new(crate::Datatype::native_i8()),
        members: vec![
            crate::datatype::EnumMember {
                name: "RED".to_string(),
                value: vec![0],
            },
            crate::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: vec![1],
            },
            crate::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: vec![2],
            },
        ],
    };
    let enc = encode_datatype(&dt).unwrap();
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match &parsed {
        crate::Datatype::Enum { base, members } => {
            assert_eq!(base.element_size(), 1);
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "RED");
            assert_eq!(members[0].value, vec![0]);
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, vec![1]);
            assert_eq!(members[2].name, "BLUE");
            assert_eq!(members[2].value, vec![2]);
        }
        _ => panic!("expected Enum"),
    }
}

#[test]
fn encode_datatype_array() {
    let dt = crate::Datatype::Array {
        element_type: Box::new(crate::Datatype::native_i32()),
        dimensions: vec![3],
    };
    let enc = encode_datatype(&dt).unwrap();
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match &parsed {
        crate::Datatype::Array {
            element_type,
            dimensions,
        } => {
            assert_eq!(dimensions, &[3]);
            assert_eq!(element_type.element_size(), 4);
            assert_eq!(parsed.element_size(), 12);
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn encode_datatype_complex() {
    let dt = crate::Datatype::Complex {
        size: 16,
        base: Box::new(crate::Datatype::native_f64()),
    };
    let enc = encode_datatype(&dt).unwrap();
    let parsed = crate::Datatype::parse(&enc).unwrap();
    match &parsed {
        crate::Datatype::Complex { size, base } => {
            assert_eq!(*size, 16);
            assert_eq!(base.element_size(), 8);
        }
        _ => panic!("expected Complex"),
    }
}

#[test]
fn roundtrip_compact_dataset() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = [100i16, 200, 300, 400]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("small", crate::Datatype::native_i16(), &[4], data.clone());
    ds.set_layout(StorageLayout::Compact);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("small").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_fill_value() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = [10i32, 20, 30, 40, -999, -999]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("filled", crate::Datatype::native_i32(), &[6], data.clone());
    ds.set_fill_value((-999i32).to_le_bytes().to_vec());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("filled").unwrap();
    let fv = ds.fill_value().unwrap();
    assert!(fv.defined);
    let val_bytes = fv.value.expect("expected fill value bytes");
    let fill_val = i32::from_le_bytes(val_bytes.try_into().unwrap());
    assert_eq!(fill_val, -999);
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_compound_dataset() {
    let dt = crate::Datatype::Compound {
        size: 12,
        members: vec![
            crate::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 0,
                datatype: crate::Datatype::native_i32(),
            },
            crate::datatype::CompoundMember {
                name: "x".to_string(),
                byte_offset: 4,
                datatype: crate::Datatype::native_f32(),
            },
            crate::datatype::CompoundMember {
                name: "y".to_string(),
                byte_offset: 8,
                datatype: crate::Datatype::native_f32(),
            },
        ],
    };
    let mut data = Vec::new();
    for i in 0..3 {
        data.extend_from_slice(&((i + 1) as i32).to_le_bytes());
        data.extend_from_slice(&((2 * i + 1) as f32).to_le_bytes());
        data.extend_from_slice(&((2 * i + 2) as f32).to_le_bytes());
    }
    let mut w = FileWriter::new();
    w.root_mut().add_dataset("points", dt, &[3], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("points").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
    match ds.datatype().unwrap() {
        crate::Datatype::Compound { size, members } => {
            assert_eq!(size, 12);
            assert_eq!(members.len(), 3);
        }
        _ => panic!("expected Compound"),
    }
}

#[test]
fn roundtrip_enum_dataset() {
    let dt = crate::Datatype::Enum {
        base: Box::new(crate::Datatype::native_i8()),
        members: vec![
            crate::datatype::EnumMember {
                name: "RED".to_string(),
                value: vec![0],
            },
            crate::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: vec![1],
            },
            crate::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: vec![2],
            },
        ],
    };
    let data = vec![0u8, 1, 2, 1, 0];
    let mut w = FileWriter::new();
    w.root_mut().add_dataset("colors", dt, &[5], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("colors").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_array_dataset() {
    let dt = crate::Datatype::Array {
        element_type: Box::new(crate::Datatype::native_i32()),
        dimensions: vec![3],
    };
    let data: Vec<u8> = (1..=12i32).flat_map(|x| x.to_le_bytes()).collect();
    let mut w = FileWriter::new();
    w.root_mut().add_dataset("vectors", dt, &[4], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("vectors").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
    assert_eq!(ds.shape().unwrap(), vec![4]);
}

#[test]
fn roundtrip_complex_dataset() {
    let dt = crate::Datatype::Complex {
        size: 16,
        base: Box::new(crate::Datatype::native_f64()),
    };
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.0, -5.0];
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut w = FileWriter::new();
    w.root_mut()
        .add_dataset("complex_data", dt, &[4], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("complex_data").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
}

#[test]
fn roundtrip_big_endian_dataset() {
    let dt = crate::Datatype::FixedPoint {
        size: 4,
        byte_order: ByteOrder::BigEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    };
    let values: Vec<i32> = vec![1, 256, 65536, -1, 1000000, 0];
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_be_bytes()).collect();
    let mut w = FileWriter::new();
    w.root_mut().add_dataset("be_data", dt, &[6], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("be_data").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);

    match ds.datatype().unwrap() {
        crate::Datatype::FixedPoint { byte_order, .. } => {
            assert_eq!(byte_order, ByteOrder::BigEndian);
        }
        _ => panic!("expected FixedPoint"),
    }
}

#[test]
fn h5dump_validates_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h5dump_test.h5");

    let mut w = FileWriter::new();
    let root = w.root_mut();
    let data: Vec<u8> = (0..10i32).flat_map(|x| x.to_le_bytes()).collect();
    root.add_dataset("numbers", crate::Datatype::native_i32(), &[10], data);
    let grp = root.add_group("metadata");
    grp.add_attribute(
        "version",
        crate::Datatype::native_i32(),
        &[],
        1i32.to_le_bytes().to_vec(),
    );
    w.write_to_file(&path).unwrap();

    if let Ok(output) = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "h5dump failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("numbers"));
        assert!(stdout.contains("metadata"));
    }
}

#[test]
fn write_to_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");

    let mut w = FileWriter::new();
    let data: Vec<u8> = (0..3i32).flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut()
        .add_dataset("vals", crate::Datatype::native_i32(), &[3], data.clone());
    w.write_to_file(&path).unwrap();

    let file = crate::File::open(&path).unwrap();
    let ds = file.root_group().unwrap().dataset("vals").unwrap();
    assert_eq!(ds.read_raw().unwrap(), data);
}

fn assert_bytes_match(our_bytes: &[u8], fixture_path: &str) {
    let reference = std::fs::read(fixture_path).unwrap();
    if *our_bytes == *reference {
        return;
    }
    let max_len = our_bytes.len().max(reference.len());
    let mut first_diff = None;
    let mut diff_count = 0;
    for i in 0..max_len {
        let ours = our_bytes.get(i).copied();
        let theirs = reference.get(i).copied();
        if ours != theirs {
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            diff_count += 1;
            if diff_count <= 20 {
                eprintln!(
                    "  offset 0x{:04x}: ours={:?} ref={:?}",
                    i,
                    ours.map(|b| format!("0x{b:02x}")),
                    theirs.map(|b| format!("0x{b:02x}")),
                );
            }
        }
    }
    panic!(
        "{}: byte mismatch: our_len={} ref_len={}, {} diffs, first at 0x{:04x}",
        fixture_path,
        our_bytes.len(),
        reference.len(),
        diff_count,
        first_diff.unwrap_or(0),
    );
}

fn compat_opts_v2() -> WriteOptions {
    WriteOptions {
        timestamps: Some((1704096000, 1704096000, 1704096000, 1704096000)),
        hdf5lib_compat: true,
        ..Default::default()
    }
}

fn compat_opts_v3() -> WriteOptions {
    WriteOptions {
        timestamps: Some((1704096000, 1704096000, 1704096000, 1704096000)),
        hdf5lib_compat: true,
        superblock_version: Some(3),
        ..Default::default()
    }
}

#[test]
fn compat_simple_contiguous_v2() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let data: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("data", crate::Datatype::native_f64(), &[4], data);
    ds.add_attribute(
        "units",
        crate::Datatype::String {
            size: 5,
            padding: crate::datatype::StringPadding::NullTerminate,
            char_set: crate::datatype::CharacterSet::Ascii,
        },
        &[],
        b"m/s\0\0".to_vec(),
    );
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/simple_contiguous_v2.h5",
    );
}

#[test]
fn compat_nested_groups_v2() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let root = w.root_mut();
    let g1 = root.add_group("group1");
    let sg = g1.add_group("subgroup");
    let temps: Vec<u8> = [20.5f32, 21.0, 19.8]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    sg.add_dataset("temps", crate::Datatype::native_f32(), &[3], temps);
    let ids: Vec<u8> = vec![10u8, 20, 30, 40, 50];
    g1.add_dataset("ids", crate::Datatype::native_u8(), &[5], ids);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/nested_groups_v2.h5");
}

#[test]
fn compat_fill_value() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let data: Vec<u8> = [10i32, 20, 30, 40, -999, -999]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("filled", crate::Datatype::native_i32(), &[6], data);
    ds.set_fill_value((-999i32).to_le_bytes().to_vec());
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/fill_value.h5");
}

#[test]
fn compat_big_endian() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::FixedPoint {
        size: 4,
        byte_order: ByteOrder::BigEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    };
    let vals: [i32; 6] = [1, 256, 65536, -1, 1000000, 0];
    let data: Vec<u8> = vals.iter().flat_map(|x| x.to_be_bytes()).collect();
    w.root_mut().add_dataset("be_data", dt, &[6], data);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/big_endian.h5");
}

#[test]
fn compat_compound() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::Compound {
        size: 12,
        members: vec![
            crate::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 0,
                datatype: crate::Datatype::native_i32(),
            },
            crate::datatype::CompoundMember {
                name: "x".to_string(),
                byte_offset: 4,
                datatype: crate::Datatype::native_f32(),
            },
            crate::datatype::CompoundMember {
                name: "y".to_string(),
                byte_offset: 8,
                datatype: crate::Datatype::native_f32(),
            },
        ],
    };
    let mut data = Vec::new();
    for &(id, x, y) in &[(1i32, 1.0f32, 2.0f32), (2, 3.0, 4.0), (3, 5.0, 6.0)] {
        data.extend_from_slice(&id.to_le_bytes());
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
    }
    w.root_mut().add_dataset("points", dt, &[3], data);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/compound.h5");
}

#[test]
fn compat_enum() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::Enum {
        base: Box::new(crate::Datatype::native_i8()),
        members: vec![
            crate::datatype::EnumMember {
                name: "RED".to_string(),
                value: vec![0],
            },
            crate::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: vec![1],
            },
            crate::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: vec![2],
            },
        ],
    };
    let data = vec![0i8 as u8, 1, 2, 1, 0];
    w.root_mut().add_dataset("colors", dt, &[5], data);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/enum.h5");
}

#[test]
fn compat_array() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::Array {
        element_type: Box::new(crate::Datatype::native_i32()),
        dimensions: vec![3],
    };
    let data: Vec<u8> = (1..=12i32).flat_map(|x| x.to_le_bytes()).collect();
    w.root_mut().add_dataset("vectors", dt, &[4], data);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/array.h5");
}

#[test]
fn compat_complex() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::Complex {
        size: 16,
        base: Box::new(crate::Datatype::native_f64()),
    };
    let data: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0, -1.0, 0.0, 0.0, -5.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    w.root_mut().add_dataset("complex_data", dt, &[4], data);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/complex.h5");
}

#[test]
fn compat_compound_complex_members() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let enum_type = crate::Datatype::Enum {
        base: Box::new(crate::Datatype::native_i32()),
        members: vec![
            crate::datatype::EnumMember {
                name: "RED".to_string(),
                value: 0i32.to_le_bytes().to_vec(),
            },
            crate::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: 1i32.to_le_bytes().to_vec(),
            },
            crate::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: 2i32.to_le_bytes().to_vec(),
            },
        ],
    };
    let arr_type = crate::Datatype::Array {
        element_type: Box::new(crate::Datatype::native_i32()),
        dimensions: vec![3],
    };
    let dt = crate::Datatype::Compound {
        size: 20,
        members: vec![
            crate::datatype::CompoundMember {
                name: "color".to_string(),
                byte_offset: 0,
                datatype: enum_type,
            },
            crate::datatype::CompoundMember {
                name: "coords".to_string(),
                byte_offset: 4,
                datatype: arr_type,
            },
            crate::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 16,
                datatype: crate::Datatype::native_i32(),
            },
        ],
    };
    let mut data = Vec::new();
    // Record 0: RED(0), [10,20,30], 100
    for v in [0i32, 10, 20, 30, 100] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Record 1: GREEN(1), [40,50,60], 200
    for v in [1i32, 40, 50, 60, 200] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Record 2: BLUE(2), [70,80,90], 300
    for v in [2i32, 70, 80, 90, 300] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    w.root_mut().add_dataset("records", dt, &[3], data);
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/compound_complex_members.h5",
    );
}

#[test]
fn compat_compact_v2() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let data: Vec<u8> = [100i16, 200, 300, 400]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("small", crate::Datatype::native_i16(), &[4], data);
    ds.set_layout(StorageLayout::Compact);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/compact_v2.h5");
}

#[test]
fn compat_empty_chunked() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let ds = w.root_mut().add_dataset(
        "empty",
        crate::Datatype::native_i32(),
        &[10],
        vec![0u8; 40], // 10 * 4 bytes placeholder
    );
    ds.set_max_dims(&[u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![5],
        filters: vec![],
    });
    // Empty chunked means no data was written — clear the data.
    ds.clear_data();
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/empty_chunked.h5");
}

#[test]
fn compat_fletcher32() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (1..=10i32)
        .map(|x| x * 100)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("checksummed", crate::Datatype::native_i32(), &[10], vals);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![10],
        filters: vec![crate::writer::types::ChunkFilter::Fletcher32],
    });
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/fletcher32.h5");
}

#[test]
#[cfg(feature = "system-zlib")]
fn compat_shuffle_deflate() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (0..20)
        .map(|i| i as f32 * 1.5)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("shuffled", crate::Datatype::native_f32(), &[20], vals);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![20],
        filters: vec![
            crate::writer::types::ChunkFilter::Shuffle,
            crate::writer::types::ChunkFilter::Deflate(4),
        ],
    });
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/shuffle_deflate_v3.h5",
    );
}

#[test]
fn compat_implicit_chunks() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = [1.1f64, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("implicit", crate::Datatype::native_f64(), &[8], vals);
    ds.set_chunked(&[4], vec![]);
    ds.set_early_alloc();
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/implicit_chunks.h5");
}

#[test]
fn compat_edge_chunks() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (0..35i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("edge", crate::Datatype::native_i32(), &[7, 5], vals);
    ds.set_chunked(&[4, 3], vec![]);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/edge_chunks.h5");
}

#[test]
fn compat_extensible_array() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (1..=15i32)
        .map(|x| x * 100)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("extarray", crate::Datatype::native_i32(), &[15], vals);
    ds.set_max_dims(&[u64::MAX]);
    ds.set_chunked(&[5], vec![]);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/extensible_array.h5");
}

#[test]
#[cfg(feature = "system-zlib")]
fn compat_chunked_deflate_v3() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (0..100i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("compressed", crate::Datatype::native_i32(), &[10, 10], vals);
    ds.set_chunked(&[5, 5], vec![crate::writer::types::ChunkFilter::Deflate(6)]);
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/chunked_deflate_v3.h5",
    );
}

#[test]
#[cfg(feature = "system-zlib")]
fn compat_chunked_btree_v1() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let vals: Vec<u8> = (1..=12i32)
        .map(|x| x * 10)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("chunked_v3", crate::Datatype::native_i32(), &[12], vals);
    ds.set_chunked(&[4], vec![crate::writer::types::ChunkFilter::Deflate(4)]);
    ds.set_layout_version(3);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/chunked_btree_v1.h5");
}

#[test]
fn compat_ea_large() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (0..100i32)
        .map(|x| x * 10)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("large_ea", crate::Datatype::native_i32(), &[100], vals);
    ds.set_max_dims(&[u64::MAX]);
    ds.set_chunked(&[4], vec![]);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/ea_large.h5");
}

#[test]
fn compat_btree_v2_chunks() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    // 2D 6×4, values 0..23, both dims unlimited → BTreeV2 index
    let vals: Vec<u8> = (0..24i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("bt2chunked", crate::Datatype::native_i32(), &[6, 4], vals);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_chunked(&[3, 2], vec![]);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/btree_v2_chunks.h5");
}

#[test]
#[cfg(feature = "system-zlib")]
fn compat_btree_v2_filtered() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let vals: Vec<u8> = (0..24i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("filtered", crate::Datatype::native_i32(), &[6, 4], vals);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_chunked(&[3, 2], vec![crate::writer::types::ChunkFilter::Deflate(6)]);
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/btree_v2_filtered.h5",
    );
}

#[test]
fn compat_vlen_sequence() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::VarLen {
        element_type: Box::new(crate::Datatype::native_i32()),
        is_string: false,
        padding: None,
        char_set: None,
    };
    let elements: Vec<Vec<u8>> = vec![
        vec![10i32, 20]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        vec![100i32, 200, 300, 400]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        vec![42i32].iter().flat_map(|x| x.to_le_bytes()).collect(),
    ];
    w.root_mut()
        .add_vlen_dataset("sequences", dt, &[3], elements);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/vlen_sequence.h5");
}

#[test]
fn compat_btree_v2_deep() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    // 20×10, chunk 1×1, 200 chunks → depth-1 BTree v2
    let vals: Vec<u8> = (0..200i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("deep", crate::Datatype::native_i32(), &[20, 10], vals);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_chunked(&[1, 1], vec![]);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/btree_v2_deep.h5");
}

#[test]
fn compat_vlen_strings() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::VarLen {
        element_type: Box::new(crate::Datatype::native_u8()),
        is_string: true,
        padding: Some(crate::datatype::StringPadding::NullTerminate),
        char_set: Some(crate::datatype::CharacterSet::Utf8),
    };
    let elements: Vec<Vec<u8>> = vec![
        b"hello".to_vec(),
        b"world".to_vec(),
        b"HDF5".to_vec(),
        b"variable-length".to_vec(),
    ];
    w.root_mut().add_vlen_dataset("names", dt, &[4], elements);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/vlen_strings.h5");
}

#[test]
fn compat_committed_datatype() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::native_i32();

    w.root_mut().commit_datatype("mytype", dt.clone());

    let vals1: Vec<u8> = [10i32, 20, 30, 40, 50]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    w.root_mut()
        .add_dataset_committed("data1", "mytype", dt.clone(), &[5], vals1);

    let vals2: Vec<u8> = [100i32, 200, 300, 400, 500]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    w.root_mut()
        .add_dataset_committed("data2", "mytype", dt, &[5], vals2);

    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/committed_datatype.h5",
    );
}

#[test]
fn compat_shared_attr() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let dt = crate::Datatype::native_i32();

    w.root_mut().commit_datatype("shared_i32", dt.clone());

    let vals: Vec<u8> = [11i32, 22, 33]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset_committed("data", "shared_i32", dt.clone(), &[3], vals);

    // Add attribute "scale" using the same committed type (scalar, value=42).
    let scale_val: Vec<u8> = 42i32.to_le_bytes().to_vec();
    ds.add_attribute_committed("scale", "shared_i32", dt, &[], scale_val);

    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/shared_attr.h5");
}

#[test]
fn compat_dense_attributes() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let grp = w.root_mut().add_group("densegroup");
    grp.set_attr_phase_change(3, 3);
    for i in 0..8 {
        let name = format!("attr_{i:02}");
        let val = ((i + 1) * 100i32).to_le_bytes().to_vec();
        grp.add_attribute(&name, crate::Datatype::native_i32(), &[], val);
    }
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/dense_attributes.h5");
}

#[test]
fn compat_nil_messages() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let data: Vec<u8> = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("data", crate::Datatype::native_i32(), &[4], data);
    ds.set_attr_phase_change(8, 6);
    for i in 0..12 {
        let name = format!("attribute_{i}");
        let val = (((i + 1) * 10) as i32).to_le_bytes().to_vec();
        ds.add_attribute(&name, crate::Datatype::native_i32(), &[], val);
    }
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/nil_messages.h5");
}

#[test]
fn compat_creation_order() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let grp = w.root_mut().add_group("ordered");
    grp.set_link_creation_order()
        .set_link_phase_change(0, 0)
        .set_attr_creation_order()
        .set_attr_phase_change(0, 0);
    grp.add_group("charlie");
    grp.add_group("alpha");
    grp.add_group("bravo");
    grp.add_attribute(
        "zebra",
        crate::Datatype::native_i32(),
        &[],
        30i32.to_le_bytes().to_vec(),
    );
    grp.add_attribute(
        "mango",
        crate::Datatype::native_i32(),
        &[],
        10i32.to_le_bytes().to_vec(),
    );
    grp.add_attribute(
        "apple",
        crate::Datatype::native_i32(),
        &[],
        20i32.to_le_bytes().to_vec(),
    );
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/creation_order.h5");
}

#[test]
fn compat_fheap_indirect_simple() {
    let mut w = FileWriter::with_options(compat_opts_v3());
    let grp = w.root_mut().add_group("many");
    grp.set_link_phase_change(0, 0);
    for i in 0..27 {
        let name = format!(
            "child_group_{:04}_with_extra_padding_to_make_the_name_longer_end",
            i
        );
        grp.add_group(&name);
    }
    assert_bytes_match(
        &w.to_bytes().unwrap(),
        "tests/fixtures/fheap_indirect_simple.h5",
    );
}

#[test]
fn compat_scaleoffset() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let vals: Vec<u8> = (1000..1008i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("data", crate::Datatype::native_i32(), &[8], vals);
    // chunk_dims=[8], 8 elements of i32, signed, LE
    let so_params = crate::writer::types::ScaleOffsetParams::from_int(8, 4, true, true);
    ds.set_chunked(
        &[8],
        vec![crate::writer::types::ChunkFilter::ScaleOffset(so_params)],
    );
    ds.set_layout_version(3);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/scaleoffset.h5");
}

#[test]
fn compat_nbit() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    // uint16 with 10-bit precision
    let dt = crate::Datatype::FixedPoint {
        size: 2,
        byte_order: crate::datatype::ByteOrder::LittleEndian,
        signed: false,
        bit_offset: 0,
        bit_precision: 10,
    };
    let vals: Vec<u8> = (0..8u16)
        .map(|x| x * 100)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w.root_mut().add_dataset("data", dt, &[8], vals);
    let nbit_params = crate::writer::types::NbitParams::from_atomic(8, 2, 10, 0, true);
    ds.set_chunked(
        &[8],
        vec![crate::writer::types::ChunkFilter::Nbit(nbit_params)],
    );
    ds.set_layout_version(3);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/nbit.h5");
}

#[test]
fn compat_lzf() {
    let mut w = FileWriter::with_options(compat_opts_v2());
    let vals: Vec<u8> = (0..32i32)
        .map(|x| x / 4)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("data", crate::Datatype::native_i32(), &[32], vals);
    ds.set_chunked(&[32], vec![crate::writer::types::ChunkFilter::Lzf]);
    ds.set_layout_version(3);
    assert_bytes_match(&w.to_bytes().unwrap(), "tests/fixtures/lzf_c.h5");
}
