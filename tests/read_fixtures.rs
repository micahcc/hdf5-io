use std::path::PathBuf;

use hdf5_pure::Datatype;
use hdf5_pure::File;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

// ── Superblock tests ──

#[test]
fn parse_superblock_v2() {
    let data = std::fs::read(fixture("simple_contiguous_v2.h5")).unwrap();
    let sb = hdf5_pure::Superblock::parse(data.as_slice(), 0).unwrap();
    assert_eq!(sb.version, 2);
    assert_eq!(sb.size_of_offsets, 8);
    assert_eq!(sb.size_of_lengths, 8);
    assert_eq!(sb.base_address, 0);
    assert!(!sb.swmr_write_in_progress());
}

#[test]
fn parse_superblock_v3() {
    let data = std::fs::read(fixture("chunked_deflate_v3.h5")).unwrap();
    let sb = hdf5_pure::Superblock::parse(data.as_slice(), 0).unwrap();
    assert_eq!(sb.version, 3);
    assert_eq!(sb.size_of_offsets, 8);
    assert_eq!(sb.size_of_lengths, 8);
}

// ── File open + root group ──

#[test]
fn open_file_and_list_root() {
    let file = File::open(fixture("simple_contiguous_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let members = root.members().unwrap();
    assert_eq!(members, vec!["data"]);
}

#[test]
fn open_nested_groups_root() {
    let file = File::open(fixture("nested_groups_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let members = root.members().unwrap();
    assert_eq!(members, vec!["group1"]);
}

// ── Dataset metadata ──

#[test]
fn contiguous_dataset_metadata() {
    let file = File::open(fixture("simple_contiguous_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    // Datatype: F64 LE
    let dt = ds.datatype().unwrap();
    match dt {
        Datatype::FloatingPoint { size, .. } => assert_eq!(size, 8),
        other => panic!("expected FloatingPoint, got {:?}", other),
    }

    // Dataspace: [4]
    let dspace = ds.dataspace().unwrap();
    assert_eq!(dspace.shape(), &[4]);
    assert_eq!(dspace.num_elements(), 4);
}

#[test]
fn compact_dataset_metadata() {
    let file = File::open(fixture("compact_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("small").unwrap();

    let dt = ds.datatype().unwrap();
    match dt {
        Datatype::FixedPoint { size, signed, .. } => {
            assert_eq!(size, 2);
            assert!(signed);
        }
        other => panic!("expected FixedPoint, got {:?}", other),
    }

    assert_eq!(ds.shape().unwrap(), vec![4]);
}

// ── Contiguous data read ──

#[test]
fn read_contiguous_f64() {
    let file = File::open(fixture("simple_contiguous_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 32); // 4 * 8 bytes
    // Verify values: 1.0, 2.0, 3.0, 4.0 as f64 LE
    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── Compact data read ──

#[test]
fn read_compact_i16() {
    let file = File::open(fixture("compact_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("small").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 8); // 4 * 2 bytes
    let values: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![100, 200, 300, 400]);
}

// ── Nested group navigation ──

#[test]
fn navigate_nested_groups() {
    let file = File::open(fixture("nested_groups_v2.h5")).unwrap();
    let root = file.root_group().unwrap();

    let g1 = root.group("group1").unwrap();
    let g1_members = g1.members().unwrap();
    assert!(g1_members.contains(&"ids".to_string()));
    assert!(g1_members.contains(&"subgroup".to_string()));

    let sub = g1.group("subgroup").unwrap();
    let sub_members = sub.members().unwrap();
    assert_eq!(sub_members, vec!["temps"]);
}

#[test]
fn read_nested_dataset() {
    let file = File::open(fixture("nested_groups_v2.h5")).unwrap();
    let root = file.root_group().unwrap();

    // Read /group1/ids (uint8)
    let g1 = root.group("group1").unwrap();
    let ds = g1.dataset("ids").unwrap();
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw, vec![10, 20, 30, 40, 50]);

    // Read /group1/subgroup/temps (f32)
    let sub = g1.group("subgroup").unwrap();
    let ds = sub.dataset("temps").unwrap();
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 12); // 3 * 4 bytes
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![20.5, 21.0, 19.8]);
}

// ── Path-based navigation ──

#[test]
fn open_by_path() {
    let file = File::open(fixture("nested_groups_v2.h5")).unwrap();

    match file.open_path("/group1/subgroup/temps").unwrap() {
        hdf5_pure::Node::Dataset(ds) => {
            assert_eq!(ds.shape().unwrap(), vec![3]);
        }
        _ => panic!("expected Dataset"),
    }

    match file.open_path("/group1").unwrap() {
        hdf5_pure::Node::Group(_) => {}
        _ => panic!("expected Group"),
    }
}

// ── Chunked data read ──

#[test]
fn read_chunked_single_chunk_shuffle_deflate() {
    let file = File::open(fixture("shuffle_deflate_v3.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("shuffled").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![20]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 80); // 20 * 4 bytes (f32)

    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<f32> = (0..20).map(|i| i as f32 * 1.5).collect();
    assert_eq!(values, expected);
}

#[test]
fn read_chunked_fixed_array_deflate() {
    let file = File::open(fixture("chunked_deflate_v3.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("compressed").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![10, 10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 400); // 100 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..100).collect();
    assert_eq!(values, expected);
}

#[test]
fn read_chunked_extensible_array() {
    let file = File::open(fixture("extensible_array.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("extarray").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![15]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 60); // 15 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (1..=15).map(|i| i * 100).collect();
    assert_eq!(values, expected);
}

#[test]
fn read_chunked_btree_v1_deflate() {
    let file = File::open(fixture("chunked_btree_v1.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("chunked_v3").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![12]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 48); // 12 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (1..=12).map(|i| i * 10).collect();
    assert_eq!(values, expected);
}

// ── Implicit chunk index ──

#[test]
fn read_chunked_implicit() {
    let file = File::open(fixture("implicit_chunks.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("implicit").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![8]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 64); // 8 * 8 bytes (f64)

    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
    assert_eq!(values, expected);
}

// ── 2D edge chunks ──

#[test]
fn read_chunked_2d_edge() {
    let file = File::open(fixture("edge_chunks.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("edge").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![7, 5]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 140); // 35 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..35).collect();
    assert_eq!(values, expected);
}

// ── Compound datatype ──

#[test]
fn read_compound_metadata() {
    let file = File::open(fixture("compound.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("points").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 12); // sizeof(Point) = 4+4+4
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "id");
            assert_eq!(members[0].byte_offset, 0);
            assert_eq!(members[1].name, "x");
            assert_eq!(members[1].byte_offset, 4);
            assert_eq!(members[2].name, "y");
            assert_eq!(members[2].byte_offset, 8);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn read_compound_data() {
    let file = File::open(fixture("compound.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("points").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 36); // 3 * 12 bytes

    // Parse each 12-byte record: { id: i32, x: f32, y: f32 }
    for (i, record) in raw.chunks_exact(12).enumerate() {
        let id = i32::from_le_bytes(record[0..4].try_into().unwrap());
        let x = f32::from_le_bytes(record[4..8].try_into().unwrap());
        let y = f32::from_le_bytes(record[8..12].try_into().unwrap());

        assert_eq!(id, (i + 1) as i32);
        assert_eq!(x, (2 * i + 1) as f32);
        assert_eq!(y, (2 * i + 2) as f32);
    }
}

// ── Enum datatype ──

#[test]
fn read_enum_metadata() {
    let file = File::open(fixture("enum.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("colors").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::Enum { base, members } => {
            // Base type: i8
            match base.as_ref() {
                Datatype::FixedPoint { size, signed, .. } => {
                    assert_eq!(*size, 1);
                    assert!(signed);
                }
                other => panic!("expected FixedPoint base, got {:?}", other),
            }
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "RED");
            assert_eq!(members[0].value, vec![0i8 as u8]);
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, vec![1i8 as u8]);
            assert_eq!(members[2].name, "BLUE");
            assert_eq!(members[2].value, vec![2i8 as u8]);
        }
        other => panic!("expected Enum, got {:?}", other),
    }
}

#[test]
fn read_enum_data() {
    let file = File::open(fixture("enum.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("colors").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 5); // 5 * 1 byte (i8)
    assert_eq!(raw, vec![0, 1, 2, 1, 0]); // RED, GREEN, BLUE, GREEN, RED
}

// ── Array datatype ──

#[test]
fn read_array_metadata() {
    let file = File::open(fixture("array.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("vectors").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            assert_eq!(dimensions, &[3]);
            match element_type.as_ref() {
                Datatype::FixedPoint { size, signed, .. } => {
                    assert_eq!(*size, 4);
                    assert!(signed);
                }
                other => panic!("expected FixedPoint element, got {:?}", other),
            }
            // element_size should be 3 * 4 = 12
            assert_eq!(dt.element_size(), 12);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn read_array_data() {
    let file = File::open(fixture("array.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("vectors").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 48); // 4 * 3 * 4 bytes (4 arrays of 3 i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (1..=12).collect();
    assert_eq!(values, expected);
}

// ── Fletcher32 checksum filter ──

#[test]
fn read_fletcher32_chunked() {
    let file = File::open(fixture("fletcher32.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("checksummed").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 40); // 10 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (1..=10).map(|i| i * 100).collect();
    assert_eq!(values, expected);
}

// ── Variable-length types ──

#[test]
fn read_vlen_string_metadata() {
    let file = File::open(fixture("vlen_strings.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("names").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::VarLen {
            is_string,
            char_set,
            ..
        } => {
            assert!(is_string);
            assert_eq!(*char_set, Some(hdf5_pure::datatype::CharacterSet::Utf8));
        }
        other => panic!("expected VarLen, got {:?}", other),
    }

    assert_eq!(ds.shape().unwrap(), vec![4]);
}

#[test]
fn read_vlen_strings() {
    let file = File::open(fixture("vlen_strings.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("names").unwrap();

    let strings = ds.read_vlen_strings().unwrap();
    assert_eq!(strings, vec!["hello", "world", "HDF5", "variable-length"]);
}

#[test]
fn read_vlen_sequence_metadata() {
    let file = File::open(fixture("vlen_sequence.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("sequences").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::VarLen {
            element_type,
            is_string,
            ..
        } => {
            assert!(!is_string);
            assert_eq!(element_type.element_size(), 4);
        }
        other => panic!("expected VarLen, got {:?}", other),
    }

    assert_eq!(ds.shape().unwrap(), vec![3]);
}

#[test]
fn read_vlen_sequence_data() {
    let file = File::open(fixture("vlen_sequence.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("sequences").unwrap();

    let vlen_data = ds.read_vlen().unwrap();
    assert_eq!(vlen_data.len(), 3);

    // Sequence 0: [10, 20]
    let seq0: Vec<i32> = vlen_data[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(seq0, vec![10, 20]);

    // Sequence 1: [100, 200, 300, 400]
    let seq1: Vec<i32> = vlen_data[1]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(seq1, vec![100, 200, 300, 400]);

    // Sequence 2: [42]
    let seq2: Vec<i32> = vlen_data[2]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(seq2, vec![42]);
}

// ── Fill value ──

#[test]
fn read_fill_value() {
    let file = File::open(fixture("fill_value.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("filled").unwrap();

    let fv = ds.fill_value().unwrap();
    assert!(fv.defined);
    let val_bytes = fv.value.expect("expected fill value bytes");
    assert_eq!(val_bytes.len(), 4);
    let fill_val = i32::from_le_bytes(val_bytes.try_into().unwrap());
    assert_eq!(fill_val, -999);

    // Also read the data
    let raw = ds.read_raw().unwrap();
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![10, 20, 30, 40, -999, -999]);
}

// ── Fill Value Old (type 0x0004) ──

#[test]
fn fill_value_old_parse() {
    use hdf5_pure::FillValue;

    // Old fill value format: size(u32 LE) + raw bytes
    let data = [
        0x04, 0x00, 0x00, 0x00, // size = 4
        0x19, 0xFC, 0xFF, 0xFF, // -999 as i32 LE
    ];
    let fv = FillValue::parse_old(&data).unwrap();
    assert!(fv.defined);
    let val = fv.value.unwrap();
    assert_eq!(val.len(), 4);
    assert_eq!(i32::from_le_bytes(val.try_into().unwrap()), -999);

    // Old fill value with size=0 (defined but no value)
    let empty = [0x00, 0x00, 0x00, 0x00];
    let fv2 = FillValue::parse_old(&empty).unwrap();
    assert!(fv2.defined);
    assert!(fv2.value.is_none());
}

// ── Dense attributes ──

#[test]
fn read_dense_attributes() {
    let file = File::open(fixture("dense_attributes.h5")).unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("densegroup").unwrap();

    let attrs = grp.attributes().unwrap();
    assert_eq!(attrs.len(), 8);

    // Verify all 8 attributes exist with correct values
    for i in 0..8 {
        let name = format!("attr_{:02}", i);
        let attr = attrs.iter().find(|a| a.name == name);
        assert!(attr.is_some(), "missing attribute {}", name);
        let attr = attr.unwrap();
        let val = i32::from_le_bytes(attr.raw_value[..4].try_into().unwrap());
        assert_eq!(val, (i + 1) * 100);
    }
}

// ── B-tree v2 chunk index ──

#[test]
fn read_btree_v2_chunked() {
    let file = File::open(fixture("btree_v2_chunks.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("bt2chunked").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![6, 4]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 96); // 24 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..24).collect();
    assert_eq!(values, expected);
}

// ── Hyperslab reads ──

#[test]
fn read_slice_contiguous_1d() {
    let file = File::open(fixture("simple_contiguous_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    // Full dataset is [1.0, 2.0, 3.0, 4.0]; read middle two
    let raw = ds.read_slice(&[1], &[2]).unwrap();
    assert_eq!(raw.len(), 16); // 2 * 8 bytes (f64)

    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![2.0, 3.0]);
}

#[test]
fn read_slice_chunked_2d() {
    // edge_chunks.h5: 7x5 i32 with values 0..34
    let file = File::open(fixture("edge_chunks.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("edge").unwrap();

    // Read a 2x3 sub-region starting at (1, 1)
    let raw = ds.read_slice(&[1, 1], &[2, 3]).unwrap();
    assert_eq!(raw.len(), 24); // 2 * 3 * 4 bytes

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // row 1 cols 1..4: [6, 7, 8], row 2 cols 1..4: [11, 12, 13]
    assert_eq!(values, vec![6, 7, 8, 11, 12, 13]);
}

#[test]
fn read_slice_compact() {
    let file = File::open(fixture("compact_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("small").unwrap();

    // Full dataset is [100, 200, 300, 400]; read last two
    let raw = ds.read_slice(&[2], &[2]).unwrap();
    assert_eq!(raw.len(), 4); // 2 * 2 bytes (i16)

    let values: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![300, 400]);
}

// ── Type conversion (byte order) ──

#[test]
fn read_native_big_endian() {
    let file = File::open(fixture("big_endian.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("be_data").unwrap();

    // read_raw returns big-endian bytes
    let raw = ds.read_raw().unwrap();
    let be_first = i32::from_be_bytes(raw[..4].try_into().unwrap());
    assert_eq!(be_first, 1);

    // read_native should byte-swap to native (little-endian on this platform)
    let native = ds.read_native().unwrap();
    let values: Vec<i32> = native
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1, 256, 65536, -1, 1000000, 0]);
}

// ── Committed (shared) datatypes ──

#[test]
fn read_committed_datatype() {
    let file = File::open(fixture("committed_datatype.h5")).unwrap();
    let root = file.root_group().unwrap();

    // Both datasets share a committed i32 type
    let ds1 = root.dataset("data1").unwrap();
    let dt1 = ds1.datatype().unwrap();
    assert!(
        matches!(dt1, Datatype::FixedPoint { size: 4, .. }),
        "expected i32, got {:?}",
        dt1
    );

    let raw1 = ds1.read_raw().unwrap();
    let vals1: Vec<i32> = raw1
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals1, vec![10, 20, 30, 40, 50]);

    let ds2 = root.dataset("data2").unwrap();
    let raw2 = ds2.read_raw().unwrap();
    let vals2: Vec<i32> = raw2
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals2, vec![100, 200, 300, 400, 500]);
}

// ── Shared datatypes in attributes ──

#[test]
fn read_shared_attribute_type() {
    let file = File::open(fixture("shared_attr.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    // Dataset uses committed type
    let dt = ds.datatype().unwrap();
    assert!(matches!(dt, Datatype::FixedPoint { size: 4, .. }));

    // Attribute also uses the same committed type
    let attrs = ds.attributes().unwrap();
    let scale = attrs
        .iter()
        .find(|a| a.name == "scale")
        .expect("missing 'scale' attr");
    assert!(
        matches!(scale.datatype, Datatype::FixedPoint { size: 4, .. }),
        "expected shared i32 type, got {:?}",
        scale.datatype
    );
    let val = i32::from_le_bytes(scale.raw_value[..4].try_into().unwrap());
    assert_eq!(val, 42);
}

// ── Empty / unallocated chunked datasets ──

#[test]
fn read_empty_chunked() {
    let file = File::open(fixture("empty_chunked.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("empty").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![10]);

    // Should return all zeros (no data written)
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 40); // 10 * 4 bytes
    assert!(
        raw.iter().all(|&b| b == 0),
        "expected all zeros for unallocated dataset"
    );

    // read_slice should also work
    let slice = ds.read_slice(&[2], &[3]).unwrap();
    assert_eq!(slice.len(), 12);
    assert!(slice.iter().all(|&b| b == 0));
}

// ── Extensible array with data blocks ──

#[test]
fn read_ea_large_dataset() {
    let file = File::open(fixture("ea_large.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("large_ea").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![100]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 400); // 100 * 4 bytes

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let expected: Vec<i32> = (0..100).map(|i| i * 10).collect();
    assert_eq!(values, expected);
}

// ── Attribute reading ──

#[test]
fn read_attribute() {
    let file = File::open(fixture("simple_contiguous_v2.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();
    let attrs = ds.attributes().unwrap();

    assert!(!attrs.is_empty(), "expected at least one attribute");
    let units = attrs.iter().find(|a| a.name == "units");
    assert!(units.is_some(), "expected 'units' attribute");
    let units = units.unwrap();
    // The raw value should contain "m/s"
    let val_str = String::from_utf8_lossy(&units.raw_value);
    assert!(
        val_str.starts_with("m/s"),
        "expected 'm/s', got {:?}",
        val_str
    );
}

// ── Creation order (B-tree v2 record types 6 & 9) ──────────────────────────

#[test]
fn read_creation_order_links() {
    let file = hdf5_pure::File::open("tests/fixtures/creation_order.h5").unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("ordered").unwrap();

    // Name index returns in hash order (not necessarily alphabetical)
    let by_name = grp.members().unwrap();
    assert_eq!(by_name.len(), 3);

    // Creation order should be: charlie (0), alpha (1), bravo (2)
    let by_order = grp.members_by_creation_order().unwrap();
    assert_eq!(by_order, vec!["charlie", "alpha", "bravo"]);

    // Verify creation order differs from name-hash order
    assert_ne!(
        by_name, by_order,
        "name order and creation order should differ"
    );
}

#[test]
fn read_creation_order_attributes() {
    let file = hdf5_pure::File::open("tests/fixtures/creation_order.h5").unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("ordered").unwrap();

    // Name index returns in hash order
    let by_name = grp.attributes().unwrap();
    assert_eq!(by_name.len(), 3);

    // Creation order should be: zebra (0), mango (1), apple (2)
    let by_order = grp.attributes_by_creation_order().unwrap();
    let names_by_order: Vec<&str> = by_order.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names_by_order, vec!["zebra", "mango", "apple"]);

    // Verify the values are correct regardless of iteration order
    let zebra = by_order.iter().find(|a| a.name == "zebra").unwrap();
    let val = i32::from_le_bytes([
        zebra.raw_value[0],
        zebra.raw_value[1],
        zebra.raw_value[2],
        zebra.raw_value[3],
    ]);
    assert_eq!(val, 30);

    let mango = by_order.iter().find(|a| a.name == "mango").unwrap();
    let val = i32::from_le_bytes([
        mango.raw_value[0],
        mango.raw_value[1],
        mango.raw_value[2],
        mango.raw_value[3],
    ]);
    assert_eq!(val, 10);
}

// ── Complex type (HDF5 2.0, class 11) ──────────────────────────────────────

#[test]
fn read_complex_dataset() {
    let file = hdf5_pure::File::open("tests/fixtures/complex.h5").unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("complex_data").unwrap();

    // Check datatype: should be Complex with f64 base
    let dt = ds.datatype().unwrap();
    match &dt {
        hdf5_pure::Datatype::Complex { size, base } => {
            assert_eq!(*size, 16); // 2 * 8 bytes
            assert_eq!(base.element_size(), 8);
        }
        other => panic!("expected Complex datatype, got {:?}", other),
    }

    // Check shape
    let shape = ds.shape().unwrap();
    assert_eq!(shape, vec![4]);

    // Read raw data: 4 complex doubles = 8 doubles = 64 bytes
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 64);

    // Parse as pairs of f64 (real, imag)
    let values: Vec<(f64, f64)> = raw
        .chunks_exact(16)
        .map(|c| {
            let re = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            let im = f64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]);
            (re, im)
        })
        .collect();

    assert_eq!(values.len(), 4);
    assert_eq!(values[0], (1.0, 2.0)); // 1+2i
    assert_eq!(values[1], (3.0, 4.0)); // 3+4i
    assert_eq!(values[2], (-1.0, 0.0)); // -1+0i
    assert_eq!(values[3], (0.0, -5.0)); // 0-5i
}

#[test]
fn read_nbit_dataset() {
    let file = hdf5_pure::File::open("tests/fixtures/nbit.h5").unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    let raw = ds.read_raw().unwrap();
    // 8 uint16 elements = 16 bytes
    assert_eq!(raw.len(), 16);

    let values: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    assert_eq!(values, vec![0, 100, 200, 300, 400, 500, 600, 700]);
}

#[test]
fn read_scaleoffset_dataset() {
    let file = hdf5_pure::File::open("tests/fixtures/scaleoffset.h5").unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    let raw = ds.read_raw().unwrap();
    // 8 int32 elements = 32 bytes
    assert_eq!(raw.len(), 32);

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(values, vec![1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]);
}

#[test]
fn read_filtered_fheap_links() {
    let file = hdf5_pure::File::open("tests/fixtures/filtered_fheap.h5").unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("filtered_group").unwrap();

    // Should be able to enumerate members through the filtered fractal heap
    let members = grp.members().unwrap();
    assert_eq!(members.len(), 31); // 30 soft links + 1 dataset

    // Should be able to access the dataset through the group
    let ds = grp.dataset("ds").unwrap();
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 32); // 4 * f64

    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    assert_eq!(values, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn read_filtered_fheap_attributes() {
    let file = hdf5_pure::File::open("tests/fixtures/filtered_fheap.h5").unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("filtered_group").unwrap();

    let attrs = grp.attributes().unwrap();
    assert_eq!(attrs.len(), 2);

    // Find attr_one and attr_two
    let a1 = attrs.iter().find(|a| a.name == "attr_one").unwrap();
    let a2 = attrs.iter().find(|a| a.name == "attr_two").unwrap();

    let v1 = i32::from_le_bytes([
        a1.raw_value[0],
        a1.raw_value[1],
        a1.raw_value[2],
        a1.raw_value[3],
    ]);
    let v2 = i32::from_le_bytes([
        a2.raw_value[0],
        a2.raw_value[1],
        a2.raw_value[2],
        a2.raw_value[3],
    ]);
    assert_eq!(v1, 42);
    assert_eq!(v2, 99);
}

// ── B-tree v2 deep (depth > 0) ──

#[test]
fn read_btree_v2_deep() {
    let file = File::open(fixture("btree_v2_deep.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("deep").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![20, 10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 800); // 200 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..200).collect();
    assert_eq!(values, expected);
}

// ── B-tree v2 filtered chunks ──

#[test]
fn read_btree_v2_filtered() {
    let file = File::open(fixture("btree_v2_filtered.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("filtered").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![6, 4]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 96); // 24 * 4 bytes (i32)

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..24).collect();
    assert_eq!(values, expected);
}

// ── Fractal heap indirect block traversal ──

#[test]
fn read_fheap_indirect_links() {
    let file = File::open(fixture("fheap_indirect.h5")).unwrap();
    let root = file.root_group().unwrap();
    let grp = root.group("many").unwrap();

    let members = grp.members().unwrap();
    // 120 child groups + 1 dataset
    assert_eq!(members.len(), 121);

    // Verify a few child groups exist
    assert!(members.contains(&"child_group_0000_padding".to_string()));
    assert!(members.contains(&"child_group_0059_padding".to_string()));
    assert!(members.contains(&"child_group_0119_padding".to_string()));

    // Verify the dataset is accessible through the group
    let ds = grp.dataset("values").unwrap();
    let raw = ds.read_raw().unwrap();
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![10, 20, 30, 40, 50]);
}

// ── Nil message handling in object headers ──

#[test]
fn read_nil_messages_dataset() {
    let file = File::open(fixture("nil_messages.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    // Read data through an object header that may contain nil padding messages
    let raw = ds.read_raw().unwrap();
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1, 2, 3, 4]);

    // All 12 attributes should be readable
    let attrs = ds.attributes().unwrap();
    assert_eq!(attrs.len(), 12);
    for i in 0..12 {
        let name = format!("attribute_{}", i);
        let attr = attrs.iter().find(|a| a.name == name);
        assert!(attr.is_some(), "missing attribute {}", name);
        let val = i32::from_le_bytes(attr.unwrap().raw_value[..4].try_into().unwrap());
        assert_eq!(val, (i + 1) * 10);
    }
}

// ── Global heap free-space entry handling ──

#[test]
fn read_gcol_free_space_vlen() {
    let file = File::open(fixture("gcol_free_space.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("strings").unwrap();

    // The dataset was overwritten with shorter strings, so the global heap
    // collection may contain free-space entries (index 0) that must be skipped.
    let strings = ds.read_vlen_strings().unwrap();
    assert_eq!(strings, vec!["hi", "there", "hdf5", "test"]);
}

// ── Compound with enum/array members (encoded_size fix) ──

#[test]
fn read_compound_complex_members_metadata() {
    let file = File::open(fixture("compound_complex_members.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("records").unwrap();

    let dt = ds.datatype().unwrap();
    match &dt {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 20); // 4 + 12 + 4
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "color");
            assert_eq!(members[0].byte_offset, 0);
            // "color" member should be an Enum type
            match &members[0].datatype {
                Datatype::Enum {
                    base,
                    members: evals,
                } => {
                    assert_eq!(base.element_size(), 4);
                    assert_eq!(evals.len(), 3);
                }
                other => panic!("expected Enum for 'color', got {:?}", other),
            }
            assert_eq!(members[1].name, "coords");
            assert_eq!(members[1].byte_offset, 4);
            // "coords" member should be an Array type
            match &members[1].datatype {
                Datatype::Array {
                    dimensions,
                    element_type,
                } => {
                    assert_eq!(dimensions, &vec![3]);
                    assert_eq!(element_type.element_size(), 4);
                }
                other => panic!("expected Array for 'coords', got {:?}", other),
            }
            assert_eq!(members[2].name, "id");
            assert_eq!(members[2].byte_offset, 16);
        }
        other => panic!("expected Compound, got {:?}", other),
    }
}

#[test]
fn read_compound_complex_members_data() {
    let file = File::open(fixture("compound_complex_members.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("records").unwrap();
    let raw = ds.read_raw().unwrap();

    assert_eq!(raw.len(), 60); // 3 records * 20 bytes
    // Record 0: color=0, coords=[10,20,30], id=100
    let r = &raw[0..20];
    assert_eq!(i32::from_le_bytes(r[0..4].try_into().unwrap()), 0);
    assert_eq!(i32::from_le_bytes(r[4..8].try_into().unwrap()), 10);
    assert_eq!(i32::from_le_bytes(r[8..12].try_into().unwrap()), 20);
    assert_eq!(i32::from_le_bytes(r[12..16].try_into().unwrap()), 30);
    assert_eq!(i32::from_le_bytes(r[16..20].try_into().unwrap()), 100);
    // Record 2: color=2, coords=[70,80,90], id=300
    let r = &raw[40..60];
    assert_eq!(i32::from_le_bytes(r[0..4].try_into().unwrap()), 2);
    assert_eq!(i32::from_le_bytes(r[4..8].try_into().unwrap()), 70);
    assert_eq!(i32::from_le_bytes(r[16..20].try_into().unwrap()), 300);
}

// ── SWMR (superblock v3 specific) ──

#[test]
fn read_swmr_superblock() {
    let data = std::fs::read(fixture("swmr.h5")).unwrap();
    let sb = hdf5_pure::Superblock::parse(data.as_slice(), 0).unwrap();
    assert_eq!(sb.version, 3);
    assert_eq!(sb.size_of_offsets, 8);
    assert_eq!(sb.size_of_lengths, 8);
    // File was cleanly closed, so SWMR flag should be clear
    assert!(!sb.swmr_write_in_progress());
}

#[test]
fn read_swmr_extended_dataset() {
    let file = File::open(fixture("swmr.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    // Dataset was created with 4 elements, then extended to 8 during SWMR write
    let shape = ds.shape().unwrap();
    assert_eq!(shape, vec![8]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 32); // 8 * 4 bytes

    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![10, 20, 30, 40, 50, 60, 70, 80]);
}

#[test]
fn read_swmr_attribute() {
    let file = File::open(fixture("swmr.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    let attrs = ds.attributes().unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "scale");
    let val = i32::from_le_bytes(attrs[0].raw_value[0..4].try_into().unwrap());
    assert_eq!(val, 100);
}

// ── LZF filter ──

#[test]
fn read_lzf_integers() {
    let file = File::open(fixture("lzf.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("integers").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![100]);
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 100 * 4);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected: Vec<i32> = (0..100).collect();
    assert_eq!(values, expected);
}

#[test]
fn read_lzf_floats_shuffled() {
    let file = File::open(fixture("lzf.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("floats").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![100]);
    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 100 * 8);
    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // np.linspace(0.0, 99.0, 100) produces 0.0, 1.0, 2.0, ..., 99.0
    for (i, v) in values.iter().enumerate() {
        assert!(
            (*v - i as f64).abs() < 1e-10,
            "index {}: expected {}, got {}",
            i,
            i as f64,
            v
        );
    }
}

#[test]
fn read_lzf_compound() {
    let file = File::open(fixture("lzf.h5")).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("compound").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![3]);
    let dt = ds.datatype().unwrap();
    assert!(matches!(dt, Datatype::Compound { .. }));

    let raw = ds.read_raw().unwrap();
    // dtype: f32 x, f32 y, u32 id = 12 bytes per element
    assert_eq!(raw.len(), 3 * 12);

    let expected = [(1.5f32, 2.5f32, 10u32), (3.5, 4.5, 20), (5.5, 6.5, 30)];
    for (i, (ex, ey, eid)) in expected.iter().enumerate() {
        let off = i * 12;
        let x = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
        let y = f32::from_le_bytes(raw[off + 4..off + 8].try_into().unwrap());
        let id = u32::from_le_bytes(raw[off + 8..off + 12].try_into().unwrap());
        assert_eq!(x, *ex, "row {} x", i);
        assert_eq!(y, *ey, "row {} y", i);
        assert_eq!(id, *eid, "row {} id", i);
    }
}
