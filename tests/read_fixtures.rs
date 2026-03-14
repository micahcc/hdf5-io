use hdf5_reader::{Datatype, File};
use std::path::PathBuf;

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
    let sb = hdf5_reader::Superblock::parse(data.as_slice(), 0).unwrap();
    assert_eq!(sb.version, 2);
    assert_eq!(sb.size_of_offsets, 8);
    assert_eq!(sb.size_of_lengths, 8);
    assert_eq!(sb.base_address, 0);
    assert!(!sb.swmr_write_in_progress());
}

#[test]
fn parse_superblock_v3() {
    let data = std::fs::read(fixture("chunked_deflate_v3.h5")).unwrap();
    let sb = hdf5_reader::Superblock::parse(data.as_slice(), 0).unwrap();
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
        hdf5_reader::Node::Dataset(ds) => {
            assert_eq!(ds.shape().unwrap(), vec![3]);
        }
        _ => panic!("expected Dataset"),
    }

    match file.open_path("/group1").unwrap() {
        hdf5_reader::Node::Group(_) => {}
        _ => panic!("expected Group"),
    }
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
