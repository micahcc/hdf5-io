# hdf5-reader TODO

Pure Rust HDF5 reader targeting superblock v2 and v3 files. WASM-compatible.

## Phase 1: Core Infrastructure
- [x] Crate skeleton, error types, I/O abstraction (`ReadAt` trait)
- [x] Superblock v2/v3 parsing (signature, version, offsets, base addr, root group addr, checksum)
- [x] Checksum validation (Jenkins lookup3 hash)
- [x] `File::open()` / `File::from_bytes()` entry points

## Phase 2: Object Headers
- [x] Object header v2 prefix (magic `OHDR`, version, flags, timestamps, chunk0 size)
- [x] Object header v2 message decoding loop (type, size, flags per message)
- [x] Continuation message (type 0x0010) — follow to next header chunk
- [x] Header chunk parsing (magic `OCHK`, messages, checksum)
- [x] Key message types:
  - [x] Dataspace (0x0001): version 1+2, rank, dimensions, max dimensions
  - [x] Datatype (0x0003): fixed-point, float, string, opaque, bitfield, reference, time
  - [x] Data Layout (0x0008): compact, contiguous, chunked (v3 + v4)
  - [x] Filter Pipeline (0x000B): v1 + v2, all built-in filter IDs
  - [x] Attribute (0x000C): v1/v2/v3, name, datatype, dataspace, value
  - [x] Link (0x0006): hard, soft, external
  - [x] Link Info (0x0002): fractal heap addr, B-tree v2 addrs
  - [x] Object Header Continuation (0x0010)
  - [ ] Fill Value (0x0005): version, space/write times, defined flag, value
  - [ ] Attribute Info (0x0015): dense attribute storage via fractal heap
  - [ ] Group Info (0x000A)
  - [ ] Fill Value Old (0x0004)

## Phase 3: B-tree v2
- [x] B-tree v2 header parsing (magic `BTHD`, version, type, node/record size, depth, root addr)
- [x] Leaf node parsing (magic `BTLF`, records)
- [x] Internal node parsing (magic `BTIN`, records + child pointers)
- [x] Tree iteration (walk all records in sorted order)
- [x] Record type 5: Link Name for indexed groups (hash + heap ID)
- [ ] Record type 6: Creation Order for indexed groups
- [ ] Record type 8: Attribute Name for indexed attributes
- [ ] Record type 9: Attribute Creation Order
- [ ] Record types 10/11: Chunked dataset indexing

## Phase 4: Fractal Heap
- [x] Fractal heap header parsing (magic `FRHP`, all fields + doubling table info)
- [x] Direct block reading (magic `FHDB`)
- [x] Indirect block traversal (magic `FHIB`)
- [x] Managed object lookup from heap ID
- [x] Tiny object decoding (inline in heap ID)
- [x] Row/column calculations for block addressing
- [ ] Huge object reading (B-tree v2 lookup)
- [ ] Filtered direct block handling

## Phase 5: Datatype Decoding (full)
- [x] Fixed-point (integer): byte order, sign, bit offset, precision
- [x] Floating-point: byte order, mantissa/exponent layout, bias
- [x] String: padding type, character set
- [x] Opaque: tag string
- [x] Bitfield: byte order, bit offset, precision
- [x] Reference: object / dataset region
- [x] Time: bit precision
- [ ] Compound: member names, offsets, types (recursive) — stubbed
- [ ] Enumeration: base type, member names, values
- [ ] Array: dimensions, element type
- [ ] Variable-length: element type, padding, character set
- [ ] Complex (HDF5 2.0): base floating-point type

## Phase 6: Data Reading
- [x] Contiguous layout: read raw bytes from file offset
- [x] Compact layout: read raw bytes from object header
- [ ] Chunked layout with B-tree v1 index
- [ ] Chunked layout with B-tree v2 index
- [ ] Chunked layout with extensible array index
- [ ] Chunked layout with fixed array index
- [ ] Single chunk optimization
- [ ] Filter pipeline application on chunked read
- [ ] Hyperslab / partial reads
- [ ] Type conversion on read (endian swap, widening)

## Phase 7: Filter Pipeline
- [x] Pipeline framework (ordered filter application)
- [x] Deflate (zlib) decompression via `flate2`
- [x] Shuffle (byte de-interleaving)
- [x] Fletcher32 checksum verification
- [ ] N-bit filter
- [ ] Scale-offset filter
- [ ] SZIP decompression (Rice coding)

## Phase 8: Navigation / Public API
- [x] `File::open()` — from filesystem path
- [x] `File::from_bytes()` — from in-memory buffer
- [x] `File::root_group()` — navigate to root group
- [x] `Group::members()` — list all child link names
- [x] `Group::find_link()` / `Group::group()` / `Group::dataset()`
- [x] `Dataset::datatype()`, `Dataset::dataspace()`, `Dataset::shape()`
- [x] `Dataset::read_raw()` — entire dataset as bytes (contiguous + compact)
- [x] `Dataset::attributes()` / `Group::attributes()`
- [x] `File::open_path("/group1/subgroup/dataset")`
- [ ] `Dataset::read_slice()` — read hyperslab

## Phase 9: WASM & Portability
- [x] `ReadAt` impl for `&[u8]` / `Vec<u8>` (in-memory buffer)
- [x] `ReadAt` impl for `std::fs::File` (native)
- [ ] Async `ReadAt` variant for HTTP range requests
- [ ] Build and test on `wasm32-unknown-unknown`
- [ ] Build and test on `wasm32-wasip1`

## Phase 10: Testing & Validation
- [x] Fixture generator (C program using libhdf5)
- [x] Fixture files: superblock v2 + v3, contiguous, compact, nested groups, attributes
- [x] Integration tests: superblock, metadata, contiguous read, compact read, navigation, path lookup, attributes
- [ ] Fixture files for chunked + compressed datasets
- [ ] Fixture files for compound, enum, array, vlen types
- [ ] Large file (>4GB) fixture
- [ ] SWMR file fixture (superblock v3 specific)
- [ ] Fuzz testing on malformed inputs
