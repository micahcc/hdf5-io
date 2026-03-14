/// Jenkins lookup3 hashlittle2 — the checksum algorithm used by HDF5.
///
/// HDF5 uses Bob Jenkins' lookup3 `hashlittle2` function for all checksums
/// in superblock v2/v3, object headers, B-tree v2 nodes, and fractal heap blocks.
///
/// Reference: H5checksum.c in the HDF5 C library.

/// Compute the HDF5 checksum (Jenkins lookup3 hashlittle2) over `data`.
///
/// Returns the hash value that HDF5 stores as the checksum field.
/// HDF5 calls `H5_checksum_lookup3(data, len, 0)` which initializes
/// both hash values to 0 and returns the first one.
pub fn lookup3(data: &[u8]) -> u32 {
    let (h1, _h2) = hashlittle2(data, 0, 0);
    h1
}

/// Jenkins lookup3 hashlittle2.
///
/// Translated from the C implementation in H5checksum.c / lookup3.c.
/// Processes the byte array in 12-byte blocks, then handles the tail.
fn hashlittle2(data: &[u8], pc: u32, pb: u32) -> (u32, u32) {
    let len = data.len();
    // Initial value: 0xdeadbeef + len + pc
    let mut a: u32 = 0xdeadbeef_u32
        .wrapping_add(len as u32)
        .wrapping_add(pc);
    let mut b: u32 = a;
    let mut c: u32 = a.wrapping_add(pb);

    let mut offset = 0usize;
    let mut remaining = len;

    // Process 12-byte blocks
    while remaining > 12 {
        a = a.wrapping_add(read_u32_le(data, offset));
        b = b.wrapping_add(read_u32_le(data, offset + 4));
        c = c.wrapping_add(read_u32_le(data, offset + 8));

        // mix
        a = a.wrapping_sub(c);
        a ^= c.rotate_left(4);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(6);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(8);
        b = b.wrapping_add(a);
        a = a.wrapping_sub(c);
        a ^= c.rotate_left(16);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(19);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(4);
        b = b.wrapping_add(a);

        offset += 12;
        remaining -= 12;
    }

    // Handle the last (possibly incomplete) block
    // We read bytes individually to avoid out-of-bounds reads
    if remaining > 0 {
        // Zero-pad a 12-byte block and fill from remaining bytes
        let mut tail = [0u8; 12];
        tail[..remaining].copy_from_slice(&data[offset..offset + remaining]);

        // The lookup3 tail handling adds bytes to a/b/c depending on count.
        // This follows the "case" fallthrough in the C switch statement.
        match remaining {
            12 => {
                c = c.wrapping_add((tail[11] as u32) << 24);
                c = c.wrapping_add((tail[10] as u32) << 16);
                c = c.wrapping_add((tail[9] as u32) << 8);
                c = c.wrapping_add(tail[8] as u32);
                b = b.wrapping_add((tail[7] as u32) << 24);
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            11 => {
                c = c.wrapping_add((tail[10] as u32) << 16);
                c = c.wrapping_add((tail[9] as u32) << 8);
                c = c.wrapping_add(tail[8] as u32);
                b = b.wrapping_add((tail[7] as u32) << 24);
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            10 => {
                c = c.wrapping_add((tail[9] as u32) << 8);
                c = c.wrapping_add(tail[8] as u32);
                b = b.wrapping_add((tail[7] as u32) << 24);
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            9 => {
                c = c.wrapping_add(tail[8] as u32);
                b = b.wrapping_add((tail[7] as u32) << 24);
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            8 => {
                b = b.wrapping_add((tail[7] as u32) << 24);
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            7 => {
                b = b.wrapping_add((tail[6] as u32) << 16);
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            6 => {
                b = b.wrapping_add((tail[5] as u32) << 8);
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            5 => {
                b = b.wrapping_add(tail[4] as u32);
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            4 => {
                a = a.wrapping_add((tail[3] as u32) << 24);
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            3 => {
                a = a.wrapping_add((tail[2] as u32) << 16);
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            2 => {
                a = a.wrapping_add((tail[1] as u32) << 8);
                a = a.wrapping_add(tail[0] as u32);
            }
            1 => {
                a = a.wrapping_add(tail[0] as u32);
            }
            _ => return (c, b), // remaining == 0, no final mix
        }

        // final mix
        c ^= b;
        c = c.wrapping_sub(b.rotate_left(14));
        a ^= c;
        a = a.wrapping_sub(c.rotate_left(11));
        b ^= a;
        b = b.wrapping_sub(a.rotate_left(25));
        c ^= b;
        c = c.wrapping_sub(b.rotate_left(16));
        a ^= c;
        a = a.wrapping_sub(c.rotate_left(4));
        b ^= a;
        b = b.wrapping_sub(a.rotate_left(14));
        c ^= b;
        c = c.wrapping_sub(b.rotate_left(24));
    }

    (c, b)
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_data() {
        // Known value: lookup3("", 0) with initval=0
        let h = lookup3(&[]);
        // Jenkins lookup3 of empty string with initval=0 => 0xdeadbeef
        assert_eq!(h, 0xdeadbeef);
    }

    #[test]
    fn known_short_value() {
        // Verify deterministic output for a small input.
        let h1 = lookup3(b"hello");
        let h2 = lookup3(b"hello");
        assert_eq!(h1, h2);
        // Must not be the empty-string hash
        assert_ne!(h1, 0xdeadbeef);
    }

    #[test]
    fn different_inputs_differ() {
        let h1 = lookup3(b"hello");
        let h2 = lookup3(b"world");
        assert_ne!(h1, h2);
    }
}
