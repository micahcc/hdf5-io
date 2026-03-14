use std::io;

/// Positional read trait — the core I/O abstraction for hdf5-reader.
///
/// All file access goes through this trait so we can support:
/// - `std::fs::File` (native)
/// - `&[u8]` / `Vec<u8>` (in-memory, WASM)
/// - Custom implementations (HTTP range requests, cloud storage)
pub trait ReadAt {
    /// Read exactly `buf.len()` bytes starting at `offset`.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Total size of the underlying data, if known.
    fn size(&self) -> io::Result<u64>;
}

// -- In-memory buffer implementation (WASM-friendly) --

impl ReadAt for [u8] {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = offset as usize;
        let end = start + buf.len();
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_exact_at: offset {} + len {} = {} exceeds buffer size {}",
                    offset,
                    buf.len(),
                    end,
                    self.len()
                ),
            ));
        }
        buf.copy_from_slice(&self[start..end]);
        Ok(())
    }

    fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}

impl ReadAt for Vec<u8> {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.as_slice().read_exact_at(offset, buf)
    }

    fn size(&self) -> io::Result<u64> {
        Ok(self.len() as u64)
    }
}

// -- std::fs::File implementation (native) --

impl ReadAt for std::fs::File {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            FileExt::read_exact_at(self, buf, offset)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut pos = 0;
            while pos < buf.len() {
                let n = self.seek_read(&mut buf[pos..], offset + pos as u64)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "unexpected EOF",
                    ));
                }
                pos += n;
            }
            Ok(())
        }
        #[cfg(not(any(unix, windows)))]
        {
            // Fallback: not available on this platform
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "positional read not supported on this platform",
            ))
        }
    }

    fn size(&self) -> io::Result<u64> {
        self.metadata().map(|m| m.len())
    }
}

/// Helper to read from any `ReadAt` into a new Vec of a given size.
pub fn read_bytes<R: ReadAt + ?Sized>(r: &R, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    r.read_exact_at(offset, &mut buf)?;
    Ok(buf)
}

/// Little-endian integer reading helpers.
pub struct Le;

impl Le {
    pub fn read_u8<R: ReadAt + ?Sized>(r: &R, offset: u64) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        r.read_exact_at(offset, &mut buf)?;
        Ok(buf[0])
    }

    pub fn read_u16<R: ReadAt + ?Sized>(r: &R, offset: u64) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        r.read_exact_at(offset, &mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    pub fn read_u32<R: ReadAt + ?Sized>(r: &R, offset: u64) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        r.read_exact_at(offset, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn read_u64<R: ReadAt + ?Sized>(r: &R, offset: u64) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        r.read_exact_at(offset, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a "size of offsets"-byte unsigned integer (4 or 8 bytes in practice).
    pub fn read_offset<R: ReadAt + ?Sized>(
        r: &R,
        offset: u64,
        size_of_offsets: u8,
    ) -> io::Result<u64> {
        match size_of_offsets {
            4 => Le::read_u32(r, offset).map(|v| v as u64),
            8 => Le::read_u64(r, offset),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported size_of_offsets: {}", size_of_offsets),
            )),
        }
    }

    /// Read a "size of lengths"-byte unsigned integer (4 or 8 bytes in practice).
    pub fn read_length<R: ReadAt + ?Sized>(
        r: &R,
        offset: u64,
        size_of_lengths: u8,
    ) -> io::Result<u64> {
        match size_of_lengths {
            4 => Le::read_u32(r, offset).map(|v| v as u64),
            8 => Le::read_u64(r, offset),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported size_of_lengths: {}", size_of_lengths),
            )),
        }
    }
}
