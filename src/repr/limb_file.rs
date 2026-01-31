use super::BigIntRam;
use anyhow::{Context, Result};
use num_bigint::BigUint;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Disk-backed base-2^32 limbs.
#[derive(Clone, Debug)]
pub struct LimbFile {
    path: PathBuf,
    limb_count: u64,
}

impl LimbFile {
    pub fn create_from_decimal_stream(path: &Path, buffer_size: usize, out: &Path) -> Result<Self> {
        let bigint = BigIntRam::from_decimal_stream(path, buffer_size)?;
        Self::create_from_biguint(bigint.as_biguint(), out)
    }

    pub fn create_from_biguint(value: &BigUint, out: &Path) -> Result<Self> {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create directory {}", parent.display()))?;
        }
        let limbs = value.to_u32_digits();
        let mut writer = BufWriter::new(
            File::create(out).with_context(|| format!("create limb file {}", out.display()))?,
        );
        let count = limbs.len() as u64;
        writer.write_all(&count.to_le_bytes())?;
        for &limb in &limbs {
            writer.write_all(&limb.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(Self {
            path: out.to_path_buf(),
            limb_count: count,
        })
    }

    #[allow(dead_code)]
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("open limb file {}", path.display()))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;
        let limb_count = u64::from_le_bytes(header);
        let meta = file.metadata()?;
        let expected = 8 + (limb_count * 4);
        if meta.len() < expected {
            anyhow::bail!("limb file {} truncated", path.display());
        }
        Ok(Self {
            path: path.to_path_buf(),
            limb_count,
        })
    }

    pub fn reader(&self) -> Result<LimbFileReader> {
        let mut file = File::open(&self.path)
            .with_context(|| format!("open limb file {}", self.path.display()))?;
        file.seek(SeekFrom::Start(8))?;
        Ok(LimbFileReader {
            file,
            remaining: self.limb_count,
        })
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn limb_count(&self) -> u64 {
        self.limb_count
    }
}

pub struct LimbFileReader {
    file: File,
    remaining: u64,
}

impl LimbFileReader {
    pub fn next_chunk(&mut self, max_limbs: usize) -> Result<Option<Vec<u32>>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        if max_limbs == 0 {
            return Ok(Some(Vec::new()));
        }
        let requested = self.remaining.min(max_limbs as u64);
        let byte_count = (requested * 4) as usize;
        let mut buffer = vec![0u8; byte_count];
        self.file
            .read_exact(&mut buffer)
            .with_context(|| "read limb chunk")?;
        let limbs = buffer
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        self.remaining -= requested;
        Ok(Some(limbs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn writes_and_reads_back_limbs() {
        let value = BigUint::from(0x1_0000_0001u64);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let file = LimbFile::create_from_biguint(&value, &path).unwrap();
        assert_eq!(file.limb_count(), 2);
        let mut reader = file.reader().unwrap();
        let chunk = reader.next_chunk(1).unwrap().unwrap();
        assert_eq!(chunk, vec![1]);
        let chunk = reader.next_chunk(10).unwrap().unwrap();
        assert_eq!(chunk, vec![1]);
        assert!(reader.next_chunk(1).unwrap().is_none());
    }

    #[test]
    fn roundtrip_from_decimal_stream() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "4294967296").unwrap();
        let src = file.path();
        let tempfile = NamedTempFile::new().unwrap();
        let out = tempfile.path().to_path_buf();
        let limb = LimbFile::create_from_decimal_stream(src, 8, &out).unwrap();
        assert_eq!(limb.limb_count(), 2);
        let mut reader = limb.reader().unwrap();
        let first = reader.next_chunk(2).unwrap().unwrap();
        assert_eq!(first, vec![0, 1]);
    }
}
