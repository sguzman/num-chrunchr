use crate::config::StreamConfig;
mod bigint_ram;
use anyhow::{Context, Result};
pub use bigint_ram::BigIntRam;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Disk-backed decimal representation. Buffer size is configurable.
#[derive(Clone, Debug)]
pub struct DecimalStream {
    path: PathBuf,
    buffer_size: usize,
}

impl DecimalStream {
    pub fn from_config(path: &Path, config: &StreamConfig) -> Result<Self> {
        Self::open(path, config.buffer_size)
    }

    pub fn open(path: &Path, buffer_size: usize) -> Result<Self> {
        let meta = std::fs::metadata(path)
            .with_context(|| format!("failed to stat input {}", path.display()))?;
        info!(input = %path.display(), bytes = meta.len(), buffer_size, "opened decimal stream");
        Ok(Self {
            path: path.to_path_buf(),
            buffer_size,
        })
    }

    pub fn decimal_len(&self) -> Result<u64> {
        let mut reader = self.open_reader()?;
        let mut buf = vec![0u8; self.buffer_size];
        let mut count: u64 = 0;

        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            for &b in &buf[..read] {
                if b.is_ascii_digit() {
                    count = count.saturating_add(1);
                }
            }
        }

        debug!(digits = count, "decimal_len computed");
        Ok(count)
    }

    pub fn leading_digits(&self, k: usize) -> Result<String> {
        let mut reader = self.open_reader()?;
        let mut buffer = vec![0u8; self.buffer_size];
        let mut out = String::with_capacity(k);

        while out.len() < k {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            for &b in &buffer[..read] {
                if b.is_ascii_digit() {
                    out.push(b as char);
                    if out.len() >= k {
                        break;
                    }
                }
            }
        }

        if out.is_empty() {
            warn!("leading_digits: no digits found; returning empty string");
        } else {
            debug!(leading = %out, "leading digits read");
        }

        Ok(out)
    }

    pub fn mod_u32(&self, p: u32) -> Result<u32> {
        anyhow::ensure!(p != 0, "modulus p must be nonzero");
        let mut reader = self.open_reader()?;
        let mut buf = vec![0u8; self.buffer_size];
        let mut remainder: u32 = 0;
        let mut seen_digit = false;

        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            for &b in &buf[..read] {
                if b.is_ascii_digit() {
                    seen_digit = true;
                    let digit = (b - b'0') as u32;
                    let tmp = (remainder as u64) * 10 + digit as u64;
                    remainder = (tmp % (p as u64)) as u32;
                }
            }
        }

        if !seen_digit {
            warn!("mod_u32: no digits found; treating input as 0");
        }
        debug!(p, remainder, "mod_u32 completed");
        Ok(remainder)
    }

    pub fn mod_u64(&self, p: u64) -> Result<u64> {
        anyhow::ensure!(p != 0, "modulus p must be nonzero");
        let mut reader = self.open_reader()?;
        let mut buf = vec![0u8; self.buffer_size];
        let mut remainder: u64 = 0;
        let mut seen_digit = false;

        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            for &b in &buf[..read] {
                if b.is_ascii_digit() {
                    seen_digit = true;
                    let digit = (b - b'0') as u64;
                    remainder = (remainder.wrapping_mul(10).wrapping_add(digit)) % p;
                }
            }
        }

        if !seen_digit {
            warn!("mod_u64: no digits found; treating input as 0");
        }
        debug!(p, remainder, "mod_u64 completed");
        Ok(remainder)
    }

    pub fn div_u32_to_path(&self, d: u32, out_path: &Path) -> Result<(PathBuf, u32)> {
        anyhow::ensure!(d != 0, "divisor d must be nonzero");
        let mut reader = self.open_reader()?;
        let mut buf = vec![0u8; self.buffer_size];
        let mut writer = std::io::BufWriter::new(
            File::create(out_path).with_context(|| format!("create {}", out_path.display()))?,
        );

        let mut remainder: u64 = 0;
        let mut started = false;
        let mut saw_digit = false;

        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            for &b in &buf[..read] {
                if !b.is_ascii_digit() {
                    continue;
                }
                saw_digit = true;
                let digit = (b - b'0') as u64;
                let current = remainder * 10 + digit;
                let q = (current / d as u64) as u8;
                remainder = current % d as u64;
                if q != 0 || started {
                    started = true;
                    writer.write_all(&[b'0' + q])?;
                }
            }
        }

        writer.flush()?;

        if !saw_digit {
            warn!("div_u32: no digits found; quotient=0");
            std::fs::write(out_path, b"0")?;
            return Ok((out_path.to_path_buf(), 0));
        }

        if !started {
            std::fs::write(out_path, b"0")?;
        }

        let rem_u32 = remainder as u32;
        info!(divisor = d, quotient = %out_path.display(), remainder = rem_u32, "division complete");
        Ok((out_path.to_path_buf(), rem_u32))
    }

    fn open_reader(&self) -> Result<BufReader<File>> {
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))?;
        Ok(BufReader::new(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn write_digits(data: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        write!(file, "{data}").expect("write digits");
        file
    }

    #[test]
    fn decimal_len_counts_digits() {
        let txt = "123 45\n6";
        let file = write_digits(txt);
        let stream = DecimalStream::open(file.path(), 16).unwrap();
        assert_eq!(stream.decimal_len().unwrap(), 6);
    }

    #[test]
    fn leading_digits_filters_non_digits() {
        let txt = "00 12a345";
        let file = write_digits(txt);
        let stream = DecimalStream::open(file.path(), 8).unwrap();
        assert_eq!(stream.leading_digits(4).unwrap(), "0012");
    }

    #[test]
    fn mod_u32_works() {
        let file = write_digits("12345\n");
        let stream = DecimalStream::open(file.path(), 8).unwrap();
        assert_eq!(stream.mod_u32(97).unwrap(), 12345 % 97);
    }

    #[test]
    fn mod_u64_works() {
        let file = write_digits("123456789012345\n");
        let stream = DecimalStream::open(file.path(), 16).unwrap();
        assert_eq!(
            stream.mod_u64(1_000_003).unwrap(),
            123456789012345 % 1_000_003
        );
    }

    #[test]
    fn div_u32_to_path_writes_quotient() {
        let tmp_dir = TempDir::new().unwrap();
        let input = write_digits("10086");
        let stream = DecimalStream::open(input.path(), 8).unwrap();
        let out_path = tmp_dir.path().join("quotient.txt");
        let (path, rem) = stream.div_u32_to_path(2, &out_path).unwrap();
        assert_eq!(rem, 0);
        let contents = fs::read_to_string(path).unwrap();
        assert_eq!(contents, "5043");
    }

    #[test]
    fn div_u32_handles_zero_quotient() {
        let tmp_dir = TempDir::new().unwrap();
        let input = write_digits("3");
        let stream = DecimalStream::open(input.path(), 8).unwrap();
        let out_path = tmp_dir.path().join("quotient.txt");
        let (path, rem) = stream.div_u32_to_path(5, &out_path).unwrap();
        assert_eq!(rem, 3);
        assert_eq!(fs::read_to_string(path).unwrap(), "0");
    }
}
