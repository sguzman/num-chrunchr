use anyhow::{Context, Result};
use num_bigint::BigUint;
use num_traits::Zero;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// RAM-backed big integer representation.
#[derive(Clone, Debug)]
pub struct BigIntRam {
    value: BigUint,
}

impl BigIntRam {
    pub fn from_decimal_stream(path: &Path, buffer_size: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = BufReader::with_capacity(buffer_size, file);
        let mut buffer = vec![0u8; buffer_size];
        let mut value = BigUint::zero();
        let ten = BigUint::from(10u8);
        let mut seen_digit = false;

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            for &b in &buffer[..read] {
                if !b.is_ascii_digit() {
                    continue;
                }
                seen_digit = true;
                let digit = BigUint::from((b - b'0') as u8);
                value = &value * &ten + digit;
            }
        }

        if !seen_digit {
            value = BigUint::zero();
        }

        Ok(Self { value })
    }

    pub fn as_biguint(&self) -> &BigUint {
        &self.value
    }

    pub fn to_decimal_string(&self) -> String {
        self.value.to_str_radix(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_decimal_stream_into_biguint() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "12345").unwrap();
        let path = file.path();
        let bigint = BigIntRam::from_decimal_stream(path, 8).unwrap();
        assert_eq!(bigint.to_decimal_string(), "12345");
    }
}
