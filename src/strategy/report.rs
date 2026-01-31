use crate::{
    config::Config,
    repr::{BigIntRam, DecimalStream},
};
use anyhow::{Context, Result};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const LOG2_10: f64 = 3.321928094887362;

#[derive(Debug, Serialize, Deserialize)]
pub struct StructureReport {
    pub generated_at: u64,
    pub decimal_len: u64,
    pub bit_estimate: f64,
    pub leading_digits: String,
    pub near_square: Option<NearSquareInfo>,
    pub near_power: Option<NearPowerInfo>,
    pub sparse_terms: Option<Vec<SparseTerm>>,
    pub special_forms: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NearSquareInfo {
    pub base: String,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NearPowerInfo {
    pub base: String,
    pub exponent: u32,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SparseTerm {
    pub digit: String,
    pub exponent: usize,
}

impl StructureReport {
    pub fn build(path: &Path, config: &Config) -> Result<Self> {
        let stream = DecimalStream::open(path, config.stream.buffer_size)?;
        let decimal_len = stream.decimal_len()?;
        let leading_digits = stream.leading_digits(config.analysis.leading_digits)?;
        let bit_estimate = decimal_len as f64 * LOG2_10;
        let mut near_square = None;
        let mut near_power = None;
        let mut sparse_terms = None;
        let mut special_forms = Vec::new();

        if decimal_len as usize <= config.policies.bigint_upgrade_digits {
            let bigint = BigIntRam::from_decimal_stream(path, config.stream.buffer_size)?;
            if let Some(info) = detect_near_square(bigint.as_biguint()) {
                special_forms.push(format!("near-square delta={}", info.delta));
                near_square = Some(info);
            }
            if let Some(info) = detect_near_power(bigint.as_biguint()) {
                special_forms.push(format!(
                    "near power base={} exp={}",
                    info.base, info.exponent
                ));
                near_power = Some(info);
            }
            if decimal_len <= 64 {
                sparse_terms = Some(build_sparse_terms(bigint.as_biguint(), 256));
            }
        }

        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|dur| dur.as_secs())
            .unwrap_or(0);

        Ok(Self {
            generated_at,
            decimal_len,
            bit_estimate,
            leading_digits,
            near_square,
            near_power,
            sparse_terms,
            special_forms,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file = fs::File::create(path).with_context(|| format!("write {}", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("serialize {}", path.display()))
    }
}

fn detect_near_square(value: &BigUint) -> Option<NearSquareInfo> {
    if value.is_zero() {
        return None;
    }
    let sqrt = bigint_floor_sqrt(value);
    let square = &sqrt * &sqrt;
    let delta = if value >= &square {
        value - &square
    } else {
        square - value
    };
    if delta.is_zero() {
        return Some(NearSquareInfo {
            base: sqrt.to_string(),
            delta: delta.to_string(),
        });
    }
    if delta.bits() * 12 < value.bits() {
        Some(NearSquareInfo {
            base: sqrt.to_string(),
            delta: delta.to_string(),
        })
    } else {
        None
    }
}

fn detect_near_power(value: &BigUint) -> Option<NearPowerInfo> {
    if value <= &BigUint::from(2u8) {
        return None;
    }
    for exponent in 2u32..=6 {
        let base = nth_root(value, exponent);
        if base <= BigUint::from(1u8) {
            continue;
        }
        let power = base.pow(exponent);
        let delta = if value >= &power {
            value - &power
        } else {
            power - value
        };
        if delta.is_zero() || (delta.bits() * 12 < value.bits()) {
            return Some(NearPowerInfo {
                base: base.to_string(),
                exponent,
                delta: delta.to_string(),
            });
        }
    }
    None
}

fn build_sparse_terms(value: &BigUint, limit: usize) -> Vec<SparseTerm> {
    let ten = BigUint::from(10u8);
    let mut terms = Vec::new();
    let mut exponent = 0usize;
    let mut current = value.clone();
    while !current.is_zero() && exponent < limit {
        let digit = (&current % &ten).to_u8().unwrap_or(0);
        if digit != 0 {
            terms.push(SparseTerm {
                digit: digit.to_string(),
                exponent,
            });
        }
        current /= &ten;
        exponent += 1;
    }
    terms.reverse();
    terms
}

fn bigint_floor_sqrt(n: &BigUint) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    let mut low = BigUint::zero();
    let mut high = BigUint::one() << ((n.bits() + 1) / 2 + 1);
    while &low < &high {
        let mid = (&low + &high + BigUint::one()) >> 1;
        let mid_sq = &mid * &mid;
        if &mid_sq > n {
            high = mid - BigUint::one();
        } else {
            low = mid;
        }
    }
    low
}

fn nth_root(n: &BigUint, k: u32) -> BigUint {
    if n.is_zero() {
        return BigUint::zero();
    }
    let mut low = BigUint::zero();
    let mut high = BigUint::one() << ((n.bits() + k as u64 - 1) / k as u64 + 1);
    while &low < &high {
        let sum = &low + &high + BigUint::one();
        let mid: BigUint = sum >> 1;
        let mid_pow = mid.pow(k);
        if &mid_pow > n {
            high = mid - BigUint::one();
        } else {
            low = mid;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn structure_report_detects_near_square() {
        let dir = tempdir().unwrap();
        let cofactor = dir.path().join("cofactor.txt");
        let mut file = File::create(&cofactor).unwrap();
        write!(file, "121").unwrap();
        let mut cfg = Config::default();
        cfg.policies.bigint_upgrade_digits = 10;
        let report = StructureReport::build(&cofactor, &cfg).unwrap();
        assert!(report.near_square.is_some());
        assert!(
            report
                .special_forms
                .iter()
                .any(|entry| entry.contains("near-square"))
        );
        assert!(report.sparse_terms.is_some());
    }

    #[test]
    fn structure_report_detects_near_power() {
        let dir = tempdir().unwrap();
        let cofactor = dir.path().join("cofactor.txt");
        let mut file = File::create(&cofactor).unwrap();
        write!(file, "243").unwrap();
        let mut cfg = Config::default();
        cfg.policies.bigint_upgrade_digits = 10;
        let report = StructureReport::build(&cofactor, &cfg).unwrap();
        assert!(report.near_power.is_some());
        assert!(
            report
                .special_forms
                .iter()
                .any(|entry| entry.contains("near power"))
        );
    }
}
