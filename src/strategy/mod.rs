use crate::{config::Config, repr::DecimalStream, source::NumberSource};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::{info, warn};

/// Run the small-factor peeling strategy using streaming mod/div.
pub fn run_peel(
    source: &NumberSource,
    config: &Config,
    primes_limit_override: Option<usize>,
    reset: bool,
) -> Result<()> {
    let strategy = &config.strategy;
    let report_dir = strategy.report_directory.clone();
    fs::create_dir_all(&report_dir)
        .with_context(|| format!("create report directory {}", report_dir.display()))?;

    let factors_path = report_dir.join("factors.json");
    let sketch_path = report_dir.join("sketch.json");
    let cofactor_path = report_dir.join("cofactor.txt");
    let primes_limit = primes_limit_override.unwrap_or(strategy.primes_limit);

    if primes_limit < 2 {
        bail!("strategy primes_limit must be >= 2");
    }

    let mut report = if !reset && factors_path.exists() {
        let loaded = FactorReport::load(&factors_path)?;
        if loaded.input_label != source.label() {
            warn!(
                existing = %loaded.input_label,
                requested = %source.label(),
                "input changed from previous run; resetting state"
            );
            fs::remove_file(&factors_path).ok();
            if cofactor_path.exists() {
                fs::remove_file(&cofactor_path).ok();
            }
            source.copy_to(&cofactor_path)?;
            FactorReport::new(source.label())
        } else {
            if !cofactor_path.exists() {
                source.copy_to(&cofactor_path)?;
            }
            loaded
        }
    } else {
        if factors_path.exists() {
            fs::remove_file(&factors_path).ok();
        }
        source.copy_to(&cofactor_path)?;
        FactorReport::new(source.label())
    };

    let primes = sieve_primes(primes_limit);
    let max_divisions = config.policies.max_divisions;
    let mut division_count = 0usize;

    'outer: for prime in primes {
        if division_count >= max_divisions {
            warn!(limit = max_divisions, "division budget reached");
            break;
        }
        loop {
            let remainder = {
                let stream = DecimalStream::open(&cofactor_path, config.stream.buffer_size)?;
                stream.mod_u32(prime)?
            };
            if remainder != 0 {
                break;
            }

            let temp_path = report_dir.join("cofactor.tmp");
            let stream = DecimalStream::open(&cofactor_path, config.stream.buffer_size)?;
            stream.div_u32_to_path(prime, &temp_path)?;
            fs::rename(&temp_path, &cofactor_path).with_context(|| {
                format!(
                    "move {} -> {}",
                    temp_path.display(),
                    cofactor_path.display()
                )
            })?;

            report.record(prime);
            report.save(&factors_path)?;
            division_count += 1;
            info!(
                prime,
                exponent = report.exponent(prime),
                divisions = division_count,
                "peeled a small factor"
            );

            if division_count >= max_divisions {
                warn!(limit = max_divisions, "division budget reached");
                break 'outer;
            }
        }
    }

    let sketch = build_sketch(
        &cofactor_path,
        config.stream.buffer_size,
        &strategy.sketch_primes,
    )?;
    sketch.save(&sketch_path)?;
    info!(
        report = %factors_path.display(),
        sketch = %sketch_path.display(),
        "peel finished"
    );

    Ok(())
}

fn sieve_primes(limit: usize) -> Vec<u32> {
    if limit < 2 {
        return Vec::new();
    }
    let mut sieve = vec![true; limit + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut primes = Vec::new();
    for num in 2..=limit {
        if sieve[num] {
            primes.push(num as u32);
            let step = num;
            if num <= limit / step {
                let mut multiple = num * num;
                while multiple <= limit {
                    sieve[multiple] = false;
                    multiple += step;
                }
            }
        }
    }
    primes
}

fn build_sketch(path: &Path, buffer_size: usize, primes: &[u64]) -> Result<Sketch> {
    let mut residues = Vec::new();
    for &prime in primes {
        let stream = DecimalStream::open(path, buffer_size)?;
        let remainder = stream.mod_u64(prime)?;
        residues.push(Residue { prime, remainder });
    }
    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(0);
    Ok(Sketch {
        generated_at,
        residues,
    })
}

#[derive(Serialize, Deserialize, Default)]
struct FactorReport {
    input_label: String,
    factors: Vec<FactorEntry>,
}

impl FactorReport {
    fn new(input_label: String) -> Self {
        Self {
            input_label,
            factors: Vec::new(),
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        serde_json::from_reader(file).with_context(|| format!("parse {}", path.display()))
    }

    fn save(&self, path: &Path) -> Result<()> {
        let file = fs::File::create(path).with_context(|| format!("write {}", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("serialize {}", path.display()))
    }

    fn record(&mut self, prime: u32) {
        if let Some(entry) = self.factors.iter_mut().find(|e| e.prime == prime) {
            entry.exponent += 1;
        } else {
            self.factors.push(FactorEntry { prime, exponent: 1 });
        }
    }

    fn exponent(&self, prime: u32) -> u32 {
        self.factors
            .iter()
            .find(|e| e.prime == prime)
            .map(|e| e.exponent)
            .unwrap_or(0)
    }
}

#[derive(Serialize, Deserialize)]
struct FactorEntry {
    prime: u32,
    exponent: u32,
}

#[derive(Serialize, Deserialize)]
struct Sketch {
    generated_at: u64,
    residues: Vec<Residue>,
}

impl Sketch {
    fn save(&self, path: &Path) -> Result<()> {
        let file = fs::File::create(path).with_context(|| format!("write {}", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("serialize {}", path.display()))
    }
}

#[derive(Serialize, Deserialize)]
struct Residue {
    prime: u64,
    remainder: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::source::NumberSource;
    use std::fs::read_to_string;
    use tempfile::tempdir;

    #[test]
    fn peel_persists_factors_and_sketch() {
        let reports = tempdir().unwrap();
        let mut cfg = Config::default();
        cfg.strategy.report_directory = reports.path().join("reports");
        cfg.strategy.primes_limit = 20;
        cfg.strategy.sketch_primes = vec![2, 3, 5];
        cfg.stream.buffer_size = 16;
        cfg.policies.max_divisions = 10;

        let source = NumberSource::Inline("2310".into());
        run_peel(&source, &cfg, None, true).unwrap();

        let factors_path = cfg.strategy.report_directory.join("factors.json");
        let sketch_path = cfg.strategy.report_directory.join("sketch.json");
        let report_text = read_to_string(factors_path).unwrap();
        assert!(report_text.contains("\"prime\": 2"));
        assert!(report_text.contains("\"prime\": 3"));
        assert!(report_text.contains("\"prime\": 5"));

        let sketch_text = read_to_string(sketch_path).unwrap();
        assert!(sketch_text.contains("\"prime\": 2"));
    }
}
