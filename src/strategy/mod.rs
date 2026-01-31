use crate::{
    config::Config, gpu::batch_mod::BatchModEngine, repr::DecimalStream, source::NumberSource,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::{debug, info, warn};

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

    let max_divisions = config.policies.max_divisions;
    let batch_size = strategy.batch_size.max(1);

    info!(
        input = %source.label(),
        primes_limit = primes_limit,
        batch_size = batch_size,
        use_gpu = strategy.use_gpu,
        max_divisions = max_divisions,
        report_directory = %report_dir.display(),
        "starting peel strategy"
    );

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
    let mut division_count = 0usize;

    'peel: loop {
        for chunk in primes.chunks(batch_size) {
            if division_count >= max_divisions {
                warn!(limit = max_divisions, "division budget reached");
                break 'peel;
            }

            let remainders =
                compute_chunk_remainders(chunk, config, &cofactor_path).with_context(|| {
                    format!(
                        "compute batch remainders for primes {}..{}",
                        chunk.first().unwrap_or(&0),
                        chunk.last().unwrap_or(&0)
                    )
                })?;

            for (idx, &prime) in chunk.iter().enumerate() {
                if remainders.get(idx).copied().unwrap_or(u32::MAX) != 0 {
                    continue;
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
                    break 'peel;
                }

                continue 'peel;
            }
        }
        break;
    }

    info!(divisions = division_count, "peeling loop completed");

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

fn compute_chunk_remainders(chunk: &[u32], config: &Config, path: &Path) -> Result<Vec<u32>> {
    let chunk_start = chunk.first().copied().unwrap_or(0);
    let chunk_end = chunk.last().copied().unwrap_or(0);
    let chunk_size = chunk.len();
    let mut engine = BatchModEngine::try_new(chunk, config.strategy.use_gpu)?;
    let used_gpu_engine = matches!(&engine, BatchModEngine::Gpu(_));
    info!(
        chunk_start,
        chunk_end,
        chunk_size,
        use_gpu_engine = used_gpu_engine,
        "initialized batch remainder engine"
    );
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(config.stream.buffer_size, file);
    let mut buffer = vec![0u8; config.stream.buffer_size];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let digits: Vec<u32> = buffer[..read]
            .iter()
            .filter(|&&b| b.is_ascii_digit())
            .map(|&b| (b - b'0') as u32)
            .collect();
        if !digits.is_empty() {
            engine.update(&digits)?;
        }
    }

    let remainders = engine.remainders()?;
    info!(
        chunk_start,
        chunk_end,
        chunk_size,
        use_gpu_engine = used_gpu_engine,
        "computed chunk remainders"
    );
    debug!(chunk_start, chunk_end, remainders = ?remainders, "chunk remainders detail");
    Ok(remainders)
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
