use anyhow::{Context, Result};
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use tracing::warn;

const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub logging: LoggingConfig,
    pub stream: StreamConfig,
    pub analysis: AnalysisConfig,
    pub policies: PoliciesConfig,
    pub strategy: StrategyConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            logging: LoggingConfig::default(),
            stream: StreamConfig::default(),
            analysis: AnalysisConfig::default(),
            policies: PoliciesConfig::default(),
            strategy: StrategyConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StrategyConfig {
    pub primes_limit: usize,
    pub report_directory: PathBuf,
    pub sketch_primes: Vec<u64>,
    pub batch_size: usize,
    pub use_gpu: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            primes_limit: 10_000,
            report_directory: PathBuf::from("reports"),
            sketch_primes: vec![2, 3, 5, 7, 11, 13, 17, 19],
            batch_size: 256,
            use_gpu: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: Option<String>,
    pub time_format: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Some("info".into()),
            time_format: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StreamConfig {
    pub buffer_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: DEFAULT_BUFFER_SIZE,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    pub leading_digits: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self { leading_digits: 32 }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PoliciesConfig {
    pub max_modulus: Option<u64>,
    pub max_divisor: Option<u32>,
    pub max_divisions: usize,
}

impl Default for PoliciesConfig {
    fn default() -> Self {
        Self {
            max_modulus: None,
            max_divisor: None,
            max_divisions: 64,
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let config_path = if let Some(p) = path {
            p.to_path_buf()
        } else {
            let cwd = env::current_dir().context("determine current directory")?;
            cwd.join("config").join("default.toml")
        };

        if !config_path.exists() {
            warn!(path = ?config_path, "config file not found; using defaults");
            return Ok(Self::default());
        }

        let raw = fs::read_to_string(&config_path)
            .with_context(|| format!("read config {}", config_path.display()))?;
        let config: Config = toml::from_str(&raw)
            .with_context(|| format!("parse config {}", config_path.display()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn loads_defaults_when_missing() {
        let dir = tempdir().unwrap();
        let missing_path = dir.path().join("missing.toml");
        let cfg = Config::load(Some(&missing_path)).unwrap();
        assert_eq!(cfg.analysis.leading_digits, 32);
        assert_eq!(cfg.stream.buffer_size, DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn loads_values_from_file() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("custom.toml");
        let mut file = File::create(&config_path).unwrap();
        writeln!(
            file,
            r#"
            [logging]
            level = "debug"
            [stream]
            buffer_size = 1234
            [analysis]
            leading_digits = 81
            [policies]
            max_modulus = 9999
            max_divisor = 7
            max_divisions = 3
            [strategy]
            primes_limit = 30
            report_directory = "reports/artifacts"
            sketch_primes = [2,3,5]
            batch_size = 512
            use_gpu = true
        "#,
        )
        .unwrap();
        let cfg = Config::load(Some(&config_path)).unwrap();
        assert_eq!(cfg.logging.level.as_deref(), Some("debug"));
        assert_eq!(cfg.stream.buffer_size, 1234);
        assert_eq!(cfg.analysis.leading_digits, 81);
        assert_eq!(cfg.policies.max_modulus, Some(9999));
        assert_eq!(cfg.policies.max_divisor, Some(7));
        assert_eq!(cfg.policies.max_divisions, 3);
        assert_eq!(cfg.strategy.primes_limit, 30);
        assert_eq!(
            cfg.strategy.report_directory,
            PathBuf::from("reports/artifacts")
        );
        assert_eq!(cfg.strategy.sketch_primes, vec![2, 3, 5]);
        assert_eq!(cfg.strategy.batch_size, 512);
        assert!(cfg.strategy.use_gpu);
    }
}
