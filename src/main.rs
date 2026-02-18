mod config;
mod gpu;
mod repr;
mod source;
mod strategy;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser, Subcommand};
use config::{Config, LoggingConfig};
use repr::DecimalStream;
use source::NumberSource;
use std::{io::Write, path::PathBuf, time::Instant};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "num-chrunchr")]
#[command(about = "Streaming factoring + number structure toolkit", long_about = None)]
#[command(group(ArgGroup::new("source").required(true).args(["input", "number"])))]
struct Cli {
    /// Path to configuration TOML (defaults to config/default.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Decimal file containing digits of the number
    #[arg(long, group = "source")]
    input: Option<PathBuf>,

    /// Inline decimal string to analyze
    #[arg(long, group = "source")]
    number: Option<String>,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compute N mod p using streaming Horner
    Mod {
        #[arg(long)]
        p: u64,
    },
    /// Divide N by a small divisor and emit quotient digits
    Div {
        #[arg(long)]
        d: u32,
        #[arg(long)]
        out: PathBuf,
    },
    /// Analyze the number (decimal length + leading digits)
    Analyze {
        #[arg(long)]
        leading: Option<usize>,
    },
    /// Scan an inclusive divisor range and print factors found in that range
    RangeFactors {
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
    },
    /// Peel small factors using streaming trial division (resumes with reports/)
    Peel {
        #[arg(long)]
        primes_limit: Option<usize>,
        #[arg(long)]
        reset: bool,
    },
}

impl Cli {
    fn number_source(&self) -> Result<NumberSource> {
        match (&self.input, &self.number) {
            (Some(path), None) => Ok(NumberSource::File(path.clone())),
            (None, Some(text)) => Ok(NumberSource::Inline(text.clone())),
            (Some(_), Some(_)) => bail!("cannot use --input and --number together"),
            (None, None) => bail!("provide --input or --number"),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let source = cli.number_source()?;
    let config = Config::load(cli.config.as_deref())?;
    init_logging(&config.logging)?;
    info!(
        buffer_size = config.stream.buffer_size,
        leading_digits = config.analysis.leading_digits,
        "configuration instantiated"
    );

    let input_label = source.label();
    info!(input = %input_label, command = ?cli.cmd, "processing request");
    let start = Instant::now();

    match cli.cmd {
        Command::Mod { p } => {
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_mod(&stream, p, &config.policies)?;
        }
        Command::Div { d, out } => {
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_div(&stream, d, &out, &config.policies)?;
        }
        Command::Analyze { leading } => {
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_analyze(&stream, leading, &config.analysis)?;
        }
        Command::RangeFactors { start, end } => {
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_range_factors(&stream, start, end, &config.policies)?;
        }
        Command::Peel {
            primes_limit,
            reset,
        } => {
            strategy::run_peel(&source, &config, primes_limit, reset)?;
        }
    }

    let duration = start.elapsed();
    info!(duration_ms = duration.as_secs_f64() * 1000.0, duration = ?duration, "command complete");
    Ok(())
}

fn init_logging(logging: &LoggingConfig) -> Result<()> {
    let level = logging.level.clone().unwrap_or_else(|| "info".into());
    let filter = std::env::var("RUST_LOG").unwrap_or(level);
    let env_filter = EnvFilter::try_from(filter).unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    Ok(())
}

fn run_mod(stream: &DecimalStream, modulus: u64, policies: &config::PoliciesConfig) -> Result<()> {
    if let Some(limit) = policies.max_modulus {
        if modulus > limit {
            warn!(modulus, limit, "modulus exceeds configured policy limit");
        }
    }
    let remainder = stream.mod_u64(modulus)?;
    info!(modulus, remainder, "streaming modulus complete");
    println!("{remainder}");
    Ok(())
}

fn run_div(
    stream: &DecimalStream,
    divisor: u32,
    out: &PathBuf,
    policies: &config::PoliciesConfig,
) -> Result<()> {
    if let Some(limit) = policies.max_divisor {
        if divisor > limit {
            warn!(divisor, limit, "divisor exceeds configured policy limit");
        }
    }

    let (_path, remainder) = stream.div_u32_to_path(divisor, out)?;
    println!("remainder={remainder}");
    Ok(())
}

fn run_analyze(
    stream: &DecimalStream,
    leading: Option<usize>,
    analysis: &config::AnalysisConfig,
) -> Result<()> {
    let leading_digits = leading.unwrap_or(analysis.leading_digits);
    let len = stream.decimal_len()?;
    let lead = stream.leading_digits(leading_digits)?;
    info!(decimal_len = len, leading_digits = %lead, "analysis complete");
    println!("decimal_len={len}");
    println!("leading_digits={lead}");
    Ok(())
}

fn run_range_factors(
    stream: &DecimalStream,
    start: u64,
    end: u64,
    policies: &config::PoliciesConfig,
) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "[")?;
    let mut first = true;

    let count = scan_factor_range(stream, start, end, policies, |factor| {
        if !first {
            write!(stdout, ",")?;
        }
        first = false;
        write!(stdout, "{factor}")?;
        Ok(())
    })?;

    writeln!(stdout, "]")?;
    info!(
        start,
        end,
        factors_found = count,
        "range factor scan complete"
    );
    Ok(())
}

fn scan_factor_range<F>(
    stream: &DecimalStream,
    start: u64,
    end: u64,
    policies: &config::PoliciesConfig,
    mut on_factor: F,
) -> Result<u64>
where
    F: FnMut(u64) -> Result<()>,
{
    if start == 0 || end == 0 {
        bail!("range bounds must be positive integers");
    }
    if start > end {
        bail!("range start must be <= end");
    }
    if let Some(limit) = policies.max_modulus {
        if end > limit {
            warn!(end, limit, "range end exceeds configured policy limit");
        }
    }

    let mut found = 0u64;
    for divisor in start..=end {
        if stream.mod_u64(divisor)? == 0 {
            on_factor(divisor)?;
            found += 1;
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::NumberSource;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn cli_parses_inline_number_source() {
        let cli = Cli::parse_from(["num-chrunchr", "--number", "3141592653589793", "analyze"]);

        let source = cli.number_source().expect("number source should parse");
        match source {
            NumberSource::Inline(text) => {
                assert_eq!(text, "3141592653589793");
            }
            NumberSource::File(path) => {
                panic!("expected inline source, got file: {}", path.display());
            }
        }
    }

    #[test]
    fn cli_parses_range_factors_command() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "range-factors",
            "--start",
            "2",
            "--end",
            "12",
        ]);
        match cli.cmd {
            Command::RangeFactors { start, end } => {
                assert_eq!(start, 2);
                assert_eq!(end, 12);
            }
            _ => panic!("expected range-factors command"),
        }
    }

    #[test]
    fn scan_factor_range_finds_expected_divisors() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "360").unwrap();
        let stream = DecimalStream::open(file.path(), 16).unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(&stream, 2, 12, &cfg.policies, |factor| {
            seen.push(factor);
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 9);
        assert_eq!(seen, vec![2, 3, 4, 5, 6, 8, 9, 10, 12]);
    }

    #[test]
    fn scan_factor_range_rejects_zero_bound() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "10").unwrap();
        let stream = DecimalStream::open(file.path(), 16).unwrap();
        let cfg = Config::default();
        let err = scan_factor_range(&stream, 0, 10, &cfg.policies, |_| Ok(())).unwrap_err();
        assert!(err.to_string().contains("positive integers"));
    }
}
