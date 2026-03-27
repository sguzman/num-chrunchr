mod config;
mod gpu;
mod repr;
mod source;
mod strategy;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser, Subcommand};
use config::{Config, LoggingConfig};
use gpu::batch_mod::{BatchModEngine, SymbolOrder};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use repr::{BigIntRam, DecimalStream};
use source::NumberSource;
use std::{
    cmp::Ordering,
    fs,
    fs::File,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "num-chrunchr")]
#[command(about = "Streaming factoring + number structure toolkit", long_about = None)]
#[command(group(ArgGroup::new("source").args(["input", "number"])))]
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

    /// Read --input as a raw binary integer (default is decimal digits)
    #[arg(long, default_value_t = false)]
    binary: bool,

    /// Interpret raw binary bytes as little-endian (default big-endian)
    #[arg(long, default_value_t = false)]
    little_endian: bool,

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
    /// Return the decimal digit count of the input number
    Digits,
    /// Compute log_base(N), optionally only returning the integer part
    Log {
        #[arg(long)]
        base: f64,
        /// Return only floor(log_base(N))
        #[arg(long, default_value_t = false)]
        integer_part: bool,
    },
    /// Compute N^exponent and emit decimal output
    Pow {
        #[arg(long)]
        exponent: u32,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Find the exponent k where base^k is closest to N
    #[command(group(ArgGroup::new("base_source").args(["base_input", "base_number"])))]
    NearPower {
        /// Decimal file containing digits of the base
        #[arg(long)]
        base_input: Option<PathBuf>,
        /// Inline decimal string for the base
        #[arg(long)]
        base_number: Option<String>,
        /// Read --base-input as a raw binary integer (default is decimal digits)
        #[arg(long, default_value_t = false)]
        base_binary: bool,
        /// Interpret raw binary base bytes as little-endian (default big-endian)
        #[arg(long, default_value_t = false)]
        base_little_endian: bool,
        /// Repeat near-power on the remaining delta N times (stop early if exact)
        #[arg(long, default_value_t = 1)]
        n_times: u32,
        /// Compress the exponent sequence with base+delta representation
        #[arg(long, default_value_t = false, conflicts_with = "compress_seq_b")]
        compress_seq_a: bool,
        /// Compress the exponent sequence with delta sequence representation
        #[arg(long, default_value_t = false, conflicts_with = "compress_seq_a")]
        compress_seq_b: bool,
        /// Compression scheme to optimize deltas
        #[arg(long, value_enum, default_value_t = CompressionScheme::MinTotalAbs)]
        compress_scheme: CompressionScheme,
        /// Disallow overshooting the target (force power <= target)
        #[arg(long, default_value_t = false)]
        no_overshoot: bool,
        /// Use prime numbers as bases for each round (2,3,5,7,11,...)
        #[arg(long, default_value_t = false)]
        prime_rounds: bool,
    },
    /// Convert raw binary input bytes to decimal text output
    WriteDecimal {
        #[arg(long)]
        out: PathBuf,
    },
    /// Estimate time/probability to find any factor under a scan budget
    EstimateAnyFactor {
        /// Number of decimal digits (use instead of --input/--number)
        #[arg(long)]
        digits: Option<u64>,
        /// Divisor scan cutoff used for the estimate
        #[arg(long)]
        max_divisor: Option<u64>,
        /// Tested divisor throughput (divisors/sec)
        #[arg(long)]
        factors_per_sec: Option<f64>,
        /// Estimation mode for success assumptions
        #[arg(long, value_enum, default_value_t = EstimateMode::Unknown)]
        mode: EstimateMode,
    },
    /// Scan an inclusive divisor range and print factors found in that range
    RangeFactors {
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        /// Return repeated factors when higher powers of a divisor also divide N
        #[arg(long, default_value_t = false)]
        all: bool,
        /// Stop after finding the first factor in the scan range
        #[arg(long, default_value_t = false)]
        first: bool,
        /// Favor lower latency for --first by reducing GPU auto-batch size
        #[arg(long, default_value_t = false)]
        low_latency_first: bool,
        /// Return only the largest factor in the scan range
        #[arg(long, default_value_t = false)]
        last: bool,
        /// Stop after this many matching factors have been emitted
        #[arg(long)]
        limit: Option<u64>,
        /// Use the batch remainder GPU engine for faster range scans
        #[arg(long, default_value_t = false)]
        use_gpu: bool,
        /// Override GPU divisor batch size for range scans (0 = auto-tune)
        #[arg(long)]
        gpu_batch_size: Option<usize>,
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
    fn number_source(&self) -> Result<Option<NumberSource>> {
        match (&self.input, &self.number) {
            (Some(path), None) => Ok(Some(NumberSource::File(path.clone()))),
            (None, Some(text)) => Ok(Some(NumberSource::Inline(text.clone()))),
            (Some(_), Some(_)) => bail!("cannot use --input and --number together"),
            (None, None) => Ok(None),
        }
    }

    fn encoding(&self, source: Option<&NumberSource>) -> Result<NumberEncoding> {
        if self.little_endian && !self.binary {
            bail!("--little-endian is only valid with --binary");
        }
        if self.binary {
            match source {
                Some(NumberSource::File(_)) => {}
                Some(NumberSource::Inline(_)) => {
                    bail!("--binary requires --input with a file path");
                }
                None => {
                    bail!("--binary requires --input with a file path");
                }
            }
            if self.little_endian {
                Ok(NumberEncoding::BinaryLittleEndian)
            } else {
                Ok(NumberEncoding::BinaryBigEndian)
            }
        } else {
            Ok(NumberEncoding::Decimal)
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let source = cli.number_source()?;
    let encoding = cli.encoding(source.as_ref())?;
    let config = Config::load(cli.config.as_deref())?;
    init_logging(&config.logging)?;
    info!(
        buffer_size = config.stream.buffer_size,
        leading_digits = config.analysis.leading_digits,
        "configuration instantiated"
    );

    let input_label = source
        .as_ref()
        .map(NumberSource::label)
        .unwrap_or_else(|| "none".to_string());
    info!(input = %input_label, command = ?cli.cmd, "processing request");
    let start = Instant::now();

    match cli.cmd {
        Command::Mod { p } => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            run_mod_from_file(
                &prepared.path,
                config.stream.buffer_size,
                p,
                encoding,
                &config.policies,
            )?;
        }
        Command::Div { d, out } => {
            if encoding != NumberEncoding::Decimal {
                bail!("div command currently supports decimal input only");
            }
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_div(&stream, d, &out, &config.policies)?;
        }
        Command::Analyze { leading } => {
            if encoding != NumberEncoding::Decimal {
                bail!("analyze command currently supports decimal input only");
            }
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_analyze(&stream, leading, &config.analysis)?;
        }
        Command::Digits => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let digits = input_decimal_digits(&prepared.path, config.stream.buffer_size, encoding)?;
            println!("{digits}");
            info!(digits, encoding = ?encoding, "digit-count complete");
        }
        Command::Log { base, integer_part } => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let value =
                load_biguint_from_input(&prepared.path, config.stream.buffer_size, encoding)?;
            let log_value = log_biguint_base(&value, base)?;
            if integer_part {
                let int_part = log_value.floor() as i128;
                println!("{int_part}");
                info!(
                    base,
                    integer_part = int_part,
                    encoding = ?encoding,
                    "log command complete"
                );
            } else {
                println!("{log_value:.12}");
                info!(base, log_value, encoding = ?encoding, "log command complete");
            }
        }
        Command::Pow { exponent, out } => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let value =
                load_biguint_from_input(&prepared.path, config.stream.buffer_size, encoding)?;
            let result = value.pow(exponent);
            let decimal = result.to_str_radix(10);
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)
                            .with_context(|| format!("failed to create {}", parent.display()))?;
                    }
                }
                fs::write(&path, decimal.as_bytes())
                    .with_context(|| format!("failed to write {}", path.display()))?;
                info!(
                    exponent,
                    output = %path.display(),
                    digits = decimal.len(),
                    "pow command complete"
                );
            } else {
                println!("{decimal}");
                info!(exponent, digits = decimal.len(), "pow command complete");
            }
        }
        Command::NearPower {
            base_input,
            base_number,
            base_binary,
            base_little_endian,
            n_times,
            compress_seq_a,
            compress_seq_b,
            compress_scheme,
            no_overshoot,
            prime_rounds,
        } => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            let value =
                load_biguint_from_input(&prepared.path, config.stream.buffer_size, encoding)?;
            if prime_rounds {
                if base_input.is_some() || base_number.is_some() {
                    bail!("--prime-rounds cannot be combined with --base-input/--base-number");
                }
                if base_binary || base_little_endian {
                    bail!("--prime-rounds cannot be combined with base binary flags");
                }
            } else if base_input.is_none() && base_number.is_none() {
                bail!("provide --base-input or --base-number");
            }
            if n_times == 0 {
                bail!("--n-times must be >= 1");
            }
            let iter_result = if prime_rounds {
                let primes = first_n_primes(n_times as usize);
                let bases: Vec<BigUint> = primes.iter().map(|&p| BigUint::from(p)).collect();
                run_near_power_iterations_with_bases(&bases, &value, no_overshoot)?
            } else {
                let base = load_base_biguint(
                    base_input.as_ref(),
                    base_number.as_deref(),
                    config.stream.buffer_size,
                    base_binary,
                    base_little_endian,
                )?;
                run_near_power_iterations(&base, &value, n_times, no_overshoot)?
            };
            let exponent_list = iter_result
                .exponents
                .iter()
                .map(|exp| exp.to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!("{exponent_list}"); // keep stdout simple but informative
            if compress_seq_a || compress_seq_b {
                let compression = if compress_seq_a {
                    compress_sequence_a(&iter_result.exponents, compress_scheme)
                } else {
                    compress_sequence_b(&iter_result.exponents, compress_scheme)
                };
                let mode_label = if compress_seq_a { "seqA" } else { "seqB" };
                let delta_list = format_i128_list(&compression.deltas);
                println!(
                    "compress_mode={mode_label} scheme={:?} base={} deltas=[{}] max_abs_delta={} total_abs_delta={} total_digit_count={}",
                    compression.scheme,
                    compression.base,
                    delta_list,
                    compression.max_abs_delta,
                    compression.total_abs_delta,
                    compression.total_digit_count
                );
                info!(
                    compress_mode = mode_label,
                    compress_scheme = ?compression.scheme,
                    compress_base = compression.base,
                    compress_max_abs_delta = compression.max_abs_delta,
                    compress_total_abs_delta = compression.total_abs_delta,
                    compress_total_digit_count = compression.total_digit_count,
                    "near-power compression complete"
                );
            }
            let coverage_percent = coverage_percent_string(&value, &iter_result.final_delta);
            let percent_delta = percent_delta_string(&value, &iter_result.final_delta);
            let exact_coverage = iter_result.final_delta.is_zero();
            let total_delta_base10_digits = biguint_decimal_digits(&iter_result.total_delta);
            let total_delta_base2_digits = iter_result.total_delta.bits().max(1);
            let final_delta_base10_digits = biguint_decimal_digits(&iter_result.final_delta);
            let final_delta_base2_digits = iter_result.final_delta.bits().max(1);
            let base_mode = if prime_rounds { "prime_rounds" } else { "fixed_base" };
            let base_bits = if prime_rounds { None } else { Some(iter_result.base_bits) };
            info!(
                base_bits = ?base_bits,
                base_mode,
                base_sequence = ?iter_result.base_sequence,
                value_bits = value.bits(),
                base_binary,
                base_little_endian,
                iterations = iter_result.iterations,
                total_exponents_checked = iter_result.total_exponents_checked,
                total_comparisons = iter_result.total_comparisons,
                fast_path_used_count = iter_result.fast_path_used_count,
                total_shift_bits = iter_result.total_shift_bits,
                total_power_bits = iter_result.total_power.bits(),
                total_delta_bits = iter_result.total_delta.bits(),
                total_delta_base10_digits,
                total_delta_base2_digits,
                final_delta_bits = iter_result.final_delta.bits(),
                final_delta_base10_digits,
                final_delta_base2_digits,
                coverage_percent = %coverage_percent,
                percent_delta = %percent_delta,
                exact_coverage,
                "near-power aggregate complete"
            );
        }
        Command::WriteDecimal { out } => {
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            run_write_decimal_from_binary(&prepared.path, &out, encoding)?;
        }
        Command::EstimateAnyFactor {
            digits,
            max_divisor,
            factors_per_sec,
            mode,
        } => {
            let prepared = source.as_ref().map(NumberSource::prepare).transpose()?;
            let path = prepared.as_ref().map(|p| p.path.as_path());
            run_estimate_any_factor(
                path,
                config.stream.buffer_size,
                encoding,
                digits,
                max_divisor,
                factors_per_sec,
                mode,
            )?;
        }
        Command::RangeFactors {
            start,
            end,
            all,
            first,
            low_latency_first,
            last,
            limit,
            use_gpu,
            gpu_batch_size,
        } => {
            if first && last {
                bail!("--first and --last cannot be combined");
            }
            if all && (first || last) {
                bail!("--all cannot be combined with --first or --last");
            }
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            let prepared = source.prepare()?;
            run_range_factors(
                &prepared.path,
                config.stream.buffer_size,
                start,
                end,
                all,
                first,
                low_latency_first,
                last,
                limit,
                use_gpu,
                gpu_batch_size.unwrap_or(config.range_factors.gpu_batch_size),
                encoding,
                &config.policies,
            )?;
        }
        Command::Peel {
            primes_limit,
            reset,
        } => {
            if encoding != NumberEncoding::Decimal {
                bail!("peel command currently supports decimal input only");
            }
            let source = source
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("provide --input or --number"))?;
            strategy::run_peel(source, &config, primes_limit, reset)?;
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

fn run_mod_from_file(
    input_path: &Path,
    buffer_size: usize,
    modulus: u64,
    encoding: NumberEncoding,
    policies: &config::PoliciesConfig,
) -> Result<()> {
    if let Some(limit) = policies.max_modulus {
        if modulus > limit {
            warn!(modulus, limit, "modulus exceeds configured policy limit");
        }
    }
    let remainder = mod_from_file(input_path, buffer_size, modulus, encoding)?;
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

fn run_write_decimal_from_binary(
    input_path: &Path,
    out: &Path,
    encoding: NumberEncoding,
) -> Result<()> {
    let bytes = fs::read(input_path)
        .with_context(|| format!("failed to read binary input {}", input_path.display()))?;
    let value = match encoding {
        NumberEncoding::BinaryBigEndian => BigUint::from_bytes_be(&bytes),
        NumberEncoding::BinaryLittleEndian => BigUint::from_bytes_le(&bytes),
        NumberEncoding::Decimal => {
            bail!("write-decimal requires --binary input (and optional --little-endian)")
        }
    };
    let decimal = value.to_str_radix(10);
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    fs::write(out, decimal.as_bytes())
        .with_context(|| format!("failed to write decimal output {}", out.display()))?;
    info!(
        input = %input_path.display(),
        output = %out.display(),
        bytes = bytes.len(),
        decimal_digits = decimal.len(),
        "write-decimal complete"
    );
    Ok(())
}

fn load_biguint_from_input(
    path: &Path,
    buffer_size: usize,
    encoding: NumberEncoding,
) -> Result<BigUint> {
    match encoding {
        NumberEncoding::Decimal => {
            let bigint = BigIntRam::from_decimal_stream(path, buffer_size)?;
            Ok(bigint.as_biguint().clone())
        }
        NumberEncoding::BinaryBigEndian => {
            let bytes = fs::read(path)
                .with_context(|| format!("failed to read binary input {}", path.display()))?;
            Ok(BigUint::from_bytes_be(&bytes))
        }
        NumberEncoding::BinaryLittleEndian => {
            let bytes = fs::read(path)
                .with_context(|| format!("failed to read binary input {}", path.display()))?;
            Ok(BigUint::from_bytes_le(&bytes))
        }
    }
}

fn input_decimal_digits(path: &Path, buffer_size: usize, encoding: NumberEncoding) -> Result<u64> {
    match encoding {
        NumberEncoding::Decimal => {
            let stream = DecimalStream::open(path, buffer_size)?;
            Ok(stream.decimal_len()?)
        }
        NumberEncoding::BinaryBigEndian | NumberEncoding::BinaryLittleEndian => {
            binary_decimal_digits(path)
        }
    }
}

fn log_biguint_base(value: &BigUint, base: f64) -> Result<f64> {
    if value == &BigUint::from(0u8) {
        bail!("log undefined for zero");
    }
    if !base.is_finite() || base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
        bail!("base must be finite, > 0, and not equal to 1");
    }

    let bit_len = value.bits();
    let shift = bit_len.saturating_sub(53);
    let top = if shift > 0 {
        (value >> shift)
            .to_u64()
            .ok_or_else(|| anyhow::anyhow!("failed to normalize input"))? as f64
    } else {
        value
            .to_u64()
            .ok_or_else(|| anyhow::anyhow!("failed to normalize input"))? as f64
    };
    let ln_n = top.ln() + (shift as f64) * std::f64::consts::LN_2;
    Ok(ln_n / base.ln())
}

struct NearPowerResult {
    exponent: u32,
    power: BigUint,
    delta: BigUint,
    exponents_checked: u64,
    comparisons: u64,
    fast_path: bool,
    power_of_two_m: Option<u32>,
    k_floor: Option<u32>,
    shift_bits_used: u64,
}

struct NearPowerIterations {
    exponents: Vec<u32>,
    iterations: u32,
    total_power: BigUint,
    total_delta: BigUint,
    total_exponents_checked: u64,
    total_comparisons: u64,
    fast_path_used_count: u64,
    total_shift_bits: u64,
    final_delta: BigUint,
    base_bits: u64,
    base_sequence: Option<Vec<u64>>,
}

struct CompressionResult {
    scheme: CompressionScheme,
    base: i128,
    deltas: Vec<i128>,
    max_abs_delta: i128,
    total_abs_delta: i128,
    total_digit_count: u64,
}

fn load_base_biguint(
    base_input: Option<&PathBuf>,
    base_number: Option<&str>,
    buffer_size: usize,
    base_binary: bool,
    base_little_endian: bool,
) -> Result<BigUint> {
    if base_little_endian && !base_binary {
        bail!("--base-little-endian is only valid with --base-binary");
    }
    match (base_input, base_number) {
        (Some(path), None) => {
            let encoding = if base_binary {
                if base_little_endian {
                    NumberEncoding::BinaryLittleEndian
                } else {
                    NumberEncoding::BinaryBigEndian
                }
            } else {
                NumberEncoding::Decimal
            };
            load_biguint_from_input(path, buffer_size, encoding)
                .with_context(|| format!("failed to load base from {}", path.display()))
        }
        (None, Some(text)) => {
            if base_binary {
                bail!("--base-binary requires --base-input");
            }
            parse_decimal_biguint(text)
        }
        (Some(_), Some(_)) => bail!("use either --base-input or --base-number"),
        (None, None) => bail!("provide --base-input or --base-number"),
    }
}

fn parse_decimal_biguint(text: &str) -> Result<BigUint> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("base number cannot be empty");
    }
    if !trimmed.bytes().all(|b| b.is_ascii_digit()) {
        bail!("base number must be decimal digits only");
    }
    BigUint::parse_bytes(trimmed.as_bytes(), 10)
        .ok_or_else(|| anyhow::anyhow!("failed to parse base number"))
}

fn power_of_two_base_exponent(base: &BigUint) -> Option<u32> {
    if base.is_zero() {
        return None;
    }
    let one = BigUint::from(1u8);
    if base == &one {
        return None;
    }
    let base_bits = base.bits();
    if base_bits == 0 {
        return None;
    }
    let base_minus_one = base - &one;
    if (base & base_minus_one).is_zero() {
        let exp = base_bits.saturating_sub(1);
        if exp <= u32::MAX as u64 {
            return Some(exp as u32);
        }
    }
    None
}

fn first_n_primes(n: usize) -> Vec<u64> {
    let mut primes = Vec::with_capacity(n);
    if n == 0 {
        return primes;
    }
    let mut candidate = 2u64;
    while primes.len() < n {
        if is_prime_with_list(candidate, &primes) {
            primes.push(candidate);
        }
        candidate = if candidate == 2 { 3 } else { candidate + 2 };
    }
    primes
}

fn is_prime_with_list(candidate: u64, primes: &[u64]) -> bool {
    if candidate < 2 {
        return false;
    }
    for &p in primes {
        if p * p > candidate {
            break;
        }
        if candidate % p == 0 {
            return false;
        }
    }
    true
}

fn nearest_power_exponent(
    base: &BigUint,
    value: &BigUint,
    no_overshoot: bool,
) -> Result<NearPowerResult> {
    let two = BigUint::from(2u8);
    if base < &two {
        bail!("base must be >= 2");
    }

    if value.is_zero() {
        let power = BigUint::from(1u8);
        let delta = &power - value;
        return Ok(NearPowerResult {
            exponent: 0,
            power,
            delta,
            exponents_checked: 1,
            comparisons: 1,
            fast_path: false,
            power_of_two_m: None,
            k_floor: Some(0),
            shift_bits_used: 0,
        });
    }

    if let Some(m) = power_of_two_base_exponent(base) {
        let value_bits = value.bits();
        let m_u64 = m as u64;
        let k_floor_u64 = if value_bits <= 1 {
            0u64
        } else {
            (value_bits - 1) / m_u64
        };
        if k_floor_u64 > u32::MAX as u64 {
            bail!("exponent exceeds u32::MAX; base too small for this input");
        }
        let k_floor = k_floor_u64 as u32;
        let shift_floor = m_u64.saturating_mul(k_floor_u64);
        if shift_floor > usize::MAX as u64 {
            bail!("shift exceeds addressable size for power-of-two fast path");
        }
        let power_floor = BigUint::from(1u8) << (shift_floor as usize);
        let (delta_floor, delta_floor_exact) = abs_diff_counted(value, &power_floor);
        let mut best = NearPowerResult {
            exponent: k_floor,
            power: power_floor,
            delta: delta_floor,
            exponents_checked: 1,
            comparisons: 1,
            fast_path: true,
            power_of_two_m: Some(m),
            k_floor: Some(k_floor),
            shift_bits_used: shift_floor,
        };

        if delta_floor_exact {
            return Ok(best);
        }

        if k_floor < u32::MAX {
            let k_hi = k_floor + 1;
            let shift_hi = m_u64.saturating_mul(k_hi as u64);
            if shift_hi > usize::MAX as u64 {
                return Ok(best);
            }
            let power_hi = BigUint::from(1u8) << (shift_hi as usize);
            if no_overshoot && power_hi > *value {
                return Ok(best);
            }
            let (delta_hi, delta_hi_exact) = abs_diff_counted(value, &power_hi);
            best.comparisons += 1;
            best.exponents_checked += 1;
            best.shift_bits_used = best.shift_bits_used.saturating_add(shift_hi);
            match delta_hi.cmp(&best.delta) {
                Ordering::Less => {
                    best = NearPowerResult {
                        exponent: k_hi,
                        power: power_hi,
                        delta: delta_hi,
                        exponents_checked: best.exponents_checked,
                        comparisons: best.comparisons,
                        fast_path: true,
                        power_of_two_m: Some(m),
                        k_floor: Some(k_floor),
                        shift_bits_used: best.shift_bits_used,
                    };
                }
                Ordering::Equal => {
                    // Keep lower exponent on ties.
                }
                Ordering::Greater => {}
            }
            if delta_hi_exact {
                return Ok(best);
            }
        }

        return Ok(best);
    }

    let (floor, checked) = floor_log_biguint(base, value)?;
    let power_floor = base.pow(floor);
    let (delta_floor, delta_floor_exact) = abs_diff_counted(value, &power_floor);
    let mut best = NearPowerResult {
        exponent: floor,
        power: power_floor,
        delta: delta_floor,
        exponents_checked: checked,
        comparisons: checked,
        fast_path: false,
        power_of_two_m: None,
        k_floor: Some(floor),
        shift_bits_used: 0,
    };

    if delta_floor_exact {
        return Ok(best);
    }

    if floor < u32::MAX {
        let hi = floor + 1;
        let power_hi = base.pow(hi);
        if no_overshoot && power_hi > *value {
            return Ok(best);
        }
        let (delta_hi, delta_hi_exact) = abs_diff_counted(value, &power_hi);
        best.exponents_checked += 1;
        best.comparisons += 1;
        match delta_hi.cmp(&best.delta) {
            Ordering::Less => {
                best = NearPowerResult {
                    exponent: hi,
                    power: power_hi,
                    delta: delta_hi,
                    exponents_checked: best.exponents_checked,
                    comparisons: best.comparisons,
                    fast_path: false,
                    power_of_two_m: None,
                    k_floor: Some(floor),
                    shift_bits_used: 0,
                };
            }
            Ordering::Equal => {
                // Keep lower exponent on ties.
            }
            Ordering::Greater => {}
        }
        if delta_hi_exact {
            return Ok(best);
        }
    }

    Ok(best)
}

fn run_near_power_iterations(
    base: &BigUint,
    value: &BigUint,
    n_times: u32,
    no_overshoot: bool,
) -> Result<NearPowerIterations> {
    let base_bits = base.bits();
    let mut remaining = value.clone();
    let mut total_power = BigUint::from(0u8);
    let mut total_delta = BigUint::from(0u8);
    let mut total_exponents_checked = 0u64;
    let mut total_comparisons = 0u64;
    let mut fast_path_used_count = 0u64;
    let mut total_shift_bits = 0u64;
    let mut exponents = Vec::new();

    for idx in 1..=n_times {
        let result = nearest_power_exponent(base, &remaining, no_overshoot)?;
        let delta_base10_digits = biguint_decimal_digits(&result.delta);
        let delta_base2_digits = result.delta.bits().max(1);
        let power_percent = percent_of_value_string(&remaining, &result.power);
        let percent_delta = percent_delta_string(&remaining, &result.delta);
        let cumulative_coverage_percent = coverage_percent_string(value, &result.delta);
        let power_over = result.power >= remaining;
        info!(
            iteration = idx,
            base_value = %base,
            base_bits = base.bits(),
            value_bits = remaining.bits(),
            fast_path = result.fast_path,
            power_of_two_m = result.power_of_two_m,
            k_floor = result.k_floor,
            exponent = result.exponent,
            power_bits = result.power.bits(),
            delta_bits = result.delta.bits(),
            delta_base10_digits,
            delta_base2_digits,
            exponents_checked = result.exponents_checked,
            comparisons = result.comparisons,
            power_over,
            percent_delta = %percent_delta,
            power_percent = %power_percent,
            cumulative_coverage_percent = %cumulative_coverage_percent,
            "near-power iteration complete"
        );

        total_power += &result.power;
        total_delta += &result.delta;
        total_exponents_checked += result.exponents_checked;
        total_comparisons += result.comparisons;
        if result.fast_path {
            fast_path_used_count += 1;
            total_shift_bits = total_shift_bits.saturating_add(result.shift_bits_used);
        }
        exponents.push(result.exponent);

        remaining = result.delta;
        if remaining.is_zero() {
            break;
        }
    }

    let iterations = exponents.len() as u32;
    Ok(NearPowerIterations {
        exponents,
        iterations,
        total_power,
        total_delta,
        total_exponents_checked,
        total_comparisons,
        fast_path_used_count,
        total_shift_bits,
        final_delta: remaining,
        base_bits,
        base_sequence: None,
    })
}

fn run_near_power_iterations_with_bases(
    bases: &[BigUint],
    value: &BigUint,
    no_overshoot: bool,
) -> Result<NearPowerIterations> {
    let mut remaining = value.clone();
    let mut total_power = BigUint::from(0u8);
    let mut total_delta = BigUint::from(0u8);
    let mut total_exponents_checked = 0u64;
    let mut total_comparisons = 0u64;
    let mut fast_path_used_count = 0u64;
    let mut total_shift_bits = 0u64;
    let mut exponents = Vec::new();

    for (idx, base) in bases.iter().enumerate() {
        let result = nearest_power_exponent(base, &remaining, no_overshoot)?;
        let delta_base10_digits = biguint_decimal_digits(&result.delta);
        let delta_base2_digits = result.delta.bits().max(1);
        let power_percent = percent_of_value_string(&remaining, &result.power);
        let percent_delta = percent_delta_string(&remaining, &result.delta);
        let cumulative_coverage_percent = coverage_percent_string(value, &result.delta);
        let power_over = result.power >= remaining;
        info!(
            iteration = (idx + 1),
            base_bits = base.bits(),
            value_bits = remaining.bits(),
            fast_path = result.fast_path,
            power_of_two_m = result.power_of_two_m,
            k_floor = result.k_floor,
            exponent = result.exponent,
            power_bits = result.power.bits(),
            delta_bits = result.delta.bits(),
            delta_base10_digits,
            delta_base2_digits,
            exponents_checked = result.exponents_checked,
            comparisons = result.comparisons,
            power_over,
            percent_delta = %percent_delta,
            power_percent = %power_percent,
            cumulative_coverage_percent = %cumulative_coverage_percent,
            "near-power iteration complete"
        );

        total_power += &result.power;
        total_delta += &result.delta;
        total_exponents_checked += result.exponents_checked;
        total_comparisons += result.comparisons;
        if result.fast_path {
            fast_path_used_count += 1;
            total_shift_bits = total_shift_bits.saturating_add(result.shift_bits_used);
        }
        exponents.push(result.exponent);

        remaining = result.delta;
        if remaining.is_zero() {
            break;
        }
    }

    let iterations = exponents.len() as u32;
    Ok(NearPowerIterations {
        exponents,
        iterations,
        total_power,
        total_delta,
        total_exponents_checked,
        total_comparisons,
        fast_path_used_count,
        total_shift_bits,
        final_delta: remaining,
        base_bits: 0,
        base_sequence: Some(bases.iter().map(|b| b.to_u64().unwrap_or(0)).collect()),
    })
}

fn compress_sequence_a(exponents: &[u32], scheme: CompressionScheme) -> CompressionResult {
    let exps: Vec<i128> = exponents.iter().map(|&e| e as i128).collect();
    let base = choose_base_for_scheme(&exps, scheme);
    let deltas: Vec<i128> = exps.iter().map(|&e| e - base).collect();
    compression_metrics(scheme, base, deltas)
}

fn compress_sequence_b(exponents: &[u32], scheme: CompressionScheme) -> CompressionResult {
    let exps: Vec<i128> = exponents.iter().map(|&e| e as i128).collect();
    let base = exps.first().copied().unwrap_or(0);
    let mut deltas = Vec::new();
    for window in exps.windows(2) {
        let delta = window[1] - window[0];
        deltas.push(delta);
    }
    compression_metrics(scheme, base, deltas)
}

fn choose_base_for_scheme(exps: &[i128], scheme: CompressionScheme) -> i128 {
    if exps.is_empty() {
        return 0;
    }
    match scheme {
        CompressionScheme::MinTotalAbs => median_i128(exps),
        CompressionScheme::MinMaxAbs => midrange_i128(exps),
        CompressionScheme::MinDigitCount
        | CompressionScheme::MinBitCount
        | CompressionScheme::MinVarintSize => best_base_by_cost(exps, scheme),
    }
}

fn compression_metrics(
    scheme: CompressionScheme,
    base: i128,
    deltas: Vec<i128>,
) -> CompressionResult {
    let mut max_abs = 0i128;
    let mut total_abs = 0i128;
    for &d in &deltas {
        let abs = d.abs();
        if abs > max_abs {
            max_abs = abs;
        }
        total_abs = total_abs.saturating_add(abs);
    }
    let total_digit_count = total_digit_count_for_scheme(scheme, base, &deltas);
    CompressionResult {
        scheme,
        base,
        deltas,
        max_abs_delta: max_abs,
        total_abs_delta: total_abs,
        total_digit_count,
    }
}

fn total_digit_count_for_scheme(
    scheme: CompressionScheme,
    base: i128,
    deltas: &[i128],
) -> u64 {
    match scheme {
        CompressionScheme::MinBitCount => {
            let mut total = bit_len_i128(base) as u64;
            for &d in deltas {
                total = total.saturating_add(bit_len_i128(d) as u64);
            }
            total
        }
        CompressionScheme::MinVarintSize => {
            let mut total = varint_len_i128(base) as u64;
            for &d in deltas {
                total = total.saturating_add(varint_len_i128(d) as u64);
            }
            total
        }
        _ => {
            let mut total = digits_i128(base) as u64;
            for &d in deltas {
                total = total.saturating_add(digits_i128(d) as u64);
            }
            total
        }
    }
}

fn best_base_by_cost(exps: &[i128], scheme: CompressionScheme) -> i128 {
    let mut candidates: Vec<i128> = Vec::new();
    let min = *exps.iter().min().unwrap_or(&0);
    let max = *exps.iter().max().unwrap_or(&0);
    candidates.push(min);
    candidates.push(max);
    candidates.push(midrange_i128(exps));
    candidates.push(median_i128(exps));
    candidates.push(0);
    for &e in exps {
        candidates.push(e);
    }
    candidates.sort_unstable();
    candidates.dedup();

    let mut best = candidates[0];
    let mut best_cost = cost_for_scheme(exps, best, scheme);
    for &cand in candidates.iter().skip(1) {
        let cost = cost_for_scheme(exps, cand, scheme);
        if cost < best_cost || (cost == best_cost && cand < best) {
            best = cand;
            best_cost = cost;
        }
    }
    best
}

fn cost_for_scheme(exps: &[i128], base: i128, scheme: CompressionScheme) -> u64 {
    match scheme {
        CompressionScheme::MinDigitCount => {
            let mut total = digits_i128(base) as u64;
            for &e in exps {
                total = total.saturating_add(digits_i128(e - base) as u64);
            }
            total
        }
        CompressionScheme::MinBitCount => {
            let mut total = bit_len_i128(base) as u64;
            for &e in exps {
                total = total.saturating_add(bit_len_i128(e - base) as u64);
            }
            total
        }
        CompressionScheme::MinVarintSize => {
            let mut total = varint_len_i128(base) as u64;
            for &e in exps {
                total = total.saturating_add(varint_len_i128(e - base) as u64);
            }
            total
        }
        _ => 0,
    }
}

fn median_i128(values: &[i128]) -> i128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    sorted[mid]
}

fn midrange_i128(values: &[i128]) -> i128 {
    let min = *values.iter().min().unwrap_or(&0);
    let max = *values.iter().max().unwrap_or(&0);
    (min + max) / 2
}

fn digits_i128(value: i128) -> u32 {
    let mut v = value.abs();
    if v == 0 {
        return 1;
    }
    let mut digits = 0u32;
    while v > 0 {
        v /= 10;
        digits += 1;
    }
    digits
}

fn bit_len_i128(value: i128) -> u32 {
    let v = value.unsigned_abs();
    let bits = 128u32.saturating_sub(v.leading_zeros());
    if bits == 0 { 1 } else { bits }
}

fn varint_len_i128(value: i128) -> u32 {
    let bits = bit_len_i128(value);
    ((bits + 6) / 7).max(1)
}

fn format_i128_list(values: &[i128]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn abs_diff_counted(a: &BigUint, b: &BigUint) -> (BigUint, bool) {
    match a.cmp(b) {
        Ordering::Greater => (a - b, false),
        Ordering::Equal => (BigUint::from(0u8), true),
        Ordering::Less => (b - a, false),
    }
}

fn biguint_decimal_digits(value: &BigUint) -> u64 {
    if value.is_zero() {
        return 1;
    }
    let bits = value.bits();
    if bits == 0 {
        return 1;
    }
    ((bits - 1) as f64 * std::f64::consts::LOG10_2).floor() as u64 + 1
}

fn percent_of_value_string(value: &BigUint, power: &BigUint) -> String {
    if value.is_zero() {
        if power.is_zero() {
            return "0.000000%".to_string();
        }
        return "inf".to_string();
    }

    let scale = BigUint::from(1_000_000u64);
    let factor = BigUint::from(100_000_000u64);
    let scaled = (power * &factor) / value;
    let integer = &scaled / &scale;
    let frac = &scaled % &scale;
    let frac_u64 = frac.to_u64().unwrap_or(0);
    format!("{}.{:06}%", integer.to_str_radix(10), frac_u64)
}

fn percent_delta_string(value: &BigUint, delta: &BigUint) -> String {
    if value.is_zero() {
        if delta.is_zero() {
            return "0.000000%".to_string();
        }
        return "inf".to_string();
    }

    let scale = BigUint::from(1_000_000u64);
    let factor = BigUint::from(100_000_000u64);
    let scaled = (delta * &factor) / value;
    let integer = &scaled / &scale;
    let frac = &scaled % &scale;
    let frac_u64 = frac.to_u64().unwrap_or(0);
    format!("{}.{:06}%", integer.to_str_radix(10), frac_u64)
}

fn coverage_percent_string(value: &BigUint, remaining: &BigUint) -> String {
    if value.is_zero() {
        return "0.000000%".to_string();
    }
    let capped = if remaining > value { value.clone() } else { remaining.clone() };
    let scale = BigUint::from(1_000_000u64);
    let factor = BigUint::from(100_000_000u64);
    let scaled = (&capped * &factor) / value;
    let coverage_scaled = if scaled > factor {
        BigUint::from(0u8)
    } else {
        &factor - scaled
    };
    let integer = &coverage_scaled / &scale;
    let frac = &coverage_scaled % &scale;
    let frac_u64 = frac.to_u64().unwrap_or(0);
    format!("{}.{:06}%", integer.to_str_radix(10), frac_u64)
}

fn floor_log_biguint(base: &BigUint, value: &BigUint) -> Result<(u32, u64)> {
    if value.is_zero() {
        return Ok((0, 1));
    }
    if base <= &BigUint::from(1u8) {
        bail!("base must be >= 2");
    }

    let value_bits = value.bits();
    let base_bits = base.bits();
    if base_bits <= 1 {
        return Ok((0, 1));
    }

    let max_k = if value_bits <= 1 {
        0u64
    } else {
        (value_bits - 1) / (base_bits - 1)
    };
    if max_k > u32::MAX as u64 {
        bail!("exponent exceeds u32::MAX; base too small for this input");
    }

    // Fast path: estimate k via logs, then adjust locally.
    let (approx_k, mut checked) = estimate_floor_log(base, value)?;
    let mut k = approx_k.min(max_k as u32);

    // Adjust downward if we overshot.
    let mut guard = 0u32;
    loop {
        let cmp = pow_cmp(base, k, value);
        checked += 1;
        match cmp {
            Ordering::Greater => {
                if k == 0 {
                    return Ok((0, checked.max(1)));
                }
                k -= 1;
            }
            Ordering::Equal | Ordering::Less => break,
        }
        guard += 1;
        if guard > 6 {
            break;
        }
    }

    // Adjust upward if we can still increase.
    guard = 0;
    loop {
        if k == u32::MAX {
            break;
        }
        let next = k + 1;
        let cmp = pow_cmp(base, next, value);
        checked += 1;
        if matches!(cmp, Ordering::Greater) {
            break;
        }
        k = next;
        guard += 1;
        if guard > 6 {
            break;
        }
    }

    // Fallback to binary search if estimate was too far off.
    if guard > 6 {
        let mut lo = 0u32;
        let mut hi = max_k as u32;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            match pow_cmp(base, mid, value) {
                Ordering::Greater => {
                    checked += 1;
                    hi = mid.saturating_sub(1);
                }
                Ordering::Equal | Ordering::Less => {
                    checked += 1;
                    lo = mid;
                }
            }
        }
        return Ok((lo, checked.max(1)));
    }

    Ok((k, checked.max(1)))
}

fn pow_cmp(base: &BigUint, exp: u32, target: &BigUint) -> Ordering {
    if exp == 0 {
        return BigUint::from(1u8).cmp(target);
    }

    let mut result = BigUint::from(1u8);
    let mut base_pow = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= &base_pow;
            if result > *target {
                return Ordering::Greater;
            }
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        base_pow = &base_pow * &base_pow;
    }
    result.cmp(target)
}

fn estimate_floor_log(base: &BigUint, value: &BigUint) -> Result<(u32, u64)> {
    if value.is_zero() {
        return Ok((0, 1));
    }
    let ln_n = ln_biguint(value)?;
    let ln_b = ln_biguint(base)?;
    if !ln_b.is_finite() || ln_b <= 0.0 {
        bail!("invalid base for log estimate");
    }
    let k = (ln_n / ln_b).floor();
    if !k.is_finite() || k < 0.0 {
        return Ok((0, 1));
    }
    let k_u64 = k as u64;
    if k_u64 > u32::MAX as u64 {
        bail!("exponent exceeds u32::MAX; base too small for this input");
    }
    Ok((k_u64 as u32, 1))
}

fn ln_biguint(value: &BigUint) -> Result<f64> {
    let bit_len = value.bits();
    if bit_len == 0 {
        return Ok(f64::NEG_INFINITY);
    }
    let shift = bit_len.saturating_sub(53);
    let top = if shift > 0 {
        (value >> shift)
            .to_u64()
            .ok_or_else(|| anyhow::anyhow!("failed to normalize input"))? as f64
    } else {
        value
            .to_u64()
            .ok_or_else(|| anyhow::anyhow!("failed to normalize input"))? as f64
    };
    Ok(top.ln() + (shift as f64) * std::f64::consts::LN_2)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum EstimateMode {
    Random,
    Adversarial,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum CompressionScheme {
    MinMaxAbs,
    MinTotalAbs,
    MinDigitCount,
    MinBitCount,
    MinVarintSize,
}

fn run_estimate_any_factor(
    input_path: Option<&Path>,
    buffer_size: usize,
    encoding: NumberEncoding,
    digits: Option<u64>,
    max_divisor: Option<u64>,
    factors_per_sec: Option<f64>,
    mode: EstimateMode,
) -> Result<()> {
    if digits.is_some() && input_path.is_some() {
        bail!("use either --digits or --input/--number for estimate-any-factor");
    }
    let digits = if let Some(digits) = digits {
        if digits == 0 {
            bail!("--digits must be > 0");
        }
        digits
    } else if let Some(path) = input_path {
        estimate_digits_from_input(path, buffer_size, encoding)?
    } else {
        bail!("estimate-any-factor requires --digits or an input source");
    };

    let max_divisor = max_divisor.unwrap_or_else(|| default_max_divisor_for_digits(digits));
    if max_divisor < 2 {
        bail!("--max-divisor must be >= 2");
    }
    let factors_per_sec = factors_per_sec.unwrap_or(350_000_000.0);
    if !factors_per_sec.is_finite() || factors_per_sec <= 0.0 {
        bail!("--factors-per-sec must be a positive finite number");
    }

    let seconds = max_divisor as f64 / factors_per_sec;
    let eta = format_duration(seconds);
    let random_prob = random_small_factor_probability(max_divisor);

    println!("digits={digits}");
    println!("max_divisor={max_divisor}");
    println!("factors_per_sec={factors_per_sec}");
    println!("scan_seconds={seconds:.3}");
    println!("scan_eta={eta}");

    match mode {
        EstimateMode::Random => {
            println!("mode=random");
            println!("estimated_success_probability={random_prob:.6}");
            println!("assumption=random integer; approximation via Mertens product");
        }
        EstimateMode::Adversarial => {
            println!("mode=adversarial");
            println!("estimated_success_probability=unknown");
            println!("assumption=adversarial input; only scan-time estimate is reliable");
        }
        EstimateMode::Unknown => {
            println!("mode=unknown");
            println!("estimated_success_probability_random={random_prob:.6}");
            println!("estimated_success_probability_adversarial=unknown");
        }
    }

    info!(
        digits,
        max_divisor,
        factors_per_sec,
        scan_seconds = seconds,
        mode = ?mode,
        random_probability = random_prob,
        "any-factor estimate complete"
    );
    Ok(())
}

fn run_range_factors(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    first: bool,
    low_latency_first: bool,
    last: bool,
    limit: Option<u64>,
    use_gpu: bool,
    gpu_batch_size: usize,
    encoding: NumberEncoding,
    policies: &config::PoliciesConfig,
) -> Result<()> {
    if first {
        let mut selected = None;
        scan_factor_range(
            input_path,
            buffer_size,
            start,
            end,
            all,
            use_gpu,
            gpu_batch_size,
            Some(1),
            low_latency_first,
            encoding,
            policies,
            |factor| {
                selected = Some(factor);
                Ok(FactorScanDecision::Stop)
            },
        )?;
        if let Some(value) = selected {
            println!("{value}");
            info!(start, end, value, "first factor scan complete");
        } else {
            println!("null");
            info!(start, end, "first factor scan complete with no match");
        }
        return Ok(());
    }

    if last {
        let mut selected = None;
        let count = scan_factor_range(
            input_path,
            buffer_size,
            start,
            end,
            all,
            use_gpu,
            gpu_batch_size,
            limit,
            low_latency_first,
            encoding,
            policies,
            |factor| {
                selected = Some(factor);
                Ok(FactorScanDecision::Continue)
            },
        )?;
        if let Some(value) = selected {
            println!("{value}");
            info!(
                start,
                end,
                value,
                factors_found = count,
                "last factor scan complete"
            );
        } else {
            println!("null");
            info!(
                start,
                end,
                factors_found = count,
                "last factor scan complete with no match"
            );
        }
        return Ok(());
    }

    let mut stdout = std::io::stdout().lock();
    write!(stdout, "[")?;
    let mut first_output = true;
    let count = scan_factor_range(
        input_path,
        buffer_size,
        start,
        end,
        all,
        use_gpu,
        gpu_batch_size,
        limit,
        low_latency_first,
        encoding,
        policies,
        |factor| {
            if !first_output {
                write!(stdout, ",")?;
            }
            first_output = false;
            write!(stdout, "{factor}")?;
            Ok(FactorScanDecision::Continue)
        },
    )?;

    writeln!(stdout, "]")?;
    info!(
        start,
        end,
        all,
        first,
        low_latency_first,
        last,
        limit,
        use_gpu,
        gpu_batch_size,
        encoding = ?encoding,
        factors_found = count,
        "range factor scan complete"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NumberEncoding {
    Decimal,
    BinaryBigEndian,
    BinaryLittleEndian,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FactorScanDecision {
    Continue,
    Stop,
}

fn estimate_digits_from_input(
    path: &Path,
    buffer_size: usize,
    encoding: NumberEncoding,
) -> Result<u64> {
    match encoding {
        NumberEncoding::Decimal => {
            let stream = DecimalStream::open(path, buffer_size)?;
            let len = stream.decimal_len()?;
            if len == 0 {
                bail!("input has no decimal digits");
            }
            Ok(len)
        }
        NumberEncoding::BinaryBigEndian | NumberEncoding::BinaryLittleEndian => {
            binary_decimal_digits(path)
        }
    }
}

fn binary_decimal_digits(path: &Path) -> Result<u64> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let file_len = metadata.len();
    if file_len == 0 {
        return Ok(1);
    }

    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let mut consumed = 0u64;
    let mut first_non_zero = None;
    let mut buf = [0u8; 8192];
    while consumed < file_len {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        for (idx, b) in buf[..read].iter().enumerate() {
            if *b != 0 {
                first_non_zero = Some((consumed + idx as u64, *b));
                break;
            }
        }
        if first_non_zero.is_some() {
            break;
        }
        consumed += read as u64;
    }

    let Some((offset, first_byte)) = first_non_zero else {
        return Ok(1);
    };
    let remaining_bytes = file_len - offset;
    let leading_bits = 8u32 - first_byte.leading_zeros();
    let bit_len = (remaining_bytes - 1) * 8 + leading_bits as u64;
    let digits = ((bit_len - 1) as f64 * std::f64::consts::LOG10_2).floor() as u64 + 1;
    Ok(digits.max(1))
}

fn default_max_divisor_for_digits(digits: u64) -> u64 {
    let sqrt_log10 = (digits as f64) / 2.0;
    let max_log10 = (u64::MAX as f64).log10();
    if sqrt_log10 >= max_log10 {
        return u64::MAX;
    }
    10f64.powf(sqrt_log10).ceil().clamp(2.0, u64::MAX as f64) as u64
}

fn random_small_factor_probability(max_divisor: u64) -> f64 {
    if max_divisor < 3 {
        return 0.0;
    }
    const MERTENS_E_NEG_GAMMA: f64 = 0.561_459_483_566_885_1;
    let ln_b = (max_divisor as f64).ln();
    let no_small_factor = (MERTENS_E_NEG_GAMMA / ln_b).clamp(0.0, 1.0);
    (1.0 - no_small_factor).clamp(0.0, 1.0)
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }
    let total = seconds.round() as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if days > 0 {
        format!("{days}d {hours}h {minutes}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

fn scan_factor_range<F>(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    use_gpu: bool,
    gpu_batch_size: usize,
    factor_limit: Option<u64>,
    low_latency_first: bool,
    encoding: NumberEncoding,
    policies: &config::PoliciesConfig,
    mut on_factor: F,
) -> Result<u64>
where
    F: FnMut(u64) -> Result<FactorScanDecision>,
{
    if matches!(factor_limit, Some(0)) {
        return Ok(0);
    }
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

    if use_gpu {
        let mut found = 0u64;
        let gpu_limit = u32::MAX as u64;
        let gpu_end = end.min(gpu_limit);

        if start <= gpu_end {
            found += scan_factor_range_batched(
                input_path,
                buffer_size,
                start,
                gpu_end,
                all,
                gpu_batch_size,
                factor_limit,
                low_latency_first,
                encoding,
                &mut on_factor,
            )?;
        }

        if end > gpu_limit {
            let tail_start = start.max(gpu_limit.saturating_add(1));
            if tail_start <= end {
                info!(
                    tail_start,
                    tail_end = end,
                    "range exceeds GPU divisor width; scanning tail on CPU"
                );
                found += scan_factor_range_scalar(
                    input_path,
                    buffer_size,
                    tail_start,
                    end,
                    all,
                    factor_limit,
                    encoding,
                    &mut on_factor,
                )?;
            }
        }
        return Ok(found);
    }

    scan_factor_range_scalar(
        input_path,
        buffer_size,
        start,
        end,
        all,
        factor_limit,
        encoding,
        &mut on_factor,
    )
}

fn scan_factor_range_batched<F>(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    gpu_batch_size: usize,
    factor_limit: Option<u64>,
    low_latency_first: bool,
    encoding: NumberEncoding,
    on_factor: &mut F,
) -> Result<u64>
where
    F: FnMut(u64) -> Result<FactorScanDecision>,
{
    const FIRST_FACTOR_BATCH_SIZE: usize = 4096;
    let start_time = Instant::now();
    let requested_batch = gpu_batch_size.max(1);
    let auto_batch = gpu_batch_size == 0;
    let mut found = 0u64;
    let mut chunk_start = start;
    let mut chunk_count = 0u64;
    let mut divisors_scanned = 0u64;
    let mut symbols_processed = 0u64;
    let mut update_ms = 0.0f64;
    let mut readback_ms = 0.0f64;
    let bootstrap_batch = requested_batch.min((end - start + 1) as usize).max(1);
    let bootstrap_end = start
        .saturating_add(bootstrap_batch as u64 - 1)
        .min(end)
        .min(u32::MAX as u64);
    let mut divisors: Vec<u32> = (chunk_start..=bootstrap_end).map(|d| d as u32).collect();
    let mut engine = BatchModEngine::try_new(&divisors, true)?;
    let mut configured_batch = requested_batch;
    if auto_batch {
        if factor_limit == Some(1) && low_latency_first {
            configured_batch = FIRST_FACTOR_BATCH_SIZE;
        } else {
            configured_batch = engine.recommended_batch_size().unwrap_or(requested_batch);
        }
        info!(configured_batch, "auto-tuned GPU range batch size");
        engine.ensure_capacity(configured_batch)?;
        let tuned_end = chunk_start
            .saturating_add(configured_batch as u64 - 1)
            .min(end)
            .min(u32::MAX as u64);
        divisors = (chunk_start..=tuned_end).map(|d| d as u32).collect();
        engine.reset_primes(&divisors)?;
    }

    while chunk_start <= end {
        chunk_count += 1;
        let chunk_end = divisors.last().copied().unwrap_or(chunk_start as u32) as u64;
        divisors_scanned += divisors.len() as u64;
        stream_symbols(input_path, buffer_size, encoding, |symbols| {
            if !symbols.is_empty() {
                symbols_processed += symbols.len() as u64;
                let (radix, order) = encoding_batch_params(encoding);
                let update_start = Instant::now();
                engine.update_symbols(symbols, radix, order)?;
                update_ms += update_start.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(())
        })?;
        let readback_start = Instant::now();
        let remainders = engine.remainders()?;
        readback_ms += readback_start.elapsed().as_secs_f64() * 1000.0;
        for (&divisor_u32, &remainder) in divisors.iter().zip(remainders.iter()) {
            let divisor = divisor_u32 as u64;
            if divisor == 1 {
                if on_factor(1)? == FactorScanDecision::Stop {
                    return Ok(found + 1);
                }
                found += 1;
                if found == factor_limit.unwrap_or(u64::MAX) {
                    return Ok(found);
                }
                continue;
            }
            if remainder != 0 {
                continue;
            }
            if !all {
                if on_factor(divisor)? == FactorScanDecision::Stop {
                    return Ok(found + 1);
                }
                found += 1;
                if found == factor_limit.unwrap_or(u64::MAX) {
                    return Ok(found);
                }
                continue;
            }
            let mut power = divisor;
            loop {
                if mod_from_file(input_path, buffer_size, power, encoding)? != 0 {
                    break;
                }
                if on_factor(divisor)? == FactorScanDecision::Stop {
                    return Ok(found + 1);
                }
                found += 1;
                if found == factor_limit.unwrap_or(u64::MAX) {
                    return Ok(found);
                }
                match power.checked_mul(divisor) {
                    Some(next) => power = next,
                    None => break,
                }
            }
        }
        if chunk_end == end {
            break;
        }
        chunk_start = chunk_end + 1;
        let next_end = chunk_start
            .saturating_add(configured_batch as u64 - 1)
            .min(end)
            .min(u32::MAX as u64);
        divisors = (chunk_start..=next_end).map(|d| d as u32).collect();
        engine.reset_primes(&divisors)?;
    }
    let elapsed = start_time.elapsed().as_secs_f64();
    let throughput = if elapsed > 0.0 {
        divisors_scanned as f64 / elapsed
    } else {
        0.0
    };
    info!(
        chunks = chunk_count,
        gpu_batch_size = configured_batch,
        divisors_scanned,
        symbols_processed,
        update_ms,
        readback_ms,
        total_ms = elapsed * 1000.0,
        divisors_per_sec = throughput,
        "gpu range batch metrics"
    );
    Ok(found)
}

fn scan_factor_range_scalar<F>(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    factor_limit: Option<u64>,
    encoding: NumberEncoding,
    on_factor: &mut F,
) -> Result<u64>
where
    F: FnMut(u64) -> Result<FactorScanDecision>,
{
    let mut found = 0u64;
    for divisor in start..=end {
        if divisor == 1 {
            if on_factor(1)? == FactorScanDecision::Stop {
                return Ok(found + 1);
            }
            found += 1;
            if found == factor_limit.unwrap_or(u64::MAX) {
                return Ok(found);
            }
            continue;
        }
        if !all {
            if mod_from_file(input_path, buffer_size, divisor, encoding)? == 0 {
                if on_factor(divisor)? == FactorScanDecision::Stop {
                    return Ok(found + 1);
                }
                found += 1;
                if found == factor_limit.unwrap_or(u64::MAX) {
                    return Ok(found);
                }
            }
            continue;
        }

        // In --all mode, emit the divisor once for each power d^k that divides N.
        let mut power = divisor;
        loop {
            if mod_from_file(input_path, buffer_size, power, encoding)? != 0 {
                break;
            }
            if on_factor(divisor)? == FactorScanDecision::Stop {
                return Ok(found + 1);
            }
            found += 1;
            if found == factor_limit.unwrap_or(u64::MAX) {
                return Ok(found);
            }
            match power.checked_mul(divisor) {
                Some(next) => power = next,
                None => break,
            }
        }
    }
    Ok(found)
}

fn encoding_batch_params(encoding: NumberEncoding) -> (u32, SymbolOrder) {
    match encoding {
        NumberEncoding::Decimal => (10, SymbolOrder::BigEndian),
        NumberEncoding::BinaryBigEndian => (256, SymbolOrder::BigEndian),
        NumberEncoding::BinaryLittleEndian => (256, SymbolOrder::LittleEndian),
    }
}

fn mod_from_file(
    input_path: &Path,
    buffer_size: usize,
    modulus: u64,
    encoding: NumberEncoding,
) -> Result<u64> {
    if modulus == 0 {
        bail!("modulus must be nonzero");
    }
    let mut reader = BufReader::new(
        File::open(input_path)
            .with_context(|| format!("failed to open {}", input_path.display()))?,
    );
    let mut buf = vec![0u8; buffer_size];
    let mut remainder = 0u64;
    let mut factor = 1u64;

    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        match encoding {
            NumberEncoding::Decimal => {
                for &byte in &buf[..read] {
                    if byte.is_ascii_digit() {
                        remainder = (remainder * 10 + (byte - b'0') as u64) % modulus;
                    }
                }
            }
            NumberEncoding::BinaryBigEndian => {
                for &byte in &buf[..read] {
                    remainder = (remainder * 256 + byte as u64) % modulus;
                }
            }
            NumberEncoding::BinaryLittleEndian => {
                for &byte in &buf[..read] {
                    remainder = (remainder + (byte as u64) * factor) % modulus;
                    factor = (factor * 256) % modulus;
                }
            }
        }
    }
    Ok(remainder)
}

fn stream_symbols<F>(
    input_path: &Path,
    buffer_size: usize,
    encoding: NumberEncoding,
    mut on_symbols: F,
) -> Result<()>
where
    F: FnMut(&[u32]) -> Result<()>,
{
    let mut reader = BufReader::new(
        File::open(input_path)
            .with_context(|| format!("failed to open {}", input_path.display()))?,
    );
    let mut raw = vec![0u8; buffer_size];
    let mut symbols = Vec::with_capacity(buffer_size);
    loop {
        let read = reader.read(&mut raw)?;
        if read == 0 {
            break;
        }
        symbols.clear();
        match encoding {
            NumberEncoding::Decimal => {
                for &byte in &raw[..read] {
                    if byte.is_ascii_digit() {
                        symbols.push((byte - b'0') as u32);
                    }
                }
            }
            NumberEncoding::BinaryBigEndian | NumberEncoding::BinaryLittleEndian => {
                symbols.extend(raw[..read].iter().map(|&b| b as u32));
            }
        }
        on_symbols(&symbols)?;
    }
    Ok(())
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

        let source = cli
            .number_source()
            .expect("number source should parse")
            .expect("source should exist");
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
            Command::RangeFactors {
                start,
                end,
                all,
                first,
                low_latency_first,
                last,
                limit,
                use_gpu,
                gpu_batch_size,
            } => {
                assert_eq!(start, 2);
                assert_eq!(end, 12);
                assert!(!all);
                assert!(!first);
                assert!(!low_latency_first);
                assert!(!last);
                assert!(limit.is_none());
                assert!(!use_gpu);
                assert!(gpu_batch_size.is_none());
            }
            _ => panic!("expected range-factors command"),
        }
        assert!(!cli.binary);
        assert!(!cli.little_endian);
    }

    #[test]
    fn cli_parses_write_decimal_command() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--input",
            "n.bin",
            "--binary",
            "write-decimal",
            "--out",
            "n.txt",
        ]);
        match cli.cmd {
            Command::WriteDecimal { out } => {
                assert_eq!(out, PathBuf::from("n.txt"));
            }
            _ => panic!("expected write-decimal command"),
        }
        assert!(cli.binary);
    }

    #[test]
    fn cli_parses_digits_log_and_pow_commands() {
        let digits_cli = Cli::parse_from(["num-chrunchr", "--number", "12345", "digits"]);
        assert!(matches!(digits_cli.cmd, Command::Digits));

        let log_cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "12345",
            "log",
            "--base",
            "10",
            "--integer-part",
        ]);
        match log_cli.cmd {
            Command::Log { base, integer_part } => {
                assert_eq!(base, 10.0);
                assert!(integer_part);
            }
            _ => panic!("expected log command"),
        }

        let pow_cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "12345",
            "pow",
            "--exponent",
            "3",
            "--out",
            "pow.txt",
        ]);
        match pow_cli.cmd {
            Command::Pow { exponent, out } => {
                assert_eq!(exponent, 3);
                assert_eq!(out, Some(PathBuf::from("pow.txt")));
            }
            _ => panic!("expected pow command"),
        }
    }

    #[test]
    fn cli_parses_near_power_command() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "12345",
            "near-power",
            "--base-number",
            "10",
            "--n-times",
            "3",
            "--compress-seqA",
            "--compress-scheme",
            "min-total-abs",
        ]);
        match cli.cmd {
            Command::NearPower {
                base_input,
                base_number,
                base_binary,
                base_little_endian,
                n_times,
                compress_seq_a,
                compress_seq_b,
                compress_scheme,
                no_overshoot,
                prime_rounds,
            } => {
                assert!(base_input.is_none());
                assert_eq!(base_number, Some("10".to_string()));
                assert!(!base_binary);
                assert!(!base_little_endian);
                assert_eq!(n_times, 3);
                assert!(compress_seq_a);
                assert!(!compress_seq_b);
                assert_eq!(compress_scheme, CompressionScheme::MinTotalAbs);
                assert!(!no_overshoot);
                assert!(!prime_rounds);
            }
            _ => panic!("expected near-power command"),
        }
    }

    #[test]
    fn cli_parses_near_power_base_binary() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "12345",
            "near-power",
            "--base-input",
            "base.bin",
            "--base-binary",
            "--base-little-endian",
        ]);
        match cli.cmd {
            Command::NearPower {
                base_input,
                base_number,
                base_binary,
                base_little_endian,
                n_times,
                compress_seq_a,
                compress_seq_b,
                compress_scheme,
                no_overshoot,
                prime_rounds,
            } => {
                assert_eq!(base_input, Some(PathBuf::from("base.bin")));
                assert!(base_number.is_none());
                assert!(base_binary);
                assert!(base_little_endian);
                assert_eq!(n_times, 1);
                assert!(!compress_seq_a);
                assert!(!compress_seq_b);
                assert_eq!(compress_scheme, CompressionScheme::MinTotalAbs);
                assert!(!no_overshoot);
                assert!(!prime_rounds);
            }
            _ => panic!("expected near-power command"),
        }
    }

    #[test]
    fn nearest_power_prefers_lower_on_tie() {
        let base = BigUint::from(2u8);
        let value = BigUint::from(6u8);
        let result = nearest_power_exponent(&base, &value, false).unwrap();
        assert_eq!(result.exponent, 2);
    }

    #[test]
    fn nearest_power_selects_closer_exponent() {
        let base = BigUint::from(10u8);
        let value = BigUint::from(900u16);
        let result = nearest_power_exponent(&base, &value, false).unwrap();
        assert_eq!(result.exponent, 3);
    }

    #[test]
    fn near_power_iterations_stop_on_exact_match() {
        let base = BigUint::from(2u8);
        let value = BigUint::from(6u8);
        let result = run_near_power_iterations(&base, &value, 5, false).unwrap();
        assert_eq!(result.exponents, vec![2, 1]);
        assert_eq!(result.iterations, 2);
        assert_eq!(result.total_power, BigUint::from(6u8));
        assert_eq!(result.total_delta, BigUint::from(2u8));
    }

    #[test]
    fn power_of_two_fast_path_matches_pow_for_small_cases() {
        let base = BigUint::from(8u8);
        let value = BigUint::from(500u16);
        let result = nearest_power_exponent(&base, &value, false).unwrap();
        let rebuilt = base.pow(result.exponent);
        assert_eq!(rebuilt, result.power);
        assert!(result.fast_path);
        assert_eq!(result.power_of_two_m, Some(3));
    }

    #[test]
    fn near_power_iterations_match_top_bits_for_base_two() {
        let base = BigUint::from(2u8);
        let value = BigUint::from(44u8); // 0b101100
        let result = run_near_power_iterations(&base, &value, 5, false).unwrap();
        assert_eq!(result.exponents, vec![5, 3, 2]);
        assert_eq!(result.iterations, 3);
        assert_eq!(result.final_delta, BigUint::from(0u8));
    }

    #[test]
    fn nearest_power_zero_returns_exponent_zero() {
        let base = BigUint::from(2u8);
        let value = BigUint::from(0u8);
        let result = nearest_power_exponent(&base, &value, false).unwrap();
        assert_eq!(result.exponent, 0);
        assert_eq!(result.delta, BigUint::from(1u8));
    }

    #[test]
    fn near_power_iterations_zero_value_stops_immediately() {
        let base = BigUint::from(4u8);
        let value = BigUint::from(0u8);
        let result = run_near_power_iterations(&base, &value, 5, false).unwrap();
        assert_eq!(result.iterations, 1);
        assert_eq!(result.final_delta, BigUint::from(1u8));
    }

    #[test]
    fn no_overshoot_keeps_power_under_target() {
        let base = BigUint::from(2u8);
        let value = BigUint::from(6u8);
        let result = nearest_power_exponent(&base, &value, true).unwrap();
        assert_eq!(result.exponent, 2);
        assert!(result.power <= value);
    }

    #[test]
    fn first_n_primes_returns_expected_sequence() {
        let primes = first_n_primes(5);
        assert_eq!(primes, vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn compress_seq_a_min_total_abs_uses_median() {
        let exps = vec![10u32, 20, 30];
        let result = compress_sequence_a(&exps, CompressionScheme::MinTotalAbs);
        assert_eq!(result.base, 20);
        assert_eq!(result.deltas, vec![-10, 0, 10]);
    }

    #[test]
    fn compress_seq_a_min_max_abs_uses_midrange() {
        let exps = vec![10u32, 25, 40];
        let result = compress_sequence_a(&exps, CompressionScheme::MinMaxAbs);
        assert_eq!(result.base, 25);
        assert_eq!(result.max_abs_delta, 15);
    }

    #[test]
    fn compress_seq_b_deltas_are_consecutive_diffs() {
        let exps = vec![10u32, 20, 15];
        let result = compress_sequence_b(&exps, CompressionScheme::MinTotalAbs);
        assert_eq!(result.base, 10);
        assert_eq!(result.deltas, vec![10, -5]);
    }

    #[test]
    fn format_i128_list_outputs_csv() {
        let values = vec![1i128, -2, 30];
        assert_eq!(format_i128_list(&values), "1,-2,30");
    }

    #[test]
    fn compression_output_fields_are_consistent() {
        let exps = vec![10u32, 20, 30];
        let result = compress_sequence_a(&exps, CompressionScheme::MinTotalAbs);
        assert_eq!(result.base, 20);
        assert_eq!(result.deltas, vec![-10, 0, 10]);
        assert_eq!(result.max_abs_delta, 10);
        assert_eq!(result.total_abs_delta, 20);
        assert!(result.total_digit_count > 0);
    }

    #[test]
    fn cli_parses_estimate_any_factor_digits_only() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "estimate-any-factor",
            "--digits",
            "1000000",
            "--factors-per-sec",
            "350000000",
        ]);
        assert!(cli.number_source().unwrap().is_none());
        assert_eq!(cli.encoding(None).unwrap(), NumberEncoding::Decimal);
        match cli.cmd {
            Command::EstimateAnyFactor {
                digits,
                max_divisor,
                factors_per_sec,
                mode,
            } => {
                assert_eq!(digits, Some(1_000_000));
                assert!(max_divisor.is_none());
                assert_eq!(factors_per_sec, Some(350_000_000.0));
                assert_eq!(mode, EstimateMode::Unknown);
            }
            _ => panic!("expected estimate-any-factor command"),
        }
    }

    #[test]
    fn cli_rejects_binary_estimate_without_input() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--binary",
            "estimate-any-factor",
            "--digits",
            "1000",
        ]);
        let err = cli.encoding(None).unwrap_err();
        assert!(err.to_string().contains("--binary requires --input"));
    }

    #[test]
    fn cli_parses_range_factors_gpu_batch_override() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "--binary",
            "--little-endian",
            "range-factors",
            "--start",
            "2",
            "--end",
            "12",
            "--use-gpu",
            "--gpu-batch-size",
            "4096",
            "--limit",
            "3",
        ]);
        match cli.cmd {
            Command::RangeFactors {
                use_gpu,
                low_latency_first,
                gpu_batch_size,
                limit,
                ..
            } => {
                assert!(use_gpu);
                assert!(!low_latency_first);
                assert_eq!(gpu_batch_size, Some(4096));
                assert_eq!(limit, Some(3));
            }
            _ => panic!("expected range-factors command"),
        }
        assert!(cli.binary);
        assert!(cli.little_endian);
    }

    #[test]
    fn cli_parses_low_latency_first_flag() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--input",
            "n.bin",
            "--binary",
            "range-factors",
            "--start",
            "2",
            "--end",
            "1000",
            "--first",
            "--low-latency-first",
            "--use-gpu",
        ]);
        match cli.cmd {
            Command::RangeFactors {
                first,
                low_latency_first,
                use_gpu,
                ..
            } => {
                assert!(first);
                assert!(low_latency_first);
                assert!(use_gpu);
            }
            _ => panic!("expected range-factors command"),
        }
    }

    #[test]
    fn cli_rejects_little_endian_without_binary() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "--little-endian",
            "mod",
            "--p",
            "7",
        ]);
        let source = cli.number_source().unwrap();
        let err = cli.encoding(source.as_ref()).unwrap_err();
        assert!(err.to_string().contains("--little-endian"));
    }

    #[test]
    fn cli_rejects_binary_inline_source() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "--binary",
            "mod",
            "--p",
            "7",
        ]);
        let source = cli.number_source().unwrap();
        let err = cli.encoding(source.as_ref()).unwrap_err();
        assert!(err.to_string().contains("--binary requires --input"));
    }

    #[test]
    fn near_power_rejects_base_little_endian_without_binary() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "near-power",
            "--base-input",
            "base.bin",
            "--base-little-endian",
        ]);
        let err = match cli.cmd {
            Command::NearPower {
                base_input,
                base_number,
                base_binary,
                base_little_endian,
                n_times: _,
                compress_seq_a: _,
                compress_seq_b: _,
                compress_scheme: _,
                no_overshoot: _,
                prime_rounds: _,
            } => load_base_biguint(
                base_input.as_ref(),
                base_number.as_deref(),
                1024,
                base_binary,
                base_little_endian,
            )
            .unwrap_err(),
            _ => panic!("expected near-power command"),
        };
        assert!(err.to_string().contains("--base-little-endian"));
    }

    #[test]
    fn near_power_rejects_base_binary_without_input() {
        let cli = Cli::parse_from([
            "num-chrunchr",
            "--number",
            "360",
            "near-power",
            "--base-number",
            "2",
            "--base-binary",
        ]);
        let err = match cli.cmd {
            Command::NearPower {
                base_input,
                base_number,
                base_binary,
                base_little_endian,
                n_times: _,
                compress_seq_a: _,
                compress_seq_b: _,
                compress_scheme: _,
                no_overshoot: _,
                prime_rounds: _,
            } => load_base_biguint(
                base_input.as_ref(),
                base_number.as_deref(),
                1024,
                base_binary,
                base_little_endian,
            )
            .unwrap_err(),
            _ => panic!("expected near-power command"),
        };
        assert!(err.to_string().contains("--base-binary requires --base-input"));
    }

    #[test]
    fn scan_factor_range_finds_expected_divisors() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "360").unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            12,
            false,
            false,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 9);
        assert_eq!(seen, vec![2, 3, 4, 5, 6, 8, 9, 10, 12]);
    }

    #[test]
    fn scan_factor_range_all_returns_multiplicity_by_power() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "64").unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            4,
            true,
            false,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 9);
        assert_eq!(seen, vec![2, 2, 2, 2, 2, 2, 4, 4, 4]);
    }

    #[test]
    fn scan_factor_range_rejects_zero_bound() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "10").unwrap();
        let cfg = Config::default();
        let err = scan_factor_range(
            file.path(),
            16,
            0,
            10,
            false,
            false,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |_| Ok(FactorScanDecision::Continue),
        )
        .unwrap_err();
        assert!(err.to_string().contains("positive integers"));
    }

    #[test]
    fn scan_factor_range_binary_big_endian() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x01, 0x00]).unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            16,
            false,
            false,
            0,
            None,
            false,
            NumberEncoding::BinaryBigEndian,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 4);
        assert_eq!(seen, vec![2, 4, 8, 16]);
    }

    #[test]
    fn scan_factor_range_binary_little_endian() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x01, 0x00]).unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            3,
            false,
            false,
            0,
            None,
            false,
            NumberEncoding::BinaryLittleEndian,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 0);
        assert!(seen.is_empty());
    }

    #[test]
    fn scan_factor_range_decimal_with_gpu_engine() {
        if crate::gpu::batch_mod::GpuBatchModEngine::new_with_force_fallback(&[2, 3], true).is_err()
        {
            eprintln!("skipping GPU range-factors test: no compatible adapter");
            return;
        }
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "360").unwrap();
        let mut seen = Vec::new();
        let cfg = Config::default();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            12,
            false,
            true,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 9);
        assert_eq!(seen, vec![2, 3, 4, 5, 6, 8, 9, 10, 12]);
    }

    #[test]
    fn scan_factor_range_use_gpu_handles_u64_tail_on_cpu() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "10").unwrap();
        let cfg = Config::default();
        let mut seen = Vec::new();
        let count = scan_factor_range(
            file.path(),
            16,
            u32::MAX as u64 + 1,
            u32::MAX as u64 + 2,
            false,
            true,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 0);
        assert!(seen.is_empty());
    }

    #[test]
    fn scan_factor_range_can_stop_after_first_factor() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "360").unwrap();
        let cfg = Config::default();
        let mut seen = Vec::new();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            12,
            false,
            false,
            0,
            None,
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Stop)
            },
        )
        .unwrap();
        assert_eq!(count, 1);
        assert_eq!(seen, vec![2]);
    }

    #[test]
    fn scan_factor_range_respects_limit() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "360").unwrap();
        let cfg = Config::default();
        let mut seen = Vec::new();
        let count = scan_factor_range(
            file.path(),
            16,
            2,
            12,
            false,
            false,
            0,
            Some(2),
            false,
            NumberEncoding::Decimal,
            &cfg.policies,
            |factor| {
                seen.push(factor);
                Ok(FactorScanDecision::Continue)
            },
        )
        .unwrap();
        assert_eq!(count, 2);
        assert_eq!(seen, vec![2, 3]);
    }

    #[test]
    fn estimate_helpers_produce_expected_values() {
        assert_eq!(default_max_divisor_for_digits(2), 10);
        assert!(random_small_factor_probability(10_000) > 0.9);
        assert_eq!(format_duration(75.0), "1m 15s");
    }

    #[test]
    fn estimate_any_factor_rejects_missing_source_and_digits() {
        let err = run_estimate_any_factor(
            None,
            1024,
            NumberEncoding::Decimal,
            None,
            Some(100),
            Some(1_000_000.0),
            EstimateMode::Unknown,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires --digits or an input source")
        );
    }

    #[test]
    fn estimate_any_factor_rejects_digits_with_input() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "12345").unwrap();
        let err = run_estimate_any_factor(
            Some(file.path()),
            1024,
            NumberEncoding::Decimal,
            Some(10),
            Some(100),
            Some(1_000_000.0),
            EstimateMode::Unknown,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("either --digits or --input/--number")
        );
    }

    #[test]
    fn digits_and_log_helpers_work_for_decimal_and_binary() {
        let mut decimal_file = NamedTempFile::new().unwrap();
        write!(decimal_file, "12345").unwrap();
        assert_eq!(
            input_decimal_digits(decimal_file.path(), 1024, NumberEncoding::Decimal).unwrap(),
            5
        );

        let mut binary_file = NamedTempFile::new().unwrap();
        binary_file.write_all(&[0x01, 0x00]).unwrap();
        assert_eq!(
            input_decimal_digits(binary_file.path(), 1024, NumberEncoding::BinaryBigEndian)
                .unwrap(),
            3
        );

        let value = BigUint::from(1000u64);
        let log10 = log_biguint_base(&value, 10.0).unwrap();
        assert!((log10 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn binary_decimal_digits_handles_leading_zero_bytes() {
        let mut binary_file = NamedTempFile::new().unwrap();
        binary_file.write_all(&[0x00, 0x00, 0x01]).unwrap();
        assert_eq!(binary_decimal_digits(binary_file.path()).unwrap(), 1);
    }

    #[test]
    fn write_decimal_from_binary_big_endian() {
        let mut input = NamedTempFile::new().unwrap();
        input.write_all(&[0x01, 0x00]).unwrap();
        let out = NamedTempFile::new().unwrap();
        run_write_decimal_from_binary(input.path(), out.path(), NumberEncoding::BinaryBigEndian)
            .unwrap();
        let text = std::fs::read_to_string(out.path()).unwrap();
        assert_eq!(text, "256");
    }

    #[test]
    fn write_decimal_from_binary_little_endian() {
        let mut input = NamedTempFile::new().unwrap();
        input.write_all(&[0x01, 0x00]).unwrap();
        let out = NamedTempFile::new().unwrap();
        run_write_decimal_from_binary(input.path(), out.path(), NumberEncoding::BinaryLittleEndian)
            .unwrap();
        let text = std::fs::read_to_string(out.path()).unwrap();
        assert_eq!(text, "1");
    }
}
