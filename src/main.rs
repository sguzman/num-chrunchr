mod config;
mod gpu;
mod repr;
mod source;
mod strategy;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser, Subcommand};
use config::{Config, LoggingConfig};
use gpu::batch_mod::{BatchModEngine, SymbolOrder};
use repr::DecimalStream;
use source::NumberSource;
use std::{
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
    fn number_source(&self) -> Result<NumberSource> {
        match (&self.input, &self.number) {
            (Some(path), None) => Ok(NumberSource::File(path.clone())),
            (None, Some(text)) => Ok(NumberSource::Inline(text.clone())),
            (Some(_), Some(_)) => bail!("cannot use --input and --number together"),
            (None, None) => bail!("provide --input or --number"),
        }
    }

    fn encoding(&self, source: &NumberSource) -> Result<NumberEncoding> {
        if self.little_endian && !self.binary {
            bail!("--little-endian is only valid with --binary");
        }
        if self.binary {
            match source {
                NumberSource::File(_) => {}
                NumberSource::Inline(_) => {
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
    let encoding = cli.encoding(&source)?;
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
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_div(&stream, d, &out, &config.policies)?;
        }
        Command::Analyze { leading } => {
            if encoding != NumberEncoding::Decimal {
                bail!("analyze command currently supports decimal input only");
            }
            let prepared = source.prepare()?;
            let stream = DecimalStream::from_config(&prepared.path, &config.stream)
                .with_context(|| format!("open input {}", prepared.path.display()))?;
            run_analyze(&stream, leading, &config.analysis)?;
        }
        Command::RangeFactors {
            start,
            end,
            all,
            first,
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
            let prepared = source.prepare()?;
            run_range_factors(
                &prepared.path,
                config.stream.buffer_size,
                start,
                end,
                all,
                first,
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

fn run_range_factors(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    first: bool,
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
            limit,
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

fn scan_factor_range<F>(
    input_path: &Path,
    buffer_size: usize,
    start: u64,
    end: u64,
    all: bool,
    use_gpu: bool,
    gpu_batch_size: usize,
    factor_limit: Option<u64>,
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
    encoding: NumberEncoding,
    on_factor: &mut F,
) -> Result<u64>
where
    F: FnMut(u64) -> Result<FactorScanDecision>,
{
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
        configured_batch = engine.recommended_batch_size().unwrap_or(requested_batch);
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
            Command::RangeFactors {
                start,
                end,
                all,
                first,
                last,
                limit,
                use_gpu,
                gpu_batch_size,
            } => {
                assert_eq!(start, 2);
                assert_eq!(end, 12);
                assert!(!all);
                assert!(!first);
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
                gpu_batch_size,
                limit,
                ..
            } => {
                assert!(use_gpu);
                assert_eq!(gpu_batch_size, Some(4096));
                assert_eq!(limit, Some(3));
            }
            _ => panic!("expected range-factors command"),
        }
        assert!(cli.binary);
        assert!(cli.little_endian);
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
        let err = cli.encoding(&source).unwrap_err();
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
        let err = cli.encoding(&source).unwrap_err();
        assert!(err.to_string().contains("--binary requires --input"));
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
}
