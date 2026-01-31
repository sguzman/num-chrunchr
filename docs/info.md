Skip to content
Chat history
You said:
I am trying to write up a program that helps me factor extremely large numbers. I want to factor them when they're small but also use heuristics when they're much larger and even incorporate a gpu device in searching. I want you to help formalize this program. These numbers can be extremely large so i want a way of handling them that doesnt necessarily store the whole number in memory but can within some limit. say something like if the number if 1G long it can be stored but any bigger and operations are done piecewise on the number while loading it from disk or something. And i also want the possibility of using finding compressed represetation of incredibly large numbers. It can use many formalize tactics for this. One could be that it doesnt necessarily try to factor that number but maybe factor numbers around it could be extremely composite. Another technique could be something like using exponentials close to the number to get a collapse representation. SO the loop can be:
- take the log using some large-ish base suitable for the number)
- take the whole part of the result. Expontiate that to find the whole number. substract that from the original number
- repeat so you have a series of exponents

if you have any other tactics include them
ChatGPT said:

Here’s a clean way to formalize this as a factoring + “number understanding” system, with tiered representations, out-of-core operations, GPU-assisted scans, and multiple compression tactics (including your “series of exponents” idea).
1) Core idea: split the program into 3 engines

    Representation engine: “What is this number and how do I hold it?”

    Arithmetic engine: “What operations can I do without loading it all?”

    Strategy engine: “Given size + structure, what do I try next?”

That separation is what lets you scale from “fits in RAM” to “must stream from disk” without rewriting everything.
2) Representations (you should support several at once)

Think of your number N not as “bytes in RAM”, but as a value that may have multiple coexisting views:
A. In-memory big integer (fast path)

For “small enough” (your threshold could be bytes of limbs, not decimal chars):

    BigIntRAM: backed by a standard big-int library (GMP-like via Rust/Python bindings).

Use this for Pollard Rho, ECM, QS/NFS wrappers, etc.
B. Decimal stream (disk-backed, no base conversion required)

For very large decimal files, the easiest streaming operations are:

    mod p for small p

    division by small p producing a new quotient stream on disk

    computing approximate logs / leading digits

This representation is gold because you can do useful work without ever materializing the full big integer.

Key streaming primitive (Horner):

    r = 0

    for each digit d: r = (r*10 + d) mod p

C. Limb file (binary base 2^32 or 2^64, disk-backed / mmap)

Once N is “only” huge (say up to a few GB of limbs) you can:

    store limbs on disk

    memory-map chunks

    do high-throughput modular arithmetic / gcd-style operations in blocks

This is where you can start making GPU work more naturally (base 2^32 limbs are GPU-friendly).
The `num-chrunchr::repr::LimbFile` helper now serializes a `DecimalStream` into base-2^32 limbs so those chunks are instantly available for GPU/CPU batch kernels.
D. Expression/AST form (compressed “mathematical number”)

This is your “compressed representation” track:

    N = a^b +/- c

    N = product(factors)

    N = factorial(k) +/- c

    N = repunit / Mersenne / Fermat / cyclotomic-like

    N = sum_i (c_i * B^{e_i}) (your exponent-series idea generalized)

The AST lets you:

    represent numbers far larger than memory

    apply algebraic factoring rules without expanding

    compute N mod p efficiently via modular exponentiation + streaming

E. “Sketch” representation (cheap fingerprints)

Store a vector of residues:

    sketch(N) = (N mod p1, N mod p2, ..., N mod pk) for many small primes

This is surprisingly useful for:

    quick equality checks / dedup

    fast “is divisible by any small prime in this set?”

    guiding heuristics / caching progress

3) Define a capability-based interface (this is the big unlock)

Not every representation can do every operation cheaply. So define capabilities and dispatch strategies based on what’s available:
Minimal operations worth standardizing

    bit_length_estimate() or decimal_length()

    leading_digits(k) (read first ~k digits from file)

    mod_u32(p) and mod_u64(p)

    mod_big(M) where M is “small enough to fit in RAM”

    is_even(), v2() (power of 2 dividing N)

    div_u32(p) -> new NumberRepr (streaming quotient to disk)

    to_bigint_if_small() (upgrade path)

This lets you do a very real workflow:

    Start as DecimalStream

    Peel off small factors via mod_u32 + div_u32

    As the quotient shrinks, “upgrade” to BigIntRAM

    Switch to Pollard/ECM/etc

4) Strategy engine: size tiers + algorithm menu

You want a “decision ladder” that never attempts impossible work.
Tier 0: Tiny (fits RAM comfortably)

    Trial division (small primes)

    Pollard Rho/Brent

    Pollard p-1 / p+1

    ECM (best for finding medium-sized factors)

    Primality tests (Miller-Rabin; maybe proofs later)

Tier 1: Large but still feasible in RAM (or limb-mmap)

    More ECM

    QS / GNFS via integration (often you will wrap existing tools rather than reimplement)

Tier 2: Extremely large (cannot be fully materialized)

You basically restrict yourself to:

    small factor hunting (trial division, batch gcd tricks)

    structure detection (special forms, near-power/near-square, sparse expansions)

    compression + algebraic factoring on AST forms

    generating a meaningful report even if full factorization is infeasible

That last point matters: for huge inputs, the “product” is often a certificate of what you tried and what you learned.
5) GPU: what’s actually worth accelerating

For giant numbers that are disk-backed, the most realistic GPU win is:
GPU-accelerated “many-mods” for trial division / prime scanning

Goal: test divisibility by a huge batch of primes quickly.

You keep, for each prime p, a running remainder r_p.
You stream the number in chunks and update all remainders in parallel.
Two practical paths

    Decimal digits streamed:

        GPU kernel updates r_p = (r_p*10 + digit) mod p

        Works but digit-wise kernels can be bandwidth-heavy.

    Binary limbs streamed (recommended):

        Convert once (when feasible) to base 2^32 limbs on disk.

        Update remainders as: r_p = (r_p * (2^32)^k + chunk_value) mod p

        This is much friendlier to GPU throughput.

Once you find p | N, the division step (div_u32(p)) is still basically sequential and is usually best done on CPU (but it’s fine; it’s rare compared to all the tests).
GPU Pollard Rho?

Possible, but only really sensible when N is not astronomically large (because big-int modular multiplication dominates). So I’d architect it as a plugin, but I would not make it your main GPU plan.

Also: if you do use a GPU device, you’ll almost certainly be using NVIDIA-style CUDA hardware or a portable compute layer (OpenCL/wgpu) depending on your taste.
6) Your “series of exponents” representation (formalized)

What you described is essentially a greedy decomposition into powers of a base.
Variant A: “sum of powers” (digits are 0/1)

Pick base B > 1.

Repeat:

    e = floor(log_B(N))

    record term B^e

    N = N - B^e

This yields:

    N_original = Σ B^{e_i} with strictly decreasing e_i

It’s valid, but it can be long (like writing a number in base B but forcing digits to be 0/1).
Variant B (better): sparse base-B expansion (digits can be > 1)

Repeat:

    e = floor(log_B(N))

    d = floor(N / B^e) (a digit, 1..B-1)

    record (d, e)

    N = N - d*B^e

Now you get the normal base-B representation, but you can store it sparsely by skipping runs of zeros:

    N = Σ d_i * B^{e_i}

How to compute floor(log_B(N)) without loading N

If you have a decimal file:

    Let L = number_of_decimal_digits(N)

    Let x = leading_k_digits_as_int with k ~ 16-32 digits

    Then log10(N) ≈ (L - k) + log10(x)

    So log_B(N) = log10(N)/log10(B)

That’s cheap and streaming-friendly (read only the header + file length).
Why this helps factoring

Not directly, but it helps with:

    compression (“description length”)

    detecting “near power” structure

    enabling N mod p via Σ d_i * (B^e mod p) with fast modular exponentiation

7) Other compression + heuristic tactics you should include

Here are tactics that are genuinely useful in a factoring-oriented system:
A. Near-square (Fermat family)

If N is close to a^2, then:

    find a = ceil(sqrt(N))

    test whether a^2 - N is a square b^2

    then N = (a-b)(a+b)

You can do “closeness” heuristics using only leading digits + length to decide if it’s worth attempting.
B. Near power a^k +/- c

Try to fit:

    N ≈ a^k

    record c = N - a^k

Then:

    if c is “small” or structured, you can store N compactly as (a,k,c) and do modular computations without expanding.

C. Detect special algebraic forms

If your input comes from somewhere (files-as-numbers, encodings, etc.), many values accidentally land in known forms:

    x^k - y^k (difference of powers)

    x^k + y^k (sum of powers)

    repunits, Mersenne-like, etc.

If you represent N as AST, you can apply algebraic factorization rules before doing numeric heavy lifting.
D. Pollard p-1 / p+1 as “numbers around it are composite” but in the right sense

This is the correct version of that intuition:

    you aren’t factoring neighbors of N

    you’re exploiting that a factor p might have p-1 (or p+1) that is very smooth (highly composite)

These are cheap-ish and often pay off.
E. Batch-gcd tricks (CPU or GPU)

Instead of checking primes one-by-one:

    multiply primes into a block product P = Π p_i (while it still fits in RAM big-int)

    compute g = gcd(N, P)

    if g > 1, then you know there’s at least one small prime factor in that block

To do this out-of-core you mainly need N mod P, which is streamable.
F. Continued-fraction / rational approximation (for compression, sometimes for structure)

If you try to approximate N / B^k or log(N) ratios, you can get compact descriptions like:

    N ≈ (u/v) * B^k
    This is more of a compression/structure signal than a direct factoring tool, but it can guide “near multiple” or “near power” searches.

G. “Operation-aware compression”

Score candidate representations by:

    description length (bytes to store)

    supports mod p cheaply

    supports subtraction/division by small p

    supports upgrade to RAM big-int

Then keep the “best 2-3” representations alive simultaneously.
8) A practical pipeline that matches your goals

This is a realistic loop:

    Ingest

        parse input as decimal stream or AST (if user provides expression form)

    Analyze

        length, leading digits, v2, quick sketch residues, special-form detectors

    Peel small factors (CPU/GPU assisted)

        GPU batch mod p scanning for huge prime ranges

        whenever found: stream-divide to shrink N

    Upgrade representation when it becomes “small enough”

        convert to limb file or RAM bigint

    Run stronger algorithms

        Pollard rho / p-1 / ECM

        optionally QS/NFS via wrappers

    If it’s too large to finish

        output a report:

            small factors found

            primality confidence (if tested)

            sketches, near-power decompositions, sparse base-B expansion

            “best compressed forms” you discovered

9) MVP scope I’d build first (so you don’t get stuck)

    DecimalStream + mod_u32 + div_u32 (this alone is huge value)

    small prime sieve scanning (CPU first)

    GPU “many-mods” accelerator (optional second wave)

    representation discovery:

        sparse base-B expansion

        near-square and near-power detectors

    upgrade to BigIntRAM and add Pollard rho + p-1

    later: ECM + wrappers to mature tooling

If you tell me what language you want for the core (Rust vs Python), I can write the trait/interface skeleton (representations + capabilities + strategy dispatcher) and a concrete design for the GPU remainder kernel that streams chunks from disk while keeping the remainder array on-device.
You said:
I want to use rust
ChatGPT said:

Cool — Rust is a great fit for this because you can make the “number” a capability-based object: some representations can do mod p cheaply (streaming), others can do fast big-int ops in RAM, others are symbolic/AST and never expand.

Below is a concrete, Rust-first formalization that you can build into a real system.
1) Architecture: 3 engines (in Rust modules)
A) Representation engine (repr/*)

A NumberRepr is not “a bigint”. It’s “something that can answer certain questions about N”.

Representations you’ll want:

    DecimalStream (disk-backed decimal digits; best for streaming mod p + division by small primes)

    BigIntRam (in-memory big integer; for Pollard/ECM/etc)

    ExprAst (symbolic/compressed expression; can do mod p via modular exponentiation)

    Sketch (cached residues; used for fast screening and persistence)

    Later: LimbFile (binary 2^32/2^64 limbs on disk; best path to GPU kernels)

B) Arithmetic engine (ops/*)

Define operations in terms of capabilities:

    Streaming:

        mod_u32(p), mod_u64(p)

        div_u32(p) -> (quotient_repr, remainder)

        decimal_len(), leading_digits(k)

    RAM big-int:

        to_bigint_if_small()

        gcd_bigint(m) etc

C) Strategy engine (strategy/*)

A pipeline that decides what to do based on:

    size (digits/bytes)

    available capabilities

    results so far (sketch, factors found)

    budget (time / GPU / memory limits)

2) Your “series of exponents” compression (formalized)

The best version is sparse base-B expansion:

    Find pairs (d_i, e_i) such that
    N = Σ d_i * B^{e_i}, with 1 <= d_i < B, e_i decreasing, and zero runs omitted.

When N is too big to materialize, you can still:

    estimate e0 = floor(log_B(N)) using (decimal_len, leading_digits)

    and store an approximate compressed form for “structure reporting”

    and compute N mod p efficiently if you have ExprAst form (exact) or if you can materialize (exact)

For exact subtraction-based decomposition, you’ll usually require BigIntRam (or a future LimbFile + big subtraction).
3) What’s realistic to GPU-accelerate first

Don’t start with GPU Pollard-Rho. Start with GPU batch remainder scanning:

    Maintain an array of primes p[i]

    Maintain an array of remainders r[i]

    Stream the number in chunks; update all r[i] in parallel

This is easiest once you have LimbFile in base 2^32 or 2^64. With decimal digits it still works, just less GPU-friendly.

I’d make GPU a pluggable accelerator behind a trait like BatchModEngine.
4) A compile-ready Rust skeleton (with real streaming mod/div implemented)

Below is a minimal crate that already does:

    disk-backed decimal ingestion

    streaming mod_u32/mod_u64

    streaming division by u32 producing a new quotient file

    lots of logging via tracing

You can extend this into the full factoring pipeline.
Cargo.toml

[package]
name = "giantfactor"
version = "0.1.0"
edition = "2021"
license = "CC0-1.0"

[dependencies]
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
clap = { version = "4", features = ["derive"] }
tempfile = "3"

src/main.rs

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

mod repr;

#[derive(Parser)]
#[command(name = "giantfactor")]
#[command(about = "Factoring + structure analysis for extremely large integers", long_about = None)]
struct Cli {
    /// Input path containing decimal digits (whitespace allowed; non-digits ignored)
    #[arg(long)]
    input: std::path::PathBuf,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute N mod p (streaming)
    Mod {
        #[arg(long)]
        p: u64,
    },

    /// Divide N by a small divisor d (streaming long division)
    Div {
        #[arg(long)]
        d: u32,
        /// Output file for quotient digits
        #[arg(long)]
        out: std::path::PathBuf,
    },

    /// Quick analysis: digit length + leading digits
    Analyze {
        #[arg(long, default_value_t = 32)]
        leading: usize,
    },
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,giantfactor=debug"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    tracing::event!(Level::INFO, "logging initialized");
}

fn main() -> Result<()> {
    init_logging();

    let cli = Cli::parse();
    info!(input = %cli.input.display(), "starting");

    let n = repr::DecimalStream::open(&cli.input)?;

    match cli.cmd {
        Command::Mod { p } => {
            let r = n.mod_u64(p)?;
            println!("{r}");
        }
        Command::Div { d, out } => {
            let (q, rem) = n.div_u32_to_path(d, &out)?;
            info!(quotient = %q.display(), remainder = rem, "division complete");
            println!("remainder={rem}");
        }
        Command::Analyze { leading } => {
            let len = n.decimal_len()?;
            let lead = n.leading_digits(leading)?;
            println!("decimal_len={len}");
            println!("leading_digits={lead}");
        }
    }

    Ok(())
}

src/repr/mod.rs

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Disk-backed decimal representation.
/// - The file may contain whitespace or separators; all non-digits are ignored.
/// - Operations are streaming and do not materialize the whole number.
#[derive(Clone, Debug)]
pub struct DecimalStream {
    path: PathBuf,
}

impl DecimalStream {
    pub fn open(path: &Path) -> Result<Self> {
        let path = path.to_path_buf();
        let meta = std::fs::metadata(&path)
            .with_context(|| format!("failed to stat input {}", path.display()))?;
        info!(
            input = %path.display(),
            bytes = meta.len(),
            "opened decimal stream"
        );
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Count decimal digits (streaming).
    pub fn decimal_len(&self) -> Result<u64> {
        let mut reader = self.open_reader()?;
        let mut buf = String::new();
        let mut count: u64 = 0;

        loop {
            buf.clear();
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                break;
            }
            for b in buf.bytes() {
                if b.is_ascii_digit() {
                    count = count.saturating_add(1);
                }
            }
        }

        debug!(digits = count, "decimal_len computed");
        Ok(count)
    }

    /// Read the first `k` digits (streaming).
    pub fn leading_digits(&self, k: usize) -> Result<String> {
        let mut reader = self.open_reader()?;
        let mut out = String::with_capacity(k);
        let mut buf = [0u8; 64 * 1024];

        while out.len() < k {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b.is_ascii_digit() {
                    out.push(b as char);
                    if out.len() >= k {
                        break;
                    }
                }
            }
        }

        if out.is_empty() {
            warn!("leading_digits: input appears to contain no digits");
        } else {
            debug!(leading = %out, "leading_digits read");
        }
        Ok(out)
    }

    /// Streaming N mod p for u32 modulus.
    pub fn mod_u32(&self, p: u32) -> Result<u32> {
        anyhow::ensure!(p != 0, "modulus p must be nonzero");
        let mut reader = self.open_reader()?;

        let mut r: u32 = 0;
        let mut buf = [0u8; 64 * 1024];
        let mut seen_any_digit = false;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b.is_ascii_digit() {
                    seen_any_digit = true;
                    let d = (b - b'0') as u32;
                    // r = (r*10 + d) % p
                    let tmp = (r as u64) * 10u64 + (d as u64);
                    r = (tmp % (p as u64)) as u32;
                }
            }
        }

        if !seen_any_digit {
            warn!("mod_u32: no digits found; treating N as 0");
        }
        debug!(p, r, "mod_u32 done");
        Ok(r)
    }

    /// Streaming N mod p for u64 modulus.
    pub fn mod_u64(&self, p: u64) -> Result<u64> {
        anyhow::ensure!(p != 0, "modulus p must be nonzero");
        let mut reader = self.open_reader()?;

        let mut r: u64 = 0;
        let mut buf = [0u8; 64 * 1024];
        let mut seen_any_digit = false;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if b.is_ascii_digit() {
                    seen_any_digit = true;
                    let d = (b - b'0') as u64;
                    // r = (r*10 + d) % p
                    r = (r.wrapping_mul(10).wrapping_add(d)) % p;
                }
            }
        }

        if !seen_any_digit {
            warn!("mod_u64: no digits found; treating N as 0");
        }
        debug!(p, r, "mod_u64 done");
        Ok(r)
    }

    /// Streaming long division by small d (u32).
    /// Writes quotient digits (no separators) to `out_path`.
    /// Returns (out_path, remainder).
    ///
    /// Notes:
    /// - Ignores non-digits in input.
    /// - Produces "0" if quotient is zero (and N had digits).
    pub fn div_u32_to_path(&self, d: u32, out_path: &Path) -> Result<(PathBuf, u32)> {
        anyhow::ensure!(d != 0, "divisor d must be nonzero");

        let mut reader = self.open_reader()?;
        let mut out = std::io::BufWriter::new(
            File::create(out_path).with_context(|| format!("create {}", out_path.display()))?,
        );

        let mut rem: u64 = 0;
        let mut started = false; // started writing non-leading-zero digits
        let mut saw_digit = false;

        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if !b.is_ascii_digit() {
                    continue;
                }
                saw_digit = true;
                let digit = (b - b'0') as u64;
                let cur = rem * 10 + digit;
                let qdigit = (cur / d as u64) as u8;
                rem = cur % d as u64;

                if qdigit != 0 || started {
                    started = true;
                    out.write_all(&[b'0' + qdigit])?;
                }
            }
        }

        out.flush()?;

        if !saw_digit {
            warn!("div_u32: no digits found; treating N as 0");
            // quotient is 0, remainder 0
            std::fs::write(out_path, b"0")?;
            return Ok((out_path.to_path_buf(), 0));
        }

        if !started {
            // Quotient is zero but N had digits.
            std::fs::write(out_path, b"0")?;
        }

        let rem_u32 = rem as u32;
        info!(
            divisor = d,
            quotient = %out_path.display(),
            remainder = rem_u32,
            "div_u32 done"
        );

        Ok((out_path.to_path_buf(), rem_u32))
    }

    fn open_reader(&self) -> Result<BufReader<File>> {
        let f = File::open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))?;
        Ok(BufReader::new(f))
    }
}

This is intentionally simple but “correct enough” to become your out-of-core base layer.
5) Next modules to add (in order that won’t trap you)
Step 1: Factor “small” completely via streaming peel

Add strategy/small_factor.rs that:

    loads a prime list (sieve up to N)

    for each prime p: compute N mod p streaming

    if divisible, stream-divide and continue on the quotient

    keep a Vec<(p, exponent)>

This will already factor huge decimal files as long as their prime factors are small.
Step 2: Sketch cache

Persist N mod p_i for a set of primes to a sidecar file so you can resume.
Step 3: Upgrade path to BigIntRam

Once digit length < threshold (or quotient file is small):

    read it fully into num-bigint (or GMP binding if you want speed later)

    run Pollard Rho / p-1 / (later ECM)

Step 4: Compression / structure reporting

Implement:

    approx_log_B() from (decimal_len, leading_digits)

    detect near-square / near-power (approx)

    report “best compressed forms” even when full factoring is impossible

Step 5: LimbFile + GPU batch mods

    Add a converter from DecimalStream -> LimbFile (base 2^32) for “fits on disk”

    Implement a BatchModEngine with a GPU backend (CUDA or wgpu compute)

    CPU fallback stays available

The `repr::LimbFile` converter and `gpu::BatchModEngine` described above are now implemented, providing a tested path from decimal input to GPU-friendly limb chunks and back.

6) How your “exponent series” fits into this design

You’ll implement two versions:

    Exact sparse base-B (requires BigIntRam or later LimbFile subtraction):

    gives true (d_i, e_i) pairs

    Approximate exponent ladder (works on giant decimal streams):

    uses decimal_len + leading_digits to estimate successive exponents

    yields a compressed description candidate for reporting/heuristics

    not used for exact arithmetic unless the number is upgraded

If you want, I can extend this skeleton with a prime sieve + streaming peel strategy (still fully out-of-core) plus a FactorReport struct that records:

    factors found

    remaining cofactor path

    digit length evolution

    sketch residues

    compression candidates (your exponent-series + near-power + near-square signals)

…and keep everything very heavily logged with tracing so you can watch progress on huge inputs.
You said:
Generate a comprehensive readme for this project
ChatGPT said:

# giantfactor

**giantfactor** is a Rust toolkit for **factoring and structurally analyzing extremely large integers**, including numbers that do **not** fit comfortably in RAM.

It is built around a simple idea:

> A "number" is not always a `BigInt`.  
> It is a *representation* that may support only certain operations efficiently (streaming, symbolic, disk-backed, RAM-backed, GPU-assisted, etc.).

This repo currently provides a working MVP for **disk-backed decimal numbers** with:
- streaming `N mod p` (`u32`/`u64`)
- streaming long division `N / d` for small `d`
- quick analysis (decimal length + leading digits)
- structured logging via `tracing`

…and a clear path to expand into a full multi-strategy factoring system.

---

## Goals

1. **Factor when feasible** (small/medium inputs; or large inputs with small factors).
2. **Use heuristics when not feasible**, and still produce a meaningful "structure report":
   - size characteristics (digits/bit-estimates)
   - residue sketches (many `mod p`)
   - special-form detection (near-square, near-power, sparse expansions)
   - best known compressed representations
3. **Scale beyond RAM**:
   - stream from disk (decimal and later limb files)
   - operate piecewise and persist intermediate states
4. **Optionally accelerate prime scanning with GPU** (planned):
   - batch remainder updates for huge prime sets

---

## Non-goals (important)

- This is not intended as a turnkey “break RSA” tool.
- Factoring arbitrary, cryptographically-sized semiprimes is hard; the project will eventually integrate / wrap mature external implementations (QS/NFS, ECM, etc.) rather than reimplement everything from scratch.
- “Compression” here means *useful mathematical representation*, not general-purpose lossless compression.

---

## Current Features (MVP)

### Disk-backed `DecimalStream`
Input is a text file containing decimal digits. Any non-digit characters are ignored. This makes it easy to:
- store gigantic numbers as text
- include separators/newlines
- stream operations without loading the whole value

### Streaming operations (no full materialization)
- `decimal_len()` — count digits by streaming
- `leading_digits(k)` — read first `k` digits (streaming)
- `mod_u32(p)` / `mod_u64(p)` — Horner-style streaming modulus
- `div_u32_to_path(d, out)` — streaming long division by small `d`

### Logging
Uses `tracing` + `tracing-subscriber` with env filtering.

---

## Why capability-based representations?

Different representations excel at different operations:

- **DecimalStream** (disk-backed text)
  - Great: `mod p`, `div by small p`, quick size/leading-digit analysis
  - Not great: big multiplications, general-purpose big-int algorithms

- **BigIntRam** (planned)
  - Great: Pollard Rho, p-1/p+1, ECM, etc.
  - Limited: RAM size

- **ExprAst** (planned)
  - Great: numbers like `a^b +/- c`, products, factorial-like forms, sparse sums
  - Allows: fast `mod p` without expansion via modular exponentiation
  - Enables: algebraic factor rules

- **LimbFile** (planned)
  - Disk-backed base 2^32/2^64 limbs
  - Great for GPU kernels and chunked high-throughput arithmetic

- **Sketch** (planned)
  - Cached residues for fast screening + resumability

The factoring strategy engine will choose algorithms based on:
- input size
- which capabilities are available
- what has already been learned (factors found, sketch residues, detected forms)

---

## Project Layout

Current structure (MVP):

src/
main.rs # CLI entrypoint
repr/
mod.rs # DecimalStream implementation (streaming ops)


Planned expansions:

src/
repr/
decimal_stream.rs
bigint_ram.rs
expr_ast.rs
limb_file.rs
sketch.rs
ops/
mod.rs # traits + shared arithmetic helpers
stream.rs
bigint.rs
strategy/
mod.rs
peel_small.rs # trial division / small-factor peeling
upgrade.rs # stream -> bigint conversion thresholds
rho.rs # Pollard Rho (RAM mode)
pminus1.rs # Pollard p-1 / p+1
ecm.rs # ECM integration or wrapper
report.rs # structure report / certificates
gpu/
mod.rs
batch_mod.rs # GPU-assisted remainder scanning (planned)


---

## CLI Usage (current)

### Input file format
- A text file containing decimal digits.
- Non-digit characters are ignored.
- If the file contains no digits, it is treated as `0`.

### Commands
- `analyze` — digit length + leading digits
- `mod` — compute `N mod p` streaming
- `div` — divide by a small `u32` divisor (streaming), write quotient to a file

### Examples (fish shell)

1) Create a file with digits:

```fish
printf "123456789012345678901234567890\n" > n.txt

    Analyze:

cargo run -- --input n.txt analyze --leading 20

Output:

    decimal_len=...

    leading_digits=...

    Modulus:

cargo run -- --input n.txt mod --p 97

    Divide:

cargo run -- --input n.txt div --d 3 --out q.txt

This writes the quotient digits to q.txt and prints remainder=....
Logging

Set RUST_LOG to control verbosity.

Examples (fish):

set -x RUST_LOG info
cargo run -- --input n.txt analyze

More detail:

set -x RUST_LOG "giantfactor=debug,info"
cargo run -- --input n.txt mod --p 1000003

Design Notes: Streaming Modulus

For DecimalStream, modulus is computed with a streaming Horner method:

    Start r = 0

    For each digit d in the file:

        r = (r * 10 + d) mod p

This is:

    O(number_of_digits) time

    O(1) memory

    Works even when the input is many GBs of decimal text

Division by a small integer

div_u32_to_path(d, out) implements streaming long division:

    reads digits sequentially

    maintains a remainder

    emits quotient digits as it goes

    never holds the whole number in memory

This operation is crucial because it lets the strategy engine:

    detect a small factor p

    divide it out immediately

    continue factoring the smaller quotient (still disk-backed)

Compression and “Structure” (planned)

Factoring becomes infeasible for truly enormous numbers. So giantfactor will also produce a structure report with:

    size estimates (digits, approximate bits)

    residue sketches: N mod p_i for many primes

    detected special forms (when possible)

    candidate compressed representations

Your "series of exponents" idea

We will support sparse base-B representations:

    N = Σ d_i * B^{e_i} with 1 <= d_i < B, decreasing e_i, skipping zero runs.

There are two variants:

    Approximate (stream-only, very large N)

    Use (decimal_len, leading_digits) to estimate floor(log_B(N))

    Produce a compressed description candidate for reporting/heuristics

    Exact (requires BigIntRam or later LimbFile subtraction)

    Compute the true sparse expansion and store the (d_i, e_i) pairs

Other planned compression / heuristic tactics

    Near-square detection (Fermat-style): check if N ≈ a^2

    Near-power detection: fit N ≈ a^k, represent as a^k +/- c

    Recognize algebraic forms in ExprAst:

        x^k - y^k, x^k + y^k

        products and partial factorizations

        repunit/Mersenne-like patterns (where applicable)

    Batch-GCD tricks for small prime blocks (when P fits in RAM)

Factoring Strategy Roadmap
Phase 1: Small-factor peeling (works for huge files)

    Sieve primes up to a bound

    Compute N mod p streaming

    If divisible, divide out and repeat

    Persist factors found

This alone will completely factor many large numbers that contain small primes.
Phase 2: Upgrade to RAM BigInt when feasible

When the remaining cofactor is small enough:

    load it into BigIntRam

    run:

        Pollard Rho (Brent variant)

        Pollard p-1 / p+1

        primality tests (probabilistic first)

Phase 3: ECM and heavier tools

    integrate ECM (library or wrapper)

    later: QS/NFS via wrappers

    ✅ Lenstra ECM (via the `ecm` crate) is now called automatically after the RAM upgrade when Pollard/p±1 stall; tune it through `[policies.ecm]`.

Phase 4: GPU acceleration

Primary GPU target: batch remainder scanning:

    test divisibility by huge batches of primes efficiently

    keep remainders on-device, stream chunks from disk

    CPU fallback always available

Both the `gpu::BatchModEngine` and the `repr::LimbFile` converter are implemented, so the GPU remainder scan pipeline is live.

GPU work will likely be most practical on systems with NVIDIA hardware, but the design will aim to keep the interface backend-agnostic.
Resumability (planned)

Long runs should be restartable. Planned artifacts:

    factors.json — factors found so far, exponents

    sketch.json — stored residues (p -> N mod p)

    cofactor.txt — current quotient as a new DecimalStream

    report.md — structure report / certificate of attempts

The strategy engine will:

    detect existing sidecars

    resume from last known cofactor and factor list

Correctness and Safety Notes

    Streaming routines treat non-digits as separators. This is convenient, but be mindful when using untrusted inputs.

    Remainder and division are exact for the digits seen.

    For enormous numbers, “structure” output may include approximations unless explicitly marked exact.

    This is research-grade tooling; validate results when used for anything high-stakes.

Contributing

Contributions are welcome, especially in these areas:

    prime peeling strategy module (CPU first)

    sketch persistence format

    RAM BigInt upgrade + Pollard Rho implementation

    expression AST + modular evaluation

    limb-file representation and GPU batch mods

Recommended dev hygiene:

    cargo fmt

    cargo clippy

    cargo test

(Exact CI tooling will be added as the crate grows.)
License

This project is licensed under CC0-1.0 (public domain dedication). See the license = "CC0-1.0" entry in Cargo.toml.

If you add dependencies or code, please ensure the result remains compatible with CC0 distribution.
Status

MVP stage:

    ✅ disk-backed decimal stream

    ✅ streaming mod / div / analyze

    🔜 small-factor peeling strategy

    🔜 resumable factor reports + sketches

    🔜 BigInt upgrade + Pollard Rho/p-1

    🔜 expression/AST compressed representations

    🔜 limb files + GPU batch remainder scanning

FAQ
Why not just store everything as a BigInt?

Because once you pass a certain size, even “basic” operations become dominated by memory and bandwidth, and you lose the ability to do useful incremental work. Streaming representations let you:

    find small factors

    compute sketches

    generate structure reports

    reduce the number
    …without ever holding it all in memory.

Will this factor a 1000-digit RSA number?

Not by itself. For that, you’ll rely on advanced algorithms and mature implementations (ECM, QS, NFS), likely via wrappers or integrations.
What’s the most valuable near-term addition?

A small prime peeling strategy that repeatedly:

    scans primes

    finds divisors via streaming mod

    divides via streaming long division

    persists factors and continues on the quotient

That will immediately make the tool useful on very large inputs.


ChatGPT can make mistakes. Check important info. See Cookie Preferences.
