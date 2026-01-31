# num-chrunchr

**num-chrunchr** is a Rust toolkit for **factoring and structurally analyzing extremely large integers**, including numbers that do **not** fit comfortably in RAM.

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

set -x RUST_LOG "num-chrunchr=debug,info"
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

Factoring becomes infeasible for truly enormous numbers. So num-chrunchr will also produce a structure report with:

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

Phase 4: GPU acceleration

Primary GPU target: batch remainder scanning:

    test divisibility by huge batches of primes efficiently

    keep remainders on-device, stream chunks from disk

    CPU fallback always available

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
```
