# Power-of-2 Base Optimizations Roadmap

## Goals
- Make nearest-power queries for bases `2^m` asymptotically cheaper than general bases.
- Preserve correctness for all encodings (decimal/binary input, endianness).
- Keep observability high (tracing metrics for the fast path and fallbacks).

## Detection
- [x] Detect `base` as an exact power of 2 (single set bit) using `base.bits()` + check `base & (base - 1) == 0`.
- [x] Record `m = log2(base)` as the index of the set bit.
- [x] Add a trace field to indicate `fast_path=power_of_two` and `m` when active.

## Exponent Selection (Fast Path)
- [x] Compute `k_floor = floor((bit_length(N) - 1) / m)` with integer arithmetic only.
- [x] Compare `N` to `2^(m*k_floor)` and `2^(m*(k_floor+1))` using bit-length and one high-bit construction (no big exponentiation loops).
- [x] Select nearest exponent (tie -> lower exponent) with only two comparisons.
- [x] Guard against `k_floor+1` overflow of `u32` and clamp if needed.

## Power Construction
- [x] Build `power = 1 << (m*k)` using BigUint bit-shift, not repeated multiplication.
- [x] Ensure shift uses `u64`/`usize` safely for very large `m*k`.
- [x] Add unit tests verifying equivalence to `base.pow(k)` for small cases.

## Delta Computation
- [x] Compute `delta = |N - power|` as today, but avoid extra clones where possible.
- [x] For `N` close to `power`, use early-exit compare to avoid full subtraction when possible.

## Iterative Near-Power (`--n-times`)
- [x] Reuse fast path for each delta iteration when base is `2^m`.
- [x] Ensure `remaining` strictly decreases unless exact (to guarantee convergence).
- [x] Add tests that successive iterations correspond to top-set-bit extraction for base 2.

## Logging & Metrics
- [x] Per-iteration tracing fields: `fast_path`, `m`, `k_floor`, `k_candidate`, `comparison_count` (expected 2).
- [x] Aggregate tracing fields: `fast_path_used_count`, `total_comparisons`, `total_shift_bits`.
- [x] Verify existing `power_percent`, `percent_delta`, `coverage_percent` remain consistent.

## Correctness Edge Cases
- [x] Handle `N = 0` (expect exponent 0, power 1, delta 1).
- [x] Handle `base = 2` (`m = 1`) without special casing.
- [x] Handle extremely large `m*k` where shift length exceeds `u32` but fits `usize`.
- [x] Ensure behavior for `base` not power-of-2 stays on current general path.

## Performance Validation
- [x] Add benchmark comparing general path vs fast path for base `2`, `4`, `8`, `16` on large binary inputs.
- [x] Validate that `exponents_checked` shrinks to constant for the fast path.
- [x] Record runtime and memory stats in `reports/` for large sample inputs.

## Cleanup
- [x] Document fast path in `README.md` (near-power section).
- [x] Add a brief note in `docs/` describing why power-of-2 bases are trivial.
