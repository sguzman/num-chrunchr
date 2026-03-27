# Power-of-2 Bases

When the base is a power of two (`b = 2^m`), nearest-power selection is much cheaper than the general case.

## Why it is fast
- Powers of `2^m` are just powers of two with exponents snapped to multiples of `m`.
- The best candidate exponent comes from the bit-length of the target, not full big-int arithmetic.
- Constructing `b^k` becomes a single bit-shift (`1 << (m*k)`) instead of repeated multiplication.

## Practical takeaway
- `base = 2` is the simplest case (every exponent is allowed).
- `base = 4, 8, 16, 256...` are still easy: you only consider every `m`-th power of two.
- For non-power-of-two bases, the algorithm falls back to the general comparison path.
