# Sequence Compression Roadmap

## Scope
- Add two compression modes:
  - `--compress-seqA`: pick a base `B` and express each exponent as `e_i = B + d_i`.
  - `--compress-seqB`: keep `e_0` and express deltas `Δe_i = e_i - e_{i-1}`.
- Add `--compress-scheme` selector for optimization objective.

## CLI + Config
- [x] Add `--compress-seqA` (mutually exclusive with `--compress-seqB`).
- [x] Add `--compress-seqB` (mutually exclusive with `--compress-seqA`).
- [x] Add `--compress-scheme` with options:
  - `min-max-abs` (minimize max |delta|)
  - `min-total-abs` (minimize sum |delta|)
  - `min-digit-count` (minimize total digit count)
- [x] Add `CompressionScheme` enum + clap ValueEnum.
- [x] Validate compression flags only apply to `near-power`.
- [x] Define default scheme (if none provided).

## Compression Mode A (Base + Deltas)
- [x] Compute base `B` and deltas `d_i = e_i - B`.
- [ ] Implement scheme selection:
  - [x] `min-max-abs`: choose `B` to minimize max |d_i|.
  - [x] `min-total-abs`: choose `B` to minimize sum |d_i|.
  - [x] `min-digit-count`: choose `B` to minimize total digit count of `B` + deltas.
- [x] Emit compressed output format for seqA.
- [x] Add stats: base, max |d|, sum |d|, digit count.

## Compression Mode B (Delta Sequence)
- [x] Compute `e_0` and `Δe_i = e_i - e_{i-1}`.
- [x] Implement scheme selection:
  - [x] `min-max-abs`: minimize max |Δe_i| (if no reorder, this is fixed).
  - [x] `min-total-abs`: minimize sum |Δe_i| (fixed without reorder).
  - [x] `min-digit-count`: minimize total digit count of `e_0` + deltas.
- [x] Decide whether sequence order is preserved (default yes).
- [x] Emit compressed output format for seqB.
- [x] Add stats: max |Δ|, sum |Δ|, digit count.

## Optimization Schemes
- [x] `min-max-abs`: center to minimize max absolute delta.
- [x] `min-total-abs`: median-based minimization.
- [x] `min-digit-count`: minimize digit length (base + deltas).
- [ ] Optional additional schemes:
  - [x] `min-bit-count`: minimize total bit-length of stored ints.
  - [x] `min-varint-size`: minimize varint-encoded byte size.
  - [ ] `min-signed-avg`: bias around mean to reduce signed average magnitude.

## Output Format
- [x] Define structured output (JSON or tagged lines).
- [x] Include scheme name, base, deltas, and summary metrics.
- [x] Preserve existing exponent list output if compression not enabled.

## Logging
- [x] Per-run compression log: scheme, mode, base, metrics.
- [x] Include compression metrics alongside existing near-power stats.

## Tests
- [x] Unit tests for seqA base selection under each scheme.
- [x] Unit tests for seqB deltas and scheme behavior.
- [x] Golden output tests for small known sequences.

## Docs
- [x] Update README with new flags.
- [x] Add a short doc describing seqA vs seqB and schemes.
