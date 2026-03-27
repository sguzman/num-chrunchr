# Sequence Compression Roadmap

## Scope
- Add two compression modes:
  - `--compress-seqA`: pick a base `B` and express each exponent as `e_i = B + d_i`.
  - `--compress-seqB`: keep `e_0` and express deltas `Δe_i = e_i - e_{i-1}`.
- Add `--compress-scheme` selector for optimization objective.

## CLI + Config
- [ ] Add `--compress-seqA` (mutually exclusive with `--compress-seqB`).
- [ ] Add `--compress-seqB` (mutually exclusive with `--compress-seqA`).
- [ ] Add `--compress-scheme` with options:
  - `min-max-abs` (minimize max |delta|)
  - `min-total-abs` (minimize sum |delta|)
  - `min-digit-count` (minimize total digit count)
- [ ] Add `CompressionScheme` enum + clap ValueEnum.
- [ ] Validate compression flags only apply to `near-power`.
- [ ] Define default scheme (if none provided).

## Compression Mode A (Base + Deltas)
- [ ] Compute base `B` and deltas `d_i = e_i - B`.
- [ ] Implement scheme selection:
  - [ ] `min-max-abs`: choose `B` to minimize max |d_i|.
  - [ ] `min-total-abs`: choose `B` to minimize sum |d_i|.
  - [ ] `min-digit-count`: choose `B` to minimize total digit count of `B` + deltas.
- [ ] Emit compressed output format for seqA.
- [ ] Add stats: base, max |d|, sum |d|, digit count.

## Compression Mode B (Delta Sequence)
- [ ] Compute `e_0` and `Δe_i = e_i - e_{i-1}`.
- [ ] Implement scheme selection:
  - [ ] `min-max-abs`: minimize max |Δe_i| (if no reorder, this is fixed).
  - [ ] `min-total-abs`: minimize sum |Δe_i| (fixed without reorder).
  - [ ] `min-digit-count`: minimize total digit count of `e_0` + deltas.
- [ ] Decide whether sequence order is preserved (default yes).
- [ ] Emit compressed output format for seqB.
- [ ] Add stats: max |Δ|, sum |Δ|, digit count.

## Optimization Schemes
- [ ] `min-max-abs`: center to minimize max absolute delta.
- [ ] `min-total-abs`: median-based minimization.
- [ ] `min-digit-count`: minimize digit length (base + deltas).
- [ ] Optional additional schemes:
  - [ ] `min-bit-count`: minimize total bit-length of stored ints.
  - [ ] `min-varint-size`: minimize varint-encoded byte size.
  - [ ] `min-signed-avg`: bias around mean to reduce signed average magnitude.

## Output Format
- [ ] Define structured output (JSON or tagged lines).
- [ ] Include scheme name, base, deltas, and summary metrics.
- [ ] Preserve existing exponent list output if compression not enabled.

## Logging
- [ ] Per-run compression log: scheme, mode, base, metrics.
- [ ] Include compression metrics alongside existing near-power stats.

## Tests
- [ ] Unit tests for seqA base selection under each scheme.
- [ ] Unit tests for seqB deltas and scheme behavior.
- [ ] Golden output tests for small known sequences.

## Docs
- [ ] Update README with new flags.
- [ ] Add a short doc describing seqA vs seqB and schemes.
