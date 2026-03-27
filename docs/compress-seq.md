# Sequence Compression

`near-power` can emit a compressed representation of the exponent sequence.

## Modes

- `--compress-seqA`: choose a base `B` and store deltas `e_i - B`.
- `--compress-seqB`: store `e_0` and deltas `e_i - e_{i-1}`.

## Schemes

Choose with `--compress-scheme`:

- `min-max-abs`: minimize max absolute delta.
- `min-total-abs`: minimize total absolute delta (default).
- `min-digit-count`: minimize total decimal digit count.
- `min-bit-count`: minimize total bit-length.
- `min-varint-size`: minimize total varint-encoded byte size.

## Output

Compression output is appended after the exponent list:

```
compress_mode=seqA scheme=MinTotalAbs base=20 deltas=[-10,0,10] max_abs_delta=10 total_abs_delta=20 total_digit_count=6
```

## Notes

- seqB preserves the original sequence order.
- Schemes operate on the exponent sequence only and do not change near-power computation.
