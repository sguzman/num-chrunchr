#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found" >&2
  exit 1
fi

INPUT=${1:-}
if [[ -z "$INPUT" ]]; then
  echo "usage: scripts/bench_power_of_two.sh <input-file>" >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "input file not found: $INPUT" >&2
  exit 1
fi

BIN=${BIN:-target/release/num-chrunchr}
if [[ ! -x "$BIN" ]]; then
  echo "building release binary" >&2
  cargo build --release
fi

run_case() {
  local base=$1
  local label=$2
  echo "== $label =="
  /usr/bin/time -f "elapsed_s=%e" "$BIN" --input "$INPUT" --binary near-power --base-number "$base" --n-times 1
}

run_case 2 "base=2"
run_case 4 "base=4"
run_case 8 "base=8"
run_case 16 "base=16"
