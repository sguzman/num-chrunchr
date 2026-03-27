#!/usr/bin/env fish

if not type -q cargo
  echo "cargo not found" >&2
  exit 1
end

set input $argv[1]
if test -z "$input"
  echo "usage: scripts/bench_power_of_two.fish <input-file>" >&2
  exit 1
end

if not test -f "$input"
  echo "input file not found: $input" >&2
  exit 1
end

set bin (set -q BIN; and echo $BIN; or echo target/release/num-chrunchr)
if not test -x "$bin"
  echo "building release binary" >&2
  cargo build --release
end

function run_case
  set base $argv[1]
  set label $argv[2]
  echo "== $label =="
  time "$bin" --input "$input" --binary near-power --base-number "$base" --n-times 1
end

run_case 2 "base=2"
run_case 4 "base=4"
run_case 8 "base=8"
run_case 16 "base=16"
