#!/bin/bash
# Runs the highs app of a pure C++ and a HIGHS_RUST build on the command
# lines of cli_cases.txt ($I: check/instances, $C: check), each in a fresh
# directory, and diffs stdout, stderr, exit code and written files, with
# times masked.
#   rust/bench/cli_compare.sh [CPP_BUILD] [RUST_BUILD]
cd "$(dirname "$0")/../.."
ROOT=$PWD
CPP=${1:-build-cpp}; RS=${2:-build-rs}
I=$ROOT/check/instances; C=$ROOT/check
OUT=$(mktemp -d)
mask() { sed -E '/^Strange: /d; s/[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?s( |$)/Ts\3/g; s/(run time *: *).*/\1T/; s/(git hash: |Githash )[0-9a-f]+/\1H/; /[Tt]ime|Timing|^Thread |sub-solver|integral|^ +[0-9.]+ \(|%\)|Sub-MIP|simplex \(|IPX \(|Total|TOTAL|^MIP  /{ s/[0-9][0-9.e+-]*/N/g; s/ +/ /g; }'; }
n=0; fail=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n+1))
  for b in "$CPP" "$RS"; do
    d="$OUT/$n/$(basename "$b")"; mkdir -p "$d"
    printf 'presolve = off\nthreads\n' > "$d/bad.set"
    printf 'log_dev_level = 3\nhighs_debug_level = 1\n' > "$d/dev.set"
    printf 'output_flag = false\n' > "$d/quiet.set"
    printf 'log_file = my.log\nlog_to_console = false\n' > "$d/logfile.set"
    printf 'write_presolved_model_file = p.mps\n' > "$d/presolved.set"
    (cd "$d" && eval "set -- $line" && "$ROOT/$b/bin/highs" "$@" > stdout 2> stderr; echo $? > exit)
    for f in "$d"/*; do mask < "$f" | sed "s|$ROOT/$b/bin/highs|HIGHS|g" > "$f.m"; mv "$f.m" "$f"; done
  done
  if ! diff -r "$OUT/$n/$(basename "$CPP")" "$OUT/$n/$(basename "$RS")" > "$OUT/$n.diff"; then
    fail=$((fail+1)); echo "DIFF [$n] $line"; head -20 "$OUT/$n.diff"
  fi
done < rust/bench/cli_cases.txt
echo "cli: $n command lines, $fail differ"
if [ -n "$KEEP" ]; then echo "$OUT"; else rm -rf "$OUT"; fi
