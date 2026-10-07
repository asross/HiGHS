#!/bin/bash
# Runs the highs app of a pure C++ and a HIGHS_RUST build (and, given a
# third build with bin/crest, the Rust crest binary) on the command lines
# of cli_cases.txt ($I: check/instances, $C: check), each in a fresh
# directory, and diffs stdout, stderr, exit code and written files against
# the C++ app, with times and the program path masked.
#   rust/bench/cli_compare.sh [CPP_BUILD] [RUST_BUILD] [CREST_BUILD]
cd "$(dirname "$0")/../.."
ROOT=$PWD
CPP=${1:-build-cpp}; RS=${2:-build-rs}
BINS=("$CPP/bin/highs" "$RS/bin/highs")
[ -n "$3" ] && BINS+=("$3/bin/crest")
I=$ROOT/check/instances; C=$ROOT/check
OUT=$(mktemp -d)
mask() { sed -E '/^Strange: /d; s/[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?s( |$)/Ts\3/g; s/(run time *: *).*/\1T/; s/(git hash: |Githash )[0-9a-f]+/\1H/; /[Tt]ime|Timing|^Thread |sub-solver|integral|^ +[0-9.]+ \(|%\)|Sub-MIP|simplex \(|IPX \(|Total|TOTAL|^MIP  /{ s/[0-9][0-9.e+-]*/N/g; s/ +/ /g; }'; }
# $F: solution and basis files to read, written by the C++ app
F=$OUT/files; mkdir -p "$F"
(cd "$F"
 "$ROOT/$CPP/bin/highs" --solution_file adl.sol --write_basis_file adl.bas --presolve off "$I/adlittle.mps" > /dev/null
 "$ROOT/$CPP/bin/highs" --solution_file egout.sol "$I/egout.mps" > /dev/null
 printf 'write_solution_style = 4\n' > sparse.set
 "$ROOT/$CPP/bin/highs" --options_file sparse.set --solution_file adl_sparse.sol "$I/adlittle.mps" > /dev/null
 { echo "=obj= 1"; sed -n '/^# Columns/,/^# Rows/p' adl.sol | sed '1d;$d' | head -20; } > adl_miplib.sol
 { echo "=obj= 1"; echo "NOSUCHCOL 3"; } > bad_miplib.sol
 head -12 adl.sol > adl_short.sol
 sed 's/^v2$/v1/' adl.bas | awk '!/^# /{print $NF; next} {print}' | sed 's/^v1$/HiGHS v1/' > adl_v1.bas
 sed '3,$s/^C/X/' adl.bas > adl_badname.bas
 sed '1s/v2/v7/' adl.bas > adl_v7.bas)
n=0; fail=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n+1))
  for k in "${!BINS[@]}"; do
    b=${BINS[$k]}
    d="$OUT/$n/$k"; mkdir -p "$d"
    printf 'presolve = off\nthreads\n' > "$d/bad.set"
    printf 'log_dev_level = 3\nhighs_debug_level = 1\n' > "$d/dev.set"
    printf 'output_flag = false\n' > "$d/quiet.set"
    printf 'log_file = my.log\nlog_to_console = false\n' > "$d/logfile.set"
    printf 'write_presolved_model_file = p.mps\n' > "$d/presolved.set"
    printf 'write_solution_style = 1\nranging = on\n' > "$d/ranging.set"
    (cd "$d" && eval "set -- $line" && "$ROOT/$b" "$@" > stdout 2> stderr; echo $? > exit)
    for f in "$d"/*; do mask < "$f" | sed "s|$ROOT/$b|HIGHS|g" > "$f.m"; mv "$f.m" "$f"; done
    if [ "$k" -gt 0 ] && ! diff -r "$OUT/$n/0" "$d" > "$OUT/$n.$k.diff"; then
      fail=$((fail+1)); echo "DIFF [$n] $b: $line"; head -20 "$OUT/$n.$k.diff"
    fi
  done
done < rust/bench/cli_cases.txt
echo "cli: $n command lines x $((${#BINS[@]} - 1)), $fail differ"
if [ -n "$KEEP" ]; then echo "$OUT"; else rm -rf "$OUT"; fi
