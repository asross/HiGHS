#!/bin/bash
# Runs the highs app of a pure C++ and a HIGHS_RUST build (and, given a
# third argument, the Rust crest binary: a build with bin/crest, or the
# binary itself, e.g. rust/target/release/crest of cargo) on the command lines
# of cli_cases.txt ($I: check/instances, $C: check), each in a fresh
# directory, and diffs stdout, stderr, exit code and written files against
# the C++ app, with times and the program path masked. A case marked
# `# dropped: ...` uses a feature Crestline leaves out (HiPO, debugging):
# it is expected to differ, and is counted apart.
#   rust/bench/cli_compare.sh [CPP_BUILD] [RUST_BUILD] [CREST_BUILD]
cd "$(dirname "$0")/../.."
ROOT=$PWD
# A path as given if absolute, else under the repository root
ab() { case $1 in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
CPP=$(ab "${1:-build-cpp}"); RS=$(ab "${2:-build-rs}")
BINS=("$CPP/bin/highs" "$RS/bin/highs")
if [ -n "$3" ]; then
  if [ -f "$3" ]; then BINS+=("$(ab "$3")"); else BINS+=("$(ab "$3")/bin/crest"); fi
fi
I=$ROOT/check/instances; C=$ROOT/check
OUT=$(mktemp -d)
mask() { sed -E '/^Strange: /d; s/[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?s( |$)/Ts\3/g; s/(run time *: *).*/\1T/; s/(git hash: |Githash )[0-9a-f]+/\1H/; /[Tt]ime|Timing|^Thread |sub-solver|integral|^ +[0-9.]+ \(|%\)|Sub-MIP|simplex \(|IPX \(|Total|TOTAL|^MIP  /{ s/[0-9][0-9.e+-]*/N/g; s/ +/ /g; }'; }
# $F: solution and basis files to read, written by the C++ app
F=$OUT/files; mkdir -p "$F"
(cd "$F"
 "$CPP/bin/highs" --solution_file adl.sol --write_basis_file adl.bas --presolve off "$I/adlittle.mps" > /dev/null
 "$CPP/bin/highs" --solution_file egout.sol "$I/egout.mps" > /dev/null
 printf 'write_solution_style = 4\n' > sparse.set
 "$CPP/bin/highs" --options_file sparse.set --solution_file adl_sparse.sol "$I/adlittle.mps" > /dev/null
 { echo "=obj= 1"; sed -n '/^# Columns/,/^# Rows/p' adl.sol | sed '1d;$d' | head -20; } > adl_miplib.sol
 { echo "=obj= 1"; echo "NOSUCHCOL 3"; } > bad_miplib.sol
 head -12 adl.sol > adl_short.sol
 sed 's/^v2$/v1/' adl.bas | awk '!/^# /{print $NF; next} {print}' | sed 's/^v1$/HiGHS v1/' > adl_v1.bas
 sed '3,$s/^C/X/' adl.bas > adl_badname.bas
 sed '1s/v2/v7/' adl.bas > adl_v7.bas
 gzip -c "$I/afiro.mps" > afiro.mps.gz
 gzip -c "$I/qptestnw.lp" > qptestnw.lp.gz
 printf 'not gzip data' > bad.mps.gz
 : > empty.mps)
n=0; fail=0; dropped=0
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
    printf 'iis_strategy = 6\nwrite_iis_model_file = iis.lp\n' > "$d/iis.set"
    printf 'iis_strategy = 2\nwrite_iis_model_file = iis.mps\n' > "$d/iis2.set"
    for k in -1 2 3; do printf 'write_solution_style = %s\n' $k > "$d/style$k.set"; done
    printf 'mip_improving_solution_file = imp.sol\n' > "$d/improving.set"
    printf 'mip_improving_solution_file = imp.sol\nmip_improving_solution_report_sparse = true\n' > "$d/improving2.set"
    printf 'write_matrix_image = true\n' > "$d/image.set"
    printf 'write_presolved_model_file = p.lp\n' > "$d/presolved_lp.set"
    printf 'mps_parser_type_free = false\n' > "$d/fixed.set"
    (cd "$d" && eval "set -- $line" && "$b" "$@" > stdout 2> stderr; echo $? > exit)
    for f in "$d"/*; do mask < "$f" | sed "s|$b|HIGHS|g" > "$f.m"; mv "$f.m" "$f"; done
    if [ "$k" -gt 0 ] && ! diff -r "$OUT/$n/0" "$d" > "$OUT/$n.$k.diff"; then
      if [[ $line == *"# dropped"* ]]; then
        dropped=$((dropped+1)); echo "EXPECTED [$n] $b: $line"
      else
        fail=$((fail+1)); echo "DIFF [$n] $b: $line"; head -20 "$OUT/$n.$k.diff"
      fi
    fi
  done
done < rust/bench/cli_cases.txt
echo "cli: $n command lines x $((${#BINS[@]} - 1)), $fail differ, $dropped dropped-feature cases differ as expected"
if [ -n "$KEEP" ]; then echo "$OUT"; else rm -rf "$OUT"; fi
