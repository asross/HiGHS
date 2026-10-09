#!/bin/bash
# Fast per-change check of a HIGHS_RUST build against a pure C++ build
# (minutes; see "Verification tiers" in rust/PORTING.md):
#   rust/bench/quick_check.sh [CPP_BUILD] [RUST_BUILD]
# Exit status 0 iff everything matches, so it also drives `git bisect run`.
set -u
cd "$(dirname "$0")/../.."
CPP=${1:-build}; RS=${2:-build-rust}
MIPLIB=${MIPLIB_DIR:-$HOME/code/miplib}
fail=0
(cd rust && cargo test --release -q 2>&1 | grep -q "test result: ok") || { echo "cargo test FAILED"; fail=1; }
tmp=$(mktemp -d)
printf "threads = 1\nrandom_seed = 0\nmip_rel_gap = 0.01\nmip_max_nodes = 200\n" > $tmp/mip.opts
filt() { grep -E "iterations:|Objective value|Model status|^  Nodes|^  LP iterations|Primal bound|Dual bound"; }
n=0
for f in 25fv47 80bau3b greenbea perold stair afiro adlittle etamacro israel shell woodinfe gas11; do
  for o in "" "--presolve off"; do
    a=$($CPP/bin/highs $o check/instances/$f.mps 2>&1 | filt); b=$($RS/bin/highs $o check/instances/$f.mps 2>&1 | filt)
    n=$((n+1)); [ "$a" == "$b" ] || { echo "LP DIFF $f $o"; fail=1; }
  done
done
for f in $MIPLIB/{air05,neos17,nu25-pr12,neos-911970,gen-ip002}.mps.gz dispatch_milp_2026_09_30/lambda_20260930_080458_22776806.mps check/instances/{egout,bell5,rgn}.mps; do
  [ -f "$f" ] || continue
  a=$($CPP/bin/highs --options_file $tmp/mip.opts "$f" 2>&1 | filt); b=$($RS/bin/highs --options_file $tmp/mip.opts "$f" 2>&1 | filt)
  n=$((n+1)); [ "$a" == "$b" ] || { echo "MIP DIFF $(basename $f)"; fail=1; }
done
rm -rf $tmp
api=$(bash rust/bench/api_compare.sh $CPP $RS 2>&1 | tail -1); echo "$api" | grep -q identical || { echo "API: $api"; fail=1; }
cli=$(bash rust/bench/cli_compare.sh $CPP $RS 2>&1 | tail -1); echo "$cli" | grep -qE " (0|1) differ" || { echo "CLI: $cli"; fail=1; }
echo "quick_check: $n solves compared, $( [ $fail = 0 ] && echo OK || echo FAILED )"
exit $fail
