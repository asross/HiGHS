#!/bin/bash
# Fast per-change check of a HIGHS_RUST build (and, given, the cargo-built
# crest binary) against a pure C++ build (minutes; see "Verification
# tiers" in rust/PORTING.md):
#   rust/bench/quick_check.sh [CPP_BUILD] [RUST_BUILD] [CREST_BINARY]
# Exit status 0 iff everything matches, so it also drives `git bisect run`.
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD
ab() { case $1 in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
CPP=$(ab "${1:-build}"); RS=$(ab "${2:-build-rust}"); CREST=${3:+$(ab "$3")}
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
    if [ -n "$CREST" ]; then
      c=$($CREST $o check/instances/$f.mps 2>&1 | filt)
      n=$((n+1)); [ "$a" == "$c" ] || { echo "crest LP DIFF $f $o"; fail=1; }
    fi
  done
done
for f in $MIPLIB/{air05,neos17,nu25-pr12,neos-911970,gen-ip002}.mps.gz dispatch_milp_2026_09_30/lambda_20260930_080458_22776806.mps check/instances/{egout,bell5,rgn}.mps; do
  [ -f "$f" ] || continue
  a=$($CPP/bin/highs --options_file $tmp/mip.opts "$f" 2>&1 | filt); b=$($RS/bin/highs --options_file $tmp/mip.opts "$f" 2>&1 | filt)
  n=$((n+1)); [ "$a" == "$b" ] || { echo "MIP DIFF $(basename $f)"; fail=1; }
  if [ -n "$CREST" ]; then
    c=$($CREST --options_file $tmp/mip.opts "$f" 2>&1 | filt)
    n=$((n+1)); [ "$a" == "$c" ] || { echo "crest MIP DIFF $(basename $f)"; fail=1; }
  fi
done
# crest: the LP files (LPs, QPs, MIPs, semi-variables) and gzip-compressed
# models, the whole output compared (times masked)
if [ -n "$CREST" ]; then
  mask() { sed -E 's/[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?s( |$)/Ts\3/g; s/(hash: )[0-9a-f]+/\1H/; /[Tt]ime|Timing|integral|%\)|Total|^ +[0-9.]+ \(/{ s/[0-9][0-9.e+-]*/N/g; s/ +/ /g; }'; }
  gzip -c check/instances/25fv47.mps > $tmp/25fv47.mps.gz
  gzip -c check/instances/qptestnw.lp > $tmp/qptestnw.lp.gz
  for f in $ROOT/check/instances/*.lp $tmp/25fv47.mps.gz $tmp/qptestnw.lp.gz; do
    for o in "" "--presolve off"; do
      a=$(cd $tmp && $CPP/bin/highs $o --options_file mip.opts "$f" 2>&1 | mask)
      c=$(cd $tmp && $CREST $o --options_file mip.opts "$f" 2>&1 | mask)
      n=$((n+1)); [ "$a" == "$c" ] || { echo "crest DIFF $(basename $f) $o"; fail=1; }
    done
  done
fi
rm -rf $tmp
api=$(bash rust/bench/api_compare.sh $CPP $RS 2>&1 | tail -1); echo "$api" | grep -q identical || { echo "API: $api"; fail=1; }
cli=$(bash rust/bench/cli_compare.sh $CPP $RS $CREST 2>&1 | tail -1); echo "$cli" | grep -qE " (0|1) differ" || { echo "CLI: $cli"; fail=1; }
echo "quick_check: $n solves compared, $( [ $fail = 0 ] && echo OK || echo FAILED )"
exit $fail
