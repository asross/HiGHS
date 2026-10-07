#!/bin/bash
# Compares the Highs API of a pure C++ and a HIGHS_RUST build: builds
# api_driver.cpp against each library, runs it in a fresh directory and
# diffs its output (times masked) and the files it writes.
#   rust/bench/api_compare.sh [CPP_BUILD] [RUST_BUILD] [CXX]
set -e
cd "$(dirname "$0")/../.."
ROOT=$PWD
CPP=${1:-build-cpp}; RS=${2:-build-rs}; CXX=${3:-c++}
OUT=$(mktemp -d)
mask() { sed -E 's/(hash: )[0-9a-f]+/\1H/; s/ *[0-9]+(\.[0-9]+)?s( |$)/ Ts\2/g; /[Tt]ime|Timing|sub-solver|integral|^ +[0-9.]+ \(|%\)|Sub-MIP|simplex \(|IPX \(|Total|TOTAL|^MIP  /{ s/[0-9][0-9.e+-]*/N/g; s/ +/ /g; }'; }
for b in "$CPP" "$RS"; do
  d="$OUT/$(basename "$b")"
  mkdir -p "$d"
  $CXX $CXXFLAGS -std=c++17 -O1 -I"$b" -Ihighs -Iextern rust/bench/api_driver.cpp \
    -L"$b/lib" -lhighs -Wl,-rpath,"$ROOT/$b/lib" -o "$OUT/driver-$(basename "$b")"
  (cd "$d" && "$OUT/driver-$(basename "$b")" "$ROOT/check/instances" 2>&1 | mask > stdout.txt)
done
diff -r "$OUT/$(basename "$CPP")" "$OUT/$(basename "$RS")" && echo "api: identical ($(cat "$OUT/$(basename "$CPP")"/* | wc -l) lines)"
if [ -n "$KEEP" ]; then echo "$OUT"; else rm -rf "$OUT"; fi
