#!/bin/bash
# Compares option and info handling of a pure C++ and a HIGHS_RUST build:
# builds options_driver.cpp against each library, runs it in a fresh
# directory and diffs its output and the files it writes.
#   rust/bench/options_compare.sh [CPP_BUILD] [RUST_BUILD] [CXX]
set -e
cd "$(dirname "$0")/../.."
ROOT=$PWD
CPP=${1:-build-cpp}; RS=${2:-build-rs}; CXX=${3:-c++}
OUT=$(mktemp -d)
for b in "$CPP" "$RS"; do
  d="$OUT/$(basename "$b")"
  mkdir -p "$d"
  $CXX $CXXFLAGS -std=c++17 -O1 -I"$b" -Ihighs -Iextern rust/bench/options_driver.cpp \
    -L"$b/lib" -lhighs -Wl,-rpath,"$ROOT/$b/lib" -o "$d/driver"
  (cd "$d" && ./driver "$ROOT/check/instances" 2>&1 | sed -E 's/(hash: )[0-9a-f]+/\1H/' > stdout.txt)
  rm "$d/driver"
done
diff -r "$OUT/$(basename "$CPP")" "$OUT/$(basename "$RS")" && echo "options: identical ($(cat "$OUT/$(basename "$CPP")"/* | wc -l) lines)"
if [ -n "$KEEP" ]; then echo "$OUT"; else rm -rf "$OUT"; fi
