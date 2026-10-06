#!/bin/bash
# Builds a pure C++ and a HIGHS_RUST HiGHS with Homebrew gcc 14 (libstdc++),
# without FMA contraction, to compare the port with a gcc production build
# on a Mac (see the gcc / libstdc++ bullet of rust/PORTING.md).
#   rust/bench/gcc_builds.sh [CPP_DIR] [RUST_DIR]
set -e
cd "$(dirname "$0")/../.."
CPP=${1:-build-gcc}; RS=${2:-build-gcc-rust}
COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
  -DCMAKE_C_FLAGS=-ffp-contract=off -DCMAKE_CXX_FLAGS=-ffp-contract=off -DZLIB=OFF
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,-ld_classic -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-ld_classic)
cmake -B "$CPP" "${COMMON[@]}" > "$CPP.log" 2>&1
cmake -B "$RS" "${COMMON[@]}" -DHIGHS_RUST=ON "-DHIGHS_RUST_FEATURES=libstdcxx no_fma" > "$RS.log" 2>&1
cmake --build "$CPP" -j3 >> "$CPP.log" 2>&1 &
cmake --build "$RS" -j3 >> "$RS.log" 2>&1
wait
ls "$CPP/bin/highs" "$RS/bin/highs"
