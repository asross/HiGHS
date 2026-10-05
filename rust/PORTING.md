# Porting HiGHS to Rust

Bottom-up, one subsystem at a time, always shippable. The C++ build with
`-DHIGHS_RUST=ON` calls the Rust code for every ported piece; without it, the
original C++ runs. The C++ of a ported piece is deleted only once the Rust is
the sole caller's path.

Order: HFactor -> simplex (HEkk, HSimplexNla) -> presolve -> MIP -> top level;
file readers in parallel. IPX, PDLP and QP last.

## Rules for each step

- **Same paths.** Ported code does the same floating-point operations in the
  same order, so solves are bit-identical: same nodes and LP iterations as C++
  (bench: `cyc.py` compares). Clang contracts `a -= b * c` into a fused
  multiply-add on arm64; use `mul_add` there. Check the C++ disassembly
  (`objdump -d build/lib/libhighs.dylib`) when unsure.
- **Safe Rust by default.** Slices, not raw pointers, outside the `extern "C"`
  shims; every pointer crosses the FFI with its length. `unsafe` only with a
  measured win and a comment saying why it is sound.
- **Rust owns ported state.** A ported class keeps its C++ header as a thin
  wrapper around an opaque Rust handle until its callers are ported.
- **Tests:** `cargo test` for each module, and the C++ unit tests
  (`ctest`) pass with `-DHIGHS_RUST=ON`.
- **Measure** cycles (not wall time) against the C++ build on the same
  instances; record results in the commit message.

Build: `cmake -B build-rust -DCMAKE_BUILD_TYPE=Release -DHIGHS_RUST=ON &&
cmake --build build-rust -j8`.
