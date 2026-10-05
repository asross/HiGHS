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
  (`objdump -d build/lib/libhighs.dylib`) when unsure: it is per compiled
  copy, not per source line. A reduction loop `s += a * b` that clang
  interleaves by 4 runs its main part unfused and its remainder loop fused
  (see `HVec::norm2`, `compute_dual_for_tableau_column`), and the same
  source inlined elsewhere may be compiled differently (`norm2_fused`).
- **Safe Rust by default.** Slices, not raw pointers, outside the `extern "C"`
  shims; every pointer crosses the FFI with its length. `unsafe` only with a
  measured win and a comment saying why it is sound. `rust/.cargo/config.toml`
  turns off LLVM's runtime loop unrolling, which slows the short sparse
  loops that clang leaves rolled.
- **C++ switch.** `HIGHS_RUST` is defined in `HConfig.h`, so every
  translation unit sees the same class layouts.
- **HEkk's data** is reached through `EkkView` (rust/src/simplex/ekk.rs,
  see its module comment), filled per call by `HEkk::rustView()`.
- **Rust owns ported state.** A ported class keeps its C++ header as a thin
  wrapper around an opaque Rust handle until its callers are ported.
- **Tests:** `cargo test` for each module, and the C++ unit tests
  (`ctest`) pass with `-DHIGHS_RUST=ON`.
- **Measure** cycles (not wall time) against the C++ build on the same
  instances; record results in the commit message.

Build: `cmake -B build-rust -DCMAKE_BUILD_TYPE=Release -DHIGHS_RUST=ON &&
cmake --build build-rust -j8`.

## The dual simplex driver

HEkkDual::solve for the serial strategy runs in Rust (simplex/dual.rs)
when no simplex analysis, timing or debugging is asked for
(HEkkDual::rustEligible); SIP and PAMI stay C++. What it still calls in
C++ is listed in the `DualCallbacks` of dual.rs: INVERT with backtracking,
the primal clean-up, the infeasibility proof and dual ray, the quad
precision refinement of a pivotal row, and logging/reports. In the common
case an iteration makes no call into C++. HEkkDual.cpp is compiled in a
unity build, so clang inlines e.g. HVector::norm2 into chooseRow
contracted: check each compiled copy.
