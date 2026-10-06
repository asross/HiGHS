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
  Also, the loop vectorizer of the LTO build splits the fused multiply-add
  again in in-order reductions (`d += x[i]*y[i]`): the first n/block*block
  terms are rounded products, the tail is fused (see
  `ipx::utils::dot_blocked`); check each call site in the final library.
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

## The simplex solve

HEkkDual::solve for the serial strategy runs in Rust (simplex/dual.rs)
when no simplex analysis, timing or debugging is asked for
(HEkkDual::rustEligible); SIP and PAMI stay C++. What it still calls in
C++ is listed in the `DualCallbacks` of dual.rs: INVERT with backtracking,
the primal clean-up, the infeasibility proof and dual ray, the quad
precision refinement of a pivotal row, and logging/reports. In the common
case an iteration makes no call into C++. HEkkDual.cpp is compiled in a
unity build, so clang inlines e.g. HVector::norm2 into chooseRow
contracted: check each compiled copy.

## PDLP (cuPDLP-C)

rust/src/pdlp ports the CPU cuPDLP-C (highs/pdlp/cupdlp) and the glue of
CupdlpWrapper.cpp: building the cuPDLP-C LP, scaling, the PDHG iterations,
the hot start and the unscaled solution. With HIGHS_RUST the C sources are
not built and highs/pdlp/CupdlpWrapperRs.cpp reads the options, calls
`pdlp_rs_solve` and sets the model status; output goes through printf like
cupdlp_printf. All reductions of libhighs's cuPDLP-C are blocked by 4
(`dot_blocked`) except those of the power method, which clang leaves
scalar and fused. The power method's logged residual stops at nrows where
the C reads ax past its end when ncols > nrows.

## The QP solver

QUASS (highs/qpsolver/) runs in Rust (rust/src/qp): solveqp in
a_quass.cpp hands the Instance to `highs_rs_qp_solve` and maps the result
back with quass2highs. Phase 1 (computeStartingPointHighs, an LP solve by
Highs, or the hot start check), the timer and the logging are C++
callbacks; basis.cpp, quass.cpp, ratiotest.cpp and the unused
perturbation.cpp and scaling.cpp are not compiled. QpVector::dot is fused
in most compiled copies but split by 4 in SteepestEdgePricing and
Instance::objval (dot_split4). The C++ Cholesky factor writes past the
size of its std::vector once the null space was empty at a recompute;
cholesky.rs models the vector's capacity to follow it.

## Cut separation (MIP)

rust/src/mip/cuts ports HighsTransformedLp, HighsLpAggregator,
HighsCutGeneration, the path, tableau and mod-k separators and
HighsGFkSolve. With HIGHS_RUST, HighsTransformedLp's constructor (a
separation round, HighsSeparation::separationRound) builds a Rust
`SepaRound` (round.rs, see its module comment for what is viewed, what is
copied and what is read live), and the separators' separateLpSolution call
Rust (highs/mip/HighsSeparationRust.cpp); HighsLpAggregator is an empty
placeholder and HighsCutGeneration only serves generateConflict for the
search. Implications (getBestVub/Vlb, cleanupVarbounds), slack bounds, LP
rows, the cut pool, the node queue and the basis inverse rows are C++
callbacks (`Host`). Speed: the per-column data of the transformation is one
array of structs, the slack bounds are cached until the next addCut (which
can change the global domain), the LP rows are copied once per round, and
the work space lives in the separators. Orders that depend on the sorting
algorithm are kept with ports of pdqsort and libc++'s heap, partial_sort
and partition (sort.rs, checked against golden_cuts.cpp). clang fuses
`100 + 0.15 * n` and `1000 + 0.1 * n`; the path mixing violation loop is
split by 8 in the LTO build.
HighsSeparation's orchestration (propagation, LP resolves, clique and
implied bound separation, cut pool separation) stays C++.

HEkk::solve runs in Rust from initialiseForSolve down (simplex/hekk.rs,
dual.rs, primal.rs) when no simplex analysis, timing or debugging is
asked for and the strategy is serial dual or primal
(HEkk::rustSolveEligible); SIP and PAMI stay C++. HEkk's data stays
C++-owned: HEkk::solveRust (highs/simplex/HEkkRustSolve.cpp) sizes every
vector the solve could resize, fills a `CHekk` of views and pointers,
calls `highs_rs_ekk_solve`, then takes what Rust left for C++ vectors
(hot start record, primal phase 1 duals, ray values to clear) and does
returnFromEkkSolve. Rust calls C++ only through `Host`: log messages
(formatted in Rust with util/printf.rs, which matches C's printf), the
analysis iteration/rebuild reports, the run clock (once per solver with a
time limit), a user interrupt callback, and the rare rank deficient
initial basis. The factor's refactorization information and the saved
INVERT of putIterate/getIterate are held by the Rust factor.
HEkkDual.cpp is compiled in a unity build, so clang inlines e.g.
HVector::norm2 into chooseRow contracted: check each compiled copy.
