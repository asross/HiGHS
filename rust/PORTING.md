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
- **x86_64.** Production runs on x86_64, where the default C++ build has no
  FMA instructions and clang fuses nothing (and the blocked reductions are
  then plain in-order sums). So write every mirrored FMA as
  `x.mul_add_c(a, b)` (util/fma.rs): `mul_add` on aarch64, `x * a + b`
  elsewhere. A test rejects raw `mul_add`. Check x86_64 paths with
  `-DCMAKE_OSX_ARCHITECTURES=x86_64 -DHIGHS_RUST_TARGET=x86_64-apple-darwin`
  builds under Rosetta (`rust/bench/perf.py` on both).
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

## The MIP domain (HighsDomain)

HighsDomain keeps its C++ class and data (external code reads col_lower_,
col_upper_ and the stack everywhere); under HIGHS_RUST it runs in Rust
(mip/domain.rs, objprop.rs, conflict.rs) on a view of that data: changeBound
with the domain change stack, its reasons and previous bounds,
backtrack/backtrackToGlobal, setDomainChangeStack, the whole propagate()
loop (model rows, cuts, conflicts, objective), the activity updates with
infinity counts and capacity thresholds, ObjectivePropagation with its
red-black trees (HighsRbTree ported exactly: the trees are built by the C++
constructor and updated in Rust), conflict analysis (ConflictSet: the
frontiers are BTreeMaps by stack position) and tightenCoefficients. Still
C++: the clique table's and implications' fixings of a fixed binary (called
back through one function, which re-enters changeBound), the conflict and
cut pools (adding a conflict, resetAge), the pseudocosts and node queue
read by conflict analysis, getPropagationConstraint/getCutoffConstraint for
external callers, and the debug solution (HIGHS_DEBUGSOL is not supported).

The view (highs/mip/HighsDomainRust.h, HighsDomainRustView.h) is cached in
HighsDomain::rsView_ and refilled where a vector it holds by data() and
size() may move: copy, assignment, computeRowActivities,
setupObjectivePropagation, adding or clearing pools; cutAdded and
conflictAdded update their pool's part in place. The vectors that grow
during propagation are passed as the std::vector objects (StdVec, the
begin/end/capacity layout, checked at runtime) and appended to in place,
with a C++ reserve when full. Rust's Dom derefs to the view and indexes its
pointer+length pairs with bounds checks; the hot loops copy the pairs they
use into locals. Only Ctx::change_bound calls back into C++ code that
re-enters Rust; Dom views are borrowed from the Ctx, so none is alive
across it. The const methods (computeMin/MaxActivity,
propagateRowUpper/Lower, tightenCoefficients), which threads may call
concurrently on the global domain, get a small Bounds struct instead.

Rust-only shortcuts (same results, less work): the row propagation skips
entries whose implied bound is surely not tighter without dividing (the
C++ divides), computes the capacity threshold in the same pass, and the
objective propagation skips columns whose implied bound is surely
rejected, by a double estimate with a margin thousands of times its
rounding error.

## Postsolve

The undo side of HighsPostsolveStack runs in Rust (rust/src/presolve/
postsolve.rs): undo, undoPrimal, undoUntil, getReducedPrimalSolution,
compressIndexMaps and DuplicateColumn::okMerge. HPresolve (C++) still
records the reductions through the inline templates of the header, so the
C++ class keeps owning the HighsDataStack bytes, the reduction list and the
index maps; Rust reads them through a view (`PostsolveRsStack`) and parses
the records with the C++ byte layout (`#[repr(C)]` copies, sizes checked by
static_assert on both sides; HighsInt must be 32 bits). Rust only reads the
stack, so thread_safe undoPrimal needs no copy. C++ resizes the solution
and basis vectors to the original space; Rust does the rest. The fused
products in plain double are `x - a*d` of ForcingRow, `x + s*y` and
`v - s*y` of DuplicateColumn, and `x + s*y` of transformToPresolvedSpace.
