# Porting HiGHS to Rust

The Rust port will become **Crestline**: its own public repository
(github.com/asross/crestline, crate `crestline`, CLI `crest`) once the
command-line binary is pure Rust. Extraction plan: `git filter-repo
--subdirectory-filter rust` to keep history; a pinned HiGHS C++ commit as a
CI test oracle for bit-identical paths; MIT licence with the HiGHS
copyright notice and attribution kept; README states it is a Rust port of
HiGHS, bit-compatible with HiGHS v1.15. Until then development continues
on the `rust-port` branch here. The binary is already `crest` (see "The
app and crest"); the crate keeps its name `highs-rs` (staticlib
`highs_rs`, which CMake links) until the extraction renames it.

**Crestline's scope** (decided 2026-10-07): not ported, left out of
Crestline: HiPO (needs BLAS/METIS; IPX covers IPM) and HiPDLP (cuPDLP is
ported), SIP/PAMI parallel simplex (simplex_strategy 2/3), iCrash,
multi-objective solves, debug/analysis-only code (simplex analysis
reports, test_kkt, HighsDebugSol, debug checks), the C API and the
C#/Fortran/Julia shims (Crestline exposes a Rust API; Python bindings
maybe later), and the fixed-format MPS reader (free MPS and LP only).
Kept: everything else, including IIS and the QP solver. The HIGHS_RUST
build (and so crest) already leaves them out, see "Left out of the
HIGHS_RUST build" below.

**Roadmap after the extraction:** (1) the leading performance
opportunities (dense/supernodal switch in HFactor's triangular solves,
concurrent FTRANs per dual iteration, x86-specific tuning), evaluated on
the dispatch suite over 16 seeds; (2) primal feasibility: the MIPFEAS
benchmark (gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark),
and whether the methods of arXiv:2609.05954 and Local-MIP
(github.com/shaowei-cai-group/Local-MIP) add to feasibility jump and the
graph LNS. Note: Local-MIP fails badly on the dispatch MILPs, presumably
because their equality chains of continuous state-of-charge / ramp
variables defeat single-variable moves (as for feasibility jump); look
for methods that handle equality-coupled continuous structure.

Bottom-up, one subsystem at a time, always shippable. The C++ build with
`-DHIGHS_RUST=ON` calls the Rust code for every ported piece; without it, the
original C++ runs. The C++ of a ported piece is deleted only once the Rust is
the sole caller's path.

Order: HFactor -> simplex (HEkk, HSimplexNla) -> presolve -> MIP -> top level;
file readers in parallel. IPX, PDLP and QP last.

## Known divergences from the C++ (deliberate)

- **dualize (simplex_dualize_strategy on, not the default):** HEkk keeps the
  dual steepest-edge weights of the dualized LP across undualize, and the
  clean-up solve reads past the end of the weight vector (undefined
  behaviour in C++, a panic in Rust). The Rust build marks the weights
  invalid when dualizing and undualizing, so they are recomputed. Results:
  correct statuses (refinery and vol1 are Infeasible; the C++ says Unknown),
  other paths differ on 6 of 82 check instances with dualize and presolve
  off. With dualize and presolve on, the primal clean-up of vol1's
  reduced dual LP fails on a rank deficient basis; undualize then kept
  has_fresh_invert, so the solve from the undualized basis skipped INVERT
  and used a factor just set up for the primal LP's dimensions (the C++
  reads stale data and ends with Not Set; the Rust panicked in FTRAN).
  Setting up the simplex NLA now clears has_invert and has_fresh_invert,
  dualize clears them too, and when the dual LP's solve fails the primal
  LP is solved from a logical basis and that solve's status is returned:
  vol1 is Infeasible. With dualize on, presolve on or off, every check
  instance has the default run's model status, and no panics.

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
- **gcc / libstdc++.** Production builds the C++ with gcc, whose libstdc++
  differs from libc++ where results depend on the standard library: heap
  and partial_sort tie orders, std::tuple layout (hashed keys),
  uniform_int_distribution, unordered container order. Anything that
  mirrors one of these needs both variants, the libstdc++ one under the
  `libstdcxx` cargo feature (CMake turns it on with gcc;
  `HIGHS_RUST_FEATURES`). Goldens: build the golden .cpp with g++ too.
  Local gcc check on the Mac: `-DCMAKE_C_COMPILER=gcc-14
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_C_FLAGS=-ffp-contract=off
  -DCMAKE_CXX_FLAGS=-ffp-contract=off -DZLIB=OFF
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,-ld_classic
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-ld_classic`, and for the Rust build
  `-DHIGHS_RUST_FEATURES="libstdcxx no_fma"` (no_fma: arm64 without
  contraction, like x86_64). zlib off because its include path puts the
  SDK's math.h ahead of gcc's; ld_classic because the new Apple linker
  crashes on gcc 14 objects. Instances: ~/code/miplib/plain/*.mps.
- **Safe Rust by default.** Slices, not raw pointers, outside the `extern "C"`
  shims; every pointer crosses the FFI with its length. `unsafe` only with a
  measured win and a comment saying why it is sound. `rust/.cargo/config.toml`
  turns off LLVM's runtime loop unrolling, which slows the short sparse
  loops that clang leaves rolled.
- **C++ switch.** `HIGHS_RUST` is defined in `HConfig.h`, so every
  translation unit sees the same class layouts.
- **HEkk's data** is Rust's: an `LpSolver` (rust/src/simplex/lp_solver.rs,
  see "The simplex engine's data") owned by the C++ HEkk shell, with the
  LP being solved (a Rust copy). The kernels work on `EkkView`s
  (rust/src/simplex/ekk.rs) that LpSolver builds from its fields and its
  LP.
- **Rust owns ported state.** A ported class keeps its C++ header as a thin
  wrapper around an opaque Rust handle until its callers are ported.
- **Tests:** `cargo test` for each module, and the C++ unit tests
  (`ctest`) pass with `-DHIGHS_RUST=ON`.
- **Measure** cycles (not wall time) against the C++ build on the same
  instances; record results in the commit message.

Build: `cmake -B build-rust -DCMAKE_BUILD_TYPE=Release -DHIGHS_RUST=ON &&
cmake --build build-rust -j8`.

## The simplex solve

The dual and primal simplex run in Rust (simplex/dual.rs, primal.rs,
from HEkk::solve, see below); SIP and PAMI are left out (they run as the
serial dual). The C++ HEkkDual, HEkkPrimal, HEkkDualRow and HEkkDualRHS
are not built with HIGHS_RUST: they were kept only as the fallback for an
INVERT with a product form update (HSimplexNlaProductForm.cpp), which
nothing sets up any more (it served the deprecated frozen bases), so the
fallback and the product form are not built either. The HEkk methods only
they called are under `#ifndef HIGHS_RUST` (found by linking the app, with
every Highs method as a root, and the unit tests with -dead_strip at -O0),
as are those of HSimplexNla, HighsSimplexAnalysis, HFactor,
HighsSparseMatrix (the C++ price), HSimplex.cpp and HighsUtils.cpp (the
value distributions and scatter data of the analysis); HEkkInterface.cpp,
HSimplexReport.cpp, HSet.cpp (and TestHSet) and HighsLinearSumBounds.cpp
are not built.

The simplex logs are Rust (simplex/report.rs): HighsSimplexAnalysis's
iteration report (dev, verbose), INVERT report (dev) and user INVERT
report (the user log's iteration lines, every 5 s, or each line with
timeless_log), with the data the dual and primal record for them. The data
and header counters are a `SimplexReport` in HighsSimplexAnalysis
(`rs_report_`), kept over solves as the C++ fields were (a solve that
records nothing reports the previous values), reset by
HighsSimplexAnalysis::setup as before; HEkk::returnFromEkkSolve reads its
densities for the simplex stats. The run time is printed as " %.1fs" (the
NDEBUG format of the C++). The sorts of HighsSort.cpp are Rust
(util/sort.rs: the heap sorts, the decreasing heap, increasingSetOk,
sortSetData; hand-written, so the tie orders do not depend on the
standard library; golden_sort.cpp checks them); HighsSort.cpp is the
wrapper.

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

QUASS (highs/qpsolver/) runs in Rust (rust/src/qp), and so does its glue
(qp/glue.rs): Highs::callSolveQp up to the objective and KKT check (the
Hessian dimension check, the Instance with triangularToSquareHessian's
square Hessian, negated for a maximization, the Settings from the options
and their log lines), quass2highs, and phase 1 (computeStartingPointHighs:
the hot start check, of which only the infeasibility counts matter, or a
feasibility LP, then the starting active set). The C++ callSolveQp
(qpsolver/QpRust.cpp) passes views and does one `op` per step on a C++
object: the profiling clock, the timer, sizing the solution and basis, and
the phase 1 LP solved by a silent `Highs`; it then computes the objective
and KKT failures and calls checkOptimality as before. a_quass.cpp and
a_asm.cpp are empty under HIGHS_RUST; basis.cpp, quass.cpp, ratiotest.cpp
and the unused perturbation.cpp and scaling.cpp are not compiled.
QpVector::dot is fused
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
HighsSeparation's loop (separationRound and separate: the order of
propagation, LP resolves, clique and implied bound separation, the
separators, the cut pools, adding cuts, aging and the termination tests)
runs in Rust (mip/separation.rs); each step on a C++ object is one
callback (`CSepaFns::op`), the LP relaxation is called directly.

## The LP relaxation (MIP)

HighsLpRelaxation's state and logic are Rust (mip/lp_relaxation.rs): the
LP rows (model rows and cuts, ages, aging and deletion, the cut pools'
LP counts), run()'s status handling (error retries, the IPM basis after
an iteration limit, unbounded points), resolveLp (fractional integers by
the clique substitutions and symmetric branching columns, in a
HighsHashTable as the C++, the rounding along the locks, the age reset of
tight cuts, the repaired point), the dual proofs (Farkas ray and
objective bound), computeBasicDegenerateDuals, computeBestEstimate and
computeLPDegneracy. The C++ class keeps the `Highs` LP solver (the solve
itself with the solver choice and the IPX race, flushDomain, the stored
basis, the playground's putIterate/getIterate, the row deletions) and
reads the Rust status, objective, rows, fractional integers and proof in
place (`LpShared`); getFractionalIntegers returns a mutable span
(`HighsFracInts`), which the heuristics sort in place. Rust sees the LP
solver through a view refetched after each call that may change it
(`CLpView`), and the MIP data through `CLpMip`; the cut pools, the clique
table and the pseudocosts are used directly. Still C++ callbacks: the
domains (fixCol, bounds, tightenCoefficients, conflict reconvergence),
clique extraction from a proof, the symmetries' branching column,
incumbents and checkSolution, and logging. No product here is
contracted: each feeds a HighsCDouble call or a comparison (the
sparse vector sum keeps the double overload of add). The search reads
the LP relaxation directly (status, objective, iterations, fractional
integers, best estimate).

HEkk::solve runs in Rust from initialiseForSolve down (simplex/hekk.rs,
dual.rs, primal.rs; simplex analysis, debugging and SIP/PAMI are left out
of the build), on the Rust-owned `LpSolver` (see "The simplex engine's
data"): LpSolver::solve sizes every vector the solve could resize, builds
the `CHekk` of views and pointers into itself, runs the solve and takes
what it left (ray values to clear, saved edge weights taken); the C++
shell copies the hot start record and primal phase 1 duals the API
returns by reference and does the simplex stats of returnFromEkkSolve.
Rust calls C++ only through `Host`: log messages (formatted in Rust with
util/printf.rs, which matches C's printf), the run clock (once per solver
with a time limit), a user interrupt callback and the CHUZC failure
reports (dev log). The factor's refactorization information and the
saved INVERT of putIterate/getIterate are held by the Rust factor.

## The simplex engine's data (LpSolver)

Everything HEkk owned but its LP is Rust's: `LpSolver`
(rust/src/simplex/lp_solver.rs) holds the status flags, HighsSimplexInfo
(work, base and random vectors, the backtracking basis, densities,
counters, perturbation and cleanup state), the simplex basis, the edge
weights (with the saved weights carried over LP edits), the row-wise
partitioned matrix, the scaled copy of the constraint matrix, the Rust
factor with HFactor's set-up parameters (HSimplexNla and the C++ HFactor
shell are gone from HEkk), the saved iterate of putIterate/getIterate, the
dual-value reuse state, the ray records and the basis records. Its methods
are the rest of HEkk.cpp, HEkkControl.cpp and HSimplexNla: clear and
invalidate, updateStatus (with the DSE weights kept over basis changes and
row additions/deletions), moveLp's checks, initialiseEkk with
setSimplexOptions/initialiseControl and the random vectors, setBasis
(logical or from a HighsBasis), getHighsBasis, getSolution, unscaleSimplex,
getUnscaledInfeasibilities, addRows/deleteRows, the basis edits of the
Highs interface (appending nonbasic columns and basic rows, the flip for a
negative scale factor), the undualized basis, initialiseSimplexLpBasisAndFactor
(the rank deficient basis handling is hekk.rs `initial_rank_deficiency`),
the NLA solves (FTRAN/BTRAN with the basis matrix scaling of the NLA's LP),
put/getIterate, computeBasisCondition and proofOfPrimalInfeasibility.

LpSolver owns the LP being solved (`lp`, an `Lp` of rust/src/lp_data/
lp.rs: dimensions, costs, bounds, the column-wise matrix, sense, offset,
integrality, scaling, the model name). moveLp copies the C++ LP into it
(`Lp::import`; the vectors keep their capacity, so a re-solve allocates
nothing), and everything that moved or scaled the C++ LP now works on the
copy: solveLpSimplex (simplex/app.rs, the driver of HApp.h:
considerScaling, the alien basis check of formSimplexLpBasisAndFactor,
dualize, the scaled solve, the unscaled clean-up solve with scaled NLA,
the proof of infeasibility, copying the solution, basis and info), dualize
and undualize (app.rs; the matrix transposes and column appends are
sparse.rs) and formSimplexLpBasisAndFactor (lp_data/form_basis.rs). The
C++ LP is never moved or scaled any more: "moving it back" unscales the
copy and gives the C++ LP its scale factors (and, after an undualized
solve, the rebuilt matrix, as the C++ moved the dual LP's rebuilt matrix
back). Copying per solve is no slower than the C++, which scaled and
unscaled the LP in place on every solve (perf.py within noise, MIPs
included). Which LP the simplex NLA scales by is the engine's LP or a C++
LP (setNlaPointersForLpAndScale), the former decided when it is set as
HSimplexNla::scale_ was.

The C++ HEkk (highs/simplex/HEkk.h under HIGHS_RUST, HEkkRust.cpp) is a
shell: the pointers to the options, callback and timer, the analysis
(whose report data the simplex logs use), a C++ LP of the simplex NLA,
the factor's log options (copied at set-up, as HFactor did), the hot
start, primal phase 1 duals and simplex stats the API returns by
reference, and the steps of solveLpSimplex on C++ objects (`op`s: HEkk's
solve with the analysis set-up and simplex stats, setBasis from a
HighsBasis, the proof of infeasibility, the profiling clocks). The
scalars C++ code reads and writes (status_, info_'s objective values,
infeasibilities, densities, pivot threshold and edge weight strategy,
model_status_, iteration_count_, exit_algorithm_, the ray indices and
signs, dual_values_valid_) are `EkkShared`, a repr(C) struct inside the
Rust object that the shell refers to with HEkk's old member names; the
basis, edge weights, work arrays and ray values are reached through
accessors. Every call passes an `LpsEnv` of the option values the simplex
reads and the host functions; Rust adds the view of its LP.

The model modification interfaces of HighsInterface.cpp are Rust
(lp_data/interface.rs: addCols/addRows with the cost, bound and matrix
assessment, the scale factors of new columns and rows (considerCol/
RowScaling), the basis updates; deleteCols/deleteRows with the basis,
scale and mask updates; changing costs, bounds (sorting a set with its
data), integrality and a coefficient; scaling a column or row). They are
generic over the LP's and basis' vectors (`LpG`, `BasisG`, as `Mat` for
HighsSparseMatrix): a C++ HighsLp and HighsBasis edited in place (`CppLp`:
std::vectors that Rust resizes through C++, HighsRust.h RsLpVec), or
Rust-owned ones, so the LP relaxation can use the same code on Rust data.
The rest of the Highs object (names, the model status, solution and info,
the Hessian) is reached through `IfaceHost`; the simplex basis edits act
on the LpSolver directly. HighsSparseMatrix's layout changes, edits,
scalings and products are lp_data/sparse.rs, generic the same way
(`CppMat` for the C++ class's methods): ensureColwise/Rowwise,
exactResize, addVec, addCols, addRows, getRow, deleteCols, deleteRows,
createRowwise(Partitioned), applyScale/ColScale/RowScale, scaleCol/Row,
hasLargeValue, product, productTranspose, alphaProductPlusY, computeDot
(blocked by 4 as clang vectorizes it), collectAj and the double-double
products; the sparse productTransposeQuad (HighsSparseVectorSum) and the
assessment messages stay C++.

The LP part of a run is on Rust data too (lp_data/lp_run.rs): LpSolver
holds an `LpRun` with the solution, the HiGHS basis, HighsInfo, the model
status and PresolveComponent's data. When run.rs's `optimize_lp` reaches
the LP solve, the Highs object's solution, basis, info and model status
are copied in (`LpRustBegin`), the rest of calledOptimizeModel's LP branch
runs on a second `CHighs` whose pointers are the Rust data and whose `op`
(`lp_op`) makes the steps on that data in Rust (refineBasis, the KKT
checks, invalidating the basis, the presolve-to-empty solution, taking the
recovered solution and basis) and passes the rest to the Highs object;
callSolveLp's solvers (the unconstrained solve, IPX, PDLP, solveLpSimplex)
read and write it through the same views they had of the C++ objects, and
at the end it is copied back (`LpRustEnd`) before returnFromOptimizeModel.
LP presolve runs on the Rust LP (lp_data/lp_presolve.rs): the reduced LP
is an `Lp` (initialised from the model, the Rust presolve's host writes
the model and matrix back into it), the postsolve stack's storage is a
Rust `PostsolveStack` (the record bytes, the (type, position) pairs, the
index maps and linearly-transformable flags in HighsPostsolveStack's
layout; `flush` and `shrink` append and compress in Rust), the reduced LP
is solved through views of it (the simplex NLA then refers to the engine's
copy of it), prepared (setMatrixDimensions, cleanBounds, assessSmallValues)
and postsolved (undo on the Rust stack, the row values from the model, the
negated duals of a maximization) in Rust. The C++ PresolveComponent is
initialised as before (its reduced LP's other members come from the model)
and given the reduced LP, stack, status and log at the end of the run
(`highs_rs_lps_presolve_export`, HighsPostsolveStack::rustSet; the names
follow the index maps), for the API's index maps, presolved model and
postsolve.

Still C++ on the LP side: the Highs object's model LP (the C++ HighsLp
stays the authoritative model: Highs.cpp, presolve, the MIP solver, IIS,
user scaling, the semi-variable and infinite-cost modifications read or
write it directly, so the "handle" of the original plan waits for those
writers), the HighsOptions (the LP run reads them through the templates
the Highs object fills per solve, and the simplex shell's LpsEnv; the
mid-run changes, saving and restoring them for the primal simplex and the
clean-up solve, are on the C++ options because the HEkk shell reads them),
the HEkk shell, the dependent equations' HFactor of presolve, and the LP
relaxation's `Highs` object (see "What remains C++" of the MIP driver).
The next steps, in order: (3) HighsLpRelaxation uses LpSolver directly
(no `Highs`): model edits (interface.rs on its Rust LP), solves with
iteration limits, basis store/recover, get/putIterate, rays, basis
inverse rows, the IPX race with the Rust IPX; there the model LP lives in
Rust only, so no copy per solve. It needs the option values in Rust first
(a Rust copy that the HEkk shell's LpsEnv and the solve templates are built
from), the run path of run.rs without the Highs object's ops (the clocks,
the model LP view, the logs), and the ~130 getLpSolver() call sites of the
heuristics, search and separators rewritten onto a Rust LP solver handle;
(4) the C++ `Highs` keeps a handle for the API.

## The MIP domain (HighsDomain)

HighsDomain's own vectors are Rust's: a `DomainVecs` (mip/domain.rs) of
the bounds, bound positions, branching positions, changed columns, the
domain change stack with its reasons and previous bounds, the row
activities, infinity counts, thresholds and propagation flags, and the
scratch of propagate(). The C++ HighsDomain owns it (made, copied,
assigned, sized for the rows and freed by Rust calls) and its old members
(col_lower_, domchgstack_, ...) are references to the fields in place:
`HighsRsArray` (HighsRsSpan.h), the begin/end/capacity layout of Rust's
StdVec, which C++ reads, writes and shrinks and only Rust grows. The
getters return these arrays. The pools' propagation domains are Rust's
too: CutpoolPropagation and ConflictPoolPropagation are shells that own a
`CutPropState` / `ConfPropState` (activities, flags, thresholds, rows to
propagate; watched literals with their column lists), refer to its
vectors in place and register with their pool as before (copies under
the parallel lock only with a pool that is not the global one); the
pools' cutAdded / cutDeleted / conflictAdded / conflictDeleted hooks run in
Rust on the domain's bounds (`Bounds`) and the Rust pool, and the C++
shell refreshes the view afterwards (also when the cut is not taken: the
pool's matrix may have moved). So is the objective propagation's state
(`ObjPropState`, objprop.rs: the contributions with their red-black trees,
the partition cliques, the lower bound and threshold), built in Rust on
the domain's bounds and owned by the C++ ObjectivePropagation shell, whose
getPropagationConstraint forwards to Rust. The
C++ bodies of the ported domain code are not compiled under HIGHS_RUST. The domain runs in Rust
(mip/domain.rs, objprop.rs, conflict.rs) on a view of that data: changeBound
with the domain change stack, its reasons and previous bounds,
backtrack/backtrackToGlobal, setDomainChangeStack, the whole propagate()
loop (model rows, cuts, conflicts, objective), the activity updates with
infinity counts and capacity thresholds, ObjectivePropagation with its
red-black trees (HighsRbTree ported exactly), conflict analysis (ConflictSet: the
frontiers are BTreeMaps by stack position) and tightenCoefficients. Still
C++: the clique table's and implications' fixings of a fixed binary (called
back through one function, which re-enters changeBound), adding a
conflict and resetAge (through the C++ pool handles, see the
branch-and-bound section), getPropagationConstraint/getCutoffConstraint for
external callers, and the debug solution (HIGHS_DEBUGSOL is not supported).

The view (highs/mip/HighsDomainRust.h, HighsDomainRustView.h) is cached in
HighsDomain::rsView_ and refilled where a vector it holds by data() and
size() may move: copy, assignment, computeRowActivities,
setupObjectivePropagation, adding or clearing pools; cutAdded and
conflictAdded update their pool's part in place. The vectors that grow
during propagation are passed as the vector objects (StdVec) and appended
to in place with the Rust allocator (`rs_reserve`). Rust's Dom derefs to the view and indexes its
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

## The clique table and implications (MIP)

HighsCliqueTable and HighsImplications are Rust-owned (mip/clique.rs,
implications.rs); the C++ classes hold a handle (HighsCliqueTable.cpp and
HighsImplications.cpp under HIGHS_RUST, struct layouts in
HighsCliqueTableRust.h). Their vectors that C++ reads and clears
(substitutions, deleted rows, clique extensions) are `HighsRsVec` views
into the Rust vectors; the generator and the neighbourhood query counter
are used in place. The domain and the rest of the MIP solver (node
queue, pseudocosts, cut pool, the other table) are C++ callbacks
(`CDom`, `CMip`, `CImp`), with column bounds read through raw pointers.
A bound change re-enters addImplications/applyImplications, which only
read; the methods that change bounds hold no Rust borrow of the table
across a callback (`Ctx`, `ICtx`; see clique.rs). queryNeighbourhood is
serial (the C++ may split it over threads; its result is the same).
clang contracts `rhs -= val * bound` (extractCliques), `x * coef +
constant` and `1 + coef * coef` (getBestVub/Vlb), `m * c - f` and
`-m * a + t` (strengthenVarBound) and `s0 * v0 + s1 * v1` (implied bound
cuts). Debug-solution checks run only for the public addVUB/addVLB and
addClique.

## The branch-and-bound search (MIP)

HighsSearch, HighsNodeQueue, HighsPseudocost, HighsRedcostFixing,
HighsCutPool (with HighsDynamicRowMatrix) and HighsConflictPool run in
Rust (mip/search.rs, nodequeue.rs, pseudocost.rs, redcost.rs, cutpool.rs,
conflictpool.rs). Each C++ class is a handle (`rust()` gives the Rust
object) with its old API; HighsDynamicRowMatrix is a view of the Rust
pool's arrays (getMatrix() returns it by value) and the pools' entry and
rhs vectors are `HighsRsSpan`s, valid until the pool changes. The Rust
domain reads the conflict pool directly and gets the cut pool's arrays in
its view (refilled on cutAdded as before); conflict analysis uses the
pseudocosts and node queue directly. The pools tell their C++
propagation domains (CutpoolPropagation, ConflictPoolPropagation, still
C++) of added and deleted rows through callbacks that read the pool, so
the Rust holds no borrow across them. The thread safe calls of the cut
pool (resetAge, lpCutRemoved, increaseNumLps, separate) touch only atomics,
as in the C++.

The node queue's red-black trees and std::sets order nodes by keys that
end with the node index, so BTreeSets of the keys give the same orders
(doubles compared with `<`, so -0 == 0). The lurking bounds of reduced cost
fixing are std::multimaps where an element inserted at the hint
lower_bound(key) precedes its equal keys (in libc++ and libstdc++): a
decreasing sequence number in the key does the same (rootReducedCost sorts
them unstably by key, so the order matters). The cut pool's hash-to-cut
std::unordered_multimap is only searched for any match and erased by
value, so its group order (different in libstdc++) does not matter.

HighsSearch keeps the local domain (external code uses it), the LP
pointer and the conflict scratch; the node stack, statistics (C++ reads
nnodes etc. through references into the Rust struct) and branching state
are Rust's. The search calls C++ through `CSearchFns` for the domain
operations, the LP relaxation (strong branching's Playground is boxed; the
fallback LP of branch() is created and swapped in C++ in steps), the
symmetries, the conflicts from LP proofs (addBoundExceedingConflict,
addInfeasibleConflict), reduced cost fixing at a node, incumbents, limits
and logging. Node bases and stabilizer orbits are std::shared_ptrs boxed on
the C++ heap (`Shared`, cloned and freed by callbacks). No callback
re-enters the search. clang fuses `cost += (1 - w) * avg` and the score
sums of the pseudocosts, `avg * count + sum` of flushPseudoCost, `minrel -
r * (minrel - 1)` of branch, the reductions over a cut in the cut pool
(none vectorized), `0.5 * lb + 0.5 * estimate` of the node queue, and `1 -
10 * feastol` and `frac * redcost + lpobj` of addRootRedcost.

Still C++: the propagation domains of the pools, HighsCutSet filling
(separate returns the selected cuts), pruneInfeasibleNodes' domain loop,
setRINS/RENSNeighbourhood, checkLimits, and the implications' and
separators' node queue and pseudocost callbacks (which call the Rust
through the handles).

## Presolve (HPresolve)

HPresolve runs in Rust (rust/src/presolve/hpresolve) for LP and MIP
presolve and the MIP restarts; HPresolve.cpp is compiled only without
HIGHS_RUST, and HPresolveRust.cpp keeps okSetInput/run as a wrapper.
Rust owns the presolve state and a copy of the model, written back to the
C++ HighsLp (`sync_model`) before C++ reads it (shrinkProblem's MIP
rebuilds, probing, the end of run). Reductions are recorded in Rust in
the HighsDataStack layout (record.rs) and appended to the C++ stack by
`flush` (HighsPostsolveStack::rustAppend); the index maps are mirrored.
The LP presolve of a run on Rust data has a Rust host instead
(lp_data/lp_presolve.rs: the model and stack are Rust's, see "The simplex
engine's data").
Still C++ behind `Host` callbacks: logging, the timer, the HFactor of the dependent equations, and the C++ parts
of the MIP solver: the domain and clique setup of prepareProbing
(setupDomainPropagation, extractCliques), the start of finaliseProbing
(cleanupFixed, runCliqueMerging), the cut pool, the lifting opportunities
of probing (storeLiftingOpportunity) and the glue of
HighsImplications::runProbing. The probing loop of runProbing (probing.rs)
and the enumeration of enumerateSolutions (enumeration.rs) run in Rust on
the Rust clique table and implications (handles passed in `MipInfo`,
borrowed only between calls into C++) and on the global domain through
its view (`mip_env`: HighsDomain::rsView_, fetched after prepareProbing,
since shrinkProblem reassigns the domain); the binaries' and rows' sort
keys are distinct, so any sort gives pdqsort's order. Orders that depend on
containers are emulated: the libc++ unordered_multimap buckets of
detectParallelRowsAndCols keep their key groups (emplace_hint inserts
before the last visited element in libc++, after it in libstdc++: with
`libstdcxx`; this picked different parallel columns on co-100's LP, which
showed up as a different IPX starting point), the lifting opportunities'
unordered_map iterates in reverse order of first insertion (one bucket per
row; the same in libstdc++, which also puts a new bucket's node first),
the std::sets are sorted vectors / BTreeSet, and `rowpositions` keeps
stale entries past its length as the C++ vector does (loops over a stored
row read it live while nested reductions store other rows).

## Postsolve

The undo side of HighsPostsolveStack runs in Rust (rust/src/presolve/
postsolve.rs): undo, undoPrimal, undoUntil, getReducedPrimalSolution,
compressIndexMaps and DuplicateColumn::okMerge. HPresolve (C++) still
records the reductions through the inline templates of the header, so the
C++ class keeps owning the HighsDataStack bytes, the reduction list and the
index maps (except after the LP presolve of a run on Rust data, whose
stack is a Rust `PostsolveStack` copied to the C++ class at the end of the
run); Rust reads them through a view (`PostsolveRsStack`) and parses
the records with the C++ byte layout (`#[repr(C)]` copies, sizes checked by
static_assert on both sides; HighsInt must be 32 bits). Rust only reads the
stack, so thread_safe undoPrimal needs no copy. C++ resizes the solution
and basis vectors to the original space; Rust does the rest. The fused
products in plain double are `x - a*d` of ForcingRow, `x + s*y` and
`v - s*y` of DuplicateColumn, and `x + s*y` of transformToPresolvedSpace.

## Symmetry detection (HighsSymmetry)

HighsSymmetryDetection, HighsSymmetries, the orbitopes and the stabilizer
orbits run in Rust (rust/src/presolve/symmetry.rs); under HIGHS_RUST the
C++ classes hold a handle (HighsSymmetry.cpp, symmetry_ffi.rs) and
StabilizerOrbits keeps its three vectors, copied from Rust, for HighsSearch.
The detection task runs on the task scheduler (a C++ lambda): Rust polls
`checkInterrupt` through a callback at each leave and returns, and C++
rethrows HighsTask::Interrupt; the result depends only on the model (no
time or work limit, only the 64e6 / columns generator cap), so not on
thread timing. Orbital fixing and orbitopal propagation reach the domain
through `SymDom` (the clique table's CDom plus the model bounds of
isGlobalBinary, the branching positions and markInfeasible); orbitope
types query the Rust clique table directly. No standard-library variant is
needed: the hash tables (also the std::tuple-keyed leave graphs) are only
searched, never iterated, so their layout only changes hash values; the
refinement queue holds distinct cells, so any min-heap pops them in the
same order; libc++ and libstdc++ share std::partition's two-ended
algorithm; and the sorts with ties use the pdqsort port (whose heapsort
fallback switches with `libstdcxx`).

## Primal heuristics (MIP)

Feasibility jump runs in Rust (mip/feasjump.rs): HighsMipSolverData::
feasibilityJump (HighsFeasibilityJump.cpp) still builds the bounds,
initial point and row-wise matrix, then calls `highs_rs_feasibility_jump`
instead of extern feasibilityjump.hh, which only the C++ build compiles.
std::mt19937 and libc++'s uniform_real_distribution are ported; the effort
limits are the wrapper's (nnz << 10 in all, nnz << 8 since the last
improvement), and the dev log lines go back through a C++ callback. The
arithmetic is the C++'s (clang fuses the LHS updates, the residual
`lhs - c*x`, the score accumulations and the jump scan's `score +=
(v - cur) * slope`; `move.score += diff` in the weight update is a separate
statement and stays unfused); the layout is CSR both ways, per-constraint
fields in one struct, no allocation per jump value, and the scores of a
constraint's old and new LHS once per constraint. Jump candidates with
equal keys can only differ in the sign of a zero, so their sort order does
not matter. Nothing here depends on the standard library's order:
libstdc++'s generate_canonical takes the same two draws to the same sum
(it only clamps a result of 1, which the `< 0.001` and `< 0.01` tests
cannot tell apart), and its mt19937 result_type is wider but holds the
same values.

ziRound and shifting run in Rust (mip/heuristics.rs) when the model is
column-wise: they return the rounded point, and the C++ tries it
(trySolution, tryRoundedPoint, ziRound after shifting). Row activities
(double-double, as calculateRowValuesQuad and getInfeasibleRows) are
recomputed only for rows whose columns moved; shifting reads the LP
relaxation's fractional integers instead of copying the whole relaxation,
and draws from the C++ HighsRandom in place. Its std::unordered_map of
shifts is only looked up, never iterated, so it needs no libstdc++
variant.

Graph LNS (HighsGraphLns.cpp) runs in Rust: the decision columns, the
neighbourhoods (seed choice from the flip promise, disagreement with the
LP, or random, and the breadth-first search, with the short-row variant),
the promise update from the reduced costs, the move type bandit (UCB,
`base + 0.5 * sqrt(...)` fused) and the flip candidates and partners
(byGain is a total order on distinct columns, so any sort reproduces
pdqsort's) are mip/lns.rs; the root dive, the neighbourhood loop with its
dives and depth-first branch and bound, the flip search with its
propagation screen and the LP re-solves are mip/graph_lns.rs.

The rest of HighsPrimalHeuristics is Rust too (mip/primal.rs): the state
(integer columns in rounding order, decision columns, LNS move statistics,
fixing-rate observations, the generator) and RENS, RINS, rootReducedCost,
randomized, central and line-search rounding, tryRoundedPoint, the
feasibility pump, crossover and solveSubMip's gaps and statistics; the C++
class is a handle whose methods forward (HighsPrimalHeuristics.cpp, the
original under `#ifndef HIGHS_RUST`). The heuristics drive the Rust search
directly (the C++ HighsSearch is only created, with its own pseudocost
copy, and holds the local domain), and reach the C++ objects through
mip/glue.rs: one C++ function per operation on a HighsDomain (copy,
assign, changeBound, fixCol, propagate, backtrack, conflict analysis, the
stack), a HighsLpRelaxation (copy or fresh, bounds, costs, options, the
root basis, resolveLp, the solution, putIterate/getIterate, the dual
infeasibility proof with its conflict), the worker and the solver, plus
`op`, a table of scalar operations. The bounds of a domain are read
through raw pointers (`Bnd`), since C++ changes them under Rust. The
sub-MIP itself (options, the HighsMipSolver, its profiling) stays C++
(`sub_mip`). Points are copied at the entry, so a C++ vector passed by
reference never aliases a Rust slice across a call that changes it.
Details that matter for the path: `changeBound`'s default reason is a
branching (rootReducedCost), `HighsIntegers::nearestInteger` returns an
int64, `1.0 - (1.0 - x) * 0.9`, `(1 - a) * p1 + a * p2`,
`1000 + avg * 5` and `0.7 * rate + 0.3 * c / e` are fused, and RINS does
not propagate after its fractional fixings (RENS does). Under the parallel
lock several workers run RENS, RINS and randomized rounding on the one
Rust object; they then only read it (UnsafeCell for the serial-only
mutations).

## The MIP driver (HighsMipSolver, HighsMipSolverData)

The scalars of HighsMipSolverData are one struct (HighsMipScalars, same
layout as glue.rs `MipScalars`, size checked on both sides); the old
fields are references into it, so the rest of the solver reads them
unchanged. Its vectors (incumbent, first and root LP solutions, analytic
centre, the row-wise matrix, row maxima and integrality, locks, column
classes) are Rust's: a `MipVecs` (mip_data.rs) that HighsMipSolverData
owns (MipVecsOwner, declared first, freed last) and refers to in place
(`HighsRsArray`), set by Rust (C++ through highs_rs_mip_vecs_set). Rust
gets them, the scalars and the model in place (`MipData`, filled per
call by `mipData()`, refetched after a restart or presolve since the
model changes).

In Rust (mip/mip_data.rs, root.rs, driver.rs): limitsToGap,
computeNewUpperLimit, limitsToBounds, updateLowerBound, the primal-dual
integral, checkLimits, moreHeuristicsAllowed, percentageInactiveIntegers,
removeFixedIndices, printDisplayLine with its key and number formats,
checkSolution, trySolution, solutionRowFeasible, the trivial heuristics,
addIncumbent, transformNewIntegerFeasibleSolution (the repair LP's
solve and the postsolve stay C++ on a scratch HighsSolution), evaluateRootLp,
rootSeparationRound, evaluateRootNode (with its restarts: one pass per
model), HighsMipSolver::run (the presolve and setup calls, the pre-root
heuristics, the root, the branch-and-bound loop: node selection, plunging,
the dives and their heuristics, the restart votes, the ramp-up of the
workers, the tree graph-LNS rounds) and cleanupSolve with the solving
report (model status strings, getGapString, highsDoubleToString).
processNode runs as a task of the C++ HighsMipSolver::runTask
(`run_process_nodes`), so the parallel search keeps its threading
model. clang
fuses `scale * ub - 0.5`, `rel * |ub + offset| * scale - eps`,
`abs * scale - eps`, `ub - rel * |ub + offset|`, `pdi += dt * gap`,
`total * effort + 10000`, `lb * scale - feastol` (the integral dual
bound), the row activities of checkSolution and trySolution, and the
separation's `scale * cur - avg` and `(1 - a) * s + a * p`;
`std::pow(1.5, n)` is libm's pow (not powi).

The setup is Rust too (mip/setup.rs): init, runMipPresolve (HPresolve
itself is one C++ step), runSetup (the row-wise matrix, the locks, the
integral rows with their rounded sides, the column classes and the model
log), basisTransfer, checkObjIntegrality, setupDomainPropagation (for
presolve's probing), performRestart (its locals, the root basis in the
original space and the pseudocost initialization, are a C++ struct for
the solver's pointers to them), the end of the analytic centre and
symmetry detection tasks (fixings at the analytic centre, the symmetry
log), saveReportMipSolution, queryExternalSolution and the user
callbacks: Rust decides when and fills data_out's values, one C++ shim
(`CMipFns::callback`) clears or sets the HighsCallback fields and calls
callbackAction, so highspy and the C API see the same callbacks
(including the cut pool callback). The workers' solutions
(HighsMipWorker::addIncumbent, trySolution and the transformation into
the original space on a per-worker scratch solution) and the
synchronization of the workers' solutions and global domains with the
solver's are in mip/workers.rs; the heuristics' addIncumbent and
trySolution go there directly under the parallel lock.

Still C++ (behind `CMipFns::op`, codes in mip_data.rs, root.rs, driver.rs,
setup.rs and workers.rs; each a step on a C++ object without decisions of
its own): the construction of the C++ objects (workers, LP relaxations,
domains, pools, the sub-MIP's HighsMipSolver with its options and model),
the pools' and pseudocosts' sync calls, the per-worker search steps (each
a call into the Rust search, with the profiling clocks around it), the
start of the analytic centre task (a `Highs` IPM solve) and of the
symmetry detection, the `Highs` solve of transformNewIntegerFeasibleSolution's
repair LP (Rust fixes the integers, sets the time limit and keeps the
counts), the profiling
clocks (HighsProfiling, shared with Highs) and HighsDebugSol (not
supported).

The concurrent LNS helper is Rust (mip/concurrent.rs): its pool
(HighsConcurrentLns: the best solution with its version, each search's
own best until the crossover, the bounds, the stop and target flags, the
root cuts in a OnceLock, the crossover's log state) and its thread, a
Rust std::thread (8 MB stack, a Linux std::thread's) owned with the pool
by the main solver (HighsMipScalars::concurrent_lns, an `Arc` shared with
the thread; joined on stop and in ~HighsMipSolverData), and start, sync,
crossoverWithMain, the root cut publish and import, the limit checks and
the offers of addIncumbent. The atomics keep the C++'s orderings. C++
keeps the object shells: the helper's options, model and root basis
(`helper_new`), its HighsMipSolver with its single-thread scheduler,
timer and profiling (`helper_run`, run on the Rust thread), the LP's cut
rows (op 135) and the cut pool's addCut and separate into the LP. The
helper's HighsMipSolver::concurrent_lns_ and a sub-MIP's
lns_target_reached_ point to the Rust pool. Single-thread solves never
start a helper (useConcurrentHelper, still C++ for
std::thread::hardware_concurrency).

A sub-MIP (solveSubMip's run: RENS, RINS, crossover) is decided in Rust
(glue.rs `sub_mip`): a `SubMipSpec` of the bounds, the start with its row
activities, and every option that differs from the caller's (limits, time
limit with the cap, objective bound, gaps, presolve, symmetry, effort,
the helper's heuristic settings, lns_target_reached_). The C++ subMip is
the shell: it copies the options and model, applies the spec, constructs
and runs the HighsMipSolver between the profiling clocks and returns the
result.

A worker's state is Rust's (workers.rs `WorkerState`: bounds, heuristic
and separation statistics, generator, the heuristics flag, the buffered
solutions); HighsMipWorker owns it, refers to its fields in place
(RsState) and keeps the pointers to the C++ objects it works on.

What remains C++ before a pure-Rust MIP solve (each an object shell or a
step on one, reached through `CMipFns`/`CLpFns`/`CSearchFns`/`CSepaFns`
callbacks; about 240 op codes):
- The LP solver of HighsLpRelaxation, a `Highs` object (passModel,
  addRows/deleteRows, changeColsBounds/Cost, setBasis/getBasis, run with
  the IPX race, getSolution/getInfo, getDualRay, getBasisInverseRow,
  putIterate/getIterate, options), and the same for the analytic centre
  (IPM) and the repair LP. The LP algorithms, the simplex engine's data,
  the LP being solved and the model modifications are Rust (`LpSolver`,
  interface.rs), and an LP run's solution, basis, info, model status and
  presolve data are Rust's (lp_run.rs, lp_presolve.rs), but the model LP,
  the options and the Highs LP API stay C++; a Rust-owned LP relaxation
  needs the options in Rust and the run path of run.rs without the Highs
  object (see "The simplex engine's data").
- The object shells: HighsMipSolver (options, models, callback, timer,
  terminator), HighsMipSolverData's containers (deques of LP relaxations,
  domains, pool and pseudocost handles, workers), HighsSearch (local
  domain shell, conflict scratch), HighsSeparation and the separators'
  objects (HighsTransformedLp, HighsCutGeneration for conflicts,
  HighsCutSet), HighsObjectiveFunction, HighsDomain's scalars and
  redundant rows, the presolve (HPresolve shell, the postsolve stack's
  storage, the restart's model rebuilds), HighsSymmetries' handle,
  HighsProfiling/HighsTimer.
- Highs::callSolveMip and its post-processing (lp_data).

## The task scheduler (highs/parallel)

The executor, the split deques, the sleeping workers' stack, the binary
semaphore and the spin waits are Rust (rust/src/parallel), with the C++'s
memory orderings, spin and sleep thresholds and random victim choice.
The header API stays (spawn, sync, TaskGroup, for_each are C++ templates):
a task slot has HighsTask's layout (56 bytes of callable, the stealer
word); push asks Rust for the slot, places the callable and publishes it;
a stolen task is run by `HighsTask::runStolen`, which catches
HighsTask::Interrupt and returns true; where the C++ scheduler throws
Interrupt (checkInterrupt, a sync whose leapfrogging ran a cancelled
task) the Rust function returns true and the inline C++ throws, so no
exception crosses Rust frames. Thread-local state (the worker's deque,
the executor handle) is Rust's; the executor is reference counted
(`Arc`) by the main thread and its workers. HighsMutex, HighsCombinable
and HighsRaceTimer are only used by C++-only code and are not compiled
into a HIGHS_RUST build's paths.

## The top level (lp_data)

The `Highs` class and its data (HighsLp, HighsSolution, HighsBasis,
HighsInfo, HighsOptions) stay C++-owned: the public API, highspy and the
C API hand out references to them. rust/src/lp_data works on views
(highs/lp_data/HighsRust.h: `RsMut` arrays, `RsLp`, `RsIndexCollection`;
HighsInfoStruct and HighsPrimalDualErrors are read and written in place,
layouts checked by static_assert and a Rust test) and logs through
highsLogUser/highsLogDev with "%s" of a message formatted by
util/printf.rs (`Log`, `log_user!`, `log_dev!`), so log files and
callbacks see the same calls. The C++ originals are under
`#ifndef HIGHS_RUST`; the glue is in HighsLpUtilsRust.cpp and
HighsSolutionRust.cpp.

In Rust: assessLp, lpDimensionsOk, assessCosts, assessBounds,
assessMatrix(+Dimensions) (duplicates found with a stamp array instead of
the C++ hash set; same result), cleanBounds, scaleLp (equilibration and
max-value scaling), HighsLp::applyScale/unapplyScale (lp_utils.rs); the
KKT checks getKktFailures, getVariableKktFailures, the basis and glpsol
error measures, getComplementarityViolations, computeDualObjectiveValue,
computeObjectiveValue, HighsLp::objectiveValue, lpKktCheck and
reportKktFailures (solution.rs). The C++ min/max (not f64::min/max) are
used for NaN fidelity. The LTO build vectorizes the objective sums:
HighsLp::objectiveValue and computeObjectiveValue are `dot_blocked` by 4,
the quadratic term of the dual objective by 8 (rounded products, fused
tail), and `dobj += bound * dual` and `+= 0.5 * quad` are fused.

Options and info (options.rs, info.rs, glue in HighsOptionsRust.cpp):
the logic of HighsOptions.cpp, HighsInfo.cpp and io/LoadOptions.cpp
(finding, checking, setting from bool/int/double/string with every
validation message, getting, resetting, passing, reporting in the full,
markdown and minimal formats, reading options files, HighsInfo's
invalidate/equal, getInfoValue, writeInfo). The records stay in the C++
headers (HighsOptions and HighsInfo are public API, read by highspy and
the C API); each call passes Rust a table of views of them
(`COptionRecord`: name, description, bounds, defaults, a pointer to the
value field) built from `records`. Rust writes bool, int and double values
through the pointers and strings through a C++ callback; highsOpenLogFile
stays a C++ callback (HiPO and HiPDLP are rejected as not available in
this build). sscanf's %d, atoi and atof are
mirrored (strtol saturated to 64 bits, truncated to 32). The one
difference: a non-numeric value for an integer option logs (at
log_dev_level > 0) the conversion's result as 0 where the C++ prints
sscanf's uninitialised variables. `rust/bench/options_compare.sh` builds
options_driver.cpp against both libraries and diffs its output and files;
`cli_compare.sh` diffs the app on the command lines of cli_cases.txt
(stdout, stderr, exit code, written files; times masked).

The app's command line (options_cli.rs) is parsed by Rust as CLI11 2.5.0
parsed it for the options HighsRuntimeOptions.h defines: classification
of `--name[=value]`, `-x[rest]`, negative numbers and `--`, the model file
as the positional, CLI11's checks in its order (help, existing-file
validators, more than one value, conversions with strtoll base 0, `_`/`'`
separators, 0o/0b, trailing spaces and strtold), leftovers, the error
messages and exit codes, and the help text (column 33, HiGHS's patched
paragraph formatter). The app's main and loadOptions are Rust too
(lp_data/app.rs, see "The app and crest" below). CLI11.hpp is still
included (by HighsAppRust.cpp) for the third-party notice and the version
in the "Command line parsed using CLI11" log line, which stay identical. Limits: strtold is strtod (x86_64's 80-bit strtold
could round a halfway --time_limit twice), and glibc 2.38's C23 strtoll
(which g++ may bind) also reads a leading "-0b".

The file writers are Rust: solution files in every style (old raw, raw,
pretty, Glpsol raw and pretty, sparse; writeSolutionFile and its parts,
writePrimalSolution and writeObjectiveValue, which the MIP improving
solution file uses), basis files (writeBasisFile) in
lp_data/writers.rs, and the MPS (writeMps, free and fixed) and LP
(FilereaderLp::writeModelToFile) model writers in io/model_write.rs.
The C++ functions are wrappers (HighsWritersRust.cpp): they open the
file, compute what is C++ data or arithmetic (the HighsCDouble objective
of the raw style, the row-wise matrix copy of the LP writer, the
getKktFailures of the Glpsol KKT report, done between two Rust calls so
messages keep their order) and pass views (`RsWriteModel`, names as
c_str/strlen). Rust sends text back through one callback (`RsOut`):
fwrite to the file in 64 KB blocks, or, when the file is stdout,
highsFprintfString of the same pieces as the C++ (one per C++
highsFprintfString), so log callbacks see the same calls; messages go
through the same callback, in order. A piece C++ builds with
highsFormatToString is cut to its 1023-byte buffer, and an LP token to
560 bytes, as vsnprintf would.

Numbers (io/write.rs) are formatted without printf: `g` is `%.{p}g`
(Rust's `{:.*e}` rounds the exact value half-to-even, as libc does, and
the digits are then laid out as %g would), `double_to_string` is
highsDoubleToString. The macOS libc has a quirk the writers mirror: its
dtoa keeps the trailing zeros of an integer below 1e15 that %g rounds
down to p <= 14 digits when its floating-point quick path cannot decide
the rounding (`%.4g` of 55005 is "5.500e+04", glibc's "5.5e+04");
`libc_keeps_zeros` simulates both dtoa paths (macOS only). It matters
for the %12g / %13.6g / %.10g columns of the pretty styles; %.15g and
highsDoubleToString never round an integer. glibc signs NaN ("-nan"),
macOS does not; `g` follows the target. (util/printf.rs, used for
messages, does not mirror the integer-tie quirk.)
writeRangingFile (the pretty and raw ranging of a solution file, and the
dev report to stdout) is writers.rs too; it fprintf's directly, so its
text is fwritten even to stdout.

The solution and basis file readers are Rust (lp_data/readers.rs:
readSolutionFile after its style check, readBasisFile, readBasisStream;
glue in HighsLpUtilsRust.cpp). The C++ read with std::ifstream's `>>`,
`ignore` and `eof()`; `IStream` follows libc++'s semantics for those (the
sentry, eofbit/failbit, a failed extraction disabling the later ones,
num_get accumulating a number's characters then strtod/strtoll, ERANGE
failing), so malformed files read the same as with libc++ (libstdc++
differs on e.g. "1.5abc", which only malformed files contain). std::stoi's
exceptions (a "# Columns" line without a number) become a panic, so an
abort, as the uncaught exception was. Name lookups go through the LP's
C++ HighsNameHash (formed by C++ at the same point as before), and the row
values of a primal-only file are the Rust calculateRowValuesQuad.
cli_compare.sh reads files written by the C++ app (raw, sparse, MIPLIB
style, truncated, wrong sizes and names, v1/v7 bases).

Highs::run's control flow (run.rs, glue in HighsRunRust.cpp): Highs
calledOptimizeModel (the QP / MIP / LP choice, infinite costs, the
inconsistent bounds of infeasibleBoundsOk, semi-variables, and for an LP
the presolve decision, the solve of the reduced LP, postsolve, the
clean-up solve and the timing report), runPresolve, runPostsolve,
returnFromOptimizeModel, returnFromHighs and reportSolvedLpQpStats; and
HighsSolve.cpp (solve.rs: solveLp's choice of simplex, IPX or PDLP
and the simplex clean-up of an unwelcome IPM status, solveUnconstrainedLp,
assessExcessiveObjectiveBoundScaling). The Highs object stays C++: Rust
reads and writes its scalars in place (model status, HighsInfo,
HighsRunData, the solution's and basis' validity flags) and calls back
for each step on a C++ object (`Op`: the solvers, presolve, postsolve,
the timer, copies of solutions, bases and options, the debug checks). The
options are read once per call, before any step changes them. A MIP calls
calledOptimizeModel for every LP of its relaxation, so the run allocates
nothing unless it logs, and formats a message only when highsLogUser /
highsLogDev would print it. A step that throws (HighsTask::Interrupt of a
cancelled task, e.g. the IPX of the root LP race) is caught by its C++
callback, Rust returns at once without further steps, and C++ rethrows
(a C++ exception must not unwind through Rust). The MIP solve itself
(callSolveMip) stays one C++ step.

The model modification and query internals (edit.rs, query.rs,
basis.rs, ranging.rs): changing costs, bounds and integrality over an
index collection, changeLpMatrixCoefficient (C++ makes room for one more
entry), deleting the LP's vectors (names moved by C++ from Rust's map),
scale factors and basis statuses, the statuses of nonbasic variables whose
bounds change or that are appended, getCoefficient, feasibleWrtBounds,
calculateRowValuesQuad / calculateColDualsQuad (CDouble), refineBasis,
isBasisConsistent, the IPX solution conversions
(ipxSolutionToHighsSolution, ipxBasicSolutionToHighsBasicSolution), the
checks and messages of getBasisInverseRow/Col, getBasisSolve,
getBasisTransposeSolve, getReducedRow/Column, the reduced row and the
extraction of basisSolveInterface's solution, setSolution, and
getRangingData (HighsRanging.cpp; the FTRAN of each nonbasic column is a
callback). Also user objective and bound scaling (user_scale.rs:
userScaleLp and its parts, userScaleStatus, HighsUserScaleData's
messages, the solution part of userScaleSolution), the semi-variables
(semi.rs: assessSemiVariables, relaxSemiVariables,
activeModifiedUpperBounds, HighsLp::unapplyMods; C++ appends the records
Rust returns to HighsLpMods) and model.rs (handleInfCost / restoreInfCost,
basisForSolution's statuses, reportModelStats). clang fuses `objective + sense * x` in the ranging (x a
product, or a product times a dual), `xi - delta * a_in`, the reduced row's
dot products, the unconstrained LP's objective and the row activities of
free rows in the IPX conversions. `rust/bench/api_compare.sh` builds
api_driver.cpp against both libraries and diffs runs with option
variations, rays, basis inverse and tableau rows, ranging (also of
maximizations), model edits, setSolution / setBasis and presolve /
postsolve through the API.

getCols / getRows' getSubVectors and getSubVectorsTranspose are edit.rs
(`get_sub_vectors*`): the outputs are C++ pointers without lengths, so
the Rust entry point first runs the extraction writing nothing to learn
the counts, then slices the outputs to them (api_driver.cpp's getColsRows
covers intervals, sets, masks, column- and row-wise matrices and absent
outputs).

The IIS is Rust (iis.rs, glue in HighsIisRust.cpp): getIisInterface
with its return (the trivial and row value bound checks, the resolve, the
elasticity filter, the deletion filter of HighsIis::compute on the LP of
the infeasible rows, setLp, setStatus and the checks indexStatusOk,
lpDataOk and lpOk with their solves) and elasticityFilter, which is also
Highs::feasibilityRelaxation. Rust holds the HighsIis data for the whole
call (loaded at its start, stored at its end), so C++'s copies and
restores of `iis_` around the model edits that clear it are not needed;
the IIS LP with its names is built by C++ from Rust's arrays and kept
aside until the end. Each step on a `Highs` object (the incumbent, or one
the IIS search creates for its LP solves) is one `Op`: options, callbacks,
passModel, solves, bound and cost changes, adding and deleting the
elastic columns and rows, the elastic solution's KKT failures. clang
fuses the row activity bounds of rowValueBounds. Left out: the developer
reports (kIisDevReport) and the dead sensitivity filter and dual ray
options. getIis, writeIisModel and extractIis stay thin C++ wrappers.

The logging sink is Rust (io/log.rs): highsLogUser and highsLogDev keep
only the vsnprintf of their variadic arguments (C++ formats a message only
when `highs_rs_log_prints` says it prints), and `highs_rs_log` decides
what prints (output_flag, log_dev_level and the detailed/verbose types),
writes the "WARNING: " / "ERROR:   " prefix and the message to the log
file and C's stdout (fwrite and fflush, so it interleaves with printf as
before) and calls the log callbacks with the C++ 1024-byte buffer (a
longer message is cut, as vsnprintf did). Rust's `Log` calls the sink
directly. The std::function user callback is invoked through two C++
functions registered at load time (the crate and its tests link no C++).

HighsHessian and HighsHessianUtils are lp_data/hessian.rs: assessHessian
(dimensions, the matrix, normaliseHessian with its in-place gathering of
a square Hessian, explicit zero diagonals), completeHessian,
okHessianDiagonal, extractTriangularHessian, triangularToSquareHessian,
deleteCols, toSquare, print, the products, objective values (clang fuses
`p += v * x`, `y += alpha * v * x` and the objective's sums; the
double-double objective is not fused) and user scaling. The C++ vectors
are resized through `RsVec` (a std::vector, its resize and its data).

LP reporting (reportLp and its parts, reportMatrix), assessLpPrimalSolution
and assessColPrimalSolution, isLessInfeasibleDSECandidate, the user data
NULL checks, applyScalingToLpCol/Row, unscaleSolution and
highsVarTypeToString are report.rs; normaliseNames, maxNameLength,
getFileType, findModelObjectiveName, the status strings,
HighsLp::objectiveCDoubleValue (for writeLpObjective / writeModelObjective),
Filereader::getFilereader's choice of reader, interpretFilereaderRetcode
and extractModelName are model_utils.rs; formStandardFormLp is api.rs
(`offset += cost * bound` and `rhs -= value * bound` fused). Names cross
as `RsName` lists (c_str and strlen, so "%s" of a name is as before;
non-UTF-8 bytes print as U+FFFD) and come back through a setter. analyseLp
and analyseModelBounds (highs_analysis_level) are left out.

The Highs methods that only orchestrate steps on the C++ object are
driven by Rust over the `Run` of run.rs (drivers.rs; the steps are `Op`s
of HighsRunRust.cpp, the state between them in its `HighsRunRust`):
callSolveMip's handling of the MIP solver's result, checkOptimality,
completeSolutionFromDiscreteAssignment, getDualRayInterface and
getPrimalRayInterface with their re-solves, presolve, crossover,
callRunPostsolve, setBasis, passModel(HighsModel), passHessian(HighsHessian),
readModel, readBasis, writeLocalModel, writeBasis,
forceHighsSolutionBasisSize, analyseSetCreateError, aFormatOk / qFormatOk
and the integrality check of passModel's arrays. A step that throws (a
cancelled task's Interrupt) ends the Rust driver at once, as in run.rs.

Still C++ in lp_data: the public API wrappers of Highs.cpp (index
collections, the getters, passModel/passHessian of arrays building the
C++ model, run()'s file steps, getFixedLp, releaseMemory, the callback
setters), the null data checks of HighsInterface.cpp's add/delete/change/
scale interfaces (the rest is interface.rs, see "The simplex engine's
data"), user scaling's sequencing, tryPdlpCleanup and HighsProfiling
(shared with the MIP solver), HighsLpUtils.cpp's withoutSemiVariables and
the getLp* copies, the struct methods of HighsLp, HighsSolution, HighsBasis,
HighsModel, HighsCallback and HighsRunData, HighsIO.cpp's string helpers
(highsBoolToString, highsDoubleToString, ...) used by C++ callers, and
callCrossover (presolve/ICrashX.cpp).

The IPX glue is Rust (lp_data/ipx_glue.rs): solveLpIpx (the IPX
parameters from the options, fillInIpxData's LP in IPX form, the solve on
the Rust IPX directly, the status reports and the checks of illegal
solved/stopped statuses, reportSolveData, the interior or basic solution
in HiGHS form) and fillInIpxData for callCrossover. ipm/IpxWrapperRust.cpp
passes the options, the LP view and HighsInfo, model status and validity
flags in place, with callbacks for the timer and for sizing the solution
and basis where the C++ resized them; the IPX hooks (logging, task and
user interrupt) are those of ipx::LpSolver, and a cancelled task returns a
code on which C++ throws HighsTask::Interrupt. IPX's arrays are null where
an empty std::vector's data() was (the Rust vectors keep the C++
capacities). formSimplexLpBasisAndFactor and accommodateAlienBasis
(lp_data/form_basis.rs; the latter factorizes with the Rust HFactor and
logs through a copy of the log options without callbacks, as HFactor's
own) are Rust too, the steps on the LP and the HEkk instance being ops of
HighsSolutionRust.cpp.

Highs::computeIllConditioning with formIllConditioningLp0/1 is Rust
(lp_data/ill_cond.rs): the analysis LP (built column-wise, transposed as
HighsSparseMatrix::ensureRowwise does for the constraint view), the
multipliers and their report (`ss << x` is `%g`). HighsIllCondRust.cpp
makes the incumbent matrix column-wise, passes views and names, solves
the analysis LP with a silent `Highs` and stores the records. The writers and readers (writers.rs, readers.rs,
io/model_write.rs), options, info and command-line parsing (options.rs,
info.rs, options_cli.rs) and the app (app.rs) are Rust, see above.

## The app and crest

The highs app's main and loadOptions are Rust (lp_data/app.rs): the
command line parse, the version / notice text, the options file and
command-line options into a separate HighsOptions, opening the log file,
passing the options, reading the model, presolve and write_presolved_model
or run, and runHighsReturn with its copyright lines, including the C++
app's quirk of calling runHighsReturn twice after a run error. The
`Highs` instance and the loaded options stay C++ (HighsAppRust.cpp in
libhighs: `highs_app_create`, one `op` per step, `highs_app_destroy`);
stdout and stderr text goes through C's stdio so it interleaves with the
C++ logging as before, and --version exits through C's exit as the C++
did. Under HIGHS_RUST app/RunHighs.cpp's main is one call of
`highs_app_main`; the CLI11 code of RunHighs.cpp and HighsRuntimeOptions.h
is compiled only without HIGHS_RUST.

`crest` (rust/src/bin/crest.rs, cargo feature `crest`) is the Rust binary:
its main calls `app_main` of the crate, so its Rust code is the crate's own
(no second copy). It links the C++ that is not ported yet as a static
library: libhighs.a of a static HIGHS_RUST build (`-DHIGHS_RUST=ON
-DBUILD_SHARED_LIBS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`, IPO so
that the C++ is compiled as in the shared library), found through
`HIGHS_LIB_DIR` by rust/build.rs, with libc++ (libstdc++, pthread and dl
on Linux) and zlib when that build found it. Such a CMake build builds it
as the `crest` target into bin/ (own cargo target dir rust-crest/); by
hand: `HIGHS_LIB_DIR=<build>/lib cargo build --release --features crest`.
The library and its tests do not need libhighs. Output is the C++ app's,
byte for byte, but for argv[0] in the usage line; the third-party notice
still lists CLI11 (whose behaviour options_cli.rs reproduces) and the log
line "Command line parsed using CLI11" is kept.

What blocks a C++-free crest: the whole C++ column of the table below.
In order of size: the `Highs` class and its data (Highs.cpp,
HighsInterface.cpp, HighsLp/HighsSolution/HighsOptions/HighsInfo records,
HighsIO logging), the MIP solver's C++ classes, init, restarts, workers
and the HighsTask scheduler (concurrent port), the HEkk shell, the IPX and
QP glue, the IIS and the utilities. highspy stays a C++ wrapper of `Highs`.
In order of size: the `Highs` class and its data (the API wrappers of
Highs.cpp and HighsInterface.cpp, the step glue of HighsRunRust.cpp,
HighsLp/HighsSolution/HighsOptions/HighsInfo records), the MIP solver's
C++ classes, init, restarts, workers and the HighsTask scheduler
(concurrent port), the HEkk shell, and the utilities. crest still links all of libhighs.a:
the C++ files of lp_data, io and model are now mostly glue between the
C++ data and the Rust logic, which goes when the data becomes Rust's.
highspy stays a C++ wrapper of `Highs`.

Comparisons: `rust/bench/cli_compare.sh build build-rust build-static`
runs the C++ app, the HIGHS_RUST app and crest on 137 command lines
(solution, basis, sparse and MIPLIB-style files to read are written first
by the C++ app). Five use what the HIGHS_RUST build leaves out (--solver
hipo, a HiPO options file, the dev.set of highs_debug_level = 1) and are
marked `# dropped` in cli_cases.txt: they differ as expected and are
counted apart.

## Left out of the HIGHS_RUST build

What Crestline does not port is not compiled with HIGHS_RUST (the pure
C++ build is unchanged), and asking for it fails cleanly:

- HiPO (highs/ipm/hipo, its BLAS/METIS extras and solveLpHipo/solveHipo
  in IpxWrapper.cpp) and HiPDLP (highs/pdlp/hipdlp, HiPdlpWrapper.cpp):
  solver / mip_lp_solver / mip_ipm_solver = "hipo" or solver = "hipdlp"
  is rejected: `The HiPO solver was requested via the "solver" option: it
  is not available in this build`. solver = "ipm" is IPX, and a QP uses
  the QP solver. CMake stops with HIPO=ON.
- SIP and PAMI (simplex_strategy 2 and 3: HEkkDualMulti.cpp, the slices
  and iterateTasks of HEkkDual.cpp): the serial dual simplex runs, with a
  warning per run (`simplex_strategy = 2 (SIP) is not available in this
  build: using the serial dual simplex`).
- iCrash (ICrash.cpp, ICrashUtil.cpp; ICrashX.cpp is callCrossover, kept
  for Highs::crossover): icrash = true warns and is ignored.
- Multi-objective solves (multiobjectiveSolve and its helpers):
  addLinearObjective / passLinearObjectives fail with `Multiple linear
  objectives are not available in this build`.
- Debugging and analysis: the debug* checks (HEkkDebug.cpp,
  HSimplexNlaDebug.cpp, HFactorDebug.cpp, HighsSolutionDebug.cpp,
  HighsInfoDebug.cpp, debugNonbasicFlagConsistent) are no-op stubs under
  `#else` of each file, the simplex analysis of HighsSimplexAnalysis
  (summary, timers, densities, records) is stubbed and its flags are off,
  test_kkt and HighsDebugSol are not compiled (CMake stops with
  DEBUGSOL=ON). highs_debug_level and highs_analysis_level are accepted
  and do nothing for the simplex; the CHUZC failure reports (dev log)
  and the analysis of the MIP and IPX logs are kept. The model data
  analysis of highs_analysis_level (analyseLp, analyseModelBounds) and
  calculateRowValuesQuad's debugging row report are left out too.
- The C API (highs_c_api.cpp) and with it the Fortran and C# interfaces
  (CMake stops with FORTRAN or CSHARP on), capi_unit_tests and the C
  examples.
- The fixed-format MPS reader (readMps and load_mpsLine of HMPSIO.cpp;
  writeModelAsMps is kept): a file the free reader sends to the fixed one
  (names with spaces) is an error `Free format reader has detected
  row/col names with spaces: the fixed format MPS reader is not available
  in this build` followed by the parser error, and mps_parser_type_free =
  false warns and reads free format.

Unit tests of these features are skipped with HIGHS_RUST: TestICrash.cpp,
TestMultiObjective.cpp and TestPdlpHi.cpp are not built (check/
CMakeLists.txt), the fixed-format reader cases of TestFilereader.cpp
(filereader-free-format-parser-qp, -lp, filereader-integrality-
constraints) are under `#ifndef HIGHS_RUST`.

## C++ still compiled

`rust/bench/cpp_inventory.py build-rust` lists every C++ source compiled
into a HIGHS_RUST libhighs (the unity batches expanded) and the app, with
its code lines (no blank or comment lines) in the pure C++ build and under
HIGHS_RUST (unifdef). Kinds: "glue" is Rust-call glue, "part ported" has
code under `#ifndef HIGHS_RUST`, "C++" is compiled whole (live, a fallback
for paths Rust does not take, or debug only, as the last column says).

By area (code lines, C++ build -> HIGHS_RUST): lp_data 19987 -> 9572, mip
19455 -> 6905, util 6360 -> 2037, simplex 1706 -> 1187, presolve 9015 ->
987, io 3460 -> 667, ipm 1304 -> 366, model 813 -> 198, qpsolver 281 ->
165, pdlp 150 -> 150, highs 99 -> 99, app 95 -> 3, parallel 28 -> 1: in
all 62753 -> 22337 lines in 103 files (the C++ build column counts only
the files still built). The LP run on Rust data (lp_run.rs, lp_presolve.rs)
moved data, not code: its steps on C++ objects (the solve templates, the
simplex shell, the clocks, the presolve export) are glue, so the count
rose from 22045. Before the solved LP, solveLpSimplex, dualize, the model
modification interfaces and HighsSparseMatrix's methods moved to Rust it
was 23386 lines; before the simplex engine's data moved to Rust
(LpSolver) 25452 lines in 110 files; before HiPO, HiPDLP, SIP/PAMI,
iCrash, multi-objective, debugging, the C API and the fixed MPS reader
were left out ("Left out of the HIGHS_RUST build"), 60977 lines in 160
files.

| file | C++ build | HIGHS_RUST | kind | what is left |
|---|---:|---:|---|---|
| highs/HighsExternalApi.cpp | 73 | 73 | C++ | live: third-party notice, extras library loader |
| highs/HighsExternalDeps.cpp | 26 | 26 | C++ | live: third-party notice, extras library loader |
| highs/io/Filereader.cpp | 71 | 48 | part ported | glue: creates the reader Rust picks (model_utils.rs) |
| highs/io/FilereaderLp.cpp | 440 | 147 | part ported | glue: calls the Rust LP reader |
| highs/io/FilereaderMps.cpp | 63 | 159 | C++ | glue: calls the Rust MPS parser (free format only) |
| highs/io/HMPSIO.cpp | 809 | 45 | part ported | live: writeModelAsMps (the fixed-format reader is left out) |
| highs/io/HMpsFF.cpp | 1725 | 5 | part ported | empty: the free MPS parser is Rust (only the class is used) |
| highs/io/HighsIO.cpp | 307 | 260 | part ported | glue: vsnprintf of the log calls (sink: io/log.rs); string helpers for C++ callers |
| highs/io/LoadOptions.cpp | 45 | 3 | part ported |  |
| highs/ipm/IpxWrapper.cpp | 1143 | 11 | part ported | empty: the IPX glue is Rust (ipx_glue.rs) |
| highs/ipm/IpxWrapperRust.cpp | 1 | 195 | glue |  |
| highs/ipm/ipx/lp_solver_rs.cc | 160 | 160 | glue |  |
| highs/lp_data/Highs.cpp | 4183 | 2021 | part ported | API wrappers; C++: passModel of arrays, run() file steps, getFixedLp, releaseMemory, callbacks |
| highs/lp_data/HighsAppRust.cpp | 1 | 129 | glue |  |
| highs/lp_data/HighsCallback.cpp | 254 | 254 | C++ | live: user callback data |
| highs/lp_data/HighsDebug.cpp | 39 | 39 | C++ | live: debug status helpers |
| highs/lp_data/HighsDeprecated.cpp | 145 | 145 | C++ | API: deprecated wrappers |
| highs/lp_data/HighsIis.cpp | 1123 | 17 | part ported | clear() only (the IIS is Rust: iis.rs) |
| highs/lp_data/HighsIisRust.cpp | 1 | 373 | glue |  |
| highs/lp_data/HighsIllCondRust.cpp | 1 | 100 | glue |  |
| highs/lp_data/HighsInfo.cpp | 396 | 3 | part ported | part ported (see "The top level") |
| highs/lp_data/HighsInfoDebug.cpp | 158 | 10 | stubs | no-op stubs (debugging is left out) |
| highs/lp_data/HighsInterface.cpp | 3750 | 918 | part ported | glue: add/delete/change/scale interfaces (interface.rs); live: user scaling, PDLP clean-up, HighsProfiling |
| highs/lp_data/HighsLp.cpp | 471 | 256 | part ported | live: HighsLp methods (equality, names, dimensions) |
| highs/lp_data/HighsLpUtils.cpp | 3272 | 411 | part ported | live: withoutSemiVariables, getLp* copies |
| highs/lp_data/HighsLpUtilsRust.cpp | 1 | 600 | glue |  |
| highs/lp_data/HighsModelUtils.cpp | 1419 | 127 | part ported | glue: names and status strings (model_utils.rs) |
| highs/lp_data/HighsOptions.cpp | 1051 | 29 | part ported | part ported (see "The top level") |
| highs/lp_data/HighsOptionsRust.cpp | 1 | 612 | glue |  |
| highs/lp_data/HighsRanging.cpp | 592 | 134 | part ported | part ported (see "The top level") |
| highs/lp_data/HighsRunData.cpp | 219 | 219 | C++ | live: run data (record of a run) |
| highs/lp_data/HighsRunRust.cpp | 1 | 1909 | glue |  |
| highs/lp_data/HighsSolution.cpp | 1894 | 218 | part ported | live: HighsSolution/HighsBasis struct methods, right-size checks |
| highs/lp_data/HighsSolutionDebug.cpp | 420 | 90 | stubs | no-op stubs (debugging is left out) |
| highs/lp_data/HighsSolutionRust.cpp | 1 | 345 | glue |  |
| highs/lp_data/HighsSolve.cpp | 555 | 18 | part ported | part ported (see "The top level") |
| highs/lp_data/HighsSolveRust.cpp | 1 | 196 | glue |  |
| highs/lp_data/HighsStatus.cpp | 37 | 37 | C++ | part ported (see "The top level") |
| highs/lp_data/HighsWritersRust.cpp | 1 | 362 | glue |  |
| highs/mip/HighsCliqueTable.cpp | 1704 | 342 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsConflictPool.cpp | 245 | 65 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsCutGeneration.cpp | 1060 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsCutPool.cpp | 511 | 113 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsDomain.cpp | 3164 | 523 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsDynamicRowMatrix.cpp | 150 | 5 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsFeasibilityJump.cpp | 111 | 119 | C++ | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsGFkSolve.cpp | 79 | 79 | C++ | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsGraphLns.cpp | 807 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsImplications.cpp | 700 | 281 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsLpAggregator.cpp | 33 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsLpRelaxation.cpp | 1401 | 921 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsMipSolver.cpp | 1210 | 780 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsMipSolverData.cpp | 2639 | 949 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsMipWorker.cpp | 161 | 143 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsModkSeparator.cpp | 197 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsNodeQueue.cpp | 361 | 53 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsObjectiveFunction.cpp | 92 | 92 | C++ | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsPathSeparator.cpp | 443 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsPrimalHeuristics.cpp | 1529 | 1018 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsPseudocost.cpp | 119 | 77 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsRedcostFixing.cpp | 252 | 137 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsSearch.cpp | 1648 | 698 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsSeparation.cpp | 164 | 164 | C++ | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsSeparationRust.cpp | 1 | 316 | glue |  |
| highs/mip/HighsSeparator.cpp | 23 | 23 | C++ | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsTableauSeparator.cpp | 183 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/mip/HighsTransformedLp.cpp | 468 | 1 | part ported | MIP (concurrent port): C++ class handles, callbacks, init/restart/workers |
| highs/model/HighsHessian.cpp | 281 | 106 | part ported | glue: HighsHessian (hessian.rs); struct methods |
| highs/model/HighsHessianUtils.cpp | 502 | 62 | part ported | glue: HighsHessian (hessian.rs); struct methods |
| highs/model/HighsModel.cpp | 30 | 30 | C++ | live: HighsModel equality, clear, objective gradient |
| highs/parallel/HighsTaskExecutor.cpp | 28 | 1 | part ported | task scheduler (concurrent port) |
| highs/pdlp/CupdlpWrapperRs.cpp | 150 | 150 | glue |  |
| highs/presolve/HPresolve.cpp | 6166 | 32 | part ported | presolve glue / C++ owner of the postsolve stack |
| highs/presolve/HPresolveAnalysis.cpp | 206 | 2 | part ported | empty: the rule analysis is the Rust presolve's |
| highs/presolve/HPresolveRust.cpp | 1 | 525 | glue |  |
| highs/presolve/HPresolveTest.cpp | 29 | 1 | part ported | presolve glue / C++ owner of the postsolve stack |
| highs/presolve/HighsPostsolveStack.cpp | 1012 | 89 | part ported | presolve glue / C++ owner of the postsolve stack |
| highs/presolve/HighsSymmetry.cpp | 1431 | 168 | part ported | presolve glue / C++ owner of the postsolve stack |
| highs/presolve/ICrashX.cpp | 142 | 142 | C++ | live: callCrossover (Highs::crossover) |
| highs/presolve/PresolveComponent.cpp | 28 | 28 | C++ | presolve glue / C++ owner of the postsolve stack |
| highs/qpsolver/QpRust.cpp | 1 | 163 | glue |  |
| highs/qpsolver/a_asm.cpp | 124 | 1 | part ported | empty: the QP glue is Rust (qp/glue.rs) |
| highs/qpsolver/a_quass.cpp | 156 | 1 | part ported | empty: the QP glue is Rust (qp/glue.rs) |
| highs/simplex/HEkkRust.cpp | 1 | 747 | glue |  |
| highs/simplex/HSimplex.cpp | 261 | 23 | part ported | live: setSolutionStatus |
| highs/simplex/HSimplexDebug.cpp | 121 | 66 | part ported | live: CHUZC failure reports (dev log) |
| highs/simplex/HighsSimplexAnalysis.cpp | 1323 | 351 | stubs | live: setup and stubs (the logs are rust/src/simplex/report.rs) |
| highs/util/HFactor.cpp | 1971 | 158 | part ported | glue: HFactor over the Rust factor |
| highs/util/HFactorDebug.cpp | 212 | 42 | stubs | no-op stubs (debugging is left out) |
| highs/util/HFactorExtend.cpp | 147 | 4 | part ported | glue: HFactor over the Rust factor |
| highs/util/HFactorRefactor.cpp | 240 | 13 | part ported | glue: HFactor over the Rust factor |
| highs/util/HFactorRust.cpp | 1 | 300 | glue |  |
| highs/util/HFactorUtils.cpp | 104 | 1 | part ported | glue: HFactor over the Rust factor |
| highs/util/HVectorBase.cpp | 169 | 169 | C++ | live: HVector container ops (setup, clear, copy) |
| highs/util/HighsDynamicLibrary.cpp | 63 | 63 | C++ | live: utilities |
| highs/util/HighsHash.cpp | 2 | 2 | C++ | live: utilities |
| highs/util/HighsMatrixPic.cpp | 131 | 131 | C++ | debug only (matrix pictures) |
| highs/util/HighsMatrixUtils.cpp | 319 | 33 | part ported | live: utilities |
| highs/util/HighsSort.cpp | 323 | 101 | part ported | glue: the sorts are Rust (rust/src/util/sort.rs) |
| highs/util/HighsSparseMatrix.cpp | 1492 | 429 | part ported | glue: HighsSparseMatrix over sparse.rs; live: assess, sparse productTransposeQuad |
| highs/util/HighsUtils.cpp | 1132 | 537 | part ported | live: index collections, value analysis logs, user data checks |
| highs/util/stringutil.cpp | 54 | 54 | C++ | live: utilities |
| app/RunHighs.cpp | 95 | 3 | part ported | main: calls the Rust app (rust/src/lp_data/app.rs) |
| **total** (103 files) | 62753 | 22337 | | |
