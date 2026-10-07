# Porting HiGHS to Rust

The Rust port will become **Crestline**: its own public repository
(github.com/asross/crestline, crate `crestline`, CLI `crest`) once the
command-line binary is pure Rust. Extraction plan: `git filter-repo
--subdirectory-filter rust` to keep history; a pinned HiGHS C++ commit as a
CI test oracle for bit-identical paths; MIT licence with the HiGHS
copyright notice and attribution kept; README states it is a Rust port of
HiGHS, bit-compatible with HiGHS v1.15. Until then development continues
on the `rust-port` branch here.

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
Still C++ behind `Host` callbacks: logging, the timer, the presolve rule
analysis setup, the HFactor of the dependent equations, and the C++ parts
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
index maps; Rust reads them through a view (`PostsolveRsStack`) and parses
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
addIncumbent, transformNewIntegerFeasibleSolution (the repair LP and the
postsolve stay C++ on a scratch HighsSolution), evaluateRootLp,
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
symmetry detection, the repair LP of transformNewIntegerFeasibleSolution, the profiling
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
and HiPO's availability stay C++ callbacks. sscanf's %d, atoi and atof are
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
paragraph formatter). C++ keeps printing them in RunHighs.cpp and the
setting of options in loadOptions. CLI11.hpp is still included for the
third-party notice and the "Command line parsed using CLI11" log line,
which stay identical. Limits: strtold is strtod (x86_64's 80-bit strtold
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

Still C++ in lp_data: Highs.cpp and HighsInterface.cpp (run /
optimizeModel orchestration, presolve/postsolve calls, the cleanup solve,
basis handling, model modification), HighsSolve.cpp (solveLp dispatch),
HighsModelUtils.cpp (solution file writers), HMPSIO/FilereaderLp writers,
HighsRanging.cpp, HighsIis.cpp, the remaining HighsLpUtils.cpp (semi
variables, user scaling, solution/basis file reading and writing, LP
reporting, vector edits), the IPX solution conversions and basis
handling of HighsSolution.cpp, and app/.
Highs::run's control flow (run.rs, glue in HighsRunRust.cpp): Highs
calledOptimizeModel (the QP / MIP / LP choice, infinite costs, the
inconsistent bounds of infeasibleBoundsOk, semi-variables, and for an LP
the presolve decision, the solve of the reduced LP, postsolve, the
clean-up solve and the timing report), runPresolve, runPostsolve,
returnFromOptimizeModel, returnFromHighs and reportSolvedLpQpStats; and
HighsSolve.cpp (solve.rs: solveLp's choice of simplex, IPX, HiPO or PDLP
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
(callSolveMip) and iCrash stay one C++ step each.

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

Still C++ in lp_data: Highs.cpp and HighsInterface.cpp outside the above
(run()'s file handling, the model passing, getStandardFormLp,
completeSolutionFromDiscreteAssignment, callSolveMip's post-processing, the API
wrappers that only call other Highs methods, getDualRay / getPrimalRay's
re-solves, setBasis on an alien basis, the IIS, ill-conditioning and
multiobjective solves), writeRangingFile, HighsIis.cpp, solution and basis
file reading (readSolutionFile, readBasisFile), LP reporting and
getSubVectors in HighsLpUtils.cpp, and the rest of app/ (main, --version,
loadOptions). The writers (writers.rs, io/model_write.rs) and options,
info and command-line parsing (options.rs, info.rs, options_cli.rs) are
Rust, see above.

