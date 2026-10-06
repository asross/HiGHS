# Performance of the Rust port

How the `-DHIGHS_RUST=ON` build (every ported piece in Rust) compares with
the pure C++ build. Re-measured after each batch of ports; newest first.

Method: `python3 rust/bench/perf.py <C++ highs> <Rust highs> --reps 3
--miplib <dir of MIPLIB .mps.gz>`. Single thread, each case run alternately
with both binaries, minimum CPU cycles of 3 runs (`/usr/bin/time -l`) on an
M1 MacBook (shared, so ±2% is noise). "Same path" compares iteration and node
counts, objective and status: the port is bit-identical, so they must match.
Both builds: Release, clang (thin LTO) for C++, rustc 1.98 (LTO) for Rust.

## 2026-10-06, path-preserving speedups of the LP kernels (7227d4a032)

A pass over the ported simplex kernels guided by a SIGPROF PC sampler
on the dispatch LPs (3c1b60d6 relaxation, MIPs 080458 and the hard tick
3c1b60d6): same floating-point operations in the same order, so every
path is unchanged. Function-level effects were measured as shares of
the profile samples of one run each (robust to the heavily loaded
machine, where whole-run cycles vary by ±5%).

| Commit | Change | Effect (share of samples, or cycles) |
|---|---|---|
| 45809efea9 | `re_index` by blocks of 4, unchecked writes | relaxation 0.986 cycles, -3.5% instructions |
| dba330cad8 | `getValueScale` max in 4 lanes (NaN: serial) | `Dual::iterate` 2.35% -> 1.28% |
| 866473fcbe | primal infeasibility by selects | `update_primal` 5.13% -> 4.10% |
| 8df46daaa6 | CHUZR tests merit before infeasibility (one branch) | `choose_normal` 4.59% -> 3.75% |
| a8dc91ec7a | CHUZC candidate test by a select | `choose_possible` 3.54% -> 2.48% |
| 02cc14ff8a | dense PRICE result indexed by selects | `price_by_row_with_switch` 2.16% -> 1.25% |
| 0e8f15ccce | per-thread pool for `OwnedHVec` buffers | hard tick: madvise 5.01% -> 1.92% |
| 7227d4a032 | `factorSolveError` keeps its work arrays | hard tick: madvise 1.67% -> 0.25% |

This build against the rust-port build before the pass (d379793f05 +
x86 FMA fix, `perf.py --reps 3`, M1, loaded): all same path.

| Group | Geomean after / before |
|---|---|
| MIP | 0.976 |
| LP dual simplex | 0.924 |
| LP primal simplex | 0.974 |
| IPM (IPX) | 0.999 |
| PDLP | 1.003 |
| Read model (time_limit 0) | 1.011 |
| **All** | **0.971** |

Dispatch cases: MIP lambda_080458 30.57 -> 28.88 Gcycles (0.945),
3c1b60d6 relaxation (dual simplex) 48.36 -> 45.07 (0.932); the hard tick
(not in perf.py) gains a further ~4% from the pool. Against pure C++
(non-dispatch cases): LP dual simplex 0.823, MIP 0.996, all 0.857.

Tried and rejected (measured slower or no change):
- Branchless HFactor solves (ftran_l/btran_l/U/hyper-sparse: select the
  kept value, branch only for a nonempty column): +7% cycles, +21%
  instructions on the relaxation; ftran_l alone 7.2% -> 9.6%. The
  zero/nonzero branch there is mostly predictable.
- Software prefetch of the RHS entry 16 pivots ahead in the dense
  triangular solves: slower (the reads hit L2; the cost is mispredicts).
- Caching the column end in the hyper-sparse DFS: no change.
- Branchless `choose_normal` without black_box: LLVM turned the rare
  update into selects, chaining the division through every row (2x
  slower); likewise `choose_possible` before black_box.
- Skipping zero terms of `update_dual`'s objective sum (exact: a sum
  started at +0 never becomes -0): slower (the extra store outweighs
  the shorter add chain).
- Selects in the DSE weight update (`aa == 0` skip): slower.
- A vectorized dense path for `update_primal`: no change (the sparse
  path is the hot one).
- IPX maxvolume `find_largest` by vectorized passes: slower (memory
  bound; one pass is best).

Not done: IPX allocates its CR and KKT work vectors per call (madvise
~3% of an IPX solve on macOS); on Linux glibc the cost is mostly a
memset, so the gain there would be small. On x86_64 (baseline SSE2)
LLVM turns some of the new selects back into branches (the first test
of the primal infeasibility, `index_dense_result`, the candidate test
of `choose_possible`), so part of those gains may not carry over to
production; native x86 measurements are still to do.

x86_64 (Rosetta, `perf.py --reps 1` against the x86_64 C++): same path on
all 31 cases; geomean 0.820 (dual simplex 0.880, IPX 0.917, MIP 1.023,
primal 1.020, PDLP 0.976, readers 0.307), cycles only indicative.

## 2026-10-05, after the simplex, IPX, PDLP and QP ports (d379793f05)

In Rust: HFactor, the whole LP simplex path from HEkk::solve down (dual and
primal), IPX + BASICLU, PDLP, the QP solver, the MPS and LP readers, PRICE.
Still C++: presolve, the MIP solver, the Highs top level, SIP/PAMI.

| Group | Geomean Rust / C++ |
|---|---|
| MIP | 0.998 |
| LP dual simplex | 0.895 |
| LP primal simplex | 0.989 |
| IPM (IPX) | 0.985 |
| PDLP | 0.955 |
| Read model (time_limit 0) | 0.399 |
| **All** | **0.852** |

| Group | Case | C++ Gcycles | Rust Gcycles | Rust / C++ | Same path |
|---|---|---|---|---|---|
| MIP | air05 | 61.71 | 62.59 | 1.014 | yes |
| MIP | neos17 | 15.16 | 15.15 | 0.999 | yes |
| MIP | nu25-pr12 | 7.00 | 7.13 | 1.019 | yes |
| MIP | neos-911970 | 20.91 | 20.79 | 0.994 | yes |
| MIP | dispatch lambda_080458 | 32.23 | 31.06 | 0.964 | yes |
| LP dual simplex | 25fv47 | 0.71 | 0.68 | 0.953 | yes |
| LP primal simplex | 25fv47 | 1.02 | 1.02 | 1.001 | yes |
| LP dual simplex | 80bau3b | 0.56 | 0.49 | 0.882 | yes |
| LP primal simplex | 80bau3b | 2.44 | 2.28 | 0.934 | yes |
| LP dual simplex | greenbea | 2.57 | 2.38 | 0.927 | yes |
| LP primal simplex | greenbea | 8.95 | 9.01 | 1.007 | yes |
| LP dual simplex | perold | 0.29 | 0.28 | 0.959 | yes |
| LP primal simplex | perold | 0.45 | 0.44 | 0.991 | yes |
| LP dual simplex | stair | 0.10 | 0.10 | 0.973 | yes |
| LP primal simplex | stair | 0.09 | 0.09 | 1.014 | yes |
| LP dual simplex | air04 relaxation | 3.30 | 3.07 | 0.932 | yes |
| LP dual simplex | rail507 relaxation | 16.35 | 13.47 | 0.824 | yes |
| LP dual simplex | co-100 relaxation | 15.68 | 10.54 | 0.672 | yes |
| LP dual simplex | dispatch 3c1b60d6 relaxation | 46.52 | 45.77 | 0.984 | yes |
| IPM (IPX) | greenbea | 1.79 | 1.93 | 1.080 | yes |
| IPM (IPX) | 80bau3b | 1.15 | 1.21 | 1.053 | yes |
| IPM (IPX) | rail507 relaxation | 16.54 | 16.23 | 0.982 | yes |
| IPM (IPX) | co-100 relaxation | 26.51 | 22.40 | 0.845 | yes |
| IPM (IPX) | dispatch 3c1b60d6 relaxation | 8.34 | 8.22 | 0.985 | yes |
| PDLP | 25fv47 | 2.55 | 2.43 | 0.953 | yes |
| PDLP | greenbea | 7.58 | 7.23 | 0.954 | yes |
| PDLP | stair | 0.81 | 0.78 | 0.959 | yes |
| Read model (time_limit 0) | co-100.mps.gz | 6.76 | 2.08 | 0.307 | yes |
| Read model (time_limit 0) | neos-5052403-cygnet.mps.gz | 12.64 | 4.87 | 0.385 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.mps | 0.78 | 0.37 | 0.471 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.lp | 1.19 | 0.54 | 0.453 | yes |

Notes:
- The large gains on whole-file cases are the readers: a byte-slice
  tokenizer without per-token allocation reads MPS/LP 2-3x faster
  (`.gz` decompression is shared C++). The relaxation solves of co-100 and
  rail507 include that read; their simplex share is ~0.95-1.0.
- The solvers themselves are a line-by-line port with identical floating
  point, so equal speed is expected; the Rust is 0-10% faster on simplex and
  PDLP (bounds checks cost instructions but not cycles on the M1) and ~5%
  slower on IPX for small LPs.
- QP (not in perf.py; generated QPs, see the QP port commit): 0.90.
- MIP is still mostly C++ (presolve, search, cuts, propagation), so its
  ratio reflects only the LP solves inside it.
- Unit tests (ctest -j1, `-DBUILD_TESTING=ON -DALL_TESTS=ON`, 168/168 pass
  in both): 197 s C++, 133 s Rust (wall time, quiet machine); unit_tests_all
  76 s -> 53 s.

## x86_64 (production architecture), 2026-10-05

Until 2026-10-05 the port mirrored clang's arm64 fused multiply-adds with
`mul_add` everywhere, which on x86_64 (no FMA in the default C++ build)
would have been a slow software `fma()` and changed the search paths. All
mirrored FMAs now go through `mul_add_c` (fused on aarch64 only). x86_64
builds (`-DCMAKE_OSX_ARCHITECTURES=x86_64
-DHIGHS_RUST_TARGET=x86_64-apple-darwin`), run under Rosetta 2 with
`perf.py --reps 1`: **same path on all 31 cases** against the x86_64 C++.
Cycles under Rosetta are translated code and one run each, so only
indicative (geomean 0.775; readers 0.26, dual simplex 0.79, IPX 0.87,
MIP 1.02, primal 1.02, PDLP 1.03); native x86 timings (e.g. EC2) still to
do.
