# Performance of the Rust port

How the `-DHIGHS_RUST=ON` build (every ported piece in Rust) compares with
the pure C++ build. Re-measured after each batch of ports; newest first.

Method: `python3 rust/bench/perf.py <C++ highs> <Rust highs> --reps 3
--miplib <dir of MIPLIB .mps.gz>`. Single thread, each case run alternately
with both binaries, minimum CPU cycles of 3 runs (`/usr/bin/time -l`) on an
M1 MacBook (shared, so ±2% is noise). "Same path" compares iteration and node
counts, objective and status: the port is bit-identical, so they must match.
Both builds: Release, clang (thin LTO) for C++, rustc 1.98 (LTO) for Rust.

## 2026-10-09, the MIP's LP solves on Rust LP solvers (LpHandle)

`perf.py <rust-port HIGHS_RUST highs (dac397718c)> <this build's highs>
--reps 3` (quiet M1): the LP relaxations, the IPX race, the IPM basis, the
analytic centre and the repair LP run on `LpHandle`s (no C++ `Highs`
inside the MIP: no copy of the solution, basis and info into and out of
the Highs object per LP solve, no C++ option or HEkk shell steps), and the
Highs object's option templates are built by Rust from a typed copy. All
64 cases same path; no measurable change.

| Group | Geomean before -> after |
|---|---|
| MIP | 1.001 |
| LP dual simplex | 1.001 |
| LP primal simplex | 1.004 |
| IPM (IPX) | 0.999 |
| PDLP | 0.999 |
| Read model (time_limit 0) | 1.002 |
| **All** | **1.001** |

Against the pure C++ build (same session), every case same path:

| Group | Geomean Rust / C++ |
|---|---|
| MIP | 0.859 |
| LP dual simplex | 0.752 |
| LP primal simplex | 0.938 |
| IPM (IPX) | 0.929 |
| PDLP | 0.964 |
| Read model (time_limit 0) | 0.384 |
| **All** | **0.776** |

## 2026-10-09, the solved LP, its run and LP presolve on Rust data

`perf.py <rust-port HIGHS_RUST highs (811de0eb90)> <this build's highs>
--reps 3` (quiet M1): the LP being solved is a Rust copy in LpSolver
(imported per solve; the C++ LP is no longer moved or scaled), the
solveLpSimplex driver, dualize, the model interfaces and HighsSparseMatrix
are Rust, and the LP part of a run works on Rust-owned solution, basis,
info, model status and presolve data (reduced LP and postsolve stack).
All 64 cases same path; no measurable cost (the per-solve LP copy replaces
the C++ in-place scale/unscale).

| Group | Geomean before -> after |
|---|---|
| MIP | 0.998 |
| LP dual simplex | 0.998 |
| LP primal simplex | 1.004 |
| IPM (IPX) | 1.004 |
| PDLP | 0.998 |
| Read model (time_limit 0) | 1.000 |
| **All** | **1.000** |

Against the pure C++ build (`build/bin/highs`, same session), every case
same path:

| Group | Geomean Rust / C++ |
|---|---|
| MIP | 0.861 |
| LP dual simplex | 0.752 |
| LP primal simplex | 0.954 |
| IPM (IPX) | 0.933 |
| PDLP | 0.957 |
| Read model (time_limit 0) | 0.388 |
| **All** | **0.780** |

## 2026-10-08, the simplex engine's data owned by Rust (LpSolver)

`perf.py <rust-port HIGHS_RUST highs> <this build's highs> --reps 3`
(the M1 shared with MIP log comparisons, so +-3% is noise): HEkk's data
(status, info, basis, work arrays, edge weights, row-wise and scaled
matrices, factor, iterate, rays) moved into the Rust `LpSolver`; the
kernels get the same views, now built in Rust, and each HEkk call passes
an `LpsEnv` (a view of the LP, the option values, the host functions)
instead of C++ filling ~150 view fields. Every case same path; no
measurable cost.

| Group | Geomean before -> after |
|---|---|
| MIP | 1.000 |
| LP dual simplex | 0.999 |
| LP primal simplex | 1.011 |
| IPM (IPX) | 0.997 |
| PDLP | 0.987 |
| Read model (time_limit 0) | 1.016 |
| **All** | **1.002** |

## 2026-10-07, crest (the Rust binary) against the C++ highs app

`python3 rust/bench/perf.py build/bin/highs build-static/bin/crest`
(reps 1, M1 loaded by gcc builds part of the time): crest is the app's
main in Rust (lp_data/app.rs) linked with libhighs.a of a static
HIGHS_RUST build with IPO, so it runs the same Rust and C++ as the
HIGHS_RUST app. The app port itself (main, readers, ranging file,
getSubVectors) is outside every timed path. Every case same path.

| Group | Case | C++ Gcycles | crest Gcycles | crest / C++ | Same path |
|---|---|---|---|---|---|
| MIP | air05 | 73.20 | 68.02 | 0.929 | yes |
| MIP | neos17 | 17.43 | 15.34 | 0.880 | yes |
| MIP | nu25-pr12 | 7.95 | 5.93 | 0.746 | yes |
| MIP | neos-911970 | 20.67 | 18.86 | 0.912 | yes |
| MIP | dispatch lambda_080458 | 31.46 | 28.40 | 0.903 | yes |
| MIP | dispatch 3c1b60d6 root | 55.25 | 52.55 | 0.951 | yes |
| LP dual simplex | 25fv47 | 0.68 | 0.57 | 0.837 | yes |
| LP primal simplex | 25fv47 | 1.00 | 0.97 | 0.964 | yes |
| LP dual simplex | 80bau3b | 0.56 | 0.43 | 0.775 | yes |
| LP primal simplex | 80bau3b | 2.41 | 2.11 | 0.875 | yes |
| LP dual simplex | greenbea | 2.49 | 2.06 | 0.828 | yes |
| LP primal simplex | greenbea | 8.48 | 8.27 | 0.975 | yes |
| LP dual simplex | perold | 0.27 | 0.23 | 0.871 | yes |
| LP primal simplex | perold | 0.43 | 0.41 | 0.964 | yes |
| LP dual simplex | stair | 0.10 | 0.09 | 0.881 | yes |
| LP primal simplex | stair | 0.09 | 0.08 | 0.871 | yes |
| LP dual simplex | air04 relaxation | 3.22 | 2.57 | 0.797 | yes |
| LP dual simplex | rail507 relaxation | 16.85 | 11.99 | 0.711 | yes |
| LP dual simplex | co-100 relaxation | 17.27 | 12.16 | 0.704 | yes |
| LP dual simplex | dispatch 3c1b60d6 relaxation | 50.78 | 46.05 | 0.907 | yes |
| IPM (IPX) | greenbea | 1.93 | 2.05 | 1.066 | yes |
| IPM (IPX) | 80bau3b | 1.22 | 1.27 | 1.039 | yes |
| IPM (IPX) | rail507 relaxation | 16.75 | 15.65 | 0.934 | yes |
| IPM (IPX) | co-100 relaxation | 15.07 | 12.16 | 0.807 | yes |
| IPM (IPX) | dispatch 3c1b60d6 relaxation | 5.45 | 5.39 | 0.990 | yes |
| PDLP | 25fv47 | 2.56 | 2.44 | 0.953 | yes |
| PDLP | greenbea | 7.83 | 7.37 | 0.941 | yes |
| PDLP | stair | 0.83 | 0.78 | 0.939 | yes |
| Read model (time_limit 0) | co-100.mps.gz | 6.64 | 2.26 | 0.340 | yes |
| Read model (time_limit 0) | neos-5052403-cygnet.mps.gz | 7.98 | 2.50 | 0.313 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.mps | 0.46 | 0.19 | 0.406 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.lp | 0.68 | 0.32 | 0.464 | yes |

Geomeans crest / C++: MIP 0.884, LP dual 0.809, LP primal 0.929, IPX
0.963, PDLP 0.944, read 0.376; all 0.796. Also: cli_compare.sh and
api_compare.sh identical on the gcc 14 pair (gcc_builds.sh).

## 2026-10-08, MIP state owned by Rust (helper, sub-MIPs, workers, domains)

The concurrent LNS helper (concurrent.rs), the sub-MIP decisions, the
workers' state, HighsDomain's vectors with the pools' propagation domains
and the objective propagation, HighsMipSolverData's vectors and the repair
LP's setup moved to Rust ownership (see PORTING.md). No change in work:
`cyc.py` per step (5 MIPs, 300 nodes, same paths) 0.999 (domain vectors),
0.995 (MIP vectors and pool propagation). `perf.py --reps 2` after merging
rust-port (crest), every case same path: clang pair MIP 0.869 (air05
0.898, neos17 0.897, nu25-pr12 0.754, neos-911970 0.941), all 0.802; gcc
pair (libstdcxx, no_fma) MIP 0.778, all 0.754. Unit tests: 166/166.

## 2026-10-07, MIP setup, workers' solutions and the task scheduler in Rust

setup.rs (init, presolve call, runSetup, performRestart, callbacks),
workers.rs and the Rust scheduler (rust/src/parallel). Setup runs once
per (re)start and the single-thread scheduler only queues and pops, so no
change is expected. `perf.py --reps 2`, M1 under heavy load (load average
20-36, so ±5%): MIP geomean 0.878 (air05 0.951, neos17 0.887, nu25-pr12
0.766, neos-911970 0.918), LP dual 0.801, primal 0.981, IPX 0.965, PDLP
0.946, all 0.833, every case same path — in line with the previous
measurement.

Same path against pure C++ (log_dev_level 1, incumbent and dev lines and
summary; periodic table lines and probing progress lines, which follow the
clock, are left out; no time limit): clang, 30 nodes, 40 MIPLIB plain
instances (less markshare_4_0, 50v-10), check/instances' 31 MIP cases;
mip_rel_gap 0.01 (graph LNS), 30 nodes: the 3 dispatch and 2 oopt MILPs,
6 MIPLIB, the check MIPs; gcc/libstdc++ (no_fma): 10 MIPLIB and the check
MIPs. Parallel: threads=4 parallel=on MIPs and SIP/PAMI LPs repeated, and
threads=2 (IPX race and concurrent LNS helper) on the dispatch MILPs, all
with correct objectives and no hang.

## 2026-10-07, the heuristics glue, graph LNS's search and the MIP driver

HighsPrimalHeuristics (primal.rs), graph LNS's dives, branch and bound
and flip search (graph_lns.rs), HighsMipSolverData's bookkeeping,
solutions, root LP and root node (mip_data.rs, root.rs) and
HighsMipSolver::run with cleanupSolve (driver.rs); see PORTING.md.
Control flow moved, the work stays in the LP solves and propagation, so
no gain is expected. Against the rust-port HIGHS_RUST build (`cyc.py 2`,
mip_rel_gap 0.01, 500 nodes, M1 loaded): neos17 1.001, nu25-pr12 1.020,
neos-911970 0.996, gen-ip002 1.005, air05 1.013, dispatch 080458 0.984
(geomean 1.003, instructions equal to 0.1%), all same path.

Same path against pure C++ (incumbent lines, restart and LNS dev lines,
final summary; 100 nodes, no time limit): clang, 55 instances (the MIPLIB
plain set less markshare_4_0 and 50v-10, nondeterministic, and co-100 and
germanrr, whose C++ runs took over 30 minutes here, plus 15 small check
MIPs); graph LNS (log_dev_level 1, 200
nodes) on hard_10-03_1340, diverse_10-01_0052, 3c1b60d6, 080458;
gcc/libstdc++ pair, 14 MIPLIB + 080458 at 100 nodes and LNS on
hard_10-03_1340 and 3c1b60d6; x86_64 under Rosetta, 6 MIPLIB + 080458
and LNS on hard_10-03_1340. The pure C++ build of this branch matches
the reference too. threads=2 is not deterministic in pure C++ (the
concurrent LNS helper), so it was only run (helper and parallel = on with
two workers: same quality, no hang). ctest 168/168.

`perf.py --reps 1`: clang all 0.838 (MIP 0.874), gcc/libstdc++ all
0.834 (MIP 0.851), 26/26 same path on both.

## 2026-10-07, the LP relaxation and the separation loop in Rust

HighsLpRelaxation (mip/lp_relaxation.rs) and HighsSeparation's loop
(separation.rs); see PORTING.md. Nothing hot moved: the LP solves stay
behind the C++ `Highs` object, so the expected effect is none. Against the
rust-port HIGHS_RUST build (`cyc.py 2`, mip_rel_gap 0.01, 2000 nodes, M1
heavily loaded): neos17 0.996, nu25-pr12 1.033, neos-911970 0.994,
gen-ip002 1.092, air05 1.007, dispatch 080458 0.966 (geomean 1.014, within
the load noise); instructions +0.0-0.7% (the LP view refetched per call),
all same path.

Same path against pure C++ (comparing incumbent lines and the final
summary; no time limit, since on this machine air05's root alone hit
1000 s, and the intermediate B&B lines are printed by time): clang, 100
nodes, 33 MIPLIB instances (the plain set less markshare_4_0, 50v-10,
which is nondeterministic in pure C++ too, and five with very slow roots);
gcc/libstdc++ pair, 14 MIPLIB + dispatch 080458; x86_64 under Rosetta,
neos17, nu25-pr12, neos-911970, gen-ip002, pk1, air05, dispatch 080458;
graph LNS at mip_rel_gap 0.01, 200 nodes, log_dev_level 1 (LNS lines and
incumbents): hard_10-03_1340, 3c1b60d6, 3c1b60d6_wind185, 080458 (clang),
hard_10-03_1340, 3c1b60d6 (gcc). After merging the symmetry and probing
ports: same path again on 12 MIPLIB (incl. the symmetric qap10, cod105,
enlight_hard) + hard_10-03_1340 and 080458 with LNS (clang), 7 MIPLIB +
080458 + hard_10-03_1340 with LNS (gcc), 5 MIPLIB + 080458 (x86_64);
ctest 168/168.

## 2026-10-07, file writers in Rust

Solution (every style), basis, MPS and LP writers in Rust
(lp_data/writers.rs, io/model_write.rs). Wall time of Highs::writeModel /
writeSolution / writeBasis, best of 3, M1 (a solve running on other
cores), seconds C++ -> Rust; solutions are of the LP relaxation:

| model | .mps | .lp | raw | pretty | glpsol raw | glpsol pretty | basis |
|---|---|---|---|---|---|---|---|
| dispatch 080458 | 0.386 -> 0.220 | 0.506 -> 0.251 | 0.458 -> 0.095 | 0.464 -> 0.189 | 0.220 -> 0.061 | 0.514 -> 0.190 | 0.112 -> 0.050 |
| blp-ar98 | 0.255 -> 0.181 | 0.327 -> 0.230 | 0.086 -> 0.015 | 0.081 -> 0.032 | 0.031 -> 0.013 | 0.107 -> 0.069 | 0.011 -> 0.004 |
| 30n20b8 | 0.147 -> 0.078 | 0.202 -> 0.125 | 0.063 -> 0.007 | 0.097 -> 0.024 | 0.017 -> 0.006 | 0.131 -> 0.025 | 0.016 -> 0.004 |
| 80bau3b | 0.029 -> 0.022 | 0.048 -> 0.018 | 0.050 -> 0.008 | 0.054 -> 0.013 | 0.042 -> 0.008 | 0.085 -> 0.016 | 0.007 -> 0.003 |

The gain is printf (one Rust `{:.*e}` per number, no locale or stream
machinery) and fwrite of 64 KB blocks instead of one fprintf per piece.
The model writers keep C++ work Rust does not replace (name checks and
hashing in Highs::writeModel, the LP writer's row-wise copy), and the
pretty Glpsol style its getKktFailures. Files byte-identical throughout.

## 2026-10-07, presolve probing and enumeration loops in Rust

runProbing's loop and enumerateSolutions in Rust (probing.rs,
enumeration.rs), merged with the search core and heuristics. M1, loaded
(load 20-100), so cycles are rough. Same path everywhere: presolve logs
(log_dev_level 1, rule logging) and presolved models identical on 82 check
instances, the MIPLIB set and 3 dispatch MILPs against clang, gcc/libstdc++
and x86_64 C++ builds (a time limit must not be reached: the loaded C++
once hit it in co-100's probing).

- Presolve cycles (read+presolve minus read, best of 3) against rust-port
  before this port and the merge: dispatch 080458 1.55 -> 1.42 G (0.915),
  3c1b60d6 0.984, 3c1b60d6_wind185 0.991, air05 1.000, nu25-pr12 1.05
  (0.2 G, noise). The probing loop itself was never the cost (the probes
  were Rust already); the gain is the crossings per probe and per
  enumerated branch.
- Full solves (cyc.py 2, against pure C++ clang): air05 0.943, neos17
  0.872, nu25-pr12 0.739, neos-911970 0.904, gen-ip002 0.821, dispatch
  080458 0.908; geomean 0.862, same path.
- `perf.py --reps 1`: clang all 0.819 (MIP 0.882), gcc/libstdc++ all
  0.824 (MIP 0.864), 32/32 same path on both.

## 2026-10-07, symmetry detection, orbitopes and orbital fixing

HighsSymmetry in Rust (presolve/symmetry.rs, on top of rust-port with the
search core and heuristics); see PORTING.md. M1, heavily loaded (load
20-55), 1 rep, so cycles are rough. Symmetric instances, `cyc.py 2` with
mip_max_nodes 60 (same path on all):

| Case | Symmetry | C++ | Rust | Rust / C++ |
|---|---|---|---|---|
| neos-3004026-krka | 64 generators | 29.44G | 29.23G | 0.993 |
| neos-1456979 | 4 generators | 110.19G | 107.05G | 0.971 |
| fastxgemm-n2r6s0t2 | 39 generators | 21.13G | 20.24G | 0.958 |
| ns1208400 | 1 full orbitope (764 cols) | 121.19G | 111.60G | 0.921 |

Same nodes and LP iterations also at 500 nodes (clang: krka, fastxgemm,
neos-1456979, ns1208400; gcc/libstdc++: krka, fastxgemm, neos-1456979),
at 60 nodes on graph20-20-1rand (orbitope; clang, gcc, x86_64) and
ns1208400 (gcc), and on x86_64 (Rosetta) on krka, neos-1456979 and
fastxgemm. `perf.py --reps 1`: all cases same path.

| Group | Geomean Rust / C++ |
|---|---|
| MIP | 0.863 |
| LP dual simplex | 0.799 |
| LP primal simplex | 0.983 |
| IPM (IPX) | 0.989 |
| PDLP | 0.936 |
| Read model (time_limit 0) | 0.388 |
| **All** | **0.837** |

## 2026-10-07, primal heuristics: feasibility jump, ziRound, shifting, graph LNS

Feasibility jump in Rust (mip/feasjump.rs), ziRound and shifting
(heuristics.rs), graph LNS neighbourhoods and scoring (lns.rs); see
PORTING.md. M1, heavily loaded (load 25-60), so cycles are rough.

Feasibility jump alone (cycles with FJ minus without, node limit 1, min of
3): C++ / rust-port / this build.

| Case | C++ | rust-port | this |
|---|---|---|---|
| neos17 | 2.19G | 2.14G | 1.42G |
| dispatch 080458 | 2.22G | 2.37G | 1.70G |
| gen-ip002 | 0.38G | 0.39G | 0.22G |
| gen-ip054 | 0.20G | 0.17G | 0.13G |
| markshare2 | 0.26G | 0.26G | 0.19G |
| markshare_4_0 | 0.09G | 0.09G | 0.06G |

Gains come from layout (CSR both ways; a variable's value, jump move and
good-set slot in one struct, so updating a neighbour's move touches one
cache line; no allocation per jump value) and from computing the scores of
a constraint's old and new LHS once per constraint.

Whole solves against the rust-port HIGHS_RUST build (2000 nodes, `cyc.py
2`, all same path): geomean 0.951; gen-ip002 0.903, gen-ip054 0.882,
markshare_4_0 0.902, markshare2 0.947, neos17 0.959, neos-911970 0.959,
nu25-pr12 0.977, air05 0.999, dispatch 080458 0.987, 3c1b60d6 1.002.

Against pure C++: `perf.py --reps 1` same path on all cases, MIP 0.869,
all 0.834. gcc/libstdc++ pair (`gcc_builds.sh`, 300 nodes, 12 MIPLIB +
dispatch 080458): same path, geomean 0.880; hard_10-03_1340 and 3c1b60d6
LNS dev logs identical. x86_64 under Rosetta (2000 nodes, the cyc.py
7-instance set): same path, geomean 0.911.

## 2026-10-06, gcc/libstdc++ builds on the M1 (4442630fe5 + HPresolve)

Both builds with Homebrew gcc 14 and libstdc++ (`rust/bench/gcc_builds.sh`:
no FMA contraction, like x86_64; Rust with `libstdcxx no_fma`),
`perf.py --reps 1` on a heavily loaded machine (load ~60), so cycles are
rough. **31/32 same path**; the exception, IPX on the co-100 relaxation,
came from presolve: libstdc++ inserts into a std::unordered_multimap group
after the hint, libc++ before it (parallel row/column buckets), so gcc's
presolve kept 6 different columns. Fixed under `libstdcxx` (5e64ec1876):
then same path on all cases against both gcc and clang builds.

| Group | Geomean Rust / C++ |
|---|---|
| MIP | 0.868 |
| LP dual simplex | 0.830 |
| LP primal simplex | 0.958 |
| IPM (IPX) | 0.994 |
| PDLP | 0.979 |
| Read model (time_limit 0) | 0.430 |
| **All** | **0.824** |

| Group | Case | C++ Gcycles | Rust Gcycles | Rust / C++ | Same path |
|---|---|---|---|---|---|
| MIP | air05 | 69.03 | 69.41 | 1.006 | yes |
| MIP | neos17 | 20.29 | 19.00 | 0.936 | yes |
| MIP | nu25-pr12 | 14.64 | 7.87 | 0.538 | yes |
| MIP | neos-911970 | 17.35 | 16.64 | 0.959 | yes |
| MIP | dispatch lambda_080458 | 31.54 | 29.82 | 0.945 | yes |
| MIP | dispatch 3c1b60d6 root | 82.63 | 77.15 | 0.934 | yes |
| LP dual simplex | 25fv47 | 0.85 | 0.70 | 0.822 | yes |
| LP primal simplex | 25fv47 | 1.11 | 1.07 | 0.968 | yes |
| LP dual simplex | 80bau3b | 0.63 | 0.51 | 0.812 | yes |
| LP primal simplex | 80bau3b | 2.38 | 2.13 | 0.893 | yes |
| LP dual simplex | greenbea | 2.54 | 2.17 | 0.857 | yes |
| LP primal simplex | greenbea | 8.93 | 8.91 | 0.998 | yes |
| LP dual simplex | perold | 0.39 | 0.34 | 0.864 | yes |
| LP primal simplex | perold | 0.53 | 0.52 | 0.986 | yes |
| LP dual simplex | stair | 0.19 | 0.18 | 0.927 | yes |
| LP primal simplex | stair | 0.18 | 0.17 | 0.949 | yes |
| LP dual simplex | air04 relaxation | 3.19 | 2.60 | 0.818 | yes |
| LP dual simplex | rail507 relaxation | 16.93 | 12.73 | 0.752 | yes |
| LP dual simplex | co-100 relaxation | 17.68 | 12.48 | 0.706 | yes |
| LP dual simplex | dispatch 3c1b60d6 relaxation | 42.54 | 40.24 | 0.946 | yes |
| IPM (IPX) | greenbea | 3.98 | 3.88 | 0.975 | yes |
| IPM (IPX) | 80bau3b | 1.34 | 1.47 | 1.097 | yes |
| IPM (IPX) | rail507 relaxation | 16.75 | 16.96 | 1.013 | yes |
| IPM (IPX) | co-100 relaxation | 28.84 | 23.84 | 0.827 | **NO** |
| IPM (IPX) | dispatch 3c1b60d6 relaxation | 8.61 | 9.30 | 1.081 | yes |
| PDLP | 25fv47 | 2.81 | 2.80 | 0.996 | yes |
| PDLP | greenbea | 8.29 | 8.03 | 0.969 | yes |
| PDLP | stair | 0.99 | 0.96 | 0.971 | yes |
| Read model (time_limit 0) | co-100.mps.gz | 7.62 | 2.75 | 0.360 | yes |
| Read model (time_limit 0) | neos-5052403-cygnet.mps.gz | 16.78 | 6.53 | 0.389 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.mps | 1.19 | 0.57 | 0.481 | yes |
| Read model (time_limit 0) | dispatch 3c1b60d6.lp | 1.46 | 0.74 | 0.506 | yes |

## 2026-10-06, x86_64 (gcc) on AWS Lambda — see `asross/oopt`

A cross-check off the M1: both builds compiled with **gcc** (the toolchain the
downstream dispatch solver actually ships, not clang) and run on a real x86_64
AWS Lambda (4096 MB), over a dispatch-MILP suite. Geomean Rust / C++ **0.945**
at one thread (18/20 bit-identical) and **0.957** at two — no regressions,
biggest wins on the LP-heavy instances. One x86 bit-identity divergence remains,
`dm_small_pert_s1_noramp`, on the newer MIP ports (same class the arm64-only
`mul_add` fix addressed for the simplex). Full tables + per-solve logs for both
builds: `asross/oopt` branch `rustport-lambda-2026-10-06`.
## 2026-10-06, HPresolve in Rust

Presolve cycles (read+presolve minus read, `write_presolved_model_file`,
best of 5, M1, loaded) against the same build with the C++ presolve
(rust-port 45bc2d0bb1): dispatch 080458 0.965, 3c1b60d6 0.964,
3c1b60d6_wind185 0.958, air05 0.972, co-100 0.82 (3 reps), 80bau3b
0.97, greenbea 1.06 (0.04 Gcycles). Same paths everywhere: presolve logs
and presolved models identical on 82 check instances, 35 MIPLIB and 3
dispatch MILPs (arm64 and x86_64), full solves and 14 MIPLIB at a node
limit. `perf.py --reps 1` against pure C++ (before the clique/implications
merge): MIP 0.925, dual simplex 0.739, primal 0.947, IPX 0.928, PDLP
0.961, readers 0.330, all 0.805, all same path.

## 2026-10-06, branch-and-bound search in Rust (b61e4b4a5f, merged with rust-port)

HighsSearch, the node queue, pseudocosts, reduced cost fixing, cut and
conflict pools in Rust. `perf.py --reps 1` against pure C++ (M1, heavily
loaded): same path on all 32 cases; geomean MIP 0.911, LP dual 0.823,
primal 0.958, IPX 0.975, PDLP 0.952, readers 0.443, all 0.828.

Against the rust-port HIGHS_RUST build (`cyc.py 2`): air05 0.998, neos17
1.000, nu25-pr12 1.005, neos-911970 1.000, dispatch 080458 1.007 (geomean
1.002), all same path. Instructions rise 0-2% (the search's calls into the
C++ domain and LP relaxation through function pointers), cycles do not.
Before the merge, against the then rust-port build: gen-ip002 0.910 (same
path); markshare_4_0 is not deterministic even for one binary (its node
count varies between runs of the same build), so it checks no path.

MIPLIB at 100 nodes (`mip_max_nodes = 100`, 30 of 34 instances, against
pure C++): all same path, geomean 0.95. The other four (eilA101-2,
neos-5052403-cygnet, germanrr, co-100) spend their time limit at the root
on the loaded machine, so the runs differ by where they stop; reruns
without a time limit did not finish (still to check).
x86_64 (Rosetta) against pure x86_64 C++: neos17 0.934, nu25-pr12 0.794,
neos-911970 1.002, air05 1.017, dispatch 080458 1.017, gen-ip002 0.872
(without the time limit, which the loaded machine hit), all same path.
gcc 14 / libstdc++ pair (`rust/bench/gcc_builds.sh`): same path on neos17,
nu25-pr12, neos-911970, air05, dispatch 080458, and at 100 nodes air04,
assign1-5-8, qap10, physiciansched6-2, mzzv11, rd-rplusc-21, gen-ip054.

## 2026-10-06, clique table and implications in Rust (8572be192b)

`perf.py --reps 1` (M1, heavily loaded): same path on all 32 cases.
Geomean Rust / C++: MIP 0.948, LP dual 0.805, primal 0.970, IPX 0.963,
PDLP 0.970, readers 0.393, all 0.818. Against the rust-port HIGHS_RUST
build at 300 nodes on clique-heavy MIPLIB (air04, air05, qap10,
assign1-5-8, physiciansched6-2, comp07-2idx, binkar10_1): geomean 0.990
(qap10 0.835, the rest 0.997-1.037). Instructions rise ~1-2% (the
callbacks into C++ domain operations), cycles do not.

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
