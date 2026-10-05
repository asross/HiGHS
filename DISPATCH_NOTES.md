# Dispatch MILP work on this branch

Goal: solve the dispatch MILPs (`~/code/oopt` instances and
`dispatch_milp_2026_09_30/`, run in production at `mip_rel_gap = 0.01`, 180 s,
about 2 vCPUs) much faster than HiGHS 1.15.1, without slowing general MIP.

## What limits the hard tick

`lambda_20260930_142050_3c1b60d6` is primal-limited, not bound-limited: the
root bound after cuts (about -85643) is already better than CPLEX's final
bound (-85686), and certifying 1% needs an incumbent of about -84790 or
better. Vanilla HiGHS's heuristics stall around -84400 (RENS takes about 75 s
there and finds -84424), so it never gets within 1% in 300 s. Improving
moves are small: one to nine binaries (single flips, swaps of two units in a
period, edges of on/off runs).

## A tick that no solver certifies in 180 s

`hard_10-03_1340` (from production, October 3) is bound-limited:
- Model: 55 periods, the first ones shorter. 28 banks of 1 to 8 storage
  modules (202 in all), one binary per bank and period for charging all its
  modules at fixed rates, and 25 per switch on or off. Discharges give heat,
  at most 0.32 per module and proportional to its state of charge. Heat
  sells up to 50 per period, above which it is vented, and below 36 a
  shortfall costs 200 per unit. Electricity is bought at the period's
  price.
- The root bound after cuts (about -251620) matches CPLEX's after 180 s
  (-251669). The best solution found by any means is -248801, after an hour
  of this branch on two threads and then 25 minutes of fix-and-optimize
  over 10-period windows. The two are 1.13% apart. Certifying 1% needs both
  a solution close to that and a bound about 330 better than the root's. In
  an hour on two threads the bound reached -251353, most of the gain in the
  25 s after a restart.
- Neither CPLEX nor this branch reaches 1% in 180 s. The limit is the
  relaxation, not the machine or the second thread. With two threads, the
  incumbent at 180 s is between -247.4k and -247.9k on this machine (4
  seeds), better than CPLEX's. Two log lines show the second thread at
  work: "Root LP: IPX won the race on two threads" (or the dual simplex),
  and "Concurrent LNS helper thread started". The line "Thread count 1 (of
  2 threads)" only describes the task scheduler. On Lambda, two full vCPUs
  need at least 3538 MB of memory. With less, the second thread shares one
  vCPU's time.
- Why the LP is weak: with a fractional switch, a bank charges part-way all
  the time. That tops up its modules and keeps their state of charge (and
  so their discharge limit) high, with no switch changes. An integer
  schedule alternates full-rate charging and idling and pays for each
  change. No inequality in the switches and their changes alone can cut
  this off, because constant part-way charging is the average of always-on
  and always-off. Useful cuts must involve the states of charge.
- Things that did not close the gap:
  - cuts on each module's room for a charge (+33 at the root);
  - substituting the discharges out;
  - raising the switching cost gradually from zero;
  - RINS at the root (128 s, no improvement);
  - a Lagrangian bound by bank, with the heat and electricity balances
    relaxed: -252124 at the root LP's duals, against -253940 for the LP and
    -251620 after cuts. It reached -251976 after 14 subgradient steps, some
    taking 45 minutes on hard bank subproblems.
- A restart raises the bound. At the restart in the hour-long run, and when
  one is forced after the root (+87, 15 s), presolve on the presolved model
  removes about 1500 more rows and columns. The new cut loop then gets
  further. The cut rows moved into the model do not cause this (without
  them the result is the same). HiGHS presolve stops short of a fixpoint:
  run on its own output, its aggregator substitutes about 1450 more
  columns, and the root LP takes 8.7G instructions instead of 15.0G.
  Neither a second presolve pass from the start nor a forced restart paid
  over the dispatch suite (see Tried and reverted).

## Changes

Graph LNS (`mip/HighsGraphLns.cpp`, option `mip_heuristic_run_graph_lns`, only
for `mip_rel_gap >= 1e-3`):
- Decision columns: integer columns that every LP vertex makes integral
  (e.g. start-up indicators bounded by +-1 rows of other integers) are not
  fixed or branched on.
- Quick search after the first root LP: a dive from the LP point (each
  chunk of fixings goes up to the first rounding that propagation rules
  out, rather than being undone and quartered: on the ramp models most
  chunks met such a conflict, and dives took hundreds of LP solves; a
  chunk whose LP is infeasible is undone together with the simplex
  iterate, so the dive goes on from the LP before it rather than from the
  infeasible one, which cost up to thousands of iterations each), then
  dived neighbourhoods of about 400 decision columns grown by breadth-first
  search over the constraint graph from a seed column, and a flip search to
  polish. Neighbourhood LPs start from the last LP solved (not the root
  basis), which made the hard tick 21% faster. On large models it spends at
  most about one root LP's worth of LP iterations, so that where
  neighbourhood LPs are expensive (ramping) the root cuts come early: their
  bound often closes the gap with the incumbent so far.
- Deep search after the root cuts, only if the quick search improved the
  incumbent and it is within three times the target gap of the bound with
  the cuts (otherwise the model does not suit it and the usual root
  heuristics run as without graph LNS). Measured against the bound before
  the cuts, the quick search on `hard_10-03_1340` ended at 2.9% or 3.1%
  depending on small changes elsewhere, so the deep search ran or did not.
  The deep search itself has small
  neighbourhoods searched by a depth-first branch and bound with a node
  limit, a flip search (single flips and pairs of opposite flips in a row,
  ordered by reduced-cost gain, resumed where the last one stopped), and
  dived neighbourhoods, chosen by a bandit on gap closed per LP iteration.
- The flip search first propagates a move's fixings over the model rows
  (no LP): on the hard tick 92% of flip LPs were infeasible, at about 2.5
  simplex iterations each, so the fixed cost of a solve was all they cost.
  A move ruled out this way counts as a quarter of an LP solve in the flip
  search's budget.
- If the deep search pays, rounds of it alternate with the tree search,
  where strong branching then starts with a smaller budget (10000 LP
  iterations rather than 100000): the bound rarely matters there, and on
  the hard tick it cost about 70000 iterations in the first dozen nodes.
- Stall rules measure progress against the gap (quick: 5% of it) or
  against what separates the incumbent from the target gap (deep: 1% of
  it). Both stop as soon as the target gap is reached. Where the bound stays
  well short of the target, as on `hard_10-03_1340`, improvements are steps
  of 1-5% of that excess, and the incumbent at the time limit is what
  counts. With a 5% threshold the deep search stopped after ten such steps.

Using a second core (option `mip_concurrent_helper`, on unless
`threads = 1`, when graph LNS runs; makes the solve non-deterministic):
- IPX with crossover races the dual simplex on the first root LP (models with
  at least 10000 nonzeros); the first to finish interrupts the other. IPX is
  2-3 times faster on large dispatch LPs (`dm_full_pert_s1_ramp`: 131 s ->
  79 s with two threads).
- A helper thread runs graph LNS on a copy of the presolved model with its
  own seed, exchanging incumbents with the main solver.
- The log says when the race ran ("Root LP: IPX won the race on two
  threads", or the dual simplex) and when the helper started ("Concurrent
  LNS helper thread started"). Both use their own thread, whatever the task
  scheduler's thread count in the "Thread count" line.
- Optionally (`mip_concurrent_crossover`, off by default), the helper and
  the main solver search independently, each with its own quick search,
  until the helper crosses its incumbent with the main solver's: the
  integer columns where they agree are fixed and the rest solved as a
  sub-MIP. Good solutions from different seeds differ in about 12% of the
  switches of `hard_10-03_1340`, and offline such sub-MIPs gained 550 to
  950 in 40 s (six pairs), where random neighbourhoods of the same size
  gained under 50 and solutions of one search 35 s apart differ in under 1%
  of the switches. In the solver, with two threads and 180 s, the
  incumbent at the time limit is better by 209 on `hard_10-03_1340` and by
  85 on `diverse_10-01_0052` (means over 4 seeds, noisy). But where the gap
  can be closed, the main solver misses the helper's solutions until the
  crossover: the dispatch suite takes 1.08 times the main thread's
  instructions (46 runs), the old hard tick 1.4-1.7 times. Hence an option.

LP re-solves (all MIP solves benefit; same simplex iterations). Graph LNS
and branch and bound solve many LPs with a few iterations each, so the fixed
cost of a solve dominates:
- Dual steepest-edge weights are kept over basis changes and cut additions
  for LPs above 20000 rows (`simplex_dse_exact_init_max_rows`), instead of
  being recomputed by one BTRAN per row.
- Values that are still fresh are not recomputed; the row-wise matrix is
  kept when the LP did not change; the simplex random vectors are kept
  (`simplex_keep_random_vectors`); work vectors are reused.
- After bound changes alone, the dual values of the last dual simplex solve
  are kept rather than recomputed (one BTRAN and a full PRICE), if it ended
  optimal without perturbed or shifted costs. The basis and the costs are
  checked by hash, since some cost changes reach the simplex unannounced.
  Over 600 re-solves of `hard_10-03_1340` after single fixings, this uses
  4.9% fewer instructions with the same iterations; MIP search paths are
  unchanged.
- Highs::run no longer assesses coefficient ranges without output, and the
  NLA and solution debug checks (which copied the LP and HighsInfo) only
  run at a debug level.
- After a solve, the MIP's LP relaxation only has the absolute primal and
  dual infeasibilities assessed, in one pass (`full_lp_kkt_check`), rather
  than all KKT measures in three.
- A proof of infeasibility is checked unscaled without rebuilding the
  row-wise matrix twice (once unscaled, once scaled again for the next
  solve), with a product over the proof's rows only. Most LNS flip LPs on
  the hard tick are infeasible.

Low-level (results bit for bit the same; search paths unchanged): the
code runs at about 3.2 instructions per cycle on the M1, so the waste is in
mispredicted branches rather than memory. After a solve whose result is
over 10% dense, `HVector::reIndex` rescans the array with a branch on each
value, and the nonzeros of a dense FTRAN result are effectively random:
it is now branchless (the hard tick's root node 4.3% fewer cycles), as is
the store in `priceByColumn` (0.6%). Row propagation computed every
element's implied bound in double-double arithmetic: a double estimate with
a wide margin now skips the elements whose bound cannot be tighter (2.2%).
Over nine dispatch and MIPLIB instances, 0.962 of the cycles (geometric
mean). Not worth it: LTO (unity builds already inline across files); PGO
(0.977, an option when building the production wheel); branchless
infeasibility sweeps (their branches predict well: 1.8% slower) and
triangular solves (the columns are long: 9% slower).

Cut separation (`mip/HighsPathSeparator.cpp`): an aggregation, or a path for
a path mixing cut, whose transformation has no integer column at a
fractional value (counting those brought in by variable bound substitution)
cannot give a violated cut, so cut generation is skipped for it (72% of the
calls on the hard tick). The cuts are the same except for the cover
separators' random tie-breaking.

The analytic centre (an IPX solve) is skipped where graph LNS suits the
model, and with a concurrent LNS helper; its time limit is what is left of
the MIP's.

Time limit: production stops at 180 s, so overruns matter. Central
rounding's line search and randomized rounding fix every integer column
with a propagation after each fixing and never checked the limits: on
MIPLIB's germanrr (vanilla too) the former ran about 1300 s past a 300 s
limit, and on blp-ar98 the latter overran by a cut round. Both now check
the MIP's limits as they go, as does presolve's dominated columns check
(which took about 150 s on the dense eilA101-2, past any time limit below
that). With a 30 s limit, no MIPLIB instance here now runs more than 8 s
over.

An LP solution whose fractional integers can all be rounded along their
locks is rounded and taken as a solution without a check, which is only
valid if the LP solution is primal feasible. After an unknown simplex
status (a taboo basis), it was taken anyway: a column fixed to 1 at 0.57
came out fractional, and the main solver rejected the point in the
original space after a repair LP (vanilla does this too, e.g. on
blp-ic98). The rounding now needs a primal feasible LP solution.

## Results

Single thread, `mip_rel_gap = 0.01`. CPU times were measured on a busy
machine, so compare them rather than reading them as absolute; instructions
retired (load-independent) are given where available.

| Dispatch suite (23 instances) | vanilla 1.15.1 | this branch |
|---|---|---|
| reach 1% within the 300 s (wall) limit | 14 | 23 |
| total CPU time | over 2264 s | 313 s |
| hard tick | time limit, gap 2.03% | 26 s |
| `_wind185` | time limit (its 0.02% solution came at the limit) | 9.7 s |
| `dm_full_pert_s1_ramp` | time limit, 9.3% | 45 s |
| `dm_full_randsoc_ramp` | time limit, no solution | 57 s |
| `lambda_..._080458` | 107 s | 17 s |
| `dm_small_pert_s1_ramp` | 106 s | 4.4 s |

Against this branch as it was before this round of work (graph LNS
already in, closer to what production runs): 823 s of CPU time and 22 of
23 (the hard tick took 173 s of CPU without reaching 1% in 300 s of wall
time).

Every instance that vanilla solves is at least 4.5 times faster (9.2 times
at the median); the 9 it does not solve within 300 s take 6 s to 57 s of
CPU time. The hard tick over 16 random seeds: all certified, mean 167G
instructions, median 160G (between 81G and 231G); vanilla does not reach
1% in 300 s with any seed.

With two threads (default threads, as in production), IPX races the dual
simplex on the root LP and a helper thread does its own root cuts and
searches neighbourhoods alongside; the main solver takes the helper's bound
while at the root, and its cuts once the helper's cut loop is done. Wall
times of two-thread runs on this machine say little (see Benchmarking), so
the main thread's own instructions retired are compared instead: what the
solve would take on two uncontended cores. Over the suite with 2 seeds, the
main thread retires 0.70 (geometric mean) of the single-thread solve's
instructions, 1670G against 2557G in total; 2 of the 46 runs need more. The
race accounts for much of that: without it the main thread needs 1.14 times
as much (IPX wins on the ramp models, the lambda ticks and the small
models: `lambda_..._080458` 18G against 51G-56G). On two hyperthreads of
one core, each thread runs slower while both are busy, which would eat
most of the gain.

Through the production script (`dispatch_milp_2026_09_30/solve_mps.py`,
highspy built from this branch with `pip wheel .` for arm64, default
threads, 180 s) on this machine, with other work running: all 23 instances
reach 1%; `dm_small_windlull_noramp` 53 s, `dm_full_randsoc_ramp` 51 s,
`dm_full_pert_s1_ramp` 39 s, the hard tick 27 s, `dm_full_pert_s2_ramp`
19 s, everything else 12 s or less.

The 21 row/column-permuted copies in `~/code/oopt/bench/perm` all reach 1%
too (single thread).

Held-out instances: two copies of each of the 23 with every distinct
objective coefficient scaled by its own lognormal(0, 0.15) factor (equal
coefficients stay equal, as prices per period do). Single thread, 300 s:
this branch reaches 1% on all 46, vanilla on 41 (not on either copy of the
hard tick, 1.54% and 1.95%, nor three full ramp models), with 0.108 of
vanilla's instructions (geometric mean; 1031G against at least 10194G in
total).

MIPLIB regression set (31 instances, 300 s), instructions retired against
vanilla, shifted geometric mean over the runs that both solve, 2 random
seeds:
- at the default gap (graph LNS off): 0.919 over 47 runs; this branch also
  solves 2 runs that vanilla does not (mas76, gmu-35-40), and vice versa
  none;
- at `mip_rel_gap = 0.01` (graph LNS active): 0.916 over 60 runs; mas76
  is solved only by this branch (both seeds).

The other 59 MIPLIB instances here (mostly not solved within 300 s), run
side by side at the default gap: this branch solves 15 and vanilla 13 (the
12 both solve: 0.962); of the rest, the final gap is better with this
branch on 11, with vanilla on 9, and about the same on 39. Vanilla runs
past the time limit on germanrr (see above).

## Options

- `mip_heuristic_run_graph_lns` (default true)
- `mip_concurrent_helper` (default true)
- `mip_concurrent_crossover` (default false)
- `simplex_dse_exact_init_max_rows` (default no limit; the MIP's LP
  relaxation uses 20000)
- `simplex_keep_random_vectors` (default false; true in the MIP's LP
  relaxation)
- `full_lp_kkt_check` (default true; false in the MIP's LP relaxation)

## Tried and reverted

- A {0,1/2}-CG (Caprara-Fischetti/KZK) separator and a richer cMIR scale
  search: no gain on the dispatch models, regressions on MIPLIB (air04,
  mik-250).
- Strong-branching lookahead: regressions on pg, mik, neos-911970.
- Approximate DSE weights on small LPs: large MIPLIB variance.
- Devex pricing, no cost perturbation, a cutoff margin, larger dive chunks
  and faster neighbourhood growth for the LNS: no gain.
- Neighbourhood LPs started from the incumbent with the primal simplex:
  twice the iterations of the dual simplex from the last LP's basis.
- In a dive, fixing a rounding that conflicts with the chunk so far the
  other way, or leaving it out, instead of undoing the chunk: the dispatch
  suite took 14-16% longer.
- With two threads, the helper running the quick search and LNS on the LP
  without cuts while the main solver goes straight to the root cuts (with
  or without passing the main solver's cuts to the helper): faster on the
  ramp models, slower on the hard tick, about even overall.
- Without the IPX race when a helper runs: 1.14 times the main thread's
  instructions over the suite; only `dm_small_windlull_noramp`, where IPX's
  crossover vertex leads the cut loop and LNS astray, gains. Without the
  import of the helper's cuts, or with a full cut loop in the helper: no
  gain there either.
- Devex pricing for the LNS LPs (again): 2-3 times cheaper per iteration
  on the ramp models' dense LPs, but the hard tick took twice the
  instructions (the dual steepest-edge weights kept between re-solves pay).
  For the root LP alone, Devex is 2-3 times faster on the ramp models and
  16% on the hard tick but 37% slower on `lambda_..._080458`; the race with
  IPX covers the ramp models when there is a second core.
- In the neighbourhood branch and bound, starting a node's other child from
  the node's simplex iterate (only the deepest node's is kept, so few
  qualify) or from its basis (a refactorization): no gain, and twice the
  iterations with the basis.
- RINS at the root after a stalled deep search on the hard tick: 128 s
  without an improvement.
- No cost perturbation at the start of a dual simplex solve that starts
  dual feasible, however large the primal infeasibilities (HiGHS skips it
  only below 1e-3). Re-solves of `hard_10-03_1340` cost 9% less, but on
  MIPLIB re-solves the effect is chaotic: mzzv42z +49%, swath1 +14%,
  physiciansched6-2 -32%, or +83% if only one infeasibility is allowed. MIP
  totals: MIPLIB 0.996 over 62 runs (the two seeds 0.95 and 1.05), with
  1.29 times the LP iterations; the dispatch suite 0.917 (seeds 1.04 and
  0.80).
- Checking the factorization's accuracy only every 50 updates: re-solves
  of `hard_10-03_1340` cost 5% less, but it changes search paths
  (neos-911970) and drops a numerical safeguard.
- A second presolve pass from the start (see the 10-03 tick above): on
  `hard_10-03_1340` the root LP is 42% cheaper and the bound after the cut
  loop is -251535 rather than -251623 (one run). Over the dispatch suite it
  took 1.034 times the instructions (the two seeds 1.18 and 0.91), with
  many of the simple models slower. Most of the extra reductions come from
  the aggregator. Running it once more at the end of the first pass, after
  a full rescan of candidates, after rescaling, or with a lower pivot
  threshold gets only 70% of them.
- A restart forced after the root when nothing is fixed: +60 to +90 on the
  bound of `hard_10-03_1340`, but the dispatch suite is unchanged (0.998;
  almost all of it finishes at the root).
- Plain triangular solves instead of hyper-sparse ones, for the dense
  inverses of the dispatch LPs: 3-5% fewer cycles per re-solve, but the
  first LP takes twice as long.
- Keeping the basic primal values over re-solves when no nonbasic value
  changed (exact, checked against a copy of the nonbasic values): only 28%
  of re-solves on `hard_10-03_1340` and 39% on neos-911970 qualify, and
  computePrimal is one FTRAN, so the gain was 0.1%.
- Other ways to a better incumbent on `hard_10-03_1340` (prototypes, single
  thread): relax-and-fix over time windows (later windows infeasible or
  slow); switching costs over a growing horizon of periods, warm-started
  (-246641 in 150 s); banks made integer in groups, lowest initial state of
  charge first, earlier groups fixed (-246358 in 97 s); proximity search
  (Fischetti and Monaci) from a 70 s solution (+25 in 110 s). The solver
  itself has about -247200 after 70 s and -247600 after 180 s.
- Incremental dual values after removing cost shifts at the end of a solve:
  the shifts are mostly on basic variables (entered on degenerate steps),
  and the cost vector is sparse, so a full recomputation costs about the
  same.

## Benchmarking

Use instructions retired (`/usr/bin/time -l`) or CPU time and deterministic
counts (nodes, LP iterations) rather than wall time on a shared machine, and
several random seeds: single runs of MIPLIB instances vary by factors of two
or more with any change to the search.

On this M1 (4 performance and 4 efficiency cores), wall times of
multi-threaded runs depend on which cores the threads get: a deterministic
single-thread solve took 3.9 s alone and 14.6 s next to three busy loops.
For two-thread runs, compare the main thread's instructions retired
(`thread_selfcounts(1, ...)` from libsystem_kernel at the end of the solve).
Also, `/opt/miniconda3/bin/python3` here is an x86_64 build running under
Rosetta: highspy wheels built with it run about 2.5 times slower than
native (use `/opt/homebrew/bin/python3.13`).
