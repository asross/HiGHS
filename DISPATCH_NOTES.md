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

## Changes

Graph LNS (`mip/HighsGraphLns.cpp`, option `mip_heuristic_run_graph_lns`, only
for `mip_rel_gap >= 1e-3`):
- Decision columns: integer columns that every LP vertex makes integral
  (e.g. start-up indicators bounded by +-1 rows of other integers) are not
  fixed or branched on.
- Quick search after the first root LP: a dive from the LP point (each
  chunk of fixings goes up to the first rounding that propagation rules
  out, rather than being undone and quartered: on the ramp models most
  chunks met such a conflict, and dives took hundreds of LP solves), then
  dived neighbourhoods of about 400 decision columns grown by breadth-first
  search over the constraint graph from a seed column, and a flip search to
  polish. Neighbourhood LPs start from the last LP solved (not the root
  basis), which made the hard tick 21% faster. On large models it spends at
  most about one root LP's worth of LP iterations, so that where
  neighbourhood LPs are expensive (ramping) the root cuts come early: their
  bound often closes the gap with the incumbent so far.
- Deep search after the root cuts, only if the quick search brought the
  incumbent within three times the target gap (otherwise the model does not
  suit it and the usual root heuristics run as without graph LNS): small
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
- Stall rules measure progress against the gap (quick) or against what
  separates the incumbent from the target gap (deep); both stop as soon as
  the target gap is reached.

Using a second core (option `mip_concurrent_helper`, on unless
`threads = 1`, when graph LNS runs; makes the solve non-deterministic):
- IPX with crossover races the dual simplex on the first root LP (models with
  at least 10000 nonzeros); the first to finish interrupts the other. IPX is
  2-3 times faster on large dispatch LPs (`dm_full_pert_s1_ramp`: 131 s ->
  79 s with two threads).
- A helper thread runs graph LNS on a copy of the presolved model with its
  own seed, exchanging incumbents with the main solver.

LP re-solves (all MIP solves benefit; same simplex iterations). Graph LNS
and branch and bound solve many LPs with a few iterations each, so the fixed
cost of a solve dominates:
- Dual steepest-edge weights are kept over basis changes and cut additions
  for LPs above 20000 rows (`simplex_dse_exact_init_max_rows`), instead of
  being recomputed by one BTRAN per row.
- Values that are still fresh are not recomputed; the row-wise matrix is
  kept when the LP did not change; the simplex random vectors are kept
  (`simplex_keep_random_vectors`); work vectors are reused.
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

Cut separation (`mip/HighsPathSeparator.cpp`): an aggregation, or a path for
a path mixing cut, whose transformation has no integer column at a
fractional value (counting those brought in by variable bound substitution)
cannot give a violated cut, so cut generation is skipped for it (72% of the
calls on the hard tick). The cuts are the same except for the cover
separators' random tie-breaking.

The analytic centre (an IPX solve) is skipped where graph LNS suits the
model, and with a concurrent LNS helper; its time limit is what is left of
the MIP's.

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

Every instance that vanilla solves is at least 4.5 times faster (9.2 times
at the median); the 9 it does not solve within 300 s take 6 s to 57 s of
CPU time. The hard tick over 16 random seeds: all certified, mean 167G
instructions, median 160G (between 81G and 231G); vanilla does not reach
1% in 300 s with any seed.

With two threads (`threads = 2`, close to production's two vCPUs), on a
lightly loaded machine: all 23 instances reach 1%, in about 300 s of wall
time for the suite; the slowest are the full ramp models (28 s to 66 s over
three seeds: `dm_full_randsoc_ramp` 43 s to 66 s, `dm_full_pert_s1_ramp`
31 s to 53 s), then the hard tick (26 s); everything else takes under 25 s.
IPX wins the root LP race on the large dispatch LPs
(`dm_full_pert_s1_ramp`: 131 s -> 79 s with the race alone, earlier), and a
helper thread searches neighbourhoods alongside.

The 21 row/column-permuted copies in `~/code/oopt/bench/perm` all reach 1%
too (single thread).

MIPLIB regression set (31 instances, 300 s), instructions retired against
vanilla, shifted geometric mean over the runs that both solve, 2 random
seeds:
- at the default gap (graph LNS off): 0.924 over 47 runs; this branch also
  solves 2 runs that vanilla does not, and vice versa none;
- at `mip_rel_gap = 0.01` (graph LNS active): 0.916 over 60 runs; mas76
  is solved only by this branch (both seeds).

## Options

- `mip_heuristic_run_graph_lns` (default true)
- `mip_concurrent_helper` (default true)
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

## Benchmarking

Use instructions retired (`/usr/bin/time -l`) or CPU time and deterministic
counts (nodes, LP iterations) rather than wall time on a shared machine, and
several random seeds: single runs of MIPLIB instances vary by factors of two
or more with any change to the search.
