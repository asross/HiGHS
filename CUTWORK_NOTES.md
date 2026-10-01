# Cut-separator experiments (WIP branch)

Experimental work on HiGHS's MIP cut separators, branched from v1.15.1. Two
independent changes, both exploratory. Nothing here is PR-ready without a
proper multi-seed MIPLIB benchmark.

## 1. `{0,1/2}`-Chvátal–Gomory (zero-half) separator — `HighsModkSeparator.cpp`

HiGHS's existing mod-k separator (`HighsGFkSolve`) enumerates a *basis* of the
GF(2) null space — it finds parity-satisfying row combinations but does not
optimize for violation. This adds a **Caprara–Fischetti / Koster–Zymolka–Kutschka**
style separation run *before* the GFk enumeration:

- Build the mod-2 system from rows with LP slack `< 1` (Lemma 2 restatement:
  a combination `v` gives a violated cut iff `v·b` is odd and
  `v·slack + (leftover odd columns)·x* < 1`).
- Preprocessing: Lemma 3 reductions (drop slack≥1 rows, unit columns fold into
  slack, …), **Proposition 5 Gaussian-elimination pivoting on slack-0 rows**
  (eliminates a column globally, folding its `x*` into slack — this is what
  reduces the general, >2-odd-column rows), and Lemma 4 (a zero row with odd
  RHS and slack<1 is a violated cut).
- Residual: bounded `≤2`-row combination enumeration.
- **Cut selection**: emit only the top-N by violation (avoids flooding the LP).
- All candidate aggregations go through the existing (validity-checked)
  `HighsCutGeneration::generateCut`, so only valid cuts are ever added.

Reference: Koster, Zymolka, Kutschka, "Algorithms to separate {0,1/2}-Chvátal-
Gomory cuts", ZIB-Report 07-10 (2007); Caprara & Fischetti, Math. Prog. 74 (1996).

Status: validated net-positive on zero-half-sensitive MIPLIB instances
(p0201 13→3 B&B nodes, mas76/mod010 improvements), neutral elsewhere; within
run-to-run variance on timeout instances. A legitimate, if modest, improvement
over the GFk enumeration; needs a full benchmark to quantify for upstream.

## 2. cMIR richer scale search — `HighsCutGeneration.cpp`

`cmirCutGenerationHeuristic` already searches `δ = |coeff|` scales, an
integral-scale candidate, and `bestδ × {2,4,8}` multipliers, plus a greedy
complementation-flip search. This adds `δ/2` and `2δ` candidates per integer
coefficient. The dedup + best-efficacy selection keeps only the strongest cut,
so it can only strengthen (never weaken) the generated cut.

Status: deterministic but tiny effect (partly duplicates the existing ×2/4/8
multiplier logic); **non-monotonic** — a small ladder helps, more candidates
(δ/3, δ/4) hurt. Marginal; likely not worth upstreaming as-is. Left in as an
experiment.

## Findings summary

The single-row cMIR/GMI engine in HiGHS is already sophisticated (scale search,
multiplier search, complementation flips, superadditive lifting, lifted covers,
zero-half, VUB substitution in `HighsTransformedLp`). The remaining strength gap
to commercial solvers appears **diffuse** (accumulated tuning), not attributable
to a single missing technique. Flow-cover cuts were considered but are largely
subsumed by HiGHS's VUB-aware cMIR (flow covers = MIR on VUB-substituted rows,
Marchand–Wolsey 2001). General multi-row cuts are reported as practically
disappointing in the literature.
