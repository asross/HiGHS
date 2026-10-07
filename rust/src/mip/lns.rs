//! Graph LNS (highs/mip/HighsGraphLns.cpp): the decision columns, the
//! neighbourhoods (seed choice and breadth-first search over the
//! variable/constraint graph), the reduced-cost promise of flips, the
//! choice of move type (a UCB bandit), and the flip candidates and their
//! partners. The search that uses them is graph_lns.rs.

use crate::util::fma::ClangFma;
use crate::util::random::HighsRandom;

/// The model: column-wise and row-wise matrix (HighsMipSolverData's ARstart_
/// etc.)
pub struct Graph<'a> {
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    pub ar_start: &'a [i32],
    pub ar_index: &'a [i32],
    pub ar_value: &'a [f64],
}

impl Graph<'_> {
    #[inline]
    fn col(&self, j: usize) -> std::ops::Range<usize> {
        self.a_start[j] as usize..self.a_start[j + 1] as usize
    }
    #[inline]
    fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.ar_start[i] as usize..self.ar_start[i + 1] as usize
    }
}

fn integral(v: f64) -> bool {
    (v - v.round()).abs() <= 1e-9
}

/// HighsPrimalHeuristics::setupDecisionCols: the integer columns that an LP
/// vertex does not always make integral (`integrality` 0 = continuous)
pub fn decision_cols(
    g: &Graph,
    row_lower: &[f64],
    row_upper: &[f64],
    integrality: &[u8],
    intcols: &[i32],
) -> Vec<i32> {
    let mut implied = vec![false; integrality.len()];
    let mut out = Vec::new();
    for &c in intcols {
        let col = c as usize;
        let mut ok = true;
        'col: for p in g.col(col) {
            if (g.a_value[p].abs() - 1.0).abs() > 1e-9 {
                ok = false;
                break;
            }
            let row = g.a_index[p] as usize;
            if (row_lower[row] != f64::NEG_INFINITY && !integral(row_lower[row]))
                || (row_upper[row] != f64::INFINITY && !integral(row_upper[row]))
            {
                ok = false;
                break;
            }
            for q in g.row(row) {
                let k = g.ar_index[q] as usize;
                if !integral(g.ar_value[q]) || integrality[k] == 0 || (k != col && implied[k]) {
                    ok = false;
                    break 'col;
                }
            }
        }
        if ok {
            implied[col] = true;
        } else {
            out.push(c);
        }
    }
    out
}

/// HighsPrimalHeuristics::LnsMove
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct LnsMove {
    pub size: f64,
    pub rate: f64,
    pub tried: i32,
    pub improved: i32,
}

/// The deep search's choice of move type: each once, then the best rate
/// of gap closed per LP iteration plus an exploration bonus. Counts the
/// try in `deep_tries`.
pub fn select_type(types: &[LnsMove; 4], deep_tries: &mut [i32; 4], flips_exhausted: bool) -> usize {
    let mut max_rate = 0.0;
    let mut num_tried = 0;
    for t in types {
        if max_rate < t.rate {
            max_rate = t.rate;
        }
        num_tried += t.tried;
    }
    let mut best_score = -1.0;
    let mut ty = 3;
    for (t, m) in types.iter().enumerate() {
        if t == 2 && flips_exhausted {
            continue;
        }
        let score = if m.tried == 0 || (deep_tries[t] == 0 && t != 3) {
            f64::INFINITY
        } else {
            let base = if max_rate > 0.0 { m.rate / max_rate } else { 0.0 };
            0.5f64.mul_add_c(((num_tried as f64).ln() / m.tried as f64).sqrt(), base)
        };
        if score > best_score {
            best_score = score;
            ty = t;
        }
    }
    deep_tries[ty] += 1;
    ty
}

/// The state of one graphLNS call
pub struct Lns {
    pub decision: Vec<i32>,
    pub is_decision: Vec<u8>,
    pub in_n: Vec<u8>,
    seen_row: Vec<u8>,
    row_decisions: Vec<i32>,
    promise: Vec<f64>,
    promise_total: f64,
    pub neighbourhood: Vec<i32>,
    frontier: Vec<i32>,
    next: Vec<i32>,
    touched_rows: Vec<i32>,
    disagree: Vec<i32>,
    pub flip_cands: Vec<(f64, i32)>,
    pub partners: Vec<(f64, i32)>,
}

/// Rows with at most this many decision columns (type 1 neighbourhoods)
const SHORT_ROW: i32 = 4;

/// byGain: larger gain first, then smaller column (a strict total order on
/// distinct columns, so any sort gives the C++ pdqsort's order)
fn by_gain(a: &(f64, i32), b: &(f64, i32)) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    if a.0 > b.0 || (a.0 == b.0 && a.1 < b.1) {
        Less
    } else if b.0 > a.0 || (a.0 == b.0 && b.1 < a.1) {
        Greater
    } else {
        Equal
    }
}

impl Lns {
    pub fn new(g: &Graph, num_row: usize, decision: &[i32]) -> Self {
        let num_col = g.a_start.len() - 1;
        let mut is_decision = vec![0u8; num_col];
        let mut row_decisions = vec![0i32; num_row];
        for &c in decision {
            is_decision[c as usize] = 1;
            for p in g.col(c as usize) {
                row_decisions[g.a_index[p] as usize] += 1;
            }
        }
        Lns {
            decision: decision.to_vec(),
            is_decision,
            in_n: vec![0; num_col],
            seen_row: vec![0; num_row],
            row_decisions,
            promise: vec![0.0; num_col],
            promise_total: 0.0,
            neighbourhood: Vec::new(),
            frontier: Vec::new(),
            next: Vec::new(),
            touched_rows: Vec::new(),
            disagree: Vec::new(),
            flip_cands: Vec::new(),
            partners: Vec::new(),
        }
    }

    /// pickSeed
    fn pick_seed(&mut self, deep: bool, inc: &[f64], relax: &[f64], glo: &[f64], gup: &[f64], rng: &mut HighsRandom) -> i32 {
        let mut seed: i32 = -1;
        let r = if deep { rng.fraction() } else { 0.5f64.mul_add_c(rng.fraction(), 0.5) };
        if r < 0.5 && self.promise_total > 0.0 {
            let mut pick = rng.fraction() * self.promise_total;
            for &c in &self.decision {
                let col = c as usize;
                pick -= self.promise[col];
                if pick <= 0.0 && self.promise[col] > 0.0 {
                    seed = c;
                    break;
                }
            }
            if seed != -1 && self.in_n[seed as usize] != 0 {
                seed = -1;
            }
        }
        if seed == -1 && r < 0.85 {
            self.disagree.clear();
            for &c in &self.decision {
                let col = c as usize;
                if self.in_n[col] == 0 && glo[col] != gup[col] && (inc[col] - relax[col]).abs() > 0.5 {
                    self.disagree.push(c);
                }
            }
            if !self.disagree.is_empty() {
                seed = self.disagree[rng.integer_below(self.disagree.len() as i32) as usize];
            }
        }
        let unusable = |s: &Self, c: i32| s.in_n[c as usize] != 0 || glo[c as usize] == gup[c as usize];
        if seed == -1 {
            let mut tries = 0;
            while tries < 50 && (seed == -1 || unusable(self, seed)) {
                seed = self.decision[rng.integer_below(self.decision.len() as i32) as usize];
                tries += 1;
            }
            if unusable(self, seed) {
                seed = -1;
            }
        }
        seed
    }

    /// The neighbourhood of `size` decision columns for move type `ty`
    /// (1: over short rows only), by breadth-first search from seeds
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        g: &Graph,
        ty: i32,
        size: usize,
        deep: bool,
        inc: &[f64],
        relax: &[f64],
        glo: &[f64],
        gup: &[f64],
        rng: &mut HighsRandom,
    ) {
        for &c in &self.neighbourhood {
            self.in_n[c as usize] = 0;
        }
        for &r in &self.touched_rows {
            self.seen_row[r as usize] = 0;
        }
        self.neighbourhood.clear();
        self.touched_rows.clear();
        while self.neighbourhood.len() < size {
            let seed = self.pick_seed(deep, inc, relax, glo, gup, rng);
            if seed == -1 {
                break;
            }
            self.neighbourhood.push(seed);
            self.in_n[seed as usize] = 1;
            self.frontier.clear();
            self.frontier.push(seed);
            while !self.frontier.is_empty() && self.neighbourhood.len() < size {
                self.next.clear();
                'front: for fi in 0..self.frontier.len() {
                    let j = self.frontier[fi] as usize;
                    for p in g.col(j) {
                        let row = g.a_index[p] as usize;
                        if self.seen_row[row] != 0 {
                            continue;
                        }
                        if ty == 1 && self.row_decisions[row] > SHORT_ROW {
                            continue;
                        }
                        self.seen_row[row] = 1;
                        self.touched_rows.push(row as i32);
                        for q in g.row(row) {
                            let k = g.ar_index[q];
                            if self.is_decision[k as usize] == 0 || self.in_n[k as usize] != 0 {
                                continue;
                            }
                            self.in_n[k as usize] = 1;
                            self.neighbourhood.push(k);
                            self.next.push(k);
                            if self.neighbourhood.len() >= size {
                                break;
                            }
                        }
                        if self.neighbourhood.len() >= size {
                            break 'front;
                        }
                    }
                }
                std::mem::swap(&mut self.frontier, &mut self.next);
            }
        }
    }

    /// The first-order gain of flipping each decision column outside the
    /// neighbourhood (fixed at the incumbent), from the reduced costs `rc`
    pub fn update_promise(&mut self, inc: &[f64], glo: &[f64], gup: &[f64], rc: &[f64]) {
        for &c in &self.decision {
            let col = c as usize;
            if self.in_n[col] != 0 {
                continue;
            }
            let mut gain = 0.0;
            if inc[col] <= glo[col] + 0.5 {
                gain = -rc[col];
            } else if inc[col] >= gup[col] - 0.5 {
                gain = rc[col];
            }
            self.promise[col] = if 0.0 < gain { gain } else { 0.0 };
        }
        let mut total = 0.0;
        for &c in &self.decision {
            total += self.promise[c as usize];
        }
        self.promise_total = total;
    }

    /// The flip search's candidates: binary decision columns whose flip
    /// from `cur` promises more than `feastol`, best first
    pub fn flip_candidates(&mut self, cur: &[f64], glo: &[f64], gup: &[f64], rc: &[f64], feastol: f64) {
        self.flip_cands.clear();
        for &c in &self.decision {
            let col = c as usize;
            if gup[col] - glo[col] != 1.0 {
                continue;
            }
            let g = gain(col, cur, glo, gup, rc);
            if g > feastol {
                self.flip_cands.push((g, c));
            }
        }
        self.flip_cands.sort_unstable_by(by_gain);
    }

    /// findPartners: binary decision columns in j's rows whose flip moves
    /// the row activity against j's, best promised gain first, once each
    pub fn find_partners(&mut self, g: &Graph, j: usize, cur: &[f64], glo: &[f64], gup: &[f64], rc: &[f64]) {
        let dj = flipped(j, cur, glo, gup) - cur[j];
        self.partners.clear();
        for p in g.col(j) {
            let row = g.a_index[p] as usize;
            let aj = g.a_value[p];
            for q in g.row(row) {
                let k = g.ar_index[q] as usize;
                if k == j || self.is_decision[k] == 0 || gup[k] - glo[k] != 1.0 {
                    continue;
                }
                if aj * dj * g.ar_value[q] * (flipped(k, cur, glo, gup) - cur[k]) >= 0.0 {
                    continue;
                }
                self.partners.push((gain(k, cur, glo, gup, rc), k as i32));
            }
        }
        self.partners.sort_unstable_by(by_gain);
        self.partners.dedup_by(|a, b| a.1 == b.1);
    }
}

#[inline]
fn flipped(col: usize, cur: &[f64], glo: &[f64], gup: &[f64]) -> f64 {
    if cur[col] <= glo[col] {
        gup[col]
    } else {
        glo[col]
    }
}

#[inline]
fn gain(col: usize, cur: &[f64], glo: &[f64], gup: &[f64], rc: &[f64]) -> f64 {
    if flipped(col, cur, glo, gup) > cur[col] {
        -rc[col]
    } else {
        rc[col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfs_and_partners() {
        // x0 + x1 <= 1, x1 + x2 <= 1 (column-wise and row-wise)
        let g = Graph {
            a_start: &[0, 1, 3, 4],
            a_index: &[0, 0, 1, 1],
            a_value: &[1.0; 4],
            ar_start: &[0, 2, 4],
            ar_index: &[0, 1, 1, 2],
            ar_value: &[1.0; 4],
        };
        let d = decision_cols(&g, &[f64::NEG_INFINITY; 2], &[1.0; 2], &[1; 3], &[0, 1, 2]);
        assert_eq!(d, vec![1]); // x0 is implied, x1 not (shares a row with x0), x2 implied
        let mut lns = Lns::new(&g, 2, &[0, 1, 2]);
        let mut rng = HighsRandom::new(0);
        let z = [0.0; 3];
        lns.build(&g, 0, 3, true, &z, &z, &z, &[1.0; 3], &mut rng);
        let mut n = lns.neighbourhood.clone();
        n.sort();
        assert_eq!(n, vec![0, 1, 2]);
        // flipping x1 up: x0 and x2 at 1 would move their rows down
        let cur = [1.0, 0.0, 1.0];
        lns.find_partners(&g, 1, &cur, &z, &[1.0; 3], &[-1.0, -2.0, -3.0]);
        assert_eq!(lns.partners.iter().map(|p| p.1).collect::<Vec<_>>(), vec![0, 2]);
    }
}
