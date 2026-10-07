//! Feasibility jump (highs/mip/feasibilityjump.hh, the solver behind
//! HighsMipSolverData::feasibilityJump in HighsFeasibilityJump.cpp).
//!
//! Same moves, scores, random draws (std::mt19937 and libc++'s
//! uniform_real_distribution, ported below) and effort counts as the C++,
//! so the same solution comes out. Only `Lte` and `Gte` rows exist (the
//! caller splits equations) and continuous columns are never relaxed.
//!
//! Layout, not arithmetic, differs: the matrix is CSR both ways (the C++
//! keeps a vector of coefficients per variable and per constraint), the
//! hot per-constraint fields are one struct, the jump value computation
//! uses no per-cell allocation, and the scores of a constraint's old and
//! new left-hand side are computed once per constraint rather than once
//! per variable in it. The one order that may differ is that of the jump
//! candidates sorted with equal keys, which only differ in the sign of a
//! zero (`ceil(-0.5)` against a bound of 0) and give the same moves.

use crate::util::fma::ClangFma;

/// std::mt19937
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// std::uniform_real_distribution<double>(0, 1) (generate_canonical
    /// from two draws; the product is exact, so fusing it does not matter)
    pub fn uniform01(&mut self) -> f64 {
        let lo = self.next_u32() as f64;
        let hi = self.next_u32() as f64;
        (lo + hi * 4294967296.0) / 18446744073709551616.0
    }
}

#[derive(Clone, Copy)]
struct Cell {
    idx: u32,
    coeff: f64,
}

#[derive(Clone, Copy)]
struct Con {
    rhs: f64,
    weight: f64,
    lhs: f64,
    violated_idx: i32,
    lte: bool,
}

/// A variable's value and its jump move (value and score), with its
/// position in the good set: what the move updates read of a neighbour,
/// in one place
#[derive(Clone, Copy)]
struct Var {
    x: f64,
    value: f64,
    score: f64,
    good_idx: i32,
}

/// Constraint::score: 0 if satisfied, else -|violation| (std::max(0., d)
/// is `0 < d ? d : 0`)
#[inline(always)]
fn score(lte: bool, rhs: f64, lhs: f64) -> f64 {
    let d = if lte { lhs - rhs } else { rhs - lhs };
    -(if 0.0 < d { d } else { 0.0 })
}

#[inline(always)]
fn eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The problem as the C++ wrapper builds it: columns with bounds
/// (rounded for integers), objective (times the sense) and integrality,
/// and the row-wise matrix with row bounds
pub struct Input<'a> {
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub col_cost: &'a [f64],
    pub col_integer: &'a [u8],
    pub ar_start: &'a [i32],
    pub ar_index: &'a [i32],
    pub ar_value: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub seed: u32,
    pub equality_tolerance: f64,
    pub violation_tolerance: f64,
    pub max_total_effort: usize,
    pub max_effort_since_improvement: usize,
}

const RANDOM_VAR_PROBABILITY: f64 = 0.001;
const RANDOM_CELL_PROBABILITY: f64 = 0.01;
const MAX_MOVES_TO_EVALUATE: usize = 25;
const LOGGING_FREQUENCY: i32 = 100000;
const LOGGING_HEADER_FREQUENCY: i32 = 50;
const MIN_EFFORT_TO_LOGGING: usize = 500 * LOGGING_FREQUENCY as usize;
const MIN_RELATIVE_OBJECTIVE_IMPROVEMENT: f64 = 1e-4;
const CALLBACK_EFFORT: usize = 500000;
const LOG_INFO: i32 = 1;
const LOG_DETAILED: i32 = 2;
const LOG_VERBOSE: i32 = 3;

struct Fj<'a> {
    lb: Vec<f64>,
    ub: Vec<f64>,
    obj: Vec<f64>,
    integer: Vec<bool>,
    vstart: Vec<usize>,
    vcells: Vec<Cell>,
    cons: Vec<Con>,
    cstart: Vec<usize>,
    ccells: Vec<Cell>,
    violated: Vec<u32>,
    incumbent_objective: f64,
    vs: Vec<Var>,
    good: Vec<u32>,
    shift: Vec<(f64, f64)>,
    rng: Mt19937,
    eq_tol: f64,
    viol_tol: f64,
    best_objective: f64,
    objective_weight: f64,
    best_violation_score: usize,
    effort_at_last_callback: usize,
    effort_at_last_improvement: usize,
    effort_at_last_logging: usize,
    total_effort: usize,
    weight_update_increment: f64,
    n_bumps: usize,
    max_total_effort: usize,
    max_effort_since_improvement: usize,
    found: bool,
    log: &'a mut dyn FnMut(i32, &str),
    logging_on: bool,
}

/// Runs feasibility jump from `x` (the initial values); on return `x`
/// holds the last solution found, if any (the return value).
/// `log(type, message)` is called with HighsLogType values only if
/// `logging_on`.
pub fn solve(inp: &Input, x: &mut [f64], logging_on: bool, log: &mut dyn FnMut(i32, &str)) -> bool {
    let n = inp.col_lower.len();
    // Constraints, in the order the C++ wrapper adds them
    let mut cons = Vec::new();
    let mut cstart = vec![0usize];
    let mut ccells = Vec::new();
    let mut vcount = vec![0usize; n + 1];
    for r in 0..inp.row_lower.len() {
        let (s, e) = (inp.ar_start[r] as usize, inp.ar_start[r + 1] as usize);
        for (lte, rhs) in [(false, inp.row_lower[r]), (true, inp.row_upper[r])] {
            if !rhs.is_finite() || s == e {
                // An empty row adds no constraint (its check is unused)
                continue;
            }
            for p in s..e {
                ccells.push(Cell { idx: inp.ar_index[p] as u32, coeff: inp.ar_value[p] });
                vcount[inp.ar_index[p] as usize + 1] += 1;
            }
            cstart.push(ccells.len());
            cons.push(Con { rhs, weight: 1.0, lhs: f64::NAN, violated_idx: -1, lte });
        }
    }
    for j in 0..n {
        vcount[j + 1] += vcount[j];
    }
    let vstart = vcount.clone();
    let mut vcells = vec![Cell { idx: 0, coeff: 0.0 }; ccells.len()];
    for c in 0..cons.len() {
        for cell in &ccells[cstart[c]..cstart[c + 1]] {
            let j = cell.idx as usize;
            vcells[vcount[j]] = Cell { idx: c as u32, coeff: cell.coeff };
            vcount[j] += 1;
        }
    }
    let mut fj = Fj {
        lb: inp.col_lower.to_vec(),
        ub: inp.col_upper.to_vec(),
        obj: inp.col_cost.to_vec(),
        integer: inp.col_integer.iter().map(|&t| t != 0).collect(),
        vstart,
        vcells,
        cons,
        cstart,
        ccells,
        violated: Vec::new(),
        incumbent_objective: f64::NAN,
        vs: x.iter().map(|&x| Var { x, value: 0.0, score: 0.0, good_idx: -1 }).collect(),
        good: Vec::new(),
        shift: Vec::new(),
        rng: Mt19937::new(inp.seed),
        eq_tol: inp.equality_tolerance,
        viol_tol: inp.violation_tolerance,
        best_objective: f64::INFINITY,
        objective_weight: 0.0,
        best_violation_score: usize::MAX,
        effort_at_last_callback: 0,
        effort_at_last_improvement: 0,
        effort_at_last_logging: 0,
        total_effort: 0,
        weight_update_increment: 1.0,
        n_bumps: 0,
        max_total_effort: inp.max_total_effort,
        max_effort_since_improvement: inp.max_effort_since_improvement,
        found: false,
        log,
        logging_on,
    };
    fj.run(x);
    fj.found
}

impl Fj<'_> {
    fn log_dev(&mut self, ty: i32, msg: &str) {
        if self.logging_on {
            (self.log)(ty, msg);
        }
    }

    fn run(&mut self, out: &mut [f64]) {
        if self.logging_on {
            // weightUpdateDecay is 1, relaxContinuous 0
            let m = crate::sprintf!(
                "Feasibility Jump: starting solve. weightUpdateDecay=%g, relaxContinuous=%d  \n",
                1.0,
                0i32
            );
            self.log_dev(LOG_INFO, &m);
        }
        self.init();
        self.effort_at_last_logging = MIN_EFFORT_TO_LOGGING.wrapping_neg();
        let mut need_logging_header = true;
        let mut lines_since_header = 0;
        let mut step: i32 = 0;
        while step < i32::MAX {
            if self.user_terminate(None) {
                break;
            }
            if step % LOGGING_FREQUENCY == 0
                && self.total_effort > self.effort_at_last_logging.wrapping_add(MIN_EFFORT_TO_LOGGING)
            {
                if need_logging_header {
                    self.logging(0, true);
                    lines_since_header = 0;
                }
                self.logging(step, false);
                lines_since_header += 1;
                need_logging_header = lines_since_header == LOGGING_HEADER_FREQUENCY;
            }
            if self.violated.len() < self.best_violation_score {
                self.effort_at_last_improvement = self.total_effort;
                self.best_violation_score = self.violated.len();
            }
            if self.violated.is_empty() && self.incumbent_objective < self.best_objective {
                let rel = (self.best_objective - self.incumbent_objective)
                    / f64::max(1.0, self.incumbent_objective.abs());
                if rel > MIN_RELATIVE_OBJECTIVE_IMPROVEMENT {
                    self.effort_at_last_improvement = self.total_effort;
                    self.best_objective = self.incumbent_objective;
                    if self.user_terminate(Some(out)) {
                        break;
                    }
                    need_logging_header = true;
                }
            }
            if self.lb.is_empty() {
                break;
            }
            let var = self.select_variable();
            self.do_variable_move(var);
            step += 1;
        }
    }

    fn logging(&mut self, step: i32, header: bool) {
        if header {
            self.log_dev(
                LOG_DETAILED,
                "Feasibility Jump:        step  violations     good    bumps       effort (per step)          Objective\n",
            );
        } else {
            if self.logging_on {
                let per = if step > 0 { self.total_effort / step as usize } else { 0 };
                let m = crate::sprintf!(
                    " %10d    %8zd   %6zd %8zd %12zd    %6zd          %10.4g\n",
                    step,
                    self.violated.len(),
                    self.good.len(),
                    self.n_bumps,
                    self.total_effort,
                    per,
                    self.incumbent_objective
                );
                self.log_dev(LOG_DETAILED, &m);
            }
            self.effort_at_last_logging = self.total_effort;
        }
    }

    /// Problem::resetIncumbent, JumpMove::init and the move reset
    fn init(&mut self) {
        let mut o = 0.0;
        for i in 0..self.lb.len() {
            o = self.obj[i].mul_add_c(self.vs[i].x, o);
        }
        self.incumbent_objective = o;
        self.violated.clear();
        for c in 0..self.cons.len() {
            let mut lhs = 0.0;
            for cell in &self.ccells[self.cstart[c]..self.cstart[c + 1]] {
                lhs = cell.coeff.mul_add_c(self.vs[cell.idx as usize].x, lhs);
            }
            let con = &mut self.cons[c];
            con.lhs = lhs;
            if score(con.lte, con.rhs, lhs) < -self.viol_tol {
                con.violated_idx = self.violated.len() as i32;
                self.violated.push(c as u32);
            } else {
                con.violated_idx = -1;
            }
        }
        self.total_effort += self.ccells.len();
        self.good.clear();
        for i in 0..self.lb.len() {
            self.reset_moves(i);
        }
    }

    fn select_variable(&mut self) -> usize {
        if !self.good.is_empty() {
            if self.rng.uniform01() < RANDOM_VAR_PROBABILITY {
                return self.good[self.rng.next_u32() as usize % self.good.len()] as usize;
            }
            let sample = MAX_MOVES_TO_EVALUATE.min(self.good.len());
            self.total_effort += sample;
            let mut best_score = f64::NEG_INFINITY;
            let mut best_var = usize::MAX;
            for _ in 0..sample {
                let v = self.good[self.rng.next_u32() as usize % self.good.len()] as usize;
                let s = self.vs[v].score;
                if s > best_score {
                    best_score = s;
                    best_var = v;
                }
            }
            debug_assert!(best_var != usize::MAX);
            return best_var;
        }
        self.update_weights();
        if !self.violated.is_empty() {
            let c = self.violated[self.rng.next_u32() as usize % self.violated.len()] as usize;
            let cells = &self.ccells[self.cstart[c]..self.cstart[c + 1]];
            if self.rng.uniform01() < RANDOM_CELL_PROBABILITY {
                return cells[self.rng.next_u32() as usize % cells.len()].idx as usize;
            }
            let mut best_score = f64::NEG_INFINITY;
            let mut best_var = usize::MAX;
            for cell in cells {
                let s = self.vs[cell.idx as usize].score;
                if s > best_score {
                    best_score = s;
                    best_var = cell.idx as usize;
                }
            }
            return best_var;
        }
        self.rng.next_u32() as usize % self.lb.len()
    }

    fn update_weights(&mut self) {
        self.log_dev(LOG_VERBOSE, "Feasibility Jump: Reached a local minimum.\n");
        self.n_bumps += 1;
        let mut rescale = false;
        let mut dt = 0usize;
        let inc = self.weight_update_increment;
        if self.violated.is_empty() {
            self.objective_weight += inc;
            if self.objective_weight > 1.0e20 {
                rescale = true;
            }
            dt += self.lb.len();
            for v in 0..self.lb.len() {
                let m = &mut self.vs[v];
                m.score = (inc * self.obj[v]).mul_add_c(m.value - m.x, m.score);
            }
        } else {
            for k in 0..self.violated.len() {
                let c = self.violated[k] as usize;
                self.cons[c].weight += inc;
                let con = self.cons[c];
                if con.weight > 1.0e20 {
                    rescale = true;
                }
                let (s, e) = (self.cstart[c], self.cstart[c + 1]);
                dt += e - s;
                let score_lhs = score(con.lte, con.rhs, con.lhs);
                for p in s..e {
                    let cell = self.ccells[p];
                    let v = cell.idx as usize;
                    let m = &mut self.vs[v];
                    let cand = cell.coeff.mul_add_c(m.value - m.x, con.lhs);
                    let diff = inc * (score(con.lte, con.rhs, cand) - score_lhs);
                    m.score += diff;
                    self.update_good_moves(v);
                }
            }
        }
        // weightUpdateDecay is 1
        self.weight_update_increment /= 1.0;
        if rescale {
            self.weight_update_increment *= 1.0e-20;
            self.objective_weight *= 1.0e-20;
            for c in &mut self.cons {
                c.weight *= 1.0e-20;
            }
            dt += self.cons.len();
            for i in 0..self.lb.len() {
                self.reset_moves(i);
            }
        }
        self.total_effort += dt;
    }

    /// bestMove(v).value: NaN if the score is not above -inf
    #[inline]
    fn best_value(&self, v: usize) -> f64 {
        let m = self.vs[v];
        if m.score > f64::NEG_INFINITY {
            m.value
        } else {
            f64::NAN
        }
    }

    fn do_variable_move(&mut self, v: usize) {
        let new_value = self.best_value(v);
        // Problem::setValue with modifyMove and updateGoodMoves
        let old_value = self.vs[v].x;
        let delta = new_value - old_value;
        self.vs[v].x = new_value;
        self.incumbent_objective = self.obj[v].mul_add_c(delta, self.incumbent_objective);
        let mut dt = 0usize;
        for q in self.vstart[v]..self.vstart[v + 1] {
            let vc = self.vcells[q];
            let ci = vc.idx as usize;
            let con = self.cons[ci];
            let old_lhs = con.lhs;
            let new_lhs = vc.coeff.mul_add_c(delta, old_lhs);
            self.cons[ci].lhs = new_lhs;
            let new_cost = score(con.lte, con.rhs, new_lhs);
            if new_cost < -self.viol_tol && con.violated_idx == -1 {
                self.cons[ci].violated_idx = self.violated.len() as i32;
                self.violated.push(ci as u32);
            }
            if new_cost >= -self.viol_tol && self.cons[ci].violated_idx != -1 {
                let last = self.violated.len() - 1;
                let last_con = self.violated[last] as usize;
                let this = self.cons[ci].violated_idx as usize;
                self.violated.swap(this, last);
                self.cons[last_con].violated_idx = this as i32;
                self.cons[ci].violated_idx = -1;
                self.violated.pop();
            }
            let (s, e) = (self.cstart[ci], self.cstart[ci + 1]);
            dt += e - s;
            let (lte, rhs, w) = (con.lte, con.rhs, con.weight);
            let old_score = score(lte, rhs, old_lhs);
            let new_score = score(lte, rhs, new_lhs);
            for p in s..e {
                let cell = self.ccells[p];
                let u = cell.idx as usize;
                if u == v {
                    continue;
                }
                // modifyMove
                let m = &mut self.vs[u];
                let t = m.value - m.x;
                let old_mod = cell.coeff.mul_add_c(t, old_lhs);
                let old_term = w * (score(lte, rhs, old_mod) - old_score);
                let new_mod = cell.coeff.mul_add_c(t, new_lhs);
                let new_term = w * (score(lte, rhs, new_mod) - new_score);
                m.score += new_term - old_term;
                self.update_good_moves(u);
            }
        }
        self.total_effort += dt;
        self.reset_moves(v);
    }

    #[inline]
    fn update_good_moves(&mut self, v: usize) {
        let any_good = self.vs[v].score > 0.0;
        let gi = self.vs[v].good_idx;
        if any_good && gi == -1 {
            self.vs[v].good_idx = self.good.len() as i32;
            self.good.push(v as u32);
        } else if !any_good && gi != -1 {
            let last = self.good.len() - 1;
            let last_var = self.good[last] as usize;
            let this = gi as usize;
            self.good.swap(this, last);
            self.vs[last_var].good_idx = this as i32;
            self.vs[v].good_idx = -1;
            self.good.pop();
        }
    }

    fn reset_moves(&mut self, v: usize) {
        let (s, e) = (self.vstart[v], self.vstart[v + 1]);
        self.total_effort += e - s;
        let value = self.jump_value(v);
        let xv = self.vs[v].x;
        let t = value - xv;
        let mut sc = (self.objective_weight * self.obj[v]).mul_add_c(t, 0.0);
        for q in s..e {
            let cell = self.vcells[q];
            let con = &self.cons[cell.idx as usize];
            let cand = cell.coeff.mul_add_c(t, con.lhs);
            sc = con
                .weight
                .mul_add_c(score(con.lte, con.rhs, cand) - score(con.lte, con.rhs, con.lhs), sc);
        }
        self.vs[v].value = value;
        self.vs[v].score = sc;
        self.update_good_moves(v);
    }

    /// JumpMove::updateValue
    fn jump_value(&mut self, v: usize) -> f64 {
        let shift = &mut self.shift;
        shift.clear();
        let inc = self.vs[v].x;
        let lower = self.lb[v];
        let upper = self.ub[v];
        let integer = self.integer[v];
        let tol = self.eq_tol;
        let current = lower;
        let mut slope = 0.0;
        for q in self.vstart[v]..self.vstart[v + 1] {
            let cell = self.vcells[q];
            let con = &self.cons[cell.idx as usize];
            let (lo, hi) = if con.lte {
                (f64::NEG_INFINITY, con.rhs)
            } else {
                (con.rhs, f64::INFINITY)
            };
            let residual = (-cell.coeff).mul_add_c(inc, con.lhs);
            let mut first = (1.0 / cell.coeff) * (lo - residual);
            let mut second = (1.0 / cell.coeff) * (hi - residual);
            if integer {
                first = (first - tol).ceil();
                second = (second + tol).floor();
            }
            if first > second {
                continue;
            }
            if first > current {
                slope -= con.weight;
                if first < upper {
                    shift.push((first, con.weight));
                }
            }
            if second <= current {
                slope += con.weight;
            } else if second < upper {
                shift.push((second, con.weight));
            }
        }
        if lower.is_finite() {
            shift.push((lower, 0.0));
        }
        if upper.is_finite() {
            shift.push((upper, 0.0));
        }
        // std::pair's operator<; no NaN arises
        shift.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut score_now = 0.0;
        let mut current = if shift.is_empty() { inc } else { shift[0].0 };
        let mut best_score = score_now;
        let mut best_value = current;
        for &(value, weight) in shift.iter() {
            score_now = (value - current).mul_add_c(slope, score_now);
            slope += weight;
            current = value;
            if eq(best_value, inc, tol) || (!eq(current, inc, tol) && score_now < best_score) {
                best_score = score_now;
                best_value = current;
            }
            if !eq(best_value, inc, tol) && slope >= 0.0 {
                break;
            }
        }
        best_value
    }

    /// user_terminate with the wrapper's callback: true to stop
    fn user_terminate(&mut self, solution: Option<&mut [f64]>) -> bool {
        if solution.is_none() && self.total_effort - self.effort_at_last_callback <= CALLBACK_EFFORT {
            return false;
        }
        self.log_dev(LOG_VERBOSE, "Feasibility Jump: calling user termination.\n");
        self.effort_at_last_callback = self.total_effort;
        let since = self.total_effort - self.effort_at_last_improvement;
        if let Some(out) = solution {
            self.found = true;
            for (o, v) in out.iter_mut().zip(&self.vs) {
                *o = v.x;
            }
        }
        if since > self.max_effort_since_improvement || self.total_effort > self.max_total_effort {
            self.log_dev(LOG_VERBOSE, "Feasibility Jump: quitting.\n");
            return true;
        }
        false
    }
}

/// The problem of HighsFeasibilityJump.cpp (HIGHS_RUST); layout
/// FjRsProblem there
#[repr(C)]
pub struct FjRsProblem {
    pub num_col: i32,
    pub num_row: i32,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    pub col_cost: *const f64,
    pub col_integer: *const u8,
    pub ar_start: *const i32,
    pub ar_index: *const i32,
    pub ar_value: *const f64,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    pub seed: u32,
    pub equality_tolerance: f64,
    pub violation_tolerance: f64,
    pub max_total_effort: u64,
    pub max_effort_since_improvement: u64,
}

/// Runs [`solve`]; `x` (num_col) holds the initial values and receives the
/// solution. Returns 1 if a solution was found. `log(ctx, type, message)`
/// is called only if `logging_on`.
///
/// # Safety
/// The pointers of `p` hold num_col, num_col + 1 (ar_start), ar_start[num_row]
/// and num_row values as named; `x` holds num_col.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_feasibility_jump(
    p: *const FjRsProblem,
    x: *mut f64,
    logging_on: i32,
    log_ctx: *mut std::ffi::c_void,
    log: extern "C" fn(*mut std::ffi::c_void, i32, *const std::ffi::c_char),
) -> i32 {
    use std::slice::from_raw_parts as sl;
    let p = &*p;
    let (n, m) = (p.num_col as usize, p.num_row as usize);
    let nnz = *p.ar_start.add(m) as usize;
    let inp = Input {
        col_lower: sl(p.col_lower, n),
        col_upper: sl(p.col_upper, n),
        col_cost: sl(p.col_cost, n),
        col_integer: sl(p.col_integer, n),
        ar_start: sl(p.ar_start, m + 1),
        ar_index: sl(p.ar_index, nnz),
        ar_value: sl(p.ar_value, nnz),
        row_lower: sl(p.row_lower, m),
        row_upper: sl(p.row_upper, m),
        seed: p.seed,
        equality_tolerance: p.equality_tolerance,
        violation_tolerance: p.violation_tolerance,
        max_total_effort: p.max_total_effort as usize,
        max_effort_since_improvement: p.max_effort_since_improvement as usize,
    };
    let x = std::slice::from_raw_parts_mut(x, n);
    let mut cb = |ty: i32, msg: &str| {
        let c = std::ffi::CString::new(msg).unwrap();
        log(log_ctx, ty, c.as_ptr());
    };
    solve(&inp, x, logging_on != 0, &mut cb) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_std() {
        // The 10000th output of a default-constructed std::mt19937
        let mut r = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = r.next_u32();
        }
        assert_eq!(v, 4123659995);
    }

    #[test]
    fn knapsack() {
        // max x0 + 2 x1 + 3 x2 (min of the negation), x0 + x1 + x2 <= 2, binaries
        let inp = Input {
            col_lower: &[0.0; 3],
            col_upper: &[1.0; 3],
            col_cost: &[-1.0, -2.0, -3.0],
            col_integer: &[1; 3],
            ar_start: &[0, 3],
            ar_index: &[0, 1, 2],
            ar_value: &[1.0; 3],
            row_lower: &[f64::NEG_INFINITY],
            row_upper: &[2.0],
            seed: 0,
            equality_tolerance: 1e-9,
            violation_tolerance: 1e-6,
            max_total_effort: 3 << 10,
            max_effort_since_improvement: 3 << 8,
        };
        let mut x = [0.0; 3];
        assert!(solve(&inp, &mut x, false, &mut |_, _| {}));
        assert!(x.iter().sum::<f64>() <= 2.0);
        assert!(x.iter().all(|&v| v == 0.0 || v == 1.0));
    }
}
