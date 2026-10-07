//! HighsPrimalHeuristics::graphLNS (highs/mip/HighsGraphLns.cpp, whose
//! comment describes the method): the root dive, the neighbourhood loop
//! with its dives and depth-first branch and bound, and the flip search
//! with its propagation screen, on a copy of the worker's LP relaxation
//! and C++ domains (glue.rs). The neighbourhoods, flip candidates and move
//! choice are lns.rs.

use super::cuts::sort::pdqsort;
use super::domain::{DomChg as Chg, Reason, LOWER as LO, REASON_BRANCHING, UPPER as UP};
use super::glue::{self, lp_status, source, Bnd, Dom, Lp, MipData, Worker};
use super::lns::{decision_cols, select_type, Graph, Lns};
use super::primal::{cmax, cmin, Heur};
use crate::lp_data::LogType;
use crate::util::fma::ClangFma;

const INF: f64 = f64::INFINITY;
const BRANCHING: Reason = Reason { kind: REASON_BRANCHING, index: 0 };
/// The LP solves of a neighbourhood's branch and bound and of a deep flip
/// search
const NODE_LIMIT: i32 = 300;

fn usable(st: i32) -> bool {
    st == lp_status::OPTIMAL || st == lp_status::UNSCALED_PRIMAL_FEASIBLE
}

/// A domain's bounds as slices (`n` columns), for reads with no change of
/// the domain in between
///
/// # Safety
/// The domain is live and not changed while the slices are used.
unsafe fn bound_slices<'a>(b: &Bnd, n: usize) -> (&'a [f64], &'a [f64]) {
    (std::slice::from_raw_parts(b.lo, n), std::slice::from_raw_parts(b.up, n))
}

/// The rounding of sol[col] within the bounds
fn rounded(b: &Bnd, sol: &[f64], col: usize) -> f64 {
    cmin(cmax(sol[col].round(), b.lo(col)), b.up(col))
}

fn fix_to(dom: &Dom, col: i32, val: f64) -> bool {
    let b = dom.bnd();
    if b.lo(col as usize) < val {
        dom.change_bound(Chg { boundval: val, column: col, boundtype: LO }, Reason::UNSPECIFIED);
    }
    if b.up(col as usize) > val {
        dom.change_bound(Chg { boundval: val, column: col, boundtype: UP }, Reason::UNSPECIFIED);
    }
    dom.propagate();
    !dom.infeasible()
}

/// fix a column as a branching decision and propagate; if that is
/// infeasible, undo it (backtracking to the decision) and return false
fn fix_try(dom: &Dom, col: i32, val: f64) -> bool {
    let mut branched = false;
    let b = dom.bnd();
    if b.lo(col as usize) < val {
        dom.change_bound(Chg { boundval: val, column: col, boundtype: LO }, BRANCHING);
        branched = true;
    }
    if b.up(col as usize) > val {
        let reason = if branched { Reason::UNSPECIFIED } else { BRANCHING };
        dom.change_bound(Chg { boundval: val, column: col, boundtype: UP }, reason);
        branched = true;
    }
    if !branched {
        return !dom.infeasible();
    }
    dom.propagate();
    if !dom.infeasible() {
        return true;
    }
    dom.backtrack();
    false
}

/// One graphLNS call
struct GraphLns<'a> {
    h: &'a mut Heur,
    m: &'a MipData,
    w: &'a Worker,
    g: Graph<'a>,
    lns: Lns,
    lp: Lp,
    deep: bool,
    lp_iters_start: i64,
    n: usize,
    feastol: f64,
    in_cands: Vec<u8>,
    // the flip search: the incumbent's decision columns, the screening
    // domain and its open set
    cur: Vec<f64>,
    screen: Dom,
    is_open: Vec<u8>,
    open_cols: Vec<i32>,
    have_base: bool,
    screened: i32,
    flip_max_iters: i64,
}

impl GraphLns<'_> {
    fn charge_iterations(&self) {
        self.w.heur().lp_iterations += self.lp.num_lp_iterations() - self.lp_iters_start;
    }

    fn solve(&self, dom: &Dom) -> i32 {
        self.lp.set_objective_limit(self.w.upper_limit());
        self.lp.resolve(Some(dom))
    }

    /// an LP solution with every integral column integral is a new
    /// incumbent
    fn try_incumbent(&self, st: i32) -> bool {
        if !usable(st) || !self.lp.frac().is_empty() {
            return false;
        }
        let sol = self.lp.col_value().to_vec();
        glue::add_incumbent(self.m, self.w, &sol, self.lp.objective(), source::GRAPH_LNS)
    }

    /// put `dom` back to `snap` and push the undone bounds back into the LP
    fn restore(&self, dom: &Dom, snap: &Dom, stack_pos: usize) {
        let cols: Vec<i32> = dom.stack()[stack_pos..].iter().map(|c| c.column).collect();
        dom.assign(snap);
        let b = dom.bnd();
        for c in cols {
            self.lp.change_col_bounds(c, b.lo(c as usize), b.up(c as usize));
        }
    }

    /// Dive: fix the most integral unfixed candidates toward the current
    /// LP point, `chunk` per LP re-solve, up to the first one whose rounding
    /// propagation rules out (the chunk then shrinks to what was fixed). A
    /// chunk whose LP is infeasible is undone, with the simplex iterate of
    /// the LP before it, and retried at a quarter of the size; a single
    /// failed fixing is flipped the other way.
    fn dive(&self, dom: &Dom, mut candidates: Vec<i32>, chunk0: i32) -> bool {
        let mut chunk = 1.max(chunk0);
        let max_solves = 5 * candidates.len() as i32 + 100;
        let mut solves = 0;
        let mut fallback = false;
        let mut order: Vec<(f64, i32)> = Vec::new();
        // the last usable LP solution
        let mut sol = self.lp.col_value().to_vec();
        while solves < max_solves {
            if self.w.terminated() || self.m.check_limits() {
                return false;
            }
            order.clear();
            let b = dom.bnd();
            for &col in &candidates {
                let c = col as usize;
                if b.lo(c) < b.up(c) {
                    order.push(((sol[c] - sol[c].round()).abs(), col));
                }
            }
            if order.is_empty() {
                if self.lp.frac().is_empty() || fallback {
                    return false;
                }
                // non-decision integers left fractional: dive on them too
                fallback = true;
                candidates = self.lp.frac().iter().map(|f| f.col).collect();
                continue;
            }
            pdqsort(&mut order, |a: &(f64, i32), b: &(f64, i32)| a.0 < b.0 || (!(b.0 < a.0) && a.1 < b.1));
            let mut nfix = chunk.min(order.len() as i32);
            let snap = Dom::copy(dom);
            let pos = dom.stack_len();
            // fix the chunk up to the first rounding that conflicts with
            // the fixings before it (which is undone)
            let mut nfixed = 0;
            for &(_, col) in &order[..nfix as usize] {
                let val = rounded(&dom.bnd(), &sol, col as usize);
                if !fix_try(dom, col, val) {
                    break;
                }
                nfixed += 1;
            }
            let feasible = nfixed > 0;
            if nfixed < nfix {
                // (the first one conflicting: try its other value, as below)
                nfix = 1.max(nfixed);
                chunk = nfix;
            }
            let mut st = lp_status::INFEASIBLE;
            let mut saved = false;
            if feasible {
                saved = self.lp.put_iterate();
                st = self.solve(dom);
                solves += 1;
            }
            if usable(st) {
                if self.try_incumbent(st) {
                    return true;
                }
                sol = self.lp.col_value().to_vec();
                chunk = chunk0.min(2 * chunk);
                continue;
            }
            self.restore(dom, &snap, pos);
            if saved {
                self.lp.get_iterate();
            }
            if nfix > 1 {
                chunk = 1.max(chunk / 4);
                continue;
            }
            // a single fixing failed: try the other rounding direction
            let col = order[0].1;
            let c = col as usize;
            let b = dom.bnd();
            let val = rounded(&b, &sol, c);
            let other = if sol[c] > val { val + 1.0 } else { val - 1.0 };
            let other = cmin(cmax(other, b.lo(c)), b.up(c));
            if other == val || !fix_to(dom, col, other) {
                return false;
            }
            st = self.solve(dom);
            solves += 1;
            if !usable(st) {
                return false;
            }
            if self.try_incumbent(st) {
                return true;
            }
            sol = self.lp.col_value().to_vec();
        }
        false
    }

    /// the solver's own test of the target gap (as in evaluateRootLp)
    fn within_gap(&self) -> bool {
        self.m.upper_bound() < INF && self.m.lower_bound() > self.m.optimality_limit()
    }

    /// Progress (which resets the stall count): in the quick search, an
    /// improvement closing at least 5% of the gap; in the deep search, one
    /// closing at least 1% of what separates the incumbent from the target
    /// gap
    fn progress_since(&self, before: f64, limit_before: f64) -> bool {
        let m = self.m;
        if !self.deep {
            return m.upper_bound() < before - cmax(self.feastol, 0.05 * (before - m.lower_bound()));
        }
        m.optimality_limit() < limit_before - cmax(self.feastol, 0.01 * (limit_before - m.lower_bound()))
    }

    fn lp_budget_exceeded(&self, cap: i64) -> bool {
        let lns_iters = self.lp.num_lp_iterations() - self.lp_iters_start;
        if self.deep {
            return lns_iters > cap;
        }
        let m = self.m;
        let heur_iters = self.w.heur().lp_iterations + lns_iters;
        heur_iters + m.heuristic_lp_iterations()
            > 100000 + ((m.total_lp_iterations() - m.heuristic_lp_iterations() - m.sb_lp_iterations()) >> 1)
            || lns_iters > cap
    }

    /// Depth-first branch and bound over the neighbourhood from the solved
    /// LP at its root, using at most `node_limit` LP solves. Branches on the
    /// fractional candidate closest to integrality, first toward its
    /// rounded value; backtracking flips the deepest open decision first.
    /// Returns the number of LP solves, and whether the neighbourhood was
    /// exhausted.
    fn search_neighbourhood(&mut self, dom: &Dom, cands: &[i32], node_limit: i32, exhausted: &mut bool) -> i32 {
        for &c in cands {
            self.in_cands[c as usize] = 1;
        }
        // the other branch of each decision, and whether it was taken
        let mut path: Vec<(Chg, bool)> = Vec::new();
        let mut nodes = 0;
        *exhausted = false;
        let mut st = lp_status::OPTIMAL;
        let mut solved = true; // the LP is solved on entry
        loop {
            if self.w.terminated() || self.m.check_limits() {
                break;
            }
            if !solved {
                st = self.solve(dom);
                nodes += 1;
            }
            solved = false;
            let prune = !usable(st) || self.lp.objective() >= self.w.upper_limit();
            if !prune {
                if self.try_incumbent(st) {
                    if self.within_gap() {
                        break;
                    }
                } else {
                    let sol = self.lp.col_value();
                    let b = dom.bnd();
                    let mut best_col = -1;
                    let mut best_frac = INF;
                    for f in self.lp.frac().iter() {
                        let col = f.col as usize;
                        if b.lo(col) == b.up(col) {
                            continue;
                        }
                        let mut frac = (sol[col] - sol[col].round()).abs();
                        // prefer neighbourhood decision columns over the rest
                        if self.in_cands[col] == 0 {
                            frac += 1.0;
                        }
                        if frac < best_frac {
                            best_frac = frac;
                            best_col = f.col;
                        }
                    }
                    if best_col == -1 || nodes >= node_limit {
                        break;
                    }
                    let x = sol[best_col as usize];
                    let up = Chg { boundval: x.ceil(), column: best_col, boundtype: LO };
                    let down = Chg { boundval: x.floor(), column: best_col, boundtype: UP };
                    let go_up = x - x.floor() >= 0.5;
                    dom.change_bound(if go_up { up } else { down }, BRANCHING);
                    path.push((if go_up { down } else { up }, false));
                    dom.propagate();
                    if !dom.infeasible() {
                        continue;
                    }
                }
            }
            // backtrack to the deepest decision whose other branch is open
            let mut open = false;
            while let Some(last) = path.last_mut() {
                dom.backtrack();
                if last.1 {
                    path.pop();
                    continue;
                }
                last.1 = true;
                let other = last.0;
                dom.change_bound(other, BRANCHING);
                dom.propagate();
                if dom.infeasible() {
                    continue;
                }
                open = true;
                break;
            }
            if !open {
                *exhausted = true;
                break;
            }
            if nodes >= node_limit {
                break;
            }
        }
        while path.pop().is_some() {
            dom.backtrack();
        }
        for &c in cands {
            self.in_cands[c as usize] = 0;
        }
        nodes
    }

    fn flips_exhausted(&self) -> bool {
        self.m.upper_bound() == self.h.lns_flip_obj && self.h.lns_flip_next == -1
    }

    fn flipped(&self, col: usize, gb: &Bnd) -> f64 {
        if self.cur[col] <= gb.lo(col) {
            gb.up(col)
        } else {
            gb.lo(col)
        }
    }

    fn set_col(&self, col: i32, val: f64) {
        self.lp.change_col_bounds(col, val, val);
    }

    /// fix a column in the screening domain, the first fixing of a set as
    /// a branching decision
    fn fix(&self, col: i32, val: f64, branched: &mut bool) {
        let b = self.screen.bnd();
        if b.lo(col as usize) < val {
            let reason = if *branched { Reason::UNSPECIFIED } else { BRANCHING };
            self.screen.change_bound(Chg { boundval: val, column: col, boundtype: LO }, reason);
            *branched = true;
        }
        if b.up(col as usize) > val {
            let reason = if *branched { Reason::UNSPECIFIED } else { BRANCHING };
            self.screen.change_bound(Chg { boundval: val, column: col, boundtype: UP }, reason);
            *branched = true;
        }
    }

    fn drop_base(&mut self) {
        if self.have_base {
            self.screen.backtrack();
        }
        self.have_base = false;
        for &c in &self.open_cols {
            self.is_open[c as usize] = 0;
        }
        self.open_cols.clear();
    }

    /// the base: every decision column fixed at the incumbent except the
    /// open set
    fn build_base(&mut self, open: &[i32]) {
        self.drop_base();
        for &c in open {
            if self.is_open[c as usize] == 0 {
                self.is_open[c as usize] = 1;
                self.open_cols.push(c);
            }
        }
        let mut branched = false;
        for i in 0..self.h.decisioncols.len() {
            if self.screen.infeasible() {
                break;
            }
            let col = self.h.decisioncols[i];
            if self.is_open[col as usize] == 0 {
                self.fix(col, self.cur[col as usize], &mut branched);
            }
        }
        if !self.screen.infeasible() {
            self.screen.propagate();
        }
        self.have_base = branched;
        // (the incumbent satisfies the base, so this is not expected)
        if self.screen.infeasible() {
            self.drop_base();
        }
    }

    /// whether propagating the move (its columns flipped, the other decision
    /// columns at the incumbent) is infeasible
    fn propagation_infeasible(&mut self, cols: &[i32], gb: &Bnd) -> bool {
        let n = cols.len();
        let in_base =
            self.have_base && self.is_open[cols[0] as usize] != 0 && (n == 1 || self.is_open[cols[1] as usize] != 0);
        if !in_base {
            self.drop_base();
        }
        let in_move = |col: i32| col == cols[0] || (n > 1 && col == cols[1]);
        let mut branched = false;
        for &c in cols {
            self.fix(c, self.flipped(c as usize, gb), &mut branched);
        }
        let list = if in_base { self.open_cols.clone() } else { self.h.decisioncols.clone() };
        for col in list {
            if self.screen.infeasible() {
                break;
            }
            if !in_move(col) {
                self.fix(col, self.cur[col as usize], &mut branched);
            }
        }
        if !self.screen.infeasible() {
            self.screen.propagate();
        }
        let infeasible = self.screen.infeasible();
        if branched {
            self.screen.backtrack();
        }
        infeasible
    }

    /// try a move: on improvement keep it, otherwise undo it
    fn try_move(&mut self, cols: &[i32], solves: &mut i32, gb: &Bnd) -> bool {
        // (moves ruled out by propagation, which cost far less than an LP
        // solve, count as one in four)
        if self.propagation_infeasible(cols, gb) {
            self.screened += 1;
            if self.screened % 4 == 0 {
                *solves += 1;
            }
            return false;
        }
        for &c in cols {
            self.set_col(c, self.flipped(c as usize, gb));
        }
        self.lp.set_objective_limit(self.w.upper_limit());
        let mst = self.lp.resolve(None);
        *solves += 1;
        if usable(mst) && self.lp.objective() < self.w.upper_limit() && self.try_incumbent(mst) {
            self.drop_base();
            for &c in cols {
                self.cur[c as usize] = self.flipped(c as usize, gb);
            }
            return true;
        }
        for &c in cols {
            self.set_col(c, self.cur[c as usize]);
        }
        false
    }

    /// Flip search: first-improvement local search from the incumbent with
    /// all decision columns fixed at it; returns the number of LP solves
    fn flip_search(&mut self, max_solves: i32) -> i32 {
        const PARTNERS: usize = 3;
        const SCREEN_BATCH: usize = 32;
        let m = self.m;
        if m.incumbent().len() != self.n || self.flips_exhausted() {
            return 0;
        }
        // go on from where the last search from this incumbent stopped
        let mut start = if m.upper_bound() == self.h.lns_flip_obj { 0.max(self.h.lns_flip_next) as usize } else { 0 };
        let flip_start_iters = self.lp.num_lp_iterations();
        let gd = self.w.globaldom();
        let gb = gd.bnd();
        {
            let inc = m.incumbent();
            for &c in &self.h.decisioncols {
                let col = c as usize;
                self.cur[col] = cmin(cmax(inc[col].round(), gb.lo(col)), gb.up(col));
            }
        }
        // the incumbent's decision columns are fixed in the LP, others get
        // their global bounds
        self.lp.change_cols_bounds_dom(&gd);
        for i in 0..self.h.decisioncols.len() {
            let c = self.h.decisioncols[i];
            self.set_col(c, self.cur[c as usize]);
        }
        self.lp.set_objective_limit(INF);
        let mut solves = 1;
        if !usable(self.lp.resolve(None)) {
            return solves;
        }
        self.screen.assign(&gd);
        self.screen.clear_pool_propagation();
        self.open_cols.clear();
        self.have_base = false;
        self.screened = 0;
        // SAFETY: the global domain does not change during the search
        let (glo, gup) = unsafe { bound_slices(&gb, self.n) };
        let mut improved = true;
        while improved {
            improved = false;
            // the LP is solved at the incumbent: gains from its reduced costs
            let rc = self.lp.col_dual().to_vec();
            self.lns.flip_candidates(&self.cur, glo, gup, &rc, self.feastol);
            let flip_cands = self.lns.flip_cands.clone();
            let mut batch_end = start;
            let mut open: Vec<i32> = Vec::new();
            for c in start..flip_cands.len() {
                if solves >= max_solves
                    || self.lp.num_lp_iterations() - flip_start_iters > self.flip_max_iters
                    || self.within_gap()
                    || self.w.terminated()
                    || m.check_limits()
                {
                    self.h.lns_flip_obj = m.upper_bound();
                    self.h.lns_flip_next = c as i32;
                    self.drop_base();
                    return solves;
                }
                if c >= batch_end {
                    // the next candidates and their partners are open in the
                    // base
                    batch_end = flip_cands.len().min(c + SCREEN_BATCH);
                    open.clear();
                    for fb in &flip_cands[c..batch_end] {
                        open.push(fb.1);
                        self.lns.find_partners(&self.g, fb.1 as usize, &self.cur, glo, gup, &rc);
                        open.extend(self.lns.partners.iter().take(PARTNERS).map(|p| p.1));
                    }
                    self.build_base(&open);
                }
                let j = flip_cands[c].1;
                if self.try_move(&[j], &mut solves, &gb) {
                    improved = true;
                    break;
                }
                self.lns.find_partners(&self.g, j as usize, &self.cur, glo, gup, &rc);
                let partners: Vec<i32> = self.lns.partners.iter().take(PARTNERS).map(|p| p.1).collect();
                for &p in &partners {
                    if improved {
                        break;
                    }
                    if solves >= max_solves {
                        self.h.lns_flip_obj = m.upper_bound();
                        self.h.lns_flip_next = c as i32;
                        self.drop_base();
                        return solves;
                    }
                    improved = self.try_move(&[j, p], &mut solves, &gb);
                }
                if improved {
                    break;
                }
            }
            start = 0;
        }
        // no improving flip from this incumbent
        self.drop_base();
        self.h.lns_flip_obj = m.upper_bound();
        self.h.lns_flip_next = -1;
        solves
    }
}

/// The model's graph from the MIP data
fn mip_graph(m: &MipData) -> Graph<'_> {
    Graph {
        a_start: m.a_start(),
        a_index: m.a_index(),
        a_value: m.a_value(),
        ar_start: m.ar_start(),
        ar_index: m.ar_index(),
        ar_value: m.ar_value(),
    }
}

impl Heur {
    /// setupDecisionCols
    fn setup_decision_cols(&mut self, m: &MipData) {
        self.decision_cols_set_up = true;
        self.decisioncols.clear();
        if !m.colwise || m.ar_start().is_empty() {
            self.decisioncols = self.intcols().to_vec();
            return;
        }
        self.decisioncols = decision_cols(&mip_graph(m), m.row_lower(), m.row_upper(), m.integrality(), self.intcols());
    }

    /// graphLNS: root dive and LNS from the incumbent: quick dived
    /// neighbourhoods (after the first root LP), or neighbourhoods searched
    /// by a depth-first branch and bound (deep, after the root cuts), using
    /// at most max_lp_iters LP iterations if that is not negative
    pub fn graph_lns(&mut self, m: &MipData, w: &Worker, relaxationsol: &[f64], deep: bool, max_lp_iters: i64) {
        if m.submip && !m.concurrent_helper {
            return;
        }
        if w.globaldom().infeasible() {
            return;
        }
        if !self.decision_cols_set_up {
            self.setup_decision_cols(m);
        }
        if self.decisioncols.is_empty() || relaxationsol.is_empty() {
            return;
        }
        let n = m.num_col as usize;
        let g = mip_graph(m);
        let lns = Lns::new(&g, m.num_row as usize, &self.decisioncols);
        // one LP relaxation copy for the whole heuristic; bounds live in a
        // domain
        let lp = Lp::copy(w.d.lp, w);
        lp.set_adjust_symmetric_branching_col(false);
        let lp_iters_start = lp.num_lp_iterations();
        let gd = w.globaldom();
        let decisioncols = self.decisioncols.clone();
        let feastol = m.feastol();
        // 1. root dive from the LP point
        let dom = Dom::copy(&gd);
        lp.change_cols_bounds_dom(&dom);
        dom.clear_changed_cols();
        // (the flip search's screening domain, a copy of the global one,
        // was made before the first solve in the C++, after the dive)
        let mut s = GraphLns {
            h: self,
            m,
            w,
            g,
            lns,
            lp,
            deep,
            lp_iters_start,
            n,
            feastol,
            in_cands: vec![0; n],
            cur: vec![0.0; n],
            screen: Dom::borrowed(std::ptr::null_mut()),
            is_open: vec![0; n],
            open_cols: Vec::new(),
            have_base: false,
            screened: 0,
            flip_max_iters: 5000i64.max(m.firstrootlpiters()),
        };
        let who = if m.concurrent_helper { "LNS(helper)" } else { "LNS" };
        let log = &m.log;
        let st = s.solve(&dom);
        if !usable(st) {
            s.charge_iterations();
            return;
        }
        // the deep search only needs a dive without an incumbent, if diving
        // can find one
        if !s.try_incumbent(st) && (!deep || (m.incumbent().len() != n && !s.h.lns_dive_failed)) {
            let dive_iters = s.lp.num_lp_iterations();
            let found = s.dive(&dom, decisioncols.clone(), 20.max(decisioncols.len() as i32 / 12));
            if !found {
                s.h.lns_dive_failed = true;
            }
            crate::log_dev!(
                log,
                LogType::Verbose,
                "%s dive %s after %lld LP iterations\n",
                who,
                if found { "found a solution" } else { "failed" },
                s.lp.num_lp_iterations() - dive_iters
            );
        }
        if m.upper_limit() == INF || m.incumbent().len() != n {
            s.charge_iterations();
            return;
        }

        // 2. neighbourhoods. The quick search dives each neighbourhood
        // once, within the heuristic LP budget; the deep search exhausts
        // smaller ones by a depth-first branch and bound with a node limit
        let max_stall = if deep { 10 } else { 5 };
        let max_it = if deep { 1000 } else { 100 };
        let nd = decisioncols.len() as f64;
        let dive_size0 = cmin(400.0, nd);
        let dfs_size0 = cmin(64.0, nd);
        let dfs_min_size = cmin(16.0, dfs_size0);
        let dfs_max_size = cmax(dfs_size0, cmin(1000.0, nd));
        let iters_fac = if deep { 10.0 } else { 3.0 };
        let mut since = 0;
        let mut heur_iters_cap = (iters_fac * m.total_lp_iterations() as f64) as i64 + if deep { 5000 } else { 1000 };
        // on large models, the quick search spends about one root LP's
        // worth of LP iterations
        if !deep {
            heur_iters_cap = heur_iters_cap.min((m.total_lp_iterations() + 1000).max(20000));
        }
        if max_lp_iters >= 0 {
            heur_iters_cap = heur_iters_cap.min(max_lp_iters);
        }
        for (t, mv) in s.h.lns_moves.iter_mut().enumerate() {
            if mv.size == 0.0 {
                mv.size = if t == 3 { dive_size0 } else { dfs_size0 };
            }
        }
        let mut deep_tries = [0i32; 4];
        s.screen = Dom::copy(&gd);
        let mut it = 0;
        while since < max_stall && it < max_it {
            m.sync_concurrent_lns();
            // a helper crosses its incumbent with the main solver's once
            // that is ready (see HighsMipSolverData::crossoverWithMain)
            if m.concurrent_helper {
                m.crossover_with_main(w);
            }
            if w.terminated() || m.check_limits() || s.lp_budget_exceeded(heur_iters_cap) || s.within_gap() {
                break;
            }
            // the quick search only dives; the deep search tries each type
            // once, then mostly the one closing the most gap per LP
            // iteration recently
            let mut ty = 3;
            if deep {
                let fe = s.flips_exhausted();
                ty = select_type(&s.h.lns_moves, &mut deep_tries, fe);
            }
            let dived = ty == 3;
            if ty == 2 {
                let before = m.upper_bound();
                let limit_before = m.optimality_limit();
                let gap_before = before - m.lower_bound();
                let start_iters = s.lp.num_lp_iterations();
                let solves = s.flip_search(NODE_LIMIT);
                let effort = 1.0 + (s.lp.num_lp_iterations() - start_iters) as f64;
                let closed = if gap_before > 0.0 { (before - m.upper_bound()) / gap_before } else { 0.0 };
                let nt = &mut s.h.lns_moves[2];
                nt.tried += 1;
                if m.upper_bound() < before - feastol {
                    nt.improved += 1;
                }
                nt.rate = 0.7f64.mul_add_c(nt.rate, 0.3 * closed / effort);
                crate::log_dev!(
                    log,
                    LogType::Verbose,
                    "%s %3d flips: %3d LP solves, objective %.4f, %lld LP iterations\n",
                    who,
                    it,
                    solves,
                    m.upper_bound(),
                    s.lp.num_lp_iterations() - s.lp_iters_start
                );
                if s.progress_since(before, limit_before) {
                    since = 0;
                } else {
                    since += 1;
                }
                it += 1;
                continue;
            }
            let size = (s.h.lns_moves[ty].size + 0.5) as i32;
            {
                // SAFETY: the global domain does not change here
                let (glo, gup) = unsafe { bound_slices(&gd.bnd(), n) };
                // graph LNS runs outside the parallel lock: the heuristics'
                // own generator
                let rng = s.h.own_rng();
                s.lns.build(&s.g, ty as i32, size.max(0) as usize, deep, m.incumbent(), relaxationsol, glo, gup, rng);
            }
            let neighbourhood = s.lns.neighbourhood.clone();

            // fix everything outside the neighbourhood to the incumbent,
            // warm from the last LP solved, near the incumbent
            let start_iters = s.lp.num_lp_iterations();
            dom.assign(&gd);
            let mut feasible = true;
            for &c in &decisioncols {
                let col = c as usize;
                if s.lns.in_n[col] != 0 {
                    continue;
                }
                let val = rounded(&dom.bnd(), m.incumbent(), col);
                if !fix_to(&dom, c, val) {
                    feasible = false;
                    break;
                }
            }
            s.h.lns_moves[ty].tried += 1;
            if !feasible {
                since += 1;
                it += 1;
                continue;
            }
            s.lp.change_cols_bounds_dom(&dom);
            dom.clear_changed_cols();
            let before = m.upper_bound();
            let limit_before = m.optimality_limit();
            let gap_before = before - m.lower_bound();
            let st = s.solve(&dom);
            let nb_lp_iters = s.lp.num_lp_iterations() - start_iters;
            let mut nodes = 0;
            let mut exhausted = true;
            if usable(st) {
                // the reduced cost of a column fixed at the incumbent is the
                // first-order gain from flipping it: remember the promising
                // ones
                // SAFETY: as above
                let (glo, gup) = unsafe { bound_slices(&gd.bnd(), n) };
                let rc = s.lp.col_dual().to_vec();
                s.lns.update_promise(m.incumbent(), glo, gup, &rc);
            }
            let pruned = !usable(st) || s.lp.objective() >= w.upper_limit();
            if !pruned && !dived {
                nodes = s.search_neighbourhood(&dom, &neighbourhood, NODE_LIMIT, &mut exhausted);
            } else if !pruned && !s.try_incumbent(st) {
                s.dive(&dom, neighbourhood.clone(), 2.max(neighbourhood.len() as i32 / 40));
            }
            // a new incumbent from a neighbourhood often has cheap flips
            // nearby
            if deep && !pruned && m.upper_bound() < before - feastol && !s.within_gap() {
                s.flip_search(30);
            }
            let improved = m.upper_bound() < before - feastol;
            let progress = s.progress_since(before, limit_before);
            // effort in LP iterations rather than time, to stay
            // deterministic
            let effort = 1.0 + (s.lp.num_lp_iterations() - start_iters) as f64;
            let closed = if gap_before > 0.0 { (before - m.upper_bound()) / gap_before } else { 0.0 };
            let nt = &mut s.h.lns_moves[ty];
            if !dived {
                // aim for neighbourhoods that the node limit just about
                // exhausts: grow one that was searched without finding
                // anything, shrink one whose search did not finish
                if !pruned && exhausted && !improved {
                    nt.size = cmin(dfs_max_size, nt.size * 1.1);
                } else if !exhausted {
                    nt.size = cmax(dfs_min_size, nt.size * 0.85);
                }
            } else if pruned {
                // the neighbourhood LP cannot beat the incumbent: look wider
                nt.size = cmin(4.0 * dive_size0, nt.size * 1.25);
            } else if progress {
                nt.size = dive_size0;
            } else {
                nt.size = cmax(dive_size0 / 2.0, nt.size * 0.8);
            }
            nt.rate = 0.7f64.mul_add_c(nt.rate, 0.3 * closed / effort);
            if improved {
                nt.improved += 1;
            }
            crate::log_dev!(
                log,
                LogType::Verbose,
                "%s %3d type %d: %4d columns, %3d nodes%s%s, objective %.4f, %lld (first LP %lld) LP iterations\n",
                who,
                it,
                ty,
                neighbourhood.len(),
                nodes,
                if pruned { ", pruned" } else { "" },
                if exhausted { ", exhausted" } else { "" },
                m.upper_bound(),
                s.lp.num_lp_iterations() - s.lp_iters_start,
                nb_lp_iters
            );
            if progress {
                since = 0;
            } else {
                since += 1;
            }
            it += 1;
        }
        // the quick search ends by polishing the incumbent with a short flip
        // search
        if !deep && !s.within_gap() && !s.lp_budget_exceeded(heur_iters_cap) && !w.terminated() && !m.check_limits() {
            let before = m.upper_bound();
            let solves = s.flip_search(100);
            crate::log_dev!(
                log,
                LogType::Verbose,
                "%s flip polish: %d LP solves, objective %.4f -> %.4f\n",
                who,
                solves,
                before,
                m.upper_bound()
            );
        }
        let t = s.h.lns_moves;
        crate::log_dev!(
            log,
            LogType::Info,
            "%s %s search: improved by %d/%d (BFS) %d/%d (short rows) %d/%d (flips) %d/%d (dives) neighbourhoods\n",
            who,
            if deep { "deep" } else { "quick" },
            t[0].improved,
            t[0].tried,
            t[1].improved,
            t[1].tried,
            t[2].improved,
            t[2].tried,
            t[3].improved,
            t[3].tried
        );
        s.charge_iterations();
    }
}

/// graphLNS
///
/// # Safety
/// as primal::ffi::highs_rs_heur_run
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_heur_graph_lns(
    h: *mut Heur,
    f: *const glue::CMipFns,
    m: *const MipData,
    w: *mut std::ffi::c_void,
    x: *const f64,
    n: i32,
    deep: bool,
    max_lp_iters: i64,
) {
    glue::set_fns(f);
    let w = Worker::new(w);
    let x = crate::ffi::sl(x, n).to_vec();
    (*h).graph_lns(&*m, &w, &x, deep, max_lp_iters);
}
