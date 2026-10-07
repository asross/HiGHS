//! HighsPrimalHeuristics (highs/mip/HighsPrimalHeuristics.cpp): the state
//! (integer columns in rounding order, graph-LNS decision columns and move
//! statistics, fixing-rate observations, the generator) and the heuristics
//! that drive the search, the domain and the LP relaxation: RENS, RINS,
//! rootReducedCost, randomized, central and line-search rounding,
//! tryRoundedPoint, the feasibility pump, crossover and the setup of
//! solveSubMip (the sub-MIP itself is a C++ HighsMipSolver). ziRound and
//! shifting are in heuristics.rs, graph LNS in lns.rs. The C++ objects are
//! reached through glue.rs.
//!
//! Under the parallel lock several workers run RENS, RINS and randomized
//! rounding at once on this object; they then only read it (they copy the
//! integer columns and draw from the worker's generator), as in the C++.

use super::cuts::sort::{partition, pdqsort};
use super::domain::{DomChg, Reason, LOWER, REASON_BRANCHING};
use super::glue::{self, lp_status, source, Dom, Lp, MipData, SearchH, Worker, MODEL_STATUS_INFEASIBLE, MODEL_STATUS_NOTSET};
use super::heuristics;
use super::lns::LnsMove;
use super::search::FracInt;
use crate::util::fma::ClangFma;
use crate::util::hash::hash_bytes;
use crate::util::random::HighsRandom;
use std::cell::UnsafeCell;
use std::collections::HashSet;

const INF: f64 = f64::INFINITY;
const IINF: i32 = i32::MAX;
const BRANCHING: Reason = Reason { kind: REASON_BRANCHING, index: 0 };

/// std::min(a, b)
#[inline(always)]
pub(crate) fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// std::max(a, b)
#[inline(always)]
pub(crate) fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// HighsHashHelpers::hash(uint64_t)
#[inline]
fn hash64(x: u64) -> u64 {
    hash_bytes(&x.to_ne_bytes())
}

/// HighsIntegers::nearestInteger (an int64_t)
#[inline]
fn nearest_integer(x: f64) -> f64 {
    (x + 0.5f64.copysign(x)) as i64 as f64
}

/// The state of HighsPrimalHeuristics
pub struct Heur {
    /// the integer columns in rounding order (setupIntCols); RENS and RINS
    /// drop the globally fixed ones when not under the parallel lock
    intcols: UnsafeCell<Vec<i32>>,
    pub(crate) decisioncols: Vec<i32>,
    pub(crate) decision_cols_set_up: bool,
    /// a graph-LNS dive from the LP point found nothing: don't dive again
    pub(crate) lns_dive_failed: bool,
    pub(crate) lns_moves: [LnsMove; 4],
    pub(crate) lns_flip_obj: f64,
    pub(crate) lns_flip_next: i32,
    success_observations: f64,
    num_success_observations: i32,
    infeas_observations: f64,
    num_infeas_observations: i32,
    randgen: UnsafeCell<HighsRandom>,
}

/// HeuristicNeighbourhood: the fixing rate of a local domain's integral
/// columns since the start
struct Neighbourhood {
    num_fixed_cols: HashSet<i32>,
    start_checked_changes: usize,
    n_checked_changes: usize,
    num_total: i32,
}

impl Neighbourhood {
    fn new(m: &MipData, localdom: &Dom) -> Self {
        let b = localdom.bnd();
        let mut num_fixed = 0;
        for &i in m.integral_cols() {
            if b.lo(i as usize) == b.up(i as usize) {
                num_fixed += 1;
            }
        }
        let start = localdom.stack_len();
        Neighbourhood {
            num_fixed_cols: HashSet::new(),
            start_checked_changes: start,
            n_checked_changes: start,
            num_total: m.integral_cols().len() as i32 - num_fixed,
        }
    }

    fn fixing_rate(&mut self, m: &MipData, localdom: &Dom) -> f64 {
        let stack = localdom.stack();
        let b = localdom.bnd();
        let integrality = m.integrality();
        while self.n_checked_changes < stack.len() {
            let col = stack[self.n_checked_changes].column;
            self.n_checked_changes += 1;
            if integrality[col as usize] == 0 {
                continue;
            }
            if b.lo(col as usize) == b.up(col as usize) {
                self.num_fixed_cols.insert(col);
            }
        }
        if self.num_total != 0 {
            self.num_fixed_cols.len() as f64 / self.num_total as f64
        } else {
            0.0
        }
    }

    fn backtracked(&mut self) {
        self.n_checked_changes = self.start_checked_changes;
        self.num_fixed_cols.clear();
    }
}

/// calcFixVal
fn calc_fix_val(rootchange: f64, fracval: f64, cost: f64) -> f64 {
    if rootchange >= 0.4 {
        fracval.ceil()
    } else if rootchange <= -0.4 {
        fracval.floor()
    } else if cost > 0.0 {
        fracval.ceil()
    } else if cost < 0.0 {
        fracval.floor()
    } else {
        (fracval + 0.5).floor()
    }
}

/// std::pair<double, uint64_t> operator<
#[inline]
fn pair_less(a: (f64, u64), b: (f64, u64)) -> bool {
    a.0 < b.0 || (!(b.0 < a.0) && a.1 < b.1)
}

impl Heur {
    pub fn new(seed: i32) -> Self {
        Heur {
            intcols: UnsafeCell::new(Vec::new()),
            decisioncols: Vec::new(),
            decision_cols_set_up: false,
            lns_dive_failed: false,
            lns_moves: [LnsMove::default(); 4],
            lns_flip_obj: INF,
            lns_flip_next: 0,
            success_observations: 0.0,
            num_success_observations: 0,
            infeas_observations: 0.0,
            num_infeas_observations: 0,
            randgen: UnsafeCell::new(HighsRandom::new(seed as u32)),
        }
    }

    pub(crate) fn intcols(&self) -> &[i32] {
        // SAFETY: changed only by setupIntCols and outside the parallel
        // lock (see the module comment)
        unsafe { &*self.intcols.get() }
    }

    /// The generator: the worker's under the parallel lock
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn rng<'a>(&'a self, m: &MipData, w: &'a Worker) -> &'a mut HighsRandom {
        if m.parallel_lock_active() {
            w.randgen()
        } else {
            // SAFETY: only one worker runs outside the parallel lock
            unsafe { &mut *self.randgen.get() }
        }
    }

    /// The heuristics' own generator
    pub(crate) fn own_rng(&mut self) -> &mut HighsRandom {
        self.randgen.get_mut()
    }

    /// getHeuristicRandom
    pub fn heuristic_random(&mut self, sup: i32) -> i32 {
        self.randgen.get_mut().integer_below(sup)
    }

    /// The integer columns for the dives (RENS, RINS): without the globally
    /// fixed ones, in place outside the parallel lock, else a copy
    fn unfixed_intcols(&self, m: &MipData, w: &Worker) -> Vec<i32> {
        let b = w.globaldom().bnd();
        let keep = |&i: &i32| b.lo(i as usize) != b.up(i as usize);
        if m.parallel_lock_active() {
            self.intcols().iter().copied().filter(keep).collect()
        } else {
            // SAFETY: outside the parallel lock nothing else uses intcols
            let v = unsafe { &mut *self.intcols.get() };
            v.retain(keep);
            v.clone()
        }
    }

    /// setupIntCols
    pub fn setup_int_cols(&mut self, m: &MipData) {
        let mut intcols = m.integer_cols().to_vec();
        self.decision_cols_set_up = false;
        self.decisioncols.clear();
        self.lns_moves = [LnsMove::default(); 4];
        self.lns_flip_obj = INF;
        let feastol = m.feastol();
        let (up, down) = (m.uplocks(), m.downlocks());
        let clq = m.clique();
        pdqsort(&mut intcols, |&c1, &c2| {
            let (u1, u2) = (c1 as usize, c2 as usize);
            let lock1 = (feastol + up[u1] as f64) * (feastol + down[u1] as f64);
            let lock2 = (feastol + up[u2] as f64) * (feastol + down[u2] as f64);
            if lock1 > lock2 {
                return true;
            }
            if lock2 > lock1 {
                return false;
            }
            let clq1 = (feastol + clq.get_num_implications_val(c1, true) as f64)
                * (feastol + clq.get_num_implications_val(c1, false) as f64);
            let clq2 = (feastol + clq.get_num_implications_val(c2, true) as f64)
                * (feastol + clq.get_num_implications_val(c2, false) as f64);
            let t1 = (clq1, hash64(c1 as i64 as u64), c1);
            let t2 = (clq2, hash64(c2 as i64 as u64), c2);
            // tuple t1 > t2, i.e. t2 < t1 lexicographically
            t2.0 < t1.0 || (!(t1.0 < t2.0) && (t2.1 < t1.1 || (!(t1.1 < t2.1) && t2.2 < t1.2)))
        });
        *self.intcols.get_mut() = intcols;
    }

    pub fn num_success_observations(&self, w: &Worker) -> i32 {
        self.num_success_observations + w.heur().num_success_observations
    }
    pub fn num_infeas_observations(&self, w: &Worker) -> i32 {
        self.num_infeas_observations + w.heur().num_infeas_observations
    }
    pub fn success_observations(&self, w: &Worker) -> f64 {
        self.success_observations + w.heur().success_observations
    }
    pub fn infeas_observations(&self, w: &Worker) -> f64 {
        self.infeas_observations + w.heur().infeas_observations
    }

    /// The observations part of flushStatistics
    pub fn add_observations(&mut self, s: f64, ns: i32, i: f64, ni: i32) {
        self.success_observations += s;
        self.num_success_observations += ns;
        self.infeas_observations += i;
        self.num_infeas_observations += ni;
    }

    /// determineTargetFixingRate
    fn determine_target_fixing_rate(&self, m: &MipData, w: &Worker) -> f64 {
        let mut low = 0.6;
        let mut high = 0.6;
        let rng = self.rng(m, w);
        if self.num_infeas_observations(w) != 0 {
            let infeas_rate = self.infeas_observations(w) / self.num_infeas_observations(w) as f64;
            high = 0.9 * infeas_rate;
            low = cmin(low, high);
        }
        if self.num_success_observations(w) != 0 {
            let success_rate = self.success_observations(w) / self.num_success_observations(w) as f64;
            low = cmin(low, 0.9 * success_rate);
            high = cmax(success_rate * 1.1, high);
        }
        rng.real(low, high)
    }

    /// solveSubMip: the gaps and the statistics around the C++ run of the
    /// sub-MIP. `lp` None: the model with the first root basis.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_sub_mip(
        &self,
        m: &MipData,
        w: &Worker,
        lp: Option<&Lp>,
        fixing_rate: f64,
        lo: &[f64],
        up: &[f64],
        maxleaves: i32,
        maxnodes: i32,
        stallnodes: i32,
        start: Option<&[f64]>,
        time_cap: f64,
    ) -> bool {
        // the gap target is the caller's, not the sub-MIP's (also for the
        // crossover of a concurrent LNS helper, itself a sub-MIP)
        let mut abs_gap = f64::NAN;
        if !m.submip || (start.is_some() && m.concurrent_helper) {
            let mut curr_abs_gap = w.upper_limit() - m.lower_bound();
            if curr_abs_gap == INF {
                curr_abs_gap = m.lower_bound().abs();
                if curr_abs_gap == INF {
                    curr_abs_gap = 0.0;
                }
            }
            abs_gap = m.feastol() * cmax(curr_abs_gap, 1000.0);
        }
        let mut sol = vec![0.0; m.num_col as usize];
        let r = glue::sub_mip(m, w, lp, lo, up, maxleaves, maxnodes, stallnodes, start, time_cap, abs_gap, &mut sol);
        let hs = w.heur();
        hs.max_submip_level = hs.max_submip_level.max(r.max_submip_level + 1);
        if r.termination_status != MODEL_STATUS_NOTSET {
            hs.termination_status = r.termination_status;
            return false;
        }
        let integral = m.integral_cols().len() as f64;
        let continuous = m.continuous_cols().len() as f64;
        let num_unfixed = integral + continuous;
        let adjustmentfactor = (1.0 - fixing_rate).mul_add_c(integral, continuous) / cmax(1.0, num_unfixed);
        let adjusted = (adjustmentfactor * r.total_lp_iterations as f64) as u64 as i64;
        hs.lp_iterations += adjusted;
        hs.total_repair_lp += r.total_repair_lp;
        hs.total_repair_lp_feasible += r.total_repair_lp_feasible;
        hs.total_repair_lp_iterations += r.total_repair_lp_iterations;
        // (not deterministic if sub-MIPs run in parallel, as in the C++)
        if m.submip {
            m.add_num_nodes(1i64.max((adjustmentfactor * r.node_count as f64) as i64));
        }
        if r.model_status == MODEL_STATUS_INFEASIBLE {
            hs.infeas_observations += fixing_rate;
            hs.num_infeas_observations += 1;
        }
        if r.node_count <= 1 && r.model_status == MODEL_STATUS_INFEASIBLE {
            return false;
        }
        let old_upper_limit = w.upper_limit();
        if r.model_status != MODEL_STATUS_INFEASIBLE && r.has_solution {
            glue::try_solution(m, w, &sol, source::SUB_MIP);
        }
        if w.upper_limit() < old_upper_limit {
            let hs = w.heur();
            hs.success_observations += fixing_rate;
            hs.num_success_observations += 1;
        }
        true
    }

    /// crossover: returns the number of integer columns where the
    /// solutions differ
    pub fn crossover(&self, m: &MipData, w: &Worker, other: &[f64], other_objective: f64, time_cap: f64) -> i32 {
        let inc = m.incumbent().to_vec();
        if inc.is_empty() || other.len() != inc.len() {
            return 0;
        }
        let gd = w.globaldom();
        if gd.infeasible() {
            return 0;
        }
        let n = m.num_col as usize;
        let b = gd.bnd();
        let mut lower: Vec<f64> = (0..n).map(|c| b.lo(c)).collect();
        let mut upper: Vec<f64> = (0..n).map(|c| b.up(c)).collect();
        let mut num_differ = 0;
        for &c in m.integral_cols() {
            let col = c as usize;
            let value = inc[col].round();
            if value != other[col].round() {
                num_differ += 1;
            } else if value >= lower[col] && value <= upper[col] {
                lower[col] = value;
                upper[col] = value;
            }
        }
        if num_differ == 0 {
            return 0;
        }
        let start = if other_objective < m.upper_bound() { other } else { &inc[..] };
        let fixing_rate = 1.0 - num_differ as f64 / m.integral_cols().len().max(1) as f64;
        self.solve_sub_mip(m, w, None, fixing_rate, &lower, &upper, IINF, IINF, 100, Some(start), time_cap);
        num_differ
    }

    /// rootReducedCost
    pub fn root_reduced_cost(&self, m: &MipData, w: &Worker) {
        let gd = w.globaldom();
        let n = m.num_col as usize;
        let mut lurking = {
            let b = gd.bnd();
            // SAFETY: the global domain's bounds, unchanged during the call
            let (lo, up) = unsafe { (std::slice::from_raw_parts(b.lo, n), std::slice::from_raw_parts(b.up, n)) };
            m.redcost().lurking_bounds(m.integral_cols(), lo, up)
        };
        if 10 * lurking.len() < m.integral_cols().len() {
            return;
        }
        pdqsort(&mut lurking, |a: &(f64, DomChg), b: &(f64, DomChg)| a.0 > b.0);
        let localdom = Dom::copy(&gd);
        let mut nb = Neighbourhood::new(m, &localdom);
        let lower_bound = m.lower_bound() + m.feastol();
        for &(key, chg) in &lurking {
            let curr_cutoff = key;
            if curr_cutoff <= lower_bound {
                break;
            }
            let b = localdom.bnd();
            let active = if chg.boundtype == LOWER {
                chg.boundval <= b.lo(chg.column as usize)
            } else {
                chg.boundval >= b.up(chg.column as usize)
            };
            if active {
                continue;
            }
            // (changeBound's default reason: a branching)
            localdom.change_bound(chg, BRANCHING);
            loop {
                localdom.propagate();
                if localdom.infeasible() {
                    localdom.conflict_analysis(w);
                    m.update_lower_bound(cmax(m.lower_bound(), curr_cutoff));
                    localdom.backtrack();
                    if localdom.branch_depth() == 0 {
                        break;
                    }
                    nb.backtracked();
                    continue;
                }
                break;
            }
            if nb.fixing_rate(m, &localdom) >= 0.5 {
                break;
            }
        }
        let fixing_rate = nb.fixing_rate(m, &localdom);
        if fixing_rate < 0.3 {
            return;
        }
        let b = localdom.bnd();
        let lo: Vec<f64> = (0..n).map(|c| b.lo(c)).collect();
        let up: Vec<f64> = (0..n).map(|c| b.up(c)).collect();
        self.solve_sub_mip(m, w, None, fixing_rate, &lo, &up, 500, 200 + (m.num_nodes() / 20) as i32, 12, None, INF);
    }

    /// The end of RENS and RINS: a depth-first search if the fixing rate
    /// is low, else a sub-MIP; returns true to retry with a lower fixing
    /// rate (updating `targetdepth` and `maxfixingrate`)
    #[allow(clippy::too_many_arguments)]
    fn finish_dive(
        &self,
        m: &MipData,
        w: &Worker,
        heur: &mut SearchH,
        heurlp: &Lp,
        localdom: &Dom,
        nb: &mut Neighbourhood,
        targetdepth: &mut i32,
        maxfixingrate: &mut f64,
    ) -> bool {
        // if there is no node left it means we backtracked to the global
        // domain and the subproblem was solved with the dive
        if !heur.s().has_node() {
            w.heur().lp_iterations += heur.stats().lpiterations;
            return false;
        }
        let fixingrate = nb.fixing_rate(m, localdom);
        if fixingrate < 0.1 || (m.submip && m.num_improving_sols() != 0) {
            heur.set_min_reliable(0);
            heur.s().solve_depth_first(10);
            w.heur().lp_iterations += heur.stats().lpiterations;
            if m.submip {
                m.add_num_nodes(heur.stats().nnodes);
            }
            return false;
        }
        heurlp.remove_obsolete_rows(false);
        let node_reduction_factor = if m.parallel_lock_active() { 1.max(m.num_workers() / 4) } else { 1 };
        let n = m.num_col as usize;
        let b = localdom.bnd();
        let lo: Vec<f64> = (0..n).map(|c| b.lo(c)).collect();
        let up: Vec<f64> = (0..n).map(|c| b.up(c)).collect();
        let maxnodes = 200 + m.num_nodes() / (node_reduction_factor as i64 * 20);
        let ret = self.solve_sub_mip(m, w, Some(heurlp), fixingrate, &lo, &up, 500, maxnodes as i32, 12, None, INF);
        if w.terminated() {
            return false;
        }
        if !ret {
            let new_lp_iterations = w.heur().lp_iterations + heur.stats().lpiterations;
            if new_lp_iterations + m.heuristic_lp_iterations()
                > 100000 + ((m.total_lp_iterations() - m.heuristic_lp_iterations() - m.sb_lp_iterations()) >> 1)
            {
                w.heur().lp_iterations = new_lp_iterations;
                return false;
            }
            *targetdepth = heur.s().current_depth() / 2;
            if *targetdepth <= 1 || (!m.parallel_lock_active() && m.check_limits()) {
                w.heur().lp_iterations = new_lp_iterations;
                return false;
            }
            *maxfixingrate = fixingrate * 0.5;
            return true;
        }
        w.heur().lp_iterations += heur.stats().lpiterations;
        false
    }

    /// The search, its local domain and LP copy for RENS and RINS
    fn dive_setup(m: &MipData, w: &Worker) -> (SearchH, Lp) {
        let mut heur = SearchH::new(w);
        heur.s().set_heuristic(true);
        let heurlp = Lp::copy(w.d.lp, w);
        // only use the global upper limit as LP limit so that dual proofs
        // are valid
        heurlp.set_objective_limit(w.upper_limit());
        heurlp.set_adjust_symmetric_branching_col(false);
        heur.set_lp(&heurlp);
        let localdom = heur.localdom();
        heurlp.change_cols_bounds_dom(&localdom);
        localdom.clear_changed_cols();
        heur.s().create_new_node();
        let _ = m;
        (heur, heurlp)
    }

    /// Branch on `col` to `val` upwards or downwards (and propagate if
    /// asked); false if the local domain became infeasible (after conflict
    /// analysis)
    #[allow(clippy::too_many_arguments)]
    fn branch(heur: &mut SearchH, w: &Worker, localdom: &Dom, col: i32, val: f64, point: f64, up: bool, propagate: bool) -> bool {
        heur.s().branch_dir(col, val, point, up);
        if propagate {
            localdom.propagate();
        }
        if localdom.infeasible() {
            localdom.conflict_analysis(w);
            return false;
        }
        true
    }

    /// RENS
    pub fn rens(&self, m: &MipData, w: &Worker) {
        if w.globaldom().infeasible() {
            return;
        }
        let (mut heur, heurlp) = Self::dive_setup(m, w);
        // (the C++ creates the search before dropping the fixed columns)
        let intcols = self.unfixed_intcols(m, w);
        let localdom = heur.localdom();
        let feastol = m.feastol();
        let mut maxfixingrate = self.determine_target_fixing_rate(m, w);
        let mut fixingrate;
        let mut targetdepth = 1;
        let mut nbacktracks = -1;
        let mut nb = Neighbourhood::new(m, &localdom);
        loop {
            // retry:
            nbacktracks += 1;
            nb.backtracked();
            if heur.s().current_depth() > targetdepth && !heur.s().backtrack_until_depth(targetdepth) {
                w.heur().lp_iterations += heur.stats().lpiterations;
                return;
            }
            loop {
                heur.s().evaluate_node();
                if heur.s().current_node_pruned() {
                    nbacktracks += 1;
                    if w.globaldom().infeasible() {
                        w.heur().lp_iterations += heur.stats().lpiterations;
                        return;
                    }
                    if !heur.s().backtrack(true) {
                        break;
                    }
                    nb.backtracked();
                    continue;
                }
                fixingrate = nb.fixing_rate(m, &localdom);
                if fixingrate >= maxfixingrate || nbacktracks >= 10 {
                    break;
                }
                let mut num_branched = 0;
                let stop_fixing_rate = cmin((-(1.0 - nb.fixing_rate(m, &localdom))).mul_add_c(0.9, 1.0), maxfixingrate);
                {
                    let relaxationsol = heurlp.col_value().to_vec();
                    for &i in &intcols {
                        let iu = i as usize;
                        let b = localdom.bnd();
                        if b.lo(iu) == b.up(iu) {
                            continue;
                        }
                        let mut downval = (relaxationsol[iu] + feastol).floor();
                        let mut upval = (relaxationsol[iu] - feastol).ceil();
                        downval = cmin(downval, b.up(iu));
                        upval = cmax(upval, b.lo(iu));
                        if b.lo(iu) < downval {
                            num_branched += 1;
                            if !Self::branch(&mut heur, w, &localdom, i, downval, downval - 0.5, true, true) {
                                break;
                            }
                        }
                        if localdom.bnd().up(iu) > upval {
                            num_branched += 1;
                            if !Self::branch(&mut heur, w, &localdom, i, upval, upval + 0.5, false, true) {
                                break;
                            }
                        }
                        if nb.fixing_rate(m, &localdom) >= stop_fixing_rate {
                            break;
                        }
                    }
                }
                if num_branched == 0 {
                    let rootlpsol = m.rootlpsol();
                    let cost = m.col_cost();
                    let get_fix_val = |col: i32, fracval: f64| {
                        let c = col as usize;
                        let rootchange = if rootlpsol.is_empty() { 0.0 } else { fracval - rootlpsol[c] };
                        let b = localdom.bnd();
                        let fixval = calc_fix_val(rootchange, fracval, cost[c]);
                        cmax(b.lo(c), cmin(b.up(c), fixval))
                    };
                    let frac = heurlp.frac();
                    let nfrac = frac.len() as u64;
                    pdqsort(frac, |a: &FracInt, b: &FracInt| {
                        pair_less(
                            ((get_fix_val(a.col, a.val) - a.val).abs(), hash64(((a.col as i64 as u64) << 32).wrapping_add(nfrac))),
                            ((get_fix_val(b.col, b.val) - b.val).abs(), hash64(((b.col as i64 as u64) << 32).wrapping_add(nfrac))),
                        )
                    });
                    let mut change = 0.0;
                    let fracs: Vec<FracInt> = heurlp.frac().to_vec();
                    for f in fracs {
                        let fixval = get_fix_val(f.col, f.val);
                        if localdom.bnd().lo(f.col as usize) < fixval {
                            num_branched += 1;
                            if !Self::branch(&mut heur, w, &localdom, f.col, fixval, f.val, true, true) {
                                break;
                            }
                            fixingrate = nb.fixing_rate(m, &localdom);
                        }
                        if localdom.bnd().up(f.col as usize) > fixval {
                            num_branched += 1;
                            if !Self::branch(&mut heur, w, &localdom, f.col, fixval, f.val, false, true) {
                                break;
                            }
                            fixingrate = nb.fixing_rate(m, &localdom);
                        }
                        if fixingrate >= maxfixingrate {
                            break;
                        }
                        change += (fixval - f.val).abs();
                        if change >= 0.5 {
                            break;
                        }
                    }
                }
                if num_branched == 0 {
                    break;
                }
                heurlp.flush_domain(&localdom);
            }
            if !self.finish_dive(m, w, &mut heur, &heurlp, &localdom, &mut nb, &mut targetdepth, &mut maxfixingrate) {
                return;
            }
        }
    }

    /// RINS
    pub fn rins(&self, m: &MipData, w: &Worker, relaxationsol: &[f64]) {
        if w.globaldom().infeasible() {
            return;
        }
        if relaxationsol.len() != m.num_col as usize {
            return;
        }
        let intcols = self.unfixed_intcols(m, w);
        let (mut heur, heurlp) = Self::dive_setup(m, w);
        let localdom = heur.localdom();
        let feastol = m.feastol();
        let mut maxfixingrate = self.determine_target_fixing_rate(m, w);
        let minfixingrate = 0.25;
        let mut fixingrate;
        let mut nbacktracks = -1;
        let mut targetdepth = 1;
        let mut nb = Neighbourhood::new(m, &localdom);
        loop {
            // retry:
            nbacktracks += 1;
            nb.backtracked();
            if heur.s().current_depth() > targetdepth && !heur.s().backtrack_until_depth(targetdepth) {
                w.heur().lp_iterations += heur.stats().lpiterations;
                return;
            }
            loop {
                heur.s().evaluate_node();
                if heur.s().current_node_pruned() {
                    nbacktracks += 1;
                    if w.globaldom().infeasible() {
                        w.heur().lp_iterations += heur.stats().lpiterations;
                        return;
                    }
                    if !heur.s().backtrack(true) {
                        break;
                    }
                    nb.backtracked();
                    continue;
                }
                fixingrate = nb.fixing_rate(m, &localdom);
                if fixingrate >= maxfixingrate || nbacktracks >= 10 {
                    break;
                }
                let incumbent = m.incumbent();
                // the fractional variables to fix first: toward the RINS
                // neighbourhood
                let mut fixcandend = partition(heurlp.frac(), |f: &FracInt| {
                    (relaxationsol[f.col as usize] - incumbent[f.col as usize]).abs() <= feastol
                });
                let mut fixtolpsol = true;
                let mut num_branched = 0;
                if fixcandend == 0 {
                    fixingrate = nb.fixing_rate(m, &localdom);
                    let stop_fixing_rate = cmin(maxfixingrate, (-(1.0 - fixingrate)).mul_add_c(0.9, 1.0));
                    let currlpsol = heurlp.col_value().to_vec();
                    for &i in &intcols {
                        let iu = i as usize;
                        let b = localdom.bnd();
                        if b.lo(iu) == b.up(iu) {
                            continue;
                        }
                        if (currlpsol[iu] - incumbent[iu]).abs() <= feastol {
                            let fixval = nearest_integer(currlpsol[iu]);
                            if b.lo(iu) < fixval {
                                num_branched += 1;
                                if !Self::branch(&mut heur, w, &localdom, i, fixval, fixval - 0.5, true, true) {
                                    break;
                                }
                                fixingrate = nb.fixing_rate(m, &localdom);
                            }
                            if localdom.bnd().up(iu) > fixval {
                                num_branched += 1;
                                if !Self::branch(&mut heur, w, &localdom, i, fixval, fixval + 0.5, false, true) {
                                    break;
                                }
                                fixingrate = nb.fixing_rate(m, &localdom);
                            }
                            if fixingrate >= stop_fixing_rate {
                                break;
                            }
                        }
                    }
                    if num_branched != 0 {
                        heurlp.flush_domain(&localdom);
                        continue;
                    }
                    if fixingrate >= minfixingrate {
                        // the RINS neighbourhood achieved a high enough
                        // fixing rate by itself
                        break;
                    }
                    fixcandend = heurlp.frac().len();
                    fixtolpsol = false;
                }
                let rootlpsol = m.rootlpsol();
                let cost = m.col_cost();
                let get_fix_val = |col: i32, fracval: f64| {
                    let c = col as usize;
                    let fixval = if fixtolpsol {
                        (relaxationsol[c] + 0.5).floor()
                    } else {
                        calc_fix_val(fracval - rootlpsol[c], fracval, cost[c])
                    };
                    let b = localdom.bnd();
                    cmax(b.lo(c), cmin(b.up(c), fixval))
                };
                let frac = heurlp.frac();
                let nfrac = frac.len() as u64;
                pdqsort(&mut frac[..fixcandend], |a: &FracInt, b: &FracInt| {
                    pair_less(
                        ((get_fix_val(a.col, a.val) - a.val).abs(), hash64(((a.col as i64 as u64) << 32).wrapping_add(nfrac))),
                        ((get_fix_val(b.col, b.val) - b.val).abs(), hash64(((b.col as i64 as u64) << 32).wrapping_add(nfrac))),
                    )
                });
                let mut change = 0.0;
                let fracs: Vec<FracInt> = heurlp.frac()[..fixcandend].to_vec();
                for f in fracs {
                    let fixval = get_fix_val(f.col, f.val);
                    if localdom.bnd().lo(f.col as usize) < fixval {
                        num_branched += 1;
                        if !Self::branch(&mut heur, w, &localdom, f.col, fixval, f.val, true, false) {
                            break;
                        }
                        fixingrate = nb.fixing_rate(m, &localdom);
                    }
                    if localdom.bnd().up(f.col as usize) > fixval {
                        num_branched += 1;
                        if !Self::branch(&mut heur, w, &localdom, f.col, fixval, f.val, false, false) {
                            break;
                        }
                        fixingrate = nb.fixing_rate(m, &localdom);
                    }
                    if fixingrate >= maxfixingrate {
                        break;
                    }
                    change += (fixval - f.val).abs();
                    if change >= 0.5 {
                        break;
                    }
                }
                if num_branched == 0 {
                    break;
                }
                heurlp.flush_domain(&localdom);
            }
            if !self.finish_dive(m, w, &mut heur, &heurlp, &localdom, &mut nb, &mut targetdepth, &mut maxfixingrate) {
                return;
            }
        }
    }

    /// The LP over the columns left free by a rounding (tryRoundedPoint,
    /// randomizedRounding): a fresh relaxation of the model at the local
    /// bounds, with presolve when few columns are left, else from the
    /// first root basis
    fn rounding_lp(m: &MipData, w: &Worker, localdom: &Dom, use_presolve: bool, origin: &[u8]) -> (Lp, i32) {
        let lprelax = Lp::new(m, w);
        lprelax.set_iteration_limit(10000i64.max(2 * m.firstrootlpiters()) as i32);
        lprelax.change_cols_bounds_dom(localdom);
        if m.root_presolve_only {
            lprelax.set_option(0);
        }
        if !m.root_presolve_only && use_presolve {
            lprelax.set_option(1);
        } else {
            lprelax.set_root_basis(origin);
        }
        let st = lprelax.resolve(None);
        (lprelax, st)
    }

    /// tryRoundedPoint
    pub fn try_rounded_point(&self, m: &MipData, w: &Worker, point: &[f64], solution_source: i32) -> bool {
        let localdom = Dom::copy(&w.globaldom());
        let mut integer_feasible = true;
        let intcols = self.intcols();
        let numintcols = intcols.len() as i32;
        let feastol = m.feastol();
        for i in 0..numintcols {
            // propagating after each fixing can take long on large models
            if (i & 1023) == 1023 && m.check_limits() {
                return false;
            }
            let col = intcols[i as usize];
            let mut intval = point[col as usize];
            let rounded = intval.round();
            let feasible = (intval - rounded).abs() <= feastol;
            integer_feasible = integer_feasible && feasible;
            if !feasible {
                continue;
            }
            let b = localdom.bnd();
            intval = cmin(b.up(col as usize), rounded);
            intval = cmax(b.lo(col as usize), intval);
            localdom.fix_col(col, intval, BRANCHING);
            if localdom.infeasible() {
                localdom.conflict_analysis(w);
                return false;
            }
            localdom.propagate();
            if localdom.infeasible() {
                localdom.conflict_analysis(w);
                return false;
            }
        }
        if numintcols != m.num_col {
            let (lprelax, st) = Self::rounding_lp(
                m,
                w,
                &localdom,
                (5 * numintcols) / m.num_col >= 1,
                b"HighsPrimalHeuristics::tryRoundedPoint\0",
            );
            if st == lp_status::INFEASIBLE {
                lprelax.infeasible_conflict(w, &localdom);
                return false;
            } else if lp_status::unscaled_primal_feasible(st) {
                let mut lpsol = lprelax.col_value().to_vec();
                if !integer_feasible {
                    // there may be fractional integer variables -> try
                    // ziRound heuristic
                    self.zi_round(m, w, &lpsol);
                    lpsol = lprelax.col_value().to_vec();
                    return glue::try_solution(m, w, &lpsol, solution_source);
                } else {
                    // all integer variables are fixed -> add incumbent
                    glue::add_incumbent(m, w, &lpsol, lprelax.objective(), solution_source);
                    return true;
                }
            }
        }
        let b = localdom.bnd();
        let lower: Vec<f64> = (0..m.num_col as usize).map(|c| b.lo(c)).collect();
        glue::try_solution(m, w, &lower, solution_source)
    }

    /// linesearchRounding
    pub fn linesearch_rounding(&self, m: &MipData, w: &Worker, point1: &[f64], point2: &[f64], solution_source: i32) -> bool {
        let n = m.num_col as usize;
        let mut roundedpoint = vec![0.0; n];
        let mut alpha = 0.0;
        let feastol = m.feastol();
        while alpha < 1.0 {
            if m.check_limits() {
                return false;
            }
            let mut nextalpha = 1.0;
            let mut reachedpoint2 = true;
            let (uplocks, downlocks) = (m.uplocks(), m.downlocks());
            for &c in self.intcols() {
                let col = c as usize;
                if uplocks[col] == 0 {
                    roundedpoint[col] = (cmax(point1[col], point2[col]) - feastol).ceil();
                    continue;
                }
                if downlocks[col] == 0 {
                    roundedpoint[col] = (cmin(point1[col], point2[col]) + feastol).floor();
                    continue;
                }
                let convexcomb = (1.0 - alpha).mul_add_c(point1[col], alpha * point2[col]);
                let intpoint2 = (point2[col] + 0.5).floor();
                roundedpoint[col] = (convexcomb + 0.5).floor();
                if roundedpoint[col] == intpoint2 {
                    continue;
                }
                reachedpoint2 = false;
                let tmpalpha = (roundedpoint[col] + 0.5 + feastol - point1[col]) / (point2[col] - point1[col]).abs();
                if tmpalpha < nextalpha && tmpalpha > alpha + 1e-2 {
                    nextalpha = tmpalpha;
                }
            }
            if self.try_rounded_point(m, w, &roundedpoint, solution_source) {
                return true;
            }
            if reachedpoint2 {
                return false;
            }
            alpha = nextalpha;
        }
        false
    }

    /// randomizedRounding
    pub fn randomized_rounding(&self, m: &MipData, w: &Worker, relaxationsol: &[f64]) {
        let n = m.num_col as usize;
        if relaxationsol.len() != n {
            return;
        }
        let localdom = Dom::copy(&w.globaldom());
        let rng = self.rng(m, w);
        let feastol = m.feastol();
        let mut num_fixed: i32 = 0;
        let intcols = self.intcols();
        for &i in intcols {
            let iu = i as usize;
            // propagating after each fixing can take long on large models
            num_fixed = num_fixed.wrapping_add(1);
            if (num_fixed & 1023) == 0 && m.check_limits() {
                return;
            }
            let mut intval = if m.uplocks()[iu] == 0 {
                (relaxationsol[iu] - feastol).ceil()
            } else if m.downlocks()[iu] == 0 {
                (relaxationsol[iu] + feastol).floor()
            } else {
                (relaxationsol[iu] + rng.real(0.1, 0.9)).floor()
            };
            let b = localdom.bnd();
            intval = cmin(b.up(iu), intval);
            intval = cmax(b.lo(iu), intval);
            localdom.fix_col(i, intval, BRANCHING);
            if localdom.infeasible() {
                localdom.conflict_analysis(w);
                return;
            }
            localdom.propagate();
            if localdom.infeasible() {
                localdom.conflict_analysis(w);
                return;
            }
        }
        if m.integer_cols().len() != n {
            let (lprelax, st) = Self::rounding_lp(
                m,
                w,
                &localdom,
                (5 * intcols.len()) / n >= 1,
                b"HighsPrimalHeuristics::randomizedRounding\0",
            );
            if st == lp_status::INFEASIBLE {
                lprelax.infeasible_conflict(w, &localdom);
            } else if lp_status::unscaled_primal_feasible(st) {
                let sol = lprelax.col_value().to_vec();
                glue::add_incumbent(m, w, &sol, lprelax.objective(), source::RANDOMIZED_ROUNDING);
            }
        } else {
            let b = localdom.bnd();
            let lower: Vec<f64> = (0..n).map(|c| b.lo(c)).collect();
            glue::try_solution(m, w, &lower, source::RANDOMIZED_ROUNDING);
        }
    }

    /// The model for ziRound and shifting
    fn heur_lp<'a>(m: &'a MipData) -> heuristics::Lp<'a> {
        heuristics::Lp {
            a_start: m.a_start(),
            a_index: m.a_index(),
            a_value: m.a_value(),
            col_lower: m.col_lower(),
            col_upper: m.col_upper(),
            col_cost: m.col_cost(),
            row_lower: m.row_lower(),
            row_upper: m.row_upper(),
            minimize: m.minimize,
        }
    }

    /// ziRound
    pub fn zi_round(&self, m: &MipData, w: &Worker, relaxationsol: &[f64]) {
        if relaxationsol.len() != m.num_col as usize {
            return;
        }
        // ponytail: the model is column-wise here (Highs and presolve make
        // it so); the row-wise C++ fallback is not ported
        debug_assert!(m.colwise);
        let mut sol = relaxationsol.to_vec();
        if heuristics::zi_round(&Self::heur_lp(m), self.intcols(), m.feastol(), &mut sol) {
            glue::try_solution(m, w, &sol, source::ZI_ROUND);
        }
    }

    /// shifting
    pub fn shifting(&self, m: &MipData, w: &Worker, relaxationsol: &[f64]) {
        let n = m.num_col as usize;
        if relaxationsol.len() != n {
            return;
        }
        debug_assert!(m.colwise);
        // the LP relaxation's fractional integers, without copying it
        let mut frac: Vec<(i32, f64)> = Lp::borrowed(w.d.lp).frac().iter().map(|f| (f.col, f.val)).collect();
        let rows = heuristics::MipRows {
            ar_start: m.ar_start(),
            ar_index: m.ar_index(),
            ar_value: m.ar_value(),
            integrality: m.integrality(),
            uplocks: m.uplocks(),
            downlocks: m.downlocks(),
            maximize: m.orig_maximize,
            num_integer_cols: m.integer_cols().len(),
        };
        let mut sol = relaxationsol.to_vec();
        let rng = self.rng(m, w);
        let infeasible = heuristics::shifting(&Self::heur_lp(m), &rows, m.feastol(), &mut frac, rng, &mut sol);
        if infeasible {
            self.try_rounded_point(m, w, &sol, source::SHIFTING);
        } else if !frac.is_empty() {
            self.zi_round(m, w, &sol);
        } else {
            glue::try_solution(m, w, &sol, source::SHIFTING);
        }
    }

    /// feasibilityPump
    pub fn feasibility_pump(&self, m: &MipData, w: &Worker) {
        let lprelax = Lp::copy(w.d.lp, w);
        let mut referencepoints: HashSet<Vec<i32>> = HashSet::new();
        let mut status = lprelax.resolve(None);
        w.heur().lp_iterations += lprelax.num_lp_iterations();
        let rng = self.rng(m, w);
        let n = m.num_col as usize;
        let mask = vec![1i32; n];
        let mut cost = vec![0.0; n];
        let feastol = m.feastol();
        lprelax.set_option(2);
        lprelax.set_objective_limit(INF);
        lprelax.set_option(3);
        lprelax.set_iteration_limit((5.0 * m.avgrootlpiters()) as i32);
        let integer_cols = m.integer_cols().to_vec();
        while !lprelax.frac().is_empty() {
            let lpsol = lprelax.col_value().to_vec();
            let mut roundedsol = lpsol.clone();
            let mut referencepoint: Vec<i32> = Vec::with_capacity(integer_cols.len());
            let localdom = Dom::copy(&w.globaldom());
            for &i in &integer_cols {
                let iu = i as usize;
                let mut intval = (roundedsol[iu] + rng.real(0.4, 0.6)).floor();
                let b = localdom.bnd();
                intval = cmax(intval, b.lo(iu));
                intval = cmin(intval, b.up(iu));
                roundedsol[iu] = intval;
                referencepoint.push(intval as i32);
                if !localdom.infeasible() {
                    localdom.fix_col(i, intval, BRANCHING);
                    if localdom.infeasible() {
                        localdom.conflict_analysis(w);
                        continue;
                    }
                    localdom.propagate();
                    if localdom.infeasible() {
                        localdom.conflict_analysis(w);
                        continue;
                    }
                }
            }
            let mut havecycle = !referencepoints.insert(referencepoint.clone());
            let gb = w.globaldom().bnd();
            let mut k = 0;
            while havecycle && k < 2 {
                for _ in 0..10 {
                    let flippos = rng.integer_below(integer_cols.len() as i32) as usize;
                    let col = integer_cols[flippos] as usize;
                    if roundedsol[col] > lpsol[col] {
                        roundedsol[col] = lpsol[col].floor() as i32 as f64;
                    } else if roundedsol[col] < lpsol[col] {
                        roundedsol[col] = lpsol[col].ceil() as i32 as f64;
                    } else if roundedsol[col] < gb.up(col) {
                        roundedsol[col] = gb.up(col);
                    } else {
                        roundedsol[col] = gb.lo(col);
                    }
                    referencepoint[flippos] = roundedsol[col] as i32;
                }
                havecycle = !referencepoints.insert(referencepoint.clone());
                k += 1;
            }
            if havecycle {
                return;
            }
            if self.linesearch_rounding(m, w, &lpsol, &roundedsol, source::FEASIBILITY_PUMP) {
                return;
            }
            if lprelax.num_lp_iterations() as f64 >= m.avgrootlpiters().mul_add_c(5.0, 1000.0) {
                break;
            }
            let (uplocks, downlocks) = (m.uplocks(), m.downlocks());
            for &i in &integer_cols {
                let iu = i as usize;
                if uplocks[iu] == 0 || downlocks[iu] == 0 {
                    cost[iu] = 0.0;
                } else if lpsol[iu] > roundedsol[iu] - feastol {
                    cost[iu] = -1.0 + rng.real(-1e-4, 1e-4);
                } else {
                    cost[iu] = 1.0 + rng.real(-1e-4, 1e-4);
                }
            }
            lprelax.change_cols_cost(&mask, &cost);
            let mut niters = -lprelax.num_lp_iterations();
            status = lprelax.resolve(None);
            niters += lprelax.num_lp_iterations();
            if niters == 0 {
                break;
            }
            w.heur().lp_iterations += niters;
        }
        if lprelax.frac().is_empty() && lp_status::unscaled_primal_feasible(status) {
            let sol = lprelax.col_value().to_vec();
            glue::add_incumbent(m, w, &sol, lprelax.objective(), source::FEASIBILITY_PUMP);
        }
    }

    /// centralRounding
    pub fn central_rounding(&self, m: &MipData, w: &Worker) {
        let ac = m.analytic_center().to_vec();
        if ac.len() != m.num_col as usize {
            return;
        }
        let first = m.firstlpsol().to_vec();
        if !first.is_empty() {
            self.linesearch_rounding(m, w, &first, &ac, source::CENTRAL_ROUNDING);
        } else {
            let root = m.rootlpsol().to_vec();
            if !root.is_empty() {
                self.linesearch_rounding(m, w, &root, &ac, source::CENTRAL_ROUNDING);
            } else {
                self.linesearch_rounding(m, w, &ac, &ac, source::CENTRAL_ROUNDING);
            }
        }
    }
}

// ---- C interface (HighsPrimalHeuristics.cpp under HIGHS_RUST) ----

pub mod ffi {
    use super::*;
    use crate::ffi::sl;
    use glue::{set_fns, CMipFns};

    #[no_mangle]
    pub extern "C" fn highs_rs_heur_new(seed: i32) -> *mut Heur {
        Box::into_raw(Box::new(Heur::new(seed)))
    }

    /// # Safety
    /// `h` from highs_rs_heur_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heur_free(h: *mut Heur) {
        if !h.is_null() {
            drop(Box::from_raw(h));
        }
    }

    /// The heuristics with a point (or none): 0 setupIntCols, 1 RENS, 2
    /// RINS(x), 3 rootReducedCost, 4 feasibilityPump, 5 centralRounding,
    /// 6 randomizedRounding(x), 7 shifting(x), 8 ziRound(x)
    ///
    /// # Safety
    /// `h` live, `f` the C++ functions, `m` filled for this call, `w` the
    /// worker, `x` `n` values. Several workers may call this at once for
    /// RENS, RINS and randomized rounding (see the module comment).
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heur_run(
        h: *mut Heur,
        f: *const CMipFns,
        m: *const MipData,
        w: *mut std::ffi::c_void,
        which: i32,
        x: *const f64,
        n: i32,
    ) {
        set_fns(f);
        let m = &*m;
        if which == 0 {
            (*h).setup_int_cols(m);
            return;
        }
        let h = &*h;
        let w = Worker::new(w);
        let x = sl(x, n).to_vec();
        match which {
            1 => h.rens(m, &w),
            2 => h.rins(m, &w, &x),
            3 => h.root_reduced_cost(m, &w),
            4 => h.feasibility_pump(m, &w),
            5 => h.central_rounding(m, &w),
            6 => h.randomized_rounding(m, &w, &x),
            7 => h.shifting(m, &w, &x),
            _ => h.zi_round(m, &w, &x),
        }
    }

    /// crossover
    ///
    /// # Safety
    /// as highs_rs_heur_run
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_heur_crossover(
        h: *mut Heur,
        f: *const CMipFns,
        m: *const MipData,
        w: *mut std::ffi::c_void,
        other: *const f64,
        n: i32,
        other_objective: f64,
        time_cap: f64,
    ) -> i32 {
        set_fns(f);
        let w = Worker::new(w);
        (*h).crossover(&*m, &w, sl(other, n), other_objective, time_cap)
    }

    /// tryRoundedPoint / linesearchRounding (`y` non-null)
    ///
    /// # Safety
    /// as highs_rs_heur_run
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_heur_rounding(
        h: *mut Heur,
        f: *const CMipFns,
        m: *const MipData,
        w: *mut std::ffi::c_void,
        x: *const f64,
        y: *const f64,
        n: i32,
        solution_source: i32,
    ) -> bool {
        set_fns(f);
        let w = Worker::new(w);
        let x = sl(x, n).to_vec();
        if y.is_null() {
            (*h).try_rounded_point(&*m, &w, &x, solution_source)
        } else {
            let y = sl(y, n).to_vec();
            (*h).linesearch_rounding(&*m, &w, &x, &y, solution_source)
        }
    }

    /// The observations of flushStatistics
    ///
    /// # Safety
    /// `h` live
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heur_add_observations(h: *mut Heur, s: f64, ns: i32, i: f64, ni: i32) {
        (*h).add_observations(s, ns, i, ni);
    }

    /// getHeuristicRandom
    ///
    /// # Safety
    /// `h` live
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heur_random(h: *mut Heur, sup: i32) -> i32 {
        (*h).heuristic_random(sup)
    }
}
