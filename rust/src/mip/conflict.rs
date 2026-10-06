//! HighsDomain::ConflictSet (highs/mip/HighsDomain.cpp): conflict analysis
//! of an infeasible local domain, explaining the infeasibility (or a dual
//! proof) by the local bound changes, resolving them depth by depth back to
//! their reasons, and adding the frontiers as conflicts (and UIP
//! reconvergence cuts) to a conflict pool.
//!
//! The frontiers, std::set<LocalDomChg> ordered by stack position, are
//! BTreeMaps from the position; the resolve queue, a heap of set iterators
//! by position, is a BinaryHeap of positions (they are unique, so the pop
//! order is that of the C++ heap). The candidates of an explanation are
//! sorted by (priority descending, position): unique keys, so the order is
//! that of the C++ pdqsort. Adding a conflict to the pool goes through
//! [`CConflict`]'s C++ function. Adding
//! a conflict resizes the conflict arrays of the domains propagating the
//! pool, so the C++ refills both views after it, and no view is held across
//! it here.

use super::domain::{max2, CDomain, Dom, DomChg, Reason, INF, LOWER};
use super::domain::{REASON_BRANCHING, REASON_CONFLICTING_BOUNDS, REASON_MODEL_ROW_LOWER, REASON_MODEL_ROW_UPPER};
use super::domain::{REASON_OBJECTIVE, REASON_UNKNOWN};
use super::nodequeue::NodeQueue;
use super::pseudocost::Pseudocost;
use crate::ffi::sl;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use std::collections::{btree_map::Entry, BTreeMap, BinaryHeap};
use std::ffi::c_void;

const REASON_CLIQUE_TABLE: i32 = -5;

/// The conflict analysis of a local domain against its global domain,
/// mirrored by highs_rs::Conflict in highs/mip/HighsDomainRust.h
#[repr(C)]
pub struct CConflict {
    local: *const CDomain,
    global: *const CDomain,
    /// the HighsConflictPool, the pseudocosts and the node queue
    pool: *mut c_void,
    pseudocost: *mut Pseudocost,
    nodequeue: *const NodeQueue,
    num_integral: i32,
    /// addConflictCut and addReconvergenceCut (if domchg is not null) of the
    /// frontier's entries, then refills both views
    add_cut: unsafe extern "C" fn(*const CConflict, *const DomChg, i32, *const DomChg),
}

type Frontier = BTreeMap<i32, DomChg>;

/// ConflictSet::ResolveCandidate
#[derive(Clone, Copy)]
struct Candidate {
    delta: f64,
    base_bound: f64,
    prio: f64,
    bound_pos: i32,
    value_pos: i32,
}

/// std::min(a, b)
#[inline(always)]
fn min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline(always)]
fn compute_prio(val: f64, bound: f64, globalbound: f64, numnodes: i64) -> f64 {
    (val * (bound - globalbound) * (1 + numnodes) as f64).abs()
}

struct ConflictSet<'c> {
    c: &'c CConflict,
    reason_side: Frontier,
    reconvergence: Frontier,
    queue: BinaryHeap<i32>,
    resolved: Vec<(i32, DomChg)>,
    buffer: Vec<Candidate>,
}

impl<'c> ConflictSet<'c> {
    fn new(c: &'c CConflict) -> Self {
        ConflictSet {
            c,
            reason_side: Frontier::new(),
            reconvergence: Frontier::new(),
            queue: BinaryHeap::new(),
            resolved: Vec::new(),
            buffer: Vec::new(),
        }
    }

    /// The views of the local and the global domain
    #[inline(always)]
    fn views(&self) -> (Dom<'c>, Dom<'c>) {
        // SAFETY: distinct domains (C++ checks), valid during the analysis
        // except across add_cut, which refills them in place
        unsafe { (CDomain::view(self.c.local as *mut CDomain), CDomain::view(self.c.global as *mut CDomain)) }
    }

    fn num_nodes(&self, col: i32, up: bool) -> i64 {
        // SAFETY: the node queue does not change during the analysis
        let q = unsafe { &*self.c.nodequeue };
        if up {
            q.num_nodes_up(col)
        } else {
            q.num_nodes_down(col)
        }
    }

    fn increase_score(&self, col: i32, up: bool) {
        // SAFETY: the pseudocosts are not otherwise borrowed during the
        // analysis
        unsafe { (*self.c.pseudocost).increase_conflict_score(col, up) };
    }

    fn add_cut(&self, frontier: &Frontier, domchg: Option<&DomChg>) {
        let entries: Vec<DomChg> = frontier.values().copied().collect();
        // SAFETY: no view is alive (the callers drop theirs)
        unsafe {
            (self.c.add_cut)(
                self.c,
                entries.as_ptr(),
                entries.len() as i32,
                domchg.map_or(std::ptr::null(), |d| d as *const DomChg),
            )
        };
    }

    /// explainBoundChangeLeq (geq = false) / explainBoundChangeGeq (geq =
    /// true)
    #[allow(clippy::too_many_arguments)]
    fn explain_bound_change_linear(
        &mut self,
        l: &Dom,
        g: &Dom,
        frontier: &Frontier,
        pos: i32,
        domchg: DomChg,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        act: f64,
        geq: bool,
    ) -> bool {
        if act == if geq { INF } else { -INF } {
            return false;
        }
        // the coefficient of the column whose bound change is explained
        let mut domchg_val = 0.0;
        self.buffer.clear();
        let lb_ = l.bounds();
        let mut m = CDouble::from(act);
        for (i, (&col, &val)) in inds.iter().zip(vals).enumerate() {
            if col == domchg.column {
                domchg_val = val;
                continue;
            }
            let cu = col as usize;
            // Leq resolves lower bounds of positive and upper bounds of
            // negative coefficients, Geq the opposite
            let upper = (val > 0.0) == geq;
            let (bound, bound_pos) = lb_.col_bound_at(cu, pos, upper);
            let gbound = if upper { g.col_upper[cu] } else { g.col_lower[cu] };
            if (if upper { gbound <= bound } else { gbound >= bound }) || bound_pos == -1 {
                continue;
            }
            let base_bound = match frontier.get(&bound_pos) {
                Some(f) => {
                    let base = f.boundval;
                    if base != gbound {
                        m += val * (base - gbound);
                    }
                    if if upper { base <= bound } else { base >= bound } {
                        continue;
                    }
                    base
                }
                None => gbound,
            };
            self.buffer.push(Candidate {
                delta: val * (bound - base_bound),
                base_bound,
                prio: compute_prio(val, bound, gbound, self.num_nodes(col, !upper)),
                bound_pos,
                value_pos: i as i32,
            });
        }

        if domchg_val == 0.0 {
            return false;
        }

        self.sort_buffer();

        // the bound constraint a0 * x0 <= b0 (>= b0), relaxed by 1 - 10 *
        // feastol for an integral column (whose bound was rounded) or by
        // epsilon
        let mut b0 = domchg.boundval;
        let relax = if !l.is_continuous(domchg.column as usize) {
            (-10.0f64).mul_add_c(l.feastol, 1.0)
        } else {
            l.epsilon
        };
        if domchg.boundtype == LOWER {
            b0 -= relax;
        } else {
            b0 += relax;
        }
        b0 *= domchg_val;

        let mbound = rhs - b0;
        let c = domchg.column as usize;
        if (domchg_val < 0.0) != geq {
            m -= domchg_val * g.col_upper[c];
        } else {
            m -= domchg_val * g.col_lower[c];
        }
        self.resolve_linear(l, m, mbound, vals, geq)
    }

    /// pdqsort of the candidates: by priority, descending, then position
    fn sort_buffer(&mut self) {
        self.buffer.sort_unstable_by(|a, b| {
            use std::cmp::Ordering::*;
            if a.prio > b.prio {
                Less
            } else if b.prio > a.prio {
                Greater
            } else {
                a.bound_pos.cmp(&b.bound_pos)
            }
        });
    }

    /// resolveLinearLeq (geq = false) / resolveLinearGeq (geq = true)
    fn resolve_linear(&mut self, l: &Dom, mut m: CDouble, mbound: f64, vals: &[f64], geq: bool) -> bool {
        self.resolved.clear();
        let feastol = l.feastol;
        let epsilon = l.epsilon;
        let stack = l.stack();
        let prev = l.prev_bounds();
        // Leq needs M >= Mlower, Geq M <= Mupper; s orients the comparisons
        let s = if geq { -1.0 } else { 1.0 };
        let mut covered = (m - mbound).to_f64();
        if s * covered < 0.0 {
            for cand in &self.buffer {
                let pos = cand.bound_pos;
                m += cand.delta;
                self.resolved.push((pos, stack[pos as usize]));
                covered = (m - mbound).to_f64();
                if s * covered >= 0.0 {
                    break;
                }
            }
            if s * covered < 0.0 {
                return false;
            }

            if if geq { covered < -feastol } else { covered > feastol } {
                // there is room for relaxing bounds / dropping unneeded bound
                // changes from the explanation
                let mut k = self.resolved.len() as i32 - 1;
                while k >= 0 {
                    let ku = k as usize;
                    k -= 1;
                    let cand = self.buffer[ku];
                    let i = cand.value_pos as usize;
                    let loc = &mut self.resolved[ku];
                    let col = loc.1.column as usize;
                    if loc.1.boundtype == LOWER {
                        let lb = loc.1.boundval;
                        let glb = cand.base_bound;
                        let mut relax_lb = (((mbound - (m - cand.delta)) / vals[i]) + glb).to_f64();
                        if !l.is_continuous(col) {
                            relax_lb = relax_lb.ceil();
                        }
                        if relax_lb - lb >= -feastol {
                            continue;
                        }
                        loc.1.boundval = relax_lb;
                        if relax_lb - glb <= epsilon {
                            // domain change can be fully removed from conflict
                            let last = self.resolved.len() - 1;
                            self.resolved.swap(last, ku);
                            self.resolved.truncate(last);
                            m -= cand.delta;
                        } else {
                            while relax_lb <= prev[loc.0 as usize].val {
                                loc.0 = prev[loc.0 as usize].pos;
                            }
                            m += vals[i] * (relax_lb - lb);
                        }
                    } else {
                        let ub = loc.1.boundval;
                        let gub = cand.base_bound;
                        let mut relax_ub = (((mbound - (m - cand.delta)) / vals[i]) + gub).to_f64();
                        if !l.is_continuous(col) {
                            relax_ub = relax_ub.floor();
                        }
                        if relax_ub - ub <= feastol {
                            continue;
                        }
                        loc.1.boundval = relax_ub;
                        if relax_ub - gub >= -epsilon {
                            // domain change can be fully removed from conflict
                            let last = self.resolved.len() - 1;
                            self.resolved.swap(last, ku);
                            self.resolved.truncate(last);
                            m -= cand.delta;
                        } else {
                            while relax_ub >= prev[loc.0 as usize].val {
                                loc.0 = prev[loc.0 as usize].pos;
                            }
                            m += vals[i] * (relax_ub - ub);
                        }
                    }
                    covered = (m - mbound).to_f64();
                    if if geq { covered >= -feastol } else { covered <= feastol } {
                        break;
                    }
                }
            }
        }
        true
    }

    /// explainInfeasibilityLeq (geq = false) / explainInfeasibilityGeq
    #[allow(clippy::too_many_arguments)]
    fn explain_infeasibility_linear(
        &mut self,
        l: &Dom,
        g: &Dom,
        inds: &[i32],
        vals: &[f64],
        rhs: f64,
        act: f64,
        geq: bool,
    ) -> bool {
        if act == if geq { INF } else { -INF } {
            return false;
        }
        let (infeasible, _, ipos) = l.infeasibility();
        let infeasible_pos = if infeasible { ipos } else { i32::MAX };
        self.buffer.clear();
        let b = l.bounds();
        for (i, (&col, &val)) in inds.iter().zip(vals).enumerate() {
            let cu = col as usize;
            let upper = (val > 0.0) == geq;
            let (bound, bound_pos) = b.col_bound_at(cu, infeasible_pos, upper);
            let base_bound = if upper { g.col_upper[cu] } else { g.col_lower[cu] };
            if (if upper { base_bound <= bound } else { base_bound >= bound }) || bound_pos == -1 {
                continue;
            }
            self.buffer.push(Candidate {
                delta: val * (bound - base_bound),
                base_bound,
                prio: compute_prio(val, bound, base_bound, self.num_nodes(col, !upper)),
                bound_pos,
                value_pos: i as i32,
            });
        }
        self.sort_buffer();
        let m = max2(10.0, rhs.abs());
        let mbound = if geq { (-m).mul_add_c(l.feastol, rhs) } else { m.mul_add_c(l.feastol, rhs) };
        self.resolve_linear(l, CDouble::from(act), mbound, vals, geq)
    }

    /// explainInfeasibilityConflict
    fn explain_infeasibility_conflict(&mut self, l: &Dom, g: &Dom, conflict: &[DomChg]) -> bool {
        self.resolved.clear();
        let (_, _, infeasible_pos) = l.infeasibility();
        let b = l.bounds();
        let prev = l.prev_bounds();
        for e in conflict {
            if g.is_active(e) {
                continue;
            }
            let upper = e.boundtype != LOWER;
            let (bound, mut pos) = b.col_bound_at(e.column as usize, infeasible_pos, upper);
            if pos == -1 || (if upper { bound > e.boundval } else { bound < e.boundval }) {
                return false;
            }
            if upper {
                while prev[pos as usize].val <= e.boundval {
                    pos = prev[pos as usize].pos;
                }
            } else {
                while prev[pos as usize].val >= e.boundval {
                    pos = prev[pos as usize].pos;
                }
            }
            self.resolved.push((pos, *e));
        }
        true
    }

    /// explainInfeasibility
    fn explain_infeasibility(&mut self) -> bool {
        let (mut l, g) = self.views();
        let (_, reason, infeasible_pos) = l.infeasibility();
        match reason.kind {
            REASON_UNKNOWN | REASON_BRANCHING => false,
            REASON_CONFLICTING_BOUNDS => {
                self.resolved.clear();
                let p = reason.index;
                let stack = l.stack();
                let chg = stack[p as usize];
                self.resolved.push((p, chg));
                let (_, other) = l.bounds().col_bound_at(chg.column as usize, p, chg.boundtype == LOWER);
                if other != -1 {
                    self.resolved.push((other, stack[other as usize]));
                }
                true
            }
            REASON_CLIQUE_TABLE => false,
            REASON_MODEL_ROW_LOWER | REASON_MODEL_ROW_UPPER => {
                let row = reason.index as usize;
                let (inds, vals) = l.row(row);
                let (rl, ru) = l.row_bounds(row);
                if reason.kind == REASON_MODEL_ROW_LOWER {
                    self.explain_infeasibility_linear(&l, &g, inds, vals, rl, g.max_activity(row), true)
                } else {
                    self.explain_infeasibility_linear(&l, &g, inds, vals, ru, g.min_activity(row), false)
                }
            }
            REASON_OBJECTIVE => {
                let (inds, vals, rhs) = l.obj_propagation_constraint(infeasible_pos, -1);
                let (ninf, act) = g.bounds().compute_activity(inds, vals, false);
                // a globally unbounded column bounded locally, e.g., by a
                // branching of a heuristic
                if ninf > 0 {
                    return false;
                }
                self.explain_infeasibility_linear(&l, &g, inds, vals, rhs, act.to_f64(), false)
            }
            kind => {
                let k = kind as usize;
                let index = reason.index as usize;
                if k < l.num_cutpools() {
                    let (inds, vals, rhs) = l.cut(k, index);
                    let act = g.min_cut_activity(l.cutpool_id(k), index);
                    self.explain_infeasibility_linear(&l, &g, inds, vals, rhs, act, false)
                } else {
                    let pool = k - l.num_cutpools();
                    if l.conflict_deleted(pool, index) {
                        return false;
                    }
                    let conflict = l.conflict(pool, index);
                    self.explain_infeasibility_conflict(&l, &g, conflict)
                }
            }
        }
    }

    /// explainBoundChangeConflict
    fn explain_bound_change_conflict(&mut self, l: &Dom, g: &Dom, pos: i32, domchg: DomChg, conflict: &[DomChg]) -> bool {
        self.resolved.clear();
        let flipped = l.flip(&domchg);
        let b = l.bounds();
        let stack = l.stack();
        let prev = l.prev_bounds();
        let mut found = false;
        for e in conflict {
            if !found && e.column == flipped.column && e.boundtype == flipped.boundtype {
                if e.boundtype == LOWER {
                    if e.boundval <= flipped.boundval {
                        found = true;
                        continue;
                    }
                } else if e.boundval >= flipped.boundval {
                    found = true;
                    continue;
                }
            }
            if g.is_active(e) {
                continue;
            }
            let upper = e.boundtype != LOWER;
            let (bound, mut p) = b.col_bound_at(e.column as usize, pos - 1, upper);
            if p == -1 || (if upper { bound > e.boundval } else { bound < e.boundval }) {
                return false;
            }
            if upper {
                while prev[p as usize].val <= e.boundval {
                    p = prev[p as usize].pos;
                }
            } else {
                while prev[p as usize].val >= e.boundval {
                    p = prev[p as usize].pos;
                }
            }
            self.resolved.push((p, stack[p as usize]));
        }
        found
    }

    /// explainBoundChange
    fn explain_bound_change(&mut self, l: &mut Dom, g: &Dom, frontier: &Frontier, pos: i32, domchg: DomChg) -> bool {
        let reason: Reason = l.reasons()[pos as usize];
        match reason.kind {
            REASON_UNKNOWN | REASON_BRANCHING | REASON_CONFLICTING_BOUNDS => false,
            REASON_CLIQUE_TABLE => {
                let col = (reason.index >> 1) as usize;
                let val = reason.index & 1;
                self.resolved.clear();
                let (_, bound_pos) = l.bounds().col_bound_at(col, pos, val == 0);
                if bound_pos != -1 {
                    self.resolved.push((bound_pos, l.stack()[bound_pos as usize]));
                }
                true
            }
            REASON_MODEL_ROW_LOWER | REASON_MODEL_ROW_UPPER => {
                let row = reason.index as usize;
                let (inds, vals) = l.row(row);
                let (rl, ru) = l.row_bounds(row);
                if reason.kind == REASON_MODEL_ROW_LOWER {
                    self.explain_bound_change_linear(l, g, frontier, pos, domchg, inds, vals, rl, g.max_activity(row), true)
                } else {
                    self.explain_bound_change_linear(l, g, frontier, pos, domchg, inds, vals, ru, g.min_activity(row), false)
                }
            }
            REASON_OBJECTIVE => {
                let (inds, vals, rhs) = l.obj_propagation_constraint(pos, domchg.column);
                let (ninf, act) = g.bounds().compute_activity(inds, vals, false);
                // todo (C++): treat case with a single infinite contribution
                // that propagated a bound
                if ninf == 1 {
                    return false;
                }
                self.explain_bound_change_linear(l, g, frontier, pos, domchg, inds, vals, rhs, act.to_f64(), false)
            }
            kind => {
                let k = kind as usize;
                let index = reason.index as usize;
                if k < l.num_cutpools() {
                    let (inds, vals, rhs) = l.cut(k, index);
                    let act = g.min_cut_activity(l.cutpool_id(k), index);
                    self.explain_bound_change_linear(l, g, frontier, pos, domchg, inds, vals, rhs, act, false)
                } else {
                    let pool = k - l.num_cutpools();
                    if l.conflict_deleted(pool, index) {
                        return false;
                    }
                    let conflict = l.conflict(pool, index);
                    self.explain_bound_change_conflict(l, g, pos, domchg, conflict)
                }
            }
        }
    }

    /// resolvable
    fn resolvable(l: &Dom, pos: i32) -> bool {
        !matches!(l.reasons()[pos as usize].kind, REASON_BRANCHING | REASON_UNKNOWN)
    }

    /// resolveDepth on the reason side (reconvergence = false) or the
    /// reconvergence frontier
    fn resolve_depth(
        &mut self,
        reconvergence: bool,
        mut depth_level: usize,
        stop_size: usize,
        min_resolve: i32,
        increase_conflict_score: bool,
    ) -> i32 {
        let (mut l, g) = self.views();
        let mut frontier = std::mem::take(if reconvergence { &mut self.reconvergence } else { &mut self.reason_side });
        let n = self.resolve_depth_in(&mut l, &g, &mut frontier, &mut depth_level, stop_size, min_resolve, increase_conflict_score);
        *(if reconvergence { &mut self.reconvergence } else { &mut self.reason_side }) = frontier;
        n
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_depth_in(
        &mut self,
        l: &mut Dom,
        g: &Dom,
        frontier: &mut Frontier,
        depth_level: &mut usize,
        stop_size: usize,
        min_resolve: i32,
        increase_conflict_score: bool,
    ) -> i32 {
        self.queue.clear();
        let branchpos = l.branch_positions();
        let stack = l.stack();
        let prev = l.prev_bounds();
        let start_pos = if *depth_level == 0 { 0 } else { branchpos[*depth_level - 1] + 1 };
        while *depth_level < branchpos.len() {
            let bp = branchpos[*depth_level] as usize;
            if stack[bp].boundval != prev[bp].val {
                break;
            }
            *depth_level += 1;
        }

        let end = if *depth_level == branchpos.len() { i32::MAX } else { branchpos[*depth_level] };
        let mut empty = true;
        for &pos in frontier.range(start_pos..=end).map(|(p, _)| p) {
            empty = false;
            if Self::resolvable(l, pos) {
                self.queue.push(pos);
            }
        }
        if empty {
            return -1;
        }

        let mut num_resolved = 0;
        while self.queue.len() > stop_size || (!self.queue.is_empty() && num_resolved < min_resolve) {
            let pos = self.queue.pop().unwrap();
            let domchg = frontier[&pos];
            if !self.explain_bound_change(l, g, frontier, pos, domchg) {
                continue;
            }
            num_resolved += 1;
            frontier.remove(&pos);
            for &(p, d) in &self.resolved {
                match frontier.entry(p) {
                    Entry::Vacant(v) => {
                        v.insert(d);
                        if increase_conflict_score {
                            let s = stack[p as usize];
                            self.increase_score(s.column, s.boundtype == LOWER);
                        }
                        if p >= start_pos && Self::resolvable(l, p) {
                            self.queue.push(p);
                        }
                    }
                    Entry::Occupied(mut o) => {
                        let e = o.get_mut();
                        e.boundval = if d.boundtype == LOWER {
                            max2(e.boundval, d.boundval)
                        } else {
                            min2(e.boundval, d.boundval)
                        };
                    }
                }
            }
        }
        num_resolved
    }

    /// computeCuts
    fn compute_cuts(&mut self, depth_level: usize) -> i32 {
        let num_branch = self.views().0.branch_positions().len();
        let num_resolved = self.resolve_depth(false, depth_level, 1, (depth_level == num_branch) as i32, true);
        if num_resolved == -1 {
            return -1;
        }
        let mut num_conflicts = 0;
        if num_resolved > 0 {
            self.add_cut(&self.reason_side, None);
            num_conflicts += 1;
        }

        // if the queue size is 1 then we have a resolvable UIP that is not
        // the branch vertex
        if self.queue.len() == 1 {
            let uip_pos = self.queue.pop().unwrap();
            let uip = self.reason_side[&uip_pos];
            self.queue.clear();

            // compute the UIP reconvergence cut
            self.reconvergence.clear();
            self.reconvergence.insert(uip_pos, uip);
            let num_resolved = self.resolve_depth(true, depth_level, 0, 0, false);
            if num_resolved > 0 && !self.reconvergence.contains_key(&uip_pos) {
                self.add_cut(&self.reconvergence, Some(&uip));
                num_conflicts += 1;
            }
        }
        num_conflicts
    }

    /// The common part of both conflictAnalysis: conflict scores of the
    /// explanation, then the cuts depth by depth
    fn analyze_resolved(&mut self) {
        // SAFETY: as increase_score
        unsafe { (*self.c.pseudocost).increase_conflict_weight() };
        for &(_, d) in &self.resolved {
            self.increase_score(d.column, d.boundtype == LOWER);
        }

        if 10 * self.resolved.len() > 1000 + 3 * self.c.num_integral as usize {
            return;
        }

        for &(p, d) in &self.resolved {
            self.reason_side.entry(p).or_insert(d);
        }

        let mut num_conflicts = 0;
        let mut last_depth = self.views().0.branch_positions().len() as i32;
        let mut curr_depth = last_depth;
        while curr_depth >= 0 {
            if curr_depth > 0 {
                // skip redundant branching changes which are just added for
                // symmetry handling
                let (l, _) = self.views();
                let bp = l.branch_positions()[curr_depth as usize - 1] as usize;
                if l.stack()[bp].boundval == l.prev_bounds()[bp].val {
                    last_depth -= 1;
                    curr_depth -= 1;
                    continue;
                }
            }
            let num_new = self.compute_cuts(curr_depth as usize);
            // if the depth level was empty, do not consider it
            if num_new == -1 {
                last_depth -= 1;
                curr_depth -= 1;
                continue;
            }
            num_conflicts += num_new;
            // if no conflict was found in the first non-empty depth level we
            // stop here
            if num_conflicts == 0 {
                break;
            }
            // if no conflict was found in this depth level and all conflicts
            // of the first 5 non-empty depth levels are generated we stop
            if last_depth - curr_depth >= 4 && num_new == 0 {
                break;
            }
            curr_depth -= 1;
        }

        // if we stopped in the highest non-empty depth no conflicts were
        // added yet: add the current conflict frontier, as the bound change
        // leading to infeasibility was the last branching itself and should
        // have been propagated in the previous depth
        if curr_depth == last_depth {
            self.add_cut(&self.reason_side, None);
        }
    }

    /// ConflictSet::conflictAnalysis of the infeasible local domain
    fn conflict_analysis(&mut self) {
        if !self.explain_infeasibility() {
            return;
        }
        self.analyze_resolved();
    }

    /// ConflictSet::conflictAnalysis of a proof constraint
    fn conflict_analysis_proof(&mut self, inds: &[i32], vals: &[f64], rhs: f64) {
        let (l, g) = self.views();
        let (ninf, act) = g.bounds().compute_activity(inds, vals, false);
        if ninf != 0 {
            return;
        }
        if !self.explain_infeasibility_linear(&l, &g, inds, vals, rhs, act.to_f64(), false) {
            return;
        }
        self.analyze_resolved();
    }

    /// HighsDomain::conflictAnalyzeReconvergence (after its checks)
    fn analyze_reconvergence(&mut self, domchg: DomChg, inds: &[i32], vals: &[f64], rhs: f64) {
        let (l, g) = self.views();
        let (ninf, act) = g.bounds().compute_activity(inds, vals, false);
        if ninf != 0 {
            return;
        }
        let pos = l.stack().len() as i32;
        let frontier = Frontier::new();
        if !self.explain_bound_change_linear(&l, &g, &frontier, pos, domchg, inds, vals, rhs, act.to_f64(), false) {
            return;
        }
        if 10 * self.resolved.len() > 1000 + 3 * self.c.num_integral as usize {
            return;
        }
        for &(p, d) in &self.resolved {
            self.reconvergence.entry(p).or_insert(d);
        }
        let branchpos = l.branch_positions();
        let (stack, prev) = (l.stack(), l.prev_bounds());
        let mut depth = branchpos.len();
        while depth > 0 {
            let bp = branchpos[depth - 1] as usize;
            if stack[bp].boundval != prev[bp].val {
                break;
            }
            depth -= 1;
        }
        self.resolve_depth(true, depth, 0, 0, false);
        self.add_cut(&self.reconvergence, Some(&domchg));
    }
}

pub mod ffi {
    //! The `extern "C"` entry points of conflict analysis
    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflict_analysis(c: *const CConflict) {
        ConflictSet::new(&*c).conflict_analysis();
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflict_analysis_proof(
        c: *const CConflict,
        inds: *const i32,
        vals: *const f64,
        len: i32,
        rhs: f64,
    ) {
        ConflictSet::new(&*c).conflict_analysis_proof(sl(inds, len), sl(vals, len), rhs);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflict_reconvergence(
        c: *const CConflict,
        domchg: DomChg,
        inds: *const i32,
        vals: *const f64,
        len: i32,
        rhs: f64,
    ) {
        ConflictSet::new(&*c).analyze_reconvergence(domchg, sl(inds, len), sl(vals, len), rhs);
    }
}
