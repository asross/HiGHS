//! HighsImplications (highs/mip/HighsImplications.cpp): the implications
//! of fixing a binary column found by probing, the variable upper and
//! lower bounds, and the implied bound cuts.
//!
//! Rust owns the data (`HighsImplications` in C++ holds a handle). As for
//! the clique table (clique.rs), the domain and the rest of the MIP solver
//! are C++ callbacks ([`CDom`], [`CImp`]), and a bound change can re-enter
//! [`Implications::apply_implications`], which reads only the
//! `implications` field, through the raw pointer. The methods that change
//! bounds work on an [`ICtx`] and never hold a reference to that field
//! across a callback; references to the other fields (vubs, vlbs) may live
//! across one.
//!
//! clang contracts `x * coef + constant` and `1 + coef * coef` in
//! getBestVub/Vlb, `m * c - f` and `-m * a + t` in strengthenVarBound, and
//! `a * b + c * d` in the violation of an implied bound cut.

use super::clique::{CDom, CliqueVar, DomChg, LOWER, REASON_CLIQUE_TABLE, REASON_UNKNOWN, UPPER};
use crate::mip::cuts::round::VarBound;
use crate::mip::cuts::sort::{partition, pdqsort};
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::hash_tree::HighsHashTree;
use std::ffi::c_void;

const INF: f64 = f64::INFINITY;
const K_HIGHS_TINY: f64 = 1e-14;
const REASON_BRANCHING: i32 = -1;

/// HighsSubstitution
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Substitution {
    pub substcol: i32,
    pub staycol: i32,
    pub scale: f64,
    pub offset: f64,
}

#[derive(Clone, Default)]
struct Implics {
    implics: Vec<DomChg>,
    computed: bool,
}

/// The MIP solver's parts the implications use
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CImp {
    pub ctx: *mut c_void,
    pub feastol: f64,
    pub epsilon: f64,
    pub num_nonzero: i32,
    /// nodequeue.numNodesDown/Up(col)
    pub num_nodes_down: unsafe extern "C" fn(*mut c_void, i32) -> i64,
    pub num_nodes_up: unsafe extern "C" fn(*mut c_void, i32) -> i64,
    /// computeImplications' lifting: start recording redundant rows
    /// (if storeLiftingOpportunity is set), and store them for (col, val)
    pub lifting_begin: unsafe extern "C" fn(*mut c_void),
    pub lifting_store: unsafe extern "C" fn(*mut c_void, i32, bool),
    /// globaldom.getDomainChangeReason()[k] as (type, index)
    pub domchg_reason: unsafe extern "C" fn(*mut c_void, i32, *mut i32) -> i32,
    /// globaldom.getChangedCols().size()
    pub changed_cols_len: unsafe extern "C" fn(*mut c_void) -> usize,
    /// globaldom.backtrack(); globaldom.clearChangedCols(changedend)
    pub backtrack: unsafe extern "C" fn(*mut c_void, usize),
    /// cliquetable.vertexInfeasible(globaldom, col, val)
    pub vertex_infeasible: unsafe extern "C" fn(*mut c_void, i32, i32),
    /// pseudocost.addInferenceObservation(col, n, val)
    pub add_inference_observation: unsafe extern "C" fn(*mut c_void, i32, i32, bool),
    /// cliquetable.getNumEntries()
    pub clique_num_entries: unsafe extern "C" fn(*mut c_void) -> i32,
    /// cliquetable.addClique(mipsolver, clique, 2) (changes clique in place)
    pub add_clique2: unsafe extern "C" fn(*mut c_void, *mut CliqueVar),
    /// cliquetable.getSubstitution(col) != nullptr
    pub clique_substituted: unsafe extern "C" fn(*mut c_void, i32) -> bool,
    pub parallel_lock_active: unsafe extern "C" fn(*mut c_void) -> bool,
    /// cliquetable.isFull()
    pub clique_is_full: unsafe extern "C" fn(*mut c_void) -> bool,
    /// &cliquetable.numNeighbourhoodQueries
    pub clique_num_queries: unsafe extern "C" fn(*mut c_void) -> *mut i64,
    /// cliquetable.runCliqueMerging(globaldom)
    pub run_clique_merging: unsafe extern "C" fn(*mut c_void),
    /// mipsolver.mipdata_->numCliqueEntriesAfterFirstPresolve
    pub num_clique_entries_after_first_presolve: unsafe extern "C" fn(*mut c_void) -> i32,
    /// profiling of kMipClockProbingImplications (start if true)
    pub probing_clock: unsafe extern "C" fn(*mut c_void, bool),
    /// cutpool.addCut(mipsolver, inds, vals, len, rhs, integral, propagate,
    /// false)
    pub add_cut: unsafe extern "C" fn(*mut c_void, *mut i32, *mut f64, i32, f64, bool, bool),
}

macro_rules! cb {
    ($m:expr, $f:ident $(, $a:expr)*) => {
        // SAFETY: a callback of the C++ side
        unsafe { ($m.$f)($m.ctx $(, $a)*) }
    };
}

/// std::max / std::min of doubles
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// HighsDomainChange::operator<
fn domchg_less(a: &DomChg, b: &DomChg) -> bool {
    if a.column != b.column {
        return a.column < b.column;
    }
    if a.boundtype != b.boundtype {
        return a.boundtype < b.boundtype;
    }
    a.boundval < b.boundval
}

pub struct Implications {
    next_cleanup_call: i32,
    implications: Vec<Implics>,
    pub num_implications: i64,
    num_var_bounds: i64,
    max_var_bounds: i64,
    vubs: Vec<HighsHashTree<i32, VarBound>>,
    vlbs: Vec<HighsHashTree<i32, VarBound>>,
    pub substitutions: Vec<Substitution>,
    colsubstituted: Vec<u8>,
}

fn calc_max_var_bounds(numcol: i32) -> i64 {
    5000000 + 10 * numcol as i64
}

impl Implications {
    pub fn new(numcol: i32, num_nonzero: i32) -> Self {
        let n = numcol as usize;
        Implications {
            next_cleanup_call: num_nonzero,
            implications: vec![Implics::default(); 2 * n],
            num_implications: 0,
            num_var_bounds: 0,
            max_var_bounds: calc_max_var_bounds(numcol),
            vubs: (0..n).map(|_| HighsHashTree::new()).collect(),
            vlbs: (0..n).map(|_| HighsHashTree::new()).collect(),
            substitutions: Vec::new(),
            colsubstituted: vec![0; n],
        }
    }

    pub fn too_many_var_bounds(&self) -> bool {
        self.num_var_bounds >= self.max_var_bounds
    }

    pub fn implications_cached(&self, col: i32, val: bool) -> bool {
        self.implications[2 * col as usize + val as usize].computed
    }

    /// strengthenVarBound
    pub fn strengthen_var_bound(vbnd: &mut VarBound, multiplier: i32) {
        if vbnd.coef.abs() == INF || vbnd.constant.abs() == INF {
            return;
        }
        const F0MIN: f64 = 0.005;
        const F0MAX: f64 = 0.995;
        let m = multiplier as f64;
        let downrhs = (m * vbnd.constant).floor();
        let f0 = m.mul_add_c(vbnd.constant, -downrhs);
        if f0 < F0MIN || f0 > F0MAX {
            return;
        }
        let mm = (-multiplier) as f64;
        let downaj = mm.mul_add_c(vbnd.coef, K_HIGHS_TINY).floor();
        let fj = mm.mul_add_c(vbnd.coef, -downaj);
        vbnd.constant = m * downrhs;
        vbnd.coef = mm * (downaj + cmax(fj - f0, 0.0) / (1.0 - f0));
    }

    /// addVUB(col, vubcol, coef, constant, colupperbound, colisintegral)
    pub fn add_vub(&mut self, col: i32, vubcol: i32, coef: f64, constant: f64, colub: f64, isint: bool, feastol: f64) {
        if self.too_many_var_bounds() {
            return;
        }
        let mut vub = VarBound { coef, constant };
        if isint {
            Self::strengthen_var_bound(&mut vub, 1);
            if vub.coef == 0.0 {
                return;
            }
        }
        let min_bound = vub.min_value();
        if min_bound >= colub - feastol {
            return;
        }
        let (cur, inserted) = self.vubs[col as usize].insert_or_get(vubcol, vub);
        if !inserted {
            if min_bound < cur.min_value() - feastol {
                *cur = vub;
            }
        } else {
            self.num_var_bounds += 1;
        }
    }

    /// addVLB(col, vlbcol, coef, constant, collowerbound, colisintegral)
    pub fn add_vlb(&mut self, col: i32, vlbcol: i32, coef: f64, constant: f64, collb: f64, isint: bool, feastol: f64) {
        if self.too_many_var_bounds() {
            return;
        }
        let mut vlb = VarBound { coef, constant };
        if isint {
            Self::strengthen_var_bound(&mut vlb, -1);
            if vlb.coef == 0.0 {
                return;
            }
        }
        let max_bound = vlb.max_value();
        if max_bound <= collb + feastol {
            return;
        }
        let (cur, inserted) = self.vlbs[col as usize].insert_or_get(vlbcol, vlb);
        if !inserted {
            if max_bound > cur.max_value() + feastol {
                *cur = vlb;
            }
        } else {
            self.num_var_bounds += 1;
        }
    }

    /// addVUB(col, vubcol, coef, constant) with the global domain's bound
    pub fn add_vub_g(&mut self, g: &CDom, col: i32, vubcol: i32, coef: f64, constant: f64) {
        self.add_vub(col, vubcol, coef, constant, g.upper(col), g.is_integral(col), g.feastol);
    }

    /// addVLB(col, vlbcol, coef, constant) with the global domain's bound
    pub fn add_vlb_g(&mut self, g: &CDom, col: i32, vlbcol: i32, coef: f64, constant: f64) {
        self.add_vlb(col, vlbcol, coef, constant, g.lower(col), g.is_integral(col), g.feastol);
    }

    pub fn column_transformed(&mut self, col: i32, scale: f64, constant: f64) {
        let c = col as usize;
        if scale < 0.0 {
            std::mem::swap(&mut self.vubs[c], &mut self.vlbs[c]);
        }
        let transform = |_: &i32, vbd: &mut VarBound| {
            vbd.constant -= constant;
            vbd.constant /= scale;
            vbd.coef /= scale;
        };
        self.vlbs[c].for_each_mut(transform);
        self.vubs[c].for_each_mut(transform);
        for s in &mut self.substitutions {
            if s.substcol == col {
                s.offset -= constant;
                s.offset /= scale;
                s.scale /= scale;
            }
        }
    }

    /// getBestVub: (column, bound) of the best variable upper bound of col
    /// for the LP solution, -1 if none; updates *best_ub
    pub fn get_best_vub(
        &self,
        imp: &CImp,
        dom: &CDom,
        col: i32,
        col_value: &[f64],
        col_dual: &[f64],
        best_ub: &mut f64,
    ) -> (i32, VarBound) {
        let feastol = imp.feastol;
        let mut best = (-1, VarBound { coef: 0.0, constant: INF });
        let mut minbest_ub = *best_ub;
        let mut best_ub_dist = INF;
        let mut best_nodes: i64 = 0;
        let mut scale = dom.upper(col) - dom.lower(col);
        scale = if scale == INF { 1.0 } else { 1.0 / scale };

        self.vubs[col as usize].for_each(|&vubcol, vub| {
            if vub.coef == INF || dom.is_fixed(vubcol) {
                return;
            }
            let x = col_value[vubcol as usize];
            let vubval = x.mul_add_c(vub.coef, vub.constant);
            let mut ub_dist = cmax(0.0, vubval - col_value[col as usize]);
            let y_dist = feastol + if vub.coef > 0.0 { 1.0 - x } else { x };
            let norm2 = vub.coef.mul_add_c(vub.coef, 1.0);
            if ub_dist * ub_dist > y_dist * y_dist * norm2 {
                return;
            }
            ub_dist *= scale;
            if ub_dist <= best_ub_dist + feastol {
                let minvubval = vub.min_value();
                let nodes = if vub.coef > 0.0 {
                    cb!(imp, num_nodes_down, vubcol)
                } else {
                    cb!(imp, num_nodes_up, vubcol)
                };
                let better = || {
                    if ub_dist < best_ub_dist - feastol {
                        return true;
                    }
                    if nodes > best_nodes {
                        return true;
                    }
                    if nodes < best_nodes {
                        return false;
                    }
                    if minvubval < minbest_ub - feastol {
                        return true;
                    }
                    if minvubval > minbest_ub + feastol {
                        return false;
                    }
                    // (the C++ reads col_dual[-1] when there is no best yet)
                    let best_dual = col_dual.get(best.0 as usize).copied().unwrap_or(0.0);
                    col_dual[vubcol as usize] / vub.coef - best_dual / best.1.coef > feastol
                };
                if better() {
                    *best_ub = vubval;
                    minbest_ub = minvubval;
                    best = (vubcol, *vub);
                    best_nodes = nodes;
                    best_ub_dist = ub_dist;
                }
            }
        });
        best
    }

    /// getBestVlb
    pub fn get_best_vlb(
        &self,
        imp: &CImp,
        dom: &CDom,
        col: i32,
        col_value: &[f64],
        col_dual: &[f64],
        best_lb: &mut f64,
    ) -> (i32, VarBound) {
        let feastol = imp.feastol;
        let mut best = (-1, VarBound { coef: 0.0, constant: -INF });
        let mut maxbest_lb = *best_lb;
        let mut best_lb_dist = INF;
        let mut best_nodes: i64 = 0;
        let mut scale = dom.upper(col) - dom.lower(col);
        scale = if scale == INF { 1.0 } else { 1.0 / scale };

        self.vlbs[col as usize].for_each(|&vlbcol, vlb| {
            if vlb.coef == -INF || dom.is_fixed(vlbcol) {
                return;
            }
            let x = col_value[vlbcol as usize];
            let vlbval = x.mul_add_c(vlb.coef, vlb.constant);
            let mut lb_dist = cmax(0.0, col_value[col as usize] - vlbval);
            let y_dist = feastol + if vlb.coef > 0.0 { x } else { 1.0 - x };
            let norm2 = vlb.coef.mul_add_c(vlb.coef, 1.0);
            if lb_dist * lb_dist > y_dist * y_dist * norm2 {
                return;
            }
            lb_dist *= scale;
            if lb_dist <= best_lb_dist + feastol {
                let maxvlbval = vlb.max_value();
                let nodes = if vlb.coef > 0.0 {
                    cb!(imp, num_nodes_up, vlbcol)
                } else {
                    cb!(imp, num_nodes_down, vlbcol)
                };
                let better = || {
                    if lb_dist < best_lb_dist - feastol {
                        return true;
                    }
                    if nodes > best_nodes {
                        return true;
                    }
                    if nodes < best_nodes {
                        return false;
                    }
                    if maxvlbval > maxbest_lb + feastol {
                        return true;
                    }
                    if maxvlbval < maxbest_lb - feastol {
                        return false;
                    }
                    let best_dual = col_dual.get(best.0 as usize).copied().unwrap_or(0.0);
                    col_dual[vlbcol as usize] / vlb.coef - best_dual / best.1.coef < -feastol
                };
                if better() {
                    *best_lb = vlbval;
                    maxbest_lb = maxvlbval;
                    best = (vlbcol, *vlb);
                    best_nodes = nodes;
                    best_lb_dist = lb_dist;
                }
            }
        });
        best
    }

    /// cleanupVub; changes the bound of col in `g` if allowed
    pub fn cleanup_vub(
        g: &CDom,
        feastol: f64,
        epsilon: f64,
        col: i32,
        vubcol: i32,
        vub: &mut VarBound,
        ub: f64,
        allow_bound_changes: bool,
    ) -> (bool, bool) {
        if vubcol == -1 {
            return (false, false);
        }
        let maxub = CDouble::from(vub.max_value());
        let minub = CDouble::from(vub.min_value());
        if minub >= ub - feastol {
            return (true, false);
        } else if maxub > ub + epsilon {
            let newcoef = (ub - minub).to_f64();
            if vub.coef > 0.0 {
                vub.coef = newcoef;
            } else {
                vub.constant = ub;
                vub.coef = -newcoef;
            }
        } else if allow_bound_changes && maxub < ub - epsilon {
            g.change_bound(UPPER, col, maxub.to_f64(), REASON_UNKNOWN, 0);
            return (false, g.infeasible());
        }
        (false, false)
    }

    /// cleanupVlb
    pub fn cleanup_vlb(
        g: &CDom,
        feastol: f64,
        epsilon: f64,
        col: i32,
        vlbcol: i32,
        vlb: &mut VarBound,
        lb: f64,
        allow_bound_changes: bool,
    ) -> (bool, bool) {
        if vlbcol == -1 {
            return (false, false);
        }
        let maxlb = CDouble::from(vlb.max_value());
        let minlb = CDouble::from(vlb.min_value());
        if maxlb <= lb + feastol {
            return (true, false);
        } else if minlb < lb - epsilon {
            let newcoef = (lb - maxlb).to_f64();
            if vlb.coef < 0.0 {
                vlb.coef = newcoef;
            } else {
                vlb.constant = lb;
                vlb.coef = -newcoef;
            }
        } else if allow_bound_changes && minlb > lb + epsilon {
            g.change_bound(LOWER, col, minlb.to_f64(), REASON_UNKNOWN, 0);
            return (false, g.infeasible());
        }
        (false, false)
    }

    /// rebuild: `transformable[col]` for the reduced columns
    pub fn rebuild(&mut self, g: &CDom, ncols: i32, orig2reducedcol: &[i32], transformable: &[u8]) {
        let oldvubs = std::mem::take(&mut self.vubs);
        let oldvlbs = std::mem::take(&mut self.vlbs);
        let n = ncols as usize;
        self.colsubstituted = vec![0; n];
        self.implications = vec![Implics::default(); 2 * n];
        self.substitutions.clear();
        self.vubs = (0..n).map(|_| HighsHashTree::new()).collect();
        self.vlbs = (0..n).map(|_| HighsHashTree::new()).collect();
        self.num_implications = 0;
        self.num_var_bounds = 0;
        self.next_cleanup_call = g.num_nonzero;

        for i in 0..oldvubs.len() {
            let newi = orig2reducedcol[i];
            if newi == -1 || transformable[newi as usize] == 0 {
                continue;
            }
            let keep = |c: i32| -> Option<i32> {
                let newc = orig2reducedcol[c as usize];
                if newc == -1 || !g.is_binary(newc) || transformable[newc as usize] == 0 {
                    None
                } else {
                    Some(newc)
                }
            };
            oldvubs[i].for_each(|&vubcol, vub| {
                if let Some(c) = keep(vubcol) {
                    self.add_vub_g(g, newi, c, vub.coef, vub.constant);
                }
            });
            oldvlbs[i].for_each(|&vlbcol, vlb| {
                if let Some(c) = keep(vlbcol) {
                    self.add_vlb_g(g, newi, c, vlb.coef, vlb.constant);
                }
            });
        }
    }

    /// buildFrom
    pub fn build_from(&mut self, g: &CDom, init: &Implications) {
        for i in 0..g.num_col {
            init.vubs[i as usize].for_each(|&vubcol, vub| {
                if g.is_binary(vubcol) {
                    self.add_vub_g(g, i, vubcol, vub.coef, vub.constant);
                }
            });
            init.vlbs[i as usize].for_each(|&vlbcol, vlb| {
                if g.is_binary(vlbcol) {
                    self.add_vlb_g(g, i, vlbcol, vlb.coef, vlb.constant);
                }
            });
        }
    }

    /// applyImplications; reads only the `implications` field of `*t`
    /// (may be re-entered from the domain)
    ///
    /// # Safety
    /// `t` a live Implications whose `implications` field nobody holds
    /// mutably
    pub unsafe fn apply_implications(t: *const Implications, dom: &CDom, col: i32, val: i32) {
        let implications = &*std::ptr::addr_of!((*t).implications);
        let im = &implications[2 * col as usize + val as usize];
        if !im.computed {
            return;
        }
        let reason = 2 * col + val;
        for d in &im.implics {
            if dom.is_fixed(d.column) {
                continue;
            }
            let isint = dom.is_integral(d.column);
            if d.boundtype == LOWER {
                if (!isint && d.boundval > dom.upper(d.column) - dom.feastol)
                    || (isint && d.boundval > dom.lower(d.column) + dom.feastol)
                {
                    dom.change_bound(LOWER, d.column, d.boundval, REASON_CLIQUE_TABLE, reason);
                }
            } else if (!isint && d.boundval < dom.lower(d.column) + dom.feastol)
                || (isint && d.boundval < dom.upper(d.column) - dom.feastol)
            {
                dom.change_bound(UPPER, d.column, d.boundval, REASON_CLIQUE_TABLE, reason);
            }
            if dom.infeasible() {
                break;
            }
        }
    }
}

/// The implications behind a raw pointer with the global domain, for the
/// methods that change bounds (see the module comment)
pub struct ICtx {
    t: *mut Implications,
    pub g: CDom,
    pub imp: CImp,
}

impl ICtx {
    /// # Safety
    /// `t` live, `g` and `imp` valid; while the ICtx is used, `*t` is only
    /// reached through it or by re-entrant apply_implications
    pub unsafe fn new(t: *mut Implications, g: CDom, imp: CImp) -> ICtx {
        ICtx { t, g, imp }
    }

    #[inline(always)]
    fn t(&mut self) -> &mut Implications {
        // SAFETY: ICtx::new's contract; borrowed from self
        unsafe { &mut *self.t }
    }

    /// computeImplications: true if infeasible
    fn compute_implications(&mut self, col: i32, val: bool) -> bool {
        let (g, imp) = (self.g, self.imp);
        g.propagate();
        if g.infeasible() || g.is_fixed(col) {
            return true;
        }
        cb!(imp, lifting_begin);
        let changedend = cb!(imp, changed_cols_len);
        let stackimplicstart = g.domchg_len() as i32 + 1;
        let mut num_implications = -stackimplicstart;
        if val {
            g.change_bound(LOWER, col, 1.0, REASON_BRANCHING, 0);
        } else {
            g.change_bound(UPPER, col, 0.0, REASON_BRANCHING, 0);
        }
        let is_infeasible = || {
            if !g.infeasible() {
                return false;
            }
            cb!(imp, lifting_store, col, val);
            cb!(imp, backtrack, changedend);
            cb!(imp, vertex_infeasible, col, val as i32);
            true
        };
        if is_infeasible() {
            return true;
        }
        g.propagate();
        if is_infeasible() {
            return true;
        }
        let stackimplicend = g.domchg_len() as i32;
        num_implications += stackimplicend;
        cb!(imp, add_inference_observation, col, num_implications, val);

        let mut implics: Vec<DomChg> = Vec::with_capacity(num_implications.max(0) as usize);
        let num_entries = cb!(imp, clique_num_entries);
        let max_entries = 100000 + imp.num_nonzero;
        for i in stackimplicstart..stackimplicend {
            let mut index = 0;
            let rtype = cb!(imp, domchg_reason, i, &mut index);
            if rtype == REASON_CLIQUE_TABLE && ((index >> 1) == col || num_entries >= max_entries) {
                continue;
            }
            implics.push(g.domchg(i as usize));
        }
        cb!(imp, lifting_store, col, val);
        cb!(imp, backtrack, changedend);

        let binstart = partition(&mut implics, |a| !g.is_binary(a.column));
        pdqsort(&mut implics[..binstart], domchg_less);

        let mut clique = [CliqueVar::new(col, val as i32), CliqueVar::default()];
        for i in binstart..implics.len() {
            let d = implics[i];
            clique[1] = CliqueVar::new(d.column, if d.boundtype == LOWER { 0 } else { 1 });
            cb!(imp, add_clique2, clique.as_mut_ptr());
            if g.infeasible() || g.is_fixed(col) {
                return true;
            }
        }

        let t = self.t();
        for d in &implics[..binstart] {
            let c = d.column;
            if d.boundtype == LOWER {
                if val {
                    if g.lower(c) != -INF {
                        t.add_vlb_g(&g, c, col, d.boundval - g.lower(c), g.lower(c));
                    }
                } else {
                    t.add_vlb_g(&g, c, col, g.lower(c) - d.boundval, d.boundval);
                }
            } else if val {
                if g.upper(c) != INF {
                    t.add_vub_g(&g, c, col, d.boundval - g.upper(c), g.upper(c));
                }
            } else {
                t.add_vub_g(&g, c, col, g.upper(c) - d.boundval, d.boundval);
            }
        }

        let loc = 2 * col as usize + val as usize;
        t.implications[loc].computed = true;
        implics.truncate(binstart);
        if !implics.is_empty() {
            t.num_implications += implics.len() as i64;
            t.implications[loc].implics = implics;
        }
        false
    }

    /// runProbing
    pub fn run_probing(&mut self, col: i32, num_reductions: &mut i32) -> bool {
        let (g, imp) = (self.g, self.imp);
        if !(g.is_binary(col)
            && !self.t().implications_cached(col, true)
            && !self.t().implications_cached(col, false)
            && !cb!(imp, clique_substituted, col))
        {
            return false;
        }
        for val in [true, false] {
            let infeasible = self.compute_implications(col, val);
            if g.infeasible() || infeasible || cb!(imp, clique_substituted, col) {
                return true;
            }
        }

        let t = self.t();
        let implicsdown = t.implications[2 * col as usize].implics.clone();
        let implicsup = t.implications[2 * col as usize + 1].implics.clone();
        let (nd, nu) = (implicsdown.len(), implicsup.len());
        let (mut u, mut d) = (0, 0);
        while u < nu && d < nd {
            if implicsup[u].column < implicsdown[d].column {
                u += 1;
            } else if implicsdown[d].column < implicsup[u].column {
                d += 1;
            } else {
                let implcol = implicsup[u].column;
                let mut lb_down = g.lower(implcol);
                let mut ub_down = g.upper(implcol);
                let mut lb_up = lb_down;
                let mut ub_up = ub_down;
                loop {
                    if implicsdown[d].boundtype == LOWER {
                        lb_down = cmax(lb_down, implicsdown[d].boundval);
                    } else {
                        ub_down = cmin(ub_down, implicsdown[d].boundval);
                    }
                    d += 1;
                    if !(d < nd && implicsdown[d].column == implcol) {
                        break;
                    }
                }
                loop {
                    if implicsup[u].boundtype == LOWER {
                        lb_up = cmax(lb_up, implicsup[u].boundval);
                    } else {
                        ub_up = cmin(ub_up, implicsup[u].boundval);
                    }
                    u += 1;
                    if !(u < nu && implicsup[u].column == implcol) {
                        break;
                    }
                }
                if self.t().colsubstituted[implcol as usize] != 0 || g.is_fixed(implcol) {
                    continue;
                }
                if lb_down == ub_down && lb_up == ub_up && (lb_down - lb_up).abs() > imp.feastol {
                    let t = self.t();
                    t.substitutions.push(Substitution {
                        substcol: implcol,
                        staycol: col,
                        offset: lb_down,
                        scale: lb_up - lb_down,
                    });
                    t.colsubstituted[implcol as usize] = 1;
                    *num_reductions += 1;
                } else if !cb!(imp, parallel_lock_active) {
                    let lb = cmin(lb_down, lb_up);
                    let ub = cmax(ub_down, ub_up);
                    if lb > g.lower(implcol) {
                        g.change_bound(LOWER, implcol, lb, REASON_UNKNOWN, 0);
                        *num_reductions += 1;
                    }
                    if ub < g.upper(implcol) {
                        g.change_bound(UPPER, implcol, ub, REASON_UNKNOWN, 0);
                        *num_reductions += 1;
                    }
                }
            }
        }
        true
    }

    /// cleanupVarbounds(col)
    pub fn cleanup_varbounds(&mut self, col: i32) {
        let g = self.g;
        let (feastol, epsilon) = (self.imp.feastol, self.imp.epsilon);
        let c = col as usize;
        let ub = g.upper(col);
        let lb = g.lower(col);
        // SAFETY: ICtx::new's contract; only the vubs/vlbs fields are
        // borrowed across the bound changes, which re-enter only
        // apply_implications (the implications field)
        let (vubs, vlbs, nvb) = unsafe {
            (
                &mut *std::ptr::addr_of_mut!((*self.t).vubs),
                &mut *std::ptr::addr_of_mut!((*self.t).vlbs),
                &mut *std::ptr::addr_of_mut!((*self.t).num_var_bounds),
            )
        };
        if ub == lb {
            let mut n = 0i64;
            vubs[c].for_each(|_, _| n += 1);
            vlbs[c].for_each(|_, _| n += 1);
            *nvb -= n;
            vlbs[c].clear();
            vubs[c].clear();
            return;
        }
        let mut del: Vec<i32> = Vec::new();
        vubs[c].for_each_mut(|&vubcol, vub| {
            let (redundant, _) = Implications::cleanup_vub(&g, feastol, epsilon, col, vubcol, vub, ub, true);
            if redundant {
                del.push(vubcol);
            }
        });
        for vubcol in &del {
            vubs[c].erase(vubcol);
        }
        *nvb -= del.len() as i64;
        del.clear();
        vlbs[c].for_each_mut(|&vlbcol, vlb| {
            let (redundant, _) = Implications::cleanup_vlb(&g, feastol, epsilon, col, vlbcol, vlb, lb, true);
            if redundant {
                del.push(vlbcol);
            }
        });
        for vlbcol in &del {
            vlbs[c].erase(vlbcol);
        }
        *nvb -= del.len() as i64;
    }

    /// separateImpliedBounds; `dom` the domain passed in (the global one)
    pub fn separate_implied_bounds(&mut self, dom: &CDom, fracints: &[(i32, f64)], sol: &[f64], feastol: f64, thread_safe: bool) {
        let imp = self.imp;
        let mut numboundchgs = 0;
        if !cb!(imp, clique_is_full) && !thread_safe {
            let nq = cb!(imp, clique_num_queries);
            // SAFETY: the clique table's counter
            let old_num_queries = unsafe { *nq };
            let old_num_entries = cb!(imp, clique_num_entries);
            for &(col, _) in fracints {
                if dom.lower(col) != 0.0
                    || dom.upper(col) != 1.0
                    || (self.t().implications_cached(col, false) && self.t().implications_cached(col, true))
                {
                    continue;
                }
                cb!(imp, probing_clock, true);
                let probing_result = self.run_probing(col, &mut numboundchgs);
                cb!(imp, probing_clock, false);
                if probing_result && dom.infeasible() {
                    return;
                }
                if cb!(imp, clique_is_full) {
                    break;
                }
            }
            let num_new_entries = cb!(imp, clique_num_entries) - old_num_entries;
            self.t().next_cleanup_call -= num_new_entries.max(0);
            if self.t().next_cleanup_call < 0 {
                if !cb!(imp, parallel_lock_active) {
                    cb!(imp, run_clique_merging);
                }
                self.t().next_cleanup_call =
                    cb!(imp, num_clique_entries_after_first_presolve).min(cb!(imp, clique_num_entries));
            }
            if !cb!(imp, parallel_lock_active) {
                // SAFETY: as above
                unsafe { *nq = old_num_queries };
            }
        }

        let mut inds = [0i32; 2];
        let mut vals = [0f64; 2];
        let viol = |inds: &[i32; 2], vals: &[f64; 2], rhs: f64| {
            sol[inds[0] as usize].mul_add_c(vals[0], sol[inds[1] as usize] * vals[1]) - rhs
        };
        for &(col, _) in fracints {
            if dom.lower(col) != 0.0 || dom.upper(col) != 1.0 {
                continue;
            }
            for val in [true, false] {
                if !self.t().implications_cached(col, val) {
                    continue;
                }
                // getImplications(col, val, infeas): cached, so feasible
                if dom.infeasible() {
                    return;
                }
                let implics = self.t().implications[2 * col as usize + val as usize].implics.clone();
                for d in &implics {
                    let c = d.column;
                    let rhs;
                    if d.boundtype == UPPER {
                        if d.boundval + feastol >= dom.upper(c) {
                            continue;
                        }
                        vals[0] = 1.0;
                        inds[0] = c;
                        vals[1] = if val { dom.upper(c) - d.boundval } else { d.boundval - dom.upper(c) };
                        inds[1] = col;
                        rhs = if val { dom.upper(c) } else { d.boundval };
                    } else {
                        if d.boundval - feastol <= dom.lower(c) {
                            continue;
                        }
                        vals[0] = -1.0;
                        inds[0] = c;
                        vals[1] = if val { d.boundval - dom.lower(c) } else { dom.lower(c) - d.boundval };
                        inds[1] = col;
                        rhs = if val { -dom.lower(c) } else { -d.boundval };
                    }
                    if viol(&inds, &vals, rhs) > feastol {
                        cb!(imp, add_cut, inds.as_mut_ptr(), vals.as_mut_ptr(), 2, rhs, dom.is_integral(c), false);
                    }
                }
            }
        }
    }
}

