//! HighsSeparation's shell (highs/mip/HighsSeparation.cpp,
//! HighsSeparationRust.cpp): the separators (tableau, path, mod-k, whose
//! work is cuts/), the cut set, the steps of the Rust separation loop
//! (`CSepaFns`), the separators' host (`Host`: implications, slack bounds,
//! LP rows, the cut pool) and the conflict generation of
//! HighsCutGeneration.

use super::dom::DomS;
use super::lp::LpS;
use super::pools::{CutPoolS, CutSet};
use super::solver::MipSolver;
use super::worker::WorkerS;
use super::INF;
use crate::mip::cuts::ffi as cf;
use crate::mip::cuts::path::PathSeparator;
use crate::mip::cuts::round::{CSepaLp, Host, SepaRound, VarBound};
use crate::mip::cuts::tableau::TableauSeparator;
use crate::mip::separation::{highs_rs_separation_round, highs_rs_separation_separate, CSepaFns};
use std::ffi::c_void;

/// HighsSeparation
pub struct SepaS {
    pub worker: *mut WorkerS,
    pub lp: *mut LpS,
    tableau: *mut TableauSeparator,
    path: *mut PathSeparator,
    /// numCalls and numCutsFound of the tableau, path and mod-k separators
    num_calls: [i32; 3],
    num_cuts_found: [i32; 3],
    cutset: CutSet,
}

// SAFETY: used by its worker's task
unsafe impl Send for SepaS {}

impl Drop for SepaS {
    fn drop(&mut self) {
        // SAFETY: the owned separators
        unsafe {
            cf::highs_rs_tableau_free(self.tableau);
            cf::highs_rs_path_free(self.path);
        }
    }
}

/// The context of the separation loop's steps
struct SepaCtx {
    sepa: *mut SepaS,
    propdomain: *mut DomS,
}

impl SepaS {
    /// HighsSeparation(mipworker)
    pub fn new(worker: *mut WorkerS) -> Box<SepaS> {
        // SAFETY: the live worker
        let seed = unsafe { (*worker).ms().opts.random_seed };
        Box::new(SepaS {
            worker,
            lp: std::ptr::null_mut(),
            tableau: cf::highs_rs_tableau_new(),
            path: cf::highs_rs_path_new(seed as u32),
            num_calls: [0; 3],
            num_cuts_found: [0; 3],
            cutset: CutSet::default(),
        })
    }

    fn fns(&self, ctx: &SepaCtx) -> CSepaFns {
        // SAFETY: the live worker and LP
        let (w, ms) = unsafe { (&*self.worker, (*self.worker).ms()) };
        let d = ms.d();
        CSepaFns {
            op: sepa_op,
            propdomain: ctx.propdomain as *mut c_void,
            rootlpsolobj: d.sc.rootlpsolobj,
            optimality_limit: &w.st().optimality_limit,
            feastol: d.sc.feastol,
        }
    }

    /// separationRound(propdomain, status)
    pub fn separation_round(&mut self, propdomain: &mut DomS, status: &mut i32) -> i32 {
        let ctx = SepaCtx { sepa: self, propdomain };
        let f = self.fns(&ctx);
        // SAFETY: the steps' context and the live LP
        unsafe {
            highs_rs_separation_round(&f, &ctx as *const SepaCtx as *mut c_void, (*self.lp).rs_ptr(), status)
        }
    }

    /// separate(propdomain)
    pub fn separate(&mut self, propdomain: &mut DomS) {
        let ctx = SepaCtx { sepa: self, propdomain };
        let f = self.fns(&ctx);
        // SAFETY: as separation_round
        unsafe { highs_rs_separation_separate(&f, &ctx as *const SepaCtx as *mut c_void, (*self.lp).rs_ptr()) }
    }

    /// The separators' runs on a transformed LP (HighsSeparator::run each)
    fn run_separators(&mut self, lp: &LpS, round: &mut Round, cutpool: &CutPoolS) -> bool {
        // SAFETY: the live worker
        let w = unsafe { &*self.worker };
        for k in 0..3 {
            self.num_calls[k] += 1;
            let curr = cutpool.num_cuts();
            let r = round.rust(cutpool);
            let seed = cutgen_seed(lp, cutpool);
            // SAFETY: the round's live separators and LP
            unsafe {
                match k {
                    0 => {
                        if lp.lph().has_invert() {
                            let d = lp.ms().d();
                            cf::highs_rs_tableau_separate(
                                self.tableau,
                                r,
                                seed,
                                self.num_calls[0],
                                lp.lph().basic_index().as_ptr(),
                                d.sc.total_lp_iterations - d.sc.heuristic_lp_iterations,
                            );
                        }
                    }
                    1 => cf::highs_rs_path_separate(self.path, r, seed),
                    _ => cf::highs_rs_modk_separate(r, seed),
                }
            }
            self.num_cuts_found[k] += cutpool.num_cuts() - curr;
            if w.get_global_domain().infeasible {
                return true;
            }
        }
        false
    }
}

/// HighsCutGeneration's seed: random_seed + LP iterations + cuts in the pool
fn cutgen_seed(lp: &LpS, cutpool: &CutPoolS) -> u32 {
    (lp.ms().opts.random_seed as i64 + lp.num_lp_iterations() + cutpool.num_cuts() as i64) as u32
}

/// The steps of the Rust separation loop (HighsSeparationAccess::op)
unsafe extern "C" fn sepa_op(p: *mut c_void, which: i32, arg: i64) -> i64 {
    let c = &*(p as *const SepaCtx);
    let s = &mut *c.sepa;
    let propdomain = &mut *c.propdomain;
    let lp = &mut *s.lp;
    let w = &*s.worker;
    let ms = lp.ms();
    let d = ms.d();
    let master = std::ptr::eq(propdomain, d.get_domain());
    match which {
        0 => (propdomain.infeasible || w.get_global_domain().infeasible) as i64,
        1 => {
            propdomain.propagate();
            propdomain.infeasible as i64
        }
        2 => {
            // only the master worker modifies the clique table
            if master {
                super::tables::cleanup_fixed(ms, d.get_domain());
            }
            w.get_global_domain().infeasible as i64
        }
        3 => {
            propdomain.clear_changed_cols();
            0
        }
        4 => propdomain.changed_cols().len() as i64,
        5 => {
            lp.set_objective_limit(w.st().upper_limit);
            0
        }
        6 | 7 => {
            if master {
                super::tables::add_root_redcost(ms, lp.col_dual().as_ptr(), lp.objective());
                let ul = if which == 6 { w.st().upper_limit } else { d.sc.upper_limit };
                if ul != INF {
                    super::tables::propagate_root_redcost(ms);
                }
            }
            0
        }
        8 => {
            let lock = d.parallel_lock_active();
            let sol = lp.col_value().to_vec();
            d.implications.separate_implied_bounds(lp, &sol, w.get_cut_pool(), d.sc.feastol, w.get_global_domain(), lock);
            0
        }
        9 => {
            let lock = d.parallel_lock_active();
            let sol = lp.col_value().to_vec();
            let (randgen, nq): (*mut crate::util::random::HighsRandom, *mut i64) = if lock {
                (&mut w.st().randgen, &mut w.st().num_neighbourhood_queries)
            } else {
                (&mut d.cliquetable.randgen, &mut d.cliquetable.num_neighbourhood_queries)
            };
            super::tables::separate_cliques(ms, &sol, w.get_cut_pool(), d.sc.feastol, randgen, nq);
            0
        }
        10 => {
            if !std::ptr::eq(propdomain, w.get_global_domain()) {
                lp.compute_basic_degenerate_duals(
                    d.sc.feastol,
                    propdomain,
                    w.get_global_domain(),
                    w.get_conflict_pool(),
                    w.get_pseudocost(),
                    true,
                );
            }
            0
        }
        11 => {
            let mut round = Round::new(lp, w.get_global_domain());
            if w.get_global_domain().infeasible {
                return 1;
            }
            s.run_separators(lp, &mut round, w.get_cut_pool()) as i64
        }
        12 => {
            let sol = lp.col_value().to_vec();
            w.get_cut_pool().separate(&sol, propdomain, &mut s.cutset, d.sc.feastol, &d.cutpools, false);
            // also the global cut pool
            if !std::ptr::eq(w.get_cut_pool(), d.get_cut_pool()) {
                d.get_cut_pool().separate(&sol, propdomain, &mut s.cutset, d.sc.feastol, &d.cutpools, true);
            }
            s.cutset.num_cuts() as i64
        }
        13 => {
            lp.add_cuts(&mut s.cutset);
            0
        }
        14 => {
            if d.parallel_lock_active() {
                w.st().sepa_lp_iterations += arg;
            } else {
                d.sc.sepa_lp_iterations += arg;
                d.sc.total_lp_iterations += arg;
            }
            0
        }
        _ => {
            w.get_cut_pool().perform_aging();
            0
        }
    }
}

// ---- HighsTransformedLp: a separation round ----

/// The state the separators' callbacks reach (HighsSepaRoundCtx)
struct RoundCtx {
    lp: *const LpS,
    globaldom: *mut DomS,
    ms: *const MipSolver,
    cutpool: *const CutPoolS,
}

/// HighsTransformedLp
struct Round {
    rs: *mut SepaRound,
    ctx: Box<RoundCtx>,
}

impl Drop for Round {
    fn drop(&mut self) {
        // SAFETY: the owned round
        unsafe { cf::highs_rs_sepa_round_free(self.rs) };
    }
}

impl Round {
    /// HighsTransformedLp(lprelaxation, implications, globaldom)
    fn new(lp: &LpS, globaldom: &mut DomS) -> Round {
        let ms = lp.ms();
        let d = ms.d();
        let mut ctx = Box::new(RoundCtx { lp, globaldom, ms, cutpool: std::ptr::null() });
        let h = lp.lph();
        let m = &h.model;
        let sol = h.solution();
        let c = CSepaLp {
            num_col: m.num_col,
            num_row: m.num_row,
            col_lower: globaldom.col_lower().as_ptr(),
            col_upper: globaldom.col_upper().as_ptr(),
            col_value: sol.col_value.as_ptr(),
            row_value: sol.row_value.as_ptr(),
            row_dual: sol.row_dual.as_ptr(),
            row_lower: m.row_lower.as_ptr(),
            row_upper: m.row_upper.as_ptr(),
            a_start: m.a.start.as_ptr(),
            a_index: m.a.index.as_ptr(),
            a_value: m.a.value.as_ptr(),
            integrality: ms.model().integrality.as_ptr(),
            continuous_cols: d.vecs.continuous_cols.as_slice().as_ptr(),
            num_continuous_cols: d.vecs.continuous_cols.len() as i32,
            integral_cols: d.vecs.integral_cols.as_slice().as_ptr(),
            num_integral_cols: d.vecs.integral_cols.len() as i32,
            feastol: d.sc.feastol,
            epsilon: d.sc.epsilon,
            small_matrix_value: ms.opts.small_matrix_value,
            parallel_lock_active: d.parallel_lock_active(),
            mip_pool_soft_limit: ms.opts.mip_pool_soft_limit,
            host: host(&mut *ctx),
            lph: h,
        };
        // SAFETY: the views live during the call
        let rs = unsafe { cf::highs_rs_sepa_round_new(&c) };
        Round { rs, ctx }
    }

    fn rust(&mut self, cutpool: &CutPoolS) -> *mut SepaRound {
        self.ctx.cutpool = cutpool;
        self.rs
    }
}

fn host(ctx: &mut RoundCtx) -> Host {
    Host {
        ctx: ctx as *mut RoundCtx as *mut c_void,
        cleanup_varbounds: h_cleanup_varbounds,
        dom_infeasible: h_dom_infeasible,
        best_vub: h_best_vub,
        best_vlb: h_best_vlb,
        slack_lower: h_slack_lower,
        slack_upper: h_slack_upper,
        get_row: h_get_row,
        add_cut: h_add_cut,
        num_cuts: h_num_cuts,
        num_available_cuts: h_num_available_cuts,
        num_nodes_down: h_num_nodes_down,
        num_nodes_up: h_num_nodes_up,
        num_lp_iterations: h_num_lp_iterations,
    }
}

fn rc<'a>(p: *mut c_void) -> &'a RoundCtx {
    // SAFETY: the round's context
    unsafe { &*(p as *const RoundCtx) }
}

unsafe extern "C" fn h_cleanup_varbounds(p: *mut c_void, col: i32) {
    (*rc(p).ms).d().implications.cleanup_varbounds(col);
}
unsafe extern "C" fn h_dom_infeasible(p: *mut c_void) -> bool {
    (*rc(p).globaldom).infeasible
}
unsafe fn best_vb(p: *mut c_void, vlb: bool, col: i32, bound: *mut f64, vb: *mut VarBound) -> i32 {
    let c = rc(p);
    let lp = &*c.lp;
    let sol = lp.lph().solution();
    let (k, coef, constant) =
        (*c.ms).d().implications.get_best_vb(vlb, col, &sol.col_value, &sol.col_dual, &mut *bound, &mut *c.globaldom);
    (*vb).coef = coef;
    (*vb).constant = constant;
    k
}
unsafe extern "C" fn h_best_vub(p: *mut c_void, col: i32, bound: *mut f64, vb: *mut VarBound) -> i32 {
    best_vb(p, false, col, bound, vb)
}
unsafe extern "C" fn h_best_vlb(p: *mut c_void, col: i32, bound: *mut f64, vb: *mut VarBound) -> i32 {
    best_vb(p, true, col, bound, vb)
}
unsafe extern "C" fn h_slack_lower(p: *mut c_void, row: i32) -> f64 {
    let c = rc(p);
    (*c.lp).slack_lower(row, &*c.globaldom)
}
unsafe extern "C" fn h_slack_upper(p: *mut c_void, row: i32) -> f64 {
    let c = rc(p);
    (*c.lp).slack_upper(row, &*c.globaldom)
}
unsafe extern "C" fn h_get_row(
    p: *mut c_void,
    row: i32,
    len: *mut i32,
    inds: *mut *const i32,
    vals: *mut *const f64,
    integral: *mut u8,
    maxabs: *mut f64,
) {
    let lp = &*rc(p).lp;
    let (i, v) = lp.get_row(row);
    *len = i.len() as i32;
    *inds = i.as_ptr();
    *vals = v.as_ptr();
    *integral = lp.is_row_integral(row) as u8;
    *maxabs = lp.get_max_abs_row_val(row);
}
unsafe extern "C" fn h_add_cut(p: *mut c_void, inds: *mut i32, vals: *mut f64, len: i32, rhs: f64, integral: bool, conflict: bool) -> i32 {
    let c = rc(p);
    let (i, v) = (std::slice::from_raw_parts_mut(inds, len as usize), std::slice::from_raw_parts_mut(vals, len as usize));
    (*c.cutpool).add_cut(&*c.ms, i, v, rhs, integral, true, true, conflict)
}
unsafe extern "C" fn h_num_cuts(p: *mut c_void) -> i32 {
    (*rc(p).cutpool).num_cuts()
}
unsafe extern "C" fn h_num_available_cuts(p: *mut c_void) -> i32 {
    (*rc(p).cutpool).num_available_cuts()
}
unsafe extern "C" fn h_num_nodes_down(p: *mut c_void, col: i32) -> i64 {
    (*rc(p).ms).d().nodequeue.num_nodes_down(col)
}
unsafe extern "C" fn h_num_nodes_up(p: *mut c_void, col: i32) -> i64 {
    (*rc(p).ms).d().nodequeue.num_nodes_up(col)
}
unsafe extern "C" fn h_num_lp_iterations(p: *mut c_void) -> i64 {
    (*rc(p).lp).num_lp_iterations()
}

/// HighsCutGeneration::generateConflict(localdom, globaldom, proof)
pub fn generate_conflict(
    lp: &LpS,
    cutpool: &CutPoolS,
    localdom: &DomS,
    globaldom: &mut DomS,
    inds: &mut [i32],
    vals: &mut [f64],
    rhs: &mut f64,
) -> bool {
    let ms = lp.ms();
    let d = ms.d();
    let mut ctx = RoundCtx { lp, globaldom, ms, cutpool };
    let c = cf::CConflict {
        num_col: ms.num_col(),
        glb: globaldom.col_lower().as_ptr(),
        gub: globaldom.col_upper().as_ptr(),
        llb: localdom.col_lower().as_ptr(),
        lub: localdom.col_upper().as_ptr(),
        integrality: ms.model().integrality.as_ptr(),
        num_lp_cols: lp.num_cols(),
        feastol: d.sc.feastol,
        epsilon: d.sc.epsilon,
        seed: cutgen_seed(lp, cutpool),
        host: host(&mut ctx),
    };
    // SAFETY: the views live during the call
    unsafe { cf::highs_rs_generate_conflict(&c, inds.as_ptr(), vals.as_ptr(), inds.len() as i32, *rhs) }
}
