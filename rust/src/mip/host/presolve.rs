//! The MIP presolve's host (highs/presolve/HPresolveRust.cpp): HPresolve
//! (presolve/hpresolve) runs on the solver's presolved model and postsolve
//! stack, both Rust's ([`Lp`], the lp_presolve.rs [`PostsolveStack`]),
//! and its callbacks reach the solver's domain, clique table,
//! implications and cut pool here. Also the postsolve stack's operations
//! the MIP solver uses (HighsPostsolveStack.h): undoPrimal,
//! getReducedPrimalSolution and the cuts appended to or removed from the
//! model on a restart.

use super::dom::DomS;
use super::solver::MipSolver;
use super::tables::{self, ObjFunc};
use crate::lp_data::lp::Lp;
use crate::lp_data::lp_presolve::PostsolveStack;
use crate::lp_data::lp_run::Solution;
use crate::lp_data::LogType;
use crate::presolve::hpresolve::driver::Input;
use crate::presolve::hpresolve::ffi::{CModel, CSlice, Host, LiftOpp, MipEnv};
use crate::presolve::hpresolve::{MipInfo, Presolve};
use crate::presolve::postsolve::{self, compress_index_maps, Reduction, Stack, Tolerances};
use std::ffi::{c_char, c_void, CStr};

/// HighsPresolveStatus::kNotSet
pub const PS_NOT_SET: i32 = 8;

fn stack_of(s: &PostsolveStack) -> Stack<'_> {
    Stack { data: &s.data, reductions: &s.reductions, orig_col_index: &s.orig_col_index, orig_row_index: &s.orig_row_index }
}

/// undoPrimal(options, solution): the primal solution expanded to the
/// original space and the reductions undone
pub fn undo_primal(ms: &MipSolver, s: &PostsolveStack, sol: &mut Solution) {
    let o = &ms.opts;
    let tol = Tolerances {
        primal_feasibility: o.primal_feasibility_tolerance,
        dual_feasibility: o.dual_feasibility_tolerance,
        mip_feasibility: o.mip_feasibility_tolerance,
    };
    sol.dual_valid = false;
    sol.col_value.resize(s.orig_num_col as usize, 0.0);
    sol.row_value.resize(s.orig_num_row as usize, 0.0);
    let mut x = postsolve::Solution {
        col_value: &mut sol.col_value,
        row_value: &mut sol.row_value,
        col_dual: &mut [],
        row_dual: &mut [],
        dual_valid: false,
    };
    let mut basis = postsolve::Basis { col_status: &mut [], row_status: &mut [], valid: false };
    postsolve::undo(&stack_of(s), &tol, &mut x, &mut basis, 0, -1);
}

/// getReducedPrimalSolution(origPrimalSolution)
pub fn reduced_primal(s: &PostsolveStack, orig: &[f64]) -> Vec<f64> {
    let mut v = orig.to_vec();
    postsolve::reduced_primal_solution(&stack_of(s), &mut v);
    v.truncate(s.orig_col_index.len());
    v
}

/// appendCutsToModel(numCuts)
pub fn append_cuts_to_model(s: &mut PostsolveStack, num_cuts: i32) {
    for _ in 0..num_cuts {
        s.orig_row_index.push(s.orig_num_row);
        s.orig_num_row += 1;
    }
}

/// removeCutsFromModel(numCuts)
pub fn remove_cuts_from_model(s: &mut PostsolveStack, num_cuts: i32) {
    s.orig_num_row -= num_cuts;
    let mut size = s.orig_row_index.len();
    for i in (1..=s.orig_row_index.len()).rev() {
        if s.orig_row_index[i - 1] < s.orig_num_row {
            break;
        }
        size -= 1;
    }
    s.orig_row_index.truncate(size);
}

/// The state of a presolve run (HPresolveRust.cpp's Ctx)
struct Ctx {
    ms: *mut MipSolver,
    ints: Vec<i32>,
    ints2: Vec<i32>,
    lifting: Vec<LiftOpp>,
}

fn cx<'a>(c: *mut c_void) -> &'a mut Ctx {
    // SAFETY: the context of the run
    unsafe { &mut *(c as *mut Ctx) }
}
fn msx<'a>(c: *mut c_void) -> &'a mut MipSolver {
    // SAFETY: the solver of the run
    unsafe { &mut *cx(c).ms }
}

/// A presolve array of n entries
unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    crate::ffi::sl(p, n as i32)
}

extern "C" fn h_log(c: *mut c_void, channel: i32, t: i32, msg: *const c_char) {
    let log = &msx(c).log;
    // SAFETY: a NUL-terminated message
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let t = match t {
        2 => LogType::Detailed,
        3 => LogType::Verbose,
        4 => LogType::Warning,
        5 => LogType::Error,
        _ => LogType::Info,
    };
    match channel {
        0 => log.user(t, &s),
        1 => log.dev(t, &s),
        _ => crate::io::log::c_stdout_flush(s.as_bytes()),
    }
}

extern "C" fn h_timer_read(c: *mut c_void) -> f64 {
    msx(c).timer.read(0)
}

extern "C" fn h_time_string(_: *mut c_void, t: f64, buf: *mut c_char, len: usize) {
    // highsTimeSecondToString
    let s = format!("{}s", t as i32);
    let n = s.len().min(len - 1);
    // SAFETY: the buffer holds len bytes
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
}

extern "C" fn h_sync_model(c: *mut c_void, m: *const CModel, _resize_row_names: bool) {
    // SAFETY: the presolve's arrays of the model's dimensions
    let m = unsafe { &*m };
    let lp = &mut msx(c).d().presolved_model;
    let (nc, nr) = (m.num_col as usize, m.num_row as usize);
    lp.num_col = m.num_col;
    lp.num_row = m.num_row;
    // SAFETY: as above
    unsafe {
        lp.col_cost = sl(m.col_cost, nc).to_vec();
        lp.col_lower = sl(m.col_lower, nc).to_vec();
        lp.col_upper = sl(m.col_upper, nc).to_vec();
        lp.row_lower = sl(m.row_lower, nr).to_vec();
        lp.row_upper = sl(m.row_upper, nr).to_vec();
        lp.integrality = sl(m.integrality, nc).to_vec();
    }
    lp.offset = m.offset;
    lp.sense = if m.maximize { -1 } else { 1 };
}

extern "C" fn h_set_matrix(c: *mut c_void, start: *const i32, ns: usize, index: *const i32, value: *const f64, nnz: usize) {
    let a = &mut msx(c).d().presolved_model.a;
    // SAFETY: the presolve's arrays of these lengths
    unsafe {
        a.start = sl(start, ns).to_vec();
        a.index = sl(index, nnz).to_vec();
        a.value = sl(value, nnz).to_vec();
    }
}

#[allow(clippy::too_many_arguments)]
extern "C" fn h_flush(
    c: *mut c_void,
    bytes: *const u8,
    len: usize,
    types: *const u8,
    pos: *const usize,
    n: usize,
    not_transformable: *const i32,
    nnt: usize,
) {
    let s = &mut msx(c).d().postsolve_stack;
    // SAFETY: the presolve's arrays of these lengths
    unsafe {
        s.data.extend_from_slice(sl(bytes, len));
        let (t, p) = (sl(types, n), sl(pos, n));
        s.reductions.extend(t.iter().zip(p).map(|(&kind, &position)| Reduction { kind, position }));
        for &col in sl(not_transformable, nnt) {
            s.linearly_transformable[col as usize] = 0;
        }
    }
}

extern "C" fn h_shrink(c: *mut c_void, new_col: *const i32, nc: usize, new_row: *const i32, nr: usize) {
    let ms = msx(c);
    let d = ms.d();
    // SAFETY: the presolve's index maps of these lengths
    let (new_col, new_row) = unsafe { (sl(new_col, nc).to_vec(), sl(new_row, nr).to_vec()) };
    let s = &mut d.postsolve_stack;
    let (num_row, num_col) = compress_index_maps(&mut s.orig_row_index, &mut s.orig_col_index, &new_row, &new_col);
    s.orig_row_index.truncate(num_row);
    s.orig_col_index.truncate(num_col);
    d.sc.row_matrix_set = false;
    d.objective_function = ObjFunc::new(ms);
    let fresh = DomS::new(ms);
    d.get_domain().assign(&fresh);
    drop(fresh);
    let lp_num_col = d.presolved_model.num_col;
    tables::rebuild_cliques(ms, lp_num_col, &new_col);
    d.implications.rebuild(lp_num_col, &new_col);
    let (age, soft) = (ms.opts.mip_pool_age_limit, ms.opts.mip_pool_soft_limit);
    let ncol = ms.num_col();
    d.get_cut_pool().replace(ncol, age, soft, 0);
    d.get_conflict_pool().replace(5 * age, soft);
    let lp = &mut d.presolved_model;
    lp.a.num_col = lp.num_col;
    lp.a.num_row = lp.num_row;
}


extern "C" fn h_profiling(c: *mut c_void, start: bool, clock: i32) {
    let p = msx(c).prof;
    let k = if clock == 0 { super::prof::PROBING_PRESOLVE } else { super::prof::ENUMERATION_PRESOLVE };
    if start {
        p.start(k);
    } else {
        p.stop(k);
    }
}

/// the C++ part of prepareProbing; true if infeasible
extern "C" fn h_probing_prepare(c: *mut c_void, _nnz: i32, first_call: *mut bool) -> bool {
    let ms = msx(c);
    let d = ms.d();
    ms.mip_data().setup_domain_propagation();
    let fc = !d.sc.cliques_extracted;
    // SAFETY: the caller's output
    unsafe { *first_call = fc };
    let gd = d.get_domain();
    gd.propagate();
    if gd.infeasible {
        return true;
    }
    if fc {
        d.sc.cliques_extracted = true;
        tables::extract_cliques(ms, true);
        if d.get_domain().infeasible {
            return true;
        }
        if d.sc.upper_limit != f64::INFINITY {
            let tmp = d.sc.upper_limit;
            d.sc.upper_limit = tmp - d.presolved_model.offset;
            tables::extract_obj_cliques(ms);
            d.sc.upper_limit = tmp;
            if d.get_domain().infeasible {
                return true;
            }
        }
        d.get_domain().propagate();
        if d.get_domain().infeasible {
            return true;
        }
    }
    tables::cleanup_fixed(ms, d.get_domain());
    d.get_domain().infeasible
}

extern "C" fn h_mip_env(c: *mut c_void, env: *mut MipEnv) {
    let ms = msx(c);
    let gd = ms.d().get_domain();
    let domain = gd.view();
    let cdom = gd.cdom();
    let cmip = tables::cmip(ms);
    // SAFETY: the caller's output
    unsafe { env.write(MipEnv { domain, cdom, cmip }) };
}

extern "C" fn h_probe(c: *mut c_void, col: i32, num_bound_chgs: *mut i32) -> bool {
    // SAFETY: the caller's output
    msx(c).d().implications.run_probing(col, unsafe { &mut *num_bound_chgs })
}

extern "C" fn h_set_lifting(c: *mut c_void, on: bool) {
    let x = cx(c);
    let imp = &mut msx(c).d().implications;
    if !on {
        imp.store_lifting = std::ptr::null_mut();
        return;
    }
    x.lifting.clear();
    imp.store_lifting = &mut x.lifting;
}

extern "C" fn h_lifting_opps(c: *mut c_void, out: *mut CSlice<LiftOpp>) {
    let x = cx(c);
    // SAFETY: the caller's output
    unsafe { *out = CSlice { ptr: x.lifting.as_ptr(), len: x.lifting.len() } };
}

extern "C" fn h_finalise_begin(c: *mut c_void, first_call: bool, deleted: *mut CSlice<i32>, extensions: *mut CSlice<i32>) {
    let x = cx(c);
    let ms = msx(c);
    let d = ms.d();
    tables::cleanup_fixed(ms, d.get_domain());
    if !first_call {
        tables::extract_cliques(ms, false);
    }
    tables::run_clique_merging(ms, d.get_domain());
    let t = &mut d.cliquetable;
    x.ints = std::mem::take(&mut t.deletedrows);
    x.ints2.clear();
    for e in &t.cliqueextensions {
        x.ints2.push(e.row);
        x.ints2.push(e.var.col());
        x.ints2.push(e.var.val());
    }
    t.cliqueextensions.clear();
    // SAFETY: the caller's outputs
    unsafe {
        *deleted = CSlice { ptr: x.ints.as_ptr(), len: x.ints.len() };
        *extensions = CSlice { ptr: x.ints2.as_ptr(), len: x.ints2.len() };
    }
}

extern "C" fn h_domain_bounds(c: *mut c_void, lower: *mut CSlice<f64>, upper: *mut CSlice<f64>) {
    let gd = msx(c).d().get_domain();
    // SAFETY: the caller's outputs
    unsafe {
        *lower = CSlice { ptr: gd.col_lower().as_ptr(), len: gd.col_lower().len() };
        *upper = CSlice { ptr: gd.col_upper().as_ptr(), len: gd.col_upper().len() };
    }
}

extern "C" fn h_mip_finish_presolve(c: *mut c_void, nnz: i32) {
    let d = msx(c).d();
    d.cliquetable.in_presolve = false;
    d.cliquetable.set_max_entries(nnz);
    let (cp, cfp) = (d.get_cut_pool() as *mut super::pools::CutPoolS, d.get_conflict_pool() as *mut super::pools::ConflictPoolS);
    // SAFETY: the solver's pools
    unsafe {
        d.get_domain().add_cutpool(&mut *cp);
        d.get_domain().add_conflict_pool(&mut *cfp);
    }
}

extern "C" fn h_add_cut(c: *mut c_void, inds: *const i32, vals: *const f64, n: usize, rhs: f64, integral: bool) {
    let ms = msx(c);
    // SAFETY: the cut's arrays
    let (mut i, mut v) = unsafe { (sl(inds, n).to_vec(), sl(vals, n).to_vec()) };
    ms.d().get_cut_pool().add_cut(ms, &mut i, &mut v, rhs, integral, true, false, false);
}

extern "C" fn h_upper_limit(c: *mut c_void) -> f64 {
    msx(c).d().sc.upper_limit
}

extern "C" fn h_set_lower_bound_zero(c: *mut c_void) {
    msx(c).d().sc.lower_bound = 0.0;
}

/// op 303: HPresolve okSetInput(mipsolver, limit) and run(postSolveStack):
/// the model status and the presolve status
pub fn run_mip_presolve(ms: &mut MipSolver, limit: i32) {
    let d = ms.d();
    // okSetInput: the presolved model is a copy of the model, or has the
    // global domain's bounds
    if !std::ptr::eq(ms.model, &d.presolved_model) {
        let m: Lp = ms.model().clone();
        d.presolved_model = m;
        ms.model = &mut d.presolved_model;
    } else {
        d.presolved_model.col_lower = d.get_domain().col_lower().to_vec();
        d.presolved_model.col_upper = d.get_domain().col_upper().to_vec();
    }
    debug_assert!(d.presolved_model.a.format != crate::lp_data::matrix_format::ROWWISE);
    let mut ctx = Ctx { ms, ints: Vec::new(), ints2: Vec::new(), lifting: Vec::new() };
    let host = Host {
        ctx: &mut ctx as *mut Ctx as *mut c_void,
        log: h_log,
        timer_read: h_timer_read,
        time_string: h_time_string,
        sync_model: h_sync_model,
        set_matrix: h_set_matrix,
        flush: h_flush,
        shrink: h_shrink,
        profiling: h_profiling,
        probing_prepare: h_probing_prepare,
        mip_env: h_mip_env,
        probe: h_probe,
        set_lifting: h_set_lifting,
        lifting_opps: h_lifting_opps,
        finalise_begin: h_finalise_begin,
        domain_bounds: h_domain_bounds,
        mip_finish_presolve: h_mip_finish_presolve,
        add_cut: h_add_cut,
        upper_limit: h_upper_limit,
        set_lower_bound_zero: h_set_lower_bound_zero,
    };
    let opts = ms.opts.presolve();
    let mi = MipInfo {
        epsilon: d.sc.epsilon,
        feastol: d.sc.feastol,
        cliquetable: &mut *d.cliquetable,
        implications: d.implications.rs(),
        orig_num_row: ms.orig().num_row,
        num_restarts: d.sc.num_restarts,
        submip: ms.submip,
    };
    let mut p = {
        // the presolve copies its input: the borrows end before the
        // callbacks write the model and stack
        let (lp, s) = (&d.presolved_model, &d.postsolve_stack);
        let input = Input {
            num_col: lp.num_col,
            num_row: lp.num_row,
            col_cost: &lp.col_cost,
            col_lower: &lp.col_lower,
            col_upper: &lp.col_upper,
            row_lower: &lp.row_lower,
            row_upper: &lp.row_upper,
            integrality: &lp.integrality,
            offset: lp.offset,
            maximize: lp.sense == -1,
            a_start: &lp.a.start[..lp.num_col as usize + 1],
            a_index: &lp.a.index,
            a_value: &lp.a.value,
            orig_col_index: &s.orig_col_index,
            orig_row_index: &s.orig_row_index,
            stack_data_size: s.data.len(),
            num_reductions: s.reductions.len(),
            presolve_reduction_limit: limit.max(-1),
            model_name: String::from_utf8_lossy(&lp.model_name).into_owned(),
        };
        Presolve::new(&host, opts, Some(mi), &input)
    };
    let status = p.run();
    let presolve_status = p.presolve_status;
    drop(p);
    ms.modelstatus = status;
    ms.d().presolve_status = presolve_status;
}
