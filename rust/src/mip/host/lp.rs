//! HighsLpRelaxation's shell (highs/mip/HighsLpRelaxation.cpp): the Rust
//! LP relaxation (lp_relaxation.rs, which owns the LP solver), the
//! stored basis shared with the search's nodes, the bound buffers of
//! flushDomain, the first solve's flags and the worker; the calls the
//! relaxation makes back (`CLpFns`: the MIP data, runSolve, the domains,
//! row deletion, incumbents) and the methods the rest of the solver uses.

use super::dom::DomS;
use super::pools::{ConflictPoolS, CutSet};
use super::solver::MipSolver;
use super::tables::PscostS;
use super::worker::WorkerS;
use super::{max2, prof, INF};
use crate::lp_data::lp::Lp;
use crate::lp_data::lp_handle::LpHandle;
use crate::lp_data::lp_run::Basis;
use crate::lp_data::opts::OptValue;
use crate::lp_data::{LogType, Status};
use crate::mip::cutpool::CutPool;
use crate::mip::domain::{DomChg, Reason};
use crate::mip::lp_relaxation::{self as lr, CLpFns, CLpMip, CSolveOut, LpRelax, LpRow};
use crate::mip::search::FracInt;
use std::ffi::c_void;
use std::sync::Arc;

/// The arguments of computeBasicDegenerateDuals' conflict analysis
struct DegenArgs {
    localdom: *mut DomS,
    globaldom: *mut DomS,
    conflictpool: *mut ConflictPoolS,
    pseudocost: *const PscostS,
}

/// HighsLpRelaxation
pub struct LpS {
    pub mipsolver: *mut MipSolver,
    rs: *mut LpRelax,
    col_lb_buffer: Vec<f64>,
    col_ub_buffer: Vec<f64>,
    basischeckpoint: Option<Arc<Basis>>,
    currentbasisstored: bool,
    solved_first_lp: bool,
    race_ipx: bool,
    pub worker: *mut WorkerS,
    cutpool_ptrs: Vec<*mut CutPool>,
    degen_args: Option<DegenArgs>,
}

// SAFETY: as the solver's
unsafe impl Send for LpS {}

impl Drop for LpS {
    fn drop(&mut self) {
        // SAFETY: the owned relaxation
        unsafe { lr::ffi::highs_rs_lprelax_free(self.rs) };
    }
}

fn status_of(s: Status) -> i32 {
    s as i32
}

impl LpS {
    pub fn ms<'a>(&self) -> &'a MipSolver {
        // SAFETY: the solver outlives its LP relaxations
        unsafe { &*self.mipsolver }
    }
    pub fn rs<'a>(&self) -> &'a mut LpRelax {
        // SAFETY: the owned relaxation, used as the C++ handle's
        unsafe { &mut *self.rs }
    }
    pub fn rs_ptr(&self) -> *mut LpRelax {
        self.rs
    }
    /// The LP solver
    pub fn lph<'a>(&self) -> &'a mut LpHandle {
        self.rs().lph()
    }

    /// HighsLpRelaxation(mipsolver)
    pub fn new(ms: &MipSolver) -> Box<LpS> {
        let mut s = Box::new(LpS {
            mipsolver: ms as *const MipSolver as *mut MipSolver,
            rs: std::ptr::null_mut(),
            col_lb_buffer: Vec::new(),
            col_ub_buffer: Vec::new(),
            basischeckpoint: None,
            currentbasisstored: false,
            solved_first_lp: true,
            race_ipx: false,
            worker: std::ptr::null_mut(),
            cutpool_ptrs: Vec::new(),
            degen_args: None,
        });
        let ctx = &mut *s as *mut LpS as *mut c_void;
        s.rs = lr::ffi::highs_rs_lprelax_new(&LP_FNS, ctx);
        let o = &ms.opts;
        s.set_option("output_flag", OptValue::Bool(false));
        s.set_option("random_seed", OptValue::Int(o.random_seed));
        s.set_option("primal_feasibility_tolerance", OptValue::Double(o.mip_feasibility_tolerance));
        s.set_option("dual_feasibility_tolerance", OptValue::Double(o.mip_feasibility_tolerance * 0.1));
        s.set_option("simplex_dse_exact_init_max_rows", OptValue::Int(20000));
        s.set_option("simplex_keep_random_vectors", OptValue::Bool(true));
        s.set_option("full_lp_kkt_check", OptValue::Bool(false));
        s
    }

    /// The copy constructor (no worker, no profiling)
    pub fn copy(other: &LpS) -> Box<LpS> {
        let mut s = Box::new(LpS {
            mipsolver: other.mipsolver,
            rs: std::ptr::null_mut(),
            col_lb_buffer: Vec::new(),
            col_ub_buffer: Vec::new(),
            basischeckpoint: other.basischeckpoint.clone(),
            currentbasisstored: other.currentbasisstored,
            solved_first_lp: true,
            race_ipx: false,
            worker: std::ptr::null_mut(),
            cutpool_ptrs: Vec::new(),
            degen_args: None,
        });
        let ctx = &mut *s as *mut LpS as *mut c_void;
        // SAFETY: the other relaxation is live
        s.rs = unsafe { lr::ffi::highs_rs_lprelax_copy(other.rs, ctx) };
        s.set_option("output_flag", OptValue::Bool(false));
        let opts = other.lph().opts.clone();
        s.lph().pass_options(&opts);
        let model = other.lph().model.clone();
        s.lph().pass_model(model);
        let basis = other.get_lp_basis();
        s.set_lp_basis(basis, "");
        let n = s.ms().num_col() as usize;
        s.col_lb_buffer.resize(n, 0.0);
        s.col_ub_buffer.resize(n, 0.0);
        s
    }

    pub fn set_option(&self, name: &str, v: OptValue) {
        let ok = self.lph().set_option(name, v);
        debug_assert!(ok, "LP option {name}");
    }
    pub fn set_profiling(&self, p: *mut c_void) {
        self.lph().profiling = p;
    }
    pub fn set_mip_worker(&mut self, w: *mut WorkerS) {
        self.worker = w;
    }

    /// loadModel: the model with the global domain's bounds (the worker's
    /// if any), no offset, no integrality
    pub fn load_model(&mut self) {
        let ms = self.ms();
        let mut lpmodel: Lp = ms.model().clone();
        let gd: &DomS =
            // SAFETY: the worker's global domain, live
            if self.worker.is_null() { ms.d().get_domain() } else { unsafe { (*self.worker).get_global_domain() } };
        lpmodel.col_lower = gd.col_lower().to_vec();
        lpmodel.col_upper = gd.col_upper().to_vec();
        lpmodel.offset = 0.0;
        // SAFETY: the owned relaxation
        unsafe { lr::ffi::highs_rs_lprelax_op(self.rs, 0, lpmodel.num_row) };
        lpmodel.integrality.clear();
        let num_col = lpmodel.num_col as usize;
        self.lph().clear_solver();
        self.lph().clear_model();
        lpmodel.is_moved = false;
        self.lph().pass_model(lpmodel);
        self.col_lb_buffer.resize(num_col, 0.0);
        self.col_ub_buffer.resize(num_col, 0.0);
    }

    // ---- the shared state ----

    pub fn status(&self) -> i32 {
        self.rs().sh.status
    }
    pub fn objective(&self) -> f64 {
        self.rs().sh.objective
    }
    pub fn num_lp_iterations(&self) -> i64 {
        self.rs().sh.numlpiters
    }
    pub fn avg_solve_iters(&self) -> f64 {
        self.rs().sh.avg_solve_iters
    }
    /// getFractionalIntegers (valid until the next solve)
    #[allow(clippy::mut_from_ref)]
    pub fn frac(&self) -> &mut [FracInt] {
        let sh = &self.rs().sh;
        // SAFETY: the relaxation's fractional integers
        unsafe { crate::ffi::sl_mut(sh.frac, sh.num_frac) }
    }
    pub fn rows(&self) -> &[LpRow] {
        let sh = &self.rs().sh;
        // SAFETY: the relaxation's rows
        unsafe { crate::ffi::sl(sh.rows, sh.num_rows) }
    }
    pub fn col_value(&self) -> &[f64] {
        &self.lph().solution().col_value
    }
    pub fn col_dual(&self) -> &[f64] {
        &self.lph().solution().col_dual
    }
    pub fn num_rows(&self) -> i32 {
        self.lph().model.num_row
    }
    pub fn num_cols(&self) -> i32 {
        self.lph().model.num_col
    }
    pub fn lp_model_status(&self) -> i32 {
        self.lph().model_status()
    }
    pub fn lp_basis_valid(&self) -> bool {
        self.lph().basis().b.valid
    }
    pub fn get_lp_basis(&self) -> Basis {
        self.lph().basis().clone()
    }
    pub fn set_lp_basis(&self, basis: Basis, origin: &str) -> Status {
        self.lph().set_basis(basis, origin)
    }
    pub fn set_adjust_symmetric_branching_col(&self, adjust: bool) {
        self.rs().sh.adjust_sym = adjust;
    }
    pub fn integer_feasible(&self) -> bool {
        let s = self.status();
        (s == lr::OPTIMAL || s == lr::UNSCALED_PRIMAL_FEASIBLE) && self.rs().sh.num_frac == 0
    }

    /// isBasisConsistent(basis)
    pub fn is_basis_consistent(&self, basis: &Basis) -> bool {
        let m = &self.lph().model;
        if basis.b.col_status.len() != m.num_col as usize || basis.b.row_status.len() != m.num_row as usize {
            return false;
        }
        let num_basic = basis.b.col_status.iter().chain(basis.b.row_status.iter()).filter(|&&s| s == 1).count();
        num_basic == m.num_row as usize
    }

    // ---- the rows ----

    /// LpRow::get: (indices, values)
    pub fn get_row(&self, row: i32) -> (&[i32], &[f64]) {
        let r = self.rows()[row as usize];
        let d = self.ms().d();
        if r.origin == lr::ROW_CUT {
            d.cutpools[r.cutpoolindex as usize].get_cut(r.index)
        } else {
            d.get_row(r.index)
        }
    }
    pub fn is_row_integral(&self, row: i32) -> bool {
        let r = self.rows()[row as usize];
        let d = self.ms().d();
        if r.origin == lr::ROW_CUT {
            d.cutpools[r.cutpoolindex as usize].cut_is_integral(r.index)
        } else {
            d.vecs.row_integral[r.index as usize] != 0
        }
    }
    pub fn get_max_abs_row_val(&self, row: i32) -> f64 {
        let r = self.rows()[row as usize];
        let d = self.ms().d();
        if r.origin == lr::ROW_CUT {
            d.cutpools[r.cutpoolindex as usize].max_abs_cut_coef(r.index)
        } else {
            d.vecs.max_abs_row_coef[r.index as usize]
        }
    }
    pub fn row_lower(&self, row: i32) -> f64 {
        self.lph().model.row_lower[row as usize]
    }
    pub fn row_upper(&self, row: i32) -> f64 {
        self.lph().model.row_upper[row as usize]
    }
    /// slackLower(row, globaldom)
    pub fn slack_lower(&self, row: i32, globaldom: &DomS) -> f64 {
        let r = self.rows()[row as usize];
        if r.origin == lr::ROW_CUT {
            let cp = &*self.ms().d().cutpools[r.cutpoolindex as usize] as *const _;
            return globaldom.get_min_cut_activity(cp, r.index);
        }
        let rowlower = self.row_lower(row);
        if rowlower != -INF {
            return rowlower;
        }
        globaldom.get_min_activity(r.index)
    }
    /// slackUpper(row, globaldom)
    pub fn slack_upper(&self, row: i32, globaldom: &DomS) -> f64 {
        let rowupper = self.row_upper(row);
        let r = self.rows()[row as usize];
        if r.origin == lr::ROW_CUT {
            return rowupper;
        }
        if rowupper != INF {
            return rowupper;
        }
        globaldom.get_max_activity(r.index)
    }

    // ---- the relaxation's operations ----

    /// computeBasicDegenerateDuals
    pub fn compute_basic_degenerate_duals(
        &mut self,
        threshold: f64,
        localdom: &mut DomS,
        globaldom: &mut DomS,
        conflictpool: &mut ConflictPoolS,
        pseudocost: &PscostS,
        getdualproof: bool,
    ) {
        // the reconvergence callback reads the arguments through the
        // relaxation's context pointer, which `&mut self`'s noalias does not
        // see: volatile, or the stores are dead to the optimiser
        let args: *mut Option<DegenArgs> = &mut self.degen_args;
        // SAFETY: a field of self; the owned relaxation
        unsafe {
            args.write_volatile(Some(DegenArgs { localdom, globaldom, conflictpool, pseudocost }));
            lr::ffi::highs_rs_lprelax_degenerate_duals(self.rs, threshold, getdualproof);
            args.write_volatile(None);
        }
    }
    pub fn compute_best_estimate(&self, ps: &PscostS) -> f64 {
        // SAFETY: live relaxation and pseudocosts
        unsafe { lr::ffi::highs_rs_lprelax_best_estimate(self.rs, ps.rs) }
    }
    pub fn compute_lp_degeneracy(&self, localdom: &DomS) -> f64 {
        let (l, u) = (localdom.col_lower(), localdom.col_upper());
        // SAFETY: the domain's bounds
        unsafe { lr::ffi::highs_rs_lprelax_degeneracy(self.rs, l.as_ptr(), u.as_ptr(), l.len() as i32) }
    }
    /// addCuts(cutset)
    pub fn add_cuts(&mut self, cutset: &mut CutSet) {
        let numcuts = cutset.num_cuts();
        if numcuts > 0 {
            self.currentbasisstored = false;
            self.basischeckpoint = None;
            // SAFETY: the cut set's arrays
            unsafe {
                lr::ffi::highs_rs_lprelax_add_cuts(
                    self.rs,
                    cutset.cutindices.as_ptr(),
                    cutset.cutpools.as_ptr(),
                    numcuts as i32,
                )
            };
            let st = self.lph().add_rows(
                numcuts as i32,
                &cutset.lower,
                &cutset.upper,
                cutset.ar_value.len() as i32,
                &cutset.ar_start,
                &cutset.ar_index,
                &cutset.ar_value,
            );
            debug_assert!(st == Status::Ok);
            cutset.clear();
        }
    }
    fn op(&self, which: i32, i: i32) {
        // SAFETY: the owned relaxation
        unsafe { lr::ffi::highs_rs_lprelax_op(self.rs, which, i) };
    }
    pub fn remove_obsolete_rows(&self, notify_pool: bool) {
        self.op(1, notify_pool as i32);
    }
    pub fn remove_worker_specific_rows(&self) {
        self.op(2, 0);
    }
    pub fn perform_aging(&self, delete_rows: bool) {
        self.op(4, delete_rows as i32);
    }
    pub fn reset_ages(&self) {
        self.op(5, 0);
    }
    pub fn notify_cut_pools_lp_copied(&self, n: i32) {
        self.op(6, n);
    }
    /// computeDualProof(globaldom, upperbound, extractCliques): (inds,
    /// vals, rhs)
    pub fn compute_dual_proof(
        &self,
        globaldom: &DomS,
        upperbound: f64,
        extract_cliques: bool,
    ) -> Option<(Vec<i32>, Vec<f64>, f64)> {
        let (glb, gub) = (globaldom.col_lower(), globaldom.col_upper());
        let (mut inds, mut vals, mut len, mut rhs) = (std::ptr::null(), std::ptr::null(), 0, 0.0);
        // SAFETY: the domain's bounds; the proof's arrays read at once
        unsafe {
            if !lr::ffi::highs_rs_lprelax_dual_proof(
                self.rs,
                globaldom as *const DomS as *const c_void,
                glb.as_ptr(),
                gub.as_ptr(),
                glb.len() as i32,
                upperbound,
                extract_cliques,
                &mut inds,
                &mut vals,
                &mut len,
                &mut rhs,
            ) {
                return None;
            }
            Some((crate::ffi::sl(inds, len).to_vec(), crate::ffi::sl(vals, len).to_vec(), rhs))
        }
    }
    /// computeDualInfProof: the proof of the last infeasible solve
    pub fn compute_dual_inf_proof(&self) -> Option<(Vec<i32>, Vec<f64>, f64)> {
        let sh = &self.rs().sh;
        if !sh.has_proof {
            return None;
        }
        // SAFETY: the relaxation's proof arrays
        unsafe {
            Some((
                crate::ffi::sl(sh.proof_inds, sh.proof_len).to_vec(),
                crate::ffi::sl(sh.proof_vals, sh.proof_len).to_vec(),
                sh.proof_rhs,
            ))
        }
    }
    pub fn run(&mut self, resolve_on_error: bool) -> i32 {
        // SAFETY: the owned relaxation
        unsafe { lr::ffi::highs_rs_lprelax_run(self.rs, resolve_on_error) }
    }
    /// resolveLp(domain or null)
    pub fn resolve_lp(&mut self, domain: Option<&mut DomS>) -> i32 {
        let d = domain.map_or(std::ptr::null_mut(), |d| d as *mut DomS as *mut c_void);
        // SAFETY: the owned relaxation and the live domain
        unsafe { lr::ffi::highs_rs_lprelax_resolve(self.rs, d) }
    }

    /// flushDomain(domain, continuous)
    pub fn flush_domain(&mut self, domain: &mut DomS, mut continuous: bool) {
        if !domain.changed_cols().is_empty() {
            if std::ptr::eq(domain, self.ms().d().get_domain()) {
                continuous = true;
            }
            self.currentbasisstored = false;
            if !continuous {
                domain.remove_continuous_changed_cols();
            }
            let n = domain.changed_cols().len();
            if n == 0 {
                return;
            }
            for (i, &col) in domain.changed_cols().iter().enumerate() {
                self.col_lb_buffer[i] = domain.col_lower()[col as usize];
                self.col_ub_buffer[i] = domain.col_upper()[col as usize];
            }
            let cols: Vec<i32> = domain.changed_cols().to_vec();
            self.lph().change_col_bounds_set(&cols, &self.col_lb_buffer[..n], &self.col_ub_buffer[..n]);
            domain.clear_changed_cols();
        }
    }

    /// resetToGlobalDomain(globaldom)
    pub fn reset_to_global_domain(&self, globaldom: &DomS) {
        let n = self.ms().num_col();
        self.lph().change_col_bounds_interval(0, n - 1, globaldom.col_lower(), globaldom.col_upper());
    }

    /// recoverBasis
    pub fn recover_basis(&mut self) {
        if let Some(b) = &self.basischeckpoint {
            let b = (**b).clone();
            self.set_lp_basis(b, "HighsLpRelaxation::recoverBasis");
            self.currentbasisstored = true;
        }
    }

    /// setObjectiveLimit(objlim)
    pub fn set_objective_limit(&self, objlim: f64) {
        let d = self.ms().d();
        let f = &d.objective_function;
        let offset =
            if f.is_integral() { 0.5 / f.integral_scale() } else { max2(1000.0 * d.sc.feastol, objlim.abs() * 1e-14) };
        self.set_option("objective_bound", OptValue::Double(objlim + offset));
    }

    pub fn set_iteration_limit(&self, limit: i32) {
        self.set_option("simplex_iteration_limit", OptValue::Int(limit));
    }
    pub fn set_solved_first_lp(&mut self, v: bool) {
        self.solved_first_lp = v;
    }
    pub fn set_race_ipx(&mut self, race: bool) {
        self.race_ipx = race;
    }

    /// storeBasis
    pub fn store_basis(&mut self) {
        if !self.currentbasisstored && self.lp_basis_valid() {
            self.basischeckpoint = Some(Arc::new(self.get_lp_basis()));
            self.currentbasisstored = true;
        }
    }
    pub fn get_stored_basis(&self) -> Option<Arc<Basis>> {
        self.basischeckpoint.clone()
    }
    pub fn set_stored_basis(&mut self, basis: Option<Arc<Basis>>) {
        self.basischeckpoint = basis;
        self.currentbasisstored = false;
    }

    /// getLpCopy: the LP solver's model (with the MIP model's names,
    /// which the C++ copied; the Rust model has none)
    pub fn get_lp_copy(&self) -> Lp {
        let mut lp = self.lph().model.clone();
        lp.is_moved = false;
        lp
    }

    /// getCutPool: (num_col, num_cut, lower, upper, start, index, value)
    #[allow(clippy::type_complexity)]
    pub fn get_cut_pool(&self) -> (i32, i32, Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<f64>) {
        let m = &self.lph().model;
        let num_lp_row = m.num_row as usize;
        let num_model_row = self.ms().num_row() as usize;
        let num_cut = num_lp_row - num_model_row;
        let mut lower = vec![0.0; num_cut];
        let mut upper = vec![0.0; num_cut];
        let mut cut_row_index = vec![-1i32; num_lp_row];
        let mut cut_num = 0usize;
        let rows = self.rows();
        for r in 0..num_lp_row {
            if rows[r].origin != lr::ROW_CUT {
                continue;
            }
            cut_row_index[r] = cut_num as i32;
            lower[cut_num] = m.row_lower[r];
            upper[cut_num] = m.row_upper[r];
            cut_num += 1;
        }
        let mut length = vec![0i32; num_cut];
        for c in 0..m.num_col as usize {
            for el in m.a.start[c] as usize..m.a.start[c + 1] as usize {
                let k = cut_row_index[m.a.index[el] as usize];
                if k >= 0 {
                    length[k as usize] += 1;
                }
            }
        }
        let mut start = vec![0i32; num_cut + 1];
        let mut nz = 0;
        for k in 0..num_cut {
            let l = length[k];
            length[k] = start[k];
            nz += l;
            start[k + 1] = nz;
        }
        let mut index = vec![0i32; nz as usize];
        let mut value = vec![0.0; nz as usize];
        for c in 0..m.num_col as usize {
            for el in m.a.start[c] as usize..m.a.start[c + 1] as usize {
                let k = cut_row_index[m.a.index[el] as usize];
                if k >= 0 {
                    let p = length[k as usize] as usize;
                    index[p] = c as i32;
                    value[p] = m.a.value[el];
                    length[k as usize] += 1;
                }
            }
        }
        (m.num_col, num_cut as i32, lower, upper, start, index, value)
    }

    /// runSolve(use_simplex, extraIterations): the call status
    fn run_solve(&mut self, out: &mut CSolveOut) {
        let ms = self.ms();
        let o = &ms.opts;
        let this_time_limit = max2(self.lph().run_time() + o.time_limit - ms.timer.read(0), 0.0);
        self.set_option("time_limit", OptValue::Double(this_time_limit));
        let valid_basis = self.lp_basis_valid();
        let mip_timing = ms.prof.mip() && !ms.submip && !self.solved_first_lp;
        if mip_timing {
            crate::log_user!(
                ms.log,
                LogType::Info,
                "MIP-Timing: %11.2g - start first LP solve (with%s basis)\n",
                ms.timer.read(0),
                if valid_basis { "" } else { "out" }
            );
        }
        let solver = match self.lph().opts.get("solver") {
            Some(OptValue::Str(s)) => s.to_vec(),
            _ => b"choose".to_vec(),
        };
        let use_solver: &[u8] = if valid_basis {
            b"simplex"
        } else if crate::lp_data::run::use_ipm(&o.mip_lp_solver) {
            if o.mip_lp_solver.as_slice() == b"hipo" {
                b"hipo"
            } else {
                b"ipx"
            }
        } else {
            b"simplex"
        };
        self.set_option("solver", OptValue::Str(use_solver));
        let use_ipm = crate::lp_data::run::use_ipm(use_solver);
        let mut use_simplex = !use_ipm;
        let mut callstatus = Status::Ok;
        if use_ipm {
            ms.prof.solve_call(1, ms.submip);
            callstatus = self.lph().optimize_lp();
            if callstatus == Status::Error {
                crate::log_dev!(
                    ms.log,
                    LogType::Info,
                    "HighsLpRelaxation::run HiPO has failed : status = %s Try IPX\n",
                    crate::mip::driver::model_status_to_string(self.lp_model_status())
                );
                self.set_option("solver", OptValue::Str(b"simplex"));
                use_simplex = true;
            }
        }
        if use_simplex {
            let profiling_submip = ms.prof.is_submip();
            ms.prof.set_submip(ms.submip);
            if ms.prof.running(prof::SUB_SOLVER_SUB_MIP) {
                crate::io::log::c_stdout_flush(
                    format!(
                        "HighsLpRelaxation::run Sub-MIP sub-solver clock running on thread {:2} and this is {}MIP\n",
                        ms.prof.my_thread(),
                        if ms.submip { "sub-" } else { "" }
                    )
                    .as_bytes(),
                );
            }
            ms.prof.set_submip(profiling_submip);
            ms.prof.solve_call(2, ms.submip);
            if self.race_ipx && !valid_basis {
                let (s, ipx_won) = self.lph().optimize_racing_ipx(o.random_seed, &mut out.extra_iterations);
                callstatus = s;
                crate::log_user!(
                    ms.log,
                    LogType::Info,
                    "Root LP: %s won the race on two threads\n",
                    if ipx_won { "IPX" } else { "the dual simplex" }
                );
            } else {
                callstatus = self.lph().optimize_lp();
            }
        }
        self.set_option("solver", OptValue::Str(&solver));
        if mip_timing {
            crate::log_user!(ms.log, LogType::Info, "MIP-Timing: %11.2g - finish first LP solve\n", ms.timer.read(0));
        }
        self.solved_first_lp = true;
        out.use_simplex = use_simplex;
        out.callstatus = status_of(callstatus);
    }

    /// ipmBasisAfterIterationLimit
    fn ipm_basis_after_iteration_limit(&self) {
        let ms = self.ms();
        ms.prof.solve_call(3, ms.submip);
        self.lph().ipm_basis_after_iteration_limit(!ms.opts.mip_root_presolve_only, ms.prof.p);
    }

    /// The LP relaxation's handle P of glue.rs
    pub fn p(&self) -> *mut c_void {
        self as *const LpS as *mut c_void
    }
}

// ---- the Rust relaxation's calls back (HighsLpRelaxationAccess) ----

fn lp<'a>(p: *mut c_void) -> &'a mut LpS {
    // SAFETY: the LpS the relaxation was made with
    unsafe { &mut *(p as *mut LpS) }
}
fn dom<'a>(p: *mut c_void) -> &'a mut DomS {
    // SAFETY: a DomS handed to the relaxation
    unsafe { &mut *(p as *mut DomS) }
}

unsafe extern "C" fn lp_mip(p: *mut c_void, m: *mut CLpMip) {
    let x = lp(p);
    let ms = &*x.mipsolver;
    let d = ms.d();
    let model = ms.model();
    x.cutpool_ptrs.clear();
    for c in &d.cutpools {
        x.cutpool_ptrs.push(c.rs());
    }
    let use_worker = !x.worker.is_null() && d.parallel_lock_active();
    let g: &DomS = if use_worker { (*x.worker).get_global_domain() } else { d.get_domain() };
    m.write(CLpMip {
        num_model_row: ms.num_row(),
        num_col: ms.num_col(),
        integral_cols: d.vecs.integral_cols.as_slice().as_ptr(),
        n_integral_cols: d.vecs.integral_cols.len() as i32,
        feastol: d.sc.feastol,
        epsilon: d.sc.epsilon,
        small_matrix_value: ms.opts.small_matrix_value,
        uplocks: d.vecs.uplocks.as_slice().as_ptr(),
        downlocks: d.vecs.downlocks.as_slice().as_ptr(),
        col_cost: model.col_cost.as_ptr(),
        integrality: model.integrality.as_ptr(),
        model_a_start: model.a.start.as_ptr(),
        ar_start: d.vecs.ar_start.as_slice().as_ptr(),
        ar_index: d.vecs.ar_index.as_slice().as_ptr(),
        ar_value: d.vecs.ar_value.as_slice().as_ptr(),
        n_ar: d.vecs.ar_index.len() as i32,
        row_integral: d.vecs.row_integral.as_slice().as_ptr(),
        max_abs_row_coef: d.vecs.max_abs_row_coef.as_slice().as_ptr(),
        age_limit: ms.opts.mip_lp_age_limit,
        parallel_lock: d.parallel_lock_active(),
        submip: ms.submip,
        has_orbitopes: !d.symmetries.column_to_orbitope.is_empty(),
        cutpools: x.cutpool_ptrs.as_ptr(),
        n_cutpools: x.cutpool_ptrs.len() as i32,
        cliquetable: &*d.cliquetable,
        global_dom: g as *const DomS as *const c_void,
        glb: g.col_lower().as_ptr(),
        gub: g.col_upper().as_ptr(),
        upper_limit: if use_worker { (*x.worker).st().upper_limit } else { d.sc.upper_limit },
        source_solve_lp: crate::mip::glue::source::SOLVE_LP,
        source_unbounded: crate::mip::glue::source::UNBOUNDED,
    });
}

unsafe extern "C" fn lp_solve(p: *mut c_void, o: *mut CSolveOut) {
    lp(p).run_solve(&mut *o);
}

unsafe extern "C" fn lp_op(p: *mut c_void, which: i32) {
    let x = lp(p);
    let ms = &*x.mipsolver;
    match which {
        3 => x.recover_basis(),
        4 => x.ipm_basis_after_iteration_limit(),
        5 => crate::log_user!(
            ms.log,
            LogType::Warning,
            "HighsLpRelaxation::run LP is unbounded with no basis, but not returning Status::kError\n"
        ),
        6 => {
            let col_value = x.col_value().to_vec();
            let src = crate::mip::glue::source::UNBOUNDED;
            if !ms.d().parallel_lock_active() || x.worker.is_null() {
                super::fns::try_solution(ms, &col_value, src);
            } else {
                (*x.worker).try_solution(&col_value, src);
            }
        }
        7 => crate::log_user!(
            ms.log,
            LogType::Warning,
            "LP solved to unexpected status: %s\n",
            crate::mip::driver::model_status_to_string(x.lp_model_status())
        ),
        8 => crate::log_dev!(ms.log, LogType::Verbose, "no dual ray stored\n"),
        _ => {
            let mut root_basis = ms.d().firstrootbasis.clone();
            let n = x.num_rows() as usize;
            root_basis.b.row_status.resize(n, 1);
            x.set_lp_basis(root_basis, "");
        }
    }
}

unsafe extern "C" fn lp_delete_rows(p: *mut c_void, mask: *mut i32) {
    let x = lp(p);
    let ms = &*x.mipsolver;
    let mut basis = x.get_lp_basis();
    let nlprows = x.num_rows() as usize;
    let n = nlprows;
    let mask_s = std::slice::from_raw_parts_mut(mask, n);
    x.lph().delete_rows_mask(mask_s);
    let mut ndelcuts = 0;
    for i in ms.num_row() as usize..nlprows {
        if mask_s[i] >= 0 {
            basis.b.row_status[mask_s[i] as usize] = basis.b.row_status[i];
        } else {
            ndelcuts += 1;
        }
    }
    let len = basis.b.row_status.len() - ndelcuts;
    basis.b.row_status.truncate(len);
    basis.origin = "HighsLpRelaxation::removeCuts".to_string();
    x.set_lp_basis(basis, "");
    ms.prof.solve_call(0, ms.submip);
    x.lph().optimize_lp();
}

unsafe extern "C" fn lp_tighten(d: *const c_void, inds: *mut i32, vals: *mut f64, len: i32, rhs: *mut f64) {
    let d = &*(d as *const DomS);
    d.tighten_coefficients(crate::ffi::sl(inds, len), crate::ffi::sl_mut(vals, len), &mut *rhs);
}

unsafe extern "C" fn lp_extract_cliques(p: *mut c_void, inds: *const i32, vals: *const f64, len: i32, rhs: f64) {
    let ms = &*lp(p).mipsolver;
    super::tables::extract_cliques_from_cut(ms, crate::ffi::sl(inds, len), crate::ffi::sl(vals, len), rhs);
}

unsafe extern "C" fn lp_reconvergence(p: *mut c_void, domchg: DomChg, inds: *const i32, vals: *const f64, len: i32, rhs: f64) {
    let a = lp(p).degen_args.as_ref().expect("computeBasicDegenerateDuals' arguments");
    (*a.localdom).conflict_analyze_reconvergence(
        domchg,
        crate::ffi::sl(inds, len),
        crate::ffi::sl(vals, len),
        rhs,
        &mut *a.conflictpool,
        &mut *a.globaldom,
        &*a.pseudocost,
    );
}

unsafe extern "C" fn lp_flush_domain(p: *mut c_void, d: *mut c_void) {
    lp(p).flush_domain(dom(d), false);
}

unsafe extern "C" fn lp_dom_fix_col(d: *mut c_void, col: i32, val: f64) -> bool {
    let d = dom(d);
    d.fix_col(col, val, Reason::UNSPECIFIED);
    d.infeasible
}

unsafe extern "C" fn lp_dom_num_changed(d: *mut c_void) -> i32 {
    dom(d).changed_cols().len() as i32
}

unsafe extern "C" fn lp_dom_bounds(d: *mut c_void, lo: *mut *const f64, up: *mut *const f64) {
    let d = dom(d);
    *lo = d.col_lower().as_ptr();
    *up = d.col_upper().as_ptr();
}

unsafe extern "C" fn lp_branching_column(p: *mut c_void, col: i32) -> i32 {
    let x = lp(p);
    let m = &x.lph().model;
    let ms = &*x.mipsolver;
    ms.d().symmetries.get_branching_column(&m.col_lower, &m.col_upper, col)
}

unsafe extern "C" fn lp_add_incumbent(p: *mut c_void, sol: *const f64, n: i32, obj: f64, source: i32) {
    let x = lp(p);
    let ms = &*x.mipsolver;
    let v = crate::ffi::sl(sol, n).to_vec();
    if !ms.d().parallel_lock_active() || x.worker.is_null() {
        super::fns::add_incumbent(ms, &v, obj, source, true, false);
    } else {
        (*x.worker).add_incumbent(&v, obj, source);
    }
}

unsafe extern "C" fn lp_check_solution(p: *mut c_void, sol: *const f64, n: i32) -> bool {
    let ms = &*lp(p).mipsolver;
    let v = crate::ffi::sl(sol, n).to_vec();
    super::fns::check_solution(ms, &v)
}

static LP_FNS: CLpFns = CLpFns {
    mip: lp_mip,
    solve: lp_solve,
    op: lp_op,
    delete_rows: lp_delete_rows,
    tighten: lp_tighten,
    extract_cliques: lp_extract_cliques,
    reconvergence: lp_reconvergence,
    flush_domain: lp_flush_domain,
    dom_fix_col: lp_dom_fix_col,
    dom_num_changed: lp_dom_num_changed,
    dom_bounds: lp_dom_bounds,
    branching_column: lp_branching_column,
    add_incumbent: lp_add_incumbent,
    check_solution: lp_check_solution,
};
