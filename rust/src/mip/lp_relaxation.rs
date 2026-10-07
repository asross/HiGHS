//! HighsLpRelaxation (highs/mip/HighsLpRelaxation.cpp): the MIP's LP, its
//! rows (model rows and cuts with their ages), the status handling of a
//! solve, the fractional integers and the rounding of the LP solution,
//! dual proofs (Farkas and objective bound), degenerate duals, the best
//! estimate and the degeneracy factor.
//!
//! The C++ class keeps the `Highs` LP solver, the stored basis and the
//! dives' playground; everything that touches them goes through
//! [`CLpFns`]: a view of the LP, its solution and basis ([`CLpView`],
//! refetched after every call that may change it), the solve itself, row
//! deletion, the basis inverse rows and the dual ray. The MIP data comes
//! in [`CLpMip`] (refetched per call). The cut pools, the clique table and
//! the pseudocosts are Rust and used directly. C++ reads [`LpShared`] in
//! place (status, objective, row and fractional integer arrays).
//!
//! No arithmetic here is contracted by clang: every product feeds a
//! HighsCDouble operation (a function call) or a comparison.

use super::clique::CliqueTable;
use super::cutpool::CutPool;
use super::domain::{DomChg, LOWER, UPPER};
use super::pseudocost::Pseudocost;
use super::search::FracInt;
use crate::util::cdouble::CDouble;
use crate::util::hash_table::HighsHashTable;
use crate::util::sparse_vector_sum::HighsSparseVectorSum;
use std::ffi::c_void;

const INF: f64 = f64::INFINITY;

/// HighsLpRelaxation::Status
pub const NOT_SET: i32 = 0;
pub const OPTIMAL: i32 = 1;
pub const INFEASIBLE: i32 = 2;
pub const UNSCALED_DUAL_FEASIBLE: i32 = 3;
pub const UNSCALED_PRIMAL_FEASIBLE: i32 = 4;
pub const UNSCALED_INFEASIBLE: i32 = 5;
pub const UNBOUNDED: i32 = 6;
pub const ERROR: i32 = 7;

pub fn scaled_optimal(s: i32) -> bool {
    matches!(s, OPTIMAL | UNSCALED_DUAL_FEASIBLE | UNSCALED_PRIMAL_FEASIBLE | UNSCALED_INFEASIBLE)
}
pub fn unscaled_primal_feasible(s: i32) -> bool {
    matches!(s, OPTIMAL | UNSCALED_PRIMAL_FEASIBLE)
}
pub fn unscaled_dual_feasible(s: i32) -> bool {
    matches!(s, OPTIMAL | UNSCALED_DUAL_FEASIBLE)
}

/// HighsModelStatus
const MS_OPTIMAL: i32 = 7;
const MS_INFEASIBLE: i32 = 8;
const MS_UNBOUNDED: i32 = 10;
const MS_OBJECTIVE_BOUND: i32 = 11;
const MS_TIME_LIMIT: i32 = 13;
const MS_ITERATION_LIMIT: i32 = 14;
const MS_UNKNOWN: i32 = 15;
/// HighsBasisStatus::kBasic
const BASIC: u8 = 1;
/// HighsStatus::kError
const CALL_ERROR: i32 = -1;
/// kSolutionStatusFeasible
const SOLUTION_FEASIBLE: i32 = 2;

/// LpRow::Origin
pub const ROW_MODEL: i32 = 0;
pub const ROW_CUT: i32 = 1;

/// HighsLpRelaxation::LpRow (same layout)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LpRow {
    pub origin: i32,
    pub index: i32,
    pub age: i32,
    pub cutpoolindex: i32,
}

/// The LP solver's model, solution, basis and info (C++-owned; valid until
/// the next call that changes the LP solver). Lengths of the solution and
/// basis vectors are their own (empty when there is none).
#[repr(C)]
pub struct CLpView {
    pub num_col: i32,
    pub num_row: i32,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    pub col_cost: *const f64,
    pub a_start: *const i32,
    pub a_index: *const i32,
    pub a_value: *const f64,
    pub col_value: *mut f64,
    pub n_col_value: i32,
    pub col_dual: *mut f64,
    pub n_col_dual: i32,
    pub row_value: *const f64,
    pub n_row_value: i32,
    pub row_dual: *const f64,
    pub n_row_dual: i32,
    pub col_status: *const u8,
    pub n_col_status: i32,
    pub row_status: *const u8,
    pub n_row_status: i32,
    pub basis_valid: bool,
    pub dual_valid: bool,
    pub basis_validity: i32,
    pub primal_solution_status: i32,
    pub simplex_iteration_count: i32,
    pub model_status: i32,
    pub max_primal_infeasibility: f64,
    pub max_dual_infeasibility: f64,
    pub dual_feasibility_tolerance: f64,
}

/// The MIP solver's data the LP relaxation reads (C++-owned, valid during
/// a call). The global domain is the worker's when the parallel lock is
/// active and the LP has a worker, else the MIP solver's (likewise the
/// upper limit).
#[repr(C)]
pub struct CLpMip {
    pub num_model_row: i32,
    pub num_col: i32,
    pub integral_cols: *const i32,
    pub n_integral_cols: i32,
    pub feastol: f64,
    pub epsilon: f64,
    pub small_matrix_value: f64,
    pub uplocks: *const i32,
    pub downlocks: *const i32,
    pub col_cost: *const f64,
    /// HighsVarType (0 continuous)
    pub integrality: *const u8,
    /// the model's column starts
    pub model_a_start: *const i32,
    pub ar_start: *const i32,
    pub ar_index: *const i32,
    pub ar_value: *const f64,
    pub n_ar: i32,
    pub row_integral: *const u8,
    pub max_abs_row_coef: *const f64,
    pub age_limit: i32,
    pub parallel_lock: bool,
    pub submip: bool,
    /// symmetries.columnToOrbitope is not empty
    pub has_orbitopes: bool,
    pub cutpools: *const *mut CutPool,
    pub n_cutpools: i32,
    pub cliquetable: *const CliqueTable,
    pub global_dom: *const c_void,
    pub glb: *const f64,
    pub gub: *const f64,
    pub upper_limit: f64,
    pub source_solve_lp: i32,
    pub source_unbounded: i32,
}

/// What the solve of run() reports
#[repr(C)]
#[derive(Default)]
pub struct CSolveOut {
    /// HighsStatus of the call
    pub callstatus: i32,
    pub use_simplex: bool,
    /// simplex iterations of a dual simplex that lost the race with IPX
    pub extra_iterations: i64,
}

/// The C++ side, called with the C++ HighsLpRelaxation
#[repr(C)]
pub struct CLpFns {
    pub view: unsafe extern "C" fn(*mut c_void, *mut CLpView),
    pub mip: unsafe extern "C" fn(*mut c_void, *mut CLpMip),
    /// run() up to the solve's end (solver choice, IPM / race / simplex)
    pub solve: unsafe extern "C" fn(*mut c_void, *mut CSolveOut),
    /// 0 clearSolver, 1 dual simplex with presolve, 2 presolve off, 3
    /// recoverBasis, 4 the IPM basis after an iteration limit, 5 the
    /// warning of an unbounded LP without basis, 6 trySolution of an
    /// unbounded LP's point, 7 the warning of an unexpected status, 8 the
    /// "no dual ray" dev log, 9 the root basis for removeWorkerSpecificRows
    pub op: unsafe extern "C" fn(*mut c_void, i32),
    /// removeCuts' deleteRows(mask) with the basis kept, and the solve
    pub delete_rows: unsafe extern "C" fn(*mut c_void, *mut i32),
    /// deleteRows(from, to)
    pub delete_row_range: unsafe extern "C" fn(*mut c_void, i32, i32),
    pub has_invert: unsafe extern "C" fn(*mut c_void) -> bool,
    pub basic_index: unsafe extern "C" fn(*mut c_void) -> *const i32,
    /// getBasisInverseRowSparse(row) into row_ep: (count, index, array)
    pub basis_inverse_row: unsafe extern "C" fn(*mut c_void, i32, *mut i32, *mut *const i32, *mut *const f64),
    /// getDualRaySparse into row_ep: has ray, (count, index, array)
    pub dual_ray: unsafe extern "C" fn(*mut c_void, *mut bool, *mut i32, *mut *const i32, *mut *const f64),
    /// tightenCoefficients of a (global) domain
    pub tighten: unsafe extern "C" fn(*const c_void, *mut i32, *mut f64, i32, *mut f64),
    /// cliquetable.extractCliquesFromCut
    pub extract_cliques: unsafe extern "C" fn(*mut c_void, *const i32, *const f64, i32, f64),
    /// conflictAnalyzeReconvergence(domchg, ...) with the arguments of
    /// computeBasicDegenerateDuals (`args`)
    pub reconvergence: unsafe extern "C" fn(*mut c_void, DomChg, *const i32, *const f64, i32, f64),
    /// the local domain of resolveLp: flushDomain
    pub flush_domain: unsafe extern "C" fn(*mut c_void, *mut c_void),
    /// fixCol(col, val); returns infeasible()
    pub dom_fix_col: unsafe extern "C" fn(*mut c_void, i32, f64) -> bool,
    pub dom_num_changed: unsafe extern "C" fn(*mut c_void) -> i32,
    pub dom_bounds: unsafe extern "C" fn(*mut c_void, *mut *const f64, *mut *const f64),
    /// symmetries.getBranchingColumn(lp bounds, col)
    pub branching_column: unsafe extern "C" fn(*mut c_void, i32) -> i32,
    /// addIncumbent (the worker's under the parallel lock)
    pub add_incumbent: unsafe extern "C" fn(*mut c_void, *const f64, i32, f64, i32),
    pub check_solution: unsafe extern "C" fn(*mut c_void, *const f64, i32) -> bool,
}

/// What C++ reads in place
#[repr(C)]
pub struct LpShared {
    pub rows: *const LpRow,
    pub num_rows: i32,
    pub frac: *mut FracInt,
    pub num_frac: i32,
    pub proof_inds: *const i32,
    pub proof_vals: *const f64,
    pub proof_len: i32,
    pub proof_rhs: f64,
    pub has_proof: bool,
    pub adjust_sym: bool,
    pub status: i32,
    pub objective: f64,
    pub numlpiters: i64,
    pub avg_solve_iters: f64,
}

pub struct LpRelax {
    pub sh: LpShared,
    fns: *const CLpFns,
    ctx: *mut c_void,
    pub rows: Vec<LpRow>,
    pub frac: Vec<FracInt>,
    proof_inds: Vec<i32>,
    proof_vals: Vec<f64>,
    row_ap: HighsSparseVectorSum,
    last_age_call: i64,
    num_solved: i64,
    epochs: u64,
    max_num_fractional: i32,
    /// computeDualProof's outputs for the C++ callers
    pub out_inds: Vec<i32>,
    pub out_vals: Vec<f64>,
}

/// std::min(a, b)
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// std::max(a, b)
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

#[inline]
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// # Safety
/// `p` valid for `n` reads (or n == 0)
unsafe fn sl<'a, T>(p: *const T, n: i32) -> &'a [T] {
    if n <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n as usize)
    }
}

/// The LP view as slices (valid until the LP solver changes)
pub struct Lp<'a> {
    pub v: &'a CLpView,
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub col_cost: &'a [f64],
    pub a_start: &'a [i32],
    pub a_index: &'a [i32],
    pub a_value: &'a [f64],
    pub col_value: &'a [f64],
    pub col_dual: &'a [f64],
    pub row_value: &'a [f64],
    pub row_dual: &'a [f64],
    pub col_status: &'a [u8],
    pub row_status: &'a [u8],
}

impl<'a> Lp<'a> {
    /// # Safety
    /// a view just filled by C++
    unsafe fn new(v: &'a CLpView) -> Self {
        let nc = v.num_col;
        let nnz = if nc > 0 { *v.a_start.add(nc as usize) } else { 0 };
        Lp {
            v,
            col_lower: sl(v.col_lower, nc),
            col_upper: sl(v.col_upper, nc),
            row_lower: sl(v.row_lower, v.num_row),
            row_upper: sl(v.row_upper, v.num_row),
            col_cost: sl(v.col_cost, nc),
            a_start: sl(v.a_start, if nc > 0 { nc + 1 } else { 0 }),
            a_index: sl(v.a_index, nnz),
            a_value: sl(v.a_value, nnz),
            col_value: sl(v.col_value, v.n_col_value),
            col_dual: sl(v.col_dual, v.n_col_dual),
            row_value: sl(v.row_value, v.n_row_value),
            row_dual: sl(v.row_dual, v.n_row_dual),
            col_status: sl(v.col_status, v.n_col_status),
            row_status: sl(v.row_status, v.n_row_status),
        }
    }
}

/// The MIP data as slices (valid during the call)
pub struct Mip<'a> {
    pub m: &'a CLpMip,
    pub integral_cols: &'a [i32],
    pub uplocks: &'a [i32],
    pub downlocks: &'a [i32],
    pub col_cost: &'a [f64],
    pub integrality: &'a [u8],
    pub model_a_start: &'a [i32],
    pub ar_start: &'a [i32],
    pub ar_index: &'a [i32],
    pub ar_value: &'a [f64],
    pub row_integral: &'a [u8],
    pub max_abs_row_coef: &'a [f64],
    pub cutpools: &'a [*mut CutPool],
    pub glb: &'a [f64],
    pub gub: &'a [f64],
}

impl<'a> Mip<'a> {
    /// # Safety
    /// a view just filled by C++
    unsafe fn new(m: &'a CLpMip) -> Self {
        let nc = m.num_col;
        let nr = m.num_model_row;
        Mip {
            m,
            integral_cols: sl(m.integral_cols, m.n_integral_cols),
            uplocks: sl(m.uplocks, nc),
            downlocks: sl(m.downlocks, nc),
            col_cost: sl(m.col_cost, nc),
            integrality: sl(m.integrality, nc),
            model_a_start: sl(m.model_a_start, nc + 1),
            ar_start: sl(m.ar_start, if nr > 0 { nr + 1 } else { 0 }),
            ar_index: sl(m.ar_index, m.n_ar),
            ar_value: sl(m.ar_value, m.n_ar),
            row_integral: sl(m.row_integral, nr),
            max_abs_row_coef: sl(m.max_abs_row_coef, nr),
            cutpools: sl(m.cutpools, m.n_cutpools),
            glb: sl(m.glb, nc),
            gub: sl(m.gub, nc),
        }
    }
    fn cutpool(&self, k: i32) -> &CutPool {
        // SAFETY: the pools outlive the call; only atomics are written
        // concurrently (see cutpool.rs)
        unsafe { &*self.cutpools[k as usize] }
    }
    fn is_continuous(&self, col: i32) -> bool {
        self.integrality[col as usize] == 0
    }
    fn cliques(&self) -> &CliqueTable {
        // SAFETY: the table outlives the call and is not changed by it
        unsafe { &*self.m.cliquetable }
    }

    /// LpRow::get
    pub fn row(&self, r: &LpRow) -> (&[i32], &[f64]) {
        if r.origin == ROW_CUT {
            self.cutpool(r.cutpoolindex).cut(r.index)
        } else {
            let s = self.ar_start[r.index as usize] as usize;
            let e = self.ar_start[r.index as usize + 1] as usize;
            (&self.ar_index[s..e], &self.ar_value[s..e])
        }
    }
    fn row_max_abs(&self, r: &LpRow) -> f64 {
        if r.origin == ROW_CUT {
            self.cutpool(r.cutpoolindex).max_abs_coef(r.index)
        } else {
            self.max_abs_row_coef[r.index as usize]
        }
    }
}

/// A view and its slices, refilled by `LpRelax::lp`
pub struct LpBox(Box<CLpView>);
pub struct MipBox(Box<CLpMip>);

impl LpBox {
    pub fn get(&self) -> Lp<'_> {
        // SAFETY: just filled by C++, and the box is dropped before the LP
        // solver changes (see LpRelax::lp)
        unsafe { Lp::new(&self.0) }
    }
}

impl MipBox {
    pub fn get(&self) -> Mip<'_> {
        // SAFETY: as LpBox::get
        unsafe { Mip::new(&self.0) }
    }
}

impl LpRelax {
    pub fn new(fns: *const CLpFns, ctx: *mut c_void) -> Box<Self> {
        let mut s = Box::new(LpRelax {
            sh: LpShared {
                rows: std::ptr::null(),
                num_rows: 0,
                frac: std::ptr::null_mut(),
                num_frac: 0,
                proof_inds: std::ptr::null(),
                proof_vals: std::ptr::null(),
                proof_len: 0,
                proof_rhs: 0.0,
                has_proof: false,
                adjust_sym: true,
                status: NOT_SET,
                objective: -INF,
                numlpiters: 0,
                avg_solve_iters: 0.0,
            },
            fns,
            ctx,
            rows: Vec::new(),
            frac: Vec::new(),
            proof_inds: Vec::new(),
            proof_vals: Vec::new(),
            row_ap: HighsSparseVectorSum::default(),
            last_age_call: 0,
            num_solved: 0,
            epochs: 0,
            max_num_fractional: 0,
            out_inds: Vec::new(),
            out_vals: Vec::new(),
        });
        s.sync();
        s
    }

    /// The copy constructor: rows, fractional integers, objective and the
    /// symmetric branching flag
    pub fn new_copy(other: &LpRelax, ctx: *mut c_void) -> Box<Self> {
        let mut s = Self::new(other.fns, ctx);
        s.rows = other.rows.clone();
        s.frac = other.frac.clone();
        s.sh.adjust_sym = other.sh.adjust_sym;
        // the C++ copies the objective and then sets it to -inf again
        s.sync();
        s
    }

    /// Republishes the arrays C++ reads
    pub fn sync(&mut self) {
        self.sh.rows = self.rows.as_ptr();
        self.sh.num_rows = self.rows.len() as i32;
        self.sh.frac = self.frac.as_mut_ptr();
        self.sh.num_frac = self.frac.len() as i32;
        self.sh.proof_inds = self.proof_inds.as_ptr();
        self.sh.proof_vals = self.proof_vals.as_ptr();
        self.sh.proof_len = self.proof_inds.len() as i32;
    }

    fn f(&self) -> &CLpFns {
        // SAFETY: the C++ table is static
        unsafe { &*self.fns }
    }

    /// The LP solver's current view
    pub fn lp(&self) -> LpBox {
        // SAFETY: CLpView is plain data, filled by the callback
        let mut v: Box<CLpView> = Box::new(unsafe { std::mem::zeroed() });
        unsafe { (self.f().view)(self.ctx, &mut *v) };
        LpBox(v)
    }

    pub fn mip(&self) -> MipBox {
        // SAFETY: as lp
        let mut m: Box<CLpMip> = Box::new(unsafe { std::mem::zeroed() });
        unsafe { (self.f().mip)(self.ctx, &mut *m) };
        MipBox(m)
    }

    fn op(&self, which: i32) {
        // SAFETY: a callback of the C++ side
        unsafe { (self.f().op)(self.ctx, which) }
    }

    /// loadModel's rows
    pub fn load_model(&mut self, num_row: i32) {
        self.rows.clear();
        self.rows.reserve(num_row as usize);
        self.rows.extend((0..num_row).map(|i| LpRow { origin: ROW_MODEL, index: i, age: 0, cutpoolindex: -1 }));
        self.sync();
    }

    /// addCuts' rows
    pub fn add_cuts(&mut self, cutindices: &[i32], cutpools: &[i32]) {
        if cutindices.is_empty() {
            return;
        }
        self.sh.status = NOT_SET;
        self.rows.reserve(cutindices.len());
        for (&i, &p) in cutindices.iter().zip(cutpools) {
            self.rows.push(LpRow { origin: ROW_CUT, index: i, age: 0, cutpoolindex: p });
        }
        self.sync();
    }

    fn lp_cut_removed(&self, m: &Mip, row: &LpRow) {
        // SAFETY: the pool is live and no borrow of it is held
        unsafe { CutPool::lp_cut_removed(m.cutpools[row.cutpoolindex as usize], row.index, m.m.parallel_lock) };
    }

    pub fn remove_obsolete_rows(&mut self, notify_pool: bool) {
        let mb = self.mip();
        let m = mb.get();
        let lb = self.lp();
        let lp = lb.get();
        let nlprows = lp.v.num_row as usize;
        let nmodel = m.m.num_model_row as usize;
        let mut mask = Vec::new();
        let mut ndel = 0;
        for i in nmodel..nlprows {
            if lp.row_status[i] == BASIC {
                if ndel == 0 {
                    mask.resize(nlprows, 0);
                }
                ndel += 1;
                mask[i] = 1;
                if notify_pool {
                    self.lp_cut_removed(&m, &self.rows[i]);
                }
            }
        }
        drop(lb);
        self.remove_cuts(ndel, &mut mask, nmodel);
    }

    pub fn remove_worker_specific_rows(&mut self) {
        let nmodel = self.mip().get().m.num_model_row as usize;
        let nlprows = self.lp().get().v.num_row as usize;
        let mut mask = Vec::new();
        let mut ndel = 0;
        for i in nmodel..nlprows {
            if self.rows[i].cutpoolindex > 0 {
                if ndel == 0 {
                    mask.resize(nlprows, 0);
                }
                ndel += 1;
                mask[i] = 1;
            }
        }
        if ndel > 0 {
            self.op(9);
        }
        self.remove_cuts(ndel, &mut mask, nmodel);
    }

    /// removeCuts(ndelcuts, deletemask)
    fn remove_cuts(&mut self, ndel: usize, mask: &mut [i32], nmodel: usize) {
        if ndel == 0 {
            return;
        }
        let nlprows = mask.len();
        // SAFETY: a callback of the C++ side; mask has nlprows entries
        unsafe { (self.f().delete_rows)(self.ctx, mask.as_mut_ptr()) };
        for i in nmodel..nlprows {
            if mask[i] >= 0 {
                self.rows[mask[i] as usize] = self.rows[i];
            }
        }
        self.rows.truncate(self.rows.len() - ndel);
        self.sync();
    }

    /// removeCuts(): all cuts
    pub fn remove_all_cuts(&mut self) {
        let mb = self.mip();
        let m = mb.get();
        let nmodel = m.m.num_model_row;
        let nlprows = self.lp().get().v.num_row;
        // SAFETY: a callback of the C++ side
        unsafe { (self.f().delete_row_range)(self.ctx, nmodel, nlprows - 1) };
        for i in nmodel as usize..nlprows as usize {
            if self.rows[i].origin == ROW_CUT {
                self.lp_cut_removed(&m, &self.rows[i]);
            }
        }
        self.rows.truncate(nmodel as usize);
        self.sync();
    }

    pub fn perform_aging(&mut self, delete_rows: bool) {
        let mb = self.mip();
        let m = mb.get();
        let lb = self.lp();
        let lp = lb.get();
        if lp.v.basis_validity == 0 || lp.v.max_dual_infeasibility > m.m.feastol || !lp.v.dual_valid {
            return;
        }
        let mut agelimit;
        if delete_rows {
            agelimit = m.m.age_limit;
            self.epochs += 1;
            if self.epochs % std::cmp::max(agelimit >> 1, 2) as u64 != 0 {
                agelimit = i32::MAX;
            } else if (self.epochs as i32) < agelimit {
                agelimit = self.epochs as i32;
            }
        } else {
            if self.last_age_call == self.sh.numlpiters {
                return;
            }
            agelimit = i32::MAX;
        }
        self.last_age_call = self.sh.numlpiters;
        let nlprows = lp.v.num_row as usize;
        let nmodel = m.m.num_model_row as usize;
        let mut mask = Vec::new();
        let mut ndel = 0;
        let dtol = lp.v.dual_feasibility_tolerance;
        for i in nmodel..nlprows {
            if lp.row_status[i] == BASIC {
                let r = &mut self.rows[i];
                r.age += (delete_rows || r.age != 0) as i32;
                if r.age > agelimit {
                    if ndel == 0 {
                        mask.resize(nlprows, 0);
                    }
                    ndel += 1;
                    mask[i] = 1;
                    let r = self.rows[i];
                    self.lp_cut_removed(&m, &r);
                }
            } else if lp.row_dual[i].abs() > dtol {
                self.rows[i].age = 0;
            }
        }
        drop(lb);
        self.remove_cuts(ndel, &mut mask, nmodel);
    }

    pub fn reset_ages(&mut self) {
        let mb = self.mip();
        let m = mb.get();
        let lb = self.lp();
        let lp = lb.get();
        if lp.v.basis_validity == 0 || lp.v.max_dual_infeasibility > m.m.feastol || !lp.v.dual_valid {
            return;
        }
        let dtol = lp.v.dual_feasibility_tolerance;
        for i in m.m.num_model_row as usize..lp.v.num_row as usize {
            if lp.row_status[i] != BASIC && lp.row_dual[i].abs() > dtol {
                self.rows[i].age = 0;
            }
        }
    }

    pub fn notify_cut_pools_lp_copied(&self, n: i32) {
        let mb = self.mip();
        let m = mb.get();
        let nlprows = self.lp().get().v.num_row as usize;
        for r in &self.rows[m.m.num_model_row as usize..nlprows] {
            if r.origin == ROW_CUT {
                m.cutpool(r.cutpoolindex).increase_num_lps(r.index, n);
            }
        }
    }

    pub fn compute_best_estimate(&self, ps: &Pseudocost) -> f64 {
        let mut estimate = CDouble::from(self.sh.objective);
        if !self.frac.is_empty() {
            let mb = self.mip();
            let m = mb.get();
            let mut increase = CDouble::from(0.0);
            let offset = m.m.feastol * cmax(self.sh.objective.abs(), 1.0) / m.integral_cols.len() as f64;
            for f in &self.frac {
                increase += cmin(
                    ps.get_pseudocost_up_offset(f.col, f.val, offset),
                    ps.get_pseudocost_down_offset(f.col, f.val, offset),
                );
            }
            estimate += increase.to_f64();
        }
        estimate.to_f64()
    }

    pub fn compute_lp_degeneracy(&self, local_lower: &[f64], local_upper: &[f64]) -> f64 {
        let lb = self.lp();
        let lp = lb.get();
        if !lp.v.dual_valid || !lp.v.basis_valid {
            return 1.0;
        }
        let tol = lp.v.max_dual_infeasibility;
        let nrows = lp.v.num_row;
        let ncols = lp.v.num_col;
        let mut num_fixed_rows = 0;
        let mut num_ineq = 0;
        let mut num_basic_eq = 0;
        for i in 0..nrows as usize {
            if lp.row_lower[i] != lp.row_upper[i] {
                num_ineq += 1;
                if lp.row_status[i] != BASIC && lp.row_dual[i].abs() > tol {
                    num_fixed_rows += 1;
                }
            } else {
                num_basic_eq += (lp.row_status[i] == BASIC) as i32;
            }
        }
        let mut num_already_fixed = 0;
        let mut num_fixed_cols = 0;
        for i in 0..ncols as usize {
            if lp.col_status[i] != BASIC {
                if lp.col_dual[i].abs() > tol {
                    num_fixed_cols += 1;
                } else if local_lower[i] == local_upper[i] {
                    num_already_fixed += 1;
                }
            }
        }
        let base = ncols - num_already_fixed + num_ineq + num_basic_eq - nrows;
        let share = if base > 0 { 1.0 - (num_fixed_cols + num_fixed_rows) as f64 / base as f64 } else { 0.0 };
        let ratio = if nrows > 0 {
            (ncols + num_ineq + num_basic_eq - num_fixed_cols - num_fixed_rows - num_already_fixed) as f64 / nrows as f64
        } else {
            1.0
        };
        let fac1 = if share < 0.8 { 1.0 } else { 10f64.powf(10.0 * (share - 0.7)) };
        let fac2 = if ratio < 2.0 { 1.0 } else { 10.0 * ratio };
        fac1 * fac2
    }

    fn ensure_row_ap(&mut self, num_col: i32) {
        if self.row_ap.values.len() < num_col as usize {
            self.row_ap.set_dimension(num_col as usize);
        }
    }

    /// computeBasicDegenerateDuals; `args` are the C++ arguments that the
    /// reconvergence callback needs
    pub fn compute_basic_degenerate_duals(&mut self, threshold: f64, getdualproof: bool) {
        // SAFETY: callbacks of the C++ side
        if !unsafe { (self.f().has_invert)(self.ctx) } {
            return;
        }
        let mb = self.mip();
        let m = mb.get();
        let feastol = m.m.feastol;
        let eps = m.m.epsilon;
        let lb = self.lp();
        let lp = lb.get();
        // the C++ writes the solution's column duals in place
        // SAFETY: n_col_dual entries, no other reference to them is alive
        // while this one is used (lp.col_dual is not read below)
        let col_dual = unsafe { std::slice::from_raw_parts_mut(lp.v.col_dual, lp.v.n_col_dual as usize) };
        let mut k = 0;
        for &col in m.integral_cols {
            let c = col as usize;
            if lp.col_status[c] != BASIC {
                continue;
            }
            let l = lp.col_lower[c];
            let u = lp.col_upper[c];
            if u - l < feastol {
                continue;
            }
            let x = lp.col_value[c];
            if x - l < u - x {
                if x > l + feastol {
                    continue;
                }
                col_dual[c] = 1.0;
                k += 1;
            } else {
                if x < u - feastol {
                    continue;
                }
                col_dual[c] = -1.0;
                k += 1;
            }
        }
        if k == 0 {
            return;
        }
        let num_col = lp.v.num_col;
        self.ensure_row_ap(num_col);
        // SAFETY: the basic index array has num_row entries
        let basic = unsafe { sl((self.f().basic_index)(self.ctx), lp.v.num_row) };
        let mut row = 0usize;
        while k > 0 {
            let var = basic[row];
            row += 1;
            if var >= num_col {
                continue;
            }
            let vu = var as usize;
            if col_dual[vu].abs() != 1.0 {
                continue;
            }
            k -= 1;
            let (mut cnt, mut idx, mut arr) = (0, std::ptr::null(), std::ptr::null());
            // SAFETY: the C++ fills its HVector and returns its arrays,
            // valid until the next call; the LP view stays valid (the
            // solve's data does not change)
            unsafe { (self.f().basis_inverse_row)(self.ctx, row as i32 - 1, &mut cnt, &mut idx, &mut arr) };
            let ep_index = unsafe { sl(idx, cnt) };
            let ep = |r: i32| unsafe { *arr.add(r as usize) };

            let sign = col_dual[vu];
            col_dual[vu] = 0.0;
            let mut deg = INF;
            for &r in ep_index {
                let ru = r as usize;
                let l = lp.row_lower[ru];
                let u = lp.row_upper[ru];
                if l == u {
                    continue;
                }
                let dual = lp.row_dual[ru];
                let val = -sign * ep(r);
                if val > 0.0 {
                    if lp.row_value[ru] - l > feastol {
                        deg = cmin(deg, -dual / val);
                        if deg < threshold {
                            break;
                        }
                    }
                } else if u - lp.row_value[ru] > feastol {
                    deg = cmin(deg, -dual / val);
                    if deg < threshold {
                        break;
                    }
                }
            }
            if deg < threshold {
                continue;
            }
            self.row_ap.clear();
            for &r in ep_index {
                let (inds, vals) = m.row(&self.rows[r as usize]);
                let w = ep(r);
                for (&j, &v) in inds.iter().zip(vals) {
                    self.row_ap.add_f64(j, w * v);
                }
            }
            self.row_ap.cleanup(|_, v| v.abs() <= eps);
            for &c in &self.row_ap.nonzeroinds {
                if c == var {
                    continue;
                }
                let cu = c as usize;
                let l = lp.col_lower[cu];
                let u = lp.col_upper[cu];
                if l == u {
                    continue;
                }
                let dual = col_dual[cu];
                let val = sign * self.row_ap.get_value(c);
                if val > eps {
                    if lp.col_value[cu] - l > feastol {
                        deg = cmin(deg, -dual / val);
                        if deg < threshold {
                            break;
                        }
                    }
                } else if val < -eps && u - lp.col_value[cu] > feastol {
                    deg = cmin(deg, -dual / val);
                    if deg < threshold {
                        break;
                    }
                }
            }
            if deg < threshold {
                continue;
            }
            if deg == INF && getdualproof {
                let mut rhs = CDouble::from(0.0);
                for &r in ep_index {
                    let ru = r as usize;
                    let l = lp.row_lower[ru];
                    let u = lp.row_upper[ru];
                    let val = sign * ep(r);
                    if u == l || val > eps {
                        rhs += val * u;
                    } else if val < -eps {
                        rhs += val * l;
                    } else {
                        rhs += val * lp.row_value[ru];
                    }
                }
                let nz = &self.row_ap.nonzeroinds;
                self.proof_vals.clear();
                self.proof_vals.extend(nz.iter().map(|&i| sign * self.row_ap.get_value(i)));
                let domchg = if sign == 1.0 {
                    DomChg { boundval: lp.col_lower[vu], column: var, boundtype: UPPER }
                } else {
                    DomChg { boundval: lp.col_upper[vu], column: var, boundtype: LOWER }
                };
                // SAFETY: a callback of the C++ side (the local domain's
                // conflict analysis; it does not touch the LP)
                unsafe {
                    (self.f().reconvergence)(
                        self.ctx,
                        domchg,
                        nz.as_ptr(),
                        self.proof_vals.as_ptr(),
                        nz.len() as i32,
                        rhs.to_f64(),
                    )
                };
                continue;
            }
            col_dual[vu] = sign * deg;
        }
        self.sync();
    }

    /// computeDualProof into out_inds / out_vals; None if there is none
    pub fn compute_dual_proof(
        &mut self,
        dom: *const c_void,
        glb: &[f64],
        gub: &[f64],
        upperbound: f64,
        extract_cliques: bool,
    ) -> Option<f64> {
        let mut inds = std::mem::take(&mut self.out_inds);
        let mut vals = std::mem::take(&mut self.out_vals);
        let r = self.dual_proof(dom, glb, gub, upperbound, extract_cliques, &mut inds, &mut vals);
        self.out_inds = inds;
        self.out_vals = vals;
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn dual_proof(
        &self,
        dom: *const c_void,
        glb: &[f64],
        gub: &[f64],
        upperbound: f64,
        extract_cliques: bool,
        inds: &mut Vec<i32>,
        vals: &mut Vec<f64>,
    ) -> Option<f64> {
        let mb = self.mip();
        let m = mb.get();
        let lb = self.lp();
        let lp = lb.get();
        let mut row_dual = lp.row_dual.to_vec();
        let mut upper = CDouble::from(upperbound);
        for i in 0..lp.v.num_row as usize {
            if row_dual[i] > 0.0 {
                if lp.row_lower[i] != -INF {
                    upper -= row_dual[i] * lp.row_lower[i];
                } else {
                    row_dual[i] = 0.0;
                }
            } else if row_dual[i] < 0.0 {
                if lp.row_upper[i] != INF {
                    upper -= row_dual[i] * lp.row_upper[i];
                } else {
                    row_dual[i] = 0.0;
                }
            }
        }
        inds.clear();
        vals.clear();
        let feastol = m.m.feastol;
        for i in 0..lp.v.num_col as usize {
            let mut sum = CDouble::from(lp.col_cost[i]);
            for j in lp.a_start[i] as usize..lp.a_start[i + 1] as usize {
                let rd = row_dual[lp.a_index[j] as usize];
                if rd == 0.0 {
                    continue;
                }
                sum -= lp.a_value[j] * rd;
            }
            let val = sum.to_f64();
            if val.abs() <= m.m.small_matrix_value {
                continue;
            }
            let mut remove = val.abs() <= feastol;
            if !remove && (glb[i] == gub[i] || m.is_continuous(i as i32)) {
                remove = if val > 0.0 {
                    lp.col_value[i] - glb[i] <= feastol
                } else {
                    gub[i] - lp.col_value[i] <= feastol
                };
            }
            if remove {
                if val < 0.0 {
                    if gub[i] == INF {
                        return None;
                    }
                    upper -= val * gub[i];
                } else {
                    if glb[i] == -INF {
                        return None;
                    }
                    upper -= val * glb[i];
                }
                continue;
            }
            vals.push(val);
            inds.push(i as i32);
        }
        let mut rhs = upper.to_f64();
        drop(lb);
        // SAFETY: callbacks of the C++ side
        unsafe {
            (self.f().tighten)(dom, inds.as_mut_ptr(), vals.as_mut_ptr(), inds.len() as i32, &mut rhs);
            if extract_cliques && !m.m.parallel_lock {
                (self.f().extract_cliques)(self.ctx, inds.as_ptr(), vals.as_ptr(), inds.len() as i32, rhs);
            }
        }
        Some(rhs)
    }

    fn store_dual_inf_proof(&mut self) {
        self.sh.has_proof = false;
        let lb = self.lp();
        if lb.get().v.basis_validity == 0 {
            return;
        }
        let num_col = lb.get().v.num_col;
        drop(lb);
        self.ensure_row_ap(num_col);
        let (mut has, mut cnt, mut idx, mut arr) = (false, 0, std::ptr::null(), std::ptr::null());
        // SAFETY: the C++ fills its HVector and returns its arrays
        unsafe { (self.f().dual_ray)(self.ctx, &mut has, &mut cnt, &mut idx, &mut arr) };
        self.sh.has_proof = has;
        if !has {
            self.op(8);
            return;
        }
        self.proof_inds.clear();
        self.proof_vals.clear();
        self.sh.proof_rhs = INF;
        let mb = self.mip();
        let m = mb.get();
        let lb = self.lp();
        let lp = lb.get();
        let eps = m.m.epsilon;
        let feastol = m.m.feastol;
        // SAFETY: the ray's arrays (valid until the next C++ call)
        let ep_index = unsafe { sl(idx, cnt) };
        let mut upper = CDouble::from(0.0);
        self.row_ap.clear();
        for &r in ep_index {
            let ru = r as usize;
            let weight = -unsafe { *arr.add(ru) };
            let row = &self.rows[ru];
            if weight.abs() * m.row_max_abs(row) <= eps {
                continue;
            } else if weight > 0.0 {
                if lp.row_upper[ru] == INF {
                    continue;
                }
                upper += weight * lp.row_upper[ru];
            } else {
                if lp.row_lower[ru] == -INF {
                    continue;
                }
                upper += weight * lp.row_lower[ru];
            }
            let (inds, vals) = m.row(row);
            for (&j, &v) in inds.iter().zip(vals) {
                self.row_ap.add_f64(j, weight * v);
            }
        }
        let (glb, gub) = (m.glb, m.gub);
        let mut ok = true;
        for &i in &self.row_ap.nonzeroinds {
            let iu = i as usize;
            let val = self.row_ap.get_value(i);
            if val.abs() <= eps {
                continue;
            }
            let mut remove = val.abs() <= feastol;
            if !remove && (glb[iu] == gub[iu] || m.is_continuous(i)) {
                remove = if val > 0.0 {
                    lp.col_lower[iu] - glb[iu] <= feastol
                } else {
                    gub[iu] - lp.col_upper[iu] <= feastol
                };
            }
            if remove {
                if val < 0.0 {
                    if gub[iu] == INF {
                        ok = false;
                        break;
                    }
                    upper -= val * gub[iu];
                } else {
                    if glb[iu] == -INF {
                        ok = false;
                        break;
                    }
                    upper -= val * glb[iu];
                }
                continue;
            }
            self.proof_vals.push(val);
            self.proof_inds.push(i);
        }
        if !ok {
            self.sh.has_proof = false;
            self.sync();
            return;
        }
        let mut rhs = upper.to_f64();
        drop(lb);
        // SAFETY: callbacks of the C++ side
        unsafe {
            (self.f().tighten)(
                m.m.global_dom,
                self.proof_inds.as_mut_ptr(),
                self.proof_vals.as_mut_ptr(),
                self.proof_inds.len() as i32,
                &mut rhs,
            );
            self.sh.proof_rhs = rhs;
            if !m.m.parallel_lock {
                (self.f().extract_cliques)(
                    self.ctx,
                    self.proof_inds.as_ptr(),
                    self.proof_vals.as_ptr(),
                    self.proof_inds.len() as i32,
                    rhs,
                );
            }
        }
        self.sync();
    }

    fn store_dual_ub_proof(&mut self) {
        self.proof_inds.clear();
        self.proof_vals.clear();
        let dual_valid = self.lp().get().v.dual_valid;
        let mut proof = None;
        if dual_valid {
            let mb = self.mip();
            let m = mb.get();
            let mut inds = std::mem::take(&mut self.proof_inds);
            let mut vals = std::mem::take(&mut self.proof_vals);
            proof = self.dual_proof(m.m.global_dom, m.glb, m.gub, m.m.upper_limit, true, &mut inds, &mut vals);
            self.proof_inds = inds;
            self.proof_vals = vals;
        }
        self.sh.has_proof = proof.is_some();
        match proof {
            Some(rhs) => self.sh.proof_rhs = rhs,
            None => self.sh.proof_rhs = INF,
        }
        self.sync();
    }

    /// computeDualInfProof: the stored proof
    pub fn has_dual_inf_proof(&self) -> bool {
        self.sh.has_proof
    }

    fn count_solve(&mut self, itercount: i64) {
        self.num_solved += 1;
        if itercount >= 0 {
            self.sh.avg_solve_iters += (itercount as f64 - self.sh.avg_solve_iters) / self.num_solved as f64;
        }
    }

    pub fn run(&mut self, resolve_on_error: bool) -> i32 {
        let mut out = CSolveOut::default();
        // SAFETY: a callback of the C++ side
        unsafe { (self.f().solve)(self.ctx, &mut out) };
        self.sh.numlpiters += out.extra_iterations;
        let lb = self.lp();
        let v = &*lb.0;
        let mut itercount = -1i64;
        if out.use_simplex {
            itercount = std::cmp::max(0, v.simplex_iteration_count) as i64;
            self.sh.numlpiters += itercount;
        }
        if out.callstatus == CALL_ERROR {
            drop(lb);
            self.op(0);
            if resolve_on_error {
                self.op(1);
                let r = self.run(false);
                self.op(2);
                return r;
            }
            self.op(3);
            return ERROR;
        }
        let model_status = v.model_status;
        let feastol = self.mip().get().m.feastol;
        match model_status {
            MS_OBJECTIVE_BOUND => {
                self.count_solve(itercount);
                drop(lb);
                self.store_dual_ub_proof();
                INFEASIBLE
            }
            MS_INFEASIBLE => {
                self.count_solve(itercount);
                drop(lb);
                self.store_dual_inf_proof();
                INFEASIBLE
            }
            MS_UNBOUNDED => {
                if v.basis_validity == 0 {
                    self.op(5);
                }
                if v.primal_solution_status == SOLUTION_FEASIBLE {
                    self.op(6);
                }
                UNBOUNDED
            }
            MS_UNKNOWN | MS_OPTIMAL => {
                if model_status == MS_UNKNOWN && v.basis_validity == 0 {
                    return ERROR;
                }
                self.count_solve(itercount);
                let p = v.max_primal_infeasibility <= feastol;
                let d = v.max_dual_infeasibility <= feastol;
                if p && d {
                    OPTIMAL
                } else if p {
                    UNSCALED_PRIMAL_FEASIBLE
                } else if d {
                    UNSCALED_DUAL_FEASIBLE
                } else if model_status == MS_OPTIMAL {
                    UNSCALED_INFEASIBLE
                } else {
                    ERROR
                }
            }
            MS_ITERATION_LIMIT => {
                if !self.mip().get().m.submip && resolve_on_error {
                    drop(lb);
                    self.op(4);
                    return self.run(false);
                }
                ERROR
            }
            MS_TIME_LIMIT => ERROR,
            _ => {
                self.op(7);
                ERROR
            }
        }
    }

    /// resolveLp; `domain` is the C++ HighsDomain (or null)
    pub fn resolve_lp(&mut self, domain: *mut c_void) -> i32 {
        self.frac.clear();
        self.sync();
        let f = self.fns;
        let ctx = self.ctx;
        // SAFETY: the C++ table is static
        let f = unsafe { &*f };
        loop {
            if !domain.is_null() {
                // SAFETY: callbacks of the C++ side
                unsafe { (f.flush_domain)(ctx, domain) };
            }
            self.sh.status = self.run(true);
            match self.sh.status {
                UNSCALED_INFEASIBLE | UNSCALED_DUAL_FEASIBLE | UNSCALED_PRIMAL_FEASIBLE | OPTIMAL => {
                    if self.process_solution(domain) {
                        continue;
                    }
                }
                INFEASIBLE => self.sh.objective = INF,
                _ => {}
            }
            break;
        }
        self.sync();
        self.sh.status
    }

    /// The solution part of resolveLp; true to solve again (fixings of
    /// substituted columns)
    fn process_solution(&mut self, domain: *mut c_void) -> bool {
        // SAFETY: the C++ table is static
        let f = unsafe { &*self.fns };
        let ctx = self.ctx;
        let mb = self.mip();
        let m = mb.get();
        let feastol = m.m.feastol;
        let mut fracints: HighsHashTable<i32, (f64, i32)> = HighsHashTable::with_capacity(self.max_num_fractional as usize);
        let mut lb = self.lp();
        let mut roundable = true;
        let cliques = m.cliques();
        let (glb, gub) = (m.glb, m.gub);
        let nmodel_col_entries = |i: usize| m.model_a_start[i + 1] - m.model_a_start[i];
        for &i in m.integral_cols {
            let iu = i as usize;
            let lp = lb.get();
            let mut val = cmax(cmin(lp.col_value[iu], lp.col_upper[iu]), lp.col_lower[iu]);
            if fractionality(val) > feastol {
                let mut col = i;
                roundable = roundable && (m.uplocks[col as usize] == 0 || m.downlocks[col as usize] == 0);
                let mut subst = cliques.get_substitution(col).copied();
                while let Some(s) = subst {
                    let rc = s.replace.col() as usize;
                    let lp = lb.get();
                    if lp.col_lower[rc] == lp.col_upper[rc] {
                        if !domain.is_null() {
                            let w = if s.replace.val() != 0 { lp.col_lower[rc] } else { 1.0 - lp.col_lower[rc] };
                            // SAFETY: callbacks of the C++ side (domain
                            // only; the LP is unchanged)
                            if unsafe { (f.dom_fix_col)(domain, col, w) } {
                                self.sh.objective = INF;
                                self.sh.status = INFEASIBLE;
                                // the C++ returns kInfeasible from resolveLp
                                self.frac.clear();
                                return false;
                            }
                        } else {
                            break;
                        }
                    }
                    if !domain.is_null() {
                        let replace_val = if s.replace.val() == 0 { 1.0 - val } else { val };
                        let (mut dl, mut du) = (std::ptr::null(), std::ptr::null());
                        // SAFETY: the domain's bounds (numCol entries)
                        let (rl, ru) = unsafe {
                            (f.dom_bounds)(domain, &mut dl, &mut du);
                            (*dl.add(rc), *du.add(rc))
                        };
                        if replace_val < rl - feastol || replace_val > ru + feastol {
                            break;
                        }
                    }
                    col = s.replace.col();
                    if s.replace.val() == 0 {
                        val = 1.0 - val;
                    }
                    subst = cliques.get_substitution(col).copied();
                }
                if self.sh.adjust_sym && m.m.has_orbitopes {
                    // SAFETY: a callback of the C++ side (reads only)
                    col = unsafe { (f.branching_column)(ctx, col) };
                }
                let pair = fracints.get_or_insert_default(col);
                pair.0 += val;
                pair.1 += 1;
            } else {
                if lp.col_status[iu] == BASIC {
                    continue;
                }
                let x = lp.col_value[iu];
                if cmin(gub[iu] - x, x - glb[iu]) <= feastol {
                    continue;
                }
                let col_start = lp.a_start[iu] + nmodel_col_entries(iu);
                let col_end = lp.a_start[iu + 1];
                if col_start == col_end {
                    continue;
                }
                for j in col_start as usize..col_end as usize {
                    let row = lp.a_index[j] as usize;
                    if self.rows[row].age == 0 {
                        continue;
                    }
                    if lp.row_value[row] < lp.row_upper[row] - feastol {
                        continue;
                    }
                    self.rows[row].age = 0;
                }
            }
        }
        self.max_num_fractional = std::cmp::max(fracints.len() as i32, self.max_num_fractional);
        // SAFETY: a callback of the C++ side
        if !domain.is_null() && unsafe { (f.dom_num_changed)(domain) } != 0 {
            return true;
        }
        for (&k, &(sum, n)) in fracints.iter() {
            self.frac.push(FracInt { col: k, val: sum / n as f64 });
        }
        self.sync();
        let ncol = m.m.num_col as usize;
        let mut objsum = CDouble::from(0.0);
        if roundable && !self.frac.is_empty() && unscaled_primal_feasible(self.sh.status) {
            let lp = lb.get();
            let mut roundsol = lp.col_value.to_vec();
            for fi in &self.frac {
                let c = fi.col as usize;
                if m.uplocks[c] == 0 && (m.col_cost[c] < 0.0 || m.downlocks[c] != 0) {
                    roundsol[c] = cmin(
                        (fi.val - feastol).ceil(),
                        if lp.col_upper[c] == INF { INF } else { (lp.col_upper[c] + feastol).floor() },
                    );
                } else {
                    roundsol[c] = cmax(
                        (fi.val + feastol).floor(),
                        if lp.col_lower[c] == -INF { -INF } else { (lp.col_lower[c] - feastol).ceil() },
                    );
                }
            }
            for s in cliques.substitutions.iter().rev() {
                let rc = s.replace.col() as usize;
                roundsol[s.substcol as usize] = if s.replace.val() == 0 { 1.0 - roundsol[rc] } else { roundsol[rc] };
            }
            for i in 0..ncol {
                objsum += roundsol[i] * m.col_cost[i];
            }
            drop(lb);
            // SAFETY: a callback of the C++ side (it may change the
            // incumbent, not the LP)
            unsafe {
                (f.add_incumbent)(ctx, roundsol.as_ptr(), roundsol.len() as i32, objsum.to_f64(), m.m.source_solve_lp)
            };
            objsum = CDouble::from(0.0);
            lb = self.lp();
        }
        let lp = lb.get();
        for i in 0..ncol {
            objsum += lp.col_value[i] * m.col_cost[i];
        }
        if self.frac.is_empty() && !unscaled_primal_feasible(self.sh.status) {
            let mut fix = lp.col_value.to_vec();
            for i in 0..ncol {
                if fix[i] < lp.col_lower[i] {
                    fix[i] = lp.col_lower[i];
                } else if fix[i] > lp.col_upper[i] {
                    fix[i] = lp.col_upper[i];
                } else if !m.is_continuous(i as i32) {
                    fix[i] = fix[i].round();
                }
            }
            // SAFETY: a callback of the C++ side (reads only)
            if unsafe { (f.check_solution)(ctx, fix.as_ptr(), fix.len() as i32) } {
                // the C++ moves the point into the LP solver's solution
                // SAFETY: n_col_value entries, no other reference alive
                unsafe { std::slice::from_raw_parts_mut(lp.v.col_value, fix.len()) }.copy_from_slice(&fix);
                self.sh.status =
                    if unscaled_dual_feasible(self.sh.status) { OPTIMAL } else { UNSCALED_PRIMAL_FEASIBLE };
            }
        }
        self.sh.objective = objsum.to_f64();
        false
    }
}

/// The FFI of the C++ HighsLpRelaxation
pub mod ffi {
    use super::*;

    #[no_mangle]
    pub extern "C" fn highs_rs_lprelax_new(fns: *const CLpFns, ctx: *mut c_void) -> *mut LpRelax {
        Box::into_raw(LpRelax::new(fns, ctx))
    }

    /// # Safety
    /// live `other`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_copy(other: *const LpRelax, ctx: *mut c_void) -> *mut LpRelax {
        Box::into_raw(LpRelax::new_copy(&*other, ctx))
    }

    /// # Safety
    /// a pointer of highs_rs_lprelax_new / copy, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_free(p: *mut LpRelax) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// # Safety
    /// live `p`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_shared(p: *mut LpRelax) -> *mut LpShared {
        &mut (*p).sh
    }

    /// 0 loadModel(i), 1 removeObsoleteRows(notify = i), 2
    /// removeWorkerSpecificRows, 3 removeCuts(), 4 performAging(i), 5
    /// resetAges, 6 notifyCutPoolsLpCopied(i)
    ///
    /// # Safety
    /// live `p`, called by its C++ object
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_op(p: *mut LpRelax, which: i32, i: i32) {
        let s = &mut *p;
        match which {
            0 => s.load_model(i),
            1 => s.remove_obsolete_rows(i != 0),
            2 => s.remove_worker_specific_rows(),
            3 => s.remove_all_cuts(),
            4 => s.perform_aging(i != 0),
            5 => s.reset_ages(),
            _ => s.notify_cut_pools_lp_copied(i),
        }
    }

    /// # Safety
    /// live `p`, `n` entries each
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_add_cuts(p: *mut LpRelax, inds: *const i32, pools: *const i32, n: i32) {
        (*p).add_cuts(sl(inds, n), sl(pools, n));
    }

    /// # Safety
    /// live `p`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_run(p: *mut LpRelax, resolve_on_error: bool) -> i32 {
        (*p).run(resolve_on_error)
    }

    /// # Safety
    /// live `p`; `domain` a C++ HighsDomain or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_resolve(p: *mut LpRelax, domain: *mut c_void) -> i32 {
        (*p).resolve_lp(domain)
    }

    /// # Safety
    /// live `p` and pseudocosts
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_best_estimate(p: *const LpRelax, ps: *const Pseudocost) -> f64 {
        (*p).compute_best_estimate(&*ps)
    }

    /// # Safety
    /// live `p`; numCol bounds
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_degeneracy(p: *const LpRelax, lower: *const f64, upper: *const f64, n: i32) -> f64 {
        (*p).compute_lp_degeneracy(sl(lower, n), sl(upper, n))
    }

    /// # Safety
    /// live `p`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lprelax_degenerate_duals(p: *mut LpRelax, threshold: f64, getdualproof: bool) {
        (*p).compute_basic_degenerate_duals(threshold, getdualproof)
    }

    /// computeDualProof with the given (global) domain; the proof is left
    /// in (inds, vals, len), valid until the next call
    ///
    /// # Safety
    /// live `p`; numCol bounds
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_lprelax_dual_proof(
        p: *mut LpRelax,
        dom: *const c_void,
        glb: *const f64,
        gub: *const f64,
        n: i32,
        upperbound: f64,
        extract_cliques: bool,
        inds: *mut *const i32,
        vals: *mut *const f64,
        len: *mut i32,
        rhs: *mut f64,
    ) -> bool {
        let s = &mut *p;
        match s.compute_dual_proof(dom, sl(glb, n), sl(gub, n), upperbound, extract_cliques) {
            Some(r) => {
                *inds = s.out_inds.as_ptr();
                *vals = s.out_vals.as_ptr();
                *len = s.out_inds.len() as i32;
                *rhs = r;
                true
            }
            None => false,
        }
    }
}
