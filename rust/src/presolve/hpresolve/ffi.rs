//! The boundary with the C++ (highs/presolve/HPresolveRust.cpp): the entry
//! point `highs_rs_presolve_run` and the C++ callbacks of [`Host`].

use super::driver::Input;
use super::{MipInfo, Options, Presolve, RULE_COUNT};
use std::ffi::{c_char, c_void, CStr, CString};
use std::slice::{from_raw_parts, from_raw_parts_mut};

type Ctx = *mut c_void;

/// The model written back to the C++ HighsLp
#[repr(C)]
pub struct CModel {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: *const f64,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    pub integrality: *const u8,
    pub offset: f64,
    pub maximize: bool,
}

/// What the probing and enumeration loops in C++ read and write
#[repr(C)]
pub struct ProbingIo {
    pub num_probes: *mut u16,
    pub num_col: i32,
    pub num_row: i32,
    pub num_nonzeros: i32,
    pub col_deleted: *const u8,
    pub integrality: *const u8,
    pub colsize_len: usize,
    pub probing_contingent: *mut i64,
    pub num_probed: *mut i32,
    pub probing_num_del_col: *mut i32,
    pub probing_early_abort: *mut bool,
    /// out: the lifting opportunities stored by the probing loop
    pub lifting: *const LiftOpp,
    pub num_lifting: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiftOpp {
    pub row: i32,
    pub col: i32,
    pub val: i32,
    pub coef: f64,
}

/// HighsSubstitution
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImplSubst {
    pub substcol: i32,
    pub staycol: i32,
    pub scale: f64,
    pub offset: f64,
}

/// (pointer, length) of a C++ array
#[repr(C)]
pub struct CSlice<T> {
    pub ptr: *const T,
    pub len: usize,
}

impl<T: Copy> CSlice<T> {
    fn empty() -> Self {
        CSlice { ptr: std::ptr::null(), len: 0 }
    }
    fn to_vec(&self) -> Vec<T> {
        if self.len == 0 {
            Vec::new()
        } else {
            // SAFETY: the C++ hands a pointer valid for len elements until
            // its next call
            unsafe { from_raw_parts(self.ptr, self.len) }.to_vec()
        }
    }
}

/// The C++ that HPresolve calls; `ctx` is the C++ RustPresolveHost
#[repr(C)]
pub struct Host {
    pub ctx: Ctx,
    /// (channel, HighsLogType, message): 0 highsLogUser, 1 highsLogDev,
    /// 3 printf
    pub log: extern "C" fn(Ctx, i32, i32, *const c_char),
    pub timer_read: extern "C" fn(Ctx) -> f64,
    /// highsTimeSecondToString(t) into the buffer
    pub time_string: extern "C" fn(Ctx, f64, *mut c_char, usize),
    /// analysis_.setup: fills allow_rule, returns allow_logging_
    pub analysis_setup: extern "C" fn(Ctx, bool, *mut u8) -> bool,
    pub sync_model: extern "C" fn(Ctx, *const CModel, bool),
    pub set_matrix: extern "C" fn(Ctx, *const i32, usize, *const i32, *const f64, usize),
    /// appends records, (type, position) pairs and the not linearly
    /// transformable columns to the postsolve stack
    pub flush: extern "C" fn(Ctx, *const u8, usize, *const u8, *const usize, usize, *const i32, usize),
    pub shrink: extern "C" fn(Ctx, *const i32, usize, *const i32, usize),
    pub dependent_equations: extern "C" fn(
        Ctx,
        usize,
        i32,
        *const i32,
        *const i32,
        *const f64,
        usize,
        f64,
        *mut f64,
        *mut CSlice<i32>,
    ) -> i32,
    pub clique_have_common_clique: extern "C" fn(Ctx, i32, i32, i32, i32) -> bool,
    pub clique_num_cliques_col: extern "C" fn(Ctx, i32, i32) -> i32,
    pub clique_num_cliques: extern "C" fn(Ctx) -> i32,
    pub clique_set_presolve_flag: extern "C" fn(Ctx, bool),
    pub clique_set_max_entries: extern "C" fn(Ctx, i32),
    pub implications_column_transformed: extern "C" fn(Ctx, i32, f64, f64),
    pub implications_add_vb: extern "C" fn(Ctx, bool, i32, i32, f64, f64, f64, bool),
    pub profiling: extern "C" fn(Ctx, bool, i32),
    pub probing_prepare: extern "C" fn(Ctx, i32, *mut bool) -> bool,
    pub probing_loop: extern "C" fn(Ctx, *mut ProbingIo) -> i32,
    /// cleanupFixed, extractCliques, runCliqueMerging: the deleted rows and
    /// the clique extensions (row, col, val)
    pub finalise_begin: extern "C" fn(Ctx, bool, *mut CSlice<i32>, *mut CSlice<i32>),
    pub domain_bounds: extern "C" fn(Ctx, *mut CSlice<f64>, *mut CSlice<f64>),
    pub impl_substitutions: extern "C" fn(Ctx, *mut CSlice<ImplSubst>),
    pub clear_impl_substitutions: extern "C" fn(Ctx),
    /// (substcol, replace.col, replace.val) triples
    pub clique_substitutions: extern "C" fn(Ctx, *mut CSlice<i32>),
    pub clear_clique_substitutions: extern "C" fn(Ctx),
    /// computeMaximalCliques of (col, val) pairs: the cliques as (col, val)
    /// pairs and their starts
    pub compute_maximal_cliques: extern "C" fn(Ctx, *const i32, usize, f64, *mut CSlice<i32>, *mut CSlice<usize>),
    pub enumerate: extern "C" fn(Ctx, *mut ProbingIo) -> bool,
    pub mip_finish_presolve: extern "C" fn(Ctx, i32),
    pub add_cut: extern "C" fn(Ctx, *const i32, *const f64, usize, f64, bool),
    pub upper_limit: extern "C" fn(Ctx) -> f64,
    pub set_lower_bound_zero: extern "C" fn(Ctx),
}

impl Host {
    pub(crate) fn log(&self, channel: i32, t: i32, msg: &str) {
        let c = CString::new(msg).unwrap_or_default();
        (self.log)(self.ctx, channel, t, c.as_ptr());
    }
    pub(crate) fn timer_read(&self) -> f64 {
        (self.timer_read)(self.ctx)
    }
    pub(crate) fn time_string(&self, t: f64) -> String {
        let mut buf = [0 as c_char; 128];
        (self.time_string)(self.ctx, t, buf.as_mut_ptr(), buf.len());
        // SAFETY: the C++ writes a NUL-terminated string into buf
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }
    pub(crate) fn analysis_setup(&self, silent: bool, allow: &mut [u8; RULE_COUNT]) -> bool {
        (self.analysis_setup)(self.ctx, silent, allow.as_mut_ptr())
    }
    pub(crate) fn sync_model(&self, p: &Presolve) {
        self.sync_model_ex(p, false);
    }
    pub(crate) fn sync_model_ex(&self, p: &Presolve, resize_row_names: bool) {
        let m = CModel {
            num_col: p.num_col,
            num_row: p.num_row,
            col_cost: p.col_cost.as_ptr(),
            col_lower: p.col_lower.as_ptr(),
            col_upper: p.col_upper.as_ptr(),
            row_lower: p.row_lower.as_ptr(),
            row_upper: p.row_upper.as_ptr(),
            integrality: p.integrality.as_ptr(),
            offset: p.offset,
            maximize: p.maximize,
        };
        (self.sync_model)(self.ctx, &m, resize_row_names);
    }
    pub(crate) fn set_matrix(&self, start: &[i32], index: &[i32], value: &[f64]) {
        (self.set_matrix)(self.ctx, start.as_ptr(), start.len(), index.as_ptr(), value.as_ptr(), value.len());
    }
    pub(crate) fn flush(&self, data: &[u8], reductions: &[(u8, usize)], not_transformable: &[i32]) {
        let types: Vec<u8> = reductions.iter().map(|r| r.0).collect();
        let pos: Vec<usize> = reductions.iter().map(|r| r.1).collect();
        (self.flush)(
            self.ctx,
            data.as_ptr(),
            data.len(),
            types.as_ptr(),
            pos.as_ptr(),
            reductions.len(),
            not_transformable.as_ptr(),
            not_transformable.len(),
        );
    }
    pub(crate) fn shrink(&self, new_col: &[i32], new_row: &[i32]) {
        (self.shrink)(self.ctx, new_col.as_ptr(), new_col.len(), new_row.as_ptr(), new_row.len());
    }
    /// (build return, time taken, var_with_no_pivot)
    pub(crate) fn dependent_equations(
        &self,
        num_col: usize,
        num_row: i32,
        start: &[i32],
        index: &[i32],
        value: &[f64],
        time_limit: f64,
    ) -> (i32, f64, Vec<i32>) {
        let mut time_taken = 0.0;
        let mut v = CSlice::empty();
        let r = (self.dependent_equations)(
            self.ctx,
            num_col,
            num_row,
            start.as_ptr(),
            index.as_ptr(),
            value.as_ptr(),
            value.len(),
            time_limit,
            &mut time_taken,
            &mut v,
        );
        (r, time_taken, v.to_vec())
    }
    pub(crate) fn clique_have_common_clique(&self, c1: i32, v1: i32, c2: i32, v2: i32) -> bool {
        (self.clique_have_common_clique)(self.ctx, c1, v1, c2, v2)
    }
    pub(crate) fn clique_num_cliques_col(&self, col: i32, val: i32) -> i32 {
        (self.clique_num_cliques_col)(self.ctx, col, val)
    }
    pub(crate) fn clique_num_cliques(&self) -> i32 {
        (self.clique_num_cliques)(self.ctx)
    }
    pub(crate) fn clique_set_presolve_flag(&self, f: bool) {
        (self.clique_set_presolve_flag)(self.ctx, f)
    }
    pub(crate) fn clique_set_max_entries(&self, n: i32) {
        (self.clique_set_max_entries)(self.ctx, n)
    }
    pub(crate) fn implications_column_transformed(&self, col: i32, scale: f64, constant: f64) {
        (self.implications_column_transformed)(self.ctx, col, scale, constant)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn implications_add_vb(
        &self,
        is_vub: bool,
        col: i32,
        bin_col: i32,
        coef: f64,
        constant: f64,
        bound: f64,
        is_int: bool,
    ) {
        (self.implications_add_vb)(self.ctx, is_vub, col, bin_col, coef, constant, bound, is_int)
    }
    pub(crate) fn profiling(&self, start: bool, clock: i32) {
        (self.profiling)(self.ctx, start, clock)
    }
    /// (infeasible, firstCall); offset unused (the model is synced)
    pub(crate) fn probing_prepare(&self, nnz: i32, _offset: f64) -> (bool, bool) {
        let mut first_call = false;
        let inf = (self.probing_prepare)(self.ctx, nnz, &mut first_call);
        (inf, first_call)
    }
    fn probing_io(p: &mut Presolve) -> ProbingIo {
        ProbingIo {
            num_probes: p.num_probes.as_mut_ptr(),
            num_col: p.num_col,
            num_row: p.num_row,
            num_nonzeros: p.num_nonzeros(),
            col_deleted: p.col_deleted.as_ptr(),
            integrality: p.integrality.as_ptr(),
            colsize_len: p.colsize.len(),
            probing_contingent: &mut p.probing_contingent,
            num_probed: &mut p.num_probed,
            probing_num_del_col: &mut p.probing_num_del_col,
            probing_early_abort: &mut p.probing_early_abort,
            lifting: std::ptr::null(),
            num_lifting: 0,
        }
    }
    /// the probing loop: (code, lifting opportunities (row, col, val, coef))
    pub(crate) fn probing_loop(&self, p: &mut Presolve) -> (i32, Vec<(i32, (i32, i32), f64)>) {
        let mut io = Self::probing_io(p);
        let code = (self.probing_loop)(self.ctx, &mut io);
        let lifting = CSlice { ptr: io.lifting, len: io.num_lifting }.to_vec();
        (code, lifting.iter().map(|l| (l.row, (l.col, l.val), l.coef)).collect())
    }
    pub(crate) fn enumerate(&self, p: &mut Presolve) -> bool {
        let mut io = Self::probing_io(p);
        (self.enumerate)(self.ctx, &mut io)
    }
    /// (deleted rows, clique extensions (row, col, val))
    pub(crate) fn finalise_begin(&self, first_call: bool) -> (Vec<i32>, Vec<(i32, i32, i32)>) {
        let mut d = CSlice::empty();
        let mut e = CSlice::empty();
        (self.finalise_begin)(self.ctx, first_call, &mut d, &mut e);
        let ext = e.to_vec();
        (d.to_vec(), ext.chunks(3).map(|c| (c[0], c[1], c[2])).collect())
    }
    pub(crate) fn domain_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let mut l = CSlice::empty();
        let mut u = CSlice::empty();
        (self.domain_bounds)(self.ctx, &mut l, &mut u);
        (l.to_vec(), u.to_vec())
    }
    pub(crate) fn impl_substitutions(&self) -> Vec<ImplSubst> {
        let mut s = CSlice::empty();
        (self.impl_substitutions)(self.ctx, &mut s);
        s.to_vec()
    }
    pub(crate) fn clear_impl_substitutions(&self) {
        (self.clear_impl_substitutions)(self.ctx)
    }
    pub(crate) fn clique_substitutions(&self) -> Vec<(i32, i32, i32)> {
        let mut s = CSlice::empty();
        (self.clique_substitutions)(self.ctx, &mut s);
        s.to_vec().chunks(3).map(|c| (c[0], c[1], c[2])).collect()
    }
    pub(crate) fn clear_clique_substitutions(&self) {
        (self.clear_clique_substitutions)(self.ctx)
    }
    pub(crate) fn compute_maximal_cliques(&self, cands: &[(i32, i32)], feastol: f64) -> Vec<Vec<(i32, i32)>> {
        let flat: Vec<i32> = cands.iter().flat_map(|&(c, v)| [c, v]).collect();
        let mut vars = CSlice::empty();
        let mut starts = CSlice::empty();
        (self.compute_maximal_cliques)(self.ctx, flat.as_ptr(), cands.len(), feastol, &mut vars, &mut starts);
        let vars = vars.to_vec();
        let starts = starts.to_vec();
        let mut out = Vec::new();
        for w in starts.windows(2) {
            out.push((w[0]..w[1]).map(|k| (vars[2 * k], vars[2 * k + 1])).collect());
        }
        out
    }
    pub(crate) fn mip_finish_presolve(&self, nnz: i32) {
        (self.mip_finish_presolve)(self.ctx, nnz)
    }
    pub(crate) fn add_cut(&self, inds: &[i32], vals: &[f64], rhs: f64, integral: bool) {
        (self.add_cut)(self.ctx, inds.as_ptr(), vals.as_ptr(), inds.len(), rhs, integral)
    }
    pub(crate) fn upper_limit(&self) -> f64 {
        (self.upper_limit)(self.ctx)
    }
    pub(crate) fn set_lower_bound_zero(&self) {
        (self.set_lower_bound_zero)(self.ctx)
    }
}

/// The model and postsolve stack state handed over by the C++
#[repr(C)]
pub struct CInput {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: *const f64,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    pub integrality: *const u8,
    pub offset: f64,
    pub maximize: bool,
    pub a_start: *const i32,
    pub a_index: *const i32,
    pub a_value: *const f64,
    pub num_nz: usize,
    pub orig_col_index: *const i32,
    pub num_orig_col_index: usize,
    pub orig_row_index: *const i32,
    pub num_orig_row_index: usize,
    pub stack_data_size: usize,
    pub num_reductions: usize,
    pub presolve_reduction_limit: i32,
    pub model_name: *const c_char,
}

/// The results for the C++ HPresolve
#[repr(C)]
pub struct COut {
    pub presolve_status: i32,
    /// (call, col_removed, row_removed) per rule
    pub log: [[i32; 3]; RULE_COUNT],
}

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        from_raw_parts(p, n)
    }
}

/// HPresolve::run: presolves the model, writing it back to the C++ HighsLp
/// and the reductions to its postsolve stack through `host`; returns the
/// HighsModelStatus
///
/// # Safety
/// The pointers are valid for their lengths (the model vectors for its
/// dimensions, a_start for num_col + 1); mip may be null; host's callbacks
/// are valid for the call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_presolve_run(
    host: *const Host,
    opt: *const Options,
    mip: *const MipInfo,
    inp: *const CInput,
    out: *mut COut,
) -> i32 {
    let host = &*host;
    let i = &*inp;
    let nc = i.num_col as usize;
    let nr = i.num_row as usize;
    let model_name =
        if i.model_name.is_null() { String::new() } else { CStr::from_ptr(i.model_name).to_string_lossy().into_owned() };
    let input = Input {
        num_col: i.num_col,
        num_row: i.num_row,
        col_cost: sl(i.col_cost, nc),
        col_lower: sl(i.col_lower, nc),
        col_upper: sl(i.col_upper, nc),
        row_lower: sl(i.row_lower, nr),
        row_upper: sl(i.row_upper, nr),
        integrality: sl(i.integrality, nc),
        offset: i.offset,
        maximize: i.maximize,
        a_start: sl(i.a_start, nc + 1),
        a_index: sl(i.a_index, i.num_nz),
        a_value: sl(i.a_value, i.num_nz),
        orig_col_index: sl(i.orig_col_index, i.num_orig_col_index),
        orig_row_index: sl(i.orig_row_index, i.num_orig_row_index),
        stack_data_size: i.stack_data_size,
        num_reductions: i.num_reductions,
        presolve_reduction_limit: i.presolve_reduction_limit,
        model_name,
    };
    let mip = if mip.is_null() { None } else { Some(*mip) };
    let mut p = Presolve::new(host, *opt, mip, &input);
    let status = p.run();
    let o = &mut *out;
    o.presolve_status = p.presolve_status;
    for (k, l) in p.analysis.log.iter().enumerate() {
        o.log[k] = [l.0, l.1, l.2];
    }
    let _ = from_raw_parts_mut::<u8>;
    status
}
