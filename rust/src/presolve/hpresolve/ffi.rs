//! The boundary with the C++ (highs/presolve/HPresolveRust.cpp): the entry
//! point `highs_rs_presolve_run` and the C++ callbacks of [`Host`].

use super::driver::Input;
use super::{MipInfo, Options, Presolve, RULE_COUNT};
use crate::mip::clique::{CDom, CMip};
use crate::mip::domain::CDomain;
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LiftOpp {
    pub row: i32,
    pub col: i32,
    pub val: i32,
    pub coef: f64,
}

/// The global domain's view and the clique table's contexts of the probing
/// and enumeration loops, valid until the next presolve reduction
#[repr(C)]
pub struct MipEnv {
    pub domain: *const CDomain,
    pub cdom: CDom,
    pub cmip: CMip,
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
    pub profiling: extern "C" fn(Ctx, bool, i32),
    pub probing_prepare: extern "C" fn(Ctx, i32, *mut bool) -> bool,
    pub mip_env: extern "C" fn(Ctx, *mut MipEnv),
    /// implications.runProbing(col, numBoundChgs)
    pub probe: extern "C" fn(Ctx, i32, *mut i32) -> bool,
    /// start collecting the lifting opportunities of probing (true) or stop
    pub set_lifting: extern "C" fn(Ctx, bool),
    /// the lifting opportunities collected
    pub lifting_opps: extern "C" fn(Ctx, *mut CSlice<LiftOpp>),
    /// cleanupFixed, extractCliques, runCliqueMerging: the deleted rows and
    /// the clique extensions (row, col, val)
    pub finalise_begin: extern "C" fn(Ctx, bool, *mut CSlice<i32>, *mut CSlice<i32>),
    pub domain_bounds: extern "C" fn(Ctx, *mut CSlice<f64>, *mut CSlice<f64>),
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
    pub(crate) fn profiling(&self, start: bool, clock: i32) {
        (self.profiling)(self.ctx, start, clock)
    }
    /// (infeasible, firstCall); offset unused (the model is synced)
    pub(crate) fn probing_prepare(&self, nnz: i32, _offset: f64) -> (bool, bool) {
        let mut first_call = false;
        let inf = (self.probing_prepare)(self.ctx, nnz, &mut first_call);
        (inf, first_call)
    }
    pub(crate) fn mip_env(&self) -> MipEnv {
        let mut env = std::mem::MaybeUninit::<MipEnv>::uninit();
        (self.mip_env)(self.ctx, env.as_mut_ptr());
        // SAFETY: the C++ fills every field
        unsafe { env.assume_init() }
    }
    pub(crate) fn probe(&self, col: i32, num_bound_chgs: &mut i32) -> bool {
        (self.probe)(self.ctx, col, num_bound_chgs)
    }
    pub(crate) fn set_lifting(&self, on: bool) {
        (self.set_lifting)(self.ctx, on)
    }
    /// the lifting opportunities collected: (row, (col, val), coef)
    pub(crate) fn lifting_opps(&self) -> Vec<(i32, (i32, i32), f64)> {
        let mut s = CSlice::empty();
        (self.lifting_opps)(self.ctx, &mut s);
        s.to_vec().iter().map(|l| (l.row, (l.col, l.val), l.coef)).collect()
    }
    /// the number of lifting opportunities collected
    pub(crate) fn num_lifting_opps(&self) -> usize {
        let mut s = CSlice::empty();
        (self.lifting_opps)(self.ctx, &mut s);
        s.len
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
