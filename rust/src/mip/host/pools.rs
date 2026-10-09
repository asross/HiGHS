//! HighsCutPool and HighsConflictPool's shells (the pools are cutpool.rs
//! and conflictpool.rs): the propagation hooks, HighsCutSet with the
//! separation of the pool's cuts into it, addCut with the clique
//! extraction of the global pool, syncCutPool, and the conflict entries'
//! relaxation of addConflictCut.

use super::dom::{ConfPropS, CutPropS, DomS};
use super::solver::MipSolver;
use crate::mip::conflictpool::ffi as cfp;
use crate::mip::conflictpool::ConflictPool;
use crate::mip::cutpool::ffi as cpf;
use crate::mip::cutpool::{CMatrixView, CutPool, ModelSize};
use crate::mip::domain::{DomChg, LOWER};
use std::ffi::c_void;

/// HighsCutSet
#[derive(Default)]
pub struct CutSet {
    pub cutindices: Vec<i32>,
    pub cutpools: Vec<i32>,
    pub ar_start: Vec<i32>,
    pub ar_index: Vec<i32>,
    pub ar_value: Vec<f64>,
    /// only ever -inf
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl CutSet {
    pub fn num_cuts(&self) -> usize {
        self.cutindices.len()
    }
    pub fn resize(&mut self, nnz: usize) {
        let n = self.num_cuts();
        self.lower.resize(n, -f64::INFINITY);
        self.upper.resize(n, 0.0);
        self.ar_start.resize(n + 1, 0);
        self.ar_index.resize(nnz, 0);
        self.ar_value.resize(nnz, 0.0);
    }
    pub fn clear(&mut self) {
        self.cutindices.clear();
        self.cutpools.clear();
        self.upper.clear();
        self.ar_start.clear();
        self.ar_index.clear();
        self.ar_value.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.cutindices.is_empty()
    }
}

/// HighsCutPool (a handle of the Rust pool)
pub struct CutPoolS {
    rs: *mut CutPool,
    pub index: i32,
}

// SAFETY: the pool's thread safe calls touch only atomics, as the C++
unsafe impl Send for CutPoolS {}

unsafe extern "C" fn cut_added(d: *mut c_void, cut: i32, propagate: bool) {
    (*(d as *mut CutPropS)).cut_added(cut, propagate);
}
unsafe extern "C" fn cut_deleted(d: *mut c_void, cut: i32, only_for_propagation: bool) {
    (*(d as *mut CutPropS)).cut_deleted(cut, only_for_propagation);
}

impl CutPoolS {
    pub fn new(ncols: i32, agelim: i32, softlimit: i32, index: i32) -> Box<CutPoolS> {
        Box::new(CutPoolS { rs: cpf::highs_rs_cutpool_new(ncols, agelim, softlimit, index, cut_added, cut_deleted), index })
    }
    /// The move assignment of a new pool (the old one freed)
    pub fn replace(&mut self, ncols: i32, agelim: i32, softlimit: i32, index: i32) {
        let old = std::mem::replace(&mut self.rs, cpf::highs_rs_cutpool_new(ncols, agelim, softlimit, index, cut_added, cut_deleted));
        self.index = index;
        // SAFETY: the old pool, no longer referred to
        unsafe { cpf::highs_rs_cutpool_free(old) };
    }
    pub fn rs(&self) -> *mut CutPool {
        self.rs
    }
    pub fn pool(&self) -> &CutPool {
        // SAFETY: the live pool
        unsafe { &*self.rs }
    }
    fn op(&self, which: i32, i: i32, j: i32, d: *mut c_void) {
        // SAFETY: the live pool
        unsafe { cpf::highs_rs_cutpool_op(self.rs, which, i, j, d) }
    }
    pub fn view(&self) -> CMatrixView {
        let mut v = std::mem::MaybeUninit::<CMatrixView>::uninit();
        // SAFETY: the live pool; the view written whole
        unsafe {
            cpf::highs_rs_cutpool_view(self.rs, v.as_mut_ptr());
            v.assume_init()
        }
    }
    pub fn reset_age(&self, cut: i32, thread_safe: bool) {
        self.op(0, cut, thread_safe as i32, std::ptr::null_mut());
    }
    pub fn lp_cut_removed(&self, cut: i32, thread_safe: bool) {
        self.op(1, cut, thread_safe as i32, std::ptr::null_mut());
    }
    pub fn increase_num_lps(&self, cut: i32, n: i32) {
        self.op(2, cut, n, std::ptr::null_mut());
    }
    pub fn set_age_limit(&self, agelim: i32) {
        self.op(3, agelim, 0, std::ptr::null_mut());
    }
    pub fn perform_aging(&self) {
        self.op(4, 0, 0, std::ptr::null_mut());
    }
    pub fn add_propagation_domain(&self, d: *mut CutPropS) {
        self.op(5, 0, 0, d as *mut c_void);
    }
    pub fn remove_propagation_domain(&self, d: *mut CutPropS) {
        self.op(6, 0, 0, d as *mut c_void);
    }
    pub fn num_cuts(&self) -> i32 {
        // SAFETY: the live pool
        unsafe { cpf::highs_rs_cutpool_geti(self.rs, 0, 0) }
    }
    pub fn num_available_cuts(&self) -> i32 {
        // SAFETY: as num_cuts
        unsafe { cpf::highs_rs_cutpool_geti(self.rs, 1, 0) }
    }
    pub fn row_length(&self, row: i32) -> i32 {
        // SAFETY: as num_cuts
        unsafe { cpf::highs_rs_cutpool_geti(self.rs, 2, row) }
    }
    pub fn cut_is_integral(&self, cut: i32) -> bool {
        // SAFETY: as num_cuts
        unsafe { cpf::highs_rs_cutpool_geti(self.rs, 3, cut) != 0 }
    }
    pub fn max_abs_cut_coef(&self, cut: i32) -> f64 {
        // SAFETY: as num_cuts
        unsafe { cpf::highs_rs_cutpool_getd(self.rs, 0, cut) }
    }
    /// getCut: (indices, values), valid until the pool changes
    pub fn get_cut(&self, cut: i32) -> (&[i32], &[f64]) {
        let v = self.view();
        // SAFETY: the pool's arrays
        unsafe {
            let r = *v.ar_range.add(cut as usize);
            let (s, e) = (r[0] as usize, r[1] as usize);
            (
                std::slice::from_raw_parts(v.ar_index.add(s), e - s),
                std::slice::from_raw_parts(v.ar_value.add(s), e - s),
            )
        }
    }
    pub fn rhs(&self, cut: i32) -> f64 {
        let v = self.view();
        // SAFETY: the pool's right-hand sides
        unsafe { *v.rhs.add(cut as usize) }
    }

    /// appendCuts: the given cuts of this pool to the cut set
    fn append_cuts(&self, cutset: &mut CutSet, cuts: &[i32], mut nnz: usize) {
        let orignumcuts = cutset.num_cuts();
        let mut offset = cutset.ar_index.len();
        for &c in cuts {
            cutset.cutindices.push(c);
            cutset.cutpools.push(self.index);
        }
        if nnz == 0 {
            nnz = offset;
            for &c in cuts {
                nnz += self.row_length(c) as usize;
            }
        }
        cutset.resize(nnz);
        let v = self.view();
        for i in orignumcuts..cutset.num_cuts() {
            cutset.ar_start[i] = offset as i32;
            let cut = cutset.cutindices[i] as usize;
            // SAFETY: the pool's arrays
            unsafe {
                let r = *v.ar_range.add(cut);
                cutset.upper[i] = *v.rhs.add(cut);
                for j in r[0] as usize..r[1] as usize {
                    cutset.ar_value[offset] = *v.ar_value.add(j);
                    cutset.ar_index[offset] = *v.ar_index.add(j);
                    offset += 1;
                }
            }
        }
        let n = cutset.num_cuts();
        cutset.ar_start[n] = offset as i32;
    }

    /// separate(sol, domain, cutset, feastol, cutpools, thread_safe)
    pub fn separate(
        &self,
        sol: &[f64],
        domain: &DomS,
        cutset: &mut CutSet,
        feastol: f64,
        cutpools: &[super::Own<CutPoolS>],
        thread_safe: bool,
    ) {
        let pools: Vec<*const CutPool> = cutpools.iter().map(|p| p.rs as *const CutPool).collect();
        let mut num = 0;
        // SAFETY: the live pools and the vectors of the given lengths
        let cuts = unsafe {
            cpf::highs_rs_cutpool_separate(
                self.rs,
                sol.as_ptr(),
                sol.len() as i32,
                domain.col_lower().as_ptr(),
                domain.col_upper().as_ptr(),
                cutset.cutindices.as_ptr(),
                cutset.cutpools.as_ptr(),
                cutset.num_cuts() as i32,
                feastol,
                pools.as_ptr(),
                pools.len() as i32,
                thread_safe,
                &mut num,
            )
        };
        if num == -1 {
            return;
        }
        // SAFETY: the buffer of num cuts
        let c = unsafe { crate::ffi::sl(cuts, num) }.to_vec();
        self.append_cuts(cutset, &c, 0);
        // SAFETY: the buffer from separate
        unsafe { cpf::highs_rs_cutpool_free_buf(cuts as *mut i32, num) };
    }

    /// separateLpCutsAfterRestart
    pub fn separate_lp_cuts_after_restart(&self, cutset: &mut CutSet) {
        let v = self.view();
        self.op(7, 0, 0, std::ptr::null_mut());
        let cuts: Vec<i32> = (0..v.num_rows).collect();
        self.append_cuts(cutset, &cuts, v.num_nz as usize);
    }

    /// addCut(mipsolver, ...): the row index or -1
    #[allow(clippy::too_many_arguments)]
    pub fn add_cut(
        &self,
        ms: &MipSolver,
        index: &mut [i32],
        value: &mut [f64],
        rhs: f64,
        integral: bool,
        propagate: bool,
        extract_cliques: bool,
        is_conflict: bool,
    ) -> i32 {
        let global = ms.d().get_cut_pool();
        let is_global = std::ptr::eq(self, global);
        let model = ModelSize { num_nonzero: ms.num_nonzero(), num_row: ms.num_row() };
        let len = index.len() as i32;
        // SAFETY: the live pools, the arrays of len entries
        let row = unsafe {
            cpf::highs_rs_cutpool_add_cut(
                self.rs,
                if is_global { std::ptr::null() } else { global.rs },
                &model,
                index.as_mut_ptr(),
                value.as_mut_ptr(),
                len,
                rhs,
                integral,
                propagate,
                is_conflict,
            )
        };
        if row == -1 {
            return -1;
        }
        if extract_cliques && is_global && len <= 100 {
            super::tables::extract_cliques_from_cut(ms, index, value, rhs);
        }
        row
    }

    /// syncCutPool(mipsolver, syncpool)
    pub fn sync_cut_pool(&self, ms: &MipSolver, syncpool: &CutPoolS) {
        let mut num = 0;
        // SAFETY: the live pool
        let cuts = unsafe { cpf::highs_rs_cutpool_cuts_to_sync(self.rs, &mut num) };
        // SAFETY: the buffer of num cuts
        let list = unsafe { crate::ffi::sl(cuts, num) }.to_vec();
        for i in list {
            let (idx, val) = self.get_cut(i);
            let (mut idxs, mut vals) = (idx.to_vec(), val.to_vec());
            let rhs = self.rhs(i);
            let integral = self.cut_is_integral(i);
            syncpool.add_cut(ms, &mut idxs, &mut vals, rhs, integral, true, true, false);
        }
        // SAFETY: the buffer from cuts_to_sync
        unsafe { cpf::highs_rs_cutpool_free_buf(cuts, num) };
    }
}

impl Drop for CutPoolS {
    fn drop(&mut self) {
        // SAFETY: the owned pool
        unsafe { cpf::highs_rs_cutpool_free(self.rs) };
    }
}

/// HighsConflictPool (a handle of the Rust pool)
pub struct ConflictPoolS {
    rs: *mut ConflictPool,
}

// SAFETY: as CutPoolS
unsafe impl Send for ConflictPoolS {}

unsafe extern "C" fn conflict_added(d: *mut c_void, conflict: i32) {
    (*(d as *mut ConfPropS)).conflict_added(conflict);
}
unsafe extern "C" fn conflict_deleted(d: *mut c_void, conflict: i32) {
    (*(d as *mut ConfPropS)).conflict_deleted(conflict);
}

impl ConflictPoolS {
    pub fn new(agelim: i32, softlimit: i32) -> Box<ConflictPoolS> {
        Box::new(ConflictPoolS { rs: cfp::highs_rs_conflictpool_new(agelim, softlimit, conflict_added, conflict_deleted) })
    }
    /// The move assignment of a new pool
    pub fn replace(&mut self, agelim: i32, softlimit: i32) {
        let old = std::mem::replace(
            &mut self.rs,
            cfp::highs_rs_conflictpool_new(agelim, softlimit, conflict_added, conflict_deleted),
        );
        // SAFETY: the old pool, no longer referred to
        unsafe { cfp::highs_rs_conflictpool_free(old) };
    }
    pub fn rs(&self) -> *mut ConflictPool {
        self.rs
    }
    fn op(&self, which: i32, i: i32, d: *mut c_void) {
        // SAFETY: the live pool
        unsafe { cfp::highs_rs_conflictpool_op(self.rs, which, i, d) }
    }
    pub fn reset_age(&self, conflict: i32) {
        self.op(0, conflict, std::ptr::null_mut());
    }
    pub fn set_age_limit(&self, agelim: i32) {
        self.op(1, agelim, std::ptr::null_mut());
    }
    pub fn set_age_lock(&self, lock: bool) {
        self.op(2, lock as i32, std::ptr::null_mut());
    }
    pub fn perform_aging(&self, thread_safe: bool) {
        self.op(3, thread_safe as i32, std::ptr::null_mut());
    }
    pub fn remove_conflict(&self, conflict: i32) {
        self.op(4, conflict, std::ptr::null_mut());
    }
    pub fn add_propagation_domain(&self, d: *mut ConfPropS) {
        self.op(5, 0, d as *mut c_void);
    }
    pub fn remove_propagation_domain(&self, d: *mut ConfPropS) {
        self.op(6, 0, d as *mut c_void);
    }
    pub fn num_conflicts(&self) -> i32 {
        // SAFETY: the live pool
        unsafe { cfp::highs_rs_conflictpool_get(self.rs, 0, 0) as i32 }
    }
    pub fn sync_conflict_pool(&self, syncpool: &ConflictPoolS) {
        // SAFETY: distinct live pools
        unsafe { cfp::highs_rs_conflictpool_sync(self.rs, syncpool.rs) }
    }

    /// The entries of a conflict: continuous bounds relaxed by the
    /// tolerance
    fn relaxed_entries(domain: &DomS, entries: &[DomChg]) -> Vec<DomChg> {
        let feastol = domain.feastol();
        // SAFETY: the domain's solver
        let ms = unsafe { &*domain.mipsolver };
        let mut relaxed = entries.to_vec();
        for e in &mut relaxed {
            if ms.is_col_continuous(e.column) {
                if e.boundtype == LOWER {
                    e.boundval += feastol;
                } else {
                    e.boundval -= feastol;
                }
            }
        }
        relaxed
    }

    /// addConflictCut(domain, entries)
    pub fn add_conflict_cut(&self, domain: &DomS, entries: &[DomChg]) {
        let relaxed = Self::relaxed_entries(domain, entries);
        // SAFETY: the live pool
        unsafe { cfp::highs_rs_conflictpool_add(self.rs, relaxed.as_ptr(), relaxed.len() as i32, std::ptr::null()) };
    }

    /// addReconvergenceCut(domain, entries, reconvergenceDomchg)
    pub fn add_reconvergence_cut(&self, domain: &DomS, entries: &[DomChg], reconvergence: &DomChg) {
        let flipped = domain.flip(reconvergence);
        let relaxed = Self::relaxed_entries(domain, entries);
        // SAFETY: the live pool
        unsafe { cfp::highs_rs_conflictpool_add(self.rs, relaxed.as_ptr(), relaxed.len() as i32, &flipped) };
    }
}

impl Drop for ConflictPoolS {
    fn drop(&mut self) {
        // SAFETY: the owned pool
        unsafe { cfp::highs_rs_conflictpool_free(self.rs) };
    }
}
