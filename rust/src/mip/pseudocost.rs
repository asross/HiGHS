//! HighsPseudocost (highs/mip/HighsPseudocost.h): the pseudocosts, inference
//! and cutoff statistics and conflict scores of the branching columns, and
//! the deltas a worker's copy collects for flushing into the master copy.
//! Rust-owned; the C++ class is a handle (HighsPseudocost.cpp under
//! HIGHS_RUST). clang contracts `cost += (1 - w) * avg`, `avg * count +
//! sum` and the products in the score sums (see `get_score`).

use crate::util::fma::ClangFma;

pub const MIN_THRESHOLD: f64 = 1e-6;

/// HighsPseudocostDelta
#[derive(Clone, Copy, Default)]
pub struct Delta {
    col: i32,
    nsamplesup: i32,
    nsamplesdown: i32,
    ninferencesup: i32,
    ninferencesdown: i32,
    ncutoffsup: i32,
    ncutoffsdown: i32,
    pseudocostup_sum: f64,
    pseudocostdown_sum: f64,
    inferencesup_sum: f64,
    inferencesdown_sum: f64,
    conflictscoreup_sum: f64,
    conflictscoredown_sum: f64,
}

#[derive(Clone, Default)]
pub struct Pseudocost {
    pub pseudocostup: Vec<f64>,
    pub pseudocostdown: Vec<f64>,
    pub nsamplesup: Vec<i32>,
    pub nsamplesdown: Vec<i32>,
    pub inferencesup: Vec<f64>,
    pub inferencesdown: Vec<f64>,
    pub ninferencesup: Vec<i32>,
    pub ninferencesdown: Vec<i32>,
    pub ncutoffsup: Vec<i32>,
    pub ncutoffsdown: Vec<i32>,
    pub conflictscoreup: Vec<f64>,
    pub conflictscoredown: Vec<f64>,
    changedpos: Vec<i32>,
    deltas: Vec<Delta>,
    pub conflict_weight: f64,
    pub conflict_avg_score: f64,
    pub cost_total: f64,
    pub inferences_total: f64,
    delta_cost_sum: f64,
    delta_inferences_sum: f64,
    pub nsamplestotal: i64,
    pub ninferencestotal: i64,
    pub ncutoffstotal: i64,
    delta_nsamplestotal: i64,
    delta_ninferencestotal: i64,
    pub minreliable: i32,
    pub degeneracy_factor: f64,
}

/// 1 - 1 / (1 + score)
#[inline(always)]
fn map_score(score: f64) -> f64 {
    1.0 - 1.0 / (1.0 + score)
}

/// std::max(a, b)
#[inline(always)]
fn max2(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// addDeltaAverage
#[inline(always)]
fn add_delta_average<C: Copy + Into<i64> + std::ops::AddAssign + std::ops::Add<Output = C>>(
    avg: &mut f64,
    count: &mut C,
    delta_sum: f64,
    delta_count: C,
) {
    if delta_count.into() <= 0 {
        return;
    }
    *avg = avg.mul_add_c((*count).into() as f64, delta_sum) / (*count + delta_count).into() as f64;
    *count += delta_count;
}

impl Pseudocost {
    /// HighsPseudocost(mipsolver) without the initialization
    pub fn new(ncol: i32, minreliable: i32) -> Self {
        let n = ncol.max(0) as usize;
        Pseudocost {
            pseudocostup: vec![0.0; n],
            pseudocostdown: vec![0.0; n],
            nsamplesup: vec![0; n],
            nsamplesdown: vec![0; n],
            inferencesup: vec![0.0; n],
            inferencesdown: vec![0.0; n],
            ninferencesup: vec![0; n],
            ninferencesdown: vec![0; n],
            ncutoffsup: vec![0; n],
            ncutoffsdown: vec![0; n],
            conflictscoreup: vec![0.0; n],
            conflictscoredown: vec![0.0; n],
            changedpos: vec![-1; n],
            deltas: Vec::with_capacity(n.min(256)),
            conflict_weight: 1.0,
            minreliable,
            degeneracy_factor: 1.0,
            ..Default::default()
        }
    }

    fn mark_changed(&mut self, col: i32) -> &mut Delta {
        let c = col as usize;
        if self.changedpos[c] == -1 {
            self.changedpos[c] = self.deltas.len() as i32;
            self.deltas.push(Delta { col, ..Default::default() });
        }
        &mut self.deltas[self.changedpos[c] as usize]
    }

    pub fn increase_conflict_weight(&mut self) {
        self.conflict_weight *= 1.02;
        if self.conflict_weight > 1000.0 {
            let scale = 1.0 / self.conflict_weight;
            self.conflict_weight = 1.0;
            self.conflict_avg_score *= scale;
            for i in 0..self.conflictscoreup.len() {
                self.conflictscoreup[i] *= scale;
                self.conflictscoredown[i] *= scale;
            }
            for d in &mut self.deltas {
                d.conflictscoreup_sum *= scale;
                d.conflictscoredown_sum *= scale;
            }
        }
    }

    pub fn increase_conflict_score(&mut self, col: i32, up: bool) {
        let w = self.conflict_weight;
        let d = self.mark_changed(col);
        if up {
            d.conflictscoreup_sum += w;
            self.conflictscoreup[col as usize] += w;
        } else {
            d.conflictscoredown_sum += w;
            self.conflictscoredown[col as usize] += w;
        }
        self.conflict_avg_score += w;
    }

    pub fn add_cutoff_observation(&mut self, col: i32, upbranch: bool) {
        let d = self.mark_changed(col);
        if upbranch {
            d.ncutoffsup += 1;
        } else {
            d.ncutoffsdown += 1;
        }
        self.ncutoffstotal += 1;
        if upbranch {
            self.ncutoffsup[col as usize] += 1;
        } else {
            self.ncutoffsdown[col as usize] += 1;
        }
    }

    pub fn add_observation(&mut self, col: i32, delta: f64, objdelta: f64) {
        debug_assert!(delta != 0.0 && objdelta >= 0.0);
        let c = col as usize;
        let unit_gain;
        if delta > 0.0 {
            unit_gain = objdelta / delta;
            let d = unit_gain - self.pseudocostup[c];
            self.nsamplesup[c] += 1;
            self.pseudocostup[c] += d / self.nsamplesup[c] as f64;
            let pd = self.mark_changed(col);
            pd.nsamplesup += 1;
            pd.pseudocostup_sum += unit_gain;
        } else {
            unit_gain = -objdelta / delta;
            let d = unit_gain - self.pseudocostdown[c];
            self.nsamplesdown[c] += 1;
            self.pseudocostdown[c] += d / self.nsamplesdown[c] as f64;
            let pd = self.mark_changed(col);
            pd.nsamplesdown += 1;
            pd.pseudocostdown_sum += unit_gain;
        }
        let d = unit_gain - self.cost_total;
        self.nsamplestotal += 1;
        self.cost_total += d / self.nsamplestotal as f64;
        self.delta_nsamplestotal += 1;
        self.delta_cost_sum += unit_gain;
    }

    pub fn add_inference_observation(&mut self, col: i32, ninferences: i32, upbranch: bool) {
        let c = col as usize;
        let n = ninferences as f64;
        let d = n - self.inferences_total;
        self.ninferencestotal += 1;
        self.inferences_total += d / self.ninferencestotal as f64;
        self.delta_ninferencestotal += 1;
        self.delta_inferences_sum += n;
        if upbranch {
            let d = n - self.inferencesup[c];
            self.ninferencesup[c] += 1;
            self.inferencesup[c] += d / self.ninferencesup[c] as f64;
            let pd = self.mark_changed(col);
            pd.ninferencesup += 1;
            pd.inferencesup_sum += n;
        } else {
            let d = n - self.inferencesdown[c];
            self.ninferencesdown[c] += 1;
            self.inferencesdown[c] += d / self.ninferencesdown[c] as f64;
            let pd = self.mark_changed(col);
            pd.ninferencesdown += 1;
            pd.inferencesdown_sum += n;
        }
    }

    #[inline]
    pub fn is_reliable(&self, col: i32) -> bool {
        let c = col as usize;
        self.nsamplesup[c].min(self.nsamplesdown[c]) >= self.minreliable
    }

    #[inline]
    pub fn is_reliable_up(&self, col: i32) -> bool {
        self.nsamplesup[col as usize] >= self.minreliable
    }

    #[inline]
    pub fn is_reliable_down(&self, col: i32) -> bool {
        self.nsamplesdown[col as usize] >= self.minreliable
    }

    /// getPseudocostUp/Down(col, frac, offset): the weighted average of the
    /// column's and the average pseudocost until it is reliable
    fn cost_offset(&self, n: i32, pc: f64, offset: f64) -> f64 {
        let cost = if n == 0 || n < self.minreliable {
            let weight = if n == 0 { 0.0 } else { 0.9 + 0.1 * n as f64 / self.minreliable as f64 };
            let cost = weight * pc;
            (1.0 - weight).mul_add_c(self.cost_total, cost)
        } else {
            pc
        };
        offset + cost
    }

    pub fn get_pseudocost_up_offset(&self, col: i32, frac: f64, offset: f64) -> f64 {
        let c = col as usize;
        let up = frac.ceil() - frac;
        up * self.cost_offset(self.nsamplesup[c], self.pseudocostup[c], offset)
    }

    pub fn get_pseudocost_down_offset(&self, col: i32, frac: f64, offset: f64) -> f64 {
        let c = col as usize;
        let down = frac - frac.floor();
        down * self.cost_offset(self.nsamplesdown[c], self.pseudocostdown[c], offset)
    }

    #[inline]
    pub fn get_pseudocost_up(&self, col: i32, frac: f64) -> f64 {
        let c = col as usize;
        let up = frac.ceil() - frac;
        if self.nsamplesup[c] == 0 {
            return up * self.cost_total;
        }
        up * self.pseudocostup[c]
    }

    #[inline]
    pub fn get_pseudocost_down(&self, col: i32, frac: f64) -> f64 {
        let c = col as usize;
        let down = frac - frac.floor();
        if self.nsamplesdown[c] == 0 {
            return down * self.cost_total;
        }
        down * self.pseudocostdown[c]
    }

    pub fn get_conflict_score_up(&self, col: i32) -> f64 {
        self.conflictscoreup[col as usize] / self.conflict_weight
    }

    pub fn get_conflict_score_down(&self, col: i32) -> f64 {
        self.conflictscoredown[col as usize] / self.conflict_weight
    }

    #[inline(always)]
    fn avg_cutoffs(&self) -> f64 {
        self.ncutoffstotal as f64 / max2(1.0, self.ncutoffstotal as f64 + self.nsamplestotal as f64)
    }

    #[inline(always)]
    fn conflict_score_avg(&self) -> f64 {
        self.conflict_avg_score / (self.conflict_weight * self.conflictscoreup.len() as f64)
    }

    /// getScore(col, upcost, downcost); clang fuses `1e-2 * a + (1e-4 *
    /// b)` and `p / f + f * q`
    pub fn get_score_costs(&self, col: i32, upcost: f64, downcost: f64) -> f64 {
        let c = col as usize;
        let t = MIN_THRESHOLD;
        let cost_score = max2(upcost, t) * max2(downcost, t) / max2(t, self.cost_total * self.cost_total);
        let inference_score = max2(self.inferencesup[c], t) * max2(self.inferencesdown[c], t)
            / max2(t, self.inferences_total * self.inferences_total);
        let cutoff_up = self.ncutoffsup[c] as f64 / max2(1.0, self.ncutoffsup[c] as f64 + self.nsamplesup[c] as f64);
        let cutoff_down =
            self.ncutoffsdown[c] as f64 / max2(1.0, self.ncutoffsdown[c] as f64 + self.nsamplesdown[c] as f64);
        let avg_cutoffs = self.avg_cutoffs();
        let cutoff_score = max2(cutoff_up, t) * max2(cutoff_down, t) / max2(t, avg_cutoffs * avg_cutoffs);
        let conflict_up = self.conflictscoreup[c] / self.conflict_weight;
        let conflict_down = self.conflictscoredown[c] / self.conflict_weight;
        let conflict_avg = self.conflict_score_avg();
        let conflict_score = max2(conflict_up, t) * max2(conflict_down, t) / max2(t, conflict_avg * conflict_avg);
        let f = self.degeneracy_factor;
        let inner = 1e-2f64.mul_add_c(map_score(conflict_score), 1e-4 * (map_score(cutoff_score) + map_score(inference_score)));
        f.mul_add_c(inner, map_score(cost_score) / f)
    }

    pub fn get_score(&self, col: i32, frac: f64) -> f64 {
        self.get_score_costs(col, self.get_pseudocost_up(col, frac), self.get_pseudocost_down(col, frac))
    }

    /// getScoreUp (up) / getScoreDown
    pub fn get_score_dir(&self, col: i32, frac: f64, up: bool) -> f64 {
        let c = col as usize;
        let t = MIN_THRESHOLD;
        let (pc, inferences, ncutoffs, nsamples, conflict) = if up {
            (
                self.get_pseudocost_up(col, frac),
                self.inferencesup[c],
                self.ncutoffsup[c],
                self.nsamplesup[c],
                self.conflictscoreup[c],
            )
        } else {
            (
                self.get_pseudocost_down(col, frac),
                self.inferencesdown[c],
                self.ncutoffsdown[c],
                self.nsamplesdown[c],
                self.conflictscoredown[c],
            )
        };
        let cost_score = pc / max2(t, self.cost_total);
        let inference_score = inferences / max2(t, self.inferences_total);
        let cutoff = ncutoffs as f64 / max2(1.0, ncutoffs as f64 + nsamples as f64);
        let cutoff_score = cutoff / max2(t, self.avg_cutoffs());
        let conflict_score = (conflict / self.conflict_weight) / max2(t, self.conflict_score_avg());
        map_score(cost_score)
            + 1e-2f64.mul_add_c(map_score(conflict_score), 1e-4 * (map_score(cutoff_score) + map_score(inference_score)))
    }

    /// flushPseudoCost: adds the deltas of `other` to this, then clears them
    pub fn flush(&mut self, other: &mut Pseudocost) {
        for d in &other.deltas {
            let c = d.col as usize;
            add_delta_average(&mut self.pseudocostup[c], &mut self.nsamplesup[c], d.pseudocostup_sum, d.nsamplesup);
            add_delta_average(
                &mut self.pseudocostdown[c],
                &mut self.nsamplesdown[c],
                d.pseudocostdown_sum,
                d.nsamplesdown,
            );
            add_delta_average(
                &mut self.inferencesup[c],
                &mut self.ninferencesup[c],
                d.inferencesup_sum,
                d.ninferencesup,
            );
            add_delta_average(
                &mut self.inferencesdown[c],
                &mut self.ninferencesdown[c],
                d.inferencesdown_sum,
                d.ninferencesdown,
            );
            let scale = self.conflict_weight / other.conflict_weight;
            let up = scale * d.conflictscoreup_sum;
            let down = scale * d.conflictscoredown_sum;
            self.conflictscoreup[c] += up;
            self.conflictscoredown[c] += down;
            self.conflict_avg_score += up + down;
            self.ncutoffsup[c] += d.ncutoffsup;
            self.ncutoffsdown[c] += d.ncutoffsdown;
            self.ncutoffstotal += (d.ncutoffsup + d.ncutoffsdown) as i64;
        }
        add_delta_average(&mut self.cost_total, &mut self.nsamplestotal, other.delta_cost_sum, other.delta_nsamplestotal);
        add_delta_average(
            &mut self.inferences_total,
            &mut self.ninferencestotal,
            other.delta_inferences_sum,
            other.delta_ninferencestotal,
        );
        other.remove_changed();
    }

    /// syncPseudoCost: copies this into `other` (same size)
    pub fn sync(&self, other: &mut Pseudocost) {
        other.pseudocostup.copy_from_slice(&self.pseudocostup);
        other.pseudocostdown.copy_from_slice(&self.pseudocostdown);
        other.nsamplesup.copy_from_slice(&self.nsamplesup);
        other.nsamplesdown.copy_from_slice(&self.nsamplesdown);
        other.inferencesup.copy_from_slice(&self.inferencesup);
        other.inferencesdown.copy_from_slice(&self.inferencesdown);
        other.ninferencesup.copy_from_slice(&self.ninferencesup);
        other.ninferencesdown.copy_from_slice(&self.ninferencesdown);
        other.ncutoffsup.copy_from_slice(&self.ncutoffsup);
        other.ncutoffsdown.copy_from_slice(&self.ncutoffsdown);
        other.conflictscoreup.copy_from_slice(&self.conflictscoreup);
        other.conflictscoredown.copy_from_slice(&self.conflictscoredown);
        other.conflict_weight = self.conflict_weight;
        other.conflict_avg_score = self.conflict_avg_score;
        other.cost_total = self.cost_total;
        other.inferences_total = self.inferences_total;
        other.nsamplestotal = self.nsamplestotal;
        other.ninferencestotal = self.ninferencestotal;
        other.ncutoffstotal = self.ncutoffstotal;
        other.remove_changed();
    }

    pub fn remove_changed(&mut self) {
        for d in &self.deltas {
            self.changedpos[d.col as usize] = -1;
        }
        self.deltas.clear();
        self.delta_cost_sum = 0.0;
        self.delta_inferences_sum = 0.0;
        self.delta_nsamplestotal = 0;
        self.delta_ninferencestotal = 0;
    }
}

/// HighsPseudocostInitialization's data (C++-owned vectors, filled by
/// Rust): arrays of `n` entries
#[repr(C)]
pub struct CPscostInit {
    pub pseudocostup: *mut f64,
    pub pseudocostdown: *mut f64,
    pub nsamplesup: *mut i32,
    pub nsamplesdown: *mut i32,
    pub inferencesup: *mut f64,
    pub inferencesdown: *mut f64,
    pub ninferencesup: *mut i32,
    pub ninferencesdown: *mut i32,
    pub conflictscoreup: *mut f64,
    pub conflictscoredown: *mut f64,
    pub n: i32,
    pub cost_total: f64,
    pub inferences_total: f64,
    pub conflict_avg_score: f64,
    pub nsamplestotal: i64,
    pub ninferencestotal: i64,
}

pub(crate) mod ffi {
    use super::*;
    use crate::ffi::{sl, sl_mut};

    #[no_mangle]
    pub extern "C" fn highs_rs_pscost_new(ncol: i32, minreliable: i32) -> *mut Pseudocost {
        Box::into_raw(Box::new(Pseudocost::new(ncol, minreliable)))
    }

    /// # Safety
    /// `p` from highs_rs_pscost_new/clone, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_free(p: *mut Pseudocost) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// # Safety
    /// `p` live or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_clone(p: *const Pseudocost) -> *mut Pseudocost {
        if p.is_null() {
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new((*p).clone()))
    }

    /// The initialization of HighsPseudocost(mipsolver): `orig[i]` the
    /// original index of column i in the init's arrays
    ///
    /// # Safety
    /// live handle, init arrays valid for `init.n`, `orig` for ncol
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_init(p: *mut Pseudocost, init: *const CPscostInit, orig: *const i32) {
        let p = &mut *p;
        let i = &*init;
        let n = p.pseudocostup.len();
        p.cost_total = i.cost_total;
        p.inferences_total = i.inferences_total;
        p.nsamplestotal = i.nsamplestotal;
        p.ninferencestotal = i.ninferencestotal;
        p.conflict_avg_score = i.conflict_avg_score * n as f64;
        let orig = sl(orig, n as i32);
        let m = i.n;
        let (pu, pd) = (sl(i.pseudocostup, m), sl(i.pseudocostdown, m));
        let (nu, nd) = (sl(i.nsamplesup, m), sl(i.nsamplesdown, m));
        let (iu, id) = (sl(i.inferencesup, m), sl(i.inferencesdown, m));
        let (niu, nid) = (sl(i.ninferencesup, m), sl(i.ninferencesdown, m));
        let (cu, cd) = (sl(i.conflictscoreup, m), sl(i.conflictscoredown, m));
        for k in 0..n {
            let o = orig[k] as usize;
            p.pseudocostup[k] = pu[o];
            p.nsamplesup[k] = nu[o];
            p.pseudocostdown[k] = pd[o];
            p.nsamplesdown[k] = nd[o];
            p.inferencesup[k] = iu[o];
            p.ninferencesup[k] = niu[o];
            p.inferencesdown[k] = id[o];
            p.ninferencesdown[k] = nid[o];
            p.conflictscoreup[k] = cu[o];
            p.conflictscoredown[k] = cd[o];
        }
    }

    /// HighsPseudocostInitialization(pscost, maxCount[, postsolveStack]):
    /// `orig` null for the first, else the original index of each column
    /// (the arrays, of the original size, zero filled by C++)
    ///
    /// # Safety
    /// live handle, init arrays valid for `init.n`, `orig` null or valid
    /// for the handle's columns
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_export(
        p: *const Pseudocost,
        max_count: i32,
        orig: *const i32,
        init: *mut CPscostInit,
    ) {
        let p = &*p;
        let i = &mut *init;
        let ncol = p.pseudocostup.len();
        i.cost_total = p.cost_total;
        i.inferences_total = p.inferences_total;
        i.conflict_avg_score = p.conflict_avg_score;
        i.nsamplestotal = p.nsamplestotal.min(1);
        i.ninferencestotal = p.ninferencestotal.min(1);
        i.conflict_avg_score /= ncol as f64 * p.conflict_weight;
        let m = i.n;
        let pu = sl_mut(i.pseudocostup, m);
        let pd = sl_mut(i.pseudocostdown, m);
        let nu = sl_mut(i.nsamplesup, m);
        let nd = sl_mut(i.nsamplesdown, m);
        let iu = sl_mut(i.inferencesup, m);
        let id = sl_mut(i.inferencesdown, m);
        let niu = sl_mut(i.ninferencesup, m);
        let nid = sl_mut(i.ninferencesdown, m);
        let cu = sl_mut(i.conflictscoreup, m);
        let cd = sl_mut(i.conflictscoredown, m);
        let presolved = !orig.is_null();
        let orig = sl(orig, ncol as i32);
        for k in 0..ncol {
            let o = if presolved { orig[k] as usize } else { k };
            pu[o] = p.pseudocostup[k];
            pd[o] = p.pseudocostdown[k];
            nu[o] = max_count.min(p.nsamplesup[k]);
            nd[o] = max_count.min(p.nsamplesdown[k]);
            iu[o] = p.inferencesup[k];
            id[o] = p.inferencesdown[k];
            niu[o] = if presolved { 1 } else { p.ninferencesup[k].min(1) };
            nid[o] = if presolved { 1 } else { p.ninferencesdown[k].min(1) };
            cu[o] = p.conflictscoreup[k] / p.conflict_weight;
            cd[o] = p.conflictscoredown[k] / p.conflict_weight;
        }
    }

    /// The scalar getters: 0 minreliable, 1 nsamplesup[col], 2
    /// nsamplesdown[col], 3 isReliable(col), 4 isReliableUp, 5
    /// isReliableDown, 6 the number of columns
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_geti(p: *const Pseudocost, which: i32, col: i32) -> i32 {
        let p = &*p;
        match which {
            0 => p.minreliable,
            1 => p.nsamplesup[col as usize],
            2 => p.nsamplesdown[col as usize],
            3 => p.is_reliable(col) as i32,
            4 => p.is_reliable_up(col) as i32,
            5 => p.is_reliable_down(col) as i32,
            _ => p.pseudocostup.len() as i32,
        }
    }

    /// The double getters: 0 getAvgPseudocost, 1 getPseudocostUp(col, x),
    /// 2 getPseudocostDown(col, x), 3 getConflictScoreUp, 4
    /// getConflictScoreDown, 5 getScore(col, x), 6 getScoreUp(col, x), 7
    /// getScoreDown(col, x), 8 getAvgInferencesUp, 9 getAvgInferencesDown
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_getd(p: *const Pseudocost, which: i32, col: i32, x: f64) -> f64 {
        let p = &*p;
        match which {
            0 => p.cost_total,
            1 => p.get_pseudocost_up(col, x),
            2 => p.get_pseudocost_down(col, x),
            3 => p.get_conflict_score_up(col),
            4 => p.get_conflict_score_down(col),
            5 => p.get_score(col, x),
            6 => p.get_score_dir(col, x, true),
            7 => p.get_score_dir(col, x, false),
            8 => p.inferencesup[col as usize],
            _ => p.inferencesdown[col as usize],
        }
    }

    /// getPseudocostUp (up) / Down (col, frac, offset)
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_offset(p: *const Pseudocost, col: i32, frac: f64, offset: f64, up: bool) -> f64 {
        let p = &*p;
        if up {
            p.get_pseudocost_up_offset(col, frac, offset)
        } else {
            p.get_pseudocost_down_offset(col, frac, offset)
        }
    }

    /// getScore(col, upcost, downcost)
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_score(p: *const Pseudocost, col: i32, up: f64, down: f64) -> f64 {
        (*p).get_score_costs(col, up, down)
    }

    /// The setters: 0 setMinReliable(i), 1 setDegeneracyFactor(x), 2
    /// increaseConflictWeight, 3 increaseConflictScoreUp(col), 4
    /// increaseConflictScoreDown(col), 5 addCutoffObservation(col, i != 0),
    /// 6 addInferenceObservation(col, i, up = x != 0), 7 removeChanged
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_set(p: *mut Pseudocost, which: i32, col: i32, i: i32, x: f64) {
        let p = &mut *p;
        match which {
            0 => p.minreliable = i,
            1 => p.degeneracy_factor = x,
            2 => p.increase_conflict_weight(),
            3 => p.increase_conflict_score(col, true),
            4 => p.increase_conflict_score(col, false),
            5 => p.add_cutoff_observation(col, i != 0),
            6 => p.add_inference_observation(col, i, x != 0.0),
            _ => p.remove_changed(),
        }
    }

    /// addObservation
    ///
    /// # Safety
    /// live handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_add_observation(p: *mut Pseudocost, col: i32, delta: f64, objdelta: f64) {
        (*p).add_observation(col, delta, objdelta);
    }

    /// flushPseudoCost (sync false) / syncPseudoCost of `p` and `other`
    ///
    /// # Safety
    /// distinct live handles
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_pscost_flush(p: *mut Pseudocost, other: *mut Pseudocost, sync: bool) {
        if sync {
            (*p).sync(&mut *other);
        } else {
            (*p).flush(&mut *other);
        }
    }
}
