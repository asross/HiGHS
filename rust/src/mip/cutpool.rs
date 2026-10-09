//! HighsCutPool and HighsDynamicRowMatrix (highs/mip/HighsCutPool.cpp,
//! HighsDynamicRowMatrix.cpp): the pool of cutting planes with duplicate
//! detection by hash, ageing, the propagation limits, and the separation of
//! violated pool cuts. Rust-owned; the C++ classes are a handle and a view
//! (HighsCutPool.h under HIGHS_RUST). The domains propagating the pool (C++
//! CutpoolPropagation) are told of added and deleted cuts through C++
//! callbacks, which read the pool: no Rust borrow of it is held across
//! them. Worker threads may call resetAge, increaseNumLps, lpCutRemoved and
//! separate in their thread safe forms at the same time as each other on
//! the global pool: those touch only the atomics (as in the C++).
//!
//! clang fuses `s += a * b` in the reductions over a cut (the norm, the
//! dot products of isDuplicate and getParallelism, the violation); none is
//! vectorized in libhighs.

use super::cuts::sort::pdqsort;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::hash::{double_hash_code, vector_hash, HighsHash};
use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI16, AtomicU8, Ordering::Relaxed};

const INF: f64 = f64::INFINITY;

/// `acc + sum x[i] * y[i]` in order, each term fused
#[inline]
fn dot_fused(n: usize, acc: f64, mut term: impl FnMut(usize) -> (f64, f64)) -> f64 {
    let mut d = acc;
    for i in 0..n {
        let (x, y) = term(i);
        d = x.mul_add_c(y, d);
    }
    d
}

/// HighsDynamicRowMatrix
#[derive(Default)]
pub struct RowMatrix {
    pub ar_range: Vec<[i32; 2]>,
    pub ar_index: Vec<i32>,
    pub ar_value: Vec<f64>,
    pub ar_rowindex: Vec<i32>,
    pub next_pos: Vec<i32>,
    prev_pos: Vec<i32>,
    pub next_neg: Vec<i32>,
    prev_neg: Vec<i32>,
    pub head_pos: Vec<i32>,
    pub head_neg: Vec<i32>,
    pub cols_linked: Vec<u8>,
    free_spaces: BTreeSet<(i32, i32)>,
    deleted_rows: Vec<i32>,
}

impl RowMatrix {
    pub fn new(ncols: i32) -> Self {
        RowMatrix { head_pos: vec![-1; ncols as usize], head_neg: vec![-1; ncols as usize], ..Default::default() }
    }

    #[inline]
    pub fn columns_linked(&self, row: i32) -> bool {
        self.cols_linked[row as usize] != 0
    }

    pub fn num_rows(&self) -> i32 {
        self.ar_range.len() as i32
    }

    pub fn num_del_rows(&self) -> i32 {
        self.deleted_rows.len() as i32
    }

    #[inline]
    pub fn row_range(&self, row: i32) -> (usize, usize) {
        let [s, e] = self.ar_range[row as usize];
        (s as usize, e as usize)
    }

    /// addRow: the row's entries (sorted by column) at a free space of the
    /// arrays, linked into the column lists if `link`
    pub fn add_row(&mut self, index: &[i32], value: &[f64], link: bool) -> i32 {
        let len = index.len() as i32;
        let start;
        let end;
        match self.free_spaces.range((len, -1)..).next().copied() {
            None => {
                start = self.ar_index.len() as i32;
                end = start + len;
                let n = end as usize;
                self.ar_index.resize(n, 0);
                self.ar_value.resize(n, 0.0);
                self.ar_rowindex.resize(n, 0);
                self.prev_pos.resize(n, -1);
                self.next_pos.resize(n, -1);
                self.prev_neg.resize(n, -1);
                self.next_neg.resize(n, -1);
            }
            Some(free) => {
                self.free_spaces.remove(&free);
                start = free.1;
                end = start + len;
                if free.0 > len {
                    self.free_spaces.insert((free.0 - len, end));
                }
            }
        }
        let row = match self.deleted_rows.pop() {
            None => {
                self.ar_range.push([start, end]);
                self.cols_linked.push(link as u8);
                self.ar_range.len() as i32 - 1
            }
            Some(row) => {
                self.ar_range[row as usize] = [start, end];
                self.cols_linked[row as usize] = link as u8;
                row
            }
        };
        let (s, e) = (start as usize, end as usize);
        self.ar_index[s..e].copy_from_slice(index);
        self.ar_value[s..e].copy_from_slice(value);
        self.ar_rowindex[s..e].fill(row);
        if !link {
            return row;
        }
        for i in s..e {
            let col = self.ar_index[i] as usize;
            let ii = i as i32;
            if self.ar_value[i] > 0.0 {
                self.prev_pos[i] = -1;
                let head = self.head_pos[col];
                self.head_pos[col] = ii;
                self.next_pos[i] = head;
                if head != -1 {
                    self.prev_pos[head as usize] = ii;
                }
            } else {
                self.prev_neg[i] = -1;
                let head = self.head_neg[col];
                self.head_neg[col] = ii;
                self.next_neg[i] = head;
                if head != -1 {
                    self.prev_neg[head as usize] = ii;
                }
            }
        }
        row
    }

    fn unlink(&mut self, s: usize, e: usize) {
        for i in s..e {
            let col = self.ar_index[i] as usize;
            if self.ar_value[i] > 0.0 {
                let (prev, next) = (self.prev_pos[i], self.next_pos[i]);
                if next != -1 {
                    self.prev_pos[next as usize] = prev;
                }
                if prev != -1 {
                    self.next_pos[prev as usize] = next;
                } else {
                    self.head_pos[col] = next;
                }
            } else {
                let (prev, next) = (self.prev_neg[i], self.next_neg[i]);
                if next != -1 {
                    self.prev_neg[next as usize] = prev;
                }
                if prev != -1 {
                    self.next_neg[prev as usize] = next;
                } else {
                    self.head_neg[col] = next;
                }
            }
        }
    }

    /// unlinkColumns
    pub fn unlink_columns(&mut self, row: i32) {
        if self.cols_linked[row as usize] == 0 {
            return;
        }
        self.cols_linked[row as usize] = 0;
        let (s, e) = self.row_range(row);
        self.unlink(s, e);
    }

    /// removeRow
    pub fn remove_row(&mut self, row: i32) {
        let (s, e) = self.row_range(row);
        if self.cols_linked[row as usize] != 0 {
            self.unlink(s, e);
        }
        self.deleted_rows.push(row);
        self.free_spaces.insert(((e - s) as i32, s as i32));
        self.ar_range[row as usize] = [-1, -1];
    }
}

/// cutAdded / cutDeleted of a C++ CutpoolPropagation
pub type CutAddedFn = unsafe extern "C" fn(*mut c_void, i32, bool);
pub type CutDeletedFn = unsafe extern "C" fn(*mut c_void, i32, bool);

pub struct CutPool {
    pub matrix: RowMatrix,
    pub rhs: Vec<f64>,
    ages: Vec<i16>,
    num_lps: Vec<AtomicI16>,
    age_reset_while_locked: Vec<AtomicU8>,
    has_synced: Vec<bool>,
    rownormalization: Vec<f64>,
    maxabscoef: Vec<f64>,
    rowintegral: Vec<u8>,
    hash_to_cut: HashMap<u64, Vec<i32>, IdHash>,
    prop_domains: Vec<*mut c_void>,
    prop_rows: BTreeSet<(i32, i32)>,
    best_observed_score: f64,
    min_score_factor: f64,
    min_density_lim: f64,
    agelim: i32,
    softlimit: i32,
    num_lp_cuts: i32,
    num_prop_nzs: i32,
    num_prop_rows: i32,
    age_distribution: Vec<i32>,
    sort_buffer: Vec<(i32, f64)>,
    value_hash_codes: Vec<u32>,
    pub index: i32,
    cut_added: CutAddedFn,
    cut_deleted: CutDeletedFn,
}

/// compute_cut_hash
fn cut_hash(index: &[i32], value: &[f64], maxabscoef: f64, codes: &mut Vec<u32>) -> u64 {
    let scale = 1.0 / maxabscoef;
    codes.clear();
    codes.extend(value.iter().map(|&v| double_hash_code(scale * v)));
    vector_hash(bytes_of(index)) ^ (vector_hash(bytes_of(codes)) >> 32)
}

/// The bytes of a slice of 4-byte integers
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    const { assert!(std::mem::size_of::<T>() == 4) };
    // SAFETY: i32 / u32 have no padding and any byte is a valid u8
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A hasher of keys that are hashes already (the cut hashes)
#[derive(Default, Clone, Copy)]
struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

type IdHash = std::hash::BuildHasherDefault<IdHasher>;

/// The model's size for addCut's propagation limits
#[repr(C)]
pub struct ModelSize {
    pub num_nonzero: i32,
    pub num_row: i32,
}

impl CutPool {
    pub fn new(ncols: i32, agelim: i32, softlimit: i32, index: i32, cut_added: CutAddedFn, cut_deleted: CutDeletedFn) -> Self {
        CutPool {
            matrix: RowMatrix::new(ncols),
            rhs: Vec::new(),
            ages: Vec::new(),
            num_lps: Vec::new(),
            age_reset_while_locked: Vec::new(),
            has_synced: Vec::new(),
            rownormalization: Vec::new(),
            maxabscoef: Vec::new(),
            rowintegral: Vec::new(),
            hash_to_cut: HashMap::default(),
            prop_domains: Vec::new(),
            prop_rows: BTreeSet::new(),
            best_observed_score: 0.0,
            min_score_factor: 0.9,
            min_density_lim: 0.1 * ncols as f64,
            agelim,
            softlimit,
            num_lp_cuts: 0,
            num_prop_nzs: 0,
            num_prop_rows: 0,
            age_distribution: vec![0; agelim as usize + 1],
            sort_buffer: Vec::new(),
            value_hash_codes: Vec::new(),
            index,
            cut_added,
            cut_deleted,
        }
    }

    pub fn num_cuts(&self) -> i32 {
        self.matrix.num_rows() - self.matrix.num_del_rows()
    }

    pub fn num_available_cuts(&self) -> i32 {
        self.num_cuts() - self.num_lp_cuts
    }

    pub fn row_length(&self, row: i32) -> i32 {
        let [s, e] = self.matrix.ar_range[row as usize];
        e - s
    }

    pub fn cut(&self, row: i32) -> (&[i32], &[f64]) {
        let (s, e) = self.matrix.row_range(row);
        (&self.matrix.ar_index[s..e], &self.matrix.ar_value[s..e])
    }

    /// getMaxAbsCutCoef
    pub fn max_abs_coef(&self, cut: i32) -> f64 {
        self.maxabscoef[cut as usize]
    }

    /// cutIsIntegral
    pub fn is_integral(&self, cut: i32) -> bool {
        self.rowintegral[cut as usize] != 0
    }

    pub fn set_age_limit(&mut self, agelim: i32) {
        self.agelim = agelim;
        self.age_distribution.resize(agelim as usize + 1, 0);
    }

    /// Moves a propagated cut's entry in propRows from its age to `age`
    #[inline]
    fn prop_row_age(&mut self, cut: i32, age: i32) {
        if self.matrix.columns_linked(cut) {
            self.prop_rows.remove(&(self.ages[cut as usize] as i32, cut));
            self.prop_rows.insert((age, cut));
        }
    }

    /// resetAge
    ///
    /// # Safety
    /// `p` live; thread_safe calls may run at the same time on other
    /// threads (only the atomic flag is written then)
    pub unsafe fn reset_age(p: *mut CutPool, cut: i32, thread_safe: bool) {
        let c = cut as usize;
        if *(*p).ages.as_ptr().add(c) > 0 {
            if thread_safe {
                (&(*p).age_reset_while_locked)[c].store(1, Relaxed);
                return;
            }
            let s = &mut *p;
            s.prop_row_age(cut, 0);
            s.age_distribution[s.ages[c] as usize] -= 1;
            s.age_distribution[0] += 1;
            s.ages[c] = 0;
            s.age_reset_while_locked[c].store(0, Relaxed);
        }
    }

    /// increaseNumLps
    pub fn increase_num_lps(&self, cut: i32, n: i32) {
        self.num_lps[cut as usize].fetch_add(n as i16, Relaxed);
    }

    /// lpCutRemoved
    ///
    /// # Safety
    /// as reset_age
    pub unsafe fn lp_cut_removed(p: *mut CutPool, cut: i32, thread_safe: bool) {
        let n = (&(*p).num_lps)[cut as usize].fetch_add(-1, Relaxed);
        if thread_safe || n > 1 {
            return;
        }
        let s = &mut *p;
        s.prop_row_age(cut, 1);
        s.ages[cut as usize] = 1;
        s.num_lp_cuts -= 1;
        s.age_distribution[1] += 1;
    }

    /// Tells the propagation domains that a cut is deleted
    ///
    /// # Safety
    /// `p` live, no borrow of it held
    unsafe fn deleted(p: *mut CutPool, cut: i32, only_for_propagation: bool) {
        let mut k = 0;
        while k < (*p).prop_domains.len() {
            let d = *(*p).prop_domains.as_ptr().add(k);
            ((*p).cut_deleted)(d, cut, only_for_propagation);
            k += 1;
        }
    }

    /// performAging
    ///
    /// # Safety
    /// as deleted
    pub unsafe fn perform_aging(p: *mut CutPool) {
        let (end, agelim) = {
            let s = &*p;
            let mut agelim = s.agelim;
            let mut available = s.num_available_cuts();
            while agelim > 5 && available > s.softlimit {
                available -= s.age_distribution[agelim as usize];
                agelim -= 1;
            }
            (s.matrix.num_rows(), agelim)
        };
        for i in 0..end {
            let iu = i as usize;
            let reset = {
                let s = &*p;
                let nlps = s.num_lps[iu].load(Relaxed);
                !(nlps > 0 && s.ages[iu] >= 0)
                    && !(nlps == 0 && s.ages[iu] == -1 && s.rhs[iu] != INF)
                    && s.age_reset_while_locked[iu].load(Relaxed) == 1
            };
            if reset {
                Self::reset_age(p, i, false);
            }
            let remove = {
                let s = &mut *p;
                let nlps = s.num_lps[iu].load(Relaxed);
                if nlps > 0 && s.ages[iu] >= 0 {
                    s.age_distribution[s.ages[iu] as usize] -= 1;
                    s.prop_row_age(i, -1);
                    s.ages[iu] = -1;
                    s.num_lp_cuts += 1;
                    s.age_reset_while_locked[iu].store(0, Relaxed);
                } else if nlps == 0 && s.ages[iu] == -1 && s.rhs[iu] != INF {
                    s.prop_row_age(i, 1);
                    s.ages[iu] = 1;
                    s.num_lp_cuts -= 1;
                    s.age_distribution[1] += 1;
                    s.age_reset_while_locked[iu].store(0, Relaxed);
                    continue;
                }
                s.age_reset_while_locked[iu].store(0, Relaxed);
                if s.ages[iu] < 0 {
                    continue;
                }
                let propagated = s.matrix.columns_linked(i);
                if propagated {
                    s.prop_rows.remove(&(s.ages[iu] as i32, i));
                }
                s.age_distribution[s.ages[iu] as usize] -= 1;
                s.ages[iu] += 1;
                if s.ages[iu] as i32 > agelim {
                    Some(propagated)
                } else {
                    if propagated {
                        s.prop_rows.insert((s.ages[iu] as i32, i));
                    }
                    s.age_distribution[s.ages[iu] as usize] += 1;
                    None
                }
            };
            if let Some(propagated) = remove {
                Self::deleted(p, i, false);
                let s = &mut *p;
                if propagated {
                    s.num_prop_rows -= 1;
                    s.num_prop_nzs -= s.row_length(i);
                }
                s.matrix.remove_row(i);
                s.ages[iu] = -1;
                s.rhs[iu] = INF;
                s.has_synced[iu] = false;
            }
        }
        debug_assert_eq!((*p).prop_rows.len() as i32, (*p).num_prop_rows);
    }

    /// isDuplicate
    fn is_duplicate(&self, hash: u64, norm: f64, index: &[i32], value: &[f64]) -> bool {
        let Some(rows) = self.hash_to_cut.get(&hash) else { return false };
        for &row in rows {
            let (s, e) = self.matrix.row_range(row);
            if e - s != index.len() || self.matrix.ar_index[s..e] != *index {
                continue;
            }
            let av = &self.matrix.ar_value[s..e];
            let dotprod = dot_fused(index.len(), 0.0, |i| (value[i], av[i]));
            let parallelism = dotprod * self.rownormalization[row as usize] * norm;
            if parallelism >= 1.0 - 1e-6 {
                return true;
            }
        }
        false
    }

    /// getParallelism of row1 of this pool and row2 of `pool2`
    fn parallelism(&self, row1: i32, row2: i32, pool2: &CutPool) -> f64 {
        let (mut i1, end1) = self.matrix.row_range(row1);
        let (mut i2, end2) = pool2.matrix.row_range(row2);
        let (index1, value1) = (&self.matrix.ar_index, &self.matrix.ar_value);
        let (index2, value2) = (&pool2.matrix.ar_index, &pool2.matrix.ar_value);
        let mut dotprod = 0.0f64;
        while i1 != end1 && i2 != end2 {
            let (c1, c2) = (index1[i1], index2[i2]);
            if c1 < c2 {
                i1 += 1;
            } else if c2 < c1 {
                i2 += 1;
            } else {
                dotprod = value1[i1].mul_add_c(value2[i2], dotprod);
                i1 += 1;
                i2 += 1;
            }
        }
        dotprod * self.rownormalization[row1 as usize] * pool2.rownormalization[row2 as usize]
    }

    /// addCut after the debug solution check, returns the row index or -1
    /// for a duplicate; `global` the global pool if this is not it. The
    /// cut's entries are sorted by column in place. The C++ extracts the
    /// cliques afterwards
    ///
    /// # Safety
    /// `p` live and `global` live or null (another pool), no borrows held
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_cut(
        p: *mut CutPool,
        global: *const CutPool,
        model: &ModelSize,
        index: &mut [i32],
        value: &mut [f64],
        rhs: f64,
        integral: bool,
        mut propagate: bool,
        is_conflict: bool,
    ) -> i32 {
        let len = index.len();
        let (h, maxabscoef, normalization, propagate) = {
            let s = &mut *p;
            s.sort_buffer.clear();
            let mut maxabscoef = 0.0f64;
            for i in 0..len {
                maxabscoef = maxabscoef.max(value[i].abs());
                s.sort_buffer.push((index[i], value[i]));
            }
            let norm = dot_fused(len, 0.0, |i| (value[i], value[i]));
            // the columns of a cut are distinct: every sort gives the order
            // of the C++ pdqsort_branchless
            s.sort_buffer.sort_unstable_by_key(|x| x.0);
            debug_assert!(s.sort_buffer.windows(2).all(|w| w[0].0 < w[1].0));
            for i in 0..len {
                index[i] = s.sort_buffer[i].0;
                value[i] = s.sort_buffer[i].1;
            }
            let h = cut_hash(index, value, maxabscoef, &mut s.value_hash_codes);
            let normalization = 1.0 / norm.sqrt();
            if !global.is_null() && (*global).is_duplicate(h, normalization, index, value) {
                return -1;
            }
            let s = &mut *p;
            if s.is_duplicate(h, normalization, index, value) {
                return -1;
            }
            if propagate {
                let new_prop_nzs = s.num_prop_nzs + len as i32;
                let avg_model_nzs = model.num_nonzero as f64 / model.num_row as f64;
                let new_avg_prop_nzs = new_prop_nzs as f64 / (s.num_prop_rows + 1) as f64;
                const ALPHA: f64 = 2.0;
                let lim = (ALPHA * avg_model_nzs).max(s.min_density_lim);
                let too_dense = if is_conflict { new_avg_prop_nzs > lim } else { len as f64 >= lim };
                if too_dense {
                    propagate = false;
                } else {
                    s.num_prop_rows += 1;
                    s.num_prop_nzs = new_prop_nzs;
                }
            }
            (h, maxabscoef, normalization, propagate)
        };

        // more than twice the model's nonzeros propagated: stop propagating
        // the oldest rows
        let mut excess = (*p).num_prop_nzs - 2 * model.num_nonzero;
        if excess > 0 {
            let mut unlinked = Vec::new();
            {
                let s = &mut *p;
                for &(age, r) in s.prop_rows.iter().rev() {
                    if excess <= 0 {
                        break;
                    }
                    let len = s.row_length(r);
                    excess -= len;
                    s.num_prop_nzs -= len;
                    s.num_prop_rows -= 1;
                    unlinked.push((age, r));
                }
            }
            for &(_, r) in &unlinked {
                (*p).matrix.unlink_columns(r);
                Self::deleted(p, r, true);
            }
            let s = &mut *p;
            for k in &unlinked {
                s.prop_rows.remove(k);
            }
        }

        let row = {
            let s = &mut *p;
            let row = s.matrix.add_row(index, value, propagate);
            s.hash_to_cut.entry(h).or_default().push(row);
            let r = row as usize;
            if r == s.rhs.len() {
                s.rhs.push(0.0);
                s.ages.push(0);
                s.num_lps.push(AtomicI16::new(0));
                s.age_reset_while_locked.push(AtomicU8::new(0));
                s.has_synced.push(false);
                s.rownormalization.push(0.0);
                s.maxabscoef.push(0.0);
                s.rowintegral.push(0);
            }
            s.rhs[r] = rhs;
            s.ages[r] = 0.max(s.agelim - 5) as i16;
            s.age_distribution[s.ages[r] as usize] += 1;
            s.rowintegral[r] = integral as u8;
            s.num_lps[r].store(0, Relaxed);
            s.age_reset_while_locked[r].store(0, Relaxed);
            s.has_synced[r] = false;
            if propagate {
                s.prop_rows.insert((s.ages[r] as i32, row));
            }
            debug_assert_eq!(s.prop_rows.len() as i32, s.num_prop_rows);
            s.rownormalization[r] = normalization;
            s.maxabscoef[r] = maxabscoef;
            row
        };
        let mut k = 0;
        while k < (*p).prop_domains.len() {
            let d = *(*p).prop_domains.as_ptr().add(k);
            ((*p).cut_added)(d, row, propagate);
            k += 1;
        }
        row
    }

    /// separate: ages the cuts not violated by `sol`, then selects the most
    /// efficacious violated ones that are not too parallel to the cuts of
    /// `cutset` (pairs of cut and pool index) or to each other. Returns the
    /// selected cuts (to be appended to the cut set by the C++), None if no
    /// cut is violated
    ///
    /// # Safety
    /// `p` live; `pools` the live pools (`pools[self.index] == p`); with
    /// `thread_safe` other threads may separate (thread safe) at the same
    /// time; else no borrows held
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn separate(
        p: *mut CutPool,
        sol: &[f64],
        col_lower: &[f64],
        col_upper: &[f64],
        cutset: &[(i32, i32)],
        feastol: f64,
        pools: &[*const CutPool],
        thread_safe: bool,
    ) -> Option<Vec<i32>> {
        let nrows = (*p).matrix.num_rows();
        let mut efficacious: Vec<(f64, i32)> = Vec::new();
        let agelim = {
            let s = &*p;
            let mut agelim = s.agelim;
            let mut num_cuts = s.num_cuts() - s.num_lp_cuts;
            while agelim > 1 && num_cuts > s.softlimit {
                num_cuts -= s.age_distribution[agelim as usize];
                agelim -= 1;
            }
            agelim
        };

        for i in 0..nrows {
            let iu = i as usize;
            if *(*p).ages.as_ptr().add(iu) < 0 {
                continue;
            }
            let (start, end) = (*p).matrix.row_range(i);
            let viol = {
                let s = &*p;
                let (ix, vx) = (&s.matrix.ar_index[start..end], &s.matrix.ar_value[start..end]);
                dot_fused(end - start, -s.rhs[iu], |j| (vx[j], sol[ix[j] as usize]))
            };
            let propagated = (*p).matrix.columns_linked(i);
            if !thread_safe {
                let s = &mut *p;
                s.age_distribution[s.ages[iu] as usize] -= 1;
                if propagated {
                    s.prop_rows.remove(&(s.ages[iu] as i32, i));
                }
            }
            if viol <= feastol {
                if thread_safe {
                    continue;
                }
                let remove = {
                    let s = &mut *p;
                    s.ages[iu] += 1;
                    if s.ages[iu] as i32 >= agelim {
                        let (st, e) = s.matrix.row_range(i);
                        let (ix, vx) = (&s.matrix.ar_index[st..e], &s.matrix.ar_value[st..e]);
                        let h = cut_hash(ix, vx, s.maxabscoef[iu], &mut s.value_hash_codes);
                        Some(h)
                    } else {
                        if propagated {
                            s.prop_rows.insert((s.ages[iu] as i32, i));
                        }
                        s.age_distribution[s.ages[iu] as usize] += 1;
                        None
                    }
                };
                if let Some(h) = remove {
                    Self::deleted(p, i, false);
                    let s = &mut *p;
                    if propagated {
                        s.num_prop_rows -= 1;
                        s.num_prop_nzs -= s.row_length(i);
                    }
                    s.matrix.remove_row(i);
                    s.ages[iu] = -1;
                    s.rhs[iu] = INF;
                    s.age_reset_while_locked[iu].store(0, Relaxed);
                    s.has_synced[iu] = false;
                    if let Some(rows) = s.hash_to_cut.get_mut(&h) {
                        if let Some(k) = rows.iter().position(|&r| r == i) {
                            rows.swap_remove(k);
                        }
                        if rows.is_empty() {
                            s.hash_to_cut.remove(&h);
                        }
                    }
                }
                continue;
            }

            // the norm over the entries not at their minimal activity
            let s = &*p;
            let mut rownorm = CDouble::from(0.0);
            let mut num_active = 0;
            for j in start..end {
                let col = s.matrix.ar_index[j] as usize;
                let v = s.matrix.ar_value[j];
                let solval = sol[col];
                let active = if v > 0.0 { solval > col_lower[col] + feastol } else { solval < col_upper[col] - feastol };
                if active {
                    rownorm += v * v;
                    num_active += 1;
                }
            }
            if !thread_safe {
                let s = &mut *p;
                s.ages[iu] = 0;
                s.age_distribution[0] += 1;
                if propagated {
                    s.prop_rows.insert((0, i));
                }
            }
            let score = viol / (num_active as f64 * rownorm.to_f64().sqrt());
            efficacious.push((score, i));
        }
        debug_assert!(thread_safe || (*p).prop_rows.len() as i32 == (*p).num_prop_rows);
        if efficacious.is_empty() {
            return None;
        }

        let n = efficacious.len() as u64;
        pdqsort(&mut efficacious, |a, b| {
            if a.0 > b.0 {
                return true;
            }
            if a.0 < b.0 {
                return false;
            }
            let ha = (((a.1 as u64) << 32).wrapping_add(n)).highs_hash();
            let hb = (((b.1 as u64) << 32).wrapping_add(n)).highs_hash();
            (ha, a.1) > (hb, b.1)
        });

        let (min_score, mut best, mut factor) = {
            let s = &*p;
            let best = efficacious[0].0.max(s.best_observed_score);
            (s.min_score_factor * best, best, s.min_score_factor)
        };
        let _ = &mut best;
        let mut numefficacious = efficacious.partition_point(|c| !(min_score > c.0));
        let lower = efficacious.len() / 20;
        let upper = efficacious.len() - 1;
        if numefficacious <= lower {
            numefficacious = (efficacious.len() / 2).max(1);
            factor = efficacious[numefficacious - 1].0 / best;
        } else if numefficacious > upper {
            factor = efficacious[upper].0 / best;
        }
        if !thread_safe {
            let s = &mut *p;
            s.best_observed_score = best;
            s.min_score_factor = factor;
        }
        efficacious.truncate(numefficacious);

        let me = (*p).index;
        let mut chosen: Vec<(i32, i32)> = cutset.to_vec();
        let mut selected = Vec::new();
        for &(_, cut) in &efficacious {
            let s = &*p;
            let mut discard = false;
            for &(c, pool) in &chosen {
                let par = if pool == me { s.parallelism(c, cut, s) } else { s.parallelism(cut, c, &*pools[pool as usize]) };
                if par > 0.1 {
                    discard = true;
                    break;
                }
            }
            if discard {
                continue;
            }
            s.num_lps[cut as usize].fetch_add(1, Relaxed);
            if !thread_safe {
                let s = &mut *p;
                let c = cut as usize;
                s.age_distribution[s.ages[c] as usize] -= 1;
                s.num_lp_cuts += 1;
                s.prop_row_age(cut, -1);
                s.ages[c] = -1;
            }
            chosen.push((cut, me));
            selected.push(cut);
        }
        Some(selected)
    }

    /// separateLpCutsAfterRestart: marks all cuts as LP rows (the C++ fills
    /// the cut set)
    pub fn lp_cuts_after_restart(&mut self) {
        let n = self.matrix.num_rows();
        for i in 0..n {
            let iu = i as usize;
            self.age_distribution[self.ages[iu] as usize] -= 1;
            self.num_lp_cuts += 1;
            self.prop_row_age(i, -1);
            self.num_lps[iu].store(1, Relaxed);
            self.ages[iu] = -1;
        }
    }

    /// The cuts syncCutPool adds to another pool (in the LP or reset while
    /// locked, not yet synced), marked as synced
    pub fn cuts_to_sync(&mut self) -> Vec<i32> {
        let mut out = Vec::new();
        for i in 0..self.matrix.num_rows() as usize {
            if (self.num_lps[i].load(Relaxed) > 0 || self.age_reset_while_locked[i].load(Relaxed) == 1) && !self.has_synced[i] {
                out.push(i as i32);
                self.has_synced[i] = true;
            }
        }
        out
    }
}

/// The view of the matrix and right-hand sides C++ reads (mirrored by
/// HighsDynamicRowMatrix and HighsCutPool::getRhs under HIGHS_RUST)
#[repr(C)]
pub struct CMatrixView {
    pub ar_range: *const [i32; 2],
    pub num_rows: i32,
    pub num_del_rows: i32,
    pub ar_index: *const i32,
    pub ar_value: *const f64,
    pub num_nz: i32,
    pub ar_rowindex: *const i32,
    pub next_pos: *const i32,
    pub next_neg: *const i32,
    pub head_pos: *const i32,
    pub head_neg: *const i32,
    pub num_cols: i32,
    pub cols_linked: *const u8,
    pub rhs: *const f64,
    pub num_rhs: i32,
}

pub(crate) mod ffi {
    use super::*;
    use crate::ffi::{sl, sl_mut};

    #[no_mangle]
    pub extern "C" fn highs_rs_cutpool_new(
        ncols: i32,
        agelim: i32,
        softlimit: i32,
        index: i32,
        added: CutAddedFn,
        deleted: CutDeletedFn,
    ) -> *mut CutPool {
        Box::into_raw(Box::new(CutPool::new(ncols, agelim, softlimit, index, added, deleted)))
    }

    /// # Safety
    /// `p` from highs_rs_cutpool_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_free(p: *mut CutPool) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// The matrix and rhs arrays
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_view(p: *const CutPool, v: *mut CMatrixView) {
        let s = &*p;
        let m = &s.matrix;
        *v = CMatrixView {
            ar_range: m.ar_range.as_ptr(),
            num_rows: m.num_rows(),
            num_del_rows: m.num_del_rows(),
            ar_index: m.ar_index.as_ptr(),
            ar_value: m.ar_value.as_ptr(),
            num_nz: m.ar_index.len() as i32,
            ar_rowindex: m.ar_rowindex.as_ptr(),
            next_pos: m.next_pos.as_ptr(),
            next_neg: m.next_neg.as_ptr(),
            head_pos: m.head_pos.as_ptr(),
            head_neg: m.head_neg.as_ptr(),
            num_cols: m.head_pos.len() as i32,
            cols_linked: m.cols_linked.as_ptr(),
            rhs: s.rhs.as_ptr(),
            num_rhs: s.rhs.len() as i32,
        };
    }

    /// 0 getNumCuts, 1 getNumAvailableCuts, 2 getRowLength(i), 3
    /// cutIsIntegral(i), 4 index
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_geti(p: *const CutPool, which: i32, i: i32) -> i32 {
        let s = &*p;
        match which {
            0 => s.num_cuts(),
            1 => s.num_available_cuts(),
            2 => s.row_length(i),
            3 => s.rowintegral[i as usize] as i32,
            _ => s.index,
        }
    }

    /// 0 getMaxAbsCutCoef(i), 1 getRowNormalization(i)
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_getd(p: *const CutPool, which: i32, i: i32) -> f64 {
        let s = &*p;
        match which {
            0 => s.maxabscoef[i as usize],
            _ => s.rownormalization[i as usize],
        }
    }

    /// 0 resetAge(i, thread_safe = j), 1 lpCutRemoved(i, j), 2
    /// increaseNumLps(i, j), 3 setAgeLimit(i), 4 performAging, 5 add / 6
    /// remove the propagation domain `d`, 7 separateLpCutsAfterRestart's
    /// bookkeeping
    ///
    /// # Safety
    /// live pool (see the module comment for the thread safe calls)
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_op(p: *mut CutPool, which: i32, i: i32, j: i32, d: *mut c_void) {
        match which {
            0 => CutPool::reset_age(p, i, j != 0),
            1 => CutPool::lp_cut_removed(p, i, j != 0),
            2 => (*p).increase_num_lps(i, j),
            3 => (*p).set_age_limit(i),
            4 => CutPool::perform_aging(p),
            5 => (*p).prop_domains.push(d),
            6 => {
                let s = &mut *p;
                if let Some(k) = s.prop_domains.iter().rposition(|&x| x == d) {
                    s.prop_domains.remove(k);
                }
            }
            _ => (*p).lp_cuts_after_restart(),
        }
    }

    /// getParallelism(row1, row2, pool2)
    ///
    /// # Safety
    /// live pools
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_parallelism(p: *const CutPool, row1: i32, row2: i32, pool2: *const CutPool) -> f64 {
        (*p).parallelism(row1, row2, &*pool2)
    }

    /// addCut (see CutPool::add_cut)
    ///
    /// # Safety
    /// live pool, `global` live or null, arrays valid for `len`
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_cutpool_add_cut(
        p: *mut CutPool,
        global: *const CutPool,
        model: *const ModelSize,
        index: *mut i32,
        value: *mut f64,
        len: i32,
        rhs: f64,
        integral: bool,
        propagate: bool,
        is_conflict: bool,
    ) -> i32 {
        CutPool::add_cut(p, global, &*model, sl_mut(index, len), sl_mut(value, len), rhs, integral, propagate, is_conflict)
    }

    /// separate: the selected cuts (a buffer to free with
    /// highs_rs_cutpool_free_buf, count in `*num`, -1 if no cut is
    /// violated)
    ///
    /// # Safety
    /// as CutPool::separate, arrays valid for their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_cutpool_separate(
        p: *mut CutPool,
        sol: *const f64,
        ncol: i32,
        col_lower: *const f64,
        col_upper: *const f64,
        cut_indices: *const i32,
        cut_pools: *const i32,
        ncuts: i32,
        feastol: f64,
        pools: *const *const CutPool,
        npools: i32,
        thread_safe: bool,
        num: *mut i32,
    ) -> *const i32 {
        let cutset: Vec<(i32, i32)> = sl(cut_indices, ncuts).iter().copied().zip(sl(cut_pools, ncuts).iter().copied()).collect();
        let sel = CutPool::separate(
            p,
            sl(sol, ncol),
            sl(col_lower, ncol),
            sl(col_upper, ncol),
            &cutset,
            feastol,
            sl(pools, npools),
            thread_safe,
        );
        let Some(sel) = sel else {
            *num = -1;
            return std::ptr::null();
        };
        *num = sel.len() as i32;
        if sel.is_empty() {
            return std::ptr::null();
        }
        Box::into_raw(sel.into_boxed_slice()) as *const i32
    }

    /// Frees the buffer of highs_rs_cutpool_separate
    ///
    /// # Safety
    /// a buffer from highs_rs_cutpool_separate with its count
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_free_buf(buf: *mut i32, num: i32) {
        if !buf.is_null() && num > 0 {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(buf, num as usize)));
        }
    }

    /// syncCutPool's cuts (marked synced): the C++ adds them to the other
    /// pool. Data returned (free with highs_rs_cutpool_free_buf), count in
    /// `*num`
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_cutpool_cuts_to_sync(p: *mut CutPool, num: *mut i32) -> *mut i32 {
        let v = (*p).cuts_to_sync();
        *num = v.len() as i32;
        if v.is_empty() {
            return std::ptr::null_mut();
        }
        Box::into_raw(v.into_boxed_slice()) as *mut i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn added(_: *mut c_void, _: i32, _: bool) {}
    unsafe extern "C" fn deleted(_: *mut c_void, _: i32, _: bool) {}

    #[test]
    fn add_duplicate_and_age_out() {
        let p = Box::into_raw(Box::new(CutPool::new(3, 10, 100, 0, added, deleted)));
        let model = ModelSize { num_nonzero: 10, num_row: 5 };
        // SAFETY: a live pool, no borrows held
        unsafe {
            let (mut i, mut v) = ([2, 0], [1.0, 2.0]);
            let r = CutPool::add_cut(p, std::ptr::null(), &model, &mut i, &mut v, 1.0, false, true, false);
            assert_eq!(r, 0);
            // sorted by column
            assert_eq!(i, [0, 2]);
            // a scaled copy is a duplicate
            let (mut i, mut v) = ([0, 2], [4.0, 2.0]);
            assert_eq!(CutPool::add_cut(p, std::ptr::null(), &model, &mut i, &mut v, 2.0, false, true, false), -1);
            assert_eq!((*p).num_cuts(), 1);
            for _ in 0..10 {
                CutPool::perform_aging(p);
            }
            assert_eq!((*p).num_cuts(), 0);
            // the index is reused
            let (mut i, mut v) = ([1], [1.0]);
            assert_eq!(CutPool::add_cut(p, std::ptr::null(), &model, &mut i, &mut v, 1.0, false, true, false), 0);
            drop(Box::from_raw(p));
        }
    }
}
