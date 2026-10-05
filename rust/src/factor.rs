//! Port of HFactor (highs/util/HFactor*.cpp): basis matrix factorization
//! PBQ = LU, update and solves. Same algorithms and floating-point
//! operations as the C++, so results are bit-identical: clang contracts
//! `x -= a * b` (and a few `a * b + c` tick formulae) into fused
//! multiply-adds on arm64, hence `mul_add` exactly there.
//!
//! The A matrix and basic_index stay owned by the caller and are passed
//! in to the calls that use them; HVectors are borrowed per call.

use crate::hvector::{HVec, OwnedHVec, K_HIGHS_TINY, K_HIGHS_ZERO};
use std::time::Instant;

pub const UPDATE_FT: i32 = 1;
pub const UPDATE_PF: i32 = 2;
pub const UPDATE_MPF: i32 = 3;
pub const UPDATE_APF: i32 = 4;

const K_HYPER_FTRAN_L: f64 = 0.15;
const K_HYPER_FTRAN_U: f64 = 0.10;
const K_HYPER_BTRAN_L: f64 = 0.10;
const K_HYPER_BTRAN_U: f64 = 0.15;
const K_HYPER_CANCEL: f64 = 0.05;
const K_RUNNING_AVERAGE_MULTIPLIER: f64 = 0.05;
const K_MC_EXTRA_ENTRIES_MULTIPLIER: i32 = 2;
const K_MR_EXTRA_ENTRIES_MULTIPLIER: i32 = 2;
const K_L_FACTOR_EXTRA_ENTRIES_MULTIPLIER: i32 = 3;
const K_U_FACTOR_EXTRA_VECTORS: i32 = 1000;
const K_U_FACTOR_EXTRA_ENTRIES_MULTIPLIER: i32 = 3;
const K_PF_PIVOT_ENTRIES: usize = 1000;
const K_PF_VECTORS: usize = 2000;
const K_PF_ENTRIES_MULTIPLIER: i32 = 4;

pub const BUILD_KERNEL_RETURN_TIMEOUT: i32 = -1;

const PIVOT_LOGICAL: i8 = 0;
const PIVOT_UNIT: i8 = 1;
const PIVOT_ROW_SINGLETON: i8 = 2;
const PIVOT_COL_SINGLETON: i8 = 3;
const PIVOT_MARKOWITZ: i8 = 4;

// std::max semantics
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// The constraint matrix, column-wise, as held by the caller
pub struct AMatrix<'a> {
    pub num_col: i32,
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

/// Refactorization information supplied by the caller (RefactorInfo)
pub struct RefactorIn<'a> {
    pub pivot_row: &'a [i32],
    pub pivot_var: &'a [i32],
    pub pivot_type: &'a [i8],
    pub build_synthetic_tick: f64,
}

/// A triangular factor in HFactor's layout: column (or row) `i` has entries
/// `index/value[start[i]..end[i]]` and pivot `pivot_index[i]`; `lookup` maps
/// a row to its pivot position. `pivot_value` is `None` for unit L.
pub struct Triangle<'a> {
    pub lookup: &'a [i32],
    pub pivot_index: &'a [i32],
    pub pivot_value: Option<&'a [f64]>,
    pub start: &'a [i32],
    pub end: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

/// Hyper-sparse triangular solve (solveHyper)
///
/// # Safety
/// `h` must be L, LR, U or UR of an HFactor with `h_size` rows (HFactor
/// invariant), so that its entries are rows < h_size and its ranges valid
unsafe fn solve_hyper(h_size: usize, h: &Triangle, rhs: &mut HVec) {
    assert!(rhs.array.len() >= h_size && h.lookup.len() >= h_size);
    let (list_index, list_stack) = rhs.iwork.split_at_mut(h_size);
    let mark = &mut *rhs.cwork;
    let mut list_count = 0;
    let mut count_pivot = 0usize;
    let mut count_entry = 0usize;

    // Depth-first search for the topological order of the nonzeros
    for &row in &rhs.index[..rhs.count as usize] {
        let mut hi = h.lookup[row as usize] as usize;
        if mark[hi] != 0 {
            continue;
        }
        let mut hk = h.start[hi] as usize;
        let mut n_stack = 0;
        mark[hi] = 1;
        loop {
            if hk < h.end[hi] as usize {
                // SAFETY: hk < end <= index.len() (range invariant), and
                // the entry is a row < h_size <= lookup.len()
                let sub = *h.lookup.get_unchecked(*h.index.get_unchecked(hk) as usize) as usize;
                hk += 1;
                if mark[sub] == 0 {
                    mark[sub] = 1;
                    list_stack[n_stack] = hi as i32;
                    list_stack[n_stack + 1] = hk as i32;
                    n_stack += 2;
                    hi = sub;
                    hk = h.start[hi] as usize;
                    if hi >= h_size {
                        count_pivot += 1;
                        count_entry += (h.end[hi] - h.start[hi]) as usize;
                    }
                }
            } else {
                list_index[list_count] = hi as i32;
                list_count += 1;
                if n_stack == 0 {
                    break;
                }
                n_stack -= 2;
                hi = list_stack[n_stack] as usize;
                hk = list_stack[n_stack + 1] as usize;
            }
        }
    }
    rhs.synthetic_tick += (count_pivot * 20 + count_entry * 10) as f64;

    // Solve in reverse topological order
    let mut count = 0;
    for &i in list_index[..list_count].iter().rev() {
        let i = i as usize;
        mark[i] = 0;
        let pivot_row = h.pivot_index[i];
        // SAFETY (pivot_row, and the column's rows and range): HFactor
        // invariant, see the function's safety condition
        let x = rhs.array.get_unchecked_mut(pivot_row as usize);
        let mut multiplier = *x;
        if multiplier.abs() > K_HIGHS_TINY {
            if let Some(pivot_value) = h.pivot_value {
                multiplier /= pivot_value[i];
                *x = multiplier;
            }
            rhs.index[count] = pivot_row;
            count += 1;
            let (idx, val) = entries(h.index, h.value, h.start[i], h.end[i]);
            axpy_unchecked(rhs.array, idx, val, multiplier);
        } else {
            *x = 0.0;
        }
    }
    rhs.count = count as i32;
}

/// Collect by X, scatter by Y (solveMatrixT, for MPF and APF)
#[allow(clippy::too_many_arguments)]
fn solve_matrix_t(
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    t_index: &[i32],
    t_value: &[f64],
    t_pivot: f64,
    rhs: &mut HVec,
) {
    let mut pivot_multiplier = 0.0;
    for k in x_start as usize..x_end as usize {
        pivot_multiplier = t_value[k].mul_add(rhs.array[t_index[k] as usize], pivot_multiplier);
    }
    if pivot_multiplier.abs() > K_HIGHS_TINY {
        let mut work_count = rhs.count as usize;
        pivot_multiplier /= t_pivot;
        for k in y_start as usize..y_end as usize {
            let index = t_index[k] as usize;
            let value0 = rhs.array[index];
            let value1 = (-pivot_multiplier).mul_add(t_value[k], value0);
            if value0 == 0.0 {
                rhs.index[work_count] = index as i32;
                work_count += 1;
            }
            rhs.array[index] = if value1.abs() < K_HIGHS_TINY {
                K_HIGHS_ZERO
            } else {
                value1
            };
        }
        rhs.count = work_count as i32;
    }
}

// The sparse solves (ftran_l, btran_l, solve_u_sparse, ftran_ft,
// btran_ft) skip bounds checks: with them, FTRAN took ~30% longer and the
// MIP lambda_080458 6% more cycles than C++. This is sound given the
// invariant documented on HFactor, plus O(1) length checks on entry.

/// The entries [start, max(start, end)) of a factor column (while
/// refactorizing, L columns not yet formed have end < start)
///
/// # Safety
/// `0 <= start, end <= index.len() <= value.len()` (range invariant)
#[inline(always)]
unsafe fn entries<'a>(
    index: &'a [i32],
    value: &'a [f64],
    start: i32,
    end: i32,
) -> (&'a [i32], &'a [f64]) {
    let (s, e) = (start as usize, end.max(start) as usize);
    (index.get_unchecked(s..e), value.get_unchecked(s..e))
}

/// `x[j] -= m * v` (fused, as clang compiles it) for the entries `(j, v)`
/// of a factor column
///
/// # Safety
/// Every `j` in `index` must be less than `x.len()` (row invariant).
#[inline(always)]
unsafe fn axpy_unchecked(x: &mut [f64], index: &[i32], value: &[f64], m: f64) {
    for (&j, &v) in index.iter().zip(value) {
        let y = x.get_unchecked_mut(j as usize);
        *y = (-m).mul_add(v, *y);
    }
}

/// Basis matrix factorization, update and solves (HFactor's data)
///
/// Invariant, relied on by the unchecked sparse solves:
/// - rows: every row index held in L (l_pivot_index, l_index, lr_index),
///   U (u_pivot_index other than -1, u_index, ur_index) and the FT update
///   (pf_pivot_index, pf_index) is less than num_row;
/// - ranges: every start/end in l_start, lr_start, u_start, u_last_p,
///   ur_start, ur_lastp and pf_start lies in [0, len] for the length len
///   of its index array, which its value array matches.
///
/// Build establishes it (ranges come from pushes; each row index passes
/// through a bounds-checked lookup of length num_row in build_finish),
/// updates check what they store, setup clears the factor and
/// check_indices checks vectors set from C++.
#[derive(Default)]
pub struct HFactor {
    pub num_row: i32,
    pub num_basic: i32,
    pub inv_num_row: f64,
    pub pivot_threshold: f64,
    pub pivot_tolerance: f64,
    pub time_limit: f64,
    pub update_method: i32,
    basis_matrix_limit_size: i32,

    // Build results, mirrored by the C++ wrapper
    pub build_synthetic_tick: f64,
    pub rank_deficiency: i32,
    pub basis_matrix_num_el: i32,
    pub invert_num_el: i32,
    pub kernel_dim: i32,
    pub kernel_num_el: i32,
    pub row_with_no_pivot: Vec<i32>,
    pub col_with_no_pivot: Vec<i32>,
    pub var_with_no_pivot: Vec<i32>,
    // Refactorization information recorded by a build from scratch
    pub refactor_pivot_row: Vec<i32>,
    pub refactor_pivot_var: Vec<i32>,
    pub refactor_pivot_type: Vec<i8>,
    pub refactor_build_synthetic_tick: f64,

    // Working buffer
    nwork: i32,
    iwork: Vec<i32>,
    dwork: Vec<f64>,

    // Basis matrix
    b_var: Vec<i32>,
    b_start: Vec<i32>,
    b_index: Vec<i32>,
    b_value: Vec<f64>,

    // Permutation
    permute: Vec<i32>,

    // Kernel matrix
    mc_var: Vec<i32>,
    mc_start: Vec<i32>,
    mc_count_a: Vec<i32>,
    mc_count_n: Vec<i32>,
    mc_space: Vec<i32>,
    mc_index: Vec<i32>,
    mc_value: Vec<f64>,
    mc_min_pivot: Vec<f64>,

    // Row wise kernel matrix
    mr_start: Vec<i32>,
    mr_count: Vec<i32>,
    mr_space: Vec<i32>,
    mr_count_before: Vec<i32>,
    mr_index: Vec<i32>,

    // Kernel column buffer
    mwz_column_index: Vec<i32>,
    mwz_column_mark: Vec<u8>,
    mwz_column_array: Vec<f64>,

    // Count link list
    col_link_first: Vec<i32>,
    col_link_next: Vec<i32>,
    col_link_last: Vec<i32>,
    row_link_first: Vec<i32>,
    row_link_next: Vec<i32>,
    row_link_last: Vec<i32>,

    // Factor L
    pub l_pivot_lookup: Vec<i32>,
    pub l_pivot_index: Vec<i32>,
    pub l_start: Vec<i32>,
    pub l_index: Vec<i32>,
    pub l_value: Vec<f64>,
    pub lr_start: Vec<i32>,
    pub lr_index: Vec<i32>,
    pub lr_value: Vec<f64>,

    // Factor U
    pub u_pivot_lookup: Vec<i32>,
    pub u_pivot_index: Vec<i32>,
    pub u_pivot_value: Vec<f64>,
    u_merit_x: i32,
    u_total_x: i32,
    pub u_start: Vec<i32>,
    pub u_last_p: Vec<i32>,
    pub u_index: Vec<i32>,
    pub u_value: Vec<f64>,
    pub ur_start: Vec<i32>,
    pub ur_lastp: Vec<i32>,
    pub ur_space: Vec<i32>,
    pub ur_index: Vec<i32>,
    pub ur_value: Vec<f64>,

    // Update buffer
    pub pf_pivot_value: Vec<f64>,
    pub pf_pivot_index: Vec<i32>,
    pub pf_start: Vec<i32>,
    pub pf_index: Vec<i32>,
    pub pf_value: Vec<f64>,
}

fn reserve<T>(v: &mut Vec<T>, n: i32) {
    v.reserve((n.max(0) as usize).saturating_sub(v.len()));
}

impl HFactor {
    /// setupGeneral: copy the problem size and allocate space for INVERT.
    /// The pivot threshold and tolerance (clamped by the caller) and the
    /// time limit are passed to build.
    pub fn setup(
        &mut self,
        num_col: i32,
        num_row: i32,
        num_basic: i32,
        a_start: &[i32],
        update_method: i32,
    ) {
        self.num_row = num_row;
        self.num_basic = num_basic;
        self.inv_num_row = 1.0 / num_row as f64;
        self.update_method = update_method;
        // No factor (of other dimensions) to solve with until build
        self.lu_clear();
        self.l_start.clear();
        self.l_pivot_index.clear();
        self.lr_start.clear();
        self.u_start.clear();
        for v in [
            &mut self.u_last_p,
            &mut self.ur_start,
            &mut self.ur_lastp,
            &mut self.pf_pivot_index,
        ] {
            v.clear();
        }
        self.pf_start.clear();
        self.pf_start.push(0);
        self.pf_index.clear();
        self.pf_value.clear();
        self.pf_pivot_value.clear();
        let nr = num_row as usize;
        let nb = num_basic as usize;

        reserve(&mut self.iwork, num_row * 2);
        self.dwork = vec![0.0; nr];

        // Find Basis matrix limit size
        let mut limit = 0;
        self.iwork.clear();
        self.iwork.resize(nr + 1, 0);
        for i in 0..num_col as usize {
            self.iwork[(a_start[i + 1] - a_start[i]) as usize] += 1;
        }
        let b_max_dim = num_row.max(num_basic);
        let (mut i, mut counted) = (num_row, 0);
        while i >= 0 && counted < b_max_dim {
            limit += i * self.iwork[i as usize];
            counted += self.iwork[i as usize];
            i -= 1;
        }
        limit += b_max_dim;
        self.basis_matrix_limit_size = limit;
        let bm = b_max_dim as usize;
        let lim = limit as usize;

        // Allocate space for basis matrix, L, U factor and Update buffer
        self.b_var.resize(bm, 0);
        self.b_start.resize(bm + 1, 0);
        self.b_index.resize(lim, 0);
        self.b_value.resize(lim, 0.0);

        // Allocate space for pivot records
        self.permute.resize(bm, 0);

        // Allocate space for Markowitz matrices
        self.mc_var.resize(nb, 0);
        self.mc_start.resize(nb, 0);
        self.mc_count_a.resize(nb, 0);
        self.mc_count_n.resize(nb, 0);
        self.mc_space.resize(nb, 0);
        self.mc_min_pivot.resize(nb, 0.0);
        self.mc_index
            .resize(lim * K_MC_EXTRA_ENTRIES_MULTIPLIER as usize, 0);
        self.mc_value
            .resize(lim * K_MC_EXTRA_ENTRIES_MULTIPLIER as usize, 0.0);

        self.mr_start.resize(nr, 0);
        self.mr_count.resize(nr, 0);
        self.mr_space.resize(nr, 0);
        self.mr_count_before.resize(nr, 0);
        self.mr_index
            .resize(lim * K_MR_EXTRA_ENTRIES_MULTIPLIER as usize, 0);

        self.mwz_column_mark = vec![0; nr];
        self.mwz_column_index.resize(nr, 0);
        self.mwz_column_array = vec![0.0; nr];

        // Allocate space for count-link-list
        self.col_link_first = vec![-1; nr + 1];
        self.col_link_next.resize(nb, 0);
        self.col_link_last.resize(nb, 0);
        self.row_link_first = vec![-1; nb + 1];
        self.row_link_next.resize(nr, 0);
        self.row_link_last.resize(nr, 0);

        // Allocate space for L factor
        self.l_pivot_lookup.resize(nr, 0);
        reserve(&mut self.l_pivot_index, num_row);
        reserve(&mut self.l_start, num_row + 1);
        reserve(
            &mut self.l_index,
            limit * K_L_FACTOR_EXTRA_ENTRIES_MULTIPLIER,
        );
        reserve(
            &mut self.l_value,
            limit * K_L_FACTOR_EXTRA_ENTRIES_MULTIPLIER,
        );
        reserve(&mut self.lr_start, num_row + 1);
        reserve(
            &mut self.lr_index,
            limit * K_L_FACTOR_EXTRA_ENTRIES_MULTIPLIER,
        );
        reserve(
            &mut self.lr_value,
            limit * K_L_FACTOR_EXTRA_ENTRIES_MULTIPLIER,
        );

        // Allocate space for U factor
        self.u_pivot_lookup.resize(nr, 0);
        let nu = num_row + K_U_FACTOR_EXTRA_VECTORS;
        let ue = limit * K_U_FACTOR_EXTRA_ENTRIES_MULTIPLIER;
        reserve(&mut self.u_pivot_index, nu);
        reserve(&mut self.u_pivot_value, nu);
        reserve(&mut self.u_start, nu + 1);
        reserve(&mut self.u_last_p, nu);
        reserve(&mut self.u_index, ue);
        reserve(&mut self.u_value, ue);
        reserve(&mut self.ur_start, nu + 1);
        reserve(&mut self.ur_lastp, nu);
        reserve(&mut self.ur_space, nu);
        reserve(&mut self.ur_index, ue);
        reserve(&mut self.ur_value, ue);

        // Allocate spaces for Update buffer
        self.pf_pivot_value.reserve(K_PF_PIVOT_ENTRIES);
        self.pf_pivot_index.reserve(K_PF_PIVOT_ENTRIES);
        self.pf_start.reserve(K_PF_VECTORS + 1);
        reserve(&mut self.pf_index, limit * K_PF_ENTRIES_MULTIPLIER);
        reserve(&mut self.pf_value, limit * K_PF_ENTRIES_MULTIPLIER);
    }

    /// Check the invariant (rows: on the live entries of U and UR, since
    /// their gaps may hold stale indices)
    pub fn check_indices(&self) {
        let n = self.num_row;
        let rows = |v: &[i32]| v.iter().all(|&i| 0 <= i && i < n);
        let ranges = |starts: &[i32], index: &[i32], value: &[f64]| {
            index.len() <= value.len()
                && starts.iter().all(|&s| 0 <= s && s as usize <= index.len())
        };
        assert!(rows(&self.l_pivot_index) && rows(&self.l_index) && rows(&self.lr_index));
        assert!(rows(&self.pf_pivot_index) && rows(&self.pf_index));
        assert!(self.u_pivot_index.iter().all(|&i| -1 <= i && i < n));
        assert!(ranges(&self.l_start, &self.l_index, &self.l_value));
        assert!(ranges(&self.lr_start, &self.lr_index, &self.lr_value));
        assert!(ranges(&self.u_start, &self.u_index, &self.u_value));
        assert!(ranges(&self.u_last_p, &self.u_index, &self.u_value));
        assert!(ranges(&self.ur_start, &self.ur_index, &self.ur_value));
        assert!(ranges(&self.ur_lastp, &self.ur_index, &self.ur_value));
        assert!(ranges(&self.pf_start, &self.pf_index, &self.pf_value));
        for i in 0..self.u_pivot_index.len() {
            assert!(rows(
                &self.u_index[self.u_start[i] as usize..self.u_last_p[i] as usize]
            ));
            assert!(rows(
                &self.ur_index[self.ur_start[i] as usize..self.ur_lastp[i] as usize]
            ));
        }
    }

    fn refactor_clear(&mut self) {
        self.refactor_build_synthetic_tick = 0.0;
        self.refactor_pivot_row.clear();
        self.refactor_pivot_var.clear();
        self.refactor_pivot_type.clear();
    }

    fn refactor_push(&mut self, row: i32, var: i32, pivot_type: i8) {
        self.refactor_pivot_row.push(row);
        self.refactor_pivot_var.push(var);
        self.refactor_pivot_type.push(pivot_type);
    }

    /// Form PBQ = LU for the basis matrix or report its rank deficiency.
    /// `refactor` is the refactorization information when it is to be
    /// used; `refactored` is set if it was used successfully. Returns the
    /// rank deficiency, or BUILD_KERNEL_RETURN_TIMEOUT.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        pivot_threshold: f64,
        pivot_tolerance: f64,
        time_limit: f64,
        a: &AMatrix,
        basic_index: &mut [i32],
        refactor: Option<&RefactorIn>,
        refactored: &mut bool,
    ) -> i32 {
        // Set up a timer to prevent build running longer than time_limit,
        // which is only finite in HPresolve::removeDependentEquations
        let build_timer = Instant::now();
        self.pivot_threshold = pivot_threshold;
        self.pivot_tolerance = pivot_tolerance;
        self.time_limit = time_limit;
        *refactored = false;
        if let Some(info) = refactor {
            self.rank_deficiency = self.rebuild(a, basic_index, info);
            if self.rank_deficiency == 0 {
                *refactored = true;
                return 0;
            }
        }
        // Refactoring from just the list of basic variables. Initialise the
        // refactorization information.
        self.refactor_clear();
        self.build_synthetic_tick = 0.0;
        self.build_simple(a, basic_index);
        let build_kernel_return = self.build_kernel(basic_index, build_timer);
        // A timeout return is negative, otherwise it's the rank deficiency
        // of the basic variables. If num_basic < num_row, the logicals
        // required to complete the basis are identified by continuing as
        // if a full-dimension set of basic variables was rank deficient.
        if build_kernel_return == BUILD_KERNEL_RETURN_TIMEOUT {
            return BUILD_KERNEL_RETURN_TIMEOUT;
        }
        self.rank_deficiency = build_kernel_return;
        let incomplete_basis = self.num_basic < self.num_row;
        if self.rank_deficiency != 0 || incomplete_basis {
            // Singular matrix B: reorder the basic variables so that the
            // singular columns are in the position corresponding to the
            // logical which replaces them
            self.build_handle_rank_deficiency(basic_index);
            self.build_mark_sing_c(a.num_col, basic_index);
        }
        if incomplete_basis {
            // Completing the factorization is not relevant if the basis
            // matrix is incomplete
            self.refactor_clear();
            return self.rank_deficiency - (self.num_row - self.num_basic);
        }
        self.build_finish(basic_index, false);
        // The refactorization information is known unless the basis was
        // rank deficient. Record build_synthetic_tick to use as the
        // value of build_synthetic_tick if refactorization is performed
        if self.rank_deficiency != 0 {
            self.refactor_clear();
        } else {
            self.refactor_build_synthetic_tick = self.build_synthetic_tick;
        }
        let nr = self.num_row as usize;
        self.invert_num_el = self.l_start[nr] + self.u_last_p[nr - 1] + self.num_row;
        self.kernel_dim -= self.rank_deficiency;
        self.rank_deficiency
    }

    fn lu_clear(&mut self) {
        self.l_start.clear();
        self.l_start.push(0);
        self.l_index.clear();
        self.l_value.clear();
        self.u_pivot_index.clear();
        self.u_pivot_value.clear();
        self.u_start.clear();
        self.u_start.push(0);
        self.u_index.clear();
        self.u_value.clear();
    }

    fn push_unit_pivot(&mut self, i_row: i32) {
        self.l_start.push(self.l_index.len() as i32);
        self.u_pivot_index.push(i_row);
        self.u_pivot_value.push(1.0);
        self.u_start.push(self.u_index.len() as i32);
    }

    fn build_simple(&mut self, a: &AMatrix, basic_index: &[i32]) {
        // 0. Clear L and U factor
        self.lu_clear();
        let num_row = self.num_row;
        let num_basic = self.num_basic;
        let num_col = a.num_col;
        let nb = num_basic as usize;

        // Set all values of permute to -1 so that unpermuted (rank
        // deficient) columns can be identified
        self.permute.clear();
        self.permute.resize(nb, -1);

        // 1. Prepare basis matrix and deal with unit columns
        let mut b_count_x: i32 = 0;
        self.mr_count_before[..num_row as usize].fill(0);
        self.nwork = 0;
        // Compile a vector iwork of the indices within basic_index of the
        // its nwork non-unit structural columns: they will be formed into
        // the B matrix as the kernel
        self.iwork.clear();
        self.iwork.resize(nb + 1, 0);
        for i_col in 0..nb {
            let i_mat = basic_index[i_col];
            let mut i_row = -1;
            let mut pivot_type = -1i8;
            // Look for unit columns as pivots. If there is already a pivot
            // corresponding to the nonzero in a unit column - evidenced by
            // mr_count_before[iRow] being negative - then it can't be used,
            // so treat it as a column to be handled in the kernel, so that
            // any rank deficiency or singularity is detected as late as
            // possible.
            if i_mat >= num_col {
                // 1.1 Logical column
                let lc_i_row = i_mat - num_col;
                if self.mr_count_before[lc_i_row as usize] >= 0 {
                    pivot_type = PIVOT_LOGICAL;
                    i_row = lc_i_row;
                } else {
                    self.mr_count_before[lc_i_row as usize] += 1;
                    self.b_index[b_count_x as usize] = lc_i_row;
                    self.b_value[b_count_x as usize] = 1.0;
                    b_count_x += 1;
                    self.iwork[self.nwork as usize] = i_col as i32;
                    self.nwork += 1;
                }
            } else {
                // 1.2 Structural column
                let start = a.start[i_mat as usize];
                let count = a.start[i_mat as usize + 1] - start;
                let s = start as usize;
                let ok_unit_col = count == 1
                    && a.value[s] == 1.0
                    && self.mr_count_before[a.index[s] as usize] >= 0;
                if ok_unit_col {
                    // Don't exploit this special case in case the matrix is
                    // re-factorized after scaling has been applied, making
                    // this column non-unit.
                    pivot_type = PIVOT_COL_SINGLETON;
                    i_row = a.index[s];
                } else {
                    for k in s..(start + count) as usize {
                        self.mr_count_before[a.index[k] as usize] += 1;
                        self.b_index[b_count_x as usize] = a.index[k];
                        self.b_value[b_count_x as usize] = a.value[k];
                        b_count_x += 1;
                    }
                    self.iwork[self.nwork as usize] = i_col as i32;
                    self.nwork += 1;
                }
            }

            if i_row >= 0 {
                // 1.3 Record unit column
                self.permute[i_col] = i_row;
                self.push_unit_pivot(i_row);
                // The negation needs to be great enough so that, starting
                // from it, the accumulated count can never reach zero
                self.mr_count_before[i_row as usize] = -num_basic;
                self.refactor_push(i_row, i_mat, pivot_type);
            }
            self.b_start[i_col + 1] = b_count_x;
            self.b_var[i_col] = i_mat;
        }
        // Record the number of elements in the basis matrix
        self.basis_matrix_num_el = num_row - self.nwork + b_count_x;
        self.build_synthetic_tick += (b_count_x * 60 + (num_row - self.nwork) * 80) as f64;

        // 2. Search for and deal with singletons
        let mut t2_search = 0.0;
        let mut t2_store_l = self.l_index.len() as f64;
        let mut t2_store_u = self.u_index.len() as f64;
        let mut t2_store_p = self.nwork as f64;
        while self.nwork > 0 {
            let nwork_last = self.nwork;
            self.nwork = 0;
            for i in 0..nwork_last as usize {
                let i_col = self.iwork[i] as usize;
                let start = self.b_start[i_col] as usize;
                let end = self.b_start[i_col + 1] as usize;
                let mut pivot_k = usize::MAX;
                let mut found_row_singleton = false;
                let mut count = 0;

                // 2.1 Search for singleton
                t2_search += (end - start) as f64;
                for k in start..end {
                    let i_row = self.b_index[k] as usize;
                    if self.mr_count_before[i_row] == 1 {
                        pivot_k = k;
                        found_row_singleton = true;
                        break;
                    }
                    if self.mr_count_before[i_row] > 1 {
                        pivot_k = k;
                        count += 1;
                    }
                }

                if found_row_singleton {
                    // 2.2 Deal with row singleton
                    let pivot_multiplier = 1.0 / self.b_value[pivot_k];
                    for k in (start..pivot_k).chain(pivot_k + 1..end) {
                        let i_row = self.b_index[k];
                        if self.mr_count_before[i_row as usize] > 0 {
                            self.l_index.push(i_row);
                            self.l_value.push(self.b_value[k] * pivot_multiplier);
                        } else {
                            self.u_index.push(i_row);
                            self.u_value.push(self.b_value[k]);
                        }
                        self.mr_count_before[i_row as usize] -= 1;
                    }
                    let i_row = self.b_index[pivot_k];
                    self.mr_count_before[i_row as usize] = 0;
                    self.permute[i_col] = i_row;
                    self.l_start.push(self.l_index.len() as i32);
                    self.u_pivot_index.push(i_row);
                    self.u_pivot_value.push(self.b_value[pivot_k]);
                    self.u_start.push(self.u_index.len() as i32);
                    self.refactor_push(i_row, basic_index[i_col], PIVOT_ROW_SINGLETON);
                } else if count == 1 {
                    // 2.3 Deal with column singleton
                    for k in (start..pivot_k).chain(pivot_k + 1..end) {
                        self.u_index.push(self.b_index[k]);
                        self.u_value.push(self.b_value[k]);
                    }
                    let i_row = self.b_index[pivot_k];
                    self.mr_count_before[i_row as usize] = 0;
                    self.permute[i_col] = i_row;
                    self.l_start.push(self.l_index.len() as i32);
                    self.u_pivot_index.push(i_row);
                    self.u_pivot_value.push(self.b_value[pivot_k]);
                    self.u_start.push(self.u_index.len() as i32);
                    self.refactor_push(i_row, basic_index[i_col], PIVOT_COL_SINGLETON);
                } else {
                    self.iwork[self.nwork as usize] = i_col as i32;
                    self.nwork += 1;
                }
            }
            // No singleton found in the last pass
            if nwork_last == self.nwork {
                break;
            }
        }
        t2_store_l = self.l_index.len() as f64 - t2_store_l;
        t2_store_u = self.u_index.len() as f64 - t2_store_u;
        t2_store_p -= self.nwork as f64;
        self.build_synthetic_tick +=
            t2_search.mul_add(20.0, (t2_store_p + t2_store_l + t2_store_u) * 80.0);

        // 3. Prepare the kernel parts
        //
        // 3.1 Prepare row links, row matrix spaces
        self.row_link_first.clear();
        self.row_link_first.resize(nb + 1, -1);
        self.mr_count.clear();
        self.mr_count.resize(num_row as usize, 0);
        let mut mr_count_x = 0;
        // Determine the number of entries in the kernel
        self.kernel_num_el = 0;
        for i_row in 0..num_row as usize {
            let count = self.mr_count_before[i_row];
            if count > 0 {
                self.mr_start[i_row] = mr_count_x;
                self.mr_space[i_row] = count * 2;
                mr_count_x += count * 2;
                self.rlink_add(i_row, count);
                self.kernel_num_el += count + 1;
            }
        }
        self.mr_index.resize(mr_count_x as usize, 0);

        // 3.2 Prepare column links, kernel matrix
        self.col_link_first.clear();
        self.col_link_first.resize(num_row as usize + 1, -1);
        self.mc_index.clear();
        self.mc_value.clear();
        self.mc_count_a.clear();
        self.mc_count_a.resize(nb, 0);
        self.mc_count_n.clear();
        self.mc_count_n.resize(nb, 0);
        let mut mc_count_x = 0;
        for i in 0..self.nwork as usize {
            let i_col = self.iwork[i] as usize;
            self.mc_var[i_col] = self.b_var[i_col];
            self.mc_start[i_col] = mc_count_x;
            self.mc_space[i_col] = (self.b_start[i_col + 1] - self.b_start[i_col]) * 2;
            mc_count_x += self.mc_space[i_col];
            self.mc_index.resize(mc_count_x as usize, 0);
            self.mc_value.resize(mc_count_x as usize, 0.0);
            for k in self.b_start[i_col] as usize..self.b_start[i_col + 1] as usize {
                let i_row = self.b_index[k];
                let value = self.b_value[k];
                if self.mr_count_before[i_row as usize] > 0 {
                    self.col_insert(i_col, i_row, value);
                    self.row_insert(i_col as i32, i_row as usize);
                } else {
                    self.col_store_n(i_col, i_row, value);
                }
            }
            self.col_fix_max(i_col);
            self.clink_add(i_col, self.mc_count_a[i_col]);
        }
        self.build_synthetic_tick +=
            ((num_row + self.nwork + mc_count_x) * 40 + mr_count_x * 20) as f64;
        // Record the kernel dimension
        self.kernel_dim = self.nwork;
    }

    fn build_kernel(&mut self, basic_index: &[i32], build_timer: Instant) -> i32 {
        // Deal with the kernel part by 'n-work' pivoting
        let mut fake_search = 0.0;
        let mut fake_fill = 0.0;
        let mut fake_eliminate = 0.0;

        // Initial timer frequency: may be reduced if iterations get slow
        let mut timer_frequency: i32 = 100;
        let mut previous_iteration_time = 0.0;
        let mut average_iteration_time = 0.0;
        let check_for_timeout = self.time_limit < f64::INFINITY;
        let mut search_k: i32 = 0;
        let num_row = self.num_row;
        let num_basic = self.num_basic;

        loop {
            // while (nwork-- > 0)
            let more = self.nwork > 0;
            self.nwork -= 1;
            if !more {
                break;
            }
            // Determine whether to return due to exceeding the time limit
            if check_for_timeout && search_k % timer_frequency == 0 {
                let current_time = build_timer.elapsed().as_secs_f64();
                let time_difference = current_time - previous_iteration_time;
                previous_iteration_time = current_time;
                let iteration_time = time_difference / timer_frequency as f64;
                average_iteration_time =
                    0.9f64.mul_add(average_iteration_time, 0.1 * iteration_time);
                if time_difference > self.time_limit / 1e3 {
                    timer_frequency = 1.max(timer_frequency / 10);
                }
                let iterations_left = self.kernel_dim - search_k + 1;
                let remaining_time_bound = average_iteration_time * iterations_left as f64;
                let total_time_bound = current_time + remaining_time_bound;
                if current_time > self.time_limit || total_time_bound > self.time_limit {
                    return BUILD_KERNEL_RETURN_TIMEOUT;
                }
            }

            // 1. Search for the pivot
            let mut j_col_pivot: i32 = -1;
            let mut i_row_pivot: i32 = -1;
            // 1.1. Setup search merits
            let search_limit = self.nwork.min(8);
            let mut search_count = 0;
            let merit_limit = 1.0 * num_basic as f64 * num_row as f64;
            let mut merit_pivot = merit_limit;
            search_k += 1;

            // 1.2. Search for local singletons
            let mut found_pivot = false;
            if self.col_link_first[1] != -1 {
                j_col_pivot = self.col_link_first[1];
                i_row_pivot = self.mc_index[self.mc_start[j_col_pivot as usize] as usize];
                found_pivot = true;
            }
            if !found_pivot && self.row_link_first[1] != -1 {
                i_row_pivot = self.row_link_first[1];
                j_col_pivot = self.mr_index[self.mr_start[i_row_pivot as usize] as usize];
                found_pivot = true;
            }

            // 1.3. Major search loop
            //
            // Row count can be more than the number of rows if num_basic >
            // num_row
            let max_count = num_row.max(num_basic);
            let mut count = 2;
            while !found_pivot && count <= max_count {
                // Column count cannot exceed the number of rows
                if count <= num_row {
                    // 1.3.1 Search for columns
                    let mut j = self.col_link_first[count as usize];
                    while j != -1 {
                        let ju = j as usize;
                        let min_pivot = self.mc_min_pivot[ju];
                        let start = self.mc_start[ju] as usize;
                        let end = start + self.mc_count_a[ju] as usize;
                        for k in start..end {
                            if self.mc_value[k].abs() >= min_pivot {
                                let i = self.mc_index[k];
                                let row_count = self.mr_count[i as usize];
                                let merit_local = 1.0 * (count - 1) as f64 * (row_count - 1) as f64;
                                if merit_pivot > merit_local {
                                    merit_pivot = merit_local;
                                    j_col_pivot = j;
                                    i_row_pivot = i;
                                    found_pivot = found_pivot || (row_count < count);
                                }
                            }
                        }
                        let sc = search_count;
                        search_count += 1;
                        if sc >= search_limit && merit_pivot < merit_limit {
                            found_pivot = true;
                        }
                        if found_pivot {
                            break;
                        }
                        fake_search += count as f64;
                        j = self.col_link_next[ju];
                    }
                }

                // Row count cannot exceed the number of basic variables
                if count <= num_basic {
                    // 1.3.2 Search for rows
                    let mut i = self.row_link_first[count as usize];
                    while i != -1 {
                        let iu = i as usize;
                        let start = self.mr_start[iu] as usize;
                        let end = start + self.mr_count[iu] as usize;
                        for k in start..end {
                            let j = self.mr_index[k];
                            let ju = j as usize;
                            let column_count = self.mc_count_a[ju];
                            let merit_local = 1.0 * (count - 1) as f64 * (column_count - 1) as f64;
                            if merit_local < merit_pivot {
                                let mut ifind = self.mc_start[ju] as usize;
                                while self.mc_index[ifind] != i {
                                    ifind += 1;
                                }
                                if self.mc_value[ifind].abs() >= self.mc_min_pivot[ju] {
                                    merit_pivot = merit_local;
                                    j_col_pivot = j;
                                    i_row_pivot = i;
                                    found_pivot = found_pivot || (column_count <= count);
                                }
                            }
                        }
                        let sc = search_count;
                        search_count += 1;
                        if sc >= search_limit && merit_pivot < merit_limit {
                            found_pivot = true;
                        }
                        if found_pivot {
                            break;
                        }
                        i = self.row_link_next[iu];
                    }
                    fake_search += count as f64;
                }
                count += 1;
            }
            // 1.4. If we found nothing: tell singular
            if i_row_pivot < 0 {
                self.rank_deficiency = self.nwork + 1;
                return self.rank_deficiency;
            }

            // 2. Elimination other elements by the pivot
            let jp = j_col_pivot as usize;
            let ip = i_row_pivot as usize;
            // 2.1. Delete the pivot
            //
            // Remove the pivot row index from the pivotal column of the
            // col-wise matrix, the pivot column index from the pivotal row
            // of the row-wise matrix, and both from their linked lists
            let pivot_multiplier = self.col_delete(jp, i_row_pivot);
            self.row_delete(j_col_pivot, ip);
            self.clink_del(jp);
            self.rlink_del(ip);
            if pivot_multiplier.abs() < self.pivot_tolerance {
                // Matrix is singular, but defer return since other valid
                // pivots may exist.
                if self.mr_count[ip] == 0 {
                    // The pivot corresponds to a singleton row. Entry is
                    // zeroed, and do no more since there may be other valid
                    // entries in the pivotal column
                    self.clink_add(jp, self.mc_count_a[jp]);
                } else {
                    // Otherwise, other entries in the pivotal column will be
                    // smaller than the pivot, so zero the column
                    self.zero_col(jp);
                    self.rlink_add(ip, self.mr_count[ip]);
                }
                // No pivot found, so have to increment nwork
                self.nwork += 1;
                continue;
            }
            self.permute[jp] = i_row_pivot;
            self.refactor_push(i_row_pivot, basic_index[jp], PIVOT_MARKOWITZ);

            // 2.2. Store active pivot column to L
            let start_a = self.mc_start[jp] as usize;
            let end_a = start_a + self.mc_count_a[jp] as usize;
            let mut mwz_column_count = 0;
            for k in start_a..end_a {
                let i_row = self.mc_index[k];
                let value = self.mc_value[k] / pivot_multiplier;
                self.mwz_column_index[mwz_column_count] = i_row;
                mwz_column_count += 1;
                self.mwz_column_array[i_row as usize] = value;
                self.mwz_column_mark[i_row as usize] = 1;
                self.l_index.push(i_row);
                self.l_value.push(value);
                self.mr_count_before[i_row as usize] = self.mr_count[i_row as usize];
                self.row_delete(j_col_pivot, i_row as usize);
            }
            self.l_start.push(self.l_index.len() as i32);
            fake_fill += (2 * self.mc_count_a[jp]) as f64;

            // 2.3. Store non active pivot column to U
            let end_n = start_a + self.mc_space[jp] as usize;
            let start_n = end_n - self.mc_count_n[jp] as usize;
            for i in start_n..end_n {
                self.u_index.push(self.mc_index[i]);
                self.u_value.push(self.mc_value[i]);
            }
            self.u_pivot_index.push(i_row_pivot);
            self.u_pivot_value.push(pivot_multiplier);
            self.u_start.push(self.u_index.len() as i32);
            fake_fill += (end_n - start_n) as f64;

            // 2.4. Loop over pivot row to eliminate other column
            let row_start = self.mr_start[ip] as usize;
            let row_end = row_start + self.mr_count[ip] as usize;
            for row_k in row_start..row_end {
                // 2.4.1. My pointer
                let i_col = self.mr_index[row_k] as usize;
                let my_count = self.mc_count_a[i_col];
                let my_start = self.mc_start[i_col] as usize;
                let my_end = my_start + my_count as usize - 1;
                let my_pivot = self.col_delete(i_col, i_row_pivot);
                self.col_store_n(i_col, i_row_pivot, my_pivot);

                // 2.4.2. Elimination on the overlapping part
                let mut n_fillin = mwz_column_count as i32;
                let mut n_cancel = 0;
                for my_k in my_start..my_end {
                    let i_row = self.mc_index[my_k] as usize;
                    let mut value = self.mc_value[my_k];
                    if self.mwz_column_mark[i_row] != 0 {
                        self.mwz_column_mark[i_row] = 0;
                        n_fillin -= 1;
                        value = (-my_pivot).mul_add(self.mwz_column_array[i_row], value);
                        if value.abs() < K_HIGHS_TINY {
                            value = 0.0;
                            n_cancel += 1;
                        }
                        self.mc_value[my_k] = value;
                    }
                }
                fake_eliminate += mwz_column_count as f64;
                fake_eliminate += (n_fillin * 2) as f64;

                // 2.4.3. Remove cancellation gaps
                if n_cancel > 0 {
                    let mut new_end = my_start;
                    for my_k in my_start..my_end {
                        if self.mc_value[my_k] != 0.0 {
                            self.mc_index[new_end] = self.mc_index[my_k];
                            self.mc_value[new_end] = self.mc_value[my_k];
                            new_end += 1;
                        } else {
                            let r = self.mc_index[my_k] as usize;
                            self.row_delete(i_col as i32, r);
                        }
                    }
                    self.mc_count_a[i_col] = (new_end - my_start) as i32;
                }

                // 2.4.4. Insert fill-in
                if n_fillin > 0 {
                    // 2.4.4.1 Check column size
                    if self.mc_count_a[i_col] + self.mc_count_n[i_col] + n_fillin
                        > self.mc_space[i_col]
                    {
                        // p1&2=active, p3&4=non active, p5=new p1, p7=new p3
                        let p1 = self.mc_start[i_col] as usize;
                        let p2 = p1 + self.mc_count_a[i_col] as usize;
                        let p3 = p1 + (self.mc_space[i_col] - self.mc_count_n[i_col]) as usize;
                        let p4 = p1 + self.mc_space[i_col] as usize;
                        self.mc_space[i_col] += self.mc_space[i_col].max(n_fillin);
                        let p5 = self.mc_index.len();
                        self.mc_start[i_col] = p5 as i32;
                        let p7 = p5 + (self.mc_space[i_col] - self.mc_count_n[i_col]) as usize;
                        let new_len = p5 + self.mc_space[i_col] as usize;
                        self.mc_index.resize(new_len, 0);
                        self.mc_value.resize(new_len, 0.0);
                        self.mc_index.copy_within(p1..p2, p5);
                        self.mc_value.copy_within(p1..p2, p5);
                        self.mc_index.copy_within(p3..p4, p7);
                        self.mc_value.copy_within(p3..p4, p7);
                    }

                    // 2.4.4.2 Fill into column copy
                    for i in 0..mwz_column_count {
                        let i_row = self.mwz_column_index[i];
                        if self.mwz_column_mark[i_row as usize] != 0 {
                            let v = -my_pivot * self.mwz_column_array[i_row as usize];
                            self.col_insert(i_col, i_row, v);
                        }
                    }

                    // 2.4.4.3 Fill into the row copy
                    for i in 0..mwz_column_count {
                        let i_row = self.mwz_column_index[i] as usize;
                        if self.mwz_column_mark[i_row] != 0 {
                            // Expand row space
                            if self.mr_count[i_row] == self.mr_space[i_row] {
                                let p1 = self.mr_start[i_row] as usize;
                                let p2 = p1 + self.mr_count[i_row] as usize;
                                let p3 = self.mr_index.len();
                                self.mr_start[i_row] = p3 as i32;
                                self.mr_space[i_row] *= 2;
                                self.mr_index.resize(p3 + self.mr_space[i_row] as usize, 0);
                                self.mr_index.copy_within(p1..p2, p3);
                            }
                            self.row_insert(i_col as i32, i_row);
                        }
                    }
                }

                // 2.4.5. Reset pivot column mark
                for i in 0..mwz_column_count {
                    self.mwz_column_mark[self.mwz_column_index[i] as usize] = 1;
                }

                // 2.4.6. Fix max value and link list
                self.col_fix_max(i_col);
                if my_count != self.mc_count_a[i_col] {
                    self.clink_del(i_col);
                    self.clink_add(i_col, self.mc_count_a[i_col]);
                }
            }

            // 2.5. Clear pivot column buffer
            for i in 0..mwz_column_count {
                self.mwz_column_mark[self.mwz_column_index[i] as usize] = 0;
            }

            // 2.6. Correct row links for the remain active part
            for i in start_a..end_a {
                let i_row = self.mc_index[i] as usize;
                if self.mr_count_before[i_row] != self.mr_count[i_row] {
                    self.rlink_del(i_row);
                    self.rlink_add(i_row, self.mr_count[i_row]);
                }
            }
        }
        self.build_synthetic_tick +=
            fake_eliminate.mul_add(80.0, fake_search.mul_add(20.0, fake_fill * 160.0));
        self.rank_deficiency = 0;
        0
    }

    fn build_handle_rank_deficiency(&mut self, basic_index: &[i32]) {
        // iwork can now be used as workspace: use it to accumulate the new
        // basic_index. iwork is set to -1 and basic_index is permuted into
        // it. Indices of iwork corresponding to missing indices in permute
        // remain -1. Hence the -1's become markers for the logicals which
        // will replace singular columns.
        //
        // On entry, rank_deficiency is the rank deficiency of basic_index,
        // which is less than the rank deficiency of the basis matrix if
        // num_basic < num_row
        let num_row = self.num_row;
        let num_basic = self.num_basic;
        if num_basic < num_row {
            self.rank_deficiency += num_row - num_basic;
        }
        let rd = self.rank_deficiency as usize;
        self.row_with_no_pivot.resize(rd, 0);
        self.col_with_no_pivot.resize(rd, 0);
        let mut lc_rank_deficiency = 0usize;
        if num_basic < num_row {
            self.iwork.resize(num_row as usize, 0);
        } else if num_basic > num_row {
            self.iwork.resize(num_basic as usize, 0);
        }
        self.iwork[..num_row as usize].fill(-1);
        for i in 0..num_basic as usize {
            let perm_i = self.permute[i];
            if perm_i >= 0 {
                self.iwork[perm_i as usize] = basic_index[i];
            } else {
                self.col_with_no_pivot[lc_rank_deficiency] = i as i32;
                lc_rank_deficiency += 1;
            }
        }
        if num_basic < num_row {
            // Resize permute and complete iwork and col_with_no_pivot with
            // fictitious indices and entries of basic_index
            self.permute.resize(num_row as usize, 0);
            for i in num_basic..num_row {
                self.col_with_no_pivot[lc_rank_deficiency] = i;
                lc_rank_deficiency += 1;
                self.permute[i as usize] = -1;
            }
        }
        lc_rank_deficiency = 0;
        for i in 0..num_row as usize {
            if self.iwork[i] < 0 {
                // Record the rows with no pivots in row_with_no_pivot and
                // indicate them within iwork by storing the negation of one
                // more than their rank deficiency counter
                self.row_with_no_pivot[lc_rank_deficiency] = i as i32;
                self.iwork[i] = -(lc_rank_deficiency as i32 + 1);
                lc_rank_deficiency += 1;
            }
        }
        if num_row < num_basic {
            // Record fictitious rows with no pivots for the excess basic
            // variables so that permute will be constructed as a
            // permutation of all entries in basic_index
            for i in num_row..num_basic {
                self.row_with_no_pivot[lc_rank_deficiency] = i;
                self.iwork[i as usize] = -(lc_rank_deficiency as i32 + 1);
                lc_rank_deficiency += 1;
            }
        }
        let row_rank_deficiency = self.rank_deficiency - (num_basic - num_row).max(0);
        // Complete the permutation using the indices of rows with no pivot,
        // the last max(num_basic-num_row, 0) of which will be fictitious
        for k in 0..rd {
            let i_row = self.row_with_no_pivot[k];
            let i_col = self.col_with_no_pivot[k];
            self.permute[i_col as usize] = i_row;
            if (k as i32) < row_rank_deficiency {
                // Only correct the factorization for the true rows
                self.push_unit_pivot(i_row);
            }
        }
    }

    fn build_mark_sing_c(&mut self, num_col: i32, basic_index: &mut [i32]) {
        // Singular matrix B: reorder the basic variables so that the
        // singular columns are in the position corresponding to the
        // logical which replaces them
        let rd = self.rank_deficiency as usize;
        self.var_with_no_pivot.resize(rd, 0);
        for k in 0..rd {
            let asm_row = self.row_with_no_pivot[k];
            let asm_col = self.col_with_no_pivot[k];
            // Store negation of 1+ASMcol so that removing column 0 can be
            // identified!
            self.iwork[asm_row as usize] = -(asm_col + 1);
            // Only update basic_index for the true entries
            if asm_col < self.num_basic {
                // Record the variable in basic_index that had no pivot, and
                // replace it with the logical
                self.var_with_no_pivot[k] = basic_index[asm_col as usize];
                basic_index[asm_col as usize] = num_col + asm_row;
            } else if self.num_basic < self.num_row {
                // Record an illegal variable when there's no index to
                // displace
                self.var_with_no_pivot[k] = -1;
            }
        }
    }

    fn build_finish(&mut self, basic_index: &mut [i32], refactor_use: bool) {
        // Must only be called in the case where there are at least as many
        // basic variables as rows
        let num_row = self.num_row;
        let nr = num_row as usize;
        // The look up table
        for i in 0..nr {
            self.u_pivot_lookup[self.u_pivot_index[i] as usize] = i as i32;
        }
        self.l_pivot_index.clone_from(&self.u_pivot_index);
        self.l_pivot_lookup.clone_from(&self.u_pivot_lookup);

        // LR space
        let l_count_x = self.l_index.len();
        self.lr_index.resize(l_count_x, 0);
        self.lr_value.resize(l_count_x, 0.0);

        // LR pointer
        self.iwork.clear();
        self.iwork.resize(nr, 0);
        for k in 0..l_count_x {
            self.iwork[self.l_pivot_lookup[self.l_index[k] as usize] as usize] += 1;
        }
        self.lr_start.clear();
        self.lr_start.resize(nr + 1, 0);
        for i in 1..=nr {
            self.lr_start[i] = self.lr_start[i - 1] + self.iwork[i - 1];
        }

        // LR elements
        self.iwork.clear();
        self.iwork.extend_from_slice(&self.lr_start[..nr]);
        for i in 0..nr {
            let index = self.l_pivot_index[i];
            for k in self.l_start[i] as usize..self.l_start[i + 1] as usize {
                let i_row = self.l_pivot_lookup[self.l_index[k] as usize] as usize;
                let i_put = self.iwork[i_row] as usize;
                self.iwork[i_row] += 1;
                self.lr_index[i_put] = index;
                self.lr_value[i_put] = self.l_value[k];
            }
        }

        // U pointer
        self.u_last_p.clear();
        self.u_last_p.extend_from_slice(&self.u_start[1..nr + 1]);
        self.u_start.truncate(nr);

        // UR space
        let u_count_x = self.u_index.len();
        let ur_stuff_size: i32 = if self.update_method == UPDATE_FT {
            5
        } else {
            0
        };
        let ur_count_size = u_count_x + (ur_stuff_size * num_row) as usize;
        self.ur_index.resize(ur_count_size, 0);
        self.ur_value.resize(ur_count_size, 0.0);

        // UR pointer
        //
        // NB ur_lastp just being used as temporary storage here
        self.ur_start.clear();
        self.ur_start.resize(nr + 1, 0);
        self.ur_lastp.clear();
        self.ur_lastp.resize(nr, 0);
        self.ur_space.clear();
        self.ur_space.resize(nr, ur_stuff_size);
        for k in 0..u_count_x {
            self.ur_lastp[self.u_pivot_lookup[self.u_index[k] as usize] as usize] += 1;
        }
        for i in 1..=nr {
            self.ur_start[i] = self.ur_start[i - 1] + self.ur_lastp[i - 1] + ur_stuff_size;
        }
        self.ur_start.truncate(nr);

        // UR element
        //
        // NB ur_lastp initialised here!
        self.ur_lastp.clone_from(&self.ur_start);
        for i in 0..nr {
            let index = self.u_pivot_index[i];
            for k in self.u_start[i] as usize..self.u_last_p[i] as usize {
                let i_row = self.u_pivot_lookup[self.u_index[k] as usize] as usize;
                let i_put = self.ur_lastp[i_row] as usize;
                self.ur_lastp[i_row] += 1;
                self.ur_index[i_put] = index;
                self.ur_value[i_put] = self.u_value[k];
            }
        }

        // Re-factor merit
        self.u_merit_x =
            ((l_count_x + u_count_x) as i32 as f64).mul_add(1.5, num_row as f64) as i32;
        self.u_total_x = u_count_x as i32;
        if self.update_method == UPDATE_PF {
            self.u_merit_x = num_row + u_count_x as i32 * 4;
        }
        if self.update_method == UPDATE_MPF {
            self.u_merit_x = num_row + u_count_x as i32 * 3;
        }

        // Clear update buffer
        self.pf_pivot_value.clear();
        self.pf_pivot_index.clear();
        self.pf_start.clear();
        self.pf_start.push(0);
        self.pf_index.clear();
        self.pf_value.clear();

        if !refactor_use {
            // Finally, if not calling buildFinish after refactorizing,
            // permute the basic variables
            let nb = self.num_basic as usize;
            self.iwork.clear();
            self.iwork.extend_from_slice(&basic_index[..nb]);
            for i in 0..nb {
                basic_index[self.permute[i] as usize] = self.iwork[i];
            }
            // Add cost of buildFinish to build_synthetic_tick
            self.build_synthetic_tick +=
                (num_row * 80 + (l_count_x + u_count_x) as i32 * 60) as f64;
        }
    }

    fn zero_col(&mut self, j_col: usize) {
        let a_start = self.mc_start[j_col] as usize;
        let a_end = a_start + self.mc_count_a[j_col] as usize;
        for i_el in a_start..a_end {
            let i_row = self.mc_index[i_el] as usize;
            // Remove the column index from this row of the row-wise matrix,
            // and move the row to the linked list for its reduced count
            self.row_delete(j_col as i32, i_row);
            self.rlink_del(i_row);
            self.rlink_add(i_row, self.mr_count[i_row]);
        }
        // Remove the column from the linked list of columns containing it
        self.clink_del(j_col);
        // Zero the counts of the active and inactive sections of the column
        self.mc_count_a[j_col] = 0;
        self.mc_count_n[j_col] = 0;
    }

    /// Rebuild using refactor information
    fn rebuild(&mut self, a: &AMatrix, basic_index: &mut [i32], info: &RefactorIn) -> i32 {
        // 0. Clear L and U factor
        self.lu_clear();
        let num_row = self.num_row;
        let nr = num_row as usize;
        self.nwork = 0;
        self.basis_matrix_num_el = 0;
        let mut stage = nr;
        let mut has_pivot = vec![false; nr];
        // Take build_synthetic_tick from the refactor info so that this
        // refactorization doesn't look unrealistically cheap.
        self.build_synthetic_tick = info.build_synthetic_tick;
        for i_k in 0..nr {
            let i_row = info.pivot_row[i_k];
            let i_var = info.pivot_var[i_k];
            let pivot_type = info.pivot_type[i_k];
            if pivot_type == PIVOT_LOGICAL || pivot_type == PIVOT_UNIT {
                // 1.1 Logical column, or 1.2 (structural) unit column
                self.basis_matrix_num_el += 1;
                // 1.3 Record unit column
                self.push_unit_pivot(i_row);
            } else if pivot_type == PIVOT_ROW_SINGLETON || pivot_type == PIVOT_COL_SINGLETON {
                // Row or column singleton
                let start = a.start[i_var as usize] as usize;
                let end = a.start[i_var as usize + 1] as usize;
                // Find where the pivot is
                let pivot_k = (start..end)
                    .find(|&k| a.index[k] == i_row)
                    .unwrap_or(usize::MAX);
                // Check that the pivot isn't too small. Shouldn't happen
                // since this is refactorization
                let abs_pivot = a.value[pivot_k].abs();
                if abs_pivot < self.pivot_tolerance {
                    return self.nwork + 1;
                }
                if pivot_type == PIVOT_ROW_SINGLETON {
                    // 2.2 Deal with row singleton
                    let pivot_multiplier = 1.0 / a.value[pivot_k];
                    for k in (start..pivot_k).chain(pivot_k + 1..end) {
                        let local_i_row = a.index[k];
                        if !has_pivot[local_i_row as usize] {
                            self.l_index.push(local_i_row);
                            self.l_value.push(a.value[k] * pivot_multiplier);
                        } else {
                            self.u_index.push(local_i_row);
                            self.u_value.push(a.value[k]);
                        }
                    }
                } else {
                    // 2.3 Deal with column singleton
                    for k in (start..pivot_k).chain(pivot_k + 1..end) {
                        self.u_index.push(a.index[k]);
                        self.u_value.push(a.value[k]);
                    }
                }
                self.l_start.push(self.l_index.len() as i32);
                self.u_pivot_index.push(i_row);
                self.u_pivot_value.push(a.value[pivot_k]);
                self.u_start.push(self.u_index.len() as i32);
            } else {
                stage = i_k;
                break;
            }
            basic_index[i_row as usize] = i_var;
            has_pivot[i_row as usize] = true;
        }
        if stage < nr {
            // Handle the remaining Markowitz pivots
            //
            // First of all complete the L factor with identity columns so
            // that FtranL counts the RHS entries in rows that don't yet
            // have pivots by running to completion. In the hyper-sparse
            // code, these will HOPEFULLY be skipped
            //
            // There are already l_start entries for the first stage rows,
            // but l_pivot_index is not assigned, as u_pivot_index gets
            // copied into it
            self.l_start.resize(nr + 1, 0);
            for i_k in stage..nr {
                self.l_start[i_k + 1] = self.l_start[i_k];
            }
            self.l_pivot_index.clear();
            self.l_pivot_index.extend_from_slice(&info.pivot_row[..nr]);
            // To do hyper-sparse FtranL operations, have to set up
            // l_pivot_lookup.
            self.l_pivot_lookup.resize(nr, 0);
            for i_row in 0..nr {
                self.l_pivot_lookup[self.l_pivot_index[i_row] as usize] = i_row as i32;
            }
            // Need to know whether to consider matrix entries for FtranL
            // operation. Initially these correspond to all the rows without
            // pivots
            let not_in_bump = has_pivot.clone();
            // Monitor density of FtranL result to possibly switch from
            // exploiting hyper-sparsity
            let mut expected_density = 0.0;
            // A vector in which the L and U entries of the pivotal column
            // will be formed
            let mut column = OwnedHVec::new(num_row);
            for i_k in stage..nr {
                let i_row = info.pivot_row[i_k];
                let i_var = info.pivot_var[i_k];
                // Set up the column for the FtranL. It contains the matrix
                // entries in rows without pivots, and the remaining entries
                // start forming the U column
                column.clear();
                for i_el in a.start[i_var as usize] as usize..a.start[i_var as usize + 1] as usize {
                    let local_i_row = a.index[i_el];
                    if not_in_bump[local_i_row as usize] {
                        self.u_index.push(local_i_row);
                        self.u_value.push(a.value[i_el]);
                    } else {
                        column.index[column.count as usize] = local_i_row;
                        column.count += 1;
                        column.array[local_i_row as usize] = a.value[i_el];
                    }
                }
                column.with(|c| self.ftran_l(c, expected_density));
                // Update the running average density
                let local_density = column.count as f64 / num_row as f64;
                expected_density = K_RUNNING_AVERAGE_MULTIPLIER.mul_add(
                    local_density,
                    (1.0 - K_RUNNING_AVERAGE_MULTIPLIER) * expected_density,
                );
                // Strip out small values
                column.with(|c| c.tight());
                // Now form the column of L
                let end = column.count as usize;
                let pivot_k = (0..end)
                    .find(|&k| column.index[k] == i_row)
                    .unwrap_or(usize::MAX);
                // Check that the pivot isn't too small. Shouldn't happen
                // since this is refactorization
                let abs_pivot = column.array[i_row as usize].abs();
                if abs_pivot < self.pivot_tolerance {
                    return num_row - i_k as i32;
                }
                let pivot_multiplier = 1.0 / column.array[i_row as usize];
                for k in (0..pivot_k).chain(pivot_k + 1..end) {
                    let local_i_row = column.index[k];
                    let v = column.array[local_i_row as usize];
                    if !has_pivot[local_i_row as usize] {
                        self.l_index.push(local_i_row);
                        self.l_value.push(v * pivot_multiplier);
                    } else {
                        self.u_index.push(local_i_row);
                        self.u_value.push(v);
                    }
                }
                self.l_start[i_k + 1] = self.l_index.len() as i32;
                self.u_pivot_index.push(i_row);
                self.u_pivot_value.push(column.array[i_row as usize]);
                self.u_start.push(self.u_index.len() as i32);
                basic_index[i_row as usize] = i_var;
                has_pivot[i_row as usize] = true;
            }
        }
        self.build_finish(basic_index, true);
        0
    }

    // Local helpers for the kernel matrix and count link lists
    fn col_insert(&mut self, i_col: usize, i_row: i32, value: f64) {
        let iput = (self.mc_start[i_col] + self.mc_count_a[i_col]) as usize;
        self.mc_count_a[i_col] += 1;
        self.mc_index[iput] = i_row;
        self.mc_value[iput] = value;
    }
    fn col_store_n(&mut self, i_col: usize, i_row: i32, value: f64) {
        self.mc_count_n[i_col] += 1;
        let iput = (self.mc_start[i_col] + self.mc_space[i_col] - self.mc_count_n[i_col]) as usize;
        self.mc_index[iput] = i_row;
        self.mc_value[iput] = value;
    }
    fn col_fix_max(&mut self, i_col: usize) {
        let start = self.mc_start[i_col] as usize;
        let end = start + self.mc_count_a[i_col] as usize;
        let mut max_value = 0.0;
        for &v in &self.mc_value[start..end] {
            max_value = cmax(max_value, v.abs());
        }
        self.mc_min_pivot[i_col] = max_value * self.pivot_threshold;
    }
    fn col_delete(&mut self, i_col: usize, i_row: i32) -> f64 {
        let mut idel = self.mc_start[i_col] as usize;
        self.mc_count_a[i_col] -= 1;
        let imov = idel + self.mc_count_a[i_col] as usize;
        while self.mc_index[idel] != i_row {
            idel += 1;
        }
        let pivot_multiplier = self.mc_value[idel];
        self.mc_index[idel] = self.mc_index[imov];
        self.mc_value[idel] = self.mc_value[imov];
        pivot_multiplier
    }
    fn row_insert(&mut self, i_col: i32, i_row: usize) {
        let iput = (self.mr_start[i_row] + self.mr_count[i_row]) as usize;
        self.mr_count[i_row] += 1;
        self.mr_index[iput] = i_col;
    }
    fn row_delete(&mut self, i_col: i32, i_row: usize) {
        let mut idel = self.mr_start[i_row] as usize;
        self.mr_count[i_row] -= 1;
        let imov = idel + self.mr_count[i_row] as usize;
        while self.mr_index[idel] != i_col {
            idel += 1;
        }
        self.mr_index[idel] = self.mr_index[imov];
    }
    fn clink_add(&mut self, index: usize, count: i32) {
        let mover = self.col_link_first[count as usize];
        self.col_link_last[index] = -2 - count;
        self.col_link_next[index] = mover;
        self.col_link_first[count as usize] = index as i32;
        if mover >= 0 {
            self.col_link_last[mover as usize] = index as i32;
        }
    }
    fn clink_del(&mut self, index: usize) {
        let xlast = self.col_link_last[index];
        let xnext = self.col_link_next[index];
        if xlast >= 0 {
            self.col_link_next[xlast as usize] = xnext;
        } else {
            self.col_link_first[(-xlast - 2) as usize] = xnext;
        }
        if xnext >= 0 {
            self.col_link_last[xnext as usize] = xlast;
        }
    }
    fn rlink_add(&mut self, index: usize, count: i32) {
        let mover = self.row_link_first[count as usize];
        self.row_link_last[index] = -2 - count;
        self.row_link_next[index] = mover;
        self.row_link_first[count as usize] = index as i32;
        if mover >= 0 {
            self.row_link_last[mover as usize] = index as i32;
        }
    }
    fn rlink_del(&mut self, index: usize) {
        let xlast = self.row_link_last[index];
        let xnext = self.row_link_next[index];
        if xlast >= 0 {
            self.row_link_next[xlast as usize] = xnext;
        } else {
            self.row_link_first[(-xlast - 2) as usize] = xnext;
        }
        if xnext >= 0 {
            self.row_link_last[xnext as usize] = xlast;
        }
    }

    /// Solve B x = b (ftranCall)
    pub fn ftran(&self, rhs: &mut HVec, expected_density: f64) {
        let use_indices = rhs.count >= 0;
        self.ftran_l(rhs, expected_density);
        self.ftran_u(rhs, expected_density);
        // Possibly find the indices in order
        if use_indices {
            rhs.re_index();
        }
    }

    /// Solve B^T x = b (btranCall)
    pub fn btran(&self, rhs: &mut HVec, expected_density: f64) {
        let use_indices = rhs.count >= 0;
        self.btran_u(rhs, expected_density);
        self.btran_l(rhs, expected_density);
        if use_indices {
            rhs.re_index();
        }
    }

    fn sparse_solve(&self, rhs: &HVec, expected_density: f64, hyper: f64) -> bool {
        let current_density = 1.0 * rhs.count as f64 * self.inv_num_row;
        rhs.count < 0 || current_density > K_HYPER_CANCEL || expected_density > hyper
    }

    fn ftran_l(&self, rhs: &mut HVec, expected_density: f64) {
        if self.update_method == UPDATE_APF {
            rhs.tight();
            rhs.pack();
            self.ftran_apf(rhs);
            rhs.tight();
        }
        if self.sparse_solve(rhs, expected_density, K_HYPER_FTRAN_L) {
            let num_row = self.num_row as usize;
            let (array, index) = (&mut *rhs.array, &mut *rhs.index);
            let (l_start, l_pivot_index) = (&self.l_start, &self.l_pivot_index);
            assert!(array.len() >= num_row && index.len() >= num_row);
            assert!(l_start.len() > num_row && l_pivot_index.len() >= num_row);
            let mut rhs_count = 0;
            for i in 0..num_row {
                // SAFETY: i < num_row and rhs_count <= i, within the lengths
                // checked above; pivot_row < num_row and the column's range
                // by the HFactor invariant
                unsafe {
                    let pivot_row = *l_pivot_index.get_unchecked(i);
                    let x = array.get_unchecked_mut(pivot_row as usize);
                    let pivot_multiplier = *x;
                    if pivot_multiplier.abs() > K_HIGHS_TINY {
                        *index.get_unchecked_mut(rhs_count) = pivot_row;
                        rhs_count += 1;
                        let (start, end) =
                            (*l_start.get_unchecked(i), *l_start.get_unchecked(i + 1));
                        let (idx, val) = entries(&self.l_index, &self.l_value, start, end);
                        axpy_unchecked(array, idx, val, pivot_multiplier);
                    } else {
                        *x = 0.0;
                    }
                }
            }
            rhs.count = rhs_count as i32;
        } else {
            let h = Triangle {
                lookup: &self.l_pivot_lookup,
                pivot_index: &self.l_pivot_index,
                pivot_value: None,
                start: &self.l_start,
                end: &self.l_start[1..],
                index: &self.l_index,
                value: &self.l_value,
            };
            // SAFETY: h is part of this factor
            unsafe { solve_hyper(self.num_row as usize, &h, rhs) };
        }
    }

    fn btran_l(&self, rhs: &mut HVec, expected_density: f64) {
        if self.sparse_solve(rhs, expected_density, K_HYPER_BTRAN_L) {
            let num_row = self.num_row as usize;
            let (array, index) = (&mut *rhs.array, &mut *rhs.index);
            let (lr_start, l_pivot_index) = (&self.lr_start, &self.l_pivot_index);
            assert!(array.len() >= num_row && index.len() >= num_row);
            assert!(lr_start.len() > num_row && l_pivot_index.len() >= num_row);
            let mut rhs_count = 0;
            for i in (0..num_row).rev() {
                // SAFETY: i < num_row and rhs_count < num_row - i, within
                // the lengths checked above; pivot_row < num_row and the
                // row's range by the HFactor invariant
                unsafe {
                    let pivot_row = *l_pivot_index.get_unchecked(i);
                    let x = array.get_unchecked_mut(pivot_row as usize);
                    let pivot_multiplier = *x;
                    if pivot_multiplier.abs() > K_HIGHS_TINY {
                        *index.get_unchecked_mut(rhs_count) = pivot_row;
                        rhs_count += 1;
                        let (start, end) =
                            (*lr_start.get_unchecked(i), *lr_start.get_unchecked(i + 1));
                        let (idx, val) = entries(&self.lr_index, &self.lr_value, start, end);
                        axpy_unchecked(array, idx, val, pivot_multiplier);
                    } else {
                        *x = 0.0;
                    }
                }
            }
            rhs.count = rhs_count as i32;
        } else {
            let h = Triangle {
                lookup: &self.l_pivot_lookup,
                pivot_index: &self.l_pivot_index,
                pivot_value: None,
                start: &self.lr_start,
                end: &self.lr_start[1..],
                index: &self.lr_index,
                value: &self.lr_value,
            };
            // SAFETY: h is part of this factor
            unsafe { solve_hyper(self.num_row as usize, &h, rhs) };
        }
        if self.update_method == UPDATE_APF {
            self.btran_apf(rhs);
            rhs.tight();
            rhs.pack();
        }
    }

    /// The sparse solve with U (or UR): `start`/`end`/`index`/`value` are
    /// the columns of U or the rows of UR, taken in `order`
    #[allow(clippy::too_many_arguments)]
    fn solve_u_sparse(
        &self,
        rhs: &mut HVec,
        start: &[i32],
        end: &[i32],
        index: &[i32],
        value: &[f64],
        order: impl Iterator<Item = usize>,
    ) {
        let num_row = self.num_row as usize;
        let array = &mut *rhs.array;
        let n = self.u_pivot_index.len();
        assert!(array.len() >= num_row);
        assert!(self.u_pivot_value.len() >= n && start.len() >= n && end.len() >= n);
        let mut rhs_synthetic_tick = 0.0;
        let mut rhs_count = 0;
        for i_logic in order {
            assert!(i_logic < n);
            // SAFETY: i_logic < n, within the lengths checked above;
            // pivot_row < num_row and the column's range by the HFactor
            // invariant
            unsafe {
                let pivot_row = *self.u_pivot_index.get_unchecked(i_logic);
                // Skip void
                if pivot_row == -1 {
                    continue;
                }
                let x = array.get_unchecked_mut(pivot_row as usize);
                let mut pivot_multiplier = *x;
                if pivot_multiplier.abs() > K_HIGHS_TINY {
                    pivot_multiplier /= *self.u_pivot_value.get_unchecked(i_logic);
                    *x = pivot_multiplier;
                    rhs.index[rhs_count] = pivot_row;
                    rhs_count += 1;
                    let (s, e) = (*start.get_unchecked(i_logic), *end.get_unchecked(i_logic));
                    if i_logic >= num_row {
                        rhs_synthetic_tick += (e - s) as f64;
                    }
                    let (idx, val) = entries(index, value, s, e);
                    axpy_unchecked(array, idx, val, pivot_multiplier);
                } else {
                    *x = 0.0;
                }
            }
        }
        rhs.count = rhs_count as i32;
        let u_pivot_count = self.u_pivot_index.len() as i32;
        rhs.synthetic_tick +=
            rhs_synthetic_tick.mul_add(15.0, ((u_pivot_count - self.num_row) * 10) as f64);
    }

    fn ftran_u(&self, rhs: &mut HVec, expected_density: f64) {
        // The update part
        if self.update_method == UPDATE_FT {
            self.ftran_ft(rhs);
            rhs.tight();
            rhs.pack();
        } else if self.update_method == UPDATE_MPF {
            self.ftran_mpf(rhs);
            rhs.tight();
            rhs.pack();
        }
        // The regular part
        if self.sparse_solve(rhs, expected_density, K_HYPER_FTRAN_U) {
            let n = self.u_pivot_index.len();
            self.solve_u_sparse(
                rhs,
                &self.u_start,
                &self.u_last_p,
                &self.u_index,
                &self.u_value,
                (0..n).rev(),
            );
        } else {
            let h = Triangle {
                lookup: &self.u_pivot_lookup,
                pivot_index: &self.u_pivot_index,
                pivot_value: Some(&self.u_pivot_value),
                start: &self.u_start,
                end: &self.u_last_p,
                index: &self.u_index,
                value: &self.u_value,
            };
            // SAFETY: h is part of this factor
            unsafe { solve_hyper(self.num_row as usize, &h, rhs) };
        }
        if self.update_method == UPDATE_PF {
            self.ftran_pf(rhs);
            rhs.tight();
            rhs.pack();
        }
    }

    fn btran_u(&self, rhs: &mut HVec, expected_density: f64) {
        if self.update_method == UPDATE_PF {
            self.btran_pf(rhs);
        }
        // The regular part
        if self.sparse_solve(rhs, expected_density, K_HYPER_BTRAN_U) {
            let n = self.u_pivot_index.len();
            self.solve_u_sparse(
                rhs,
                &self.ur_start,
                &self.ur_lastp,
                &self.ur_index,
                &self.ur_value,
                0..n,
            );
        } else {
            let h = Triangle {
                lookup: &self.u_pivot_lookup,
                pivot_index: &self.u_pivot_index,
                pivot_value: Some(&self.u_pivot_value),
                start: &self.ur_start,
                end: &self.ur_lastp,
                index: &self.ur_index,
                value: &self.ur_value,
            };
            // SAFETY: h is part of this factor
            unsafe { solve_hyper(self.num_row as usize, &h, rhs) };
        }
        // The update part
        if self.update_method == UPDATE_FT {
            rhs.tight();
            rhs.pack();
            self.btran_ft(rhs);
            rhs.tight();
        }
        if self.update_method == UPDATE_MPF {
            rhs.tight();
            rhs.pack();
            self.btran_mpf(rhs);
            rhs.tight();
        }
    }

    fn ftran_ft(&self, rhs: &mut HVec) {
        let mut rhs_count = rhs.count as usize;
        let array = &mut *rhs.array;
        let pf_pivot_count = self.pf_pivot_index.len();
        assert!(array.len() >= self.num_row as usize && self.pf_start.len() > pf_pivot_count);
        for (i, &i_row) in self.pf_pivot_index.iter().enumerate() {
            // SAFETY: i < pf_pivot_count, within the length checked above;
            // i_row and the entries' rows < num_row, and their range, by the
            // HFactor invariant
            let (value0, idx, val) = unsafe {
                let (start, end) = (
                    *self.pf_start.get_unchecked(i),
                    *self.pf_start.get_unchecked(i + 1),
                );
                let (idx, val) = entries(&self.pf_index, &self.pf_value, start, end);
                (*array.get_unchecked(i_row as usize), idx, val)
            };
            let mut value1 = value0;
            for (&j, &v) in idx.iter().zip(val) {
                // SAFETY: j < num_row (HFactor invariant)
                value1 = (-unsafe { *array.get_unchecked(j as usize) }).mul_add(v, value1);
            }
            // This would skip the situation where they are both zeros
            if value0 != 0.0 || value1 != 0.0 {
                if value0 == 0.0 {
                    rhs.index[rhs_count] = i_row;
                    rhs_count += 1;
                }
                array[i_row as usize] = if value1.abs() < K_HIGHS_TINY {
                    K_HIGHS_ZERO
                } else {
                    value1
                };
            }
        }
        rhs.count = rhs_count as i32;
        let pf_count = pf_pivot_count as i32;
        let pf_end = self.pf_start[pf_pivot_count];
        rhs.synthetic_tick += (pf_count * 20 + pf_end * 5) as f64;
        if pf_end / (pf_count + 1) < 5 {
            rhs.synthetic_tick += (pf_end * 5) as f64;
        }
    }

    fn btran_ft(&self, rhs: &mut HVec) {
        let mut rhs_count = rhs.count as usize;
        let pf_pivot_count = self.pf_pivot_index.len();
        assert!(rhs.array.len() >= self.num_row as usize && self.pf_start.len() > pf_pivot_count);
        // Apply row ETA backward
        let mut rhs_synthetic_tick = 0.0;
        for i in (0..pf_pivot_count).rev() {
            let pivot_row = self.pf_pivot_index[i] as usize;
            let pivot_multiplier = rhs.array[pivot_row];
            if pivot_multiplier != 0.0 {
                // SAFETY: i < pf_pivot_count, within the length checked
                // above; the range by the HFactor invariant
                let (start, end) = unsafe {
                    (
                        *self.pf_start.get_unchecked(i),
                        *self.pf_start.get_unchecked(i + 1),
                    )
                };
                rhs_synthetic_tick += (end - start) as f64;
                let (idx, val) = unsafe { entries(&self.pf_index, &self.pf_value, start, end) };
                for (&j, &v) in idx.iter().zip(val) {
                    // SAFETY: pf_index entries are < num_row <= array.len()
                    // (HFactor invariant, length asserted above)
                    let x = unsafe { rhs.array.get_unchecked_mut(j as usize) };
                    let value0 = *x;
                    let value1 = (-pivot_multiplier).mul_add(v, value0);
                    *x = if value1.abs() < K_HIGHS_TINY {
                        K_HIGHS_ZERO
                    } else {
                        value1
                    };
                    if value0 == 0.0 {
                        rhs.index[rhs_count] = j;
                        rhs_count += 1;
                    }
                }
            }
        }
        rhs.synthetic_tick += rhs_synthetic_tick.mul_add(15.0, (pf_pivot_count as i32 * 10) as f64);
        rhs.count = rhs_count as i32;
    }

    fn ftran_pf(&self, rhs: &mut HVec) {
        let mut rhs_count = rhs.count as usize;
        // Forwardly
        for i in 0..self.pf_pivot_index.len() {
            let pivot_row = self.pf_pivot_index[i] as usize;
            let mut pivot_multiplier = rhs.array[pivot_row];
            if pivot_multiplier.abs() > K_HIGHS_TINY {
                pivot_multiplier /= self.pf_pivot_value[i];
                rhs.array[pivot_row] = pivot_multiplier;
                for k in self.pf_start[i] as usize..self.pf_start[i + 1] as usize {
                    let index = self.pf_index[k] as usize;
                    let value0 = rhs.array[index];
                    let value1 = (-pivot_multiplier).mul_add(self.pf_value[k], value0);
                    if value0 == 0.0 {
                        rhs.index[rhs_count] = index as i32;
                        rhs_count += 1;
                    }
                    rhs.array[index] = if value1.abs() < K_HIGHS_TINY {
                        K_HIGHS_ZERO
                    } else {
                        value1
                    };
                }
            }
        }
        rhs.count = rhs_count as i32;
    }

    fn btran_pf(&self, rhs: &mut HVec) {
        let mut rhs_count = rhs.count as usize;
        // Backwardly
        for i in (0..self.pf_pivot_index.len()).rev() {
            let pivot_row = self.pf_pivot_index[i] as usize;
            let mut pivot_multiplier = rhs.array[pivot_row];
            for k in self.pf_start[i] as usize..self.pf_start[i + 1] as usize {
                pivot_multiplier = (-self.pf_value[k])
                    .mul_add(rhs.array[self.pf_index[k] as usize], pivot_multiplier);
            }
            pivot_multiplier /= self.pf_pivot_value[i];
            if rhs.array[pivot_row] == 0.0 {
                rhs.index[rhs_count] = pivot_row as i32;
                rhs_count += 1;
            }
            rhs.array[pivot_row] = if pivot_multiplier.abs() < K_HIGHS_TINY {
                1e-100
            } else {
                pivot_multiplier
            };
        }
        rhs.count = rhs_count as i32;
    }

    fn mpf_step(&self, rhs: &mut HVec, i: usize, forward_x: bool) {
        let s = &self.pf_start;
        let (a, b, c) = (s[i * 2], s[i * 2 + 1], s[i * 2 + 2]);
        let (xs, xe, ys, ye) = if forward_x {
            (b, c, a, b)
        } else {
            (a, b, b, c)
        };
        solve_matrix_t(
            xs,
            xe,
            ys,
            ye,
            &self.pf_index,
            &self.pf_value,
            self.pf_pivot_value[i],
            rhs,
        );
    }

    fn ftran_mpf(&self, rhs: &mut HVec) {
        for i in 0..self.pf_pivot_value.len() {
            self.mpf_step(rhs, i, true);
        }
    }

    fn btran_mpf(&self, rhs: &mut HVec) {
        for i in (0..self.pf_pivot_value.len()).rev() {
            self.mpf_step(rhs, i, false);
        }
    }

    fn ftran_apf(&self, rhs: &mut HVec) {
        for i in (0..self.pf_pivot_value.len()).rev() {
            self.mpf_step(rhs, i, true);
        }
    }

    fn btran_apf(&self, rhs: &mut HVec) {
        for i in 0..self.pf_pivot_value.len() {
            self.mpf_step(rhs, i, false);
        }
    }

    /// Update according to B' = B + (a_q - B e_p) e_p^T. More than one
    /// (aq, ep) pair means a multiple (CFT) update. `a` and
    /// `basic_index` are only used by APF.
    pub fn update(
        &mut self,
        aq: &[HVec],
        ep: &[HVec],
        i_row: &[i32],
        hint: &mut i32,
        a: Option<&AMatrix>,
        basic_index: &[i32],
    ) {
        if aq.len() > 1 {
            self.update_cft(aq, ep, i_row);
            return;
        }
        match self.update_method {
            UPDATE_FT => self.update_ft(&aq[0], &ep[0], i_row[0]),
            UPDATE_PF => self.update_pf(&aq[0], i_row[0], hint),
            UPDATE_MPF => self.update_mpf(&aq[0], &ep[0], i_row[0], hint),
            UPDATE_APF => self.update_apf(
                &aq[0],
                &ep[0],
                i_row[0],
                a.expect("APF needs A"),
                basic_index,
            ),
            _ => {}
        }
    }

    /// Delete the pivotal row (row `c_index`, logical pivot `p_logic`)
    /// from U
    fn delete_pivotal_row_from_u(&mut self, p_logic: usize, c_index: i32) {
        // Delete pivotal row from U
        for k in self.ur_start[p_logic] as usize..self.ur_lastp[p_logic] as usize {
            // Find the pivotal position
            let i_logic = self.u_pivot_lookup[self.ur_index[k] as usize] as usize;
            let mut i_find = self.u_start[i_logic] as usize;
            self.u_last_p[i_logic] -= 1;
            let i_last = self.u_last_p[i_logic] as usize;
            while i_find <= i_last {
                if self.u_index[i_find] == c_index {
                    break;
                }
                i_find += 1;
            }
            // Put last to find, and delete last
            self.u_index[i_find] = self.u_index[i_last];
            self.u_value[i_find] = self.u_value[i_last];
        }
    }

    fn delete_pivotal_col_from_ur(&mut self, p_logic: usize, c_index: i32) {
        // Delete pivotal column from UR
        for k in self.u_start[p_logic] as usize..self.u_last_p[p_logic] as usize {
            // Find the pivotal position
            let i_logic = self.u_pivot_lookup[self.u_index[k] as usize] as usize;
            let mut i_find = self.ur_start[i_logic] as usize;
            self.ur_lastp[i_logic] -= 1;
            let i_last = self.ur_lastp[i_logic] as usize;
            while i_find <= i_last {
                if self.ur_index[i_find] == c_index {
                    break;
                }
                i_find += 1;
            }
            // Put last to find, and delete last
            self.ur_space[i_logic] += 1;
            self.ur_index[i_find] = self.ur_index[i_last];
            self.ur_value[i_find] = self.ur_value[i_last];
        }
    }

    /// Store U column entries [u_start_x, u_end_x) as UR elements in row
    /// `c_index`
    fn store_column_as_ur(&mut self, u_start_x: usize, u_end_x: usize, c_index: i32) {
        for k in u_start_x..u_end_x {
            // Which ETA file
            let i_logic = self.u_pivot_lookup[self.u_index[k] as usize] as usize;
            // Move row to the end if necessary
            if self.ur_space[i_logic] == 0 {
                let row_start = self.ur_start[i_logic] as usize;
                let row_count = self.ur_lastp[i_logic] as usize - row_start;
                let new_start = self.ur_index.len();
                let new_space = (row_count as i32 as f64).mul_add(1.1, 5.0) as i32 as usize;
                self.ur_index.resize(new_start + new_space, 0);
                self.ur_value.resize(new_start + new_space, 0.0);
                self.ur_index
                    .copy_within(row_start..row_start + row_count, new_start);
                self.ur_value
                    .copy_within(row_start..row_start + row_count, new_start);
                self.ur_start[i_logic] = new_start as i32;
                self.ur_lastp[i_logic] = (new_start + row_count) as i32;
                self.ur_space[i_logic] = (new_space - row_count) as i32;
            }
            // Put into the next available space
            self.ur_space[i_logic] -= 1;
            let i_put = self.ur_lastp[i_logic] as usize;
            self.ur_lastp[i_logic] += 1;
            self.ur_index[i_put] = c_index;
            self.ur_value[i_put] = self.u_value[k];
        }
    }

    /// Save the UR pointers for the new U column replacing `p_logic`
    fn push_ur_pointers(&mut self, p_logic: usize) {
        let s = self.ur_start[p_logic];
        self.ur_start.push(s);
        self.ur_lastp.push(s);
        self.ur_space
            .push(self.ur_space[p_logic] + self.ur_lastp[p_logic] - s);
    }

    fn update_cft(&mut self, aq_work: &[HVec], ep_work: &[HVec], i_row: &[i32]) {
        // In the major update loop, the prefix
        //
        // c(p) = current working pivot
        // p(p) = previous pivot  (0 =< pp < cp)
        let num_update = aq_work.len();

        // Pivot related buffers
        let pf_np0 = self.pf_pivot_index.len();
        let mut p_logic = vec![0usize; num_update];
        let mut p_value = vec![0.0; num_update];
        let mut p_alpha = vec![0.0; num_update];
        for cp in 0..num_update {
            let c_row = i_row[cp] as usize;
            let i_logic = self.u_pivot_lookup[c_row] as usize;
            p_logic[cp] = i_logic;
            p_value[cp] = self.u_pivot_value[i_logic];
            p_alpha[cp] = aq_work[cp].array[c_row];
        }

        // Temporary U pointers
        let mut t_start = vec![0usize; num_update + 1];
        let mut t_pivot = vec![0.0; num_update];
        t_start[0] = self.u_index.len();

        // Logically sorted previous row_ep
        let mut sorted_pp: Vec<(usize, usize)> = Vec::new();

        // Major update loop
        for cp in 0..num_update {
            // 1. Expand partial FTRAN result to buffer
            self.iwork.clear();
            let aq = &aq_work[cp];
            for i in 0..aq.pack_count as usize {
                let index = aq.pack_index[i];
                self.iwork.push(index);
                self.dwork[index as usize] = aq.pack_value[i];
            }

            // 2. Update partial FTRAN result by recent FT matrix
            for pp in 0..cp {
                let p_row = i_row[pp];
                let mut value = self.dwork[p_row as usize];
                let pf_pp = pp + pf_np0;
                for i in self.pf_start[pf_pp] as usize..self.pf_start[pf_pp + 1] as usize {
                    value =
                        (-self.dwork[self.pf_index[i] as usize]).mul_add(self.pf_value[i], value);
                }
                self.iwork.push(p_row); // OK to duplicate
                self.dwork[p_row as usize] = value;
            }

            // 3. Store the partial FTRAN result to matrix U
            let ppaq = self.dwork[i_row[cp] as usize]; // pivot of the partial aq
            self.dwork[i_row[cp] as usize] = 0.0;
            let u_start_x = t_start[cp];
            for &index in &self.iwork {
                let value = self.dwork[index as usize];
                self.dwork[index as usize] = 0.0; // This effectively removes all duplication
                if value.abs() > K_HIGHS_TINY {
                    self.u_index.push(index);
                    self.u_value.push(value);
                }
            }
            let u_count_x = self.u_index.len();
            t_start[cp + 1] = u_count_x;
            t_pivot[cp] = p_value[cp] * p_alpha[cp];

            // 4. Expand partial BTRAN result to buffer
            self.iwork.clear();
            let ep = &ep_work[cp];
            for i in 0..ep.pack_count as usize {
                let index = ep.pack_index[i];
                self.iwork.push(index);
                self.dwork[index as usize] = ep.pack_value[i];
            }

            // 5. Delete logical later rows (in logical order)
            for &(_, pp) in &sorted_pp[..cp] {
                let p_row = i_row[pp] as usize;
                let multiplier = -p_value[pp] * self.dwork[p_row];
                if self.dwork[p_row].abs() > K_HIGHS_TINY {
                    let epp = &ep_work[pp];
                    for i in 0..epp.pack_count as usize {
                        let index = epp.pack_index[i];
                        self.iwork.push(index);
                        let d = &mut self.dwork[index as usize];
                        *d = epp.pack_value[i].mul_add(multiplier, *d);
                    }
                }
                self.dwork[p_row] = 0.0; // Force to be 0
            }

            // 6. Update partial BTRAN result by recent U columns
            for pp in 0..cp {
                let kpivot = i_row[pp];
                let mut value = self.dwork[kpivot as usize];
                for k in t_start[pp]..t_start[pp + 1] {
                    value = (-self.dwork[self.u_index[k] as usize]).mul_add(self.u_value[k], value);
                }
                value /= t_pivot[pp];
                self.iwork.push(kpivot);
                self.dwork[kpivot as usize] = value; // Again OK to duplicate
            }

            // 6.x compute current alpha
            let mut thex = 0.0;
            for k in u_start_x..u_count_x {
                thex = self.dwork[self.u_index[k] as usize].mul_add(self.u_value[k], thex);
            }
            t_pivot[cp] = thex.mul_add(p_value[cp], ppaq);

            // 7. Store BTRAN result to FT elimination, update logic helper
            self.dwork[i_row[cp] as usize] = 0.0;
            let pivot_multiplier = -p_value[cp];
            for &index in &self.iwork {
                let value = self.dwork[index as usize];
                self.dwork[index as usize] = 0.0;
                if value.abs() > K_HIGHS_TINY {
                    self.pf_index.push(index);
                    self.pf_value.push(value * pivot_multiplier);
                }
            }
            self.pf_pivot_index.push(i_row[cp]);
            self.u_total_x += self.pf_index.len() as i32 - *self.pf_start.last().unwrap();
            self.pf_start.push(self.pf_index.len() as i32);

            // 8. Update the sorted ep
            sorted_pp.push((p_logic[cp], cp));
            sorted_pp.sort_unstable();
        }

        // Now modify the U matrix
        for cp in 0..num_update {
            // 1. Delete pivotal row from U
            let c_index = i_row[cp];
            let c_logic = p_logic[cp];
            self.u_total_x -= self.ur_lastp[c_logic] - self.ur_start[c_logic];
            self.delete_pivotal_row_from_u(c_logic, c_index);

            // 2. Delete pivotal column from UR
            self.u_total_x -= self.u_last_p[c_logic] - self.u_start[c_logic];
            self.delete_pivotal_col_from_ur(c_logic, c_index);

            // 3. Insert the (stored) partial FTRAN to the row matrix
            let u_start_x = t_start[cp];
            let u_end_x = t_start[cp + 1];
            self.u_total_x += (u_end_x - u_start_x) as i32;
            self.store_column_as_ur(u_start_x, u_end_x, c_index);

            // 4. Save pointers
            self.u_start.push(u_start_x as i32);
            self.u_last_p.push(u_end_x as i32);
            self.push_ur_pointers(c_logic);
            self.u_pivot_lookup[c_index as usize] = self.u_pivot_index.len() as i32;
            self.u_pivot_index[c_logic] = -1;
            self.u_pivot_index.push(c_index);
            self.u_pivot_value.push(t_pivot[cp]);
        }
    }

    fn update_ft(&mut self, aq: &HVec, ep: &HVec, i_row: i32) {
        // Store pivot
        let p_logic = self.u_pivot_lookup[i_row as usize] as usize;
        let pivot = self.u_pivot_value[p_logic];
        let alpha = aq.array[i_row as usize];
        self.u_pivot_index[p_logic] = -1;

        // Delete pivotal row from U, and pivotal column from UR
        self.delete_pivotal_row_from_u(p_logic, i_row);
        self.delete_pivotal_col_from_ur(p_logic, i_row);

        // Store column to U
        self.u_start.push(self.u_index.len() as i32);
        for i in 0..aq.pack_count as usize {
            if aq.pack_index[i] != i_row {
                self.u_index.push(aq.pack_index[i]);
                self.u_value.push(aq.pack_value[i]);
            }
        }
        self.u_last_p.push(self.u_index.len() as i32);
        let u_start_x = *self.u_start.last().unwrap();
        let u_end_x = *self.u_last_p.last().unwrap();
        self.u_total_x += u_end_x - u_start_x + 1;

        // Store column as UR elements
        self.store_column_as_ur(u_start_x as usize, u_end_x as usize, i_row);

        // Store UR pointers
        self.push_ur_pointers(p_logic);

        // Update pivot count
        self.u_pivot_lookup[i_row as usize] = self.u_pivot_index.len() as i32;
        self.u_pivot_index.push(i_row);
        self.u_pivot_value.push(pivot * alpha);

        // Store row_ep as R matrix
        for i in 0..ep.pack_count as usize {
            if ep.pack_index[i] != i_row {
                assert!((ep.pack_index[i] as u32) < self.num_row as u32);
                self.pf_index.push(ep.pack_index[i]);
                self.pf_value.push(-ep.pack_value[i] * pivot);
            }
        }
        self.u_total_x += self.pf_index.len() as i32 - *self.pf_start.last().unwrap();

        // Store R matrix pivot
        self.pf_pivot_index.push(i_row);
        self.pf_start.push(self.pf_index.len() as i32);

        // Update total countX
        self.u_total_x -= self.u_last_p[p_logic] - self.u_start[p_logic];
        self.u_total_x -= self.ur_lastp[p_logic] - self.ur_start[p_logic];
    }

    fn update_pf(&mut self, aq: &HVec, i_row: i32, hint: &mut i32) {
        // Copy the pivotal column
        for i in 0..aq.pack_count as usize {
            let index = aq.pack_index[i];
            if index != i_row {
                self.pf_index.push(index);
                self.pf_value.push(aq.pack_value[i]);
            }
        }
        // Save pivot
        self.pf_pivot_index.push(i_row);
        self.pf_pivot_value.push(aq.array[i_row as usize]);
        self.pf_start.push(self.pf_index.len() as i32);
        // Check refactor
        self.u_total_x += aq.pack_count;
        if self.u_total_x > self.u_merit_x {
            *hint = 1;
        }
    }

    fn update_mpf(&mut self, aq: &HVec, ep: &HVec, i_row: i32, hint: &mut i32) {
        // Store elements
        let n = aq.pack_count as usize;
        self.pf_index.extend_from_slice(&aq.pack_index[..n]);
        self.pf_value.extend_from_slice(&aq.pack_value[..n]);
        let p_logic = self.u_pivot_lookup[i_row as usize] as usize;
        let u_start_x = self.u_start[p_logic] as usize;
        let u_end_x = self.u_start[p_logic + 1] as usize;
        for k in u_start_x..u_end_x {
            self.pf_index.push(self.u_index[k]);
            self.pf_value.push(-self.u_value[k]);
        }
        self.pf_index.push(i_row);
        self.pf_value.push(-self.u_pivot_value[p_logic]);
        self.pf_start.push(self.pf_index.len() as i32);

        let n = ep.pack_count as usize;
        self.pf_index.extend_from_slice(&ep.pack_index[..n]);
        self.pf_value.extend_from_slice(&ep.pack_value[..n]);
        self.pf_start.push(self.pf_index.len() as i32);

        // Store pivot
        self.pf_pivot_value.push(aq.array[i_row as usize]);

        // Refactor or not
        self.u_total_x += aq.pack_count + ep.pack_count;
        if self.u_total_x > self.u_merit_x {
            *hint = 1;
        }
    }

    fn update_apf(&mut self, aq: &HVec, ep: &HVec, i_row: i32, a: &AMatrix, basic_index: &[i32]) {
        // Store elements
        let n = aq.pack_count as usize;
        self.pf_index.extend_from_slice(&aq.pack_index[..n]);
        self.pf_value.extend_from_slice(&aq.pack_value[..n]);

        let variable_out = basic_index[i_row as usize];
        if variable_out >= a.num_col {
            self.pf_index.push(variable_out - a.num_col);
            self.pf_value.push(-1.0);
        } else {
            let v = variable_out as usize;
            for k in a.start[v] as usize..a.start[v + 1] as usize {
                self.pf_index.push(a.index[k]);
                self.pf_value.push(-a.value[k]);
            }
        }
        self.pf_start.push(self.pf_index.len() as i32);

        let n = ep.pack_count as usize;
        self.pf_index.extend_from_slice(&ep.pack_index[..n]);
        self.pf_value.extend_from_slice(&ep.pack_value[..n]);
        self.pf_start.push(self.pf_index.len() as i32);

        // Store pivot
        self.pf_pivot_value.push(aq.array[i_row as usize]);
    }

    /// Updates the factor with respect to new rows in the constraint matrix
    /// (assuming slacks are basic). `ar_*` are the new rows, row-wise.
    pub fn add_rows(
        &mut self,
        num_col: i32,
        basic_index: &[i32],
        ar_start: &[i32],
        ar_index: &[i32],
        ar_value: &[f64],
    ) {
        let num_row = self.num_row;
        let nr = num_row as usize;
        let num_new_row = ar_start.len() as i32 - 1;
        let new_num_row = num_row + num_new_row;
        let nnr = new_num_row as usize;

        // Need to know where (if) a column is basic
        let mut in_basis = vec![-1; num_col as usize];
        for i_row in 0..nr {
            let i_var = basic_index[i_row];
            if i_var < num_col {
                in_basis[i_var as usize] = i_row as i32;
            }
        }

        // Create a row-wise sparse matrix containing the new rows of the L
        // matrix - so that a column-wise version can be created (after
        // inserting the rows into the LR matrix) allowing the new column
        // entries to be inserted efficiently into the L matrix
        let mut new_lr_start = vec![0];
        let mut new_lr_index = Vec::new();
        let mut new_lr_value = Vec::new();
        let mut expected_density = 0.0;
        let mut rhs = OwnedHVec::new(num_row);
        reserve(&mut self.lr_start, new_num_row + 1);
        for inew_row in 0..num_new_row as usize {
            // Prepare RHS for system U^T.v = r
            rhs.clear();
            rhs.pack_flag = true;
            for i_el in ar_start[inew_row] as usize..ar_start[inew_row + 1] as usize {
                let basis_index = in_basis[ar_index[i_el] as usize];
                if basis_index >= 0 {
                    rhs.array[basis_index as usize] = ar_value[i_el];
                    rhs.index[rhs.count as usize] = basis_index;
                    rhs.count += 1;
                }
            }
            // Solve U^T.v = r
            rhs.with(|r| self.btran_u(r, expected_density));
            let local_density = rhs.count as f64 / num_row as f64;
            expected_density = K_RUNNING_AVERAGE_MULTIPLIER.mul_add(
                local_density,
                (1.0 - K_RUNNING_AVERAGE_MULTIPLIER) * expected_density,
            );
            rhs.with(|r| r.tight());
            // Append v to the matrix containing the new rows of L, and to
            // the L matrix
            for i_x in 0..rhs.count as usize {
                let i_col = rhs.index[i_x];
                let v = rhs.array[i_col as usize];
                new_lr_index.push(i_col);
                new_lr_value.push(v);
                self.lr_index.push(i_col);
                self.lr_value.push(v);
            }
            new_lr_start.push(new_lr_index.len() as i32);
            self.lr_start.push(self.lr_index.len() as i32);
        }
        // Now create a column-wise copy of the new rows (as
        // HighsSparseMatrix::ensureColwise)
        let num_nz = new_lr_index.len();
        let mut col_start = vec![0i32; nr + 1];
        for &c in &new_lr_index {
            col_start[c as usize + 1] += 1;
        }
        for i in 0..nr {
            col_start[i + 1] += col_start[i];
        }
        let mut col_index = vec![0i32; num_nz];
        let mut col_value = vec![0.0; num_nz];
        let mut put = col_start.clone();
        for i_row in 0..num_new_row as usize {
            for i_el in new_lr_start[i_row] as usize..new_lr_start[i_row + 1] as usize {
                let c = new_lr_index[i_el] as usize;
                col_index[put[c] as usize] = i_row as i32;
                col_value[put[c] as usize] = new_lr_value[i_el];
                put[c] += 1;
            }
        }
        //
        // Insert the column-wise copy into the L matrix
        //
        // Add pivot indices for the new columns
        self.l_pivot_index.resize(nnr, 0);
        for i_col in nr..nnr {
            self.l_pivot_index[i_col] = i_col as i32;
        }
        //
        // Add starts for the identity columns
        let l_matrix_new_num_nz = self.lr_index.len();
        self.l_start.resize(nnr + 1, 0);
        let mut to_el = l_matrix_new_num_nz;
        for i_col in nr + 1..nnr + 1 {
            self.l_start[i_col] = l_matrix_new_num_nz as i32;
        }
        //
        // Insert the new entries, remembering to offset the index values by
        // num_row, since the column-wise copy only has the new rows
        self.l_index.resize(l_matrix_new_num_nz, 0);
        self.l_value.resize(l_matrix_new_num_nz, 0.0);
        for i_col in (0..nr).rev() {
            let from_el = self.l_start[i_col + 1] as usize;
            self.l_start[i_col + 1] = to_el as i32;
            for i_el in (col_start[i_col] as usize..col_start[i_col + 1] as usize).rev() {
                to_el -= 1;
                self.l_index[to_el] = num_row + col_index[i_el];
                self.l_value[to_el] = col_value[i_el];
            }
            for i_el in (self.l_start[i_col] as usize..from_el).rev() {
                to_el -= 1;
                self.l_index[to_el] = self.l_index[i_el];
                self.l_value[to_el] = self.l_value[i_el];
            }
        }
        self.l_pivot_lookup.resize(nnr, 0);
        for i_row in nr..nnr {
            self.l_pivot_lookup[self.l_pivot_index[i_row] as usize] = i_row as i32;
        }
        //
        // Now add pivots corresponding to identity columns in U. U(R)lastp
        // needs to be equal to the start since there are no non-pivotal
        // entries
        let u_count_x = self.u_index.len();
        let u_pivot_lookup_offset = self.u_pivot_index.len() as i32 - num_row;
        for i_row in num_row..new_num_row {
            self.u_pivot_lookup.push(u_pivot_lookup_offset + i_row);
            self.u_pivot_index.push(i_row);
            self.u_pivot_value.push(1.0);
            self.u_start.push(u_count_x as i32);
            self.u_last_p.push(u_count_x as i32);
        }

        // Now, to extend UR, borrowing names from buildFinish()
        let ur_stuff_size: i32 = if self.update_method == UPDATE_FT {
            5
        } else {
            0
        };
        let ur_size = self.ur_index.len() as i32;
        let ur_count_size = ur_size + ur_stuff_size * num_new_row;
        self.ur_index.resize(ur_count_size as usize, 0);
        self.ur_value.resize(ur_count_size as usize, 0.0);

        // Need to refer to just the new UR vectors
        let ur_cur_num_vec = self.ur_start.len();
        let ur_new_num_vec = ur_cur_num_vec + num_new_row as usize;
        // Allow space to the start of new rows, including the start for
        // the fictitious ur_new_num_vec'th row
        self.ur_start.resize(ur_new_num_vec + 1, 0);
        for i_row in ur_cur_num_vec + 1..ur_new_num_vec + 1 {
            self.ur_start[i_row] = ur_size;
        }
        // NB ur_temp plays the role of ur_lastp when it could be used as
        // temporary storage in buildFinish()
        let mut ur_temp = vec![0; ur_new_num_vec];
        self.ur_space.resize(ur_new_num_vec, 0);
        for i_row in ur_cur_num_vec..ur_new_num_vec {
            self.ur_space[i_row] = ur_stuff_size;
        }
        for k in 0..u_count_x {
            ur_temp[self.u_pivot_lookup[self.u_index[k] as usize] as usize] += 1;
        }
        let mut i_start = ur_size;
        self.ur_start[ur_cur_num_vec] = i_start;
        for i_row in ur_cur_num_vec + 1..ur_new_num_vec + 1 {
            let gap = ur_temp[i_row - 1] + ur_stuff_size;
            self.ur_start[i_row] = i_start + gap;
            i_start += gap;
        }
        // Lose the start for the fictitious ur_new_num_vec'th row
        self.ur_start.truncate(ur_new_num_vec);
        // Resize ur_lastp and initialise its new entries to be the ur_start
        // values since the rows are empty
        self.ur_lastp.resize(ur_new_num_vec, 0);
        for i_row in ur_cur_num_vec..ur_new_num_vec {
            self.ur_lastp[i_row] = self.ur_start[i_row];
        }
        // Increase the number of rows
        self.num_row += num_new_row;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // B = [[2, 1, 0], [1, 3, 1], [0, 1, 4]], column-wise; variables 3..6
    // are the slacks
    const START: [i32; 4] = [0, 2, 5, 7];
    const INDEX: [i32; 7] = [0, 1, 0, 1, 2, 1, 2];
    const VALUE: [f64; 7] = [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 4.0];
    const A: AMatrix = AMatrix {
        num_col: 3,
        start: &START,
        index: &INDEX,
        value: &VALUE,
    };

    fn column(var: i32) -> Vec<f64> {
        let mut c = vec![0.0; 3];
        if var >= 3 {
            c[(var - 3) as usize] = 1.0;
        } else {
            for k in START[var as usize] as usize..START[var as usize + 1] as usize {
                c[INDEX[k] as usize] = VALUE[k];
            }
        }
        c
    }

    fn solve(f: &HFactor, b: &[f64], transpose: bool) -> OwnedHVec {
        let mut v = OwnedHVec::new(3);
        for (i, &x) in b.iter().enumerate() {
            if x != 0.0 {
                v.array[i] = x;
                v.index[v.count as usize] = i as i32;
                v.count += 1;
            }
        }
        v.pack_flag = true;
        v.with(|h| {
            if transpose {
                f.btran(h, 1.0)
            } else {
                f.ftran(h, 1.0)
            }
        });
        v
    }

    /// B x = b and B^T y = b, with x[i] the value of basic_index[i]
    fn check(f: &HFactor, basic_index: &[i32]) {
        let b = [1.0, 2.0, 3.0];
        let x = solve(f, &b, false);
        for r in 0..3 {
            let bx: f64 = (0..3).map(|i| column(basic_index[i])[r] * x.array[i]).sum();
            assert!((bx - b[r]).abs() < 1e-12, "row {r}: {bx}");
        }
        let y = solve(f, &b, true);
        for i in 0..3 {
            let c = column(basic_index[i]);
            let by: f64 = (0..3).map(|r| c[r] * y.array[r]).sum();
            assert!((by - b[i]).abs() < 1e-12, "col {i}: {by}");
        }
    }

    fn build(f: &mut HFactor, basic_index: &mut [i32], info: Option<&RefactorIn>) -> (i32, bool) {
        let mut refactored = false;
        let r = f.build(
            0.1,
            1e-10,
            f64::INFINITY,
            &A,
            basic_index,
            info,
            &mut refactored,
        );
        (r, refactored)
    }

    fn new_factor() -> HFactor {
        let mut f = HFactor::default();
        f.setup(3, 3, 3, &START, UPDATE_FT);
        f
    }

    #[test]
    fn update_methods() {
        for method in [UPDATE_FT, UPDATE_PF, UPDATE_MPF, UPDATE_APF] {
            let mut f = HFactor::default();
            f.setup(3, 3, 3, &START, method);
            let mut basic_index = [0, 1, 2];
            assert_eq!(build(&mut f, &mut basic_index, None), (0, false));
            check(&f, &basic_index);
            // Slack 3 enters in the position of row p
            let mut aq = solve(&f, &column(3), false);
            let p = (0..3).find(|&i| aq.array[i].abs() > 0.1).unwrap();
            let mut e_p = [0.0; 3];
            e_p[p] = 1.0;
            let mut ep = solve(&f, &e_p, true);
            let mut hint = 0;
            f.update(
                &[aq.view()],
                &[ep.view()],
                &[p as i32],
                &mut hint,
                Some(&A),
                &basic_index,
            );
            basic_index[p] = 3;
            check(&f, &basic_index);
        }
    }

    #[test]
    fn build_solve_update_refactor() {
        let mut f = new_factor();
        let mut basic_index = [0, 1, 2];
        assert_eq!(build(&mut f, &mut basic_index, None), (0, false));
        check(&f, &basic_index);

        // FT update: slack 3 enters in the position of row p
        let mut aq = solve(&f, &column(3), false);
        let p = (0..3).find(|&i| aq.array[i].abs() > 0.1).unwrap();
        let mut e_p = [0.0; 3];
        e_p[p] = 1.0;
        let mut ep = solve(&f, &e_p, true);
        let mut hint = 0;
        f.update(
            &[aq.view()],
            &[ep.view()],
            &[p as i32],
            &mut hint,
            None,
            &[],
        );
        basic_index[p] = 3;
        check(&f, &basic_index);

        // Refactorization from the recorded pivots
        let mut g = new_factor();
        let mut bi = [0, 1, 2];
        build(&mut g, &mut bi, None);
        let (row, var, ty) = (
            g.refactor_pivot_row.clone(),
            g.refactor_pivot_var.clone(),
            g.refactor_pivot_type.clone(),
        );
        let info = RefactorIn {
            pivot_row: &row,
            pivot_var: &var,
            pivot_type: &ty,
            build_synthetic_tick: 0.0,
        };
        let mut bi2 = [2, 0, 1];
        assert_eq!(build(&mut g, &mut bi2, Some(&info)), (0, true));
        assert_eq!(bi2, bi);
        check(&g, &bi2);
    }

    #[test]
    fn rank_deficient() {
        // Column 0 twice: one is replaced by a slack
        let mut f = new_factor();
        let mut basic_index = [0, 0, 2];
        assert_eq!(build(&mut f, &mut basic_index, None), (1, false));
        assert_eq!(f.var_with_no_pivot, [0]);
        assert!(basic_index.contains(&(3 + f.row_with_no_pivot[0])));
        check(&f, &basic_index);
    }
}
