//! LU factorization with updates of the basis matrix: lu_update.h/.cc
//! (the interface), basiclu_wrapper.cc (BasicLu: BASICLU's own
//! Forrest-Tomlin updates, the default), and forrest_tomlin.cc with
//! lu_factorization.cc and basiclu_kernel.cc (ForrestTomlin: a generic
//! Forrest-Tomlin update on top of a BASICLU factorization, lu_kernel > 0).
//! BASICLU is called directly through its Rust port.

use crate::util::fma::ClangFma;

use super::control::Control;
use super::fmt::sci2;
use super::indexed_vector::IndexedVector;
use super::sparse_matrix::{
    dot_column, onenorm, infnorm, multiply_add, normest_inverse, remove_diagonal, scatter_column,
    triangular_solve, SparseMatrix,
};
use super::{cmax, utils, Int, FT_DIAG_ERROR_TOL, LU_DEPENDENCY_TOL, LU_STABILITY_THRESHOLD};
use crate::basiclu::{self as blu, Lu};

/// Errors that the C++ code throws as std::logic_error (they do not occur
/// for the matrices IPX factorizes).
pub(crate) type LuResult<T> = Result<T, String>;

/// The BASICLU stores and factor files, grown on request of BASICLU.
struct Store {
    m: Int,
    istore: Vec<Int>,
    xstore: Vec<f64>,
    li: Vec<Int>,
    lx: Vec<f64>,
    ui: Vec<Int>,
    ux: Vec<f64>,
    wi: Vec<Int>,
    wx: Vec<f64>,
}

impl Store {
    /// basiclu_initialize with factor files of fmem elements
    fn new(m: Int, fmem: usize) -> LuResult<Self> {
        if m <= 0 {
            return Err("basiclu_initialize failed".into());
        }
        // the C++ sizes both stores BASICLU_SIZE_ISTORE_1 +
        // BASICLU_SIZE_ISTORE_M * dim; only the first xstore_len(m) of
        // xstore are used
        let mut istore = vec![0; blu::istore_len(m)];
        let mut xstore = vec![0.0; blu::xstore_len(m)];
        blu::initialize(m, &mut istore, &mut xstore);
        xstore[blu::MEMORYL] = fmem as f64;
        xstore[blu::MEMORYU] = fmem as f64;
        xstore[blu::MEMORYW] = fmem as f64;
        Ok(Store {
            m,
            istore,
            xstore,
            li: vec![0; fmem],
            lx: vec![0.0; fmem],
            ui: vec![0; fmem],
            ux: vec![0.0; fmem],
            wi: vec![0; fmem],
            wx: vec![0.0; fmem],
        })
    }

    /// lu_load on the stores
    fn lu(&mut self) -> Lu<'_> {
        match Lu::load(
            &mut self.istore,
            &mut self.xstore,
            &mut self.li,
            &mut self.lx,
            &mut self.ui,
            &mut self.ux,
            &mut self.wi,
            &mut self.wx,
        ) {
            Ok(lu) => lu,
            Err(_) => unreachable!("BASICLU store initialized by Store::new"),
        }
    }

    /// Reallocates (Li,Lx), (Ui,Ux) and/or (Wi,Wx) as requested by BASICLU,
    /// to 1.5 times the required amount.
    fn reallocate(&mut self) {
        const REALLOC_FACTOR: f64 = 1.5;
        let x = &mut self.xstore;
        for (mem, add, vi, vx) in [
            (blu::MEMORYL, blu::ADD_MEMORYL, &mut self.li, &mut self.lx),
            (blu::MEMORYU, blu::ADD_MEMORYU, &mut self.ui, &mut self.ux),
            (blu::MEMORYW, blu::ADD_MEMORYW, &mut self.wi, &mut self.wx),
        ] {
            if x[add] > 0.0 {
                let mut new_size = (x[mem] + x[add]) as Int;
                new_size = (new_size as f64 * REALLOC_FACTOR) as Int;
                vi.resize(new_size as usize, 0);
                vx.resize(new_size as usize, 0.0);
                x[mem] = new_size as f64;
            }
        }
    }

    /// basiclu_factorize, repeated on reallocation
    fn factorize(&mut self, bbegin: &[Int], bend: &[Int], bi: &[Int], bx: &[f64]) -> Int {
        let mut ncall = 0;
        loop {
            let mut lu = self.lu();
            let status = lu.factorize(bbegin, bend, bi, bx, ncall != 0);
            let status = lu.save(status);
            if status != blu::REALLOCATE {
                return status;
            }
            self.reallocate();
            ncall += 1;
        }
    }

    /// basiclu_get_factors into L (with unit diagonal), U, rowperm, colperm
    fn get_factors(
        &mut self,
        l: Option<&mut SparseMatrix>,
        u: Option<&mut SparseMatrix>,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
    ) -> LuResult<()> {
        let m = self.m;
        let lnz = self.xstore[blu::LNZ] as Int;
        let unz = self.xstore[blu::UNZ] as Int;
        let mut lu = self.lu();
        if lu.nupdate != 0 {
            lu.save(blu::ERROR_INVALID_CALL);
            return Err("basiclu_get_factors failed".into());
        }
        let l = l.map(|l| {
            l.resize(m, m, m + lnz);
            (&mut l.colptr[..], &mut l.rowidx[..], &mut l.values[..])
        });
        let u = u.map(|u| {
            u.resize(m, m, m + unz);
            (&mut u.colptr[..], &mut u.rowidx[..], &mut u.values[..])
        });
        lu.get_factors(rowperm, colperm, l, u);
        // the C code does not save on success
        Ok(())
    }

    /// basiclu_solve_dense; rhs None solves in place
    fn solve_dense(&mut self, rhs: Option<&[f64]>, lhs: &mut [f64], trans: u8) -> LuResult<()> {
        let mut lu = self.lu();
        let status = if lu.nupdate < 0 {
            blu::ERROR_INVALID_CALL
        } else {
            lu.solve_dense(rhs, lhs, trans);
            blu::OK
        };
        if lu.save(status) != blu::OK {
            return Err("basiclu_solve_dense failed".into());
        }
        Ok(())
    }

    /// basiclu_solve_for_update, repeated on reallocation
    fn solve_for_update(
        &mut self,
        irhs: &[Int],
        xrhs: &[f64],
        mut out: Option<(&mut Int, &mut [Int], &mut [f64])>,
        trans: u8,
    ) -> Int {
        loop {
            let m = self.m;
            let mut lu = self.lu();
            let is_t = trans == b't' || trans == b'T';
            let mut status = blu::OK;
            if lu.nupdate < 0 {
                status = blu::ERROR_INVALID_CALL;
            } else if lu.nforrest == m {
                status = blu::ERROR_MAXIMUM_UPDATES;
            } else {
                let n = if is_t { 1 } else { irhs.len() };
                if !irhs[..n].iter().all(|&i| i >= 0 && i < m) {
                    status = blu::ERROR_INVALID_ARGUMENT;
                }
            }
            if status == blu::OK {
                let o = out.as_mut().map(|(n, i, x)| (&mut **n, &mut **i, &mut **x));
                status = lu.solve_for_update(irhs, xrhs, o, trans);
            }
            let status = lu.save(status);
            if status != blu::REALLOCATE {
                return status;
            }
            self.reallocate();
        }
    }

    /// basiclu_update, repeated on reallocation
    fn update(&mut self, xtbl: f64) -> Int {
        loop {
            let mut lu = self.lu();
            let status = if lu.nupdate < 0 || lu.ftran_for_update < 0 || lu.btran_for_update < 0 {
                blu::ERROR_INVALID_CALL
            } else {
                lu.update(xtbl)
            };
            let status = lu.save(status);
            if status != blu::REALLOCATE {
                return status;
            }
            self.reallocate();
        }
    }
}

/// class BasicLu
struct BasicLu {
    s: Store,
    fill_factor: f64,
}

impl BasicLu {
    fn new(dim: Int) -> LuResult<Self> {
        // Initial size of the BASICLU work arrays is 1 element.
        Ok(BasicLu {
            s: Store::new(dim, 1)?,
            fill_factor: 0.0,
        })
    }

    fn factorize(
        &mut self,
        control: &Control,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
        strict_abs_pivottol: bool,
    ) -> LuResult<Int> {
        let x = &mut self.s.xstore;
        if strict_abs_pivottol {
            x[blu::REMOVE_COLUMNS] = 1.0;
            x[blu::ABS_PIVOT_TOLERANCE] = LU_DEPENDENCY_TOL;
        } else {
            x[blu::REMOVE_COLUMNS] = 0.0;
            x[blu::ABS_PIVOT_TOLERANCE] = 1e-14; // BASICLU default
        }
        let status = self.s.factorize(bbegin, bend, bi, bx);
        if status != blu::OK && status != blu::WARNING_SINGULAR_MATRIX {
            return Err("basiclu_factorize failed".into());
        }
        let x = &self.s.xstore;
        let matrix_nz = x[blu::MATRIX_NZ] as Int;
        let lnz = x[blu::LNZ] as Int;
        let unz = x[blu::UNZ] as Int;
        let dim = x[blu::DIM] as Int;
        self.fill_factor = 1.0 * (lnz + unz + dim) as f64 / matrix_nz as f64;

        let stability = x[blu::RESIDUAL_TEST];
        if control.debug(3) {
            control.debug_out(
                3,
                &format!(
                    " normLinv = {}, normUinv = {}, stability = {}\n",
                    sci2(x[blu::NORMEST_LINV]),
                    sci2(x[blu::NORMEST_UINV]),
                    sci2(stability)
                ),
            );
        }
        let mut ret = 0;
        if stability > LU_STABILITY_THRESHOLD {
            ret |= 1;
        }
        if status == blu::WARNING_SINGULAR_MATRIX {
            ret |= 2;
        }
        Ok(ret)
    }

    fn get_factors(
        &mut self,
        l: Option<&mut SparseMatrix>,
        u: Option<&mut SparseMatrix>,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
        dependent_cols: Option<&mut Vec<Int>>,
    ) -> LuResult<()> {
        let dim = self.s.xstore[blu::DIM] as Int;
        let has_l = l.is_some();
        // keep L to drop its unit diagonal afterwards
        let mut l = l;
        self.s.get_factors(l.as_deref_mut(), u, rowperm, colperm)?;
        if has_l {
            let num_dropped = remove_diagonal(l.unwrap(), None);
            debug_assert_eq!(num_dropped, dim);
        }
        if let Some(dc) = dependent_cols {
            // Dependent columns are at the end of the BASICLU pivot sequence.
            let rank = self.s.xstore[blu::RANK] as Int;
            dc.clear();
            dc.extend(rank..dim);
        }
        Ok(())
    }

    fn update(&mut self, control: &Control, pivot: f64) -> LuResult<Int> {
        let max_eta_old = self.s.xstore[blu::MAX_ETA];
        let status = self.s.update(pivot);
        if status != blu::OK && status != blu::ERROR_SINGULAR_UPDATE {
            return Err("basiclu_update failed".into());
        }
        if status == blu::ERROR_SINGULAR_UPDATE {
            return Ok(-1);
        }
        // Print a debugging message if a new eta entry is large.
        let max_eta = self.s.xstore[blu::MAX_ETA];
        if max_eta > 1e10 && max_eta > max_eta_old {
            control.debug_out(3, &format!(" max eta = {}\n", sci2(max_eta)));
        }
        // stability check
        let pivot_error = self.s.xstore[blu::PIVOT_ERROR];
        if pivot_error > FT_DIAG_ERROR_TOL {
            control.debug_out(
                3,
                &format!(" relative error in new diagonal entry of U = {}\n", sci2(pivot_error)),
            );
            return Ok(1);
        }
        Ok(0)
    }

    fn need_fresh_factorization(&self) -> bool {
        let x = &self.s.xstore;
        let dim = x[blu::DIM] as Int;
        let nforrest = x[blu::NFORREST] as Int;
        let update_cost = x[blu::UPDATE_COST];
        nforrest == dim || update_cost > 1.0
    }
}

// --- lu_factorization.cc / basiclu_kernel.cc ---

/// Returns the matrix which in exact arithmetic would be L*U.
fn permuted_matrix(
    bbegin: &[Int],
    bend: &[Int],
    bi: &[Int],
    bx: &[f64],
    rowperm: &[Int],
    colperm: &[Int],
    dependent_cols: &[Int],
) -> SparseMatrix {
    let dim = rowperm.len();
    let permuted_row = utils::inverse_perm(rowperm);
    let mut dependent = vec![false; dim];
    for &k in dependent_cols {
        dependent[k as usize] = true;
    }
    let mut b = SparseMatrix::new(dim as Int, 0);
    for k in 0..dim {
        if dependent[k] {
            b.push_back(k as Int, 1.0);
        } else {
            let j = colperm[k] as usize;
            for p in bbegin[j] as usize..bend[j] as usize {
                b.push_back(permuted_row[bi[p] as usize], bx[p]);
            }
        }
        b.add_column();
    }
    b
}

/// Chooses rhs entries +/-1 such that (L+I)\rhs becomes large and computes
/// lhs = U\(L+I)\rhs.
fn solve_forward(l: &SparseMatrix, u: &SparseMatrix, rhs: &mut [f64], lhs: &mut [f64]) {
    lhs.fill(0.0);
    for i in 0..rhs.len() {
        rhs[i] = if lhs[i] >= 0.0 { 1.0 } else { -1.0 };
        lhs[i] += rhs[i];
        scatter_column(l, i, -lhs[i], lhs);
    }
    triangular_solve(u, lhs, b'n', true, false);
}

/// Chooses rhs entries +/-1 such that U'\rhs becomes large and computes
/// lhs = (L+I)'\U'\rhs.
fn solve_backward(l: &SparseMatrix, u: &SparseMatrix, rhs: &mut [f64], lhs: &mut [f64]) {
    lhs.fill(0.0);
    for j in 0..rhs.len() {
        lhs[j] -= dot_column(u, j, lhs);
        rhs[j] = if lhs[j] >= 0.0 { 1.0 } else { -1.0 };
        lhs[j] += rhs[j];
        let p = u.end(j) - 1;
        lhs[j] /= u.value(p);
    }
    triangular_solve(l, lhs, b't', false, true);
}

/// Stability measure for the factorization L*U=B: the maximum of the scaled
/// residuals norm(b-Bx) / (norm(b) + norm(B)*norm(x)) in 1-norm for two
/// right-hand sides (as in BASICLU).
#[allow(clippy::too_many_arguments)]
fn stability_estimate(
    bbegin: &[Int],
    bend: &[Int],
    bi: &[Int],
    bx: &[f64],
    l: &SparseMatrix,
    u: &SparseMatrix,
    rowperm: &[Int],
    colperm: &[Int],
    dependent_cols: &[Int],
) -> f64 {
    let dim = rowperm.len();
    let mut rhs = vec![0.0; dim];
    let mut lhs = vec![0.0; dim];

    let b = permuted_matrix(bbegin, bend, bi, bx, rowperm, colperm, dependent_cols);
    let onenorm_b = onenorm(&b);
    let infnorm_b = infnorm(&b);

    solve_forward(l, u, &mut rhs, &mut lhs);
    let norm_ftran = utils::onenorm(&lhs);
    multiply_add(&b, &lhs, -1.0, &mut rhs, b'N');
    let norm_ftran_res = utils::onenorm(&rhs);

    solve_backward(l, u, &mut rhs, &mut lhs);
    let norm_btran = utils::onenorm(&lhs);
    multiply_add(&b, &lhs, -1.0, &mut rhs, b'T');
    let norm_btran_res = utils::onenorm(&rhs);

    cmax(
        norm_ftran_res / onenorm_b.mul_add_c(norm_ftran, dim as f64),
        norm_btran_res / infnorm_b.mul_add_c(norm_btran, dim as f64),
    )
}

/// LuFactorization::Factorize with the BasicLuKernel implementation:
/// B[rowperm,colperm] = (L+I)*U, dependent columns replaced by unit
/// columns. Returns the stability estimate.
#[allow(clippy::too_many_arguments)]
fn kernel_factorize(
    dim: Int,
    bbegin: &[Int],
    bend: &[Int],
    bi: &[Int],
    bx: &[f64],
    pivottol: f64,
    strict_abs_pivottol: bool,
    l: &mut SparseMatrix,
    u: &mut SparseMatrix,
    rowperm: &mut Vec<Int>,
    colperm: &mut Vec<Int>,
    dependent_cols: &mut Vec<Int>,
) -> LuResult<f64> {
    // basiclu_obj_initialize: factor files of dim elements
    let mut s = Store::new(dim, dim as usize)?;
    s.xstore[blu::REL_PIVOT_TOLERANCE] = pivottol;
    if strict_abs_pivottol {
        s.xstore[blu::ABS_PIVOT_TOLERANCE] = LU_DEPENDENCY_TOL;
        s.xstore[blu::REMOVE_COLUMNS] = 1.0;
    }
    let status = s.factorize(bbegin, bend, bi, bx);
    if status != blu::OK && status != blu::WARNING_SINGULAR_MATRIX {
        return Err("basiclu_obj_factorize failed".into());
    }
    let rank = s.xstore[blu::RANK] as Int;
    dependent_cols.clear();
    dependent_cols.extend(rank..dim);
    rowperm.resize(dim as usize, 0);
    colperm.resize(dim as usize, 0);
    s.get_factors(Some(l), Some(u), Some(rowperm), Some(colperm))?;
    // Remove unit diagonal from L.
    remove_diagonal(l, None);
    Ok(stability_estimate(
        bbegin,
        bend,
        bi,
        bx,
        l,
        u,
        rowperm,
        colperm,
        dependent_cols,
    ))
}

// --- forrest_tomlin.cc ---

/// Maximum # updates before refactorization is required.
const FT_MAX_UPDATES: usize = 5000;

/// Generic Forrest-Tomlin update on top of an LU factorization; does not
/// exploit hypersparsity. L and U are stored with permuted indices.
struct ForrestTomlin {
    dim: usize,
    rowperm: Vec<Int>,
    colperm: Vec<Int>,
    rowperm_inv: Vec<Int>,
    colperm_inv: Vec<Int>,
    dependent_cols: Vec<Int>,
    stability: f64,
    l: SparseMatrix, // L from factorization
    u: SparseMatrix, // U from factorization with spike cols appended
    r: SparseMatrix, // cols of R build row eta matrices from updates
    // replaced[k] == p if update k replaced position p in pivot sequence
    replaced: Vec<Int>,
    replace_next: Int,
    have_btran: bool,
    have_ftran: bool,
    fill_factor: f64,
    pivottol: f64,
    work: Vec<f64>, // size dim + kMaxUpdates workspace
}

impl ForrestTomlin {
    fn new(dim: Int) -> Self {
        ForrestTomlin {
            dim: dim as usize,
            rowperm: vec![],
            colperm: vec![],
            rowperm_inv: vec![],
            colperm_inv: vec![],
            dependent_cols: vec![],
            stability: 0.0,
            l: SparseMatrix::default(),
            u: SparseMatrix::default(),
            r: SparseMatrix::default(),
            replaced: vec![],
            replace_next: -1,
            have_btran: false,
            have_ftran: false,
            fill_factor: 0.0,
            pivottol: 0.1,
            work: vec![0.0; dim as usize + FT_MAX_UPDATES],
        }
    }

    fn factorize(
        &mut self,
        control: &Control,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
        strict_abs_pivottol: bool,
    ) -> LuResult<Int> {
        let dim = self.dim;
        // Reset updates.
        self.r.resize(dim as Int, 0, 0);
        self.replaced.clear();
        self.replace_next = -1;
        self.have_btran = false;
        self.have_ftran = false;

        self.stability = kernel_factorize(
            dim as Int,
            bbegin,
            bend,
            bi,
            bx,
            self.pivottol,
            strict_abs_pivottol,
            &mut self.l,
            &mut self.u,
            &mut self.rowperm,
            &mut self.colperm,
            &mut self.dependent_cols,
        )?;
        self.rowperm_inv = utils::inverse_perm(&self.rowperm);
        self.colperm_inv = utils::inverse_perm(&self.colperm);

        // Compute fill factor.
        let mut bnz: Int = 0;
        for j in 0..dim {
            bnz += bend[j] - bbegin[j];
        }
        self.fill_factor = 1.0 * (self.l.entries() + self.u.entries()) as f64 / bnz as f64;

        if control.debug(3) {
            let norm_linv = normest_inverse(&self.l, false, true);
            let norm_uinv = normest_inverse(&self.u, true, false);
            control.debug_out(
                3,
                &format!(
                    " normLinv = {}, normUinv = {}, stability = {}\n",
                    sci2(norm_linv),
                    sci2(norm_uinv),
                    sci2(self.stability)
                ),
            );
        }
        let mut ret = 0;
        if self.stability > LU_STABILITY_THRESHOLD {
            ret |= 1;
        }
        if !self.dependent_cols.is_empty() {
            ret |= 2;
        }
        Ok(ret)
    }

    fn get_factors(
        &self,
        l: Option<&mut SparseMatrix>,
        u: Option<&mut SparseMatrix>,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
        dependent_cols: Option<&mut Vec<Int>>,
    ) {
        if let Some(l) = l {
            *l = self.l.clone();
        }
        if let Some(u) = u {
            *u = self.u.clone();
        }
        if let Some(rp) = rowperm {
            rp[..self.dim].copy_from_slice(&self.rowperm);
        }
        if let Some(cp) = colperm {
            cp[..self.dim].copy_from_slice(&self.colperm);
        }
        if let Some(dc) = dependent_cols {
            *dc = self.dependent_cols.clone();
        }
    }

    fn solve_dense(&mut self, rhs: Option<&[f64]>, lhs: &mut [f64], trans: u8) {
        let mut work = std::mem::take(&mut self.work);
        if trans == b't' || trans == b'T' {
            utils::permute_back(&self.colperm, rhs.unwrap_or(lhs), &mut work);
            self.solve_permuted(&mut work, b'T');
            utils::permute(&self.rowperm, &work, lhs);
        } else {
            utils::permute_back(&self.rowperm, rhs.unwrap_or(lhs), &mut work);
            self.solve_permuted(&mut work, b'N');
            utils::permute(&self.colperm, &work, lhs);
        }
        self.work = work;
    }

    fn ftran_for_update(&mut self, bi: &[Int], bx: &[f64], lhs: Option<&mut IndexedVector>) {
        self.compute_spike(bi, bx);
        if let Some(lhs) = lhs {
            triangular_solve(&self.u, &mut self.work, b'n', true, false);
            // Move extra variables from updates to replaced positions.
            let num_updates = self.replaced.len();
            for k in (0..num_updates).rev() {
                self.work[self.replaced[k] as usize] = self.work[self.dim + k];
            }
            // Return lhs without pattern.
            for p in 0..self.dim {
                lhs[self.colperm[p] as usize] = self.work[p];
            }
            lhs.invalidate_pattern();
        }
    }

    fn btran_for_update(&mut self, j: Int, lhs: Option<&mut IndexedVector>) {
        self.compute_eta(j);
        if let Some(lhs) = lhs {
            // Apply update etas and solve with L'.
            let num_updates = self.replaced.len();
            let dim = self.dim;
            for k in (0..num_updates).rev() {
                let a = -self.work[dim + k];
                scatter_column(&self.r, k, a, &mut self.work);
                self.work[self.replaced[k] as usize] = self.work[dim + k];
                self.work[dim + k] = 0.0;
            }
            triangular_solve(&self.l, &mut self.work, b't', false, true);
            // Return lhs without pattern.
            for p in 0..dim {
                lhs[self.rowperm[p] as usize] = self.work[p];
            }
            lhs.invalidate_pattern();
        }
    }

    fn update(&mut self, control: &Control, pivot: f64) -> Int {
        let num_updates = self.replaced.len();
        let dim = self.dim;
        debug_assert!(self.have_ftran && self.have_btran);

        // Find the position of the entry with row index replace_next in the
        // spike. If not present, we will have where_ == qend.
        let rn = self.replace_next;
        let qend = self.u.queue_size();
        let mut where_ = 0;
        while where_ < qend && self.u.qindex(where_) != rn {
            where_ += 1;
        }

        // Compute new diagonal entry of U. newdiag1 will be inserted into U;
        // newdiag2 would be the same in exact arithmetic and is for
        // monitoring numerical stability.
        let rn = rn as usize;
        let olddiag = self.u.value(self.u.end(rn) - 1);
        let newdiag1 = pivot * olddiag;
        let newdiag2 = (if where_ == qend {
            0.0
        } else {
            self.u.qvalue(where_)
        }) - sparse_dot(&self.u, &self.r);
        let newdiag_err = (newdiag1 - newdiag2).abs();
        let rel_newdiag_err = newdiag_err / newdiag1.abs();

        // Put new diagonal entry at end of spike.
        if where_ < qend {
            for l in where_..qend - 1 {
                let (i, x) = (self.u.qindex(l + 1), self.u.qvalue(l + 1));
                self.u.set_qentry(l, i, x);
            }
            self.u.set_qentry(qend - 1, (dim + num_updates) as Int, newdiag1);
        } else {
            self.u.push_back((dim + num_updates) as Int, newdiag1);
        }

        // Overwrite replaced column by unit column in U.
        let end = self.u.end(rn);
        for l in self.u.begin(rn)..end - 1 {
            self.u.values[l] = 0.0;
        }
        self.u.values[end - 1] = 1.0;

        // Finish update.
        self.u.add_column();
        self.r.add_column();
        self.replaced.push(rn as Int);
        self.replace_next = -1;
        self.have_btran = false;
        self.have_ftran = false;

        if newdiag1 == 0.0 {
            return -1;
        }

        // Print a debugging message if a new eta entry is large.
        let mut max_eta = 0.0;
        for l in self.r.begin(num_updates)..self.r.end(num_updates) {
            max_eta = cmax(max_eta, self.r.value(l).abs());
        }
        if max_eta > 1e10 {
            control.debug_out(3, &format!(" max eta = {}\n", sci2(max_eta)));
        }

        // stability check
        if rel_newdiag_err > FT_DIAG_ERROR_TOL {
            control.debug_out(
                3,
                &format!(
                    " relative error in new diagonal entry of U = {}\n",
                    sci2(rel_newdiag_err)
                ),
            );
            return 1;
        }
        0
    }

    fn need_fresh_factorization(&self) -> bool {
        let num_updates = self.replaced.len();
        let rnz = self.r.entries(); // nnz in accumulated row etas
        let lnz = self.l.entries() + self.dim as Int; // nnz(L) incl. diagonal
        let unz = self.u.entries(); // nnz(U) incl. zeroed out columns
        let u0nz = self.u.begin(self.dim) as Int; // nnz(U) after factorization

        if num_updates == FT_MAX_UPDATES {
            return true;
        }
        if num_updates < 100 {
            return false;
        }
        if rnz as f64 > 1.0 * lnz as f64 {
            return true;
        }
        if unz as f64 > 1.7 * u0nz as f64 {
            return true;
        }
        false
    }

    /// Solves with the basis matrix; lhs holds the permuted right-hand side
    /// on entry and the permuted solution on return, and has dim + #
    /// updates entries (the extra ones are workspace).
    fn solve_permuted(&self, lhs: &mut [f64], trans: u8) {
        let num_updates = self.replaced.len();
        let dim = self.dim;
        if trans == b't' || trans == b'T' {
            // Move replaced entries to the end of the pivot sequence and
            // zero out their old position. Because the corresponding columns
            // of U are unit columns now, the replaced positions remain zero
            // when solving with U'. This is crucial because we neglected to
            // zero out the corresponding rows of U in the update, so there
            // are entries in U which actually should not be there.
            for k in 0..num_updates {
                let r = self.replaced[k] as usize;
                lhs[dim + k] = lhs[r];
                lhs[r] = 0.0;
            }
            triangular_solve(&self.u, lhs, b't', true, false);
            // Solve backwards with row eta matrices (leading scatter
            // operations) and put the entry from the end of the pivot
            // sequence back into the position that its update replaced.
            for k in (0..num_updates).rev() {
                let a = -lhs[dim + k];
                scatter_column(&self.r, k, a, lhs);
                lhs[self.replaced[k] as usize] = lhs[dim + k];
                lhs[dim + k] = 0.0;
            }
            triangular_solve(&self.l, lhs, b't', false, true);
        } else {
            triangular_solve(&self.l, lhs, b'n', false, true);
            // Solve forward with row eta matrices (leading gather
            // operations) and put the newly computed entry at the end of the
            // pivot sequence.
            for k in 0..num_updates {
                let r = self.replaced[k] as usize;
                lhs[dim + k] = lhs[r] - dot_column(&self.r, k, lhs);
                lhs[r] = 0.0;
            }
            // The triangular solve with U fills the replaced positions with
            // garbage, which is not propagated further since the columns of
            // U are unit columns; overwrite with the values from the end of
            // the pivot sequence.
            triangular_solve(&self.u, lhs, b'n', true, false);
            for k in (0..num_updates).rev() {
                lhs[self.replaced[k] as usize] = lhs[dim + k];
                lhs[dim + k] = 0.0;
            }
        }
    }

    /// Computes the spike R_k^{-1} * ... * R_1^{-1} * L^{-1} * b, stores it
    /// at the end of U and returns it as a full vector in work.
    fn compute_spike(&mut self, bi: &[Int], bx: &[f64]) {
        let num_updates = self.replaced.len();
        let dim = self.dim;
        let work = &mut self.work;

        // Solve L*lhs=b.
        work.fill(0.0);
        for p in 0..bi.len() {
            work[self.rowperm_inv[bi[p] as usize] as usize] = bx[p];
        }
        triangular_solve(&self.l, work, b'n', false, true);

        // Apply update etas.
        for k in 0..num_updates {
            let r = self.replaced[k] as usize;
            work[dim + k] = work[r] - dot_column(&self.r, k, work);
            work[r] = 0.0;
        }

        // Store spike in U. Indices are sorted, which is required for the
        // sparse dot product in update().
        self.u.clear_queue();
        for p in 0..dim + num_updates {
            if work[p] != 0.0 {
                self.u.push_back(p as Int, work[p]);
            }
        }
        self.have_ftran = true;
    }

    /// Computes the partial BTRAN solution r = ep' * U^{-1} and stores the
    /// row eta -r/r[p] (without unit diagonal) at the end of R.
    fn compute_eta(&mut self, j: Int) {
        let num_updates = self.replaced.len();
        let dim = self.dim;

        // Find permuted position of j.
        let mut pos = self.colperm_inv[j as usize] as usize;
        for k in 0..num_updates {
            if self.replaced[k] as usize == pos {
                pos = dim + k;
            }
        }

        // Solve lhs'U=e_pos'. Replaced positions remain zero.
        let work = &mut self.work;
        work.fill(0.0);
        work[pos] = 1.0;
        triangular_solve(&self.u, work, b't', true, false);

        // Queue eta at end of R (sorted indices).
        self.r.clear_queue();
        let pivot = work[pos];
        for i in pos + 1..dim + num_updates {
            if work[i] != 0.0 {
                self.r.push_back(i as Int, -work[i] / pivot);
            }
        }
        self.have_btran = true;
        self.replace_next = pos as Int;
    }
}

/// Dot product of the entries in the queues of A1 and A2 (sorted indices).
fn sparse_dot(a1: &SparseMatrix, a2: &SparseMatrix) -> f64 {
    let (q1, q2) = (a1.queue_size(), a2.queue_size());
    let (mut p1, mut p2) = (0, 0);
    let mut d = 0.0f64;
    while p1 < q1 && p2 < q2 {
        let (i1, i2) = (a1.qindex(p1), a2.qindex(p2));
        if i1 == i2 {
            d = a1.qvalue(p1).mul_add_c(a2.qvalue(p2), d);
            p1 += 1;
            p2 += 1;
        } else if i1 < i2 {
            p1 += 1;
        } else {
            p2 += 1;
        }
    }
    d
}

enum Kernel {
    BasicLu(BasicLu),
    ForrestTomlin(Box<ForrestTomlin>),
}

/// class LuUpdate: factorization + update of the basis matrix
pub(crate) struct LuUpdate {
    kernel: Kernel,
    updates: Int, // counts updates since factorization
}

impl LuUpdate {
    /// BasicLu if lu_kernel <= 0, else ForrestTomlin over BasicLuKernel
    pub fn new(lu_kernel: Int, dim: Int) -> LuResult<Self> {
        let kernel = if lu_kernel <= 0 {
            Kernel::BasicLu(BasicLu::new(dim)?)
        } else {
            Kernel::ForrestTomlin(Box::new(ForrestTomlin::new(dim)))
        };
        Ok(LuUpdate { kernel, updates: 0 })
    }

    /// Factorizes the matrix given in 4-array notation. Returns 0 if OK,
    /// bit 1 if the factorization is unstable, bit 2 if singularities were
    /// replaced by unit columns.
    pub fn factorize(
        &mut self,
        control: &Control,
        bbegin: &[Int],
        bend: &[Int],
        bi: &[Int],
        bx: &[f64],
        strict_abs_pivottol: bool,
    ) -> LuResult<Int> {
        self.updates = 0;
        match &mut self.kernel {
            Kernel::BasicLu(k) => k.factorize(control, bbegin, bend, bi, bx, strict_abs_pivottol),
            Kernel::ForrestTomlin(k) => {
                k.factorize(control, bbegin, bend, bi, bx, strict_abs_pivottol)
            }
        }
    }

    /// Exports L, U and permutations of a fresh factorization:
    /// B[rowperm,colperm] = (L+I)*U.
    pub fn get_factors(
        &mut self,
        l: Option<&mut SparseMatrix>,
        u: Option<&mut SparseMatrix>,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
        dependent_cols: Option<&mut Vec<Int>>,
    ) -> LuResult<()> {
        match &mut self.kernel {
            Kernel::BasicLu(k) => k.get_factors(l, u, rowperm, colperm, dependent_cols),
            Kernel::ForrestTomlin(k) => {
                k.get_factors(l, u, rowperm, colperm, dependent_cols);
                Ok(())
            }
        }
    }

    /// Solves with dense right-hand side (None: in place in lhs).
    pub fn solve_dense(&mut self, rhs: Option<&[f64]>, lhs: &mut [f64], trans: u8) -> LuResult<()> {
        match &mut self.kernel {
            Kernel::BasicLu(k) => k.s.solve_dense(rhs, lhs, trans),
            Kernel::ForrestTomlin(k) => {
                k.solve_dense(rhs, lhs, trans);
                Ok(())
            }
        }
    }

    /// Solves B*x=b in preparation for replacing a column of B by b,
    /// returning x in lhs if given.
    pub fn ftran_for_update(
        &mut self,
        bi: &[Int],
        bx: &[f64],
        lhs: Option<&mut IndexedVector>,
    ) -> LuResult<()> {
        match &mut self.kernel {
            Kernel::BasicLu(k) => {
                let status = match lhs {
                    None => k.s.solve_for_update(bi, bx, None, b'N'),
                    Some(lhs) => {
                        let mut nzlhs = 0;
                        lhs.set_to_zero();
                        let st = k.s.solve_for_update(
                            bi,
                            bx,
                            Some((&mut nzlhs, &mut lhs.pattern, &mut lhs.elements)),
                            b'N',
                        );
                        if st == blu::OK {
                            lhs.set_nnz(nzlhs);
                        }
                        st
                    }
                };
                if status != blu::OK {
                    return Err("basiclu_solve_for_update (ftran) failed".into());
                }
                Ok(())
            }
            Kernel::ForrestTomlin(k) => {
                k.ftran_for_update(bi, bx, lhs);
                Ok(())
            }
        }
    }

    /// Solves B'*y=ej in preparation for replacing column j of B.
    pub fn btran_for_update(&mut self, j: Int, lhs: Option<&mut IndexedVector>) -> LuResult<()> {
        match &mut self.kernel {
            Kernel::BasicLu(k) => {
                let irhs = [j];
                let status = match lhs {
                    None => k.s.solve_for_update(&irhs, &[], None, b'T'),
                    Some(lhs) => {
                        let mut nzlhs = 0;
                        lhs.set_to_zero();
                        let st = k.s.solve_for_update(
                            &irhs,
                            &[],
                            Some((&mut nzlhs, &mut lhs.pattern, &mut lhs.elements)),
                            b'T',
                        );
                        if st == blu::OK {
                            lhs.set_nnz(nzlhs);
                        }
                        st
                    }
                };
                if status != blu::OK {
                    return Err("basiclu_solve_for_update (btran) failed".into());
                }
                Ok(())
            }
            Kernel::ForrestTomlin(k) => {
                k.btran_for_update(j, lhs);
                Ok(())
            }
        }
    }

    /// Updates the factorization (column from the last BtranForUpdate,
    /// replaced by the column from the last FtranForUpdate). Returns < 0 if
    /// singular, > 0 if the update looks unstable, 0 otherwise.
    pub fn update(&mut self, control: &Control, pivot: f64) -> LuResult<Int> {
        self.updates += 1;
        match &mut self.kernel {
            Kernel::BasicLu(k) => k.update(control, pivot),
            Kernel::ForrestTomlin(k) => Ok(k.update(control, pivot)),
        }
    }

    pub fn need_fresh_factorization(&self) -> bool {
        match &self.kernel {
            Kernel::BasicLu(k) => k.need_fresh_factorization(),
            Kernel::ForrestTomlin(k) => k.need_fresh_factorization(),
        }
    }

    pub fn fill_factor(&self) -> f64 {
        match &self.kernel {
            Kernel::BasicLu(k) => k.fill_factor,
            Kernel::ForrestTomlin(k) => k.fill_factor,
        }
    }

    pub fn pivottol(&self) -> f64 {
        match &self.kernel {
            Kernel::BasicLu(k) => k.s.xstore[blu::REL_PIVOT_TOLERANCE],
            Kernel::ForrestTomlin(k) => k.pivottol,
        }
    }

    pub fn set_pivottol(&mut self, new_pivottol: f64) {
        match &mut self.kernel {
            Kernel::BasicLu(k) => k.s.xstore[blu::REL_PIVOT_TOLERANCE] = new_pivottol,
            Kernel::ForrestTomlin(k) => k.pivottol = new_pivottol,
        }
    }

    /// # updates since the last factorization
    pub fn updates(&self) -> Int {
        self.updates
    }
}
