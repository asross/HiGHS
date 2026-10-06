//! Basis (basis.h/.cc): an ordered set of m columns of AI forming a
//! nonsingular matrix, with simplex-type linear algebra, plus the
//! construction of a starting basis (crash, repair, pivoting free variables
//! in and fixed variables out).

use crate::util::fma::ClangFma;

use super::control::Control;
use super::fmt::{fmt, g, sci2, textline, time};
use super::guess_basis::guess_basis;
use super::indexed_vector::{self, IndexedVector};
use super::lu::{LuResult, LuUpdate};
use super::model::Model;
use super::sparse_matrix::{dot_column, scatter_column, SparseMatrix};
use super::symbolic_invert::symbolic_invert;
use super::utils::{all_finite, find_max_abs, twonorm};
use super::{
    cmax, Info, Int, ERROR_BASIS_SINGULAR, ERROR_BASIS_TOO_ILL_CONDITIONED,
    HYPERSPARSE_THRESHOLD,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// Basis repair terminates when the maximum absolute entry in inverse(B)
/// is smaller than kBasisRepairThreshold. At most kMaxBasisRepair repair
/// operations are performed.
const BASIS_REPAIR_THRESHOLD: f64 = 1e5;
const MAX_BASIS_REPAIR: Int = 200;

/// Status of a variable. NONBASIC_FIXED never enters, BASIC_FREE never
/// leaves (by convention of the callers).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BasicStatus {
    NonbasicFixed,
    Nonbasic,
    Basic,
    BasicFree,
}

pub struct Basis {
    control: Rc<Control>,
    model: Rc<Model>,
    basis: Vec<Int>, // m column indices of AI
    // For 0 <= j < n+m, map2basis[j] is
    //  -2:            variable is NONBASIC_FIXED
    //  -1:            variable is NONBASIC
    //   0 <= p < m:   variable is BASIC and at position p in the basis
    //   m <= p < 2*m: variable is BASIC_FREE and at position p-m
    map2basis: Vec<Int>,
    lu: RefCell<LuUpdate>,
    factorization_is_fresh: bool,

    num_factorizations: Int,
    num_updates: Int,
    num_ftran: Int,
    num_btran: Int,
    num_ftran_sparse: Int,
    num_btran_sparse: Int,
    time_ftran: f64,
    time_btran: f64,
    time_update: f64,
    time_factorize: f64,
    fill_factors: Vec<f64>,
    sum_ftran_density: f64,
    sum_btran_density: f64,
}

/// Power method: the largest eigenvalue of the (symmetric positive
/// definite) operator func, v returning the eigenvector (power_method.h).
fn power_method(mut func: impl FnMut(&[f64], &mut [f64]) -> LuResult<()>, v: &mut [f64]) -> LuResult<f64> {
    const MAXITER: Int = 100;
    const TOL: f64 = 1e-3;
    let dim = v.len();
    let mut fv = vec![0.0; dim];
    for i in 0..dim {
        v[i] = 1.0 + 1.0 / (i + 1) as f64;
    }
    let norm = twonorm(v);
    for x in v.iter_mut() {
        *x /= norm;
    }
    let mut lambda = 0.0f64;
    let mut iter = 0;
    while iter < MAXITER {
        iter += 1;
        func(v, &mut fv)?;
        let lambda_old = lambda;
        lambda = twonorm(&fv);
        for i in 0..dim {
            v[i] = fv[i] / lambda;
        }
        if (lambda - lambda_old).abs() <= TOL * lambda {
            break;
        }
    }
    Ok(lambda)
}

impl Basis {
    /// Initializes to the slack basis.
    pub fn new(control: Rc<Control>, model: Rc<Model>) -> LuResult<Basis> {
        let m = model.rows();
        let n = model.cols();
        let mut lu = LuUpdate::new(control.lu_kernel(), m as Int)?;
        lu.set_pivottol(control.lu_pivottol());
        let mut basis = Basis {
            control,
            model,
            basis: vec![0; m],
            map2basis: vec![0; n + m],
            lu: RefCell::new(lu),
            factorization_is_fresh: false,
            num_factorizations: 0,
            num_updates: 0,
            num_ftran: 0,
            num_btran: 0,
            num_ftran_sparse: 0,
            num_btran_sparse: 0,
            time_ftran: 0.0,
            time_btran: 0.0,
            time_update: 0.0,
            time_factorize: 0.0,
            fill_factors: vec![],
            sum_ftran_density: 0.0,
            sum_btran_density: 0.0,
        };
        basis.set_to_slack_basis()?;
        Ok(basis)
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The variable at position p
    #[inline]
    pub fn at(&self, p: usize) -> usize {
        self.basis[p] as usize
    }

    #[inline]
    pub fn status_of(&self, j: usize) -> BasicStatus {
        let m = self.model.rows() as Int;
        let p = self.map2basis[j];
        if p < 0 {
            if p == -1 {
                BasicStatus::Nonbasic
            } else {
                BasicStatus::NonbasicFixed
            }
        } else if p < m {
            BasicStatus::Basic
        } else {
            BasicStatus::BasicFree
        }
    }

    /// Position of variable j in the basis, or -1 if nonbasic
    #[inline]
    pub fn position_of(&self, j: usize) -> Int {
        let m = self.model.rows() as Int;
        let p = self.map2basis[j];
        if p < 0 {
            -1
        } else if p < m {
            p
        } else {
            p - m
        }
    }

    #[inline]
    pub fn is_basic(&self, j: usize) -> bool {
        self.map2basis[j] >= 0
    }

    #[inline]
    pub fn is_nonbasic(&self, j: usize) -> bool {
        self.map2basis[j] < 0
    }

    pub fn fix_nonbasic_variable(&mut self, j: usize) {
        if self.status_of(j) == BasicStatus::NonbasicFixed {
            return;
        }
        self.map2basis[j] = -2;
    }

    pub fn free_basic_variable(&mut self, j: usize) {
        if self.status_of(j) == BasicStatus::BasicFree {
            return;
        }
        self.map2basis[j] += self.model.rows() as Int;
    }

    pub fn set_to_slack_basis(&mut self) -> LuResult<()> {
        let m = self.model.rows();
        let n = self.model.cols();
        for i in 0..m {
            self.basis[i] = (n + i) as Int;
        }
        for j in 0..n {
            self.map2basis[j] = -1;
        }
        for i in 0..m {
            self.map2basis[n + i] = i as Int;
        }
        // factorization of slack basis cannot fail other than out of memory
        self.factorize()?;
        Ok(())
    }

    /// Column pointers of the basis matrix in AI
    fn basis_columns(&self) -> (Vec<Int>, Vec<Int>) {
        let ai = self.model.ai();
        let m = self.model.rows();
        let mut begin = vec![0; m];
        let mut end = vec![0; m];
        for i in 0..m {
            // A negative index means an empty slot (crash procedure).
            if self.basis[i] >= 0 {
                begin[i] = ai.colptr[self.basis[i] as usize];
                end[i] = ai.colptr[self.basis[i] as usize + 1];
            }
        }
        (begin, end)
    }

    /// Factorizes the basis matrix from scratch, tightening the pivot
    /// tolerance and repeating if unstable. Returns
    /// IPX_ERROR_basis_singular if slack columns were inserted, 0 otherwise.
    pub fn factorize(&mut self) -> LuResult<Int> {
        let control = Rc::clone(&self.control);
        let model = Rc::clone(&self.model);
        let ai = model.ai();
        let timer = Instant::now();

        let (begin, end) = self.basis_columns();
        let mut basis_num_nz: Int = 0;
        for i in 0..begin.len() {
            basis_num_nz += end[i] - begin[i];
        }
        control.interval_log(&format!(
            "    Start  factorization {}: nonzeros in basis = {}{}{}\n",
            fmt(self.num_factorizations + 1, 3),
            fmt(basis_num_nz, 9),
            fmt("", 14),
            time(control.elapsed())
        ));

        let mut err = 0; // return code
        loop {
            let flag = self
                .lu
                .get_mut()
                .factorize(&control, &begin, &end, &ai.rowidx, &ai.values, false)?;
            self.num_factorizations += 1;
            self.fill_factors.push(self.lu.get_mut().fill_factor());
            if flag & 2 != 0 {
                self.adapt_to_singular_factorization()?;
                err = ERROR_BASIS_SINGULAR;
                break;
            }
            if flag & 1 != 0 && self.tighten_lu_pivot_tol() {
                // The factorization was numerically unstable and the pivot
                // tolerance could be tightened. Repeat.
                continue;
            }
            if flag & 1 != 0 {
                control.debug_out(
                    3,
                    &format!(
                        " LU factorization unstable with pivot tolerance {}\n",
                        g(self.lu.get_mut().pivottol())
                    ),
                );
                // ignore instability
            }
            break;
        }
        self.time_factorize += timer.elapsed().as_secs_f64();
        self.factorization_is_fresh = true;
        control.interval_log(&format!(
            "    Finish factorization {}: fill factor = {}{}{}\n",
            fmt(self.num_factorizations, 3),
            super::fmt::fixed(self.lu.get_mut().fill_factor(), 6, 2),
            fmt("", 23),
            time(control.elapsed())
        ));
        Ok(err)
    }

    pub fn factorization_is_fresh(&self) -> bool {
        self.factorization_is_fresh
    }

    /// L, U and permutations with B[rowperm,colperm] = (L+I)*U, B the
    /// current basis matrix (fresh factorization only).
    pub fn get_lu_factors(
        &self,
        l: Option<&mut SparseMatrix>,
        u: Option<&mut SparseMatrix>,
        rowperm: Option<&mut [Int]>,
        colperm: Option<&mut [Int]>,
    ) -> LuResult<()> {
        debug_assert!(self.factorization_is_fresh);
        self.lu.borrow_mut().get_factors(l, u, rowperm, colperm, None)
    }

    /// Solves with the basis matrix ('t'/'T' transposed): lhs = B\rhs.
    pub fn solve_dense(&self, rhs: &[f64], lhs: &mut [f64], trans: u8) -> LuResult<()> {
        self.lu.borrow_mut().solve_dense(Some(rhs), lhs, trans)
    }

    /// In-place SolveDense(x, x, trans)
    pub fn solve_dense_inplace(&self, x: &mut [f64], trans: u8) -> LuResult<()> {
        self.lu.borrow_mut().solve_dense(None, x, trans)
    }

    /// Solves the linear system in preparation for an update: BTRAN with
    /// the unit vector of j's position if j is basic, else FTRAN with
    /// AI[:,j]. The solution is returned in lhs if given.
    pub fn solve_for_update(&mut self, j: usize, lhs: Option<&mut IndexedVector>) -> LuResult<()> {
        let p = self.position_of(j);
        let timer = Instant::now();
        let dim = self.model.rows();
        let lu = self.lu.get_mut();
        if p < 0 {
            // ftran
            let ai = self.model.ai();
            let (b, e) = (ai.begin(j), ai.end(j));
            let (bi, bx) = (&ai.rowidx[b..e], &ai.values[b..e]);
            match lhs {
                Some(lhs) => {
                    lu.ftran_for_update(bi, bx, Some(&mut *lhs))?;
                    self.num_ftran += 1;
                    self.sum_ftran_density += (1.0 * lhs.nnz() as f64) / dim as f64;
                    if lhs.sparse() {
                        self.num_ftran_sparse += 1;
                    }
                }
                None => lu.ftran_for_update(bi, bx, None)?,
            }
            self.time_ftran += timer.elapsed().as_secs_f64();
        } else {
            // btran
            match lhs {
                Some(lhs) => {
                    lu.btran_for_update(p, Some(&mut *lhs))?;
                    self.num_btran += 1;
                    self.sum_btran_density += (1.0 * lhs.nnz() as f64) / dim as f64;
                    if lhs.sparse() {
                        self.num_btran_sparse += 1;
                    }
                }
                None => lu.btran_for_update(p, None)?,
            }
            self.time_btran += timer.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// Computes row p of the tableau matrix for the basic variable jb at
    /// position p (BTRAN in btran, prepared for update). Basic variables get
    /// zero; with ignore_fixed also NONBASIC_FIXED ones. Chooses a sparse
    /// (pattern set) or dense (pattern invalid) product.
    pub fn tableau_row(
        &mut self,
        jb: usize,
        btran: &mut IndexedVector,
        row: &mut IndexedVector,
        ignore_fixed: bool,
    ) -> LuResult<()> {
        let model = Rc::clone(&self.model);
        let m = model.rows();
        let n = model.cols();
        self.solve_for_update(jb, Some(btran))?;

        // Estimate if tableau row is sparse.
        let mut is_sparse = btran.sparse();
        let ait = model.ait();
        if is_sparse {
            let mut nz: Int = 0;
            for k in 0..btran.nnz() as usize {
                let i = btran.pattern[k] as usize;
                nz += ait.col_entries(i);
            }
            nz /= 2; // guess for overlap
            if nz as f64 > HYPERSPARSE_THRESHOLD * n as f64 {
                is_sparse = false;
            }
        }

        let map2basis = &mut self.map2basis;
        if is_sparse {
            // sparse vector * sparse matrix: accesses A rowwise
            let ati = &ait.rowidx;
            let atx = &ait.values;
            row.set_to_zero();
            let mut nz = 0;
            for &i in &btran.pattern[..btran.nnz() as usize] {
                let i = i as usize;
                let temp = btran.elements[i];
                let (b, e) = (ait.begin(i), ait.end(i));
                for (&j, &v) in ati[b..e].iter().zip(&atx[b..e]) {
                    let j = j as usize;
                    let mb = &mut map2basis[j];
                    if *mb == -1 || (*mb == -2 && !ignore_fixed) {
                        *mb -= 2; // mark column
                        row.pattern[nz] = j as Int;
                        nz += 1;
                    }
                    if *mb < -2 {
                        // marked column
                        row.elements[j] = temp.mul_add_c(v, row.elements[j]);
                    }
                }
            }
            for k in 0..nz {
                // reset marked
                map2basis[row.pattern[k] as usize] += 2;
            }
            row.set_nnz(nz as Int);
        } else {
            // dense vector * sparse matrix: accesses A columnwise
            let ai = model.ai();
            let (aidx, ax) = (&ai.rowidx, &ai.values);
            let btr = &btran.elements[..m];
            let cols = ai.colptr[..n + m + 1].windows(2);
            for ((c, &mb), r) in cols.zip(&map2basis[..n + m]).zip(&mut row.elements[..n + m]) {
                let mut result = 0.0f64;
                if mb == -1 || (mb == -2 && !ignore_fixed) {
                    let (b, e) = (c[0] as usize, c[1] as usize);
                    for (&i, &v) in aidx[b..e].iter().zip(&ax[b..e]) {
                        result = v.mul_add_c(btr[i as usize], result);
                    }
                }
                *r = result;
            }
            row.invalidate_pattern();
        }
        Ok(())
    }

    /// Exchanges basic variable jb with nonbasic jn if the LU update is
    /// stable; otherwise refactorizes (possibly tightening the pivot
    /// tolerance) or returns IPX_ERROR_basis_too_ill_conditioned. sys > 0
    /// (< 0) if the forward (transposed) system still needs to be solved.
    /// Returns (errflag, exchanged).
    pub fn exchange_if_stable(
        &mut self,
        jb: usize,
        jn: usize,
        tableau_entry: f64,
        sys: i32,
    ) -> LuResult<(Int, bool)> {
        if sys > 0 {
            self.solve_for_update(jn, None)?;
        }
        if sys < 0 {
            self.solve_for_update(jb, None)?;
        }

        // Update factorization.
        let timer = Instant::now();
        let err = self.lu.get_mut().update(&self.control, tableau_entry)?;
        self.time_update += timer.elapsed().as_secs_f64();
        if err != 0 {
            if self.factorization_is_fresh && !self.tighten_lu_pivot_tol() {
                return Ok((ERROR_BASIS_TOO_ILL_CONDITIONED, false));
            }
            self.control.debug_out(
                3,
                &format!(
                    " stability check forced refactorization after {} updates\n",
                    self.lu.get_mut().updates() - 1
                ),
            );
            return Ok((self.factorize()?, false)); // refactorizes the old basis
        }

        // Update basis.
        let ib = self.position_of(jb);
        self.basis[ib as usize] = jn as Int;
        self.map2basis[jn] = ib; // status now BASIC
        self.map2basis[jb] = -1; // status now NONBASIC
        self.num_updates += 1;
        self.factorization_is_fresh = false;

        if self.lu.get_mut().need_fresh_factorization() {
            return Ok((self.factorize()?, true));
        }
        Ok((0, true))
    }

    /// Computes x[basic], y and z[nonbasic] such that Ax=b and A'y+z=c,
    /// given x[nonbasic] and z[basic].
    pub fn compute_basic_solution(&self, x: &mut [f64], y: &mut [f64], z: &mut [f64]) -> LuResult<()> {
        let model = &self.model;
        let m = model.rows();
        let n = model.cols();
        let (b, c, ai) = (model.b(), model.c(), model.ai());

        // Compute x[basic] so that Ax=b. Use y as workspace.
        y.copy_from_slice(b);
        for j in 0..n + m {
            if self.is_nonbasic(j) {
                scatter_column(ai, j, -x[j], y);
            }
        }
        self.solve_dense_inplace(y, b'N')?;
        for p in 0..m {
            x[self.at(p)] = y[p];
        }

        // Compute y and z[nonbasic] so that AI'y+z=c.
        for p in 0..m {
            y[p] = c[self.at(p)] - z[self.at(p)];
        }
        self.solve_dense_inplace(y, b'T')?;
        for j in 0..n + m {
            if self.is_nonbasic(j) {
                z[j] = c[j] - dot_column(ai, j, y);
            }
        }
        Ok(())
    }

    /// Constructs a nonsingular basis preferring columns of larger weight
    /// (infinite weight: basic unless singular; zero: nonbasic unless
    /// required).
    pub fn construct_basis_from_weights(&mut self, colscale: &[f64], info: &mut Info) -> LuResult<()> {
        info.errflag = 0;
        info.dependent_rows = 0;
        info.dependent_cols = 0;

        if self.control.crash_basis() != 0 {
            self.crash_basis(colscale)?;
            let sigma = self.min_singular_value()?;
            self.control.debug_out(
                1,
                &format!("{}{}\n", textline("Minimum singular value of crash basis:"), sci2(sigma)),
            );
            self.repair(info)?;
            if info.basis_repairs < 0 {
                self.control.log(" discarding crash basis\n");
                self.set_to_slack_basis()?;
            } else if info.basis_repairs > 0 {
                let sigma = self.min_singular_value()?;
                self.control.debug_out(
                    1,
                    &format!(
                        "{}{}\n",
                        textline("Minimum singular value of repaired crash basis:"),
                        sci2(sigma)
                    ),
                );
            }
        } else {
            self.set_to_slack_basis()?;
        }
        self.pivot_free_variables_into_basis(colscale, info)?;
        if info.errflag != 0 {
            return Ok(());
        }
        self.pivot_fixed_variables_out_of_basis(colscale, info)?;
        Ok(())
    }

    /// Estimates the smallest singular value of the basis matrix.
    pub fn min_singular_value(&self) -> LuResult<f64> {
        let m = self.model.rows();
        let mut v = vec![0.0; m];
        // Computes maximum eigenvalue of inverse(B*B').
        let lambda = power_method(
            |x, fx| {
                self.solve_dense(x, fx, b'N')?;
                self.solve_dense_inplace(fx, b'T')
            },
            &mut v,
        )?;
        Ok((1.0 / lambda).sqrt())
    }

    /// # structural nonzeros per row and column of inverse(B).
    pub fn symbolic_invert(&self, rowcounts: Option<&mut [Int]>, colcounts: Option<&mut [Int]>) {
        symbolic_invert(&self.model, &self.basis, rowcounts, colcounts);
    }

    /// Structural density of inverse(B).
    pub fn density_inverse(&self) -> f64 {
        let m = self.model.rows();
        let mut rowcounts = vec![0; m];
        self.symbolic_invert(Some(&mut rowcounts), None);
        // Accumulating rowcounts would result in overflow for large LPs.
        let mut density = 0.0;
        for &c in &rowcounts {
            density += 1.0 * c as f64 / m as f64;
        }
        density / m as f64
    }

    pub fn factorizations(&self) -> Int {
        self.num_factorizations
    }
    pub fn updates_total(&self) -> Int {
        self.num_updates
    }
    pub fn frac_ftran_sparse(&self) -> f64 {
        1.0 * self.num_ftran_sparse as f64 / self.num_ftran as f64
    }
    pub fn frac_btran_sparse(&self) -> f64 {
        1.0 * self.num_btran_sparse as f64 / self.num_btran as f64
    }
    pub fn time_factorize(&self) -> f64 {
        self.time_factorize
    }
    pub fn time_ftran(&self) -> f64 {
        self.time_ftran
    }
    pub fn time_btran(&self) -> f64 {
        self.time_btran
    }
    pub fn time_update(&self) -> f64 {
        self.time_update
    }
    /// Geometric mean of LU fill factors
    pub fn mean_fill(&self) -> f64 {
        if self.fill_factors.is_empty() {
            return 0.0;
        }
        let mut mean = 1.0;
        let num_factors = self.fill_factors.len() as f64;
        for &f in &self.fill_factors {
            mean *= f.powf(1.0 / num_factors);
        }
        mean
    }
    pub fn max_fill(&self) -> f64 {
        // std::max_element: the first largest
        let mut it = self.fill_factors.iter();
        let Some(&first) = it.next() else {
            return 0.0;
        };
        it.fold(first, |a, &b| if a < b { b } else { a })
    }

    /// Adjusts basis and map2basis after a singular factorization (the
    /// dependent columns were replaced by slack columns). Returns the # slack
    /// variables inserted.
    fn adapt_to_singular_factorization(&mut self) -> LuResult<Int> {
        let m = self.model.rows();
        let n = self.model.cols();
        let mut rowperm = vec![0; m];
        let mut colperm = vec![0; m];
        let mut dependent_cols = vec![];
        self.lu.get_mut().get_factors(
            None,
            None,
            Some(&mut rowperm),
            Some(&mut colperm),
            Some(&mut dependent_cols),
        )?;
        for &k in &dependent_cols {
            // Column p of the basis matrix was replaced by the i-th unit
            // column. Insert the corresponding slack variable jn into
            // position p of the basis.
            let p = colperm[k as usize] as usize;
            let i = rowperm[k as usize] as usize;
            let jb = self.basis[p];
            let jn = n + i;
            self.basis[p] = jn as Int;
            self.map2basis[jn] = p as Int; // now BASIC at position p
            if jb >= 0 {
                self.map2basis[jb as usize] = -1; // now NONBASIC
            }
        }
        Ok(dependent_cols.len() as Int)
    }

    /// If possible, tightens the LU pivot tolerance; returns true if so.
    fn tighten_lu_pivot_tol(&mut self) -> bool {
        let lu = self.lu.get_mut();
        let tol = lu.pivottol();
        if tol <= 0.05 {
            lu.set_pivottol(0.1);
        } else if tol <= 0.25 {
            lu.set_pivottol(0.3);
        } else if tol <= 0.5 {
            lu.set_pivottol(0.9);
        } else {
            return false;
        }
        self.control
            .log(&format!(" LU pivot tolerance tightened to {}\n", g(lu.pivottol())));
        true
    }

    /// "Crashes" a basis with preference for variables of larger weight,
    /// nonsingular in exact arithmetic.
    fn crash_basis(&mut self, colweights: &[f64]) -> LuResult<()> {
        // Make a guess for a basis. Then use LU factorization with a strict
        // absolute pivot tolerance to remove dependent columns.
        let cols_guessed = guess_basis(&self.control, &self.model, colweights);

        // Initialize the Basis object and factorize the (partial) basis. If
        // basis[p] is negative, the p-th column of the basis matrix is zero,
        // and a slack column will be inserted by crash_factorize().
        self.basis.fill(-1);
        self.map2basis.fill(-1);
        for (k, &j) in cols_guessed.iter().enumerate() {
            self.basis[k] = j;
            self.map2basis[j as usize] = k as Int;
        }
        let num_dropped = self.crash_factorize()?;
        self.control.debug_out(
            1,
            &format!(
                "{}{}\n",
                textline("Number of columns dropped from guessed basis:"),
                num_dropped
            ),
        );
        Ok(())
    }

    /// Repairs singularities by replacing basic columns by slack columns;
    /// info.basis_repairs >= 0 if repaired successfully, < 0 if failed.
    fn repair(&mut self, info: &mut Info) -> LuResult<()> {
        let m = self.model.rows();
        let n = self.model.cols();
        let mut work = vec![0.0; m];
        info.basis_repairs = 0;

        loop {
            let (pmax, imax, pivot) = inverse_search(self, &mut work)?;
            if pmax < 0 || imax < 0 || !pivot.is_finite() {
                info.basis_repairs = -1;
                break;
            }
            if pivot.abs() < BASIS_REPAIR_THRESHOLD {
                break;
            }
            let jb = self.at(pmax as usize);
            let jn = n + imax as usize;
            if !self.is_nonbasic(jn) {
                info.basis_repairs = -2;
                break;
            }
            if info.basis_repairs >= MAX_BASIS_REPAIR {
                info.basis_repairs = -3;
                break;
            }
            self.solve_for_update(jb, None)?;
            self.solve_for_update(jn, None)?;
            self.crash_exchange(jb, jn, pivot, 0)?;
            info.basis_repairs += 1;
            self.control
                .debug_out(3, &format!(" basis repair: |pivot| = {}\n", sci2(pivot.abs())));
        }
        Ok(())
    }

    /// Factorizes with a strict absolute pivot tolerance (no stability
    /// check); returns the # columns dropped and replaced by slacks.
    fn crash_factorize(&mut self) -> LuResult<Int> {
        let control = Rc::clone(&self.control);
        let model = Rc::clone(&self.model);
        let ai = model.ai();
        let timer = Instant::now();

        let (begin, end) = self.basis_columns();
        let flag = self
            .lu
            .get_mut()
            .factorize(&control, &begin, &end, &ai.rowidx, &ai.values, true)?;
        self.num_factorizations += 1;
        self.fill_factors.push(self.lu.get_mut().fill_factor());
        let mut ndropped = 0;
        if flag & 2 != 0 {
            ndropped = self.adapt_to_singular_factorization()?;
        }
        self.time_factorize += timer.elapsed().as_secs_f64();
        self.factorization_is_fresh = true;
        Ok(ndropped)
    }

    /// Like exchange_if_stable but always exchanges jb and jn;
    /// refactorizes with crash_factorize() if required.
    fn crash_exchange(&mut self, jb: usize, jn: usize, tableau_entry: f64, sys: i32) -> LuResult<()> {
        if sys > 0 {
            self.solve_for_update(jn, None)?;
        }
        if sys < 0 {
            self.solve_for_update(jb, None)?;
        }

        // Update basis.
        let ib = self.position_of(jb);
        self.basis[ib as usize] = jn as Int;
        self.map2basis[jn] = ib; // status now BASIC
        self.map2basis[jb] = -1; // status now NONBASIC
        self.num_updates += 1;
        self.factorization_is_fresh = false;

        // Update factorization.
        let timer = Instant::now();
        let err = self.lu.get_mut().update(&self.control, tableau_entry)?;
        self.time_update += timer.elapsed().as_secs_f64();
        if err != 0 || self.lu.get_mut().need_fresh_factorization() {
            self.control
                .debug_out(3, " refactorization required in CrashExchange()\n");
            self.crash_factorize()?;
        }
        Ok(())
    }

    /// Pivots free variables (infinite weight) into the basis if they can
    /// replace a nonfree basic variable; those that cannot are linearly
    /// dependent on free basic columns (info.dependent_cols).
    fn pivot_free_variables_into_basis(&mut self, colweights: &[f64], info: &mut Info) -> LuResult<()> {
        let control = Rc::clone(&self.control);
        let model = Rc::clone(&self.model);
        let m = model.rows();
        let n = model.cols();
        let mut ftran = IndexedVector::new(m);
        let dependency_tol = cmax(0.0, control.dependency_tol());
        info.errflag = 0;
        info.dependent_cols = 0;
        let mut stability_pivots = 0;

        // Maintain stack of free nonbasic variables.
        let mut remaining: Vec<usize> = (0..n + m)
            .filter(|&j| colweights[j].is_infinite() && self.map2basis[j] < 0)
            .collect();
        control.debug_out(
            1,
            &format!("{}{}\n", textline("Number of free variables nonbasic:"), remaining.len()),
        );

        control.reset_print_interval();
        while let Some(&jn) = remaining.last() {
            info.errflag = control.interrupt_check(-1);
            if info.errflag != 0 {
                return Ok(());
            }

            self.solve_for_update(jn, Some(&mut ftran))?;
            let mut pmax: Int = -1;
            let mut pmax_nonfree: Int = -1;
            let mut fmax = 0.0;
            let mut fmax_nonfree = 0.0;
            ftran.for_each_nonzero(|p, f| {
                let f = f.abs();
                if f > fmax {
                    fmax = f;
                    pmax = p as Int;
                }
                if !colweights[self.basis[p] as usize].is_infinite() && f > fmax_nonfree {
                    fmax_nonfree = f;
                    pmax_nonfree = p as Int;
                }
            });

            if fmax > 4.0 && fmax_nonfree < 1.0 {
                let jb = self.at(pmax as usize);
                let (err, exchanged) = self.exchange_if_stable(jb, jn, ftran[pmax as usize], -1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                remaining.pop();
                remaining.push(jb);
                info.updates_start += 1;
                stability_pivots += 1;
            } else if fmax_nonfree <= dependency_tol {
                // jn cannot be pivoted into the basis. If we do not have an
                // unbounded primal ray yet, then test if column jn yields
                // one: the change in the primal objective caused by a unit
                // increase of x[jn] with corresponding adjustment of free
                // basic variables.
                if info.cols_inconsistent == 0 {
                    let c = model.c();
                    let mut delta_obj = c[jn];
                    ftran.for_each_nonzero(|p, f| {
                        let j = self.at(p);
                        if colweights[j].is_infinite() {
                            delta_obj = (-c[j]).mul_add_c(f, delta_obj);
                        }
                    });
                    if delta_obj.abs() > dependency_tol {
                        control.debug_out(
                            1,
                            &format!(
                                "{}{}\n",
                                textline("Unbounded primal ray with objective change:"),
                                sci2(delta_obj)
                            ),
                        );
                        info.cols_inconsistent = 1;
                    }
                }
                info.dependent_cols += 1;
                remaining.pop();
            } else {
                let jb = self.at(pmax_nonfree as usize);
                let (err, exchanged) =
                    self.exchange_if_stable(jb, jn, ftran[pmax_nonfree as usize], -1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    continue;
                }
                remaining.pop();
                info.updates_start += 1;
            }
            control.interval_log(&format!(" {} free variables remaining\n", remaining.len()));
        }
        control.debug_out(
            1,
            &format!(
                "{}{}\n",
                textline("Number of free variables swapped for stability:"),
                stability_pivots
            ),
        );
        Ok(())
    }

    /// Pivots fixed variables (zero weight) out of the basis if they can be
    /// replaced by a nonfixed variable; those that cannot make the rows of
    /// AI without them dependent (info.dependent_rows).
    fn pivot_fixed_variables_out_of_basis(&mut self, colweights: &[f64], info: &mut Info) -> LuResult<()> {
        let control = Rc::clone(&self.control);
        let model = Rc::clone(&self.model);
        let m = model.rows();
        let n = model.cols();
        let (ai, lb, ub) = (model.ai(), model.lb(), model.ub());
        let mut btran = IndexedVector::new(m);
        let mut row = IndexedVector::new(n + m);
        let dependency_tol = cmax(0.0, control.dependency_tol());
        info.errflag = 0;
        info.dependent_rows = 0;
        let mut stability_pivots = 0;

        // Build model right-hand side after subtracting fixed columns.
        // Needed for dual unboundedness test.
        let mut b_minus_fixed_columns = model.b().to_vec();
        for j in 0..n + m {
            if lb[j] == ub[j] && lb[j] != 0.0 {
                scatter_column(ai, j, -lb[j], &mut b_minus_fixed_columns);
            }
        }

        // Maintain stack of fixed basic variables.
        let mut remaining: Vec<usize> = (n..n + m)
            .filter(|&j| colweights[j] == 0.0 && self.map2basis[j] >= 0)
            .collect();
        control.debug_out(
            1,
            &format!("{}{}\n", textline("Number of fixed variables basic:"), remaining.len()),
        );

        control.reset_print_interval();
        while let Some(&jb) = remaining.last() {
            info.errflag = control.interrupt_check(-1);
            if info.errflag != 0 {
                return Ok(());
            }

            self.tableau_row(jb, &mut btran, &mut row, false)?;
            let mut jmax: Int = -1;
            let mut jmax_nonfixed: Int = -1;
            let mut rmax = 0.0;
            let mut rmax_nonfixed = 0.0;
            row.for_each_nonzero(|j, r| {
                // Ignore structural variables with zero weight.
                if j >= n || colweights[j] != 0.0 {
                    let r = r.abs();
                    if r > rmax {
                        rmax = r;
                        jmax = j as Int;
                    }
                    if colweights[j] != 0.0 && r > rmax_nonfixed {
                        rmax_nonfixed = r;
                        jmax_nonfixed = j as Int;
                    }
                }
            });

            if rmax > 4.0 && rmax_nonfixed < 1.0 {
                let jmax = jmax as usize;
                let (err, exchanged) = self.exchange_if_stable(jb, jmax, row[jmax], 1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                remaining.pop();
                remaining.push(jmax);
                info.updates_start += 1;
                stability_pivots += 1;
            } else if rmax_nonfixed <= dependency_tol {
                // jb cannot be pivoted out of the basis. If we do not have an
                // unbounded dual ray yet, then test if row jb-n yields one:
                // the change in the dual objective caused by a unit increase
                // of y[jb-n] with corresponding adjustment of the remaining
                // y[i].
                if info.rows_inconsistent == 0 {
                    // Fix for #280: use b minus the fixed columns
                    let delta_obj = indexed_vector::dot(&btran, &b_minus_fixed_columns);
                    if delta_obj.abs() > dependency_tol {
                        control.debug_out(
                            1,
                            &format!(
                                "{}{}\n",
                                textline("Unbounded dual ray with objective change:"),
                                sci2(delta_obj)
                            ),
                        );
                        info.rows_inconsistent = 1;
                    }
                }
                info.dependent_rows += 1;
                remaining.pop();
            } else {
                // jb can be exchanged by a non-fixed variable. Among all
                // numerically stable pivots, choose the one that maximizes
                // volume of the basis matrix.
                let mut jmax_scaled: Int = -1;
                let mut rmax_scaled = 0.0;
                row.for_each_nonzero(|j, r| {
                    let r = r.abs();
                    let rscaled = r * colweights[j];
                    if r >= 0.1 * rmax_nonfixed && rscaled > rmax_scaled {
                        rmax_scaled = rscaled;
                        jmax_scaled = j as Int;
                    }
                });
                let jmax_scaled = jmax_scaled as usize;
                let pivot = row[jmax_scaled];
                let (err, exchanged) = self.exchange_if_stable(jb, jmax_scaled, pivot, 1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    continue;
                }
                remaining.pop();
                info.updates_start += 1;
            }
            control.interval_log(&format!(
                "{} fixed variables remaining{}{}\n",
                fmt(remaining.len(), 9),
                fmt("", 38),
                time(control.elapsed())
            ));
        }
        control.debug_out(
            1,
            &format!(
                "{}{}\n",
                textline("Number of fixed variables swapped for stability:"),
                stability_pivots
            ),
        );
        Ok(())
    }

    pub fn report_basis_data(&self) {
        let mut s = String::from("\nBasis data\n");
        s += &format!("    Num factorizations = {}\n", self.factorizations());
        s += &format!("    Num updates = {}\n", self.updates_total());
        if self.num_ftran != 0 {
            s += &format!(
                "    Average density of {:7} FTRANs is {:6.4}; sparse proportion = {:6.4}\n",
                self.num_ftran,
                self.sum_ftran_density / self.num_ftran as f64,
                self.frac_ftran_sparse()
            );
        }
        if self.num_btran != 0 {
            s += &format!(
                "    Average density of {:7} BTRANs is {:6.4}; sparse proportion = {:6.4}\n",
                self.num_btran,
                self.sum_btran_density / self.num_btran as f64,
                self.frac_btran_sparse()
            );
        }
        s += &format!("    Mean fill-in {:>11}\n", g4(self.mean_fill()));
        s += &format!("    Max  fill-in {:>11}\n", g4(self.max_fill()));
        self.control.print(&s);
    }
}

/// printf's %.4g
fn g4(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return g(v);
    }
    let e = format!("{v:.3e}");
    let (mantissa, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..4).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exp.abs())
    } else {
        strip(&format!("{:.*}", (3 - exp) as usize, v))
    }
}

/// Rook search for a large entry in inverse(B): returns (p, i, x) with x
/// the entry at (p,i) of inverse(B) (p a column, i a row of B), or
/// (-1,-1,INFINITY) on overflow (Higham and Relton, "Estimating the Largest
/// Elements of a Matrix", 2015).
fn inverse_search(basis: &Basis, work: &mut [f64]) -> LuResult<(Int, Int, f64)> {
    let m = work.len();
    let mut inverse_max = 0.0;
    for i in 0..m {
        work[i] = 1.0 / (i + 1) as f64;
    }
    loop {
        basis.solve_dense_inplace(work, b'N')?;
        if !all_finite(work) {
            break;
        }
        let pmax = find_max_abs(work);
        work.fill(0.0);
        work[pmax] = 1.0;
        basis.solve_dense_inplace(work, b'T')?;
        if !all_finite(work) {
            break;
        }
        let imax = find_max_abs(work);
        let inverse_entry = work[imax];
        if inverse_entry.abs() <= 2.0 * inverse_max {
            return Ok((pmax as Int, imax as Int, inverse_entry));
        }
        inverse_max = inverse_entry.abs();
        work.fill(0.0);
        work[imax] = 1.0;
    }
    Ok((-1, -1, f64::INFINITY)) // failure
}

/// CopyBasic: x[basis]
pub fn copy_basic(x: &[f64], basis: &Basis) -> Vec<f64> {
    (0..basis.model().rows()).map(|p| x[basis.at(p)]).collect()
}
