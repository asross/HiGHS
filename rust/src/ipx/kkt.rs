//! The KKT solvers of the IPM and their linear operators:
//! kkt_solver.h/.cc (the interface), kkt_solver_diag.cc (normal equations
//! with diagonal preconditioner, normal_matrix.cc, diagonal_precond.cc),
//! kkt_solver_basis.cc (basis preconditioning, splitted_normal_matrix.cc),
//! and conjugate_residuals.cc (the CR method).

use super::basis::{BasicStatus, Basis};
use super::control::Control;
use super::fmt::sci2;
use super::indexed_vector::IndexedVector;
use super::iterate::Iterate;
use super::lu::LuResult;
use super::maxvolume::Maxvolume;
use super::model::Model;
use super::sparse_matrix::{
    add_normal_product, backward_solve, copy_columns, dot_column, forward_solve, permute_rows,
    scale_column, scatter_column, SparseMatrix,
};
use super::utils::{dot, dot_fused, infnorm, inverse_perm};
use super::{
    cmax, Info, Int, ERROR_CR_INF_OR_NAN, ERROR_CR_ITER_LIMIT, ERROR_CR_MATRIX_NOT_POSDEF,
    ERROR_CR_NO_PROGRESS, ERROR_CR_PRECOND_NOT_POSDEF,
};
use std::time::Instant;

/// LinearOperator: lhs = op(rhs), optionally returning dot(rhs, lhs)
pub trait LinearOperator {
    fn apply(&mut self, rhs: &[f64], lhs: &mut [f64], rhs_dot_lhs: Option<&mut f64>);
}

/// NormalMatrix: AI*AI' or AI*W*AI' (W covering all n+m columns of AI)
pub struct NormalMatrix<'a> {
    model: &'a Model,
    w: Option<&'a [f64]>,
    pub time: f64,
}

impl LinearOperator for NormalMatrix<'_> {
    fn apply(&mut self, rhs: &[f64], lhs: &mut [f64], rhs_dot_lhs: Option<&mut f64>) {
        let m = self.model.rows();
        let n = self.model.cols();
        let ai = self.model.ai();
        let (ap, aidx, ax) = (&ai.colptr, &ai.rowidx, &ai.values);
        let timer = Instant::now();

        let rhs = &rhs[..m];
        let lhs = &mut lhs[..m];
        match self.w {
            Some(w) => {
                for ((l, r), wi) in lhs.iter_mut().zip(rhs).zip(&w[n..n + m]) {
                    *l = r * wi;
                }
                for (j, c) in ap[..n + 1].windows(2).enumerate() {
                    normal_column(&aidx[c[0] as usize..c[1] as usize], &ax[c[0] as usize..c[1] as usize], Some(w[j]), rhs, lhs);
                }
            }
            None => {
                lhs.fill(0.0);
                for c in ap[..n + 1].windows(2) {
                    normal_column(&aidx[c[0] as usize..c[1] as usize], &ax[c[0] as usize..c[1] as usize], None, rhs, lhs);
                }
            }
        }
        if let Some(r) = rhs_dot_lhs {
            *r = dot(rhs, lhs);
        }
        self.time += timer.elapsed().as_secs_f64();
    }
}

/// lhs += d * a with d = (a'*rhs) * w, a a column of AI given by (idx,
/// val)
#[inline(always)]
fn normal_column(idx: &[Int], val: &[f64], w: Option<f64>, rhs: &[f64], lhs: &mut [f64]) {
    let mut d = 0.0f64;
    for (&i, &v) in idx.iter().zip(val) {
        d = rhs[i as usize].mul_add(v, d);
    }
    if let Some(w) = w {
        d *= w;
    }
    for (&i, &v) in idx.iter().zip(val) {
        let i = i as usize;
        lhs[i] = d.mul_add(v, lhs[i]);
    }
}

/// DiagonalPrecond: inverse of the diagonal of AI*W*AI'
pub struct DiagonalPrecond<'a> {
    diagonal: &'a [f64],
    pub time: f64,
}

/// DiagonalPrecond::Factorize: the diagonal of AI*W*AI' (W = 0 on the
/// slack columns if None)
fn diagonal_of_normal_matrix(model: &Model, w: Option<&[f64]>, diagonal: &mut [f64]) {
    let m = model.rows();
    let n = model.cols();
    let ai = model.ai();
    if let Some(w) = w {
        diagonal[..m].copy_from_slice(&w[n..n + m]);
        for j in 0..n {
            let wj = w[j];
            for p in ai.begin(j)..ai.end(j) {
                let i = ai.index(p);
                diagonal[i] = (ai.value(p) * wj).mul_add(ai.value(p), diagonal[i]);
            }
        }
    } else {
        diagonal.fill(0.0); // rightmost m columns have weight zero
        for j in 0..n {
            for p in ai.begin(j)..ai.end(j) {
                let i = ai.index(p);
                diagonal[i] = ai.value(p).mul_add(ai.value(p), diagonal[i]);
            }
        }
    }
}

impl LinearOperator for DiagonalPrecond<'_> {
    fn apply(&mut self, rhs: &[f64], lhs: &mut [f64], rhs_dot_lhs: Option<&mut f64>) {
        let timer = Instant::now();
        // The C++ loop is vectorized by 8 for m >= 8, which splits the
        // fused multiply-add of the in-order reduction (utils::dot_blocked)
        let m = rhs.len();
        let nb = if m >= 8 { m - m % 8 } else { 0 };
        let mut rldot = 0.0f64;
        for i in 0..nb {
            lhs[i] = rhs[i] / self.diagonal[i];
            rldot += lhs[i] * rhs[i];
        }
        for i in nb..m {
            lhs[i] = rhs[i] / self.diagonal[i];
            rldot = lhs[i].mul_add(rhs[i], rldot);
        }
        if let Some(r) = rhs_dot_lhs {
            *r = rldot;
        }
        self.time += timer.elapsed().as_secs_f64();
    }
}

/// SplittedNormalMatrix: the normal matrix preconditioned with the basis,
/// I + inverse(B)*N*N'*inverse(B') (column scaling included), with zero
/// rows/columns at the positions of free basic variables, in the
/// permutation of the LU factors of B.
#[derive(Default)]
pub struct SplittedNormalMatrix {
    l: SparseMatrix, // lower triangular factor without unit diagonal
    u: SparseMatrix, // upper triangular factor with scaled columns
    n: SparseMatrix, // N with scaled columns and permuted row indices
    free_positions: Vec<usize>,
    colperm: Vec<Int>,
    rowperm_inv: Vec<Int>,
    work: Vec<f64>,
    time_b: f64,
    time_bt: f64,
    time_nnt: f64,
}

impl SplittedNormalMatrix {
    fn new(m: usize) -> Self {
        SplittedNormalMatrix {
            colperm: vec![0; m],
            rowperm_inv: vec![0; m],
            work: vec![0.0; m],
            ..Default::default()
        }
    }

    fn prepare(&mut self, basis: &Basis, colscale: &[f64]) -> LuResult<()> {
        let model = basis.model();
        let m = model.rows();
        let n = model.cols();
        let ai = model.ai();
        self.n.clear(); // deallocate old memory

        basis.get_lu_factors(
            Some(&mut self.l),
            Some(&mut self.u),
            Some(&mut self.rowperm_inv),
            Some(&mut self.colperm),
        )?;
        self.rowperm_inv = inverse_perm(&self.rowperm_inv);

        for k in 0..m {
            let p = self.colperm[k] as usize;
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::Basic {
                scale_column(&mut self.u, k, colscale[j]);
            }
        }

        let nonbasic_vars: Vec<Int> = (0..n + m)
            .filter(|&j| basis.status_of(j) == BasicStatus::Nonbasic)
            .map(|j| j as Int)
            .collect();
        self.n = copy_columns(ai, &nonbasic_vars);
        permute_rows(&mut self.n, &self.rowperm_inv);
        for (k, &j) in nonbasic_vars.iter().enumerate() {
            scale_column(&mut self.n, k, colscale[j as usize]);
        }

        self.free_positions.clear();
        for k in 0..m {
            let p = self.colperm[k] as usize;
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::BasicFree {
                self.free_positions.push(k);
            }
        }
        Ok(())
    }

    fn reset_time(&mut self) {
        self.time_b = 0.0;
        self.time_bt = 0.0;
        self.time_nnt = 0.0;
    }
}

impl LinearOperator for SplittedNormalMatrix {
    fn apply(&mut self, rhs: &[f64], lhs: &mut [f64], rhs_dot_lhs: Option<&mut f64>) {
        self.work.copy_from_slice(rhs);
        let timer = Instant::now();
        backward_solve(&self.l, &self.u, &mut self.work);
        self.time_bt += timer.elapsed().as_secs_f64();

        lhs.fill(0.0);
        let timer = Instant::now();
        add_normal_product(&self.n, None, &self.work, lhs);
        self.time_nnt += timer.elapsed().as_secs_f64();

        let timer = Instant::now();
        forward_solve(&self.l, &self.u, lhs);
        self.time_b += timer.elapsed().as_secs_f64();

        for i in 0..lhs.len() {
            lhs[i] += rhs[i];
        }
        for &i in &self.free_positions {
            lhs[i] = 0.0;
        }
        if let Some(r) = rhs_dot_lhs {
            *r = dot(rhs, lhs);
        }
    }
}

/// The conjugate residuals method (optionally preconditioned) for C*lhs =
/// rhs, C symmetric positive definite; lhs is the starting point on entry.
pub struct ConjugateResiduals<'a> {
    control: &'a Control,
    pub errflag: Int,
    pub iter: Int,
    pub time: f64,
}

/// y += a*x (valarray `y += a*x`, multiply and add separately; y -= a*x
/// is y += (-a)*x exactly)
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// y = x + b*y (valarray `y = x + b*y`)
fn xpby(y: &mut [f64], x: &[f64], b: f64) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi = xi + b * *yi;
    }
}

/// max_i |resscale[i]*residual[i]| (or infnorm(residual))
fn residual_norm(resscale: Option<&[f64]>, residual: &[f64]) -> f64 {
    match resscale {
        Some(s) => {
            let mut resnorm = 0.0;
            for i in 0..residual.len() {
                resnorm = cmax(resnorm, (s[i] * residual[i]).abs());
            }
            resnorm
        }
        None => infnorm(residual),
    }
}

impl<'a> ConjugateResiduals<'a> {
    pub fn new(control: &'a Control) -> Self {
        ConjugateResiduals {
            control,
            errflag: 0,
            iter: 0,
            time: 0.0,
        }
    }

    /// Terminates when max |resscale[i]*residual[i]| <= tol; maxiter < 0
    /// means m+100.
    pub fn solve(
        &mut self,
        c: &mut dyn LinearOperator,
        rhs: &[f64],
        tol: f64,
        resscale: Option<&[f64]>,
        maxiter: Int,
        lhs: &mut [f64],
    ) {
        let m = rhs.len();
        let mut residual = vec![0.0; m]; // rhs - C*lhs
        let mut step = vec![0.0; m]; // update to lhs
        let mut cresidual = vec![0.0; m]; // C * residual
        let mut cstep = vec![0.0; m]; // C * step
        let mut cdot = 0.0; // dot product from C.Apply
        let timer = Instant::now();

        self.errflag = 0;
        self.iter = 0;
        self.time = 0.0;
        let maxiter = if maxiter < 0 { m as Int + 100 } else { maxiter };

        if infnorm(lhs) == 0.0 {
            residual.copy_from_slice(rhs); // saves a matrix-vector op
        } else {
            c.apply(lhs, &mut residual, None);
            for i in 0..m {
                residual[i] = rhs[i] - residual[i];
            }
        }
        c.apply(&residual, &mut cresidual, Some(&mut cdot));
        step.copy_from_slice(&residual);
        cstep.copy_from_slice(&cresidual);

        loop {
            let resnorm = residual_norm(resscale, &residual);
            if resnorm <= tol {
                break;
            }
            if self.iter == maxiter {
                self.control.debug_out(
                    3,
                    &format!(
                        " CR method not converged in {} iterations. residual = {}, tolerance = {}\n",
                        maxiter,
                        sci2(resnorm),
                        sci2(tol)
                    ),
                );
                self.errflag = ERROR_CR_ITER_LIMIT;
                break;
            }
            if cdot <= 0.0 {
                self.errflag = ERROR_CR_MATRIX_NOT_POSDEF;
                break;
            }

            let denom = dot_fused(&cstep, &cstep);
            let alpha = cdot / denom;
            if !alpha.is_finite() {
                self.errflag = ERROR_CR_INF_OR_NAN;
                break;
            }
            axpy(lhs, alpha, &step);
            axpy(&mut residual, -alpha, &cstep);
            let mut cdotnew = 0.0;
            c.apply(&residual, &mut cresidual, Some(&mut cdotnew));

            let beta = cdotnew / cdot;
            xpby(&mut step, &residual, beta);
            xpby(&mut cstep, &cresidual, beta);
            cdot = cdotnew;

            self.iter += 1;
            self.errflag = self.control.interrupt_check(-1);
            if self.errflag != 0 {
                break;
            }
        }
        self.time = timer.elapsed().as_secs_f64();
    }

    /// Preconditioned CR with preconditioner P.
    pub fn solve_precond(
        &mut self,
        c: &mut dyn LinearOperator,
        p: &mut dyn LinearOperator,
        rhs: &[f64],
        tol: f64,
        resscale: Option<&[f64]>,
        maxiter: Int,
        lhs: &mut [f64],
    ) {
        let m = rhs.len();
        let mut residual = vec![0.0; m]; // rhs - C*lhs
        let mut sresidual = vec![0.0; m]; // preconditioned residual
        let mut step = vec![0.0; m]; // update to lhs
        let mut csresidual = vec![0.0; m]; // C * sresidual
        let mut cstep = vec![0.0; m]; // C * step
        let mut cdot = 0.0; // dot product from C.Apply
        let timer = Instant::now();

        let mut resnorm_precond_system = 0.0;

        self.errflag = 0;
        self.iter = 0;
        self.time = 0.0;
        let maxiter = if maxiter < 0 { m as Int + 100 } else { maxiter };

        if infnorm(lhs) == 0.0 {
            residual.copy_from_slice(rhs); // saves a matrix-vector op
        } else {
            c.apply(lhs, &mut residual, None);
            for i in 0..m {
                residual[i] = rhs[i] - residual[i];
            }
        }
        p.apply(&residual, &mut sresidual, Some(&mut resnorm_precond_system));
        c.apply(&sresidual, &mut csresidual, Some(&mut cdot));
        step.copy_from_slice(&sresidual);
        cstep.copy_from_slice(&csresidual);

        loop {
            let resnorm = residual_norm(resscale, &residual);
            if resnorm <= tol {
                break;
            }
            if self.iter == maxiter {
                self.control.debug_out(
                    3,
                    &format!(
                        " PCR method not converged in {} iterations. residual = {}, tolerance = {}\n",
                        maxiter,
                        sci2(resnorm),
                        sci2(tol)
                    ),
                );
                self.errflag = ERROR_CR_ITER_LIMIT;
                break;
            }
            if cdot <= 0.0 {
                self.control.debug_out(
                    3,
                    &format!(
                        " matrix in PCR method not posdef. cdot = {}, infnorm(sresidual) = {}, infnorm(residual) = {}\n",
                        sci2(cdot),
                        sci2(infnorm(&sresidual)),
                        sci2(infnorm(&residual))
                    ),
                );
                self.errflag = ERROR_CR_MATRIX_NOT_POSDEF;
                break;
            }

            let mut cdotnew = 0.0;
            {
                // csresidual is used as workspace for P*Cstep
                let mut pdot = 0.0;
                p.apply(&cstep, &mut csresidual, Some(&mut pdot));
                if pdot <= 0.0 {
                    self.errflag = ERROR_CR_PRECOND_NOT_POSDEF;
                    break;
                }
                let alpha = cdot / pdot;
                if !alpha.is_finite() {
                    self.errflag = ERROR_CR_INF_OR_NAN;
                    break;
                }
                axpy(lhs, alpha, &step);
                axpy(&mut residual, -alpha, &cstep);
                axpy(&mut sresidual, -alpha, &csresidual);
                c.apply(&sresidual, &mut csresidual, Some(&mut cdotnew));
            }

            let beta = cdotnew / cdot;
            xpby(&mut step, &sresidual, beta);
            xpby(&mut cstep, &csresidual, beta);
            cdot = cdotnew;

            self.iter += 1;
            if self.iter % 5 == 0 {
                let mut rsdot = 0.0;
                p.apply(&residual, &mut sresidual, Some(&mut rsdot));
                if rsdot >= resnorm_precond_system {
                    self.control.debug_out(
                        3,
                        &format!(
                            " resnorm_precond_system old = {}\n resnorm_precond_system new = {}\n",
                            sci2(resnorm_precond_system),
                            sci2(rsdot)
                        ),
                    );
                    self.errflag = ERROR_CR_NO_PROGRESS;
                    break;
                }
                resnorm_precond_system = rsdot;
            }

            self.errflag = self.control.interrupt_check(-1);
            if self.errflag != 0 {
                break;
            }
        }
        self.time = timer.elapsed().as_secs_f64();
    }
}

/// KKTSolver: solves the KKT system
///   [ G  AI' ] [x]   [a]
///   [ AI  0  ] [y] = [b]
/// with G the diagonal of the current iterate (identity without iterate).
pub trait KktSolver {
    fn factorize_impl(&mut self, iterate: Option<&mut Iterate>, info: &mut Info) -> LuResult<()>;
    fn solve_impl(&mut self, a: &[f64], b: &[f64], tol: f64, x: &mut [f64], y: &mut [f64], info: &mut Info) -> LuResult<()>;
    /// # iterations of the linear solver since the last Factorize()
    fn iter(&self) -> Int;
    fn basis_changes(&self) -> Int {
        0
    }
    fn basis(&self) -> Option<&Basis> {
        None
    }

    fn factorize(&mut self, iterate: Option<&mut Iterate>, info: &mut Info) -> LuResult<()> {
        let timer = Instant::now();
        let r = self.factorize_impl(iterate, info);
        info.time_kkt_factorize += timer.elapsed().as_secs_f64();
        r
    }

    fn solve(&mut self, a: &[f64], b: &[f64], tol: f64, x: &mut [f64], y: &mut [f64], info: &mut Info) -> LuResult<()> {
        let timer = Instant::now();
        let r = self.solve_impl(a, b, tol, x, y, info);
        info.time_kkt_solve += timer.elapsed().as_secs_f64();
        r
    }
}

/// KKTSolverDiag: normal equations AI*W*AI' with W = inverse(G), solved by
/// CR with a diagonal preconditioner.
pub struct KktSolverDiag<'a> {
    control: &'a Control,
    model: &'a Model,
    w: Vec<f64>,        // diagonal matrix in AI*W*AI'
    diagonal: Vec<f64>, // diagonal of the normal matrix (preconditioner)
    resscale: Vec<f64>, // residual scaling factors for CR termination test
    factorized: bool,
    maxiter: Int,
    iter: Int, // # CR iterations since last Factorize()
}

impl<'a> KktSolverDiag<'a> {
    pub fn new(control: &'a Control, model: &'a Model) -> Self {
        let m = model.rows();
        let n = model.cols();
        KktSolverDiag {
            control,
            model,
            w: vec![0.0; n + m],
            diagonal: vec![0.0; m],
            resscale: vec![0.0; m],
            factorized: false,
            maxiter: -1,
            iter: 0,
        }
    }

    pub fn set_maxiter(&mut self, maxiter: Int) {
        self.maxiter = maxiter;
    }
}

impl KktSolver for KktSolverDiag<'_> {
    fn factorize_impl(&mut self, iterate: Option<&mut Iterate>, _info: &mut Info) -> LuResult<()> {
        let m = self.model.rows();
        let n = self.model.cols();
        self.iter = 0;
        self.factorized = false;

        if let Some(pt) = iterate {
            let (xl, xu, zl, zu) = (pt.xl(), pt.xu(), pt.zl(), pt.zu());
            let mut regval = pt.mu();
            for j in 0..n + m {
                let g = zl[j] / xl[j] + zu[j] / xu[j];
                if g != 0.0 && g < regval {
                    regval = g;
                }
                self.w[j] = 1.0 / g; // infinity if g is zero
            }
            for j in 0..n + m {
                if self.w[j].is_infinite() {
                    self.w[j] = 1.0 / regval;
                }
            }
        } else {
            self.w.fill(1.0);
        }

        for i in 0..m {
            self.resscale[i] = 1.0 / self.w[n + i].sqrt();
        }
        diagonal_of_normal_matrix(self.model, Some(&self.w), &mut self.diagonal);
        self.factorized = true;
        Ok(())
    }

    fn solve_impl(&mut self, a: &[f64], b: &[f64], tol: f64, x: &mut [f64], y: &mut [f64], info: &mut Info) -> LuResult<()> {
        let m = self.model.rows();
        let n = self.model.cols();
        let ai = self.model.ai();

        let mut rhs: Vec<f64> = b.iter().map(|v| -v).collect();
        for j in 0..n + m {
            scatter_column(ai, j, self.w[j] * a[j], &mut rhs);
        }

        y.fill(0.0);
        let mut normal_matrix = NormalMatrix {
            model: self.model,
            w: Some(&self.w),
            time: 0.0,
        };
        let mut precond = DiagonalPrecond {
            diagonal: &self.diagonal,
            time: 0.0,
        };
        let mut cr = ConjugateResiduals::new(self.control);
        cr.solve_precond(
            &mut normal_matrix,
            &mut precond,
            &rhs,
            tol,
            Some(&self.resscale),
            self.maxiter,
            y,
        );
        info.errflag = cr.errflag;
        info.kktiter1 += cr.iter;
        info.time_cr1 += cr.time;
        info.time_cr1_aat += normal_matrix.time;
        info.time_cr1_pre += precond.time;
        self.iter += cr.iter;

        x[n..n + m].copy_from_slice(&b[..m]);
        for j in 0..n {
            let aty = dot_column(ai, j, y);
            x[j] = self.w[j] * (a[j] - aty);
            for p in ai.begin(j)..ai.end(j) {
                let i = n + ai.index(p);
                x[i] = (-x[j]).mul_add(ai.value(p), x[i]);
            }
        }
        Ok(())
    }

    fn iter(&self) -> Int {
        self.iter
    }
}

/// KKTSolverBasis: normal equations preconditioned with a basis that is
/// maintained by the maxvolume algorithm (Schork, "Basis Preconditioning in
/// Interior Point Methods", PhD thesis, 2018).
pub struct KktSolverBasis<'a> {
    control: &'a Control,
    basis: &'a mut Basis,
    splitted_normal_matrix: SplittedNormalMatrix,
    colscale: Vec<f64>, // interior point column scaling factors
    factorized: bool,
    maxiter: Int,
    iter: Int,
    basis_changes: Int,
}

const PIVOT_ZERO_TOL: f64 = 1e-7;

impl<'a> KktSolverBasis<'a> {
    pub fn new(control: &'a Control, basis: &'a mut Basis) -> Self {
        let m = basis.model().rows();
        let n = basis.model().cols();
        KktSolverBasis {
            control,
            basis,
            splitted_normal_matrix: SplittedNormalMatrix::new(m),
            colscale: vec![0.0; n + m],
            factorized: false,
            maxiter: -1,
            iter: 0,
            basis_changes: 0,
        }
    }

    /// Pivots basic variables close to a bound out of the basis, or makes
    /// them "implied" at the bound.
    fn drop_primal(&mut self, iterate: &mut Iterate, info: &mut Info) -> LuResult<()> {
        let m = self.basis.model().rows();
        let n = self.basis.model().cols();
        let mut btran = IndexedVector::new(m);
        let mut row = IndexedVector::new(n + m);
        let drop_primal = self.control.ipm_drop_primal();
        const VOLUME_TOL: f64 = 2.0;
        info.errflag = 0;

        let mut candidates = Vec::new();
        for p in 0..m {
            let jb = self.basis.at(p);
            if self.basis.status_of(jb) != BasicStatus::Basic {
                // ignore free variables
                continue;
            }
            // choose which bound is nearer
            let (xj, zj) = if iterate.xl()[jb] <= iterate.xu()[jb] {
                (iterate.xl()[jb], iterate.zl()[jb])
            } else {
                (iterate.xu()[jb], iterate.zu()[jb])
            };
            if xj < 0.01 * zj && xj <= drop_primal {
                candidates.push(jb);
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }

        // Maintain a copy of the inverse scaling factors of basic variables
        // for faster access.
        let mut invscale_basic: Vec<f64> = (0..m).map(|p| 1.0 / self.colscale[self.basis.at(p)]).collect();

        while let Some(&jb) = candidates.last() {
            let p = self.basis.position_of(jb) as usize;
            // Pivot jb out of the basis if the volume increases sufficiently.
            let s = invscale_basic[p];
            self.basis.tableau_row(jb, &mut btran, &mut row, true)?;
            let mut jmax: Int = -1;
            let mut vmax = VOLUME_TOL;
            let colscale = &self.colscale;
            row.for_each_nonzero(|j, pivot| {
                let pivot = pivot.abs();
                if pivot > PIVOT_ZERO_TOL {
                    let v = pivot * colscale[j] * s;
                    if v > vmax {
                        vmax = v;
                        jmax = j as Int;
                    }
                }
            });
            if jmax >= 0 {
                // Pivot jb out of the basis.
                let jmax = jmax as usize;
                let pivot = row[jmax];
                if pivot.abs() < 1e-3 {
                    self.control.debug_out(
                        3,
                        &format!(" |pivot| = {} (primal basic variable close to bound)\n", sci2(pivot.abs())),
                    );
                }
                let (err, exchanged) = self.basis.exchange_if_stable(jb, jmax, pivot, 1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                invscale_basic[p] = 1.0 / self.colscale[jmax];
                info.updates_ipm += 1;
                self.basis_changes += 1;
            } else {
                // Make variable jb "implied" at a bound.
                if iterate.zl()[jb] / iterate.xl()[jb] > iterate.zu()[jb] / iterate.xu()[jb] {
                    iterate.make_implied_lb(jb);
                } else {
                    iterate.make_implied_ub(jb);
                }
                self.basis.free_basic_variable(jb);
                invscale_basic[p] = 0.0;
                self.colscale[jb] = f64::INFINITY;
                info.primal_dropped += 1;
            }
            candidates.pop();
        }
        Ok(())
    }

    /// Pivots nonbasic variables with dual close to zero into the basis, or
    /// fixes them at their current value.
    fn drop_dual(&mut self, iterate: &mut Iterate, info: &mut Info) -> LuResult<()> {
        let m = self.basis.model().rows();
        let n = self.basis.model().cols();
        let mut ftran = IndexedVector::new(m);
        let drop_dual = self.control.ipm_drop_dual();
        const VOLUME_TOL: f64 = 2.0;
        info.errflag = 0;

        let mut candidates = Vec::new();
        for jn in 0..n + m {
            if self.basis.status_of(jn) != BasicStatus::Nonbasic {
                continue;
            }
            // choose larger dual variable
            let (xj, zj) = if iterate.zl()[jn] >= iterate.zu()[jn] {
                (iterate.xl()[jn], iterate.zl()[jn])
            } else {
                (iterate.xu()[jn], iterate.zu()[jn])
            };
            if zj < 0.01 * xj && zj <= drop_dual {
                candidates.push(jn);
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }

        let mut invscale_basic: Vec<f64> = (0..m).map(|p| 1.0 / self.colscale[self.basis.at(p)]).collect();

        while let Some(&jn) = candidates.last() {
            // Pivot jn into the basis if volume increases sufficiently.
            let s = self.colscale[jn];
            self.basis.solve_for_update(jn, Some(&mut ftran))?;
            let mut pmax: Int = -1;
            let mut vmax = VOLUME_TOL;
            ftran.for_each_nonzero(|p, pivot| {
                let pivot = pivot.abs();
                if pivot > PIVOT_ZERO_TOL {
                    let v = pivot * invscale_basic[p] * s;
                    if v > vmax {
                        vmax = v;
                        pmax = p as Int;
                    }
                }
            });
            if pmax >= 0 {
                let pmax = pmax as usize;
                let pivot = ftran[pmax];
                if pivot.abs() < 1e-3 {
                    self.control.debug_out(
                        3,
                        &format!(" |pivot| = {} (dual nonbasic variable close to zero)\n", sci2(pivot.abs())),
                    );
                }
                let jb = self.basis.at(pmax);
                // Pivot jn into the basis.
                let (err, exchanged) = self.basis.exchange_if_stable(jb, jn, pivot, -1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    return Ok(());
                }
                if !exchanged {
                    continue;
                }
                invscale_basic[pmax] = 1.0 / self.colscale[jn];
                info.updates_ipm += 1;
                self.basis_changes += 1;
            } else {
                // Make variable jn "fixed" at its current value.
                iterate.make_fixed(jn);
                self.basis.fix_nonbasic_variable(jn);
                self.colscale[jn] = 0.0;
                info.dual_dropped += 1;
            }
            candidates.pop();
        }
        Ok(())
    }
}

impl KktSolver for KktSolverBasis<'_> {
    fn factorize_impl(&mut self, iterate: Option<&mut Iterate>, info: &mut Info) -> LuResult<()> {
        let iterate = iterate.expect("KKTSolverBasis requires an iterate");
        let m = self.basis.model().rows();
        let n = self.basis.model().cols();
        info.errflag = 0;
        self.factorized = false;
        self.iter = 0;
        self.basis_changes = 0;

        for j in 0..n + m {
            self.colscale[j] = iterate.scaling_factor(j);
        }

        // Remove degenerate variables unless the primal objective is smaller
        // than the dual objective (the model might be infeasible or
        // unbounded then).
        if iterate.pobjective() >= iterate.dobjective() {
            self.drop_primal(iterate, info)?;
            if info.errflag != 0 {
                return Ok(());
            }
            self.drop_dual(iterate, info)?;
            if info.errflag != 0 {
                return Ok(());
            }
        }

        // Run maxvolume ("Russian algorithm").
        let mut maxvol = Maxvolume::new(self.control);
        info.errflag = if self.control.update_heuristic() == 0 {
            maxvol.run_sequential(Some(&self.colscale), self.basis)?
        } else {
            maxvol.run_heuristic(Some(&self.colscale), self.basis)?
        };
        info.updates_ipm += maxvol.updates();
        info.time_maxvol += maxvol.time();
        self.basis_changes += maxvol.updates();
        if info.errflag != 0 {
            return Ok(());
        }

        // Refactorize and build preconditioned normal matrix.
        if !self.basis.factorization_is_fresh() {
            info.errflag = self.basis.factorize()?;
            if info.errflag != 0 {
                return Ok(());
            }
        }
        self.splitted_normal_matrix.prepare(self.basis, &self.colscale)?;
        self.factorized = true;
        Ok(())
    }

    /// Reduces the KKT system to preconditioned normal equations, taking
    /// free (basic) variables into account [Schork, Section 6.4].
    fn solve_impl(&mut self, a: &[f64], b: &[f64], tol: f64, x: &mut [f64], y: &mut [f64], info: &mut Info) -> LuResult<()> {
        let basis = &*self.basis;
        let model = basis.model();
        let m = model.rows();
        let n = model.cols();
        let ai = model.ai();
        let colscale = &self.colscale;
        let mut rhs = vec![0.0; m]; // unpermuted right-hand side
        let mut work = vec![0.0; m];
        info.errflag = 0;

        // Compute work = inverse(B')*v, where v[p] = a[basis[p]] if variable
        // basis[p] is free, and v[p] = 0 otherwise.
        let mut num_free = 0;
        for p in 0..m {
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::BasicFree {
                work[p] = a[j];
                num_free += 1;
            }
        }
        if num_free > 0 {
            basis.solve_dense_inplace(&mut work, b'T')?;
        }

        // Compute rhs = inverse(B)*(N*D2[nonbasic]*(a[nonbasic]-N'*work)).
        if num_free > 0 {
            for j in 0..n + m {
                if basis.status_of(j) == BasicStatus::Nonbasic {
                    let d2 = colscale[j] * colscale[j];
                    let mut alpha = a[j] - dot_column(ai, j, &work);
                    alpha *= d2;
                    scatter_column(ai, j, alpha, &mut rhs);
                }
            }
        } else {
            for j in 0..n + m {
                if basis.status_of(j) == BasicStatus::Nonbasic {
                    let d2 = colscale[j] * colscale[j];
                    let alpha = d2 * a[j];
                    scatter_column(ai, j, alpha, &mut rhs);
                }
            }
        }
        basis.solve_dense_inplace(&mut rhs, b'N')?;

        // Compute work = inverse(B)*b.
        basis.solve_dense(b, &mut work, b'N')?;

        // Build rhs[p] = (rhs[p]-work[p])/D[j] + D[j]*a[j], where j =
        // basis[p] is not a free variable, and rhs[p] = 0 otherwise.
        for p in 0..m {
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::Basic {
                let d = colscale[j];
                rhs[p] = a[j].mul_add(d, (rhs[p] - work[p]) / d);
            } else {
                rhs[p] = 0.0;
            }
        }

        // Build permuted rhs in work.
        let colperm = &self.splitted_normal_matrix.colperm;
        for k in 0..m {
            work[k] = rhs[colperm[k] as usize];
        }

        // Solve normal equations.
        self.splitted_normal_matrix.reset_time();
        let mut lhs = rhs; // don't need rhs any more
        lhs.fill(0.0);
        let mut cr = ConjugateResiduals::new(self.control);
        cr.solve(&mut self.splitted_normal_matrix, &work, tol, None, self.maxiter, &mut lhs);
        info.errflag = cr.errflag;
        info.kktiter2 += cr.iter;
        info.time_cr2 += cr.time;
        info.time_cr2_nnt += self.splitted_normal_matrix.time_nnt;
        info.time_cr2_b += self.splitted_normal_matrix.time_b;
        info.time_cr2_bt += self.splitted_normal_matrix.time_bt;
        self.iter += cr.iter;

        // Permute back solution to normal equations.
        let colperm = &self.splitted_normal_matrix.colperm;
        for k in 0..m {
            y[colperm[k] as usize] = lhs[k];
        }

        // Recover dual solution to KKT system.
        for p in 0..m {
            let j = basis.at(p);
            if basis.status_of(j) == BasicStatus::Basic {
                y[p] /= colscale[j];
            } else {
                y[p] = a[j]; // slot in solution to free basic variable
            }
        }
        basis.solve_dense_inplace(y, b'T')?;

        // Compute x[nonbasic] and work = b - N*x[nonbasic].
        work.copy_from_slice(&b[..m]);
        for j in 0..n + m {
            let mut xj = 0.0;
            if basis.status_of(j) == BasicStatus::Nonbasic {
                xj = a[j] - dot_column(ai, j, y);
                xj *= colscale[j] * colscale[j];
                scatter_column(ai, j, -xj, &mut work);
            }
            x[j] = xj;
        }

        // Compute x[basic].
        basis.solve_dense_inplace(&mut work, b'N')?;
        for p in 0..m {
            x[basis.at(p)] = work[p];
        }
        Ok(())
    }

    fn iter(&self) -> Int {
        self.iter
    }
    fn basis_changes(&self) -> Int {
        self.basis_changes
    }
    fn basis(&self) -> Option<&Basis> {
        Some(self.basis)
    }
}
