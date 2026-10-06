//! lu_condest.c, lu_matrix_norm.c, lu_residual_test.c: condition estimates,
//! norms and the stability test of a fresh factorization.

use crate::util::fma::ClangFma;

use super::{Int, Lu};

/// lu_condest: given m-by-m U such that U[perm,perm] is triangular, return
/// (estimate of the 1-norm condition number, 1-norm of U, estimated 1-norm
/// of U^{-1}). Arguments as for normest().
#[allow(clippy::too_many_arguments)]
pub(crate) fn condest(
    m: Int,
    ubegin: &[Int],
    ui: &[Int],
    ux: &[f64],
    pivot: Option<&[f64]>,
    perm: &[Int],
    upper: bool,
    work: &mut [f64],
) -> (f64, f64, f64) {
    // compute 1-norm of U
    let mut unorm: f64 = 0.0;
    for j in 0..m as usize {
        let mut colsum = match pivot {
            Some(pivot) => pivot[j].abs(),
            None => 1.0,
        };
        let mut p = ubegin[j] as usize;
        while ui[p] >= 0 {
            colsum += ux[p].abs();
            p += 1;
        }
        unorm = unorm.max(colsum);
    }

    // estimate 1-norm of U^{-1}
    let uinvnorm = normest(m, ubegin, ui, ux, pivot, perm, upper, work);
    (unorm * uinvnorm, unorm, uinvnorm)
}

/// lu_normest: given m-by-m U such that U[perm,perm] is triangular
/// (`upper` or lower), estimate the 1-norm of U^{-1} by computing
/// U'x = b, Uy = x, normest = max{norm(y)_1/norm(x)_1, norm(x)_inf}, where
/// the entries of b are +/-1 chosen dynamically to make x large (I. Duff,
/// A. Erisman, J. Reid, "Direct Methods for Sparse Matrices").
///
/// U is in compressed column format without pivots, columns terminated by
/// a negative index; `pivot` holds the pivots by column index (None: unit
/// pivots); `work` is size m workspace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normest(
    m: Int,
    ubegin: &[Int],
    ui: &[Int],
    ux: &[f64],
    pivot: Option<&[f64]>,
    perm: &[Int],
    upper: bool,
    work: &mut [f64],
) -> f64 {
    let m = m as usize;
    let mut x1norm: f64 = 0.0;
    let mut xinfnorm: f64 = 0.0;
    for t in 0..m {
        let k = if upper { t } else { m - 1 - t };
        let j = perm[k] as usize;
        let mut temp: f64 = 0.0;
        let mut p = ubegin[j] as usize;
        loop {
            let i = ui[p];
            if i < 0 {
                break;
            }
            temp = (-work[i as usize]).mul_add_c(ux[p], temp);
            p += 1;
        }
        temp += if temp >= 0.0 { 1.0 } else { -1.0 }; // choose b[i] = 1 or -1
        if let Some(pivot) = pivot {
            temp /= pivot[j];
        }
        work[j] = temp;
        x1norm += temp.abs();
        xinfnorm = xinfnorm.max(temp.abs());
    }

    let mut y1norm: f64 = 0.0;
    for t in 0..m {
        let k = if upper { m - 1 - t } else { t };
        let j = perm[k] as usize;
        if let Some(pivot) = pivot {
            work[j] /= pivot[j];
        }
        let temp = work[j];
        let mut p = ubegin[j] as usize;
        loop {
            let i = ui[p];
            if i < 0 {
                break;
            }
            work[i as usize] = (-temp).mul_add_c(ux[p], work[i as usize]);
            p += 1;
        }
        y1norm += temp.abs();
    }

    (y1norm / x1norm).max(xinfnorm)
}

fn onenorm(x: &[f64]) -> f64 {
    let mut d = 0.0;
    for &x in x {
        d += x.abs();
    }
    d
}

impl Lu<'_> {
    /// lu_matrix_norm: 1-norm and infinity-norm of the freshly factorized
    /// matrix; unit columns inserted by the factorization are handled
    /// implicitly
    fn matrix_norm(&mut self, bbegin: &[Int], bend: &[Int], bi: &[Int], bx: &[f64]) {
        let m = self.m as usize;
        let rank = self.rank as usize;
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let rowsum = &mut *self.work1;
        debug_assert!(self.nupdate == 0);

        rowsum.fill(0.0);
        let mut onenorm: f64 = 0.0;
        let mut infnorm: f64 = 0.0;
        for k in 0..rank {
            let jpivot = pivotcol[k] as usize;
            let mut colsum = 0.0;
            for pos in bbegin[jpivot] as usize..bend[jpivot] as usize {
                colsum += bx[pos].abs();
                rowsum[bi[pos] as usize] += bx[pos].abs();
            }
            onenorm = onenorm.max(colsum);
        }
        for k in rank..m {
            rowsum[pivotrow[k] as usize] += 1.0;
            onenorm = onenorm.max(1.0);
        }
        for &r in rowsum.iter() {
            infnorm = infnorm.max(r);
        }

        self.onenorm = onenorm;
        self.infnorm = infnorm;
    }

    /// lu_residual_test: stability test of a fresh LU factorization based on
    /// the relative residual
    pub(crate) fn residual_test(&mut self, bbegin: &[Int], bend: &[Int], bi: &[Int], bx: &[f64]) {
        let m = self.m as usize;
        let rank = self.rank as usize;
        let p = &self.wblink[m + 1..];
        let pivotcol = &*self.colcount_flink;
        let pivotrow = &*self.colcount_blink;
        let lbegin_p = &*self.lbegin_p;
        let ltbegin_p = &self.wflink[m + 1..];
        let ubegin = &*self.ubegin;
        let row_pivot = &*self.row_pivot;
        let lindex = &*self.lindex;
        let lvalue = &*self.lvalue;
        let uindex = &*self.uindex;
        let uvalue = &*self.uvalue;
        let rhs = &mut *self.work0;
        let lhs = &mut *self.work1;
        debug_assert!(self.nupdate == 0);

        // Residual Test with Forward System

        // Compute lhs = L\rhs and build rhs on-the-fly.
        for k in 0..m {
            let mut d: f64 = 0.0;
            let mut pos = ltbegin_p[k] as usize;
            while lindex[pos] >= 0 {
                d = lhs[lindex[pos] as usize].mul_add_c(lvalue[pos], d);
                pos += 1;
            }
            let ipivot = p[k] as usize;
            rhs[ipivot] = if d <= 0.0 { 1.0 } else { -1.0 };
            lhs[ipivot] = rhs[ipivot] - d;
        }

        // Overwrite lhs by U\lhs.
        for k in (0..m).rev() {
            let ipivot = pivotrow[k] as usize;
            lhs[ipivot] /= row_pivot[ipivot];
            let d = lhs[ipivot];
            let mut pos = ubegin[ipivot] as usize;
            while uindex[pos] >= 0 {
                let i = uindex[pos] as usize;
                lhs[i] = (-d).mul_add_c(uvalue[pos], lhs[i]);
                pos += 1;
            }
        }

        // Overwrite rhs by the residual rhs-B*lhs.
        for k in 0..rank {
            let ipivot = pivotrow[k] as usize;
            let jpivot = pivotcol[k] as usize;
            let d = lhs[ipivot];
            for pos in bbegin[jpivot] as usize..bend[jpivot] as usize {
                let i = bi[pos] as usize;
                rhs[i] = (-d).mul_add_c(bx[pos], rhs[i]);
            }
        }
        for k in rank..m {
            let ipivot = pivotrow[k] as usize;
            rhs[ipivot] -= lhs[ipivot];
        }
        let norm_ftran = onenorm(lhs);
        let norm_ftran_res = onenorm(rhs);

        // Residual Test with Backward System

        // Compute lhs = U'\rhs and build rhs on-the-fly.
        for k in 0..m {
            let ipivot = pivotrow[k] as usize;
            let mut d: f64 = 0.0;
            let mut pos = ubegin[ipivot] as usize;
            while uindex[pos] >= 0 {
                d = lhs[uindex[pos] as usize].mul_add_c(uvalue[pos], d);
                pos += 1;
            }
            rhs[ipivot] = if d <= 0.0 { 1.0 } else { -1.0 };
            lhs[ipivot] = (rhs[ipivot] - d) / row_pivot[ipivot];
        }

        // Overwrite lhs by L'\lhs.
        for k in (0..m).rev() {
            let mut d: f64 = 0.0;
            let mut pos = lbegin_p[k] as usize;
            while lindex[pos] >= 0 {
                d = lhs[lindex[pos] as usize].mul_add_c(lvalue[pos], d);
                pos += 1;
            }
            lhs[p[k] as usize] -= d;
        }

        // Overwrite rhs by the residual rhs-B'*lhs.
        for k in 0..rank {
            let ipivot = pivotrow[k] as usize;
            let jpivot = pivotcol[k] as usize;
            let mut d: f64 = 0.0;
            for pos in bbegin[jpivot] as usize..bend[jpivot] as usize {
                d = lhs[bi[pos] as usize].mul_add_c(bx[pos], d);
            }
            rhs[ipivot] -= d;
        }
        for k in rank..m {
            let ipivot = pivotrow[k] as usize;
            rhs[ipivot] -= lhs[ipivot];
        }
        let norm_btran = onenorm(lhs);
        let norm_btran_res = onenorm(rhs);

        // Finalize

        self.matrix_norm(bbegin, bend, bi, bx);
        let mf = m as f64;
        self.residual_test = (norm_ftran_res / self.onenorm.mul_add_c(norm_ftran, mf))
            .max(norm_btran_res / self.infnorm.mul_add_c(norm_btran, mf));

        // reset workspace
        self.work0.fill(0.0);
    }
}
