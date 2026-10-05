//! CholeskyFactor (qpsolver/factor.hpp): the dense factor L (row-major,
//! leading dimension `k_max`) of the reduced Hessian Z'QZ, kept up to date
//! as the null space grows (expand) and shrinks (reduce).

use super::basis::Basis;
use super::vector::{fma, MatrixBase, QpVector};
use super::SolverStatus;

pub struct CholeskyFactor {
    uptodate: bool,
    numberofreduces: usize,
    current_k: usize,
    k_max: usize,
    /// Row-major with leading dimension k_max (see resize)
    l: Vec<f64>,
}

impl CholeskyFactor {
    pub fn new(num_var: usize, num_inactive: usize) -> Self {
        let k_max = ((num_var as f64 / 16.0).ceil() as usize).min(1000).max(num_inactive);
        CholeskyFactor { uptodate: false, numberofreduces: 0, current_k: 0, k_max, l: vec![0.0; k_max * k_max] }
    }

    /// L.clear(); L.resize(new_k_max^2) and copy back. `l` models the
    /// whole capacity of the C++ std::vector (libc++ growth), since the
    /// factor writes past its size once the null space has been empty at
    /// a recompute (k_max = 0): such writes land in the old allocation.
    fn resize(&mut self, new_k_max: usize) {
        let old = self.l[..self.k_max * self.k_max].to_vec();
        let new_size = new_k_max * new_k_max;
        if new_size > self.l.len() {
            // (C++ leaves the new capacity past new_size uninitialized)
            self.l = vec![0.0; new_size.max(2 * self.l.len())];
        } else {
            self.l[..new_size].fill(0.0);
        }
        let m = new_k_max.min(self.k_max);
        for i in 0..m {
            for j in 0..m {
                self.l[i * new_k_max + j] = old[i * self.k_max + j];
            }
        }
        self.k_max = new_k_max;
    }

    pub fn recompute(&mut self, q: &MatrixBase, num_var: usize, basis: &mut Basis) -> SolverStatus {
        let dim_ns = basis.inactive().len();
        self.numberofreduces = 0;
        let mut orig = vec![vec![0.0; dim_ns]; dim_ns];
        self.resize(dim_ns);

        let mut temp = MatrixBase { num_row: dim_ns, ..Default::default() };
        let mut buffer_qcol = QpVector::new(num_var);
        let mut buffer_ztqi = QpVector::new(dim_ns);
        for i in 0..num_var {
            q.extractcol(i, &mut buffer_qcol);
            basis.ztprod(&buffer_qcol, &mut buffer_ztqi, false);
            temp.append(&buffer_ztqi);
        }
        let temp_t = temp.transpose();
        for (i, row) in orig.iter_mut().enumerate() {
            temp_t.extractcol(i, &mut buffer_qcol);
            basis.ztprod(&buffer_qcol, &mut buffer_ztqi, false);
            for &j in buffer_ztqi.nz() {
                row[j] = buffer_ztqi.value[j];
            }
        }

        let km = self.k_max;
        let l = &mut self.l;
        for col in 0..dim_ns {
            for row in 0..=col {
                let mut sum = 0.0;
                if row == col {
                    for &x in l[row..].iter().step_by(km).take(row) {
                        sum = fma(x, x, sum);
                    }
                    let d_value = orig[row][row] - sum;
                    if d_value <= 0.0 {
                        return SolverStatus::NotPositiveDefinite;
                    }
                    l[row * km + row] = d_value.sqrt();
                } else {
                    let cols = l[col..].iter().step_by(km).zip(l[row..].iter().step_by(km));
                    for (&a, &b) in cols.take(row) {
                        sum = fma(a, b, sum);
                    }
                    l[row * km + col] = (orig[col][row] - sum) / l[row * km + row];
                }
            }
        }
        self.current_k = dim_ns;
        self.uptodate = true;
        SolverStatus::Ok
    }

    pub fn expand(&mut self, yp: &QpVector, gyp: &QpVector, l: &mut QpVector) -> SolverStatus {
        if !self.uptodate {
            return SolverStatus::Ok;
        }
        let mu = gyp.dot(yp);
        l.resparsify();
        let lambda = mu - l.norm2();
        if lambda > 0.0 {
            if self.k_max <= self.current_k + 1 {
                self.resize(self.k_max * 2);
            }
            let (k, km) = (self.current_k, self.k_max);
            for i in 0..k {
                self.l[i * km + k] = l.value[i];
            }
            self.l[k * km + k] = lambda.sqrt();
            self.current_k += 1;
            SolverStatus::Ok
        } else {
            SolverStatus::NotPositiveDefinite
        }
    }

    pub fn solve_l(&mut self, q: &MatrixBase, num_var: usize, basis: &mut Basis, rhs: &mut QpVector) -> SolverStatus {
        if !self.uptodate {
            let status = self.recompute(q, num_var, basis);
            if status != SolverStatus::Ok {
                return status;
            }
        }
        if self.current_k != rhs.dim {
            return SolverStatus::Error;
        }
        let km = self.k_max;
        let x = &mut rhs.value;
        for r in 0..rhs.dim {
            let mut xr = x[r];
            if km == 0 {
                // (L aliased after an empty null space, see resize)
                for j in 0..r {
                    xr = fma(-x[j], self.l[r], xr);
                }
            } else {
                for (&xj, &lj) in x[..r].iter().zip(self.l[r..].iter().step_by(km)) {
                    xr = fma(-xj, lj, xr);
                }
            }
            x[r] = xr / self.l[r * km + r];
        }
        SolverStatus::Ok
    }

    /// Solve L' u = v
    pub fn solve_lt(&self, rhs: &mut QpVector) {
        let km = self.k_max;
        for i in (0..rhs.dim).rev() {
            let mut sum = 0.0;
            let row = &self.l[i * km + i + 1..i * km + rhs.dim];
            for (&x, &l) in rhs.value[i + 1..rhs.dim].iter().zip(row).rev() {
                sum = fma(x, l, sum);
            }
            rhs.value[i] = (rhs.value[i] - sum) / self.l[i * km + i];
        }
    }

    pub fn solve(&mut self, q: &MatrixBase, num_var: usize, basis: &mut Basis, rhs: &mut QpVector) -> SolverStatus {
        // has_negative_eigenvalue is always false
        if !self.uptodate || self.numberofreduces >= num_var / 2 {
            let status = self.recompute(q, num_var, basis);
            if status != SolverStatus::Ok {
                return status;
            }
        }
        let status = self.solve_l(q, num_var, basis, rhs);
        if status != SolverStatus::Ok {
            return status;
        }
        self.solve_lt(rhs);
        rhs.resparsify();
        SolverStatus::Ok
    }

    /// Givens rotation of rows i and j of m to zero m[j][i]
    fn eliminate(m: &mut [f64], i: usize, j: usize, kmax: usize, current_k: usize) {
        if m[j * kmax + i] == 0.0 {
            return;
        }
        let (mii, mji) = (m[i * kmax + i], m[j * kmax + i]);
        let z = fma(mii, mii, mji * mji).sqrt();
        let (cos_, sin_) = if z == 0.0 { (1.0, 0.0) } else { (mii / z, -mji / z) };
        if sin_ == 0.0 {
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN negates
            if !(cos_ > 0.0) {
                for k in 0..current_k {
                    m[i * kmax + k] = -m[i * kmax + k];
                    m[j * kmax + k] = -m[j * kmax + k];
                }
            }
        } else if cos_ == 0.0 {
            for k in 0..current_k {
                let a_ik = m[i * kmax + k];
                if sin_ > 0.0 {
                    m[i * kmax + k] = -m[j * kmax + k];
                    m[j * kmax + k] = a_ik;
                } else {
                    m[i * kmax + k] = m[j * kmax + k];
                    m[j * kmax + k] = -a_ik;
                }
            }
        } else if i.abs_diff(j) * kmax >= current_k {
            // Disjoint rows
            let (lo, hi) = (i.min(j) * kmax, i.max(j) * kmax);
            let (a, b) = m.split_at_mut(hi);
            let (lo_row, hi_row) = (&mut a[lo..lo + current_k], &mut b[..current_k]);
            let (ri, rj) = if i < j { (lo_row, hi_row) } else { (hi_row, lo_row) };
            for (x, y) in ri.iter_mut().zip(rj.iter_mut()) {
                let (a_ik, a_jk) = (*x, *y);
                *x = fma(cos_, a_ik, -(sin_ * a_jk));
                *y = fma(sin_, a_ik, cos_ * a_jk);
            }
        } else {
            for k in 0..current_k {
                let a_ik = m[i * kmax + k];
                let a_jk = m[j * kmax + k];
                m[i * kmax + k] = fma(cos_, a_ik, -(sin_ * a_jk));
                m[j * kmax + k] = fma(sin_, a_ik, cos_ * a_jk);
            }
        }
        m[j * kmax + i] = 0.0;
    }

    pub fn reduce(&mut self, buffer_d: &QpVector, maxabsd: usize, p_in_v: bool) {
        if self.current_k == 0 || !self.uptodate {
            return;
        }
        self.numberofreduces += 1;
        let (k, km, p) = (self.current_k, self.k_max, maxabsd);
        let l = &mut self.l;

        // Move row p to the bottom
        let row_p: Vec<f64> = l[p * km..p * km + k].to_vec();
        for row in p..k - 1 {
            l.copy_within((row + 1) * km..(row + 1) * km + k, row * km);
        }
        l[(k - 1) * km..(k - 1) * km + k].copy_from_slice(&row_p);

        // Move column p to the right in each row
        for row in 0..k {
            let p_entry = l[row * km + p];
            l.copy_within(row * km + p + 1..row * km + k, row * km + p);
            l[row * km + k - 1] = p_entry;
        }

        if k == 1 {
            self.current_k -= 1;
            return;
        }

        if !p_in_v {
            // Remove the nonzeros in the last column but the diagonal
            for r in (0..p).rev() {
                Self::eliminate(l, k - 1, r, km, k);
            }
            // New last row: old last row plus r * R[k-1][k-1]
            for &idx in buffer_d.nz() {
                if idx == maxabsd {
                    continue;
                }
                let at = if idx < maxabsd { idx } else { idx - 1 };
                let factor = -buffer_d.value[idx] / buffer_d.value[maxabsd];
                l[(k - 1) * km + at] = fma(factor, l[(k - 1) * km + k - 1], l[(k - 1) * km + at]);
            }
        }
        // Eliminate the last row
        for i in 0..k - 1 {
            Self::eliminate(l, i, k - 1, km, k);
        }
        self.current_k -= 1;
    }
}
