//! The CPU kernels of cupdlp_linalg.c, with the floating-point operations of
//! libhighs: the vector updates `y[i] += a * x[i]` are contracted (fused),
//! and the in-order reductions `d += x[i] * y[i]` are blocked by 4 (clang
//! splits the fused multiply-add in the vectorized main part, see
//! `ipx::utils::dot_blocked`). Every copy of `dot` and `nrm2` inlined into
//! cupdlp_solver.c, cupdlp_step.c and cupdlp_restart.c compiles this way.

use crate::ipx::utils::dot_blocked;

/// dot() and cupdlp_dot
pub(super) fn dot(x: &[f64], y: &[f64]) -> f64 {
    dot_blocked(x, y, 0.0, 4)
}

/// nrm2() and cupdlp_twoNorm
pub(super) fn nrm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// nrminf(), and infNorm of CupdlpWrapper.cpp (std::max with the same
/// result for the non-NaN entries)
pub(super) fn nrminf(x: &[f64]) -> f64 {
    let mut nrm = 0.0;
    for &xi in x {
        let tmp = xi.abs();
        if tmp > nrm {
            nrm = tmp;
        }
    }
    nrm
}

/// cupdlp_axpy: y += alpha * x (fused)
pub(super) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = alpha.mul_add(xi, *yi);
    }
}

/// cupdlp_scaleVector: x *= w
pub(super) fn scale(w: f64, x: &mut [f64]) {
    x.iter_mut().for_each(|xi| *xi *= w);
}

/// cupdlp_edot: x .*= y
pub(super) fn edot(x: &mut [f64], y: &[f64]) {
    x.iter_mut().zip(y).for_each(|(xi, &yi)| *xi *= yi);
}

/// cupdlp_ediv: x ./= y
pub(super) fn ediv(x: &mut [f64], y: &[f64]) {
    x.iter_mut().zip(y).for_each(|(xi, &yi)| *xi /= yi);
}

/// cupdlp_projub: x = min(x, ub) as `x < ub ? x : ub`
pub(super) fn proj_ub(x: &mut [f64], ub: &[f64]) {
    for (xi, &u) in x.iter_mut().zip(ub) {
        *xi = if *xi < u { *xi } else { u };
    }
}

/// cupdlp_projlb: x = max(x, lb) as `x > lb ? x : lb`
pub(super) fn proj_lb(x: &mut [f64], lb: &[f64]) {
    for (xi, &l) in x.iter_mut().zip(lb) {
        *xi = if *xi > l { *xi } else { l };
    }
}

/// cupdlp_projPos: `x > 0 ? x : 0`
pub(super) fn proj_pos(x: &mut [f64]) {
    x.iter_mut()
        .for_each(|xi| *xi = if *xi > 0.0 { *xi } else { 0.0 });
}

/// cupdlp_projNeg: `x < 0 ? x : 0`
pub(super) fn proj_neg(x: &mut [f64]) {
    x.iter_mut()
        .for_each(|xi| *xi = if *xi < 0.0 { *xi } else { 0.0 });
}

/// A compressed sparse matrix: CSC (columns as the outer index) or, as
/// its transpose, CSR
#[derive(Clone, Debug, Default)]
pub(super) struct Sparse {
    pub(super) start: Vec<usize>,
    pub(super) index: Vec<u32>,
    pub(super) value: Vec<f64>,
}

impl Sparse {
    /// Number of outer vectors
    pub(super) fn outer(&self) -> usize {
        self.start.len() - 1
    }

    /// AxCPU / ATyCPU: out = M x, scattering each outer vector times x[j]
    /// (ScatterCol / ScatterRow, fused)
    pub(super) fn mul(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (j, &xj) in x[..self.outer()].iter().enumerate() {
            let (b, e) = (self.start[j], self.start[j + 1]);
            for (&i, &v) in self.index[b..e].iter().zip(&self.value[b..e]) {
                let t = &mut out[i as usize];
                *t = v.mul_add(xj, *t);
            }
        }
    }

    /// cupdlp_dcs_transpose, for an inner dimension of n
    pub(super) fn transpose(&self, n: usize) -> Sparse {
        let mut count = vec![0usize; n];
        for &i in &self.index {
            count[i as usize] += 1;
        }
        let mut start = vec![0usize; n + 1];
        for i in 0..n {
            start[i + 1] = start[i] + count[i];
        }
        let mut next = start[..n].to_vec();
        let nnz = self.index.len();
        let (mut index, mut value) = (vec![0u32; nnz], vec![0.0; nnz]);
        for j in 0..self.outer() {
            for p in self.start[j]..self.start[j + 1] {
                let q = &mut next[self.index[p] as usize];
                index[*q] = j as u32;
                value[*q] = self.value[p];
                *q += 1;
            }
        }
        Sparse {
            start,
            index,
            value,
        }
    }
}
