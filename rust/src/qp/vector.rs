//! QpVector (qpsolver/qpvector.hpp) and the column-wise matrices of
//! qpsolver/matrix.hpp. A QpVector is dense `value` with the list `index`
//! of its first `num_nz` entries; `dim` may be less than `value.len()`.

/// `s + a * b` as one fused multiply-add (clang contracts these)
#[inline(always)]
pub(crate) fn fma(a: f64, b: f64, s: f64) -> f64 {
    a.mul_add(b, s)
}

#[derive(Clone, Debug)]
pub struct QpVector {
    pub num_nz: usize,
    pub dim: usize,
    pub index: Vec<usize>,
    pub value: Vec<f64>,
}

impl QpVector {
    pub fn new(dim: usize) -> Self {
        QpVector { num_nz: 0, dim, index: vec![0; dim], value: vec![0.0; dim] }
    }

    /// A vector with the nonzeros of `value`, in order
    pub fn from_dense(value: &[f64]) -> Self {
        let mut v = QpVector { num_nz: 0, dim: value.len(), index: vec![0; value.len()], value: value.to_vec() };
        v.resparsify();
        v
    }

    pub fn nz(&self) -> &[usize] {
        &self.index[..self.num_nz]
    }

    pub fn reset(&mut self) {
        for i in 0..self.num_nz {
            self.value[self.index[i]] = 0.0;
            self.index[i] = 0;
        }
        self.num_nz = 0;
    }

    pub fn repopulate(&mut self, other: &QpVector) -> &mut Self {
        self.reset();
        for i in 0..other.num_nz {
            self.index[i] = other.index[i];
            self.value[self.index[i]] = other.value[self.index[i]];
        }
        self.num_nz = other.num_nz;
        self
    }

    /// QpVector::unit into `self` (dim unchanged)
    pub fn set_unit(&mut self, u: usize) {
        self.reset();
        self.index[0] = u;
        self.value[u] = 1.0;
        self.num_nz = 1;
    }

    pub fn unit(dim: usize, u: usize) -> Self {
        let mut v = QpVector::new(dim);
        v.set_unit(u);
        v
    }

    /// Sum of squares over the nonzeros
    pub fn norm2(&self) -> f64 {
        self.dot(self)
    }

    pub fn sanitize(&mut self, threshold: f64) {
        let mut new_idx = 0;
        for i in 0..self.num_nz {
            let j = self.index[i];
            if self.value[j].abs() > threshold {
                self.index[new_idx] = j;
                new_idx += 1;
            } else {
                self.value[j] = 0.0;
                self.index[i] = 0;
            }
        }
        self.num_nz = new_idx;
    }

    pub fn resparsify(&mut self) {
        self.num_nz = 0;
        for i in 0..self.dim {
            if self.value[i] != 0.0 {
                self.index[self.num_nz] = i;
                self.num_nz += 1;
            }
        }
    }

    pub fn scale(&mut self, a: f64) -> &mut Self {
        for i in 0..self.num_nz {
            self.value[self.index[i]] *= a;
        }
        self
    }

    /// self = a * self + b * x
    pub fn saxpy2(&mut self, a: f64, b: f64, x: &QpVector) -> &mut Self {
        self.scale(a);
        self.saxpy(b, x)
    }

    /// self += a * x
    pub fn saxpy(&mut self, a: f64, x: &QpVector) -> &mut Self {
        self.sanitize(0.0);
        for &j in x.nz() {
            if self.value[j] == 0.0 {
                self.index[self.num_nz] = j;
                self.num_nz += 1;
            }
            self.value[j] = fma(a, x.value[j], self.value[j]);
        }
        self.resparsify();
        self
    }

    /// -self (operator-)
    pub fn neg(&self) -> QpVector {
        let mut r = QpVector::new(self.dim);
        for i in 0..self.num_nz {
            let j = self.index[i];
            r.index[i] = j;
            r.value[j] = -self.value[j];
        }
        r.num_nz = self.num_nz;
        r
    }

    /// Dot product over the nonzeros of `self`, fused throughout as in
    /// most of its compiled copies
    pub fn dot(&self, other: &QpVector) -> f64 {
        let mut s = 0.0;
        for &j in self.nz() {
            s = fma(self.value[j], other.value[j], s);
        }
        s
    }

    /// dot as clang compiles it in SteepestEdgePricing and
    /// Instance::objval: interleaved by 4, so the products of the first
    /// len/4*4 terms are rounded and the remainder fused
    pub fn dot_split4(&self, other: &QpVector) -> f64 {
        let index = self.nz();
        let nb = index.len() & !3;
        let mut s = 0.0;
        for &j in &index[..nb] {
            s += self.value[j] * other.value[j];
        }
        for &j in &index[nb..] {
            s = fma(self.value[j], other.value[j], s);
        }
        s
    }

    /// operator+=
    pub fn add_assign(&mut self, other: &QpVector) {
        for &j in other.nz() {
            self.value[j] += other.value[j];
        }
        self.resparsify();
    }
}

/// MatrixBase: column-wise
#[derive(Clone, Debug, Default)]
pub struct MatrixBase {
    pub num_row: usize,
    pub num_col: usize,
    pub start: Vec<usize>,
    pub index: Vec<usize>,
    pub value: Vec<f64>,
}

impl MatrixBase {
    /// target = this * other
    pub fn mat_vec(&self, other: &QpVector, target: &mut QpVector) {
        target.reset();
        for &col in other.nz() {
            for idx in self.start[col]..self.start[col + 1] {
                let row = self.index[idx];
                target.value[row] = fma(self.value[idx], other.value[col], target.value[row]);
            }
        }
        target.resparsify();
    }

    /// target = other^T * this (vec_mat_1)
    pub fn vec_mat(&self, other: &QpVector, target: &mut QpVector) {
        target.reset();
        for col in 0..self.num_col {
            let mut dot = 0.0;
            for j in self.start[col]..self.start[col + 1] {
                dot = fma(other.value[self.index[j]], self.value[j], dot);
            }
            target.value[col] = dot;
        }
        target.resparsify();
    }

    pub fn extractcol(&self, col: usize, target: &mut QpVector) {
        debug_assert_eq!(target.dim, self.num_row);
        target.reset();
        if col >= self.num_col {
            target.index[0] = col - self.num_col;
            target.value[col - self.num_col] = 1.0;
            target.num_nz = 1;
        } else {
            let (s, e) = (self.start[col], self.start[col + 1]);
            for i in 0..e - s {
                target.index[i] = self.index[s + i];
                target.value[target.index[i]] = self.value[s + i];
            }
            target.num_nz = e - s;
        }
    }

    /// The transpose (Matrix::transpose)
    pub fn transpose(&self) -> MatrixBase {
        let mut count = vec![0usize; self.num_row + 1];
        for col in 0..self.num_col {
            for &r in &self.index[self.start[col]..self.start[col + 1]] {
                count[r + 1] += 1;
            }
        }
        for r in 0..self.num_row {
            count[r + 1] += count[r];
        }
        let start = count.clone();
        let nnz = start[self.num_row];
        let mut index = vec![0; nnz];
        let mut value = vec![0.0; nnz];
        for col in 0..self.num_col {
            for e in self.start[col]..self.start[col + 1] {
                let r = self.index[e];
                index[count[r]] = col;
                value[count[r]] = self.value[e];
                count[r] += 1;
            }
        }
        MatrixBase { num_row: self.num_col, num_col: self.num_row, start, index, value }
    }

    /// Matrix::append
    pub fn append(&mut self, vec: &QpVector) {
        if self.num_col == 0 && self.start.is_empty() {
            self.start.push(0);
        }
        for &j in vec.nz() {
            self.index.push(j);
            self.value.push(vec.value[j]);
        }
        self.start.push(self.start[self.num_col] + vec.num_nz);
        self.num_col += 1;
    }
}
