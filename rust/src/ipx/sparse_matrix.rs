//! sparse_matrix.h/.cc: sparse matrix in CSC format and kernels on it.

use super::{cmax, utils, Int};

/// Sparse matrix in CSC format, with a queue for building a new column.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    nrow: Int,
    pub(crate) colptr: Vec<Int>,
    pub(crate) rowidx: Vec<Int>,
    pub(crate) values: Vec<f64>,
    rowidx_queue: Vec<Int>,
    values_queue: Vec<f64>,
}

impl Default for SparseMatrix {
    /// The empty 0-by-0 matrix
    fn default() -> Self {
        SparseMatrix {
            nrow: 0,
            colptr: vec![0],
            rowidx: vec![],
            values: vec![],
            rowidx_queue: vec![],
            values_queue: vec![],
        }
    }
}

impl SparseMatrix {
    pub fn new(nrow: Int, ncol: Int) -> Self {
        let mut a = SparseMatrix::default();
        a.resize(nrow, ncol, 0);
        a
    }

    pub fn rows(&self) -> Int {
        self.nrow
    }
    pub fn cols(&self) -> Int {
        self.colptr.len() as Int - 1
    }
    pub fn entries(&self) -> Int {
        *self.colptr.last().unwrap()
    }
    /// # entries in column j
    pub fn col_entries(&self, j: usize) -> Int {
        self.colptr[j + 1] - self.colptr[j]
    }
    pub fn capacity(&self) -> usize {
        self.rowidx.len()
    }

    #[inline]
    pub fn begin(&self, j: usize) -> usize {
        self.colptr[j] as usize
    }
    #[inline]
    pub fn end(&self, j: usize) -> usize {
        self.colptr[j + 1] as usize
    }
    #[inline]
    pub fn index(&self, p: usize) -> usize {
        self.rowidx[p] as usize
    }
    #[inline]
    pub fn value(&self, p: usize) -> f64 {
        self.values[p]
    }

    /// Increases capacity if necessary such that capacity() >= min_capacity.
    pub fn reserve(&mut self, min_capacity: usize) {
        if min_capacity > self.capacity() {
            self.rowidx.resize(min_capacity, 0);
            self.values.resize(min_capacity, 0.0);
        }
    }

    /// Changes matrix dimensions. Matrix becomes empty.
    pub fn resize(&mut self, nrow: Int, ncol: Int, min_capacity: Int) {
        debug_assert!(nrow >= 0 && ncol >= 0 && min_capacity >= 0);
        self.nrow = nrow;
        self.colptr.clear();
        self.colptr.resize(ncol as usize + 1, 0);
        self.rowidx.clear();
        self.rowidx.resize(min_capacity as usize, 0);
        self.values.clear();
        self.values.resize(min_capacity as usize, 0.0);
    }

    pub fn clear(&mut self) {
        self.resize(0, 0, 0);
    }

    /// Builds matrix from data in compressed column format, dropping zero
    /// entries. The row indices in the output matrix are sorted.
    pub fn load_from_arrays(
        &mut self,
        nrow: Int,
        ncol: Int,
        abegin: &[Int],
        aend: &[Int],
        ai: &[Int],
        ax: &[f64],
    ) {
        let ncol = ncol as usize;
        let mut nz = 0;
        for j in 0..ncol {
            nz += aend[j] - abegin[j];
        }
        self.resize(nrow, ncol as Int, nz);
        let mut put = 0;
        for j in 0..ncol {
            self.colptr[j] = put as Int;
            for p in abegin[j] as usize..aend[j] as usize {
                if ax[p] != 0.0 {
                    self.rowidx[put] = ai[p];
                    self.values[put] = ax[p];
                    put += 1;
                }
            }
        }
        self.colptr[ncol] = put as Int;
        self.sort_indices();
    }

    /// Stores the entries in each column in increasing order of index.
    pub fn sort_indices(&mut self) {
        if self.is_sorted() {
            return;
        }
        let mut work: Vec<(Int, f64)> = Vec::with_capacity(self.nrow as usize);
        for j in 0..self.cols() as usize {
            let (b, e) = (self.begin(j), self.end(j));
            work.clear();
            for p in b..e {
                work.push((self.rowidx[p], self.values[p]));
            }
            // indices within a column are distinct
            work.sort_unstable_by_key(|&(i, _)| i);
            for (k, p) in (b..e).enumerate() {
                self.rowidx[p] = work[k].0;
                self.values[p] = work[k].1;
            }
        }
    }

    fn is_sorted(&self) -> bool {
        for j in 0..self.cols() as usize {
            for p in self.begin(j) + 1..self.end(j) {
                if self.rowidx[p - 1] > self.rowidx[p] {
                    return false;
                }
            }
        }
        true
    }

    // The queue for adding new columns. Entries in the queue are not part of
    // the matrix.
    pub fn push_back(&mut self, i: Int, x: f64) {
        self.rowidx_queue.push(i);
        self.values_queue.push(x);
    }
    pub fn queue_size(&self) -> usize {
        self.rowidx_queue.len()
    }
    pub fn qindex(&self, pos: usize) -> Int {
        self.rowidx_queue[pos]
    }
    pub fn qvalue(&self, pos: usize) -> f64 {
        self.values_queue[pos]
    }
    pub fn set_qentry(&mut self, pos: usize, i: Int, x: f64) {
        self.rowidx_queue[pos] = i;
        self.values_queue[pos] = x;
    }

    /// Makes a new column from the queue.
    pub fn add_column(&mut self) {
        let nz = self.entries() as usize;
        let nznew = nz + self.queue_size();
        self.reserve(nznew);
        self.rowidx[nz..nznew].copy_from_slice(&self.rowidx_queue);
        self.values[nz..nznew].copy_from_slice(&self.values_queue);
        self.colptr.push(nznew as Int);
        self.clear_queue();
    }

    pub fn clear_queue(&mut self) {
        self.rowidx_queue.clear();
        self.values_queue.clear();
    }
}

/// Builds transpose of matrix.
pub fn transpose(a: &SparseMatrix) -> SparseMatrix {
    let mut at = SparseMatrix::default();
    transpose_into(a, &mut at);
    at
}

/// Resizes AT as necessary and fills with the transpose of A.
pub fn transpose_into(a: &SparseMatrix, at: &mut SparseMatrix) {
    let m = a.rows() as usize;
    let n = a.cols() as usize;
    let nz = a.entries() as usize;
    at.resize(n as Int, m as Int, nz as Int);

    // Compute row counts of A in workspace.
    let mut work = vec![0 as Int; m];
    for p in 0..nz {
        work[a.index(p)] += 1;
    }

    // Set column pointers for AT.
    let mut sum = 0;
    for i in 0..m {
        at.colptr[i] = sum;
        sum += work[i];
        work[i] = at.colptr[i];
    }
    at.colptr[m] = sum;

    // Fill AT with one column of A at a time.
    for j in 0..n {
        for p in a.begin(j)..a.end(j) {
            let i = a.index(p);
            let put = work[i] as usize;
            work[i] += 1;
            at.rowidx[put] = j as Int;
            at.values[put] = a.values[p];
        }
    }
}

/// Returns a copy of A[:,cols].
pub fn copy_columns(a: &SparseMatrix, cols: &[Int]) -> SparseMatrix {
    let mut a2 = SparseMatrix::new(a.rows(), 0);
    for &j in cols {
        for p in a.begin(j as usize)..a.end(j as usize) {
            a2.push_back(a.rowidx[p], a.values[p]);
        }
        a2.add_column();
    }
    a2
}

/// Permutes rows in place so that row i becomes row perm[i].
pub fn permute_rows(a: &mut SparseMatrix, perm: &[Int]) {
    let nz = a.entries() as usize;
    for p in 0..nz {
        a.rowidx[p] = perm[a.rowidx[p] as usize];
    }
}

/// Multiplies column j by s.
pub fn scale_column(a: &mut SparseMatrix, j: usize, s: f64) {
    for p in a.begin(j)..a.end(j) {
        a.values[p] *= s;
    }
}

/// Removes diagonal entries from A, returning them in diag (zero if not
/// present) and the # entries removed.
pub fn remove_diagonal(a: &mut SparseMatrix, mut diag: Option<&mut [f64]>) -> Int {
    let ncol = a.cols() as usize;
    let mut get = 0usize;
    let mut put = 0usize;
    for j in 0..ncol {
        if let Some(d) = diag.as_deref_mut() {
            d[j] = 0.0;
        }
        a.colptr[j] = put as Int;
        let end = a.colptr[j + 1] as usize;
        while get < end {
            if a.rowidx[get] as usize == j {
                if let Some(d) = diag.as_deref_mut() {
                    d[j] = a.values[get];
                }
            } else {
                a.rowidx[put] = a.rowidx[get];
                a.values[put] = a.values[get];
                put += 1;
            }
            get += 1;
        }
    }
    a.colptr[ncol] = put as Int;
    (get - put) as Int
}

/// Returns dot(A[:,j], rhs).
#[inline(always)]
pub fn dot_column(a: &SparseMatrix, j: usize, rhs: &[f64]) -> f64 {
    let (b, e) = (a.begin(j), a.end(j));
    let mut d = 0.0f64;
    for (&i, &v) in a.rowidx[b..e].iter().zip(&a.values[b..e]) {
        d = rhs[i as usize].mul_add(v, d);
    }
    d
}

/// lhs := lhs + alpha * A[:,j].
#[inline(always)]
pub fn scatter_column(a: &SparseMatrix, j: usize, alpha: f64, lhs: &mut [f64]) {
    let (b, e) = (a.begin(j), a.end(j));
    for (&i, &v) in a.rowidx[b..e].iter().zip(&a.values[b..e]) {
        let i = i as usize;
        lhs[i] = alpha.mul_add(v, lhs[i]);
    }
}

/// lhs := lhs + alpha*A*rhs or lhs := lhs + alpha*A'*rhs ('t'/'T').
pub fn multiply_add(a: &SparseMatrix, rhs: &[f64], alpha: f64, lhs: &mut [f64], trans: u8) {
    let n = a.cols() as usize;
    if trans == b't' || trans == b'T' {
        for j in 0..n {
            lhs[j] = alpha.mul_add(dot_column(a, j, rhs), lhs[j]);
        }
    } else {
        for j in 0..n {
            scatter_column(a, j, alpha * rhs[j], lhs);
        }
    }
}

/// lhs := lhs + A*A'*rhs or lhs := lhs + A*D*D*A'*rhs.
pub fn add_normal_product(a: &SparseMatrix, d: Option<&[f64]>, rhs: &[f64], lhs: &mut [f64]) {
    let n = a.cols() as usize;
    for j in 0..n {
        let mut temp = dot_column(a, j, rhs);
        if let Some(d) = d {
            temp *= d[j] * d[j];
        }
        scatter_column(a, j, temp, lhs);
    }
}

/// Triangular solve with sparse matrix; x is rhs on entry, solution on
/// return. upper: A is upper triangular (else lower); unitdiag: A has a unit
/// diagonal that is not stored. Returns the # nonzeros in the solution.
pub fn triangular_solve(a: &SparseMatrix, x: &mut [f64], trans: u8, upper: bool, unitdiag: bool) -> Int {
    let ncol = a.cols() as usize;
    let ap = &a.colptr;
    let ai = &a.rowidx;
    let ax = &a.values;
    let mut nz = 0;
    if trans == b't' || trans == b'T' {
        if upper {
            // transposed solve with upper triangular matrix
            for i in 0..ncol {
                let begin = ap[i] as usize;
                let end = ap[i + 1] as usize - usize::from(!unitdiag);
                let mut d = 0.0f64;
                for (&i, &v) in ai[begin..end].iter().zip(&ax[begin..end]) {
                    d = x[i as usize].mul_add(v, d);
                }
                x[i] -= d;
                if !unitdiag {
                    x[i] /= ax[end];
                }
                if x[i] != 0.0 {
                    nz += 1;
                }
            }
        } else {
            // transposed solve with lower triangular matrix
            for i in (0..ncol).rev() {
                let begin = ap[i] as usize + usize::from(!unitdiag);
                let end = ap[i + 1] as usize;
                let mut d = 0.0f64;
                for (&i, &v) in ai[begin..end].iter().zip(&ax[begin..end]) {
                    d = x[i as usize].mul_add(v, d);
                }
                x[i] -= d;
                if !unitdiag {
                    x[i] /= ax[begin - 1];
                }
                if x[i] != 0.0 {
                    nz += 1;
                }
            }
        }
    } else if upper {
        // forward solve with upper triangular matrix
        for j in (0..ncol).rev() {
            let begin = ap[j] as usize;
            let end = ap[j + 1] as usize - usize::from(!unitdiag);
            if !unitdiag {
                x[j] /= ax[end];
            }
            let temp = x[j];
            if temp != 0.0 {
                for (&i, &v) in ai[begin..end].iter().zip(&ax[begin..end]) {
                    let i = i as usize;
                    x[i] = (-v).mul_add(temp, x[i]);
                }
                nz += 1;
            }
        }
    } else {
        // forward solve with lower triangular matrix
        for j in 0..ncol {
            let begin = ap[j] as usize + usize::from(!unitdiag);
            let end = ap[j + 1] as usize;
            if !unitdiag {
                x[j] /= ax[begin - 1];
            }
            let temp = x[j];
            if temp != 0.0 {
                for (&i, &v) in ai[begin..end].iter().zip(&ax[begin..end]) {
                    let i = i as usize;
                    x[i] = (-v).mul_add(temp, x[i]);
                }
                nz += 1;
            }
        }
    }
    nz
}

/// Solves (L*U) x = x; L unit lower triangular stored without diagonal, U
/// upper triangular with the diagonal element at the end of each column.
pub fn forward_solve(l: &SparseMatrix, u: &SparseMatrix, x: &mut [f64]) {
    triangular_solve(l, x, b'n', false, true);
    triangular_solve(u, x, b'n', true, false);
}

/// Solves (L*U)' x = x.
pub fn backward_solve(l: &SparseMatrix, u: &SparseMatrix, x: &mut [f64]) {
    triangular_solve(u, x, b't', true, false);
    triangular_solve(l, x, b't', false, true);
}

pub fn onenorm(a: &SparseMatrix) -> f64 {
    let mut norm = 0.0;
    for j in 0..a.cols() as usize {
        let mut colsum = 0.0;
        for p in a.begin(j)..a.end(j) {
            colsum += a.values[p].abs();
        }
        norm = cmax(norm, colsum);
    }
    norm
}

pub fn infnorm(a: &SparseMatrix) -> f64 {
    let mut rowsum = vec![0.0; a.rows() as usize];
    for j in 0..a.cols() as usize {
        for p in a.begin(j)..a.end(j) {
            rowsum[a.index(p)] += a.values[p].abs();
        }
    }
    utils::infnorm(&rowsum)
}

/// Estimates the 1-norm of inverse(A) for triangular A.
pub fn normest_inverse(a: &SparseMatrix, upper: bool, unitdiag: bool) -> f64 {
    let m = a.rows() as usize;
    let mut x = vec![0.0f64; m];

    // Solve A'x=b, where the entries of b are +/-1 chosen dynamically to
    // make x large.
    if upper {
        for j in 0..m {
            let begin = a.begin(j);
            let mut end = a.end(j);
            if !unitdiag {
                end -= 1;
            }
            let mut temp = 0.0f64;
            for p in begin..end {
                temp = (-x[a.index(p)]).mul_add(a.values[p], temp);
            }
            temp += if temp >= 0.0 { 1.0 } else { -1.0 };
            if !unitdiag {
                temp /= a.values[end];
            }
            x[j] = temp;
        }
    } else {
        for j in (0..m).rev() {
            let mut begin = a.begin(j);
            let end = a.end(j);
            if !unitdiag {
                begin += 1;
            }
            let mut temp = 0.0f64;
            for p in begin..end {
                temp = (-x[a.index(p)]).mul_add(a.values[p], temp);
            }
            temp += if temp >= 0.0 { 1.0 } else { -1.0 };
            if !unitdiag {
                temp /= a.values[begin - 1];
            }
            x[j] = temp;
        }
    }
    let x1norm = utils::onenorm(&x);
    let xinfnorm = utils::infnorm(&x);

    // Solve Ay=x, solution overwrites x.
    triangular_solve(a, &mut x, b'n', upper, unitdiag);
    let y1norm = utils::onenorm(&x);

    cmax(y1norm / x1norm, xinfnorm)
}
