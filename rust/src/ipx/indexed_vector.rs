//! indexed_vector.h/.cc: a dense vector with an optional pattern of its
//! nonzero entries.

use crate::util::fma::ClangFma;

use super::{Int, HYPERSPARSE_THRESHOLD};

#[derive(Clone, Debug, Default)]
pub struct IndexedVector {
    pub(crate) elements: Vec<f64>,
    pub(crate) pattern: Vec<Int>,
    // If nnz >= 0, then pattern[0..nnz-1] are the indices of (possible)
    // nonzeros in elements. If nnz < 0, then the pattern is unknown.
    nnz: Int,
}

impl IndexedVector {
    pub fn new(dim: usize) -> Self {
        IndexedVector {
            elements: vec![0.0; dim],
            pattern: vec![0; dim],
            nnz: 0,
        }
    }

    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    /// True if the pattern is known and sparse.
    pub fn sparse(&self) -> bool {
        self.nnz >= 0 && (self.nnz as f64) <= HYPERSPARSE_THRESHOLD * self.dim() as f64
    }

    pub fn nnz(&self) -> Int {
        self.nnz
    }

    pub fn invalidate_pattern(&mut self) {
        self.nnz = -1;
    }

    pub fn set_nnz(&mut self, new_nnz: Int) {
        self.nnz = new_nnz;
    }

    /// Sets all entries to zero and the pattern to empty.
    pub fn set_to_zero(&mut self) {
        if self.sparse() {
            for p in 0..self.nnz as usize {
                self.elements[self.pattern[p] as usize] = 0.0;
            }
        } else {
            self.elements.fill(0.0);
        }
        self.nnz = 0;
    }

    /// for_each_nonzero: calls f(i, v[i]) over the pattern if it is known
    /// and sparse, else over all entries.
    #[inline]
    pub fn for_each_nonzero(&self, mut f: impl FnMut(usize, f64)) {
        if self.sparse() {
            for &i in &self.pattern[..self.nnz as usize] {
                f(i as usize, self.elements[i as usize]);
            }
        } else {
            for (i, &x) in self.elements.iter().enumerate() {
                f(i, x);
            }
        }
    }

    /// for_each_nonzero with a reference to the entry
    #[inline]
    pub fn for_each_nonzero_mut(&mut self, mut f: impl FnMut(usize, &mut f64)) {
        if self.sparse() {
            for &i in &self.pattern[..self.nnz as usize] {
                f(i as usize, &mut self.elements[i as usize]);
            }
        } else {
            for (i, x) in self.elements.iter_mut().enumerate() {
                f(i, x);
            }
        }
    }
}

impl std::ops::Index<usize> for IndexedVector {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.elements[i]
    }
}

impl std::ops::IndexMut<usize> for IndexedVector {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.elements[i]
    }
}

/// Dot(IndexedVector, Vector). Both loops are interleaved by 4 in the C++
/// build, which splits the fused multiply-add of the first nnz/4*4 terms
/// (see utils::dot_blocked).
pub fn dot(x: &IndexedVector, y: &[f64]) -> f64 {
    if x.sparse() {
        let pattern = &x.pattern[..x.nnz as usize];
        let n = pattern.len();
        let nb = if n >= 4 { n - n % 4 } else { 0 };
        let mut d = 0.0f64;
        for &i in &pattern[..nb] {
            d += y[i as usize] * x.elements[i as usize];
        }
        for &i in &pattern[nb..] {
            d = y[i as usize].mul_add_c(x.elements[i as usize], d);
        }
        d
    } else {
        super::utils::dot_blocked(&x.elements, y, 0.0, 4)
    }
}
