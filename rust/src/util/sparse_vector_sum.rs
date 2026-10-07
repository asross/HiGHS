//! HighsSparseVectorSum: a dense accumulator of double-double values with
//! the list of its nonzero positions.

use super::cdouble::CDouble;

#[derive(Clone, Default)]
pub struct HighsSparseVectorSum {
    pub values: Vec<CDouble>,
    pub nonzeroinds: Vec<i32>,
}

impl HighsSparseVectorSum {
    pub fn new(dimension: usize) -> Self {
        let mut s = Self::default();
        s.set_dimension(dimension);
        s
    }

    pub fn set_dimension(&mut self, dimension: usize) {
        self.values.resize(dimension, CDouble::default());
        self.nonzeroinds.reserve(dimension);
    }

    /// Adds a double or a CDouble. A sum that cancels to zero is stored as
    /// the smallest normal double so that the index stays listed once.
    pub fn add(&mut self, index: i32, value: impl Into<CDouble>) {
        let value = value.into();
        let v = &mut self.values[index as usize];
        if *v != 0.0 {
            *v += value;
        } else {
            *v = value;
            self.nonzeroinds.push(index);
        }
        if *v == 0.0 {
            *v = CDouble::from(f64::MIN_POSITIVE);
        }
    }

    /// add(HighsInt, double): the double overload (a CDouble += double)
    pub fn add_f64(&mut self, index: i32, value: f64) {
        let v = &mut self.values[index as usize];
        if *v != 0.0 {
            *v += value;
        } else {
            *v = CDouble::from(value);
            self.nonzeroinds.push(index);
        }
        if *v == 0.0 {
            *v = CDouble::from(f64::MIN_POSITIVE);
        }
    }

    pub fn get_nonzeros(&self) -> &[i32] {
        &self.nonzeroinds
    }

    pub fn get_value(&self, index: i32) -> f64 {
        self.values[index as usize].to_f64()
    }

    pub fn clear(&mut self) {
        if 10 * self.nonzeroinds.len() < 3 * self.values.len() {
            for &i in &self.nonzeroinds {
                self.values[i as usize] = CDouble::default();
            }
        } else {
            self.values.fill(CDouble::default());
        }
        self.nonzeroinds.clear();
    }

    /// std::partition of the nonzero indices (the bidirectional algorithm of
    /// libc++ and libstdc++, which fixes the order); returns the number of
    /// indices satisfying pred.
    pub fn partition(&mut self, mut pred: impl FnMut(i32) -> bool) -> usize {
        let v = &mut self.nonzeroinds;
        let (mut first, mut last) = (0, v.len());
        loop {
            loop {
                if first == last {
                    return first;
                }
                if !pred(v[first]) {
                    break;
                }
                first += 1;
            }
            loop {
                last -= 1;
                if first == last {
                    return first;
                }
                if pred(v[last]) {
                    break;
                }
            }
            v.swap(first, last);
            first += 1;
        }
    }

    /// Drops the entries for which is_zero(index, value) holds, scanning
    /// backwards and swapping each dropped index to the end.
    pub fn cleanup(&mut self, mut is_zero: impl FnMut(i32, f64) -> bool) {
        let mut num_nz = self.nonzeroinds.len();
        for i in (0..num_nz).rev() {
            let pos = self.nonzeroinds[i];
            if is_zero(pos, self.values[pos as usize].to_f64()) {
                self.values[pos as usize] = CDouble::default();
                num_nz -= 1;
                self.nonzeroinds.swap(num_nz, i);
            }
        }
        self.nonzeroinds.truncate(num_nz);
    }
}
