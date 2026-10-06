//! HVector's data as seen from Rust: either borrowed from a C++ HVector
//! for the duration of a call, or owned (for the factor's own work vectors).

use crate::util::fma::ClangFma;

pub const K_HIGHS_TINY: f64 = 1e-14;
pub const K_HIGHS_ZERO: f64 = 1e-50;

/// The arrays and scalars of an HVector (util/HVectorBase.h).
pub struct HVec<'a> {
    pub size: i32,
    pub count: i32,
    pub index: &'a mut [i32],
    pub array: &'a mut [f64],
    pub cwork: &'a mut [u8],
    pub iwork: &'a mut [i32],
    pub synthetic_tick: f64,
    pub pack_flag: bool,
    pub pack_count: i32,
    pub pack_index: &'a mut [i32],
    pub pack_value: &'a mut [f64],
}

impl HVec<'_> {
    /// HVectorBase::clear (but `next`, which stays with C++)
    pub fn clear(&mut self) {
        if self.count < 0 || self.count as f64 > self.size as f64 * 0.3 {
            self.array.fill(0.0);
        } else {
            for &i in &self.index[..self.count as usize] {
                self.array[i as usize] = 0.0;
            }
        }
        self.pack_flag = false;
        self.count = 0;
        self.synthetic_tick = 0.0;
    }

    /// HVectorBase::norm2: the squared 2-norm, as compiled out of line,
    /// where clang interleaves the loop by 4 without contracting and
    /// contracts the remainder loop
    pub fn norm2(&self) -> f64 {
        self.norm2_split(true)
    }

    /// norm2 as clang compiles it inline in some callers (e.g.
    /// HEkk::computeDualSteepestEdgeWeights): contracted throughout
    pub fn norm2_fused(&self) -> f64 {
        self.norm2_split(false)
    }

    fn norm2_split(&self, interleaved: bool) -> f64 {
        let index = &self.index[..self.count.max(0) as usize];
        let unfused = if interleaved && index.len() >= 4 { index.len() & !3 } else { 0 };
        let mut result = 0.0;
        for &i in &index[..unfused] {
            let value = self.array[i as usize];
            result += value * value;
        }
        for &i in &index[unfused..] {
            let value = self.array[i as usize];
            result = value.mul_add_c(value, result);
        }
        result
    }

    /// Zero values that do not exceed kHighsTiny in magnitude, maintaining
    /// the index if it is well defined
    pub fn tight(&mut self) {
        if self.count < 0 {
            for v in self.array.iter_mut() {
                if v.abs() < K_HIGHS_TINY {
                    *v = 0.0;
                }
            }
        } else {
            let index = &mut self.index[..self.count as usize];
            let mut total = 0;
            for i in 0..index.len() {
                let my_index = index[i];
                let value = &mut self.array[my_index as usize];
                if value.abs() >= K_HIGHS_TINY {
                    index[total] = my_index;
                    total += 1;
                } else {
                    *value = 0.0;
                }
            }
            self.count = total as i32;
        }
    }

    /// If pack_flag is set, pack the nonzeros into pack_index/pack_value
    pub fn pack(&mut self) {
        if !self.pack_flag {
            return;
        }
        self.pack_flag = false;
        let n = self.count as usize;
        for i in 0..n {
            let ipack = self.index[i];
            self.pack_index[i] = ipack;
            self.pack_value[i] = self.array[ipack as usize];
        }
        self.pack_count = n as i32;
    }

    /// Possibly determine the indices from scratch by passing through the
    /// array
    pub fn re_index(&mut self) {
        if self.count >= 0 && self.count as f64 <= self.size as f64 * 0.1 {
            return;
        }
        let size = self.size as usize;
        let mut num = 0;
        for (i, &v) in self.array[..size].iter().enumerate() {
            self.index[num] = i as i32;
            num += (v != 0.0) as usize;
        }
        self.count = num as i32;
    }
}

/// An HVector owned by Rust: setup/clear as in HVectorBase.
pub struct OwnedHVec {
    pub size: i32,
    pub count: i32,
    pub index: Vec<i32>,
    pub array: Vec<f64>,
    pub cwork: Vec<u8>,
    pub iwork: Vec<i32>,
    pub synthetic_tick: f64,
    pub pack_flag: bool,
    pub pack_count: i32,
    pub pack_index: Vec<i32>,
    pub pack_value: Vec<f64>,
}

impl OwnedHVec {
    pub fn new(size: i32) -> Self {
        let n = size as usize;
        OwnedHVec {
            size,
            count: 0,
            index: vec![0; n],
            array: vec![0.0; n],
            cwork: vec![0; n + 6400],
            iwork: vec![0; n * 4],
            synthetic_tick: 0.0,
            pack_flag: false,
            pack_count: 0,
            pack_index: vec![0; n],
            pack_value: vec![0.0; n],
        }
    }

    pub fn clear(&mut self) {
        if self.count < 0 || self.count as f64 > self.size as f64 * 0.3 {
            self.array.fill(0.0);
        } else {
            for &i in &self.index[..self.count as usize] {
                self.array[i as usize] = 0.0;
            }
        }
        self.pack_flag = false;
        self.count = 0;
        self.synthetic_tick = 0.0;
    }

    /// Run `f` on a borrowed view, then copy the scalars back
    pub fn with<R>(&mut self, f: impl FnOnce(&mut HVec) -> R) -> R {
        let mut v = self.view();
        let r = f(&mut v);
        let (count, tick, flag, pc) = (v.count, v.synthetic_tick, v.pack_flag, v.pack_count);
        self.count = count;
        self.synthetic_tick = tick;
        self.pack_flag = flag;
        self.pack_count = pc;
        r
    }

    /// A borrowed view (scalars are copies)
    pub fn view(&mut self) -> HVec<'_> {
        HVec {
            size: self.size,
            count: self.count,
            index: &mut self.index,
            array: &mut self.array,
            cwork: &mut self.cwork,
            iwork: &mut self.iwork,
            synthetic_tick: self.synthetic_tick,
            pack_flag: self.pack_flag,
            pack_count: self.pack_count,
            pack_index: &mut self.pack_index,
            pack_value: &mut self.pack_value,
        }
    }
}
