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
        let (array, index) = (&self.array[..size], &mut self.index[..size]);
        let mut num = 0;
        // By blocks of 4, so that the count of nonzeros is a chain of one
        // add per block rather than one per entry
        let mut blocks = array.chunks_exact(4);
        let mut i = 0;
        for b in &mut blocks {
            let nz = [(b[0] != 0.0) as usize, (b[1] != 0.0) as usize, (b[2] != 0.0) as usize, (b[3] != 0.0) as usize];
            let (p1, p2) = (nz[0], nz[0] + nz[1]);
            let p3 = p2 + nz[2];
            // SAFETY: each position is num plus the nonzeros among the
            // entries before i + k, so at most i + k < size = index.len()
            unsafe {
                *index.get_unchecked_mut(num) = i as i32;
                *index.get_unchecked_mut(num + p1) = i as i32 + 1;
                *index.get_unchecked_mut(num + p2) = i as i32 + 2;
                *index.get_unchecked_mut(num + p3) = i as i32 + 3;
            }
            num += p3 + nz[3];
            i += 4;
        }
        for &v in blocks.remainder() {
            index[num] = i as i32;
            num += (v != 0.0) as usize;
            i += 1;
        }
        self.count = num as i32;
    }
}

/// An HVector owned by Rust: setup/clear as in HVectorBase.
///
/// Its buffers come from, and on drop go back to, a small per-thread pool:
/// the solvers set up their work vectors per solve (and per call of e.g.
/// computeDual), as in the C++, and allocating and zeroing them anew (41
/// bytes per entry) cost several percent of the dispatch MIPs' many short
/// LP solves. A vector goes back cleared (HVector::clear, on which the
/// solvers rely between uses anyway), and the solves leave cwork (the
/// marks of the hyper-sparse solve) zero, so a reused vector is
/// indistinguishable from a new one.
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

/// The buffers of a pooled OwnedHVec, with `array` and `cwork` zero
struct Buffers {
    index: Vec<i32>,
    array: Vec<f64>,
    cwork: Vec<u8>,
    iwork: Vec<i32>,
    pack_index: Vec<i32>,
    pack_value: Vec<f64>,
}

impl Buffers {
    fn new(n: usize) -> Self {
        Buffers {
            index: vec![0; n],
            array: vec![0.0; n],
            cwork: vec![0; n + 6400],
            iwork: vec![0; n * 4],
            pack_index: vec![0; n],
            pack_value: vec![0.0; n],
        }
    }

    fn bytes(&self) -> usize {
        41 * self.array.len() + self.cwork.len()
    }
}

/// At most this many vectors, and bytes, are kept per thread (the oldest
/// go first)
const POOL_VECTORS: usize = 16;
const POOL_BYTES: usize = 64 << 20;

thread_local! {
    static POOL: std::cell::RefCell<Vec<Buffers>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl OwnedHVec {
    pub fn new(size: i32) -> Self {
        let n = size as usize;
        let pooled = POOL
            .try_with(|p| {
                let mut p = p.borrow_mut();
                let i = p.iter().rposition(|b| b.array.len() == n)?;
                Some(p.remove(i))
            })
            .ok()
            .flatten();
        let b = pooled.unwrap_or_else(|| Buffers::new(n));
        OwnedHVec {
            size,
            count: 0,
            index: b.index,
            array: b.array,
            cwork: b.cwork,
            iwork: b.iwork,
            synthetic_tick: 0.0,
            pack_flag: false,
            pack_count: 0,
            pack_index: b.pack_index,
            pack_value: b.pack_value,
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

impl Drop for OwnedHVec {
    fn drop(&mut self) {
        let n = self.size.max(0) as usize;
        let intact = self.array.len() == n
            && self.index.len() == n
            && self.cwork.len() == n + 6400
            && self.iwork.len() == 4 * n
            && self.pack_index.len() == n
            && self.pack_value.len() == n;
        if !intact {
            return;
        }
        self.clear();
        debug_assert!(self.array.iter().all(|&x| x == 0.0) && self.cwork.iter().all(|&c| c == 0));
        let b = Buffers {
            index: std::mem::take(&mut self.index),
            array: std::mem::take(&mut self.array),
            cwork: std::mem::take(&mut self.cwork),
            iwork: std::mem::take(&mut self.iwork),
            pack_index: std::mem::take(&mut self.pack_index),
            pack_value: std::mem::take(&mut self.pack_value),
        };
        if b.bytes() > POOL_BYTES / 4 {
            return;
        }
        let _ = POOL.try_with(|p| {
            let mut p = p.borrow_mut();
            let mut bytes = b.bytes() + p.iter().map(Buffers::bytes).sum::<usize>();
            while !p.is_empty() && (p.len() >= POOL_VECTORS || bytes > POOL_BYTES) {
                bytes -= p.remove(0).bytes();
            }
            p.push(b);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_index_finds_the_nonzeros_in_order() {
        for n in 0..12 {
            let mut v = OwnedHVec::new(n);
            let want: Vec<i32> = (0..n).filter(|i| i % 3 != 1).collect();
            for &i in &want {
                v.array[i as usize] = 1.0 + i as f64;
            }
            v.count = -1;
            v.view().re_index();
            let mut view = v.view();
            view.re_index();
            assert_eq!(&view.index[..view.count as usize], &want[..]);
        }
    }
}
