//! HighsSort (highs/util/HighsSort.cpp): the heap sorts of HiGHS, on
//! 1-based arrays (entry 0 unused, or the "is a heap" flag of the
//! decreasing heap), and the set checks and sorts of the model edits.
//! Hand-written sorts, so the tie orders are the C++'s on every standard
//! library.

/// addToDecreasingHeap: keep the mx_n largest values in a min-heap
pub fn add_to_decreasing_heap(n: &mut i32, mx_n: i32, heap_v: &mut [f64], heap_ix: &mut [i32], v: f64, ix: i32) {
    if *n < mx_n {
        // The heap is not full so put the new value at the bottom of the
        // heap and let it rise up to its correct level
        *n += 1;
        let mut cd_p = *n as usize;
        let mut pa_p = cd_p / 2;
        while pa_p > 0 && v < heap_v[pa_p] {
            heap_v[cd_p] = heap_v[pa_p];
            heap_ix[cd_p] = heap_ix[pa_p];
            cd_p = pa_p;
            pa_p /= 2;
        }
        heap_v[cd_p] = v;
        heap_ix[cd_p] = ix;
    } else if v > heap_v[1] {
        // The heap is full so replace the least value with the new value
        // and let it sink down to its correct level
        let n = *n as usize;
        let mut pa_p = 1;
        let mut cd_p = pa_p + pa_p;
        while cd_p <= n {
            if cd_p < n && heap_v[cd_p] > heap_v[cd_p + 1] {
                cd_p += 1;
            }
            if v > heap_v[cd_p] {
                heap_v[pa_p] = heap_v[cd_p];
                heap_ix[pa_p] = heap_ix[cd_p];
                pa_p = cd_p;
                cd_p += cd_p;
                continue;
            }
            break;
        }
        heap_v[pa_p] = v;
        heap_ix[pa_p] = ix;
    }
    // Set heap_ix[0]=1 to indicate that the values form a heap
    heap_ix[0] = 1;
}

/// sortDecreasingHeap: sort heap_v[1..=n] (a heap if heap_ix[0] == 1)
/// into decreasing order
pub fn sort_decreasing_heap(n: i32, heap_v: &mut [f64], heap_ix: &mut [i32]) {
    if n <= 1 {
        return;
    }
    let n = n as usize;
    let (mut fo_p, mut srt_p) = if heap_ix[0] != 1 { (n / 2 + 1, n) } else { (1, n) };
    loop {
        let (v, ix);
        if fo_p > 1 {
            fo_p -= 1;
            v = heap_v[fo_p];
            ix = heap_ix[fo_p];
        } else {
            v = heap_v[srt_p];
            ix = heap_ix[srt_p];
            heap_v[srt_p] = heap_v[1];
            heap_ix[srt_p] = heap_ix[1];
            srt_p -= 1;
            if srt_p == 1 {
                heap_v[1] = v;
                heap_ix[1] = ix;
                return;
            }
        }
        let mut pa_p = fo_p;
        let mut cd_p = fo_p + fo_p;
        while cd_p <= srt_p {
            if cd_p < srt_p && heap_v[cd_p] > heap_v[cd_p + 1] {
                cd_p += 1;
            }
            if v > heap_v[cd_p] {
                heap_v[pa_p] = heap_v[cd_p];
                heap_ix[pa_p] = heap_ix[cd_p];
                pa_p = cd_p;
                cd_p += cd_p;
                continue;
            }
            break;
        }
        heap_v[pa_p] = v;
        heap_ix[pa_p] = ix;
    }
}

/// The index array carried along by a heap sort, if any
pub trait HeapIndex {
    fn swap(&mut self, a: usize, b: usize);
    fn get(&self, a: usize) -> i32;
    fn set(&mut self, a: usize, x: i32);
}

impl HeapIndex for () {
    fn swap(&mut self, _: usize, _: usize) {}
    fn get(&self, _: usize) -> i32 {
        0
    }
    fn set(&mut self, _: usize, _: i32) {}
}

impl HeapIndex for &mut [i32] {
    fn swap(&mut self, a: usize, b: usize) {
        <[i32]>::swap(self, a, b)
    }
    fn get(&self, a: usize) -> i32 {
        self[a]
    }
    fn set(&mut self, a: usize, x: i32) {
        self[a] = x
    }
}

/// maxHeapify. As the C++, a NaN value never settles (it loops)
pub fn max_heapify<V: Copy + PartialOrd, I: HeapIndex>(heap_v: &mut [V], heap_i: &mut I, i: usize, n: usize) {
    let temp_v = heap_v[i];
    let temp_i = heap_i.get(i);
    let mut j = 2 * i;
    while j <= n {
        if j < n && heap_v[j + 1] > heap_v[j] {
            j += 1;
        }
        if temp_v > heap_v[j] {
            break;
        } else if temp_v <= heap_v[j] {
            heap_v[j / 2] = heap_v[j];
            heap_i.set(j / 2, heap_i.get(j));
            j *= 2;
        }
    }
    heap_v[j / 2] = temp_v;
    heap_i.set(j / 2, temp_i);
}

/// buildMaxheap
pub fn build_maxheap<V: Copy + PartialOrd, I: HeapIndex>(heap_v: &mut [V], heap_i: &mut I, n: usize) {
    for i in (1..=n / 2).rev() {
        max_heapify(heap_v, heap_i, i, n);
    }
}

/// maxHeapsort: sort a max-heap into increasing order
pub fn max_heapsort_heap<V: Copy + PartialOrd, I: HeapIndex>(heap_v: &mut [V], heap_i: &mut I, n: usize) {
    for i in (2..=n).rev() {
        heap_v.swap(i, 1);
        heap_i.swap(i, 1);
        max_heapify(heap_v, heap_i, 1, i - 1);
    }
}

/// maxheapsort: sort heap_v[1..=n] (with heap_i) into increasing order
pub fn maxheapsort<V: Copy + PartialOrd, I: HeapIndex>(heap_v: &mut [V], heap_i: &mut I, n: usize) {
    build_maxheap(heap_v, heap_i, n);
    max_heapsort_heap(heap_v, heap_i, n);
}

/// increasingSetOk for integers
pub fn increasing_set_ok_int(set: &[i32], lower: i32, upper: i32, strict: bool) -> bool {
    let check_bounds = lower <= upper;
    let mut previous = if check_bounds {
        if strict {
            lower.wrapping_sub(1)
        } else {
            lower
        }
    } else {
        -i32::MAX
    };
    for &entry in set {
        if strict {
            if entry <= previous {
                return false;
            }
        } else if entry < previous {
            return false;
        }
        if check_bounds && entry > upper {
            return false;
        }
        previous = entry;
    }
    true
}

/// increasingSetOk for doubles
pub fn increasing_set_ok_double(set: &[f64], lower: f64, upper: f64, strict: bool) -> bool {
    const TINY: f64 = 1e-14;
    let check_bounds = lower <= upper;
    let mut previous = if check_bounds {
        if strict {
            if lower < 0.0 {
                (1.0 + TINY) * lower
            } else if lower > 0.0 {
                (1.0 - TINY) * lower
            } else {
                -TINY
            }
        } else {
            lower
        }
    } else {
        -f64::INFINITY
    };
    for &entry in set {
        if strict {
            if entry <= previous {
                return false;
            }
        } else if entry < previous {
            return false;
        }
        if check_bounds && entry > upper {
            return false;
        }
        previous = entry;
    }
    true
}

/// The permutation that sortSetData applies: set sorted (increasing) by
/// maxheapsort, perm[k] the original position of its k-th entry
pub fn sort_set_permutation(set: &mut [i32]) -> Vec<i32> {
    let n = set.len();
    let mut sort_set = vec![0; n + 1];
    let mut perm = vec![0; n + 1];
    for ix in 0..n {
        sort_set[1 + ix] = set[ix];
        perm[1 + ix] = ix as i32;
    }
    maxheapsort(&mut sort_set, &mut &mut perm[..], n);
    set.copy_from_slice(&sort_set[1..]);
    perm.remove(0);
    perm
}

/// `extern "C"` shims for highs/util/HighsSort.cpp. Heap arrays are
/// 1-based: they have n + 1 entries
mod ffi {
    use super::*;
    use crate::ffi::{sl, sl_mut};

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_add_to_decreasing_heap(
        n: *mut i32, mx_n: i32, heap_v: *mut f64, heap_ix: *mut i32, len: i32, v: f64, ix: i32,
    ) {
        add_to_decreasing_heap(&mut *n, mx_n, sl_mut(heap_v, len), sl_mut(heap_ix, len), v, ix)
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_sort_decreasing_heap(n: i32, heap_v: *mut f64, heap_ix: *mut i32, len: i32) {
        sort_decreasing_heap(n, sl_mut(heap_v, len), sl_mut(heap_ix, len))
    }

    /// what: 0 maxheapsort, 1 buildMaxheap, 2 maxHeapsort, 3 maxHeapify
    /// (at i); heap_i may be null
    unsafe fn heap_op<V: Copy + PartialOrd>(heap_v: *mut V, heap_i: *mut i32, i: i32, n: i32, what: i32) {
        if n < 1 && what != 3 {
            return;
        }
        let len = n.max(i) + 1;
        let v = sl_mut(heap_v, len);
        let nu = n.max(0) as usize;
        let iu = i.max(0) as usize;
        if heap_i.is_null() {
            let x = &mut ();
            match what {
                0 => maxheapsort(v, x, nu),
                1 => build_maxheap(v, x, nu),
                2 => max_heapsort_heap(v, x, nu),
                _ => max_heapify(v, x, iu, nu),
            }
        } else {
            let x = &mut sl_mut(heap_i, len);
            match what {
                0 => maxheapsort(v, x, nu),
                1 => build_maxheap(v, x, nu),
                2 => max_heapsort_heap(v, x, nu),
                _ => max_heapify(v, x, iu, nu),
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heap_int(heap_v: *mut i32, heap_i: *mut i32, i: i32, n: i32, what: i32) {
        heap_op(heap_v, heap_i, i, n, what)
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_heap_double(heap_v: *mut f64, heap_i: *mut i32, i: i32, n: i32, what: i32) {
        heap_op(heap_v, heap_i, i, n, what)
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_increasing_set_ok_int(set: *const i32, n: i32, lower: i32, upper: i32, strict: bool) -> bool {
        increasing_set_ok_int(sl(set, n), lower, upper, strict)
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_increasing_set_ok_double(set: *const f64, n: i32, lower: f64, upper: f64, strict: bool) -> bool {
        increasing_set_ok_double(sl(set, n), lower, upper, strict)
    }

    /// sortSetData: sort set[0..n] and gather each non-null data array
    /// (of `width` bytes per entry) into its sorted array
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_sort_set_data(
        n: i32, set: *mut i32, data: *const *const u8, sorted: *const *mut u8, num_data: i32, width: i32,
    ) {
        if n <= 0 {
            return;
        }
        let perm = sort_set_permutation(sl_mut(set, n));
        let w = width as usize;
        for d in 0..num_data as usize {
            let src = *data.add(d);
            if src.is_null() {
                continue;
            }
            let dst = *sorted.add(d);
            for (ix, &p) in perm.iter().enumerate() {
                std::ptr::copy_nonoverlapping(src.add(p as usize * w), dst.add(ix * w), w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxheapsort_sorts_with_indices() {
        let mut v = [0.0, 3.0, 1.0, 2.0, 1.0, 5.0];
        let mut i = [0, 10, 11, 12, 13, 14];
        maxheapsort(&mut v, &mut &mut i[..], 5);
        assert_eq!(v, [0.0, 1.0, 1.0, 2.0, 3.0, 5.0]);
        assert_eq!(i[3..], [12, 10, 14]);
    }

    #[test]
    fn sort_set_permutation_and_checks() {
        let mut set = [4, 1, 3];
        assert_eq!(sort_set_permutation(&mut set), vec![1, 2, 0]);
        assert_eq!(set, [1, 3, 4]);
        assert!(increasing_set_ok_int(&set, 0, 4, true));
        assert!(!increasing_set_ok_int(&set, 2, 4, true));
        assert!(increasing_set_ok_double(&[0.5, 1.0], 0.5, 2.0, true));
        assert!(!increasing_set_ok_double(&[1.0, 1.0], 0.0, 2.0, true));
    }

    /// The checksum of golden_sort.cpp (ties in all the heap sorts)
    #[test]
    fn golden() {
        let mut h: u64 = 1469598103934665603;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(1099511628211);
        let mut s: u64 = 1;
        let mut next = |m: i32| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % m as u64) as i32
        };
        for _ in 0..2000 {
            let n = next(40) as usize;
            let m = 1 + next(10);
            let (mut a, mut ai, mut b, mut d) = (vec![0; n + 1], vec![0; n + 1], vec![0; n + 1], vec![0.0; n + 1]);
            let (mut dv, mut di) = (vec![0.0; n + 2], vec![0; n + 2]);
            for k in 1..=n {
                a[k] = next(m);
                b[k] = a[k];
                ai[k] = k as i32;
                d[k] = next(m) as f64 * 0.5;
            }
            let mut ci = ai.clone();
            maxheapsort(&mut a, &mut &mut ai[..], n);
            maxheapsort(&mut b, &mut (), n);
            maxheapsort(&mut d, &mut &mut ci[..], n);
            let mut hn = 0;
            for k in 1..=n {
                let v = next(m) as f64 * 0.25;
                add_to_decreasing_heap(&mut hn, n as i32 / 2 + 1, &mut dv, &mut di, v, k as i32);
            }
            sort_decreasing_heap(hn, &mut dv, &mut di);
            for k in 1..=n {
                for x in [a[k] as u64, ai[k] as u64, b[k] as u64, (d[k] * 4.0) as u64, ci[k] as u64] {
                    mix(x);
                }
            }
            for k in 1..=hn as usize {
                mix((dv[k] * 4.0) as u64);
                mix(di[k] as u64);
            }
        }
        assert_eq!(h, 13511892912542497810);
    }

    #[test]
    fn decreasing_heap_keeps_largest_sorted() {
        let mut v = vec![0.0; 4];
        let mut ix = vec![0; 4];
        let mut n = 0;
        for (k, x) in [5.0, 1.0, 7.0, 3.0, 6.0].iter().enumerate() {
            add_to_decreasing_heap(&mut n, 3, &mut v, &mut ix, *x, k as i32);
        }
        sort_decreasing_heap(n, &mut v, &mut ix);
        assert_eq!(&v[1..], &[7.0, 6.0, 5.0]);
        assert_eq!(&ix[1..], &[2, 4, 0]);
    }
}
