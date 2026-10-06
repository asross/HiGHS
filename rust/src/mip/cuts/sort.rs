//! The sorting algorithms of the C++ code, step for step, so that elements
//! that compare equal (or comparators that are not strict weak orders) end
//! up in the same order: pdqsort (extern/pdqsort/pdqsort.h, both the plain
//! and the branchless partition) and libc++'s heap algorithms and
//! partial_sort (Apple clang's libc++).

const INSERTION_SORT_THRESHOLD: usize = 24;
const NINTHER_THRESHOLD: usize = 128;
const PARTIAL_INSERTION_SORT_LIMIT: usize = 8;
const BLOCK_SIZE: usize = 64;

/// pdqsort(begin, end, comp) with a non-default comparator.
pub fn pdqsort<T: Copy>(v: &mut [T], mut comp: impl FnMut(&T, &T) -> bool) {
    if v.is_empty() {
        return;
    }
    let bad = log2(v.len());
    pdqsort_loop::<T, _, false>(v, 0, v.len(), &mut comp, bad, true);
}

/// pdqsort_branchless(begin, end, comp); also pdqsort with std::less on an
/// arithmetic type.
pub fn pdqsort_branchless<T: Copy>(v: &mut [T], mut comp: impl FnMut(&T, &T) -> bool) {
    if v.is_empty() {
        return;
    }
    let bad = log2(v.len());
    pdqsort_loop::<T, _, true>(v, 0, v.len(), &mut comp, bad, true);
}

fn log2(mut n: usize) -> i32 {
    let mut log = 0;
    loop {
        n >>= 1;
        if n == 0 {
            return log;
        }
        log += 1;
    }
}

fn insertion_sort<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], begin: usize, end: usize, comp: &mut C) {
    if begin == end {
        return;
    }
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                if sift == begin {
                    break;
                }
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
        }
    }
}

fn unguarded_insertion_sort<T: Copy, C: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut C,
) {
    if begin == end {
        return;
    }
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
        }
    }
}

fn partial_insertion_sort<T: Copy, C: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut C,
) -> bool {
    if begin == end {
        return true;
    }
    let mut limit = 0usize;
    for cur in begin + 1..end {
        let mut sift = cur;
        let mut sift_1 = cur - 1;
        if comp(&v[sift], &v[sift_1]) {
            let tmp = v[sift];
            loop {
                v[sift] = v[sift_1];
                sift -= 1;
                if sift == begin {
                    break;
                }
                sift_1 -= 1;
                if !comp(&tmp, &v[sift_1]) {
                    break;
                }
            }
            v[sift] = tmp;
            limit += cur - sift;
        }
        if limit > PARTIAL_INSERTION_SORT_LIMIT {
            return false;
        }
    }
    true
}

#[inline]
fn sort2<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], a: usize, b: usize, comp: &mut C) {
    if comp(&v[b], &v[a]) {
        v.swap(a, b);
    }
}

#[inline]
fn sort3<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], a: usize, b: usize, c: usize, comp: &mut C) {
    sort2(v, a, b, comp);
    sort2(v, b, c, comp);
    sort2(v, a, b, comp);
}

fn swap_offsets<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    offsets_l: &[u8],
    offsets_r: &[u8],
    num: usize,
    use_swaps: bool,
) {
    if use_swaps {
        for i in 0..num {
            v.swap(first + offsets_l[i] as usize, last - offsets_r[i] as usize);
        }
    } else if num > 0 {
        let mut l = first + offsets_l[0] as usize;
        let mut r = last - offsets_r[0] as usize;
        let tmp = v[l];
        v[l] = v[r];
        for i in 1..num {
            l = first + offsets_l[i] as usize;
            v[r] = v[l];
            r = last - offsets_r[i] as usize;
            v[l] = v[r];
        }
        v[r] = tmp;
    }
}

fn partition_right_branchless<T: Copy, C: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut C,
) -> (usize, bool) {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        first += 1;
        if !comp(&v[first], &pivot) {
            break;
        }
    }
    if first - 1 == begin {
        while first < last {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    if !already_partitioned {
        v.swap(first, last);
        first += 1;
        let mut offsets_l_buf = [0u8; BLOCK_SIZE];
        let mut offsets_r_buf = [0u8; BLOCK_SIZE];
        let mut offsets_l_base = first;
        let mut offsets_r_base = last;
        let (mut num_l, mut num_r, mut start_l, mut start_r) = (0usize, 0usize, 0usize, 0usize);
        while first < last {
            let num_unknown = last - first;
            let left_split = if num_l == 0 {
                if num_r == 0 {
                    num_unknown / 2
                } else {
                    num_unknown
                }
            } else {
                0
            };
            let right_split = if num_r == 0 { num_unknown - left_split } else { 0 };
            let ln = if left_split >= BLOCK_SIZE { BLOCK_SIZE } else { left_split };
            for i in 0..ln {
                offsets_l_buf[num_l] = i as u8;
                num_l += !comp(&v[first], &pivot) as usize;
                first += 1;
            }
            let rn = if right_split >= BLOCK_SIZE { BLOCK_SIZE } else { right_split };
            for i in 0..rn {
                offsets_r_buf[num_r] = (i + 1) as u8;
                last -= 1;
                num_r += comp(&v[last], &pivot) as usize;
            }
            let num = num_l.min(num_r);
            swap_offsets(
                v,
                offsets_l_base,
                offsets_r_base,
                &offsets_l_buf[start_l..],
                &offsets_r_buf[start_r..],
                num,
                num_l == num_r,
            );
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if num_l == 0 {
                start_l = 0;
                offsets_l_base = first;
            }
            if num_r == 0 {
                start_r = 0;
                offsets_r_base = last;
            }
        }
        if num_l != 0 {
            while num_l > 0 {
                num_l -= 1;
                last -= 1;
                v.swap(offsets_l_base + offsets_l_buf[start_l + num_l] as usize, last);
            }
            first = last;
        }
        if num_r != 0 {
            while num_r > 0 {
                num_r -= 1;
                v.swap(offsets_r_base - offsets_r_buf[start_r + num_r] as usize, first);
                first += 1;
            }
            last = first;
        }
        let _ = last;
    }
    let pivot_pos = first - 1;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    (pivot_pos, already_partitioned)
}

fn partition_right<T: Copy, C: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut C,
) -> (usize, bool) {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        first += 1;
        if !comp(&v[first], &pivot) {
            break;
        }
    }
    if first - 1 == begin {
        while first < last {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if !comp(&v[first], &pivot) {
                break;
            }
        }
        loop {
            last -= 1;
            if comp(&v[last], &pivot) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    (pivot_pos, already_partitioned)
}

fn partition_left<T: Copy, C: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    begin: usize,
    end: usize,
    comp: &mut C,
) -> usize {
    let pivot = v[begin];
    let mut first = begin;
    let mut last = end;
    loop {
        last -= 1;
        if !comp(&pivot, &v[last]) {
            break;
        }
    }
    if last + 1 == end {
        while first < last {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    } else {
        loop {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    }
    while first < last {
        v.swap(first, last);
        loop {
            last -= 1;
            if !comp(&pivot, &v[last]) {
                break;
            }
        }
        loop {
            first += 1;
            if comp(&pivot, &v[first]) {
                break;
            }
        }
    }
    let pivot_pos = last;
    v[begin] = v[pivot_pos];
    v[pivot_pos] = pivot;
    pivot_pos
}

fn pdqsort_loop<T: Copy, C: FnMut(&T, &T) -> bool, const BRANCHLESS: bool>(
    v: &mut [T],
    mut begin: usize,
    end: usize,
    comp: &mut C,
    mut bad_allowed: i32,
    mut leftmost: bool,
) {
    loop {
        let size = end - begin;
        if size < INSERTION_SORT_THRESHOLD {
            if leftmost {
                insertion_sort(v, begin, end, comp);
            } else {
                unguarded_insertion_sort(v, begin, end, comp);
            }
            return;
        }
        let s2 = size / 2;
        if size > NINTHER_THRESHOLD {
            sort3(v, begin, begin + s2, end - 1, comp);
            sort3(v, begin + 1, begin + (s2 - 1), end - 2, comp);
            sort3(v, begin + 2, begin + (s2 + 1), end - 3, comp);
            sort3(v, begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
            v.swap(begin, begin + s2);
        } else {
            sort3(v, begin + s2, begin, end - 1, comp);
        }
        if !leftmost && !comp(&v[begin - 1], &v[begin]) {
            begin = partition_left(v, begin, end, comp) + 1;
            continue;
        }
        let (pivot_pos, already_partitioned) = if BRANCHLESS {
            partition_right_branchless(v, begin, end, comp)
        } else {
            partition_right(v, begin, end, comp)
        };
        let l_size = pivot_pos - begin;
        let r_size = end - (pivot_pos + 1);
        let highly_unbalanced = l_size < size / 8 || r_size < size / 8;
        if highly_unbalanced {
            bad_allowed -= 1;
            if bad_allowed == 0 {
                make_heap(&mut v[begin..end], comp);
                sort_heap(&mut v[begin..end], comp);
                return;
            }
            if l_size >= INSERTION_SORT_THRESHOLD {
                v.swap(begin, begin + l_size / 4);
                v.swap(pivot_pos - 1, pivot_pos - l_size / 4);
                if l_size > NINTHER_THRESHOLD {
                    v.swap(begin + 1, begin + (l_size / 4 + 1));
                    v.swap(begin + 2, begin + (l_size / 4 + 2));
                    v.swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    v.swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }
            if r_size >= INSERTION_SORT_THRESHOLD {
                v.swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                v.swap(end - 1, end - r_size / 4);
                if r_size > NINTHER_THRESHOLD {
                    v.swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    v.swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    v.swap(end - 2, end - (1 + r_size / 4));
                    v.swap(end - 3, end - (2 + r_size / 4));
                }
            }
        } else if already_partitioned
            && partial_insertion_sort(v, begin, pivot_pos, comp)
            && partial_insertion_sort(v, pivot_pos + 1, end, comp)
        {
            return;
        }
        pdqsort_loop::<T, C, BRANCHLESS>(v, begin, pivot_pos, comp, bad_allowed, leftmost);
        begin = pivot_pos + 1;
        leftmost = false;
    }
}

// ---- libc++ heap algorithms ----

/// std::__sift_up on v[..len]
fn sift_up<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], len: usize, comp: &mut C) {
    if len > 1 {
        let mut l = (len - 2) / 2;
        let mut ptr = l;
        let mut last = len - 1;
        if comp(&v[ptr], &v[last]) {
            let t = v[last];
            loop {
                v[last] = v[ptr];
                last = ptr;
                if l == 0 {
                    break;
                }
                l = (l - 1) / 2;
                ptr = l;
                if !comp(&v[ptr], &t) {
                    break;
                }
            }
            v[last] = t;
        }
    }
}

/// std::__sift_down of position start in the heap v[..len]
fn sift_down<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C, len: usize, mut start: usize) {
    let mut child = start;
    if len < 2 || (len - 2) / 2 < child {
        return;
    }
    child = 2 * child + 1;
    if child + 1 < len && comp(&v[child], &v[child + 1]) {
        child += 1;
    }
    if comp(&v[child], &v[start]) {
        return;
    }
    let top = v[start];
    loop {
        v[start] = v[child];
        start = child;
        if (len - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        if child + 1 < len && comp(&v[child], &v[child + 1]) {
            child += 1;
        }
        if comp(&v[child], &top) {
            break;
        }
    }
    v[start] = top;
}

/// std::__floyd_sift_down; returns the hole
fn floyd_sift_down<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C, len: usize) -> usize {
    // the C++ iterator __child_i always points at __child
    let mut hole = 0usize;
    let mut child = 0usize;
    loop {
        child = 2 * child + 1;
        if child + 1 < len && comp(&v[child], &v[child + 1]) {
            child += 1;
        }
        v[hole] = v[child];
        hole = child;
        if child > (len - 2) / 2 {
            return hole;
        }
    }
}

/// std::__pop_heap on v[..len]
fn pop_heap_len<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], len: usize, comp: &mut C) {
    if len > 1 {
        let top = v[0];
        let hole = floyd_sift_down(v, comp, len);
        let last = len - 1;
        if hole == last {
            v[hole] = top;
        } else {
            v[hole] = v[last];
            v[last] = top;
            sift_up(v, hole + 1, comp);
        }
    }
}

pub fn make_heap<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C) {
    let n = v.len();
    if n > 1 {
        let mut start = (n - 2) / 2;
        loop {
            sift_down(v, comp, n, start);
            if start == 0 {
                break;
            }
            start -= 1;
        }
    }
}

pub fn sort_heap<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C) {
    let mut n = v.len();
    while n > 1 {
        pop_heap_len(v, n, comp);
        n -= 1;
    }
}

/// std::push_heap after v.push(x)
pub fn push_heap<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C) {
    let n = v.len();
    sift_up(v, n, comp);
}

/// std::pop_heap: moves the top to the back
pub fn pop_heap<T: Copy, C: FnMut(&T, &T) -> bool>(v: &mut [T], comp: &mut C) {
    let n = v.len();
    pop_heap_len(v, n, comp);
}

/// std::partial_sort(first, first + middle, last, comp)
pub fn partial_sort<T: Copy>(v: &mut [T], middle: usize, mut comp: impl FnMut(&T, &T) -> bool) {
    if middle == 0 {
        return;
    }
    make_heap(&mut v[..middle], &mut comp);
    for i in middle..v.len() {
        if comp(&v[i], &v[0]) {
            v.swap(i, 0);
            sift_down(v, &mut comp, middle, 0);
        }
    }
    sort_heap(&mut v[..middle], &mut comp);
}

/// std::priority_queue with libc++'s heap operations; `less(a, b)` is the
/// queue's comparator (the top is a maximal element).
pub struct PriorityQueue<T: Copy, C: FnMut(&T, &T) -> bool> {
    pub data: Vec<T>,
    less: C,
}

impl<T: Copy, C: FnMut(&T, &T) -> bool> PriorityQueue<T, C> {
    pub fn new(less: C) -> Self {
        PriorityQueue { data: Vec::new(), less }
    }
    pub fn push(&mut self, x: T) {
        self.data.push(x);
        push_heap(&mut self.data, &mut self.less);
    }
    pub fn top(&self) -> T {
        self.data[0]
    }
    pub fn pop(&mut self) {
        pop_heap(&mut self.data, &mut self.less);
        self.data.pop();
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// std::partition (libc++'s bidirectional algorithm); returns the number
/// of elements satisfying pred
pub fn partition<T: Copy>(v: &mut [T], mut pred: impl FnMut(&T) -> bool) -> usize {
    let (mut first, mut last) = (0, v.len());
    loop {
        loop {
            if first == last {
                return first;
            }
            if !pred(&v[first]) {
                break;
            }
            first += 1;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if pred(&v[last]) {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

/// std::upper_bound(first, last, value, comp), given comp(value, element)
pub fn upper_bound<T>(v: &[T], mut comp: impl FnMut(&T) -> bool) -> usize {
    let (mut first, mut len) = (0usize, v.len());
    while len != 0 {
        let half = len / 2;
        let mid = first + half;
        if comp(&v[mid]) {
            len = half;
        } else {
            first = mid + 1;
            len -= half + 1;
        }
    }
    first
}
