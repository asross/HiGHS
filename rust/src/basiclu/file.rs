//! Doubly linked lists (lu_list.h) and the data file (lu_file.c).
//!
//! Lists: maintain nelem elements in nlist doubly linked lists, each element
//! in at most one list. flink/blink hold nelem+nlist entries: the leading
//! nelem store links, the trailing nlist store heads (flink[nelem+j] is the
//! first, blink[nelem+j] the last element of list j). The last element of a
//! list links forward to its head, the first backward to its head; empty
//! lists and elements in no list point to themselves. Optionally
//! min_list >= 1 is kept such that lists 1..min_list-1 are empty (list 0 is
//! not covered). See I. Maros, Computational Techniques of the Simplex Method,
//! section 5.5.
//!
//! File: lines of (index,value) pairs, each line contiguous in memory, lines
//! in any order with gaps between them. begin[k]/end[k] delimit line k <
//! nlines, begin[nlines] is the start of unused space and end[nlines] the
//! file size. next/prev link the lines in memory order (next[nlines] and
//! prev[nlines] are the first and last line).

use crate::util::fma::ClangFma;

use super::Int;

/// lu_list_init: initialize all lists to empty; returns the initial
/// min_list
pub(crate) fn list_init(flink: &mut [Int], blink: &mut [Int], nelem: Int, nlist: Int) -> Int {
    for i in 0..(nelem + nlist) as usize {
        flink[i] = i as Int;
        blink[i] = i as Int;
    }
    nlist.max(1)
}

/// lu_list_add: append `elem` (in no list yet) to `list`; if list > 0
/// lower `min_list` to list
#[inline]
pub(crate) fn list_add(
    elem: Int,
    list: Int,
    flink: &mut [Int],
    blink: &mut [Int],
    nelem: Int,
    min_list: Option<&mut Int>,
) {
    debug_assert!(flink[elem as usize] == elem);
    debug_assert!(blink[elem as usize] == elem);
    let head = (nelem + list) as usize;
    let temp = blink[head];
    blink[head] = elem;
    blink[elem as usize] = temp;
    flink[temp as usize] = elem;
    flink[elem as usize] = nelem + list;
    if let Some(min_list) = min_list {
        if list > 0 && list < *min_list {
            *min_list = list;
        }
    }
}

/// lu_list_remove: remove `elem` from its list (no-op if in none)
#[inline]
pub(crate) fn list_remove(flink: &mut [Int], blink: &mut [Int], elem: Int) {
    let e = elem as usize;
    let (f, b) = (flink[e], blink[e]);
    flink[b as usize] = f;
    blink[f as usize] = b;
    flink[e] = elem;
    blink[e] = elem;
}

/// lu_list_move: remove `elem` from its list (if any) and add it to `list`
#[inline]
pub(crate) fn list_move(
    elem: Int,
    list: Int,
    flink: &mut [Int],
    blink: &mut [Int],
    nelem: Int,
    min_list: Option<&mut Int>,
) {
    list_remove(flink, blink, elem);
    list_add(elem, list, flink, blink, nelem, min_list);
}

/// lu_list_swap: swap elements e1 and e2, which both must be in a list. In
/// the same list their positions are swapped, otherwise each moves to the
/// other's list.
pub(crate) fn list_swap(flink: &mut [Int], blink: &mut [Int], e1: Int, e2: Int) {
    let (u1, u2) = (e1 as usize, e2 as usize);
    let e1next = flink[u1];
    let e2next = flink[u2];
    let e1prev = blink[u1];
    let e2prev = blink[u2];
    debug_assert!(e1next != e1 && e2next != e2); // must be in a list

    if e1next == e2 {
        flink[u2] = e1;
        blink[u1] = e2;
        flink[e1prev as usize] = e2;
        blink[u2] = e1prev;
        flink[u1] = e2next;
        blink[e2next as usize] = e1;
    } else if e2next == e1 {
        flink[u1] = e2;
        blink[u2] = e1;
        flink[u2] = e1next;
        blink[e1next as usize] = e2;
        flink[e2prev as usize] = e1;
        blink[u1] = e2prev;
    } else {
        flink[u2] = e1next;
        blink[e1next as usize] = e2;
        flink[e2prev as usize] = e1;
        blink[u1] = e2prev;
        flink[e1prev as usize] = e2;
        blink[u2] = e1prev;
        flink[u1] = e2next;
        blink[e2next as usize] = e1;
    }
}

/// lu_file_empty: initialize an empty file with `fmem` memory space
pub(crate) fn file_empty(
    nlines: Int,
    begin: &mut [Int],
    end: &mut [Int],
    next: &mut [Int],
    prev: &mut [Int],
    fmem: Int,
) {
    let n = nlines as usize;
    begin[n] = 0;
    end[n] = fmem;
    begin[..n].fill(0);
    end[..n].fill(0);
    for i in 0..n {
        next[i] = i as Int + 1;
        prev[i + 1] = i as Int;
    }
    next[n] = 0;
    prev[0] = nlines;
}

/// lu_file_reappend: move `line` to the file end and add `extra_space`
/// elements room. The file must have at least length(line) + extra_space
/// elements free space.
#[allow(clippy::too_many_arguments)]
pub(crate) fn file_reappend(
    line: Int,
    nlines: Int,
    begin: &mut [Int],
    end: &mut [Int],
    next: &mut [Int],
    prev: &mut [Int],
    index: &mut [Int],
    value: &mut [f64],
    extra_space: Int,
) {
    let n = nlines as usize;
    let l = line as usize;
    let fmem = end[n];
    let mut used = begin[n];
    let ibeg = begin[l]; // old beginning of line
    let iend = end[l];
    begin[l] = used; // new beginning of line
    debug_assert!(iend - ibeg <= fmem - used);
    for pos in ibeg as usize..iend as usize {
        index[used as usize] = index[pos];
        value[used as usize] = value[pos];
        used += 1;
    }
    end[l] = used;
    debug_assert!(fmem - used >= extra_space);
    used += extra_space;
    begin[n] = used; // beginning of unused space
    list_move(line, 0, next, prev, nlines, None);
}

/// lu_file_compress: compress the file to reuse memory gaps, keeping the
/// order of lines. To each line with nz entries add stretch*nz+pad elements
/// extra space, chopped if it would overlap the following line. Returns
/// the number of entries in the file.
#[allow(clippy::too_many_arguments)]
pub(crate) fn file_compress(
    nlines: Int,
    begin: &mut [Int],
    end: &mut [Int],
    next: &[Int],
    index: &mut [Int],
    value: &mut [f64],
    stretch: f64,
    pad: Int,
) -> Int {
    let n = nlines as usize;
    let mut nz = 0;
    let mut used: Int = 0;
    let mut extra_space: Int = 0;
    let mut i = next[n];
    while i < nlines {
        // move line i
        let iu = i as usize;
        let ibeg = begin[iu];
        let iend = end[iu];
        debug_assert!(ibeg >= used);
        used += extra_space;
        if used > ibeg {
            used = ibeg; // chop extra space added before
        }
        begin[iu] = used;
        for pos in ibeg as usize..iend as usize {
            index[used as usize] = index[pos];
            value[used as usize] = value[pos];
            used += 1;
        }
        end[iu] = used;
        extra_space = stretch.mul_add_c((iend - ibeg) as f64, pad as f64) as Int;
        nz += iend - ibeg;
        i = next[iu];
    }
    debug_assert!(used <= begin[n]);
    used += extra_space;
    if used > begin[n] {
        used = begin[n]; // never use more space than before
    }
    begin[n] = used;
    nz
}
