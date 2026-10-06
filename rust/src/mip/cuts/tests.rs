//! Checks of the sorting algorithms against libc++/pdqsort (golden_cuts.cpp
//! prints the same checksums) and of GFkSolve on a small system.

use super::sort::*;
use crate::util::random::HighsRandom;

fn mix(h: &mut u64, v: u64) {
    *h = (*h ^ v).wrapping_mul(1099511628211);
}

/// Pairs (key, id) with many equal keys, sorted by key only
fn tie_data(r: &mut HighsRandom, n: usize, keys: i32) -> Vec<(i32, i32)> {
    (0..n).map(|i| (r.integer_below(keys), i as i32)).collect()
}

#[test]
fn sorts_match_cpp() {
    let mut h = 1469598103934665603u64;
    let mut r = HighsRandom::new(99);
    for round in 0..300 {
        let n = (round * 7) % 400 + 1;
        let keys = (round % 13 + 1) as i32;
        let base = tie_data(&mut r, n, keys);
        let mut v = base.clone();
        pdqsort(&mut v, |a, b| a.0 < b.0);
        v.iter().for_each(|x| mix(&mut h, x.1 as u64));
        let mut v = base.clone();
        pdqsort_branchless(&mut v, |a, b| a.0 > b.0);
        v.iter().for_each(|x| mix(&mut h, x.1 as u64));
        let mut v = base.clone();
        let mid = n / 3;
        partial_sort(&mut v, mid, |a, b| a.0 < b.0);
        v.iter().for_each(|x| mix(&mut h, x.1 as u64));
        let mut v = base.clone();
        let p = partition(&mut v, |a| a.0 % 2 == 0);
        mix(&mut h, p as u64);
        v.iter().for_each(|x| mix(&mut h, x.1 as u64));
        let mut q = PriorityQueue::new(|a: &(i32, i32), b: &(i32, i32)| a.0 > b.0);
        for &x in &base {
            q.push(x);
            if x.1 % 3 == 0 {
                mix(&mut h, q.top().1 as u64);
                q.pop();
            }
        }
        while !q.is_empty() {
            mix(&mut h, q.top().1 as u64);
            q.pop();
        }
    }
    // golden_cuts.cpp
    assert_eq!(h, GOLDEN_SORTS);
}

const GOLDEN_SORTS: u64 = 9754718482623447556;

#[test]
fn ldexp_matches_libm() {
    extern "C" {
        fn ldexp(x: f64, e: i32) -> f64;
    }
    let mut r = HighsRandom::new(5);
    for _ in 0..200000 {
        let x = (r.fraction() - 0.5) * 2f64.powi(r.integer_below(200) - 100);
        let e = r.integer_below(2200) - 1100;
        // SAFETY: libm
        let want = unsafe { ldexp(x, e) };
        assert_eq!(super::integers::ldexp(x, e).to_bits(), want.to_bits(), "{x} {e}");
    }
}

#[test]
fn gfk_solves_mod2() {
    use super::gfk::GfkSolve;
    // weights u of two constraint columns over rows 0..2 and the rhs row 3:
    // u0 (e0 + e1 + e3) + u1 (e0 + e1) = e3 (mod 2) needs u0 = u1 = 1
    let aval = [1i64, 1, 1, 1, 3];
    let aindex = [0, 1, 3, 0, 1];
    let astart = [0, 3, 5];
    let mut g = GfkSolve::default();
    g.from_csc(2, &aval, &aindex, &astart, 4);
    g.set_rhs(2, 3, 1);
    let mut sols = Vec::new();
    g.solve(2, |sol| {
        sol.sort();
        sols.push(sol.clone());
    });
    assert!(!sols.is_empty());
    for s in sols {
        let w: Vec<_> = s.iter().map(|e| (e.index, e.weight)).collect();
        assert_eq!(w, vec![(0, 1), (1, 1)]);
    }
}
