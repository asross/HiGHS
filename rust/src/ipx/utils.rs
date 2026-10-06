//! utils.h/.cc: norms and permutations of dense vectors.

use crate::util::fma::ClangFma;

use super::{cmax, Int};
use std::cmp::Ordering;

pub(crate) fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

pub(crate) fn onenorm(x: &[f64]) -> f64 {
    let mut norm = 0.0;
    for &xi in x {
        norm += xi.abs();
    }
    norm
}

/// sqrt(x·x), summed as dot_blocked(x, x, 0.0, 4)
pub(crate) fn twonorm(x: &[f64]) -> f64 {
    dot_blocked(x, x, 0.0, 4).sqrt()
}

pub(crate) fn infnorm(x: &[f64]) -> f64 {
    let mut norm = 0.0;
    for &xi in x {
        norm = cmax(norm, xi.abs());
    }
    norm
}

/// x·y as dot_blocked(x, y, 0.0, 4) (Dot() at all its call sites but one,
/// see dot_fused)
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    dot_blocked(x, y, 0.0, 4)
}

/// acc + x·y as clang compiles the C++ reduction loops `acc += x[i]*y[i]`
/// of IPX: the source contracts into llvm.fmuladd, but the loop vectorizer
/// (or interleaver) splits fmuladd in an in-order reduction into a rounded
/// product and an addition. So the first (n/block)*block terms are added
/// unfused, in order, and the scalar remainder fused; with n < block all
/// terms are fused. block is VF*UF of the C++ loop (from the disassembly of
/// libhighs).
pub(crate) fn dot_blocked(x: &[f64], y: &[f64], acc: f64, block: usize) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let nb = if n >= block { n - n % block } else { 0 };
    let mut d = acc;
    for i in 0..nb {
        d += x[i] * y[i];
    }
    for i in nb..n {
        d = x[i].mul_add_c(y[i], d);
    }
    d
}

/// x·y with every term fused (Dot(Cstep,Cstep) in the unpreconditioned
/// CR method, which clang does not vectorize)
pub(crate) fn dot_fused(x: &[f64], y: &[f64]) -> f64 {
    let mut d = 0.0f64;
    for i in 0..x.len() {
        d = x[i].mul_add_c(y[i], d);
    }
    d
}

/// Index of an entry of maximum absolute value (the first one)
pub(crate) fn find_max_abs(x: &[f64]) -> usize {
    let mut xmax = 0.0;
    let mut imax = 0;
    for (i, &xi) in x.iter().enumerate() {
        if xi.abs() > xmax {
            xmax = xi.abs();
            imax = i;
        }
    }
    imax
}

/// lhs[permuted_index] = rhs
pub(crate) fn permute(permuted_index: &[Int], rhs: &[f64], lhs: &mut [f64]) {
    for (i, &p) in permuted_index.iter().enumerate() {
        lhs[p as usize] = rhs[i];
    }
}

/// lhs = rhs[permuted_index]
pub(crate) fn permute_back(permuted_index: &[Int], rhs: &[f64], lhs: &mut [f64]) {
    for (i, &p) in permuted_index.iter().enumerate() {
        lhs[i] = rhs[p as usize];
    }
}

pub(crate) fn inverse_perm(perm: &[Int]) -> Vec<Int> {
    let mut invperm = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        invperm[p as usize] = i as Int;
    }
    invperm
}

/// The permutation that puts values[0..m-1] in increasing (or, if reverse,
/// decreasing) order, ties broken by index as the C++ code compares
/// (value, index) pairs; this is a total order on non-NaN values, so any
/// sort gives pdqsort's result. Identity if values is None.
pub(crate) fn sortperm(m: usize, values: Option<&[f64]>, reverse: bool) -> Vec<Int> {
    let mut perm: Vec<Int> = (0..m as Int).collect();
    let Some(values) = values else {
        return perm;
    };
    let less = |i: &Int, j: &Int| -> Ordering {
        let (vi, vj) = (values[*i as usize], values[*j as usize]);
        // std::pair operator<: vi < vj || (!(vj < vi) && i < j)
        if vi < vj {
            Ordering::Less
        } else if vj < vi {
            Ordering::Greater
        } else {
            i.cmp(j)
        }
    };
    if reverse {
        perm.sort_unstable_by(|i, j| less(j, i));
    } else {
        perm.sort_unstable_by(less);
    }
    perm
}
