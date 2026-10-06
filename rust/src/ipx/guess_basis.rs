//! guess_basis.h/.cc: guesses a basis from column weights: free columns
//! first (with a sparse LU of their own), then singletons, then a maximum
//! matching on the remaining columns in decreasing order of weight; slack
//! columns fill the remaining rows.

use crate::util::fma::ClangFma;

use super::control::Control;
use super::fmt::textline;
use super::model::Model;
use super::sparse_matrix::SparseMatrix;
use super::sparse_utils::{augmenting_path, depth_first_search};
use super::utils::sortperm;
use super::{cmax, Int, LU_DEPENDENCY_TOL};

/// Pattern of L\AI[:,j] into pattern[top..m-1]; returns top.
fn compute_pattern(
    l: &SparseMatrix,
    ai: &SparseMatrix,
    j: usize,
    rownumber: &[Int],
    pattern: &mut [Int],
    pstack: &mut [Int],
    marked: &mut [Int],
    marker: Int,
) -> Int {
    let m = l.rows();
    let mut top = m;
    for p in ai.begin(j)..ai.end(j) {
        let i = ai.index(p);
        if marked[i] != marker {
            top = depth_first_search(
                i as Int,
                &l.colptr,
                &l.rowidx,
                Some(rownumber),
                top,
                pattern,
                marked,
                marker,
                pstack,
            );
        }
    }
    top
}

/// Values of L\AI[:,j] on the pattern; returns the position of the
/// largest entry that has not been pivotal (-1 if none).
fn compute_values(
    l: &SparseMatrix,
    ai: &SparseMatrix,
    j: usize,
    rownumber: &[Int],
    pattern: &[Int],
    top: Int,
    lhs: &mut [f64],
) -> Int {
    let m = l.rows() as usize;
    let top = top as usize;
    for t in top..m {
        lhs[pattern[t] as usize] = 0.0;
    }
    for p in ai.begin(j)..ai.end(j) {
        // scatter RHS into lhs
        lhs[ai.index(p)] = ai.value(p);
    }
    let mut lhsmax = 0.0; // maximum entry that has not been pivotal
    let mut imax: Int = -1; // corresponding position in lhs
    for t in top..m {
        let i = pattern[t] as usize;
        let temp = lhs[i];
        let k = rownumber[i];
        if temp != 0.0 {
            if k >= 0 {
                let k = k as usize;
                for p in l.begin(k)..l.end(k) {
                    let r = l.index(p);
                    lhs[r] = (-l.value(p)).mul_add_c(temp, lhs[r]);
                }
            } else if temp.abs() > lhsmax {
                lhsmax = temp.abs();
                imax = i as Int;
            }
        }
    }
    imax
}

fn process_free_columns(
    control: &Control,
    model: &Model,
    weights: &[f64],
    basis: &mut Vec<Int>,
    rownumber: &mut [Int],
    active: &mut [bool],
) {
    let m = model.rows();
    let n = model.cols();
    let ai = model.ai();
    let mut pattern = vec![0; m];
    let mut work = vec![0; m];
    let mut marked = vec![-1; m];
    let mut lhs = vec![0.0; m];
    let mut l = SparseMatrix::new(m as Int, 0);

    let mut num_free = 0;
    for j in 0..n + m {
        if weights[j] != f64::INFINITY {
            continue;
        }
        let top = compute_pattern(&l, ai, j, rownumber, &mut pattern, &mut work, &mut marked, j as Int);
        let imax = compute_values(&l, ai, j, rownumber, &pattern, top, &mut lhs);
        let pivot = if imax >= 0 { lhs[imax as usize] } else { 0.0 };
        if pivot.abs() > LU_DEPENDENCY_TOL {
            rownumber[imax as usize] = basis.len() as Int;
            basis.push(j as Int);
            for t in ai.begin(j)..ai.end(j) {
                let i = ai.index(t);
                if rownumber[i] < 0 && lhs[i] != 0.0 {
                    l.push_back(i as Int, lhs[i] / pivot);
                }
            }
            l.add_column();
            num_free += 1;
        }
        active[j] = false;
    }
    control.debug_out(
        1,
        &format!("{}{}\n", textline("Number of free variables in starting basis:"), num_free),
    );
}

fn process_singletons(
    control: &Control,
    model: &Model,
    weights: &[f64],
    basis: &mut Vec<Int>,
    rownumber: &mut [Int],
    active: &mut [bool],
) {
    let m = model.rows();
    let ai = model.ai();
    let at = model.ait();
    let mut num_singletons = 0;
    for i in 0..m {
        if rownumber[i] >= 0 {
            continue;
        }
        let mut rowmax = 0.0;
        let mut max_singleton = 0.0;
        let mut jsingleton: Int = -1;
        for p in at.begin(i)..at.end(i) {
            let j = at.index(p);
            if !active[j] {
                continue;
            }
            let a = at.value(p).abs() * weights[j];
            rowmax = cmax(rowmax, a);
            if a > max_singleton && ai.end(j) == ai.begin(j) + 1 {
                max_singleton = a;
                jsingleton = j as Int;
            }
        }
        if max_singleton > 0.0 && max_singleton >= 0.5 * rowmax {
            rownumber[i] = basis.len() as Int;
            basis.push(jsingleton);
            active[jsingleton as usize] = false;
            num_singletons += 1;
        }
    }
    control.debug_out(
        1,
        &format!("{}{}\n", textline("Number of singletons in starting basis:"), num_singletons),
    );
}

fn process_remaining(
    control: &Control,
    model: &Model,
    weights: &[f64],
    basis: &mut Vec<Int>,
    rownumber: &mut [Int],
    active: &[bool],
) {
    let m = model.rows();
    let n = model.cols();
    let ai = model.ai();
    let colperm = sortperm(n + m, Some(weights), true);

    let mut jmatch = vec![-1; m];
    for i in 0..m {
        if rownumber[i] >= 0 {
            jmatch[i] = -2;
        }
    }
    let mut marked = vec![-1; n + m];
    let ap = &ai.colptr;
    let aidx = &ai.rowidx;
    let mut cheap = ap[..n + m].to_vec();
    let mut work = vec![0; m];
    let mut work2 = vec![0; m + 1];
    let mut work3 = vec![0; m + 1];
    let mut num_matched = 0;
    let mut num_failed: Int = 0;

    for &j in &colperm {
        let ju = j as usize;
        if !active[ju] {
            continue;
        }
        if weights[ju] == 0.0 {
            break;
        }
        let matched = augmenting_path(
            j,
            ap,
            aidx,
            &mut jmatch,
            &mut cheap,
            &mut marked,
            &mut work,
            &mut work2,
            &mut work3,
        );
        if matched {
            basis.push(j);
            num_matched += 1;
        } else {
            num_failed += 1;
        }
        if num_failed >= 10 * (m as Int - basis.len() as Int) {
            break;
        }
    }
    for i in 0..m {
        if jmatch[i] >= 0 {
            rownumber[i] = m as Int;
        }
    }
    control.debug_out(
        1,
        &format!(
            "{}{}\n{}{}\n",
            textline("Number of other columns matched:"),
            num_matched,
            textline("Number of other columns failed:"),
            num_failed
        ),
    );
}

/// Returns m column indices of AI forming a guess for a basis, preferring
/// columns with larger weight.
pub fn guess_basis(control: &Control, model: &Model, colweights: &[f64]) -> Vec<Int> {
    let m = model.rows();
    let n = model.cols();
    let mut basis = Vec::new();
    let mut rownumber = vec![-1; m];
    let mut active = vec![true; n + m];

    process_free_columns(control, model, colweights, &mut basis, &mut rownumber, &mut active);
    process_singletons(control, model, colweights, &mut basis, &mut rownumber, &mut active);
    process_remaining(control, model, colweights, &mut basis, &mut rownumber, &active);

    for i in 0..m {
        if rownumber[i] < 0 {
            basis.push((n + i) as Int);
        }
    }
    basis
}
