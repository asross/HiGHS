//! starting_basis.h/.cc: constructs the starting basis for the main IPM
//! from the iterate after the initial iterations, and adjusts the iterate
//! to dependent rows and columns.

use super::basis::{BasicStatus, Basis};
use super::iterate::Iterate;
use super::lu::LuResult;
use super::sparse_matrix::scatter_column;
use super::Info;
use std::time::Instant;

/// Removes the components of the iterate along dependent columns (free
/// variables that could not become basic: fixed at zero) and dependent
/// rows (equality slacks that stay basic: their dual fixed at zero).
fn postprocess_dependencies(iterate: &mut Iterate, basis: &Basis, info: &Info) -> LuResult<()> {
    let model = basis.model();
    let m = model.rows();
    let n = model.cols();
    let (ai, lb, ub) = (model.ai(), model.lb(), model.ub());
    let mut dependent_rows = Vec::new();
    let mut dependent_cols = Vec::new();
    let mut dx = vec![0.0; n + m];
    let mut dy = vec![0.0; m];

    if info.dependent_cols > 0 {
        let mut dxbasic = vec![0.0; m];
        let x = iterate.x();
        for j in 0..n {
            if lb[j].is_infinite() && ub[j].is_infinite() && basis.is_nonbasic(j) {
                dx[j] = -x[j];
                scatter_column(ai, j, x[j], &mut dxbasic);
                dependent_cols.push(j);
            }
        }
        basis.solve_dense_inplace(&mut dxbasic, b'N')?;
        for p in 0..m {
            dx[basis.at(p)] = dxbasic[p];
        }
    }

    if info.dependent_rows > 0 {
        let y = iterate.y();
        for p in 0..m {
            let j = basis.at(p);
            if j >= n && lb[j] == ub[j] {
                dy[p] = -y[j - n];
                dependent_rows.push(j - n);
            }
        }
        basis.solve_dense_inplace(&mut dy, b'T')?;
        for &i in &dependent_rows {
            dy[i] = -y[i]; // would be already in exact arithmetic
        }
    }

    iterate.update(1.0, Some(&dx), None, None, 1.0, Some(&dy), None, None);

    for &j in &dependent_cols {
        iterate.make_fixed_at(j, 0.0);
    }
    for &i in &dependent_rows {
        iterate.make_implied_eq(n + i); // sets zl[n+i] = zu[n+i] = 0
    }
    Ok(())
}

/// StartingBasis: constructs a basis from the scaling factors of the
/// iterate (fixed variables weight 0, free variables infinite), frees or
/// fixes the variables with zero or infinite weight, and postprocesses
/// dependencies.
pub fn starting_basis(iterate: &mut Iterate, basis: &mut Basis, info: &mut Info) -> LuResult<()> {
    let m = basis.model().rows();
    let n = basis.model().cols();
    let mut colscale = vec![0.0; n + m];
    info.errflag = 0;
    let timer = Instant::now();

    {
        let (lb, ub) = (basis.model().lb(), basis.model().ub());
        for j in 0..n + m {
            colscale[j] = iterate.scaling_factor(j);
            if lb[j] == ub[j] {
                colscale[j] = 0.0;
            }
        }
    }
    basis.construct_basis_from_weights(&colscale, info)?;
    if info.errflag != 0 {
        return Ok(());
    }

    for j in 0..n + m {
        if colscale[j] == 0.0 || colscale[j].is_infinite() {
            if basis.is_basic(j) {
                basis.free_basic_variable(j);
            } else {
                basis.fix_nonbasic_variable(j);
            }
        }
    }

    for j in 0..n + m {
        let (lbj, ubj) = (basis.model().lb()[j], basis.model().ub()[j]);
        if lbj == ubj && basis.status_of(j) == BasicStatus::NonbasicFixed {
            iterate.make_fixed_at(j, lbj);
        }
    }

    postprocess_dependencies(iterate, basis, info)?;
    info.time_starting_basis += timer.elapsed().as_secs_f64();
    Ok(())
}
