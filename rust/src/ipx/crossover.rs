//! Crossover (crossover.h/.cc): from a complementary primal-dual point and
//! a basis to a basic solution, by dual pushes of basic variables with
//! nonzero dual and primal pushes of nonbasic variables between their
//! bounds.

use super::basis::{copy_basic, Basis};
use super::control::Control;
use super::fmt::{fmt, sci2, textline};
use super::indexed_vector::IndexedVector;
use super::lu::LuResult;
use super::model::{dual_infeasibility, dual_residual, primal_infeasibility, primal_residual};
use super::utils::sortperm;
use super::{
    cmax, cmin, Info, Int, ERROR_TIME_INTERRUPT, ERROR_USER_INTERRUPT, STATUS_FAILED,
    STATUS_OPTIMAL, STATUS_TIME_LIMIT, STATUS_USER_INTERRUPT,
};
use std::time::Instant;

const PIVOT_ZERO_TOL: f64 = 1e-5;

pub struct Crossover<'a> {
    control: &'a Control,
    pub primal_pushes: Int,
    pub dual_pushes: Int,
    pub primal_pivots: Int,
    pub dual_pivots: Int,
    pub time_primal: f64,
    pub time_dual: f64,
}

/// Sets status_crossover from errflag after a push phase. (A user interrupt
/// sets status_ipm, as the C++ code does.)
fn push_status(info: &mut Info) {
    if info.errflag == ERROR_USER_INTERRUPT {
        info.errflag = 0;
        info.status_ipm = STATUS_USER_INTERRUPT;
    } else if info.errflag == ERROR_TIME_INTERRUPT {
        info.errflag = 0;
        info.status_crossover = STATUS_TIME_LIMIT;
    } else if info.errflag != 0 {
        info.status_crossover = STATUS_FAILED;
    } else {
        info.status_crossover = STATUS_OPTIMAL;
    }
}

impl<'a> Crossover<'a> {
    pub fn new(control: &'a Control) -> Self {
        Crossover {
            control,
            primal_pushes: 0,
            dual_pushes: 0,
            primal_pivots: 0,
            dual_pivots: 0,
            time_primal: 0.0,
            time_dual: 0.0,
        }
    }

    /// Runs the dual push phase in increasing order of weights, then the
    /// primal push phase in decreasing order (index order without weights).
    pub fn push_all(
        &mut self,
        basis: &mut Basis,
        x: &mut [f64],
        y: &mut [f64],
        z: &mut [f64],
        weights: Option<&[f64]>,
        info: &mut Info,
    ) -> LuResult<()> {
        let model_rc = basis.model();
        let m = model_rc.rows();
        let n = model_rc.cols();
        let perm = sortperm(n + m, weights, false);

        self.control.log(&format!(
            "{}{}\n{}{}\n",
            textline("Primal residual before push phase:"),
            sci2(primal_residual(basis.model(), x)),
            textline("Dual residual before push phase:"),
            sci2(dual_residual(basis.model(), y, z))
        ));

        // Run dual push phase.
        let dual_superbasics: Vec<usize> = perm
            .iter()
            .map(|&j| j as usize)
            .filter(|&j| basis.is_basic(j) && z[j] != 0.0)
            .collect();
        self.control.log(&format!(
            "{}{}\n",
            textline("Number of dual pushes required:"),
            dual_superbasics.len()
        ));
        let sign_restrict: Vec<i32> = {
            let (lb, ub) = (basis.model().lb(), basis.model().ub());
            (0..n + m)
                .map(|j| i32::from(x[j] != ub[j]) | (i32::from(x[j] != lb[j]) << 1))
                .collect()
        };
        self.push_dual(basis, y, z, &dual_superbasics, &sign_restrict, info)?;
        debug_assert_eq!(dual_infeasibility(basis.model(), x, z), 0.0);
        if info.status_crossover != STATUS_OPTIMAL {
            return Ok(());
        }

        // Run primal push phase. Because z[j]==0 for all basic variables,
        // none of the primal variables is fixed at its bound.
        let primal_superbasics: Vec<usize> = {
            let (lb, ub) = (basis.model().lb(), basis.model().ub());
            perm.iter()
                .rev()
                .map(|&j| j as usize)
                .filter(|&j| {
                    basis.is_nonbasic(j)
                        && x[j] != lb[j]
                        && x[j] != ub[j]
                        && !(lb[j].is_infinite() && ub[j].is_infinite() && x[j] == 0.0)
                })
                .collect()
        };
        self.control.log(&format!(
            "{}{}\n",
            textline("Number of primal pushes required:"),
            primal_superbasics.len()
        ));
        self.push_primal(basis, x, &primal_superbasics, None, info)?;
        debug_assert_eq!(primal_infeasibility(basis.model(), x), 0.0);
        if info.status_crossover != STATUS_OPTIMAL {
            return Ok(());
        }

        if self.control.debug(1) {
            self.control.debug_out(
                1,
                &format!(
                    "{}{}\n{}{}\n",
                    textline("Primal residual after push phase:"),
                    sci2(primal_residual(basis.model(), x)),
                    textline("Dual residual after push phase:"),
                    sci2(dual_residual(basis.model(), y, z))
                ),
            );
        }
        info.status_crossover = STATUS_OPTIMAL;
        Ok(())
    }

    /// Pushes the nonbasic variables to a bound (or zero if free) by
    /// pivoting them into the basis when blocked. Variables with
    /// fixed_at_bound[j] must stay at their bound.
    pub fn push_primal(
        &mut self,
        basis: &mut Basis,
        x: &mut [f64],
        variables: &[usize],
        fixed_at_bound: Option<&[bool]>,
        info: &mut Info,
    ) -> LuResult<()> {
        let timer = Instant::now();
        let model = basis.model();
        let m = model.rows();
        let n = model.cols();
        let lb = model.lb().to_vec();
        let ub = model.ub().to_vec();
        let mut ftran = IndexedVector::new(m);
        let feastol = if model.dualized() {
            self.control.dfeasibility_tol()
        } else {
            self.control.pfeasibility_tol()
        };
        self.primal_pushes = 0;
        self.primal_pivots = 0;

        // Check that variables are nonbasic and that x satisfies bound
        // condition.
        if variables.iter().any(|&j| !basis.is_nonbasic(j)) {
            return Err("invalid variable in Crossover::PushPrimal".into());
        }
        for j in 0..n + m {
            if x[j] < lb[j] || x[j] > ub[j] {
                return Err("bound condition violated in Crossover::PushPrimal".into());
            }
            if let Some(f) = fixed_at_bound {
                if f[j] && x[j] != lb[j] && x[j] != ub[j] {
                    return Err("bound condition violated in Crossover::PushPrimal".into());
                }
            }
        }

        // Maintain a copy of primal basic variables and their bounds for
        // faster ratio test. Fixed-at-bound variables are handled by setting
        // their bounds equal.
        let mut xbasic = copy_basic(x, basis);
        let mut lbbasic = copy_basic(&lb, basis);
        let mut ubbasic = copy_basic(&ub, basis);
        if let Some(f) = fixed_at_bound {
            for p in 0..m {
                let j = basis.at(p);
                if f[j] {
                    lbbasic[p] = x[j];
                    ubbasic[p] = x[j];
                }
            }
        }

        self.control.reset_print_interval();
        let mut next = 0;
        while next < variables.len() {
            info.errflag = self.control.interrupt_check(-1);
            if info.errflag != 0 {
                break;
            }

            let jn = variables[next];
            if x[jn] == lb[jn] || x[jn] == ub[jn] || (x[jn] == 0.0 && lb[jn].is_infinite() && ub[jn].is_infinite()) {
                // nothing to do
                next += 1;
                continue;
            }
            // Choose bound to push to. If the variable has two finite
            // bounds, move to the nearer. If it has none, move to zero.
            let mut move_to = 0.0;
            if lb[jn].is_finite() && ub[jn].is_finite() {
                move_to = if x[jn] - lb[jn] <= ub[jn] - x[jn] { lb[jn] } else { ub[jn] };
            } else if lb[jn].is_finite() {
                move_to = lb[jn];
            } else if ub[jn].is_finite() {
                move_to = ub[jn];
            }

            // A full step is such that x[jn]-step is at its bound.
            let mut step = x[jn] - move_to;

            basis.solve_for_update(jn, Some(&mut ftran))?;
            let (pblock, block_at_lb) = primal_ratio_test(&xbasic, &ftran, &lbbasic, &ubbasic, step, feastol);
            let jb = if pblock >= 0 { basis.at(pblock as usize) } else { usize::MAX };

            // If step was blocked, update basis and compute step size.
            if pblock >= 0 {
                let pblock = pblock as usize;
                let pivot = ftran[pblock];
                if pivot.abs() < 1e-4 {
                    self.control
                        .debug_out(3, &format!(" |pivot| = {}\n", sci2(pivot.abs())));
                }
                let (err, exchanged) = basis.exchange_if_stable(jb, jn, pivot, -1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    if self.control.debug(1) {
                        let s = basis.min_singular_value()?;
                        self.control.debug_out(
                            1,
                            &format!("{}{}\n", textline("Minimum singular value of basis matrix:"), sci2(s)),
                        );
                    }
                    break;
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                self.primal_pivots += 1;
                // We must use lbbasic[pblock] and ubbasic[pblock] (and not
                // lb[jb] and ub[jb]) so that step is 0.0 if a fixed-at-bound
                // variable blocked.
                step = if block_at_lb {
                    (lbbasic[pblock] - xbasic[pblock]) / ftran[pblock]
                } else {
                    (ubbasic[pblock] - xbasic[pblock]) / ftran[pblock]
                };
            }
            // Update solution.
            if step != 0.0 {
                ftran.for_each_nonzero(|p, pivot| {
                    xbasic[p] = step.mul_add(pivot, xbasic[p]);
                    xbasic[p] = cmax(xbasic[p], lbbasic[p]);
                    xbasic[p] = cmin(xbasic[p], ubbasic[p]);
                });
                x[jn] -= step;
            }
            if pblock >= 0 {
                let pblock = pblock as usize;
                // make clean
                x[jb] = if block_at_lb { lbbasic[pblock] } else { ubbasic[pblock] };
                // Update copy of basic variables and bounds. Note: jn cannot
                // be a fixed-at-bound variable since it was pushed to a
                // bound.
                xbasic[pblock] = x[jn];
                lbbasic[pblock] = lb[jn];
                ubbasic[pblock] = ub[jn];
            } else {
                x[jn] = move_to; // make clean
            }

            self.primal_pushes += 1;
            next += 1;
            self.control.interval_log(&format!(
                " {} primal pushes remaining ({} pivots)\n",
                fmt(variables.len() - next, 8),
                fmt(self.primal_pivots, 7)
            ));
        }
        for p in 0..m {
            x[basis.at(p)] = xbasic[p];
        }

        push_status(info);
        self.time_primal = timer.elapsed().as_secs_f64();
        Ok(())
    }

    /// Pushes the duals of basic variables to zero by pivoting them out of
    /// the basis when blocked. sign_restrict[j] & 1 (& 2): z[j] must stay
    /// >= 0 (<= 0).
    pub fn push_dual(
        &mut self,
        basis: &mut Basis,
        y: &mut [f64],
        z: &mut [f64],
        variables: &[usize],
        sign_restrict: &[i32],
        info: &mut Info,
    ) -> LuResult<()> {
        let timer = Instant::now();
        let m = basis.model().rows();
        let n = basis.model().cols();
        let mut btran = IndexedVector::new(m);
        let mut row = IndexedVector::new(n + m);
        let feastol = if basis.model().dualized() {
            self.control.pfeasibility_tol()
        } else {
            self.control.dfeasibility_tol()
        };
        self.dual_pushes = 0;
        self.dual_pivots = 0;

        // Check that variables are basic and that z satisfies sign
        // condition.
        if variables.iter().any(|&j| !basis.is_basic(j)) {
            return Err("invalid variable in Crossover::PushDual".into());
        }
        for j in 0..n + m {
            if ((sign_restrict[j] & 1) != 0 && z[j] < 0.0) || ((sign_restrict[j] & 2) != 0 && z[j] > 0.0) {
                return Err("sign condition violated in Crossover::PushDual".into());
            }
        }

        self.control.reset_print_interval();
        let mut next = 0;
        while next < variables.len() {
            info.errflag = self.control.interrupt_check(-1);
            if info.errflag != 0 {
                break;
            }

            let jb = variables[next];
            if z[jb] == 0.0 {
                // nothing to do
                next += 1;
                continue;
            }
            // The update operation applied below is
            // y := y + step*btran, z := z - step*row, z[jb] := z[jb] - step,
            // where row is the tableau row for variable jb. In exact
            // arithmetic this leaves A'y+z unchanged.
            basis.tableau_row(jb, &mut btran, &mut row, false)?;
            let mut step = z[jb];
            let jn = dual_ratio_test(z, &row, sign_restrict, step, feastol);

            // If step was blocked, update basis and compute step size.
            if jn >= 0 {
                let jn = jn as usize;
                let pivot = row[jn];
                if pivot.abs() < 1e-4 {
                    self.control
                        .debug_out(3, &format!(" |pivot| = {}\n", sci2(pivot.abs())));
                }
                let (err, exchanged) = basis.exchange_if_stable(jb, jn, pivot, 1)?;
                info.errflag = err;
                if info.errflag != 0 {
                    if self.control.debug(1) {
                        let s = basis.min_singular_value()?;
                        self.control.debug_out(
                            1,
                            &format!("{}{}\n", textline("Minimum singular value of basis matrix:"), sci2(s)),
                        );
                    }
                    break;
                }
                if !exchanged {
                    // factorization was unstable, try again
                    continue;
                }
                self.dual_pivots += 1;
                step = z[jn] / row[jn];
            }
            // Update solution.
            if step != 0.0 {
                btran.for_each_nonzero(|i, x| {
                    y[i] = step.mul_add(x, y[i]);
                });
                row.for_each_nonzero(|j, pivot| {
                    z[j] = (-step).mul_add(pivot, z[j]);
                    if sign_restrict[j] & 1 != 0 {
                        z[j] = cmax(z[j], 0.0);
                    }
                    if sign_restrict[j] & 2 != 0 {
                        z[j] = cmin(z[j], 0.0);
                    }
                });
                z[jb] -= step;
            }
            if jn >= 0 {
                z[jn as usize] = 0.0; // make clean
            }

            self.dual_pushes += 1;
            next += 1;
            self.control.interval_log(&format!(
                " {} dual pushes remaining ({} pivots)\n",
                fmt(variables.len() - next, 8),
                fmt(self.dual_pivots, 7)
            ));
        }

        push_status(info);
        self.time_dual = timer.elapsed().as_secs_f64();
        Ok(())
    }
}

/// Primal ratio test: the position of the blocking basic variable (-1 if
/// none) and whether it blocks at its lower bound. A first pass determines
/// the maximum step exploiting the feasibility tolerance; a second pass
/// chooses the maximum pivot among all that block within that step.
fn primal_ratio_test(
    xbasic: &[f64],
    ftran: &IndexedVector,
    lbbasic: &[f64],
    ubbasic: &[f64],
    mut step: f64,
    feastol: f64,
) -> (Int, bool) {
    let mut pblock: Int = -1;
    let mut block_at_lb = true;

    // First pass: determine maximum step size exploiting feasibility tol.
    ftran.for_each_nonzero(|p, pivot| {
        if pivot.abs() > PIVOT_ZERO_TOL {
            // test block at lower bound
            if step.mul_add(pivot, xbasic[p]) < lbbasic[p] - feastol {
                step = (lbbasic[p] - xbasic[p] - feastol) / pivot;
                pblock = p as Int;
                block_at_lb = true;
            }
            // test block at upper bound
            if step.mul_add(pivot, xbasic[p]) > ubbasic[p] + feastol {
                step = (ubbasic[p] - xbasic[p] + feastol) / pivot;
                pblock = p as Int;
                block_at_lb = false;
            }
        }
    });

    // If the step was not blocked, we are done.
    if pblock < 0 {
        return (pblock, block_at_lb);
    }

    // Second pass: choose maximum pivot among all that block within step.
    pblock = -1;
    let mut max_pivot = PIVOT_ZERO_TOL;
    ftran.for_each_nonzero(|p, pivot| {
        if pivot.abs() > max_pivot {
            // test block at lower bound
            if step * pivot < 0.0 {
                let step_p = (lbbasic[p] - xbasic[p]) / pivot;
                if step_p.abs() <= step.abs() {
                    pblock = p as Int;
                    block_at_lb = true;
                    max_pivot = pivot.abs();
                }
            }
            // test block at upper bound
            if step * pivot > 0.0 {
                let step_p = (ubbasic[p] - xbasic[p]) / pivot;
                if step_p.abs() <= step.abs() {
                    pblock = p as Int;
                    block_at_lb = false;
                    max_pivot = pivot.abs();
                }
            }
        }
    });
    (pblock, block_at_lb)
}

/// Dual ratio test: the blocking nonbasic variable (-1 if none), in two
/// passes as the primal ratio test.
fn dual_ratio_test(z: &[f64], row: &IndexedVector, sign_restrict: &[i32], mut step: f64, feastol: f64) -> Int {
    let mut jblock: Int = -1;

    // First pass: determine maximum step size exploiting feasibility tol.
    row.for_each_nonzero(|j, pivot| {
        if pivot.abs() > PIVOT_ZERO_TOL {
            if (sign_restrict[j] & 1) != 0 && (-step).mul_add(pivot, z[j]) < -feastol {
                step = (z[j] + feastol) / pivot;
                jblock = j as Int;
            }
            if (sign_restrict[j] & 2) != 0 && (-step).mul_add(pivot, z[j]) > feastol {
                step = (z[j] - feastol) / pivot;
                jblock = j as Int;
            }
        }
    });

    // If step was not blocked, we are done.
    if jblock < 0 {
        return jblock;
    }

    // Second pass: choose maximum pivot among all that block within step.
    jblock = -1;
    let mut max_pivot = PIVOT_ZERO_TOL;
    row.for_each_nonzero(|j, pivot| {
        if pivot.abs() > max_pivot && (z[j] / pivot).abs() <= step.abs() {
            if (sign_restrict[j] & 1) != 0 && step * pivot > 0.0 {
                jblock = j as Int;
                max_pivot = pivot.abs();
            }
            if (sign_restrict[j] & 2) != 0 && step * pivot < 0.0 {
                jblock = j as Int;
                max_pivot = pivot.abs();
            }
        }
    });
    jblock
}
