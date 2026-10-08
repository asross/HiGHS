//! HighsRanging.cpp: getRangingData, the cost and bound ranging of an
//! optimal basis from the (unscaled) simplex data. The FTRAN of each
//! nonbasic column is a C++ callback (HEkk). writeRangingFile is in
//! writers.rs.
//!
//! clang fuses `objective + sense * x` (x a product, or a product times
//! the dual), `xi - delta * a_in` and `objective + sense * delta * dual`;
//! `xi + delta * a_in / a_out` divides last, so is not fused.

use super::ffi::RsMut;
use super::lp_utils::{cmax, cmin};
use super::{Log, LogType, Status, INF};
use crate::log_user;
use crate::util::fma::ClangFma;
use std::ffi::c_void;

/// An FTRANned column: the HVector's count, index and array
#[repr(C)]
pub struct CColumn {
    pub count: i32,
    pub index: *const i32,
    pub array: *const f64,
}

/// HighsRangingRecord, sized by C++
#[repr(C)]
pub struct CRecord {
    pub value: RsMut<f64>,
    pub objective: RsMut<f64>,
    pub in_var: RsMut<i32>,
    pub ou_var: RsMut<i32>,
}

/// The simplex data of getRangingData and the ranging to fill
#[repr(C)]
pub struct CRanging {
    pub log: Log,
    pub optimal: bool,
    pub initialised_for_solve: bool,
    pub num_col: i32,
    pub num_row: i32,
    /// -1 for a maximization
    pub sense: i32,
    pub objective: f64,
    pub work_value: RsMut<f64>,
    pub work_dual: RsMut<f64>,
    pub work_cost: RsMut<f64>,
    pub work_lower: RsMut<f64>,
    pub work_upper: RsMut<f64>,
    pub base_value: RsMut<f64>,
    pub base_lower: RsMut<f64>,
    pub base_upper: RsMut<f64>,
    pub nonbasic_flag: RsMut<i8>,
    pub nonbasic_move: RsMut<i8>,
    pub basic_index: RsMut<i32>,
    /// The updated column of variable j (collectAj and ftran)
    pub ftran: unsafe extern "C" fn(*mut c_void, i32, *mut CColumn),
    pub ctx: *mut c_void,
    /// col_cost_up/dn (num_col + num_row), col_bound_up/dn (num_col),
    /// row_bound_up/dn (num_row)
    pub out: [CRecord; 6],
}

/// infProduct
fn inf_product(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value * INF
    }
}

/// possInfProduct
fn poss_inf_product(poss_inf: f64, value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        poss_inf * value
    }
}

/// getRangingData after ranging.clear() and unscaleSimplex
///
/// # Safety
/// The arrays of `c` valid for the call
pub unsafe fn get_ranging_data(c: &CRanging) -> Status {
    if !c.optimal {
        log_user!(c.log, LogType::Error, "Cannot get ranging without an optimal solution\n");
        return Status::Error;
    }
    if !c.initialised_for_solve {
        log_user!(c.log, LogType::Error, "Cannot get ranging without a valid Simplex instance\n");
        return Status::Error;
    }
    let value_ = c.work_value.get();
    let dual_ = c.work_dual.get();
    let cost_ = c.work_cost.get();
    let lower_ = c.work_lower.get();
    let upper_ = c.work_upper.get();
    let bvalue = c.base_value.get();
    let blower = c.base_lower.get();
    let bupper = c.base_upper.get();
    let nflag = c.nonbasic_flag.get();
    let nmove = c.nonbasic_move.get();
    let bindex = c.basic_index.get();

    let num_row = c.num_row as usize;
    let num_col = c.num_col as usize;
    let num_total = num_col + num_row;
    const H_TT: f64 = 1e-13;
    const H_INF: f64 = INF;
    let objective = c.objective;
    let sense = c.sense as f64;

    let mut iwork = vec![0usize; num_total];
    let mut dwork = vec![0f64; num_total];

    let mut xi = bvalue[..num_row].to_vec();
    for i in 0..num_row {
        xi[i] = cmax(xi[i], blower[i]);
        xi[i] = cmin(xi[i], bupper[i]);
    }
    let mut dj = dual_[..num_total].to_vec();
    for j in 0..num_total {
        if nflag[j] != 0 && lower_[j] != upper_[j] {
            if value_[j] == lower_[j] {
                dj[j] = cmax(dj[j], 0.0);
            }
            if value_[j] == upper_[j] {
                dj[j] = cmin(dj[j], 0.0);
            }
            if lower_[j] == -H_INF && upper_[j] == H_INF {
                dj[j] = 0.0;
            }
        }
    }
    let mut dxi_inc = vec![0f64; num_row];
    let mut dxi_dec = vec![0f64; num_row];
    for i in 0..num_row {
        dxi_inc[i] = bupper[i] - xi[i];
        dxi_dec[i] = blower[i] - xi[i];
    }
    let mut ddj_inc = vec![0f64; num_total];
    let mut ddj_dec = vec![0f64; num_total];
    for j in 0..num_total {
        if nflag[j] != 0 {
            ddj_inc[j] = if value_[j] == lower_[j] { H_INF } else { -dj[j] };
            ddj_dec[j] = if value_[j] == upper_[j] { -H_INF } else { -dj[j] };
        }
    }
    const TOL_A: f64 = 1e-9;
    const THETA_INF: f64 = H_INF / 1e40;

    let mut txj_inc = vec![THETA_INF; num_total];
    let mut axj_inc = vec![0f64; num_total];
    let mut ixj_inc = vec![-1i32; num_total];
    let mut wxj_inc = vec![0i32; num_total];
    let mut jxj_inc = vec![-1i32; num_total];
    let mut txj_dec = vec![-THETA_INF; num_total];
    let mut axj_dec = vec![0f64; num_total];
    let mut ixj_dec = vec![-1i32; num_total];
    let mut wxj_dec = vec![0i32; num_total];
    let mut jxj_dec = vec![-1i32; num_total];

    let mut tci_inc = vec![THETA_INF; num_row];
    let mut aci_inc = vec![0f64; num_row];
    let mut jci_inc = vec![-1i32; num_row];
    let mut tci_dec = vec![-THETA_INF; num_row];
    let mut aci_dec = vec![0f64; num_row];
    let mut jci_dec = vec![-1i32; num_row];

    // Major "theta" loop
    for j in 0..num_total {
        if nflag[j] == 0 {
            continue;
        }
        let mut col = CColumn { count: 0, index: std::ptr::null(), array: std::ptr::null() };
        (c.ftran)(c.ctx, j as i32, &mut col);
        let count = col.count.max(0) as usize;
        let index = if count > 0 { std::slice::from_raw_parts(col.index, count) } else { &[] };
        let mut n_work = 0;
        for &i_row in index {
            let alpha = *col.array.add(i_row as usize);
            if alpha.abs() > TOL_A {
                iwork[n_work] = i_row as usize;
                dwork[n_work] = alpha;
                n_work += 1;
            }
        }
        // Standard primal ratio test
        let mut myt_inc = THETA_INF;
        let mut myt_dec = -THETA_INF;
        let mut myk_inc: Option<usize> = None;
        let mut myk_dec: Option<usize> = None;
        for k in 0..n_work {
            let i = iwork[k];
            let alpha = dwork[k];
            let theta_inc = (if alpha < 0.0 { dxi_inc[i] } else { dxi_dec[i] }) / -alpha;
            let theta_dec = (if alpha > 0.0 { dxi_inc[i] } else { dxi_dec[i] }) / -alpha;
            if myt_inc > theta_inc {
                myt_inc = theta_inc;
                myk_inc = Some(k);
            }
            if myt_dec < theta_dec {
                myt_dec = theta_dec;
                myk_dec = Some(k);
            }
        }
        if let Some(k) = myk_inc {
            let i = iwork[k];
            let alpha = dwork[k];
            ixj_inc[j] = i as i32;
            axj_inc[j] = alpha;
            txj_inc[j] = (if alpha < 0.0 { dxi_inc[i] } else { dxi_dec[i] }) / -alpha;
            wxj_inc[j] = if alpha < 0.0 { 1 } else { -1 };
        }
        if let Some(k) = myk_dec {
            let i = iwork[k];
            let alpha = dwork[k];
            ixj_dec[j] = i as i32;
            axj_dec[j] = alpha;
            txj_dec[j] = (if alpha > 0.0 { dxi_inc[i] } else { dxi_dec[i] }) / -alpha;
            wxj_dec[j] = if alpha > 0.0 { 1 } else { -1 };
        }
        // Accumulated dual ratio test
        let myd_inc = ddj_inc[j];
        let myd_dec = ddj_dec[j];
        for k in 0..n_work {
            let i = iwork[k];
            let alpha = dwork[k];
            let theta_inc = (if alpha < 0.0 { myd_inc } else { myd_dec }) / -alpha;
            let theta_dec = (if alpha > 0.0 { myd_inc } else { myd_dec }) / -alpha;
            if tci_inc[i] > theta_inc {
                tci_inc[i] = theta_inc;
                aci_inc[i] = alpha;
                jci_inc[i] = j as i32;
            }
            if tci_dec[i] < theta_dec {
                tci_dec[i] = theta_dec;
                aci_dec[i] = alpha;
                jci_dec[i] = j as i32;
            }
        }
    }

    // Additional j-out for primal ratio test (considering bound flip)
    for j in 0..num_total {
        if nflag[j] == 0 {
            continue;
        }
        if nmove[j] == 1 {
            let value = value_[j] + txj_inc[j];
            if ixj_inc[j] != -1 && value <= upper_[j] {
                jxj_inc[j] = bindex[ixj_inc[j] as usize];
            } else if value > upper_[j] {
                jxj_inc[j] = j as i32;
            }
        }
        if nmove[j] == -1 {
            let value = value_[j] + txj_dec[j];
            if ixj_dec[j] != -1 && value >= lower_[j] {
                jxj_dec[j] = bindex[ixj_dec[j] as usize];
            } else if value < lower_[j] {
                jxj_dec[j] = j as i32;
            }
        }
        if lower_[j] == -H_INF && upper_[j] == H_INF {
            if ixj_inc[j] != -1 {
                jxj_inc[j] = bindex[ixj_inc[j] as usize];
                jxj_dec[j] = jxj_inc[j];
            }
            if ixj_dec[j] != -1 {
                jxj_inc[j] = bindex[ixj_dec[j] as usize];
                jxj_dec[j] = jxj_inc[j];
            }
        }
    }

    // Cost ranging
    let mut c_up_c = vec![0f64; num_total];
    let mut c_dn_c = vec![0f64; num_total];
    let mut c_up_f = vec![0f64; num_total];
    let mut c_dn_f = vec![0f64; num_total];
    let mut c_up_e = vec![0i32; num_total];
    let mut c_dn_e = vec![0i32; num_total];
    let mut c_up_l = vec![0i32; num_total];
    let mut c_dn_l = vec![0i32; num_total];

    // objective + sense * x, fused
    let obj_plus = |x: f64| sense.mul_add_c(x, objective);
    let obj_minus = |x: f64| (-sense).mul_add_c(x, objective);

    // Nonbasic cost ranging
    for j in 0..num_col {
        if nflag[j] == 0 {
            continue;
        }
        let value = value_[j];
        let vsign = if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        };
        if ddj_inc[j] != H_INF {
            c_up_c[j] = cost_[j] + ddj_inc[j];
            c_up_f[j] = obj_plus(poss_inf_product(ddj_inc[j], value));
            c_up_e[j] = j as i32;
            c_up_l[j] = jxj_dec[j];
        } else {
            c_up_c[j] = H_INF;
            c_up_f[j] = obj_plus(inf_product(vsign));
            c_up_e[j] = -1;
            c_up_l[j] = -1;
        }
        if ddj_dec[j] != H_INF {
            c_dn_c[j] = cost_[j] + ddj_dec[j];
            c_dn_f[j] = obj_plus(poss_inf_product(ddj_dec[j], value));
            c_dn_e[j] = j as i32;
            c_dn_l[j] = jxj_inc[j];
        } else {
            // As in the C++, the up values are written here
            c_up_c[j] = -H_INF;
            c_up_f[j] = obj_minus(inf_product(vsign));
            c_up_e[j] = -1;
            c_up_l[j] = -1;
        }
    }

    // Basic cost ranging
    for i in 0..num_row {
        if (bindex[i] as usize) >= num_col {
            continue;
        }
        let j = bindex[i] as usize;
        let value = xi[i];
        let vsign = if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        };
        if jci_inc[i] != -1 {
            c_up_c[j] = cost_[j] + tci_inc[i];
            c_up_f[j] = obj_plus(poss_inf_product(tci_inc[i], value));
            let je = jci_inc[i] as usize;
            c_up_e[j] = je as i32;
            c_up_l[j] = if nmove[je] > 0 { jxj_inc[je] } else { jxj_dec[je] };
        } else {
            c_up_c[j] = H_INF;
            c_up_f[j] = obj_plus(inf_product(vsign));
            c_up_e[j] = -1;
            c_up_l[j] = -1;
        }
        if jci_dec[i] != -1 {
            c_dn_c[j] = cost_[j] + tci_dec[i];
            c_dn_f[j] = obj_plus(poss_inf_product(tci_dec[i], value));
            let je = jci_dec[i] as usize;
            c_dn_e[j] = je as i32;
            c_dn_l[j] = if nmove[je] > 0 { jxj_inc[je] } else { jxj_dec[je] };
        } else {
            c_dn_c[j] = -H_INF;
            c_dn_f[j] = obj_minus(inf_product(vsign));
            c_dn_e[j] = -1;
            c_dn_l[j] = -1;
        }
    }

    // Bounds ranging
    let mut b_up_b = vec![0f64; num_total];
    let mut b_dn_b = vec![0f64; num_total];
    let mut b_up_f = vec![0f64; num_total];
    let mut b_dn_f = vec![0f64; num_total];
    let mut b_up_e = vec![0i32; num_total];
    let mut b_dn_e = vec![0i32; num_total];
    let mut b_up_l = vec![0i32; num_total];
    let mut b_dn_l = vec![0i32; num_total];

    // Nonbasic bounds ranging
    for j in 0..num_total {
        if nflag[j] == 0 {
            continue;
        }
        if lower_[j] == -H_INF && upper_[j] == H_INF {
            b_up_b[j] = H_INF;
            b_up_f[j] = objective;
            b_up_e[j] = -1;
            b_up_l[j] = -1;
            b_dn_b[j] = -H_INF;
            b_dn_f[j] = objective;
            b_dn_e[j] = -1;
            b_dn_l[j] = -1;
            continue;
        }
        let dualv = dj[j];
        let dsign = if dualv > 0.0 {
            1.0
        } else if dualv < 0.0 {
            -1.0
        } else {
            0.0
        };
        if ixj_inc[j] != -1 {
            let i = ixj_inc[j] as usize;
            b_up_b[j] = value_[j] + txj_inc[j];
            b_up_f[j] = obj_plus(poss_inf_product(txj_inc[j], dualv));
            b_up_e[j] = if wxj_inc[j] > 0 { jci_inc[i] } else { jci_dec[i] };
            b_up_l[j] = bindex[i];
        } else {
            b_up_b[j] = H_INF;
            b_up_f[j] = obj_plus(inf_product(dsign));
            b_up_e[j] = -1;
            b_up_l[j] = -1;
        }
        if value_[j] != upper_[j] && b_up_b[j] > upper_[j] {
            b_up_b[j] = upper_[j];
            b_up_f[j] = (sense * (upper_[j] - lower_[j])).mul_add_c(dualv, objective);
            b_up_e[j] = j as i32;
            b_up_l[j] = j as i32;
        }
        if ixj_dec[j] != -1 {
            let i = ixj_dec[j] as usize;
            b_dn_b[j] = value_[j] + txj_dec[j];
            b_dn_f[j] = obj_plus(poss_inf_product(txj_dec[j], dualv));
            b_dn_e[j] = if wxj_dec[j] > 0 { jci_inc[i] } else { jci_dec[i] };
            b_dn_l[j] = bindex[i];
        } else {
            b_dn_b[j] = -H_INF;
            b_dn_f[j] = obj_minus(inf_product(dsign));
            b_dn_e[j] = -1;
            b_dn_l[j] = -1;
        }
        if value_[j] != lower_[j] && b_dn_b[j] < lower_[j] {
            b_dn_b[j] = lower_[j];
            b_dn_f[j] = (sense * (lower_[j] - upper_[j])).mul_add_c(dualv, objective);
            b_dn_e[j] = j as i32;
            b_dn_l[j] = j as i32;
        }
    }

    // Basic bounds ranging
    for i in 0..num_row {
        for dir in [-1i32, 1] {
            let j = bindex[i] as usize;
            let j_in = if dir == -1 { jci_inc[i] } else { jci_dec[i] };
            let a_in = if dir == -1 { aci_inc[i] } else { aci_dec[i] };
            let (newx, newf, j_enter, j_leave);
            if j_in != -1 {
                let ji = j_in as usize;
                let jmove = nmove[ji] as i32;
                let i_out = if jmove > 0 { ixj_inc[ji] } else { ixj_dec[ji] };
                let j_out = if jmove > 0 { jxj_inc[ji] } else { jxj_dec[ji] };
                let w_out = if jmove > 0 { wxj_inc[ji] } else { wxj_dec[ji] };
                let tt = if jmove > 0 { txj_inc[ji] } else { txj_dec[ji] };
                if j_out == j_in {
                    // Bound flip
                    let delta = jmove as f64 * (upper_[ji] - lower_[ji]);
                    newx = (-delta).mul_add_c(a_in, xi[i]);
                    newf = (sense * delta).mul_add_c(dual_[ji], objective);
                    j_enter = j_in;
                    j_leave = j_out;
                } else if j_out != -1 {
                    // Regular
                    let io = i_out as usize;
                    let delta = if w_out > 0 { dxi_inc[io] } else { dxi_dec[io] };
                    let a_out = if jmove > 0 { axj_inc[ji] } else { axj_dec[ji] };
                    newx = xi[i] + delta * a_in / a_out;
                    newf = (sense * tt).mul_add_c(dual_[ji], objective);
                    j_enter = j_in;
                    j_leave = j_out;
                } else {
                    // Primal ratio test failed - change unlimitedly
                    newx = if dir == -1 { lower_[j] } else { upper_[j] };
                    newf = objective;
                    j_enter = -1;
                    j_leave = -1;
                }
            } else {
                // Dual ratio test failed - just stay
                newx = xi[i];
                newf = objective;
                j_enter = -1;
                j_leave = -1;
            }
            if dir == -1 {
                b_dn_b[j] = newx;
                b_dn_f[j] = newf;
                b_dn_e[j] = j_enter;
                b_dn_l[j] = j_leave;
            } else {
                b_up_b[j] = newx;
                b_up_f[j] = newf;
                b_up_e[j] = j_enter;
                b_up_l[j] = j_leave;
            }
        }
    }

    // Trim small values to zero
    for j in 0..num_col {
        if c_up_c[j].abs() < H_TT {
            c_up_c[j] = 0.0;
        }
        if c_dn_c[j].abs() < H_TT {
            c_dn_c[j] = 0.0;
        }
        if b_up_b[j].abs() < H_TT {
            b_up_b[j] = 0.0;
        }
        if b_dn_b[j].abs() < H_TT {
            b_dn_b[j] = 0.0;
        }
    }
    for j in num_col..num_total {
        if b_up_b[j].abs() < H_TT {
            b_up_b[j] = 0.0;
        }
        if b_dn_b[j].abs() < H_TT {
            b_dn_b[j] = 0.0;
        }
    }

    // Output
    let [cost_up, cost_dn, bound_up, bound_dn, row_up, row_dn] = &c.out;
    let put = |r: &CRecord, v: &[f64], f: &[f64], e: &[i32], l: &[i32], negate: bool| {
        let (rv, rf, re, rl) = (r.value.get_mut(), r.objective.get_mut(), r.in_var.get_mut(), r.ou_var.get_mut());
        for k in 0..rv.len() {
            rv[k] = if negate { -v[k] } else { v[k] };
        }
        rf.copy_from_slice(&f[..rf.len()]);
        re.copy_from_slice(&e[..re.len()]);
        rl.copy_from_slice(&l[..rl.len()]);
    };
    if c.sense > 0 {
        put(cost_up, &c_up_c, &c_up_f, &c_up_e, &c_up_l, false);
        put(cost_dn, &c_dn_c, &c_dn_f, &c_dn_e, &c_dn_l, false);
    } else {
        // For maximization problems, flip data and negate the cost values
        put(cost_up, &c_dn_c, &c_dn_f, &c_dn_e, &c_dn_l, true);
        put(cost_dn, &c_up_c, &c_up_f, &c_up_e, &c_up_l, true);
    }
    put(bound_up, &b_up_b, &b_up_f, &b_up_e, &b_up_l, false);
    put(bound_dn, &b_dn_b, &b_dn_f, &b_dn_e, &b_dn_l, false);
    // Flip all row data and negate the row bound values
    let n = num_col;
    put(row_up, &b_dn_b[n..], &b_dn_f[n..], &b_dn_e[n..], &b_dn_l[n..], true);
    put(row_dn, &b_up_b[n..], &b_up_f[n..], &b_up_e[n..], &b_up_l[n..], true);
    Status::Ok
}

/// # Safety
/// `c` valid, its arrays sized as documented
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_ranging_data(c: *const CRanging) -> i32 {
    get_ranging_data(&*c) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products() {
        assert_eq!(inf_product(0.0), 0.0);
        assert_eq!(inf_product(-1.0), -INF);
        assert_eq!(poss_inf_product(INF, 0.0), 0.0);
        assert_eq!(poss_inf_product(INF, 2.0), INF);
    }
}
