//! HighsTransformedLp::transform and untransform (bound substitution with
//! simple and variable bounds, complementation of integers), and
//! HighsLpAggregator, on a [`SepaRound`].

use super::round::*;
use crate::util::cdouble::CDouble;

impl SepaRound {
    /// HighsImplications::cleanupVub(col, vubCol, vub, ub, ..., false)
    fn cleanup_vub(&mut self, col: usize, ub: f64) {
        let vub = &mut self.vub[col];
        let maxub = CDouble::from(vub.max_value());
        let minub = CDouble::from(vub.min_value());
        if minub >= ub - self.feastol {
            // redundant
        } else if maxub > ub + self.epsilon {
            let newcoef = (ub - minub).to_f64();
            if vub.coef > 0.0 {
                vub.coef = newcoef;
            } else {
                vub.constant = ub;
                vub.coef = -newcoef;
            }
        }
    }

    /// HighsImplications::cleanupVlb(col, vlbCol, vlb, lb, ..., false)
    fn cleanup_vlb(&mut self, col: usize, lb: f64) {
        let vlb = &mut self.vlb[col];
        let maxlb = CDouble::from(vlb.max_value());
        let minlb = CDouble::from(vlb.min_value());
        if maxlb <= lb + self.feastol {
            // redundant
        } else if minlb < lb - self.epsilon {
            let newcoef = (lb - maxlb).to_f64();
            if vlb.coef < 0.0 {
                vlb.coef = newcoef;
            } else {
                vlb.constant = lb;
                vlb.coef = -newcoef;
            }
        }
    }

    /// HighsTransformedLp::transform
    #[allow(clippy::too_many_arguments)]
    pub fn transform(
        &mut self,
        vals: &mut Vec<f64>,
        upper: &mut Vec<f64>,
        solval: &mut Vec<f64>,
        inds: &mut Vec<i32>,
        rhs: &mut f64,
        integers_positive: &mut bool,
        prefer_vbds: bool,
    ) -> bool {
        debug_assert!(self.vectorsum.nonzeroinds.is_empty());
        let mut tmp_rhs = CDouble::from(*rhs);
        let feastol = self.feastol;
        let small = self.small_matrix_value;
        let mut num_nz = inds.len();

        let mut i = 0;
        while i < num_nz {
            let col = inds[i] as usize;
            let (lb, ub) = self.bounds(col);

            macro_rules! remove {
                () => {{
                    num_nz -= 1;
                    inds[i] = inds[num_nz];
                    vals[i] = vals[num_nz];
                    inds[num_nz] = 0;
                    vals[num_nz] = 0.0;
                }};
            }

            if ub - lb < small {
                tmp_rhs -= lb.min_cpp(ub) * vals[i];
                remove!();
                continue;
            }

            if lb == -K_HIGHS_INF && ub == K_HIGHS_INF {
                self.vectorsum.clear();
                return false;
            }

            let d = self.cols[col];
            if d.vub_col != -1 && self.vub[col].max_value() > ub + feastol {
                self.cleanup_vub(col, ub);
            }
            if d.vlb_col != -1 && self.vlb[col].min_value() < lb - feastol {
                self.cleanup_vlb(col, lb);
            }

            let old_bound_type = d.bound_type;
            let v = vals[i];
            let bt;
            if d.integral {
                if ub - lb <= 1.5
                    || d.bound_dist() != 0.0
                    || d.simple_lb_dist == 0.0
                    || d.simple_ub_dist == 0.0
                {
                    self.cols[col].bound_type = if d.simple_lb_dist < d.simple_ub_dist - feastol {
                        K_SIMPLE_LB
                    } else if d.simple_ub_dist < d.simple_lb_dist - feastol {
                        K_SIMPLE_UB
                    } else if v > 0.0 {
                        K_SIMPLE_LB
                    } else {
                        K_SIMPLE_UB
                    };
                    i += 1;
                    continue;
                }
                bt = if d.vlb_col == -1 || d.ub_dist < d.lb_dist - feastol {
                    K_VARIABLE_UB
                } else if d.vub_col == -1 || d.lb_dist < d.ub_dist - feastol {
                    K_VARIABLE_LB
                } else if v > 0.0 {
                    K_VARIABLE_UB
                } else {
                    K_VARIABLE_LB
                };
            } else if d.lb_dist < d.ub_dist - feastol {
                bt = if d.vlb_col == -1 {
                    K_SIMPLE_LB
                } else if prefer_vbds || v > 0.0 || d.simple_lb_dist > d.lb_dist + feastol {
                    K_VARIABLE_LB
                } else {
                    K_SIMPLE_LB
                };
            } else if d.ub_dist < d.lb_dist - feastol {
                bt = if d.vub_col == -1 {
                    K_SIMPLE_UB
                } else if prefer_vbds || v < 0.0 || d.simple_ub_dist > d.ub_dist + feastol {
                    K_VARIABLE_UB
                } else {
                    K_SIMPLE_UB
                };
            } else if v > 0.0 {
                bt = if d.vlb_col != -1 {
                    K_VARIABLE_LB
                } else if prefer_vbds && d.vub_col != -1 {
                    K_VARIABLE_UB
                } else {
                    K_SIMPLE_LB
                };
            } else {
                bt = if d.vub_col != -1 {
                    K_VARIABLE_UB
                } else if prefer_vbds && d.vlb_col != -1 {
                    K_VARIABLE_LB
                } else {
                    K_SIMPLE_UB
                };
            }

            match bt {
                K_SIMPLE_LB => {
                    if v > 0.0 {
                        tmp_rhs -= lb * v;
                        self.cols[col].bound_type = old_bound_type;
                        remove!();
                        continue;
                    }
                }
                K_SIMPLE_UB => {
                    if v < 0.0 {
                        tmp_rhs -= ub * v;
                        self.cols[col].bound_type = old_bound_type;
                        remove!();
                        continue;
                    }
                }
                K_VARIABLE_LB => {
                    let vb = self.vlb[col];
                    tmp_rhs -= vb.constant * v;
                    self.vectorsum.add(d.vlb_col, v * vb.coef);
                    if v > 0.0 {
                        self.cols[col].bound_type = old_bound_type;
                        remove!();
                        continue;
                    }
                }
                _ => {
                    let vb = self.vub[col];
                    tmp_rhs -= vb.constant * v;
                    self.vectorsum.add(d.vub_col, v * vb.coef);
                    vals[i] = -v;
                    if vals[i] > 0.0 {
                        self.cols[col].bound_type = old_bound_type;
                        remove!();
                        continue;
                    }
                }
            }
            self.cols[col].bound_type = bt;
            i += 1;
        }

        if !self.vectorsum.nonzeroinds.is_empty() {
            for j in 0..num_nz {
                if vals[j] != 0.0 {
                    self.vectorsum.add(inds[j], vals[j]);
                }
            }
            self.vectorsum.cleanup(|_, val| val.abs() <= small);
            inds.clear();
            inds.extend_from_slice(&self.vectorsum.nonzeroinds);
            num_nz = inds.len();
            vals.clear();
            vals.extend(inds.iter().map(|&j| self.vectorsum.get_value(j)));
            self.vectorsum.clear();
        } else {
            vals.truncate(num_nz);
            inds.truncate(num_nz);
        }

        for j in 0..num_nz {
            let col = inds[j] as usize;
            let d = &self.cols[col];
            if !d.integral {
                continue;
            }
            if d.bound_type == K_VARIABLE_LB || d.bound_type == K_VARIABLE_UB {
                continue;
            }
            let (lb, ub) = self.bounds(col);
            let d = &mut self.cols[col];
            d.bound_type = if *integers_positive {
                if (lb != -K_HIGHS_INF && vals[j] > 0.0) || ub == K_HIGHS_INF {
                    K_SIMPLE_LB
                } else {
                    K_SIMPLE_UB
                }
            } else if d.lb_dist < d.ub_dist {
                K_SIMPLE_LB
            } else {
                K_SIMPLE_UB
            };
        }

        upper.clear();
        upper.resize(num_nz, 0.0);
        solval.clear();
        solval.resize(num_nz, 0.0);

        for j in 0..num_nz {
            let col = inds[j] as usize;
            let (lb, ub) = self.bounds(col);
            upper[j] = ub - lb;
            let d = &self.cols[col];
            match d.bound_type {
                K_SIMPLE_LB => {
                    tmp_rhs -= lb * vals[j];
                    solval[j] = d.lb_dist;
                }
                K_SIMPLE_UB => {
                    tmp_rhs -= ub * vals[j];
                    vals[j] = -vals[j];
                    solval[j] = d.ub_dist;
                }
                K_VARIABLE_LB => solval[j] = d.lb_dist,
                _ => solval[j] = d.ub_dist,
            }
            if d.integral {
                *integers_positive = *integers_positive && vals[j] > 0.0;
            }
        }

        *rhs = tmp_rhs.to_f64();
        !(num_nz == 0 && *rhs >= -feastol)
    }

    /// HighsTransformedLp::untransform
    pub fn untransform(&mut self, vals: &mut Vec<f64>, inds: &mut Vec<i32>, rhs: &mut f64, integral: bool) -> bool {
        let mut tmp_rhs = CDouble::from(*rhs);
        let num_col = self.num_col;
        for i in 0..inds.len() {
            if vals[i] == 0.0 {
                continue;
            }
            let col = inds[i] as usize;
            let d = self.cols[col];
            match d.bound_type {
                K_VARIABLE_LB => {
                    let vb = self.vlb[col];
                    tmp_rhs += vb.constant * vals[i];
                    self.vectorsum.add(d.vlb_col, -vals[i] * vb.coef);
                    self.vectorsum.add(col as i32, vals[i]);
                }
                K_VARIABLE_UB => {
                    let vb = self.vub[col];
                    tmp_rhs -= vb.constant * vals[i];
                    self.vectorsum.add(d.vub_col, vals[i] * vb.coef);
                    self.vectorsum.add(col as i32, -vals[i]);
                }
                K_SIMPLE_LB => {
                    if col < num_col {
                        tmp_rhs += vals[i] * self.col_lower.at(col);
                        self.vectorsum.add(col as i32, vals[i]);
                    } else {
                        let row = col - num_col;
                        tmp_rhs += vals[i] * self.slack_bounds(row).0;
                        let v = vals[i];
                        let (s, e) = (self.ar_start[row], self.ar_start[row + 1]);
                        for k in s..e {
                            self.vectorsum.add(self.ar_index[k], v * self.ar_value[k]);
                        }
                    }
                }
                _ => {
                    if col < num_col {
                        tmp_rhs -= vals[i] * self.col_upper.at(col);
                        self.vectorsum.add(col as i32, -vals[i]);
                    } else {
                        let row = col - num_col;
                        tmp_rhs -= vals[i] * self.slack_bounds(row).1;
                        vals[i] = -vals[i];
                        let v = vals[i];
                        let (s, e) = (self.ar_start[row], self.ar_start[row + 1]);
                        for k in s..e {
                            self.vectorsum.add(self.ar_index[k], v * self.ar_value[k]);
                        }
                    }
                }
            }
        }

        if integral {
            self.vectorsum.cleanup(|_, val| val.abs() < 0.5);
            *rhs = tmp_rhs.to_f64().round();
        } else {
            let mut abort = false;
            let (small, feastol) = (self.small_matrix_value, self.feastol);
            let (lower, upper) = (self.col_lower, self.col_upper);
            self.vectorsum.cleanup(|col, val| {
                let absval = val.abs();
                if absval <= small {
                    return true;
                }
                if absval <= feastol {
                    if val > 0.0 {
                        let lb = lower.at(col as usize);
                        if lb == -K_HIGHS_INF {
                            abort = true;
                        } else {
                            tmp_rhs -= val * lb;
                        }
                    } else {
                        let ub = upper.at(col as usize);
                        if ub == K_HIGHS_INF {
                            abort = true;
                        } else {
                            tmp_rhs -= val * ub;
                        }
                    }
                    return true;
                }
                false
            });
            if abort {
                self.vectorsum.clear();
                return false;
            }
            *rhs = tmp_rhs.to_f64();
        }

        inds.clear();
        inds.extend_from_slice(&self.vectorsum.nonzeroinds);
        vals.clear();
        if integral {
            vals.extend(inds.iter().map(|&j| self.vectorsum.get_value(j).round()));
        } else {
            vals.extend(inds.iter().map(|&j| self.vectorsum.get_value(j)));
        }
        self.vectorsum.clear();
        true
    }

    /// HighsLpAggregator::addRow
    pub fn aggr_add_row(&mut self, row: usize, weight: f64) {
        let (s, e) = (self.ar_start[row], self.ar_start[row + 1]);
        for k in s..e {
            self.aggr.add(self.ar_index[k], weight * self.ar_value[k]);
        }
        self.aggr.add((self.num_col + row) as i32, -weight);
    }

    /// HighsLpAggregator::getCurrentAggregation
    pub fn aggr_get(&mut self, inds: &mut Vec<i32>, vals: &mut Vec<f64>, negate: bool) {
        let droptol = self.small_matrix_value;
        let num_col = self.num_col as i32;
        self.aggr.cleanup(|col, val| col < num_col && val.abs() <= droptol);
        inds.clear();
        inds.extend_from_slice(&self.aggr.nonzeroinds);
        vals.clear();
        if negate {
            vals.extend(inds.iter().map(|&j| -self.aggr.get_value(j)));
        } else {
            vals.extend(inds.iter().map(|&j| self.aggr.get_value(j)));
        }
    }

    pub fn aggr_clear(&mut self) {
        self.aggr.clear();
    }

    pub fn aggr_is_empty(&self) -> bool {
        self.aggr.nonzeroinds.is_empty()
    }
}
