//! HighsLinearSumBounds (highs/util/HighsLinearSumBounds.{h,cpp}): bounds on
//! linear sums of variables with finite or infinite bounds, counting the
//! infinite contributions. "Orig" sums use the variable bounds only, the
//! others the tighter of the variable and implied bounds (an implied bound
//! that comes from the sum itself is ignored for that sum).
//!
//! The C++ keeps pointers to the bound arrays; here each call gets them as a
//! [`VarBounds`]. The eight per-sum values are one struct (same arithmetic,
//! one cache line per sum).

use crate::util::cdouble::CDouble;

const INF: f64 = f64::INFINITY;

/// The bound arrays of the variables (setBoundArrays)
#[derive(Clone, Copy)]
pub struct VarBounds<'a> {
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub impl_lower: &'a [f64],
    pub impl_upper: &'a [f64],
    pub impl_lower_src: &'a [i32],
    pub impl_upper_src: &'a [i32],
}

impl VarBounds<'_> {
    #[inline]
    pub fn impl_var_upper(&self, sum: i32, var: usize) -> f64 {
        impl_upper(sum, self.upper[var], self.impl_upper[var], self.impl_upper_src[var])
    }
    #[inline]
    pub fn impl_var_lower(&self, sum: i32, var: usize) -> f64 {
        impl_lower(sum, self.lower[var], self.impl_lower[var], self.impl_lower_src[var])
    }
}

#[inline]
fn impl_upper(sum: i32, var_upper: f64, impl_var_upper: f64, src: i32) -> f64 {
    if src == sum {
        var_upper
    } else {
        // std::min(a, b): b < a ? b : a
        if var_upper < impl_var_upper {
            var_upper
        } else {
            impl_var_upper
        }
    }
}

#[inline]
fn impl_lower(sum: i32, var_lower: f64, impl_var_lower: f64, src: i32) -> f64 {
    if src == sum {
        var_lower
    } else {
        // std::max(a, b): a < b ? b : a
        if impl_var_lower < var_lower {
            var_lower
        } else {
            impl_var_lower
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Sum {
    lower_orig: CDouble,
    upper_orig: CDouble,
    lower: CDouble,
    upper: CDouble,
    ninf_lower_orig: i32,
    ninf_upper_orig: i32,
    ninf_lower: i32,
    ninf_upper: i32,
}

#[derive(Default)]
pub struct LinearSumBounds {
    sums: Vec<Sum>,
}

#[inline]
fn update(num_inf: &mut i32, activity: &mut CDouble, direction: i32, bound: f64, coefficient: f64) {
    if bound.abs() != INF {
        *activity += CDouble::from(bound) * direction as f64 * coefficient;
    } else {
        *num_inf += direction;
    }
}

#[inline]
fn update2(num_inf: &mut i32, activity: &mut CDouble, old: f64, new: f64, coefficient: f64) {
    if old == new {
        return;
    }
    update(num_inf, activity, -1, old, coefficient);
    update(num_inf, activity, 1, new, coefficient);
}

impl LinearSumBounds {
    pub fn set_num_sums(&mut self, n: usize) {
        self.sums.resize(n, Sum::default());
    }

    pub fn len(&self) -> usize {
        self.sums.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sums.is_empty()
    }

    pub fn sum_scaled(&mut self, sum: i32, scale: f64) {
        let s = &mut self.sums[sum as usize];
        s.lower_orig *= scale;
        s.upper_orig *= scale;
        s.lower *= scale;
        s.upper *= scale;
        if scale < 0.0 {
            std::mem::swap(&mut s.lower, &mut s.upper);
            std::mem::swap(&mut s.lower_orig, &mut s.upper_orig);
            std::mem::swap(&mut s.ninf_lower, &mut s.ninf_upper);
            std::mem::swap(&mut s.ninf_lower_orig, &mut s.ninf_upper_orig);
        }
    }

    #[inline]
    fn handle_var_upper(&mut self, sum: i32, coef: f64, ub: f64, dir: i32) {
        let s = &mut self.sums[sum as usize];
        if coef > 0.0 {
            update(&mut s.ninf_upper_orig, &mut s.upper_orig, dir, ub, coef);
        } else {
            update(&mut s.ninf_lower_orig, &mut s.lower_orig, dir, ub, coef);
        }
    }

    #[inline]
    fn handle_var_lower(&mut self, sum: i32, coef: f64, lb: f64, dir: i32) {
        let s = &mut self.sums[sum as usize];
        if coef > 0.0 {
            update(&mut s.ninf_lower_orig, &mut s.lower_orig, dir, lb, coef);
        } else {
            update(&mut s.ninf_upper_orig, &mut s.upper_orig, dir, lb, coef);
        }
    }

    #[inline]
    fn handle_impl_var_upper(&mut self, sum: i32, coef: f64, ub: f64, dir: i32) {
        let s = &mut self.sums[sum as usize];
        if coef > 0.0 {
            update(&mut s.ninf_upper, &mut s.upper, dir, ub, coef);
        } else {
            update(&mut s.ninf_lower, &mut s.lower, dir, ub, coef);
        }
    }

    #[inline]
    fn handle_impl_var_lower(&mut self, sum: i32, coef: f64, lb: f64, dir: i32) {
        let s = &mut self.sums[sum as usize];
        if coef > 0.0 {
            update(&mut s.ninf_lower, &mut s.lower, dir, lb, coef);
        } else {
            update(&mut s.ninf_upper, &mut s.upper, dir, lb, coef);
        }
    }

    pub fn add(&mut self, sum: i32, var: i32, coef: f64, b: &VarBounds) {
        let v = var as usize;
        self.handle_var_upper(sum, coef, b.upper[v], 1);
        self.handle_var_lower(sum, coef, b.lower[v], 1);
        self.handle_impl_var_upper(sum, coef, b.impl_var_upper(sum, v), 1);
        self.handle_impl_var_lower(sum, coef, b.impl_var_lower(sum, v), 1);
    }

    pub fn remove(&mut self, sum: i32, var: i32, coef: f64, b: &VarBounds) {
        let v = var as usize;
        self.handle_var_upper(sum, coef, b.upper[v], -1);
        self.handle_var_lower(sum, coef, b.lower[v], -1);
        self.handle_impl_var_upper(sum, coef, b.impl_var_upper(sum, v), -1);
        self.handle_impl_var_lower(sum, coef, b.impl_var_lower(sum, v), -1);
    }

    pub fn updated_var_upper(&mut self, sum: i32, var: i32, coef: f64, old_var_upper: f64, b: &VarBounds) {
        let v = var as usize;
        self.handle_var_upper(sum, coef, old_var_upper, -1);
        self.handle_var_upper(sum, coef, b.upper[v], 1);
        self.updated_impl_var_upper4(sum, var, coef, old_var_upper, b.impl_upper[v], b.impl_upper_src[v], b);
    }

    pub fn updated_var_lower(&mut self, sum: i32, var: i32, coef: f64, old_var_lower: f64, b: &VarBounds) {
        let v = var as usize;
        self.handle_var_lower(sum, coef, old_var_lower, -1);
        self.handle_var_lower(sum, coef, b.lower[v], 1);
        self.updated_impl_var_lower4(sum, var, coef, old_var_lower, b.impl_lower[v], b.impl_lower_src[v], b);
    }

    pub fn updated_impl_var_upper(
        &mut self,
        sum: i32,
        var: i32,
        coef: f64,
        old_impl_upper: f64,
        old_src: i32,
        b: &VarBounds,
    ) {
        self.updated_impl_var_upper4(sum, var, coef, b.upper[var as usize], old_impl_upper, old_src, b);
    }

    pub fn updated_impl_var_lower(
        &mut self,
        sum: i32,
        var: i32,
        coef: f64,
        old_impl_lower: f64,
        old_src: i32,
        b: &VarBounds,
    ) {
        self.updated_impl_var_lower4(sum, var, coef, b.lower[var as usize], old_impl_lower, old_src, b);
    }

    #[allow(clippy::too_many_arguments)]
    fn updated_impl_var_upper4(
        &mut self,
        sum: i32,
        var: i32,
        coef: f64,
        old_var_upper: f64,
        old_impl_upper: f64,
        old_src: i32,
        b: &VarBounds,
    ) {
        let old_v = impl_upper(sum, old_var_upper, old_impl_upper, old_src);
        let v = b.impl_var_upper(sum, var as usize);
        if v == old_v {
            return;
        }
        self.handle_impl_var_upper(sum, coef, old_v, -1);
        self.handle_impl_var_upper(sum, coef, v, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn updated_impl_var_lower4(
        &mut self,
        sum: i32,
        var: i32,
        coef: f64,
        old_var_lower: f64,
        old_impl_lower: f64,
        old_src: i32,
        b: &VarBounds,
    ) {
        let old_v = impl_lower(sum, old_var_lower, old_impl_lower, old_src);
        let v = b.impl_var_lower(sum, var as usize);
        if v == old_v {
            return;
        }
        self.handle_impl_var_lower(sum, coef, old_v, -1);
        self.handle_impl_var_lower(sum, coef, v, 1);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn residual(
        num_inf: &mut i32,
        activity: &mut CDouble,
        var_bound: f64,
        coef: f64,
        bound_var: i32,
        bound_var_coef: f64,
        old_bv_bound: f64,
        new_bv_bound: f64,
    ) {
        update(num_inf, activity, -1, var_bound, coef);
        if bound_var != -1 {
            update2(num_inf, activity, old_bv_bound, new_bv_bound, bound_var_coef);
        }
    }

    /// getResidualSumLower (bound_var = -1 for none)
    #[allow(clippy::too_many_arguments)]
    pub fn residual_sum_lower(
        &self,
        sum: i32,
        var: i32,
        coef: f64,
        bound_var: i32,
        bound_var_coef: f64,
        bound_var_value: f64,
        b: &VarBounds,
    ) -> f64 {
        let s = &self.sums[sum as usize];
        let mut activity = s.lower;
        let mut ninf = s.ninf_lower;
        let vb = if coef > 0.0 { b.impl_var_lower(sum, var as usize) } else { b.impl_var_upper(sum, var as usize) };
        let bvb = if bound_var != -1 {
            if bound_var_coef > 0.0 {
                b.impl_var_lower(sum, bound_var as usize)
            } else {
                b.impl_var_upper(sum, bound_var as usize)
            }
        } else {
            INF
        };
        Self::residual(&mut ninf, &mut activity, vb, coef, bound_var, bound_var_coef, bvb, bound_var_value);
        if ninf == 0 {
            activity.to_f64()
        } else {
            -INF
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_sum_upper(
        &self,
        sum: i32,
        var: i32,
        coef: f64,
        bound_var: i32,
        bound_var_coef: f64,
        bound_var_value: f64,
        b: &VarBounds,
    ) -> f64 {
        let s = &self.sums[sum as usize];
        let mut activity = s.upper;
        let mut ninf = s.ninf_upper;
        let vb = if coef < 0.0 { b.impl_var_lower(sum, var as usize) } else { b.impl_var_upper(sum, var as usize) };
        let bvb = if bound_var != -1 {
            if bound_var_coef < 0.0 {
                b.impl_var_lower(sum, bound_var as usize)
            } else {
                b.impl_var_upper(sum, bound_var as usize)
            }
        } else {
            INF
        };
        Self::residual(&mut ninf, &mut activity, vb, coef, bound_var, bound_var_coef, bvb, bound_var_value);
        if ninf == 0 {
            activity.to_f64()
        } else {
            INF
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_sum_lower_orig(
        &self,
        sum: i32,
        var: i32,
        coef: f64,
        bound_var: i32,
        bound_var_coef: f64,
        bound_var_value: f64,
        b: &VarBounds,
    ) -> f64 {
        let s = &self.sums[sum as usize];
        let mut activity = s.lower_orig;
        let mut ninf = s.ninf_lower_orig;
        let v = var as usize;
        let vb = if coef > 0.0 { b.lower[v] } else { b.upper[v] };
        let bvb = if bound_var != -1 {
            if bound_var_coef > 0.0 {
                b.lower[bound_var as usize]
            } else {
                b.upper[bound_var as usize]
            }
        } else {
            INF
        };
        Self::residual(&mut ninf, &mut activity, vb, coef, bound_var, bound_var_coef, bvb, bound_var_value);
        if ninf == 0 {
            activity.to_f64()
        } else {
            -INF
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_sum_upper_orig(
        &self,
        sum: i32,
        var: i32,
        coef: f64,
        bound_var: i32,
        bound_var_coef: f64,
        bound_var_value: f64,
        b: &VarBounds,
    ) -> f64 {
        let s = &self.sums[sum as usize];
        let mut activity = s.upper_orig;
        let mut ninf = s.ninf_upper_orig;
        let v = var as usize;
        let vb = if coef < 0.0 { b.lower[v] } else { b.upper[v] };
        let bvb = if bound_var != -1 {
            if bound_var_coef < 0.0 {
                b.lower[bound_var as usize]
            } else {
                b.upper[bound_var as usize]
            }
        } else {
            INF
        };
        Self::residual(&mut ninf, &mut activity, vb, coef, bound_var, bound_var_coef, bvb, bound_var_value);
        if ninf == 0 {
            activity.to_f64()
        } else {
            INF
        }
    }

    /// getSumLowerOrig(sum) (offset 0: CDouble + 0.0)
    #[inline]
    pub fn sum_lower_orig(&self, sum: i32) -> f64 {
        self.sum_lower_orig_off(sum, CDouble::from(0.0))
    }
    #[inline]
    pub fn sum_upper_orig(&self, sum: i32) -> f64 {
        self.sum_upper_orig_off(sum, CDouble::from(0.0))
    }
    #[inline]
    pub fn sum_lower(&self, sum: i32) -> f64 {
        self.sum_lower_off(sum, CDouble::from(0.0))
    }
    #[inline]
    pub fn sum_upper(&self, sum: i32) -> f64 {
        self.sum_upper_off(sum, CDouble::from(0.0))
    }

    #[inline]
    pub fn sum_lower_orig_off(&self, sum: i32, off: CDouble) -> f64 {
        let s = &self.sums[sum as usize];
        if s.ninf_lower_orig == 0 {
            (s.lower_orig + off).to_f64()
        } else {
            -INF
        }
    }
    #[inline]
    pub fn sum_upper_orig_off(&self, sum: i32, off: CDouble) -> f64 {
        let s = &self.sums[sum as usize];
        if s.ninf_upper_orig == 0 {
            (s.upper_orig + off).to_f64()
        } else {
            INF
        }
    }
    #[inline]
    pub fn sum_lower_off(&self, sum: i32, off: CDouble) -> f64 {
        let s = &self.sums[sum as usize];
        if s.ninf_lower == 0 {
            (s.lower + off).to_f64()
        } else {
            -INF
        }
    }
    #[inline]
    pub fn sum_upper_off(&self, sum: i32, off: CDouble) -> f64 {
        let s = &self.sums[sum as usize];
        if s.ninf_upper == 0 {
            (s.upper + off).to_f64()
        } else {
            INF
        }
    }

    #[inline]
    pub fn num_inf_sum_lower(&self, sum: i32) -> i32 {
        self.sums[sum as usize].ninf_lower
    }
    #[inline]
    pub fn num_inf_sum_upper(&self, sum: i32) -> i32 {
        self.sums[sum as usize].ninf_upper
    }
    #[inline]
    pub fn num_inf_sum_lower_orig(&self, sum: i32) -> i32 {
        self.sums[sum as usize].ninf_lower_orig
    }
    #[inline]
    pub fn num_inf_sum_upper_orig(&self, sum: i32) -> i32 {
        self.sums[sum as usize].ninf_upper_orig
    }

    pub fn shrink(&mut self, new_indices: &[i32], new_size: usize) {
        for (i, &ni) in new_indices.iter().enumerate() {
            if ni != -1 {
                self.sums[ni as usize] = self.sums[i];
            }
        }
        self.sums.resize(new_size, Sum::default());
    }
}
