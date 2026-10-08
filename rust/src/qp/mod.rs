//! Port of the QP solver QUASS (highs/qpsolver/): a primal active-set
//! method with a dense Cholesky factor of the reduced Hessian. Highs::
//! callSolveQp's glue (the instance, settings, logging, phase 1 and the
//! result in HiGHS form) is glue.rs; the phase 1 LP solve, the timer and
//! the solution's storage stay in C++ (qpsolver/QpRust.cpp). Same
//! floating-point operations as the C++, so solves are
//! bit-identical: clang contracts `a + b * c` (fma here) and compiles the
//! QpVector dot product fused in most copies (see QpVector::dot_split4).

mod basis;
mod cholesky;
pub mod glue;
mod pricing;
mod quass;
pub mod vector;
#[cfg(test)]
mod tests;

pub use quass::{solve, Callbacks, Phase1Start, QpOutcome};
use vector::{MatrixBase, QpVector};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolverStatus {
    Ok,
    NotPositiveDefinite,
    Degenerate,
    Error,
}

/// QpModelStatus (qpconst.hpp), with the C++ values
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum ModelStatus {
    NotSet = 0,
    Undetermined,
    Optimal,
    Unbounded,
    Infeasible,
    IterationLimit,
    TimeLimit,
    LargeNullspace,
    NonConvex,
    Interrupt,
    Error,
}

impl ModelStatus {
    pub fn from_i32(v: i32) -> ModelStatus {
        use ModelStatus::*;
        [NotSet, Undetermined, Optimal, Unbounded, Infeasible, IterationLimit, TimeLimit, LargeNullspace, NonConvex, Interrupt, Error]
            .get(v as usize)
            .copied()
            .unwrap_or(Error)
    }
}

/// BasisStatus (qpconst.hpp), with the C++ values
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum BasisStatus {
    Inactive = 0,
    ActiveAtLower = 1,
    ActiveAtUpper = 2,
    InactiveInBasis = 3,
}

impl BasisStatus {
    pub fn from_i32(v: i32) -> BasisStatus {
        match v {
            1 => BasisStatus::ActiveAtLower,
            2 => BasisStatus::ActiveAtUpper,
            3 => BasisStatus::InactiveInBasis,
            _ => BasisStatus::Inactive,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PricingStrategy {
    SteepestEdge,
    DantzigWolfe,
    Devex,
}

/// The numerical Settings (settings.hpp) as the solver reads them
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Settings {
    /// 0: two pass, 1: textbook
    pub ratiotest: i32,
    /// PricingStrategy in C++ order
    pub pricing: i32,
    pub reportingfequency: i32,
    pub nullspace_limit: i32,
    pub reinvertfrequency: i32,
    pub gradientrecomputefrequency: i32,
    pub iteration_limit: i32,
    pub ratiotest_t: f64,
    pub ratiotest_d: f64,
    pub pnorm_zero_threshold: f64,
    pub d_zero_threshold: f64,
    pub lambda_zero_threshold: f64,
    pub pqp_zero_threshold: f64,
    pub hessian_regularization_value: f64,
    pub time_limit: f64,
}

impl Settings {
    pub fn pricing_strategy(&self) -> PricingStrategy {
        match self.pricing {
            0 => PricingStrategy::SteepestEdge,
            1 => PricingStrategy::DantzigWolfe,
            _ => PricingStrategy::Devex,
        }
    }
}

/// The QP: min c'x + x'Qx/2 + offset s.t. con_lo <= Ax <= con_up,
/// var_lo <= x <= var_up (instance.hpp)
pub struct Instance {
    pub num_var: usize,
    pub num_con: usize,
    pub offset: f64,
    pub c: QpVector,
    pub q: MatrixBase,
    pub a: MatrixBase,
    pub con_lo: Vec<f64>,
    pub con_up: Vec<f64>,
    pub var_lo: Vec<f64>,
    pub var_up: Vec<f64>,
}

impl Instance {
    pub fn objval(&self, x: &QpVector) -> f64 {
        let mut qx = QpVector::new(self.num_var);
        self.q.vec_mat(x, &mut qx);
        vector::fma(0.5, qx.dot_split4(x), self.c.dot_split4(x)) + self.offset
    }

    pub fn is_equality(&self, i: usize) -> bool {
        if i < self.num_con {
            self.con_lo[i] == self.con_up[i]
        } else {
            self.var_lo[i - self.num_con] == self.var_up[i - self.num_con]
        }
    }
}
