//! Choice of the active constraint to drop (qpsolver/dantzigpricing.hpp,
//! devexpricing.hpp, steepestedgepricing.hpp)

use super::basis::Basis;
use super::vector::{fma, QpVector};
use super::{BasisStatus, Instance, PricingStrategy};

pub enum Pricing {
    Dantzig,
    Devex(Vec<f64>),
    SteepestEdge(Vec<f64>),
}

impl Pricing {
    pub fn new(strategy: PricingStrategy, num_var: usize, basis: &mut Basis) -> Self {
        match strategy {
            PricingStrategy::DantzigWolfe => Pricing::Dantzig,
            PricingStrategy::Devex => Pricing::Devex(vec![1.0; num_var]),
            PricingStrategy::SteepestEdge => {
                // compute_exact_weights
                let mut y = QpVector::new(num_var);
                let weights = (0..num_var)
                    .map(|i| {
                        basis.btran(&QpVector::unit(num_var, i), &mut y);
                        y.dot_split4(&y)
                    })
                    .collect();
                Pricing::SteepestEdge(weights)
            }
        }
    }

    /// The active constraint to drop, or None if the basis is optimal
    pub fn price(&self, inst: &Instance, basis: &Basis, lambda: &QpVector, lambda_zero_threshold: f64) -> Option<usize> {
        let index_in_factor = basis.index_in_factor();
        let mut minidx = None;
        let mut maxval = 0.0;
        for &con in basis.active() {
            let ib = index_in_factor[con];
            if ib == -1 {
                println!("error");
            }
            if inst.is_equality(con) {
                continue;
            }
            let l = lambda.value[ib as usize];
            let status = basis.status(con);
            match self {
                Pricing::Dantzig => {
                    if status == BasisStatus::ActiveAtLower && -l > maxval {
                        minidx = Some(con);
                        maxval = -l;
                    } else if status == BasisStatus::ActiveAtUpper && l > maxval {
                        minidx = Some(con);
                        maxval = l;
                    }
                }
                Pricing::Devex(weights) | Pricing::SteepestEdge(weights) => {
                    let val = l * l / weights[ib as usize];
                    if val > maxval
                        && l.abs() > lambda_zero_threshold
                        && ((status == BasisStatus::ActiveAtLower && -l > 0.0)
                            || (status == BasisStatus::ActiveAtUpper && l > 0.0))
                    {
                        minidx = Some(con);
                        maxval = val;
                    }
                }
            }
        }
        if let Pricing::Dantzig = self {
            if maxval <= lambda_zero_threshold {
                return None;
            }
        }
        minidx
    }

    /// Update the weights for the basis change dropping `p`, with the
    /// pivotal column `aq` and row `ep` (before the factor is updated)
    pub fn update_weights(&mut self, basis: &mut Basis, aq: &QpVector, ep: &QpVector, p: usize) {
        let rowindex_p = basis.index_in_factor()[p] as usize;
        match self {
            Pricing::Dantzig => {}
            Pricing::Devex(weights) => {
                let weight_p = weights[rowindex_p];
                let t_p = aq.value[rowindex_p];
                for (i, w) in weights.iter_mut().enumerate() {
                    if i == rowindex_p {
                        *w = weight_p / (t_p * t_p);
                    } else {
                        let t_i = aq.value[i];
                        *w = fma((t_i * t_i) / (t_p * t_p) * weight_p, weight_p, *w);
                    }
                    if *w > 1e7 {
                        *w = 1.0;
                    }
                }
            }
            Pricing::SteepestEdge(weights) => {
                let mut delta = QpVector::new(weights.len());
                basis.ftran(aq, &mut delta, false);
                let weight_p = ep.dot_split4(ep);
                let t_p = aq.value[rowindex_p];
                for (i, w) in weights.iter_mut().enumerate() {
                    if i != rowindex_p {
                        let t_i = aq.value[i];
                        let v = fma(-(2.0 * (t_i / t_p)), delta.value[i], *w);
                        *w = fma((t_i * t_i) / (t_p * t_p), weight_p, v);
                    }
                }
                weights[rowindex_p] = weight_p / (t_p * t_p);
            }
        }
    }
}
