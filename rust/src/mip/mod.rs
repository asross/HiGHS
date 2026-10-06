//! The MIP solver (highs/mip), ported bottom-up: so far the propagation
//! engine of HighsDomain (domain.rs) and cut separation (cuts/).

pub mod cuts;
pub mod domain;
