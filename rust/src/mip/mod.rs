//! The MIP solver (highs/mip), ported bottom-up: so far HighsDomain
//! (domain.rs) with its objective propagation (objprop.rs) and conflict
//! analysis (conflict.rs).

pub mod conflict;
pub mod domain;
pub mod objprop;
