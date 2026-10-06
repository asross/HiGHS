//! The MIP solver (highs/mip), ported bottom-up: so far HighsDomain
//! (domain.rs) with its objective propagation (objprop.rs) and conflict
//! analysis (conflict.rs), cut separation (cuts/), the clique table
//! (clique.rs) and the implications (implications.rs).

pub mod clique;
mod clique_ffi;
pub mod conflict;
pub mod cuts;
pub mod domain;
pub mod feasjump;
pub mod heuristics;
pub mod implications;
mod implications_ffi;
pub mod lns;
pub mod objprop;
