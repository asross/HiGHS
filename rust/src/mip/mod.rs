//! The MIP solver (highs/mip), ported bottom-up: so far the propagation
//! engine of HighsDomain (domain.rs), cut separation (cuts/), the clique
//! table (clique.rs) and the implications (implications.rs).

pub mod cuts;
pub mod domain;
pub mod clique;
mod clique_ffi;
pub mod implications;
mod implications_ffi;
