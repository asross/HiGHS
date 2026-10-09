//! The MIP solver (highs/mip), ported bottom-up: so far HighsDomain
//! (domain.rs) with its objective propagation (objprop.rs) and conflict
//! analysis (conflict.rs), cut separation (cuts/), the clique table
//! (clique.rs), the implications (implications.rs), the pseudocosts
//! (pseudocost.rs), the conflict and cut pools (conflictpool.rs,
//! cutpool.rs), the node queue (nodequeue.rs) and reduced cost fixing
//! (redcost.rs).

pub mod clique;
pub mod concurrent;
pub(crate) mod clique_ffi;
pub mod conflict;
pub mod cuts;
pub mod domain;
pub mod driver;
pub mod feasjump;
pub mod glue;
pub mod graph_lns;
pub mod host;
pub mod heuristics;
pub mod implications;
pub(crate) mod implications_ffi;
pub mod lns;
pub mod lp_relaxation;
pub mod mip_data;
pub mod objprop;
pub mod primal;
pub mod pseudocost;
pub mod conflictpool;
pub mod cutpool;
pub mod nodequeue;
pub mod redcost;
pub mod root;
pub mod search;
pub mod separation;
pub mod setup;
pub mod workers;
