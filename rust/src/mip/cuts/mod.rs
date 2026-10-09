//! MIP cut separation: HighsTransformedLp, HighsLpAggregator,
//! HighsCutGeneration, the path, tableau and mod-k separators and
//! HighsGFkSolve. One separation round's data is a [`round::SepaRound`]
//! (see its module comment for the boundary with C++).

// index loops mirror the C++; negated comparisons keep its NaN semantics
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord, clippy::too_many_arguments)]

pub mod cut_generation;
pub(crate) mod ffi;
pub mod gfk;
pub mod integers;
pub mod modk;
pub mod path;
pub mod round;
pub mod sort;
pub mod tableau;
mod transform;

#[cfg(test)]
mod tests;
