/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file presolve/HPresolveRust.cpp
 * @brief Empty under HIGHS_RUST: the LP presolve of a Highs object runs on
 * its engine's data (rust/src/lp_data/lp_presolve.rs) and the MIP presolve
 * is Rust's (rust/src/mip/host), so no C++ HPresolve wrapper is called. The
 * file stays in the source list so that the unity batches keep their
 * composition.
 */
#include "presolve/HPresolve.h"


