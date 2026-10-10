/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsMipHost.h
 * @brief The MIP solver of a HIGHS_RUST build is Rust's
 * (rust/src/mip/host): Highs::callSolveMip and Highs::runPresolve pass it
 * the model, the options and the Highs object's callback and profiling,
 * and take its result. HighsMipHost.cpp is the Highs object's side: the
 * profiling clocks, the user callback and the improving solution file.
 */
#ifndef MIP_HIGHS_MIP_HOST_H_
#define MIP_HIGHS_MIP_HOST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsCallback.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
#include "lp_data/HighsSolution.h"
#include "presolve/HighsPostsolveStack.h"


/// The MIP solver's host of a Highs object's engine (rust/src/lp_data/top.rs:
/// HighsFns' context): its callback, options and model (with
/// semi-variables, the LP withoutSemiVariables makes), made and freed
void* highsMipHostNew(HighsCallback& callback, const HighsOptions& options,
                      const HighsLp& lp, const bool semi);
void highsMipHostFree(void* host);


#endif  // HIGHS_RUST
#endif
