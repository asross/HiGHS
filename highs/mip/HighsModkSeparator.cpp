/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsModKSeparator.cpp
 */

#include "mip/HighsModkSeparator.h"

#include <array>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../extern/pdqsort/pdqsort.h"
#include "mip/HighsCutGeneration.h"
#include "mip/HighsGFkSolve.h"
#include "mip/HighsLpAggregator.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsTransformedLp.h"
#include "util/HighsHash.h"
#include "util/HighsIntegers.h"

template <HighsInt k, typename FoundModKCut>
static bool separateModKCuts(const std::vector<int64_t>& intSystemValue,
                             const std::vector<HighsInt>& intSystemIndex,
                             const std::vector<HighsInt>& intSystemStart,
                             const HighsCutPool& cutpool, HighsInt numCol,
                             FoundModKCut&& foundModKCut) {
  HighsGFkSolve GFkSolve;

  HighsInt numCuts = cutpool.getNumCuts();

  GFkSolve.fromCSC<k>(intSystemValue, intSystemIndex, intSystemStart,
                      numCol + 1);
  GFkSolve.setRhs<k>(numCol, 1);
  GFkSolve.solve<k>(foundModKCut);

  return cutpool.getNumCuts() != numCuts;
}

void HighsModkSeparator::separateLpSolution(HighsLpRelaxation& lpRelaxation,
                                            HighsLpAggregator& lpAggregator,
                                            HighsTransformedLp& transLp,
                                            HighsCutPool& cutpool) {
  const HighsMipSolver& mipsolver = lpRelaxation.getMipSolver();
  const HighsLp& lp = lpRelaxation.getLp();

  std::vector<uint8_t> skipRow(lp.num_row_);

  // mark all rows that have continuous variables with a nonzero solution value
  // in the transformed LP to be skipped
  for (HighsInt col : mipsolver.mipdata_->continuous_cols) {
    if (transLp.boundDistance(col) == 0) continue;

    const HighsInt start = lp.a_matrix_.start_[col];
    const HighsInt end = lp.a_matrix_.start_[col + 1];

    for (HighsInt i = start; i != end; ++i)
      skipRow[lp.a_matrix_.index_[i]] = true;
  }

  HighsCutGeneration cutGen(lpRelaxation, cutpool);

  std::vector<std::pair<HighsInt, double>> integralScales;
  std::vector<int64_t> intSystemValue;
  std::vector<HighsInt> intSystemIndex;
  std::vector<HighsInt> intSystemStart;

  intSystemValue.reserve(lp.a_matrix_.value_.size() + lp.num_row_);
  intSystemIndex.reserve(intSystemValue.size());
  intSystemStart.reserve(lp.num_row_ + 1);
  intSystemStart.push_back(0);

  std::vector<HighsInt> inds;
  std::vector<double> vals;
  std::vector<double> scaleVals;

  inds.reserve(lp.num_col_);
  vals.reserve(lp.num_col_);
  scaleVals.reserve(lp.num_col_);

  std::vector<double> upper;
  std::vector<double> solval;
  double rhs;

  const HighsSolution& lpSolution = lpRelaxation.getSolution();
  HighsInt numNonzeroRhs = 0;
  HighsInt maxIntRowLen = 1000 + 0.1 * lp.num_col_;

  for (HighsInt row = 0; row != lp.num_row_; ++row) {
    if (skipRow[row]) continue;

    bool leqRow;

    if (lp.row_upper_[row] - lpSolution.row_value[row] <=
        mipsolver.mipdata_->feastol)
      leqRow = true;
    else if (lpSolution.row_value[row] - lp.row_lower_[row] <=
             mipsolver.mipdata_->feastol)
      leqRow = false;
    else
      continue;

    HighsInt rowlen;
    const HighsInt* rowinds;
    const double* rowvals;

    lpRelaxation.getRow(row, rowlen, rowinds, rowvals);

    if (leqRow) {
      rhs = lp.row_upper_[row];
      inds.assign(rowinds, rowinds + rowlen);
      vals.assign(rowvals, rowvals + rowlen);
    } else {
      assert(lpSolution.row_value[row] - lp.row_lower_[row] <=
             mipsolver.mipdata_->feastol);

      rhs = -lp.row_lower_[row];
      inds.assign(rowinds, rowinds + rowlen);
      vals.resize(rowlen);
      std::transform(rowvals, rowvals + rowlen, vals.begin(),
                     [](double x) { return -x; });
    }

    bool integralPositive = false;
    if (!transLp.transform(vals, upper, solval, inds, rhs, integralPositive,
                           true))
      continue;

    rowlen = inds.size();
    if (rowlen > maxIntRowLen) {
      HighsInt intRowLen = 0;
      for (HighsInt i = 0; i < rowlen; ++i) {
        if (solval[i] <= mipsolver.mipdata_->feastol) continue;
        if (mipsolver.isColContinuous(inds[i])) continue;
        ++intRowLen;
      }

      // skip row if either too long or 0 = 0 row
      if (intRowLen > maxIntRowLen ||
          (intRowLen == 0 && fabs(rhs) <= mipsolver.mipdata_->epsilon))
        continue;
    }

    double intscale;
    int64_t intrhs;

    if (!lpRelaxation.isRowIntegral(row)) {
      scaleVals.clear();
      for (HighsInt i = 0; i != rowlen; ++i) {
        if (mipsolver.isColContinuous(inds[i])) continue;
        if (solval[i] > mipsolver.mipdata_->feastol) {
          scaleVals.push_back(vals[i]);
        }
      }

      if (fabs(rhs) > mipsolver.mipdata_->epsilon) scaleVals.push_back(-rhs);
      if (scaleVals.empty()) continue;

      intscale = HighsIntegers::integralScale(
          scaleVals, mipsolver.mipdata_->feastol, mipsolver.mipdata_->epsilon);
      if (intscale == 0.0 || intscale > 1e6) continue;

      intrhs = HighsIntegers::nearestInteger(intscale * rhs);

      for (HighsInt i = 0; i != rowlen; ++i) {
        if (mipsolver.isColContinuous(inds[i])) continue;
        if (solval[i] > mipsolver.mipdata_->feastol) {
          intSystemIndex.push_back(inds[i]);
          intSystemValue.push_back(
              HighsIntegers::nearestInteger(intscale * vals[i]));
        }
      }
    } else {
      intscale = 1.0;
      intrhs = HighsIntegers::nearestInteger(rhs);

      for (HighsInt i = 0; i != rowlen; ++i) {
        if (solval[i] > mipsolver.mipdata_->feastol) {
          intSystemIndex.push_back(inds[i]);
          intSystemValue.push_back(HighsIntegers::nearestInteger(vals[i]));
        }
      }
    }

    numNonzeroRhs += (intrhs != 0);

    intSystemIndex.push_back(lp.num_col_);
    intSystemValue.push_back(intrhs);
    intSystemStart.push_back(intSystemValue.size());
    integralScales.emplace_back(row, intscale);
  }

  if (integralScales.empty() || numNonzeroRhs == 0) return;

  std::vector<HighsInt> tmpinds;
  std::vector<double> tmpvals;

  HighsHashTable<std::vector<HighsGFkSolve::SolutionEntry>> usedWeights;
  // std::unordered_set<std::vector<HighsGFkSolve::SolutionEntry>,
  //                   HighsVectorHasher, HighsVectorEqual>
  //    usedWeights;
  HighsInt k;
  auto foundCut = [&](std::vector<HighsGFkSolve::SolutionEntry>& weights,
                      int rhsIndex) {
    // cuts which come from a single row can already be found with the
    // aggregation heuristic
    if (weights.empty()) return;

    pdqsort(weights.begin(), weights.end());
    if (!usedWeights.insert(weights)) return;

    assert(lpAggregator.isEmpty());
    for (const auto& w : weights) {
      double weight = integralScales[w.index].second *
                      (double((w.weight * (k - 1)) % k) / k);
      HighsInt row = integralScales[w.index].first;
      lpAggregator.addRow(row, weight);
    }

    lpAggregator.getCurrentAggregation(inds, vals, false);

    rhs = 0.0;
    cutGen.generateCut(transLp, inds, vals, rhs, true);

    if (k != 2) {
      lpAggregator.clear();
      for (const auto& w : weights) {
        double weight = integralScales[w.index].second * (double(w.weight) / k);
        HighsInt row = integralScales[w.index].first;
        lpAggregator.addRow(row, weight);
      }
    }

    lpAggregator.getCurrentAggregation(inds, vals, true);

    rhs = 0.0;
    cutGen.generateCut(transLp, inds, vals, rhs, true);

    lpAggregator.clear();
  };

  k = 2;

  // ======================================================================
  // Caprara-Fischetti / Koster-Zymolka-Kutschka {0,1/2} separation.
  //
  // Restated separation (KZK 2007, Lemma 2): over the mod-2 system (A mod 2,
  // b mod 2, row slacks s), a combination v in {0,1}^m gives a violated
  // {0,1/2} cut iff v.b is odd and  z = v.s + (leftover odd columns).x* < 1.
  // We preprocess with the reduction rules (Lemma 3), emit single-row cuts
  // (Lemma 4), and run the Gaussian-elimination pivoting on slack-0 rows
  // (Proposition 5) that eliminates a column globally and folds its x* into
  // slack -- this reduces the general (>2 odd column) rows that defeated the
  // plain odd-cycle. Surviving residual rows are enumerated in small
  // combinations. Original rows behind a cut (tracked per reduced row in R)
  // are aggregated with weight 1/2 and handed to the validity-checked cut
  // generator, exactly as foundCut does for the GFk enumeration below.
  // ======================================================================
  {
    const double kCFSlackLimit = 1.0;
    const double cfeps = 1e-6;

    struct Cand {
      HighsInt origRow;
      double scale;
    };
    std::vector<Cand> cand;
    std::vector<std::vector<HighsInt>> rcols;  // reduced row -> sorted odd cols
    std::vector<uint8_t> rpar;                 // reduced row rhs parity
    std::vector<double> rslk;                  // reduced row slack
    std::vector<std::vector<HighsInt>> rR;     // reduced row -> cand indices
    std::unordered_map<HighsInt, double> xstar;

    auto symdiff = [](const std::vector<HighsInt>& a,
                      const std::vector<HighsInt>& b) {
      std::vector<HighsInt> out;
      out.reserve(a.size() + b.size());
      size_t ia = 0, ib = 0;
      while (ia < a.size() && ib < b.size()) {
        if (a[ia] < b[ib])
          out.push_back(a[ia++]);
        else if (b[ib] < a[ia])
          out.push_back(b[ib++]);
        else {
          ++ia;
          ++ib;
        }
      }
      while (ia < a.size()) out.push_back(a[ia++]);
      while (ib < b.size()) out.push_back(b[ib++]);
      return out;
    };

    for (HighsInt row = 0; row != lp.num_row_; ++row) {
      if (skipRow[row]) continue;
      double slackLeq = lp.row_upper_[row] - lpSolution.row_value[row];
      double slackGeq = lpSolution.row_value[row] - lp.row_lower_[row];
      bool leqRow = slackLeq <= slackGeq;
      double rowSlack = std::max(0.0, leqRow ? slackLeq : slackGeq);
      if (rowSlack >= kCFSlackLimit) continue;

      HighsInt rlen;
      const HighsInt* rinds;
      const double* rvals;
      lpRelaxation.getRow(row, rlen, rinds, rvals);
      if (leqRow) {
        rhs = lp.row_upper_[row];
        inds.assign(rinds, rinds + rlen);
        vals.assign(rvals, rvals + rlen);
      } else {
        rhs = -lp.row_lower_[row];
        inds.assign(rinds, rinds + rlen);
        vals.resize(rlen);
        std::transform(rvals, rvals + rlen, vals.begin(),
                       [](double x) { return -x; });
      }
      bool integralPositive = false;
      if (!transLp.transform(vals, upper, solval, inds, rhs, integralPositive,
                             true))
        continue;
      const HighsInt tlen = (HighsInt)inds.size();
      const bool rowIntegral = lpRelaxation.isRowIntegral(row);
      double intscale;
      int64_t intrhs;
      if (!rowIntegral) {
        scaleVals.clear();
        for (HighsInt i = 0; i != tlen; ++i) {
          if (mipsolver.isColContinuous(inds[i])) continue;
          if (solval[i] > mipsolver.mipdata_->feastol)
            scaleVals.push_back(vals[i]);
        }
        if (fabs(rhs) > mipsolver.mipdata_->epsilon) scaleVals.push_back(-rhs);
        if (scaleVals.empty()) continue;
        intscale = HighsIntegers::integralScale(
            scaleVals, mipsolver.mipdata_->feastol, mipsolver.mipdata_->epsilon);
        if (intscale == 0.0 || intscale > 1e6) continue;
        intrhs = HighsIntegers::nearestInteger(intscale * rhs);
      } else {
        intscale = 1.0;
        intrhs = HighsIntegers::nearestInteger(rhs);
      }

      std::vector<HighsInt> oddCols;
      for (HighsInt i = 0; i != tlen; ++i) {
        if (mipsolver.isColContinuous(inds[i])) continue;
        if (solval[i] <= mipsolver.mipdata_->feastol) continue;
        int64_t coeff = rowIntegral
                            ? HighsIntegers::nearestInteger(vals[i])
                            : HighsIntegers::nearestInteger(intscale * vals[i]);
        if (coeff & 1) {
          oddCols.push_back(inds[i]);
          xstar[inds[i]] = solval[i];
        }
      }
      std::sort(oddCols.begin(), oddCols.end());
      uint8_t parity = (uint8_t)(intrhs & 1);
      if (oddCols.empty() && parity == 0) continue;  // Lemma 3(ii)

      cand.push_back(Cand{row, intscale});
      rcols.push_back(std::move(oddCols));
      rpar.push_back(parity);
      rslk.push_back(rowSlack);
      rR.push_back(std::vector<HighsInt>{(HighsInt)cand.size() - 1});
    }

    const HighsInt nrow = (HighsInt)rcols.size();
    std::vector<uint8_t> dead(nrow, 0);
    std::vector<std::pair<double, std::vector<HighsInt>>> cuts;  // (z, R)

    // ---- preprocessing: Lemma 3 + Lemma 4 + Proposition 5 ----
    bool changed = true;
    HighsInt guard = 0;
    while (changed && guard++ < 4 * nrow + 16) {
      changed = false;

      for (HighsInt r = 0; r < nrow; ++r) {
        if (dead[r]) continue;
        if (rslk[r] >= 1.0 - cfeps) {  // Lemma 3(vi)
          dead[r] = 1;
          changed = true;
        } else if (rcols[r].empty()) {
          if (rpar[r] == 1 && !rR[r].empty())
            cuts.push_back({rslk[r], rR[r]});  // Lemma 4 (z = slack)
          dead[r] = 1;
          changed = true;
        }
      }

      // Lemma 3(v): a column odd in exactly one live row folds into its slack.
      std::unordered_map<HighsInt, HighsInt> colCount;
      for (HighsInt r = 0; r < nrow; ++r)
        if (!dead[r])
          for (HighsInt c : rcols[r]) ++colCount[c];
      for (HighsInt r = 0; r < nrow; ++r) {
        if (dead[r] || rcols[r].empty()) continue;
        std::vector<HighsInt> keep;
        keep.reserve(rcols[r].size());
        for (HighsInt c : rcols[r]) {
          if (colCount[c] == 1) {
            rslk[r] += xstar[c];
            changed = true;
          } else {
            keep.push_back(c);
          }
        }
        if (keep.size() != rcols[r].size()) rcols[r].swap(keep);
      }

      // Proposition 5: pivot on a slack-0 row to eliminate one column.
      HighsInt pivot = -1, pivCol = -1;
      double bestx = -1.0;
      for (HighsInt r = 0; r < nrow; ++r) {
        if (dead[r] || rslk[r] > cfeps || rcols[r].empty()) continue;
        for (HighsInt c : rcols[r]) {
          double xv = xstar[c];
          if (xv > bestx) {
            bestx = xv;
            pivot = r;
            pivCol = c;
          }
        }
      }
      if (pivot != -1) {
        for (HighsInt r = 0; r < nrow; ++r) {
          if (r == pivot || dead[r]) continue;
          if (!std::binary_search(rcols[r].begin(), rcols[r].end(), pivCol))
            continue;
          rcols[r] = symdiff(rcols[r], rcols[pivot]);
          rpar[r] ^= rpar[pivot];
          rslk[r] += rslk[pivot];  // == 0
          rR[r] = symdiff(rR[r], rR[pivot]);
        }
        std::vector<HighsInt> keep;
        keep.reserve(rcols[pivot].size());
        for (HighsInt c : rcols[pivot])
          if (c != pivCol) keep.push_back(c);
        rcols[pivot].swap(keep);
        rslk[pivot] = xstar[pivCol];  // Prop 5 / Cor 6
        changed = true;
      }
    }

    // ---- residual enumeration (KZK 3.2), combinations of <= 2 rows ----
    auto cost = [&](const std::vector<HighsInt>& cols, double slk) {
      double z = slk;
      for (HighsInt c : cols) z += xstar[c];
      return z;
    };
    std::vector<HighsInt> live;
    for (HighsInt r = 0; r < nrow; ++r)
      if (!dead[r] && rslk[r] < 1.0 - cfeps) live.push_back(r);

    const size_t kMaxCuts = 80;
    for (HighsInt r : live)
      if (rpar[r] == 1) {
        double z = cost(rcols[r], rslk[r]);
        if (z < 1.0 - cfeps) cuts.push_back({z, rR[r]});
      }
    for (size_t a = 0; a < live.size() && cuts.size() < kMaxCuts; ++a) {
      for (size_t b = a + 1; b < live.size() && cuts.size() < kMaxCuts; ++b) {
        HighsInt r1 = live[a], r2 = live[b];
        if ((rpar[r1] ^ rpar[r2]) != 1) continue;
        double slk = rslk[r1] + rslk[r2];
        if (slk >= 1.0 - cfeps) continue;
        std::vector<HighsInt> cc = symdiff(rcols[r1], rcols[r2]);
        double z = cost(cc, slk);
        if (z < 1.0 - cfeps) cuts.push_back({z, symdiff(rR[r1], rR[r2])});
      }
    }

    // ---- cut selection: emit only the most-violated (smallest z) cuts, so a
    // flood of weakly-violated zero-half cuts cannot bloat the LP. Aggregate
    // each kept cut's original rows with weight 1/2. ----
    std::sort(cuts.begin(), cuts.end(),
              [](const std::pair<double, std::vector<HighsInt>>& a,
                 const std::pair<double, std::vector<HighsInt>>& b) {
                return a.first < b.first;
              });
    const size_t kEmit = 12;  // top-N by violation per separation round
    const HighsInt poolBefore = cutpool.getNumCuts();
    size_t emitted = 0;
    for (const std::pair<double, std::vector<HighsInt>>& zc : cuts) {
      if (emitted >= kEmit) break;
      const std::vector<HighsInt>& R = zc.second;
      if (R.empty()) continue;
      assert(lpAggregator.isEmpty());
      for (HighsInt idx : R)
        lpAggregator.addRow(cand[idx].origRow, cand[idx].scale * 0.5);
      lpAggregator.getCurrentAggregation(inds, vals, false);
      rhs = 0.0;
      cutGen.generateCut(transLp, inds, vals, rhs, true);
      lpAggregator.getCurrentAggregation(inds, vals, true);
      rhs = 0.0;
      cutGen.generateCut(transLp, inds, vals, rhs, true);
      lpAggregator.clear();
      ++emitted;
    }
    (void)poolBefore;
  }

  if (separateModKCuts<2>(intSystemValue, intSystemIndex, intSystemStart,
                          cutpool, lp.num_col_, foundCut))
    return;

  usedWeights.clear();
  k = 3;
  if (separateModKCuts<3>(intSystemValue, intSystemIndex, intSystemStart,
                          cutpool, lp.num_col_, foundCut))
    return;

  usedWeights.clear();
  k = 5;
  if (separateModKCuts<5>(intSystemValue, intSystemIndex, intSystemStart,
                          cutpool, lp.num_col_, foundCut))
    return;

  usedWeights.clear();
  k = 7;
  if (separateModKCuts<7>(intSystemValue, intSystemIndex, intSystemStart,
                          cutpool, lp.num_col_, foundCut))
    return;
}
