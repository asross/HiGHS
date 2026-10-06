/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_PSEUDOCOST_H_
#define HIGHS_PSEUDOCOST_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "HConfig.h"
#include "util/HighsInt.h"

class HighsMipSolver;
namespace presolve {
class HighsPostsolveStack;
}

class HighsPseudocost;

constexpr double minThreshold = 1e-6;

struct HighsPseudocostInitialization {
  std::vector<double> pseudocostup;
  std::vector<double> pseudocostdown;
  std::vector<HighsInt> nsamplesup;
  std::vector<HighsInt> nsamplesdown;
  std::vector<double> inferencesup;
  std::vector<double> inferencesdown;
  std::vector<HighsInt> ninferencesup;
  std::vector<HighsInt> ninferencesdown;
  std::vector<double> conflictscoreup;
  std::vector<double> conflictscoredown;
  double cost_total;
  double inferences_total;
  double conflict_avg_score;
  int64_t nsamplestotal;
  int64_t ninferencestotal;

  HighsPseudocostInitialization(const HighsPseudocost& pscost,
                                HighsInt maxCount);
  HighsPseudocostInitialization(
      const HighsPseudocost& pscost, HighsInt maxCount,
      const presolve::HighsPostsolveStack& postsolveStack);
};

struct HighsPseudocostDelta {
  HighsInt col = -1;
  HighsInt nsamplesup = 0;
  HighsInt nsamplesdown = 0;
  HighsInt ninferencesup = 0;
  HighsInt ninferencesdown = 0;
  HighsInt ncutoffsup = 0;
  HighsInt ncutoffsdown = 0;
  double pseudocostup_sum = 0.0;
  double pseudocostdown_sum = 0.0;
  double inferencesup_sum = 0.0;
  double inferencesdown_sum = 0.0;
  double conflictscoreup_sum = 0.0;
  double conflictscoredown_sum = 0.0;
};

#ifdef HIGHS_RUST
namespace highs_rs {
struct Pseudocost;
// HighsPseudocostInitialization's arrays (rust/src/mip/pseudocost.rs)
struct PscostInit {
  double* pseudocostup;
  double* pseudocostdown;
  HighsInt* nsamplesup;
  HighsInt* nsamplesdown;
  double* inferencesup;
  double* inferencesdown;
  HighsInt* ninferencesup;
  HighsInt* ninferencesdown;
  double* conflictscoreup;
  double* conflictscoredown;
  HighsInt n;
  double cost_total;
  double inferences_total;
  double conflict_avg_score;
  int64_t nsamplestotal;
  int64_t ninferencestotal;
};
extern "C" {
Pseudocost* highs_rs_pscost_new(HighsInt ncol, HighsInt minreliable);
void highs_rs_pscost_free(Pseudocost* p);
Pseudocost* highs_rs_pscost_clone(const Pseudocost* p);
void highs_rs_pscost_init(Pseudocost* p, const PscostInit* init,
                          const HighsInt* orig);
void highs_rs_pscost_export(const Pseudocost* p, HighsInt maxCount,
                            const HighsInt* orig, PscostInit* init);
HighsInt highs_rs_pscost_geti(const Pseudocost* p, int which, HighsInt col);
double highs_rs_pscost_getd(const Pseudocost* p, int which, HighsInt col,
                            double x);
double highs_rs_pscost_offset(const Pseudocost* p, HighsInt col, double frac,
                              double offset, bool up);
double highs_rs_pscost_score(const Pseudocost* p, HighsInt col, double up,
                             double down);
void highs_rs_pscost_set(Pseudocost* p, int which, HighsInt col, HighsInt i,
                         double x);
void highs_rs_pscost_add_observation(Pseudocost* p, HighsInt col, double delta,
                                     double objdelta);
void highs_rs_pscost_flush(Pseudocost* p, Pseudocost* other, bool sync);
}
}  // namespace highs_rs

// The pseudocosts are Rust's (rust/src/mip/pseudocost.rs); this class is a
// handle
class HighsPseudocost {
  friend struct HighsPseudocostInitialization;
  highs_rs::Pseudocost* rs_ = nullptr;

  using Rs = highs_rs::Pseudocost;
  void set(int which, HighsInt col = 0, HighsInt i = 0, double x = 0) {
    highs_rs::highs_rs_pscost_set(rs_, which, col, i, x);
  }
  double getd(int which, HighsInt col = 0, double x = 0) const {
    return highs_rs::highs_rs_pscost_getd(rs_, which, col, x);
  }
  HighsInt geti(int which, HighsInt col = 0) const {
    return highs_rs::highs_rs_pscost_geti(rs_, which, col);
  }

 public:
  HighsPseudocost() = default;
  HighsPseudocost(const HighsMipSolver& mipsolver);
  HighsPseudocost(const HighsPseudocost& other)
      : rs_(highs_rs::highs_rs_pscost_clone(other.rs_)) {}
  HighsPseudocost(HighsPseudocost&& other) : rs_(other.rs_) {
    other.rs_ = nullptr;
  }
  HighsPseudocost& operator=(const HighsPseudocost& other) {
    if (this != &other) {
      highs_rs::highs_rs_pscost_free(rs_);
      rs_ = highs_rs::highs_rs_pscost_clone(other.rs_);
    }
    return *this;
  }
  HighsPseudocost& operator=(HighsPseudocost&& other) {
    std::swap(rs_, other.rs_);
    return *this;
  }
  ~HighsPseudocost() { highs_rs::highs_rs_pscost_free(rs_); }

  Rs* rust() const { return rs_; }

  void increaseConflictWeight() { set(2); }
  void setDegeneracyFactor(double degeneracyFactor) {
    assert(degeneracyFactor >= 1.0);
    set(1, 0, 0, degeneracyFactor);
  }
  void increaseConflictScoreUp(HighsInt col) { set(3, col); }
  void increaseConflictScoreDown(HighsInt col) { set(4, col); }
  void setMinReliable(HighsInt minreliable) { set(0, 0, minreliable); }
  HighsInt getMinReliable() const { return geti(0); }
  HighsInt getNumObservations(HighsInt col) const {
    return geti(1, col) + geti(2, col);
  }
  HighsInt getNumObservationsUp(HighsInt col) const { return geti(1, col); }
  HighsInt getNumObservationsDown(HighsInt col) const { return geti(2, col); }
  void addCutoffObservation(HighsInt col, bool upbranch) {
    set(5, col, upbranch);
  }
  void addObservation(HighsInt col, double delta, double objdelta) {
    highs_rs::highs_rs_pscost_add_observation(rs_, col, delta, objdelta);
  }
  void addInferenceObservation(HighsInt col, HighsInt ninferences,
                               bool upbranch) {
    set(6, col, ninferences, upbranch ? 1.0 : 0.0);
  }
  bool isReliable(HighsInt col) const { return geti(3, col) != 0; }
  bool isReliableUp(HighsInt col) const { return geti(4, col) != 0; }
  bool isReliableDown(HighsInt col) const { return geti(5, col) != 0; }
  double getAvgPseudocost() const { return getd(0); }
  double getPseudocostUp(HighsInt col, double frac, double offset) const {
    return highs_rs::highs_rs_pscost_offset(rs_, col, frac, offset, true);
  }
  double getPseudocostDown(HighsInt col, double frac, double offset) const {
    return highs_rs::highs_rs_pscost_offset(rs_, col, frac, offset, false);
  }
  double getPseudocostUp(HighsInt col, double frac) const {
    return getd(1, col, frac);
  }
  double getPseudocostDown(HighsInt col, double frac) const {
    return getd(2, col, frac);
  }
  double getConflictScoreUp(HighsInt col) const { return getd(3, col); }
  double getConflictScoreDown(HighsInt col) const { return getd(4, col); }
  double getScore(HighsInt col, double upcost, double downcost) const {
    return highs_rs::highs_rs_pscost_score(rs_, col, upcost, downcost);
  }
  double getScore(HighsInt col, double frac) const {
    return getd(5, col, frac);
  }
  double getScoreUp(HighsInt col, double frac) const {
    return getd(6, col, frac);
  }
  double getScoreDown(HighsInt col, double frac) const {
    return getd(7, col, frac);
  }
  double getAvgInferencesUp(HighsInt col) const { return getd(8, col); }
  double getAvgInferencesDown(HighsInt col) const { return getd(9, col); }
  void flushPseudoCost(HighsPseudocost& pseudocost) {
    highs_rs::highs_rs_pscost_flush(rs_, pseudocost.rs_, false);
  }
  void syncPseudoCost(HighsPseudocost& pseudocost) {
    highs_rs::highs_rs_pscost_flush(rs_, pseudocost.rs_, true);
  }
  void removeChanged() { set(7); }
};
#else
class HighsPseudocost {
  friend struct HighsPseudocostInitialization;
  std::vector<double> pseudocostup;
  std::vector<double> pseudocostdown;
  std::vector<HighsInt> nsamplesup;
  std::vector<HighsInt> nsamplesdown;
  std::vector<double> inferencesup;
  std::vector<double> inferencesdown;
  std::vector<HighsInt> ninferencesup;
  std::vector<HighsInt> ninferencesdown;
  std::vector<HighsInt> ncutoffsup;
  std::vector<HighsInt> ncutoffsdown;
  std::vector<double> conflictscoreup;
  std::vector<double> conflictscoredown;
  std::vector<HighsInt> changedpos;
  std::vector<HighsPseudocostDelta> deltas;

  double conflict_weight;
  double conflict_avg_score;
  double cost_total;
  double inferences_total;
  double delta_cost_sum;
  double delta_inferences_sum;
  int64_t nsamplestotal;
  int64_t ninferencestotal;
  int64_t ncutoffstotal;
  int64_t delta_nsamplestotal;
  int64_t delta_ninferencestotal;
  HighsInt minreliable;
  double degeneracyFactor;

  HighsPseudocostDelta& markChanged(HighsInt col) {
    assert(col >= 0 && col < static_cast<HighsInt>(changedpos.size()));
    if (changedpos[col] == -1) {
      changedpos[col] = static_cast<HighsInt>(deltas.size());
      deltas.emplace_back();
      deltas.back().col = col;
    }
    return deltas[changedpos[col]];
  }

  template <typename CountType>
  static void addDeltaAverage(double& avg, CountType& count, double delta_sum,
                              CountType delta_count) {
    if (delta_count <= 0) return;
    avg = (avg * static_cast<double>(count) + delta_sum) /
          static_cast<double>(count + delta_count);
    count += delta_count;
  }

 public:
  HighsPseudocost() = default;
  HighsPseudocost(const HighsMipSolver& mipsolver);

  void increaseConflictWeight() {
    conflict_weight *= 1.02;

    if (conflict_weight > 1000.0) {
      double scale = 1.0 / conflict_weight;
      conflict_weight = 1.0;
      conflict_avg_score *= scale;

      for (size_t i = 0; i != conflictscoreup.size(); ++i) {
        conflictscoreup[i] *= scale;
        conflictscoredown[i] *= scale;
      }
      for (HighsPseudocostDelta& delta : deltas) {
        delta.conflictscoreup_sum *= scale;
        delta.conflictscoredown_sum *= scale;
      }
    }
  }

  void setDegeneracyFactor(double degeneracyFactor) {
    assert(degeneracyFactor >= 1.0);
    this->degeneracyFactor = degeneracyFactor;
  }

  void increaseConflictScoreUp(HighsInt col) {
    HighsPseudocostDelta& delta = markChanged(col);
    conflictscoreup[col] += conflict_weight;
    conflict_avg_score += conflict_weight;
    delta.conflictscoreup_sum += conflict_weight;
  }

  void increaseConflictScoreDown(HighsInt col) {
    HighsPseudocostDelta& delta = markChanged(col);
    conflictscoredown[col] += conflict_weight;
    conflict_avg_score += conflict_weight;
    delta.conflictscoredown_sum += conflict_weight;
  }

  void setMinReliable(HighsInt minreliable) { this->minreliable = minreliable; }

  HighsInt getMinReliable() const { return minreliable; }

  HighsInt getNumObservations(HighsInt col) const {
    return nsamplesup[col] + nsamplesdown[col];
  }

  HighsInt getNumObservationsUp(HighsInt col) const { return nsamplesup[col]; }

  HighsInt getNumObservationsDown(HighsInt col) const {
    return nsamplesdown[col];
  }

  void addCutoffObservation(HighsInt col, bool upbranch) {
    HighsPseudocostDelta& delta = markChanged(col);
    ++ncutoffstotal;
    if (upbranch) {
      ncutoffsup[col] += 1;
      delta.ncutoffsup += 1;
    } else {
      ncutoffsdown[col] += 1;
      delta.ncutoffsdown += 1;
    }
  }

  void addObservation(HighsInt col, double delta, double objdelta) {
    assert(delta != 0.0);
    assert(objdelta >= 0.0);
    HighsPseudocostDelta& ps_delta = markChanged(col);
    if (delta > 0.0) {
      double unit_gain = objdelta / delta;
      double d = unit_gain - pseudocostup[col];
      nsamplesup[col] += 1;
      pseudocostup[col] += d / nsamplesup[col];
      ps_delta.nsamplesup += 1;
      ps_delta.pseudocostup_sum += unit_gain;

      d = unit_gain - cost_total;
      ++nsamplestotal;
      cost_total += d / static_cast<double>(nsamplestotal);
      ++delta_nsamplestotal;
      delta_cost_sum += unit_gain;
    } else {
      double unit_gain = -objdelta / delta;
      double d = unit_gain - pseudocostdown[col];
      nsamplesdown[col] += 1;
      pseudocostdown[col] += d / nsamplesdown[col];
      ps_delta.nsamplesdown += 1;
      ps_delta.pseudocostdown_sum += unit_gain;

      d = unit_gain - cost_total;
      ++nsamplestotal;
      cost_total += d / static_cast<double>(nsamplestotal);
      ++delta_nsamplestotal;
      delta_cost_sum += unit_gain;
    }
  }

  void addInferenceObservation(HighsInt col, HighsInt ninferences,
                               bool upbranch) {
    HighsPseudocostDelta& delta = markChanged(col);
    double d = ninferences - inferences_total;
    ++ninferencestotal;
    inferences_total += d / static_cast<double>(ninferencestotal);
    ++delta_ninferencestotal;
    delta_inferences_sum += ninferences;
    if (upbranch) {
      d = ninferences - inferencesup[col];
      ninferencesup[col] += 1;
      inferencesup[col] += d / ninferencesup[col];
      delta.ninferencesup += 1;
      delta.inferencesup_sum += ninferences;
    } else {
      d = ninferences - inferencesdown[col];
      ninferencesdown[col] += 1;
      inferencesdown[col] += d / ninferencesdown[col];
      delta.ninferencesdown += 1;
      delta.inferencesdown_sum += ninferences;
    }
  }

  bool isReliable(HighsInt col) const {
    return std::min(nsamplesup[col], nsamplesdown[col]) >= minreliable;
  }

  bool isReliableUp(HighsInt col) const {
    return nsamplesup[col] >= minreliable;
  }

  bool isReliableDown(HighsInt col) const {
    return nsamplesdown[col] >= minreliable;
  }

  double getAvgPseudocost() const { return cost_total; }

  double getPseudocostUp(HighsInt col, double frac, double offset) const {
    double up = std::ceil(frac) - frac;
    double cost;

    if (nsamplesup[col] == 0 || nsamplesup[col] < minreliable) {
      double weightPs =
          nsamplesup[col] == 0
              ? 0
              : 0.9 + 0.1 * nsamplesup[col] / static_cast<double>(minreliable);
      cost = weightPs * pseudocostup[col];
      cost += (1.0 - weightPs) * getAvgPseudocost();
    } else
      cost = pseudocostup[col];
    return up * (offset + cost);
  }

  double getPseudocostDown(HighsInt col, double frac, double offset) const {
    double down = frac - std::floor(frac);
    double cost;

    if (nsamplesdown[col] == 0 || nsamplesdown[col] < minreliable) {
      double weightPs = nsamplesdown[col] == 0
                            ? 0
                            : 0.9 + 0.1 * nsamplesdown[col] /
                                        static_cast<double>(minreliable);
      cost = weightPs * pseudocostdown[col];
      cost += (1.0 - weightPs) * getAvgPseudocost();
    } else
      cost = pseudocostdown[col];

    return down * (offset + cost);
  }

  double getPseudocostUp(HighsInt col, double frac) const {
    double up = std::ceil(frac) - frac;
    if (nsamplesup[col] == 0) return up * cost_total;
    return up * pseudocostup[col];
  }

  double getPseudocostDown(HighsInt col, double frac) const {
    double down = frac - std::floor(frac);
    if (nsamplesdown[col] == 0) return down * cost_total;
    return down * pseudocostdown[col];
  }

  double getConflictScoreUp(HighsInt col) const {
    return conflictscoreup[col] / conflict_weight;
  }

  double getConflictScoreDown(HighsInt col) const {
    return conflictscoredown[col] / conflict_weight;
  }

  double getScore(HighsInt col, double upcost, double downcost) const {
    double costScore = std::max(upcost, minThreshold) *
                       std::max(downcost, minThreshold) /
                       std::max(minThreshold, cost_total * cost_total);
    double inferenceScore =
        std::max(inferencesup[col], minThreshold) *
        std::max(inferencesdown[col], minThreshold) /
        std::max(minThreshold, inferences_total * inferences_total);

    double cutOffScoreUp =
        ncutoffsup[col] /
        std::max(1.0, static_cast<double>(ncutoffsup[col]) +
                          static_cast<double>(nsamplesup[col]));
    double cutOffScoreDown =
        ncutoffsdown[col] /
        std::max(1.0, static_cast<double>(ncutoffsdown[col]) +
                          static_cast<double>(nsamplesdown[col]));
    double avgCutoffs = static_cast<double>(ncutoffstotal) /
                        std::max(1.0, static_cast<double>(ncutoffstotal) +
                                          static_cast<double>(nsamplestotal));

    double cutoffScore = std::max(cutOffScoreUp, minThreshold) *
                         std::max(cutOffScoreDown, minThreshold) /
                         std::max(minThreshold, avgCutoffs * avgCutoffs);

    double conflictScoreUp = conflictscoreup[col] / conflict_weight;
    double conflictScoreDown = conflictscoredown[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score /
        (conflict_weight * static_cast<double>(conflictscoreup.size()));
    double conflictScore =
        std::max(conflictScoreUp, minThreshold) *
        std::max(conflictScoreDown, minThreshold) /
        std::max(minThreshold, conflictScoreAvg * conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };
    return mapScore(costScore) / degeneracyFactor +
           degeneracyFactor *
               (1e-2 * mapScore(conflictScore) +
                1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getScore(HighsInt col, double frac) const {
    double upcost = getPseudocostUp(col, frac);
    double downcost = getPseudocostDown(col, frac);

    return getScore(col, upcost, downcost);
  }

  double getScoreUp(HighsInt col, double frac) const {
    double costScore =
        getPseudocostUp(col, frac) / std::max(minThreshold, cost_total);
    double inferenceScore =
        inferencesup[col] / std::max(minThreshold, inferences_total);

    double cutOffScoreUp =
        ncutoffsup[col] /
        std::max(1.0, static_cast<double>(ncutoffsup[col]) +
                          static_cast<double>(nsamplesup[col]));
    double avgCutoffs = static_cast<double>(ncutoffstotal) /
                        std::max(1.0, static_cast<double>(ncutoffstotal) +
                                          static_cast<double>(nsamplestotal));

    double cutoffScore = cutOffScoreUp / std::max(minThreshold, avgCutoffs);

    double conflictScoreUp = conflictscoreup[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score /
        (conflict_weight * static_cast<double>(conflictscoreup.size()));
    double conflictScore =
        conflictScoreUp / std::max(minThreshold, conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };

    return mapScore(costScore) +
           (1e-2 * mapScore(conflictScore) +
            1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getScoreDown(HighsInt col, double frac) const {
    double costScore =
        getPseudocostDown(col, frac) / std::max(minThreshold, cost_total);
    double inferenceScore =
        inferencesdown[col] / std::max(minThreshold, inferences_total);

    double cutOffScoreDown =
        ncutoffsdown[col] /
        std::max(1.0, static_cast<double>(ncutoffsdown[col]) +
                          static_cast<double>(nsamplesdown[col]));
    double avgCutoffs = static_cast<double>(ncutoffstotal) /
                        std::max(1.0, static_cast<double>(ncutoffstotal) +
                                          static_cast<double>(nsamplestotal));

    double cutoffScore = cutOffScoreDown / std::max(minThreshold, avgCutoffs);

    double conflictScoreDown = conflictscoredown[col] / conflict_weight;
    double conflictScoreAvg =
        conflict_avg_score /
        (conflict_weight * static_cast<double>(conflictscoredown.size()));
    double conflictScore =
        conflictScoreDown / std::max(minThreshold, conflictScoreAvg);

    auto mapScore = [](double score) { return 1.0 - 1.0 / (1.0 + score); };

    return mapScore(costScore) +
           (1e-2 * mapScore(conflictScore) +
            1e-4 * (mapScore(cutoffScore) + mapScore(inferenceScore)));
  }

  double getAvgInferencesUp(HighsInt col) const { return inferencesup[col]; }

  double getAvgInferencesDown(HighsInt col) const {
    return inferencesdown[col];
  }

  void flushPseudoCost(HighsPseudocost& pseudocost) {
    assert(pseudocost.ncutoffsup.size() == this->ncutoffsup.size());
    for (const HighsPseudocostDelta& delta : pseudocost.deltas) {
      const HighsInt col = delta.col;
      assert(col >= 0 &&
             col < static_cast<HighsInt>(pseudocost.ncutoffsup.size()));
      addDeltaAverage(this->pseudocostup[col], this->nsamplesup[col],
                      delta.pseudocostup_sum, delta.nsamplesup);
      addDeltaAverage(this->pseudocostdown[col], this->nsamplesdown[col],
                      delta.pseudocostdown_sum, delta.nsamplesdown);
      addDeltaAverage(this->inferencesup[col], this->ninferencesup[col],
                      delta.inferencesup_sum, delta.ninferencesup);
      addDeltaAverage(this->inferencesdown[col], this->ninferencesdown[col],
                      delta.inferencesdown_sum, delta.ninferencesdown);
      const double conflict_scale =
          this->conflict_weight / pseudocost.conflict_weight;
      const double conflict_delta_up =
          conflict_scale * delta.conflictscoreup_sum;
      const double conflict_delta_down =
          conflict_scale * delta.conflictscoredown_sum;
      this->conflictscoreup[col] += conflict_delta_up;
      this->conflictscoredown[col] += conflict_delta_down;
      this->conflict_avg_score += conflict_delta_up + conflict_delta_down;
      this->ncutoffsup[col] += delta.ncutoffsup;
      this->ncutoffsdown[col] += delta.ncutoffsdown;
      this->ncutoffstotal += delta.ncutoffsup + delta.ncutoffsdown;
    }
    addDeltaAverage(this->cost_total, this->nsamplestotal,
                    pseudocost.delta_cost_sum, pseudocost.delta_nsamplestotal);
    addDeltaAverage(this->inferences_total, this->ninferencestotal,
                    pseudocost.delta_inferences_sum,
                    pseudocost.delta_ninferencestotal);
    pseudocost.removeChanged();
  }

  void syncPseudoCost(HighsPseudocost& pseudocost) {
    std::copy(pseudocostup.begin(), pseudocostup.end(),
              pseudocost.pseudocostup.begin());
    std::copy(pseudocostdown.begin(), pseudocostdown.end(),
              pseudocost.pseudocostdown.begin());
    std::copy(nsamplesup.begin(), nsamplesup.end(),
              pseudocost.nsamplesup.begin());
    std::copy(nsamplesdown.begin(), nsamplesdown.end(),
              pseudocost.nsamplesdown.begin());
    std::copy(inferencesup.begin(), inferencesup.end(),
              pseudocost.inferencesup.begin());
    std::copy(inferencesdown.begin(), inferencesdown.end(),
              pseudocost.inferencesdown.begin());
    std::copy(ninferencesup.begin(), ninferencesup.end(),
              pseudocost.ninferencesup.begin());
    std::copy(ninferencesdown.begin(), ninferencesdown.end(),
              pseudocost.ninferencesdown.begin());
    std::copy(ncutoffsup.begin(), ncutoffsup.end(),
              pseudocost.ncutoffsup.begin());
    std::copy(ncutoffsdown.begin(), ncutoffsdown.end(),
              pseudocost.ncutoffsdown.begin());
    std::copy(conflictscoreup.begin(), conflictscoreup.end(),
              pseudocost.conflictscoreup.begin());
    std::copy(conflictscoredown.begin(), conflictscoredown.end(),
              pseudocost.conflictscoredown.begin());
    pseudocost.conflict_weight = conflict_weight;
    pseudocost.conflict_avg_score = conflict_avg_score;
    pseudocost.cost_total = cost_total;
    pseudocost.inferences_total = inferences_total;
    pseudocost.nsamplestotal = nsamplestotal;
    pseudocost.ninferencestotal = ninferencestotal;
    pseudocost.ncutoffstotal = ncutoffstotal;
    pseudocost.removeChanged();
  }

  void removeChanged() {
    for (const HighsPseudocostDelta& delta : deltas) {
      changedpos[delta.col] = -1;
    }
    deltas.clear();
    delta_cost_sum = 0.0;
    delta_inferences_sum = 0.0;
    delta_nsamplestotal = 0;
    delta_ninferencestotal = 0;
  }
};

#endif  // HIGHS_RUST

#endif
