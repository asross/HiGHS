/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsCliqueTableRust.h
 * @brief The C side of the Rust clique table and implications
 * (rust/src/mip/clique.rs, implications.rs)
 */
#ifndef MIP_HIGHSCLIQUETABLERUST_H_
#define MIP_HIGHSCLIQUETABLERUST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST

#include <cstdint>

#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"
#include "mip/HighsDomainChange.h"
#include "util/HighsRandom.h"

namespace highs_rs {

using ClqVar = HighsCliqueTable::CliqueVar;

/// A HighsDomain (rust CDom)
struct CliqueDom {
  void* ctx;
  const double* col_lower;
  const double* col_upper;
  const uint8_t* integrality;
  HighsInt num_col;
  HighsInt num_nonzero;
  double feastol;
  bool (*infeasible)(void*);
  void (*change_bound)(void*, HighsInt, HighsInt, double, HighsInt, HighsInt);
  void (*fix_col)(void*, HighsInt, double);
  void (*propagate)(void*);
  const HighsDomainChange* (*domchg_stack)(void*, HighsInt*);
};

/// The MIP solver's parts the clique table uses (rust CMip)
struct CliqueMip {
  void* ctx;
  double feastol;
  double epsilon;
  HighsInt num_clique_entries_after_presolve;
  HighsInt num_nonzero;
  void (*prune_edge)(void*, ClqVar, ClqVar);
  bool (*too_many_var_bounds)(void*);
  void (*add_vub)(void*, HighsInt, HighsInt, double, double);
  void (*add_vlb)(void*, HighsInt, HighsInt, double, double);
};

/// rust CRows
struct CliqueRows {
  HighsInt num_row;
  const HighsInt* ar_start;
  const HighsInt* ar_index;
  const double* ar_value;
  HighsInt num_nz;
  const double* row_lower;
  const double* row_upper;
};

/// rust CSepaCliques
struct CliqueSepa {
  const double* sol;
  HighsInt num_col;
  const HighsInt* integral_cols;
  HighsInt num_integral_cols;
  double feastol;
  int64_t max_neighbourhood_queries;
  void* ctx;
  void (*add_cut)(void*, HighsInt*, double*, HighsInt, double);
};

CliqueDom cliqueDom(HighsDomain& dom);
CliqueMip cliqueMip(const HighsMipSolver& mipsolver);

extern "C" {
CliqueTable* highs_rs_clique_new(HighsInt ncols);
void highs_rs_clique_free(CliqueTable* t);
HighsInt highs_rs_clique_get(const CliqueTable* t, HighsInt which);
void highs_rs_clique_set(CliqueTable* t, HighsInt which, HighsInt value);
HighsRandom* highs_rs_clique_randgen(CliqueTable* t);
int64_t* highs_rs_clique_num_queries(CliqueTable* t);
void* highs_rs_clique_vec(CliqueTable* t, int which, size_t* len);
void highs_rs_clique_vec_clear(CliqueTable* t, int which);
const void* highs_rs_clique_substitution(const CliqueTable* t, HighsInt col);
HighsInt highs_rs_clique_num_cliques_var(const CliqueTable* t, ClqVar v);
bool highs_rs_clique_have_common(CliqueTable* t, int64_t* numQueries,
                                 ClqVar v1, ClqVar v2);
const ClqVar* highs_rs_clique_find_common(CliqueTable* t, ClqVar v1,
                                          ClqVar v2, HighsInt* len);
void highs_rs_clique_resolve_subst_val(const CliqueTable* t, HighsInt* col,
                                       double* val, double* offset);
void highs_rs_clique_resolve_subst(const CliqueTable* t, ClqVar* v);
HighsInt highs_rs_clique_num_implications(const CliqueTable* t, HighsInt col,
                                          HighsInt val);
void highs_rs_clique_add_implications(const CliqueTable* t,
                                      const CliqueDom* dom, HighsInt col,
                                      HighsInt val);
void highs_rs_clique_add_clique(CliqueTable* t, const CliqueDom* dom,
                                const CliqueMip* mip, ClqVar* vars, HighsInt n,
                                bool equality, HighsInt origin);
void highs_rs_clique_remove_clique(CliqueTable* t, HighsInt cliqueid);
void highs_rs_clique_do_add_clique(CliqueTable* t, const ClqVar* vars,
                                   HighsInt n, bool equality, HighsInt origin);
HighsInt highs_rs_clique_partition(CliqueTable* t, const double* objective,
                                   HighsInt num_col, ClqVar* vars, HighsInt n,
                                   HighsInt* starts);
bool highs_rs_clique_found_cover(CliqueTable* t, const CliqueDom* dom,
                                 ClqVar v1, ClqVar v2);
void highs_rs_clique_extract_cliques(CliqueTable* t, const CliqueDom* dom,
                                     const CliqueMip* mip,
                                     const CliqueRows* rows,
                                     bool transform_rows);
void highs_rs_clique_extract_from_cut(CliqueTable* t, const CliqueDom* dom,
                                      const CliqueMip* mip,
                                      const HighsInt* inds, const double* vals,
                                      HighsInt len, double rhs);
void highs_rs_clique_extract_obj(CliqueTable* t, const CliqueDom* dom,
                                 const CliqueMip* mip, HighsInt nbin,
                                 const double* vals, const HighsInt* inds,
                                 HighsInt len, double rhs, double minact_hi,
                                 double minact_lo);
void highs_rs_clique_vertex_infeasible(CliqueTable* t, const CliqueDom* dom,
                                       HighsInt col, HighsInt val);
void highs_rs_clique_cleanup_fixed(CliqueTable* t, const CliqueDom* dom);
void highs_rs_clique_run_merging(CliqueTable* t, const CliqueDom* dom);
void highs_rs_clique_separate(CliqueTable* t, const CliqueDom* dom,
                              const CliqueSepa* s, HighsRandom* randgen,
                              int64_t* local_num_queries);
void highs_rs_clique_maximal_cliques(const CliqueTable* t, const ClqVar* vars,
                                     HighsInt n, double feastol, void* out,
                                     void (*push)(void*, const ClqVar*,
                                                  HighsInt));
void highs_rs_clique_rebuild(CliqueTable* t, HighsInt ncols,
                             const HighsInt* orig2reducedcol, HighsInt norig,
                             const uint8_t* keep);
void highs_rs_clique_build_from(CliqueTable* t, const double* orig_lower,
                                const double* orig_upper, HighsInt num_col,
                                const CliqueTable* init);
}

/// The MIP solver's parts the implications use (rust CImp)
struct ImplicsHost {
  void* ctx;
  double feastol;
  double epsilon;
  HighsInt num_nonzero;
  int64_t (*num_nodes_down)(void*, HighsInt);
  int64_t (*num_nodes_up)(void*, HighsInt);
  void (*lifting_begin)(void*);
  void (*lifting_store)(void*, HighsInt, bool);
  HighsInt (*domchg_reason)(void*, HighsInt, HighsInt*);
  size_t (*changed_cols_len)(void*);
  void (*backtrack)(void*, size_t);
  void (*vertex_infeasible)(void*, HighsInt, HighsInt);
  void (*add_inference_observation)(void*, HighsInt, HighsInt, bool);
  HighsInt (*clique_num_entries)(void*);
  void (*add_clique2)(void*, ClqVar*);
  bool (*clique_substituted)(void*, HighsInt);
  bool (*parallel_lock_active)(void*);
  bool (*clique_is_full)(void*);
  int64_t* (*clique_num_queries)(void*);
  void (*run_clique_merging)(void*);
  HighsInt (*num_clique_entries_after_first_presolve)(void*);
  void (*probing_clock)(void*, bool);
  void (*add_cut)(void*, HighsInt*, double*, HighsInt, double, bool, bool);
};

struct ImplicsVarBound {
  double coef;
  double constant;
};

extern "C" {
Implications* highs_rs_implics_new(HighsInt numcol, HighsInt num_nonzero);
void highs_rs_implics_free(Implications* t);
void* highs_rs_implics_vec(Implications* t, int which, size_t* len);
void highs_rs_implics_vec_clear(Implications* t, int which);
int64_t highs_rs_implics_get(const Implications* t, HighsInt which);
void highs_rs_implics_add_vb(Implications* t, bool vlb, HighsInt col,
                             HighsInt vbcol, double coef, double constant,
                             double bound, bool isint, double feastol);
void highs_rs_implics_column_transformed(Implications* t, HighsInt col,
                                         double scale, double constant);
HighsInt highs_rs_implics_best_vb(const Implications* t, bool vlb,
                                  const ImplicsHost* imp, const CliqueDom* dom,
                                  HighsInt col, const double* col_value,
                                  const double* col_dual, HighsInt num_col,
                                  double* bound, ImplicsVarBound* vb);
void highs_rs_implics_cleanup_vb(bool vlb, const CliqueDom* g, double feastol,
                                 double epsilon, HighsInt col, HighsInt vbcol,
                                 ImplicsVarBound* vb, double bound,
                                 bool allow_bound_changes, bool* redundant,
                                 bool* infeasible);
bool highs_rs_implics_run_probing(Implications* t, const CliqueDom* g,
                                  const ImplicsHost* imp, HighsInt col,
                                  HighsInt* num_reductions);
void highs_rs_implics_cleanup_varbounds(Implications* t, const CliqueDom* g,
                                        const ImplicsHost* imp, HighsInt col);
void highs_rs_implics_separate(Implications* t, const CliqueDom* g,
                               const ImplicsHost* imp, const CliqueDom* dom,
                               const std::pair<HighsInt, double>* fracints,
                               HighsInt num_frac, const double* sol,
                               HighsInt num_sol, double feastol,
                               bool thread_safe);
void highs_rs_implics_apply(const Implications* t, const CliqueDom* dom,
                            HighsInt col, HighsInt val);
void highs_rs_implics_rebuild(Implications* t, const CliqueDom* g,
                              HighsInt ncols, const HighsInt* orig2reducedcol,
                              HighsInt norig, const uint8_t* transformable);
void highs_rs_implics_build_from(Implications* t, const CliqueDom* g,
                                 const Implications* init);
}

}  // namespace highs_rs

#endif  // HIGHS_RUST
#endif  // MIP_HIGHSCLIQUETABLERUST_H_
