//! The option values as a typed Rust copy ([`Opts`]): one field per
//! HighsOptions record, named as the option, in the records' order. The
//! C++ HighsOptions stays the public store (the API, highspy and the C API
//! hand out references to it): a `Highs` object's values are copied in
//! ([`Opts::sync`]) when a run builds what its solvers read, and the LP
//! solver of the MIP's LP relaxation ([`super::lp_handle::LpHandle`]) owns
//! its options only here. What the solvers read is built from an `Opts`:
//! the simplex's [`LpsOptions`], the KKT check's, assessLp's, IPX's, the
//! LP presolve's and the run's option views.

use super::ffi::CLpOptions;
use super::ipx_glue::CIpxOptions;
use super::options::{COptionRecord, RsStr};
use super::run::ROptions;
use super::solution::CKktOptions;
use super::{Log, INF};
use crate::simplex::lp_solver::LpsOptions;

/// A value for a name-based set
#[derive(Clone, Copy, Debug)]
pub enum OptValue<'a> {
    Bool(bool),
    Int(i32),
    Double(f64),
    Str(&'a [u8]),
}

// HighsOptionType
const TYPE_BOOL: i32 = 0;
const TYPE_INT: i32 = 1;
const TYPE_DOUBLE: i32 = 2;

macro_rules! opt_ty {
    (bool) => { bool };
    (int) => { i32 };
    (double) => { f64 };
    (string) => { Vec<u8> };
}

macro_rules! opt_val {
    (bool, $v:expr) => { $v };
    (int, $v:expr) => { $v };
    (double, $v:expr) => { $v };
    (string, $v:expr) => {{
        // room for any value, so that a value set later keeps the buffer
        let mut s = Vec::with_capacity(32);
        s.extend_from_slice($v);
        s
    }};
}

/// The record's value into the field (the record's type is the field's)
macro_rules! opt_take {
    (bool, $f:expr, $r:expr) => { $f = *($r.value as *const bool) };
    (int, $f:expr, $r:expr) => { $f = *($r.value as *const i32) };
    (double, $f:expr, $r:expr) => { $f = *($r.value as *const f64) };
    (string, $f:expr, $r:expr) => {{
        $f.clear();
        $f.extend_from_slice($r.str_value.get());
    }};
}

/// Whether the field holds the record's value
macro_rules! opt_same {
    (bool, $f:expr, $r:expr) => { $f == *($r.value as *const bool) };
    (int, $f:expr, $r:expr) => { $f == *($r.value as *const i32) };
    (double, $f:expr, $r:expr) => { $f.to_bits() == (*($r.value as *const f64)).to_bits() };
    (string, $f:expr, $r:expr) => { $f.as_slice() == $r.str_value.get() };
}

macro_rules! opt_set {
    (bool, $f:expr, $v:expr) => {
        match $v {
            OptValue::Bool(b) => $f = b,
            _ => return false,
        }
    };
    (int, $f:expr, $v:expr) => {
        match $v {
            OptValue::Int(i) => $f = i,
            _ => return false,
        }
    };
    (double, $f:expr, $v:expr) => {
        match $v {
            OptValue::Double(d) => $f = d,
            // an int for a double option, as Highs::setOptionValue
            OptValue::Int(i) => $f = i as f64,
            _ => return false,
        }
    };
    (string, $f:expr, $v:expr) => {
        match $v {
            OptValue::Str(s) => {
                $f.clear();
                $f.extend_from_slice(s);
            }
            _ => return false,
        }
    };
}

/// The field set from another options' field, a string keeping its
/// buffer (C++ string assignment keeps an SSO buffer, so views of the
/// value taken at a run's start stay valid)
macro_rules! opt_assign {
    (string, $f:expr, $o:expr) => {{
        $f.clear();
        $f.extend_from_slice(&$o);
    }};
    ($kind:ident, $f:expr, $o:expr) => { $f = $o };
}

macro_rules! opt_get {
    (bool, $f:expr) => { OptValue::Bool($f) };
    (int, $f:expr) => { OptValue::Int($f) };
    (double, $f:expr) => { OptValue::Double($f) };
    (string, $f:expr) => { OptValue::Str(&$f) };
}

macro_rules! options {
    ($($name:ident: $kind:ident = $default:expr;)*) => {
        /// HighsOptions' values (see the module comment)
        #[allow(non_snake_case)]
        #[derive(Clone, Debug, PartialEq)]
        pub struct Opts {
            $(pub $name: opt_ty!($kind),)*
        }

        impl Default for Opts {
            /// HighsOptions()
            fn default() -> Opts {
                Opts { $($name: opt_val!($kind, $default),)* }
            }
        }

        /// The option names in the records' order
        pub const NAMES: &[&str] = &[$(stringify!($name),)*];

        /// The option names and their kinds
        const KINDS: &[(&str, &str)] = &[$((stringify!($name), stringify!($kind)),)*];

        impl Opts {
            /// The value of record `r` into field `i` (`NAMES[i]` is its
            /// name)
            ///
            /// # Safety
            /// The record's value pointer and strings are valid
            unsafe fn take(&mut self, i: usize, r: &COptionRecord) {
                let mut k = 0usize;
                $(
                    if k == i {
                        opt_take!($kind, self.$name, r);
                        return;
                    }
                    k += 1;
                )*
                let _ = k;
            }

            /// Whether field `i` holds record `r`'s value
            ///
            /// # Safety
            /// As take
            unsafe fn same(&self, i: usize, r: &COptionRecord) -> bool {
                let mut k = 0usize;
                $(
                    if k == i {
                        return opt_same!($kind, self.$name, r);
                    }
                    k += 1;
                )*
                let _ = k;
                false
            }

            /// Every value from `o` (passOptions; restoring saved options)
            pub fn assign(&mut self, o: &Opts) {
                $(opt_assign!($kind, self.$name, o.$name);)*
            }

            /// setOptionValue(name, value): false for an unknown name or a
            /// value of the wrong type (an int is taken for a double)
            pub fn set(&mut self, name: &str, value: OptValue) -> bool {
                match name {
                    $(stringify!($name) => opt_set!($kind, self.$name, value),)*
                    _ => return false,
                }
                true
            }

            /// getOptionValue(name)
            pub fn get(&self, name: &str) -> Option<OptValue<'_>> {
                match name {
                    $(stringify!($name) => Some(opt_get!($kind, self.$name)),)*
                    _ => None,
                }
            }
        }
    };
}

fn record_type(kind: &str) -> i32 {
    match kind {
        "bool" => TYPE_BOOL,
        "int" => TYPE_INT,
        "double" => TYPE_DOUBLE,
        _ => 3,
    }
}

impl Opts {
    /// The index of a name's field
    fn index(name: &[u8]) -> Option<usize> {
        NAMES.iter().position(|n| n.as_bytes() == name)
    }

    /// The values of the C++ records (HighsOptions::records) into the
    /// fields: the records come in the fields' order, so each name is
    /// checked once (a name out of place is looked up)
    ///
    /// # Safety
    /// The records' value pointers and strings are valid
    pub unsafe fn sync(&mut self, recs: &[COptionRecord]) {
        for (i, r) in recs.iter().enumerate() {
            let name = r.name.get();
            if NAMES.get(i).is_some_and(|n| n.as_bytes() == name) {
                self.take(i, r);
            } else if let Some(j) = Self::index(name) {
                self.take(j, r);
            }
        }
    }

    /// The names whose field differs from the C++ record, and the records
    /// without a field (a check of the defaults and of the table)
    ///
    /// # Safety
    /// As sync
    pub unsafe fn diff(&self, recs: &[COptionRecord]) -> Vec<String> {
        let mut d = Vec::new();
        for r in recs {
            let name = r.name.get();
            match Self::index(name) {
                Some(i) if self.same(i, r) => {}
                _ => d.push(String::from_utf8_lossy(name).into_owned()),
            }
        }
        if recs.len() != NAMES.len() {
            d.push(format!("{} records, {} fields", recs.len(), NAMES.len()));
        }
        d
    }

    /// The kind of a name's field (HighsOptionType), -1 for none
    pub fn kind(name: &str) -> i32 {
        match KINDS.iter().find(|(n, _)| *n == name) {
            Some((_, k)) => record_type(k),
            None => -1,
        }
    }

    /// LpsOptions: what the simplex reads
    pub fn lps(&self) -> LpsOptions {
        LpsOptions {
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            dual_feasibility_tolerance: self.dual_feasibility_tolerance,
            time_limit: self.time_limit,
            objective_bound: self.objective_bound,
            dual_simplex_pivot_growth_tolerance: self.dual_simplex_pivot_growth_tolerance,
            small_matrix_value: self.small_matrix_value,
            dual_steepest_edge_weight_error_tolerance: self.dual_steepest_edge_weight_error_tolerance,
            rebuild_refactor_solution_error_tolerance: self.rebuild_refactor_solution_error_tolerance,
            factor_pivot_tolerance: self.factor_pivot_tolerance,
            factor_pivot_threshold: self.factor_pivot_threshold,
            dual_simplex_cost_perturbation_multiplier: self.dual_simplex_cost_perturbation_multiplier,
            primal_simplex_bound_perturbation_multiplier: self.primal_simplex_bound_perturbation_multiplier,
            dual_steepest_edge_weight_log_error_threshold: self.dual_steepest_edge_weight_log_error_threshold,
            cost_scale_factor: self.cost_scale_factor,
            log_dev_level: self.log_dev_level,
            dev_level: if self.output_flag { self.log_dev_level } else { 0 },
            simplex_primal_edge_weight_strategy: self.simplex_primal_edge_weight_strategy,
            simplex_iteration_limit: self.simplex_iteration_limit,
            simplex_update_limit: self.simplex_update_limit,
            max_dual_simplex_cleanup_level: self.max_dual_simplex_cleanup_level,
            max_dual_simplex_phase1_cleanup_level: self.max_dual_simplex_phase1_cleanup_level,
            simplex_dse_exact_init_max_rows: self.simplex_dse_exact_init_max_rows,
            simplex_strategy: self.simplex_strategy,
            simplex_min_concurrency: self.simplex_min_concurrency,
            simplex_max_concurrency: self.simplex_max_concurrency,
            simplex_dual_edge_weight_strategy: self.simplex_dual_edge_weight_strategy,
            simplex_price_strategy: self.simplex_price_strategy,
            random_seed: self.random_seed,
            output_flag: self.output_flag,
            no_unnecessary_rebuild_refactor: self.no_unnecessary_rebuild_refactor,
            allow_unbounded_or_infeasible: self.allow_unbounded_or_infeasible,
            less_infeasible_dse_check: self.less_infeasible_DSE_check,
            less_infeasible_dse_choose_row: self.less_infeasible_DSE_choose_row,
            simplex_keep_random_vectors: self.simplex_keep_random_vectors,
        }
    }

    /// The options of the KKT checks (rsKktOptions)
    pub fn kkt(&self, log: Log) -> CKktOptions {
        CKktOptions {
            log,
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            dual_feasibility_tolerance: self.dual_feasibility_tolerance,
            mip_feasibility_tolerance: self.mip_feasibility_tolerance,
            primal_residual_tolerance: self.primal_residual_tolerance,
            dual_residual_tolerance: self.dual_residual_tolerance,
            optimality_tolerance: self.optimality_tolerance,
            kkt_tolerance: self.kkt_tolerance,
            log_dev_level: self.log_dev_level,
            full_lp_kkt_check: self.full_lp_kkt_check,
        }
    }

    /// The options lp_utils reads (rsLpOptions)
    pub fn lp_options(&self, log: Log) -> CLpOptions {
        CLpOptions {
            log,
            infinite_cost: self.infinite_cost,
            infinite_bound: self.infinite_bound,
            small_matrix_value: self.small_matrix_value,
            large_matrix_value: self.large_matrix_value,
            simplex_scale_strategy: self.simplex_scale_strategy,
            allowed_matrix_scale_factor: self.allowed_matrix_scale_factor,
            highs_analysis_level: self.highs_analysis_level,
            log_dev_level: self.log_dev_level,
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
        }
    }

    /// The options solveLpIpx reads (ipxOptions); `log_options` goes to
    /// IPX's log hook
    pub fn ipx(&self, log: Log, log_options: *const std::ffi::c_void) -> CIpxOptions {
        CIpxOptions {
            log,
            log_options,
            output_flag: self.output_flag,
            log_to_console: self.log_to_console,
            timeless_log: self.timeless_log,
            run_centring: self.run_centring,
            log_dev_level: self.log_dev_level,
            ipx_dualize_strategy: self.ipx_dualize_strategy,
            highs_analysis_level: self.highs_analysis_level,
            ipm_iteration_limit: self.ipm_iteration_limit,
            run_crossover: match self.run_crossover.as_slice() {
                b"on" => 1,
                b"off" => 0,
                _ => -1,
            },
            max_centring_steps: self.max_centring_steps,
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            dual_feasibility_tolerance: self.dual_feasibility_tolerance,
            ipm_optimality_tolerance: self.ipm_optimality_tolerance,
            start_crossover_tolerance: self.start_crossover_tolerance,
            kkt_tolerance: self.kkt_tolerance,
            time_limit: self.time_limit,
            centring_ratio_tolerance: self.centring_ratio_tolerance,
        }
    }

    /// The option values HPresolve reads (rsOptions)
    pub fn presolve(&self) -> crate::presolve::hpresolve::Options {
        crate::presolve::hpresolve::Options {
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            dual_feasibility_tolerance: self.dual_feasibility_tolerance,
            mip_feasibility_tolerance: self.mip_feasibility_tolerance,
            small_matrix_value: self.small_matrix_value,
            time_limit: self.time_limit,
            presolve_pivot_threshold: self.presolve_pivot_threshold,
            presolve_substitution_maxfillin: self.presolve_substitution_maxfillin,
            presolve_rule_test: self.presolve_rule_test,
            presolve_rule_off: self.presolve_rule_off,
            log_dev_level: self.log_dev_level,
            random_seed: self.random_seed,
            mip_lifting_for_probing: self.mip_lifting_for_probing,
            presolve_off: self.presolve == b"off",
            lp_presolve_requires_basis_postsolve: self.lp_presolve_requires_basis_postsolve,
            presolve_remove_slacks: self.presolve_remove_slacks,
            output_flag: self.output_flag,
            timeless_log: self.timeless_log,
            use_implied_bounds_from_presolve: self.use_implied_bounds_from_presolve,
            presolve_rule_logging: self.presolve_rule_logging,
        }
    }

    /// The run's option view, its pointers into this copy (valid while it
    /// is not moved)
    pub fn r_options(&mut self) -> ROptions {
        ROptions {
            solver: RsStr::of(&self.solver),
            run_crossover: RsStr::of(&self.run_crossover),
            presolve: RsStr::of(&self.presolve),
            use_warm_start: self.use_warm_start,
            icrash: self.icrash,
            solve_relaxation: self.solve_relaxation,
            allow_unbounded_or_infeasible: self.allow_unbounded_or_infeasible,
            timeless_log: self.timeless_log,
            large_matrix_value: self.large_matrix_value,
            time_limit: self.time_limit,
            primal_feasibility_tolerance: self.primal_feasibility_tolerance,
            mip_feasibility_tolerance: self.mip_feasibility_tolerance,
            highs_debug_level: &mut self.highs_debug_level,
            objective_bound: &mut self.objective_bound,
            lp_presolve_requires_basis_postsolve: &mut self.lp_presolve_requires_basis_postsolve,
            output_flag: &self.output_flag,
            log_dev_level: &self.log_dev_level,
            simplex_strategy: self.simplex_strategy,
        }
    }
}

options! {
    presolve: string = b"choose";
    solver: string = b"choose";
    parallel: string = b"choose";
    threads: int = 0;
    run_crossover: string = b"on";
    time_limit: double = INF;
    ranging: string = b"off";
    infinite_cost: double = 1e+20;
    infinite_bound: double = 1e+20;
    small_matrix_value: double = 1e-09;
    large_matrix_value: double = 1000000000000000.0;
    kkt_tolerance: double = 1e-07;
    primal_feasibility_tolerance: double = 1e-07;
    dual_feasibility_tolerance: double = 1e-07;
    primal_residual_tolerance: double = 1e-07;
    dual_residual_tolerance: double = 1e-07;
    optimality_tolerance: double = 1e-07;
    objective_bound: double = INF;
    objective_target: double = -INF;
    random_seed: int = 0;
    user_objective_scale: int = 0;
    user_bound_scale: int = 0;
    highs_debug_level: int = 0;
    highs_analysis_level: int = 0;
    simplex_strategy: int = 1;
    simplex_scale_strategy: int = 2;
    simplex_crash_strategy: int = 0;
    simplex_dual_edge_weight_strategy: int = -1;
    simplex_primal_edge_weight_strategy: int = -1;
    simplex_iteration_limit: int = i32::MAX;
    simplex_update_limit: int = 5000;
    simplex_min_concurrency: int = 1;
    simplex_max_concurrency: int = 8;
    output_flag: bool = true;
    log_to_console: bool = true;
    timeless_log: bool = false;
    log_file: string = b"";
    write_model_to_file: bool = false;
    write_presolved_model_to_file: bool = false;
    write_solution_to_file: bool = false;
    write_solution_style: int = 0;
    glpsol_cost_row_location: int = 0;
    icrash: bool = false;
    icrash_dualize: bool = false;
    icrash_strategy: string = b"ICA";
    icrash_starting_weight: double = 0.001;
    icrash_iterations: int = 30;
    icrash_approx_iter: int = 50;
    icrash_exact: bool = false;
    icrash_breakpoints: bool = false;
    read_solution_file: string = b"";
    read_basis_file: string = b"";
    write_model_file: string = b"";
    solution_file: string = b"";
    write_basis_file: string = b"";
    write_presolved_model_file: string = b"";
    write_iis_model_file: string = b"";
    mip_detect_symmetry: bool = true;
    mip_allow_restart: bool = true;
    mip_max_nodes: int = i32::MAX;
    mip_max_stall_nodes: int = i32::MAX;
    mip_max_start_nodes: int = 500;
    mip_improving_solution_save: bool = false;
    mip_improving_solution_report_sparse: bool = false;
    mip_improving_solution_file: string = b"";
    mip_root_presolve_only: bool = false;
    mip_lifting_for_probing: int = -1;
    mip_max_leaves: int = i32::MAX;
    mip_max_improving_sols: int = i32::MAX;
    mip_lp_age_limit: int = 10;
    mip_pool_age_limit: int = 30;
    mip_pool_soft_limit: int = 10000;
    mip_pscost_minreliable: int = 8;
    mip_min_cliquetable_entries_for_parallelism: int = 100000;
    mip_report_level: int = 1;
    mip_feasibility_tolerance: double = 1e-06;
    mip_heuristic_effort: double = 0.05;
    mip_heuristic_run_feasibility_jump: bool = true;
    mip_heuristic_run_rins: bool = true;
    mip_heuristic_run_rens: bool = true;
    mip_heuristic_run_graph_lns: bool = true;
    mip_concurrent_helper: bool = true;
    mip_concurrent_crossover: bool = false;
    mip_heuristic_run_root_reduced_cost: bool = true;
    mip_heuristic_run_zi_round: bool = false;
    mip_heuristic_run_shifting: bool = false;
    mip_allow_cut_separation_at_nodes: bool = true;
    mip_rel_gap: double = 0.0001;
    mip_abs_gap: double = 1e-06;
    mip_min_logging_interval: double = 5.0;
    mip_lp_solver: string = b"choose";
    mip_ipm_solver: string = b"choose";
    ipm_optimality_tolerance: double = 1e-08;
    mip_search_simulate_concurrency: bool = false;
    ipm_iteration_limit: int = i32::MAX;
    hipo_system: string = b"choose";
    hipo_parallel_type: string = b"both";
    hipo_ordering: string = b"choose";
    hipo_block_size: int = 128;
    pdlp_iteration_limit: int = i32::MAX;
    pdlp_scaling_mode: int = 5;
    pdlp_ruiz_iterations: int = 10;
    pdlp_restart_strategy: int = 2;
    pdlp_cupdlpc_restart_method: int = 1;
    pdlp_step_size_strategy: int = 1;
    pdlp_optimality_tolerance: double = 1e-07;
    qp_allow_hot_start: bool = false;
    qp_iteration_limit: int = i32::MAX;
    qp_nullspace_limit: int = 4000;
    qp_regularization_value: double = 1e-07;
    iis_strategy: int = 0;
    iis_time_limit: double = INF;
    blend_multi_objectives: bool = true;
    log_dev_level: int = 0;
    log_githash: bool = true;
    solve_relaxation: bool = false;
    allow_unbounded_or_infeasible: bool = false;
    use_implied_bounds_from_presolve: bool = false;
    lp_presolve_requires_basis_postsolve: bool = true;
    mps_parser_type_free: bool = true;
    use_warm_start: bool = true;
    write_matrix_image: bool = false;
    write_hessian_image: bool = false;
    keep_n_rows: int = -1;
    cost_scale_factor: int = 0;
    allowed_matrix_scale_factor: int = 20;
    allowed_cost_scale_factor: int = 0;
    ipx_dualize_strategy: int = 2;
    simplex_dualize_strategy: int = -1;
    simplex_permute_strategy: int = -1;
    max_dual_simplex_cleanup_level: int = 1;
    max_dual_simplex_phase1_cleanup_level: int = 2;
    simplex_price_strategy: int = 3;
    simplex_dse_exact_init_max_rows: int = i32::MAX;
    simplex_keep_random_vectors: bool = false;
    full_lp_kkt_check: bool = true;
    simplex_unscaled_solution_strategy: int = 1;
    no_unnecessary_rebuild_refactor: bool = true;
    rebuild_refactor_solution_error_tolerance: double = 1e-08;
    dual_steepest_edge_weight_error_tolerance: double = INF;
    dual_steepest_edge_weight_log_error_threshold: double = 10.0;
    dual_simplex_cost_perturbation_multiplier: double = 1.0;
    primal_simplex_bound_perturbation_multiplier: double = 1.0;
    dual_simplex_pivot_growth_tolerance: double = 1e-09;
    presolve_pivot_threshold: double = 0.01;
    presolve_reduction_limit: int = -1;
    restart_presolve_reduction_limit: int = -1;
    presolve_rule_off: int = 0;
    presolve_rule_test: int = 0;
    presolve_rule_logging: bool = false;
    presolve_remove_slacks: bool = false;
    presolve_substitution_maxfillin: int = 10;
    factor_pivot_threshold: double = 0.1;
    factor_pivot_tolerance: double = 1e-10;
    start_crossover_tolerance: double = 1e-08;
    use_original_HFactor_logic: bool = true;
    less_infeasible_DSE_check: bool = true;
    less_infeasible_DSE_choose_row: bool = true;
    run_centring: bool = false;
    max_centring_steps: int = 5;
    centring_ratio_tolerance: double = 100.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_set_and_get() {
        let mut o = Opts::default();
        assert_eq!(NAMES.len(), 161);
        assert!(o.set("presolve", OptValue::Str(b"off")));
        assert_eq!(o.presolve, b"off");
        assert!(o.set("time_limit", OptValue::Int(3)));
        assert_eq!(o.time_limit, 3.0);
        assert!(!o.set("time_limit", OptValue::Bool(true)));
        assert!(!o.set("no_such_option", OptValue::Int(1)));
        assert!(matches!(o.get("simplex_strategy"), Some(OptValue::Int(1))));
        assert_eq!(Opts::kind("output_flag"), TYPE_BOOL);
        assert_eq!(Opts::kind("solver"), 3);
        assert_eq!(o.lps().dev_level, 0);
    }
}
