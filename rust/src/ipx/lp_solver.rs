//! LpSolver (lp_solver.h/.cc): the driver. Loads the model, runs the IPM
//! (initial iterations with the diagonal KKT solver, then the main IPM with
//! basis preconditioning) and crossover, and postsolves the solution. Also
//! info.cc (StatusString, the Info dump).

use crate::util::fma::ClangFma;

use super::basis::Basis;
use super::control::{Control, Hooks};
use super::crossover::Crossover;
use super::fmt::{fix2, sci2, sci8, textline};
use super::ipm::Ipm;
use super::iterate::{Iterate, State};
use super::kkt::{KktSolverBasis, KktSolverDiag};
use super::lu::LuResult;
use super::model::{dual_infeasibility, primal_infeasibility, Model, UserLp};
use super::starting_basis::starting_basis;
use super::{
    Info, Int, Parameters, BASIC, ERROR_INVALID_VECTOR, ERROR_TIME_INTERRUPT, ERROR_USER_INTERRUPT,
    NONBASIC_LB, NONBASIC_UB, STATUS_DEBUG, STATUS_DUAL_INFEAS, STATUS_FAILED, STATUS_IMPRECISE,
    STATUS_INTERNAL_ERROR, STATUS_ITER_LIMIT, STATUS_NOT_RUN, STATUS_NO_MODEL, STATUS_NO_PROGRESS,
    STATUS_OPTIMAL, STATUS_PRIMAL_INFEAS, STATUS_SOLVED, STATUS_STOPPED, STATUS_TIME_LIMIT,
    STATUS_USER_INTERRUPT, SUPERBASIC,
};
use std::rc::Rc;
use std::time::Instant;

pub struct LpSolver {
    control: Rc<Control>,
    info: Info,
    model: Rc<Model>,
    iterate: Option<Iterate>,
    basis: Option<Basis>,
    // Basic solution computed by crossover and basic status of each
    // variable. If crossover was not run or failed, basic_statuses is
    // empty. If crossover_weights is non-empty at RunCrossover(), it holds
    // the weights that define the order of primal and dual pushes.
    x_crossover: Vec<f64>,
    y_crossover: Vec<f64>,
    z_crossover: Vec<f64>,
    crossover_weights: Vec<f64>,
    basic_statuses: Vec<Int>,
    // IPM starting point provided by user (presolved).
    x_start: Vec<f64>,
    xl_start: Vec<f64>,
    xu_start: Vec<f64>,
    y_start: Vec<f64>,
    zl_start: Vec<f64>,
    zu_start: Vec<f64>,
}

impl Default for LpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LpSolver {
    pub fn new() -> Self {
        LpSolver {
            control: Rc::new(Control::new()),
            info: Info::default(),
            model: Rc::new(Model::default()),
            iterate: None,
            basis: None,
            x_crossover: vec![],
            y_crossover: vec![],
            z_crossover: vec![],
            crossover_weights: vec![],
            basic_statuses: vec![],
            x_start: vec![],
            xl_start: vec![],
            xu_start: vec![],
            y_start: vec![],
            zl_start: vec![],
            zu_start: vec![],
        }
    }

    /// Loads an LP model; returns 0 or IPX_ERROR_*.
    pub fn load_model(&mut self, lp: &UserLp) -> Int {
        self.clear_model();
        let (model, errflag) = Model::load(&self.control, lp);
        self.model = Rc::new(model);
        self.model.get_info(&mut self.info);
        errflag
    }

    /// Loads a primal-dual point (x, xl, xu, slack, y, zl, zu of the user
    /// model) as starting point for the IPM.
    pub fn load_ipm_starting_point(&mut self, user: [Option<&[f64]>; 7]) -> Int {
        let m = self.model.rows();
        let n = self.model.cols();
        self.x_start = vec![0.0; n + m];
        self.xl_start = vec![0.0; n + m];
        self.xu_start = vec![0.0; n + m];
        self.y_start = vec![0.0; m];
        self.zl_start = vec![0.0; n + m];
        self.zu_start = vec![0.0; n + m];
        let errflag = self.model.presolve_ipm_starting_point(
            user,
            &mut self.x_start,
            &mut self.xl_start,
            &mut self.xu_start,
            &mut self.y_start,
            &mut self.zl_start,
            &mut self.zu_start,
        );
        if errflag != 0 {
            self.clear_ipm_starting_point();
            return errflag;
        }
        self.make_ipm_starting_point_valid();
        0
    }

    /// Solves the loaded model; returns info.status.
    pub fn solve(&mut self) -> Int {
        if self.model.empty() {
            self.info.status = STATUS_NO_MODEL;
            return self.info.status;
        }
        self.clear_solution();
        self.control.reset_timer();
        self.control.open_logfile();
        self.control.log("IPX version 1.0\n");
        if let Err(e) = self.solve_body() {
            self.control.log(&format!(" internal error: {e}\n"));
            self.info.status = STATUS_INTERNAL_ERROR;
        }
        self.info.time_total = self.control.elapsed();
        if self.control.debug(2) {
            self.control.debug_out(2, &info_dump(&self.info));
        }
        self.control.close_logfile();
        if self.control.analyse_basis_data() {
            if let Some(basis) = &self.basis {
                basis.report_basis_data();
            }
        }
        self.info.status
    }

    fn solve_body(&mut self) -> LuResult<()> {
        self.interior_point_solve()?;
        let run_crossover_on = self.control.run_crossover() == 1;
        let run_crossover_choose = self.control.run_crossover() == -1;
        let run_crossover_not_off = run_crossover_choose || run_crossover_on;
        let run_crossover = (self.info.status_ipm == STATUS_OPTIMAL && run_crossover_on)
            || (self.info.status_ipm == STATUS_IMPRECISE && run_crossover_not_off);
        if run_crossover {
            if run_crossover_on {
                self.control.log("Running crossover as requested\n");
            } else if run_crossover_choose {
                self.control.log("Running crossover since IPX is imprecise\n");
            }
            self.build_crossover_starting_point();
            self.run_crossover()?;
        }
        if let Some(basis) = &self.basis {
            self.info.ftran_sparse = basis.frac_ftran_sparse();
            self.info.btran_sparse = basis.frac_btran_sparse();
            self.info.time_lu_invert = basis.time_factorize();
            self.info.time_lu_update = basis.time_update();
            self.info.time_ftran = basis.time_ftran();
            self.info.time_btran = basis.time_btran();
            self.info.mean_fill = basis.mean_fill();
            self.info.max_fill = basis.max_fill();
        }
        let info = &mut self.info;
        if info.status_ipm == STATUS_PRIMAL_INFEAS
            || info.status_ipm == STATUS_DUAL_INFEAS
            || info.status_crossover == STATUS_PRIMAL_INFEAS
            || info.status_crossover == STATUS_DUAL_INFEAS
        {
            // When IPM or crossover detect the model to be infeasible, then
            // the problem is solved.
            info.status = STATUS_SOLVED;
        } else {
            let method_status = if run_crossover {
                info.status_crossover
            } else {
                info.status_ipm
            };
            info.status = if method_status == STATUS_OPTIMAL || method_status == STATUS_IMPRECISE {
                STATUS_SOLVED
            } else {
                STATUS_STOPPED
            };
        }
        self.print_summary();
        Ok(())
    }

    pub fn get_info(&self) -> Info {
        self.info
    }

    /// The final IPM iterate postsolved: [x, xl, xu, slack, y, zl, zu];
    /// -1 if no iterate is available.
    pub fn get_interior_solution(&self, out: [Option<&mut [f64]>; 7]) -> Int {
        let Some(it) = &self.iterate else {
            return -1;
        };
        self.model
            .postsolve_interior_solution([it.x(), it.xl(), it.xu(), it.y(), it.zl(), it.zu()], out);
        0
    }

    /// The basic solution and basis from crossover; -1 if not available.
    pub fn get_basic_solution(
        &self,
        x: Option<&mut [f64]>,
        slack: Option<&mut [f64]>,
        y: Option<&mut [f64]>,
        z: Option<&mut [f64]>,
        cbasis: Option<&mut [Int]>,
        vbasis: Option<&mut [Int]>,
    ) -> Int {
        if self.basic_statuses.is_empty() {
            return -1;
        }
        self.model.postsolve_basic_solution(
            &self.x_crossover,
            &self.y_crossover,
            &self.z_crossover,
            &self.basic_statuses,
            x,
            slack,
            y,
            z,
        );
        self.model.postsolve_basis(&self.basic_statuses, cbasis, vbasis);
        0
    }

    pub fn get_parameters(&self) -> Parameters {
        self.control.parameters()
    }

    pub fn set_parameters(&mut self, p: Parameters) {
        self.control.set_parameters(p);
    }

    pub fn set_hooks(&mut self, hooks: Hooks) {
        self.control.set_hooks(hooks);
    }

    pub fn set_timer_offset(&mut self, offset: f64) {
        self.control.set_timer_offset(offset);
    }

    /// True if the HiGHS task running the solver was cancelled (C++ then
    /// rethrows HighsTask::Interrupt).
    pub fn cancelled(&self) -> bool {
        self.control.cancelled()
    }

    /// Discards the model and solution (if any) but keeps the parameters.
    pub fn clear_model(&mut self) {
        self.clear_solution();
        self.model = Rc::new(Model::default());
        self.clear_ipm_starting_point();
    }

    pub fn clear_ipm_starting_point(&mut self) {
        self.x_start.clear();
        self.xl_start.clear();
        self.xu_start.clear();
        self.y_start.clear();
        self.zl_start.clear();
        self.zu_start.clear();
    }

    /// Runs crossover from the given (complementary) starting point; None
    /// components are zero. Returns 0 or IPX_ERROR_invalid_vector.
    pub fn crossover_from_starting_point(
        &mut self,
        x_start: Option<&[f64]>,
        slack_start: Option<&[f64]>,
        y_start: Option<&[f64]>,
        z_start: Option<&[f64]>,
    ) -> Int {
        let model = Rc::clone(&self.model);
        let m = model.rows();
        let n = model.cols();
        let (lb, ub, ai) = (model.lb(), model.ub(), model.ai());

        self.clear_solution();
        self.control.log("Crossover from starting point\n");

        self.x_crossover = vec![0.0; n + m];
        self.y_crossover = vec![0.0; m];
        self.z_crossover = vec![0.0; n + m];
        self.crossover_weights.clear();
        model.presolve_starting_point(
            x_start,
            slack_start,
            y_start,
            z_start,
            &mut self.x_crossover,
            &mut self.y_crossover,
            &mut self.z_crossover,
        );

        // Check that starting point is complementary and satisfies bound
        // and sign conditions.
        let (x, z) = (&self.x_crossover, &self.z_crossover);
        for j in 0..n + m {
            if x[j] < lb[j] || x[j] > ub[j] {
                return ERROR_INVALID_VECTOR;
            }
            if x[j] != lb[j] && z[j] > 0.0 {
                return ERROR_INVALID_VECTOR;
            }
            if x[j] != ub[j] && z[j] < 0.0 {
                return ERROR_INVALID_VECTOR;
            }
        }

        let r: LuResult<()> = (|| {
            // Construct starting basis.
            let mut basis = Basis::new(Rc::clone(&self.control), Rc::clone(&model))?;
            if self.control.crash_basis() != 0 {
                // Take columns in the following order of priority:
                // - free columns
                // - columns between their bounds, in increasing number of
                //   nonzeros
                // - columns with zero dual, in increasing number of nonzeros
                // - Fixed columns and those with nonzero dual
                let timer = Instant::now();
                let mut colweight = vec![0.0; n + m];
                for j in 0..n + m {
                    let nz = ai.col_entries(j) as usize;
                    colweight[j] = if lb[j] == ub[j] {
                        0.0
                    } else if lb[j].is_infinite() && ub[j].is_infinite() {
                        f64::INFINITY
                    } else if self.z_crossover[j] != 0.0 {
                        0.0
                    } else if self.x_crossover[j] != lb[j] && self.x_crossover[j] != ub[j] {
                        (m + (m - nz + 1)) as f64
                    } else {
                        (m - nz + 1) as f64
                    };
                }
                basis.construct_basis_from_weights(&colweight, &mut self.info)?;
                self.info.time_starting_basis += timer.elapsed().as_secs_f64();
                if self.info.errflag != 0 {
                    self.clear_solution();
                    return Ok(());
                }
            }
            self.basis = Some(basis);
            self.run_crossover()
        })();
        if let Err(e) = r {
            // ponytail: the C++ propagates this exception to the caller;
            // report it as an internal error instead
            self.control.log(&format!(" internal error: {e}\n"));
            self.info.status_crossover = STATUS_FAILED;
        }
        0
    }

    /// The current IPM iterate without postsolve; -1 if not available.
    pub fn get_iterate(&self, out: [Option<&mut [f64]>; 6]) -> Int {
        let Some(it) = &self.iterate else {
            return -1;
        };
        let [x, y, zl, zu, xl, xu] = out;
        for (o, v) in [x, y, zl, zu, xl, xu]
            .into_iter()
            .zip([it.x(), it.y(), it.zl(), it.zu(), it.xl(), it.xu()])
        {
            if let Some(o) = o {
                o[..v.len()].copy_from_slice(v);
            }
        }
        0
    }

    /// The current basis postsolved; -1 if no basis is available.
    pub fn get_basis(&self, cbasis: Option<&mut [Int]>, vbasis: Option<&mut [Int]>) -> Int {
        let Some(basis) = &self.basis else {
            return -1;
        };
        if !self.basic_statuses.is_empty() {
            // crossover provides basic statuses
            self.model.postsolve_basis(&self.basic_statuses, cbasis, vbasis);
        } else {
            self.model.postsolve_basis(&build_basic_statuses(basis), cbasis, vbasis);
        }
        0
    }

    /// The constraint matrix of the solver and the diagonal of the (1,1)
    /// block of the KKT matrix; -1 if no iterate is available.
    pub fn get_kkt_matrix(
        &self,
        aip: Option<&mut [Int]>,
        aii: Option<&mut [Int]>,
        aix: Option<&mut [f64]>,
        g: Option<&mut [f64]>,
    ) -> Int {
        let Some(it) = &self.iterate else {
            return -1;
        };
        if let (Some(p), Some(i), Some(x)) = (aip, aii, aix) {
            let ai = self.model.ai();
            let nz = ai.entries() as usize;
            p[..ai.colptr.len()].copy_from_slice(&ai.colptr);
            i[..nz].copy_from_slice(&ai.rowidx[..nz]);
            x[..nz].copy_from_slice(&ai.values[..nz]);
        }
        if let Some(g) = g {
            for j in 0..it.x().len() {
                g[j] = match it.state_of(j) {
                    State::Fixed => f64::INFINITY,
                    State::Free => 0.0,
                    State::Barrier => it.zl()[j] / it.xl()[j] + it.zu()[j] / it.xu()[j],
                };
            }
        }
        0
    }

    /// Row and column counts of the symbolic inverse of the basis; -1 if no
    /// basis is available.
    pub fn symbolic_invert(&self, rowcounts: Option<&mut [Int]>, colcounts: Option<&mut [Int]>) -> Int {
        let Some(basis) = &self.basis else {
            return -1;
        };
        basis.symbolic_invert(rowcounts, colcounts);
        0
    }

    fn clear_solution(&mut self) {
        self.iterate = None;
        self.basis = None;
        self.x_crossover.clear();
        self.y_crossover.clear();
        self.z_crossover.clear();
        self.crossover_weights.clear();
        self.basic_statuses = Vec::new();
        self.info = Info::default();
        // Restore info entries that belong to model.
        self.model.get_info(&mut self.info);
    }

    fn interior_point_solve(&mut self) -> LuResult<()> {
        if self.control.run_centring() != 0 {
            self.control.log("Interior point solve for analytic centre\n");
        } else {
            self.control.log("Interior point solve\n");
        }

        // Allocate new iterate and set tolerances for IPM termination test.
        let mut iterate = Iterate::new(Rc::clone(&self.model));
        iterate.set_feasibility_tol(self.control.ipm_feasibility_tol());
        iterate.set_optimality_tol(self.control.ipm_optimality_tol());
        if self.control.run_crossover() != 0 {
            iterate.set_start_crossover_tol(self.control.start_crossover_tol());
        }
        self.iterate = Some(iterate);

        self.run_ipm()?;

        let iterate = self.iterate.as_mut().unwrap();
        iterate.postprocess();
        iterate.evaluate_postsolved(&mut self.info);

        // Declare status_ipm "imprecise" if the IPM terminated optimal but
        // the solution after postprocessing/postsolve does not satisfy
        // tolerances.
        let info = &mut self.info;
        if info.status_ipm == STATUS_OPTIMAL
            && (info.rel_objgap.abs() > self.control.ipm_optimality_tol()
                || info.rel_presidual > self.control.ipm_feasibility_tol()
                || info.rel_dresidual > self.control.ipm_feasibility_tol())
        {
            info.status_ipm = STATUS_IMPRECISE;
        }

        // Assess the success of analytic centre calculation
        if info.centring_tried != 0 {
            info.status_ipm = if info.centring_success != 0 {
                STATUS_OPTIMAL
            } else {
                STATUS_IMPRECISE
            };
        }
        Ok(())
    }

    fn run_ipm(&mut self) -> LuResult<()> {
        let control = Rc::clone(&self.control);
        let mut ipm = Ipm::new(&control);
        self.info.centring_tried = 0;
        self.info.centring_success = 0;

        if !self.x_start.is_empty() {
            control.log(" Using starting point provided by user. Skipping initial iterations.\n");
            self.iterate.as_mut().unwrap().initialize(
                &self.x_start,
                &self.xl_start,
                &self.xu_start,
                &self.y_start,
                &self.zl_start,
                &self.zu_start,
            );
        } else {
            self.compute_starting_point(&mut ipm)?;
            if self.info.status_ipm != STATUS_NOT_RUN {
                return Ok(());
            }
            self.run_initial_ipm(&mut ipm)?;
            if self.info.status_ipm != STATUS_NOT_RUN {
                return Ok(());
            }
        }
        self.build_starting_basis()?;
        if self.info.status_ipm != STATUS_NOT_RUN || self.info.centring_tried != 0 {
            return Ok(());
        }
        self.run_main_ipm(&mut ipm)
    }

    /// Makes zero complementarity pairs of the user starting point positive.
    fn make_ipm_starting_point_valid(&mut self) {
        let (lb, ub) = (self.model.lb(), self.model.ub());
        let (xl, xu, zl, zu) = (&mut self.xl_start, &mut self.xu_start, &mut self.zl_start, &mut self.zu_start);
        let nm = xl.len();

        let mut num_products = 0;
        let mut sum_products = 0.0f64;
        for j in 0..nm {
            if xl[j] > 0.0 && zl[j] > 0.0 {
                sum_products = xl[j].mul_add_c(zl[j], sum_products);
                num_products += 1;
            }
            if xu[j] > 0.0 && zu[j] > 0.0 {
                sum_products = xu[j].mul_add_c(zu[j], sum_products);
                num_products += 1;
            }
        }
        let mu = if num_products != 0 {
            sum_products / num_products as f64
        } else {
            1.0
        };

        for j in 0..nm {
            if lb[j].is_finite() {
                if xl[j] == 0.0 && zl[j] == 0.0 {
                    xl[j] = mu.sqrt();
                    zl[j] = xl[j];
                } else if xl[j] == 0.0 {
                    xl[j] = mu / zl[j];
                } else if zl[j] == 0.0 {
                    zl[j] = mu / xl[j];
                }
            }
            if ub[j].is_finite() {
                if xu[j] == 0.0 && zu[j] == 0.0 {
                    xu[j] = mu.sqrt();
                    zu[j] = xu[j];
                } else if xu[j] == 0.0 {
                    xu[j] = mu / zu[j];
                } else if zu[j] == 0.0 {
                    zu[j] = mu / xu[j];
                }
            }
        }
    }

    fn compute_starting_point(&mut self, ipm: &mut Ipm) -> LuResult<()> {
        let timer = Instant::now();
        let mut kkt = KktSolverDiag::new(&self.control, &self.model);
        // If the starting point procedure fails, then the iterate remains as
        // initialized by the constructor, which is a valid state for
        // postprocessing/postsolving.
        ipm.starting_point(&mut kkt, self.iterate.as_mut().unwrap(), &mut self.info)?;
        self.info.time_ipm1 += timer.elapsed().as_secs_f64();
        Ok(())
    }

    fn run_initial_ipm(&mut self, ipm: &mut Ipm) -> LuResult<()> {
        let timer = Instant::now();
        let mut kkt = KktSolverDiag::new(&self.control, &self.model);

        let switchiter = self.control.switchiter();
        if switchiter < 0 {
            // Switch iteration not specified by user. Run as long as KKT
            // solver converges within min(500,10+m/20) iterations.
            let m = self.model.rows() as Int;
            kkt.set_maxiter(std::cmp::min(500, 10 + m / 20));
            ipm.set_maxiter(self.control.ipm_maxiter());
        } else {
            ipm.set_maxiter(std::cmp::min(switchiter, self.control.ipm_maxiter()));
        }
        ipm.driver(&mut kkt, self.iterate.as_mut().unwrap(), &mut self.info)?;
        let info = &mut self.info;
        match info.status_ipm {
            // If the IPM reached its termination criterion in the initial
            // iterations (happens rarely), we still call the IPM again with
            // basis preconditioning. A starting basis is then available for
            // crossover.
            STATUS_OPTIMAL | STATUS_NO_PROGRESS => info.status_ipm = STATUS_NOT_RUN,
            STATUS_FAILED => {
                info.status_ipm = STATUS_NOT_RUN;
                info.errflag = 0;
            }
            STATUS_ITER_LIMIT => {
                if info.iter < self.control.ipm_maxiter() {
                    // stopped at switchiter
                    info.status_ipm = STATUS_NOT_RUN;
                }
            }
            _ => {}
        }
        info.time_ipm1 += timer.elapsed().as_secs_f64();
        Ok(())
    }

    fn build_starting_basis(&mut self) -> LuResult<()> {
        if self.control.stop_at_switch() < 0 {
            self.info.status_ipm = STATUS_DEBUG;
            return Ok(());
        }
        self.control.log(" Constructing starting basis...\n");
        self.basis = None;
        let mut basis = Basis::new(Rc::clone(&self.control), Rc::clone(&self.model))?;
        let r = starting_basis(self.iterate.as_mut().unwrap(), &mut basis, &mut self.info);
        self.basis = Some(basis);
        r?;
        let info = &mut self.info;
        if info.errflag == ERROR_USER_INTERRUPT {
            info.errflag = 0;
            info.status_ipm = STATUS_USER_INTERRUPT;
            return Ok(());
        } else if info.errflag == ERROR_TIME_INTERRUPT {
            info.errflag = 0;
            info.status_ipm = STATUS_TIME_LIMIT;
            return Ok(());
        } else if info.errflag != 0 {
            info.status_ipm = STATUS_FAILED;
            return Ok(());
        }
        if self.model.dualized() {
            std::mem::swap(&mut info.dependent_rows, &mut info.dependent_cols);
            std::mem::swap(&mut info.rows_inconsistent, &mut info.cols_inconsistent);
        }
        if self.control.stop_at_switch() > 0 {
            info.status_ipm = STATUS_DEBUG;
            return Ok(());
        }
        if info.rows_inconsistent != 0 {
            info.status_ipm = STATUS_PRIMAL_INFEAS;
            return Ok(());
        }
        if info.cols_inconsistent != 0 {
            info.status_ipm = STATUS_DUAL_INFEAS;
        }
        Ok(())
    }

    fn run_main_ipm(&mut self, ipm: &mut Ipm) -> LuResult<()> {
        let basis = self.basis.as_mut().unwrap();
        let mut kkt = KktSolverBasis::new(&self.control, basis);
        let timer = Instant::now();
        ipm.print_header();
        ipm.set_maxiter(self.control.ipm_maxiter());
        let r = ipm.driver(&mut kkt, self.iterate.as_mut().unwrap(), &mut self.info);
        self.info.time_ipm2 = timer.elapsed().as_secs_f64();
        r
    }

    fn build_crossover_starting_point(&mut self) {
        let m = self.model.rows();
        let n = self.model.cols();
        let iterate = self.iterate.as_ref().unwrap();

        // Construct a complementary primal-dual point from the final IPM
        // iterate. This usually increases the residuals to Ax=b and A'y+z=c.
        self.x_crossover = vec![0.0; n + m];
        self.y_crossover = vec![0.0; m];
        self.z_crossover = vec![0.0; n + m];
        iterate.drop_to_complementarity(&mut self.x_crossover, &mut self.y_crossover, &mut self.z_crossover);

        // Perform dual pushes in increasing order and primal pushes in
        // decreasing order of the scaling factors from the final IPM
        // iterate.
        self.crossover_weights = (0..n + m).map(|j| iterate.scaling_factor(j)).collect();
    }

    fn run_crossover(&mut self) -> LuResult<()> {
        let model = Rc::clone(&self.model);
        let control = Rc::clone(&self.control);
        let m = model.rows();
        let n = model.cols();
        let (lb, ub) = (model.lb(), model.ub());
        self.basic_statuses.clear();
        let basis = self.basis.as_mut().expect("crossover requires a basis");

        let weights = if self.crossover_weights.is_empty() {
            None
        } else {
            Some(&self.crossover_weights[..])
        };

        let mut crossover = Crossover::new(&control);
        crossover.push_all(
            basis,
            &mut self.x_crossover,
            &mut self.y_crossover,
            &mut self.z_crossover,
            weights,
            &mut self.info,
        )?;
        self.info.time_crossover = crossover.time_primal + crossover.time_dual;
        self.info.updates_crossover = crossover.primal_pivots + crossover.dual_pivots;
        if self.info.status_crossover != STATUS_OPTIMAL {
            // Crossover failed. Discard solution.
            self.x_crossover.clear();
            self.y_crossover.clear();
            self.z_crossover.clear();
            return Ok(());
        }

        // Recompute vertex solution and set basic statuses.
        basis.compute_basic_solution(&mut self.x_crossover, &mut self.y_crossover, &mut self.z_crossover)?;
        let (x, z) = (&self.x_crossover, &self.z_crossover);
        self.basic_statuses = (0..n + m)
            .map(|j| {
                if basis.is_basic(j) {
                    BASIC
                } else if lb[j] == ub[j] {
                    if z[j] >= 0.0 {
                        NONBASIC_LB
                    } else {
                        NONBASIC_UB
                    }
                } else if x[j] == lb[j] {
                    NONBASIC_LB
                } else if x[j] == ub[j] {
                    NONBASIC_UB
                } else {
                    SUPERBASIC
                }
            })
            .collect();
        if control.debug(1) {
            control.debug_out(
                1,
                &format!(
                    "{}{}\n{}{}\n",
                    textline("Bound violation of basic solution:"),
                    sci2(primal_infeasibility(&model, x)),
                    textline("Dual sign violation of basic solution:"),
                    sci2(dual_infeasibility(&model, x, z))
                ),
            );
            control.debug_out(
                1,
                &format!(
                    "{}{}\n",
                    textline("Minimum singular value of basis matrix:"),
                    sci2(basis.min_singular_value()?)
                ),
            );
        }

        // Declare crossover status "imprecise" if the vertex solution
        // defined by the final basis does not satisfy tolerances.
        model.evaluate_basic_solution(
            &self.x_crossover,
            &self.y_crossover,
            &self.z_crossover,
            &self.basic_statuses,
            &mut self.info,
        );
        if self.info.primal_infeas > control.pfeasibility_tol() || self.info.dual_infeas > control.dfeasibility_tol() {
            self.info.status_crossover = STATUS_IMPRECISE;
        }
        Ok(())
    }

    fn print_summary(&self) {
        let c = &self.control;
        let info = &self.info;
        let mut s = String::from("Summary\n");
        if !c.timeless_log() {
            s += &format!("{}{}s\n", textline("Runtime:"), fix2(c.elapsed()));
        }
        s += &format!(
            "{}{}\n{}{}\n",
            textline("Status interior point solve:"),
            status_string(info.status_ipm),
            textline("Status crossover:"),
            status_string(info.status_crossover)
        );
        c.log(&s);
        if info.status_ipm == STATUS_OPTIMAL || info.status_ipm == STATUS_IMPRECISE {
            c.log(&format!(
                "{}{}\n{}{} / {}\n{}{} / {}\n{}{} / {}\n",
                textline("objective value:"),
                sci8(info.pobjval),
                textline("interior solution primal residual (abs/rel):"),
                sci2(info.abs_presidual),
                sci2(info.rel_presidual),
                textline("interior solution dual residual (abs/rel):"),
                sci2(info.abs_dresidual),
                sci2(info.rel_dresidual),
                textline("interior solution objective gap (abs/rel):"),
                sci2(info.pobjval - info.dobjval),
                sci2(info.rel_objgap)
            ));
        }
        if info.status_crossover == STATUS_OPTIMAL || info.status_crossover == STATUS_IMPRECISE {
            c.log(&format!(
                "{}{}\n{}{}\n",
                textline("basic solution primal infeasibility:"),
                sci2(info.primal_infeas),
                textline("basic solution dual infeasibility:"),
                sci2(info.dual_infeas)
            ));
        }
    }
}

/// Basic statuses consistent with the basis and the bounds from the model.
fn build_basic_statuses(basis: &Basis) -> Vec<Int> {
    let model = basis.model();
    let (lb, ub) = (model.lb(), model.ub());
    (0..model.rows() + model.cols())
        .map(|j| {
            if basis.is_basic(j) {
                BASIC
            } else if lb[j].is_finite() {
                NONBASIC_LB
            } else if ub[j].is_finite() {
                NONBASIC_UB
            } else {
                SUPERBASIC
            }
        })
        .collect()
}

/// StatusString: a text for each status code
pub fn status_string(status: Int) -> &'static str {
    match status {
        STATUS_NOT_RUN => "not run",
        STATUS_SOLVED => "solved",
        STATUS_STOPPED => "stopped",
        STATUS_NO_MODEL => "no model",
        super::STATUS_OUT_OF_MEMORY => "out of memory",
        STATUS_INTERNAL_ERROR => "internal error",
        STATUS_OPTIMAL => "optimal",
        STATUS_IMPRECISE => "imprecise",
        STATUS_PRIMAL_INFEAS => "primal infeas",
        STATUS_DUAL_INFEAS => "dual infeas",
        STATUS_TIME_LIMIT => "time limit",
        STATUS_ITER_LIMIT => "iter limit",
        STATUS_NO_PROGRESS => "no progress",
        STATUS_FAILED => "failed",
        STATUS_DEBUG => "debug",
        _ => "unknown",
    }
}

/// operator<<(ostream, Info): one line "info.<name>  <value>" per member
fn info_dump(info: &Info) -> String {
    let mut s = String::new();
    let mut dump = |name: &str, v: String| {
        s += &format!("{}{}\n", textline(&format!("info.{name}")), v);
    };
    macro_rules! ints {
        ($($f:ident),*) => { $(dump(stringify!($f), info.$f.to_string());)* };
    }
    macro_rules! with {
        ($fmt:ident: $($f:ident),*) => { $(dump(stringify!($f), $fmt(info.$f));)* };
    }
    ints!(status, status_ipm, status_crossover, errflag, num_var, num_constr, num_entries,
        num_rows_solver, num_cols_solver, num_entries_solver, dualized, dense_cols,
        dependent_rows, dependent_cols, rows_inconsistent, cols_inconsistent,
        primal_dropped, dual_dropped);
    with!(sci2: abs_presidual, abs_dresidual, rel_presidual, rel_dresidual);
    with!(sci8: pobjval, dobjval);
    with!(sci2: rel_objgap, complementarity, normx, normy, normz);
    with!(sci8: objval);
    with!(sci2: primal_infeas, dual_infeas);
    ints!(iter, kktiter1, kktiter2, basis_repairs, updates_start, updates_ipm, updates_crossover);
    with!(fix2: time_total, time_ipm1, time_ipm2, time_starting_basis, time_crossover,
        time_kkt_factorize, time_kkt_solve, time_maxvol, time_cr1);
    dump("time_cr1_AAt", fix2(info.time_cr1_aat));
    dump("time_cr1_pre", fix2(info.time_cr1_pre));
    dump("time_cr2", fix2(info.time_cr2));
    dump("time_cr2_NNt", fix2(info.time_cr2_nnt));
    dump("time_cr2_B", fix2(info.time_cr2_b));
    dump("time_cr2_Bt", fix2(info.time_cr2_bt));
    with!(fix2: ftran_sparse, btran_sparse, time_ftran, time_btran, time_lu_invert,
        time_lu_update, mean_fill, max_fill, time_symb_invert);
    ints!(maxvol_updates, maxvol_skipped, maxvol_passes, tbl_nnz);
    with!(sci2: tbl_max, frobnorm_squared, lambdamax, volume_increase);
    s
}
