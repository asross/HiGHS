//! Iterate (iterate.h/.cc): the IPM iterate (x, xl, xu, y, zl, zu) and its
//! residuals, objectives and complementarity, evaluated lazily.

use super::model::Model;
use super::sparse_matrix::{dot_column, multiply_add};
use super::utils::{dot, infnorm};
use super::{cmax, cmin, Info};
use std::cell::{Ref, RefCell};
use std::rc::Rc;

const INF: f64 = f64::INFINITY;
const BARRIER_MIN: f64 = 1e-30;

/// The state of a variable as seen by the IPM
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    /// treated as non-existent (xl = xu = zl = zu = 0)
    Fixed,
    /// no barrier term (free or implied)
    Free,
    /// with barrier term
    Barrier,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StateDetail {
    BarrierLb,
    BarrierUb,
    BarrierBoxed,
    Free,
    Fixed,
    ImpliedLb,
    ImpliedUb,
    ImpliedEq,
}

/// The lazily evaluated quantities
#[derive(Default)]
pub struct Eval {
    pub rb: Vec<f64>, // b-AI*x
    pub rl: Vec<f64>, // lb-x+xl
    pub ru: Vec<f64>, // ub-x-xu
    pub rc: Vec<f64>, // c-AI'y-zl+zu
    pub pobjective: f64,
    pub dobjective: f64,
    pub presidual: f64,
    pub dresidual: f64,
    pub offset: f64,
    pub complementarity: f64,
    pub mu: f64,
    pub mu_min: f64,
    pub mu_max: f64,
    evaluated: bool,
}

pub struct Iterate {
    model: Rc<Model>,
    x: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    y: Vec<f64>,
    zl: Vec<f64>,
    zu: Vec<f64>,
    variable_state: Vec<StateDetail>,
    ev: RefCell<Eval>,
    postprocessed: bool,
    feasibility_tol: f64,
    optimality_tol: f64,
    start_crossover_tol: f64,
    pub bounds_measure: f64,
    pub costs_measure: f64,
}

impl Iterate {
    /// Initial iterate: xl, xu, zl, zu one (or INF/zero if no bound).
    pub fn new(model: Rc<Model>) -> Self {
        let m = model.rows();
        let n = model.cols();
        let mut it = Iterate {
            x: vec![0.0; n + m],
            xl: vec![0.0; n + m],
            xu: vec![0.0; n + m],
            y: vec![0.0; m],
            zl: vec![0.0; n + m],
            zu: vec![0.0; n + m],
            variable_state: vec![StateDetail::Free; n + m],
            ev: RefCell::new(Eval {
                rb: vec![0.0; m],
                rl: vec![0.0; n + m],
                ru: vec![0.0; n + m],
                rc: vec![0.0; n + m],
                ..Default::default()
            }),
            postprocessed: false,
            feasibility_tol: 1e-6,
            optimality_tol: 1e-8,
            start_crossover_tol: -1.0,
            bounds_measure: 1.0 + model.norm_bounds(),
            costs_measure: 1.0 + model.norm_c(),
            model,
        };
        let (lb, ub) = (it.model.lb(), it.model.ub());
        for j in 0..n + m {
            let (state, xl, xu, zl, zu) = if lb[j].is_finite() && ub[j].is_finite() {
                (StateDetail::BarrierBoxed, 1.0, 1.0, 1.0, 1.0)
            } else if lb[j].is_finite() {
                (StateDetail::BarrierLb, 1.0, INF, 1.0, 0.0)
            } else if ub[j].is_finite() {
                (StateDetail::BarrierUb, INF, 1.0, 0.0, 1.0)
            } else {
                (StateDetail::Free, INF, INF, 0.0, 0.0)
            };
            it.variable_state[j] = state;
            it.xl[j] = xl;
            it.xu[j] = xu;
            it.zl[j] = zl;
            it.zu[j] = zu;
        }
        it
    }

    /// Sets the iterate; variables become barrier or free according to
    /// their bounds.
    pub fn initialize(&mut self, x: &[f64], xl: &[f64], xu: &[f64], y: &[f64], zl: &[f64], zu: &[f64]) {
        self.x.copy_from_slice(x);
        self.xl.copy_from_slice(xl);
        self.xu.copy_from_slice(xu);
        self.y.copy_from_slice(y);
        self.zl.copy_from_slice(zl);
        self.zu.copy_from_slice(zu);
        let (lb, ub) = (self.model.lb(), self.model.ub());
        for j in 0..self.x.len() {
            self.variable_state[j] = if lb[j] == ub[j] {
                StateDetail::BarrierBoxed
            } else if lb[j].is_finite() && ub[j].is_finite() {
                StateDetail::BarrierBoxed
            } else if lb[j].is_finite() {
                StateDetail::BarrierLb
            } else if ub[j].is_finite() {
                StateDetail::BarrierUb
            } else {
                StateDetail::Free
            };
        }
        self.ev.get_mut().evaluated = false;
        self.postprocessed = false;
    }

    /// Updates the iterate by step sizes sp (primal) and sd (dual); the
    /// barrier variables are kept >= kBarrierMin.
    pub fn update(
        &mut self,
        sp: f64,
        dx: Option<&[f64]>,
        dxl: Option<&[f64]>,
        dxu: Option<&[f64]>,
        sd: f64,
        dy: Option<&[f64]>,
        dzl: Option<&[f64]>,
        dzu: Option<&[f64]>,
    ) {
        let nm = self.x.len();
        if let Some(dx) = dx {
            for j in 0..nm {
                if self.state_of(j) != State::Fixed {
                    self.x[j] = sp.mul_add(dx[j], self.x[j]);
                }
            }
        }
        if let Some(dxl) = dxl {
            for j in 0..nm {
                if self.has_barrier_lb(j) {
                    self.xl[j] = sp.mul_add(dxl[j], self.xl[j]);
                    self.xl[j] = cmax(self.xl[j], BARRIER_MIN);
                }
            }
        }
        if let Some(dxu) = dxu {
            for j in 0..nm {
                if self.has_barrier_ub(j) {
                    self.xu[j] = sp.mul_add(dxu[j], self.xu[j]);
                    self.xu[j] = cmax(self.xu[j], BARRIER_MIN);
                }
            }
        }
        if let Some(dy) = dy {
            for i in 0..self.y.len() {
                self.y[i] = sd.mul_add(dy[i], self.y[i]);
            }
        }
        if let Some(dzl) = dzl {
            for j in 0..nm {
                if self.has_barrier_lb(j) {
                    self.zl[j] = sd.mul_add(dzl[j], self.zl[j]);
                    self.zl[j] = cmax(self.zl[j], BARRIER_MIN);
                }
            }
        }
        if let Some(dzu) = dzu {
            for j in 0..nm {
                if self.has_barrier_ub(j) {
                    self.zu[j] = sd.mul_add(dzu[j], self.zu[j]);
                    self.zu[j] = cmax(self.zu[j], BARRIER_MIN);
                }
            }
        }
        self.ev.get_mut().evaluated = false;
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn x(&self) -> &[f64] {
        &self.x
    }
    pub fn xl(&self) -> &[f64] {
        &self.xl
    }
    pub fn xu(&self) -> &[f64] {
        &self.xu
    }
    pub fn y(&self) -> &[f64] {
        &self.y
    }
    pub fn zl(&self) -> &[f64] {
        &self.zl
    }
    pub fn zu(&self) -> &[f64] {
        &self.zu
    }

    /// The evaluated residuals, objectives and complementarity
    pub fn eval(&self) -> Ref<'_, Eval> {
        if !self.ev.borrow().evaluated {
            let mut ev = self.ev.borrow_mut();
            self.compute_residuals(&mut ev);
            self.compute_objectives(&mut ev);
            self.compute_complementarity(&mut ev);
            ev.evaluated = true;
        }
        self.ev.borrow()
    }

    #[inline]
    pub fn state_of(&self, j: usize) -> State {
        match self.variable_state[j] {
            StateDetail::Fixed => State::Fixed,
            StateDetail::Free
            | StateDetail::ImpliedLb
            | StateDetail::ImpliedUb
            | StateDetail::ImpliedEq => State::Free,
            _ => State::Barrier,
        }
    }

    #[inline]
    pub fn has_barrier_lb(&self, j: usize) -> bool {
        matches!(
            self.variable_state[j],
            StateDetail::BarrierLb | StateDetail::BarrierBoxed
        )
    }

    #[inline]
    pub fn has_barrier_ub(&self, j: usize) -> bool {
        matches!(
            self.variable_state[j],
            StateDetail::BarrierUb | StateDetail::BarrierBoxed
        )
    }

    pub fn is_implied(&self, j: usize) -> bool {
        matches!(
            self.variable_state[j],
            StateDetail::ImpliedLb | StateDetail::ImpliedUb | StateDetail::ImpliedEq
        )
    }

    /// Fixes variable j at its current value (removes it from the IPM).
    pub fn make_fixed(&mut self, j: usize) {
        self.xl[j] = 0.0;
        self.xu[j] = 0.0;
        self.zl[j] = 0.0;
        self.zu[j] = 0.0;
        self.variable_state[j] = StateDetail::Fixed;
        self.ev.get_mut().evaluated = false;
    }

    pub fn make_fixed_at(&mut self, j: usize, value: f64) {
        self.x[j] = value;
        self.make_fixed(j);
    }

    /// Variable j becomes "implied" at its lower bound: free in the IPM,
    /// with x[j] set to lb[j] in postprocessing.
    pub fn make_implied_lb(&mut self, j: usize) {
        self.xl[j] = INF;
        self.xu[j] = INF;
        self.variable_state[j] = StateDetail::ImpliedLb;
        self.ev.get_mut().evaluated = false;
    }

    pub fn make_implied_ub(&mut self, j: usize) {
        self.xl[j] = INF;
        self.xu[j] = INF;
        self.variable_state[j] = StateDetail::ImpliedUb;
        self.ev.get_mut().evaluated = false;
    }

    pub fn make_implied_eq(&mut self, j: usize) {
        self.xl[j] = INF;
        self.xu[j] = INF;
        self.zl[j] = 0.0;
        self.zu[j] = 0.0;
        self.variable_state[j] = StateDetail::ImpliedEq;
        self.ev.get_mut().evaluated = false;
    }

    /// Scaling factor of column j: 0 if fixed, INF if free, else
    /// 1/sqrt(zl/xl + zu/xu).
    pub fn scaling_factor(&self, j: usize) -> f64 {
        match self.state_of(j) {
            State::Fixed => 0.0,
            State::Free => INF,
            State::Barrier => {
                let g = self.zl[j] / self.xl[j] + self.zu[j] / self.xu[j];
                1.0 / g.sqrt()
            }
        }
    }

    pub fn pobjective(&self) -> f64 {
        self.eval().pobjective
    }
    pub fn dobjective(&self) -> f64 {
        self.eval().dobjective
    }
    pub fn pobjective_after_postproc(&self) -> f64 {
        let ev = self.eval();
        ev.pobjective + ev.offset
    }
    pub fn dobjective_after_postproc(&self) -> f64 {
        let ev = self.eval();
        ev.dobjective + ev.offset
    }
    pub fn presidual(&self) -> f64 {
        self.eval().presidual
    }
    pub fn dresidual(&self) -> f64 {
        self.eval().dresidual
    }
    pub fn complementarity(&self) -> f64 {
        self.eval().complementarity
    }
    pub fn mu(&self) -> f64 {
        self.eval().mu
    }

    pub fn feasible(&self) -> bool {
        let ev = self.eval();
        let primal_feasible = ev.presidual <= self.feasibility_tol * self.bounds_measure;
        let dual_feasible = ev.dresidual <= self.feasibility_tol * self.costs_measure;
        primal_feasible && dual_feasible
    }

    pub fn optimal(&self) -> bool {
        let pobj = self.pobjective_after_postproc();
        let dobj = self.dobjective_after_postproc();
        let ave_obj = 0.5 * (pobj + dobj);
        let gap = pobj - dobj;
        let abs_gap = gap.abs();
        let obj_measure = 1.0 + ave_obj.abs();
        abs_gap <= self.optimality_tol * obj_measure
    }

    /// IPM termination test: feasible, optimal and (if a crossover start
    /// tolerance is set) small residuals after dropping to complementarity.
    pub fn term_crit_reached(&self) -> bool {
        if self.feasible() && self.optimal() {
            if self.start_crossover_tol <= 0.0 {
                return true;
            }
            let (pres, dres) = self.residuals_from_dropping();
            if pres <= self.start_crossover_tol * (1.0 + self.model.norm_bounds())
                && dres <= self.start_crossover_tol * (1.0 + self.model.norm_c())
            {
                return true;
            }
        }
        false
    }

    pub fn set_feasibility_tol(&mut self, tol: f64) {
        self.feasibility_tol = tol;
    }
    pub fn set_optimality_tol(&mut self, tol: f64) {
        self.optimality_tol = tol;
    }
    pub fn set_start_crossover_tol(&mut self, tol: f64) {
        self.start_crossover_tol = tol;
    }

    /// Computes xl, xu (and zl, zu) of fixed and implied variables so that
    /// the iterate is a solution of the original model.
    pub fn postprocess(&mut self) {
        let model = Rc::clone(&self.model);
        let (c, lb, ub, ai) = (model.c(), model.lb(), model.ub(), model.ai());
        let nm = self.x.len();

        // For fixed variables compute xl[j] and xu[j] from x[j]. If the
        // lower and upper bound are equal, set zl[j] or zu[j] such that the
        // variable is dual feasible. Otherwise leave them zero.
        for j in 0..nm {
            if self.state_of(j) == State::Fixed {
                self.xl[j] = self.x[j] - lb[j];
                self.xu[j] = ub[j] - self.x[j];
                if lb[j] == ub[j] {
                    let z = c[j] - dot_column(ai, j, &self.y);
                    if z >= 0.0 {
                        self.zl[j] = z;
                    } else {
                        self.zu[j] = -z;
                    }
                }
            }
        }
        // For implied variables set x[j] to the bound at which it was implied
        // and compute zl[j] or zu[j]. If the variable was implied at both
        // bounds, choose between zl and zu depending on sign.
        for j in 0..nm {
            if self.is_implied(j) {
                let z = c[j] - dot_column(ai, j, &self.y);
                match self.variable_state[j] {
                    StateDetail::ImpliedEq => {
                        if z >= 0.0 {
                            self.zl[j] = z;
                            self.zu[j] = 0.0;
                        } else {
                            self.zl[j] = 0.0;
                            self.zu[j] = -z;
                        }
                        self.x[j] = lb[j];
                    }
                    StateDetail::ImpliedLb => {
                        self.zl[j] = z;
                        self.zu[j] = 0.0;
                        self.x[j] = lb[j];
                    }
                    _ => {
                        self.zl[j] = 0.0;
                        self.zu[j] = -z;
                        self.x[j] = ub[j];
                    }
                }
                self.xl[j] = self.x[j] - lb[j];
                self.xu[j] = ub[j] - self.x[j];
            }
        }
        self.postprocessed = true;
        self.ev.get_mut().evaluated = false;
    }

    /// Evaluates the postsolved interior solution into info.
    pub fn evaluate_postsolved(&self, info: &mut Info) {
        self.model.evaluate_interior_solution(
            [&self.x, &self.xl, &self.xu, &self.y, &self.zl, &self.zu],
            info,
        );
    }

    /// Constructs a complementary primal-dual point (x, y, z) from the
    /// postprocessed iterate.
    pub fn drop_to_complementarity(&self, x: &mut [f64], y: &mut [f64], z: &mut [f64]) {
        let (lb, ub) = (self.model.lb(), self.model.ub());
        y.copy_from_slice(&self.y);
        for j in 0..self.x.len() {
            let xlj = self.xl[j];
            let xuj = self.xu[j];
            let zlj = self.zl[j];
            let zuj = self.zu[j];
            let mut xj = self.x[j];
            xj = cmax(xj, lb[j]);
            xj = cmin(xj, ub[j]);

            if lb[j] == ub[j] {
                // fixed variable
                x[j] = lb[j];
                z[j] = zlj - zuj;
            } else if lb[j].is_finite() && ub[j].is_finite() {
                // boxed variable
                if zlj * xuj >= zuj * xlj {
                    // either active at lower bound or inactive
                    if zlj >= xlj {
                        x[j] = lb[j];
                        z[j] = cmax(0.0, zlj - zuj);
                    } else {
                        x[j] = xj;
                        z[j] = 0.0;
                    }
                } else {
                    // either active at upper bound or inactive
                    if zuj >= xuj {
                        x[j] = ub[j];
                        z[j] = cmin(0.0, zlj - zuj);
                    } else {
                        x[j] = xj;
                        z[j] = 0.0;
                    }
                }
            } else if lb[j].is_finite() {
                // lower bound only
                if zlj >= xlj {
                    x[j] = lb[j];
                    z[j] = cmax(0.0, zlj - zuj);
                } else {
                    x[j] = xj;
                    z[j] = 0.0;
                }
            } else if ub[j].is_finite() {
                // upper bound only
                if zuj >= xuj {
                    x[j] = ub[j];
                    z[j] = cmin(0.0, zlj - zuj);
                } else {
                    x[j] = xj;
                    z[j] = 0.0;
                }
            } else {
                // free variable
                x[j] = xj;
                z[j] = 0.0;
            }
        }
    }

    /// The maximum primal and dual residual caused by dropping to
    /// complementarity (as in DropToComplementarity, before
    /// postprocessing).
    fn residuals_from_dropping(&self) -> (f64, f64) {
        let ai = self.model.ai();
        let (lb, ub) = (self.model.lb(), self.model.ub());
        let mut presmax = 0.0;
        let mut dresmax = 0.0;
        for j in 0..self.x.len() {
            let mut xdrop = 0.0; // xnew = xold - xdrop
            let mut zdrop = 0.0;
            let (x, xl, xu, zl, zu) = (self.x[j], self.xl[j], self.xu[j], self.zl[j], self.zu[j]);
            match self.variable_state[j] {
                StateDetail::BarrierLb => {
                    if zl >= xl {
                        xdrop = x - lb[j]; // active at lower bound
                    } else {
                        zdrop = zl - zu; // inactive
                    }
                }
                StateDetail::BarrierUb => {
                    if zu >= xu {
                        xdrop = x - ub[j]; // active at upper bound
                    } else {
                        zdrop = zl - zu; // inactive
                    }
                }
                StateDetail::BarrierBoxed => {
                    if zl / xl >= zu / xu {
                        if zl >= xl {
                            xdrop = x - lb[j];
                        } else {
                            zdrop = zl - zu;
                        }
                    } else if zu >= xu {
                        xdrop = x - ub[j];
                    } else {
                        zdrop = zl - zu;
                    }
                }
                _ => {}
            }
            let mut amax = 0.0;
            for p in ai.begin(j)..ai.begin(j + 1) {
                amax = cmax(amax, ai.value(p).abs());
            }
            presmax = cmax(presmax, xdrop.abs() * amax);
            dresmax = cmax(dresmax, zdrop.abs());
        }
        (presmax, dresmax)
    }

    fn compute_residuals(&self, ev: &mut Eval) {
        let model = &self.model;
        let (lb, ub, ai) = (model.lb(), model.ub(), model.ai());
        let nm = self.x.len();

        // Primal residual: rb = b-AI*x.
        ev.rb.copy_from_slice(model.b());
        multiply_add(ai, &self.x, -1.0, &mut ev.rb, b'N');

        // Dual residual: rc = c-AI'y-zl+zu. If the iterate has not been
        // postprocessed, then the dual residual for fixed variables is zero
        // because these variables are treated as non-existent by the IPM.
        let c = model.c();
        for j in 0..nm {
            ev.rc[j] = c[j] - self.zl[j] + self.zu[j];
        }
        multiply_add(ai, &self.y, -1.0, &mut ev.rc, b'T');
        if !self.postprocessed {
            for j in 0..nm {
                if self.state_of(j) == State::Fixed {
                    ev.rc[j] = 0.0;
                }
            }
        }

        // Bound residuals: rl = lb-x+xl and ru = ub-x-xu, zero if the
        // variable has no barrier term for the bound.
        for j in 0..nm {
            ev.rl[j] = if self.has_barrier_lb(j) {
                lb[j] - self.x[j] + self.xl[j]
            } else {
                0.0
            };
            ev.ru[j] = if self.has_barrier_ub(j) {
                ub[j] - self.x[j] - self.xu[j]
            } else {
                0.0
            };
        }

        ev.presidual = infnorm(&ev.rb);
        ev.dresidual = infnorm(&ev.rc);
        ev.presidual = cmax(ev.presidual, infnorm(&ev.rl));
        ev.presidual = cmax(ev.presidual, infnorm(&ev.ru));
    }

    fn compute_objectives(&self, ev: &mut Eval) {
        let model = &self.model;
        let (b, c, lb, ub, ai) = (model.b(), model.c(), model.lb(), model.ub(), model.ai());
        let nm = self.x.len();
        let (x, y, zl, zu) = (&self.x, &self.y, &self.zl, &self.zu);

        if self.postprocessed {
            // Compute objective values as defined for the LP model.
            ev.offset = 0.0;
            ev.pobjective = model.offset() + dot(c, x);
            let mut d = model.offset() + dot(b, y);
            for j in 0..nm {
                if lb[j].is_finite() {
                    d = lb[j].mul_add(zl[j], d);
                }
                if ub[j].is_finite() {
                    d = (-ub[j]).mul_add(zu[j], d);
                }
            }
            ev.dobjective = d;
        } else {
            // Compute objective values for the LP that is solved at the very
            // moment (after fixing and implying variables). The offset is
            // such that pobjective + offset is the primal objective after
            // postprocessing.
            let mut offset = 0.0f64;
            let mut p = model.offset();
            for j in 0..nm {
                if self.state_of(j) != State::Fixed {
                    p = c[j].mul_add(x[j], p);
                } else {
                    offset = c[j].mul_add(x[j], offset);
                }
                if self.is_implied(j) {
                    // At the moment, we are solving an LP with the cost
                    // coefficient for variable j decreased by zl[j]-zu[j].
                    p = (-(zl[j] - zu[j])).mul_add(x[j], p);
                    offset = (zl[j] - zu[j]).mul_add(x[j], offset);
                }
            }
            let mut d = model.offset() + dot(b, y);
            for j in 0..nm {
                if self.has_barrier_lb(j) {
                    d = lb[j].mul_add(zl[j], d);
                }
                if self.has_barrier_ub(j) {
                    d = (-ub[j]).mul_add(zu[j], d);
                }
                if self.state_of(j) == State::Fixed {
                    // At the moment, we are solving the LP without variable
                    // j, but with the RHS decreased by AI[:,j]*x[j].
                    d = (-x[j]).mul_add(dot_column(ai, j, y), d);
                }
            }
            ev.offset = offset;
            ev.pobjective = p;
            ev.dobjective = d;
        }
    }

    fn compute_complementarity(&self, ev: &mut Eval) {
        let nm = self.x.len();
        let mut comp = 0.0f64;
        let mut mu_min = INF;
        let mut mu_max = 0.0;
        let mut num_finite = 0;
        for j in 0..nm {
            if self.has_barrier_lb(j) {
                comp = self.xl[j].mul_add(self.zl[j], comp);
                mu_min = cmin(mu_min, self.xl[j] * self.zl[j]);
                mu_max = cmax(mu_max, self.xl[j] * self.zl[j]);
                num_finite += 1;
            }
        }
        for j in 0..nm {
            if self.has_barrier_ub(j) {
                comp = self.xu[j].mul_add(self.zu[j], comp);
                mu_min = cmin(mu_min, self.xu[j] * self.zu[j]);
                mu_max = cmax(mu_max, self.xu[j] * self.zu[j]);
                num_finite += 1;
            }
        }
        ev.complementarity = comp;
        if num_finite > 0 {
            ev.mu = comp / num_finite as f64;
            ev.mu_min = mu_min;
        } else {
            ev.mu = 0.0;
            ev.mu_min = 0.0;
        }
        ev.mu_max = mu_max;
    }
}
