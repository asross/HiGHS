//! The PDHG method of cupdlp_solver.c, cupdlp_step.c, cupdlp_restart.c and
//! cupdlp_proj.c (CPU paths): residuals, termination and infeasibility
//! checks, the adaptive (or constant) step, averaging, restarts, and the
//! hot start / unscaling around the solve (PDHG_PreSolve, PDHG_PostSolve).
//!
//! The C code keeps the timers of time(NULL): whole seconds, checked
//! against the time limit; nothing else depends on the clock. Products are
//! fused where clang contracts the C expression (`a*b + c` with the left
//! operand preferred), see each `mul_add`.

use crate::util::fma::ClangFma;

use super::linalg::*;
use super::scaling::Scaling;
use super::{e, g, plus, Log, Params};
use crate::ipx::fmt::fixed;
use crate::ipx::utils::dot_fused;
use std::time::{SystemTime, UNIX_EPOCH};

/// termination_code of cupdlp_defs.h
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TermCode {
    Optimal = 0,
    Infeasible,
    Unbounded,
    InfeasibleOrUnbounded,
    TimeOrIterLimit,
    Feasible,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TermIterate {
    Last,
    Average,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Restart {
    No,
    ToCurrent,
    ToAverage,
}

/// The scaled LP in cuPDLP form: min c'x, A x = b (first neqs rows),
/// A x >= b (the others), l <= x <= u
pub(super) struct Problem {
    pub(super) nrows: usize,
    pub(super) ncols: usize,
    pub(super) neqs: usize,
    pub(super) csc: Sparse,
    pub(super) csr: Sparse,
    pub(super) mat_inf_norm: f64,
    pub(super) cost: Vec<f64>,
    pub(super) rhs: Vec<f64>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) has_lower: Vec<f64>,
    pub(super) has_upper: Vec<f64>,
    pub(super) offset: f64,
    pub(super) sense: f64,
}

/// getTimeStamp(): time(NULL)
pub(super) fn time_stamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs() as f64)
}

/// CUPDLPtimers
#[derive(Default)]
pub(super) struct Timers {
    pub(super) scaling_time: f64,
    n_iter: i32,
    solving_time: f64,
    solving_beg: f64,
    ax_time: f64,
    aty_time: f64,
    update_iterate_time: f64,
    n_ax: i32,
    n_aty: i32,
    n_update_iterate: i32,
}

impl Timers {
    /// Ax()
    fn ax(&mut self, p: &Problem, x: &[f64], ax: &mut [f64]) {
        let begin = time_stamp();
        p.csc.mul(x, ax);
        self.ax_time += time_stamp() - begin;
        self.n_ax += 1;
    }

    /// ATy()
    fn aty(&mut self, p: &Problem, y: &[f64], aty: &mut [f64]) {
        let begin = time_stamp();
        p.csr.mul(y, aty);
        self.aty_time += time_stamp() - begin;
        self.n_aty += 1;
    }
}

/// The scalars of CUPDLPresobj
struct Res {
    feas_tol: f64,
    primal_obj: f64,
    dual_obj: f64,
    gap: f64,
    primal_feas: f64,
    dual_feas: f64,
    rel_gap: f64,
    primal_obj_avg: f64,
    dual_obj_avg: f64,
    gap_avg: f64,
    primal_feas_avg: f64,
    dual_feas_avg: f64,
    rel_gap_avg: f64,
    primal_feas_last_restart: f64,
    dual_feas_last_restart: f64,
    gap_last_restart: f64,
    primal_feas_last_cand: f64,
    dual_feas_last_cand: f64,
    gap_last_cand: f64,
    primal_code: TermCode,
    dual_code: TermCode,
    term_infeas_iterate: TermIterate,
    primal_infeas_obj: f64,
    dual_infeas_obj: f64,
    primal_infeas_res: f64,
    dual_infeas_res: f64,
    primal_infeas_obj_avg: f64,
    dual_infeas_obj_avg: f64,
    primal_infeas_res_avg: f64,
    dual_infeas_res_avg: f64,
    term_code: TermCode,
    term_iterate: TermIterate,
}

impl Res {
    /// resobj_Alloc
    fn new() -> Self {
        Res {
            feas_tol: 1e-8,
            primal_obj: 0.0,
            dual_obj: 0.0,
            gap: 0.0,
            primal_feas: 0.0,
            dual_feas: 0.0,
            rel_gap: 0.0,
            primal_obj_avg: 0.0,
            dual_obj_avg: 0.0,
            gap_avg: 0.0,
            primal_feas_avg: 0.0,
            dual_feas_avg: 0.0,
            rel_gap_avg: 0.0,
            primal_feas_last_restart: 0.0,
            dual_feas_last_restart: 0.0,
            gap_last_restart: 0.0,
            primal_feas_last_cand: 0.0,
            dual_feas_last_cand: 0.0,
            gap_last_cand: 0.0,
            primal_code: TermCode::Feasible,
            dual_code: TermCode::Feasible,
            term_infeas_iterate: TermIterate::Last,
            primal_infeas_obj: 0.0,
            dual_infeas_obj: 0.0,
            primal_infeas_res: 1.0,
            dual_infeas_res: 1.0,
            primal_infeas_obj_avg: 0.0,
            dual_infeas_obj_avg: 0.0,
            primal_infeas_res_avg: 1.0,
            dual_infeas_res_avg: 1.0,
            term_code: TermCode::TimeOrIterLimit,
            term_iterate: TermIterate::Last,
        }
    }
}

/// The vectors of CUPDLPresobj, and scratch for the infeasibility checks
struct ResVecs {
    primal_residual: Vec<f64>,
    dual_residual: Vec<f64>,
    primal_residual_avg: Vec<f64>,
    dual_residual_avg: Vec<f64>,
    slack_pos: Vec<f64>,
    slack_neg: Vec<f64>,
    slack_pos_avg: Vec<f64>,
    slack_neg_avg: Vec<f64>,
    lower_filtered: Vec<f64>,
    upper_filtered: Vec<f64>,
    col1: Vec<f64>,
    col2: Vec<f64>,
    row1: Vec<f64>,
}

/// CUPDLPiterates: iterate k lives in x[k % 2], its successor in the other
struct Iterates {
    x: [Vec<f64>; 2],
    y: [Vec<f64>; 2],
    ax: [Vec<f64>; 2],
    aty: [Vec<f64>; 2],
    x_avg: Vec<f64>,
    y_avg: Vec<f64>,
    ax_avg: Vec<f64>,
    aty_avg: Vec<f64>,
    x_sum: Vec<f64>,
    y_sum: Vec<f64>,
    x_last_restart: Vec<f64>,
    y_last_restart: Vec<f64>,
    last_restart_iter: i32,
}

/// CUPDLPstepsize
struct Step {
    adaptive: bool,
    primal: f64,
    dual: f64,
    sum_primal: f64,
    sum_dual: f64,
    beta: f64,
    n_step_iter: i32,
}

/// (current, next) of a pair of iterates
fn cur_next(a: &mut [Vec<f64>; 2], k: usize) -> (&mut Vec<f64>, &mut Vec<f64>) {
    let [a0, a1] = a;
    if k == 0 {
        (a0, a1)
    } else {
        (a1, a0)
    }
}

/// PDHG_Compute_Primal_Feasibility: (||scaled (Ax - b)^-||, objective)
fn primal_feasibility(
    p: &Problem,
    rs: Option<&[f64]>,
    ax: &[f64],
    x: &[f64],
    r: &mut [f64],
) -> (f64, f64) {
    let obj = dot(x, &p.cost).mul_add_c(p.sense, p.offset);
    for ((ri, &a), &b) in r.iter_mut().zip(ax).zip(&p.rhs) {
        *ri = a - b;
    }
    proj_neg(&mut r[p.neqs..]);
    if let Some(rs) = rs {
        edot(r, rs);
    }
    (nrm2(r), obj)
}

/// PDHG_Compute_Dual_Feasibility: (dual residual norm, dual objective);
/// the reduced costs split into the slacks sp (at lower) and sn (at upper)
#[allow(clippy::too_many_arguments)]
fn dual_feasibility(
    p: &Problem,
    cs: Option<&[f64]>,
    lower_filtered: &[f64],
    upper_filtered: &[f64],
    aty: &[f64],
    y: &[f64],
    r: &mut [f64],
    sp: &mut [f64],
    sn: &mut [f64],
) -> (f64, f64) {
    let mut obj = dot(y, &p.rhs);
    for ((ri, &a), &c) in r.iter_mut().zip(aty).zip(&p.cost) {
        *ri = -a + c;
    }
    sp.copy_from_slice(r);
    proj_pos(sp);
    edot(sp, &p.has_lower);
    obj += dot(sp, lower_filtered);
    sn.copy_from_slice(r);
    proj_neg(sn);
    scale(-1.0, sn);
    edot(sn, &p.has_upper);
    obj -= dot(sn, upper_filtered);
    let obj = obj.mul_add_c(p.sense, p.offset);
    for ((ri, &a), &b) in r.iter_mut().zip(sp.iter()).zip(sn.iter()) {
        *ri = *ri - a + b;
    }
    if let Some(cs) = cs {
        edot(r, cs);
    }
    (nrm2(r), obj)
}

/// PDHG_Compute_Primal_Infeasibility: (objective, residual) of the dual
/// ray (y, sp, sn) normalized
#[allow(clippy::too_many_arguments)]
fn primal_infeasibility(
    p: &Problem,
    cs: Option<&[f64]>,
    y: &[f64],
    sp: &[f64],
    sn: &[f64],
    aty: &[f64],
    dual_obj: f64,
    constr: &mut [f64],
) -> (f64, f64) {
    let mut d_scale = (dot(y, y) + dot(sp, sp) + dot(sn, sn)).sqrt();
    if d_scale < 1e-8 {
        d_scale = 1.0;
    }
    let w = 1.0 / d_scale;
    let obj = (dual_obj - p.offset) / p.sense / d_scale;
    for (((c, &a), &l), &u) in constr.iter_mut().zip(aty).zip(sp).zip(sn) {
        *c = a * w + l * w - u * w;
    }
    if let Some(cs) = cs {
        edot(constr, cs);
    }
    (obj, nrm2(constr))
}

/// PDHG_Compute_Dual_Infeasibility: (objective, residual) of the primal
/// ray x normalized
#[allow(clippy::too_many_arguments)]
fn dual_infeasibility(
    p: &Problem,
    rs: Option<&[f64]>,
    cs: Option<&[f64]>,
    x: &[f64],
    ax: &[f64],
    primal_obj: f64,
    ray: &mut [f64],
    bound: &mut [f64],
    constr: &mut [f64],
) -> (f64, f64) {
    let mut p_scale = nrm2(x);
    if p_scale < 1e-8 {
        p_scale = 1.0;
    }
    let w = 1.0 / p_scale;
    for (r, &xi) in ray.iter_mut().zip(x) {
        *r = xi * w;
    }
    let obj = (primal_obj - p.offset) / p.sense / p_scale;
    for (c, &a) in constr.iter_mut().zip(ax) {
        *c = a * w;
    }
    proj_neg(&mut constr[p.neqs..]);
    if let Some(rs) = rs {
        edot(constr, rs);
    }
    let constr_sq = dot(constr, constr);
    let mut bound_sq = |proj: fn(&mut [f64]), has: &[f64]| {
        bound.copy_from_slice(ray);
        proj(bound);
        edot(bound, has);
        if let Some(cs) = cs {
            ediv(bound, cs);
        }
        dot(bound, bound)
    };
    let lb_sq = bound_sq(proj_neg, &p.has_lower);
    let ub_sq = bound_sq(proj_pos, &p.has_upper);
    (obj, (constr_sq + lb_sq + ub_sq).sqrt())
}

/// PDHG_Restart_Score_GPU (contracted twice)
fn restart_score(w2: f64, primal_feas: f64, dual_feas: f64, gap: f64) -> f64 {
    gap.mul_add_c(
        gap,
        (w2 * primal_feas).mul_add_c(primal_feas, dual_feas * dual_feas / w2),
    )
    .sqrt()
}

/// PDHG_primalGradientStep: xu = proj(x - s (c - A'y))
fn primal_step(p: &Problem, xu: &mut [f64], x: &[f64], aty: &[f64], s: f64) {
    for i in 0..p.ncols {
        let t = s.mul_add_c(aty[i], (-s).mul_add_c(p.cost[i], x[i]));
        let t = if t < p.upper[i] { t } else { p.upper[i] };
        xu[i] = if t > p.lower[i] { t } else { p.lower[i] };
    }
}

/// PDHG_dualGradientStep: yu = proj(y + d (b - A (2 xu - x)))
fn dual_step(p: &Problem, yu: &mut [f64], y: &[f64], ax: &[f64], axu: &[f64], d: f64) {
    let m2d = -2.0 * d;
    for i in 0..p.nrows {
        yu[i] = d.mul_add_c(ax[i], m2d.mul_add_c(axu[i], d.mul_add_c(p.rhs[i], y[i])));
    }
    proj_pos(&mut yu[p.neqs..p.nrows]);
}

/// time string of PDHG_Print_Iter: "%6.2fs", or "%6ds" from 100s, in 7
/// characters
fn time_string(t: f64) -> String {
    let mut s = if t < 100.0 {
        format!("{}s", fixed(t, 6, 2))
    } else {
        format!("{:>6}s", t as i32)
    };
    s.truncate(7);
    s
}

/// CUPDLPwork
pub(super) struct Pdhg<'a> {
    pub(super) p: Problem,
    pub(super) sc: Scaling,
    log: &'a Log,
    iter_lim: i32,
    time_lim: f64,
    log_interval: i32,
    primal_tol: f64,
    dual_tol: f64,
    gap_tol: f64,
    restart: bool,
    r: Res,
    v: ResVecs,
    it: Iterates,
    st: Step,
    pub(super) t: Timers,
    buffer: Vec<f64>,
    buffer2: Vec<f64>,
    buffer3: Vec<f64>,
}

impl<'a> Pdhg<'a> {
    /// PDHG_Alloc and PDHG_SetUserParam
    pub(super) fn new(p: Problem, sc: Scaling, log: &'a Log, params: &Params) -> Self {
        let (n, m) = (p.ncols, p.nrows);
        let lower_filtered = p
            .lower
            .iter()
            .map(|&l| if l > f64::NEG_INFINITY { l } else { 0.0 });
        let upper_filtered = p
            .upper
            .iter()
            .map(|&u| if u < f64::INFINITY { u } else { 0.0 });
        let v = ResVecs {
            primal_residual: vec![0.0; m],
            dual_residual: vec![0.0; n],
            primal_residual_avg: vec![0.0; m],
            dual_residual_avg: vec![0.0; n],
            slack_pos: vec![0.0; n],
            slack_neg: vec![0.0; n],
            slack_pos_avg: vec![0.0; n],
            slack_neg_avg: vec![0.0; n],
            lower_filtered: lower_filtered.collect(),
            upper_filtered: upper_filtered.collect(),
            col1: vec![0.0; n],
            col2: vec![0.0; n],
            row1: vec![0.0; m],
        };
        let pair = |len| [vec![0.0; len], vec![0.0; len]];
        let it = Iterates {
            x: pair(n),
            y: pair(m),
            ax: pair(m),
            aty: pair(n),
            x_avg: vec![0.0; n],
            y_avg: vec![0.0; m],
            ax_avg: vec![0.0; m],
            aty_avg: vec![0.0; n],
            x_sum: vec![0.0; n],
            y_sum: vec![0.0; m],
            x_last_restart: vec![0.0; n],
            y_last_restart: vec![0.0; m],
            last_restart_iter: 0,
        };
        let st = Step {
            adaptive: params.line_search == 2,
            primal: 0.0,
            dual: 0.0,
            sum_primal: 0.0,
            sum_dual: 0.0,
            beta: 0.0,
            n_step_iter: 0,
        };
        let buf = n.max(m).max(2048);
        let w = Pdhg {
            p,
            sc,
            log,
            iter_lim: params.iter_lim,
            time_lim: params.time_lim,
            log_interval: 100,
            primal_tol: params.primal_tol,
            dual_tol: params.dual_tol,
            gap_tol: params.gap_tol,
            restart: params.restart == 1,
            r: Res::new(),
            v,
            it,
            st,
            t: Timers::default(),
            buffer: vec![0.0; m],
            buffer2: vec![0.0; buf],
            buffer3: vec![0.0; buf],
        };
        w.print_param(params);
        w
    }

    /// PDHG_PrintPDHGParam
    fn print_param(&self, params: &Params) {
        if self.log.level < 2 {
            return;
        }
        let dashes = "--------------------------------------------------\n";
        let mut s = format!("\n\n{dashes}CUPDHG Parameters:\n{dashes}\n");
        s += &format!("    nIterLim:          {}\n", self.iter_lim);
        s += &format!("    dTimeLim (sec):    {}\n", fixed(self.time_lim, 0, 2));
        s += &format!("    ifScaling:         {}\n", params.scaling);
        s += "    ifRuizScaling:     1\n    ifL2Scaling:       0\n";
        s += "    ifPcScaling:       1\n";
        s += &format!("    eLineSearchMethod: {}\n", params.line_search);
        s += &format!("    dPrimalTol:        {}\n", e(self.primal_tol, 4));
        s += &format!("    dDualTol:          {}\n", e(self.dual_tol, 4));
        s += &format!("    dGapTol:           {}\n", e(self.gap_tol, 4));
        s += &format!("    dFeasTol:          {}\n", e(self.r.feas_tol, 4));
        s += &format!("    eRestartMethod:    {}\n", params.restart);
        s += &format!("    nLogLevel:    {}\n", self.log.level);
        s += &format!("    nLogInterval:    {}\n", self.log_interval);
        s += &format!("\n{dashes}\n");
        self.log.out(&s);
    }

    /// The index of the current iterate
    fn k(&self) -> usize {
        (self.t.n_iter % 2) as usize
    }

    /// PDHG_Compute_Residuals
    fn compute_residuals(&mut self) {
        let k = self.k();
        let (rs, cs) = self.sc.active();
        let (p, it, v, r) = (&self.p, &self.it, &mut self.v, &mut self.r);
        (r.primal_feas, r.primal_obj) =
            primal_feasibility(p, rs, &it.ax[k], &it.x[k], &mut v.primal_residual);
        (r.dual_feas, r.dual_obj) = dual_feasibility(
            p,
            cs,
            &v.lower_filtered,
            &v.upper_filtered,
            &it.aty[k],
            &it.y[k],
            &mut v.dual_residual,
            &mut v.slack_pos,
            &mut v.slack_neg,
        );
        (r.primal_feas_avg, r.primal_obj_avg) =
            primal_feasibility(p, rs, &it.ax_avg, &it.x_avg, &mut v.primal_residual_avg);
        (r.dual_feas_avg, r.dual_obj_avg) = dual_feasibility(
            p,
            cs,
            &v.lower_filtered,
            &v.upper_filtered,
            &it.aty_avg,
            &it.y_avg,
            &mut v.dual_residual_avg,
            &mut v.slack_pos_avg,
            &mut v.slack_neg_avg,
        );
        r.gap = r.primal_obj - r.dual_obj;
        r.rel_gap =
            (r.primal_obj - r.dual_obj).abs() / (1.0 + r.primal_obj.abs() + r.dual_obj.abs());
        r.gap_avg = r.primal_obj_avg - r.dual_obj_avg;
        r.rel_gap_avg = (r.primal_obj_avg - r.dual_obj_avg).abs()
            / (1.0 + r.primal_obj_avg.abs() + r.dual_obj_avg.abs());
    }

    /// PDHG_Compute_Infeas_Residuals
    fn compute_infeas_residuals(&mut self) {
        let k = self.k();
        let (rs, cs) = self.sc.active();
        let (p, it, v, r) = (&self.p, &self.it, &mut self.v, &mut self.r);
        (r.primal_infeas_obj, r.primal_infeas_res) = primal_infeasibility(
            p,
            cs,
            &it.y[k],
            &v.slack_pos,
            &v.slack_neg,
            &it.aty[k],
            r.dual_obj,
            &mut v.col1,
        );
        (r.dual_infeas_obj, r.dual_infeas_res) = dual_infeasibility(
            p,
            rs,
            cs,
            &it.x[k],
            &it.ax[k],
            r.primal_obj,
            &mut v.col1,
            &mut v.col2,
            &mut v.row1,
        );
        (r.primal_infeas_obj_avg, r.primal_infeas_res_avg) = primal_infeasibility(
            p,
            cs,
            &it.y_avg,
            &v.slack_pos_avg,
            &v.slack_neg_avg,
            &it.aty_avg,
            r.dual_obj_avg,
            &mut v.col1,
        );
        (r.dual_infeas_obj_avg, r.dual_infeas_res_avg) = dual_infeasibility(
            p,
            rs,
            cs,
            &it.x_avg,
            &it.ax_avg,
            r.primal_obj_avg,
            &mut v.col1,
            &mut v.col2,
            &mut v.row1,
        );
    }

    /// PDHG_Init_Variables
    fn init_variables(&mut self, has_variables: bool) {
        let k = self.k();
        let (p, it) = (&self.p, &mut self.it);
        let x = &mut it.x[k];
        if !has_variables {
            x.fill(0.0);
        }
        let project = |x: &mut [f64]| {
            proj_ub(x, &p.upper);
            proj_lb(x, &p.lower);
        };
        project(x);
        if self.log.level > 0 {
            self.log.out(&format!("||x0||_2 = {}\n", e(nrm2(x), 6)));
        }
        if !has_variables {
            it.y[k].fill(0.0);
        }
        self.t.ax(p, &it.x[k], &mut it.ax[k]);
        self.t.aty(p, &it.y[k], &mut it.aty[k]);
        it.x_sum.fill(0.0);
        it.y_sum.fill(0.0);
        it.x_avg.fill(0.0);
        it.y_avg.fill(0.0);
        project(&mut it.x_sum);
        project(&mut it.x_avg);
        self.st.sum_primal = 0.0;
        self.st.sum_dual = 0.0;
        it.x_last_restart.fill(0.0);
        it.y_last_restart.fill(0.0);
    }

    /// PDHG_Power_Method: an estimate of the largest eigenvalue of AA'.
    /// clang leaves its reductions scalar, so all fused (dot_fused)
    fn power_method(&mut self) -> f64 {
        if self.log.level > 0 {
            self.log.out("Power Method:\n");
        }
        let k = self.k();
        let (p, it, q) = (&self.p, &mut self.it, &mut self.buffer);
        let (ax, aty) = (&mut it.ax[k], &mut it.aty[k]);
        q.fill(1.0);
        let mut lambda = 0.0;
        let mut previous_lambda = 0.0;
        let log_iters = self.log.level > 1;
        if log_iters {
            self.log.out("It       lambda   dl_lambda    residual\n");
        }
        for iter in 0..20 {
            self.t.aty(p, q, aty);
            self.t.ax(p, aty, ax);
            q.copy_from_slice(ax);
            let q_norm = dot_fused(q, q).sqrt();
            scale(1.0 / q_norm, q);
            self.t.aty(p, q, aty);
            lambda = dot_fused(aty, aty);
            axpy(-lambda, q, ax);
            // ponytail: the C sums the first ncols entries of ax (reading
            // past its nrows when ncols > nrows); this logged value stops
            // at nrows
            let n = p.ncols.min(p.nrows);
            let res = dot_fused(&ax[..n], &ax[..n]);
            let dl_lambda = (lambda - previous_lambda).abs();
            previous_lambda = lambda;
            if log_iters {
                self.log.out(&format!(
                    "{iter:>2} {:>12} {:>11} {:>11}\n",
                    g(lambda, 6),
                    g(dl_lambda, 4),
                    g(res, 4)
                ));
            }
        }
        lambda
    }

    /// PDHG_Init_Step_Sizes
    fn init_step_sizes(&mut self) {
        let a = dot(&self.p.cost, &self.p.cost);
        let b = dot(&self.p.rhs, &self.p.rhs);
        let beta = if a.min(b) > 1e-6 { a / b } else { 1.0 };
        if !self.st.adaptive {
            let lambda = self.power_method();
            let st = &mut self.st;
            st.beta = beta;
            st.primal = 0.8 / lambda.sqrt();
            st.dual = st.primal;
            st.primal /= st.beta.sqrt();
            st.dual *= st.beta.sqrt();
            if self.log.level > 1 {
                self.log.out(&format!(
                    "Initial step sizes from power method lambda = {}: primal = {}; dual = {}\n",
                    g(lambda, 6),
                    g(st.primal, 6),
                    g(st.dual, 6)
                ));
            }
        } else {
            let st = &mut self.st;
            st.beta = beta;
            st.primal = (1.0 / self.p.mat_inf_norm) / st.beta.sqrt();
            st.dual = st.primal * st.beta;
        }
        self.it.last_restart_iter = 0;
        self.st.sum_primal = 0.0;
        self.st.sum_dual = 0.0;
    }

    /// PDHG_Compute_Average_Iterate
    fn compute_average_iterate(&mut self) {
        let st = &self.st;
        let primal_scale = if st.sum_primal > 0.0 {
            1.0 / st.sum_primal
        } else {
            1.0
        };
        let dual_scale = if st.sum_dual > 0.0 {
            1.0 / st.sum_dual
        } else {
            1.0
        };
        let it = &mut self.it;
        it.x_avg.copy_from_slice(&it.x_sum);
        it.y_avg.copy_from_slice(&it.y_sum);
        scale(primal_scale, &mut it.x_avg);
        scale(dual_scale, &mut it.y_avg);
        self.t.ax(&self.p, &it.x_avg, &mut it.ax_avg);
        self.t.aty(&self.p, &it.y_avg, &mut it.aty_avg);
    }

    /// PDHG_Update_Average
    fn update_average(&mut self) {
        let kn = 1 - self.k();
        let st = &mut self.st;
        let it = &mut self.it;
        let mean = (st.primal * st.dual).sqrt();
        axpy(mean, &it.x[kn], &mut it.x_sum);
        axpy(mean, &it.y[kn], &mut it.y_sum);
        st.sum_primal += mean;
        st.sum_dual += mean;
    }

    /// PDHG_Update_Iterate_Constant_Step_Size
    fn update_constant(&mut self) {
        let k = self.k();
        let (p, it, t) = (&self.p, &mut self.it, &mut self.t);
        let (x, xu) = cur_next(&mut it.x, k);
        let (y, yu) = cur_next(&mut it.y, k);
        let (ax, axu) = cur_next(&mut it.ax, k);
        let (aty, atyu) = cur_next(&mut it.aty, k);
        t.ax(p, x, ax);
        t.aty(p, y, aty);
        primal_step(p, xu, x, aty, self.st.primal);
        t.ax(p, xu, axu);
        dual_step(p, yu, y, ax, axu, self.st.dual);
        t.aty(p, yu, atyu);
    }

    /// PDHG_Update_Iterate_Adaptive_Step_Size: false on the time limit
    fn update_adaptive(&mut self) -> bool {
        let k = self.k();
        let (p, it) = (&self.p, &mut self.it);
        let (x, xu) = cur_next(&mut it.x, k);
        let (y, yu) = cur_next(&mut it.y, k);
        let (ax, axu) = cur_next(&mut it.ax, k);
        let (aty, atyu) = cur_next(&mut it.aty, k);
        let st = &mut self.st;
        let mut step = (st.primal * st.dual).sqrt();
        loop {
            st.n_step_iter += 1;
            let primal_step_update = step / st.beta.sqrt();
            let dual_step_update = step * st.beta.sqrt();
            primal_step(p, xu, x, aty, primal_step_update);
            self.t.ax(p, xu, axu);
            dual_step(p, yu, y, ax, axu, dual_step_update);
            self.t.aty(p, yu, atyu);

            // cupdlp_compute_interaction_and_movement
            let (n, m) = (p.ncols, p.nrows);
            let beta = st.beta.sqrt();
            let dx = &mut self.buffer2[..n];
            dx.iter_mut()
                .zip(x.iter().zip(xu.iter()))
                .for_each(|(d, (a, b))| *d = a - b);
            let d_x = dot(dx, dx);
            let daty = &mut self.buffer3[..n];
            daty.iter_mut()
                .zip(aty.iter().zip(atyu.iter()))
                .for_each(|(d, (a, b))| *d = a - b);
            let interaction = dot(dx, daty);
            let dy = &mut self.buffer3[..m];
            dy.iter_mut()
                .zip(y.iter().zip(yu.iter()))
                .for_each(|(d, (a, b))| *d = a - b);
            let d_y = dot(dy, dy);
            let movement = (d_x * 0.5).mul_add_c(beta, d_y / (2.0 * beta));

            let limit = if interaction != 0.0 {
                movement / interaction.abs()
            } else {
                f64::INFINITY
            };
            let done = step <= limit;
            if !done {
                // CUPDLP_CHECK_TIMEOUT
                self.t.solving_time = time_stamp() - self.t.solving_beg;
                if self.t.solving_time > self.time_lim {
                    return false;
                }
            }
            let n1 = st.n_step_iter as f64 + 1.0;
            let first = (1.0 - n1.powf(-0.3)) * limit;
            let second = (1.0 + n1.powf(-0.6)) * step;
            step = first.min(second);
            if done {
                break;
            }
        }
        st.primal = step / st.beta.sqrt();
        st.dual = step * st.beta.sqrt();
        true
    }

    /// PDHG_Update_Iterate
    fn update_iterate(&mut self) {
        self.t.n_update_iterate += 1;
        let start = time_stamp();
        if self.st.adaptive {
            if !self.update_adaptive() {
                // the C returns from the time-out without averaging
                return;
            }
        } else {
            self.update_constant();
        }
        self.update_average();
        self.t.update_iterate_time += time_stamp() - start;
    }

    /// PDHG_Check_Restart_GPU
    fn check_restart(&mut self) -> Restart {
        let r = &mut self.r;
        let n_iter = self.t.n_iter;
        let last = self.it.last_restart_iter;
        if n_iter == last {
            r.primal_feas_last_restart = r.primal_feas;
            r.dual_feas_last_restart = r.dual_feas;
            r.gap_last_restart = r.gap;
            r.primal_feas_last_cand = r.primal_feas;
            r.dual_feas_last_cand = r.dual_feas;
            r.gap_last_cand = r.gap;
            return Restart::No;
        }
        let beta = self.st.beta;
        let mu_current = restart_score(beta, r.primal_feas, r.dual_feas, r.gap);
        let mu_average = restart_score(beta, r.primal_feas_avg, r.dual_feas_avg, r.gap_avg);
        let (mut choice, mu_candidate) = if mu_current < mu_average {
            (Restart::ToCurrent, mu_current)
        } else {
            (Restart::ToAverage, mu_average)
        };
        if ((n_iter - last) as f64) < 0.36 * n_iter as f64 {
            let mu_last_restart = restart_score(
                beta,
                r.primal_feas_last_restart,
                r.dual_feas_last_restart,
                r.gap_last_restart,
            );
            // sufficient decay, else necessary decay
            if mu_candidate >= 0.2 * mu_last_restart {
                let mu_last_candidate = restart_score(
                    beta,
                    r.primal_feas_last_cand,
                    r.dual_feas_last_cand,
                    r.gap_last_cand,
                );
                if !(mu_candidate < 0.8 * mu_last_restart && mu_candidate > mu_last_candidate) {
                    choice = Restart::No;
                }
            }
        }
        if mu_current < mu_average {
            r.primal_feas_last_cand = r.primal_feas;
            r.dual_feas_last_cand = r.dual_feas;
            r.gap_last_cand = r.gap;
        } else {
            r.primal_feas_last_cand = r.primal_feas_avg;
            r.dual_feas_last_cand = r.dual_feas_avg;
            r.gap_last_cand = r.gap_avg;
        }
        if choice != Restart::No && self.log.level > 1 {
            let which = if mu_current < mu_average {
                "current"
            } else {
                "average"
            };
            self.log
                .out(&format!("Last restart was iter {last}: {which}\n"));
        }
        choice
    }

    /// PDHG_Compute_Step_Size_Ratio
    fn compute_step_size_ratio(&mut self) {
        let k = self.k();
        let (it, st) = (&self.it, &mut self.st);
        let mean = (st.primal * st.dual).sqrt();
        let diff_norm = |a: &[f64], b: &[f64], buf: &mut [f64]| {
            let d = &mut buf[..a.len()];
            d.iter_mut()
                .zip(a.iter().zip(b))
                .for_each(|(d, (a, b))| *d = a - b);
            nrm2(d)
        };
        let diff_primal = diff_norm(&it.x[k], &it.x_last_restart, &mut self.buffer2);
        let diff_dual = diff_norm(&it.y[k], &it.y_last_restart, &mut self.buffer2);
        if diff_primal.min(diff_dual) > 1e-10 {
            let beta_update = diff_dual / diff_primal;
            let log_beta = 0.5f64.mul_add_c(beta_update.ln(), 0.5 * st.beta.sqrt().ln());
            let e = log_beta.exp();
            st.beta = e * e;
        }
        st.primal = mean / st.beta.sqrt();
        st.dual = st.primal * st.beta;
    }

    /// PDHG_Restart_Iterate (with eRestartMethod GPU, or none)
    fn restart_iterate(&mut self) {
        if !self.restart {
            return;
        }
        let choice = self.check_restart();
        if choice == Restart::No {
            return;
        }
        let k = self.k();
        let (it, r) = (&mut self.it, &mut self.r);
        self.st.sum_primal = 0.0;
        self.st.sum_dual = 0.0;
        it.x_sum.fill(0.0);
        it.y_sum.fill(0.0);
        if choice == Restart::ToAverage {
            r.primal_feas_last_restart = r.primal_feas_avg;
            r.dual_feas_last_restart = r.dual_feas_avg;
            r.gap_last_restart = r.gap_avg;
            it.x[k].copy_from_slice(&it.x_avg);
            it.y[k].copy_from_slice(&it.y_avg);
            it.ax[k].copy_from_slice(&it.ax_avg);
            it.aty[k].copy_from_slice(&it.aty_avg);
        } else {
            r.primal_feas_last_restart = r.primal_feas;
            r.dual_feas_last_restart = r.dual_feas;
            r.gap_last_restart = r.gap;
        }
        self.compute_step_size_ratio();
        let it = &mut self.it;
        it.x_last_restart.copy_from_slice(&it.x[k]);
        it.y_last_restart.copy_from_slice(&it.y[k]);
        it.last_restart_iter = self.t.n_iter;
        self.compute_residuals();
    }

    fn tol_primal(&self) -> f64 {
        self.primal_tol * (1.0 + self.sc.norm_rhs)
    }

    fn tol_dual(&self) -> f64 {
        self.dual_tol * (1.0 + self.sc.norm_cost)
    }

    /// PDHG_Check_Termination(_Average) on (primal feas, dual feas, rel gap)
    fn check_termination(&self, print: bool, (pf, df, gap): (f64, f64, f64)) -> bool {
        let (tp, td) = (self.tol_primal(), self.tol_dual());
        if print {
            self.log.out(&format!(
                "Termination check: {}|{}  {}|{}  {}|{}\n",
                e(pf, 6),
                e(tp, 6),
                e(df, 6),
                e(td, 6),
                e(gap, 6),
                e(self.gap_tol, 6)
            ));
        }
        pf < tp && df < td && gap < self.gap_tol
    }

    /// PDHG_Check_Infeasibility: whether infeasibility or unboundedness is
    /// detected
    fn check_infeasibility(&mut self) -> bool {
        let r = &mut self.r;
        let primal_infeasible = |obj: f64, res: f64| obj > 0.0 && res < r.feas_tol * obj;
        let dual_infeasible = |obj: f64, res: f64| obj < 0.0 && res < -r.feas_tol * obj;
        let checks = [
            (
                true,
                TermIterate::Last,
                primal_infeasible(r.primal_infeas_obj, r.primal_infeas_res),
            ),
            (
                false,
                TermIterate::Last,
                dual_infeasible(r.dual_infeas_obj, r.dual_infeas_res),
            ),
            (
                true,
                TermIterate::Average,
                primal_infeasible(r.primal_infeas_obj_avg, r.primal_infeas_res_avg),
            ),
            (
                false,
                TermIterate::Average,
                dual_infeasible(r.dual_infeas_obj_avg, r.dual_infeas_res_avg),
            ),
        ];
        let mut found = false;
        for (primal, iterate, infeasible) in checks {
            if infeasible {
                if primal {
                    r.primal_code = TermCode::Infeasible;
                } else {
                    r.dual_code = TermCode::Infeasible;
                }
                r.term_infeas_iterate = iterate;
                found = true;
            }
        }
        found
    }

    fn print_header(&self) {
        self.log.out(&format!(
            "{:>9}  {:>15}  {:>15}   {:>8}  {:>10}  {:>8} {:>7}\n",
            "Iter", "Primal.Obj", "Dual.Obj", "Gap", "Primal.Inf", "Dual.Inf", "Time"
        ));
    }

    /// PDHG_Print_Iter (average: PDHG_Print_Iter_Average)
    fn print_iter(&self, average: bool) {
        let r = &self.r;
        let (po, d_o, gap, pf, df, tag) = if average {
            (
                r.primal_obj_avg,
                r.dual_obj_avg,
                r.rel_gap_avg,
                r.primal_feas_avg,
                r.dual_feas_avg,
                "[A]",
            )
        } else {
            (
                r.primal_obj,
                r.dual_obj,
                r.rel_gap,
                r.primal_feas,
                r.dual_feas,
                "[L]",
            )
        };
        self.log.out(&format!(
            "{:>9}  {:>15}  {:>15}  {:>8}  {:>10}  {:>8} {:>7} {tag}\n",
            self.t.n_iter,
            plus(e(po, 8)),
            plus(e(d_o, 8)),
            plus(e(gap, 2)),
            e(pf / (1.0 + self.sc.norm_rhs), 2),
            e(df / (1.0 + self.sc.norm_cost), 2),
            time_string(self.t.solving_time)
        ));
    }

    /// PDHG_Solve
    pub(super) fn solve(&mut self, has_variables: bool) {
        self.t.n_iter = 0;
        self.t.solving_beg = time_stamp();
        self.init_step_sizes();
        self.init_variables(has_variables);

        let level = self.log.level;
        let full_print = level >= 2;
        // The header repeats every 50 logged iterations at level 1
        const ITER_LOG_BETWEEN_HEADER: i32 = 50;
        let mut iter_log_since_header = ITER_LOG_BETWEEN_HEADER;
        self.t.n_iter = 0;
        while self.t.n_iter < self.iter_lim {
            let n_iter = self.t.n_iter;
            self.t.solving_time = time_stamp() - self.t.solving_beg;
            let at_limit = n_iter == self.iter_lim - 1 || self.t.solving_time > self.time_lim;
            // CUPDLP_RELEASE_INTERVAL = 40
            let checking = n_iter < 10 || at_limit || n_iter % 40 == 0;
            let print =
                level > 0 && ((checking && n_iter % (40 * self.log_interval) == 0) || at_limit);
            if checking {
                self.compute_average_iterate();
                self.compute_residuals();
                self.compute_infeas_residuals();
                if print {
                    if full_print || iter_log_since_header == ITER_LOG_BETWEEN_HEADER {
                        self.print_header();
                        iter_log_since_header = 0;
                    }
                    if n_iter == 0 || full_print {
                        self.print_iter(false);
                    }
                    if n_iter > 0 || full_print {
                        self.print_iter(true);
                    }
                    iter_log_since_header += 1;
                }
                let termination_print = print && full_print;
                let r = &self.r;
                if self
                    .check_termination(termination_print, (r.primal_feas, r.dual_feas, r.rel_gap))
                {
                    self.r.term_iterate = TermIterate::Last;
                    self.r.term_code = TermCode::Optimal;
                    break;
                }
                if self.check_termination(
                    termination_print,
                    (r.primal_feas_avg, r.dual_feas_avg, r.rel_gap_avg),
                ) {
                    let k = self.k();
                    let (it, v) = (&mut self.it, &mut self.v);
                    it.x[k].copy_from_slice(&it.x_avg);
                    it.y[k].copy_from_slice(&it.y_avg);
                    it.ax[k].copy_from_slice(&it.ax_avg);
                    it.aty[k].copy_from_slice(&it.aty_avg);
                    v.slack_pos.copy_from_slice(&v.slack_pos_avg);
                    v.slack_neg.copy_from_slice(&v.slack_neg_avg);
                    self.r.term_iterate = TermIterate::Average;
                    self.r.term_code = TermCode::Optimal;
                    break;
                }
                if self.check_infeasibility() {
                    self.r.term_code = TermCode::InfeasibleOrUnbounded;
                    break;
                }
                if self.t.solving_time > self.time_lim || n_iter >= self.iter_lim - 1 {
                    self.r.term_code = TermCode::TimeOrIterLimit;
                    break;
                }
                self.restart_iterate();
            }
            self.update_iterate();
            self.t.n_iter += 1;
        }
        self.print_summary();
    }

    /// The logging at the end of PDHG_Solve
    fn print_summary(&self) {
        let level = self.log.level;
        let (r, t) = (&self.r, &self.t);
        if level > 0 && t.n_iter > 0 {
            let full_print = level >= 2;
            if full_print {
                self.print_header();
            }
            if r.term_iterate == TermIterate::Last || full_print {
                self.print_iter(false);
            }
            if r.term_iterate == TermIterate::Average || full_print {
                self.print_iter(true);
            }
        }
        if level > 0 {
            let mut s = format!("\n{:<27} ", "Solving information:");
            match r.term_code {
                TermCode::Optimal => {
                    s += match r.term_iterate {
                        TermIterate::Last => "Optimal current solution.\n",
                        TermIterate::Average => "Optimal average solution.\n",
                    }
                }
                TermCode::TimeOrIterLimit => {
                    if t.solving_time > self.time_lim {
                        s += "Time limit reached.\n";
                    } else if t.n_iter >= self.iter_lim - 1 {
                        s += "Iteration limit reached.\n";
                    }
                }
                TermCode::InfeasibleOrUnbounded => {
                    s += match (r.primal_code, r.dual_code) {
                        (TermCode::Infeasible, TermCode::Feasible) => {
                            "Infeasible or unbounded: primal infeasible."
                        }
                        (TermCode::Feasible, TermCode::Infeasible) => {
                            "Infeasible or unbounded: dual infeasible."
                        }
                        _ => "Infeasible or unbounded: both primal and dual infeasible.",
                    };
                    s += match r.term_infeas_iterate {
                        TermIterate::Last => " [L]\n",
                        TermIterate::Average => " [A]\n",
                    };
                }
                _ => s += "Unexpected.\n",
            }
            let average =
                r.term_code == TermCode::Optimal && r.term_iterate == TermIterate::Average;
            let (po, d_o, pf, df, gap, rel_gap) = if average {
                (
                    r.primal_obj_avg,
                    r.dual_obj_avg,
                    r.primal_feas_avg,
                    r.dual_feas_avg,
                    r.gap_avg,
                    r.rel_gap_avg,
                )
            } else {
                (
                    r.primal_obj,
                    r.dual_obj,
                    r.primal_feas,
                    r.dual_feas,
                    r.gap,
                    r.rel_gap,
                )
            };
            let obj = |name: &str, v: f64| format!("{name:>27} {:>15}\n", plus(e(v, 8)));
            let pair = |name: &str, a: f64, b: f64| {
                format!("{name:>27} {:>8} / {:>8}\n", e(a, 2), e(b, 2))
            };
            s += &obj("Primal objective:", po);
            s += &obj("Dual objective:", d_o);
            s += &pair(
                "Primal infeas (abs/rel):",
                pf,
                pf / (1.0 + self.sc.norm_rhs),
            );
            s += &pair("Dual infeas (abs/rel):", df, df / (1.0 + self.sc.norm_cost));
            s += &pair("Duality gap (abs/rel):", gap.abs(), rel_gap);
            s += &format!("{:>27} {}\n\n", "Number of iterations:", t.n_iter);
            self.log.out(&s);
        }
        if level > 1 {
            let mut s = "Timing information:\n".to_string();
            s += &format!(
                "{:>21} {} in {} iterations\n",
                "Total solver time",
                e(t.solving_time + t.scaling_time, 6),
                t.n_iter
            );
            s += &format!(
                "{:>21} {} in {} iterations\n",
                "Solve time",
                e(t.solving_time, 6),
                t.n_iter
            );
            s += &format!(
                "{:>21} {} \n",
                "Iters per sec",
                e(t.n_iter as f64 / t.solving_time, 6)
            );
            s += &format!("{:>21} {}\n", "Scaling time", e(t.scaling_time, 6));
            s += &format!("{:>21} {}\n", "Presolve time", e(0.0, 6));
            s += &format!("{:>21} {} in {} calls\n", "Ax", e(t.ax_time, 6), t.n_ax);
            s += &format!("{:>21} {} in {} calls\n", "Aty", e(t.aty_time, 6), t.n_aty);
            s += &format!("{:>21} {} in {} calls\n", "ComputeResiduals", e(0.0, 6), 0);
            s += &format!(
                "{:>21} {} in {} calls\n",
                "UpdateIterates",
                e(t.update_iterate_time, 6),
                t.n_update_iterate
            );
            self.log.out(&s);
        }
    }

    /// PDHG_PreSolve: the hot start from a HiGHS solution, when both its
    /// values and duals are valid
    pub(super) fn presolve(&mut self, lp: &super::Formulated, sol: &super::Solution) {
        let (p, it) = (&self.p, &mut self.it);
        let ncols_origin = lp.ncols_origin;
        let x = &mut it.x[0];
        let y = &mut it.y[0];
        x[..ncols_origin].copy_from_slice(&sol.col_value[..ncols_origin]);
        let mut i_col = ncols_origin;
        for (i_row, (&ty, &new_idx)) in lp.constraint_type.iter().zip(&lp.new_idx).enumerate() {
            let mu = if ty == super::LEQ { -1.0 } else { 1.0 };
            y[new_idx] = p.sense * mu * sol.row_dual[i_row];
            if ty == super::BOUND {
                x[i_col] = sol.row_value[i_row];
                i_col += 1;
            }
        }
        if self.sc.scaled {
            edot(x, &self.sc.col_scale);
            edot(y, &self.sc.row_scale);
        }
    }

    /// PDHG_PostSolve: unscale the final iterate and recover the HiGHS
    /// solution
    pub(super) fn postsolve(&mut self, lp: &super::Formulated, sol: &mut super::Solution) {
        let k = self.k();
        let (p, it, v) = (&self.p, &mut self.it, &mut self.v);
        let (x, y, ax) = (&mut it.x[k], &mut it.y[k], &mut it.ax[k]);
        if self.sc.scaled {
            let (cs, rs) = (&self.sc.col_scale, &self.sc.row_scale);
            ediv(x, cs);
            ediv(y, rs);
            edot(&mut v.slack_pos, cs);
            edot(&mut v.slack_neg, cs);
            edot(ax, rs);
            edot(&mut it.aty[k], cs);
        }
        let n0 = lp.ncols_origin;
        sol.col_value[..n0].copy_from_slice(&x[..n0]);
        let mut j = n0;
        for (i, &ty) in lp.constraint_type.iter().enumerate() {
            let value = ax[lp.new_idx[i]];
            sol.row_value[i] = if ty == super::LEQ {
                -value
            } else if ty == super::BOUND {
                j += 1;
                value + x[j - 1]
            } else {
                value
            };
        }
        for i in 0..n0 {
            sol.col_dual[i] = (v.slack_pos[i] - v.slack_neg[i]) * p.sense;
        }
        for (i, &ty) in lp.constraint_type.iter().enumerate() {
            let dual = y[lp.new_idx[i]] * p.sense;
            sol.row_dual[i] = if ty == super::LEQ { -dual } else { dual };
        }
    }

    pub(super) fn result(&self) -> (TermCode, i32) {
        (self.r.term_code, self.t.n_iter)
    }
}
