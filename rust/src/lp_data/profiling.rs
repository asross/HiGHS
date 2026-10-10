//! HighsProfiling (lp_data/HStruct.h, HighsInterface.cpp) for crest's
//! Rust Highs object: per-thread clocks of the presolve, solve and
//! postsolve (the MIP's timing report reads them), the sub-solvers
//! (log_dev_level > 0) and the MIP (with highs_analysis_level's MIP time
//! bit). The C++ builds use the C++ class through the same callbacks.
//!
//! Each thread writes only its own record, as in the C++, so the records
//! are reached through raw pointers without a lock.

use super::lp_handle::Timer;

// The clocks (HConst.h, mip/MipTimer.h)
pub const PRESOLVE_TIME: usize = 0;
pub const SOLVE_TIME: usize = 1;
pub const POSTSOLVE_TIME: usize = 2;
const TO_PRESOLVE_SOLVE_POSTSOLVE: usize = 3;
pub const SUB_SOLVER_MIP: usize = 3;
pub const SUB_SOLVER_DU_SIMPLEX_BASIS: usize = 4;
pub const SUB_SOLVER_DU_SIMPLEX_NO_BASIS: usize = 5;
pub const SUB_SOLVER_PR_SIMPLEX_BASIS: usize = 6;
pub const SUB_SOLVER_PR_SIMPLEX_NO_BASIS: usize = 7;
const SUB_SOLVER_HIPO_AC: usize = 10;
const SUB_SOLVER_IPX_AC: usize = 11;
pub const SUB_SOLVER_PDLP: usize = 12;
pub const SUB_SOLVER_QP_ASM: usize = 13;
const TO_SUB_SOLVER: usize = 15;
const TO_MIP_CLOCK: usize = 88;
/// kHighsAnalysisLevelMipTime
pub const ANALYSIS_LEVEL_MIP_TIME: i32 = 128;

/// The clock ids of the Rust MIP solver's clock indices (root.rs and
/// driver.rs mod clk, mip/host/mod.rs mod prof): HighsMipHost.cpp's
/// kMipClocks
pub const MIP_CLOCKS: [usize; 54] = [
    51, 52, 53, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 0, 18, 19, 1, 20, 22, 21,
    23, 24, 25, 33, 66, 64, 67, 57, 54, 55, 59, 60, 61, 56, 30, 31, 32, 2, 86, 14, 11, 87, 26, 27,
];

/// HighsProfilingRecord
#[derive(Clone)]
struct Record {
    num_call: Vec<i32>,
    run_time: Vec<f64>,
    start_time: Vec<f64>,
}

/// HighsProfiling
pub struct Profiling {
    timer: *const Timer,
    pub multi_threaded: bool,
    pub sub_solver: bool,
    pub mip: bool,
    num_clock: usize,
    submip: Vec<bool>,
    record: Vec<Record>,
    submip_record: Vec<Record>,
}

/// highs::parallel::thread_num()
fn thread_num() -> usize {
    crate::parallel::thread_num().max(0) as usize
}

impl Profiling {
    pub fn new() -> Profiling {
        Profiling {
            timer: std::ptr::null(),
            multi_threaded: true,
            sub_solver: false,
            mip: false,
            num_clock: 0,
            submip: Vec::new(),
            record: Vec::new(),
            submip_record: Vec::new(),
        }
    }

    /// initialize (multi_threaded is set by the caller)
    pub fn initialize(&mut self, timer: &Timer, sub_solver: bool, mip: bool) {
        self.timer = timer;
        self.num_clock = TO_PRESOLVE_SOLVE_POSTSOLVE;
        self.sub_solver = sub_solver;
        if sub_solver {
            self.num_clock = TO_SUB_SOLVER;
        }
        self.mip = sub_solver && mip;
        if self.mip {
            self.num_clock = TO_MIP_CLOCK;
        }
        let n = self.num_clock;
        let r = Record { num_call: vec![0; n], run_time: vec![0.0; n], start_time: vec![1.0; n] };
        let num_thread = self.num_thread();
        self.submip = vec![false; num_thread];
        self.record = vec![r.clone(); num_thread];
        self.submip_record = vec![r; num_thread];
    }

    /// clear
    pub fn clear(&mut self) {
        *self = Profiling::new();
    }

    fn num_thread(&self) -> usize {
        if self.multi_threaded {
            super::lp_handle::num_threads().max(1) as usize
        } else {
            1
        }
    }

    pub fn my_thread(&self) -> usize {
        if self.multi_threaded {
            thread_num()
        } else {
            0
        }
    }

    fn time(&self) -> f64 {
        // SAFETY: the Highs object's timer outlives its profiling
        unsafe { (*self.timer).read(0) }
    }

    /// The record of this thread: the sub-MIP's when `submip` (None: as
    /// the thread's flag says)
    ///
    /// # Safety
    /// Only this thread writes its records
    unsafe fn rec(p: *mut Profiling, submip: Option<bool>) -> *mut Record {
        let t = (*p).my_thread();
        let sub = submip.unwrap_or_else(|| *(*p).submip.as_ptr().add(t));
        if sub {
            (*p).submip_record.as_mut_ptr().add(t)
        } else {
            (*p).record.as_mut_ptr().add(t)
        }
    }

    /// # Safety (all the functions on a `*mut Profiling`)
    /// `p` a live Profiling, written concurrently only at the calling
    /// thread's index
    pub unsafe fn set_submip(p: *mut Profiling, submip: bool) {
        let t = (*p).my_thread();
        *(*p).submip.as_mut_ptr().add(t) = submip;
    }

    pub unsafe fn is_submip(p: *mut Profiling) -> bool {
        let t = (*p).my_thread();
        *(*p).submip.as_ptr().add(t)
    }

    pub unsafe fn start(p: *mut Profiling, clock: usize, restart: bool) {
        if clock >= (*p).num_clock || (Self::is_submip(p) && clock >= TO_SUB_SOLVER) {
            return;
        }
        let r = &mut *Self::rec(p, None);
        let time_start = (*p).time();
        if clock == 86 {
            crate::io::log::c_stdout(
                format!(
                    "HighsProfiling::start SubMipSolve on thread {:2} with submip = {}\n",
                    (*p).my_thread(),
                    if Self::is_submip(p) { "T" } else { "F" }
                )
                .as_bytes(),
            );
        }
        let running = r.start_time[clock].is_sign_negative();
        if running && clock != SUB_SOLVER_HIPO_AC && clock != SUB_SOLVER_IPX_AC {
            crate::io::log::c_stdout(
                format!(
                    "HighsProfiling: clock running for thread {} when starting clock {} and subMip = {}\n",
                    (*p).my_thread(),
                    clock,
                    if Self::is_submip(p) { "T" } else { "F" }
                )
                .as_bytes(),
            );
        }
        r.start_time[clock] = -time_start;
        if restart {
            r.num_call[clock] -= 1;
        }
    }

    pub unsafe fn stop(p: *mut Profiling, clock: usize) {
        if clock >= (*p).num_clock || (Self::is_submip(p) && clock >= TO_SUB_SOLVER) {
            return;
        }
        let r = &mut *Self::rec(p, None);
        let time_stop = (*p).time();
        let time_start = r.start_time[clock];
        if !time_start.is_sign_negative() {
            crate::io::log::c_stdout(
                format!(
                    "HighsProfiling: clock not running for thread {} when stopping clock {} \n",
                    (*p).my_thread(),
                    clock
                )
                .as_bytes(),
            );
        } else {
            r.num_call[clock] += 1;
            r.run_time[clock] += time_stop + time_start;
        }
        r.start_time[clock] = time_stop;
    }

    /// The record of `record`: -1 the MIP's, 1 the sub-MIP's, 0 as the
    /// thread's flag says
    unsafe fn pick(p: *mut Profiling, record: i64) -> *mut Record {
        Self::rec(p, if record == 0 { None } else { Some(record > 0) })
    }

    pub unsafe fn running(p: *mut Profiling, clock: usize, record: i64) -> bool {
        if clock >= (*p).num_clock {
            return false;
        }
        let r = &*Self::pick(p, record);
        r.start_time[clock].is_sign_negative()
    }

    pub unsafe fn read(p: *mut Profiling, clock: usize, record: i64) -> f64 {
        if clock >= (*p).num_clock {
            return -f64::INFINITY;
        }
        let r = &*Self::pick(p, record);
        let current = if Self::running(p, clock, record) { r.start_time[clock] + (*p).time() } else { 0.0 };
        r.run_time[clock] + current
    }

    pub unsafe fn num_call(p: *mut Profiling, clock: usize, record: i64) -> i32 {
        if clock >= (*p).num_clock {
            return -i32::MAX;
        }
        let r = &*Self::pick(p, record);
        r.num_call[clock] + Self::running(p, clock, record) as i32
    }
}

/// The sub-solver clocks' names (kFromSubSolver to kToSubSolver)
const SUB_SOLVER_NAMES: [&str; TO_SUB_SOLVER - SUB_SOLVER_MIP] = [
    "MIP",
    "Du simplex (basis)",
    "Du simplex (no basis)",
    "Pr simplex (basis)",
    "Pr simplex (no basis)",
    "HiPO",
    "IPX",
    "HiPO (AC)",
    "IPX (AC)",
    "PDLP",
    "QP ASM",
    "Sub-MIP",
];

impl Profiling {
    /// Highs::reportProfiling: the sub-solver times by thread (only with
    /// sub-solver profiling, log_dev_level > 0)
    pub fn report(&self, log: &super::Log) {
        use super::LogType::Info;
        use crate::log_user;
        use crate::util::printf::sprintf;
        if !self.sub_solver {
            return;
        }
        const SUB_MIP: usize = TO_SUB_SOLVER - 1;
        let num_thread = self.num_thread().min(self.record.len());
        let (mut mip_time, mut max_submip_time) = (0.0f64, 0.0f64);
        for t in 0..num_thread {
            mip_time = mip_time.max(self.record[t].run_time[SUB_SOLVER_MIP]);
            max_submip_time = max_submip_time.max(self.record[t].run_time[SUB_MIP]);
        }
        let used_thread: Vec<usize> = (0..num_thread)
            .filter(|&t| {
                (SUB_SOLVER_MIP..TO_SUB_SOLVER)
                    .any(|i| self.record[t].num_call[i] != 0 || self.submip_record[t].num_call[i] != 0)
            })
            .collect();
        let name = |i: usize| SUB_SOLVER_NAMES[i - SUB_SOLVER_MIP];
        let mut used = [[false; TO_SUB_SOLVER]; 2];
        let to_k = if max_submip_time > 0.0 { 2 } else { 1 };
        let mut sum_sum = 0.0;
        for k in 0..to_k {
            if k == 0 {
                log_user!(log, Info, "\nMIP sub-solver profiling: number of threads used = %d\n", used_thread.len() as i32);
            } else {
                log_user!(log, Info, "\nSub-MIP sub-solver profiling\n");
            }
            let records = if k == 0 { &self.record } else { &self.submip_record };
            for &t in &used_thread {
                let ideal_time = if k == 0 { mip_time } else { self.record[t].run_time[SUB_MIP] };
                if ideal_time <= 0.0 {
                    continue;
                }
                let r = &records[t];
                let mut s = sprintf("\nThread %d\nSolver                    Calls    Time       Time/call", &[(t as i32).into()]);
                s.push_str(if k == 0 { "      MIP%" } else { "  Sub-MIP%" });
                log_user!(log, Info, "%s\n", s.as_str());
                let mut sum = 0.0;
                for i in SUB_SOLVER_MIP..TO_SUB_SOLVER {
                    if r.num_call[i] == 0 {
                        continue;
                    }
                    used[k][i] = true;
                    let mut s = sprintf(
                        "%-21s %9d %11.4e %11.4e",
                        &[name(i).into(), r.num_call[i].into(), r.run_time[i].into(), (r.run_time[i] / r.num_call[i] as f64).into()],
                    );
                    if i != SUB_SOLVER_MIP {
                        sum += r.run_time[i];
                        s.push_str(&sprintf("     %5.1f", &[(1e2 * r.run_time[i] / ideal_time).into()]));
                    }
                    log_user!(log, Info, "%s\n", s.as_str());
                }
                sum_sum += sum;
                if sum > 0.0 {
                    log_user!(log, Info, "TOTAL                           %11.4e                 %5.1f\n", sum, 1e2 * sum / ideal_time);
                }
            }
        }
        if mip_time <= 0.0 || sum_sum <= 0.0 {
            return;
        }
        let hrule = || {
            let s = format!("====================={}", "======".repeat(used_thread.len()));
            log_user!(log, Info, "%s\n", s.as_str());
        };
        log_user!(log, Info, "\nPercent (sub-)MIP time by thread\n");
        for k in 0..to_k {
            let mut s = String::from(if k == 0 { "\nMIP sub-solver       " } else { "\nSub-MIP sub-solver   " });
            if k == 1 && max_submip_time <= 0.0 {
                continue;
            }
            for &t in &used_thread {
                s.push_str(&sprintf("%6d", &[(t as i32).into()]));
            }
            log_user!(log, Info, "%s\n", s.as_str());
            let records = if k == 0 { &self.record } else { &self.submip_record };
            let mut total = vec![0.0; used_thread.len()];
            for i in SUB_SOLVER_MIP + 1..TO_SUB_SOLVER {
                if !used[k][i] {
                    continue;
                }
                let mut s = sprintf("%-21s", &[name(i).into()]);
                for (x, &t) in used_thread.iter().enumerate() {
                    let ideal_time = if k == 0 { mip_time } else { self.record[t].run_time[SUB_MIP] };
                    let (num_call, run_time) = (records[t].num_call[i], records[t].run_time[i]);
                    if num_call != 0 && ideal_time > 0.0 {
                        let pct = 1e2 * run_time / ideal_time;
                        total[x] += pct;
                        s.push_str(&sprintf(" %5.1f", &[pct.into()]));
                    } else {
                        s.push_str("      ");
                    }
                }
                log_user!(log, Info, "%s\n", s.as_str());
            }
            hrule();
            let mut s = String::from("Total                ");
            for &p in &total {
                if p != 0.0 {
                    s.push_str(&sprintf(" %5.1f", &[p.into()]));
                } else {
                    s.push_str("      ");
                }
            }
            log_user!(log, Info, "%s\n", s.as_str());
            hrule();
        }
    }
}

impl Default for Profiling {
    fn default() -> Self {
        Profiling::new()
    }
}

/// The HighsFns profiling operations (mip/host/mod.rs prof) and the
/// simplex and PDLP steps (lp_handle.rs ProfilingFn) on a Profiling
pub mod fns {
    use super::*;
    use crate::mip::host::prof;
    use std::ffi::c_void;

    /// HighsMipHost.cpp recordType: 0 the MIP record, else the sub-MIP's
    fn record(arg: i64) -> i64 {
        if arg == prof::MIP_RECORD {
            -1
        } else {
            1
        }
    }

    /// HighsFns::profiling
    ///
    /// # Safety
    /// `p` a live Profiling
    pub unsafe extern "C" fn mip(p: *mut c_void, code: i32, clock: i32, arg: i64) -> f64 {
        let p = p as *mut Profiling;
        let k = MIP_CLOCKS[clock.max(0) as usize];
        match code {
            prof::START => Profiling::start(p, k, false),
            prof::STOP => Profiling::stop(p, k),
            prof::RUNNING => return Profiling::running(p, k, 0) as i32 as f64,
            prof::READ => return Profiling::read(p, k, record(arg)),
            prof::NUM_CALL => return Profiling::num_call(p, k, record(arg)) as f64,
            prof::MIP => return (*p).mip as i32 as f64,
            prof::IS_SUBMIP => return Profiling::is_submip(p) as i32 as f64,
            prof::SET_SUBMIP => Profiling::set_submip(p, arg != 0),
            prof::SUB_SOLVER => return (*p).sub_solver as i32 as f64,
            prof::MY_THREAD => return (*p).my_thread() as f64,
            prof::RESTART => Profiling::start(p, k, true),
            // solveCall: only its consistency check prints
            _ => {
                if (*p).num_clock > TO_PRESOLVE_SOLVE_POSTSOLVE && Profiling::is_submip(p) != (clock != 0) {
                    const MODEL: [&str; 5] = ["LP0", "LP1", "LP2", "LP3", "MIP"];
                    let sub = |b: bool| if b { "sub-" } else { "" };
                    crate::io::log::c_stdout(
                        format!(
                            "Solving {:>3} for {:>4}MIP on thread {} with isSubMip() = {:>4}MIP\n",
                            MODEL[arg.clamp(0, 4) as usize],
                            sub(clock != 0),
                            (*p).my_thread(),
                            sub(Profiling::is_submip(p))
                        )
                        .as_bytes(),
                    );
                }
            }
        }
        0.0
    }

    /// rsSimplexProfiling: the simplex solve's clock (code 1 start, arg
    /// whether there is a basis; 2 stop), PDLP's (3 start, 4 stop)
    ///
    /// # Safety
    /// `p` null or a live Profiling
    pub unsafe extern "C" fn simplex(p: *mut c_void, code: i32, simplex_strategy: i32, arg: i64) {
        let p = p as *mut Profiling;
        if p.is_null() {
            return;
        }
        // kSimplexStrategyPrimal
        let primal = simplex_strategy == 4;
        match code {
            1 => {
                let k = match (primal, arg != 0) {
                    (true, true) => SUB_SOLVER_PR_SIMPLEX_BASIS,
                    (true, false) => SUB_SOLVER_PR_SIMPLEX_NO_BASIS,
                    (false, true) => SUB_SOLVER_DU_SIMPLEX_BASIS,
                    (false, false) => SUB_SOLVER_DU_SIMPLEX_NO_BASIS,
                };
                Profiling::start(p, k, false)
            }
            2 => {
                if (*p).sub_solver {
                    let r = &*Profiling::rec(p, None);
                    let mut k = usize::MAX;
                    for c in [
                        SUB_SOLVER_DU_SIMPLEX_BASIS,
                        SUB_SOLVER_DU_SIMPLEX_NO_BASIS,
                        SUB_SOLVER_PR_SIMPLEX_BASIS,
                        SUB_SOLVER_PR_SIMPLEX_NO_BASIS,
                    ] {
                        if r.start_time[c].is_sign_negative() {
                            k = c;
                        }
                    }
                    Profiling::stop(p, k);
                }
            }
            3 => Profiling::start(p, SUB_SOLVER_PDLP, false),
            _ => Profiling::stop(p, SUB_SOLVER_PDLP),
        }
    }
}
