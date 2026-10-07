//! The top level of HiGHS (highs/lp_data): LP validation and scaling
//! (lp_utils.rs). The C++ `Highs` class keeps owning its data (its API
//! hands out references to it); the Rust works on views of it
//! (ffi.rs: `RsMut` arrays, `CLp`) and logs through C++'s highsLogUser
//! (`Log`), so messages are those of the C++ formats, byte for byte.

pub mod ffi;
pub mod info;
pub mod lp_utils;
pub mod options;
pub mod solution;

use crate::util::printf::{sprintf, Arg};

/// HighsStatus
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Error = -1,
    Ok = 0,
    Warning = 1,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "OK",
            Status::Warning => "Warning",
            Status::Error => "Error",
        }
    }
    /// worseStatus
    pub fn worse(self, other: Status) -> Status {
        if self == Status::Error || other == Status::Error {
            Status::Error
        } else if self == Status::Warning || other == Status::Warning {
            Status::Warning
        } else {
            Status::Ok
        }
    }
}

/// HighsLogType
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum LogType {
    Info = 1,
    Detailed = 2,
    Verbose = 3,
    Warning = 4,
    Error = 5,
}

pub type LogFn = unsafe extern "C" fn(*const std::ffi::c_void, i32, i32, *const u8, usize);

/// HighsLogOptions, as a handle and the C++ function that logs a
/// formatted message through highsLogUser (dev = 0) or highsLogDev
/// (dev = 1)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Log {
    pub opts: *const std::ffi::c_void,
    pub log: Option<LogFn>,
}

impl Log {
    /// A log that drops everything (tests)
    pub fn none() -> Log {
        Log { opts: std::ptr::null(), log: None }
    }
    fn emit(&self, dev: i32, t: LogType, msg: &str) {
        if let Some(f) = self.log {
            // SAFETY: the C++ side gave a function and its handle
            unsafe { f(self.opts, dev, t as i32, msg.as_ptr(), msg.len()) }
        }
    }
    /// highsLogUser(log_options, t, "%s", msg)
    pub fn user(&self, t: LogType, msg: &str) {
        self.emit(0, t, msg)
    }
    /// highsLogDev(log_options, t, "%s", msg)
    pub fn dev(&self, t: LogType, msg: &str) {
        self.emit(1, t, msg)
    }
    /// highsLogUser(log_options, t, fmt, args...)
    pub fn userf(&self, t: LogType, fmt: &str, args: &[Arg]) {
        if self.log.is_some() {
            self.user(t, &sprintf(fmt, args))
        }
    }
    /// highsLogDev(log_options, t, fmt, args...)
    pub fn devf(&self, t: LogType, fmt: &str, args: &[Arg]) {
        if self.log.is_some() {
            self.dev(t, &sprintf(fmt, args))
        }
    }
    /// interpretCallStatus
    pub fn interpret(&self, call: Status, from: Status, message: &str) -> Status {
        if call != Status::Ok {
            self.devf(
                LogType::Warning,
                "%s return of HighsStatus::%s\n",
                &[message.into(), call.as_str().into()],
            );
        }
        call.worse(from)
    }
}

/// `log_user!(log, type, fmt, args...)`
#[macro_export]
macro_rules! log_user {
    ($log:expr, $t:expr, $fmt:expr) => { $log.user($t, $fmt) };
    ($log:expr, $t:expr, $fmt:expr, $($a:expr),+ $(,)?) => {
        $log.userf($t, $fmt, &[$($crate::util::printf::Arg::from($a)),+])
    };
}

/// `log_dev!(log, type, fmt, args...)`
#[macro_export]
macro_rules! log_dev {
    ($log:expr, $t:expr, $fmt:expr) => { $log.dev($t, $fmt) };
    ($log:expr, $t:expr, $fmt:expr, $($a:expr),+ $(,)?) => {
        $log.devf($t, $fmt, &[$($crate::util::printf::Arg::from($a)),+])
    };
}

pub const INF: f64 = f64::INFINITY;

/// HighsVarType
pub mod var_type {
    pub const CONTINUOUS: u8 = 0;
    pub const INTEGER: u8 = 1;
    pub const SEMI_CONTINUOUS: u8 = 2;
    pub const SEMI_INTEGER: u8 = 3;
    pub const IMPLICIT_INTEGER: u8 = 4;
}

/// MatrixFormat
pub mod matrix_format {
    pub const COLWISE: i32 = 1;
    pub const ROWWISE: i32 = 2;
    pub const ROWWISE_PARTITIONED: i32 = 3;
}
