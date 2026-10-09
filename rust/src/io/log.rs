//! The logging sink of highsLogUser / highsLogDev (io/HighsIO.cpp): which
//! messages print, the "WARNING: " / "ERROR:   " prefix, writing to the
//! log file and to C's stdout (fflush after each), and the log callbacks
//! with the 1024-byte buffer of the C++. The C++ variadic functions only
//! format the message (vsnprintf) and hand it here; Rust's `Log` calls
//! `highs_rs_log` directly.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A C `FILE`
#[repr(C)]
pub struct File {
    _private: [u8; 0],
}

extern "C" {
    #[cfg_attr(target_os = "macos", link_name = "__stdoutp")]
    static stdout: *mut File;
    fn fwrite(p: *const c_void, size: usize, n: usize, f: *mut File) -> usize;
    fn fflush(f: *mut File) -> i32;
}

type ActiveFn = unsafe extern "C" fn(*const c_void) -> bool;
type CallFn = unsafe extern "C" fn(*const c_void, i32, *const c_char);

/// The C++ functions on HighsLogOptions' std::function user_callback:
/// whether it is set and active, and calling it (kCallbackLogging, with
/// data_out.log_type = t); registered by HighsIO.cpp at load time (the
/// library and its tests link no C++)
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static CALL: AtomicUsize = AtomicUsize::new(0);

/// Registers the C++ functions of the std::function user_callback
#[no_mangle]
pub extern "C" fn highs_rs_log_init(active: ActiveFn, call: CallFn) {
    ACTIVE.store(active as usize, Ordering::Relaxed);
    CALL.store(call as usize, Ordering::Relaxed);
}

unsafe fn highs_log_user_callback_active(opts: *const c_void) -> bool {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => false,
        // SAFETY: stored from an ActiveFn by highs_rs_log_init
        f => std::mem::transmute::<usize, ActiveFn>(f)(opts),
    }
}

unsafe fn highs_log_user_callback(opts: *const c_void, t: i32, msg: *const c_char) {
    let f = CALL.load(Ordering::Relaxed);
    if f != 0 {
        // SAFETY: stored from a CallFn by highs_rs_log_init
        std::mem::transmute::<usize, CallFn>(f)(opts, t, msg)
    }
}

/// The fields of HighsLogOptions before its std::function (offsets
/// checked by static_assert in HighsIO.cpp)
#[repr(C)]
struct LogOptionsHead {
    log_stream: *mut File,
    output_flag: *const bool,
    log_to_console: *const bool,
    log_dev_level: *const i32,
    user_log_callback: Option<unsafe extern "C" fn(i32, *const c_char, *mut c_void)>,
    user_log_callback_data: *mut c_void,
}

const WARNING: i32 = 4;
const ERROR: i32 = 5;
const DETAILED: i32 = 2;
const VERBOSE: i32 = 3;
/// kIoBufferSize
const BUFFER: usize = 1024;

struct Targets {
    file: bool,
    console: bool,
    callback: bool,
}

/// Where a message goes, or None if it does not print
unsafe fn targets(opts: *const c_void, dev: bool, t: i32) -> Option<Targets> {
    let o = &*(opts as *const LogOptionsHead);
    if !*o.output_flag {
        return None;
    }
    if dev && *o.log_dev_level == 0 {
        return None;
    }
    let file = !o.log_stream.is_null() && o.log_stream != stdout;
    let console = *o.log_to_console;
    let callback = o.user_log_callback.is_some() || highs_log_user_callback_active(opts);
    if !file && !console && !callback {
        return None;
    }
    if dev && ((t == DETAILED && *o.log_dev_level < DETAILED) || (t == VERBOSE && *o.log_dev_level < VERBOSE)) {
        return None;
    }
    Some(Targets { file, console, callback })
}

/// Whether highsLogUser (dev = 0) or highsLogDev (dev = 1) prints a
/// message of type t, so the C++ formats it only then
///
/// # Safety
/// `opts` is a HighsLogOptions whose pointers are set
#[no_mangle]
pub unsafe extern "C" fn highs_rs_log_prints(opts: *const c_void, dev: i32, t: i32) -> bool {
    targets(opts, dev != 0, t).is_some()
}

/// fwrite to C's stdout (printf's stream), unflushed as printf
pub fn c_stdout(s: &[u8]) {
    // SAFETY: C's stdout is open for the program's life
    unsafe { put(stdout, s) }
}

/// printf("%s", s); fflush(stdout)
pub fn c_stdout_flush(s: &[u8]) {
    // SAFETY: as c_stdout
    unsafe {
        put(stdout, s);
        fflush(stdout);
    }
}

unsafe fn put(f: *mut File, s: &[u8]) {
    if !s.is_empty() {
        fwrite(s.as_ptr() as *const c_void, 1, s.len(), f);
    }
}

/// highsLogUser (dev = 0) or highsLogDev (dev = 1) of the formatted
/// message `msg` (cut at a NUL, as "%s" would)
///
/// # Safety
/// `opts` is a HighsLogOptions whose pointers are set; `msg` holds `len`
/// bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_log(opts: *const c_void, dev: i32, t: i32, msg: *const u8, len: usize) {
    let dev = dev != 0;
    let Some(to) = targets(opts, dev, t) else { return };
    let o = &*(opts as *const LogOptionsHead);
    let mut msg = if len == 0 { &[][..] } else { std::slice::from_raw_parts(msg, len) };
    if let Some(n) = msg.iter().position(|&c| c == 0) {
        msg = &msg[..n];
    }
    let prefix: &[u8] = match (dev, t) {
        (false, WARNING) => b"WARNING: ",
        (false, ERROR) => b"ERROR:   ",
        _ => b"",
    };
    if to.file {
        put(o.log_stream, prefix);
        put(o.log_stream, msg);
        fflush(o.log_stream);
    }
    if to.console {
        put(stdout, prefix);
        put(stdout, msg);
        fflush(stdout);
    }
    if to.callback {
        // The C++ buffer: the prefix, then what vsnprintf fits, NUL ended
        let mut buf = Vec::with_capacity(BUFFER);
        buf.extend_from_slice(prefix);
        let room = BUFFER - 1 - buf.len();
        buf.extend_from_slice(&msg[..msg.len().min(room)]);
        buf.push(0);
        let p = buf.as_ptr() as *const c_char;
        if let Some(cb) = o.user_log_callback {
            cb(t, p, o.user_log_callback_data);
            if !dev && highs_log_user_callback_active(opts) {
                highs_log_user_callback(opts, t, p);
            }
        } else if highs_log_user_callback_active(opts) {
            highs_log_user_callback(opts, t, p);
        }
    }
}
