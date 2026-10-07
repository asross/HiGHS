//! HighsOptions.cpp and LoadOptions.cpp: finding, checking, setting,
//! getting, resetting, passing and reporting options, and reading an
//! options file. The option records (names, descriptions, bounds,
//! defaults) and the values stay in the C++ HighsOptions (its fields are
//! public API); each call gets a table of views of the records
//! (`COptionRecord`, highs/lp_data/HighsOptionsRust.cpp), and Rust writes
//! bool, int and double values through their pointers and string values
//! through a C++ callback.
//!
//! Strings are handled as bytes, as std::string is; only the formatting of
//! messages and reports converts them (lossily, for non-UTF-8 bytes) to
//! `str`.

use super::{Log, LogType};
use crate::{log_dev, log_user, sprintf};
use std::borrow::Cow;
use std::ffi::c_void;
use std::slice::from_raw_parts;

// OptionStatus
pub const OK: i32 = 0;
pub const UNKNOWN_OPTION: i32 = 1;
pub const ILLEGAL_VALUE: i32 = 2;

// HighsOptionType
pub const BOOL: i32 = 0;
pub const INT: i32 = 1;
pub const DOUBLE: i32 = 2;
pub const STRING: i32 = 3;

// HighsFileType
pub const FILE_MINIMAL: i32 = 0;
pub const FILE_FULL: i32 = 1;
pub const FILE_MD: i32 = 4;

const OFF: &[u8] = b"off";
const CHOOSE: &[u8] = b"choose";
const ON: &[u8] = b"on";
const SIMPLEX: &[u8] = b"simplex";
const IPM: &[u8] = b"ipm";
const HIPO: &[u8] = b"hipo";
const IPX: &[u8] = b"ipx";
const PDLP: &[u8] = b"pdlp";
const QPASM: &[u8] = b"qpasm";
const HIPDLP: &[u8] = b"hipdlp";
const LOG_FILE: &[u8] = b"log_file";
const MODEL_FILE: &[u8] = b"model_file";
const HIGHS_RUN_LOG_FILE: &[u8] = b"Highs.log";
/// kIoBufferSize: highsFormatToString keeps at most this less one byte
const IO_BUFFER_SIZE: usize = 1024;

/// A C++ string: data() and size()
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsStr {
    pub ptr: *const u8,
    pub len: usize,
}

impl RsStr {
    /// # Safety
    /// `ptr` valid for `len` bytes while the slice lives
    pub unsafe fn get<'a>(&self) -> &'a [u8] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            from_raw_parts(self.ptr, self.len)
        }
    }
    pub fn of(s: &[u8]) -> RsStr {
        RsStr { ptr: s.as_ptr(), len: s.len() }
    }
}

/// An OptionRecord (Bool, Int, Double or String): `value` points to the
/// option's field of HighsOptions; `str_value` is the string value when
/// the table was made (read only before any write of that string)
#[repr(C)]
pub struct COptionRecord {
    pub type_: i32,
    pub advanced: bool,
    pub name: RsStr,
    pub description: RsStr,
    pub value: *mut c_void,
    pub str_value: RsStr,
    pub str_default: RsStr,
    pub bool_default: bool,
    pub int_lower: i32,
    pub int_default: i32,
    pub int_upper: i32,
    pub dbl_lower: f64,
    pub dbl_default: f64,
    pub dbl_upper: f64,
}

pub type SetStringFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);
pub type OpenLogFileFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);
pub type HipoAvailableFn = unsafe extern "C" fn() -> bool;
pub type HipoUnavailableFn = unsafe extern "C" fn(*const Log, *const u8, usize);
pub type WriteFn = unsafe extern "C" fn(*mut c_void, *const u8, usize);

/// What the option functions call in C++: `log` is the report log
/// options; `ctx` (the HighsLogOptions to change and the record vector)
/// goes to `open_log_file` (highsOpenLogFile)
#[repr(C)]
pub struct COptionHost {
    pub log: Log,
    pub ctx: *mut c_void,
    pub set_string: Option<SetStringFn>,
    pub open_log_file: Option<OpenLogFileFn>,
    pub hipo_available: Option<HipoAvailableFn>,
    pub hipo_unavailable: Option<HipoUnavailableFn>,
    pub write: Option<WriteFn>,
}

impl COptionHost {
    fn hipo_available(&self) -> bool {
        // SAFETY: a C++ function without arguments
        self.hipo_available.map_or(false, |f| unsafe { f() })
    }
    fn hipo_unavailable(&self, name: &[u8]) {
        if let Some(f) = self.hipo_unavailable {
            // SAFETY: the log handle and a byte string
            unsafe { f(&self.log, name.as_ptr(), name.len()) }
        }
    }
    fn write(&self, file: *mut c_void, s: &str) {
        if let Some(f) = self.write {
            // SAFETY: the C++ FILE* and a byte string
            unsafe { f(file, s.as_ptr(), s.len()) }
        }
    }
}

/// A byte string for formatting (`%s`)
pub fn st(b: &[u8]) -> Cow<'_, str> {
    // ponytail: non-UTF-8 bytes print as U+FFFD; byte-exact output would
    // need a byte-string printf
    String::from_utf8_lossy(b)
}

impl COptionRecord {
    /// # Safety (all accessors)
    /// The record's strings and value pointer must be valid
    pub fn name(&self) -> &[u8] {
        // SAFETY: the C++ record outlives the call
        unsafe { self.name.get() }
    }
    fn description(&self) -> &[u8] {
        // SAFETY: as name
        unsafe { self.description.get() }
    }
    fn str_value(&self) -> &[u8] {
        // SAFETY: as name
        unsafe { self.str_value.get() }
    }
    fn str_default(&self) -> &[u8] {
        // SAFETY: as name
        unsafe { self.str_default.get() }
    }
    fn get_bool(&self) -> bool {
        // SAFETY: a bool option's value points to its bool field
        unsafe { *(self.value as *const bool) }
    }
    fn set_bool(&self, v: bool) {
        // SAFETY: as get_bool
        unsafe { *(self.value as *mut bool) = v }
    }
    fn get_int(&self) -> i32 {
        // SAFETY: an int option's value points to its HighsInt field
        unsafe { *(self.value as *const i32) }
    }
    fn set_int(&self, v: i32) {
        // SAFETY: as get_int
        unsafe { *(self.value as *mut i32) = v }
    }
    fn get_double(&self) -> f64 {
        // SAFETY: a double option's value points to its double field
        unsafe { *(self.value as *const f64) }
    }
    fn set_double(&self, v: f64) {
        // SAFETY: as get_double
        unsafe { *(self.value as *mut f64) = v }
    }
    fn set_string(&self, host: &COptionHost, v: &[u8]) {
        if let Some(f) = host.set_string {
            // SAFETY: a string option's value points to its std::string
            unsafe { f(self.value, v.as_ptr(), v.len()) }
        }
    }
}

/// optionEntryTypeToString
fn type_name(t: i32) -> &'static str {
    match t {
        BOOL => "bool",
        INT => "HighsInt",
        DOUBLE => "double",
        _ => "string",
    }
}

/// highsBoolToString (field width 2)
fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

/// trim(s, chars): strip leading and trailing bytes in `chars`
pub fn trim<'a>(s: &'a [u8], chars: &[u8]) -> &'a [u8] {
    let Some(first) = s.iter().position(|c| !chars.contains(c)) else {
        return &s[..0];
    };
    let last = s.iter().rposition(|c| !chars.contains(c)).unwrap();
    &s[first..=last]
}

/// optionOffChooseOnOk
pub fn off_choose_on_ok(log: &Log, name: &[u8], value: &[u8]) -> bool {
    if value == OFF || value == CHOOSE || value == ON {
        return true;
    }
    log_user!(
        log,
        LogType::Error,
        "Value \"%s\" for %s option is not one of \"%s\", \"%s\" or \"%s\"\n",
        &*st(value),
        &*st(name),
        "off",
        "choose",
        "on"
    );
    false
}

/// optionOffOnOk
pub fn off_on_ok(log: &Log, name: &[u8], value: &[u8]) -> bool {
    if value == OFF || value == ON {
        return true;
    }
    log_user!(
        log,
        LogType::Error,
        "Value \"%s\" for %s option is not one of \"%s\" or \"%s\"\n",
        &*st(value),
        &*st(name),
        "off",
        "on"
    );
    false
}

fn hipo_prefix(host: &COptionHost) -> &'static str {
    if host.hipo_available() {
        "hipo\", \""
    } else {
        ""
    }
}

/// optionSolverOk
pub fn solver_ok(host: &COptionHost, value: &[u8]) -> bool {
    if value == CHOOSE
        || value == SIMPLEX
        || value == IPM
        || (value == HIPO && host.hipo_available())
        || value == IPX
        || value == PDLP
        || value == QPASM
        || value == HIPDLP
    {
        true
    } else if value == HIPO && !host.hipo_available() {
        host.hipo_unavailable(b"solver");
        false
    } else {
        log_user!(
            host.log,
            LogType::Warning,
            "Value \"%s\" for LP solver option (\"%s\") is not one of %s\"%s\", \"%s\", \"%s\", \"%s\" or \"%s\"\n",
            &*st(value),
            "solver",
            hipo_prefix(host),
            "choose",
            "simplex",
            "ipm",
            "ipx",
            "pdlp",
            "qpasm",
            "hipdlp"
        );
        false
    }
}

/// optionMipLpSolverOk
pub fn mip_lp_solver_ok(host: &COptionHost, value: &[u8]) -> bool {
    if value == CHOOSE || value == SIMPLEX || value == IPM || (value == HIPO && host.hipo_available()) || value == IPX
    {
        true
    } else if value == HIPO && !host.hipo_available() {
        host.hipo_unavailable(b"mip_lp_solver");
        false
    } else {
        log_user!(
            host.log,
            LogType::Error,
            "Value \"%s\" for MIP LP solver option (\"%s\") is not one of %s\"%s\", \"%s\", \"%s\" or \"%s\"\n",
            &*st(value),
            "mip_lp_solver",
            hipo_prefix(host),
            "choose",
            "simplex",
            "ipm",
            "ipx"
        );
        false
    }
}

/// optionMipIpmSolverOk
pub fn mip_ipm_solver_ok(host: &COptionHost, value: &[u8]) -> bool {
    if value == CHOOSE || value == IPM || (value == HIPO && host.hipo_available()) || value == IPX {
        true
    } else if value == HIPO && !host.hipo_available() {
        host.hipo_unavailable(b"mip_ipm_solver");
        false
    } else {
        log_user!(
            host.log,
            LogType::Error,
            "Value \"%s\" for MIP IPM solver (\"%s\") option is not one of %s\"%s\", \"%s\" or \"%s\"\n",
            &*st(value),
            "mip_ipm_solver",
            hipo_prefix(host),
            "choose",
            "ipm",
            "ipx"
        );
        false
    }
}

/// optionHipoParallelTypeOk
pub fn hipo_parallel_type_ok(log: &Log, value: &[u8]) -> bool {
    if value == b"node" || value == b"tree" || value == b"both" {
        return true;
    }
    log_user!(
        log,
        LogType::Error,
        "Value \"%s\" for %s option is not one of \"%s\", \"%s\" or \"%s\"\n",
        &*st(value),
        "hipo_parallel_type",
        "tree",
        "node",
        "both"
    );
    false
}

/// optionHipoSystemOk
pub fn hipo_system_ok(log: &Log, value: &[u8]) -> bool {
    if value == b"normaleq" || value == b"augmented" || value == CHOOSE {
        return true;
    }
    log_user!(
        log,
        LogType::Error,
        "Value \"%s\" for %s option is not one of \"%s\", \"%s\" or \"%s\"\n",
        &*st(value),
        "hipo_system",
        "normaleq",
        "augmented",
        "choose"
    );
    false
}

/// optionHipoOrderingOk
pub fn hipo_ordering_ok(log: &Log, value: &[u8]) -> bool {
    if value == b"amd" || value == b"metis" || value == b"rcm" || value == CHOOSE {
        return true;
    }
    log_user!(
        log,
        LogType::Error,
        "Value \"%s\" for %s option is not one of \"%s\", \"%s\", \"%s\" or \"%s\"\n",
        &*st(value),
        "hipo_ordering",
        "amd",
        "metis",
        "rcm",
        "choose"
    );
    false
}

/// boolFromString
pub fn bool_from_string(value: &[u8]) -> Option<bool> {
    let v = value.to_ascii_lowercase();
    match &v[..] {
        b"t" | b"true" | b"1" | b"on" => Some(true),
        b"f" | b"false" | b"0" | b"off" => Some(false),
        _ => None,
    }
}

/// getOptionIndex
pub fn option_index(log: &Log, name: &[u8], recs: &[COptionRecord]) -> Result<usize, i32> {
    if let Some(i) = recs.iter().position(|r| r.name() == name) {
        return Ok(i);
    }
    log_user!(log, LogType::Error, "getOptionIndex: Option \"%s\" is unknown\n", &*st(name));
    Err(UNKNOWN_OPTION)
}

/// checkOptions
pub fn check_options(log: &Log, recs: &[COptionRecord]) -> i32 {
    let mut error_found = false;
    for (index, r) in recs.iter().enumerate() {
        let name = r.name();
        for (check_index, c) in recs.iter().enumerate() {
            if check_index != index && c.name() == name {
                log_user!(
                    log,
                    LogType::Error,
                    "checkOptions: Option %d (\"%s\") has the same name as option %d \"%s\"\n",
                    index,
                    &*st(name),
                    check_index,
                    &*st(c.name())
                );
                error_found = true;
            }
        }
        if (r.type_ == INT || r.type_ == DOUBLE) && check_option(log, r) != OK {
            error_found = true;
        }
        for (check_index, c) in recs.iter().enumerate() {
            if check_index != index && c.type_ == r.type_ && c.value == r.value {
                log_user!(
                    log,
                    LogType::Error,
                    "checkOptions: Option %d (\"%s\") has the same value pointer as option %d (\"%s\")\n",
                    index,
                    &*st(name),
                    check_index,
                    &*st(c.name())
                );
                error_found = true;
            }
        }
    }
    if error_found {
        return ILLEGAL_VALUE;
    }
    log_user!(log, LogType::Info, "checkOptions: Options are OK\n");
    OK
}

/// checkOption (HighsInt and double records)
pub fn check_option(log: &Log, r: &COptionRecord) -> i32 {
    let name = st(r.name());
    if r.type_ == INT {
        let (lo, de, up) = (r.int_lower, r.int_default, r.int_upper);
        if lo > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has inconsistent bounds [%d, %d]\n",
                &*name,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
        if de < lo || de > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has default value %d inconsistent with bounds [%d, %d]\n",
                &*name,
                de,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
        let v = r.get_int();
        if v < lo || v > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has value %d inconsistent with bounds [%d, %d]\n",
                &*name,
                v,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
    } else {
        let (lo, de, up) = (r.dbl_lower, r.dbl_default, r.dbl_upper);
        if lo > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has inconsistent bounds [%g, %g]\n",
                &*name,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
        if de < lo || de > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has default value %g inconsistent with bounds [%g, %g]\n",
                &*name,
                de,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
        let v = r.get_double();
        if v < lo || v > up {
            log_user!(
                log,
                LogType::Error,
                "checkOption: Option \"%s\" has value %g inconsistent with bounds [%g, %g]\n",
                &*name,
                v,
                lo,
                up
            );
            return ILLEGAL_VALUE;
        }
    }
    OK
}

/// checkOptionValue (HighsInt)
pub fn check_int_value(log: &Log, r: &COptionRecord, value: i32) -> i32 {
    if value < r.int_lower {
        log_user!(
            log,
            LogType::Error,
            "checkOptionValue: Value %d for option \"%s\" is below lower bound of %d\n",
            value,
            &*st(r.name()),
            r.int_lower
        );
        return ILLEGAL_VALUE;
    } else if value > r.int_upper {
        log_user!(
            log,
            LogType::Error,
            "checkOptionValue: Value %d for option \"%s\" is above upper bound of %d\n",
            value,
            &*st(r.name()),
            r.int_upper
        );
        return ILLEGAL_VALUE;
    }
    OK
}

/// checkOptionValue (double)
pub fn check_double_value(log: &Log, r: &COptionRecord, value: f64) -> i32 {
    if value < r.dbl_lower {
        log_user!(
            log,
            LogType::Error,
            "checkOptionValue: Value %g for option \"%s\" is below lower bound of %g\n",
            value,
            &*st(r.name()),
            r.dbl_lower
        );
        return ILLEGAL_VALUE;
    } else if value > r.dbl_upper {
        log_user!(
            log,
            LogType::Error,
            "checkOptionValue: Value %g for option \"%s\" is above upper bound of %g\n",
            value,
            &*st(r.name()),
            r.dbl_upper
        );
        return ILLEGAL_VALUE;
    }
    OK
}

/// checkOptionValue (string): the values some options allow
pub fn check_string_value(host: &COptionHost, r: &COptionRecord, value: &[u8]) -> i32 {
    let log = &host.log;
    let name = r.name();
    let ok = match name {
        b"presolve" => off_choose_on_ok(log, name, value) || value == b"mip",
        b"solver" => solver_ok(host, value),
        b"mip_lp_solver" => mip_lp_solver_ok(host, value),
        b"mip_ipm_solver" => mip_ipm_solver_ok(host, value),
        b"parallel" | b"run_crossover" => off_choose_on_ok(log, name, value),
        b"ranging" => off_on_ok(log, name, value),
        b"hipo_parallel_type" => hipo_parallel_type_ok(log, value),
        b"hipo_system" => hipo_system_ok(log, value),
        b"hipo_ordering" => hipo_ordering_ok(log, value),
        _ => true,
    };
    if ok {
        OK
    } else {
        ILLEGAL_VALUE
    }
}

/// possibleLowerCaseOptionValue: lower case, except file names
pub fn possible_lower_case(name: &[u8], value: &mut Vec<u8>) {
    const FILES: [&[u8]; 11] = [
        b"model_file",
        b"read_basis_file",
        b"write_basis_file",
        b"options_file",
        b"solution_file",
        b"write_model_file",
        b"write_presolved_model_file",
        b"write_iis_model_file",
        b"read_solution_file",
        b"log_file",
        b"mip_improving_solution_file",
    ];
    if !FILES.contains(&name) {
        value.make_ascii_lowercase();
    }
}

/// setLocalOptionValue(OptionRecordInt&, value)
pub fn set_int_record(log: &Log, r: &COptionRecord, value: i32) -> i32 {
    let s = check_int_value(log, r, value);
    if s == OK {
        r.set_int(value);
    }
    s
}

/// setLocalOptionValue(OptionRecordDouble&, value)
pub fn set_double_record(log: &Log, r: &COptionRecord, value: f64) -> i32 {
    let s = check_double_value(log, r, value);
    if s == OK {
        r.set_double(value);
    }
    s
}

/// setLocalOptionValue(OptionRecordString&, value)
pub fn set_string_record(host: &COptionHost, r: &COptionRecord, value: &[u8]) -> i32 {
    let mut v = trim(value, b" ").to_vec();
    possible_lower_case(r.name(), &mut v);
    let s = check_string_value(host, r, &v);
    if s == OK {
        r.set_string(host, &v);
    }
    s
}

/// setLocalOptionValue(name, bool)
pub fn set_bool(log: &Log, name: &[u8], recs: &[COptionRecord], value: bool) -> i32 {
    let index = match option_index(log, name, recs) {
        Ok(i) => i,
        Err(s) => return s,
    };
    let r = &recs[index];
    if r.type_ != BOOL {
        log_user!(
            log,
            LogType::Error,
            "setLocalOptionValue: Option \"%s\" cannot be assigned a bool\n",
            &*st(name)
        );
        return ILLEGAL_VALUE;
    }
    r.set_bool(value);
    OK
}

/// setLocalOptionValue(name, HighsInt): a double option takes it as double
pub fn set_int(log: &Log, name: &[u8], recs: &[COptionRecord], value: i32) -> i32 {
    let index = match option_index(log, name, recs) {
        Ok(i) => i,
        Err(s) => return s,
    };
    let r = &recs[index];
    if r.type_ != INT {
        if r.type_ == DOUBLE {
            return set_double_record(log, r, value as f64);
        }
        log_user!(
            log,
            LogType::Error,
            "setLocalOptionValue: Option \"%s\" cannot be assigned an int\n",
            &*st(name)
        );
        return ILLEGAL_VALUE;
    }
    set_int_record(log, r, value)
}

/// setLocalOptionValue(name, double)
pub fn set_double(log: &Log, name: &[u8], recs: &[COptionRecord], value: f64) -> i32 {
    let index = match option_index(log, name, recs) {
        Ok(i) => i,
        Err(s) => return s,
    };
    let r = &recs[index];
    if r.type_ != DOUBLE {
        log_user!(
            log,
            LogType::Error,
            "setLocalOptionValue: Option \"%s\" cannot be assigned a double\n",
            &*st(name)
        );
        return ILLEGAL_VALUE;
    }
    set_double_record(log, r, value)
}

/// C's strtol on the longest prefix it accepts (no leading white space
/// here): (value saturated to i64, bytes read; 0 if no number)
fn strtol(s: &[u8]) -> (i64, usize) {
    let mut i = 0;
    let neg = match s.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let start = i;
    let mut v: i64 = 0;
    let mut overflow = false;
    while i < s.len() && s[i].is_ascii_digit() {
        let d = (s[i] - b'0') as i64;
        // Accumulate negatively so that i64::MIN fits
        match v.checked_mul(10).and_then(|x| x.checked_sub(d)) {
            Some(x) => v = x,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, 0);
    }
    let v = if overflow {
        if neg {
            i64::MIN
        } else {
            i64::MAX
        }
    } else if neg {
        v
    } else {
        v.checked_neg().unwrap_or(i64::MAX)
    };
    (v, i)
}

/// sscanf(s, "%d%n"): the int (strtol truncated to 32 bits, as the C
/// libraries do) and the characters scanned, or None if no conversion
fn scan_int(s: &[u8]) -> Option<(i32, usize)> {
    let (v, n) = strtol(s);
    if n == 0 {
        None
    } else {
        Some((v as i32, n))
    }
}

/// atoi
fn atoi(s: &[u8]) -> i32 {
    strtol(s).0 as i32
}

/// atof on a string of "+-.0123456789eE" (no white space, inf or nan):
/// strtod of the longest prefix that is a decimal number
fn atof(s: &[u8]) -> f64 {
    let mut i = 0;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < s.len() && s[i].is_ascii_digit() {
        i += 1;
    }
    let mut digits = i - int_start;
    let mut end = i;
    if i < s.len() && s[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
        }
        digits += i - frac_start;
        end = i;
    }
    if digits == 0 {
        return 0.0;
    }
    if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
        let mut j = i + 1;
        if j < s.len() && (s[j] == b'+' || s[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            end = j;
        }
    }
    // Rust's parser rounds correctly, as strtod does; it wants a digit
    // before the exponent
    let mut t = String::from_utf8_lossy(&s[..end]).into_owned();
    if let Some(p) = t.find(['e', 'E']) {
        if t[..p].ends_with('.') {
            t.insert(p, '0');
        }
    } else if t.ends_with('.') {
        t.push('0');
    }
    t.parse::<f64>().unwrap_or(0.0)
}

/// setLocalOptionValue(name, string) with the HighsLogOptions to change
/// when log_file changes (in `host.ctx`)
pub fn set_from_string(host: &COptionHost, name: &[u8], recs: &[COptionRecord], value_passed: &[u8]) -> i32 {
    let log = &host.log;
    let value_trim = trim(value_passed, b" ");
    let index = match option_index(log, name, recs) {
        Ok(i) => i,
        Err(s) => return s,
    };
    let r = &recs[index];
    match r.type_ {
        BOOL => match bool_from_string(value_trim) {
            Some(b) => {
                r.set_bool(b);
                OK
            }
            None => {
                log_user!(
                    log,
                    LogType::Error,
                    "setLocalOptionValue: Value \"%s\" cannot be interpreted as a bool\n",
                    &*st(value_trim)
                );
                ILLEGAL_VALUE
            }
        },
        INT => {
            if value_trim.iter().any(|c| !b"+-0123456789eE".contains(c)) {
                return ILLEGAL_VALUE;
            }
            let scanned = scan_int(value_trim);
            if scanned.map_or(true, |(_, n)| n != value_trim.len()) {
                // ponytail: without a conversion the C++ prints
                // uninitialised values; this prints zeros
                let (v, n) = scanned.unwrap_or((0, 0));
                log_dev!(
                    log,
                    LogType::Error,
                    "setLocalOptionValue: Value = \"%s\" converts via sscanf as %d by scanning %d of %d characters\n",
                    &*st(value_trim),
                    v,
                    n,
                    value_trim.len()
                );
                return ILLEGAL_VALUE;
            }
            set_int_record(log, r, scanned.unwrap().0)
        }
        DOUBLE => {
            let v = value_trim.to_ascii_lowercase();
            let value = if v == b"inf" || v == b"+inf" {
                f64::INFINITY
            } else if v == b"-inf" {
                f64::NEG_INFINITY
            } else {
                if v.iter().any(|c| !b"+-.0123456789eE".contains(c)) {
                    return ILLEGAL_VALUE;
                }
                let value_int = atoi(&v);
                let value_double = atof(&v);
                let value_int_double = value_int as f64;
                if value_double == value_int_double {
                    log_dev!(
                        log,
                        LogType::Info,
                        "setLocalOptionValue: Value = \"%s\" converts via atoi as %d so is %g as double, and %g via atof\n",
                        &*st(&v),
                        value_int,
                        value_int_double,
                        value_double
                    );
                }
                value_double
            };
            set_double_record(log, r, value)
        }
        _ => {
            if name == LOG_FILE && value_passed != r.str_value() {
                if let Some(f) = host.open_log_file {
                    // SAFETY: C++'s context and a byte string
                    unsafe { f(host.ctx, value_passed.as_ptr(), value_passed.len()) }
                }
            }
            if name == MODEL_FILE {
                log_user!(log, LogType::Error, "setLocalOptionValue: model filename cannot be set\n");
                return UNKNOWN_OPTION;
            }
            set_string_record(host, r, value_passed)
        }
    }
}

/// passLocalOptions: check all values of `from` for `to`, then set them
pub fn pass_options(host: &COptionHost, from: &[COptionRecord], to: &[COptionRecord]) -> i32 {
    let log = &host.log;
    for (f, t) in from.iter().zip(to) {
        let s = match t.type_ {
            INT => check_int_value(log, t, f.get_int()),
            DOUBLE => check_double_value(log, t, f.get_double()),
            STRING => check_string_value(host, t, f.str_value()),
            _ => OK,
        };
        if s != OK {
            return s;
        }
    }
    for (f, t) in from.iter().zip(to) {
        let s = match t.type_ {
            BOOL => {
                t.set_bool(f.get_bool());
                OK
            }
            INT => set_int_record(log, t, f.get_int()),
            DOUBLE => set_double_record(log, t, f.get_double()),
            _ => set_string_record(host, t, f.str_value()),
        };
        if s != OK {
            return s;
        }
    }
    OK
}

/// getLocalOptionValues: the record of `name` if it has type `want`
pub fn get_record<'a>(log: &Log, name: &[u8], recs: &'a [COptionRecord], want: i32) -> Result<&'a COptionRecord, i32> {
    let r = &recs[option_index(log, name, recs)?];
    if r.type_ != want {
        let not = match want {
            BOOL => "bool",
            INT => "HighsInt",
            DOUBLE => "double",
            _ => "string",
        };
        log_user!(
            log,
            LogType::Error,
            "getLocalOptionValue: Option \"%s\" requires value of type %s, not %s\n",
            &*st(name),
            type_name(r.type_),
            not
        );
        return Err(ILLEGAL_VALUE);
    }
    Ok(r)
}

/// resetLocalOptions
pub fn reset(host: &COptionHost, recs: &[COptionRecord]) {
    for r in recs {
        match r.type_ {
            BOOL => r.set_bool(r.bool_default),
            INT => r.set_int(r.int_default),
            DOUBLE => r.set_double(r.dbl_default),
            _ => r.set_string(host, r.str_default()),
        }
    }
}

/// highsInsertMdEscapes
pub fn md_escapes(s: &str) -> String {
    s.replace('_', "\\_")
}

/// highsInsertMdId
pub fn md_id(s: &str) -> String {
    s.replace('_', "-")
}

/// highsFormatToString's truncation to its buffer
fn truncate_buffer(mut s: String) -> String {
    if s.len() >= IO_BUFFER_SIZE {
        let mut n = IO_BUFFER_SIZE - 1;
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        s.truncate(n);
    }
    s
}

/// reportOption: the text for one record ("" if not reported)
pub fn report_option(r: &COptionRecord, only_deviations: bool, file_type: i32) -> String {
    let name = st(r.name());
    let desc = st(r.description());
    let adv = bool_str(r.advanced);
    match r.type_ {
        BOOL => {
            let (v, d) = (r.get_bool(), r.bool_default);
            if only_deviations && v == d {
                return String::new();
            }
            match file_type {
                FILE_MD => sprintf!(
                    "## [%s](@id option-%s)\n- %s\n- Type: boolean\n- Default: \"%s\"\n\n",
                    &md_escapes(&name),
                    &md_id(&name),
                    &md_escapes(&desc),
                    bool_str(d)
                ),
                FILE_FULL => {
                    sprintf!("\n# %s\n", &*desc)
                        + &sprintf!(
                            "# [type: bool, advanced: %s, range: {false, true}, default: %s]\n",
                            adv,
                            bool_str(d)
                        )
                        + &sprintf!("%s = %s\n", &*name, bool_str(v))
                }
                _ => truncate_buffer(sprintf!("Set option %s to %s\n", &*name, bool_str(v))),
            }
        }
        INT => {
            let v = r.get_int();
            if only_deviations && v == r.int_default {
                return String::new();
            }
            match file_type {
                FILE_MD => sprintf!(
                    "## [%s](@id option-%s)\n- %s\n- Type: integer\n- Range: {%d, %d}\n- Default: %d\n\n",
                    &md_escapes(&name),
                    &md_id(&name),
                    &md_escapes(&desc),
                    r.int_lower,
                    r.int_upper,
                    r.int_default
                ),
                FILE_FULL => {
                    sprintf!("\n# %s\n", &*desc)
                        + &sprintf!(
                            "# [type: integer, advanced: %s, range: {%d, %d}, default: %d]\n",
                            adv,
                            r.int_lower,
                            r.int_upper,
                            r.int_default
                        )
                        + &sprintf!("%s = %d\n", &*name, v)
                }
                _ => truncate_buffer(sprintf!("Set option %s to %d\n", &*name, v)),
            }
        }
        DOUBLE => {
            let v = r.get_double();
            if only_deviations && v == r.dbl_default {
                return String::new();
            }
            match file_type {
                FILE_MD => sprintf!(
                    "## [%s](@id option-%s)\n- %s\n- Type: double\n- Range: [%g, %g]\n- Default: %g\n\n",
                    &md_escapes(&name),
                    &md_id(&name),
                    &md_escapes(&desc),
                    r.dbl_lower,
                    r.dbl_upper,
                    r.dbl_default
                ),
                FILE_FULL => {
                    sprintf!("\n# %s\n", &*desc)
                        + &sprintf!(
                            "# [type: double, advanced: %s, range: [%g, %g], default: %g]\n",
                            adv,
                            r.dbl_lower,
                            r.dbl_upper,
                            r.dbl_default
                        )
                        + &sprintf!("%s = %g\n", &*name, v)
                }
                _ => truncate_buffer(sprintf!("Set option %s to %g\n", &*name, v)),
            }
        }
        _ => {
            if r.name() == b"options_file" {
                return String::new();
            }
            let (v, d) = (st(r.str_value()), st(r.str_default()));
            if only_deviations && r.str_value() == r.str_default() {
                return String::new();
            }
            match file_type {
                FILE_MD => sprintf!(
                    "## [%s](@id option-%s)\n- %s\n- Type: string\n- Default: \"%s\"\n\n",
                    &md_escapes(&name),
                    &md_id(&name),
                    &md_escapes(&desc),
                    &*d
                ),
                FILE_FULL => {
                    sprintf!("\n# %s\n", &*desc)
                        + &sprintf!("# [type: string, advanced: %s, default: \"%s\"]\n", adv, &*d)
                        + &sprintf!("%s = %s\n", &*name, &*v)
                }
                _ => truncate_buffer(sprintf!("Set option %s to \"%s\"\n", &*name, &*v)),
            }
        }
    }
}

/// reportOption to a FILE: the minimal report to stdout goes to the log
pub fn write_option(host: &COptionHost, file: *mut c_void, file_is_stdout: bool, r: &COptionRecord, only_deviations: bool, file_type: i32) {
    let text = report_option(r, only_deviations, file_type);
    if text.is_empty() {
        return;
    }
    if file_type != FILE_MD && file_type != FILE_FULL && file_is_stdout {
        host.log.user(LogType::Info, &text);
    } else {
        host.write(file, &text);
    }
}

/// reportOptions
pub fn write_options(
    host: &COptionHost,
    file: *mut c_void,
    file_is_stdout: bool,
    recs: &[COptionRecord],
    only_deviations: bool,
    file_type: i32,
) {
    let not_md_or_full = file_type != FILE_MD && file_type != FILE_FULL;
    if file_type == FILE_MD {
        host.write(file, "# [List of options](@id option-definitions)\n\n");
    }
    for r in recs {
        if not_md_or_full && r.name() == LOG_FILE && r.str_value() == HIGHS_RUN_LOG_FILE {
            continue;
        }
        // Advanced options are not reported (kAdvancedInDocumentation)
        if r.advanced {
            continue;
        }
        write_option(host, file, file_is_stdout, r, only_deviations, file_type);
    }
}

/// loadOptionsFromFile: HighsLoadOptionsStatus (-1 kError, 0 kOk, 1 kEmpty)
pub fn load_options_from_file(host: &COptionHost, recs: &[COptionRecord], filename: &[u8]) -> i32 {
    const LOAD_OK: i32 = 0;
    const LOAD_ERROR: i32 = -1;
    const LOAD_EMPTY: i32 = 1;
    let log = &host.log;
    if filename.is_empty() {
        return LOAD_EMPTY;
    }
    let content = match open_bytes(filename) {
        Some(c) => c,
        None => {
            log_user!(log, LogType::Error, "Options file not found\n");
            return LOAD_ERROR;
        }
    };
    // trim's characters for the options file: \" and \' too
    const NON_CHARS: &[u8] = b"\t\n\x0b\x0c\r\"' ";
    // getline while good(): each piece between newlines, including the
    // (empty) one after a final newline
    for (k, line) in content.split(|&c| c == b'\n').enumerate() {
        let line_count = k + 1;
        if line.is_empty() || line[0] == b'#' || line.iter().all(|&c| c == b' ') {
            continue;
        }
        let equals = line.iter().position(|&c| c == b'=');
        let Some(equals) = equals.filter(|&e| e + 1 < line.len()) else {
            log_user!(
                log,
                LogType::Error,
                "Error on line %d (\"%s\") of options file\n",
                line_count,
                &*st(line)
            );
            return LOAD_ERROR;
        };
        let option = trim(&line[..equals], NON_CHARS);
        let value = trim(&line[equals + 1..], NON_CHARS);
        if set_from_string(host, option, recs, value) != OK {
            log_user!(
                log,
                LogType::Error,
                "Cannot read value \"%s\" for option \"%s\"\n",
                &*st(value),
                &*st(option)
            );
            return LOAD_ERROR;
        }
    }
    LOAD_OK
}

/// The bytes of a file, None if it cannot be opened; a read error ends
/// the content (as the stream's getline would)
fn open_bytes(filename: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    #[cfg(unix)]
    let path = {
        use std::os::unix::ffi::OsStrExt;
        std::path::PathBuf::from(std::ffi::OsStr::from_bytes(filename))
    };
    #[cfg(not(unix))]
    let path = std::path::PathBuf::from(String::from_utf8_lossy(filename).into_owned());
    let mut f = std::fs::File::open(path).ok()?;
    let mut content = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match f.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => content.extend_from_slice(&buf[..n]),
        }
    }
    Some(content)
}

/// warnSolverInvalid
pub fn warn_solver_invalid(log: &Log, solver: &[u8], problem_type: &[u8]) {
    log_user!(
        log,
        LogType::Warning,
        "Solver \"%s\" is not available for %s. Option \"%s\" ignored\n",
        &*st(solver),
        &*st(problem_type),
        "solver"
    );
}

/// solverValidForLp (0), solverValidForMip (1), solverValidForQp (2)
pub fn solver_valid(solver: &[u8], problem: i32) -> bool {
    match problem {
        0 => [CHOOSE, SIMPLEX, IPM, IPX, HIPO, PDLP, HIPDLP].contains(&solver),
        1 => solver == CHOOSE,
        _ => [CHOOSE, QPASM, IPM, HIPO].contains(&solver),
    }
}

// The C++ entry points (highs/lp_data/HighsOptionsRust.cpp)

/// # Safety
/// `ptr` valid for `len` bytes
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    RsStr { ptr, len }.get()
}

/// # Safety
/// `recs` valid for `n` records
unsafe fn table<'a>(recs: *const COptionRecord, n: usize) -> &'a [COptionRecord] {
    if n == 0 {
        &[]
    } else {
        from_raw_parts(recs, n)
    }
}

/// # Safety (all entry points)
/// The host, the record tables (with their strings and value pointers)
/// and the byte strings must be valid for the call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_option_index(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    recs: *const COptionRecord,
    n: usize,
    index: *mut i32,
) -> i32 {
    match option_index(&*log, bytes(name, name_len), table(recs, n)) {
        Ok(i) => {
            *index = i as i32;
            OK
        }
        Err(s) => {
            *index = n as i32;
            s
        }
    }
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_options(log: *const Log, recs: *const COptionRecord, n: usize) -> i32 {
    check_options(&*log, table(recs, n))
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_option(log: *const Log, rec: *const COptionRecord) -> i32 {
    check_option(&*log, &*rec)
}

/// checkOptionValue for HighsInt and double records
///
/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_option_value_number(
    log: *const Log,
    rec: *const COptionRecord,
    int_value: i32,
    double_value: f64,
) -> i32 {
    if (*rec).type_ == INT {
        check_int_value(&*log, &*rec, int_value)
    } else {
        check_double_value(&*log, &*rec, double_value)
    }
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_option_value_string(
    host: *const COptionHost,
    rec: *const COptionRecord,
    value: *const u8,
    len: usize,
) -> i32 {
    check_string_value(&*host, &*rec, bytes(value, len))
}

/// The string option validators: 0 off/choose/on, 1 off/on, 2 solver,
/// 3 MIP LP solver, 4 MIP IPM solver, 5 HiPO parallel type, 6 HiPO
/// system, 7 HiPO ordering
///
/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_option_value_ok(
    host: *const COptionHost,
    which: i32,
    name: *const u8,
    name_len: usize,
    value: *const u8,
    value_len: usize,
) -> bool {
    let host = &*host;
    let (name, value) = (bytes(name, name_len), bytes(value, value_len));
    match which {
        0 => off_choose_on_ok(&host.log, name, value),
        1 => off_on_ok(&host.log, name, value),
        2 => solver_ok(host, value),
        3 => mip_lp_solver_ok(host, value),
        4 => mip_ipm_solver_ok(host, value),
        5 => hipo_parallel_type_ok(&host.log, value),
        6 => hipo_system_ok(&host.log, value),
        _ => hipo_ordering_ok(&host.log, value),
    }
}

/// boolFromString
///
/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_bool_from_string(value: *const u8, len: usize, b: *mut bool) -> bool {
    match bool_from_string(bytes(value, len)) {
        Some(v) => {
            *b = v;
            true
        }
        None => false,
    }
}

/// possibleLowerCaseOptionValue: lower cases `value` in place
///
/// # Safety
/// As highs_rs_option_index; `value` writable
#[no_mangle]
pub unsafe extern "C" fn highs_rs_possible_lower_case(name: *const u8, name_len: usize, value: *mut u8, len: usize) {
    let mut v = bytes(value, len).to_vec();
    possible_lower_case(bytes(name, name_len), &mut v);
    if len > 0 {
        std::ptr::copy_nonoverlapping(v.as_ptr(), value, len);
    }
}

/// setLocalOptionValue by name: `kind` 0 bool, 1 HighsInt, 2 double, 3
/// string (the bytes `value`)
///
/// # Safety
/// As highs_rs_option_index
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_set_option(
    host: *const COptionHost,
    name: *const u8,
    name_len: usize,
    recs: *const COptionRecord,
    n: usize,
    kind: i32,
    bool_value: bool,
    int_value: i32,
    double_value: f64,
    value: *const u8,
    value_len: usize,
) -> i32 {
    let host = &*host;
    let (name, recs) = (bytes(name, name_len), table(recs, n));
    match kind {
        BOOL => set_bool(&host.log, name, recs, bool_value),
        INT => set_int(&host.log, name, recs, int_value),
        DOUBLE => set_double(&host.log, name, recs, double_value),
        _ => set_from_string(host, name, recs, bytes(value, value_len)),
    }
}

/// setLocalOptionValue of a record (the value of its type)
///
/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_set_option_record(
    host: *const COptionHost,
    rec: *const COptionRecord,
    bool_value: bool,
    int_value: i32,
    double_value: f64,
    value: *const u8,
    value_len: usize,
) -> i32 {
    let (host, r) = (&*host, &*rec);
    match r.type_ {
        BOOL => {
            r.set_bool(bool_value);
            OK
        }
        INT => set_int_record(&host.log, r, int_value),
        DOUBLE => set_double_record(&host.log, r, double_value),
        _ => set_string_record(host, r, bytes(value, value_len)),
    }
}

/// # Safety
/// As highs_rs_option_index; both tables have `n` records
#[no_mangle]
pub unsafe extern "C" fn highs_rs_pass_options(
    host: *const COptionHost,
    from: *const COptionRecord,
    to: *const COptionRecord,
    n: usize,
) -> i32 {
    pass_options(&*host, table(from, n), table(to, n))
}

/// getLocalOptionValues of type `want`: the out pointers may be null;
/// string values go to the std::strings `str_current` and `str_default`
/// through set_string
///
/// # Safety
/// As highs_rs_option_index; out pointers null or writable
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_get_option_values(
    host: *const COptionHost,
    name: *const u8,
    name_len: usize,
    recs: *const COptionRecord,
    n: usize,
    want: i32,
    current: *mut c_void,
    min: *mut c_void,
    max: *mut c_void,
    default: *mut c_void,
) -> i32 {
    let host = &*host;
    let r = match get_record(&host.log, bytes(name, name_len), table(recs, n), want) {
        Ok(r) => r,
        Err(s) => return s,
    };
    unsafe fn put<T>(p: *mut c_void, v: T) {
        if !p.is_null() {
            *(p as *mut T) = v;
        }
    }
    match want {
        BOOL => {
            put(current, r.get_bool());
            put(default, r.bool_default);
        }
        INT => {
            put(current, r.get_int());
            put(min, r.int_lower);
            put(max, r.int_upper);
            put(default, r.int_default);
        }
        DOUBLE => {
            put(current, r.get_double());
            put(min, r.dbl_lower);
            put(max, r.dbl_upper);
            put(default, r.dbl_default);
        }
        _ => {
            if let Some(f) = host.set_string {
                let (v, d) = (r.str_value(), r.str_default());
                if !current.is_null() {
                    f(current, v.as_ptr(), v.len());
                }
                if !default.is_null() {
                    f(default, d.as_ptr(), d.len());
                }
            }
        }
    }
    OK
}

/// getLocalOptionType
///
/// # Safety
/// As highs_rs_option_index; `type_` null or writable
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_option_type(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    recs: *const COptionRecord,
    n: usize,
    type_: *mut i32,
) -> i32 {
    let recs = table(recs, n);
    match option_index(&*log, bytes(name, name_len), recs) {
        Ok(i) => {
            if !type_.is_null() {
                *type_ = recs[i].type_;
            }
            OK
        }
        Err(s) => s,
    }
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_reset_options(host: *const COptionHost, recs: *const COptionRecord, n: usize) {
    reset(&*host, table(recs, n))
}

/// reportOptions (`all` true) or reportOption of the single record
///
/// # Safety
/// As highs_rs_option_index; `file` a FILE* for host.write
#[no_mangle]
pub unsafe extern "C" fn highs_rs_report_options(
    host: *const COptionHost,
    file: *mut c_void,
    file_is_stdout: bool,
    recs: *const COptionRecord,
    n: usize,
    all: bool,
    only_deviations: bool,
    file_type: i32,
) {
    let recs = table(recs, n);
    if all {
        write_options(&*host, file, file_is_stdout, recs, only_deviations, file_type)
    } else if let Some(r) = recs.first() {
        write_option(&*host, file, file_is_stdout, r, only_deviations, file_type)
    }
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_load_options_from_file(
    host: *const COptionHost,
    recs: *const COptionRecord,
    n: usize,
    filename: *const u8,
    len: usize,
) -> i32 {
    load_options_from_file(&*host, table(recs, n), bytes(filename, len))
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_warn_solver_invalid(
    log: *const Log,
    solver: *const u8,
    solver_len: usize,
    problem: *const u8,
    problem_len: usize,
) {
    warn_solver_invalid(&*log, bytes(solver, solver_len), bytes(problem, problem_len))
}

/// # Safety
/// As highs_rs_option_index
#[no_mangle]
pub unsafe extern "C" fn highs_rs_solver_valid(solver: *const u8, len: usize, problem: i32) -> bool {
    solver_valid(bytes(solver, len), problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static LOG: RefCell<String> = const { RefCell::new(String::new()) };
        static OUT: RefCell<String> = const { RefCell::new(String::new()) };
    }

    unsafe extern "C" fn log_fn(_: *const c_void, dev: i32, _t: i32, msg: *const u8, len: usize) {
        let s = std::str::from_utf8(from_raw_parts(msg, len)).unwrap();
        LOG.with(|l| l.borrow_mut().push_str(&format!("{dev}:{s}")));
    }
    unsafe extern "C" fn set_string(p: *mut c_void, v: *const u8, len: usize) {
        *(p as *mut Vec<u8>) = bytes(v, len).to_vec();
    }
    unsafe extern "C" fn write(_: *mut c_void, v: *const u8, len: usize) {
        OUT.with(|o| o.borrow_mut().push_str(std::str::from_utf8(bytes(v, len)).unwrap()));
    }
    unsafe extern "C" fn no_hipo() -> bool {
        false
    }

    fn host() -> COptionHost {
        COptionHost {
            log: Log { opts: std::ptr::null(), log: Some(log_fn) },
            ctx: std::ptr::null_mut(),
            set_string: Some(set_string),
            open_log_file: None,
            hipo_available: Some(no_hipo),
            hipo_unavailable: None,
            write: Some(write),
        }
    }

    fn rec(type_: i32, name: &'static str, value: *mut c_void) -> COptionRecord {
        COptionRecord {
            type_,
            advanced: false,
            name: RsStr::of(name.as_bytes()),
            description: RsStr::of(b"Desc_x"),
            value,
            str_value: RsStr::of(b""),
            str_default: RsStr::of(b"choose"),
            bool_default: true,
            int_lower: 0,
            int_default: 5,
            int_upper: 10,
            dbl_lower: 0.0,
            dbl_default: 1e-7,
            dbl_upper: f64::INFINITY,
        }
    }

    fn take_log() -> String {
        LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    #[test]
    fn set_and_report() {
        let (mut b, mut i, mut d, mut s) = (true, 5i32, 1e-7f64, b"choose".to_vec());
        let recs = [
            rec(BOOL, "output_flag", &mut b as *mut bool as *mut c_void),
            rec(INT, "threads", &mut i as *mut i32 as *mut c_void),
            rec(DOUBLE, "time_limit", &mut d as *mut f64 as *mut c_void),
            rec(STRING, "presolve", &mut s as *mut Vec<u8> as *mut c_void),
        ];
        let h = host();
        assert_eq!(set_from_string(&h, b"output_flag", &recs, b" FALSE "), OK);
        assert!(!b);
        assert_eq!(set_from_string(&h, b"threads", &recs, b"1e3"), ILLEGAL_VALUE);
        assert_eq!(set_from_string(&h, b"threads", &recs, b"11"), ILLEGAL_VALUE);
        assert_eq!(take_log(), "1:setLocalOptionValue: Value = \"1e3\" converts via sscanf as 1 by scanning 1 of 3 characters\n0:checkOptionValue: Value 11 for option \"threads\" is above upper bound of 10\n");
        assert_eq!(set_from_string(&h, b"threads", &recs, b"+7"), OK);
        assert_eq!(i, 7);
        assert_eq!(set_from_string(&h, b"time_limit", &recs, b"2.5e1"), OK);
        assert_eq!(d, 25.0);
        assert_eq!(set_from_string(&h, b"time_limit", &recs, b"-1"), ILLEGAL_VALUE);
        assert_eq!(set_from_string(&h, b"presolve", &recs, b"OFF"), OK);
        assert_eq!(s, b"off");
        assert_eq!(set_from_string(&h, b"presolve", &recs, b"maybe"), ILLEGAL_VALUE);
        assert_eq!(set_from_string(&h, b"nope", &recs, b"1"), UNKNOWN_OPTION);
        assert_eq!(set_int(&h.log, b"time_limit", &recs, 3), OK);
        assert_eq!(d, 3.0);
        take_log();
        let full = report_option(&recs[2], false, FILE_FULL);
        assert_eq!(full, "\n# Desc_x\n# [type: double, advanced: false, range: [0, inf], default: 1e-07]\ntime_limit = 3\n");
        let md = report_option(&recs[1], false, FILE_MD);
        assert_eq!(md, "## [threads](@id option-threads)\n- Desc\\_x\n- Type: integer\n- Range: {0, 10}\n- Default: 5\n\n");
        assert_eq!(report_option(&recs[0], true, FILE_MINIMAL), "Set option output_flag to false\n");
        reset(&h, &recs);
        assert!(b && i == 5 && d == 1e-7 && s == b"choose");
        assert_eq!(check_options(&h.log, &recs), OK);
        assert_eq!(take_log(), "0:checkOptions: Options are OK\n");
    }

    #[test]
    fn c_conversions() {
        assert_eq!(scan_int(b"99999999999"), Some((1215752191, 11)));
        assert_eq!(scan_int(b"-2147483649"), Some((2147483647, 11)));
        assert_eq!(scan_int(b"99999999999999999999999"), Some((-1, 23)));
        assert_eq!(scan_int(b"-99999999999999999999999"), Some((0, 24)));
        assert_eq!(scan_int(b"--5"), None);
        assert_eq!(scan_int(b"12e"), Some((12, 2)));
        assert_eq!(atof(b"1.5e"), 1.5);
        assert_eq!(atof(b"1e+"), 1.0);
        assert_eq!(atof(b"..5"), 0.0);
        assert_eq!(atof(b"-.5"), -0.5);
        assert_eq!(atof(b"5."), 5.0);
        assert_eq!(atof(b"5.e2"), 500.0);
        assert_eq!(atof(b"1e400"), f64::INFINITY);
        assert_eq!(atof(b"1.2.3"), 1.2);
        assert_eq!(atof(b"2.4703282292062328e-324"), 4.9406564584124654e-324);
        assert_eq!(trim(b"  a b ", b" "), b"a b");
        assert_eq!(trim(b"   ", b" "), b"");
    }
}
