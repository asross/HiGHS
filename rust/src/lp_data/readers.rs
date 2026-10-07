//! Solution and basis file readers (HighsLpUtils.cpp: readSolutionFile,
//! readBasisFile, readBasisStream and their line helpers). The C++ read
//! them with std::ifstream's `>>`, `ignore` and `eof()`; `IStream`
//! follows libc++'s semantics for those (sentry, eofbit/failbit, a failed
//! extraction leaving later ones without effect, num_get's accumulation of
//! a number then strtod/strtoll), so a file reads the same, malformed
//! ones included. (libstdc++ differs on malformed numbers, e.g. "1.5abc"
//! reads 1.5 there and fails here, as in libc++.) Name lookups use the
//! LP's C++ name hash (`Names`), formed by C++ where it was.

use super::edit::calculate_row_values_quad;
use super::ffi::RsMut;
use super::{Log, LogType, Status};
use crate::log_user;
use std::ffi::{c_char, c_void};

/// std::ifstream over a file's bytes
pub struct IStream<'a> {
    buf: &'a [u8],
    pos: usize,
    eof: bool,
    fail: bool,
}

/// isspace in the C locale
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

extern "C" {
    fn strtod(s: *const c_char, end: *mut *mut c_char) -> f64;
    fn strtoll(s: *const c_char, end: *mut *mut c_char, base: i32) -> i64;
    #[cfg_attr(any(target_os = "macos", target_os = "ios"), link_name = "__error")]
    #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), link_name = "__errno_location")]
    fn errno_location() -> *mut i32;
}

const ERANGE: i32 = 34;

/// strtod/strtoll(s, &end, 10) of a whole token: (value, all of it read,
/// ERANGE)
fn c_parse<T>(s: &[u8], f: impl FnOnce(*const c_char, *mut *mut c_char) -> T) -> (T, bool, bool) {
    let c = std::ffi::CString::new(s).unwrap_or_default();
    let mut end = std::ptr::null_mut();
    // SAFETY: c is NUL-terminated and outlives the call; errno is this
    // thread's
    unsafe {
        *errno_location() = 0;
        let v = f(c.as_ptr(), &mut end);
        let range = *errno_location() == ERANGE;
        let all = end as usize - c.as_ptr() as usize == s.len();
        (v, all, range)
    }
}

impl<'a> IStream<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        IStream { buf, pos: 0, eof: false, fail: false }
    }
    pub fn eof(&self) -> bool {
        self.eof
    }
    fn peek(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }
    /// istream::sentry, skipping white space unless `noskipws`
    fn sentry(&mut self, noskipws: bool) -> bool {
        if self.eof || self.fail {
            self.fail = true;
            return false;
        }
        if !noskipws {
            while let Some(c) = self.peek() {
                if !is_space(c) {
                    return true;
                }
                self.pos += 1;
            }
            self.eof = true;
            self.fail = true;
            return false;
        }
        true
    }
    /// `>> std::string`
    pub fn word(&mut self, s: &mut Vec<u8>) {
        if !self.sentry(false) {
            return;
        }
        s.clear();
        while let Some(c) = self.peek() {
            if is_space(c) {
                return;
            }
            s.push(c);
            self.pos += 1;
        }
        self.eof = true;
    }
    /// The characters num_get accumulates (libc++'s stage 2): for an
    /// integer the decimal digits after an optional sign, for a double
    /// any of "0123456789abcdefABCDEFxX+-pPiInN" and the point, a sign
    /// only first or after an exponent mark
    fn stage2(&mut self, float: bool) -> Vec<u8> {
        let mut t = Vec::new();
        while let Some(c) = self.peek() {
            let ok = if c == b'+' || c == b'-' {
                t.is_empty() || (float && matches!(t.last(), Some(b'e' | b'E' | b'p' | b'P')))
            } else if float {
                c.is_ascii_hexdigit() || b".xXpPiInN".contains(&c)
            } else {
                c.is_ascii_digit()
            };
            if !ok {
                return t;
            }
            t.push(c);
            self.pos += 1;
        }
        self.eof = true;
        t
    }
    /// `>> HighsInt`: long's num_get, then the int range check
    pub fn int(&mut self, v: &mut i32) {
        if !self.sentry(false) {
            return;
        }
        let t = self.stage2(false);
        let (x, all, range) = c_parse(&t, |p, e| unsafe { strtoll(p, e, 10) });
        if t.is_empty() || !all || t == b"+" || t == b"-" {
            *v = 0;
            self.fail = true;
        } else if range || x > i32::MAX as i64 || x < i32::MIN as i64 {
            *v = if x < 0 { i32::MIN } else { i32::MAX };
            self.fail = true;
        } else {
            *v = x as i32;
        }
    }
    /// `>> double`: strtod of what num_get accumulated, which must all
    /// be read; ERANGE also fails, keeping strtod's value
    pub fn double(&mut self, v: &mut f64) {
        if !self.sentry(false) {
            return;
        }
        let t = self.stage2(true);
        let (x, all, range) = c_parse(&t, |p, e| unsafe { strtod(p, e) });
        if t.is_empty() || !all {
            *v = 0.0;
            self.fail = true;
        } else {
            *v = x;
            if range {
                self.fail = true;
            }
        }
    }
    /// `ignore(n, '\n')`
    pub fn ignore(&mut self, n: usize) {
        if !self.sentry(true) {
            return;
        }
        for _ in 0..n {
            match self.peek() {
                None => {
                    self.eof = true;
                    return;
                }
                Some(c) => {
                    self.pos += 1;
                    if c == b'\n' {
                        return;
                    }
                }
            }
        }
    }
}

/// kMaxLineLength
const MAX_LINE_LENGTH: usize = 80;

/// The LP's names as the readers use them: whether there are names, and
/// their C++ hash (HighsNameHash), formed by C++ on demand
pub struct Names<'a> {
    pub have_col: bool,
    pub have_row: bool,
    pub host: &'a CNames,
}

/// The C++ side of `Names`: `op(ctx, 0, ..)` forms the hashes of the
/// names there are and that are not formed; `op(ctx, 1 or 2, name)` is
/// the index of a column or row name, kHashIsDuplicate (-1) or -2 if
/// not found
#[repr(C)]
pub struct CNames {
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, *const u8, usize) -> i32,
    pub have_col: bool,
    pub have_row: bool,
}

impl Names<'_> {
    fn form(&self) {
        // SAFETY: C++ gave the function and its context
        unsafe { (self.host.op)(self.host.ctx, 0, std::ptr::null(), 0) };
    }
    /// getIndexFromName
    fn index(&self, log: &Log, from: &str, is_col: bool, name: &[u8]) -> Result<usize, Status> {
        // SAFETY: as form
        let i = unsafe { (self.host.op)(self.host.ctx, if is_col { 1 } else { 2 }, name.as_ptr(), name.len()) };
        let what = if is_col { "column" } else { "row" };
        let name = String::from_utf8_lossy(name);
        match i {
            -2 => {
                log_user!(log, LogType::Error, "%s: %s name %s is not found\n", from, what, name.as_ref());
                Err(Status::Error)
            }
            -1 => {
                log_user!(log, LogType::Error, "%s: %s name %s is duplicated\n", from, what, name.as_ref());
                Err(Status::Error)
            }
            i => Ok(i as usize),
        }
    }
}

/// HighsBasis's statuses, sized by C++
pub struct BasisMut<'a> {
    pub valid: &'a mut bool,
    pub col_status: &'a mut [u8],
    pub row_status: &'a mut [u8],
}

/// readBasisStream
pub fn read_basis_stream(log: &Log, names: &Names, basis: &mut BasisMut, s: &mut IStream) -> Status {
    let from = "readBasisStream";
    let (mut highs, mut version) = (Vec::new(), Vec::new());
    s.word(&mut highs);
    s.word(&mut version);
    let v1 = version == b"v1";
    let v2 = version == b"v2";
    names.form();
    if !(v1 || v2) {
        log_user!(
            log,
            LogType::Error,
            "readBasisFile: Cannot read basis file for HiGHS %s\n",
            String::from_utf8_lossy(&version).as_ref()
        );
        return Status::Error;
    }
    let mut status = Status::Ok;
    if v1 {
        log_user!(log, LogType::Warning, "readBasisFile: Basis file format %s is deprecated\n", "v1");
        status = Status::Warning;
    }
    let mut keyword = Vec::new();
    s.word(&mut keyword);
    if keyword == b"None" {
        *basis.valid = false;
        return status;
    }
    let mut int_status = 0;
    let mut name = Vec::new();
    for (is_col, statuses) in [(true, &mut *basis.col_status), (false, &mut *basis.row_status)] {
        s.word(&mut keyword);
        s.word(&mut keyword);
        let mut num = 0;
        s.int(&mut num);
        let basis_num = statuses.len() as i32;
        if num != basis_num {
            log_user!(
                log,
                LogType::Error,
                "readBasisFile: Basis file is for %d %s, not %d\n",
                num,
                if is_col { "columns" } else { "rows" },
                basis_num
            );
            return Status::Error;
        }
        let have_names = if is_col { names.have_col } else { names.have_row };
        for x in 0..num as usize {
            let i = if v1 {
                s.int(&mut int_status);
                x
            } else {
                s.word(&mut name);
                s.int(&mut int_status);
                if have_names {
                    match names.index(log, from, is_col, &name) {
                        Ok(i) => i,
                        Err(e) => return e,
                    }
                } else {
                    x
                }
            };
            statuses[i] = int_status as u8;
        }
    }
    status
}

/// HighsSolution's vectors, sized by C++
pub struct SolutionMut<'a> {
    pub col_value: &'a mut [f64],
    pub col_dual: &'a mut [f64],
    pub row_value: &'a mut [f64],
    pub row_dual: &'a mut [f64],
}

/// The column-wise matrix for the row values (None if not column-wise)
pub struct Matrix<'a> {
    pub start: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

/// How readSolutionFile ended: an error, or the read solution and basis
/// are copied (`Ok` with `value_valid`) or not (a warning or the basis'
/// status)
#[derive(Debug, PartialEq)]
pub enum SolutionRead {
    Error,
    Return(Status),
}

/// readSolutionFileHashKeywordIntLineOk
fn hash_keyword_int(s: &mut IStream, hash: &mut Vec<u8>, keyword: &mut Vec<u8>, value_string: &mut Vec<u8>, value: &mut i32) -> bool {
    hash.clear();
    keyword.clear();
    value_string.clear();
    if s.eof() {
        return false;
    }
    s.word(hash);
    if hash != b"#" {
        return false;
    }
    if s.eof() {
        return false;
    }
    s.word(keyword);
    if s.eof() {
        return false;
    }
    s.word(value_string);
    if value_string.iter().any(|c| !b"-0123456789".contains(c)) {
        return false;
    }
    *value = stoi(value_string);
    true
}

/// std::stoi: a C++ exception (std::terminate) where it throws
fn stoi(s: &[u8]) -> i32 {
    let (x, _, range) = c_parse(s, |p, e| unsafe { strtoll(p, e, 10) });
    let digits = s.iter().skip(usize::from(matches!(s.first(), Some(b'-' | b'+')))).take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        panic!("terminate called after throwing an instance of 'std::invalid_argument': stoi");
    }
    if range || x > i32::MAX as i64 || x < i32::MIN as i64 {
        panic!("terminate called after throwing an instance of 'std::out_of_range': stoi");
    }
    x as i32
}

/// readSolutionFile from the second check on (C++ checked the style and
/// opened the file); `sol` and `basis` are the cleared and sized
/// read_solution and read_basis, `value_valid` read_solution's flag
#[allow(clippy::too_many_arguments)]
pub fn read_solution_file(
    log: &Log,
    names: &Names,
    a: Option<&Matrix>,
    style_sparse: bool,
    sol: &mut SolutionMut,
    value_valid: &mut bool,
    basis: &mut BasisMut,
    s: &mut IStream,
) -> SolutionRead {
    use SolutionRead::*;
    let from = "readSolutionFile";
    let lp_num_col = sol.col_value.len() as i32;
    let lp_num_row = sol.row_value.len() as i32;
    let (mut hash, mut keyword, mut value_string, mut name) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut value = 0.0;
    let mut num_col: i32 = -1;
    let mut num_row: i32 = -1;
    let ignore = |s: &mut IStream| {
        if s.eof() {
            return false;
        }
        s.ignore(MAX_LINE_LENGTH);
        true
    };
    let keyword_line = |s: &mut IStream, k: &mut Vec<u8>| {
        if s.eof() {
            return false;
        }
        s.word(k);
        true
    };
    let id_double = |s: &mut IStream, id: &mut Vec<u8>, v: &mut f64| {
        if s.eof() {
            return false;
        }
        s.word(id);
        if s.eof() {
            return false;
        }
        s.double(v);
        true
    };
    let row_values = |sol: &mut SolutionMut| match a {
        Some(a) => {
            calculate_row_values_quad(a.start, a.index, a.value, sol.col_value, sol.row_value);
            true
        }
        None => false,
    };
    let error_line = |hash: &[u8], keyword: &[u8], value_string: &[u8]| {
        log_user!(
            log,
            LogType::Error,
            "readSolutionFile: Error reading line \"%s %s %s\"\n",
            String::from_utf8_lossy(hash).as_ref(),
            String::from_utf8_lossy(keyword).as_ref(),
            String::from_utf8_lossy(value_string).as_ref()
        );
    };
    let index = |is_col: bool, name: &[u8]| names.index(log, from, is_col, name);

    let mut section_name = Vec::new();
    if s.eof() {
        return Error;
    }
    s.word(&mut section_name);
    s.ignore(MAX_LINE_LENGTH);
    let miplib_sol = section_name == b"=obj=";
    if miplib_sol && !names.have_col {
        log_user!(
            log,
            LogType::Error,
            "readSolutionFile: Cannot read a MIPLIB solution file without column names in the model\n"
        );
        return Return(Status::Error);
    }
    names.form();
    let mut sparse = false;
    if !miplib_sol {
        for _ in 0..3 {
            if !ignore(s) {
                return Error;
            }
        }
        if !keyword_line(s, &mut keyword) {
            return Error;
        }
        if keyword == b"None" {
            return Return(Status::Warning);
        }
        for _ in 0..2 {
            if !ignore(s) {
                return Error;
            }
        }
        if !hash_keyword_int(s, &mut hash, &mut keyword, &mut value_string, &mut num_col) {
            error_line(&hash, &keyword, &value_string);
            return Error;
        }
        sparse = num_col <= 0;
        debug_assert!(!style_sparse || sparse);
        if sparse {
            num_col = -num_col;
        } else if num_col != lp_num_col {
            log_user!(
                log,
                LogType::Error,
                "readSolutionFile: Solution file is for %d columns, not %d\n",
                num_col,
                lp_num_col
            );
            return Error;
        }
    }

    if miplib_sol {
        sol.col_value.fill(0.0);
        loop {
            if !id_double(s, &mut name, &mut value) {
                break;
            }
            match index(true, &name) {
                Ok(i) => sol.col_value[i] = value,
                Err(e) => return Return(e),
            }
            if s.eof() {
                break;
            }
        }
    } else if sparse {
        sol.col_value.fill(0.0);
        let mut i_read = 0;
        for _ in 0..num_col {
            // readSolutionFileIdDoubleIntLineOk
            let ok = !s.eof() && {
                s.word(&mut name);
                !s.eof() && {
                    s.double(&mut value);
                    !s.eof() && {
                        s.int(&mut i_read);
                        true
                    }
                }
            };
            if !ok {
                return Error;
            }
            let i_col = if names.have_col {
                match index(true, &name) {
                    Ok(i) => i,
                    Err(e) => return Return(e),
                }
            } else {
                i_read as usize
            };
            sol.col_value[i_col] = value;
        }
    } else {
        for x in 0..num_col as usize {
            if !id_double(s, &mut name, &mut value) {
                return Error;
            }
            let i_col = if names.have_col {
                match index(true, &name) {
                    Ok(i) => i,
                    Err(e) => return Return(e),
                }
            } else {
                x
            };
            sol.col_value[i_col] = value;
        }
    }
    *value_valid = true;
    if miplib_sol || sparse {
        return if row_values(sol) { Return(Status::Ok) } else { Error };
    }
    if !hash_keyword_int(s, &mut hash, &mut keyword, &mut value_string, &mut num_row) {
        return if row_values(sol) { Return(Status::Ok) } else { Error };
    }
    let num_row_ok = num_row == lp_num_row;
    for x in 0..num_row.max(0) as usize {
        if !id_double(s, &mut name, &mut value) {
            return Error;
        }
        if num_row_ok {
            let i_row = if names.have_row {
                match index(false, &name) {
                    Ok(i) => i,
                    Err(e) => return Return(e),
                }
            } else {
                x
            };
            sol.row_value[i_row] = value;
        }
    }
    if !num_row_ok {
        log_user!(
            log,
            LogType::Warning,
            "readSolutionFile: Solution file is for %d rows, not %d: row values ignored\n",
            num_row,
            lp_num_row
        );
        if !row_values(sol) {
            return Error;
        }
    }
    for _ in 0..3 {
        if !ignore(s) {
            return Return(Status::Ok);
        }
    }
    if !keyword_line(s, &mut keyword) {
        return Error;
    }
    if keyword != b"None" {
        if !ignore(s) {
            return Error;
        }
        if !hash_keyword_int(s, &mut hash, &mut keyword, &mut value_string, &mut num_col) {
            error_line(&hash, &keyword, &value_string);
            return Return(Status::Ok);
        }
        let mut dual = 0.0;
        for x in 0..num_col.max(0) as usize {
            if !id_double(s, &mut name, &mut dual) {
                return Error;
            }
            let i_col = if names.have_col {
                match index(true, &name) {
                    Ok(i) => i,
                    Err(e) => return Return(e),
                }
            } else {
                x
            };
            sol.col_dual[i_col] = dual;
        }
        if !hash_keyword_int(s, &mut hash, &mut keyword, &mut value_string, &mut num_col) {
            error_line(&hash, &keyword, &value_string);
            return Return(Status::Ok);
        }
        for x in 0..num_row.max(0) as usize {
            if !id_double(s, &mut name, &mut dual) {
                return Error;
            }
            let i_row = if names.have_row {
                match index(false, &name) {
                    Ok(i) => i,
                    Err(e) => return Return(e),
                }
            } else {
                x
            };
            sol.row_dual[i_row] = dual;
        }
    }
    for _ in 0..3 {
        if !ignore(s) {
            return Return(Status::Ok);
        }
    }
    Return(read_basis_stream(log, names, basis, s))
}

/// The file's bytes, or None if it cannot be opened (an unreadable one,
/// e.g. a directory, reads as empty, as an ifstream)
fn file_bytes(name: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    #[cfg(unix)]
    let path = {
        use std::os::unix::ffi::OsStrExt;
        std::path::PathBuf::from(std::ffi::OsStr::from_bytes(name))
    };
    #[cfg(not(unix))]
    let path = std::path::PathBuf::from(String::from_utf8_lossy(name).into_owned());
    let mut f = std::fs::File::open(path).ok()?;
    let mut v = Vec::new();
    if f.read_to_end(&mut v).is_err() {
        v.clear();
    }
    Some(v)
}

/// What C++ passes the readers (HighsLpUtilsRust.cpp)
#[repr(C)]
pub struct CRead {
    pub log: Log,
    pub names: CNames,
    pub filename: RsMut<u8>,
    pub basis_valid: *mut bool,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
}

/// readBasisFile
///
/// # Safety
/// The views must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_read_basis_file(r: *const CRead) -> i32 {
    let r = &*r;
    let filename = r.filename.get();
    let Some(bytes) = file_bytes(filename) else {
        log_user!(
            r.log,
            LogType::Error,
            "readBasisFile: Cannot open readable file \"%s\"\n",
            String::from_utf8_lossy(filename).as_ref()
        );
        return Status::Error as i32;
    };
    let names = Names { have_col: r.names.have_col, have_row: r.names.have_row, host: &r.names };
    let mut basis = BasisMut { valid: &mut *r.basis_valid, col_status: r.col_status.get_mut(), row_status: r.row_status.get_mut() };
    read_basis_stream(&r.log, &names, &mut basis, &mut IStream::new(&bytes)) as i32
}

/// The rest of readSolutionFile's input
#[repr(C)]
pub struct CReadSolution {
    pub style_sparse: bool,
    /// Whether the matrix is column-wise
    pub colwise: bool,
    pub a_start: RsMut<i32>,
    pub a_index: RsMut<i32>,
    pub a_value: RsMut<f64>,
    pub value_valid: *mut bool,
    pub col_value: RsMut<f64>,
    pub col_dual: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub row_dual: RsMut<f64>,
}

/// readSolutionFile after the style check: -2 for
/// readSolutionFileErrorReturn, otherwise the HighsStatus of
/// readSolutionFileReturn (C++ takes the read solution and basis on kOk)
///
/// # Safety
/// The views must be valid, as C++ passes them
#[no_mangle]
pub unsafe extern "C" fn highs_rs_read_solution_file(r: *const CRead, c: *const CReadSolution) -> i32 {
    let (r, c) = (&*r, &*c);
    let filename = r.filename.get();
    let Some(bytes) = file_bytes(filename) else {
        log_user!(
            r.log,
            LogType::Error,
            "readSolutionFile: Cannot open readable file \"%s\"\n",
            String::from_utf8_lossy(filename).as_ref()
        );
        return Status::Error as i32;
    };
    let names = Names { have_col: r.names.have_col, have_row: r.names.have_row, host: &r.names };
    let mut basis = BasisMut { valid: &mut *r.basis_valid, col_status: r.col_status.get_mut(), row_status: r.row_status.get_mut() };
    let a = Matrix { start: c.a_start.get(), index: c.a_index.get(), value: c.a_value.get() };
    let mut sol = SolutionMut {
        col_value: c.col_value.get_mut(),
        col_dual: c.col_dual.get_mut(),
        row_value: c.row_value.get_mut(),
        row_dual: c.row_dual.get_mut(),
    };
    match read_solution_file(
        &r.log,
        &names,
        c.colwise.then_some(&a),
        c.style_sparse,
        &mut sol,
        &mut *c.value_valid,
        &mut basis,
        &mut IStream::new(&bytes),
    ) {
        SolutionRead::Error => -2,
        SolutionRead::Return(s) => s as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn istream() {
        let mut s = IStream::new(b" ab 12 -3.5e1 x\n  1.5abc 7");
        let (mut w, mut i, mut d) = (Vec::new(), 0, 0.0);
        s.word(&mut w);
        assert_eq!(w, b"ab");
        s.int(&mut i);
        assert_eq!(i, 12);
        s.double(&mut d);
        assert_eq!(d, -35.0);
        s.ignore(80);
        s.double(&mut d);
        // libc++ accumulates "1.5abc", which strtod does not read whole
        assert_eq!((d, s.fail, s.eof), (0.0, true, false));
        s.int(&mut i);
        assert_eq!(i, 12);
        let mut s = IStream::new(b"5");
        s.int(&mut i);
        assert_eq!((i, s.eof, s.fail), (5, true, false));
        s.word(&mut w);
        assert_eq!((w.as_slice(), s.fail), (&b"ab"[..], true));
        let mut s = IStream::new(b"99999999999 x");
        s.int(&mut i);
        assert_eq!((i, s.fail), (i32::MAX, true));
    }
}
