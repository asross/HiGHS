//! Free-format MPS reader: a port of HMpsFF.cpp. C++ reads the file
//! (decompressing .gz through zstr) and hands the bytes over; `read` parses
//! them into the same model, warnings and status as HMpsFF::loadProblem,
//! including its quirks (see the comments), so the two are interchangeable.
//! Names are slices of the input, so nothing is copied per token.

use std::collections::hash_map::{Entry, HashMap};
use std::time::Instant;

/// FreeFormatParserReturnCode
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Status {
    Success = 0,
    ParserError = 1,
    FixedFormat = 3,
    Timeout = 4,
}

/// Message kinds: HighsLogType values, and 0 for highsLogDev(kInfo).
pub const LOG_DEV: i32 = 0;
pub const LOG_INFO: i32 = 1;
pub const LOG_WARNING: i32 = 4;
pub const LOG_ERROR: i32 = 5;

/// HighsVarType values
pub(super) const CONTINUOUS: u8 = 0;
pub(super) const INTEGER: u8 = 1;
pub(super) const SEMI_CONTINUOUS: u8 = 2;
pub(super) const SEMI_INTEGER: u8 = 3;

const INF: f64 = f64::INFINITY;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Key {
    Name,
    Objsense,
    Max,
    Min,
    Rows,
    Cols,
    Rhs,
    Bounds,
    Ranges,
    Qsection,
    Qmatrix,
    Quadobj,
    Qcmatrix,
    Csection,
    Delayedrows,
    Modelcuts,
    Usercuts,
    Indicators,
    Sets,
    Sos,
    Gencons,
    Pwlobj,
    Pwlnam,
    Pwlcon,
    None,
    End,
    Fail,
    FixedFormat,
    Timeout,
}

const KEYWORDS: &[(&[u8], Key)] = &[
    (b"NAME", Key::Name),
    (b"OBJSENSE", Key::Objsense),
    (b"ROWS", Key::Rows),
    (b"COLUMNS", Key::Cols),
    (b"RHS", Key::Rhs),
    (b"BOUNDS", Key::Bounds),
    (b"RANGES", Key::Ranges),
    (b"QSECTION", Key::Qsection),
    (b"QMATRIX", Key::Qmatrix),
    (b"QUADOBJ", Key::Quadobj),
    (b"QCMATRIX", Key::Qcmatrix),
    (b"CSECTION", Key::Csection),
    (b"DELAYEDROWS", Key::Delayedrows),
    (b"MODELCUTS", Key::Modelcuts),
    (b"USERCUTS", Key::Usercuts),
    (b"INDICATORS", Key::Indicators),
    (b"SETS", Key::Sets),
    (b"SOS", Key::Sos),
    (b"GENCONS", Key::Gencons),
    (b"PWLOBJ", Key::Pwlobj),
    (b"PWLNAM", Key::Pwlnam),
    (b"PWLCON", Key::Pwlcon),
    (b"ENDATA", Key::End),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowType {
    Le,
    Eq,
    Ge,
}

/// A count of ignored entries whose warning is reported at counts 1, 2, 4, ...
struct Counter {
    n: usize,
    freq: usize,
}

impl Default for Counter {
    fn default() -> Self {
        Counter { n: 0, freq: 1 }
    }
}

impl Counter {
    fn hit(&mut self) -> bool {
        self.n += 1;
        let report = self.n % self.freq == 0;
        if report {
            self.freq *= 2;
        }
        report
    }
}

/// The parsed model, or the parser state when `status` is not Success.
pub struct Mps<'a> {
    pub status: Status,
    pub warning_issued: bool,
    pub messages: Vec<(i32, String)>,
    pub num_row: usize,
    pub num_col: usize,
    pub maximize: bool,
    pub offset: f64,
    pub cost_row_location: i32,
    pub a_start: Vec<i32>,
    pub a_index: Vec<i32>,
    pub a_value: Vec<f64>,
    pub col_cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    /// HighsVarType per column; empty unless some column is not continuous
    pub integrality: Vec<u8>,
    /// Square Hessian, column-wise; empty when there are no entries
    pub q_dim: usize,
    pub q_start: Vec<i32>,
    pub q_index: Vec<i32>,
    pub q_value: Vec<f64>,
    pub objective_name: &'a [u8],
    /// Cleared if a name is duplicated
    pub row_names: Vec<&'a [u8]>,
    pub col_names: Vec<&'a [u8]>,

    rest: &'a [u8],
    start_time: Instant,
    time_limit: f64,
    num_nz: usize,
    mps_name: &'a [u8],
    col_binary: Vec<bool>,
    duplicate_row: Option<(&'a [u8], i32, i32)>,
    duplicate_col: Option<(&'a [u8], i32, i32)>,
    has_obj_entry: bool,
    has_row_entry: Vec<bool>,
    row_type: Vec<RowType>,
    // (col, row, value) in column order
    entries: Vec<(i32, i32, f64)>,
    // (row, col, value)
    q_entries: Vec<(i32, i32, f64)>,
    coeffobj: Vec<(usize, f64)>,
    // Quadratic rows, SOS and cones are parsed only to be rejected
    has_qrows: bool,
    sos_names: Vec<&'a [u8]>,
    has_cones: bool,
    rowname2idx: HashMap<&'a [u8], i32>,
    colname2idx: HashMap<&'a [u8], i32>,
    section_args: &'a [u8],
}

fn is_ws(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ')
}

fn trim(l: &[u8]) -> &[u8] {
    let s = l.iter().position(|&b| !is_ws(b)).unwrap_or(l.len());
    let e = l.iter().rposition(|&b| !is_ws(b)).map_or(s, |e| e + 1);
    &l[s..e]
}

/// first_word and first_word_end of stringutil: the word starting at or
/// after `start`, and where it ends (the line length if there is none).
fn word_at(l: &[u8], start: usize) -> (&[u8], usize) {
    if start >= l.len() {
        return (&[], l.len());
    }
    let s = l[start..].iter().position(|&b| !is_ws(b)).map_or(l.len(), |p| start + p);
    let e = l[s..].iter().position(|&b| is_ws(b)).map_or(l.len(), |p| s + p);
    (&l[s..e], e)
}

fn is_end(l: &[u8], end: usize) -> bool {
    end >= l.len() || l[end..].iter().all(|&b| is_ws(b))
}

fn lossy(b: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(b)
}

/// printf's %g
pub(crate) fn g(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.into();
    }
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    let e = format!("{v:.5e}");
    let (mantissa, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exp.abs())
    } else {
        strip(&format!("{:.*}", (5 - exp) as usize, v))
    }
}

/// HMpsFF::getValue: atof, after replacing the first D (or, failing that,
/// the first d) by E for Fortran-style exponents. Never yields an error, so
/// the C++ NaN checks after it are dead and not ported.
fn get_value(word: &[u8]) -> f64 {
    let d = word.iter().position(|&b| b == b'D');
    match d.or_else(|| word.iter().position(|&b| b == b'd')) {
        None => atof(word),
        Some(i) => {
            let mut w = word.to_vec();
            w[i] = b'E';
            atof(&w)
        }
    }
}

/// Length of the longest prefix of `w` that is a plain decimal number
/// [+-]digits[.digits][e[+-]digits], or 0.
pub(super) fn decimal_prefix(w: &[u8]) -> usize {
    let digits = |mut i: usize| {
        while w.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        i
    };
    let mut i = usize::from(matches!(w.first(), Some(b'+' | b'-')));
    let int_end = digits(i);
    let mut num_digits = int_end - i;
    i = int_end;
    if w.get(i) == Some(&b'.') {
        let frac_end = digits(i + 1);
        num_digits += frac_end - i - 1;
        i = frac_end;
    }
    if num_digits == 0 {
        return 0;
    }
    if matches!(w.get(i), Some(b'e' | b'E')) {
        let j = i + 1 + usize::from(matches!(w.get(i + 1), Some(b'+' | b'-')));
        let exp_end = digits(j);
        if exp_end > j {
            i = exp_end;
        }
    }
    i
}

/// C's atof. A word that is wholly a decimal number (all real MPS data) is
/// parsed by Rust, which rounds correctly, as strtod does; anything else
/// (hex, inf, nan, trailing junk, empty) goes to libc for the same answer.
fn atof(w: &[u8]) -> f64 {
    if !w.is_empty() && decimal_prefix(w) == w.len() {
        // ASCII by construction
        return std::str::from_utf8(w).unwrap().parse().unwrap();
    }
    extern "C" {
        fn atof(s: *const std::ffi::c_char) -> f64;
    }
    let w = &w[..w.iter().position(|&b| b == 0).unwrap_or(w.len())];
    let s = std::ffi::CString::new(w).unwrap();
    // SAFETY: s is a valid NUL-terminated string that outlives the call.
    unsafe { atof(s.as_ptr()) }
}

/// Parse a free-format MPS file's contents. A positive finite `time_limit`
/// (seconds) stops the parse with Status::Timeout.
pub fn read(input: &[u8], time_limit: f64) -> Mps<'_> {
    let mut mps = Mps {
        status: Status::Success,
        warning_issued: false,
        messages: Vec::new(),
        num_row: 0,
        num_col: 0,
        maximize: false,
        offset: 0.0,
        cost_row_location: -1,
        a_start: Vec::new(),
        a_index: Vec::new(),
        a_value: Vec::new(),
        col_cost: Vec::new(),
        col_lower: Vec::new(),
        col_upper: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        integrality: Vec::new(),
        q_dim: 0,
        q_start: Vec::new(),
        q_index: Vec::new(),
        q_value: Vec::new(),
        objective_name: b"",
        row_names: Vec::new(),
        col_names: Vec::new(),
        rest: input,
        start_time: Instant::now(),
        time_limit,
        num_nz: 0,
        mps_name: b"",
        col_binary: Vec::new(),
        duplicate_row: None,
        duplicate_col: None,
        has_obj_entry: false,
        has_row_entry: Vec::new(),
        row_type: Vec::new(),
        entries: Vec::new(),
        q_entries: Vec::new(),
        coeffobj: Vec::new(),
        has_qrows: false,
        sos_names: Vec::new(),
        has_cones: false,
        rowname2idx: HashMap::new(),
        colname2idx: HashMap::new(),
        section_args: b"",
    };
    mps.status = mps.load();
    mps
}

impl<'a> Mps<'a> {
    fn log(&mut self, kind: i32, msg: String) {
        self.messages.push((kind, msg));
    }

    /// HMpsFF::loadProblem, less the copy into HighsModel
    fn load(&mut self) -> Status {
        let status = self.parse();
        if status != Status::Success {
            return status;
        }
        if self.has_qrows {
            self.log(LOG_ERROR, "Quadratic rows not supported by HiGHS\n".into());
            return Status::ParserError;
        }
        if !self.sos_names.is_empty() {
            self.log(LOG_ERROR, "SOS not supported by HiGHS\n".into());
            return Status::ParserError;
        }
        if self.has_cones {
            self.log(LOG_ERROR, "Cones not supported by HiGHS\n".into());
            return Status::ParserError;
        }
        // Duplicate row and column names in MPS files occur if the same row
        // name appears twice in the ROWS section, or if a column name
        // reoccurs in the COLUMNS section after another column has been
        // defined. Some solvers only warn, so HiGHS does the same: the
        // duplicates are distinct rows (columns), and the names array,
        // being invalid, is cleared. Values in other sections can only
        // be given for the first occurrence of a name.
        if let Some((name, i0, i1)) = self.duplicate_row {
            self.warning_issued = true;
            let msg = format!("Linear constraints {i0} and {i1} have the same name \"{}\"\n", lossy(name));
            self.log(LOG_WARNING, msg);
            self.row_names.clear();
        }
        if let Some((name, i0, i1)) = self.duplicate_col {
            self.warning_issued = true;
            let msg = format!("Variables {i0} and {i1} have the same name \"{}\"\n", lossy(name));
            self.log(LOG_WARNING, msg);
            self.col_names.clear();
        }
        self.col_cost = vec![0.0; self.num_col];
        for &(col, cost) in &self.coeffobj {
            self.col_cost[col] = cost;
        }
        if !self.fill_matrix() {
            return Status::ParserError;
        }
        self.fill_hessian();
        if self.integrality.iter().all(|&t| t == CONTINUOUS) {
            self.integrality.clear();
        }
        Status::Success
    }

    fn fill_matrix(&mut self) -> bool {
        let num_nz = self.num_nz;
        if self.entries.len() != num_nz {
            return false;
        }
        let num_col = self.num_col;
        self.a_value = self.entries.iter().map(|e| e.2).collect();
        self.a_index = self.entries.iter().map(|e| e.1).collect();
        self.a_start = vec![0; num_col + 1];
        // Nothing more to do if there are no entries in the matrix
        let Some(&(first_col, _, _)) = self.entries.first() else {
            return true;
        };
        let mut new_col = first_col as usize;
        for (k, &(col, _, _)) in self.entries.iter().enumerate() {
            let col = col as usize;
            if col != new_col {
                let num_empty_cols = col.wrapping_sub(new_col);
                new_col = col;
                if new_col >= num_col {
                    return false;
                }
                // Columns skipped since the last nonzero are empty
                for c in new_col + 1 - num_empty_cols..=new_col {
                    self.a_start[c] = k as i32;
                }
            }
        }
        for c in new_col + 1..=num_col {
            self.a_start[c] = num_nz as i32;
        }
        if self.a_start.windows(2).any(|w| w[0] > w[1]) {
            self.log(LOG_ERROR, "Non-monotonic starts in MPS file reader\n".into());
            return false;
        }
        true
    }

    fn fill_hessian(&mut self) {
        if self.q_entries.is_empty() {
            self.q_dim = 0;
            return;
        }
        let dim = self.num_col;
        self.q_dim = dim;
        // Count the entries in each column, then use the counts as
        // pointers to the next entry to be filled in each column
        let mut q_length = vec![0i32; dim];
        for &(_, col, _) in &self.q_entries {
            q_length[col as usize] += 1;
        }
        self.q_start = vec![0; dim + 1];
        for col in 0..dim {
            self.q_start[col + 1] = self.q_start[col] + q_length[col];
            q_length[col] = self.q_start[col];
        }
        self.q_index = vec![0; self.q_entries.len()];
        self.q_value = vec![0.0; self.q_entries.len()];
        for &(row, col, value) in &self.q_entries {
            let k = q_length[col as usize] as usize;
            self.q_index[k] = row;
            self.q_value[k] = value;
            q_length[col as usize] += 1;
        }
    }

    /// The next line that is not blank or a comment, trimmed (getMpsLine
    /// plus the timeout check of each section loop). Err(Fail) at the end
    /// of the file.
    fn next_line(&mut self) -> Result<&'a [u8], Key> {
        loop {
            if self.rest.is_empty() {
                return Err(Key::Fail);
            }
            let (line, rest) = match self.rest.iter().position(|&b| b == b'\n') {
                Some(i) => (&self.rest[..i], &self.rest[i + 1..]),
                None => (self.rest, &[][..]),
            };
            self.rest = rest;
            // Only a '*' in the first column marks a comment
            if line.first() == Some(&b'*') {
                continue;
            }
            let line = trim(line);
            if line.is_empty() {
                continue;
            }
            if self.time_limit < INF && self.start_time.elapsed().as_secs_f64() > self.time_limit {
                return Err(Key::Timeout);
            }
            return Ok(line);
        }
    }

    /// HMpsFF::checkFirstWord on a trimmed line: the key, first word, and
    /// its end.
    fn check_first_word(&mut self, l: &'a [u8]) -> (Key, &'a [u8], usize) {
        if l.len() == 1 || is_ws(l[1]) {
            return (Key::None, &l[..1], 1);
        }
        let end = word_at(l, 1).1;
        let word = &l[..end];
        // Keywords are read as if they were in upper case
        let is = |k: &[u8]| word.eq_ignore_ascii_case(k);
        // Store the rest of the line for keywords that have arguments
        if is(b"QCMATRIX") || is(b"QSECTION") || is(b"CSECTION") {
            self.section_args = &l[end..];
        }
        let key = if let Some(&(_, key)) = KEYWORDS.iter().find(|(k, _)| is(k)) {
            key
        } else if word.len() >= 3 && word[..3].eq_ignore_ascii_case(b"MAX") {
            Key::Max
        } else if word.len() >= 3 && word[..3].eq_ignore_ascii_case(b"MIN") {
            Key::Min
        } else {
            return (Key::None, word, end);
        };
        // Keywords can be used as column names or names of RHS, BOUND,
        // RANGES etc, so assume this if there are non-blanks after the
        // apparent keyword. Only NAME, OBJSENSE, QCMATRIX, QSECTION and
        // CSECTION can be followed by text
        if matches!(key, Key::Name | Key::Objsense | Key::Qcmatrix | Key::Qsection | Key::Csection)
            || is_end(l, end)
        {
            return (key, word, end);
        }
        (Key::None, word, end)
    }

    /// Index of a column, adding it (continuous, default bounds) if new
    fn get_col_idx(&mut self, name: &'a [u8]) -> i32 {
        if let Some(&idx) = self.colname2idx.get(name) {
            return idx;
        }
        let idx = self.num_col as i32;
        self.colname2idx.insert(name, idx);
        self.num_col += 1;
        self.col_names.push(name);
        self.integrality.push(CONTINUOUS);
        self.col_binary.push(false);
        self.col_lower.push(0.0);
        self.col_upper.push(INF);
        idx
    }

    fn parse(&mut self) -> Status {
        let mut key = Key::None;
        while !matches!(key, Key::Fail | Key::End | Key::Timeout) {
            if self.cannot_parse_section(key) {
                return Status::ParserError;
            }
            key = match key {
                Key::Objsense => self.parse_objsense(),
                Key::Rows => self.parse_rows(),
                Key::Cols => self.parse_cols(),
                Key::Rhs => self.parse_rhs(),
                Key::Bounds => self.parse_bounds(),
                Key::Ranges => self.parse_ranges(),
                Key::Qmatrix => self.parse_quad_matrix("QMATRIX", true),
                Key::Quadobj => self.parse_quad_matrix("QUADOBJ", true),
                Key::Qsection => self.parse_quad_rows("QSECTION"),
                Key::Qcmatrix => self.parse_quad_rows("QCMATRIX"),
                Key::Csection => self.parse_cones(),
                Key::Sets | Key::Sos => self.parse_sos(key),
                Key::FixedFormat => return Status::FixedFormat,
                _ => self.parse_default(),
            };
        }
        // Assign bounds to columns that remain binary by default
        for col in 0..self.num_col {
            if self.col_binary[col] {
                self.col_lower[col] = 0.0;
                self.col_upper[col] = 1.0;
            }
        }
        match key {
            Key::Fail => Status::ParserError,
            Key::Timeout => Status::Timeout,
            _ => Status::Success,
        }
    }

    fn cannot_parse_section(&mut self, key: Key) -> bool {
        let section = match key {
            Key::Delayedrows => "DELAYEDROWS",
            Key::Modelcuts => "MODELCUTS",
            Key::Usercuts => "USERCUTS",
            Key::Indicators => "INDICATORS",
            Key::Gencons => "GENCONS",
            Key::Pwlobj => "PWLOBJ",
            Key::Pwlnam => "PWLNAM",
            Key::Pwlcon => "PWLCON",
            _ => return false,
        };
        self.log(LOG_ERROR, format!("MPS file reader cannot parse {section} section\n"));
        true
    }

    fn parse_default(&mut self) -> Key {
        let l = match self.next_line() {
            Ok(l) => l,
            Err(key) => return key,
        };
        let (key, _, end) = self.check_first_word(l);
        if key == Key::Name {
            // Save name of the MPS file
            if end < l.len() {
                self.mps_name = word_at(l, end).0;
            }
            self.log(LOG_DEV, "readMPS: Read NAME    OK\n".into());
            return Key::None;
        }
        // Look for Gurobi-style definition of MAX/MIN on OBJSENSE line.
        // Return the key anyway, in case there's a redefinition of
        // OBJSENSE on the "proper" line. If there's no such line, the
        // ROWS keyword is read OK
        if key == Key::Objsense && end < l.len() {
            let sense = word_at(l, end).0;
            if sense.eq_ignore_ascii_case(b"MAX") {
                self.maximize = true;
            } else if sense.eq_ignore_ascii_case(b"MIN") {
                self.maximize = false;
            }
        }
        key
    }

    fn parse_objsense(&mut self) -> Key {
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let key = self.check_first_word(l).0;
            match key {
                Key::Max => self.maximize = true,
                Key::Min => self.maximize = false,
                _ => {
                    self.log(LOG_DEV, "readMPS: Read OBJSENSE OK\n".into());
                    if key != Key::None {
                        return key;
                    }
                }
            }
        }
    }

    fn parse_rows(&mut self) -> Key {
        let mut has_obj = false;
        self.objective_name = b"Objective";
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(Key::Fail) => {
                    // As in the C++, which names the wrong section
                    self.log(LOG_ERROR, "Anomalous exit when parsing BOUNDS section of MPS file\n".into());
                    return Key::Fail;
                }
                Err(key) => return key,
            };
            let key = self.check_first_word(l).0;
            if key != Key::None {
                self.log(LOG_DEV, "readMPS: Read ROWS    OK\n".into());
                if !has_obj {
                    self.warning_issued = true;
                    self.log(LOG_WARNING, "No objective row found\n".into());
                    self.rowname2idx.entry(b"artificial_empty_objective").or_insert(-1);
                }
                return key;
            }
            let mut is_obj = false;
            let mut is_free = false;
            let (lower, upper, row_type) = match l[0] {
                b'G' => (0.0, INF, RowType::Ge),
                b'E' => (0.0, 0.0, RowType::Eq),
                b'L' => (-INF, 0.0, RowType::Le),
                b'N' => {
                    if !has_obj {
                        is_obj = true;
                        has_obj = true;
                        self.cost_row_location = self.num_row as i32;
                    } else {
                        is_free = true;
                    }
                    (0.0, 0.0, RowType::Eq)
                }
                _ => {
                    let msg = format!("Entry \"{}\" in ROWS section of MPS file is unidentified\n", lossy(l));
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
            };
            if l[0] != b'N' {
                self.row_lower.push(lower);
                self.row_upper.push(upper);
                self.row_type.push(row_type);
            }
            let (rowname, rowname_end) = word_at(l, 1);
            // Detect if file is in fixed format.
            if !is_end(l, rowname_end) {
                return if trim(&l[1..]).len() > 8 { Key::Fail } else { Key::FixedFormat };
            }
            // Free rows are not added to the matrix: in rowname2idx, -1 is
            // the objective and -2 all the free rows
            if is_free {
                self.rowname2idx.entry(rowname).or_insert(-2);
                continue;
            }
            let idx = if is_obj { -1 } else { self.num_row as i32 };
            if !is_obj {
                self.num_row += 1;
            }
            let inserted = match self.rowname2idx.entry(rowname) {
                Entry::Vacant(v) => {
                    v.insert(idx);
                    true
                }
                Entry::Occupied(_) => false,
            };
            if is_obj {
                self.objective_name = rowname;
            } else {
                self.row_names.push(rowname);
            }
            if !inserted && self.duplicate_row.is_none() {
                let first = self.rowname2idx[rowname];
                self.duplicate_row = Some((rowname, first, self.num_row as i32 - 1));
            }
        }
    }

    fn parse_cols(&mut self) -> Key {
        let mut cols = Cols {
            value: vec![0.0; self.num_row],
            index: Vec::new(),
            cost: 0.0,
            ignored_row_name: Counter::default(),
            ignored_duplicate_cost: Counter::default(),
            ignored_duplicate_nz: Counter::default(),
        };
        let mut colname: &[u8] = b"";
        let mut integral_cols = false;
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, word, end) = self.check_first_word(l);
            if key != Key::None {
                if self.num_col > 0 {
                    cols.flush(self);
                }
                let (r, c, n) = (
                    cols.ignored_row_name.n,
                    cols.ignored_duplicate_cost.n,
                    cols.ignored_duplicate_nz.n,
                );
                // Overwrites, so can clear a warning from an earlier section
                self.warning_issued = r > 0 || c > 0 || n > 0;
                if self.warning_issued {
                    let msg = format!("COLUMNS section: ignored {r} undefined rows {c} duplicate cost values and {n} duplicate matrix values\n");
                    self.log(LOG_WARNING, msg);
                }
                self.log(LOG_DEV, "readMPS: Read COLUMNS OK\n".into());
                return key;
            }
            // Check for integrality marker
            let (marker, end_marker) = word_at(l, end);
            if marker == b"'MARKER'" {
                let marker = word_at(l, end_marker).0;
                if (integral_cols && marker != b"'INTEND'") || (!integral_cols && marker != b"'INTORG'") {
                    self.log(LOG_ERROR, "Integrality marker error in COLUMNS section of MPS file\n".into());
                    return Key::Fail;
                }
                integral_cols = !integral_cols;
                continue;
            }
            // Detect whether the file is in fixed format with spaces in
            // names. end_marker, the end of the row name, would be more
            // than 9 for 8-character names, but free format MPS can have
            // names with only one character (pyomo.mps), so assume short
            // names if marker is a row name
            if end_marker < 9 && !self.rowname2idx.contains_key(marker) {
                let name = trim(&l[..l.len().min(10)]);
                if name.len() > 8 {
                    let msg = format!("Row name \"{}\" with spaces exceeds fixed format name length of 8\n", lossy(name));
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
                self.warning_issued = true;
                let msg = format!(
                    "Row name \"{}\" with spaces has length {}, so assume fixed format\n",
                    lossy(name),
                    name.len()
                );
                self.log(LOG_WARNING, msg);
                return Key::FixedFormat;
            }
            // Test for new column
            if word != colname {
                if self.num_col > 0 {
                    cols.flush(self);
                }
                colname = word;
                let idx = self.num_col as i32;
                self.num_col += 1;
                self.col_names.push(colname);
                if let Entry::Vacant(v) = self.colname2idx.entry(colname) {
                    v.insert(idx);
                } else if self.duplicate_col.is_none() {
                    self.duplicate_col = Some((colname, self.colname2idx[colname], idx));
                }
                self.integrality.push(if integral_cols { INTEGER } else { CONTINUOUS });
                // Integer columns from markers are binary by default
                self.col_binary.push(integral_cols);
                self.col_lower.push(0.0);
                self.col_upper.push(INF);
            }
            let (value, end) = word_at(l, end_marker);
            if value.is_empty() {
                let msg = format!("No coefficient given for column \"{}\"\n", lossy(marker));
                self.log(LOG_ERROR, msg);
                return Key::Fail;
            }
            cols.add(self, colname, marker, value, marker);
            if !is_end(l, end) {
                // Second coefficient. As in the C++, the character after
                // the row name is skipped, and a missing value reads as 0
                let (marker, end_marker) = word_at(l, end);
                let value = word_at(l, end_marker + 1).0;
                let objective_name = self.objective_name;
                cols.add(self, colname, marker, value, objective_name);
            }
        }
    }

    fn parse_rhs(&mut self) -> Key {
        // Track duplicate entries
        self.has_row_entry = vec![false; self.num_row];
        self.has_obj_entry = false;
        let mut ignored_row_name = Counter::default();
        let mut ignored_duplicate = Counter::default();
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, word, mut end) = self.check_first_word(l);
            if key != Key::None && key != Key::Rhs {
                let (r, d) = (ignored_row_name.n, ignored_duplicate.n);
                self.warning_issued = r > 0 || d > 0;
                if self.warning_issued {
                    let msg = format!("RHS section: ignored {r} undefined rows and {d} duplicate values\n");
                    self.log(LOG_WARNING, msg);
                }
                self.log(LOG_DEV, "readMPS: Read RHS     OK\n".into());
                return key;
            }
            // Ignore lack of name for SIF format; we know we have this
            // case when "word" is a row name
            if key == Key::None && self.rowname2idx.contains_key(word) {
                end = 0;
            }
            let (mut marker, end_marker) = word_at(l, end);
            let (mut word, mut end) = word_at(l, end_marker);
            if word.is_empty() {
                self.log(LOG_ERROR, format!("No bound given for row \"{}\"\n", lossy(marker)));
                return Key::Fail;
            }
            let mut row = self.rowname2idx.get(marker).copied();
            // SIF format sometimes has the name of the MPS file prepended
            // to the RHS entry; remove it here if that's the case
            if row.is_none() && marker == self.mps_name {
                marker = word;
                (word, end) = word_at(l, end);
                if word.is_empty() {
                    self.log(LOG_ERROR, format!("No bound given for SIF row \"{}\"\n", lossy(marker)));
                    return Key::Fail;
                }
                row = self.rowname2idx.get(marker).copied();
            }
            self.rhs_entry(row, marker, word, &mut ignored_row_name, &mut ignored_duplicate);
            if !is_end(l, end) {
                // Second coefficient, skipping the character after the
                // row name as in the C++
                let (marker, end_marker) = word_at(l, end);
                let word = word_at(l, end_marker + 1).0;
                let row = self.rowname2idx.get(marker).copied();
                self.rhs_entry(row, marker, word, &mut ignored_row_name, &mut ignored_duplicate);
            }
        }
    }

    fn rhs_entry(
        &mut self,
        row: Option<i32>,
        marker: &[u8],
        word: &[u8],
        ignored_row_name: &mut Counter,
        ignored_duplicate: &mut Counter,
    ) {
        let Some(row) = row else {
            if ignored_row_name.hit() {
                let msg = format!("Row name \"{}\" in RHS section is not defined: ignored\n", lossy(marker));
                self.log(LOG_WARNING, msg);
            }
            return;
        };
        let value = get_value(word);
        // Free rows (-2) share the objective's entry flag, and their RHS
        // sets the objective offset, as in the C++
        let has_entry = if row > -1 { self.has_row_entry[row as usize] } else { self.has_obj_entry };
        if has_entry {
            if ignored_duplicate.hit() {
                let msg = format!(
                    "Row name \"{}\" in RHS section has duplicate value {}: ignored\n",
                    lossy(marker),
                    g(value)
                );
                self.log(LOG_WARNING, msg);
            }
        } else if row > -1 {
            let r = row as usize;
            let t = self.row_type[r];
            if t == RowType::Eq || t == RowType::Le {
                self.row_upper[r] = value;
            }
            if t == RowType::Eq || t == RowType::Ge {
                self.row_lower[r] = value;
            }
            self.has_row_entry[r] = true;
        } else {
            // Objective shift
            self.offset = -value;
            self.has_obj_entry = true;
        }
    }

    fn parse_bounds(&mut self) -> Key {
        let mut has_lower = vec![false; self.num_col];
        let mut has_upper = vec![false; self.num_col];
        // MI, PL, BV, LI, UI, SI, SC
        let mut counts = [0usize; 7];
        let mut ignored_duplicate = Counter::default();
        let mut fractional_integer = Counter::default();
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, word, end) = self.check_first_word(l);
            if key != Key::None {
                for (count, t) in counts.iter().zip(["MI", "PL", "BV", "LI", "UI", "SI", "SC"]) {
                    if *count > 0 {
                        let msg = format!("Number of {t} entries in BOUNDS section is {count}\n");
                        self.log(LOG_INFO, msg);
                    }
                }
                let (d, f) = (ignored_duplicate.n, fractional_integer.n);
                self.warning_issued = d > 0 || f > 0;
                if self.warning_issued {
                    let msg = format!(
                        "BOUNDS section: ignored {d} duplicate values and {f} fractional integer bounds\n"
                    );
                    self.log(LOG_WARNING, msg);
                }
                self.log(LOG_DEV, "readMPS: Read BOUNDS  OK\n".into());
                return key;
            }
            // (is_lb, is_ub, is_integral, is_semi, is_defaultbound)
            let (is_lb, is_ub, is_integral, is_semi, is_default) = match word {
                b"UP" => (false, true, false, false, false),
                b"LO" => (true, false, false, false, false),
                b"FX" => (true, true, false, false, false),
                b"MI" => (true, false, false, false, true),
                b"PL" => (false, true, false, false, true),
                b"BV" => (true, true, true, false, true),
                b"LI" => (true, false, true, false, false),
                b"UI" => (false, true, true, false, false),
                b"FR" => (true, true, false, false, true),
                b"SI" => (false, true, true, true, false),
                b"SC" => (false, true, false, true, false),
                _ => {
                    let msg = format!("Entry in BOUNDS section of MPS file is of type \"{}\"\n", lossy(word));
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
            };
            if let Some(i) = [b"MI", b"PL", b"BV", b"LI", b"UI", b"SI", b"SC"].iter().position(|t| *t == word) {
                counts[i] += 1;
            }
            let (bound_name, end_bound_name) = word_at(l, end);
            // SIF format might not have the bound name, so skip it if it
            // is a column name
            let (marker, end_marker) = if self.colname2idx.contains_key(bound_name) {
                (bound_name, end_bound_name)
            } else {
                word_at(l, end_bound_name)
            };
            let col = self.get_col_idx(marker) as usize;
            if col == has_lower.len() {
                has_lower.push(false);
                has_upper.push(false);
            }
            if (is_lb && has_lower[col]) || (is_ub && has_upper[col]) {
                if ignored_duplicate.hit() {
                    let msg = format!(
                        "Column name \"{}\" in BOUNDS section has duplicate {} bound definition: ignored\n",
                        lossy(marker),
                        if is_lb { "lower" } else { "upper" }
                    );
                    self.log(LOG_WARNING, msg);
                }
                continue;
            }
            if is_default {
                if is_integral {
                    // BV: integer and binary; the lower bound is untouched
                    self.integrality[col] = INTEGER;
                    self.col_binary[col] = true;
                    self.col_upper[col] = 1.0;
                } else {
                    // MI, PL or FR
                    self.col_binary[col] = false;
                    if is_lb {
                        self.col_lower[col] = -INF;
                    }
                    if is_ub {
                        self.col_upper[col] = INF;
                    }
                }
                has_lower[col] |= is_lb;
                has_upper[col] |= is_ub;
                continue;
            }
            // UP, LO, FX, LI, UI, SI or SC
            let word_value = word_at(l, end_marker).0;
            if word_value.is_empty() {
                let msg = format!("No bound given for {} row \"{}\"\n", lossy(word), lossy(marker));
                self.log(LOG_ERROR, msg);
                return Key::Fail;
            }
            let value = get_value(word_value);
            if is_integral {
                // LI, UI or SI, whose value should be integer. `as`
                // saturates, as does the C++ cast on arm64
                if value - f64::from(value as i32) != 0.0 && fractional_integer.hit() {
                    let msg = format!(
                        "Bound for LI/UI/SI column \"{}\" is {}: not integer\n",
                        lossy(marker),
                        g(value)
                    );
                    self.log(LOG_WARNING, msg);
                }
                self.integrality[col] = if is_semi { SEMI_INTEGER } else { INTEGER };
            } else if is_semi {
                self.integrality[col] = SEMI_CONTINUOUS;
            }
            if is_lb {
                self.col_lower[col] = value;
                has_lower[col] = true;
            }
            if is_ub {
                self.col_upper[col] = value;
                has_upper[col] = true;
            }
            self.col_binary[col] = false;
        }
    }

    fn parse_ranges(&mut self) -> Key {
        self.has_row_entry = vec![false; self.num_row];
        let mut ignored_row_name = Counter::default();
        let mut ignored_duplicate = Counter::default();
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, _, end) = self.check_first_word(l);
            if key != Key::None {
                let (r, d) = (ignored_row_name.n, ignored_duplicate.n);
                self.warning_issued = r > 0 || d > 0;
                if self.warning_issued {
                    let msg = format!("RANGES section: ignored {r} undefined/illegal rows and {d} duplicate values\n");
                    self.log(LOG_WARNING, msg);
                }
                self.log(LOG_DEV, "readMPS: Read RANGES  OK\n".into());
                return key;
            }
            let mut end = end;
            for pair in 0..2 {
                let (marker, end_marker) = word_at(l, end);
                let word;
                (word, end) = word_at(l, end_marker);
                if word.is_empty() {
                    self.log(LOG_ERROR, format!("No range given for row \"{}\"\n", lossy(marker)));
                    return Key::Fail;
                }
                self.range_entry(marker, word, &mut ignored_row_name, &mut ignored_duplicate);
                if is_end(l, end) {
                    break;
                }
                if pair == 1 {
                    let msg = format!("Unknown specifiers in RANGES section for row \"{}\"\n", lossy(marker));
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
            }
        }
    }

    fn range_entry(&mut self, marker: &[u8], word: &[u8], ignored_row_name: &mut Counter, ignored_duplicate: &mut Counter) {
        let Some(&row) = self.rowname2idx.get(marker) else {
            if ignored_row_name.hit() {
                let msg = format!("Row name \"{}\" in RANGES section is not defined: ignored\n", lossy(marker));
                self.log(LOG_WARNING, msg);
            }
            return;
        };
        if row < 0 {
            if ignored_row_name.hit() {
                let msg = format!("Row name \"{}\" in RANGES section is not valid: ignored\n", lossy(marker));
                self.log(LOG_WARNING, msg);
            }
            return;
        }
        let r = row as usize;
        let value = get_value(word);
        if self.has_row_entry[r] {
            if ignored_duplicate.hit() {
                let msg = format!(
                    "Row name \"{}\" in RANGES section has duplicate value {}: ignored\n",
                    lossy(marker),
                    g(value)
                );
                self.log(LOG_WARNING, msg);
            }
            return;
        }
        let t = self.row_type[r];
        if (t == RowType::Eq && value < 0.0) || t == RowType::Le {
            self.row_lower[r] = self.row_upper[r] - value.abs();
        } else if (t == RowType::Eq && value > 0.0) || t == RowType::Ge {
            self.row_upper[r] = self.row_lower[r] + value.abs();
        }
        self.has_row_entry[r] = true;
    }

    /// QSECTION or QCMATRIX: quadratic terms of a row, of the objective
    /// if the row is the objective
    fn parse_quad_rows(&mut self, section: &str) -> Key {
        let rowname = word_at(self.section_args, 0).0;
        if rowname.is_empty() {
            self.log(LOG_ERROR, format!("No row name given in argument of {section}\n"));
            return Key::Fail;
        }
        match self.rowname2idx.get(rowname).copied() {
            Some(-1) => self.parse_quad_matrix(section, true),
            Some(row) if row >= 0 => {
                // Quadratic rows are not supported: parse to report errors
                self.has_qrows = true;
                self.parse_quad_matrix(section, false)
            }
            row => {
                // Skip the section of an undefined or free row
                if row.is_none() {
                    self.warning_issued = true;
                    let msg = format!("Row name \"{}\" in {section} section is not defined: ignored\n", lossy(rowname));
                    self.log(LOG_WARNING, msg);
                }
                loop {
                    let l = match self.next_line() {
                        Ok(l) => l,
                        Err(key) => return key,
                    };
                    let key = self.check_first_word(l).0;
                    if key != Key::None {
                        self.log(LOG_DEV, format!("readMPS: Read {section}  OK\n"));
                        return key;
                    }
                }
            }
        }
    }

    /// Hessian entries, kept if `keep`. QMATRIX/QCMATRIX should have all
    /// entries, whereas every off-diagonal QUADOBJ/QSECTION entry also
    /// defines its entry in the opposite triangle, so add this explicitly
    /// to unify the record regardless of section
    fn parse_quad_matrix(&mut self, section: &str, keep: bool) -> Key {
        let triangular = section == "QUADOBJ" || section == "QSECTION";
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, col_name, mut end) = self.check_first_word(l);
            if key != Key::None {
                self.log(LOG_DEV, format!("readMPS: Read {section} OK\n"));
                return key;
            }
            let col = self.get_col_idx(col_name);
            // At most two entries per line
            for _ in 0..2 {
                let (row_name, end_row_name) = word_at(l, end);
                if row_name.is_empty() {
                    break;
                }
                let (coeff, end_coeff) = word_at(l, end_row_name);
                if coeff.is_empty() {
                    let msg = format!(
                        "{section} has no coefficient for entry \"{}\" in column \"{}\"\n",
                        lossy(row_name),
                        lossy(col_name)
                    );
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
                let row = self.get_col_idx(row_name);
                let coeff = get_value(coeff);
                if coeff != 0.0 && keep {
                    self.q_entries.push((row, col, coeff));
                    if triangular && row != col {
                        self.q_entries.push((col, row, coeff));
                    }
                }
                end = end_coeff;
                if end == l.len() {
                    break;
                }
            }
        }
    }

    /// CSECTION: cones are not supported, so parse only to report errors
    fn parse_cones(&mut self) -> Key {
        let args = self.section_args;
        // Cone name, optional parameter, cone type
        let (cone_name, end) = word_at(args, 0);
        if cone_name.is_empty() {
            self.log(LOG_ERROR, "Cone name missing in CSECTION\n".into());
            return Key::Fail;
        }
        let (second, end) = word_at(args, end);
        let third = word_at(args, end).0;
        let cone_type = if third.is_empty() { second } else { third };
        if cone_type.is_empty() {
            self.log(LOG_ERROR, format!("Cone type missing in CSECTION {}\n", lossy(trim(args))));
            return Key::Fail;
        }
        if ![&b"ZERO"[..], b"QUAD", b"RQUAD", b"PEXP", b"PPOW", b"DEXP", b"DPOW"].contains(&cone_type) {
            self.log(LOG_ERROR, format!("Unrecognized cone type {}\n", lossy(cone_type)));
            return Key::Fail;
        }
        self.has_cones = true;
        // One column per line
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, col_name, _) = self.check_first_word(l);
            if key != Key::None {
                self.log(LOG_DEV, "readMPS: Read CSECTION OK\n".into());
                return key;
            }
            self.get_col_idx(col_name);
        }
    }

    /// SETS or SOS: not supported, so parse only to report errors
    fn parse_sos(&mut self, section: Key) -> Key {
        loop {
            let l = match self.next_line() {
                Ok(l) => l,
                Err(key) => return key,
            };
            let (key, word, end) = self.check_first_word(l);
            if key != Key::None {
                self.log(LOG_DEV, "readMPS: Read SETS    OK\n".into());
                return key;
            }
            if word == b"S1" || word == b"S2" {
                // A new SOS is starting
                let sos_name = word_at(l, end).0;
                if sos_name.is_empty() {
                    self.log(LOG_ERROR, "No name given for SOS\n".into());
                    return Key::Fail;
                }
                self.sos_names.push(sos_name);
                continue;
            }
            // An SOS is continuing
            let Some(&current) = self.sos_names.last() else {
                let msg = format!("SOS type specification missing before {}.\n", lossy(l));
                self.log(LOG_ERROR, msg);
                return Key::Fail;
            };
            let col_name = if section == Key::Sos {
                // First word is the column name
                word
            } else {
                // SOS name, column name, weight; SOS definitions must be
                // contiguous
                if word != current {
                    let msg = format!(
                        "SOS specification for SOS {} mixed with SOS {}. This is currently not supported.\n",
                        lossy(current),
                        lossy(word)
                    );
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
                if is_end(l, end) {
                    let msg = format!("Missing variable in SOS specification line {}.\n", lossy(l));
                    self.log(LOG_ERROR, msg);
                    return Key::Fail;
                }
                word_at(l, end).0
            };
            self.get_col_idx(col_name);
        }
    }
}

/// Scatter of the current column in the COLUMNS section
struct Cols {
    value: Vec<f64>,
    index: Vec<usize>,
    cost: f64,
    ignored_row_name: Counter,
    ignored_duplicate_cost: Counter,
    ignored_duplicate_nz: Counter,
}

impl Cols {
    /// Record the nonzeros of the last column
    fn flush(&mut self, p: &mut Mps) {
        let col = p.num_col - 1;
        if self.cost != 0.0 {
            p.coeffobj.push((col, self.cost));
            self.cost = 0.0;
        }
        for &row in &self.index {
            p.entries.push((col as i32, row as i32, self.value[row]));
            self.value[row] = 0.0;
        }
        self.index.clear();
    }

    /// An entry `marker` (row name) `word` (value) of column `colname`;
    /// `cost_row_name` is the row named in a duplicate cost warning
    fn add(&mut self, p: &mut Mps, colname: &[u8], marker: &[u8], word: &[u8], cost_row_name: &[u8]) {
        let Some(&row) = p.rowname2idx.get(marker) else {
            if self.ignored_row_name.hit() {
                let msg = format!("Row name \"{}\" in COLUMNS section is not defined: ignored\n", lossy(marker));
                p.log(LOG_WARNING, msg);
            }
            return;
        };
        let value = get_value(word);
        // NaN counts as nonzero, as in the C++
        if value == 0.0 {
            return;
        }
        if row >= 0 {
            let r = row as usize;
            if self.value[r] != 0.0 {
                if self.ignored_duplicate_nz.hit() {
                    let msg = format!(
                        "Column \"{}\" has duplicate nonzero {} in row \"{}\": ignored\n",
                        lossy(colname),
                        g(value),
                        lossy(marker)
                    );
                    p.log(LOG_WARNING, msg);
                }
            } else {
                p.num_nz += 1;
                self.value[r] = value;
                self.index.push(r);
            }
        } else if row == -1 {
            if self.cost != 0.0 {
                if self.ignored_duplicate_cost.hit() {
                    let msg = format!(
                        "Column \"{}\" has duplicate nonzero {} in objective row \"{}\": ignored\n",
                        lossy(colname),
                        g(value),
                        lossy(cost_row_name)
                    );
                    p.log(LOG_WARNING, msg);
                }
            } else {
                self.cost = value;
            }
        }
    }
}

/// A borrowed array, as C++ reads it
#[repr(C)]
pub struct Slice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> Slice<T> {
    pub(super) fn new(s: &[T]) -> Self {
        Slice { ptr: s.as_ptr(), len: s.len() }
    }
}

#[repr(C)]
pub struct Message {
    pub(super) kind: i32,
    pub(super) text: Slice<u8>,
}

/// What C++ copies into the HighsModel; mirrored in FilereaderMps.cpp
#[repr(C)]
pub struct MpsView {
    status: i32,
    warning_issued: bool,
    maximize: bool,
    num_row: i32,
    num_col: i32,
    cost_row_location: i32,
    q_dim: i32,
    offset: f64,
    a_start: Slice<i32>,
    a_index: Slice<i32>,
    a_value: Slice<f64>,
    col_cost: Slice<f64>,
    col_lower: Slice<f64>,
    col_upper: Slice<f64>,
    row_lower: Slice<f64>,
    row_upper: Slice<f64>,
    integrality: Slice<u8>,
    q_start: Slice<i32>,
    q_index: Slice<i32>,
    q_value: Slice<f64>,
    objective_name: Slice<u8>,
    row_names: Slice<Slice<u8>>,
    col_names: Slice<Slice<u8>>,
    messages: Slice<Message>,
}

/// Owner of everything an MpsView points to
pub struct MpsHandle {
    _mps: Mps<'static>,
    _row_names: Vec<Slice<u8>>,
    _col_names: Vec<Slice<u8>>,
    _messages: Vec<Message>,
}

/// Parse `len` bytes at `buf` and fill `view`. Returns the handle owning
/// the view's arrays, to be freed with highs_rs_mps_free.
///
/// # Safety
/// `buf` must be valid for `len` bytes and outlive the handle (names point
/// into it); `view` must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mps_read(
    buf: *const u8,
    len: usize,
    time_limit: f64,
    view: *mut MpsView,
) -> *mut MpsHandle {
    let input: &'static [u8] = if len == 0 { &[] } else { std::slice::from_raw_parts(buf, len) };
    let mps = read(input, time_limit);
    let names = |n: &[&[u8]]| n.iter().map(|s| Slice::new(s)).collect::<Vec<_>>();
    let row_names = names(&mps.row_names);
    let col_names = names(&mps.col_names);
    let messages: Vec<Message> = mps
        .messages
        .iter()
        .map(|(kind, text)| Message { kind: *kind, text: Slice::new(text.as_bytes()) })
        .collect();
    *view = MpsView {
        status: mps.status as i32,
        warning_issued: mps.warning_issued,
        maximize: mps.maximize,
        num_row: mps.num_row as i32,
        num_col: mps.num_col as i32,
        cost_row_location: mps.cost_row_location,
        q_dim: mps.q_dim as i32,
        offset: mps.offset,
        a_start: Slice::new(&mps.a_start),
        a_index: Slice::new(&mps.a_index),
        a_value: Slice::new(&mps.a_value),
        col_cost: Slice::new(&mps.col_cost),
        col_lower: Slice::new(&mps.col_lower),
        col_upper: Slice::new(&mps.col_upper),
        row_lower: Slice::new(&mps.row_lower),
        row_upper: Slice::new(&mps.row_upper),
        integrality: Slice::new(&mps.integrality),
        q_start: Slice::new(&mps.q_start),
        q_index: Slice::new(&mps.q_index),
        q_value: Slice::new(&mps.q_value),
        objective_name: Slice::new(mps.objective_name),
        row_names: Slice::new(&row_names),
        col_names: Slice::new(&col_names),
        messages: Slice::new(&messages),
    };
    // Moving the Vecs into the box keeps their heap buffers in place
    Box::into_raw(Box::new(MpsHandle {
        _mps: mps,
        _row_names: row_names,
        _col_names: col_names,
        _messages: messages,
    }))
}

/// # Safety
/// `handle` must come from highs_rs_mps_read and not be freed already.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mps_free(handle: *mut MpsHandle) {
    drop(Box::from_raw(handle));
}

#[cfg(test)]
mod tests {
    use super::*;

    const LP: &str = "\
NAME          TINY
* comment
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
COLUMNS
    x         obj       1.0          c1        2.0
    x         c2        1D1
    MARKER    'MARKER'  'INTORG'
    y         obj       3            c3        -1
    y         c1        1
    MARKER    'MARKER'  'INTEND'
    z         c2        1.5
RHS
    RHS       obj       2.5          c1        4
    RHS       c2        1            c3        -2
RANGES
    RNG       c1        3            c3        -5
BOUNDS
 UP BND       x         8
 MI BND       z
 LI BND       w         2.5
ENDATA
";

    #[test]
    fn tiny_lp() {
        let m = read(LP.as_bytes(), INF);
        assert_eq!(m.status, Status::Success);
        assert!(m.maximize);
        assert_eq!(m.objective_name, b"obj");
        assert_eq!(m.row_names, [&b"c1"[..], b"c2", b"c3"]);
        assert_eq!(m.col_names, [&b"x"[..], b"y", b"z", b"w"]);
        assert_eq!(m.offset, -2.5);
        assert_eq!(m.col_cost, [1.0, 3.0, 0.0, 0.0]);
        assert_eq!(m.a_start, [0, 2, 4, 5, 5]);
        assert_eq!(m.a_index, [0, 1, 2, 0, 1]);
        assert_eq!(m.a_value, [2.0, 10.0, -1.0, 1.0, 1.5]);
        // L: [4-3, 4]; G: [1, inf]; E with negative range: [-2-5, -2]
        assert_eq!(m.row_lower, [1.0, 1.0, -7.0]);
        assert_eq!(m.row_upper, [4.0, INF, -2.0]);
        // y is binary by default; w is LI with a fractional bound
        assert_eq!(m.col_lower, [0.0, 0.0, -INF, 2.5]);
        assert_eq!(m.col_upper, [8.0, 1.0, INF, INF]);
        assert_eq!(m.integrality, [CONTINUOUS, INTEGER, CONTINUOUS, INTEGER]);
        assert!(m.warning_issued);
        assert!(m.messages.iter().any(|(k, s)| *k == LOG_WARNING && s.contains("is 2.5: not integer")));
    }

    #[test]
    fn errors_and_fixed_format() {
        assert_eq!(read(b"NAME x\nROWS\n N obj\n", INF).status, Status::ParserError);
        let bad = "ROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n XX BND x 1\nENDATA\n";
        let m = read(bad.as_bytes(), INF);
        assert_eq!(m.status, Status::ParserError);
        assert_eq!(m.messages.last().unwrap().1, "Entry in BOUNDS section of MPS file is of type \"XX\"\n");
        assert_eq!(read(b"ROWS\n N obj\n L c 1\nENDATA\n", INF).status, Status::FixedFormat);
    }

    #[test]
    fn printf_g() {
        assert_eq!(g(2.5), "2.5");
        assert_eq!(g(1e-5), "1e-05");
        assert_eq!(g(123456.0), "123456");
        assert_eq!(g(1234567.0), "1.23457e+06");
        assert_eq!(g(-0.0001), "-0.0001");
        assert_eq!(atof(b"1.5e"), 1.5);
        assert_eq!(atof(b"0x10"), 16.0);
    }
}
