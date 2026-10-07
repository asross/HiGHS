//! The command line of the highs app (app/HighsRuntimeOptions.h, which
//! used CLI11 2.5.0): CLI11's parse of exactly the options the app defines
//! (long options with one value, `--name=value`, the -v/--version,
//! --notice and -h/--help flags, the model file as the positional, `--`),
//! its conversions (strtoll/strtold with separators, 0o/0b prefixes and
//! trailing spaces), validators (existing file), errors with their
//! messages and exit codes, and its help text with the app's formatter
//! settings (HiGHS's CLI11 indents description words starting with `"`).
//! Arguments are bytes, as in C.

use std::ffi::c_char;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    File,
    Text,
    Int,
    Float,
    Flag,
}

struct Opt {
    names: &'static str,
    kind: Kind,
    desc: &'static str,
}

/// The app's options in CLI11's order (the help flag is re-added last)
const OPTS: [Opt; 18] = [
    Opt { names: "--model_file", kind: Kind::File, desc: "File of model to solve" },
    Opt { names: "--options_file", kind: Kind::File, desc: "File containing HiGHS options" },
    Opt { names: "--read_solution_file", kind: Kind::File, desc: "File of solution to read" },
    Opt { names: "--read_basis_file", kind: Kind::File, desc: "File of initial basis to read" },
    Opt { names: "--write_model_file", kind: Kind::Text, desc: "File for writing out model" },
    Opt { names: "--solution_file", kind: Kind::Text, desc: "File for writing out solution" },
    Opt { names: "--write_basis_file", kind: Kind::Text, desc: "File for writing out final basis" },
    Opt {
        names: "--presolve",
        kind: Kind::Text,
        desc: "Set presolve option to:\n\"choose\" * default\n\"on\"\n\"off\"",
    },
    Opt {
        names: "--solver",
        kind: Kind::Text,
        desc: "Set solver option to:\n\"choose\" * default\n\"simplex\"\n\"hipo\"\n\"ipm\"",
    },
    Opt {
        names: "--parallel",
        kind: Kind::Text,
        desc: "Set parallel option to:\n\"choose\" * default\n\"on\"\n\"off\"",
    },
    Opt {
        names: "--threads",
        kind: Kind::Int,
        desc: "Set maximum number of threads to use:\n0: automatic * default",
    },
    Opt {
        names: "--run_crossover",
        kind: Kind::Text,
        desc: "Set run_crossover option to:\n\"choose\"\n\"on\" * default\n\"off\"",
    },
    Opt { names: "--time_limit", kind: Kind::Float, desc: "Run time limit (seconds - double)" },
    Opt { names: "--random_seed", kind: Kind::Int, desc: "Seed to initialize random number\ngeneration" },
    Opt {
        names: "--ranging",
        kind: Kind::Text,
        desc: "Compute cost, bound, RHS and basic\nsolution ranging:\n\"on\"\n\"off\" * default",
    },
    Opt { names: "-v,--version", kind: Kind::Flag, desc: "Print version" },
    Opt { names: "--notice", kind: Kind::Flag, desc: "Print third-party information" },
    Opt { names: "-h,--help", kind: Kind::Flag, desc: "Print help" },
];
const MODEL_FILE: usize = 0;
pub const THREADS: usize = 10;
pub const TIME_LIMIT: usize = 12;
pub const RANDOM_SEED: usize = 13;
const VERSION: usize = 15;
const NOTICE: usize = 16;
const HELP: usize = 17;

impl Opt {
    fn long(&self) -> &'static str {
        self.names.split(',').find(|n| n.starts_with("--")).unwrap()
    }
    fn short(&self) -> Option<&'static str> {
        self.names.split(',').find(|n| !n.starts_with("--"))
    }
    /// CLI11's type name (for errors) and its help label
    fn type_name(&self) -> (&'static str, &'static str) {
        match self.kind {
            Kind::File => ("TEXT:FILE", "file"),
            Kind::Text => ("TEXT", "text"),
            Kind::Int => ("INT", "int"),
            Kind::Float => ("FLOAT", "float"),
            Kind::Flag => ("", ""),
        }
    }
}

/// How the parse ended: the app's catch clauses
pub enum Outcome {
    Ok,
    Help(Vec<u8>),
    Extras(Vec<u8>),
    Mismatch(Vec<u8>),
    /// Another CLI::ParseError: the message and the exit code
    Error(Vec<u8>, i32),
}

const EXIT_CONVERSION: i32 = 104;
const EXIT_VALIDATION: i32 = 105;

/// The values the app reads (HighsCommandLineOptions and the counts)
#[derive(Default)]
pub struct CommandLine {
    pub strings: [Vec<u8>; 15],
    pub threads: i32,
    pub random_seed: i32,
    pub time_limit: f64,
    pub version: bool,
    pub notice: bool,
    /// The number of results of each option (app.count)
    pub count: [usize; 18],
}

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Mark,
    Long,
    Short,
    None,
}

/// detail::valid_first_char
fn valid_first_char(c: u8) -> bool {
    c != b'-' && c > 33
}

/// detail::split_long: (name, value)
fn split_long(cur: &[u8]) -> Option<(&[u8], &[u8])> {
    if cur.len() > 2 && cur.starts_with(b"--") && valid_first_char(cur[2]) {
        Some(match cur.iter().position(|&c| c == b'=') {
            Some(l) => (&cur[2..l], &cur[l + 1..]),
            None => (&cur[2..], &b""[..]),
        })
    } else {
        None
    }
}

/// detail::split_short: (name, rest)
fn split_short(cur: &[u8]) -> Option<(&[u8], &[u8])> {
    if cur.len() > 1 && cur[0] == b'-' && valid_first_char(cur[1]) {
        Some((&cur[1..2], &cur[2..]))
    } else {
        None
    }
}

fn find_short(name: &[u8]) -> Option<usize> {
    OPTS.iter().position(|o| o.short().is_some_and(|s| &s.as_bytes()[1..] == name))
}

fn find_long(name: &[u8]) -> Option<usize> {
    OPTS.iter().position(|o| &o.long().as_bytes()[2..] == name)
}

/// App::_recognize (no subcommands)
fn recognize(cur: &[u8]) -> Class {
    if cur == b"--" {
        return Class::Mark;
    }
    if split_long(cur).is_some() {
        return Class::Long;
    }
    if let Some((name, _)) = split_short(cur) {
        if name[0].is_ascii_digit() && find_short(name).is_none() {
            return Class::None;
        }
        return Class::Short;
    }
    Class::None
}

/// Option::_add_result: the `[[x]]` escape of a doubled string
fn add_result(results: &mut Vec<Vec<u8>>, r: &[u8]) {
    let n = r.len();
    if n >= 4 && r.starts_with(b"[[") && r.ends_with(b"]]") {
        let mut s = vec![b'['];
        let mut duplicated = true;
        let mut i = 2;
        while i < n - 2 {
            if r[i] == r[i + 1] {
                s.push(r[i]);
            } else {
                duplicated = false;
                break;
            }
            i += 2;
        }
        if duplicated {
            s.push(b']');
            results.push(s);
            return;
        }
    }
    results.push(r.to_vec());
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// CLI11's parse of the app's command line (argv without the program)
pub fn parse(argv: &[Vec<u8>], program: &[u8], out: &mut CommandLine) -> Outcome {
    let mut results: Vec<Vec<Vec<u8>>> = vec![Vec::new(); OPTS.len()];
    // (is a `--` mark, argument) of App::missing_
    let mut missing: Vec<(bool, Vec<u8>)> = Vec::new();
    let mut args: Vec<Vec<u8>> = argv.iter().rev().cloned().collect();
    let mut positional_only = false;
    while let Some(cur) = args.last().cloned() {
        let class = if positional_only { Class::None } else { recognize(&cur) };
        match class {
            Class::Mark => {
                args.pop();
                positional_only = true;
                missing.push((true, cur));
            }
            Class::Long | Class::Short => {
                let (index, value, rest) = if class == Class::Long {
                    let (name, value) = split_long(&cur).unwrap();
                    (find_long(name), value.to_vec(), Vec::new())
                } else {
                    let (name, rest) = split_short(&cur).unwrap();
                    (find_short(name), Vec::new(), rest.to_vec())
                };
                args.pop();
                let Some(i) = index else {
                    missing.push((false, cur));
                    continue;
                };
                if OPTS[i].kind == Kind::Flag {
                    // get_flag_value: an empty value is "true"
                    let v = if value.is_empty() || value == b"{}" { b"true".to_vec() } else { value };
                    add_result(&mut results[i], &v);
                } else {
                    let mut collected = 0;
                    if !value.is_empty() {
                        add_result(&mut results[i], &value);
                        collected = 1;
                    }
                    if collected == 0 {
                        match args.pop() {
                            Some(a) => add_result(&mut results[i], &a),
                            None => {
                                let msg = cat(&[
                                    OPTS[i].long().as_bytes(),
                                    b": 1 required ",
                                    OPTS[i].type_name().0.as_bytes(),
                                    b" missing",
                                ]);
                                return Outcome::Mismatch(msg);
                            }
                        }
                    }
                }
                if !rest.is_empty() {
                    args.push(cat(&[b"-", &rest]));
                }
            }
            Class::None => {
                args.pop();
                if results[MODEL_FILE].is_empty() {
                    add_result(&mut results[MODEL_FILE], &cur);
                } else {
                    missing.push((false, cur));
                }
            }
        }
    }
    // _process_help_flags
    if !results[HELP].is_empty() {
        return Outcome::Help(help(program));
    }
    // _process_callbacks, in the options' order
    for (i, o) in OPTS.iter().enumerate() {
        let res = &results[i];
        out.count[i] = res.len();
        if res.is_empty() || i == HELP {
            continue;
        }
        let name = o.long().as_bytes();
        if o.kind == Kind::File {
            for r in res {
                if let Some(e) = existing_file(r) {
                    return Outcome::Error(cat(&[name, b": ", &e]), EXIT_VALIDATION);
                }
            }
        }
        // _reduce_results: flags take the last, options throw on more
        // than one (but for CLI11's {} %% empty-container trap)
        let value: &[u8] = if o.kind == Kind::Flag {
            res.last().unwrap()
        } else {
            if res.len() > 1 && !(res.len() == 2 && res[1] == b"%%" && res[0] == b"{}") {
                let msg = cat(&[name, b": At Most 1 required but received ", res.len().to_string().as_bytes()]);
                return Outcome::Mismatch(msg);
            }
            &res[0]
        };
        let ok = match o.kind {
            Kind::Flag => match flag_to_bool(value) {
                Some(b) => {
                    if i == VERSION {
                        out.version = b;
                    } else if i == NOTICE {
                        out.notice = b;
                    }
                    true
                }
                None => false,
            },
            Kind::Int => match if value.is_empty() { Some(0) } else { integral_conversion(value) } {
                Some(v) => {
                    if i == THREADS {
                        out.threads = v;
                    } else {
                        out.random_seed = v;
                    }
                    true
                }
                None => false,
            },
            Kind::Float => {
                let (v, ok) = if value.is_empty() { (0.0, true) } else { float_conversion(value) };
                out.time_limit = v;
                ok
            }
            Kind::File | Kind::Text => {
                out.strings[i] = value.to_vec();
                true
            }
        };
        if !ok {
            let msg = cat(&[b"Could not convert: ", name, b" = ", &res.join(&b","[..])]);
            return Outcome::Error(msg, EXIT_CONVERSION);
        }
    }
    // _process_extras: the `--` marks are listed but not counted
    if missing.iter().any(|(mark, _)| !mark) {
        let mut msg: Vec<u8> = if missing.len() > 1 {
            b"The following arguments were not expected: ".to_vec()
        } else {
            b"The following argument was not expected: ".to_vec()
        };
        let words: Vec<&[u8]> = missing.iter().rev().map(|(_, a)| &a[..]).collect();
        msg.extend(words.join(&b" "[..]));
        return Outcome::Extras(msg);
    }
    Outcome::Ok
}

/// ExistingFileValidator (stat, following links)
fn existing_file(name: &[u8]) -> Option<Vec<u8>> {
    #[cfg(unix)]
    let path = {
        use std::os::unix::ffi::OsStrExt;
        std::path::PathBuf::from(std::ffi::OsStr::from_bytes(name))
    };
    #[cfg(not(unix))]
    let path = std::path::PathBuf::from(String::from_utf8_lossy(name).into_owned());
    match std::fs::metadata(path) {
        Err(_) => Some(cat(&[b"File does not exist: ", name])),
        Ok(m) if m.is_dir() => Some(cat(&[b"File is actually a directory: ", name])),
        Ok(_) => None,
    }
}

extern "C" {
    fn strtoll(s: *const c_char, end: *mut *mut c_char, base: i32) -> i64;
    fn strtod(s: *const c_char, end: *mut *mut c_char) -> f64;
}

extern "C" {
    #[cfg_attr(any(target_os = "macos", target_os = "ios"), link_name = "__error")]
    #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), link_name = "__errno_location")]
    fn errno_location() -> *mut i32;
}

const ERANGE: i32 = 34;

/// C's strtoll(s, &end, base) on bytes: (value, bytes consumed, ERANGE)
fn c_strtoll(s: &[u8], base: i32) -> (i64, usize, bool) {
    // A C string ends at the first NUL (none come from argv)
    let s = &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())];
    let c = std::ffi::CString::new(s).unwrap();
    let mut end = std::ptr::null_mut();
    // SAFETY: errno is this thread's; c is NUL-terminated and outlives the
    // call, end points into it
    unsafe {
        *errno_location() = 0;
        let v = strtoll(c.as_ptr(), &mut end, base);
        let erange = *errno_location() == ERANGE;
        (v, end as usize - c.as_ptr() as usize, erange)
    }
}

/// strtold, as a double (ponytail: strtod; x86_64's 80-bit strtold could
/// round a halfway decimal twice)
fn c_strtod(s: &[u8]) -> (f64, usize) {
    let s = &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())];
    let c = std::ffi::CString::new(s).unwrap();
    let mut end = std::ptr::null_mut();
    // SAFETY: c is NUL-terminated and outlives the call
    let v = unsafe { strtod(c.as_ptr(), &mut end) };
    (v, end as usize - c.as_ptr() as usize)
}

/// C's isspace in the C locale
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn without_separators(s: &[u8]) -> Vec<u8> {
    s.iter().copied().filter(|&c| c != b'_' && c != b'\'').collect()
}

/// detail::integral_conversion for int
fn integral_conversion(input: &[u8]) -> Option<i32> {
    if input.is_empty() {
        return None;
    }
    let (v, n, erange) = c_strtoll(input, 0);
    if erange {
        return None;
    }
    if n == input.len() && v as i32 as i64 == v {
        return Some(v as i32);
    }
    if input == b"true" {
        return Some(1);
    }
    if input.iter().any(|&c| c == b'_' || c == b'\'') {
        return integral_conversion(&without_separators(input));
    }
    if is_space(*input.last().unwrap()) {
        let first = input.iter().position(|&c| !is_space(c));
        let t = match first {
            Some(f) => &input[f..=input.iter().rposition(|&c| !is_space(c)).unwrap()],
            None => &input[..0],
        };
        return integral_conversion(t);
    }
    for (prefix, base) in [(b"0o", 8), (b"0O", 8), (b"0b", 2), (b"0B", 2)] {
        if input.starts_with(prefix) {
            let (v, n, erange) = c_strtoll(&input[2..], base);
            if erange {
                return None;
            }
            return (n == input.len() - 2 && v as i32 as i64 == v).then_some(v as i32);
        }
    }
    None
}

/// detail::lexical_cast for a floating point value: (value, converted)
fn float_conversion(input: &[u8]) -> (f64, bool) {
    if input.is_empty() {
        return (0.0, false);
    }
    let (v, n) = c_strtod(input);
    if n == input.len() || input[n..].iter().all(|&c| is_space(c)) && is_space(input[n]) {
        return (v, true);
    }
    if input.iter().any(|&c| c == b'_' || c == b'\'') {
        return float_conversion(&without_separators(input));
    }
    (v, false)
}

/// detail::lexical_cast for a bool (to_flag_value), None if not converted
fn flag_to_bool(input: &[u8]) -> Option<bool> {
    let (v, einval, erange) = to_flag_value(input);
    if erange {
        Some(input.first() != Some(&b'-'))
    } else if einval {
        None
    } else {
        Some(v > 0)
    }
}

/// detail::to_flag_value: (value, EINVAL, ERANGE)
fn to_flag_value(input: &[u8]) -> (i64, bool, bool) {
    if input == b"true" {
        return (1, false, false);
    }
    if input == b"false" {
        return (-1, false, false);
    }
    let val = input.to_ascii_lowercase();
    if val.len() == 1 {
        return match val[0] {
            c @ b'1'..=b'9' => ((c - b'0') as i64, false, false),
            b'0' | b'f' | b'n' | b'-' => (-1, false, false),
            b't' | b'y' | b'+' => (1, false, false),
            _ => (-1, true, false),
        };
    }
    match &val[..] {
        b"true" | b"on" | b"yes" | b"enable" => (1, false, false),
        b"false" | b"off" | b"no" | b"disable" => (-1, false, false),
        _ => {
            let (v, n, erange) = c_strtoll(&val, 0);
            (v, n != val.len() && !erange, erange)
        }
    }
}

/// detail::streamOutAsParagraph (HiGHS's CLI11: words starting with a
/// double quote are indented by two)
fn paragraph(out: &mut String, text: &str, width: usize, prefix: &str, skip_prefix_on_first_line: bool) {
    if !skip_prefix_on_first_line {
        out.push_str(prefix);
    }
    let lines: Vec<&str> = text.split('\n').collect();
    // getline sees no line after a final newline
    let n = if text.ends_with('\n') { lines.len() - 1 } else { lines.len() };
    for (k, line) in lines[..n].iter().enumerate() {
        let mut chars = 0;
        for word in line.split(|c: char| c.is_ascii_whitespace() || c == '\x0b').filter(|w| !w.is_empty()) {
            if word.len() + chars > width {
                out.push('\n');
                out.push_str(prefix);
                chars = 0;
            }
            if word.starts_with('"') {
                out.push_str("  ");
            }
            out.push_str(word);
            out.push(' ');
            chars += word.len() + 1;
        }
        // !lss.eof(): a newline followed this line
        if k + 1 < lines.len() {
            out.push('\n');
            out.push_str(prefix);
        }
    }
}

/// App::help(): the usage the app sets and the "options" group, with
/// column width 33 and CLI11's default right column width of 60
fn help(program: &[u8]) -> Vec<u8> {
    const COLUMN: usize = 33;
    const RIGHT: usize = 60;
    let mut out = cat(&[b"usage:\n      ", program, b" [options] [file]\n"]);
    let mut s = String::from("\noptions:\n");
    for o in &OPTS {
        let mut line = String::new();
        let short_width = COLUMN / 5;
        let long_width = (COLUMN as f32 / 5.0f32 * 4.0f32).ceil() as usize;
        let mut over = 0;
        let label = o.type_name().1;
        let opts = if o.kind == Kind::Flag { String::new() } else { format!(" {label}") };
        match o.short() {
            Some(sn) => {
                let mut sn = format!("  {sn},");
                if sn.len() >= short_width {
                    sn.push(' ');
                    over = sn.len() - short_width;
                }
                line.push_str(&format!("{sn:<short_width$}"));
            }
            None => line.push_str(&" ".repeat(short_width)),
        }
        let adjusted = long_width - over.min(long_width);
        let mut ln = format!("{}{}", o.long(), opts);
        if ln.len() >= adjusted {
            ln.push(' ');
        }
        line.push_str(&format!("{ln:<adjusted$}"));
        let skip = line.len() <= COLUMN;
        if !skip {
            line.push('\n');
        }
        paragraph(&mut line, o.desc, RIGHT, &" ".repeat(COLUMN), skip);
        line.push('\n');
        s.push_str(&line);
    }
    out.extend(s.into_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> (CommandLine, Outcome) {
        let a: Vec<Vec<u8>> = args.iter().map(|s| s.as_bytes().to_vec()).collect();
        let mut c = CommandLine::default();
        let o = parse(&a, b"highs", &mut c);
        (c, o)
    }

    fn msg(o: &Outcome) -> String {
        match o {
            Outcome::Ok => "ok".into(),
            Outcome::Help(_) => "help".into(),
            Outcome::Extras(m) | Outcome::Mismatch(m) | Outcome::Error(m, _) => String::from_utf8_lossy(m).into(),
        }
    }

    #[test]
    fn command_lines() {
        let (c, o) = run(&["--threads=0x10", "--time_limit", "1_0", "--presolve", "On", "-v"]);
        assert_eq!(msg(&o), "ok");
        assert_eq!((c.threads, c.time_limit, c.version), (16, 10.0, true));
        assert_eq!(c.strings[7], b"On");
        assert_eq!(c.count[THREADS], 1);
        assert_eq!(msg(&run(&["--threads", "abc"]).1), "Could not convert: --threads = abc");
        assert_eq!(msg(&run(&["--threads"]).1), "--threads: 1 required INT missing");
        assert_eq!(msg(&run(&["--threads", "1", "--threads=2"]).1), "--threads: At Most 1 required but received 2");
        assert_eq!(msg(&run(&["Cargo.toml", "b", "--", "c"]).1), "The following arguments were not expected: c -- b");
        assert_eq!(msg(&run(&["--x"]).1), "The following argument was not expected: --x");
        assert_eq!(msg(&run(&["--model_file", "/nonexistent/x"]).1), "--model_file: File does not exist: /nonexistent/x");
        assert_eq!(msg(&run(&["--options_file", "/"]).1), "--options_file: File is actually a directory: /");
        assert_eq!(msg(&run(&["-vh"]).1), "help");
        assert_eq!(msg(&run(&["--version=abc"]).1), "Could not convert: --version = abc");
        assert_eq!(run(&["--threads", "5 "]).0.threads, 5);
        assert_eq!(run(&["--threads", "0b11"]).0.threads, 3);
        assert_eq!(run(&["--threads", "true"]).0.threads, 1);
        assert_eq!(msg(&run(&["--threads", "99999999999"]).1), "Could not convert: --threads = 99999999999");
    }

    #[test]
    fn help_text() {
        let h = String::from_utf8(help(b"highs")).unwrap();
        assert!(h.starts_with("usage:\n      highs [options] [file]\n\noptions:\n      --model_file file          File of model to solve \n"));
        assert!(h.contains("      --presolve text            Set presolve option to: \n                                   \"choose\" * default \n"));
        assert!(h.ends_with("  -h, --help                     Print help \n"));
    }
}
