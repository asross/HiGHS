//! HighsModelUtils.cpp: names (normaliseNames, maxNameLength,
//! findModelObjectiveName), the file type of a file name, the status
//! strings, and HighsLp::objectiveCDoubleValue (for writeLpObjective /
//! writeModelObjective). Names come as `RsName` lists and go back to C++
//! through `CNamesOut`.

use super::ffi::{RsMut, RsName};
use super::{Log, LogType, Status};
use crate::log_user;
use crate::util::cdouble::CDouble;
use std::collections::HashSet;
use std::ffi::c_void;

/// HighsFileType
pub const FILE_MINIMAL: i32 = 0;
pub const FILE_MPS: i32 = 2;
pub const FILE_LP: i32 = 3;

/// kLegalLpFileColRowNameChar (the bytes of its UTF-8 spelling)
const LEGAL_LP_NAME_CHARS: &str =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!\"#$%&(),.;?@_\u{2018}\u{2019}{}~";

fn legal_lp_byte(c: u8) -> bool {
    LEGAL_LP_NAME_CHARS.as_bytes().contains(&c)
}

/// hasIllegalNameForLpFile
fn has_illegal_name_for_lp_file(names: &[Vec<u8>]) -> bool {
    names.iter().any(|n| !n.iter().all(|&c| legal_lp_byte(c)))
}

/// maxNameLength
pub fn max_name_length(names: &[RsName]) -> i32 {
    names.iter().map(|n| n.len as i32).max().unwrap_or(0).max(0)
}

/// getFilenameExt and getFileType: "mps" or "lp" (any case) after the
/// last '.'
pub fn file_type(filename: &[u8]) -> i32 {
    let ext = match filename.iter().rposition(|&c| c == b'.') {
        Some(k) => &filename[k + 1..],
        None => &[][..],
    };
    let lower: Vec<u8> = ext.iter().map(|c| c.to_ascii_lowercase()).collect();
    match lower.as_slice() {
        b"mps" => FILE_MPS,
        b"lp" => FILE_LP,
        _ => FILE_MINIMAL,
    }
}

/// The result of normaliseNames for one dimension
pub struct Normalised {
    pub status: Status,
    /// The names, if changed
    pub names: Option<Vec<Vec<u8>>>,
    pub prefix: Option<&'static str>,
    pub suffix: i32,
}

/// normaliseNames of the column (or row) names: blank names get the
/// prefix "c_ekk" / "r_ekk" and a running suffix, MPS names have their
/// spaces replaced, and names that are absent, duplicated or (LP) illegal
/// are replaced by "c" / "r" and a suffix from 0
pub fn normalise_names(
    log: &Log,
    column: bool,
    num_name_required: usize,
    name_prefix: &str,
    name_suffix: i32,
    names_in: &[RsName],
    file: i32,
) -> Normalised {
    let max_name_length = max_name_length(names_in);
    let mut invalid_names = max_name_length == 0;
    let mut names: Vec<Vec<u8>> = names_in.iter().map(|n| n.bytes().to_vec()).collect();
    names.resize(num_name_required, Vec::new());
    let mut changed = names_in.len() != num_name_required;
    let mut num_blank = 0;
    let mut num_names_with_spaces = 0;
    let from_name_suffix = name_suffix;
    let mut suffix = name_suffix;
    let mut prefix: Option<&'static str> = None;
    if !invalid_names {
        invalid_names = file == FILE_LP && has_illegal_name_for_lp_file(&names);
        if !invalid_names {
            for name in names.iter_mut() {
                if name.is_empty() {
                    num_blank += 1;
                    let p = if column { "c_ekk" } else { "r_ekk" };
                    prefix = Some(p);
                    *name = format!("{}{}", p, suffix).into_bytes();
                    suffix += 1;
                    changed = true;
                } else if file == FILE_MPS && name.contains(&b' ') {
                    for c in name.iter_mut() {
                        if *c == b' ' {
                            *c = b'_';
                        }
                    }
                    num_names_with_spaces += 1;
                    changed = true;
                }
            }
        }
    }
    if !invalid_names {
        let mut seen = HashSet::new();
        invalid_names = !names.iter().all(|n| seen.insert(n.as_slice()));
    }
    let what = if column { "column" } else { "row" };
    if !invalid_names {
        let used_prefix = prefix.unwrap_or(name_prefix);
        let status = if num_blank != 0 || num_names_with_spaces != 0 {
            if num_blank != 0 {
                log_user!(
                    log,
                    LogType::Warning,
                    "Replaced %d blank %-6s name%s by one%s with prefix \"%s\", beginning with suffix %d\n",
                    num_blank,
                    what,
                    if num_blank == 1 { "" } else { "s" },
                    if num_blank == 1 { "" } else { "s" },
                    used_prefix,
                    from_name_suffix
                );
            }
            if num_names_with_spaces != 0 {
                log_user!(
                    log,
                    LogType::Warning,
                    "Replaced spaces in %d %-6s name%s by underscores\n",
                    num_names_with_spaces,
                    what,
                    if num_names_with_spaces == 1 { "" } else { "s" }
                );
            }
            Status::Warning
        } else {
            Status::Ok
        };
        return Normalised { status, names: if changed { Some(names) } else { None }, prefix, suffix };
    }
    let p = if column { "c" } else { "r" };
    log_user!(
        log,
        LogType::Warning,
        "%s names are not present, or contain %sduplicates: using names with prefix \"%s\", beginning with suffix %d\n",
        if column { "Column" } else { "Row   " },
        if file == FILE_LP { "invalid characters or " } else { "" },
        p,
        0
    );
    let names = (0..num_name_required).map(|k| format!("{}{}", p, k).into_bytes()).collect();
    Normalised { status: Status::Warning, names: Some(names), prefix: Some(p), suffix: num_name_required as i32 }
}

/// trim of " \t\n\v\f\r"
fn trim(s: &[u8]) -> &[u8] {
    let ws = |c: &u8| b"\t\n\x0b\x0c\r ".contains(c);
    let a = s.iter().position(|c| !ws(c)).unwrap_or(s.len());
    let b = s.iter().rposition(|c| !ws(c)).map_or(a, |k| k + 1);
    &s[a..b.max(a)]
}

/// findModelObjectiveName when the LP has none: "Obj" or "NoObj", with a
/// pass number appended until it differs from every (trimmed) row name
pub fn find_model_objective_name(has_objective: bool, row_names: &[RsName], num_row: usize) -> String {
    let mut pass = 0;
    loop {
        let mut name = String::from(if has_objective { "Obj" } else { "NoObj" });
        if row_names.is_empty() {
            return name;
        }
        if pass != 0 {
            name += &pass.to_string();
        }
        if !row_names[..num_row].iter().any(|r| trim(r.bytes()) == name.as_bytes()) {
            return name;
        }
        pass += 1;
    }
}

/// HighsLp::objectiveCDoubleValue
pub fn lp_objective_cdouble(offset: f64, cost: &[f64], x: &[f64]) -> CDouble {
    let mut f = CDouble::from(offset);
    for (&c, &v) in cost.iter().zip(x) {
        f += CDouble::from(c) * v;
    }
    f
}

/// utilSolutionStatusToString
pub fn solution_status_string(s: i32) -> &'static str {
    match s {
        0 => "None",
        1 => "Infeasible",
        2 => "Feasible",
        _ => "Unrecognised solution status",
    }
}

/// utilBasisStatusToString
pub fn basis_status_string(s: i32) -> &'static str {
    match s {
        0 => "At lower/fixed bound",
        1 => "Basic",
        2 => "At upper bound",
        3 => "Free at zero",
        4 => "Nonbasic",
        _ => "Unrecognised solution status",
    }
}

/// utilPresolveRuleTypeToString
pub fn presolve_rule_type_string(r: i32) -> &'static str {
    const NAMES: [&str; 20] = [
        "Empty row",
        "Singleton row",
        "Redundant row",
        "Empty column",
        "Fixed column",
        "Dominated col",
        "Forcing row",
        "Forcing col",
        "Free col substitution",
        "Doubleton equation",
        "Dependent equations",
        "Dependent free columns",
        "Aggregator",
        "Parallel rows and columns",
        "Sparsify",
        "Probing",
        "Enumeration",
        "Dual fixing",
        "Col stuffing",
        "Initial sweep",
    ];
    NAMES.get(r as usize).copied().filter(|_| r >= 0).unwrap_or("????")
}

/// interpretFilereaderRetcode
pub fn interpret_filereader_retcode(log: &Log, filename: &str, code: i32) {
    match code {
        2 => log_user!(log, LogType::Error, "File %s not found\n", filename),
        3 => log_user!(log, LogType::Error, "Parser error reading %s\n", filename),
        4 => log_user!(log, LogType::Error, "Parser not implemented for %s", filename),
        5 => log_user!(log, LogType::Error, "Parser reached timeout\n"),
        _ => {}
    }
}

/// extractModelName: the file name without its directory and
/// extension(s) (".gz" then one more)
pub fn extract_model_name(filename: &str) -> String {
    let mut name = filename;
    if let Some(k) = name.rfind(['/', '\\']) {
        name = &name[k + 1..];
    }
    let mut name = name.to_string();
    let found = name.rfind('.');
    let ext = match found {
        Some(k) => &name[k + 1..],
        None => name.as_str(),
    };
    let mut found = found;
    if ext == "gz" {
        // std::string::erase(npos) throws: a name "gz" without a '.'
        let k = found.expect("extractModelName: std::out_of_range");
        name.truncate(k);
        found = name.rfind('.');
    }
    if let Some(k) = found {
        name.truncate(k);
    }
    name
}

/// The reader of a file (Filereader::getFilereader): 1 MPS, 2 LP, 0 none;
/// a ".gz" file without zlib support has an error
pub fn file_reader_kind(log: &Log, filename: &[u8], zlib: bool) -> i32 {
    let ext = |f: &[u8]| -> Vec<u8> {
        match f.iter().rposition(|&c| c == b'.') {
            Some(k) => f[k + 1..].to_vec(),
            None => Vec::new(),
        }
    };
    let mut extension = ext(filename);
    if extension == b"gz" {
        if zlib {
            extension = ext(&filename[..filename.len() - 3]);
        } else {
            log_user!(log, LogType::Error, "HiGHS build without zlib support. Cannot read .gz file.\n");
        }
    }
    let lower: Vec<u8> = extension.iter().map(|c| c.to_ascii_lowercase()).collect();
    match lower.as_slice() {
        b"mps" => 1,
        b"lp" => 2,
        _ => 0,
    }
}

// ------------------------------------------------------------ entry points

/// A C++ vector of names that Rust replaces (resize, then set each name)
#[repr(C)]
pub struct CNamesOut {
    pub vec: *mut c_void,
    pub resize: unsafe extern "C" fn(*mut c_void, usize),
    pub set: unsafe extern "C" fn(*mut c_void, usize, *const u8, usize),
}

/// normaliseNames for one dimension: names are replaced through `out`,
/// the prefix through `set_prefix(prefix_ctx, ..)` and the suffix in
/// place; the C++ clears the name hash
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_normalise_names(
    log: *const Log,
    column: bool,
    num_name_required: i32,
    name_prefix: RsName,
    name_suffix: *mut i32,
    names: RsMut<RsName>,
    file: i32,
    out: *const CNamesOut,
    set_prefix: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    prefix_ctx: *mut c_void,
) -> i32 {
    let prefix = name_prefix.text();
    let r = normalise_names(
        &*log,
        column,
        num_name_required.max(0) as usize,
        &prefix,
        *name_suffix,
        names.get(),
        file,
    );
    if let Some(names) = &r.names {
        let out = &*out;
        (out.resize)(out.vec, names.len());
        for (k, n) in names.iter().enumerate() {
            (out.set)(out.vec, k, n.as_ptr(), n.len());
        }
    }
    if let Some(p) = r.prefix {
        set_prefix(prefix_ctx, p.as_ptr(), p.len());
    }
    *name_suffix = r.suffix;
    r.status as i32
}

/// Filereader::getFilereader's reader kind (file_reader_kind),
/// interpretFilereaderRetcode (code >= 0) and extractModelName (into
/// set(ctx, ..))
///
/// # Safety
/// `log` is valid; `name` holds `len` bytes
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_filereader(
    log: *const Log,
    which: i32,
    name: *const u8,
    len: usize,
    arg: i32,
    set: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    ctx: *mut c_void,
) -> i32 {
    let bytes = if len == 0 { &[][..] } else { std::slice::from_raw_parts(name, len) };
    match which {
        0 => file_reader_kind(&*log, bytes, arg != 0),
        1 => {
            interpret_filereader_retcode(&*log, &String::from_utf8_lossy(bytes), arg);
            0
        }
        _ => {
            let n = extract_model_name(&String::from_utf8_lossy(bytes));
            set(ctx, n.as_ptr(), n.len());
            0
        }
    }
}

/// maxNameLength
///
/// # Safety
/// The view is the C++ list's
#[no_mangle]
pub unsafe extern "C" fn highs_rs_max_name_length(names: RsMut<RsName>) -> i32 {
    max_name_length(names.get())
}

/// getFileType
///
/// # Safety
/// `name` holds `len` bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_file_type(name: *const u8, len: usize) -> i32 {
    file_type(if len == 0 { &[] } else { std::slice::from_raw_parts(name, len) })
}

/// findModelObjectiveName of an LP without one, through `set(ctx, ..)`
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_find_model_objective_name(
    cost: RsMut<f64>,
    hessian_dim: i32,
    row_names: RsMut<RsName>,
    num_row: i32,
    set: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    ctx: *mut c_void,
) {
    let has_objective = cost.get().iter().any(|&c| c != 0.0) || hessian_dim != 0;
    let name = find_model_objective_name(has_objective, row_names.get(), num_row.max(0) as usize);
    set(ctx, name.as_ptr(), name.len());
}

/// HighsLp::objectiveCDoubleValue: (hi, lo) into out
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lp_objective_cdouble(
    offset: f64,
    cost: RsMut<f64>,
    x: RsMut<f64>,
    out: *mut CDouble,
) {
    *out = lp_objective_cdouble(offset, cost.get(), x.get());
}

/// The status strings: which 0 solution status, 1 basis status, 2 basis
/// validity, 3 model status, 4 presolve rule
#[no_mangle]
pub extern "C" fn highs_rs_status_string(which: i32, value: i32) -> RsName {
    let s = match which {
        0 => solution_status_string(value),
        1 => basis_status_string(value),
        2 => {
            if value != 0 {
                "Valid"
            } else {
                "Not valid"
            }
        }
        3 => crate::simplex::hekk::model_status_string(value),
        _ => presolve_rule_type_string(value),
    };
    RsName { ptr: s.as_ptr(), len: s.len() }
}

/// highsStatusFromHighsModelStatus
#[no_mangle]
pub extern "C" fn highs_rs_status_from_model_status(model_status: i32) -> i32 {
    super::run::status_from_model_status(model_status) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs(names: &[&'static str]) -> Vec<RsName> {
        names.iter().map(|n| RsName { ptr: n.as_ptr(), len: n.len() }).collect()
    }

    #[test]
    fn names() {
        let r = normalise_names(&Log::none(), true, 3, "", 0, &rs(&["a b", "", "c"]), FILE_MPS);
        assert_eq!(r.status, Status::Warning);
        assert_eq!(r.names.unwrap(), vec![b"a_b".to_vec(), b"c_ekk0".to_vec(), b"c".to_vec()]);
        let r = normalise_names(&Log::none(), false, 2, "", 0, &rs(&["x", "x"]), FILE_MINIMAL);
        assert_eq!(r.names.unwrap(), vec![b"r0".to_vec(), b"r1".to_vec()]);
        let r = normalise_names(&Log::none(), false, 2, "", 0, &rs(&["x", "y"]), FILE_LP);
        assert_eq!(r.status, Status::Ok);
        assert!(r.names.is_none());
        assert_eq!(file_type(b"a.b.MPS"), FILE_MPS);
        assert_eq!(file_type(b"lp"), FILE_MINIMAL);
        assert_eq!(find_model_objective_name(true, &rs(&[" Obj ", "Obj1"]), 2), "Obj2");
        assert_eq!(presolve_rule_type_string(19), "Initial sweep");
    }
}
