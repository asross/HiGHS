//! HighsInfo.cpp: invalidating and comparing HighsInfo, and finding,
//! checking, getting and reporting info values. As for the options
//! (options.rs), the records stay C++ (HighsInfo's fields are public
//! API) and each call gets a table of views of them (`CInfoRecord`); the
//! data is HighsInfoStruct, read and written in place (solution.rs: Info).

use super::options::{md_escapes, st, RsStr, FILE_FULL, FILE_MD};
use super::solution::Info;
use super::{Log, LogType, Status, INF};
use crate::{log_user, sprintf};
use std::ffi::c_void;

// InfoStatus
pub const OK: i32 = 0;
pub const UNKNOWN_INFO: i32 = 1;
pub const ILLEGAL_VALUE: i32 = 2;
pub const UNAVAILABLE: i32 = 3;

// HighsInfoType
pub const INT64: i32 = -1;
pub const INT: i32 = 1;
pub const DOUBLE: i32 = 2;

/// An InfoRecord (Int64, Int or Double): `value` points to its field of
/// HighsInfo
#[repr(C)]
pub struct CInfoRecord {
    pub type_: i32,
    pub advanced: bool,
    pub name: RsStr,
    pub description: RsStr,
    pub value: *mut c_void,
}

impl CInfoRecord {
    fn name(&self) -> &[u8] {
        // SAFETY: the C++ record outlives the call
        unsafe { self.name.get() }
    }
    fn description(&self) -> &[u8] {
        // SAFETY: as name
        unsafe { self.description.get() }
    }
    fn int64(&self) -> i64 {
        // SAFETY: an int64_t record's value points to its int64_t field
        unsafe { *(self.value as *const i64) }
    }
    fn int(&self) -> i32 {
        // SAFETY: a HighsInt record's value points to its HighsInt field
        unsafe { *(self.value as *const i32) }
    }
    fn double(&self) -> f64 {
        // SAFETY: a double record's value points to its double field
        unsafe { *(self.value as *const f64) }
    }
}

impl Info {
    /// HighsInfo::invalidate
    pub fn invalidate(&mut self) {
        self.valid = false;
        self.mip_node_count = -1;
        self.simplex_iteration_count = -1;
        self.ipm_iteration_count = -1;
        self.crossover_iteration_count = -1;
        self.pdlp_iteration_count = -1;
        self.qp_iteration_count = -1;
        self.basis_validity = 0; // kBasisValidityInvalid
        self.objective_function_value = 0.0;
        self.mip_dual_bound = 0.0;
        self.mip_gap = INF;
        self.max_integrality_violation = INF;
        self.invalidate_kkt();
        self.primal_dual_integral = -INF;
    }

    /// HighsInfo::equal (all but primal_dual_integral)
    pub fn equal(&self, o: &Info) -> bool {
        self.valid == o.valid
            && self.mip_node_count == o.mip_node_count
            && self.simplex_iteration_count == o.simplex_iteration_count
            && self.ipm_iteration_count == o.ipm_iteration_count
            && self.crossover_iteration_count == o.crossover_iteration_count
            && self.pdlp_iteration_count == o.pdlp_iteration_count
            && self.qp_iteration_count == o.qp_iteration_count
            && self.primal_solution_status == o.primal_solution_status
            && self.dual_solution_status == o.dual_solution_status
            && self.basis_validity == o.basis_validity
            && self.objective_function_value == o.objective_function_value
            && self.mip_dual_bound == o.mip_dual_bound
            && self.mip_gap == o.mip_gap
            && self.max_integrality_violation == o.max_integrality_violation
            && self.num_primal_infeasibilities == o.num_primal_infeasibilities
            && self.max_primal_infeasibility == o.max_primal_infeasibility
            && self.sum_primal_infeasibilities == o.sum_primal_infeasibilities
            && self.num_dual_infeasibilities == o.num_dual_infeasibilities
            && self.max_dual_infeasibility == o.max_dual_infeasibility
            && self.sum_dual_infeasibilities == o.sum_dual_infeasibilities
            && self.num_semi_infeasibilities == o.num_semi_infeasibilities
            && self.max_semi_infeasibility == o.max_semi_infeasibility
            && self.sum_semi_infeasibilities == o.sum_semi_infeasibilities
            && self.num_relative_primal_infeasibilities == o.num_relative_primal_infeasibilities
            && self.max_relative_primal_infeasibility == o.max_relative_primal_infeasibility
            && self.num_relative_dual_infeasibilities == o.num_relative_dual_infeasibilities
            && self.max_relative_dual_infeasibility == o.max_relative_dual_infeasibility
            && self.num_primal_residual_errors == o.num_primal_residual_errors
            && self.max_primal_residual_error == o.max_primal_residual_error
            && self.num_dual_residual_errors == o.num_dual_residual_errors
            && self.max_dual_residual_error == o.max_dual_residual_error
            && self.num_relative_primal_residual_errors == o.num_relative_primal_residual_errors
            && self.max_relative_primal_residual_error == o.max_relative_primal_residual_error
            && self.num_relative_dual_residual_errors == o.num_relative_dual_residual_errors
            && self.max_relative_dual_residual_error == o.max_relative_dual_residual_error
            && self.num_complementarity_violations == o.num_complementarity_violations
            && self.max_complementarity_violation == o.max_complementarity_violation
            && self.primal_dual_objective_error == o.primal_dual_objective_error
    }
}

/// infoEntryTypeToString
fn type_name(t: i32) -> &'static str {
    match t {
        INT64 => "int64_t",
        INT => "HighsInt",
        _ => "double",
    }
}

/// getInfoIndex
pub fn info_index(log: &Log, name: &[u8], recs: &[CInfoRecord]) -> Result<usize, i32> {
    if let Some(i) = recs.iter().position(|r| r.name() == name) {
        return Ok(i);
    }
    log_user!(log, LogType::Error, "getInfoIndex: Info \"%s\" is unknown\n", &*st(name));
    Err(UNKNOWN_INFO)
}

/// checkInfo
pub fn check_info(log: &Log, recs: &[CInfoRecord]) -> i32 {
    let mut error_found = false;
    for (index, r) in recs.iter().enumerate() {
        for (check_index, c) in recs.iter().enumerate() {
            if check_index != index && c.name() == r.name() {
                log_user!(
                    log,
                    LogType::Error,
                    "checkInfo: Info %d (\"%s\") has the same name as info %d \"%s\"\n",
                    index,
                    &*st(r.name()),
                    check_index,
                    &*st(c.name())
                );
                error_found = true;
            }
        }
        for (check_index, c) in recs.iter().enumerate() {
            if check_index != index && c.type_ == r.type_ && c.value == r.value {
                log_user!(
                    log,
                    LogType::Error,
                    "checkInfo: Info %d (\"%s\") has the same value pointer as info %d (\"%s\")\n",
                    index,
                    &*st(r.name()),
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
    log_user!(log, LogType::Info, "checkInfo: Info are OK\n");
    OK
}

/// A value of getLocalInfoValue
pub enum Value {
    I64(i64),
    I(i32),
    D(f64),
}

/// getLocalInfoValue for `want` (INT64, INT or DOUBLE)
pub fn get_info_value(log: &Log, name: &[u8], valid: bool, recs: &[CInfoRecord], want: i32) -> Result<Value, i32> {
    let r = &recs[info_index(log, name, recs)?];
    if !valid {
        return Err(UNAVAILABLE);
    }
    if r.type_ != want {
        let not = match want {
            INT64 => "int64_t",
            INT => "HighsInt",
            _ => "double",
        };
        log_user!(
            log,
            LogType::Error,
            "getInfoValue: Info \"%s\" requires value of type %s, not %s\n",
            &*st(name),
            type_name(r.type_),
            not
        );
        return Err(ILLEGAL_VALUE);
    }
    Ok(match want {
        INT64 => Value::I64(r.int64()),
        INT => Value::I(r.int()),
        _ => Value::D(r.double()),
    })
}

/// reportInfo of one record
pub fn report_info(r: &CInfoRecord, file_type: i32) -> String {
    let name = st(r.name());
    let desc = st(r.description());
    let (md_type, full_type) = match r.type_ {
        INT64 => ("long integer", "int64_t"),
        INT => ("integer", "HighsInt"),
        _ => ("double", "double"),
    };
    if file_type == FILE_MD {
        return sprintf!("## %s\n- %s\n- Type: %s\n\n", &md_escapes(&name), &md_escapes(&desc), md_type);
    }
    let head = if file_type == FILE_FULL {
        sprintf!("\n# %s\n# [type: %s]\n%s = ", &*desc, full_type, &*name)
    } else {
        sprintf!("%-30s = ", &*name)
    };
    head + &match r.type_ {
        INT64 => sprintf!("%lld\n", r.int64()),
        INT => sprintf!("%d\n", r.int()),
        _ => sprintf!("%g\n", r.double()),
    }
}

/// writeInfoToFile: the report (none if not valid, unless documentation)
pub fn write_info(valid: bool, recs: &[CInfoRecord], file_type: i32) -> (Status, String) {
    let documentation_file = file_type == FILE_MD;
    if !documentation_file && !valid {
        return (Status::Warning, String::new());
    }
    (Status::Ok, recs.iter().map(|r| report_info(r, file_type)).collect())
}

// The C++ entry points (highs/lp_data/HighsOptionsRust.cpp)

/// # Safety (all entry points)
/// The log, the record tables (with their strings and value pointers),
/// the byte strings and the out pointers must be valid
unsafe fn table<'a>(recs: *const CInfoRecord, n: usize) -> &'a [CInfoRecord] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(recs, n)
    }
}

unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    RsStr { ptr, len }.get()
}

/// # Safety
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_info_invalidate(info: *mut Info, what: i32) {
    let info = &mut *info;
    match what {
        0 => info.invalidate(),
        1 => info.invalidate_kkt(),
        2 => info.invalidate_primal_kkt(),
        _ => info.invalidate_dual_kkt(),
    }
}

/// # Safety
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_info_equal(a: *const Info, b: *const Info) -> bool {
    (*a).equal(&*b)
}

/// # Safety
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_info_index(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    recs: *const CInfoRecord,
    n: usize,
    index: *mut i32,
) -> i32 {
    match info_index(&*log, bytes(name, name_len), table(recs, n)) {
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
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_info(log: *const Log, recs: *const CInfoRecord, n: usize) -> i32 {
    check_info(&*log, table(recs, n))
}

/// getLocalInfoValue (`want`: INT64, INT or DOUBLE; `value` its type)
///
/// # Safety
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_info_value(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    valid: bool,
    recs: *const CInfoRecord,
    n: usize,
    want: i32,
    value: *mut c_void,
) -> i32 {
    match get_info_value(&*log, bytes(name, name_len), valid, table(recs, n), want) {
        Ok(Value::I64(v)) => *(value as *mut i64) = v,
        Ok(Value::I(v)) => *(value as *mut i32) = v,
        Ok(Value::D(v)) => *(value as *mut f64) = v,
        Err(s) => return s,
    }
    OK
}

/// getLocalInfoType
///
/// # Safety
/// See `table`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_info_type(
    log: *const Log,
    name: *const u8,
    name_len: usize,
    recs: *const CInfoRecord,
    n: usize,
    type_: *mut i32,
) -> i32 {
    let recs = table(recs, n);
    match info_index(&*log, bytes(name, name_len), recs) {
        Ok(i) => {
            *type_ = recs[i].type_;
            OK
        }
        Err(s) => s,
    }
}

/// writeInfoToFile (`check_valid`) or reportInfo: the text goes to
/// `write(file, ...)`; returns the HighsStatus
///
/// # Safety
/// See `table`; `write` takes `file`
#[no_mangle]
pub unsafe extern "C" fn highs_rs_write_info(
    file: *mut c_void,
    write: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    check_valid: bool,
    valid: bool,
    recs: *const CInfoRecord,
    n: usize,
    file_type: i32,
) -> i32 {
    let (status, text) = write_info(valid || !check_valid, table(recs, n), file_type);
    if !text.is_empty() {
        write(file, text.as_ptr(), text.len());
    }
    status as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_and_invalidate() {
        let (mut i64v, mut iv, mut dv) = (12i64, -1i32, 0.5f64);
        let rec = |type_, name: &'static str, value: *mut c_void| CInfoRecord {
            type_,
            advanced: false,
            name: RsStr::of(name.as_bytes()),
            description: RsStr::of(b"Some_desc"),
            value,
        };
        let recs = [
            rec(INT64, "mip_node_count", &mut i64v as *mut i64 as *mut c_void),
            rec(INT, "simplex_iteration_count", &mut iv as *mut i32 as *mut c_void),
            rec(DOUBLE, "mip_gap", &mut dv as *mut f64 as *mut c_void),
        ];
        let (s, t) = write_info(true, &recs, 0);
        assert_eq!(s, Status::Ok);
        assert_eq!(
            t,
            "mip_node_count                 = 12\nsimplex_iteration_count        = -1\nmip_gap                        = 0.5\n"
        );
        assert_eq!(report_info(&recs[1], FILE_FULL), "\n# Some_desc\n# [type: HighsInt]\nsimplex_iteration_count = -1\n");
        assert_eq!(report_info(&recs[0], FILE_MD), "## mip\\_node\\_count\n- Some\\_desc\n- Type: long integer\n\n");
        assert_eq!(write_info(false, &recs, FILE_MD).0, Status::Ok);
        assert_eq!(write_info(false, &recs, FILE_FULL), (Status::Warning, String::new()));
        assert!(matches!(get_info_value(&Log::none(), b"mip_gap", true, &recs, DOUBLE), Ok(Value::D(v)) if v == 0.5));
        assert!(matches!(get_info_value(&Log::none(), b"mip_gap", false, &recs, DOUBLE), Err(UNAVAILABLE)));
        assert!(matches!(get_info_value(&Log::none(), b"mip_gap", true, &recs, INT), Err(ILLEGAL_VALUE)));
        assert!(matches!(get_info_value(&Log::none(), b"x", true, &recs, INT), Err(UNKNOWN_INFO)));
    }
}
