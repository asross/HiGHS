//! BASICLU (highs/ipm/basiclu/): the sparse LU factorization with
//! Forrest-Tomlin updates used by IPX. A line-by-line port of the C code:
//! same operations in the same order (clang contracts `a ± b*c` within an
//! expression into a fused multiply-add on arm64, mirrored here by
//! `mul_add`), so IPX runs bit-identically. The C entry points
//! (basiclu_factorize, basiclu_solve_*, basiclu_update, basiclu_obj_*, ...)
//! are exported from ffi.rs under their C names.
//!
//! All state lives in caller-owned arrays: the integer and double stores
//! `istore`, `xstore` and the factor files Li/Lx, Ui/Ux, Wi/Wx. `Lu::load`
//! partitions the stores into the subarrays the routines work on (some of
//! which are reused under a second name between factorization and
//! solves/updates, see the field comments) and copies the scalars out;
//! `Lu::save` copies the scalars back.

// Faithful port: index loops and in-place compaction as in the C code
#![allow(
    clippy::needless_range_loop,
    clippy::mut_range_bound,
    clippy::explicit_counter_loop
)]

mod build_factors;
mod condest;
mod factorize;
mod ffi;
mod file;
mod get_factors;
mod markowitz;
mod pivot;
mod setup_bump;
mod singletons;
mod solve;
mod update;

#[cfg(test)]
mod tests;

/// lu_int: HighsInt (32 bit; IPX static_asserts that its Int matches)
pub type Int = i32;

pub const SIZE_ISTORE_1: Int = 1024;
pub const SIZE_ISTORE_M: Int = 21;
pub const SIZE_XSTORE_1: Int = 1024;
pub const SIZE_XSTORE_M: Int = 4;

// status codes
pub const OK: Int = 0;
pub const REALLOCATE: Int = 1;
pub const WARNING_SINGULAR_MATRIX: Int = 2;
pub const ERROR_INVALID_STORE: Int = -1;
pub const ERROR_INVALID_CALL: Int = -2;
pub const ERROR_ARGUMENT_MISSING: Int = -3;
pub const ERROR_INVALID_ARGUMENT: Int = -4;
pub const ERROR_MAXIMUM_UPDATES: Int = -5;
pub const ERROR_SINGULAR_UPDATE: Int = -6;
pub const ERROR_INVALID_OBJECT: Int = -8;
pub const ERROR_OUT_OF_MEMORY: Int = -9;

// public entries in xstore: user parameters
pub const MEMORYL: usize = 1;
pub const MEMORYU: usize = 2;
pub const MEMORYW: usize = 3;
pub const DROP_TOLERANCE: usize = 4;
pub const ABS_PIVOT_TOLERANCE: usize = 5;
pub const REL_PIVOT_TOLERANCE: usize = 6;
pub const BIAS_NONZEROS: usize = 7;
pub const MAXN_SEARCH_PIVOT: usize = 8;
pub const PAD: usize = 9;
pub const STRETCH: usize = 10;
pub const COMPRESSION_THRESHOLD: usize = 11;
pub const SPARSE_THRESHOLD: usize = 12;
pub const REMOVE_COLUMNS: usize = 13;
pub const SEARCH_ROWS: usize = 14;

// user readable
pub const DIM: usize = 64;
pub const STATUS: usize = 65;
pub const ADD_MEMORYL: usize = 66;
pub const ADD_MEMORYU: usize = 67;
pub const ADD_MEMORYW: usize = 68;
pub const NUPDATE: usize = 70;
pub const NFORREST: usize = 71;
pub const NFACTORIZE: usize = 72;
pub const NUPDATE_TOTAL: usize = 73;
pub const NFORREST_TOTAL: usize = 74;
pub const NSYMPERM_TOTAL: usize = 75;
pub const LNZ: usize = 76;
pub const UNZ: usize = 77;
pub const RNZ: usize = 78;
pub const MIN_PIVOT: usize = 79;
pub const MAX_PIVOT: usize = 80;
pub const UPDATE_COST: usize = 81;
pub const TIME_FACTORIZE: usize = 82;
pub const TIME_SOLVE: usize = 83;
pub const TIME_UPDATE: usize = 84;
pub const TIME_FACTORIZE_TOTAL: usize = 85;
pub const TIME_SOLVE_TOTAL: usize = 86;
pub const TIME_UPDATE_TOTAL: usize = 87;
pub const LFLOPS: usize = 88;
pub const UFLOPS: usize = 89;
pub const RFLOPS: usize = 90;
pub const CONDEST_L: usize = 91;
pub const CONDEST_U: usize = 92;
pub const MAX_ETA: usize = 93;
pub const NORM_L: usize = 94;
pub const NORM_U: usize = 95;
pub const NORMEST_LINV: usize = 96;
pub const NORMEST_UINV: usize = 97;
pub const MATRIX_ONENORM: usize = 98;
pub const MATRIX_INFNORM: usize = 99;
pub const MATRIX_NZ: usize = 100;
pub const RANK: usize = 101;
pub const BUMP_SIZE: usize = 102;
pub const BUMP_NZ: usize = 103;
pub const NSEARCH_PIVOT: usize = 104;
pub const NEXPAND: usize = 105;
pub const NGARBAGE: usize = 106;
pub const FACTOR_FLOPS: usize = 107;
pub const TIME_SINGLETONS: usize = 108;
pub const TIME_SEARCH_PIVOT: usize = 109;
pub const TIME_ELIM_PIVOT: usize = 110;
pub const RESIDUAL_TEST: usize = 111;
pub const PIVOT_ERROR: usize = 120;

// private entries in xstore
const TASK: usize = 256;
const FTCOLUMN_IN: usize = 257;
const FTCOLUMN_OUT: usize = 258;
const PIVOT_ROW: usize = 259;
const PIVOT_COL: usize = 260;
const RANKDEF: usize = 261;
const MIN_COLNZ: usize = 262;
const MIN_ROWNZ: usize = 263;
const MARKER: usize = 266;
const UPDATE_COST_NUMER: usize = 267;
const UPDATE_COST_DENOM: usize = 268;
const PIVOTLEN: usize = 269;

/// hash in istore[0], xstore[0]
pub const HASH: Int = 7743090;

// the part of factorization in progress
const NO_TASK: Int = 0;
const SINGLETONS: Int = 1;
const SETUP_BUMP: Int = 2;
const FACTORIZE_BUMP: Int = 3;
const BUILD_FACTORS: Int = 4;

/// Length of istore for dimension m
pub fn istore_len(m: Int) -> usize {
    (SIZE_ISTORE_1 + SIZE_ISTORE_M * m) as usize
}

/// Length of xstore for dimension m
pub fn xstore_len(m: Int) -> usize {
    (SIZE_XSTORE_1 + SIZE_XSTORE_M * m) as usize
}

/// struct lu: scalars copied from xstore and the store partitions.
///
/// Fields documented as "a | b" are one array used as `a` during
/// factorization and as `b` during solves/updates.
pub(crate) struct Lu<'a> {
    // user parameters, not modified
    pub lmem: Int,
    pub umem: Int,
    pub wmem: Int,
    pub droptol: f64,
    pub abstol: f64,
    pub reltol: f64,
    pub nzbias: Int,
    pub maxsearch: Int,
    pub pad: Int,
    pub stretch: f64,
    pub compress_thres: f64,
    pub sparse_thres: f64,
    pub search_rows: Int,

    // user readable
    pub m: Int,
    pub addmem_l: Int,
    pub addmem_u: Int,
    pub addmem_w: Int,
    pub nupdate: Int,
    pub nforrest: Int,
    pub nfactorize: Int,
    pub nupdate_total: Int,
    pub nforrest_total: Int,
    pub nsymperm_total: Int,
    /// nz in L excluding diagonal
    pub lnz: Int,
    /// nz in U excluding diagonal
    pub unz: Int,
    /// nz in update etas excluding diagonal
    pub rnz: Int,
    pub min_pivot: f64,
    pub max_pivot: f64,
    pub max_eta: f64,
    pub update_cost_numer: f64,
    pub update_cost_denom: f64,
    pub time_factorize: f64,
    pub time_solve: f64,
    pub time_update: f64,
    pub time_factorize_total: f64,
    pub time_solve_total: f64,
    pub time_update_total: f64,
    pub lflops: Int,
    pub uflops: Int,
    pub rflops: Int,
    pub condest_l: f64,
    pub condest_u: f64,
    pub norm_l: f64,
    pub norm_u: f64,
    pub normest_linv: f64,
    pub normest_uinv: f64,
    /// 1-norm and inf-norm of matrix after fresh factorization with
    /// dependent cols replaced
    pub onenorm: f64,
    pub infnorm: f64,
    pub residual_test: f64,
    /// nz in basis matrix when factorized
    pub matrix_nz: Int,
    /// rank of basis matrix when factorized
    pub rank: Int,
    pub bump_size: Int,
    pub bump_nz: Int,
    /// # rows/cols searched for pivot
    pub nsearch_pivot: Int,
    /// # rows/cols expanded in factorize
    pub nexpand: Int,
    /// # garbage collections in factorize
    pub ngarbage: Int,
    /// # flops in factorize
    pub factor_flops: Int,
    pub time_singletons: f64,
    pub time_search_pivot: f64,
    pub time_elim_pivot: f64,
    /// error estimate for pivot in last update
    pub pivot_error: f64,

    // private
    pub task: Int,
    pub pivot_row: Int,
    pub pivot_col: Int,
    /// >= 0 if FTRAN prepared for update
    pub ftran_for_update: Int,
    /// >= 0 if BTRAN prepared for update
    pub btran_for_update: Int,
    /// see `iwork0` (marked)
    pub marker: Int,
    /// length of pivotcol, pivotrow; <= 2*m
    pub pivotlen: Int,
    /// # columns removed from active submatrix because maximum was 0 or
    /// < abstol
    pub rankdef: Int,
    /// colcount lists 1..min_colnz-1 are empty
    pub min_colnz: Int,
    /// rowcount lists 1..min_rownz-1 are empty
    pub min_rownz: Int,

    // user arrays
    pub lindex: &'a mut [Int],
    pub lvalue: &'a mut [f64],
    pub uindex: &'a mut [Int],
    pub uvalue: &'a mut [f64],
    pub windex: &'a mut [Int],
    pub wvalue: &'a mut [f64],

    // istore partitions
    /// colcount_flink (2m+2) | pivotcol
    pub colcount_flink: &'a mut [Int],
    /// colcount_blink (2m+2) | pivotrow
    pub colcount_blink: &'a mut [Int],
    /// rowcount_flink (2m+2) | Rbegin [..m+1], eta_row [m+1..]
    pub rowcount_flink: &'a mut [Int],
    /// rowcount_blink (2m+2) | iwork1
    pub rowcount_blink: &'a mut [Int],
    /// Wbegin (2m+1) | Wbegin [..m+1], Lbegin [m+1..]
    pub wbegin: &'a mut [Int],
    /// Wend (2m+1) | Wend [..m+1], Ltbegin [m+1..]
    pub wend: &'a mut [Int],
    /// Wflink (2m+1) | Wflink [..m+1], Ltbegin_p [m+1..]
    pub wflink: &'a mut [Int],
    /// Wblink (2m+1) | Wblink [..m+1], p [m+1..]
    pub wblink: &'a mut [Int],
    /// pinv (m) | pmap
    pub pinv: &'a mut [Int],
    /// qinv (m) | qmap
    pub qinv: &'a mut [Int],
    pub lbegin_p: &'a mut [Int],
    pub ubegin: &'a mut [Int],
    /// iwork0: size m workspace, zeroed | marked: 0 <= marked[i] <= marker
    pub iwork0: &'a mut [Int],

    // xstore partitions
    /// size m workspace, zeroed
    pub work0: &'a mut [f64],
    /// size m workspace, uninitialized
    pub work1: &'a mut [f64],
    /// pivot elements by column index
    pub col_pivot: &'a mut [f64],
    /// pivot elements by row index
    pub row_pivot: &'a mut [f64],

    /// xstore[0..512], where the scalars are saved
    xhdr: &'a mut [f64],
}

/// Split the first `n` elements off `s`
fn take<'a, T>(s: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (a, b) = std::mem::take(s).split_at_mut(n);
    *s = b;
    a
}

/// Reinterpret a double workspace as integer workspace of the same length
/// (C: `lu_int *pstack = (void *) this->work1`). The C code never reads
/// these doubles afterwards before writing them.
pub(crate) fn as_int_mut(x: &mut [f64]) -> &mut [Int] {
    // SAFETY: f64 is at least as aligned and twice as large as i32, every
    // bit pattern is a valid i32, and the result borrows x exclusively.
    // Avoids an O(m) allocation in every sparse solve.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut Int, x.len()) }
}

impl<'a> Lu<'a> {
    /// lu_load: initialize from `istore`, `xstore` if these are a valid
    /// BASICLU instance (Err(ERROR_INVALID_STORE) otherwise). `istore` must
    /// hold istore_len(m), `xstore` xstore_len(m) elements; the user arrays
    /// are aliased only and can be empty.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        istore: &'a mut [Int],
        xstore: &'a mut [f64],
        li: &'a mut [Int],
        lx: &'a mut [f64],
        ui: &'a mut [Int],
        ux: &'a mut [f64],
        wi: &'a mut [Int],
        wx: &'a mut [f64],
    ) -> Result<Lu<'a>, Int> {
        if istore.is_empty() || istore[0] != HASH || xstore.is_empty() || xstore[0] != HASH as f64 {
            return Err(ERROR_INVALID_STORE);
        }
        let x = &*xstore;
        let m = x[DIM] as Int;
        let mu = m as usize;

        // partition istore
        let mut is = &mut istore[1..];
        let colcount_flink = take(&mut is, 2 * mu + 2);
        let colcount_blink = take(&mut is, 2 * mu + 2);
        let rowcount_flink = take(&mut is, 2 * mu + 2);
        let rowcount_blink = take(&mut is, 2 * mu + 2);
        let wbegin = take(&mut is, 2 * mu + 1);
        let wend = take(&mut is, 2 * mu + 1);
        let wflink = take(&mut is, 2 * mu + 1);
        let wblink = take(&mut is, 2 * mu + 1);
        let pinv = take(&mut is, mu);
        let qinv = take(&mut is, mu);
        let lbegin_p = take(&mut is, mu + 1);
        let ubegin = take(&mut is, mu + 1);
        let iwork0 = take(&mut is, mu);

        // partition xstore
        let (xhdr, mut xs) = xstore.split_at_mut(512);
        let work0 = take(&mut xs, mu);
        let work1 = take(&mut xs, mu);
        let col_pivot = take(&mut xs, mu);
        let row_pivot = take(&mut xs, mu);

        let x = &*xhdr;
        let mut this = Lu {
            lmem: x[MEMORYL] as Int,
            umem: x[MEMORYU] as Int,
            wmem: x[MEMORYW] as Int,
            droptol: x[DROP_TOLERANCE],
            abstol: x[ABS_PIVOT_TOLERANCE],
            reltol: x[REL_PIVOT_TOLERANCE].min(1.0),
            nzbias: x[BIAS_NONZEROS] as Int,
            maxsearch: x[MAXN_SEARCH_PIVOT] as Int,
            pad: x[PAD] as Int,
            stretch: x[STRETCH],
            compress_thres: x[COMPRESSION_THRESHOLD],
            sparse_thres: x[SPARSE_THRESHOLD],
            search_rows: (x[SEARCH_ROWS] != 0.0) as Int,

            m,
            addmem_l: 0,
            addmem_u: 0,
            addmem_w: 0,
            nupdate: x[NUPDATE] as Int,
            nforrest: x[NFORREST] as Int,
            nfactorize: x[NFACTORIZE] as Int,
            nupdate_total: x[NUPDATE_TOTAL] as Int,
            nforrest_total: x[NFORREST_TOTAL] as Int,
            nsymperm_total: x[NSYMPERM_TOTAL] as Int,
            lnz: x[LNZ] as Int,
            unz: x[UNZ] as Int,
            rnz: x[RNZ] as Int,
            min_pivot: x[MIN_PIVOT],
            max_pivot: x[MAX_PIVOT],
            max_eta: x[MAX_ETA],
            update_cost_numer: x[UPDATE_COST_NUMER],
            update_cost_denom: x[UPDATE_COST_DENOM],
            time_factorize: x[TIME_FACTORIZE],
            time_solve: x[TIME_SOLVE],
            time_update: x[TIME_UPDATE],
            time_factorize_total: x[TIME_FACTORIZE_TOTAL],
            time_solve_total: x[TIME_SOLVE_TOTAL],
            time_update_total: x[TIME_UPDATE_TOTAL],
            lflops: x[LFLOPS] as Int,
            uflops: x[UFLOPS] as Int,
            rflops: x[RFLOPS] as Int,
            condest_l: x[CONDEST_L],
            condest_u: x[CONDEST_U],
            norm_l: x[NORM_L],
            norm_u: x[NORM_U],
            normest_linv: x[NORMEST_LINV],
            normest_uinv: x[NORMEST_UINV],
            onenorm: x[MATRIX_ONENORM],
            infnorm: x[MATRIX_INFNORM],
            residual_test: x[RESIDUAL_TEST],
            matrix_nz: x[MATRIX_NZ] as Int,
            rank: x[RANK] as Int,
            bump_size: x[BUMP_SIZE] as Int,
            bump_nz: x[BUMP_NZ] as Int,
            nsearch_pivot: x[NSEARCH_PIVOT] as Int,
            nexpand: x[NEXPAND] as Int,
            ngarbage: x[NGARBAGE] as Int,
            factor_flops: x[FACTOR_FLOPS] as Int,
            time_singletons: x[TIME_SINGLETONS],
            time_search_pivot: x[TIME_SEARCH_PIVOT],
            time_elim_pivot: x[TIME_ELIM_PIVOT],
            pivot_error: x[PIVOT_ERROR],

            task: x[TASK] as Int,
            pivot_row: x[PIVOT_ROW] as Int,
            pivot_col: x[PIVOT_COL] as Int,
            ftran_for_update: x[FTCOLUMN_IN] as Int,
            btran_for_update: x[FTCOLUMN_OUT] as Int,
            marker: x[MARKER] as Int,
            pivotlen: x[PIVOTLEN] as Int,
            rankdef: x[RANKDEF] as Int,
            min_colnz: x[MIN_COLNZ] as Int,
            min_rownz: x[MIN_ROWNZ] as Int,

            lindex: li,
            lvalue: lx,
            uindex: ui,
            uvalue: ux,
            windex: wi,
            wvalue: wx,
            colcount_flink,
            colcount_blink,
            rowcount_flink,
            rowcount_blink,
            wbegin,
            wend,
            wflink,
            wblink,
            pinv,
            qinv,
            lbegin_p,
            ubegin,
            iwork0,
            work0,
            work1,
            col_pivot,
            row_pivot,
            xhdr,
        };

        // Reset marked if increasing marker by four causes overflow.
        if this.marker > Int::MAX - 4 {
            this.iwork0.fill(0);
            this.marker = 0;
        }

        // One past the final position in Wend must hold the file size.
        // The file has 2*m lines while factorizing and m lines otherwise.
        if this.nupdate >= 0 {
            this.wend[mu] = this.wmem;
        } else {
            this.wend[2 * mu] = this.wmem;
        }
        Ok(this)
    }

    /// lu_save: copy scalar entries (except for user parameters) back to
    /// xstore and store the status code. Returns `status`.
    pub fn save(&mut self, status: Int) -> Int {
        let x = &mut *self.xhdr;
        x[STATUS] = status as f64;
        x[ADD_MEMORYL] = self.addmem_l as f64;
        x[ADD_MEMORYU] = self.addmem_u as f64;
        x[ADD_MEMORYW] = self.addmem_w as f64;

        x[NUPDATE] = self.nupdate as f64;
        x[NFORREST] = self.nforrest as f64;
        x[NFACTORIZE] = self.nfactorize as f64;
        x[NUPDATE_TOTAL] = self.nupdate_total as f64;
        x[NFORREST_TOTAL] = self.nforrest_total as f64;
        x[NSYMPERM_TOTAL] = self.nsymperm_total as f64;
        x[LNZ] = self.lnz as f64;
        x[UNZ] = self.unz as f64;
        x[RNZ] = self.rnz as f64;
        x[MIN_PIVOT] = self.min_pivot;
        x[MAX_PIVOT] = self.max_pivot;
        x[MAX_ETA] = self.max_eta;
        x[UPDATE_COST_NUMER] = self.update_cost_numer;
        x[UPDATE_COST_DENOM] = self.update_cost_denom;
        x[UPDATE_COST] = self.update_cost_numer / self.update_cost_denom;
        x[TIME_FACTORIZE] = self.time_factorize;
        x[TIME_SOLVE] = self.time_solve;
        x[TIME_UPDATE] = self.time_update;
        x[TIME_FACTORIZE_TOTAL] = self.time_factorize_total;
        x[TIME_SOLVE_TOTAL] = self.time_solve_total;
        x[TIME_UPDATE_TOTAL] = self.time_update_total;
        x[LFLOPS] = self.lflops as f64;
        x[UFLOPS] = self.uflops as f64;
        x[RFLOPS] = self.rflops as f64;
        x[CONDEST_L] = self.condest_l;
        x[CONDEST_U] = self.condest_u;
        x[NORM_L] = self.norm_l;
        x[NORM_U] = self.norm_u;
        x[NORMEST_LINV] = self.normest_linv;
        x[NORMEST_UINV] = self.normest_uinv;
        x[MATRIX_ONENORM] = self.onenorm;
        x[MATRIX_INFNORM] = self.infnorm;
        x[RESIDUAL_TEST] = self.residual_test;

        x[MATRIX_NZ] = self.matrix_nz as f64;
        x[RANK] = self.rank as f64;
        x[BUMP_SIZE] = self.bump_size as f64;
        x[BUMP_NZ] = self.bump_nz as f64;
        x[NSEARCH_PIVOT] = self.nsearch_pivot as f64;
        x[NEXPAND] = self.nexpand as f64;
        x[NGARBAGE] = self.ngarbage as f64;
        x[FACTOR_FLOPS] = self.factor_flops as f64;
        x[TIME_SINGLETONS] = self.time_singletons;
        x[TIME_SEARCH_PIVOT] = self.time_search_pivot;
        x[TIME_ELIM_PIVOT] = self.time_elim_pivot;

        x[PIVOT_ERROR] = self.pivot_error;

        x[TASK] = self.task as f64;
        x[PIVOT_ROW] = self.pivot_row as f64;
        x[PIVOT_COL] = self.pivot_col as f64;
        x[FTCOLUMN_IN] = self.ftran_for_update as f64;
        x[FTCOLUMN_OUT] = self.btran_for_update as f64;
        x[MARKER] = self.marker as f64;
        x[PIVOTLEN] = self.pivotlen as f64;
        x[RANKDEF] = self.rankdef as f64;
        x[MIN_COLNZ] = self.min_colnz as f64;
        x[MIN_ROWNZ] = self.min_rownz as f64;
        status
    }

    /// lu_reset: reset for a new factorization, invalidating the current one
    pub fn reset(&mut self) {
        self.nupdate = -1; // invalidate factorization
        self.nforrest = 0;
        self.lnz = 0;
        self.unz = 0;
        self.rnz = 0;
        self.min_pivot = 0.0;
        self.max_pivot = 0.0;
        self.max_eta = 0.0;
        self.update_cost_numer = 0.0;
        self.update_cost_denom = 1.0;
        self.time_factorize = 0.0;
        self.time_solve = 0.0;
        self.time_update = 0.0;
        self.lflops = 0;
        self.uflops = 0;
        self.rflops = 0;
        self.condest_l = 0.0;
        self.condest_u = 0.0;
        self.norm_l = 0.0;
        self.norm_u = 0.0;
        self.normest_linv = 0.0;
        self.normest_uinv = 0.0;
        self.onenorm = 0.0;
        self.infnorm = 0.0;
        self.residual_test = 0.0;

        self.matrix_nz = 0;
        self.rank = 0;
        self.bump_size = 0;
        self.bump_nz = 0;
        self.nsearch_pivot = 0;
        self.nexpand = 0;
        self.ngarbage = 0;
        self.factor_flops = 0;
        self.time_singletons = 0.0;
        self.time_search_pivot = 0.0;
        self.time_elim_pivot = 0.0;

        self.pivot_error = 0.0;

        self.task = NO_TASK;
        self.pivot_row = -1;
        self.pivot_col = -1;
        self.ftran_for_update = -1;
        self.btran_for_update = -1;
        self.marker = 0;
        self.pivotlen = 0;
        self.rankdef = 0;
        self.min_colnz = 1;
        self.min_rownz = 1;

        // One past the final position in Wend must hold the file size.
        // The file has 2*m lines during factorization.
        let m = self.m as usize;
        self.wend[2 * m] = self.wmem;

        // The integer workspace iwork0 must be zeroed for a new
        // factorization. work0 needs to be zero as well; the C code clears
        // m*sizeof(lu_int) bytes of it, i.e. the first m/2 doubles and,
        // for odd m, the low half of the next one. Mirror that exactly.
        self.iwork0.fill(0);
        self.work0[..m / 2].fill(0.0);
        if m % 2 == 1 {
            let x = &mut self.work0[m / 2];
            *x = f64::from_bits(x.to_bits() & 0xFFFF_FFFF_0000_0000);
        }
    }
}

/// lu_initialize: make istore, xstore a BASICLU instance. Set parameters to
/// defaults and initialize global counters. Reset instance for a fresh
/// factorization.
pub(crate) fn initialize(m: Int, istore: &mut [Int], xstore: &mut [f64]) {
    // set constant entries
    istore[0] = HASH;
    xstore[0] = HASH as f64;
    xstore[DIM] = m as f64;

    // set default parameters
    xstore[MEMORYL] = 0.0;
    xstore[MEMORYU] = 0.0;
    xstore[MEMORYW] = 0.0;
    xstore[DROP_TOLERANCE] = 1e-20;
    xstore[ABS_PIVOT_TOLERANCE] = 1e-14;
    xstore[REL_PIVOT_TOLERANCE] = 0.1;
    xstore[BIAS_NONZEROS] = 1.0;
    xstore[MAXN_SEARCH_PIVOT] = 3.0;
    xstore[PAD] = 4.0;
    xstore[STRETCH] = 0.3;
    xstore[COMPRESSION_THRESHOLD] = 0.5;
    xstore[SPARSE_THRESHOLD] = 0.05;
    xstore[REMOVE_COLUMNS] = 0.0;
    xstore[SEARCH_ROWS] = 1.0;

    // initialize global counters
    xstore[NFACTORIZE] = 0.0;
    xstore[NUPDATE_TOTAL] = 0.0;
    xstore[NFORREST_TOTAL] = 0.0;
    xstore[NSYMPERM_TOTAL] = 0.0;
    xstore[TIME_FACTORIZE_TOTAL] = 0.0;
    xstore[TIME_SOLVE_TOTAL] = 0.0;
    xstore[TIME_UPDATE_TOTAL] = 0.0;

    // reset() and save() initialize the remaining slots
    let mut this = Lu::load(
        istore,
        xstore,
        &mut [],
        &mut [],
        &mut [],
        &mut [],
        &mut [],
        &mut [],
    )
    .expect("store was just initialized");
    this.reset();
    this.save(OK);
}
