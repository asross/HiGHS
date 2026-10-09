//! HighsLp's data that the solvers use (dimensions, costs, bounds, the
//! constraint matrix, sense, offset, integrality and scaling), over
//! vectors that Rust owns or C++ std::vectors that Rust resizes
//! ([`LpG`], like [`Mat`]): the same code edits both. The simplex engine's
//! LP ([`crate::simplex::lp_solver::LpSolver::lp`], an [`Lp`]) is copied
//! from the C++ HighsLp being solved when the solve starts
//! ([`Lp::import`]), scaled and dualized in place, and only its scale
//! factors go back to the C++ LP, which is never moved or scaled any more.
//! A C++ HighsLp that Rust edits in place is a [`CppLp`] (HighsRust.h:
//! RsLpVec; C++ copies the scalars back). Rust functions written for C++
//! views ([`CLp`]) take [`LpG::view`].

use super::ffi::{CLp, CMatrix, RsMut, RsVec};
#[cfg(test)]
use super::matrix_format;
use super::sparse::{Buf, Mat};

pub use super::sparse::SparseMatrix;

/// The view of a vector for the CLp/CMatrix views
fn vw<T: Copy + Default, B: Buf<T>>(v: &mut B) -> RsMut<T> {
    let s = v.sl_mut();
    RsMut { ptr: s.as_mut_ptr(), len: s.len() }
}

pub(crate) fn rs<T>(v: &mut Vec<T>) -> RsMut<T> {
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

impl<I: Buf<i32>, F: Buf<f64>> Mat<I, F> {
    pub fn view(&mut self) -> CMatrix {
        CMatrix {
            format: self.format,
            num_col: self.num_col,
            num_row: self.num_row,
            start: vw(&mut self.start),
            p_end: vw(&mut self.p_end),
            index: vw(&mut self.index),
            value: vw(&mut self.value),
        }
    }
}

/// HighsScale
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleG<F> {
    pub strategy: i32,
    pub has_scaling: bool,
    pub num_col: i32,
    pub num_row: i32,
    pub cost: f64,
    pub col: F,
    pub row: F,
}

pub type Scale = ScaleG<Vec<f64>>;

impl Default for Scale {
    fn default() -> Self {
        ScaleG { strategy: 0, has_scaling: false, num_col: 0, num_row: 0, cost: 0.0, col: Vec::new(), row: Vec::new() }
    }
}

/// HighsLp's data for the solvers: see the module comment
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct LpG<F, I, U> {
    pub num_col: i32,
    pub num_row: i32,
    pub col_cost: F,
    pub col_lower: F,
    pub col_upper: F,
    pub row_lower: F,
    pub row_upper: F,
    pub a: Mat<I, F>,
    /// ObjSense: 1 minimize, -1 maximize
    pub sense: i32,
    pub offset: f64,
    pub integrality: U,
    pub scale: ScaleG<F>,
    pub is_scaled: bool,
    pub is_moved: bool,
    pub has_infinite_cost: bool,
}

/// A C++ HighsLp edited in place (HighsRust.h: RsLpVec)
pub type CppLp = LpG<RsVec<f64>, RsVec<i32>, RsVec<u8>>;

// The layouts HighsRust.h mirrors
const _: () = assert!(std::mem::size_of::<CppLp>() == 456);
const _: () = assert!(std::mem::size_of::<Mat<RsVec<i32>, RsVec<f64>>>() == 144);
const _: () = assert!(std::mem::size_of::<ScaleG<RsVec<f64>>>() == 88);

pub type LpCore = LpG<Vec<f64>, Vec<i32>, Vec<u8>>;

impl Default for LpCore {
    /// HighsLp() (clear)
    fn default() -> Self {
        LpG {
            num_col: 0,
            num_row: 0,
            col_cost: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            a: SparseMatrix::default(),
            sense: 1,
            offset: 0.0,
            integrality: Vec::new(),
            scale: Scale::default(),
            is_scaled: false,
            is_moved: false,
            has_infinite_cost: false,
        }
    }
}

/// The Rust-owned LP: the data and the model name
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Lp {
    pub g: LpCore,
    pub model_name: Vec<u8>,
}

impl std::ops::Deref for Lp {
    type Target = LpCore;
    fn deref(&self) -> &LpCore {
        &self.g
    }
}

impl std::ops::DerefMut for Lp {
    fn deref_mut(&mut self) -> &mut LpCore {
        &mut self.g
    }
}

fn copy_into<T: Copy>(dst: &mut Vec<T>, src: &RsMut<T>) {
    dst.clear();
    // SAFETY: the C++ view is valid for the call
    dst.extend_from_slice(unsafe { src.get() });
}

impl<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>> LpG<F, I, U> {
    /// The view that Rust functions written for C++ LPs take: valid while
    /// the LP's vectors are not resized
    pub fn view(&mut self) -> CLp {
        CLp {
            num_col: self.num_col,
            num_row: self.num_row,
            col_cost: vw(&mut self.col_cost),
            col_lower: vw(&mut self.col_lower),
            col_upper: vw(&mut self.col_upper),
            row_lower: vw(&mut self.row_lower),
            row_upper: vw(&mut self.row_upper),
            a: self.a.view(),
            sense: self.sense,
            offset: self.offset,
            integrality: vw(&mut self.integrality),
            scale_strategy: self.scale.strategy,
            scale_has_scaling: self.scale.has_scaling,
            scale_num_col: self.scale.num_col,
            scale_num_row: self.scale.num_row,
            scale_cost: self.scale.cost,
            scale_col: vw(&mut self.scale.col),
            scale_row: vw(&mut self.scale.row),
            is_scaled: self.is_scaled,
            is_moved: self.is_moved,
            has_infinite_cost: self.has_infinite_cost,
        }
    }

    /// Take the scalars that a Rust function changed in a view (rsLpBack)
    pub fn take_scalars(&mut self, v: &CLp) {
        self.scale.strategy = v.scale_strategy;
        self.scale.has_scaling = v.scale_has_scaling;
        self.scale.num_col = v.scale_num_col;
        self.scale.num_row = v.scale_num_row;
        self.scale.cost = v.scale_cost;
        self.is_scaled = v.is_scaled;
        self.has_infinite_cost = v.has_infinite_cost;
    }

    /// HighsLp::clearScale
    pub fn clear_scale(&mut self) {
        let s = &mut self.scale;
        s.strategy = super::lp_utils::SCALE_OFF;
        s.has_scaling = false;
        s.num_col = 0;
        s.num_row = 0;
        s.cost = 0.0;
        s.col.clear();
        s.row.clear();
    }

    /// HighsLp::applyScale
    pub fn apply_scale(&mut self) {
        let mut v = self.view();
        super::lp_utils::apply_scale(&mut v);
        self.take_scalars(&v);
    }

    /// HighsLp::unapplyScale
    pub fn unapply_scale(&mut self) {
        let mut v = self.view();
        super::lp_utils::unapply_scale(&mut v);
        self.take_scalars(&v);
    }

    /// HighsLp::clearScaling
    pub fn clear_scaling(&mut self) {
        self.unapply_scale();
        self.clear_scale();
    }

    pub fn is_colwise(&self) -> bool {
        self.a.is_colwise()
    }
}

impl Lp {
    /// A copy of a C++ LP (the vectors keep their capacity)
    ///
    /// # Safety
    /// The view's arrays must be valid
    pub unsafe fn import(&mut self, v: &CLp, model_name: &[u8]) {
        let lp = &mut self.g;
        lp.num_col = v.num_col;
        lp.num_row = v.num_row;
        copy_into(&mut lp.col_cost, &v.col_cost);
        copy_into(&mut lp.col_lower, &v.col_lower);
        copy_into(&mut lp.col_upper, &v.col_upper);
        copy_into(&mut lp.row_lower, &v.row_lower);
        copy_into(&mut lp.row_upper, &v.row_upper);
        let a = &mut lp.a;
        a.format = v.a.format;
        a.num_col = v.a.num_col;
        a.num_row = v.a.num_row;
        copy_into(&mut a.start, &v.a.start);
        copy_into(&mut a.p_end, &v.a.p_end);
        copy_into(&mut a.index, &v.a.index);
        copy_into(&mut a.value, &v.a.value);
        lp.sense = v.sense;
        lp.offset = v.offset;
        copy_into(&mut lp.integrality, &v.integrality);
        self.import_scale(v);
        let lp = &mut self.g;
        lp.is_scaled = v.is_scaled;
        lp.is_moved = false;
        lp.has_infinite_cost = v.has_infinite_cost;
        self.model_name.clear();
        self.model_name.extend_from_slice(model_name);
    }

    /// The LP data into a C++ LP (the inverse of import; its is_moved_
    /// and what Lp does not hold are kept)
    pub fn export(&self, c: &mut CppLp) {
        fn set<T: Copy + Default, B: Buf<T>>(v: &mut B, s: &[T]) {
            v.resize(s.len());
            v.sl_mut().copy_from_slice(s);
        }
        let r = &self.g;
        c.num_col = r.num_col;
        c.num_row = r.num_row;
        set(&mut c.col_cost, &r.col_cost);
        set(&mut c.col_lower, &r.col_lower);
        set(&mut c.col_upper, &r.col_upper);
        set(&mut c.row_lower, &r.row_lower);
        set(&mut c.row_upper, &r.row_upper);
        let (a, ca) = (&r.a, &mut c.a);
        ca.format = a.format;
        ca.num_col = a.num_col;
        ca.num_row = a.num_row;
        set(&mut ca.start, &a.start);
        set(&mut ca.p_end, &a.p_end);
        set(&mut ca.index, &a.index);
        set(&mut ca.value, &a.value);
        c.sense = r.sense;
        c.offset = r.offset;
        set(&mut c.integrality, &r.integrality);
        let (s, cs) = (&r.scale, &mut c.scale);
        cs.strategy = s.strategy;
        cs.has_scaling = s.has_scaling;
        cs.num_col = s.num_col;
        cs.num_row = s.num_row;
        cs.cost = s.cost;
        set(&mut cs.col, &s.col);
        set(&mut cs.row, &s.row);
        c.is_scaled = r.is_scaled;
        c.has_infinite_cost = r.has_infinite_cost;
    }

    /// Which of the LP data differ from a C++ LP's (none: empty)
    ///
    /// # Safety
    /// The view's arrays must be valid
    pub unsafe fn differences(&self, v: &CLp, model_name: &[u8]) -> String {
        let mut t = Lp::default();
        t.import(v, model_name);
        let (a, b) = (&self.g, &t.g);
        let mut d = Vec::new();
        let verbose = std::env::var("HIGHS_RS_CHECK_SYNC").is_ok_and(|v| v == "2");
        macro_rules! cmp {
            ($($f:ident).+) => {
                if a.$($f).+ != b.$($f).+ {
                    d.push(stringify!($($f).+));
                    if verbose {
                        eprintln!("{}: engine {:?} C++ {:?}", stringify!($($f).+), a.$($f).+, b.$($f).+);
                    }
                }
            };
        }
        cmp!(num_col);
        cmp!(num_row);
        cmp!(col_cost);
        cmp!(col_lower);
        cmp!(col_upper);
        cmp!(row_lower);
        cmp!(row_upper);
        cmp!(a.format);
        cmp!(a.num_col);
        cmp!(a.num_row);
        cmp!(a.start);
        cmp!(a.p_end);
        cmp!(a.index);
        cmp!(a.value);
        cmp!(sense);
        cmp!(offset);
        cmp!(integrality);
        cmp!(scale);
        cmp!(is_scaled);
        cmp!(has_infinite_cost);
        if self.model_name != t.model_name {
            d.push("model_name");
        }
        d.join(" ")
    }

    /// The scale of a C++ LP
    ///
    /// # Safety
    /// The view's scale vectors must be valid
    pub unsafe fn import_scale(&mut self, v: &CLp) {
        let s = &mut self.g.scale;
        s.strategy = v.scale_strategy;
        s.has_scaling = v.scale_has_scaling;
        s.num_col = v.scale_num_col;
        s.num_row = v.scale_num_row;
        s.cost = v.scale_cost;
        copy_into(&mut s.col, &v.scale_col);
        copy_into(&mut s.row, &v.scale_row);
    }

    /// HighsLp::clear (the vectors keep their capacity)
    pub fn clear(&mut self) {
        let lp = &mut self.g;
        for v in [&mut lp.col_cost, &mut lp.col_lower, &mut lp.col_upper, &mut lp.row_lower, &mut lp.row_upper] {
            v.clear();
        }
        lp.integrality.clear();
        lp.a.clear();
        lp.num_col = 0;
        lp.num_row = 0;
        lp.sense = 1;
        lp.offset = 0.0;
        lp.clear_scale();
        lp.is_scaled = false;
        lp.is_moved = false;
        lp.has_infinite_cost = false;
        self.model_name.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_and_view_round_trip() {
        let mut src = Lp {
            g: LpG {
                num_col: 2,
                num_row: 1,
                col_cost: vec![1.0, -2.0],
                col_lower: vec![0.0, 0.0],
                col_upper: vec![4.0, f64::INFINITY],
                row_lower: vec![-f64::INFINITY],
                row_upper: vec![3.0],
                a: Mat {
                    format: matrix_format::COLWISE,
                    num_col: 2,
                    num_row: 1,
                    start: vec![0, 1, 2],
                    p_end: vec![],
                    index: vec![0, 0],
                    value: vec![1.0, 3.0],
                },
                sense: 1,
                offset: 0.5,
                ..Default::default()
            },
            model_name: b"m".to_vec(),
        };
        let v = src.view();
        let mut dst = Lp::default();
        // SAFETY: src lives
        unsafe { dst.import(&v, b"m") };
        assert_eq!(dst, src);
        assert_eq!(dst.a.num_nz(), 2);
    }
}
