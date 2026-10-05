//! LP-format reader: a port of io/filereaderlp/reader.cpp together with the
//! model assembly in FilereaderLp::readModelFromFile. C++ reads the file
//! (decompressing .gz through zstr) and hands the bytes over; `read` returns
//! the same model, messages and status, quirks included (see the comments).
//! Names are slices of the input, so nothing is copied per token.

use super::mps::{decimal_prefix, g, Message, Slice, LOG_ERROR, LOG_INFO, LOG_WARNING};
use super::mps::{CONTINUOUS, INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER};
use std::borrow::Cow;
use std::collections::HashMap;

/// Message kind for text the C++ prints with printf rather than logs
pub const PRINTF: i32 = -1;

/// FilereaderRetcode
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(i32)]
pub enum Status {
    #[default]
    Ok = 0,
    Warning = 1,
    ParserError = 3,
}

const INF: f64 = f64::INFINITY;

/// Any lpassert failure: the C++ throws, and the read is a parser error
type R<T> = Result<T, ()>;

fn check(condition: bool) -> R<()> {
    if condition {
        Ok(())
    } else {
        Err(())
    }
}

#[derive(Clone, Copy, Debug)]
enum Raw<'a> {
    Str(&'a [u8]),
    /// Value and the text it was parsed from, which may be a row name
    Cons(f64, &'a [u8]),
    Less,
    Greater,
    Equal,
    Colon,
    FlEnd,
    BrkOp,
    BrkCl,
    Plus,
    Minus,
    Hat,
    Slash,
    Asterisk,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sec {
    ObjMin,
    ObjMax,
    Con,
    Bounds,
    Gen,
    Bin,
    Semi,
    Sos,
    End,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cmp {
    Leq,
    L,
    Eq,
    G,
    Geq,
}

#[derive(Clone, Copy, Debug)]
enum Tok<'a> {
    SecId(Sec),
    SosType,
    VarId(&'a [u8]),
    ConId(&'a [u8]),
    Const(f64),
    Free,
    BrkOp,
    BrkCl,
    Comp(Cmp),
    Slash,
    Asterisk,
    Hat,
}

/// sectionkeywordmap (case sensitive, as is its lookup)
fn keyword(s: &[u8]) -> Option<Sec> {
    Some(match s {
        b"minimize" | b"min" | b"minimum" => Sec::ObjMin,
        b"maximize" | b"max" | b"maximum" => Sec::ObjMax,
        b"subject to" | b"such that" | b"st" | b"s.t." => Sec::Con,
        b"bounds" | b"bound" => Sec::Bounds,
        b"binary" | b"binaries" | b"bin" => Sec::Bin,
        b"general" | b"generals" | b"gen" | b"integer" | b"integers" => Sec::Gen,
        b"semi-continuous" | b"semi" | b"semis" => Sec::Semi,
        b"sos" => Sec::Sos,
        b"end" => Sec::End,
        _ => return None,
    })
}

fn keyword_lc(s: &[u8]) -> Option<Sec> {
    let mut lc = [0u8; 16];
    let lc = lc.get_mut(..s.len())?;
    lc.copy_from_slice(s);
    lc.make_ascii_lowercase();
    keyword(lc)
}

/// The C++ keeps names via strdup, so they end at the first NUL
fn cstr(s: &[u8]) -> &[u8] {
    &s[..s.iter().position(|&b| b == 0).unwrap_or(s.len())]
}

/// C's strtod at the start of `s`: the value and the number of bytes
/// consumed (0 when there is no number). Plain decimals are parsed by Rust,
/// which rounds correctly as strtod does; whatever else strtod might accept
/// (leading \r\v\f, inf, nan, hex) goes to libc. Note that strtod takes the
/// "inf" of a name like "inflow", so the C++ does too.
fn strtod(s: &[u8]) -> (f64, usize) {
    let n = decimal_prefix(s);
    let hex = s.len() > 1 && s[0] == b'0' && matches!(s[1], b'x' | b'X');
    if n > 0 && !hex {
        // ASCII by construction
        return (std::str::from_utf8(&s[..n]).unwrap().parse().unwrap(), n);
    }
    if !matches!(s.first(), Some(b'\r' | 0x0b | 0x0c | b'0' | b'i' | b'I' | b'n' | b'N')) {
        return (0.0, 0);
    }
    // strtod cannot read past a byte outside these, so neither does the copy
    let ws = s.iter().take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')).count();
    let len = ws
        + s[ws..]
            .iter()
            .take_while(|&&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'+' | b'-' | b'(' | b')'))
            .count();
    let c = std::ffi::CString::new(&s[..len]).unwrap();
    extern "C" {
        fn strtod(s: *const std::ffi::c_char, end: *mut *mut std::ffi::c_char) -> f64;
    }
    let mut end = std::ptr::null_mut();
    // SAFETY: c is NUL-terminated and outlives the call; end points into it.
    let v = unsafe { strtod(c.as_ptr(), &mut end) };
    (v, end as usize - c.as_ptr() as usize)
}

/// Reader::readnexttoken, over the input split into lines as getline does
struct Lexer<'a> {
    input: &'a [u8],
    next_line: usize,
    eof: bool,
    line: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn next(&mut self) -> R<Raw<'a>> {
        loop {
            if self.pos == self.line.len() {
                if self.eof {
                    return Ok(Raw::FlEnd);
                }
                let rest = &self.input[self.next_line..];
                let line = match rest.iter().position(|&b| b == b'\n') {
                    Some(k) => &rest[..k],
                    None => {
                        self.eof = true;
                        rest
                    }
                };
                self.next_line += line.len() + 1;
                self.line = line.strip_suffix(b"\r").unwrap_or(line);
                self.pos = 0;
            }
            let (l, p) = (self.line, self.pos);
            let Some(&c) = l.get(p) else { continue }; // empty line
            let t = match c {
                b'\\' | b';' => {
                    self.pos = l.len();
                    continue;
                }
                b' ' | b'\t' => {
                    self.pos += 1;
                    continue;
                }
                0 => return Err(()),
                b'[' => Raw::BrkOp,
                b']' => Raw::BrkCl,
                b'<' => Raw::Less,
                b'>' => Raw::Greater,
                b'=' => Raw::Equal,
                b':' => Raw::Colon,
                b'+' => Raw::Plus,
                b'^' => Raw::Hat,
                b'/' => Raw::Slash,
                b'*' => Raw::Asterisk,
                b'-' => Raw::Minus,
                _ => {
                    let (v, n) = strtod(&l[p..]);
                    if n > 0 {
                        self.pos += n;
                        return Ok(Raw::Cons(v, &l[p..p + n]));
                    }
                    // Identifier, up to the next of "\t\n\\:+<>^= /-*[]";
                    // c is none of those, so it is not empty
                    let end = l[p..]
                        .iter()
                        .position(|b| b"\t\n\\:+<>^= /-*[]".contains(b))
                        .map_or(l.len(), |k| p + k);
                    self.pos = end;
                    return Ok(Raw::Str(&l[p..end]));
                }
            };
            self.pos += 1;
            return Ok(t);
        }
    }
}

/// The three raw tokens of lookahead
struct Window<'a> {
    lexer: Lexer<'a>,
    r: [Raw<'a>; 3],
}

impl Window<'_> {
    fn shift(&mut self, n: usize) -> R<()> {
        for _ in 0..n {
            self.r.rotate_left(1);
            self.r[2] = self.lexer.next()?;
        }
        Ok(())
    }
}

fn eqi(s: &[u8], word: &str) -> bool {
    s.eq_ignore_ascii_case(word.as_bytes())
}

/// Reader::processtokens
fn tokenize<'a>(input: &'a [u8], messages: &mut Vec<(i32, Vec<u8>)>) -> R<Vec<Tok<'a>>> {
    use Raw::*;
    let mut lexer = Lexer { input, next_line: 0, eof: false, line: &[], pos: 0 };
    let r = [lexer.next()?, lexer.next()?, lexer.next()?];
    let mut w = Window { lexer, r };
    let mut toks = Vec::new();
    while !matches!(w.r[0], FlEnd) {
        // A section keyword followed by a colon is a row name
        if let (Str(s), Colon) = (w.r[0], w.r[1]) {
            if keyword(s).is_some() {
                w.r[0] = Cons(0.0, s);
            }
        }
        // /* comment */, matched two tokens at a time
        if let (Slash, Asterisk) = (w.r[0], w.r[1]) {
            loop {
                w.shift(2)?;
                if matches!((w.r[0], w.r[1]), (Asterisk, Slash) | (FlEnd, _)) {
                    break;
                }
            }
            w.shift(2)?;
            continue;
        }
        let (tok, n) = match w.r {
            [Str(a), Minus, Str(c)] if eqi(a, "semi") && eqi(c, "continuous") => (Tok::SecId(Sec::Semi), 3),
            [Str(a), Str(b), _] if (eqi(a, "subject") && eqi(b, "to")) || (eqi(a, "such") && eqi(b, "that")) => {
                (Tok::SecId(Sec::Con), 2)
            }
            [Str(a), ..] if keyword_lc(a).is_some() => (Tok::SecId(keyword_lc(a).unwrap()), 1),
            [Str(a), Colon, Colon] => {
                check(a.len() == 2 && matches!(a[0], b'S' | b's') && matches!(a[1], b'1' | b'2'))?;
                (Tok::SosType, 3)
            }
            [Str(a) | Cons(_, a), Colon, _] => (Tok::ConId(cstr(a)), 2),
            [Str(a), ..] if eqi(a, "free") => (Tok::Free, 1),
            [Str(a), ..] if eqi(a, "infinity") || eqi(a, "inf") => (Tok::Const(INF), 1),
            [Str(a), ..] => (Tok::VarId(cstr(a)), 1),
            [Plus | Minus, ..] => {
                let mut sign = if matches!(w.r[0], Plus) { 1.0 } else { -1.0 };
                w.shift(1)?;
                // another + or - for #948, #950
                if matches!(w.r[0], Plus | Minus) {
                    sign *= if matches!(w.r[0], Plus) { 1.0 } else { -1.0 };
                    w.shift(1)?;
                }
                match w.r[0] {
                    Cons(v, _) => (Tok::Const(sign * v), 1),
                    BrkOp if sign == 1.0 => (Tok::BrkOp, 1),
                    // The sign is a coefficient of the variable that follows
                    Str(_) => (Tok::Const(sign), 0),
                    Greater => {
                        messages.push((
                            PRINTF,
                            b"File appears to contain indicator constraints: cannot currently be handled by HiGHS\n"
                                .to_vec(),
                        ));
                        return Err(());
                    }
                    _ => return Err(()),
                }
            }
            [Cons(..), BrkOp, _] => return Err(()),
            [Cons(v, _), ..] => (Tok::Const(v), 1),
            [BrkOp, ..] => (Tok::BrkOp, 1),
            [BrkCl, ..] => (Tok::BrkCl, 1),
            [Slash, ..] => (Tok::Slash, 1),
            [Asterisk, ..] => (Tok::Asterisk, 1),
            [Hat, ..] => (Tok::Hat, 1),
            [Less, Equal, _] => (Tok::Comp(Cmp::Leq), 2),
            [Less, ..] => (Tok::Comp(Cmp::L), 1),
            [Greater, Equal, _] => (Tok::Comp(Cmp::Geq), 2),
            [Greater, ..] => (Tok::Comp(Cmp::G), 1),
            [Equal, ..] => (Tok::Comp(Cmp::Eq), 1),
            _ => return Err(()),
        };
        toks.push(tok);
        w.shift(n)?;
    }
    Ok(toks)
}

/// Reader::splittokens: the token range of each section. Tokens before the
/// first section keyword are ignored, and a repeated keyword of the same
/// section (e.g. "general x general y") stays inside its range.
fn split(toks: &[Tok]) -> R<[Option<(usize, usize)>; 9]> {
    let mut sections: [Option<(usize, usize)>; 9] = [None; 9];
    let mut current: Option<Sec> = None;
    let mut open = false;
    for (i, t) in toks.iter().enumerate() {
        let Tok::SecId(kw) = *t else { continue };
        let new_section_type = current != Some(kw);
        if new_section_type {
            if let Some(c) = current {
                check(open)?;
                sections[c as usize].as_mut().unwrap().1 = i;
                open = false;
                current = None;
            }
        }
        // End of the tokens (where the C++ reads past the end: taken as a
        // different section), or the new section is empty
        if let next @ (None | Some(Tok::SecId(_))) = toks.get(i + 1) {
            if let Some(c) = current {
                if !matches!(next, Some(&Tok::SecId(k)) if k == c) {
                    check(open)?;
                    sections[c as usize].as_mut().unwrap().1 = i;
                    open = false;
                }
            }
            current = None;
            check(!open)?;
            continue;
        }
        if new_section_type {
            current = Some(kw);
            check(sections[kw as usize].is_none())?;
            check(!open)?;
            sections[kw as usize] = Some((i + 1, i + 1));
            open = true;
        }
        check(open != current.is_none())?;
    }
    check(current.is_none())?;
    Ok(sections)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VarType {
    Continuous,
    Binary,
    General,
    SemiContinuous,
    SemiInteger,
}

struct Var<'a> {
    name: &'a [u8],
    kind: VarType,
    lower: f64,
    upper: f64,
}

/// Builder: variables in order of first appearance
#[derive(Default)]
struct Builder<'a> {
    index: HashMap<&'a [u8], usize>,
    vars: Vec<Var<'a>>,
}

impl<'a> Builder<'a> {
    fn var(&mut self, name: &'a [u8]) -> usize {
        let n = self.vars.len();
        *self.index.entry(name).or_insert_with(|| {
            self.vars.push(Var { name, kind: VarType::Continuous, lower: 0.0, upper: INF });
            n
        })
    }
}

#[derive(Default)]
struct Expr<'a> {
    name: &'a [u8],
    lin: Vec<(usize, f64)>,
    quad: Vec<(usize, usize, f64)>,
    offset: f64,
}

/// Reader::parseexpression over the section `t` from `*i`
fn expression<'a>(t: &[Tok<'a>], i: &mut usize, b: &mut Builder<'a>, is_obj: bool) -> R<Expr<'a>> {
    use Tok::*;
    let mut e = Expr::default();
    if let Some(&ConId(name)) = t.get(*i) {
        e.name = name;
        *i += 1;
    }
    while let Some(&tok) = t.get(*i) {
        match (tok, t.get(*i + 1)) {
            (Const(c), Some(&VarId(v))) => {
                e.lin.push((b.var(v), c));
                *i += 2;
            }
            (Const(c), _) => {
                e.offset += c;
                *i += 1;
            }
            (VarId(v), _) => {
                e.lin.push((b.var(v), 1.0));
                *i += 1;
            }
            (BrkOp, Some(_)) => {
                *i += 1;
                while let Some(&tok) = t.get(*i) {
                    if matches!(tok, BrkCl) {
                        break;
                    }
                    match (tok, t.get(*i + 1), t.get(*i + 2), t.get(*i + 3)) {
                        (Const(c), Some(&VarId(v)), Some(Hat), Some(&Const(p))) => {
                            check(p == 2.0)?;
                            let x = b.var(v);
                            e.quad.push((x, x, c));
                            *i += 4;
                        }
                        (VarId(v), Some(Hat), Some(&Const(p)), _) => {
                            check(p == 2.0)?;
                            let x = b.var(v);
                            e.quad.push((x, x, 1.0));
                            *i += 3;
                        }
                        (Const(c), Some(&VarId(v1)), Some(Asterisk), Some(&VarId(v2))) => {
                            let x = b.var(v1);
                            e.quad.push((x, b.var(v2), c));
                            *i += 4;
                        }
                        (VarId(v1), Some(Asterisk), Some(&VarId(v2)), _) => {
                            let x = b.var(v1);
                            e.quad.push((x, b.var(v2), 1.0));
                            *i += 3;
                        }
                        _ => break,
                    }
                }
                // Only in the objective is a quadratic term followed by "/2"
                if is_obj {
                    check(matches!(
                        (t.get(*i), t.get(*i + 1), t.get(*i + 2)),
                        (Some(BrkCl), Some(Slash), Some(&Const(p))) if p == 2.0
                    ))?;
                    *i += 3;
                } else {
                    check(matches!(t.get(*i), Some(BrkCl)))?;
                    *i += 1;
                }
            }
            _ => break,
        }
    }
    Ok(e)
}

/// The model as FilereaderLp::readModelFromFile leaves it
#[derive(Default)]
pub struct Lp<'a> {
    pub status: Status,
    /// (kind, text): PRINTF or a HighsLogType, in output order
    pub messages: Vec<(i32, Vec<u8>)>,
    pub maximize: bool,
    pub offset: f64,
    pub objective_name: &'a [u8],
    pub col_names: Vec<&'a [u8]>,
    pub col_cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    /// Empty unless some column is not continuous
    pub integrality: Vec<u8>,
    /// Empty if made-up names would clash with given ones
    pub row_names: Vec<Cow<'a, [u8]>>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    /// Column-wise; index and value keep their length when duplicate
    /// entries are merged, as in the C++
    pub a_start: Vec<i32>,
    pub a_index: Vec<i32>,
    pub a_value: Vec<f64>,
    /// Empty when the Hessian has no nonzeros
    pub q_start: Vec<i32>,
    pub q_index: Vec<i32>,
    pub q_value: Vec<f64>,
}

/// Parse the contents of an .lp file
pub fn read(input: &[u8]) -> Lp<'_> {
    let mut lp = Lp::default();
    if parse(input, &mut lp).is_err() {
        lp.status = Status::ParserError;
    }
    lp
}

/// Compressed columns of the nonzero (column, row, value) entries, in their
/// order within each column
fn by_column(n: usize, entries: &[(usize, i32, f64)]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut start = vec![0i32; n + 1];
    for &(j, _, v) in entries {
        if v != 0.0 {
            start[j + 1] += 1;
        }
    }
    for j in 0..n {
        start[j + 1] += start[j];
    }
    let nnz = start[n] as usize;
    let (mut index, mut value) = (vec![0; nnz], vec![0.0; nnz]);
    let mut next: Vec<usize> = start[..n].iter().map(|&s| s as usize).collect();
    for &(j, i, v) in entries {
        if v != 0.0 {
            index[next[j]] = i;
            value[next[j]] = v;
            next[j] += 1;
        }
    }
    (start, index, value)
}

fn parse<'a>(input: &'a [u8], lp: &mut Lp<'a>) -> R<()> {
    let toks = tokenize(input, &mut lp.messages)?;
    let sections = split(&toks)?;
    // Nothing may follow "end" but other sections
    check(sections[Sec::End as usize].is_none())?;
    let section = |s: Sec| sections[s as usize].map(|(from, to)| &toks[from..to]);
    let mut b = Builder::default();

    // Objective: a minimize section wins over a maximize one
    let mut objective = Expr::default();
    if let Some((t, max)) = section(Sec::ObjMin).map(|t| (t, false)).or(section(Sec::ObjMax).map(|t| (t, true))) {
        lp.maximize = max;
        let mut i = 0;
        objective = expression(t, &mut i, &mut b, true)?;
        check(i == t.len())?;
    }

    // Constraints: (column, row, value) entries; a row's constant is ignored
    let mut entries = Vec::new();
    let mut quadratic_rows = false;
    if let Some(t) = section(Sec::Con) {
        let mut i = 0;
        while i < t.len() {
            let e = expression(t, &mut i, &mut b, false)?;
            let Some(&Tok::Comp(dir)) = t.get(i) else { return Err(()) };
            let Some(&Tok::Const(v)) = t.get(i + 1) else { return Err(()) };
            let (lower, upper) = match dir {
                Cmp::Eq => (v, v),
                Cmp::Leq => (-INF, v),
                Cmp::Geq => (v, INF),
                _ => return Err(()),
            };
            i += 2;
            let row = lp.row_lower.len() as i32;
            entries.extend(e.lin.iter().map(|&(j, c)| (j, row, c)));
            quadratic_rows |= !e.quad.is_empty();
            lp.row_names.push(Cow::Borrowed(e.name));
            lp.row_lower.push(lower);
            lp.row_upper.push(upper);
        }
    }

    if let Some(t) = section(Sec::Bounds) {
        use Tok::*;
        let mut i = 0;
        while i < t.len() {
            match (t[i], t.get(i + 1), t.get(i + 2), t.get(i + 3), t.get(i + 4)) {
                (VarId(v), Some(Free), ..) => {
                    let j = b.var(v);
                    let var = &mut b.vars[j];
                    (var.lower, var.upper) = (-INF, INF);
                    i += 2;
                }
                (Const(l), Some(&Comp(d1)), Some(&VarId(v)), Some(&Comp(d2)), Some(&Const(u))) => {
                    check(d1 == Cmp::Leq && d2 == Cmp::Leq)?;
                    let j = b.var(v);
                    let var = &mut b.vars[j];
                    (var.lower, var.upper) = (l, u);
                    i += 5;
                }
                (Const(x), Some(&Comp(d)), Some(&VarId(v)), ..) => {
                    let j = b.var(v);
                    let var = &mut b.vars[j];
                    match d {
                        Cmp::Leq => var.lower = x,
                        Cmp::Geq => var.upper = x,
                        Cmp::Eq => (var.lower, var.upper) = (x, x),
                        _ => return Err(()),
                    }
                    i += 3;
                }
                (VarId(v), Some(&Comp(d)), Some(&Const(x)), ..) => {
                    let j = b.var(v);
                    let var = &mut b.vars[j];
                    match d {
                        Cmp::Leq => var.upper = x,
                        Cmp::Geq => var.lower = x,
                        Cmp::Eq => (var.lower, var.upper) = (x, x),
                        _ => return Err(()),
                    }
                    i += 3;
                }
                _ => return Err(()),
            }
        }
    }

    // Variable lists, in the C++ order general, binary, semi-continuous
    for sec in [Sec::Gen, Sec::Bin, Sec::Semi] {
        for &tok in section(sec).unwrap_or(&[]) {
            match tok {
                // Possible to have repeat of keyword for this section type
                Tok::SecId(k) => check(k == sec)?,
                Tok::VarId(v) => {
                    let j = b.var(v);
                    let var = &mut b.vars[j];
                    var.kind = match (sec, var.kind) {
                        (Sec::Gen, VarType::SemiContinuous) | (Sec::Semi, VarType::General) => VarType::SemiInteger,
                        (Sec::Gen, _) => VarType::General,
                        (Sec::Semi, _) => VarType::SemiContinuous,
                        _ => {
                            // Respect any bounds already declared
                            if var.upper == INF {
                                var.upper = 1.0;
                            }
                            VarType::Binary
                        }
                    };
                }
                _ => return Err(()),
            }
        }
    }

    // SOS: "name: S1:: x1:1 x2:2" (a "var:" is a ConId here). Parsed only
    // to tell a malformed section from an unsupported one.
    let mut sos = false;
    if let Some(t) = section(Sec::Sos) {
        let mut i = 0;
        while i < t.len() {
            check(matches!(t[i], Tok::ConId(_)))?;
            check(matches!(t.get(i + 1), Some(Tok::SosType)))?;
            i += 2;
            while let (Some(&Tok::ConId(v)), Some(Tok::Const(_))) = (t.get(i), t.get(i + 1)) {
                b.var(v);
                i += 2;
            }
            sos = true;
        }
    }
    if sos {
        lp.messages.push((LOG_ERROR, b"SOS not supported by HiGHS\n".to_vec()));
        return Err(());
    }
    if quadratic_rows {
        lp.messages.push((LOG_ERROR, b"Quadratic constraints not supported by HiGHS\n".to_vec()));
        return Err(());
    }

    // Columns
    let num_col = b.vars.len();
    for v in &b.vars {
        lp.col_names.push(v.name);
        lp.col_lower.push(v.lower);
        lp.col_upper.push(v.upper);
        lp.integrality.push(match v.kind {
            VarType::Binary | VarType::General => INTEGER,
            VarType::SemiContinuous => SEMI_CONTINUOUS,
            VarType::SemiInteger => SEMI_INTEGER,
            VarType::Continuous => CONTINUOUS,
        });
    }
    if lp.integrality.iter().all(|&k| k == CONTINUOUS) {
        lp.integrality.clear();
    }
    lp.objective_name = objective.name;
    lp.offset = objective.offset;
    lp.col_cost = vec![0.0; num_col];
    // A repeated variable's last coefficient wins
    for &(j, c) in &objective.lin {
        lp.col_cost[j] = c;
    }

    // Hessian: both triangles of each term, unsummed, in term order
    let mut q = Vec::new();
    for &(x, y, c) in &objective.quad {
        if x != y {
            q.push((x, y as i32, c / 2.0));
            q.push((y, x as i32, c / 2.0));
        } else {
            q.push((x, x as i32, c));
        }
    }
    if q.iter().any(|e| e.2 != 0.0) {
        (lp.q_start, lp.q_index, lp.q_value) = by_column(num_col, &q);
    }

    // Empty row names become HiGHS_R<row>, unless given names have that prefix
    let mut prefix_ok = true;
    let mut used_prefix = false;
    for (i, name) in lp.row_names.iter_mut().enumerate() {
        if name.starts_with(b"HiGHS_R") {
            lp.messages.push((PRINTF, [&b"Name "[..], name, b" begins with \"HiGHS_R\"\n"].concat()));
            prefix_ok = false;
        } else if name.is_empty() {
            *name = Cow::Owned(format!("HiGHS_R{i}").into_bytes());
            used_prefix = true;
        }
    }
    if used_prefix && !prefix_ok {
        lp.row_names.clear();
        lp.messages.push((
            LOG_WARNING,
            b"Cannot create row name beginning \"HiGHS_R\" due to others with same prefix: row names cleared\n"
                .to_vec(),
        ));
    }

    // Matrix, summing repeated entries of a column in place
    let (mut start, mut index, mut value) = by_column(num_col, &entries);
    let num_row = lp.row_lower.len();
    let mut column = vec![0.0; num_row];
    let mut nz_count = vec![0; num_row];
    let mut zero_count = vec![0; num_row];
    let (mut sum_num_duplicate, mut sum_num_zero, mut sum_cancellation) = (0, 0, 0);
    let mut num_report = 0;
    const MAX_NUM_REPORT: i32 = 10;
    let mut num_nz = 0;
    for j in 0..num_col {
        let (from, to) = (start[j] as usize, start[j + 1] as usize);
        for k in from..to {
            let i = index[k] as usize;
            if value[k] != 0.0 {
                column[i] += value[k];
                nz_count[i] += 1;
            } else {
                zero_count[i] += 1;
            }
        }
        start[j] = num_nz as i32;
        for k in from..to {
            let i = index[k] as usize;
            if column[i] != 0.0 {
                index[num_nz] = i as i32;
                value[num_nz] = column[i];
                num_nz += 1;
            }
            let num_occurrence = zero_count[i] + nz_count[i];
            if num_occurrence > 1 {
                // The C++ indexes cleared row names out of bounds
                let row_name = lp.row_names.get(i).map_or(&b""[..], |n| &n[..]);
                if nz_count[i] > 1 {
                    if num_report < MAX_NUM_REPORT {
                        let text = [
                            format!("Column {j} (name \"").as_bytes(),
                            lp.col_names[j],
                            format!("\") occurs {num_occurrence} times in row {i} (name \"").as_bytes(),
                            row_name,
                            format!("\"): values summed to {}\n", g(column[i])).as_bytes(),
                        ]
                        .concat();
                        lp.messages.push((LOG_WARNING, text));
                    }
                    num_report += 1;
                }
                if zero_count[i] > 0 {
                    if num_report < MAX_NUM_REPORT {
                        let s = if zero_count[i] > 1 { "s" } else { "" };
                        let text = [
                            format!("Column {j} (name \"").as_bytes(),
                            lp.col_names[j],
                            format!("\") contains {} explicit zero coefficient{s} in row {i} (name \"", zero_count[i])
                                .as_bytes(),
                            row_name,
                            b"\")\n",
                        ]
                        .concat();
                        lp.messages.push((LOG_WARNING, text));
                    }
                    num_report += 1;
                }
                sum_num_duplicate += num_occurrence - 1;
                sum_num_zero += zero_count[i];
                if column[i] == 0.0 && nz_count[i] > 0 {
                    sum_cancellation += 1;
                }
            }
            zero_count[i] = 0;
            nz_count[i] = 0;
            column[i] = 0.0;
        }
    }
    start[num_col] = num_nz as i32;
    (lp.a_start, lp.a_index, lp.a_value) = (start, index, value);

    let plural = |n: i32| if n > 1 { "s" } else { "" };
    let num_report_skipped = num_report - MAX_NUM_REPORT;
    if num_report_skipped > 0 {
        let text = format!("Skipped {num_report_skipped} further warning{} of this kind\n", plural(num_report_skipped));
        lp.messages.push((LOG_INFO, text.into_bytes()));
    }
    if sum_num_duplicate > 0 {
        let text = format!(
            "lp file contains {sum_num_duplicate} repeated variable{} in constraints: summing them yielded \
             {sum_cancellation} cancellation{}\n",
            plural(sum_num_duplicate),
            if sum_cancellation == 1 { "" } else { "s" }
        );
        lp.messages.push((LOG_WARNING, text.into_bytes()));
    }
    if sum_num_zero > 0 {
        let text = format!("lp file contains {sum_num_zero} explicit zero{}\n", plural(sum_num_zero));
        lp.messages.push((LOG_WARNING, text.into_bytes()));
    }
    if sum_num_duplicate > 0 || sum_num_zero > 0 {
        lp.status = Status::Warning;
    }
    Ok(())
}

/// What C++ copies into the HighsModel; mirrored in FilereaderLp.cpp
#[repr(C)]
pub struct LpView {
    status: i32,
    maximize: bool,
    num_row: i32,
    num_col: i32,
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

/// Owner of everything an LpView points to
pub struct LpHandle {
    _lp: Lp<'static>,
    _row_names: Vec<Slice<u8>>,
    _col_names: Vec<Slice<u8>>,
    _messages: Vec<Message>,
}

/// Parse `len` bytes at `buf` and fill `view`. Returns the handle owning
/// the view's arrays, to be freed with highs_rs_lp_free.
///
/// # Safety
/// `buf` must be valid for `len` bytes and outlive the handle (names point
/// into it); `view` must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lp_read(buf: *const u8, len: usize, view: *mut LpView) -> *mut LpHandle {
    let input: &'static [u8] = if len == 0 { &[] } else { std::slice::from_raw_parts(buf, len) };
    let lp = read(input);
    let row_names: Vec<_> = lp.row_names.iter().map(|s| Slice::new(s)).collect();
    let col_names: Vec<_> = lp.col_names.iter().map(|s| Slice::new(s)).collect();
    let messages: Vec<_> = lp.messages.iter().map(|(kind, text)| Message { kind: *kind, text: Slice::new(text) }).collect();
    *view = LpView {
        status: lp.status as i32,
        maximize: lp.maximize,
        num_row: lp.row_lower.len() as i32,
        num_col: lp.col_lower.len() as i32,
        offset: lp.offset,
        a_start: Slice::new(&lp.a_start),
        a_index: Slice::new(&lp.a_index),
        a_value: Slice::new(&lp.a_value),
        col_cost: Slice::new(&lp.col_cost),
        col_lower: Slice::new(&lp.col_lower),
        col_upper: Slice::new(&lp.col_upper),
        row_lower: Slice::new(&lp.row_lower),
        row_upper: Slice::new(&lp.row_upper),
        integrality: Slice::new(&lp.integrality),
        q_start: Slice::new(&lp.q_start),
        q_index: Slice::new(&lp.q_index),
        q_value: Slice::new(&lp.q_value),
        objective_name: Slice::new(lp.objective_name),
        row_names: Slice::new(&row_names),
        col_names: Slice::new(&col_names),
        messages: Slice::new(&messages),
    };
    // Moving the Vecs into the box keeps their heap buffers in place
    Box::into_raw(Box::new(LpHandle { _lp: lp, _row_names: row_names, _col_names: col_names, _messages: messages }))
}

/// # Safety
/// `handle` must come from highs_rs_lp_read and not be freed already.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lp_free(handle: *mut LpHandle) {
    drop(Box::from_raw(handle));
}

#[cfg(test)]
mod tests {
    use super::*;

    const LP: &str = "\\ comment
Maximize
 obj: 2 x + - 3.5e0 y + 1 + [ x^2 + 4 x * y ] / 2
Subject To
 c1: x + y + x <= 4
 - y >= -inf
 r: x - y = 1.5
Bounds
 -1 <= x <= 3
 y free
General
 x
Semi-continuous
 z
End
";

    #[test]
    fn small_qp() {
        let m = read(LP.as_bytes());
        assert_eq!(m.status, Status::Warning);
        assert!(m.maximize);
        assert_eq!(m.objective_name, b"obj");
        assert_eq!(m.offset, 1.0);
        assert_eq!(m.col_names, [&b"x"[..], b"y", b"z"]);
        assert_eq!(m.col_cost, [2.0, -3.5, 0.0]);
        assert_eq!((m.col_lower, m.col_upper), (vec![-1.0, -INF, 0.0], vec![3.0, INF, INF]));
        assert_eq!(m.integrality, [INTEGER, CONTINUOUS, SEMI_CONTINUOUS]);
        assert_eq!(m.row_names, [&b"c1"[..], b"HiGHS_R1", b"r"]);
        assert_eq!((m.row_lower, m.row_upper), (vec![-INF, -INF, 1.5], vec![4.0, INF, 1.5]));
        // x occurs twice in c1: summed in place, index/value keep their length
        assert_eq!(m.a_start, [0, 2, 5, 5]);
        assert_eq!(m.a_index[..5], [0, 2, 0, 1, 2]);
        assert_eq!(m.a_value[..5], [2.0, 1.0, 1.0, -1.0, -1.0]);
        assert_eq!(m.a_index.len(), 6);
        assert_eq!((m.q_start, m.q_index, m.q_value), (vec![0, 2, 3, 3], vec![0, 1, 0], vec![1.0, 2.0, 2.0]));
        assert!(m.messages[0].1.starts_with(b"Column 0 (name \"x\") occurs 2 times in row 0"));
    }

    #[test]
    fn errors_and_strtod() {
        assert_eq!(read(b"min\n x\nst\n x + y\n").status, Status::ParserError);
        assert_eq!(read(b"min\n x\nst\n c: x >= 1\nsos\n s: S1:: x:1\nend\n").status, Status::ParserError);
        assert_eq!(read(b"min x\0\n").status, Status::ParserError);
        assert_eq!(strtod(b"1.e5x"), (1e5, 4));
        assert_eq!(strtod(b"inflow"), (INF, 3));
        assert_eq!(strtod(b"0x10 "), (16.0, 4));
        assert_eq!(strtod(b"\r -2"), (-2.0, 4));
        assert_eq!(strtod(b"x1").1, 0);
    }
}
