//! HighsRandom: xorshift state with hashed outputs. Draws are identical to
//! the C++ (HighsInt is 32 bits).

use crate::util::fma::ClangFma;

use super::hash::{log2i, pair_hash};

/// Layout of the C++ HighsRandom (one uint64_t), so C++ generators can be
/// used in place
#[repr(transparent)]
#[derive(Clone)]
pub struct HighsRandom {
    state: u64,
}

impl HighsRandom {
    pub fn new(seed: u32) -> Self {
        let mut r = HighsRandom { state: 0 };
        r.initialise(seed);
        r
    }

    pub fn initialise(&mut self, seed: u32) {
        self.state = seed as u64;
        loop {
            self.state = pair_hash::<0>(self.state as u32, (self.state >> 32) as u32);
            self.state ^= pair_hash::<1>((self.state >> 32) as u32, seed) >> 32;
            if self.state != 0 {
                break;
            }
        }
    }

    /// The generator with a given state, such as a C++ HighsRandom's
    pub fn from_state(state: u64) -> Self {
        HighsRandom { state }
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    #[inline]
    fn advance(&mut self) {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
    }

    #[inline]
    fn lo_hi(&self) -> (u32, u32) {
        (self.state as u32, (self.state >> 32) as u32)
    }

    /// Uniform in [0, sup), using every output of a state before advancing
    /// (output 8 is skipped, as in the C++).
    fn draw_uniform(&mut self, sup: u32, nbits: u32) -> u32 {
        loop {
            self.advance();
            let (lo, hi) = self.lo_hi();
            macro_rules! try_output {
                ($($k:literal)*) => {$(
                    let val = (pair_hash::<$k>(lo, hi) >> (64 - nbits)) as u32;
                    if val < sup {
                        return val;
                    }
                )*};
            }
            try_output!(0 1 2 3 4 5 6 7 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);
        }
    }

    /// Random integer in [0, 2^31).
    pub fn integer(&mut self) -> i32 {
        self.advance();
        let (lo, hi) = self.lo_hi();
        (pair_hash::<0>(lo, hi) >> 33) as i32
    }

    /// Random integer in [0, sup).
    pub fn integer_below(&mut self, sup: i32) -> i32 {
        if sup <= 1 {
            return 0;
        }
        let nbits = log2i((sup - 1) as u32) + 1;
        self.draw_uniform(sup as u32, nbits) as i32
    }

    /// Random integer in [min, sup).
    pub fn integer_between(&mut self, min: i32, sup: i32) -> i32 {
        min + self.integer_below(sup - min)
    }

    /// Random real in (0, 1).
    pub fn fraction(&mut self) -> f64 {
        self.advance();
        let (lo, hi) = self.lo_hi();
        let output = (pair_hash::<0>(lo, hi) >> (64 - 52)) ^ (pair_hash::<1>(lo, hi) >> (64 - 26));
        (1 + output) as f64 * 2.2204460492503125e-16
    }

    /// Random real in [0, 1].
    pub fn closed_fraction(&mut self) -> f64 {
        self.advance();
        let (lo, hi) = self.lo_hi();
        let output = (pair_hash::<0>(lo, hi) >> (64 - 53)) ^ (pair_hash::<1>(lo, hi) >> 32);
        output as f64 * 1.1102230246251566e-16
    }

    /// Random real in [a, b] (clang fuses a + (b - a) * f).
    pub fn real(&mut self, a: f64, b: f64) -> f64 {
        (b - a).mul_add_c(self.closed_fraction(), a)
    }

    pub fn bit(&mut self) -> bool {
        self.advance();
        let (lo, hi) = self.lo_hi();
        pair_hash::<0>(lo, hi) >> 63 != 0
    }

    pub fn shuffle<T>(&mut self, data: &mut [T]) {
        for i in (2..=data.len()).rev() {
            let pos = self.integer_below(i as i32) as usize;
            data.swap(pos, i - 1);
        }
    }
}
