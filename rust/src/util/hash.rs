//! Hash helpers of HighsHash.h.

/// Random constants of HighsHashHelpers.
pub const C: [u64; 64] = [
    0xc8497d2a400d9551, 0x80c8963be3e4c2f3, 0x042d8680e260ae5b,
    0x8a183895eeac1536, 0xa94e9c75f80ad6de, 0x7e92251dec62835e,
    0x07294165cb671455, 0x89b0f6212b0a4292, 0x31900011b96bf554,
    0xa44540f8eee2094f, 0xce7ffd372e4c64fc, 0x51c9d471bfe6a10f,
    0x758c2a674483826f, 0xf91a20abe63f8b02, 0xc2a069024a1fcc6f,
    0xd5bb18b70c5dbd59, 0xd510adac6d1ae289, 0x571d069b23050a79,
    0x60873b8872933e06, 0x780481cc19670350, 0x7a48551760216885,
    0xb5d68b918231e6ca, 0xa7e5571699aa5274, 0x7b6d309b2cfdcf01,
    0x04e77c3d474daeff, 0x4dbf099fd7247031, 0x5d70dca901130beb,
    0x9f8b5f0df4182499, 0x293a74c9686092da, 0xd09bdab6840f52b3,
    0xc05d47f3ab302263, 0x6b79e62b884b65d6, 0xa581106fc980c34d,
    0xf081b7145ea2293e, 0xfb27243dd7c3f5ad, 0x5211bf8860ea667f,
    0x9455e65cb2385e7f, 0x0dfaf6731b449b33, 0x4ec98b3c6f5e68c7,
    0x007bfd4a42ae936b, 0x65c93061f8674518, 0x640816f17127c5d1,
    0x6dd4bab17b7c3a74, 0x34d9268c256fa1ba, 0x0b4d0c6b5b50d7f4,
    0x30aa965bc9fadaff, 0xc0ac1d0c2771404d, 0xc5e64509abb76ef2,
    0xd606b11990624a36, 0x0d3f05d242ce2fb7, 0x469a803cb276fe32,
    0xa4a44d177a3e23f4, 0xb9d9a120dcc1ca03, 0x2e15af8165234a2e,
    0x10609ba2720573d4, 0xaa4191b60368d1d5, 0x333dd2300bc57762,
    0xdf6ec48f79fb402f, 0x5ed20fcef1b734fa, 0x4c94924ec8be21ee,
    0x5abe6ad9d131e631, 0xbe10136a522e602d, 0x53671115c340e779,
    0x9f392fe43e2144da,
];

/// Strongly universal hash of a pair of 32-bit values, with constants k.
#[inline]
pub fn pair_hash<const K: usize>(a: u32, b: u32) -> u64 {
    pair_hash_k(K, a, b)
}

#[inline]
fn pair_hash_k(k: usize, a: u32, b: u32) -> u64 {
    (a as u64).wrapping_add(C[2 * k]).wrapping_mul((b as u64).wrapping_add(C[2 * k + 1]))
}

#[inline]
pub fn log2i(n: u32) -> u32 {
    31 - n.leading_zeros()
}

#[inline]
pub fn log2i64(n: u64) -> u32 {
    63 - n.leading_zeros()
}

/// Mersenne prime 2^61 - 1.
pub const M61: u64 = 0x1fffffffffffffff;
/// Mersenne prime 2^31 - 1.
pub const M31: u64 = 0x7fffffff;
pub const FIBONACCI_MULTIPLIER: u64 = 0x9e3779b97f4a7c15;

/// a * b mod 2^61 - 1.
pub fn multiply_mod_m61(a: u64, b: u64) -> u64 {
    let (ahi, bhi) = (a >> 32, b >> 32);
    let (alo, blo) = (a & 0xffffffff, b & 0xffffffff);

    // the different order terms with adicities 2^64, 2^32, 2^0
    let term_64 = ahi.wrapping_mul(bhi);
    let term_32 = ahi.wrapping_mul(blo).wrapping_add(bhi.wrapping_mul(alo));
    let mut term_0 = alo.wrapping_mul(blo);

    // Partially reduce term_0 and term_32 modulo M61 individually so that no
    // carry bit is lost; the final reduction happens at the end.
    term_0 = (term_0 & M61).wrapping_add(term_0 >> 61);
    term_0 = term_0.wrapping_add(((term_32 >> 29).wrapping_add(term_32 << 32)) & M61);

    // The lower 61 bits of term_0 are the lower 61 bits of the result; the
    // upper ones fold back since q * 2^61 + r = q + r (mod 2^61 - 1).
    let ab61 = (term_64 << 3) | (term_0 >> 61);
    // unsigned wrap-around as in the C++
    let mut result = (term_0 & M61).wrapping_add(ab61);
    if result >= M61 {
        result -= M61;
    }
    result
}

/// a^e mod 2^61 - 1, e > 0.
pub fn modexp_m61(a: u64, mut e: u64) -> u64 {
    debug_assert!(e > 0);
    let mut result = a;
    while e != 1 {
        result = multiply_mod_m61(result, result);
        if e & 1 != 0 {
            result = multiply_mod_m61(result, a);
        }
        e >>= 1;
    }
    result
}

/// a * b mod 2^31 - 1.
pub fn multiply_mod_m31(a: u32, b: u32) -> u32 {
    let mut result = a as u64 * b as u64;
    result = (result >> 31) + (result & M31);
    if result >= M31 {
        result -= M31;
    }
    result as u32
}

/// a^e mod 2^31 - 1, e > 0.
pub fn modexp_m31(a: u32, mut e: u64) -> u32 {
    debug_assert!(e > 0);
    let mut result = a;
    while e != 1 {
        result = multiply_mod_m31(result, result);
        if e & 1 != 0 {
            result = multiply_mod_m31(result, a);
        }
        e >>= 1;
    }
    result
}

/// c[index % 64]^(index / 64 + 1) mod M61: the monomial of a sparse entry.
fn monomial_m61(index: i32) -> u64 {
    modexp_m61(C[(index & 63) as usize] & M61, (index as i64 as u64 >> 6) + 1)
}

fn add_mod_m61(hash: &mut u64, term: u64) {
    *hash += term;
    *hash = (*hash >> 61) + (*hash & M61);
    if *hash >= M61 {
        *hash -= M61;
    }
}

/// Each value of a sparse vector is the coefficient of a polynomial over the
/// field modulo 2^61 - 1 whose monomial for an entry has the degree of its
/// index (spread over 64 variables, one per random constant), evaluated at
/// the random constants. Entries contribute independently, so the hash can
/// be computed in any order and updated (see sparse_inverse_combine).
pub fn sparse_combine(hash: &mut u64, index: i32, value: u64) {
    // make sure the value is never zero and uses at most 61 bits
    let value = ((value << 1) & M61) | 1;
    add_mod_m61(hash, multiply_mod_m61(value, monomial_m61(index)));
}

/// Undoes sparse_combine by adding the additive inverse.
pub fn sparse_inverse_combine(hash: &mut u64, index: i32, value: u64) {
    let value = ((value << 1) & M61) | 1;
    add_mod_m61(hash, M61 - multiply_mod_m61(value, monomial_m61(index)));
}

/// sparse_combine without a value, for sparse bit vectors.
pub fn sparse_combine_index(hash: &mut u64, index: i32) {
    add_mod_m61(hash, monomial_m61(index));
}

pub fn sparse_inverse_combine_index(hash: &mut u64, index: i32) {
    add_mod_m61(hash, M61 - monomial_m61(index));
}

fn sparse_term_m31(index: i32, value: u64) -> u64 {
    // make sure the value is never zero and uses at most 31 bits
    let value = (pair_hash::<0>(value as u32, (value >> 32) as u32) >> 33) | 1;
    let a = (C[(index & 63) as usize] & M31) as u32;
    multiply_mod_m31(value as u32, modexp_m31(a, (index as i64 as u64 >> 6) + 1)) as u64
}

fn add_mod_m31(hash: &mut u32, term: u64) {
    let mut result = *hash as u64 + term;
    result = (result >> 31) + (result & M31);
    if result >= M31 {
        result -= M31;
    }
    *hash = result as u32;
}

/// sparse_combine modulo 2^31 - 1.
pub fn sparse_combine32(hash: &mut u32, index: i32, value: u64) {
    add_mod_m31(hash, sparse_term_m31(index, value));
}

pub fn sparse_inverse_combine32(hash: &mut u32, index: i32, value: u64) {
    add_mod_m31(hash, M31 - sparse_term_m31(index, value));
}

fn split_pair(p: &[u8]) -> (u32, u32) {
    (
        u32::from_ne_bytes([p[0], p[1], p[2], p[3]]),
        u32::from_ne_bytes([p[4], p[5], p[6], p[7]]),
    )
}

/// HighsHashHelpers::vector_hash over the bytes of the elements.
pub fn vector_hash(data: &[u8]) -> u64 {
    // as in the C++, a short last pair keeps the trailing bytes of the
    // previous pair
    let mut pair = [0u8; 8];
    let mut hash = 0u64;
    let mut k = 0;
    for chunk in data.chunks(256) {
        let num_pairs = chunk.len().div_ceil(8);
        let mut chunkhash = [0u64; 2];
        // reduce mod M61 before multiplying with the next random constant;
        // only vectors longer than 256 bytes get here
        if num_pairs == 32 && hash != 0 {
            if hash >= M61 {
                hash -= M61;
            }
            hash = multiply_mod_m61(hash, C[k & 63] & M61);
            k += 1;
        }
        for (i, bytes) in chunk.chunks(8).enumerate() {
            let n = num_pairs - i;
            pair[..bytes.len()].copy_from_slice(bytes);
            let (a, b) = split_pair(&pair);
            chunkhash[n & 1] = chunkhash[n & 1].wrapping_add(pair_hash_k(32 - n, a, b));
        }
        hash = hash.wrapping_add((chunkhash[0] >> 3) ^ (chunkhash[1] >> 32));
    }
    hash.wrapping_mul(FIBONACCI_MULTIPLIER)
}

/// HighsHashHelpers::hash of a trivially copyable value of 1 to 64 bytes:
/// its bytes, zero padded to whole 8-byte pairs, go through pair_hash with
/// the constants of each pair's position.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let n = bytes.len();
    assert!((1..=64).contains(&n));
    let mut buf = [0u8; 64];
    buf[..n].copy_from_slice(bytes);
    let pair = |k: usize| {
        let (a, b) = split_pair(&buf[8 * k..]);
        pair_hash_k(k, a, b)
    };
    let num_pairs = n.div_ceil(8);
    if num_pairs == 1 {
        let (a, b) = split_pair(&buf);
        return pair_hash_k(1, a, b) ^ (pair_hash_k(0, a, b) >> 32);
    }
    let sum = |r: std::ops::Range<usize>| r.fold(0u64, |s, k| s.wrapping_add(pair(k)));
    let half = num_pairs / 2;
    (sum(0..half) ^ (sum(half..num_pairs) >> 32)).wrapping_mul(FIBONACCI_MULTIPLIER)
}

/// A trivially copyable value with the byte layout of its C++ counterpart
/// (tuples: libc++ layout, members in order; padding-free types only).
pub trait Pod: Copy {
    const SIZE: usize;
    fn write(&self, out: &mut [u8]);
}

macro_rules! pod_primitive {
    ($($t:ty)*) => {$(
        impl Pod for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn write(&self, out: &mut [u8]) {
                out[..Self::SIZE].copy_from_slice(&self.to_ne_bytes());
            }
        }
    )*};
}
pod_primitive!(i8 u8 i16 u16 i32 u32 i64 u64 f32 f64);

impl<A: Pod, B: Pod> Pod for (A, B) {
    const SIZE: usize = A::SIZE + B::SIZE;
    fn write(&self, out: &mut [u8]) {
        self.0.write(out);
        self.1.write(&mut out[A::SIZE..]);
    }
}

impl<A: Pod, B: Pod, D: Pod> Pod for (A, B, D) {
    const SIZE: usize = A::SIZE + B::SIZE + D::SIZE;
    fn write(&self, out: &mut [u8]) {
        self.0.write(out);
        self.1.write(&mut out[A::SIZE..]);
        self.2.write(&mut out[A::SIZE + B::SIZE..]);
    }
}

impl<T: Pod, const N: usize> Pod for [T; N] {
    const SIZE: usize = T::SIZE * N;
    fn write(&self, out: &mut [u8]) {
        for (i, x) in self.iter().enumerate() {
            x.write(&mut out[i * T::SIZE..]);
        }
    }
}

/// Keys of HighsHashTable and HighsHashTree: HighsHashHelpers::hash, with
/// equality by `==` as in the C++.
pub trait HighsHash: PartialEq {
    fn highs_hash(&self) -> u64;
}

impl<T: Pod + PartialEq> HighsHash for T {
    fn highs_hash(&self) -> u64 {
        let mut buf = [0u8; 64];
        self.write(&mut buf);
        hash_bytes(&buf[..T::SIZE])
    }
}

impl<T: Pod + PartialEq> HighsHash for Vec<T> {
    fn highs_hash(&self) -> u64 {
        // ponytail: copies the bytes; hash in place if it shows in profiles
        let mut bytes = vec![0u8; self.len() * T::SIZE];
        for (i, x) in self.iter().enumerate() {
            x.write(&mut bytes[i * T::SIZE..]);
        }
        vector_hash(&bytes)
    }
}

/// Hash code of a double, equal for values that agree in the leading 15
/// bits of the mantissa. Multiplying by the reciprocal of the golden ratio
/// keeps bucket borders off powers of two: 0.5 and 0.5 - 1e-9 differ in the
/// exponent, but not after the multiplication.
#[allow(clippy::excessive_precision)] // the C++ literal
pub fn double_hash_code(val: f64) -> u32 {
    let (hashbits, exponent) = frexp(val * 0.61803398874989484);
    // upper 16 bits: the exponent; lower 16 bits: sign and leading 15 bits
    // of the mantissa
    let hashvalue = exponent as i16 as u16 as u32;
    (hashvalue << 16) | (hashbits * 32768.0) as i16 as u16 as u32
}

/// C frexp: x = m * 2^e with 0.5 <= |m| < 1.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        let (m, e) = frexp(x * 2f64.powi(54));
        return (m, e - 54);
    }
    (f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52)), e - 1022)
}
