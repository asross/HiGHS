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
    (a as u64).wrapping_add(C[2 * K]).wrapping_mul((b as u64).wrapping_add(C[2 * K + 1]))
}

#[inline]
pub fn log2i(n: u32) -> u32 {
    31 - n.leading_zeros()
}
