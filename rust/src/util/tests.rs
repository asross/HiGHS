//! Bit-identity with the C++: checksums produced by the same sequence of
//! operations in golden.cpp (clang++ -O3 -std=c++11 -Ihighs -Ibuild) and
//! golden2.cpp (build command in its header).

use super::cdouble::CDouble;
use super::disjoint_sets::HighsDisjointSets;
use super::hash::*;
use super::hash_table::HighsHashTable;
use super::hash_tree::HighsHashTree;
use super::hset::HSet;
use super::random::HighsRandom;
use super::sparse_vector_sum::HighsSparseVectorSum;

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(1469598103934665603)
    }
    fn mix(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(1099511628211);
    }
    fn mixd(&mut self, d: f64) {
        self.mix(d.to_bits());
    }
    fn mixc(&mut self, c: CDouble) {
        self.mixd(c.hi);
        self.mixd(c.lo);
    }
}

#[test]
fn random_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(12345);
    for i in 0..100000 {
        h.mix(r.integer() as u64);
        h.mix(r.integer_below(i % 1000 + 1) as u64);
        h.mix(r.integer_between(-5, i % 77 + 3) as i64 as u64);
        h.mixd(r.fraction());
        h.mixd(r.closed_fraction());
        h.mixd(r.real(-3.5, 1e3));
        h.mix(r.bit() as u64);
    }
    // HighsRandom::real is fused on arm64 only (see util/fma.rs)
    let expected = if cfg!(target_arch = "aarch64") { 16995212409544007823 } else { 6400714799036924485 };
    assert_eq!(h.0, expected);
}

#[test]
fn cdouble_matches_cpp() {
    let mut h = Fnv::new();
    let mut s = HighsRandom::new(7);
    let mut val = || {
        let f = s.fraction() - 0.5;
        f * 2f64.powi(s.integer_below(80) - 40)
    };
    for _ in 0..100000 {
        let (a, b, c) = (val(), val(), val());
        let x = CDouble::from(a) + b;
        let y = CDouble::from(c) * a;
        for v in [x, y, x * y, x * c, x / y, x / b, x - y, x - c, a - y, a + y, a * y, b / x, -x] {
            h.mixc(v);
        }
        let mut z = x;
        z += y;
        z -= c;
        z *= y;
        z /= x;
        z *= b;
        z /= c;
        z -= x;
        h.mixc(z);
        h.mixc(x.abs().sqrt());
        h.mixc((x * 1e3).floor());
        h.mixc((y * 1e3).ceil());
        h.mixc(x.round());
        let mut z = x;
        z.renormalize();
        h.mixc(z);
        h.mix((x < y) as u64);
        h.mix((x == a) as u64);
        h.mix((x >= b) as u64);
    }
    assert_eq!(h.0, 11651025907869496546);
}

fn rand_double(r: &mut HighsRandom, range: i32) -> f64 {
    let f = r.fraction() - 0.5;
    f * 2f64.powi(r.integer_below(2 * range) - range)
}

#[test]
fn hash_helpers_match_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(1);
    let mut buf = [0u8; 1024];
    for _ in 0..2000 {
        for b in buf.iter_mut() {
            *b = r.integer_below(256) as u8;
        }
        for n in 1..=64 {
            h.mix(hash_bytes(&buf[..n]));
        }
        let len = r.integer_below(1025) as usize;
        h.mix(vector_hash(&buf[..len]));

        let a = r.integer();
        let b = r.integer();
        h.mix(a.highs_hash());
        h.mix((a as u16).highs_hash());
        h.mix(((a as u64) << 32 | b as u64).highs_hash());
        h.mix((a, b).highs_hash());
        h.mix((a, b, a as u32).highs_hash());
        let d = rand_double(&mut r, 100);
        h.mix(d.highs_hash());
        h.mix(double_hash_code(d) as u64);
        let n = r.integer_below(100);
        let v: Vec<i32> = (0..n).map(|_| r.integer()).collect();
        h.mix(v.highs_hash());

        let (mut s, mut s32) = (0u64, 0u32);
        for _ in 0..20 {
            let idx = r.integer_below(5000);
            let hi = r.integer() as u64;
            let val = (hi << 33) ^ r.integer() as u64;
            sparse_combine(&mut s, idx, val);
            h.mix(s);
            sparse_combine_index(&mut s, idx);
            h.mix(s);
            sparse_combine32(&mut s32, idx, val);
            h.mix(s32 as u64);
            if r.bit() {
                sparse_inverse_combine(&mut s, idx, val);
                h.mix(s);
                sparse_inverse_combine_index(&mut s, idx);
                h.mix(s);
                sparse_inverse_combine32(&mut s32, idx, val);
                h.mix(s32 as u64);
            }
            h.mix(multiply_mod_m61(val, hi.wrapping_mul(FIBONACCI_MULTIPLIER)));
            h.mix(modexp_m61(val & M61, r.integer_below(1000) as u64 + 1));
        }
    }
    assert_eq!(h.0, 3050368128204157702);
}

#[test]
fn hash_table_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(2);
    let mut t = HighsHashTable::<i32, i32>::new();
    let mut s = HighsHashTable::<(i32, i32, u32)>::new();
    let set_key = |k: i32| (k, k ^ 0x55, (k as u32).wrapping_mul(2654435761));
    for phase in 0..12 {
        let range = [50, 2000, 20000][phase % 3];
        let ins_prob = if phase % 2 == 1 { 70 } else { 30 };
        for _ in 0..15000 {
            let op = r.integer_below(100);
            let k = r.integer_below(range);
            if op < ins_prob {
                let v = r.integer_below(1000000);
                h.mix(t.insert(k, v) as u64);
                h.mix(s.insert(set_key(k), ()) as u64);
            } else if op < ins_prob + 15 {
                h.mix(t.erase(&k) as u64);
                h.mix(s.erase(&set_key(k)) as u64);
            } else if op < ins_prob + 25 {
                *t.get_or_insert_default(k) += 1;
                h.mix(*t.get_or_insert_default(k) as u64);
            } else {
                h.mix(t.find(&k).map_or(-1, |v| *v) as u64);
                h.mix(s.find(&set_key(k)).is_some() as u64);
            }
            h.mix(t.len() as u64);
            h.mix(s.len() as u64);
        }
        for (k, v) in t.iter() {
            h.mix(*k as u64);
            h.mix(*v as u64);
        }
        for (k, _) in s.iter() {
            h.mix(k.2 as u64);
        }
        if phase % 4 == 3 {
            let keys: Vec<i32> = t.iter().map(|(k, _)| *k).collect();
            for k in keys {
                if r.integer_below(10) != 0 {
                    h.mix(t.erase(&k) as u64);
                    h.mix(s.erase(&set_key(k)) as u64);
                }
            }
            for (k, _) in t.iter() {
                h.mix(*k as u64);
            }
            h.mix(t.len() as u64);
        }
        if phase == 6 {
            t.clear();
            s.clear();
        }
    }

    let mut c = HighsHashTable::<i32, i32>::with_capacity(1000);
    for i in 0..900 {
        c.insert(r.integer(), i);
    }
    for (k, _) in c.iter() {
        h.mix(*k as u64);
    }

    let mut vt = HighsHashTable::<Vec<i32>, i32>::new();
    for i in 0..20000 {
        let n = r.integer_between(2, 6);
        let v: Vec<i32> = (0..n).map(|_| r.integer_below(4)).collect();
        h.mix(vt.insert(v.clone(), i) as u64);
        h.mix(*vt.find(&v).unwrap() as u64);
    }
    for (_, v) in vt.iter() {
        h.mix(*v as u64);
    }
    assert_eq!(h.0, 3235967778451609503);
}

#[test]
fn hash_tree_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(3);
    let mut trees: Vec<HighsHashTree<i32, i32>> = (0..8).map(|_| HighsHashTree::new()).collect();
    let mut sets: Vec<HighsHashTree<i32>> = (0..8).map(|_| HighsHashTree::new()).collect();
    for phase in 0..10 {
        let range = [100, 3000, 1000000][phase % 3];
        let ins_prob = if phase % 2 == 1 { 60 } else { 35 };
        for _ in 0..40000 {
            let w = r.integer_below(8) as usize;
            let op = r.integer_below(100);
            let k = r.integer_below(range);
            if op < ins_prob {
                let v = r.integer_below(1000000);
                let (value, inserted) = trees[w].insert_or_get(k, v);
                h.mix(*value as u64);
                h.mix(inserted as u64);
                *value += 1;
                h.mix(sets[w].insert(k, ()) as u64);
            } else if op < ins_prob + 20 {
                trees[w].erase(&k);
                sets[w].erase(&k);
            } else if op < ins_prob + 30 {
                h.mix(trees[w].find(&k).map_or(-1, |v| *v) as u64);
                h.mix(sets[w].contains(&k) as u64);
            } else {
                let w2 = r.integer_below(8) as usize;
                h.mix(trees[w].find_common(&trees[w2]).map_or(-1, |(k, _)| *k) as u64);
                h.mix(sets[w].find_common(&sets[w2]).map_or(-1, |(k, _)| *k) as u64);
            }
        }
        for w in 0..8 {
            trees[w].for_each(|k, v| {
                h.mix(*k as u64);
                h.mix(*v as u64);
            });
            sets[w].for_each(|k, _| h.mix(*k as u64));
            h.mix(trees[w].is_empty() as u64);
            if phase % 4 == 3 {
                let mut keys = Vec::new();
                trees[w].for_each(|k, _| keys.push(*k));
                for k in keys {
                    if r.integer_below(10) != 0 {
                        trees[w].erase(&k);
                        sets[w].erase(&k);
                    }
                }
                sets[w].for_each(|k, _| h.mix(*k as u64));
            }
        }
    }
    assert_eq!(h.0, 15343388911382925190);
}

/// Random keys always burst a leaf into children sized exactly. Force the
/// other bursts: keys with distinct leading 6 hash bits (all children
/// small) and keys with equal leading bits (a single child).
#[test]
fn hash_tree_bursts_match_cpp() {
    let mut h = Fnv::new();
    for variant in 0..2 {
        let mut keys = Vec::new();
        let mut used = 0u64;
        let mut k = 0i32;
        while keys.len() < if variant == 1 { 300 } else { 60 } {
            let chunk = k.highs_hash() >> 58;
            if if variant == 1 { chunk == 0 } else { used >> chunk & 1 == 0 } {
                used |= 1 << chunk;
                keys.push(k);
            }
            k += 1;
        }
        let mut t = HighsHashTree::<i32, i32>::new();
        for &k in &keys {
            h.mix(t.insert(k, k) as u64);
        }
        t.for_each(|k, _| h.mix(*k as u64));
        for k in keys.iter().step_by(2) {
            t.erase(k);
        }
        t.for_each(|k, _| h.mix(*k as u64));
        for &k in &keys {
            h.mix(t.insert(k, k) as u64);
        }
        t.for_each(|k, _| h.mix(*k as u64));
    }
    assert_eq!(h.0, 16555990373533190415);
}

/// Keys whose hashes agree in all the bits the levels use end in list
/// leaves at the maximal depth; no 32-bit key gets there, so this checks
/// the path against a reference set only.
#[test]
fn hash_tree_list_leaves() {
    #[derive(PartialEq, Clone, Copy)]
    struct Colliding(u32);
    impl super::hash::HighsHash for Colliding {
        fn highs_hash(&self) -> u64 {
            (self.0 % 3) as u64 // equal leading 54 bits
        }
    }
    let mut t = HighsHashTree::<Colliding, u32>::new();
    let mut other = HighsHashTree::<Colliding, u32>::new();
    for i in 0..200 {
        assert!(t.insert(Colliding(i), i));
        assert!(!t.insert(Colliding(i), 0));
    }
    other.insert(Colliding(1000), 0);
    assert!(t.find_common(&other).is_none());
    other.insert(Colliding(150), 0);
    assert_eq!(t.find_common(&other).map(|(k, _)| k.0), Some(150));
    for i in (0..200).step_by(2) {
        t.erase(&Colliding(i));
    }
    for i in 0..200 {
        assert_eq!(t.find(&Colliding(i)), (i % 2 == 1).then_some(&i));
    }
    let mut n = 0;
    t.for_each(|k, v| {
        assert_eq!(k.0, *v);
        n += 1;
    });
    assert_eq!(n, 100);
    for i in (1..200).step_by(2) {
        t.erase(&Colliding(i));
    }
    // as in the C++, a branch losing its last child becomes an empty leaf,
    // so the tree is not is_empty()
    let mut n = 0;
    t.for_each(|_, _| n += 1);
    assert_eq!(n, 0);
}

#[test]
fn sparse_vector_sum_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(4);
    let mut s = HighsSparseVectorSum::new(1000);
    for _ in 0..3000 {
        let n = r.integer_between(2, 400);
        for _ in 0..n {
            let idx = r.integer_below(1000);
            let d = rand_double(&mut r, 20);
            match r.integer_below(3) {
                0 => s.add(idx, d),
                1 => s.add(idx, CDouble::from(d) * 3.0 + 1e-17),
                _ => s.add(idx, -s.get_value(idx)),
            }
        }
        match r.integer_below(3) {
            0 => s.cleanup(|_, v| v.abs() < 1e-3),
            1 => h.mix(s.partition(|i| i % 3 == 0) as u64),
            _ => {}
        }
        for &i in s.get_nonzeros() {
            h.mix(i as u64);
            h.mixc(s.values[i as usize]);
        }
        s.clear();
    }
    assert_eq!(h.0, 3746380646038202678);
}

#[test]
fn hset_matches_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(5);
    let mut set = HSet::default();
    set.setup(100, 50);
    for i in 0..200000 {
        let op = r.integer_below(3);
        let e = r.integer_between(-2, 300);
        h.mix(match op {
            0 => set.add(e),
            1 => set.remove(e),
            _ => set.contains(e),
        } as u64);
        h.mix(set.count() as u64);
        if i % 1000 == 999 {
            for &x in set.entries() {
                h.mix(x as u64);
            }
        }
        if i % 50000 == 49999 {
            set.clear();
        }
    }
    assert_eq!(h.0, 17446011844000705433);
}

#[test]
fn disjoint_sets_match_cpp() {
    let mut h = Fnv::new();
    let mut r = HighsRandom::new(6);
    let mut a = HighsDisjointSets::<false>::new(5000);
    let mut b = HighsDisjointSets::<true>::new(5000);
    for _ in 0..20000 {
        let x = r.integer_below(5000);
        let y = r.integer_below(5000);
        a.merge(x, y);
        b.merge(x, y);
        let z = r.integer_below(5000);
        h.mix(a.get_set(z) as u64);
        h.mix(b.get_set(z) as u64);
        let sa = a.get_set(z);
        h.mix(a.get_set_size(sa) as u64);
        let sb = b.get_set(z);
        h.mix(b.get_set_size(sb) as u64);
    }
    assert_eq!(h.0, 17995560140726397141);
}

#[test]
fn no_raw_mul_add() {
    // Mirrored FMAs must go through ClangFma::mul_add_c (util/fma.rs): a raw
    // mul_add would be a slow software fma() on x86_64 and change the paths.
    fn walk(dir: &std::path::Path, bad: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, bad);
            } else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("util/fma.rs") && !p.ends_with("util/tests.rs") {
                let s = std::fs::read_to_string(&p).unwrap();
                if s.contains(".mul_add(") {
                    bad.push(p.display().to_string());
                }
            }
        }
    }
    let mut bad = Vec::new();
    walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut bad);
    assert!(bad.is_empty(), "raw mul_add in {bad:?}");
}
