//! HighsDisjointSets: union-find with path compression. Merging links the
//! smaller set below the larger one, or, with MINIMAL_REPRESENTATIVE, the
//! set with the larger representative below the other.

#[derive(Clone, Default)]
pub struct HighsDisjointSets<const MINIMAL_REPRESENTATIVE: bool = false> {
    sizes: Vec<i32>,
    sets: Vec<i32>,
    link_compression_stack: Vec<i32>,
}

impl<const MINIMAL_REPRESENTATIVE: bool> HighsDisjointSets<MINIMAL_REPRESENTATIVE> {
    pub fn new(num_items: usize) -> Self {
        let mut s = Self::default();
        s.reset(num_items);
        s
    }

    pub fn reset(&mut self, num_items: usize) {
        self.sizes = vec![1; num_items];
        self.sets = (0..num_items as i32).collect();
    }

    pub fn get_set(&mut self, mut item: i32) -> i32 {
        let sets = &mut self.sets;
        let mut repr = sets[item as usize];
        if repr != sets[repr as usize] {
            loop {
                self.link_compression_stack.push(item);
                item = repr;
                repr = sets[repr as usize];
                if repr == sets[repr as usize] {
                    break;
                }
            }
            while let Some(i) = self.link_compression_stack.pop() {
                sets[i as usize] = repr;
            }
            sets[item as usize] = repr;
        }
        repr
    }

    pub fn get_set_size(&self, set: i32) -> i32 {
        debug_assert_eq!(self.sets[set as usize], set);
        self.sizes[set as usize]
    }

    pub fn merge(&mut self, item1: i32, item2: i32) {
        let repr1 = self.get_set(item1);
        let repr2 = self.get_set(item2);
        if repr1 == repr2 {
            return;
        }
        let keep_repr1 = if MINIMAL_REPRESENTATIVE {
            repr2 > repr1
        } else {
            self.sizes[repr1 as usize] > self.sizes[repr2 as usize]
        };
        let (root, child) = if keep_repr1 { (repr1, repr2) } else { (repr2, repr1) };
        self.sets[child as usize] = root;
        self.sizes[root as usize] += self.sizes[child as usize];
    }
}
