//! HighsHashTable of HighsHash.h: Robin Hood hashing with a metadata byte
//! per slot. Same probing, growth and shrinking as the C++, so the slot of
//! every entry, and therefore the iteration order, is identical. A set is
//! `HighsHashTable<K>` (value `()`).

use super::hash::{log2i64, HighsHash};

/// An item never travels further than this from its ideal slot.
const MAX_DISTANCE: u64 = 127;

#[derive(Clone)]
pub struct HighsHashTable<K, V = ()> {
    /// Slots; only those with an occupied metadata byte hold an entry.
    entries: Vec<(K, V)>,
    /// 0 for an empty slot, else 0x80 | the low 7 bits of the ideal slot.
    metadata: Vec<u8>,
    table_size_mask: u64,
    num_hash_shift: u32,
    num_elements: usize,
}

/// Probe state of find_position.
struct Probe {
    meta: u8,
    start_pos: u64,
    max_pos: u64,
    pos: u64,
}

fn occupied(meta: u8) -> bool {
    meta & 0x80 != 0
}

impl<K: HighsHash + Default, V: Default> Default for HighsHashTable<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: HighsHash + Default, V: Default> HighsHashTable<K, V> {
    pub fn new() -> Self {
        let mut t = HighsHashTable {
            entries: Vec::new(),
            metadata: Vec::new(),
            table_size_mask: 0,
            num_hash_shift: 0,
            num_elements: 0,
        };
        t.make_empty_table(128);
        t
    }

    pub fn with_capacity(min_capacity: usize) -> Self {
        let mut t = Self::new();
        let capacity = 128f64.max(8.0 * min_capacity as f64 / 7.0).log2().ceil();
        t.make_empty_table(1 << capacity as u64);
        t
    }

    fn make_empty_table(&mut self, capacity: u64) {
        self.table_size_mask = capacity - 1;
        self.num_hash_shift = 64 - log2i64(capacity);
        self.num_elements = 0;
        self.metadata = vec![0; capacity as usize];
        self.entries = (0..capacity).map(|_| Default::default()).collect();
    }

    fn to_metadata(&self, hash: u64) -> u8 {
        (hash >> self.num_hash_shift) as u8 | 0x80
    }

    /// The metadata holds 7 bits of the ideal slot. Assuming an item never
    /// travels a full cycle of 128 slots, the distance is the difference of
    /// the low 7 bits, ignoring the overflow.
    fn distance_from_ideal_slot(&self, pos: u64) -> u64 {
        pos.wrapping_sub(self.metadata[pos as usize] as u64) & 0x7f
    }

    fn rebuild(&mut self, capacity: u64) {
        let old_entries = std::mem::take(&mut self.entries);
        let old_metadata = std::mem::take(&mut self.metadata);
        self.make_empty_table(capacity);
        for (meta, entry) in old_metadata.into_iter().zip(old_entries) {
            if occupied(meta) {
                self.insert_entry(entry);
            }
        }
    }

    fn grow_table(&mut self) {
        self.rebuild(2 * (self.table_size_mask + 1));
    }

    fn find_position(&self, key: &K) -> (bool, Probe) {
        let hash = key.highs_hash();
        let start_pos = hash >> self.num_hash_shift;
        let mut p = Probe {
            meta: self.to_metadata(hash),
            start_pos,
            max_pos: (start_pos + MAX_DISTANCE) & self.table_size_mask,
            pos: start_pos,
        };
        loop {
            let meta = self.metadata[p.pos as usize];
            if !occupied(meta) {
                return (false, p);
            }
            if meta == p.meta && *key == self.entries[p.pos as usize].0 {
                return (true, p);
            }
            let current_distance = p.pos.wrapping_sub(p.start_pos) & self.table_size_mask;
            if current_distance > self.distance_from_ideal_slot(p.pos) {
                return (false, p);
            }
            p.pos = (p.pos + 1) & self.table_size_mask;
            if p.pos == p.max_pos {
                return (false, p);
            }
        }
    }

    fn is_full(&self) -> bool {
        self.num_elements as u64 == (self.table_size_mask + 1) * 7 / 8
    }

    /// Robin Hood placement from the probe's slot on: an entry further from
    /// its ideal slot than the current occupant steals the slot. Returns the
    /// entry left over when the probe runs out of range.
    fn place(&mut self, mut entry: (K, V), mut p: Probe) -> Option<(K, V)> {
        let mask = self.table_size_mask;
        loop {
            let pos = p.pos as usize;
            if !occupied(self.metadata[pos]) {
                self.metadata[pos] = p.meta;
                self.entries[pos] = entry;
                return None;
            }
            let current_distance = p.pos.wrapping_sub(p.start_pos) & mask;
            let distance_of_occupant = self.distance_from_ideal_slot(p.pos);
            if current_distance > distance_of_occupant {
                std::mem::swap(&mut entry, &mut self.entries[pos]);
                std::mem::swap(&mut p.meta, &mut self.metadata[pos]);
                p.start_pos = p.pos.wrapping_sub(distance_of_occupant) & mask;
                p.max_pos = (p.start_pos + MAX_DISTANCE) & mask;
            }
            p.pos = (p.pos + 1) & mask;
            if p.pos == p.max_pos {
                return Some(entry);
            }
        }
    }

    fn insert_entry(&mut self, entry: (K, V)) -> bool {
        let (found, p) = self.find_position(&entry.0);
        if found {
            return false;
        }
        if self.is_full() || p.pos == p.max_pos {
            self.grow_table();
            return self.insert_entry(entry);
        }
        self.num_elements += 1;
        if let Some(entry) = self.place(entry, p) {
            self.grow_table();
            self.insert_entry(entry);
        }
        true
    }

    /// Inserts if the key is new; returns whether it was.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        self.insert_entry((key, value))
    }

    /// operator[]: the value of the key, inserted as default if new.
    pub fn get_or_insert_default(&mut self, key: K) -> &mut V
    where
        K: Clone,
    {
        let (found, p) = self.find_position(&key);
        if found {
            return &mut self.entries[p.pos as usize].1;
        }
        if self.is_full() || p.pos == p.max_pos {
            self.grow_table();
            return self.get_or_insert_default(key);
        }
        let insert_location = p.pos as usize;
        self.num_elements += 1;
        match self.place((key.clone(), V::default()), p) {
            None => &mut self.entries[insert_location].1,
            Some(entry) => {
                self.grow_table();
                self.insert_entry(entry);
                self.get_or_insert_default(key)
            }
        }
    }

    pub fn find(&self, key: &K) -> Option<&V> {
        match self.find_position(key) {
            (true, p) => Some(&self.entries[p.pos as usize].1),
            _ => None,
        }
    }

    pub fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.find_position(key) {
            (true, p) => Some(&mut self.entries[p.pos as usize].1),
            _ => None,
        }
    }

    pub fn erase(&mut self, key: &K) -> bool {
        let (found, p) = self.find_position(key);
        if !found {
            return false;
        }
        let mut pos = p.pos as usize;
        self.entries[pos] = Default::default();
        self.metadata[pos] = 0;

        // keep at least a quarter of the slots occupied, otherwise shrink the
        // table unless it is at its minimum size
        self.num_elements -= 1;
        let capacity = self.table_size_mask + 1;
        if capacity != 128 && (self.num_elements as u64) < capacity / 4 {
            self.rebuild(capacity / 2);
            return true;
        }

        // shift the following elements backwards
        loop {
            let shift = (pos + 1) & self.table_size_mask as usize;
            if !occupied(self.metadata[shift]) || self.distance_from_ideal_slot(shift as u64) == 0 {
                return true;
            }
            self.entries.swap(pos, shift);
            self.metadata[pos] = self.metadata[shift];
            self.metadata[shift] = 0;
            pos = shift;
        }
    }

    pub fn clear(&mut self) {
        if self.num_elements != 0 {
            if self.table_size_mask + 1 == 128 {
                self.metadata.fill(0);
                self.num_elements = 0;
            } else {
                self.make_empty_table(128);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.num_elements
    }

    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Entries in slot order, as the C++ iterators.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.metadata
            .iter()
            .zip(&self.entries)
            .filter(|(m, _)| occupied(**m))
            .map(|(_, (k, v))| (k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.metadata
            .iter()
            .zip(&mut self.entries)
            .filter(|(m, _)| occupied(**m))
            .map(|(_, (k, v))| (&*k, v))
    }
}
