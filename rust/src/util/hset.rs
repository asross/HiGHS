//! HSet: a set of nonnegative integers with O(1) add, remove and membership
//! and its entries in a list (removal moves the last entry into the gap).
//! The debug consistency check and printing of the C++ are not ported.

const NO_POINTER: i32 = -1;

#[derive(Clone, Default)]
pub struct HSet {
    count: usize,
    entry: Vec<i32>,
    setup: bool,
    max_entry: i32,
    pointer: Vec<i32>,
}

impl HSet {
    /// Room for size entries up to max_entry; false for invalid arguments.
    pub fn setup(&mut self, size: i32, max_entry: i32) -> bool {
        self.setup = false;
        if size <= 0 || max_entry < 0 {
            return false;
        }
        self.max_entry = max_entry;
        self.entry.resize(size as usize, 0);
        self.pointer = vec![NO_POINTER; max_entry as usize + 1];
        self.count = 0;
        self.setup = true;
        true
    }

    pub fn clear(&mut self) {
        if !self.setup {
            self.setup(1, 0);
        }
        self.pointer.fill(NO_POINTER);
        self.count = 0;
    }

    pub fn add(&mut self, entry: i32) -> bool {
        if entry < 0 {
            return false;
        }
        if !self.setup {
            self.setup(1, entry);
        }
        if entry > self.max_entry {
            self.pointer.resize(entry as usize + 1, NO_POINTER);
            self.max_entry = entry;
        } else if self.pointer[entry as usize] > NO_POINTER {
            return false;
        }
        if self.count == self.entry.len() {
            self.entry.push(0);
        }
        self.pointer[entry as usize] = self.count as i32;
        self.entry[self.count] = entry;
        self.count += 1;
        true
    }

    pub fn remove(&mut self, entry: i32) -> bool {
        if !self.setup {
            self.setup(1, 0);
            return false;
        }
        if !self.contains(entry) {
            return false;
        }
        let pointer = self.pointer[entry as usize];
        self.pointer[entry as usize] = NO_POINTER;
        if (pointer as usize) < self.count - 1 {
            let last_entry = self.entry[self.count - 1];
            self.entry[pointer as usize] = last_entry;
            self.pointer[last_entry as usize] = pointer;
        }
        self.count -= 1;
        true
    }

    /// HSet::in
    pub fn contains(&self, entry: i32) -> bool {
        entry >= 0 && entry <= self.max_entry && self.pointer[entry as usize] != NO_POINTER
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// The entries, in list order.
    pub fn entries(&self) -> &[i32] {
        &self.entry[..self.count]
    }
}
