//! HighsHashTree of HighsHashTree.h: a hash array mapped trie. Each level
//! uses 6 bits of the hash; leaves hold up to 54 entries ordered by 16 bits
//! of the hash, in four size classes; at depth 9 colliding entries go into
//! list leaves. Node splits, merges and entry orders are as in the C++, so
//! for_each and find_common visit entries in the same order. A set is
//! `HighsHashTree<K>` (value `()`).

use super::hash::{log2i64, HighsHash};

const BITS_PER_LEVEL: u32 = 6;
// Up to depth 9 get_hash_chunks16 shifts right by a non-negative amount.
const MAX_DEPTH: u32 = 9;
const MIN_LEAF_SIZE: usize = 6;
const LEAF_BURST_THRESHOLD: usize = 54;
const SIZE_CLASS_STEP: usize = (LEAF_BURST_THRESHOLD - MIN_LEAF_SIZE) / 3;

fn capacity(size_class: u8) -> usize {
    MIN_LEAF_SIZE + (size_class as usize - 1) * SIZE_CLASS_STEP
}

fn entries_to_size_class(num_entries: usize) -> u8 {
    (1 + (num_entries + SIZE_CLASS_STEP - MIN_LEAF_SIZE - 1) / SIZE_CLASS_STEP) as u8
}

fn get_hash_chunk(hash: u64, pos: u32) -> u32 {
    ((hash >> (64 - BITS_PER_LEVEL - pos * BITS_PER_LEVEL)) & 63) as u32
}

fn get_hash_chunks16(hash: u64, pos: u32) -> u16 {
    (hash >> (48 - pos * BITS_PER_LEVEL)) as u16
}

fn first_chunk16(chunks: u16) -> u32 {
    (chunks >> (16 - BITS_PER_LEVEL)) as u32
}

/// Number of occupied chunks at or above pos.
fn num_set_until(occupation: u64, pos: u32) -> usize {
    (occupation >> pos).count_ones() as usize
}

fn test(occupation: u64, pos: u32) -> bool {
    occupation >> pos & 1 != 0
}

fn compute_hash<K: HighsHash>(key: &K) -> u64 {
    key.highs_hash()
}

/// Entries ordered by descending 16-bit hash chunks; the occupation flags
/// of the leading 6 bits locate a chunk's run, and collisions are scanned
/// like linear probing.
#[derive(Clone)]
struct InnerLeaf<K, V> {
    size_class: u8,
    occupation: u64,
    hashes: Vec<u16>,
    entries: Vec<(K, V)>,
}

impl<K: HighsHash, V> InnerLeaf<K, V> {
    fn new(size_class: u8) -> Self {
        InnerLeaf { size_class, occupation: 0, hashes: Vec::new(), entries: Vec::new() }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// The C++ keeps a 0 sentinel after the last hash.
    fn hash_at(&self, i: usize) -> u16 {
        self.hashes.get(i).copied().unwrap_or(0)
    }

    fn find_key(&self, key: &K, hash: u16, pos: &mut usize) -> bool {
        while *pos != self.len() && self.hashes[*pos] == hash {
            if *key == self.entries[*pos].0 {
                return true;
            }
            *pos += 1;
        }
        false
    }

    /// Returns the position of the key and whether it was inserted.
    fn insert_entry(&mut self, full_hash: u64, hash_pos: u32, entry: (K, V)) -> (usize, bool) {
        debug_assert!(self.len() < capacity(self.size_class));
        let hash = get_hash_chunks16(full_hash, hash_pos);
        let chunk = first_chunk16(hash);
        let mut pos = num_set_until(self.occupation, chunk);
        if test(self.occupation, chunk) {
            // the chunk exists, so its run starts at pos - 1 or later
            pos -= 1;
            while self.hash_at(pos) > hash {
                pos += 1;
            }
            if self.find_key(&entry.0, hash, &mut pos) {
                return (pos, false);
            }
        } else {
            self.occupation |= 1 << chunk;
            if pos < self.len() {
                while self.hash_at(pos) > hash {
                    pos += 1;
                }
            }
        }
        self.hashes.insert(pos, hash);
        self.entries.insert(pos, entry);
        (pos, true)
    }

    fn find_entry(&self, full_hash: u64, hash_pos: u32, key: &K) -> Option<usize> {
        let hash = get_hash_chunks16(full_hash, hash_pos);
        let chunk = first_chunk16(hash);
        if !test(self.occupation, chunk) {
            return None;
        }
        let mut pos = num_set_until(self.occupation, chunk) - 1;
        while self.hash_at(pos) > hash {
            pos += 1;
        }
        self.find_key(key, hash, &mut pos).then_some(pos)
    }

    fn erase_entry(&mut self, full_hash: u64, hash_pos: u32, key: &K) -> bool {
        let hash = get_hash_chunks16(full_hash, hash_pos);
        let chunk = first_chunk16(hash);
        if !test(self.occupation, chunk) {
            return false;
        }
        let mut start_pos = num_set_until(self.occupation, chunk) - 1;
        while first_chunk16(self.hashes[start_pos]) > chunk {
            start_pos += 1;
        }
        let mut pos = start_pos;
        while self.hash_at(pos) > hash {
            pos += 1;
        }
        if !self.find_key(key, hash, &mut pos) {
            return false;
        }
        self.hashes.remove(pos);
        self.entries.remove(pos);
        // clear the flag when the chunk's run is gone
        if pos < self.len() {
            if first_chunk16(self.hashes[start_pos]) != chunk {
                self.occupation ^= 1 << chunk;
            }
        } else if start_pos == pos {
            self.occupation ^= 1 << chunk;
        }
        true
    }

    /// Reorders the entries by the hash chunks of the next level. They are
    /// most likely in order already, since 10 of the 16 bits stay the same;
    /// the exact order matters for finding them.
    fn rehash(&mut self, hash_pos: u32) {
        self.occupation = 0;
        for i in 0..self.len() {
            self.hashes[i] = get_hash_chunks16(compute_hash(&self.entries[i].0), hash_pos);
            self.occupation |= 1 << first_chunk16(self.hashes[i]);
        }
        let mut i = 0;
        while i < self.len() {
            let mut pos = num_set_until(self.occupation, first_chunk16(self.hashes[i])) - 1;
            // an element belonging after i is swapped there, and i is
            // examined again
            if pos > i {
                self.hashes.swap(pos, i);
                self.entries.swap(pos, i);
                continue;
            }
            // insertion sort, starting at the position the flags suggest
            while pos < i && self.hashes[pos] >= self.hashes[i] {
                pos += 1;
            }
            if pos < i {
                self.hashes[pos..=i].rotate_right(1);
                self.entries[pos..=i].rotate_right(1);
            }
            i += 1;
        }
    }
}

#[derive(Clone)]
struct Branch<K, V> {
    occupation: u64,
    /// One per occupied chunk, by descending chunk.
    children: Vec<Node<K, V>>,
}

#[derive(Clone)]
enum Node<K, V> {
    Empty,
    /// Leaf at the maximal depth, in list order.
    List(Vec<(K, V)>),
    Leaf(Box<InnerLeaf<K, V>>),
    Branch(Branch<K, V>),
}

impl<K: HighsHash, V> Node<K, V> {
    /// The C++ node type tag, ordered Empty < List < leaf classes < Branch.
    fn rank(&self) -> u8 {
        match self {
            Node::Empty => 0,
            Node::List(_) => 1,
            Node::Leaf(l) => 1 + l.size_class,
            Node::Branch(_) => 6,
        }
    }

    /// Estimate from the node type alone; branches count as large so that
    /// a parent never merges them.
    fn num_entries_estimate(&self) -> usize {
        match self {
            Node::Empty => 0,
            Node::List(_) => 1,
            Node::Leaf(l) => capacity(l.size_class),
            Node::Branch(_) => 64,
        }
    }

    fn num_entries(&self) -> usize {
        match self {
            Node::Empty => 0,
            Node::List(l) => l.len(),
            Node::Leaf(l) => l.len(),
            Node::Branch(_) => 64,
        }
    }
}

fn leaf_ref<K, V>(node: &mut Node<K, V>, i: usize) -> &mut V {
    match node {
        Node::Leaf(l) => &mut l.entries[i].1,
        Node::List(l) => &mut l[i].1,
        _ => unreachable!(),
    }
}

fn insert_recurse<K: HighsHash, V>(
    node: &mut Node<K, V>,
    hash: u64,
    hash_pos: u32,
    entry: (K, V),
) -> (&mut V, bool) {
    if let Node::Leaf(l) = &*node {
        if l.size_class == 4
            && l.len() == capacity(4)
            && l.find_entry(hash, hash_pos, &entry.0).is_none()
        {
            return burst_leaf(node, hash, hash_pos, entry);
        }
    }
    match node {
        Node::Empty => {
            if hash_pos == MAX_DEPTH {
                *node = Node::List(vec![entry]);
                (leaf_ref(node, 0), true)
            } else {
                let mut leaf = InnerLeaf::new(1);
                let (i, inserted) = leaf.insert_entry(hash, hash_pos, entry);
                *node = Node::Leaf(Box::new(leaf));
                (leaf_ref(node, i), inserted)
            }
        }
        Node::List(list) => match list.iter().position(|e| e.0 == entry.0) {
            Some(i) => (&mut list[i].1, false),
            None => {
                list.push(entry);
                (&mut list.last_mut().unwrap().1, true)
            }
        },
        Node::Leaf(leaf) => {
            if leaf.len() == capacity(leaf.size_class) {
                if let Some(i) = leaf.find_entry(hash, hash_pos, &entry.0) {
                    return (&mut leaf.entries[i].1, false);
                }
                leaf.size_class += 1;
            }
            let (i, inserted) = leaf.insert_entry(hash, hash_pos, entry);
            (&mut leaf.entries[i].1, inserted)
        }
        Node::Branch(branch) => {
            let chunk = get_hash_chunk(hash, hash_pos);
            let mut location = num_set_until(branch.occupation, chunk);
            if test(branch.occupation, chunk) {
                location -= 1;
            } else {
                branch.children.insert(location, Node::Empty);
                branch.occupation |= 1 << chunk;
            }
            insert_recurse(&mut branch.children[location], hash, hash_pos + 1, entry)
        }
    }
}

/// Replaces a full leaf of the largest class that lacks the entry by a
/// branch node.
fn burst_leaf<K: HighsHash, V>(
    node: &mut Node<K, V>,
    hash: u64,
    hash_pos: u32,
    entry: (K, V),
) -> (&mut V, bool) {
    let Node::Leaf(mut leaf) = std::mem::replace(node, Node::Empty) else { unreachable!() };
    let chunk = get_hash_chunk(hash, hash_pos);
    let occupation = leaf.occupation | 1 << chunk;
    let branch_size = occupation.count_ones() as usize;
    let child_of = |chunks16: u16| num_set_until(occupation, first_chunk16(chunks16)) - 1;
    let new_pos = num_set_until(occupation, chunk) - 1;
    let hashes = std::mem::take(&mut leaf.hashes);
    let entries = std::mem::take(&mut leaf.entries);

    if hash_pos + 1 == MAX_DEPTH {
        // children are list leaves; each entry goes to the front of its list
        let mut children: Vec<Node<K, V>> = (0..branch_size).map(|_| Node::Empty).collect();
        let items = hashes.iter().map(|&h| child_of(h)).zip(entries);
        for (pos, e) in items.chain(std::iter::once((new_pos, entry))) {
            match &mut children[pos] {
                Node::List(list) => list.insert(0, e),
                child => *child = Node::List(vec![e]),
            }
        }
        *node = Node::Branch(Branch { occupation, children });
        let Node::Branch(branch) = node else { unreachable!() };
        return (leaf_ref(&mut branch.children[new_pos], 0), true);
    }

    if branch_size == 1 {
        // extremely unlikely: the branch gets a single child, so the leaf
        // moves down a level and the insertion is retried there
        leaf.hashes = hashes;
        leaf.entries = entries;
        leaf.rehash(hash_pos + 1);
        *node = Node::Branch(Branch { occupation, children: vec![Node::Leaf(leaf)] });
        let Node::Branch(branch) = node else { unreachable!() };
        return insert_recurse(&mut branch.children[0], hash, hash_pos + 1, entry);
    }

    // the largest child has at most all items but one per other child
    let max_entries_per_leaf = 2 + entries.len() - branch_size;
    let mut leaves: Vec<InnerLeaf<K, V>> = if max_entries_per_leaf <= capacity(1) {
        (0..branch_size).map(|_| InnerLeaf::new(1)).collect()
    } else {
        // many collisions: size the children exactly
        let mut sizes = vec![0; branch_size];
        sizes[new_pos] += 1;
        for &h in &hashes {
            sizes[child_of(h)] += 1;
        }
        sizes.iter().map(|&s| InnerLeaf::new(entries_to_size_class(s))).collect()
    };
    for (&h, e) in hashes.iter().zip(entries) {
        let full_hash = compute_hash(&e.0);
        leaves[child_of(h)].insert_entry(full_hash, hash_pos + 1, e);
    }
    let children = leaves.into_iter().map(|l| Node::Leaf(Box::new(l))).collect();
    *node = Node::Branch(Branch { occupation, children });
    let Node::Branch(branch) = node else { unreachable!() };
    insert_recurse(&mut branch.children[new_pos], hash, hash_pos + 1, entry)
}

fn merge_into_leaf<K: HighsHash, V>(leaf: &mut InnerLeaf<K, V>, hash_pos: u32, node: Node<K, V>) {
    let entries = match node {
        Node::List(list) => list,
        Node::Leaf(l) => l.entries,
        _ => return,
    };
    for e in entries {
        leaf.insert_entry(compute_hash(&e.0), hash_pos, e);
    }
}

/// Removes the empty child at location (its flag is already cleared), or
/// merges all children into one leaf if they fit.
fn remove_child<K: HighsHash, V>(
    mut branch: Branch<K, V>,
    location: usize,
    hash_pos: u32,
) -> Node<K, V> {
    let new_num_child = branch.occupation.count_ones() as usize;
    if new_num_child * capacity(1) <= LEAF_BURST_THRESHOLD {
        // a cheap estimate from the node types first
        let mut child_entries = 0;
        for child in &branch.children {
            child_entries += child.num_entries_estimate();
            if child_entries > LEAF_BURST_THRESHOLD {
                break;
            }
        }
        if child_entries < LEAF_BURST_THRESHOLD {
            child_entries = branch.children.iter().map(Node::num_entries).sum();
            if child_entries < LEAF_BURST_THRESHOLD {
                let mut leaf = InnerLeaf::new(entries_to_size_class(child_entries));
                for child in branch.children {
                    merge_into_leaf(&mut leaf, hash_pos, child);
                }
                return Node::Leaf(Box::new(leaf));
            }
        }
    }
    branch.children.remove(location);
    Node::Branch(branch)
}

fn erase_recurse<K: HighsHash, V>(node: &mut Node<K, V>, hash: u64, hash_pos: u32, key: &K) {
    match node {
        Node::Empty => {}
        Node::List(list) => {
            if let Some(i) = list.iter().position(|e| e.0 == *key) {
                list.remove(i);
            }
            if list.is_empty() {
                *node = Node::Empty;
            }
        }
        Node::Leaf(leaf) => {
            if leaf.erase_entry(hash, hash_pos, key) {
                if leaf.size_class == 1 {
                    if leaf.len() == 0 {
                        *node = Node::Empty;
                    }
                } else if leaf.len() == capacity(leaf.size_class - 1) {
                    leaf.size_class -= 1;
                }
            }
        }
        Node::Branch(branch) => {
            let chunk = get_hash_chunk(hash, hash_pos);
            if !test(branch.occupation, chunk) {
                return;
            }
            let location = num_set_until(branch.occupation, chunk) - 1;
            erase_recurse(&mut branch.children[location], hash, hash_pos + 1, key);
            if !matches!(branch.children[location], Node::Empty) {
                return;
            }
            branch.occupation ^= 1 << chunk;
            let branch = std::mem::replace(branch, Branch { occupation: 0, children: Vec::new() });
            *node = remove_child(branch, location, hash_pos);
        }
    }
}

fn find_recurse<'a, K: HighsHash, V>(
    mut node: &'a Node<K, V>,
    hash: u64,
    mut hash_pos: u32,
    key: &K,
) -> Option<&'a V> {
    loop {
        match node {
            Node::Empty => return None,
            Node::List(list) => return list.iter().find(|e| e.0 == *key).map(|e| &e.1),
            Node::Leaf(leaf) => {
                return leaf.find_entry(hash, hash_pos, key).map(|i| &leaf.entries[i].1)
            }
            Node::Branch(branch) => {
                let chunk = get_hash_chunk(hash, hash_pos);
                if !test(branch.occupation, chunk) {
                    return None;
                }
                node = &branch.children[num_set_until(branch.occupation, chunk) - 1];
                hash_pos += 1;
            }
        }
    }
}

/// Matches the runs of equal leading chunks of two leaves.
fn find_common_leaves<'a, K: HighsHash, V>(
    leaf1: &'a InnerLeaf<K, V>,
    leaf2: &'a InnerLeaf<K, V>,
) -> Option<&'a (K, V)> {
    let mut match_mask = leaf1.occupation & leaf2.occupation;
    let (mut offset1, mut offset2) = (-1isize, -1isize);
    while match_mask != 0 {
        let pos = log2i64(match_mask);
        match_mask ^= 1 << pos;

        let mut i = (num_set_until(leaf1.occupation, pos) as isize + offset1) as usize;
        while first_chunk16(leaf1.hashes[i]) != pos {
            i += 1;
            offset1 += 1;
        }
        let mut j = (num_set_until(leaf2.occupation, pos) as isize + offset2) as usize;
        while first_chunk16(leaf2.hashes[j]) != pos {
            j += 1;
            offset2 += 1;
        }

        let run_ends = |leaf: &InnerLeaf<K, V>, i: usize| {
            i == leaf.len() || first_chunk16(leaf.hashes[i]) != pos
        };
        loop {
            if leaf1.hashes[i] > leaf2.hashes[j] {
                i += 1;
                if run_ends(leaf1, i) {
                    break;
                }
            } else if leaf2.hashes[j] > leaf1.hashes[i] {
                j += 1;
                if run_ends(leaf2, j) {
                    break;
                }
            } else {
                if leaf1.entries[i].0 == leaf2.entries[j].0 {
                    return Some(&leaf1.entries[i]);
                }
                i += 1;
                if run_ends(leaf1, i) {
                    break;
                }
                j += 1;
                if run_ends(leaf2, j) {
                    break;
                }
            }
        }
    }
    None
}

fn find_common_in_leaf<'a, K: HighsHash, V>(
    leaf: &'a InnerLeaf<K, V>,
    n2: &'a Node<K, V>,
    hash_pos: u32,
) -> Option<&'a (K, V)> {
    match n2 {
        Node::Leaf(leaf2) => find_common_leaves(leaf, leaf2),
        Node::Branch(branch) => {
            let mut match_mask = branch.occupation & leaf.occupation;
            let mut offset = -1isize;
            while match_mask != 0 {
                let pos = log2i64(match_mask);
                match_mask ^= 1 << pos;

                let mut i = (num_set_until(leaf.occupation, pos) as isize + offset) as usize;
                while first_chunk16(leaf.hashes[i]) != pos {
                    i += 1;
                    offset += 1;
                }
                let child = &branch.children[num_set_until(branch.occupation, pos) - 1];
                loop {
                    let key = &leaf.entries[i].0;
                    if find_recurse(child, compute_hash(key), hash_pos + 1, key).is_some() {
                        return Some(&leaf.entries[i]);
                    }
                    i += 1;
                    if !(i < leaf.len() && first_chunk16(leaf.hashes[i]) == pos) {
                        break;
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn find_common_recurse<'a, K: HighsHash, V>(
    mut n1: &'a Node<K, V>,
    mut n2: &'a Node<K, V>,
    hash_pos: u32,
) -> Option<&'a (K, V)> {
    if n1.rank() > n2.rank() {
        std::mem::swap(&mut n1, &mut n2);
    }
    match n1 {
        Node::Empty => None,
        Node::List(list) => {
            list.iter().find(|e| find_recurse(n2, compute_hash(&e.0), hash_pos, &e.0).is_some())
        }
        Node::Leaf(leaf) => find_common_in_leaf(leaf, n2, hash_pos),
        Node::Branch(branch1) => {
            let Node::Branch(branch2) = n2 else { unreachable!() };
            let mut match_mask = branch1.occupation & branch2.occupation;
            while match_mask != 0 {
                let pos = log2i64(match_mask);
                match_mask ^= 1 << pos;
                let child1 = &branch1.children[num_set_until(branch1.occupation, pos) - 1];
                let child2 = &branch2.children[num_set_until(branch2.occupation, pos) - 1];
                if let Some(e) = find_common_recurse(child1, child2, hash_pos + 1) {
                    return Some(e);
                }
            }
            None
        }
    }
}

fn find_map_recurse<K, V, R>(
    node: &Node<K, V>,
    f: &mut impl FnMut(&K, &V) -> Option<R>,
) -> Option<R> {
    match node {
        Node::Empty => None,
        Node::List(list) => list.iter().find_map(|(k, v)| f(k, v)),
        Node::Leaf(leaf) => leaf.entries.iter().find_map(|(k, v)| f(k, v)),
        Node::Branch(branch) => branch.children.iter().find_map(|c| find_map_recurse(c, f)),
    }
}

fn for_each_mut_recurse<K, V>(node: &mut Node<K, V>, f: &mut impl FnMut(&K, &mut V)) {
    match node {
        Node::Empty => {}
        Node::List(list) => list.iter_mut().for_each(|(k, v)| f(k, v)),
        Node::Leaf(leaf) => leaf.entries.iter_mut().for_each(|(k, v)| f(k, v)),
        Node::Branch(branch) => branch.children.iter_mut().for_each(|c| for_each_mut_recurse(c, f)),
    }
}

#[derive(Clone)]
pub struct HighsHashTree<K, V = ()> {
    root: Node<K, V>,
}

impl<K: HighsHash, V> Default for HighsHashTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: HighsHash, V> HighsHashTree<K, V> {
    pub fn new() -> Self {
        HighsHashTree { root: Node::Empty }
    }

    /// Inserts if the key is new; returns whether it was.
    pub fn insert(&mut self, key: K, value: V) -> bool {
        self.insert_or_get(key, value).1
    }

    /// The value of the key, inserted if new, and whether it was inserted.
    pub fn insert_or_get(&mut self, key: K, value: V) -> (&mut V, bool) {
        let hash = compute_hash(&key);
        insert_recurse(&mut self.root, hash, 0, (key, value))
    }

    pub fn erase(&mut self, key: &K) {
        erase_recurse(&mut self.root, compute_hash(key), 0, key);
    }

    pub fn contains(&self, key: &K) -> bool {
        self.find(key).is_some()
    }

    pub fn find(&self, key: &K) -> Option<&V> {
        find_recurse(&self.root, compute_hash(key), 0, key)
    }

    /// Some entry present in both trees, the same one as the C++ finds.
    pub fn find_common<'a>(&'a self, other: &'a Self) -> Option<(&'a K, &'a V)> {
        find_common_recurse(&self.root, &other.root, 0).map(|(k, v)| (k, v))
    }

    pub fn is_empty(&self) -> bool {
        matches!(self.root, Node::Empty)
    }

    pub fn clear(&mut self) {
        self.root = Node::Empty;
    }

    pub fn for_each(&self, mut f: impl FnMut(&K, &V)) {
        find_map_recurse(&self.root, &mut |k, v| {
            f(k, v);
            None::<()>
        });
    }

    pub fn for_each_mut(&mut self, mut f: impl FnMut(&K, &mut V)) {
        for_each_mut_recurse(&mut self.root, &mut f);
    }

    /// for_each that stops at the first `Some` the callback returns, as the
    /// C++ for_each with a non-void callback.
    pub fn find_map<R>(&self, mut f: impl FnMut(&K, &V) -> Option<R>) -> Option<R> {
        find_map_recurse(&self.root, &mut f)
    }
}
