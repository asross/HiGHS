//! highs_splay, highs_splay_link and highs_splay_unlink (highs/util/
//! HighsSplay.h): a top down splay tree stored in arrays of left/right
//! child links, step for step as the C++ (the shapes of the trees decide
//! the iteration orders of HPresolve's rows).
//!
//! The C++ walks with pointers that point either at a local root of the
//! left/right trees or at a child link of a node; [`Slot`] names them.

#[derive(Clone, Copy)]
enum Slot {
    /// the local Nleft / Nright
    NLeft,
    NRight,
    Left(usize),
    Right(usize),
}

/// The child links and keys of a splay tree
pub struct Tree<'a> {
    pub left: &'a mut [i32],
    pub right: &'a mut [i32],
    pub key: &'a [i32],
}

impl Tree<'_> {
    #[inline]
    fn set(&mut self, s: Slot, v: i32, nleft: &mut i32, nright: &mut i32) {
        match s {
            Slot::NLeft => *nleft = v,
            Slot::NRight => *nright = v,
            Slot::Left(i) => self.left[i] = v,
            Slot::Right(i) => self.right[i] = v,
        }
    }

    /// highs_splay: returns the new root
    pub fn splay(&mut self, key: i32, mut root: i32) -> i32 {
        if root == -1 {
            return -1;
        }
        let mut nleft = -1i32;
        let mut nright = -1i32;
        let mut lright = Slot::NRight;
        let mut rleft = Slot::NLeft;
        loop {
            let r = root as usize;
            let rk = self.key[r];
            if key < rk {
                let left = self.left[r];
                if left == -1 {
                    break;
                }
                if key < self.key[left as usize] {
                    let y = left as usize;
                    self.left[r] = self.right[y];
                    self.right[y] = root;
                    root = y as i32;
                    if self.left[y] == -1 {
                        break;
                    }
                }
                self.set(rleft, root, &mut nleft, &mut nright);
                rleft = Slot::Left(root as usize);
                root = self.left[root as usize];
            } else if key > rk {
                let right = self.right[r];
                if right == -1 {
                    break;
                }
                if key > self.key[right as usize] {
                    let y = right as usize;
                    self.right[r] = self.left[y];
                    self.left[y] = root;
                    root = y as i32;
                    if self.right[y] == -1 {
                        break;
                    }
                }
                self.set(lright, root, &mut nleft, &mut nright);
                lright = Slot::Right(root as usize);
                root = self.right[root as usize];
            } else {
                break;
            }
        }
        let r = root as usize;
        let (l, rr) = (self.left[r], self.right[r]);
        self.set(lright, l, &mut nleft, &mut nright);
        self.set(rleft, rr, &mut nleft, &mut nright);
        self.left[r] = nright;
        self.right[r] = nleft;
        root
    }

    /// highs_splay_link
    pub fn link(&mut self, node: i32, root: &mut i32) {
        let n = node as usize;
        if *root == -1 {
            self.left[n] = -1;
            self.right[n] = -1;
            *root = node;
            return;
        }
        *root = self.splay(self.key[n], *root);
        let r = *root as usize;
        if self.key[n] < self.key[r] {
            self.left[n] = self.left[r];
            self.right[n] = *root;
            self.left[r] = -1;
        } else {
            debug_assert!(self.key[n] > self.key[r]);
            self.right[n] = self.right[r];
            self.left[n] = *root;
            self.right[r] = -1;
        }
        *root = node;
    }

    /// highs_splay_unlink
    pub fn unlink(&mut self, node: i32, root: &mut i32) {
        debug_assert!(*root != -1);
        let n = node as usize;
        *root = self.splay(self.key[n], *root);
        if *root != node {
            // equal keys are in the right subtree
            let r = *root as usize;
            let mut sub = self.right[r];
            self.unlink(node, &mut sub);
            self.right[r] = sub;
            return;
        }
        if self.left[n] == -1 {
            *root = self.right[n];
        } else {
            *root = self.splay(self.key[n], self.left[n]);
            let r = *root as usize;
            self.right[r] = self.right[n];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tree;

    fn in_order(t: &Tree, n: i32, out: &mut Vec<i32>) {
        if n == -1 {
            return;
        }
        in_order(t, t.left[n as usize], out);
        out.push(t.key[n as usize]);
        in_order(t, t.right[n as usize], out);
    }

    #[test]
    fn link_unlink_keeps_order() {
        let key: Vec<i32> = (0..50).map(|i| (i * 37) % 50).collect();
        let (mut left, mut right) = (vec![-1; 50], vec![-1; 50]);
        let mut t = Tree { left: &mut left, right: &mut right, key: &key };
        let mut root = -1;
        for n in 0..50 {
            t.link(n, &mut root);
        }
        for n in (0..50).step_by(3) {
            t.unlink(n, &mut root);
        }
        let mut v = Vec::new();
        in_order(&t, root, &mut v);
        let mut want: Vec<i32> = (0..50).filter(|n| n % 3 != 0).map(|n| key[n as usize]).collect();
        want.sort();
        assert_eq!(v, want);
        let r = t.splay(want[5], root);
        assert_eq!(t.key[r as usize], want[5]);
    }
}
