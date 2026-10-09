//! HighsConflictPool (highs/mip/HighsConflictPool.cpp): the conflicts (sets
//! of bound changes that cannot hold together), their ages and the free
//! space of the entry array. Rust-owned; the C++ class is a handle. The
//! domains propagating the pool (C++ ConflictPoolPropagation) are told of
//! added and deleted conflicts through the C++ callbacks; no Rust borrow of
//! the pool is held across them, as they read the pool.

use super::domain::DomChg;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

/// conflictAdded / conflictDeleted of a C++ ConflictPoolPropagation
pub type PropFn = unsafe extern "C" fn(*mut c_void, i32);

pub struct ConflictPool {
    agelim: i32,
    softlimit: i32,
    age_lock: bool,
    age_distribution: Vec<i32>,
    ages: Vec<i16>,
    modification: Vec<u32>,
    age_reset_while_locked: Vec<AtomicU8>,
    pub entries: Vec<DomChg>,
    pub ranges: Vec<[i32; 2]>,
    free_spaces: BTreeSet<(i32, i32)>,
    deleted: Vec<i32>,
    prop_domains: Vec<*mut c_void>,
    conflict_added: PropFn,
    conflict_deleted: PropFn,
}

impl ConflictPool {
    pub fn new(agelim: i32, softlimit: i32, conflict_added: PropFn, conflict_deleted: PropFn) -> Self {
        ConflictPool {
            agelim,
            softlimit,
            age_lock: false,
            age_distribution: vec![0; agelim as usize + 1],
            ages: Vec::new(),
            modification: Vec::new(),
            age_reset_while_locked: Vec::new(),
            entries: Vec::new(),
            ranges: Vec::new(),
            free_spaces: BTreeSet::new(),
            deleted: Vec::new(),
            prop_domains: Vec::new(),
            conflict_added,
            conflict_deleted,
        }
    }

    pub fn num_conflicts(&self) -> i32 {
        (self.ranges.len() - self.deleted.len()) as i32
    }

    pub fn modification_count(&self, c: i32) -> u32 {
        self.modification[c as usize]
    }

    pub fn set_age_limit(&mut self, agelim: i32) {
        self.agelim = agelim;
        self.age_distribution.resize(agelim as usize + 1, 0);
    }

    pub fn set_age_lock(&mut self, lock: bool) {
        self.age_lock = lock;
    }

    pub fn add_propagation_domain(&mut self, d: *mut c_void) {
        self.prop_domains.push(d);
    }

    pub fn remove_propagation_domain(&mut self, d: *mut c_void) {
        if let Some(k) = self.prop_domains.iter().rposition(|&x| x == d) {
            self.prop_domains.remove(k);
        }
    }

    /// resetAge
    ///
    /// # Safety
    /// `p` a live pool; under the age lock other threads may call this at
    /// the same time (only the atomic flag is written then)
    pub unsafe fn reset_age(p: *mut ConflictPool, conflict: i32) {
        let c = conflict as usize;
        if *(*p).ages.as_ptr().add(c) > 0 {
            if (*p).age_lock {
                (&(*p).age_reset_while_locked)[c].store(1, Relaxed);
                return;
            }
            let s = &mut *p;
            s.age_distribution[s.ages[c] as usize] -= 1;
            s.age_distribution[0] += 1;
            s.ages[c] = 0;
        }
    }

    /// The range of a new conflict of `len` entries and its index (the
    /// common start of the add functions)
    fn new_conflict(&mut self, len: i32) -> (i32, i32) {
        let start;
        let end;
        let slot = self.free_spaces.range((len, -1)..).next().copied();
        match slot {
            None => {
                start = self.entries.len() as i32;
                end = start + len;
                self.entries.resize(end as usize, DomChg::default());
            }
            Some(free) => {
                self.free_spaces.remove(&free);
                start = free.1;
                end = start + len;
                if free.0 > len {
                    self.free_spaces.insert((free.0 - len, end));
                }
            }
        }
        let index = match self.deleted.pop() {
            None => {
                let index = self.ranges.len() as i32;
                self.ranges.push([start, end]);
                let n = self.ranges.len();
                self.ages.resize(n, 0);
                self.modification.resize(n, 0);
                self.age_reset_while_locked.resize_with(n, || AtomicU8::new(0));
                index
            }
            Some(index) => {
                self.ranges[index as usize] = [start, end];
                index
            }
        };
        let i = index as usize;
        self.age_reset_while_locked[i].store(0, Relaxed);
        self.modification[i] = self.modification[i].wrapping_add(1);
        self.ages[i] = 0;
        self.age_distribution[0] += 1;
        (index, start)
    }

    /// Tells the propagation domains of a new conflict
    ///
    /// # Safety
    /// `p` live, no borrow of it held
    unsafe fn added(p: *mut ConflictPool, index: i32) {
        let mut k = 0;
        while k < (*p).prop_domains.len() {
            let d = *(*p).prop_domains.as_ptr().add(k);
            ((*p).conflict_added)(d, index);
            k += 1;
        }
    }

    /// addConflictCut (reconvergence `None`) / addReconvergenceCut (the
    /// flipped reconvergence domain change first); the C++ relaxed the
    /// continuous entries by the feasibility tolerance
    ///
    /// # Safety
    /// `p` live, no borrow of it held
    pub unsafe fn add_conflict(p: *mut ConflictPool, entries: &[DomChg], flipped: Option<DomChg>) {
        let index = {
            let s = &mut *p;
            let len = entries.len() as i32 + flipped.is_some() as i32;
            let (index, start) = s.new_conflict(len);
            let mut i = start as usize;
            if let Some(f) = flipped {
                s.entries[i] = f;
                i += 1;
            }
            for &e in entries {
                s.entries[i] = e;
                i += 1;
            }
            index
        };
        Self::added(p, index);
    }

    /// addConflictFromOtherPool
    ///
    /// # Safety
    /// as add_conflict
    pub unsafe fn add_from_other(p: *mut ConflictPool, entries: &[DomChg]) {
        let index = {
            let s = &mut *p;
            let (index, start) = s.new_conflict(entries.len() as i32);
            s.entries[start as usize..start as usize + entries.len()].copy_from_slice(entries);
            index
        };
        Self::added(p, index);
    }

    /// removeConflict
    ///
    /// # Safety
    /// as add_conflict
    pub unsafe fn remove_conflict(p: *mut ConflictPool, conflict: i32) {
        let mut k = 0;
        while k < (*p).prop_domains.len() {
            let d = *(*p).prop_domains.as_ptr().add(k);
            ((*p).conflict_deleted)(d, conflict);
            k += 1;
        }
        let s = &mut *p;
        let c = conflict as usize;
        if s.ages[c] >= 0 {
            s.age_distribution[s.ages[c] as usize] -= 1;
            s.ages[c] = -1;
        }
        let [start, end] = s.ranges[c];
        s.deleted.push(conflict);
        s.free_spaces.insert((end - start, start));
        s.ranges[c] = [-1, -1];
        s.modification[c] = s.modification[c].wrapping_add(1);
    }

    /// performAging
    ///
    /// # Safety
    /// as add_conflict
    pub unsafe fn perform_aging(p: *mut ConflictPool, thread_safe: bool) {
        if (*p).age_lock {
            return;
        }
        let (max_index, agelim) = {
            let s = &*p;
            let mut agelim = s.agelim;
            let mut active = s.num_conflicts();
            while agelim > 5 && active > s.softlimit {
                active -= s.age_distribution[agelim as usize];
                agelim -= 1;
            }
            (s.ranges.len(), agelim)
        };
        for i in 0..max_index {
            if (&(*p).ages)[i] < 0 {
                continue;
            }
            if thread_safe && (&(*p).age_reset_while_locked)[i].load(Relaxed) == 1 {
                Self::reset_age(p, i as i32);
            }
            let remove = {
                let s = &mut *p;
                s.age_distribution[s.ages[i] as usize] -= 1;
                s.ages[i] += 1;
                s.age_reset_while_locked[i].store(0, Relaxed);
                if s.ages[i] as i32 > agelim {
                    s.ages[i] = -1;
                    true
                } else {
                    s.age_distribution[s.ages[i] as usize] += 1;
                    false
                }
            };
            if remove {
                Self::remove_conflict(p, i as i32);
            }
        }
    }

    /// syncConflictPool: moves the conflicts of `p` into `sync`
    ///
    /// # Safety
    /// distinct live pools, no borrows held
    pub unsafe fn sync(p: *mut ConflictPool, sync: *mut ConflictPool) {
        let n = (*p).ranges.len();
        for i in 0..n {
            if (&(*p).ages)[i] < 0 {
                continue;
            }
            let [start, end] = (&(*p).ranges)[i];
            let entries: Vec<DomChg> = (&(*p).entries)[start as usize..end as usize].to_vec();
            Self::add_from_other(sync, &entries);
            Self::remove_conflict(p, i as i32);
        }
        let s = &mut *p;
        s.deleted.clear();
        s.free_spaces.clear();
        s.ranges.clear();
        s.entries.clear();
        s.modification.clear();
        s.ages.clear();
        s.age_reset_while_locked.clear();
    }
}

pub(crate) mod ffi {
    use super::*;
    use crate::ffi::sl;

    #[no_mangle]
    pub extern "C" fn highs_rs_conflictpool_new(
        agelim: i32,
        softlimit: i32,
        added: PropFn,
        deleted: PropFn,
    ) -> *mut ConflictPool {
        Box::into_raw(Box::new(ConflictPool::new(agelim, softlimit, added, deleted)))
    }

    /// # Safety
    /// `p` from highs_rs_conflictpool_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_free(p: *mut ConflictPool) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// The entries (`which` 0) or ranges (1): data, length in `*len`
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_data(p: *const ConflictPool, which: i32, len: *mut usize) -> *const c_void {
        if which == 0 {
            *len = (*p).entries.len();
            (*p).entries.as_ptr() as *const c_void
        } else {
            *len = (*p).ranges.len();
            (*p).ranges.as_ptr() as *const c_void
        }
    }

    /// 0 getNumConflicts, 1 getModificationCount(i)
    ///
    /// # Safety
    /// live pool
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_get(p: *const ConflictPool, which: i32, i: i32) -> u32 {
        match which {
            0 => (*p).num_conflicts() as u32,
            _ => (*p).modification_count(i),
        }
    }

    /// 0 resetAge(i), 1 setAgeLimit(i), 2 setAgeLock(i), 3 performAging
    /// (thread_safe = i), 4 removeConflict(i), 5 add / 6 remove the
    /// propagation domain `d`
    ///
    /// # Safety
    /// live pool, `d` a live ConflictPoolPropagation for 5
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_op(p: *mut ConflictPool, which: i32, i: i32, d: *mut c_void) {
        match which {
            0 => ConflictPool::reset_age(p, i),
            1 => (*p).set_age_limit(i),
            2 => (*p).set_age_lock(i != 0),
            3 => ConflictPool::perform_aging(p, i != 0),
            4 => ConflictPool::remove_conflict(p, i),
            5 => (*p).add_propagation_domain(d),
            _ => (*p).remove_propagation_domain(d),
        }
    }

    /// addConflictCut (flipped null) / addReconvergenceCut with the flipped
    /// reconvergence domain change
    ///
    /// # Safety
    /// live pool, arrays valid for their lengths
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_add(
        p: *mut ConflictPool,
        entries: *const DomChg,
        len: i32,
        flipped: *const DomChg,
    ) {
        let flipped = if flipped.is_null() { None } else { Some(*flipped) };
        ConflictPool::add_conflict(p, sl(entries, len), flipped);
    }

    /// addConflictFromOtherPool
    ///
    /// # Safety
    /// live pool, `entries` valid for `len`
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_add_from_other(p: *mut ConflictPool, entries: *const DomChg, len: i32) {
        ConflictPool::add_from_other(p, sl(entries, len));
    }

    /// syncConflictPool
    ///
    /// # Safety
    /// distinct live pools
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_conflictpool_sync(p: *mut ConflictPool, sync: *mut ConflictPool) {
        ConflictPool::sync(p, sync);
    }
}
