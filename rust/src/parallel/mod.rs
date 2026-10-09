//! The HiGHS task scheduler (highs/parallel: HighsTaskExecutor,
//! HighsSplitDeque, HighsTask, HighsBinarySemaphore, HighsSpinMutex,
//! HighsSchedulerConstants, HighsCacheAlign): a work-stealing executor with
//! one split deque per worker thread (Wagner and Ferrara's split deque: the
//! owner pushes and pops at the head without atomics, stealers take from
//! the tail below the split point), sleeping workers on a lock-free stack
//! that new work wakes, and the leapfrogging sync of stolen tasks.
//!
//! The memory orderings, the spin and sleep thresholds and the random
//! victim choice are the C++'s. The C++ keeps its header API (spawn, sync,
//! TaskGroup, for_each are templates that place the callable in a task
//! slot): a task slot has the C++ HighsTask layout (56 bytes of callable,
//! then the stealer word), and a stolen task is run through the function
//! given at initialization (`RunFn`, the C++ calls the callable and catches
//! HighsTask::Interrupt, returning true then). Where the C++ throws
//! HighsTask::Interrupt from the scheduler (checkInterrupt, a sync whose
//! leapfrogging ran a cancelled task), the Rust functions return true and
//! the C++ header throws.
//!
//! Results never depend on the scheduling: the tasks of the solvers write
//! disjoint data or are reduced in a fixed order.

use crate::util::random::HighsRandom;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

use Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};

/// HighsSplitDeque::kTaskArraySize
pub const TASK_ARRAY_SIZE: u32 = 8192;
/// HighsSchedulerConstants
const NUM_TRY_FAC: i32 = 16;
const MICRO_SECS_BEFORE_SLEEP: u128 = 5000;
const MICRO_SECS_BEFORE_GLOBAL_SYNC: u128 = 1000;

/// HighsSpinMutex::yieldProcessor: _mm_pause on x86_64, a thread yield
/// elsewhere
#[inline(always)]
fn yield_processor() {
    #[cfg(target_arch = "x86_64")]
    std::hint::spin_loop();
    #[cfg(not(target_arch = "x86_64"))]
    std::thread::yield_now();
}

fn micros_since(t: Instant) -> u128 {
    t.elapsed().as_micros()
}

// ---- tasks ----

const FINISHED_FLAG: usize = 1;
const CANCEL_FLAG: usize = 2;
const PTR_MASK: usize = !(FINISHED_FLAG | CANCEL_FLAG);

/// Runs a stolen task's callable; true if it was interrupted
/// (HighsTask::Interrupt)
pub type RunFn = unsafe extern "C" fn(*mut Task) -> bool;

static RUN_FN: AtomicUsize = AtomicUsize::new(0);

/// A task slot (HighsTask): the callable, placed by the C++, and the
/// stealer word (the stealing deque, with the finished and cancel flags)
#[repr(C, align(64))]
pub struct Task {
    data: UnsafeCell<[u64; 7]>,
    stealer: AtomicUsize,
}

impl Task {
    fn call(&self) -> bool {
        let f = RUN_FN.load(Relaxed);
        // SAFETY: the function set at initialization; the slot holds a
        // callable placed by its owner before it was published
        unsafe {
            let f: RunFn = std::mem::transmute::<usize, RunFn>(f);
            f(self as *const Task as *mut Task)
        }
    }

    fn mark_as_finished(&self, stealer: *const Deque) -> *const Deque {
        let state = self.stealer.swap(FINISHED_FLAG, Release);
        let waiting_owner = (state & PTR_MASK) as *const Deque;
        if waiting_owner != stealer {
            waiting_owner
        } else {
            std::ptr::null()
        }
    }

    /// run(stealer): (the owner to notify, interrupted)
    fn run_stolen(&self, stealer: *const Deque) -> (*const Deque, bool) {
        let state = self.stealer.fetch_or(stealer as usize, Acquire);
        if state == 0 && self.call() {
            return (std::ptr::null(), true);
        }
        (self.mark_as_finished(stealer), false)
    }

    fn cancel(&self) {
        self.stealer.fetch_or(CANCEL_FLAG, Release);
    }

    fn request_notify_when_finished(&self, owner: *const Deque, stealer: *const Deque) -> bool {
        let xormask = (owner as usize) ^ (stealer as usize);
        let state = self.stealer.fetch_xor(xormask, Acquire);
        debug_assert!(!stealer.is_null());
        state & FINISHED_FLAG == 0
    }

    fn is_finished(&self) -> bool {
        self.stealer.load(Acquire) & FINISHED_FLAG != 0
    }

    fn is_cancelled(&self) -> bool {
        self.stealer.load(Relaxed) & CANCEL_FLAG != 0
    }

    /// getStealerIfUnfinished: (stealer or null, cancelled)
    fn get_stealer_if_unfinished(&self) -> (*const Deque, bool) {
        let mut state = self.stealer.load(Acquire);
        if state & FINISHED_FLAG != 0 {
            return (std::ptr::null(), false);
        }
        while state & !CANCEL_FLAG == 0 {
            yield_processor();
            state = self.stealer.load(Acquire);
        }
        if state & FINISHED_FLAG != 0 {
            return (std::ptr::null(), false);
        }
        ((state & PTR_MASK) as *const Deque, state & CANCEL_FLAG != 0)
    }
}

// ---- HighsBinarySemaphore ----

struct BinarySemaphore {
    count: AtomicI32,
    mutex: Mutex<()>,
    condvar: Condvar,
}

impl BinarySemaphore {
    fn new() -> Self {
        BinarySemaphore { count: AtomicI32::new(0), mutex: Mutex::new(()), condvar: Condvar::new() }
    }

    fn release(&self) {
        let prev = self.count.swap(1, Release);
        if prev < 0 {
            let _lg = self.mutex.lock().unwrap();
            self.condvar.notify_one();
        }
    }

    fn try_acquire(&self) -> bool {
        self.count.compare_exchange_weak(1, 0, Acquire, Relaxed).is_ok()
    }

    fn acquire(&self) {
        if self.try_acquire() {
            return;
        }
        let t_start = Instant::now();
        let mut spin_iters = 10;
        loop {
            for _ in 0..spin_iters {
                if self.count.load(Acquire) == 1 && self.try_acquire() {
                    return;
                }
                yield_processor();
            }
            if micros_since(t_start) < MICRO_SECS_BEFORE_SLEEP {
                spin_iters *= 2;
            } else {
                break;
            }
        }
        let lg = self.mutex.lock().unwrap();
        self.acquire_locked(lg, Acquire);
    }

    fn lock_mutex_for_acquire(&self) -> MutexGuard<'_, ()> {
        self.mutex.lock().unwrap()
    }

    /// acquire with the mutex held (the wake-up check reads with `order`)
    fn acquire_locked(&self, mut lg: MutexGuard<'_, ()>, order: Ordering) {
        let prev = self.count.swap(-1, Relaxed);
        if prev == 1 {
            self.count.store(0, Relaxed);
            return;
        }
        loop {
            lg = self.condvar.wait(lg).unwrap();
            if self.count.load(order) == 1 {
                break;
            }
        }
        self.count.store(0, Relaxed);
    }
}

// ---- the split deque ----

#[repr(C, align(64))]
struct Padded<T>(T);

struct OwnerData {
    worker_bunk: *const WorkerBunk,
    workers: *const Box<Deque>,
    randgen: HighsRandom,
    head: u32,
    split_copy: u32,
    num_workers: i32,
    owner_id: i32,
    root_task: *mut Task,
    all_stolen_copy: bool,
}

struct StealerData {
    semaphore: BinarySemaphore,
    injected_task: AtomicPtr<Task>,
    ts: AtomicU64,
    all_stolen: AtomicBool,
}

struct WorkerBunkData {
    next_sleeper: AtomicPtr<Deque>,
    owner_id: i32,
}

#[inline(always)]
fn make_tail_split(tail: u32, split: u32) -> u64 {
    ((tail as u64) << 32) | split as u64
}
#[inline(always)]
fn tail(ts: u64) -> u32 {
    (ts >> 32) as u32
}
#[inline(always)]
fn split(ts: u64) -> u32 {
    ts as u32
}

/// HighsSplitDeque::Status
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Empty = 0,
    Stolen = 1,
    Work = 2,
    Overflown = 3,
}

/// A worker's deque (HighsSplitDeque). The owner data is only used by the
/// owning thread; the stealer data, the split request and the bunk data
/// are shared.
#[repr(C, align(64))]
pub struct Deque {
    owner: Padded<UnsafeCell<OwnerData>>,
    split_request: Padded<AtomicBool>,
    stealer: Padded<StealerData>,
    bunk_data: Padded<WorkerBunkData>,
    tasks: [Task; TASK_ARRAY_SIZE as usize],
}

// SAFETY: the owner data is touched by the owning thread only; everything
// shared is atomic or behind the semaphore's mutex, and a task slot's data
// is written by the owner before it is published to stealers
unsafe impl Sync for Deque {}
unsafe impl Send for Deque {}

/// HighsSplitDeque::WorkerBunk: the number of deques with work, and the
/// stack of sleeping workers (an index with an ABA tag)
struct WorkerBunk {
    have_jobs: Padded<AtomicI32>,
    sleeper_stack: Padded<AtomicU64>,
}

const ABA_TAG_SHIFT: u64 = 20;
const INDEX_MASK: u64 = (1u64 << ABA_TAG_SHIFT) - 1;

impl WorkerBunk {
    fn push_sleeper(&self, deque: &Deque) {
        let mut stack_state = self.sleeper_stack.0.load(Relaxed);
        loop {
            let head: *mut Deque = if stack_state & INDEX_MASK != 0 {
                deque.worker((stack_state & INDEX_MASK) as i32 - 1) as *const Deque as *mut Deque
            } else {
                std::ptr::null_mut()
            };
            deque.bunk_data.0.next_sleeper.store(head, Relaxed);
            let mut new_state = (stack_state >> ABA_TAG_SHIFT) + 1;
            new_state = (new_state << ABA_TAG_SHIFT) | (deque.bunk_data.0.owner_id + 1) as u64;
            match self.sleeper_stack.0.compare_exchange_weak(stack_state, new_state, Release, Relaxed) {
                Ok(_) => return,
                Err(s) => stack_state = s,
            }
        }
    }

    fn pop_sleeper(&self, local: &Deque) -> *const Deque {
        let mut stack_state = self.sleeper_stack.0.load(Relaxed);
        let head;
        loop {
            if stack_state & INDEX_MASK == 0 {
                return std::ptr::null();
            }
            let h = local.worker((stack_state & INDEX_MASK) as i32 - 1);
            let new_head = h.bunk_data.0.next_sleeper.load(Relaxed);
            let new_head_id = if new_head.is_null() {
                0
                // SAFETY: a deque of the executor
            } else {
                unsafe { (*new_head).bunk_data.0.owner_id + 1 }
            };
            let mut new_state = (stack_state >> ABA_TAG_SHIFT) + 1;
            new_state = (new_state << ABA_TAG_SHIFT) | new_head_id as u64;
            match self.sleeper_stack.0.compare_exchange_weak(stack_state, new_state, Acquire, Relaxed) {
                Ok(_) => {
                    head = h;
                    break;
                }
                Err(s) => stack_state = s,
            }
        }
        head.bunk_data.0.next_sleeper.store(std::ptr::null_mut(), Relaxed);
        head
    }

    fn publish_work(&self, local: &Deque) {
        let mut sleeper = self.pop_sleeper(local);
        while !sleeper.is_null() {
            // SAFETY: a deque of the executor
            let sl = unsafe { &*sleeper };
            let t = local.self_steal_and_get_tail();
            let o = local.o();
            if t == o.split_copy {
                if o.head == o.split_copy {
                    o.all_stolen_copy = true;
                    local.stealer.0.all_stolen.store(true, Relaxed);
                    self.have_jobs.0.fetch_add(-1, Release);
                }
                self.push_sleeper(sl);
                return;
            } else {
                sl.inject_task_and_notify(local.task(t));
            }
            if t == o.split_copy - 1 {
                if o.head == o.split_copy {
                    o.all_stolen_copy = true;
                    local.stealer.0.all_stolen.store(true, Relaxed);
                    self.have_jobs.0.fetch_add(-1, Release);
                }
                return;
            }
            sleeper = self.pop_sleeper(local);
        }
    }

    fn wait_for_new_task(&self, local: &Deque) -> *mut Task {
        self.push_sleeper(local);
        local.stealer.0.semaphore.acquire();
        local.stealer.0.injected_task.load(Relaxed)
    }
}

impl Deque {
    /// A deque on the heap (512 KB of task slots)
    fn new_boxed(bunk: *const WorkerBunk, workers: *const Box<Deque>, owner_id: i32, num_workers: i32) -> Box<Deque> {
        let layout = std::alloc::Layout::new::<Deque>();
        // SAFETY: a zeroed Deque is a valid one except for the fields
        // written below (all-zero atomics, null pointers, empty slots); the
        // semaphore and owner data are written in place before use
        unsafe {
            let p = std::alloc::alloc_zeroed(layout) as *mut Deque;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            std::ptr::addr_of_mut!((*p).owner).write(Padded(UnsafeCell::new(OwnerData {
                worker_bunk: bunk,
                workers,
                randgen: HighsRandom::new(owner_id as u32),
                head: 0,
                split_copy: 0,
                num_workers,
                owner_id,
                root_task: std::ptr::null_mut(),
                all_stolen_copy: true,
            })));
            std::ptr::addr_of_mut!((*p).split_request).write(Padded(AtomicBool::new(false)));
            std::ptr::addr_of_mut!((*p).stealer).write(Padded(StealerData {
                semaphore: BinarySemaphore::new(),
                injected_task: AtomicPtr::new(std::ptr::null_mut()),
                ts: AtomicU64::new(0),
                all_stolen: AtomicBool::new(true),
            }));
            std::ptr::addr_of_mut!((*p).bunk_data)
                .write(Padded(WorkerBunkData { next_sleeper: AtomicPtr::new(std::ptr::null_mut()), owner_id }));
            Box::from_raw(p)
        }
    }

    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    fn o(&self) -> &mut OwnerData {
        // SAFETY: the owner data is used by the owning thread only, and no
        // reference to it is held across a call that takes another
        unsafe { &mut *self.owner.0.get() }
    }

    #[inline(always)]
    fn bunk(&self) -> &WorkerBunk {
        // SAFETY: the executor's bunk outlives its deques
        unsafe { &*self.o().worker_bunk }
    }

    #[inline(always)]
    fn worker(&self, id: i32) -> &Deque {
        // SAFETY: the executor's deques (a vector that is not resized)
        // outlive each other's uses
        unsafe { &**self.o().workers.add(id as usize) }
    }

    #[inline(always)]
    fn task(&self, i: u32) -> *mut Task {
        &self.tasks[i as usize] as *const Task as *mut Task
    }


    fn grow_shared(&self) {
        let o = self.o();
        let have_jobs = self.bunk().have_jobs.0.load(Relaxed);
        let mut split_rq = false;
        if have_jobs == o.num_workers {
            split_rq = self.split_request.0.load(Relaxed);
            if !split_rq {
                return;
            }
        }
        let new_split = TASK_ARRAY_SIZE.min(o.head);
        debug_assert!(new_split > o.split_copy);
        let xor_mask = (o.split_copy ^ new_split) as u64;
        self.stealer.0.ts.fetch_xor(xor_mask, Release);
        o.split_copy = new_split;
        if split_rq {
            self.split_request.0.store(false, Relaxed);
        } else {
            self.bunk().publish_work(self);
        }
    }

    fn shrink_shared(&self) -> bool {
        let o = self.o();
        let mut t = tail(self.stealer.0.ts.load(Relaxed));
        let s = o.split_copy;
        if t != s {
            o.split_copy = (t + s) / 2;
            t = tail(self.stealer.0.ts.fetch_add((o.split_copy as u64).wrapping_sub(s as u64), AcqRel));
            if t != s {
                if t > o.split_copy {
                    o.split_copy = (t + s) / 2;
                    self.stealer.0.ts.store(make_tail_split(t, o.split_copy), Relaxed);
                }
                return false;
            }
        }
        self.stealer.0.all_stolen.store(true, Relaxed);
        o.all_stolen_copy = true;
        self.bunk().have_jobs.0.fetch_add(-1, Relaxed);
        true
    }

    /// checkInterrupt: true where the C++ throws HighsTask::Interrupt
    pub fn check_interrupt(&self) -> bool {
        let o = self.o();
        // SAFETY: the root task is a live slot of a deque
        if !o.root_task.is_null() && unsafe { (*o.root_task).is_cancelled() } {
            o.root_task = std::ptr::null_mut();
            return true;
        }
        false
    }

    pub fn cancel_task(&self, task_index: i32) {
        debug_assert!(task_index >= 0 && (task_index as u32) < self.o().head);
        self.tasks[task_index as usize].cancel();
    }

    pub fn set_root_task(&self, new_root: *mut Task) -> *mut Task {
        let o = self.o();
        std::mem::replace(&mut o.root_task, new_root)
    }

    /// The first part of push: the slot for the task, or null if the deque
    /// overflowed (the caller then runs the task itself)
    pub fn push_slot(&self) -> *mut Task {
        let o = self.o();
        if o.head >= TASK_ARRAY_SIZE {
            if o.split_copy < TASK_ARRAY_SIZE && !o.all_stolen_copy {
                self.grow_shared();
            }
            self.o().head += 1;
            return std::ptr::null_mut();
        }
        self.task(o.head)
    }

    /// The second part of push, once the slot holds the task
    pub fn push_publish(&self) {
        let o = self.o();
        self.tasks[o.head as usize].stealer.store(0, Relaxed);
        o.head += 1;
        if o.all_stolen_copy {
            self.stealer.0.ts.store(make_tail_split(o.head - 1, o.head), Release);
            self.stealer.0.all_stolen.store(false, Relaxed);
            o.split_copy = o.head;
            o.all_stolen_copy = false;
            if self.split_request.0.load(Relaxed) {
                self.split_request.0.store(false, Relaxed);
            }
            let have_jobs = self.bunk().have_jobs.0.fetch_add(1, Release);
            if have_jobs < o.num_workers - 1 {
                self.bunk().publish_work(self);
            }
        } else {
            self.grow_shared();
        }
    }

    pub fn pop(&self) -> (Status, *mut Task) {
        let o = self.o();
        if o.head == 0 {
            return (Status::Empty, std::ptr::null_mut());
        }
        if o.head > TASK_ARRAY_SIZE {
            o.head -= 1;
            return (Status::Overflown, std::ptr::null_mut());
        }
        if o.all_stolen_copy {
            return (Status::Stolen, self.task(o.head - 1));
        }
        if o.split_copy == o.head && self.shrink_shared() {
            return (Status::Stolen, self.task(self.o().head - 1));
        }
        let o = self.o();
        o.head -= 1;
        if o.head == 0 {
            if !o.all_stolen_copy {
                o.all_stolen_copy = true;
                self.stealer.0.all_stolen.store(true, Relaxed);
                self.bunk().have_jobs.0.fetch_add(-1, Release);
            }
        } else if o.head != o.split_copy {
            self.grow_shared();
        }
        (Status::Work, self.task(self.o().head))
    }

    fn pop_stolen(&self) {
        let o = self.o();
        o.head -= 1;
        if !o.all_stolen_copy {
            o.all_stolen_copy = true;
            self.stealer.0.all_stolen.store(true, Relaxed);
            self.bunk().have_jobs.0.fetch_add(-1, Release);
        }
    }

    fn steal(&self) -> *mut Task {
        if self.stealer.0.all_stolen.load(Relaxed) {
            return std::ptr::null_mut();
        }
        let ts = self.stealer.0.ts.load(Relaxed);
        let mut t = tail(ts);
        let s = split(ts);
        if t < s {
            match self.stealer.0.ts.compare_exchange_weak(ts, make_tail_split(t + 1, s), Acquire, Relaxed) {
                Ok(_) => return self.task(t),
                Err(ts) => {
                    t = tail(ts);
                    if t < split(ts) {
                        return std::ptr::null_mut();
                    }
                }
            }
        }
        if t < TASK_ARRAY_SIZE && !self.split_request.0.load(Relaxed) {
            self.split_request.0.store(true, Relaxed);
        }
        std::ptr::null_mut()
    }

    fn steal_with_retry_loop(&self) -> *mut Task {
        if self.stealer.0.all_stolen.load(Relaxed) {
            return std::ptr::null_mut();
        }
        let mut ts = self.stealer.0.ts.load(Relaxed);
        let mut t = tail(ts);
        let mut s = split(ts);
        while t < s {
            match self.stealer.0.ts.compare_exchange_weak(ts, make_tail_split(t + 1, s), Acquire, Relaxed) {
                Ok(_) => return self.task(t),
                Err(cur) => {
                    ts = cur;
                    t = tail(ts);
                    s = split(ts);
                }
            }
        }
        if t < TASK_ARRAY_SIZE && !self.split_request.0.load(Relaxed) {
            self.split_request.0.store(true, Relaxed);
        }
        std::ptr::null_mut()
    }

    fn self_steal_and_get_tail(&self) -> u32 {
        let o = self.o();
        if o.all_stolen_copy {
            return o.split_copy;
        }
        // the deque is not all stolen, so tail < split: take the tail, or
        // undo the increment if that raced with all being stolen
        let t = tail(self.stealer.0.ts.fetch_add(make_tail_split(1, 0), Relaxed));
        if t == o.split_copy {
            self.stealer.0.ts.store(make_tail_split(t, o.split_copy), Relaxed);
        }
        t
    }

    fn random_steal(&self) -> *mut Task {
        let o = self.o();
        let mut next = o.randgen.integer_below(o.num_workers - 1);
        next += (next >= o.owner_id) as i32;
        debug_assert!(next != o.owner_id && next >= 0 && next < o.num_workers);
        self.worker(next).steal()
    }

    fn inject_task_and_notify(&self, t: *mut Task) {
        self.stealer.0.injected_task.store(t, Relaxed);
        self.stealer.0.semaphore.release();
    }

    fn notify(&self) {
        self.stealer.0.semaphore.release();
    }

    /// runStolenTask: true where the C++ throws HighsTask::Interrupt (the
    /// end's checkInterrupt)
    fn run_stolen_task(&self, task: *mut Task) -> bool {
        let prev_root_task = self.o().root_task;
        self.o().root_task = task;
        let current_head = self.o().head;
        // SAFETY: a slot of a live deque, stolen by this thread
        let tk = unsafe { &*task };
        let (owner, interrupted) = tk.run_stolen(self);
        if !interrupted {
            if !owner.is_null() {
                // SAFETY: a deque of the executor
                unsafe { (*owner).notify() };
            }
        } else {
            // the interrupted task's own tasks: cancelled, and waited for
            // if stolen
            for i in current_head..self.o().head {
                self.tasks[i as usize].cancel();
            }
            while self.o().head != current_head {
                let (status, t) = self.pop();
                debug_assert!(status != Status::Empty);
                if status != Status::Stolen {
                    continue;
                }
                // SAFETY: a slot of this deque
                let tt = unsafe { &*t };
                let (stealer, _) = tt.get_stealer_if_unfinished();
                if stealer.is_null() {
                    self.pop_stolen();
                    continue;
                }
                let mut num_tries = NUM_TRY_FAC;
                let t_start = Instant::now();
                let mut is_finished = tt.is_finished();
                while !is_finished {
                    for _ in 0..num_tries {
                        yield_processor();
                        is_finished = tt.is_finished();
                        if is_finished {
                            break;
                        }
                    }
                    if !is_finished {
                        if micros_since(t_start) < MICRO_SECS_BEFORE_SLEEP {
                            num_tries *= 2;
                        } else {
                            self.wait_for_task_to_finish(tt, stealer);
                            break;
                        }
                    }
                }
                self.pop_stolen();
            }
            let owner = tk.mark_as_finished(self);
            if !owner.is_null() {
                // SAFETY: a deque of the executor
                unsafe { (*owner).notify() };
            }
        }
        self.o().root_task = prev_root_task;
        self.check_interrupt()
    }

    /// leapfrogStolenTask: (finished, the stealer, interrupted)
    fn leapfrog_stolen_task(&self, task: &Task) -> (bool, *const Deque, bool) {
        let (stealer, cancelled) = task.get_stealer_if_unfinished();
        if stealer.is_null() {
            return (true, stealer, false);
        }
        if !cancelled {
            loop {
                // SAFETY: a deque of the executor
                let t = unsafe { (*stealer).steal_with_retry_loop() };
                if t.is_null() {
                    break;
                }
                if self.run_stolen_task(t) {
                    return (false, stealer, true);
                }
                if task.is_finished() {
                    break;
                }
            }
        }
        (task.is_finished(), stealer, false)
    }

    fn wait_for_task_to_finish(&self, t: &Task, stealer: *const Deque) {
        let lg = self.stealer.0.semaphore.lock_mutex_for_acquire();
        if !t.request_notify_when_finished(self, stealer) {
            return;
        }
        self.stealer.0.semaphore.acquire_locked(lg, Relaxed);
    }

    /// yield: true where the C++ throws HighsTask::Interrupt
    fn yield_(&self) -> bool {
        let t = self.random_steal();
        !t.is_null() && self.run_stolen_task(t)
    }

    pub fn owner_id(&self) -> i32 {
        self.o().owner_id
    }
    pub fn num_workers(&self) -> i32 {
        self.o().num_workers
    }
    pub fn current_head(&self) -> i32 {
        self.o().head as i32
    }

    /// HighsTaskExecutor::sync_stolen_task: true where the C++ throws
    /// HighsTask::Interrupt
    pub fn sync_stolen_task(&self, stolen_task: *mut Task) -> bool {
        // SAFETY: a slot of this deque
        let st = unsafe { &*stolen_task };
        let (finished, stealer, interrupted) = self.leapfrog_stolen_task(st);
        if interrupted {
            return true;
        }
        if !finished {
            let num_workers = self.num_workers();
            let mut num_tries = NUM_TRY_FAC * (num_workers - 1);
            let t_start = Instant::now();
            loop {
                for _ in 0..num_tries {
                    if st.is_finished() {
                        self.pop_stolen();
                        return false;
                    }
                    if self.yield_() {
                        return true;
                    }
                }
                if micros_since(t_start) < MICRO_SECS_BEFORE_SLEEP {
                    num_tries *= 2;
                } else {
                    break;
                }
            }
            if !st.is_finished() {
                self.wait_for_task_to_finish(st, stealer);
            }
        }
        self.pop_stolen();
        false
    }
}

// ---- the executor ----

/// HighsTaskExecutor: the deques and the worker threads
pub struct Executor {
    has_stopped: AtomicBool,
    worker_bunk: Box<WorkerBunk>,
    worker_deques: Vec<Box<Deque>>,
    worker_threads: Mutex<Vec<JoinHandle<()>>>,
}

/// ExecutorHandle: the thread's reference to its executor
struct Handle {
    ptr: Option<Arc<Executor>>,
    is_main: bool,
}

impl Handle {
    fn dispose(&mut self) {
        let Some(ptr) = self.ptr.clone() else {
            return;
        };
        if self.is_main {
            ptr.stop_worker_threads(false, self);
        }
        self.ptr = None;
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.dispose();
    }
}

thread_local! {
    static WORKER_DEQUE: Cell<*mut Deque> = const { Cell::new(std::ptr::null_mut()) };
    static EXECUTOR_HANDLE: RefCell<Handle> = const { RefCell::new(Handle { ptr: None, is_main: false }) };
}

impl Executor {
    fn new(num_threads: i32) -> Arc<Executor> {
        debug_assert!(num_threads > 0);
        let worker_bunk = Box::new(WorkerBunk { have_jobs: Padded(AtomicI32::new(0)), sleeper_stack: Padded(AtomicU64::new(0)) });
        let mut worker_deques: Vec<Box<Deque>> = Vec::with_capacity(num_threads as usize);
        let workers_ptr = worker_deques.as_ptr();
        for i in 0..num_threads {
            worker_deques.push(Deque::new_boxed(&*worker_bunk, workers_ptr, i, num_threads));
        }
        debug_assert!(worker_deques.as_ptr() == workers_ptr);
        let ex = Arc::new(Executor {
            has_stopped: AtomicBool::new(false),
            worker_bunk,
            worker_deques,
            worker_threads: Mutex::new(Vec::new()),
        });
        WORKER_DEQUE.with(|d| d.set(&*ex.worker_deques[0] as *const Deque as *mut Deque));
        let mut threads = ex.worker_threads.lock().unwrap();
        for i in 1..num_threads {
            let ex2 = Arc::clone(&ex);
            threads.push(std::thread::spawn(move || run_worker(i, ex2)));
        }
        drop(threads);
        ex
    }

    fn random_steal_loop(&self, local: &Deque) -> *mut Task {
        let num_workers = self.worker_deques.len() as i32;
        let mut num_tries = 16 * (num_workers - 1);
        let t_start = Instant::now();
        loop {
            for _ in 0..num_tries {
                let task = local.random_steal();
                if !task.is_null() {
                    return task;
                }
            }
            if self.worker_bunk.have_jobs.0.load(Relaxed) == 0 {
                break;
            }
            if micros_since(t_start) < MICRO_SECS_BEFORE_GLOBAL_SYNC {
                num_tries *= 2;
            } else {
                break;
            }
        }
        std::ptr::null_mut()
    }

    /// stopWorkerThreads (`handle` is the calling thread's)
    fn stop_worker_threads(&self, blocking: bool, handle: &Handle) {
        if handle.ptr.is_none() || self.has_stopped.swap(true, SeqCst) {
            return;
        }
        for d in &self.worker_deques {
            d.inject_task_and_notify(std::ptr::null_mut());
        }
        let threads = std::mem::take(&mut *self.worker_threads.lock().unwrap());
        if blocking && handle.is_main {
            for t in threads {
                let _ = t.join();
            }
        }
        // otherwise the threads are detached (their handles dropped)
    }
}

fn run_worker(worker_id: i32, ex: Arc<Executor>) {
    EXECUTOR_HANDLE.with(|h| h.borrow_mut().ptr = Some(Arc::clone(&ex)));
    if !ex.has_stopped.load(SeqCst) {
        let local: &Deque = &ex.worker_deques[worker_id as usize];
        WORKER_DEQUE.with(|d| d.set(local as *const Deque as *mut Deque));
        let mut current = ex.worker_bunk.wait_for_new_task(local);
        while !current.is_null() {
            local.run_stolen_task(current);
            current = ex.random_steal_loop(local);
            if !current.is_null() {
                continue;
            }
            current = ex.worker_bunk.wait_for_new_task(local);
        }
    }
    drop(ex);
    EXECUTOR_HANDLE.with(|h| h.borrow_mut().dispose());
}

/// HighsTaskExecutor::initialize: an executor with `num_threads` workers
/// (the calling thread is worker 0) unless the thread has one; `run` runs a
/// stolen task
pub fn initialize(num_threads: i32, run: RunFn) {
    RUN_FN.store(run as usize, Relaxed);
    EXECUTOR_HANDLE.with(|h| {
        let mut h = h.borrow_mut();
        if h.ptr.is_none() {
            h.is_main = true;
            h.ptr = Some(Executor::new(num_threads));
        }
    });
}

/// initialize_scheduler(num_threads) on a thread Rust made (the helper
/// of the IPX race), with the run function the C++ registered (nothing if
/// there is none: no task is run then)
pub fn initialize_thread(num_threads: i32) {
    let f = RUN_FN.load(Relaxed);
    if f != 0 {
        // SAFETY: stored from a RunFn by initialize
        initialize(num_threads, unsafe { std::mem::transmute::<usize, RunFn>(f) });
    }
}

/// HighsTaskExecutor::shutdown
pub fn shutdown(blocking: bool) {
    EXECUTOR_HANDLE.with(|h| {
        let mut h = h.borrow_mut();
        if let Some(ptr) = h.ptr.clone() {
            ptr.stop_worker_threads(blocking, &h);
            h.dispose();
        }
    });
}

/// HighsTaskExecutor::getThisWorkerDeque (null if none)
pub fn this_worker_deque() -> *mut Deque {
    WORKER_DEQUE.with(|d| d.get())
}

// ---- C interface (highs/parallel/*.h under HIGHS_RUST) ----

pub mod ffi {
    use super::*;

    /// # Safety
    /// `run` runs the callable of a task slot
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_sched_initialize(num_threads: i32, run: RunFn) {
        initialize(num_threads, run)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_sched_shutdown(blocking: bool) {
        shutdown(blocking)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_sched_this_deque() -> *mut Deque {
        this_worker_deque()
    }

    /// # Safety
    /// `d` the calling thread's deque (for all the deque functions)
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_push_slot(d: *mut Deque) -> *mut Task {
        (*d).push_slot()
    }

    /// # Safety
    /// as push_slot, after the slot was filled
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_push_publish(d: *mut Deque) {
        (*d).push_publish()
    }

    /// # Safety
    /// as push_slot
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_pop(d: *mut Deque, task: *mut *mut Task) -> i32 {
        let (s, t) = (*d).pop();
        *task = t;
        s as i32
    }

    /// # Safety
    /// as push_slot; `t` the popped stolen task
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_sync_stolen(d: *mut Deque, t: *mut Task) -> bool {
        (*d).sync_stolen_task(t)
    }

    /// # Safety
    /// as push_slot
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_check_interrupt(d: *mut Deque) -> bool {
        (*d).check_interrupt()
    }

    /// # Safety
    /// as push_slot
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_cancel_task(d: *mut Deque, i: i32) {
        (*d).cancel_task(i)
    }

    /// # Safety
    /// as push_slot
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_set_root_task(d: *mut Deque, t: *mut Task) -> *mut Task {
        (*d).set_root_task(t)
    }

    /// # Safety
    /// `d` a deque; which: 0 owner id, 1 number of workers, 2 current head
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_deque_info(d: *const Deque, which: i32) -> i32 {
        match which {
            0 => (*d).owner_id(),
            1 => (*d).num_workers(),
            _ => (*d).current_head(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;

    // A Rust "callable" for the tests: the slot's first word is a function
    // pointer, the second its argument
    unsafe extern "C" fn run_test_task(t: *mut Task) -> bool {
        let d = &*(*t).data.get();
        let f: fn(usize) = std::mem::transmute::<u64, fn(usize)>(d[0]);
        f(d[1] as usize);
        false
    }

    fn spawn(d: &Deque, f: fn(usize), arg: usize) {
        let t = d.push_slot();
        if t.is_null() {
            f(arg);
            return;
        }
        // SAFETY: the slot reserved for this owner
        unsafe {
            let data = &mut *(*t).data.get();
            data[0] = f as usize as u64;
            data[1] = arg as u64;
        }
        d.push_publish();
    }

    fn sync(d: &Deque) {
        let (s, t) = d.pop();
        match s {
            Status::Empty => panic!("sync without a task"),
            Status::Overflown => {}
            Status::Stolen => assert!(!d.sync_stolen_task(t)),
            // SAFETY: the owner's own task
            Status::Work => unsafe {
                if (*t).stealer.load(Relaxed) == 0 {
                    run_test_task(t);
                }
            },
        }
    }

    static SUM: AtomicI64 = AtomicI64::new(0);

    fn fib_task(n: usize) {
        if n < 2 {
            SUM.fetch_add(n as i64, Relaxed);
            return;
        }
        let d = this_worker_deque();
        // SAFETY: the calling worker's deque
        let d = unsafe { &*d };
        spawn(d, fib_task, n - 1);
        fib_task(n - 2);
        sync(d);
    }

    #[test]
    fn fib_on_four_threads() {
        std::thread::spawn(|| {
            initialize(4, run_test_task);
            assert_eq!(this_worker_deque().is_null(), false);
            // SAFETY: the main worker's deque
            assert_eq!(unsafe { (*this_worker_deque()).num_workers() }, 4);
            for _ in 0..3 {
                SUM.store(0, Relaxed);
                fib_task(25);
                assert_eq!(SUM.load(Relaxed), 75025);
            }
            shutdown(true);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn layout() {
        assert_eq!(std::mem::size_of::<Task>(), 64);
        assert_eq!(std::mem::offset_of!(Task, stealer), 56);
        assert_eq!(std::mem::align_of::<Deque>(), 64);
    }
}
