//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// (`VICTORY_BVH_PREP=1`) Title allocations whose extent has been
    /// mapped whole, by their first byte.
    allocations: std::collections::BTreeMap<u64, (u64, u64, u64)>,
    /// (`VICTORY_BVH_PREP=1`) Per host list (by address): how many copies
    /// were numbered when it last brought them up to date.
    seen: HashMap<usize, usize>,
    /// Lost reads and lost stores (pages, as noted) since the last log,
    /// the first page and the frames counted.
    missed: (u64, u64, u64, u64),
    /// Bytes of the copies the prep numbered.
    prepped: u64,
}

/// `VICTORY_BVH_PREP=1`: before a dispatch that reads title memory by
/// address (the ray-tracing structure builder, the traces):
/// - a named address or a faulted page gets the copies of the whole title
///   allocation around it ([`allocation`], at most `VICTORY_BVH_PREP_MB`,
///   default 256, centred on the address; all made ready by it together at
///   most `VICTORY_BVH_PREP_TOTAL_MB`, default 1024), not only its 4 MiB: what a stage
///   reaches through pointers it reads out of memory is mostly in the same
///   allocation as what it was given;
/// - every copy numbered since the list last brought them up to date
///   (by another list too) is brought up to date before this dispatch,
///   and, for a dispatch that writes, declared written (so its bytes go
///   back to title memory after the list, where the title's CPU reads the
///   nodes).
fn prep() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_PREP").is_some_and(|v| v == "1"))
}

fn watch() -> &'static [u64] {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_BVH_WATCH")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                u64::from_str_radix(s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s), 16).ok()
            })
            .collect()
    })
}

/// `VICTORY_BVH_PREP_TOTAL_MB` (default 1024): bytes of copies the prep
/// may number in all. (Unbounded, a run took 29.7 GB of private memory in
/// the menus.)
fn prep_total() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_PREP_TOTAL_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1024) << 20
    })
}

fn prep_bytes() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_PREP_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The title allocation holding `address`: its first byte and the end of
/// the run of pages in the address's state (one `VirtualQuery`; it walks
/// the page tables, so it is asked once per allocation).
fn allocation(address: u64) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory as mem;
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(address as *const std::ffi::c_void, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != mem::MEM_COMMIT {
        return None;
    }
    let base = info.AllocationBase as u64;
    let end = info.BaseAddress as u64 + info.RegionSize as u64;
    Some((base, end))
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            allocations: Default::default(),
            seen: HashMap::new(),
            missed: (0, 0, u64::MAX, 0),
            prepped: 0,
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                shadow.by_address.store(true, std::sync::atomic::Ordering::Relaxed);
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                let store = page & victory_shader::emit::FAULT_STORE != 0;
                let page = page & !victory_shader::emit::FAULT_STORE;
                if store {
                    self.missed.1 += 1;
                } else {
                    self.missed.0 += 1;
                }
                self.missed.2 = self.missed.2.min(u64::from(page) * PAGE);
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        // Every 64 frames: how much the shaders that read title memory by
        // address lost (a page counts once per frame, whatever the lanes).
        self.missed.3 += 1;
        if self.missed.3 >= 64 {
            let (reads, stores, first, frames) = self.missed;
            if reads + stores > 0 {
                tracing::warn!(
                    "bvh raw: {reads} reads/{stores} stores missed (pages without a copy, over {frames} frames), first at {first:#x}"
                );
            }
            let back = crate::shadow::BY_ADDRESS_BACK.swap(0, std::sync::atomic::Ordering::Relaxed);
            let in_place = self.copies.iter().filter(|c| c.in_place()).count();
            if !self.copies.is_empty() {
                tracing::info!(
                    "bvh raw: {} copies numbered ({in_place} in place, written straight to title memory); {:.1} KB/frame copied back to title memory from the others",
                    self.copies.len(),
                    back as f64 / frames as f64 / 1e3
                );
            }
            // VICTORY_BVH_WATCH=0x<va>[,...]: whether the page of each
            // address (where the title's CPU reads nodes) has a copy.
            for &va in watch() {
                let page = va / PAGE;
                let n = if page < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    unsafe { self.directory.1.add(page as usize * 2).read() }
                } else {
                    0
                };
                let what = match n {
                    0 => "NO copy (shaders read 0 and drop stores there)".to_owned(),
                    n => match self.copies.get(n as usize - 1) {
                        Some(c) if c.in_place() => format!("copy {n}, in place"),
                        Some(c) => format!("copy {n} at {:#x}+{:#x}, copied back", c.span().0, c.span().1),
                        None => format!("copy {n}?"),
                    },
                };
                tracing::info!("bvh watch {va:#x}: {what}");
            }
            self.missed = (0, 0, u64::MAX, 0);
        }
        for page in pages {
            let at = u64::from(page) * PAGE;
            self.map(at);
            if prep() {
                self.map_allocation(at);
            }
        }
    }

    /// (`VICTORY_BVH_PREP=1`) Copies for the title allocation around
    /// `address`: all of it when it is at most `VICTORY_BVH_PREP_MB`
    /// (default 256), else that much centred on the address. Each
    /// allocation is asked of Windows once and remembered with the span
    /// already made ready: an address inside it costs a map lookup, and
    /// only cells outside the span are mapped. (It was the cell of the
    /// address and on only: the builder's pointers into the title's big
    /// block lay 400 MB below the address it was given, and each new
    /// address asked again.)
    fn map_allocation(&mut self, address: u64) {
        let known = self
            .allocations
            .range(..=address)
            .next_back()
            .filter(|(_, a)| address < a.0)
            .map(|(&base, &a)| (base, a));
        let (base, (end, lo, hi)) = match known {
            Some(k) => k,
            None => {
                let Some((base, end)) = allocation(address) else { return };
                (base, (end, address, address))
            }
        };
        let half = prep_bytes() / 2;
        let (want_lo, want_hi) = if end - base <= 2 * half {
            (base, end)
        } else {
            let lo = address.saturating_sub(half).max(base);
            (lo, (lo + 2 * half).min(end))
        };
        let want_lo = want_lo & !(CELL - 1);
        let covered = lo < hi;
        if covered && want_lo >= lo && want_hi <= hi {
            return;
        }
        let before = self.copies.len();
        let mut at = want_lo;
        while at < want_hi {
            if self.prepped >= prep_total() {
                static FULL: std::sync::Once = std::sync::Once::new();
                FULL.call_once(|| {
                    tracing::warn!(
                        mb = self.prepped >> 20,
                        "bvh prep: VICTORY_BVH_PREP_TOTAL_MB reached; further allocations are not made ready ahead (faulted pages still get copies)"
                    )
                });
                break;
            }
            if !(covered && at >= lo && at + CELL <= hi) {
                let n = self.copies.len();
                self.map(at.max(base));
                self.prepped += self.copies[n..].iter().map(|c| c.span().1).sum::<u64>();
            }
            at += CELL;
        }
        let want_hi = want_hi.min(at);
        let (lo, hi) = if covered { (lo.min(want_lo), hi.max(want_hi)) } else { (want_lo, want_hi) };
        self.allocations.insert(base, (end, lo, hi));
        let made = self.copies.len() - before;
        if made > 0 {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 64 {
                tracing::info!(
                    "bvh prep: allocation at {base:#x} ({:.1} MB) ready from {lo:#x} to {hi:#x}, {made} copies more ({} numbered)",
                    (end - base) as f64 / 1e6,
                    self.copies.len()
                );
            }
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
        if prep() && !self.refused.contains(&(address / PAGE)) {
            self.map_allocation(address);
        }
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    let numbered = m.copies.len();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    // With VICTORY_BVH_PREP=1, also the copies numbered (by any list) since
    // this list last brought them up to date.
    let list_key = std::ptr::from_ref(list) as usize;
    let since = if prep() && !first_read {
        m.seen.get(&list_key).copied().unwrap_or(numbered).min(numbered)
    } else {
        numbered
    };
    for (n, shadow) in m.copies.iter().enumerate() {
        let (start, len) = shadow.span();
        if first_read || n >= since {
            shadow.refresh(list, start, len);
        }
        if writes && (first_write || n >= since) {
            shadow.gpu_writes(list, start, len);
        }
    }
    if prep() {
        if m.seen.len() > 4096 {
            m.seen.clear();
        }
        m.seen.insert(list_key, m.copies.len());
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}
