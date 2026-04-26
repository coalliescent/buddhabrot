//! Background sampler: uniform warmup + Metropolis-Hastings chain.
//!
//! The worker owns the iteration; the main thread only snapshots the View
//! under a read lock and reads atomics for presentation.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use parking_lot::RwLock;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::orbit;
use crate::view::View;

/// Atomic histogram. One `u32` counter per pixel; scatter via fetch_add.
pub struct Histogram {
    pub data: Vec<AtomicU32>,
    pub width: u32,
    pub height: u32,
}

impl Histogram {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(n);
        for _ in 0..n { data.push(AtomicU32::new(0)); }
        Self { data, width, height }
    }

    pub fn clear(&self) {
        for c in &self.data { c.store(0, Ordering::Relaxed); }
    }
}

/// How many early iterates to drop from every scatter. `z_0 = 0` and
/// `z_1 = c`, so the first few iterates track `c` closely. Dropping them
/// kills the "c-sampling shadow" (uniform rectangle of brightness in the
/// c-plane where the uniform-warmup sampler draws from). Applied
/// unconditionally to live, composer preview, and video renders since
/// they all share `worker_loop`.
pub const SKIP_EARLY_N: usize = 2;

pub struct SamplerHandle {
    pub hist: Arc<RwLock<Arc<Histogram>>>,
    pub view: Arc<RwLock<View>>,
    pub running: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
    pub samples: Arc<AtomicU64>,
    pub max_count: Arc<AtomicU32>,
    /// Auto-pause threshold. When `samples` crosses this, workers flip
    /// `paused=true` and disarm by setting this to `u64::MAX` so the user
    /// can manually resume without being re-paused immediately. Re-armed on
    /// `invalidate` (new view = fresh budget).
    pub auto_pause_at: Arc<AtomicU64>,
    pub n_workers: usize,
    threads: Vec<JoinHandle<()>>,
}

/// Default UI auto-pause budget: ~10B proposals. At 1024×768 that's
/// ~12.7k proposals/pixel — enough for a well-formed preview.
pub const UI_AUTO_PAUSE_SAMPLES: u64 = 10_000_000_000;

impl SamplerHandle {
    /// Spawn `available_parallelism() - 1` workers (min 1). Leaves one core
    /// for the event loop and present.
    pub fn spawn(view: Arc<RwLock<View>>) -> Self {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .saturating_sub(1)
            .max(1);
        Self::spawn_n(view, n)
    }

    pub fn spawn_n(view: Arc<RwLock<View>>, n_workers: usize) -> Self {
        let (w, h) = { let v = view.read(); (v.width, v.height) };
        let hist = Arc::new(RwLock::new(Arc::new(Histogram::new(w, h))));
        let running = Arc::new(AtomicBool::new(true));
        let paused = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(AtomicU64::new(0));
        let max_count = Arc::new(AtomicU32::new(1));
        let auto_pause_at = Arc::new(AtomicU64::new(UI_AUTO_PAUSE_SAMPLES));

        let n_workers = n_workers.max(1);
        let mut threads = Vec::with_capacity(n_workers);
        for tid in 0..n_workers {
            let hist_c = hist.clone();
            let view_c = view.clone();
            let running_c = running.clone();
            let paused_c = paused.clone();
            let samples_c = samples.clone();
            let max_count_c = max_count.clone();
            let auto_pause_c = auto_pause_at.clone();
            threads.push(thread::spawn(move || {
                worker_loop(tid, hist_c, view_c, running_c, Some(paused_c),
                            samples_c, max_count_c, None, Some(auto_pause_c));
            }));
        }

        Self {
            hist,
            view,
            running,
            paused,
            samples,
            max_count,
            auto_pause_at,
            n_workers,
            threads,
        }
    }

    /// Reset generation / histogram for view changes. Workers observe the
    /// generation atomically and start over.
    pub fn invalidate(&self, reason: InvalidationReason) {
        let is_resize = matches!(reason, InvalidationReason::Resize { .. });
        if is_resize {
            // Resize: allocate a new histogram of the right size.
            let (w, h) = {
                let v = self.view.read();
                (v.width, v.height)
            };
            let new_hist = Arc::new(Histogram::new(w, h));
            *self.hist.write() = new_hist;
        } else {
            // Same dimensions, just zero.
            let h = self.hist.read().clone();
            h.clear();
        }
        self.samples.store(0, Ordering::Relaxed);
        self.max_count.store(1, Ordering::Relaxed);
        self.auto_pause_at.store(UI_AUTO_PAUSE_SAMPLES, Ordering::Relaxed);
        // Unpause only on camera-driven invalidations. A window resize is
        // not a user signal to start rendering, and must not override a
        // composer-mode auto-pause or a manual pause.
        if !is_resize {
            self.paused.store(false, Ordering::Relaxed);
        }
    }

    pub fn hist_snapshot(&self) -> Arc<Histogram> {
        self.hist.read().clone()
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) { self.stop(); }
}

pub enum InvalidationReason {
    ViewChange,
    Resize { _w: u32, _h: u32 },
}

const BATCH_UNIFORM: usize = 128;
const MH_STEPS: usize = 1024;
const MAX_ORBIT: usize = 8192;

/// Runs a sampler on the calling thread.
///
/// If `target_samples` is `Some(n)`, the loop exits once `samples >= n`; this
/// is the mode used by the standalone renderer. Otherwise the loop runs until
/// `running` is cleared (pool mode used by `SamplerHandle`).
///
/// If `paused` is `Some`, the worker idles (sleeps 50ms per tick) whenever
/// that flag is set. Lets the UI free CPU for a background render.
///
/// If `auto_pause_at` is `Some` (requires `paused` also `Some`), the worker
/// sets `paused=true` when `samples` crosses the stored threshold, then
/// disarms by swapping the threshold to `u64::MAX`. The UI re-arms on view
/// invalidation.
pub fn worker_loop(
    _tid: usize,
    hist: Arc<RwLock<Arc<Histogram>>>,
    view: Arc<RwLock<View>>,
    running: Arc<AtomicBool>,
    paused: Option<Arc<AtomicBool>>,
    samples: Arc<AtomicU64>,
    max_count: Arc<AtomicU32>,
    target_samples: Option<u64>,
    auto_pause_at: Option<Arc<AtomicU64>>,
) {
    let mut rng = SmallRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut orbit_buf: Vec<[f64; 2]> = vec![[0.0; 2]; MAX_ORBIT];
    let mut pixel_buf: Vec<u32> = vec![0u32; MAX_ORBIT];

    // MH chain state.
    let mut chain_cr: f64 = 0.0;
    let mut chain_ci: f64 = 0.0;
    let mut chain_contrib: u32 = 0;
    let mut chain_valid: bool = false;

    let reached = |s: &AtomicU64| -> bool {
        target_samples.map_or(false, |t| s.load(Ordering::Relaxed) >= t)
    };

    while running.load(Ordering::Relaxed) && !reached(&samples) {
        // Respect pause: idle the thread so the OS schedules other work.
        if paused.as_ref().map_or(false, |p| p.load(Ordering::Relaxed)) {
            thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }
        // Auto-pause: one worker flips `paused` when samples crosses the
        // threshold; disarming (swap to MAX) guarantees a single trigger
        // per view. Other workers see `paused` on the next iteration.
        if let (Some(p), Some(ap)) = (paused.as_ref(), auto_pause_at.as_ref()) {
            let t = ap.load(Ordering::Relaxed);
            if t != u64::MAX && samples.load(Ordering::Relaxed) >= t {
                ap.store(u64::MAX, Ordering::Relaxed);
                p.store(true, Ordering::Relaxed);
                continue;
            }
        }
        // Snapshot view.
        let (gen, max_iter, width, height, sigma) = {
            let v = view.read();
            // Proposal std-dev: scale with pixel pitch so MH stays effective
            // when zoomed in.
            let pixel_pitch = 2.0 * v.half_width / v.width as f64;
            let sigma = pixel_pitch * 2.0;
            (v.generation, v.max_iter, v.width, v.height, sigma)
        };
        // Guard: clamp orbit length.
        let max_iter = (max_iter as usize).min(MAX_ORBIT) as u32;

        let h = hist.read().clone();
        // Handle a resize race: if the histogram's dims don't match the view's,
        // skip this batch (main thread will reallocate promptly).
        if h.width != width || h.height != height {
            thread::yield_now();
            continue;
        }

        // Phase 1: uniform warmup batch.
        for _ in 0..BATCH_UNIFORM {
            if view_changed(&view, gen) { break; }
            let cr = rng.gen_range(-2.2..1.2);
            let ci = rng.gen_range(-1.5..1.5);
            if orbit::in_main_bulb(cr, ci) { continue; }

            let (esc, len) = orbit::iterate(cr, ci, max_iter, &mut orbit_buf);
            if !esc || len == 0 { continue; }
            let start = SKIP_EARLY_N.min(len);

            let snap = view.read();
            if snap.generation != gen { break; }
            let k = orbit::pixel_contributions(
                &snap, cr, ci, &orbit_buf[start..len], &mut pixel_buf);
            drop(snap);

            if k > 0 {
                scatter(&h, &pixel_buf[..k as usize], &max_count);
                // Seed the MH chain with the best uniform find so far.
                if k > chain_contrib || !chain_valid {
                    chain_cr = cr;
                    chain_ci = ci;
                    chain_contrib = k;
                    chain_valid = true;
                }
            }
            samples.fetch_add(1, Ordering::Relaxed);
        }

        if view_changed(&view, gen) { continue; }

        // Phase 2: MH chain — local mutations around `chain_*`.
        if !chain_valid {
            // Nothing contributes yet; loop back to uniform.
            continue;
        }

        let mut accept_streak = 0u32;
        let mut last_acceptance = 0u32;
        let mut proposals = 0u32;
        for step in 0..MH_STEPS {
            if step & 63 == 0 && (view_changed(&view, gen) || reached(&samples)) { break; }

            let dcr = normal.sample(&mut rng) * sigma;
            let dci = normal.sample(&mut rng) * sigma;
            let cr = chain_cr + dcr;
            let ci = chain_ci + dci;
            if orbit::in_main_bulb(cr, ci) {
                proposals += 1;
                continue;
            }

            let (esc, len) = orbit::iterate(cr, ci, max_iter, &mut orbit_buf);
            proposals += 1;
            if !esc || len == 0 { continue; }
            let start = SKIP_EARLY_N.min(len);

            let snap = view.read();
            if snap.generation != gen { break; }
            let k = orbit::pixel_contributions(
                &snap, cr, ci, &orbit_buf[start..len], &mut pixel_buf);
            drop(snap);

            // Acceptance: min(1, k / chain_contrib). If chain_contrib is 0 we
            // always accept (escape valley).
            let accept = if chain_contrib == 0 {
                true
            } else {
                let p = (k as f64) / (chain_contrib as f64);
                p >= 1.0 || rng.gen::<f64>() < p
            };

            if accept {
                if k > 0 {
                    scatter(&h, &pixel_buf[..k as usize], &max_count);
                }
                chain_cr = cr;
                chain_ci = ci;
                chain_contrib = k;
                accept_streak += 1;
                last_acceptance += 1;
            }
            samples.fetch_add(1, Ordering::Relaxed);

            // Periodic escape: re-seed from uniform if the chain gets stuck.
            if step % 256 == 255 {
                if last_acceptance < 8 {
                    // Force a re-seed on the next iteration by invalidating.
                    chain_valid = false;
                    break;
                }
                last_acceptance = 0;
            }
            let _ = accept_streak;
        }
        let _ = proposals;
    }
}

#[inline]
fn view_changed(view: &RwLock<View>, expected: u64) -> bool {
    view.read().generation != expected
}

fn scatter(hist: &Histogram, pixels: &[u32], max_count: &AtomicU32) {
    let mut local_max = 0u32;
    for &idx in pixels {
        let prev = hist.data[idx as usize].fetch_add(1, Ordering::Relaxed);
        let v = prev + 1;
        if v > local_max { local_max = v; }
    }
    if local_max > max_count.load(Ordering::Relaxed) {
        // Race-safe: fetch_max is monotonic, so concurrent workers converge
        // to the actual peak.
        let _ = max_count.fetch_max(local_max, Ordering::Relaxed);
    }
}
