//! Standalone frame renderer used by the CLI (`render-frame`, `render-video`).
//!
//! `render_frame` owns the sampler for the duration of a single frame: spawns
//! N threads (or runs inline when N == 1), accumulates up to `samples_target`
//! samples, then palette-applies to RGBA. No event loop, no UI.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use parking_lot::RwLock;

use crate::palette::{self, Palette};
use crate::sampler::{self, Histogram};
use crate::view::View;

#[derive(Clone)]
pub struct FrameSpec {
    pub view: View,
    pub palette: Palette,
    pub gamma: f32,
    pub samples_target: u64,
    pub n_workers: usize, // 1 = inline (no spawned threads)
}

/// Render a single frame; returns tightly-packed RGBA of size `w*h*4`.
///
/// `external_running` lets the caller cancel mid-render: once it's flipped
/// to `false`, workers exit on their next iteration and the function
/// returns whatever has been accumulated so far. Pass `None` for
/// uncancellable renders.
pub fn render_frame(spec: &FrameSpec, external_running: Option<Arc<AtomicBool>>)
    -> Vec<u8>
{
    let (w, h) = (spec.view.width, spec.view.height);
    let view = Arc::new(RwLock::new(spec.view.clone()));
    let hist = Arc::new(RwLock::new(Arc::new(Histogram::new(w, h))));
    let running = external_running
        .unwrap_or_else(|| Arc::new(AtomicBool::new(true)));
    let samples = Arc::new(AtomicU64::new(0));
    let max_count = Arc::new(AtomicU32::new(1));

    let n = spec.n_workers.max(1);
    let target = spec.samples_target;
    if n == 1 {
        sampler::worker_loop(
            0,
            hist.clone(),
            view.clone(),
            running.clone(),
            None,
            samples.clone(),
            max_count.clone(),
            Some(target),
            None,
        );
    } else {
        let mut threads = Vec::with_capacity(n);
        for tid in 0..n {
            let hist_c = hist.clone();
            let view_c = view.clone();
            let running_c = running.clone();
            let samples_c = samples.clone();
            let max_count_c = max_count.clone();
            threads.push(thread::spawn(move || {
                // Workers share the `samples` counter; first one to cross the
                // global target signals all the rest via the `reached` check.
                sampler::worker_loop(
                    tid,
                    hist_c,
                    view_c,
                    running_c,
                    None,
                    samples_c,
                    max_count_c,
                    Some(target),
                    None,
                );
            }));
        }
        for t in threads {
            let _ = t.join();
        }
    }

    let hist = hist.read().clone();
    let max = max_count.load(Ordering::Relaxed).max(1);
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    palette::apply(&hist, max, spec.gamma, &spec.palette, &mut rgba);
    rgba
}
