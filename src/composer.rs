//! Composer mode: screenshot browser + timeline + controls.
//!
//! Layout (when in this mode):
//! ```
//! +-------------------+------------------+
//! | thumb browser     | preview / ctls   |
//! |  (scrolling grid) | (future)         |
//! +-------------------+------------------+
//! |               timeline               |
//! +--------------------------------------+
//! ```
//!
//! Thumbnails are loaded on a worker thread from `~/buddhabrot/*.png`. Each
//! one's metadata is also parsed so a thumbnail click can add a `FrameSpec`
//! to the timeline without re-opening the file.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;

use crate::gizmo::Rect;
use crate::metadata;
use crate::palette::{self, Palette};
use crate::render::{self, FrameSpec};
use crate::session::SessionFile;
use crate::videorender;

pub const THUMB_MAX_DIM: u32 = 320; // storage size (we scale down to display)
pub const THUMB_SIZE_MIN: u32 = 80;
pub const THUMB_SIZE_MAX: u32 = 300;
pub const THUMB_SIZE_DEFAULT: u32 = 140;

pub const PREVIEW_SPP_MIN: f64 = 0.5;
pub const PREVIEW_SPP_MAX: f64 = 100.0;
pub const PREVIEW_SPP_DEFAULT: f64 = 10.0;

pub const SPLIT_FRAC_MIN: f64 = 0.35;
pub const SPLIT_FRAC_MAX: f64 = 0.80;
pub const SPLIT_FRAC_DEFAULT: f64 = 0.60;

/// Which composer slider is currently being dragged.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ComposerSliderDrag {
    None,
    ThumbSize,
    Playhead,
    PreviewQuality,
    PreviewSpp,
    PreviewGamma,
    RenderSpp,
    PathRho,
}

/// Inclusive range of the path-ρ slider. ρ = 0 is pure log-linear (no
/// pullback); √2 is the V-N paper default; 2.0 is pronounced pullback.
pub const RHO_MIN: f64 = 0.0;
pub const RHO_MAX: f64 = 2.0;

/// Gamma slider range — matches the live mode bracket (`g`/`G` keys).
pub const GAMMA_MIN: f32 = 0.1;
pub const GAMMA_MAX: f32 = 2.5;

/// Resolution presets surfaced as radio buttons next to the width/height
/// fields. Laid out column-major in a 3-row × 2-column grid (entries 0..3
/// in the left column, 3..6 in the right). The active preset (if any) is
/// derived by matching the current width/height fields exactly.
pub const RESOLUTION_PRESETS: &[(&str, u32, u32)] = &[
    ("4K",     3840, 2160),
    ("1440p",  2560, 1440),
    ("1080p",  1920, 1080),
    ("720p",   1280,  720),
    ("480p",    854,  480),
    ("240p",    426,  240),
];

/// Number of rows in the resolution-preset radio grid (column-major).
pub const RESOLUTION_GRID_ROWS: usize = 3;

/// Status of an auto-test frame render. Used both by the embedded render
/// panel and the popup render dialog (in live mode).
#[derive(Clone, Debug)]
pub enum TestStatus {
    NotRun,
    Running,
    Done { frame_time: Duration },
    #[allow(dead_code)]
    Failed(String),
}

/// Which text field in the embedded render panel has keyboard focus.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RenderFieldFocus {
    None,
    Width,
    Height,
}

/// Stop in the composer's editable palette: normalized position + RGB color.
#[derive(Clone, Copy, Debug)]
pub struct PaletteStop {
    pub pos: f32,
    pub rgb: [u8; 3],
}

/// Which RGB channel the user is dragging in the stop-color popover.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ColorChannelDrag {
    None,
    R,
    G,
    B,
}

/// Open right-click menu targeting one screenshot in the grid.
#[derive(Clone, Debug)]
pub struct GridContextMenu {
    /// Physical-pixel position where the menu is anchored (cursor on open).
    pub anchor_x: f64,
    pub anchor_y: f64,
    /// The screenshot this menu operates on.
    pub path: PathBuf,
}

#[derive(Clone)]
pub struct Thumbnail {
    pub rgba: Vec<u8>,
    pub w: u32,
    pub h: u32,
    // Eagerly parsed metadata so the UI doesn't pay a PNG re-decode when a
    // user adds a thumbnail to the timeline or to the preview.
    #[allow(dead_code)]
    pub spec: Option<FrameSpec>,
}

pub struct TimelineItem {
    pub source: PathBuf,
    pub spec: FrameSpec,
    /// Absolute seconds for the spline to travel from this item to the next.
    /// Ignored on the last item.
    pub segment_seconds: f64,
    /// Per-keyframe Hermite tangent multiplier. `1.0` (default) gives
    /// the classical centered-difference tangent — the camera passes
    /// through this keyframe at the natural curve speed with no
    /// slowdown. `0.0` zeros the tangent so the camera comes to rest
    /// at the keyframe (ease-in/ease-out). `> 1.0` exaggerates the
    /// tangent so the camera barrels through faster, with more
    /// overshoot. Range typically `[0.0, 3.0]`.
    pub tension: f64,
    /// True when the backing PNG is no longer on disk. Set at load/rescan
    /// time. Items stay in the timeline so the user can see what went
    /// missing (rendered with an X overlay); render is blocked until the
    /// file comes back or the item is removed.
    pub missing: bool,
}

/// A saved-timeline entry surfaced in the load-dialog list.
#[derive(Clone)]
pub struct LoadDialogEntry {
    pub path: PathBuf,
    /// Shown in the dialog: `timeline-1776923077.state`.
    pub display: String,
    /// How many keyframes this save contains — lets the user distinguish
    /// saves at a glance without having to load each one.
    pub item_count: usize,
}

#[derive(Clone)]
pub struct LoadDialog {
    pub entries: Vec<LoadDialogEntry>,
}

#[derive(Clone)]
pub struct TimelineDrag {
    /// Original index of the timeline item being dragged.
    pub from_idx: usize,
    /// Current cursor x in window pixels.
    pub cursor_x: f64,
    /// Current cursor y in window pixels (used to detect drag-off-timeline).
    pub cursor_y: f64,
    /// Whether the cursor has moved far enough to count as a reorder drag.
    pub moved: bool,
}

/// A preview request sent to the preview render thread.
pub struct PreviewRequest {
    pub spec: FrameSpec,
    pub gen: u64,
}

/// The most recent finished preview.
#[derive(Default)]
pub struct PreviewState {
    pub rgba: Option<Vec<u8>>,
    pub w: u32,
    pub h: u32,
    #[allow(dead_code)]
    pub gen: u64,
    pub elapsed: Option<Duration>,
}

pub struct Composer {
    /// All PNG paths known in `~/buddhabrot/`, recent-first.
    pub png_paths: Vec<PathBuf>,
    /// Loaded thumbnails keyed by path (Arc so the main thread can blit
    /// without locking).
    pub thumbs: Arc<RwLock<HashMap<PathBuf, Arc<Thumbnail>>>>,
    /// Set of path: tracks which thumbs are on the timeline (for the check
    /// overlay in the browser).
    pub on_timeline: HashSet<PathBuf>,
    pub timeline: Vec<TimelineItem>,
    pub thumb_size: u32,
    pub grid_scroll: f64,
    /// Editable palette stops (always applied to the preview). Seeded from
    /// the first builtin palette; user edits via the palette editor.
    pub palette_stops: Vec<PaletteStop>,
    /// Derived 1024-entry LUT. Rebuilt from `palette_stops` whenever stops
    /// change.
    pub palette: palette::Palette,
    /// Index of the stop currently being dragged (position change).
    pub palette_drag_stop: Option<usize>,
    /// Index of the stop whose color popover is open.
    pub palette_color_edit: Option<usize>,
    /// Drag state for the RGB sliders in the color popover.
    pub color_channel_drag: ColorChannelDrag,

    /// Right-click context menu in the thumbnail grid (if open).
    pub context_menu: Option<GridContextMenu>,

    /// Open load-timeline dialog (click-outside or Esc to dismiss). Entry
    /// list is populated when the dialog opens and not refreshed afterward.
    pub load_dialog: Option<LoadDialog>,

    // --- Embedded render panel state (mirrors RenderDialogState fields).
    pub rp_width_text: String,
    pub rp_height_text: String,
    pub rp_spp: f64,
    pub rp_focus: RenderFieldFocus,
    pub rp_test: Arc<RwLock<TestStatus>>,
    /// Earliest instant at which the next auto-test may fire. Updated whenever
    /// a render-panel param edit happens (debounced 500 ms).
    pub rp_test_due: Option<Instant>,

    /// Which slider (if any) the mouse is currently dragging.
    pub slider_drag: ComposerSliderDrag,
    /// Pane-splitter drag state: `Some((start_cursor_x, start_frac))`.
    pub splitter_drag: Option<(f64, f64)>,
    /// Horizontal split fraction between the thumbnail grid (left) and the
    /// preview/controls (right). Session-only.
    pub split_frac: f64,
    /// Drag state for the timeline (reorder; may drag off-timeline to delete).
    pub timeline_drag: Option<TimelineDrag>,
    /// Horizontal scroll offset for the timeline content (physical pixels).
    pub timeline_scroll_x: f64,
    /// Middle-click pan drag: `Some((start_cursor_x, start_scroll_x))`.
    pub timeline_pan_drag: Option<(f64, f64)>,
    /// Last single-click on a timeline slot; used for double-click detection
    /// (same idx within 500 ms = remove).
    pub last_timeline_click: Option<(Instant, usize)>,
    /// Playhead position along the timeline, in the range [0, 1].
    pub playhead: f64,
    /// Whether the preview is auto-advancing.
    pub playing: bool,
    /// Samples-per-pixel-per-frame for the preview render. Preview sample
    /// count = `pw * ph * spp`. Exposed as a log-scaled slider in the
    /// controls pane.
    pub spp: f64,
    /// Target FPS for timeline playback + video render (future: playback).
    #[allow(dead_code)]
    pub fps: u32,
    /// Preview resolution as a percentage of the preview pane size, 10–100.
    pub preview_quality_pct: u32,
    /// Gamma applied between histogram and palette in the composer preview.
    /// Overrides the per-keyframe gamma for editing, just like
    /// `palette` overrides per-keyframe palettes.
    pub gamma: f32,
    /// V-N path shape parameter for intra-segment pan/zoom. See
    /// `zoompan` for details. Tunable via the composer's motion panel
    /// slider so a bouncy timeline can be flattened toward log-linear.
    pub rho: f64,

    /// Last tick time, for auto-advance dt computation.
    pub last_tick: Instant,
    /// Monotonic generation counter; bumped whenever the preview should refresh.
    preview_gen: u64,
    /// Set to true whenever the preview should re-render (playhead change,
    /// timeline edit, palette change, resize, etc.).
    pub preview_dirty: bool,
    /// Latest finished preview image (written by the worker thread).
    pub preview_state: Arc<RwLock<PreviewState>>,
    /// While true, the preview worker is allowed to render. The render-time
    /// estimate path flips this to false to cancel any in-flight preview
    /// (and prevent new ones) so the test gets all the CPU.
    pub preview_active: Arc<AtomicBool>,

    load_tx: Sender<PathBuf>,
    #[allow(dead_code)]
    load_thread: thread::JoinHandle<()>,
    preview_tx: Sender<PreviewRequest>,
    #[allow(dead_code)]
    preview_thread: thread::JoinHandle<()>,

    /// Set whenever some persisted state changed (timeline edits, render-panel
    /// fields, palette, gamma). The tick loop flushes to disk — rate-limited
    /// by `last_save_at` so a slider drag doesn't hammer the filesystem.
    save_dirty: bool,
    last_save_at: Option<Instant>,
}

impl Composer {
    pub fn new() -> Self {
        let thumbs: Arc<RwLock<HashMap<PathBuf, Arc<Thumbnail>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let (tx, rx) = channel::<PathBuf>();
        let thumbs_for_thread = thumbs.clone();
        let load_thread = thread::spawn(move || thumbnail_loader(rx, thumbs_for_thread));

        let preview_state = Arc::new(RwLock::new(PreviewState::default()));
        let preview_active = Arc::new(AtomicBool::new(true));
        let (ptx, prx) = channel::<PreviewRequest>();
        let pstate_for_thread = preview_state.clone();
        let pactive_for_thread = preview_active.clone();
        let preview_thread = thread::spawn(move ||
            preview_loop(prx, pstate_for_thread, pactive_for_thread));

        let mut c = Self {
            png_paths: Vec::new(),
            thumbs,
            on_timeline: HashSet::new(),
            timeline: Vec::new(),
            thumb_size: THUMB_SIZE_DEFAULT,
            grid_scroll: 0.0,
            palette_stops: default_palette_stops(),
            palette: palette::builtin()[0].clone(),
            palette_drag_stop: None,
            palette_color_edit: None,
            color_channel_drag: ColorChannelDrag::None,

            context_menu: None,
            load_dialog: None,

            rp_width_text: "1920".to_string(),
            rp_height_text: "1080".to_string(),
            rp_spp: 10.0,
            rp_focus: RenderFieldFocus::None,
            rp_test: Arc::new(RwLock::new(TestStatus::NotRun)),
            rp_test_due: None,

            slider_drag: ComposerSliderDrag::None,
            splitter_drag: None,
            split_frac: SPLIT_FRAC_DEFAULT,
            timeline_drag: None,
            timeline_scroll_x: 0.0,
            timeline_pan_drag: None,
            last_timeline_click: None,
            playhead: 0.0,
            playing: false,
            spp: PREVIEW_SPP_DEFAULT,
            fps: 30,
            preview_quality_pct: 30,
            gamma: 0.75,
            rho: crate::zoompan::RHO_DEFAULT,
            last_tick: Instant::now(),
            preview_gen: 0,
            preview_dirty: true,
            preview_state,
            preview_active,
            load_tx: tx,
            load_thread,
            preview_tx: ptx,
            preview_thread,

            save_dirty: false,
            last_save_at: None,
        };
        c.rescan();
        c.try_restore_saved_state();
        c
    }

    /// Advance the playhead for auto-play, mark preview dirty if anything
    /// changed. Caller passes the current instant.
    pub fn tick(&mut self, now: Instant) {
        let dt = now.duration_since(self.last_tick);
        self.last_tick = now;
        if self.playing && self.timeline.len() >= 2 {
            let duration_s = self.total_duration_seconds();
            if duration_s > 0.0 {
                let adv = dt.as_secs_f64() / duration_s;
                let new = self.playhead + adv;
                self.playhead = new - new.floor();
                self.preview_dirty = true;
            }
        }
        self.flush_save_if_due(now);
    }

    /// Flag the composer as having state worth persisting. The tick loop
    /// batches writes so back-to-back edits (e.g. a slider drag) don't spam
    /// the filesystem.
    pub fn mark_save_dirty(&mut self) {
        self.save_dirty = true;
    }

    fn flush_save_if_due(&mut self, now: Instant) {
        if !self.save_dirty { return; }
        // Rate-limit: at most one write every 250ms. A trailing write after
        // the user stops editing is ensured because `save_dirty` stays true
        // until the throttle lets us through.
        if let Some(t) = self.last_save_at {
            if now.duration_since(t) < Duration::from_millis(250) {
                return;
            }
        }
        crate::timeline_save::write_best_effort(&self.snapshot_for_save());
        self.save_dirty = false;
        self.last_save_at = Some(now);
    }

    fn snapshot_for_save(&self) -> crate::timeline_save::TimelineSave {
        use crate::timeline_save::{PaletteStopSave, TimelineItemSave, TimelineSave};
        TimelineSave {
            fps: self.fps,
            rp_width: self.rp_width(),
            rp_height: self.rp_height(),
            rp_spp: self.rp_spp,
            gamma: self.gamma,
            rho: self.rho,
            playhead: self.playhead,
            palette_stops: self.palette_stops.iter()
                .map(|s| PaletteStopSave { pos: s.pos, rgb: s.rgb })
                .collect(),
            items: self.timeline.iter().map(|t| TimelineItemSave {
                source: t.source.clone(),
                segment_seconds: t.segment_seconds,
                tension: t.tension,
            }).collect(),
        }
    }

    /// Load the autosave slot (`~/buddhabrot/timeline.state`) if it exists.
    /// Silent no-op on first run.
    fn try_restore_saved_state(&mut self) {
        match crate::timeline_save::read() {
            Ok(Some(s)) => self.apply_loaded_state(&s),
            Ok(None) => {}
            Err(e) => eprintln!("timeline.state: load failed: {e}"),
        }
    }

    /// Overwrite the composer's edit state from a loaded save. Items whose
    /// PNG is missing are kept with `missing: true` and a placeholder
    /// FrameSpec, so the user can see exactly what vanished.
    pub fn apply_loaded_state(&mut self, saved: &crate::timeline_save::TimelineSave) {
        self.rp_width_text = saved.rp_width.to_string();
        self.rp_height_text = saved.rp_height.to_string();
        self.rp_spp = saved.rp_spp;
        self.fps = saved.fps;
        self.gamma = saved.gamma;
        self.rho = saved.rho.clamp(RHO_MIN, RHO_MAX);
        self.playhead = saved.playhead.clamp(0.0, 1.0);

        if !saved.palette_stops.is_empty() {
            self.palette_stops = saved.palette_stops.iter().map(|s| PaletteStop {
                pos: s.pos,
                rgb: s.rgb,
            }).collect();
            self.rebuild_palette();
        }

        self.on_timeline.clear();
        self.timeline.clear();
        let placeholder_spec = Self::placeholder_spec();
        for it in &saved.items {
            let exists = crate::timeline_save::source_exists(&it.source);
            let spec = if exists {
                match metadata::read_spec(&it.source, 0, 1) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("timeline: read_spec({}) failed: {}",
                            it.source.display(), e);
                        placeholder_spec.clone()
                    }
                }
            } else {
                placeholder_spec.clone()
            };
            if exists {
                self.on_timeline.insert(it.source.clone());
                let _ = self.load_tx.send(it.source.clone());
            }
            self.timeline.push(TimelineItem {
                source: it.source.clone(),
                spec,
                segment_seconds: it.segment_seconds,
                tension: it.tension,
                missing: !exists,
            });
        }
        self.preview_dirty = true;
        self.schedule_render_test();
        // The save we just applied is already on disk — don't mark dirty.
    }

    /// Write a timestamped snapshot to `~/buddhabrot/timelines/`. Returns the
    /// resulting path (or `None` on I/O failure, which is already logged).
    pub fn save_manual(&self) -> Option<PathBuf> {
        match crate::timeline_save::save_manual(&self.snapshot_for_save()) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("timeline manual-save failed: {e}");
                None
            }
        }
    }

    /// Populate and open the load-timeline dialog. Cheap enough to do
    /// per-click — only stats the saves directory.
    pub fn open_load_dialog(&mut self) {
        let paths = crate::timeline_save::list_saves();
        let mut entries = Vec::with_capacity(paths.len());
        for p in paths {
            let display = p.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string());
            // Peek at item_count without fully loading; cheap vs. a full
            // read_spec walk, and keeps dialog-open latency flat.
            let item_count = match crate::timeline_save::read_from(&p) {
                Ok(Some(s)) => s.items.len(),
                _ => 0,
            };
            entries.push(LoadDialogEntry { path: p, display, item_count });
        }
        self.load_dialog = Some(LoadDialog { entries });
    }

    pub fn close_load_dialog(&mut self) {
        self.load_dialog = None;
    }

    /// Load the save at `path` into this composer. Returns true on success.
    pub fn load_from_path(&mut self, path: &Path) -> bool {
        match crate::timeline_save::read_from(path) {
            Ok(Some(s)) => {
                self.apply_loaded_state(&s);
                self.mark_save_dirty(); // sync the autosave slot
                true
            }
            Ok(None) => {
                eprintln!("load: {} not found", path.display());
                false
            }
            Err(e) => {
                eprintln!("load: {} failed: {}", path.display(), e);
                false
            }
        }
    }

    fn placeholder_spec() -> FrameSpec {
        FrameSpec {
            view: crate::view::View::new(1920, 1080),
            palette: palette::builtin()[0].clone(),
            gamma: 0.75,
            samples_target: 0,
            n_workers: 1,
        }
    }

    /// Refresh each timeline item's `missing` flag by stat-ing the file. Call
    /// this when the set of on-disk PNGs may have changed (e.g. a `rescan`).
    pub fn refresh_missing_flags(&mut self) {
        let mut any_changed = false;
        for it in self.timeline.iter_mut() {
            let missing = !crate::timeline_save::source_exists(&it.source);
            if missing != it.missing {
                it.missing = missing;
                any_changed = true;
            }
        }
        if any_changed {
            self.preview_dirty = true;
            self.schedule_render_test();
        }
    }

    /// Paths of currently-missing keyframes, in timeline order. Used to gate
    /// render spawns so we fail loudly at the UI instead of with a cryptic
    /// subprocess error on the first keyframe open.
    pub fn missing_paths(&self) -> Vec<PathBuf> {
        self.timeline.iter().filter(|t| t.missing)
            .map(|t| t.source.clone()).collect()
    }

    /// Total playback/render duration in seconds: just the sum of
    /// segment_seconds (last item's segment is ignored).
    pub fn total_duration_seconds(&self) -> f64 {
        let n = self.timeline.len();
        if n < 2 { return 0.0; }
        self.timeline[..n - 1].iter()
            .map(|t| t.segment_seconds.max(0.05)).sum()
    }

    /// Mark the preview dirty; the next `request_preview_if_dirty` call will
    /// send a new render job.
    pub fn mark_preview_dirty(&mut self) {
        self.preview_dirty = true;
    }

    /// Rebuild the derived 1024-LUT from `palette_stops`. Call after any stop
    /// edit. Also marks the preview dirty.
    pub fn rebuild_palette(&mut self) {
        // Sort stops by position (may have moved past neighbors mid-drag).
        self.palette_stops.sort_by(|a, b|
            a.pos.partial_cmp(&b.pos).unwrap_or(std::cmp::Ordering::Equal));
        let stops: Vec<(f32, [u8; 3])> = self.palette_stops.iter()
            .map(|s| (s.pos, s.rgb)).collect();
        let name = self.palette.name;
        self.palette = palette::Palette::from_stops_dyn(name, &stops);
        self.mark_preview_dirty();
        self.mark_save_dirty();
    }

    /// Mark the render-panel's test estimate as needing a re-measure. Called
    /// whenever a dependent parameter changes (w/h/scale/spp/timeline).
    /// Debounced by `rp_test_due`; actual test runs from the app's tick loop.
    pub fn schedule_render_test(&mut self) {
        self.rp_test_due = Some(Instant::now() + Duration::from_millis(500));
        // Mark current reading as stale.
        let mut t = self.rp_test.write();
        *t = TestStatus::NotRun;
    }

    /// Set width/height text fields to a resolution preset and re-measure.
    pub fn rp_apply_resolution(&mut self, w: u32, h: u32) {
        self.rp_width_text = w.to_string();
        self.rp_height_text = h.to_string();
        self.schedule_render_test();
        self.mark_save_dirty();
    }

    pub fn rp_width(&self) -> u32 {
        self.rp_width_text.parse().unwrap_or(1920).max(16)
    }
    pub fn rp_height(&self) -> u32 {
        self.rp_height_text.parse().unwrap_or(1080).max(16)
    }
    pub fn rp_samples_per_frame(&self) -> u64 {
        (self.rp_width() as f64 * self.rp_height() as f64 * self.rp_spp)
            .round() as u64
    }

    /// If the preview is dirty, send a new PreviewRequest to the worker.
    /// `w, h` is the target preview size in physical pixels.
    pub fn request_preview_if_dirty(&mut self, w: u32, h: u32, n_workers: usize) {
        if !self.preview_dirty { return; }
        if self.timeline.len() < 2 { return; }
        // Pause preview generation while the render-time-estimate test is
        // running so the two don't fight for CPU. `preview_dirty` stays
        // true, so the next tick after the test completes will dispatch.
        if matches!(*self.rp_test.read(), TestStatus::Running) { return; }
        // Scale by preview_quality_pct.
        let q = (self.preview_quality_pct.clamp(10, 100)) as f64 / 100.0;
        let pw = ((w as f64 * q).round() as u32).max(64) & !1;
        let ph = ((h as f64 * q).round() as u32).max(64) & !1;
        // Samples scale with pixel count × spp so perceptual grain stays
        // roughly constant as the user drags the quality slider. 20k floor
        // keeps the smallest preview from being pure noise.
        let spp = self.spp.clamp(PREVIEW_SPP_MIN, PREVIEW_SPP_MAX);
        let samples = (((pw as f64) * (ph as f64) * spp).round() as u64)
            .max(20_000);
        // Build per-keyframe FrameSpecs (using existing specs cloned).
        let kfs: Vec<FrameSpec> = self.timeline.iter().map(|t| t.spec.clone()).collect();
        let tensions: Vec<f64> = self.timeline.iter()
            .map(|t| t.tension).collect();
        let segments: Vec<f64> = self.timeline.iter()
            .map(|t| t.segment_seconds.max(0.05)).collect();
        // Matching first/last source = loop intent; tangents wrap.
        let cyclic = self.timeline.len() >= 3
            && self.timeline.first().map(|t| &t.source)
                == self.timeline.last().map(|t| &t.source);
        // Always override with the composer's editable palette.
        let palette_over = Some(self.palette.clone());
        let mut spec = interpolate_frame_spec(
            &kfs, &tensions, &segments, self.rho, self.playhead,
            pw, ph, samples, n_workers,
            palette_over.as_ref(), cyclic,
        );
        // Override gamma the same way — the slider is the source of truth for
        // the composer preview, regardless of per-keyframe gamma.
        spec.gamma = self.gamma;
        self.preview_gen = self.preview_gen.wrapping_add(1);
        let _ = self.preview_tx.send(PreviewRequest { spec, gen: self.preview_gen });
        self.preview_dirty = false;
    }

    pub fn rescan(&mut self) {
        self.png_paths = list_screenshots();
        // Queue every thumbnail not already loaded.
        let loaded: HashSet<PathBuf> =
            self.thumbs.read().keys().cloned().collect();
        for p in &self.png_paths {
            if !loaded.contains(p) {
                let _ = self.load_tx.send(p.clone());
            }
        }
        self.refresh_missing_flags();
    }

    /// Append a screenshot as a new keyframe. Duplicates are allowed — the
    /// same screenshot can appear on the timeline multiple times.
    pub fn add_to_timeline(&mut self, path: PathBuf) {
        match metadata::read_spec(&path, 0, 1) {
            Ok(spec) => {
                self.on_timeline.insert(path.clone());
                // First added? Seed render-panel native size from the
                // keyframe's dims so the resolution display starts on
                // something sensible.
                if self.timeline.is_empty() {
                    self.rp_width_text = spec.view.width.to_string();
                    self.rp_height_text = spec.view.height.to_string();
                }
                self.timeline.push(TimelineItem {
                    source: path,
                    spec,
                    segment_seconds: 2.0,
                    tension: 1.0,
                    missing: false,
                });
                self.preview_dirty = true;
                self.schedule_render_test();
                self.mark_save_dirty();
            }
            Err(e) => {
                eprintln!("error: add_to_timeline: failed to read {}: {}",
                    path.display(), e);
            }
        }
    }

    pub fn remove_from_timeline(&mut self, idx: usize) {
        if idx < self.timeline.len() {
            let item = self.timeline.remove(idx);
            // Only purge `on_timeline` entry when no other occurrence of
            // this path remains — so grid star count stays correct.
            let still_present = self.timeline.iter()
                .any(|t| t.source == item.source);
            if !still_present {
                self.on_timeline.remove(&item.source);
            }
            self.preview_dirty = true;
            self.schedule_render_test();
            self.mark_save_dirty();
        }
    }

    pub fn clear_timeline(&mut self) {
        self.on_timeline.clear();
        self.timeline.clear();
        self.preview_dirty = true;
        self.schedule_render_test();
        self.mark_save_dirty();
    }

    /// Move a screenshot's file to the system trash (if it's on disk) and
    /// purge every reference to it from composer state: png_paths, thumbs,
    /// timeline, on_timeline. Safe to call even if the file is already gone.
    pub fn delete_screenshot(&mut self, path: &Path) {
        // Trash the file. We intentionally ignore the result — if the file
        // is missing or the user's desktop doesn't expose a trash, we still
        // want to purge our list entry.
        if path.exists() {
            if let Err(e) = trash::delete(path) {
                eprintln!("trash {} failed: {}", path.display(), e);
            }
        }
        self.png_paths.retain(|p| p != path);
        self.thumbs.write().remove(path);
        // Remove any timeline entries that pointed to this file.
        let removed = self.on_timeline.remove(path);
        self.timeline.retain(|t| t.source != path);
        if removed {
            self.preview_dirty = true;
            self.schedule_render_test();
        }
    }
}

/// Enumerate `*.png` files in `~/buddhabrot/`, sorted newest-first by mtime.
pub fn list_screenshots() -> Vec<PathBuf> {
    let root = SessionFile::root();
    let Ok(rd) = fs::read_dir(&root) else { return Vec::new(); };
    let mut items: Vec<(PathBuf, SystemTime)> = Vec::new();
    for ent in rd.flatten() {
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("png") { continue; }
        let mtime = ent.metadata().and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        items.push((p, mtime));
    }
    items.sort_by(|a, b| b.1.cmp(&a.1));
    items.into_iter().map(|(p, _)| p).collect()
}

/// Preview render worker: receives `PreviewRequest`s and writes the most
/// recent render into `state`. When multiple requests are queued it drains
/// them and only renders the newest, so a rapidly dragging playhead doesn't
/// create a backlog. `active` lets the main thread cancel an in-flight
/// preview so a render-time-estimate test can run alone.
fn preview_loop(
    rx: Receiver<PreviewRequest>,
    state: Arc<RwLock<PreviewState>>,
    active: Arc<AtomicBool>,
) {
    while let Ok(mut req) = rx.recv() {
        while let Ok(next) = rx.try_recv() {
            req = next;
        }
        if !active.load(Ordering::Relaxed) {
            // Skip while the test render holds the budget.
            continue;
        }
        let t0 = Instant::now();
        let rgba = render::render_frame(&req.spec, Some(active.clone()));
        let elapsed = t0.elapsed();
        if !active.load(Ordering::Relaxed) {
            // Cancelled mid-render; keep the previous good frame on screen.
            continue;
        }
        *state.write() = PreviewState {
            rgba: Some(rgba),
            w: req.spec.view.width,
            h: req.spec.view.height,
            gen: req.gen,
            elapsed: Some(elapsed),
        };
    }
}

/// Build a FrameSpec at playhead `t ∈ [0,1]` by time-aware cubic
/// Hermite interpolation through the timeline keyframes, reusing the
/// same math as the video renderer so the preview matches the final
/// output exactly.
pub fn interpolate_frame_spec(
    keyframes: &[FrameSpec],
    tensions: &[f64],
    segments: &[f64],
    rho: f64,
    t: f64,
    ow: u32, oh: u32,
    samples: u64,
    n_workers: usize,
    palette_override: Option<&Palette>,
    cyclic: bool,
) -> FrameSpec {
    let n = keyframes.len();
    if n == 0 {
        let mut s = FrameSpec {
            view: crate::view::View::new(ow.max(1), oh.max(1)),
            palette: palette::builtin()[0].clone(),
            gamma: 0.75,
            samples_target: samples,
            n_workers,
        };
        if let Some(p) = palette_override { s.palette = p.clone(); }
        return s;
    }
    if n == 1 {
        let mut s = keyframes[0].clone();
        s.view.width = ow; s.view.height = oh;
        s.samples_target = samples;
        s.n_workers = n_workers;
        if let Some(p) = palette_override { s.palette = p.clone(); }
        return s;
    }
    let times = videorender::cumulative_times(segments, n);
    let total = *times.last().unwrap_or(&0.0);
    let (from, local_u) = videorender::segment_for_playhead(&times, total, t);
    let indices = videorender::pick_neighbors(n, from, cyclic);
    let nt = videorender::neighbor_times(&times, total, indices, from, cyclic);
    let ntens = videorender::neighbor_tensions(tensions, indices);
    // Build the sign-aligned bi-quaternion path across the whole
    // keyframe sequence so shared seams don't flip. Cheap enough to
    // redo per preview request — n is small (a dozen or so).
    let rotations: Vec<crate::view::Mat4> = keyframes.iter()
        .map(|f| f.view.rotation).collect();
    let (l_path, r_path) =
        crate::rotation_interp::build_aligned_biquats(&rotations);
    let rotation = crate::rotation_interp::eval_hermite(
        &l_path, &r_path, indices, nt, ntens, local_u,
    );
    let p0 = &keyframes[indices[0]];
    let p1 = &keyframes[indices[1]];
    let p2 = &keyframes[indices[2]];
    let p3 = &keyframes[indices[3]];
    let view = videorender::interp_view(&p1.view, &p2.view,
        &p0.view, &p3.view, rotation, nt, ntens, rho, local_u, ow, oh);
    let gamma = videorender::hermite_time_aware(
        [p0.gamma as f64, p1.gamma as f64, p2.gamma as f64, p3.gamma as f64],
        nt, ntens, local_u,
    ) as f32;
    let palette = match palette_override {
        Some(p) => p.clone(),
        None => videorender::interp_palette(&p1.palette, &p2.palette, local_u),
    };
    FrameSpec {
        view, palette, gamma,
        samples_target: samples,
        n_workers,
    }
}

fn thumbnail_loader(
    rx: Receiver<PathBuf>,
    thumbs: Arc<RwLock<HashMap<PathBuf, Arc<Thumbnail>>>>,
) {
    while let Ok(path) = rx.recv() {
        if thumbs.read().contains_key(&path) { continue; }
        match load_thumb_and_spec(&path) {
            Ok(thumb) => {
                thumbs.write().insert(path, Arc::new(thumb));
            }
            Err(e) => {
                eprintln!("error: thumbnail load {}: {}", path.display(), e);
            }
        }
    }
}

fn load_thumb_and_spec(path: &Path) -> io::Result<Thumbnail> {
    let (rgba, w, h) = decode_png_scaled(path, THUMB_MAX_DIM)?;
    let spec = metadata::read_spec(path, 0, 1).ok();
    Ok(Thumbnail { rgba, w, h, spec })
}

fn decode_png_scaled(path: &Path, max_dim: u32) -> io::Result<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let src_w = info.width;
    let src_h = info.height;
    let bpp = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported png color type: {:?}", info.color_type),
            ))
        }
    };

    // Integer box-downsample to target max_dim on the longer axis.
    let scale = (src_w.max(src_h) as f64 / max_dim as f64).ceil().max(1.0) as u32;
    let dst_w = (src_w + scale - 1) / scale;
    let dst_h = (src_h + scale - 1) / scale;
    let mut out = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            let mut count = 0u32;
            let y0 = dy * scale;
            let x0 = dx * scale;
            for sy in y0..(y0 + scale).min(src_h) {
                for sx in x0..(x0 + scale).min(src_w) {
                    let idx = ((sy * src_w + sx) * bpp) as usize;
                    r += buf[idx] as u32;
                    g += buf[idx + 1] as u32;
                    b += buf[idx + 2] as u32;
                    count += 1;
                }
            }
            let o = ((dy * dst_w + dx) * 4) as usize;
            out[o] = (r / count.max(1)) as u8;
            out[o + 1] = (g / count.max(1)) as u8;
            out[o + 2] = (b / count.max(1)) as u8;
            out[o + 3] = 255;
        }
    }
    Ok((out, dst_w, dst_h))
}

/// Blit a thumbnail scaled to fit `dst_rect`, preserving aspect ratio and
/// centering. Uses nearest-neighbor scaling (fine at thumb sizes).
pub fn blit_thumb(
    frame: &mut [u8], fw: u32, fh: u32,
    t: &Thumbnail, dst: Rect,
) {
    // Fit within dst keeping aspect.
    let src_aspect = t.w as f64 / t.h as f64;
    let dst_aspect = dst.w as f64 / dst.h as f64;
    let (tw, th) = if src_aspect > dst_aspect {
        (dst.w, ((dst.w as f64 / src_aspect).round() as u32).max(1))
    } else {
        (((dst.h as f64 * src_aspect).round() as u32).max(1), dst.h)
    };
    let tx = dst.x + (dst.w as i32 - tw as i32) / 2;
    let ty = dst.y + (dst.h as i32 - th as i32) / 2;

    for py in 0..th {
        let sy = ((py as f64 + 0.5) * t.h as f64 / th as f64) as u32;
        let sy = sy.min(t.h - 1);
        let dy = ty + py as i32;
        if dy < 0 || dy as u32 >= fh { continue; }
        for px in 0..tw {
            let sx = ((px as f64 + 0.5) * t.w as f64 / tw as f64) as u32;
            let sx = sx.min(t.w - 1);
            let dx = tx + px as i32;
            if dx < 0 || dx as u32 >= fw { continue; }
            let si = ((sy * t.w + sx) * 4) as usize;
            let di = ((dy as u32 * fw + dx as u32) * 4) as usize;
            frame[di]     = t.rgba[si];
            frame[di + 1] = t.rgba[si + 1];
            frame[di + 2] = t.rgba[si + 2];
            frame[di + 3] = 255;
        }
    }
}

/// Apply the composer's palette override to each timeline item's FrameSpec.
/// Callers get a list of `FrameSpec`s they can feed to the video renderer.
#[allow(dead_code)]
pub fn resolved_timeline_specs(
    timeline: &[TimelineItem],
    palette_override: Option<usize>,
) -> Vec<FrameSpec> {
    let palettes = palette::builtin();
    timeline.iter().map(|item| {
        let mut s = item.spec.clone();
        if let Some(idx) = palette_override {
            if let Some(p) = palettes.get(idx) {
                s.palette = p.clone();
            }
        }
        s
    }).collect()
}

/// Build a ring of keyframes rotating around each of the 6 SO(4) planes and
/// returning to start. Writes the PNGs into `~/buddhabrot/` so they appear
/// in the browser.
pub fn generate_auto_rotation(
    base_view: &crate::view::View,
    palette: &Palette,
    gamma: f32,
    samples: u64,
    n_per_plane: u32,
) -> io::Result<Vec<PathBuf>> {
    use crate::view::Plane;
    use crate::savepng;
    use crate::render;

    let mut out = Vec::new();
    let planes = [
        Plane::XZ, Plane::YW,
        Plane::XW, Plane::YZ,
        Plane::XY, Plane::ZW,
    ];
    for plane in planes {
        for step in 0..n_per_plane {
            let mut v = base_view.clone();
            let theta = 2.0 * std::f64::consts::PI * (step as f64 / n_per_plane as f64);
            v.apply_plane_rotation(plane, theta);
            let spec = FrameSpec {
                view: v.clone(),
                palette: palette.clone(),
                gamma,
                samples_target: samples,
                n_workers: std::thread::available_parallelism()
                    .map(|n| n.get()).unwrap_or(4).saturating_sub(1).max(1),
            };
            let rgba = render::render_frame(&spec, None);
            let ts = SessionFile::now() + out.len() as u64;
            let home = std::env::var_os("HOME")
                .map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            let dir = home.join("buddhabrot");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join(format!("buddhabrot-auto-{ts:016}.png"));
            savepng::save_png_to(&path, &rgba, v.width, v.height,
                savepng::SaveMeta {
                    view: &v,
                    palette_name: spec.palette.name,
                    gamma,
                    samples,
                })?;
            out.push(path);
        }
    }
    Ok(out)
}


/// Seed the editable palette with a reasonable default (fire gradient).
pub fn default_palette_stops() -> Vec<PaletteStop> {
    vec![
        PaletteStop { pos: 0.0,  rgb: [0, 0, 0] },
        PaletteStop { pos: 0.15, rgb: [40, 8, 60] },
        PaletteStop { pos: 0.35, rgb: [140, 20, 30] },
        PaletteStop { pos: 0.6,  rgb: [230, 110, 20] },
        PaletteStop { pos: 0.85, rgb: [250, 220, 110] },
        PaletteStop { pos: 1.0,  rgb: [255, 255, 240] },
    ]
}
