//! Top-level application: window, event loop, rendering, input wiring.
//!
//! Uses winit 0.30's `ApplicationHandler` trait. The window and softbuffer
//! surface are created lazily in `resumed()` because winit 0.30 no longer
//! allows creating windows before the event loop is active.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::composer::{self, Composer};
use crate::gizmo::{Gizmo, Rect};
use crate::input;
use crate::onion::Onion;
use crate::overlay::{self, TextRenderer};
use crate::palette::{self, Palette};
use crate::sampler::{InvalidationReason, SamplerHandle};
use crate::savepng;
use crate::session::{self, SessionFile};
use crate::view::{Mat4, View};

const INIT_W: u32 = 1024;
const INIT_H: u32 = 768;

// Base sizes in logical pixels; multiplied by `scale_factor` at draw time.
const HUD_BASE_PX: u32 = 13;
const HELP_BASE_PX: u32 = 15;
const GIZMO_BASE_PX: u32 = 160;
const BUTTON_BASE_W: u32 = 92;
const BUTTON_BASE_H: u32 = 44;
const MARGIN_BASE: i32 = 12;

const MIN_ONION_SAMPLES: u64 = 50_000;
const ONION_FADE: f32 = 0.85;

const VIDEO_FPS: u32 = 30;

const SPP_MIN: f64 = 0.1;
const SPP_MAX: f64 = 200.0;
const SPP_DEFAULT: f64 = 10.0;
const SCALE_MIN: u32 = 10;
const SCALE_MAX: u32 = 100;

struct DragState {
    button: MouseButton,
    last: PhysicalPosition<f64>,
    start: PhysicalPosition<f64>,
    moved: bool,
}

/// Active drag inside the live-mode palette/gamma HUD. Suppresses the
/// rotation / pan path so dragging a stop or slider doesn't also spin the
/// view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LiveWidgetDrag {
    None,
    /// Repositioning palette stop `idx`.
    Stop(usize),
    /// Dragging the gamma slider thumb.
    Gamma,
    /// Dragging an RGB channel inside the color popover.
    ColorR,
    ColorG,
    ColorB,
}

/// Press in flight on a live-mode top-bar button. Recorded at press time so
/// the cursor-moved rotation path is bypassed (no `DragState` is armed),
/// preventing a 1–2px wobble during a click from invalidating the
/// histogram. Fires the click on release iff the cursor is still over the
/// same button.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LiveButton {
    Save,
    Pause,
    Resume,
}

struct SnapAnim {
    from: Mat4,
    to: Mat4,
    start: Instant,
    duration_ms: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SaveResult {
    None,
    Ok,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Live,
    Composer,
}

#[derive(Clone)]
struct RenderProgress {
    cur: u32,
    total: u32,
    started: Instant,
    fb_w: u32,
    fb_h: u32,
    fps: u32,
    samples_per_frame: u64,
    session_dir: PathBuf,
    output: PathBuf,
}

#[derive(Clone)]
struct RenderDone {
    path: PathBuf,
    elapsed: Duration,
    total_frames: u32,
    fb_w: u32,
    fb_h: u32,
    fps: u32,
    samples_per_frame: u64,
    file_size: Option<u64>,
}

#[derive(Clone)]
enum RenderResult {
    None,
    Ok(RenderDone),
    Failed(String),
}

#[derive(Clone)]
struct RenderJob {
    keyframes: Vec<PathBuf>,
    fb_w: u32,
    fb_h: u32,
    fps: u32,
    samples_per_frame: u64,
    total_frames: u32,
    output: PathBuf,
    session_dir: PathBuf,
    /// Per-keyframe Hermite tangent multiplier (1.0 = no slowdown).
    /// Length == keyframes.len().
    tensions: Vec<f64>,
    /// Per-keyframe travel seconds (last entry ignored). Length ==
    /// keyframes.len().
    segments: Vec<f64>,
    /// Composer gamma override. Applied to every frame (matches preview),
    /// replacing the interpolated per-keyframe gamma.
    gamma_override: Option<f32>,
    /// Composer palette-stops override. Applied to every frame (matches
    /// preview), replacing per-keyframe palette interpolation.
    palette_stops_override: Option<Vec<(f32, [u8; 3])>>,
    /// V-N path shape parameter for pan/zoom. √2 default; 0 flattens
    /// to log-linear. Set by the composer's motion panel slider.
    rho: f64,
}

impl RenderJob {
    fn new(
        keyframes: Vec<PathBuf>,
        fb_w: u32, fb_h: u32, fps: u32, samples_per_frame: u64,
    ) -> Self {
        let dur_s = 2.0 * (keyframes.len().saturating_sub(1)) as f64;
        let total_frames = (dur_s * fps as f64).round().max(2.0) as u32;
        let ts = SessionFile::now();
        let home = std::env::var_os("HOME")
            .map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        let output = home.join("buddhabrot").join(format!("buddhabrot-{ts}.mp4"));
        let session_dir = SessionFile::new_dir(ts);
        Self {
            keyframes, fb_w, fb_h, fps, samples_per_frame,
            total_frames, output, session_dir,
            tensions: Vec::new(),
            segments: Vec::new(),
            gamma_override: None,
            palette_stops_override: None,
            rho: crate::zoompan::RHO_DEFAULT,
        }
    }

    fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    fn with_schedule(mut self, tensions: Vec<f64>, segments: Vec<f64>) -> Self {
        self.tensions = tensions;
        self.segments = segments;
        self
    }

    fn with_overrides(
        mut self,
        gamma: Option<f32>,
        palette_stops: Option<Vec<(f32, [u8; 3])>>,
    ) -> Self {
        self.gamma_override = gamma;
        self.palette_stops_override = palette_stops;
        self
    }
}

/// Build a RenderJob from the composer's embedded render panel + timeline.
/// Returns None when the timeline has fewer than 2 keyframes.
/// If any timeline keyframe's PNG is missing from disk, returns a
/// user-facing error string listing the offenders. Used to gate render spawns
/// so we fail loudly at the UI instead of with a cryptic subprocess error.
fn missing_blocker_msg(composer: &Composer) -> Option<String> {
    let missing = composer.missing_paths();
    if missing.is_empty() { return None; }
    let names: Vec<String> = missing.iter()
        .map(|p| p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string()))
        .collect();
    Some(format!(
        "render blocked: {} missing keyframe(s): {}",
        missing.len(),
        names.join(", "),
    ))
}

fn build_render_job_from_composer(composer: &Composer) -> Option<RenderJob> {
    if composer.timeline.len() < 2 { return None; }
    let keyframes: Vec<PathBuf> = composer.timeline.iter()
        .map(|t| t.source.clone()).collect();
    let tensions: Vec<f64> = composer.timeline.iter()
        .map(|t| t.tension).collect();
    let segments: Vec<f64> = composer.timeline.iter()
        .map(|t| t.segment_seconds.max(0.05)).collect();
    let w = composer.rp_width();
    let h = composer.rp_height();
    let samples = composer.rp_samples_per_frame();
    // Use the composer's live duration (sum of segment_seconds) so the
    // frame count matches the preview exactly.
    let duration = composer.total_duration_seconds();
    let total_frames = (duration * composer.fps as f64).round().max(2.0) as u32;
    let palette_stops: Vec<(f32, [u8; 3])> = composer.palette_stops.iter()
        .map(|s| (s.pos, s.rgb)).collect();
    let mut job = RenderJob::new(keyframes, w, h, composer.fps, samples)
        .with_schedule(tensions, segments)
        .with_overrides(Some(composer.gamma), Some(palette_stops))
        .with_rho(composer.rho);
    job.total_frames = total_frames;
    Some(job)
}

#[derive(Clone)]
struct ViewSnapshot {
    rotation: Mat4,
    center: [f64; 2],
    half_width: f64,
    max_iter: u32,
}

impl ViewSnapshot {
    fn of(v: &View) -> Self {
        Self {
            rotation: v.rotation,
            center: v.center,
            half_width: v.half_width,
            max_iter: v.max_iter,
        }
    }
    fn matches(&self, o: &ViewSnapshot) -> bool {
        self.rotation == o.rotation
            && self.center == o.center
            && self.half_width == o.half_width
            && self.max_iter == o.max_iter
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum DialogFocus { None, Width, Height }

#[derive(Copy, Clone, PartialEq, Eq)]
enum DialogDrag { None, ScaleSlider, SppSlider }

#[derive(Clone)]
enum TestState {
    NotRun,
    Running,
    Done { frame_time: std::time::Duration },
    Failed(String),
}

struct RenderDialogState {
    native_w: u32,
    native_h: u32,
    width_text: String,   // edited buffer; parsed on commit
    height_text: String,
    scale_pct: u32,       // SCALE_MIN..=SCALE_MAX
    spp: f64,             // SPP_MIN..=SPP_MAX
    focus: DialogFocus,
    test: Arc<RwLock<TestState>>,
    keyframes: Vec<PathBuf>, // keyframes captured at open time
    tensions: Vec<f64>,      // per-keyframe tangent multiplier
    segments: Vec<f64>,      // per-keyframe travel seconds (last ignored)
}

impl RenderDialogState {
    fn width(&self) -> u32 {
        self.width_text.parse().unwrap_or(self.native_w).max(16)
    }
    fn height(&self) -> u32 {
        self.height_text.parse().unwrap_or(self.native_h).max(16)
    }
    fn samples_per_frame(&self) -> u64 {
        (self.width() as f64 * self.height() as f64 * self.spp).round() as u64
    }
    fn apply_scale(&mut self) {
        let pct = self.scale_pct as f64 / 100.0;
        let w = ((self.native_w as f64 * pct).round() as u32).max(16);
        let h = ((self.native_h as f64 * pct).round() as u32).max(16);
        self.width_text = w.to_string();
        self.height_text = h.to_string();
    }
}

struct App {
    view: Arc<RwLock<View>>,
    sampler: SamplerHandle,
    /// Editable palette stops driving the live histogram → RGBA mapping.
    /// Mirrors the composer's editor: click bar to add, drag stop to move,
    /// click stop to color-edit, right-click to delete. The HUD lives in
    /// the lower-left of the live pane.
    palette_stops: Vec<composer::PaletteStop>,
    /// Derived 1024-LUT rebuilt from `palette_stops` on every edit.
    palette: Palette,
    /// Index into `palette::builtin_stops()` so the `p` / shift-`p` cycle
    /// keys can advance through preset stop sets.
    palette_preset_idx: usize,
    /// Active live-HUD widget drag (palette stop, gamma slider, color
    /// channel). When `Some`, view rotation/pan is suppressed for that
    /// drag.
    live_widget_drag: LiveWidgetDrag,
    /// Cursor position when the active `live_widget_drag` was armed, used
    /// to promote "click a stop without moving" → "open color popover" on
    /// release (mirrors the composer's `did_move` check).
    live_widget_press_origin: Option<PhysicalPosition<f64>>,
    /// Press in flight on a top-bar button (save / pause / resume). When
    /// `Some`, no `DragState` was armed, so cursor-move never bleeds into
    /// the rotation path — protects render progress from sub-threshold
    /// wobble during the click.
    live_button_press: Option<LiveButton>,
    /// When the color popover is open, the index of the stop being edited.
    palette_color_edit: Option<usize>,
    gamma: f32,
    text: TextRenderer,

    drag: Option<DragState>,
    cursor_pos: PhysicalPosition<f64>,
    mods_shift: bool,
    mods_ctrl: bool,
    mods_super: bool,
    help_visible: bool,

    snap: Option<SnapAnim>,
    onion: Option<Onion>,
    onion_enabled: bool,
    show_axes: bool,
    save_pending: bool,
    last_save: SaveResult,

    keyframes: Vec<PathBuf>,
    render_progress: Arc<RwLock<Option<RenderProgress>>>,
    last_render: Arc<RwLock<RenderResult>>,
    render_queue: Vec<RenderJob>,
    render_dialog: Option<RenderDialogState>,
    dialog_drag: DialogDrag,
    resume_candidate: Option<PathBuf>,

    undo_stack: Vec<ViewSnapshot>,
    undo_cursor: usize,
    last_undo_push: Option<Instant>,

    mode: Mode,
    composer: Composer,
    /// Was the composer's render-time-estimate test running on the previous
    /// tick? Used to detect the Running→Done transition so we can dirty the
    /// preview (which got cancelled when the test took the floor).
    composer_test_was_running: bool,

    last_frame: Instant,
    last_sample_count: u64,
    samples_per_sec: f64,

    cursor_icon: winit::window::CursorIcon,

    /// True while the sampler is auto-paused because the composer tab is
    /// active. Set on Live→Composer if the sampler wasn't already paused;
    /// cleared on Composer→Live after the resume. Lets us distinguish
    /// "user paused manually" from "we paused you for tab-switch"; the
    /// former is preserved across tab switches.
    auto_paused_by_composer: bool,

    ws: Option<WindowState>,
}

struct WindowState {
    window: Arc<Window>,
    surface: Surface<Arc<Window>, Arc<Window>>,
    fb_w: u32,
    fb_h: u32,
    scale_factor: f64,
    rgba: Vec<u8>,
    gizmo: Gizmo,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let view = Arc::new(RwLock::new(View::new(INIT_W, INIT_H)));
    let sampler = SamplerHandle::spawn(view.clone());
    let initial_snapshot = ViewSnapshot::of(&view.read());

    let presets = palette::builtin_stops();
    let (preset_name, preset_stops) = presets[0].clone();
    let palette_stops: Vec<composer::PaletteStop> = preset_stops.iter()
        .map(|(p, rgb)| composer::PaletteStop { pos: *p, rgb: *rgb })
        .collect();
    let initial_palette = Palette::from_stops_dyn(preset_name, &preset_stops);
    let mut app = App {
        view,
        sampler,
        palette_stops,
        palette: initial_palette,
        palette_preset_idx: 0,
        live_widget_drag: LiveWidgetDrag::None,
        live_widget_press_origin: None,
        live_button_press: None,
        palette_color_edit: None,
        gamma: 0.75,
        text: TextRenderer::new(),
        drag: None,
        cursor_pos: PhysicalPosition::new(0.0, 0.0),
        mods_shift: false,
        mods_ctrl: false,
        mods_super: false,
        help_visible: false,
        snap: None,
        onion: None,
        onion_enabled: false,
        show_axes: false,
        save_pending: false,
        last_save: SaveResult::None,
        keyframes: Vec::new(),
        render_progress: Arc::new(RwLock::new(None)),
        last_render: Arc::new(RwLock::new(RenderResult::None)),
        render_queue: Vec::new(),
        render_dialog: None,
        dialog_drag: DialogDrag::None,
        resume_candidate: None,
        undo_stack: vec![initial_snapshot],
        undo_cursor: 0,
        last_undo_push: None,
        mode: Mode::Live,
        composer: Composer::new(),
        composer_test_was_running: false,
        last_frame: Instant::now(),
        last_sample_count: 0,
        samples_per_sec: 0.0,
        cursor_icon: winit::window::CursorIcon::Default,
        auto_paused_by_composer: false,
        ws: None,
    };

    app.scan_for_sessions();

    event_loop.run_app(&mut app)?;
    Ok(())
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.ws.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("buddhabrot")
            .with_inner_size(LogicalSize::new(INIT_W, INIT_H));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => { eprintln!("create_window: {e}"); event_loop.exit(); return; }
        };
        let sz = window.inner_size();
        let fb_w = sz.width.max(1);
        let fb_h = sz.height.max(1);
        let scale_factor = window.scale_factor();

        let context = match Context::new(window.clone()) {
            Ok(c) => c,
            Err(e) => { eprintln!("softbuffer context: {e}"); event_loop.exit(); return; }
        };
        let mut surface: Surface<Arc<Window>, Arc<Window>> =
            match Surface::new(&context, window.clone()) {
                Ok(s) => s,
                Err(e) => { eprintln!("softbuffer surface: {e}"); event_loop.exit(); return; }
            };
        if let Err(e) = surface.resize(
            NonZeroU32::new(fb_w).unwrap(),
            NonZeroU32::new(fb_h).unwrap(),
        ) {
            eprintln!("softbuffer resize: {e}");
            event_loop.exit();
            return;
        }

        let rgba = vec![0u8; (fb_w as usize) * (fb_h as usize) * 4];
        let gizmo = Gizmo::new(gizmo_rect_for(fb_w, fb_h, scale_factor));

        // The sampler was spawned with placeholder dimensions; swap in the real
        // ones now that the window is up.
        self.view.write().resize(fb_w, fb_h);
        self.sampler.invalidate(InvalidationReason::Resize { _w: fb_w, _h: fb_h });

        self.ws = Some(WindowState {
            window,
            surface,
            fb_w,
            fb_h,
            scale_factor,
            rgba,
            gizmo,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(ws) = self.ws.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                let w = size.width.max(1);
                let h = size.height.max(1);
                if w != ws.fb_w || h != ws.fb_h {
                    ws.fb_w = w;
                    ws.fb_h = h;
                    let _ = ws.surface.resize(
                        NonZeroU32::new(w).unwrap(),
                        NonZeroU32::new(h).unwrap(),
                    );
                    ws.rgba.resize((w as usize) * (h as usize) * 4, 0);
                    self.view.write().resize(w, h);
                    self.sampler
                        .invalidate(InvalidationReason::Resize { _w: w, _h: h });
                    ws.gizmo.set_rect(gizmo_rect_for(w, h, ws.scale_factor));
                    self.onion = None;
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor: sf, .. } => {
                ws.scale_factor = sf;
                ws.gizmo.set_rect(gizmo_rect_for(ws.fb_w, ws.fb_h, ws.scale_factor));
            }

            WindowEvent::ModifiersChanged(m) => {
                let s = m.state();
                self.mods_shift = s.shift_key();
                self.mods_ctrl = s.control_key();
                self.mods_super = s.super_key();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = position;
                if self.mode == Mode::Composer {
                    composer_handle_cursor_moved(
                        &mut self.composer, position, ws.fb_w, ws.fb_h, ws.scale_factor);
                    // Cursor icon: col-resize over the splitter (or while
                    // dragging it). Default everywhere else.
                    let l = composer_layout(
                        ws.fb_w, ws.fb_h, ws.scale_factor, &self.composer);
                    let cx = position.x as i32;
                    let cy = position.y as i32;
                    let over_splitter = composer_splitter_rect(&l).contains(cx, cy)
                        || self.composer.splitter_drag.is_some();
                    let want = if over_splitter {
                        winit::window::CursorIcon::ColResize
                    } else {
                        winit::window::CursorIcon::Default
                    };
                    if self.cursor_icon != want {
                        ws.window.set_cursor(winit::window::Cursor::Icon(want));
                        self.cursor_icon = want;
                    }
                    return;
                }
                // Live-mode HUD widget drag (palette stop / gamma slider /
                // RGB channel) — applies before the rotation/pan path so
                // the view doesn't spin while editing the palette.
                if self.live_widget_drag != LiveWidgetDrag::None {
                    let layout = live_palette_hud_layout(
                        ws.fb_w, ws.fb_h, ws.scale_factor);
                    match self.live_widget_drag {
                        LiveWidgetDrag::Stop(idx) => {
                            let bar = layout.palette_bar;
                            let t = ((position.x as i32 - bar.x) as f64
                                / bar.w.max(1) as f64).clamp(0.0, 1.0) as f32;
                            if let Some(s) = self.palette_stops.get_mut(idx) {
                                s.pos = t;
                            }
                            self.rebuild_palette();
                        }
                        LiveWidgetDrag::Gamma => {
                            apply_live_gamma_slider(
                                &mut self.gamma, layout.gamma_slider,
                                position.x as i32);
                        }
                        ch @ (LiveWidgetDrag::ColorR
                            | LiveWidgetDrag::ColorG
                            | LiveWidgetDrag::ColorB) => {
                            if let Some(stop_idx) = self.palette_color_edit {
                                let sliders = live_color_popover_sliders(
                                    layout.color_popover, ws.scale_factor);
                                let rect = match ch {
                                    LiveWidgetDrag::ColorR => sliders[0],
                                    LiveWidgetDrag::ColorG => sliders[1],
                                    LiveWidgetDrag::ColorB => sliders[2],
                                    _ => Rect { x: 0, y: 0, w: 0, h: 0 },
                                };
                                apply_live_color_channel(
                                    &mut self.palette_stops, stop_idx, ch,
                                    rect, position.x as i32);
                                self.rebuild_palette();
                            }
                        }
                        LiveWidgetDrag::None => {}
                    }
                    return;
                }
                if let Some(d) = self.drag.as_mut() {
                    let dx = position.x - d.last.x;
                    let dy = position.y - d.last.y;
                    d.last = position;
                    let moved_dist = ((position.x - d.start.x).powi(2)
                        + (position.y - d.start.y).powi(2))
                        .sqrt();
                    if moved_dist > 3.0 {
                        d.moved = true;
                    }

                    // Dialog slider drag, if active.
                    if self.dialog_drag != DialogDrag::None {
                        let l = dialog_layout(ws.fb_w, ws.fb_h, ws.scale_factor);
                        let cx = position.x as i32;
                        if let Some(dlg) = self.render_dialog.as_mut() {
                            match self.dialog_drag {
                                DialogDrag::ScaleSlider => {
                                    let t = ((cx - l.scale_slider.x) as f64
                                        / l.scale_slider.w.max(1) as f64).clamp(0.0, 1.0);
                                    dlg.scale_pct = (SCALE_MIN as f64
                                        + t * (SCALE_MAX - SCALE_MIN) as f64)
                                        .round() as u32;
                                    dlg.apply_scale();
                                    *dlg.test.write() = TestState::NotRun;
                                }
                                DialogDrag::SppSlider => {
                                    let t = ((cx - l.spp_slider.x) as f64
                                        / l.spp_slider.w.max(1) as f64).clamp(0.0, 1.0);
                                    dlg.spp = (SPP_MIN.ln()
                                        + t * (SPP_MAX.ln() - SPP_MIN.ln())).exp();
                                    *dlg.test.write() = TestState::NotRun;
                                }
                                DialogDrag::None => {}
                            }
                        }
                        return;
                    }

                    // View rotation / pan only when no dialog is open.
                    if self.render_dialog.is_some() { return; }

                    match d.button {
                        MouseButton::Left => {
                            let (ph, pv) = input::drag_planes(self.mods_shift, self.mods_ctrl);
                            let mut v = self.view.write();
                            let zs = input::zoom_rotation_scale(v.half_width);
                            v.apply_plane_rotation(ph, input::drag_to_angle(dx, zs));
                            v.apply_plane_rotation(pv, input::drag_to_angle(dy, zs));
                            drop(v);
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                        }
                        MouseButton::Right => {
                            let (ph, pv) = input::right_drag_planes(self.mods_shift);
                            let mut v = self.view.write();
                            let zs = input::zoom_rotation_scale(v.half_width);
                            v.apply_plane_rotation(ph, input::drag_to_angle(-dx, zs));
                            v.apply_plane_rotation(pv, input::drag_to_angle(-dy, zs));
                            drop(v);
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                        }
                        MouseButton::Middle => {
                            self.view.write().pan_pixels(dx, dy);
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    // Tab click trumps everything else.
                    if button == MouseButton::Left {
                        let tabs = tab_bar_rects(ws.fb_w, ws.scale_factor);
                        let cx = self.cursor_pos.x as i32;
                        let cy = self.cursor_pos.y as i32;
                        if tabs[0].contains(cx, cy) {
                            self.mode = Mode::Live;
                            self.composer.context_menu = None;
                            // Resume the live sampler if we were the ones
                            // who paused it on the way into composer.
                            if self.auto_paused_by_composer {
                                self.sampler.paused.store(false, Ordering::Relaxed);
                                self.auto_paused_by_composer = false;
                            }
                            return;
                        }
                        if tabs[1].contains(cx, cy) {
                            self.mode = Mode::Composer;
                            self.composer.rescan();
                            // Pause the live sampler so the composer's
                            // preview lane has full CPU. Only pause if it
                            // wasn't already paused (preserves a manual
                            // pause across the tab-switch).
                            let already_paused =
                                self.sampler.paused.load(Ordering::Relaxed);
                            if !already_paused {
                                self.sampler.paused.store(true, Ordering::Relaxed);
                                self.auto_paused_by_composer = true;
                            }
                            return;
                        }
                    }

                    // In composer mode, route clicks to the composer handler
                    // rather than live-mode rotation/pan/save/etc.
                    if self.mode == Mode::Composer {
                        // Clicks in composer fire on release so they behave
                        // like buttons. Only the slider drag start needs to
                        // see the press. A DragState is still created so the
                        // shared Released block can dispatch click actions.
                        self.drag = Some(DragState {
                            button,
                            last: self.cursor_pos,
                            start: self.cursor_pos,
                            moved: false,
                        });
                        if button == MouseButton::Right
                            && self.render_dialog.is_none()
                        {
                            let cx = self.cursor_pos.x as i32;
                            let cy = self.cursor_pos.y as i32;
                            let l = composer_layout(
                                ws.fb_w, ws.fb_h, ws.scale_factor, &self.composer);
                            // Right-click on a palette stop → delete (min 2
                            // stops enforced).
                            if l.preview.contains(cx, cy) {
                                let pp = preview_pane_layout(l.preview, ws.scale_factor);
                                if let Some((i, _)) = palette_stop_hit(
                                    &self.composer, pp.palette_bar,
                                    pp.palette_stops_y, pp.palette_stops_h,
                                    cx, cy,
                                ) {
                                    if self.composer.palette_stops.len() > 2 {
                                        self.composer.palette_stops.remove(i);
                                        if self.composer.palette_color_edit == Some(i) {
                                            self.composer.palette_color_edit = None;
                                        }
                                        self.composer.rebuild_palette();
                                    }
                                    return;
                                }
                            }
                            // Right-click on a grid thumbnail → open its
                            // context menu. (The "+" tile has no menu.)
                            if let Some(Some(idx)) = grid_thumb_hit(
                                &self.composer, l.grid, ws.scale_factor, cx, cy)
                            {
                                let path = self.composer.png_paths[idx].clone();
                                self.composer.context_menu =
                                    Some(composer::GridContextMenu {
                                        anchor_x: self.cursor_pos.x,
                                        anchor_y: self.cursor_pos.y,
                                        path,
                                    });
                                return;
                            }
                            // Right-click on a timeline slot → same menu,
                            // anchored at the cursor, operating on that
                            // item's source PNG.
                            if l.timeline.contains(cx, cy) {
                                let tl = timeline_layout(
                                    &self.composer, l.timeline, ws.scale_factor);
                                for (idx, slot) in tl.slots.iter().enumerate() {
                                    if slot.contains(cx, cy) {
                                        if let Some(it) =
                                            self.composer.timeline.get(idx)
                                        {
                                            self.composer.context_menu =
                                                Some(composer::GridContextMenu {
                                                    anchor_x: self.cursor_pos.x,
                                                    anchor_y: self.cursor_pos.y,
                                                    path: it.source.clone(),
                                                });
                                        }
                                        return;
                                    }
                                }
                            }
                            // Right-click anywhere else dismisses an open
                            // grid menu.
                            if self.composer.context_menu.is_some() {
                                self.composer.context_menu = None;
                                return;
                            }
                        }
                        if button == MouseButton::Left
                            && self.render_dialog.is_none()
                        {
                            // If a grid context menu is open, this press
                            // either invokes a menu item or dismisses it.
                            if let Some(menu) = self.composer.context_menu.clone() {
                                let cx = self.cursor_pos.x as i32;
                                let cy = self.cursor_pos.y as i32;
                                let ml = grid_context_menu_layout(
                                    &menu, ws.fb_w, ws.fb_h, ws.scale_factor);
                                if ml.use_in_live.contains(cx, cy) {
                                    self.use_screenshot_in_live(&menu.path);
                                    self.composer.context_menu = None;
                                    return;
                                }
                                if ml.trash.contains(cx, cy) {
                                    self.composer.delete_screenshot(&menu.path);
                                    self.composer.context_menu = None;
                                    return;
                                }
                                if !ml.frame.contains(cx, cy) {
                                    self.composer.context_menu = None;
                                    // Fall through to normal press handling.
                                } else {
                                    // Click on menu background — consume.
                                    return;
                                }
                            }
                            let _ = composer_handle_press(
                                &mut self.composer, self.cursor_pos,
                                ws.fb_w, ws.fb_h, ws.scale_factor,
                            );
                        }
                        if button == MouseButton::Middle
                            && self.render_dialog.is_none()
                        {
                            let cx = self.cursor_pos.x as i32;
                            let cy = self.cursor_pos.y as i32;
                            let l = composer_layout(
                                ws.fb_w, ws.fb_h, ws.scale_factor, &self.composer);
                            if l.timeline.contains(cx, cy) {
                                self.composer.timeline_pan_drag = Some((
                                    self.cursor_pos.x,
                                    self.composer.timeline_scroll_x,
                                ));
                            }
                        }
                        return;
                    }

                    // Live-mode palette/gamma HUD (lower-left) intercepts
                    // presses before the rotation/pan path so dragging a
                    // stop or slider doesn't also spin the view. All mouse
                    // buttons are intercepted while over the panel; only
                    // left/right have meaningful actions, the rest are
                    // swallowed to prevent middle-pan bleeding through.
                    if self.render_dialog.is_none() {
                        let cx = self.cursor_pos.x as i32;
                        let cy = self.cursor_pos.y as i32;
                        let layout = live_palette_hud_layout(
                            ws.fb_w, ws.fb_h, ws.scale_factor);
                        if layout.panel.contains(cx, cy)
                            || (self.palette_color_edit.is_some()
                                && layout.color_popover.contains(cx, cy))
                        {
                            // Color popover (if open): RGB slider press.
                            if button == MouseButton::Left {
                                if let Some(stop_idx) = self.palette_color_edit {
                                    if layout.color_popover.contains(cx, cy) {
                                        let sliders = live_color_popover_sliders(
                                            layout.color_popover, ws.scale_factor);
                                        let mut hit = false;
                                        for (drag, rect) in [
                                            (LiveWidgetDrag::ColorR, sliders[0]),
                                            (LiveWidgetDrag::ColorG, sliders[1]),
                                            (LiveWidgetDrag::ColorB, sliders[2]),
                                        ] {
                                            if rect.contains(cx, cy) {
                                                self.live_widget_drag = drag;
                                                self.live_widget_press_origin =
                                                    Some(self.cursor_pos);
                                                apply_live_color_channel(
                                                    &mut self.palette_stops,
                                                    stop_idx, drag, rect, cx);
                                                self.rebuild_palette();
                                                hit = true;
                                                break;
                                            }
                                        }
                                        if hit { return; }
                                        // Click inside popover but not a slider — consume.
                                        return;
                                    }
                                }
                            }
                            // Reset button.
                            if button == MouseButton::Left
                                && layout.palette_reset_btn.contains(cx, cy)
                            {
                                let presets = palette::builtin_stops();
                                if !presets.is_empty() {
                                    let (name, stops) =
                                        presets[self.palette_preset_idx
                                            .min(presets.len() - 1)].clone();
                                    self.palette_stops = stops.iter()
                                        .map(|(p, rgb)| composer::PaletteStop {
                                            pos: *p, rgb: *rgb,
                                        })
                                        .collect();
                                    self.palette = Palette::from_stops_dyn(
                                        name, &stops);
                                    self.palette_color_edit = None;
                                }
                                return;
                            }
                            // Stop markers: left = drag, right = delete (≥2).
                            if let Some(i) = live_palette_stop_hit(
                                &self.palette_stops, layout.palette_bar,
                                layout.palette_stops_y, layout.palette_stops_h,
                                cx, cy)
                            {
                                if button == MouseButton::Right {
                                    if self.palette_stops.len() > 2 {
                                        self.palette_stops.remove(i);
                                        if self.palette_color_edit == Some(i) {
                                            self.palette_color_edit = None;
                                        } else if let Some(sel) = self.palette_color_edit {
                                            if sel > i {
                                                self.palette_color_edit = Some(sel - 1);
                                            }
                                        }
                                        self.rebuild_palette();
                                    }
                                } else {
                                    self.live_widget_drag = LiveWidgetDrag::Stop(i);
                                    self.live_widget_press_origin = Some(self.cursor_pos);
                                }
                                return;
                            }
                            // Bar (not on a marker): left-click adds a stop.
                            if button == MouseButton::Left
                                && layout.palette_bar.contains(cx, cy)
                            {
                                let bar = layout.palette_bar;
                                let t = ((cx - bar.x) as f64
                                    / bar.w.max(1) as f64).clamp(0.0, 1.0) as f32;
                                let rgb = self.palette.sample_rgb(t);
                                self.palette_stops.push(
                                    composer::PaletteStop { pos: t, rgb });
                                self.rebuild_palette();
                                return;
                            }
                            // Gamma slider.
                            if button == MouseButton::Left
                                && layout.gamma_slider.contains(cx, cy)
                            {
                                self.live_widget_drag = LiveWidgetDrag::Gamma;
                                self.live_widget_press_origin = Some(self.cursor_pos);
                                apply_live_gamma_slider(
                                    &mut self.gamma, layout.gamma_slider, cx);
                                return;
                            }
                            // Click inside the panel but not on any control —
                            // close the color popover if it's open.
                            if button == MouseButton::Left
                                && self.palette_color_edit.is_some()
                            {
                                self.palette_color_edit = None;
                            }
                            return;
                        }
                        // Click outside the HUD with popover open: close it
                        // (matches composer behavior). Then fall through so
                        // the click still does its normal thing.
                        if button == MouseButton::Left
                            && self.palette_color_edit.is_some()
                        {
                            self.palette_color_edit = None;
                        }
                    }

                    // Top-bar button intercept: a press that lands on save /
                    // pause / resume must not arm a rotation drag, or a
                    // sub-threshold wobble during the click would spin the
                    // view by a pixel and invalidate the histogram (losing
                    // render progress).
                    if self.render_dialog.is_none() && button == MouseButton::Left {
                        let cx = self.cursor_pos.x as i32;
                        let cy = self.cursor_pos.y as i32;
                        let hit = if save_button_rect(ws.fb_w, ws.scale_factor)
                            .contains(cx, cy)
                        {
                            Some(LiveButton::Save)
                        } else if pause_button_rect(ws.fb_w, ws.scale_factor)
                            .contains(cx, cy)
                        {
                            Some(LiveButton::Pause)
                        } else if self.resume_candidate.is_some()
                            && self.render_progress.read().is_none()
                            && resume_button_rect(ws.fb_w, ws.scale_factor)
                                .contains(cx, cy)
                        {
                            Some(LiveButton::Resume)
                        } else {
                            None
                        };
                        if let Some(b) = hit {
                            self.live_button_press = Some(b);
                            return;
                        }
                    }

                    self.drag = Some(DragState {
                        button,
                        last: self.cursor_pos,
                        start: self.cursor_pos,
                        moved: false,
                    });

                    // If dialog is open and press is on a slider, start slider
                    // drag + snap value to cursor immediately.
                    if self.render_dialog.is_some() && button == MouseButton::Left {
                        let l = dialog_layout(ws.fb_w, ws.fb_h, ws.scale_factor);
                        let cx = self.cursor_pos.x as i32;
                        let cy = self.cursor_pos.y as i32;
                        if l.scale_slider.contains(cx, cy) {
                            self.dialog_drag = DialogDrag::ScaleSlider;
                            let t = ((cx - l.scale_slider.x) as f64
                                / l.scale_slider.w.max(1) as f64).clamp(0.0, 1.0);
                            let dlg = self.render_dialog.as_mut().unwrap();
                            dlg.scale_pct = (SCALE_MIN as f64
                                + t * (SCALE_MAX - SCALE_MIN) as f64).round() as u32;
                            dlg.apply_scale();
                            *dlg.test.write() = TestState::NotRun;
                        } else if l.spp_slider.contains(cx, cy) {
                            self.dialog_drag = DialogDrag::SppSlider;
                            let t = ((cx - l.spp_slider.x) as f64
                                / l.spp_slider.w.max(1) as f64).clamp(0.0, 1.0);
                            let dlg = self.render_dialog.as_mut().unwrap();
                            dlg.spp = (SPP_MIN.ln()
                                + t * (SPP_MAX.ln() - SPP_MIN.ln())).exp();
                            *dlg.test.write() = TestState::NotRun;
                        }
                    }

                    if self.onion_enabled
                        && matches!(button, MouseButton::Left | MouseButton::Middle | MouseButton::Right)
                    {
                        let cur = self.sampler.samples.load(Ordering::Relaxed);
                        if self.onion.is_none() || cur >= MIN_ONION_SAMPLES {
                            let mut snap_rgba = vec![0u8;
                                (ws.fb_w as usize) * (ws.fb_h as usize) * 4];
                            let hist = self.sampler.hist_snapshot();
                            let max = self.sampler.max_count.load(Ordering::Relaxed).max(1);
                            palette::apply(
                                &hist,
                                max,
                                self.gamma,
                                &self.palette,
                                &mut snap_rgba,
                            );
                            self.onion = Some(Onion::capture(
                                &snap_rgba,
                                ws.fb_w,
                                ws.fb_h,
                                &self.view.read(),
                            ));
                        }
                    }
                }
                ElementState::Released => {
                    if self.mode == Mode::Composer {
                        self.composer.slider_drag =
                            composer::ComposerSliderDrag::None;
                        // A press on a stop marker arms `palette_drag_stop`.
                        // If the user released without moving, promote to a
                        // click and open the color popover for that stop.
                        let pending_stop = self.composer.palette_drag_stop.take();
                        if let Some(stop_idx) = pending_stop {
                            let did_move = self.drag.as_ref()
                                .map(|d| d.moved).unwrap_or(false);
                            if !did_move && button == MouseButton::Left {
                                self.composer.palette_color_edit = Some(stop_idx);
                            }
                        }
                        self.composer.color_channel_drag =
                            composer::ColorChannelDrag::None;
                        // Finalize a pane-splitter drag.
                        if self.composer.splitter_drag.take().is_some() {
                            return;
                        }
                        // Finalize a timeline-pan drag (middle-click).
                        if self.composer.timeline_pan_drag.take().is_some() {
                            return;
                        }
                        // Finalize timeline drag-reorder OR drag-off-to-remove.
                        if let Some(d) = self.composer.timeline_drag.take() {
                            if d.moved {
                                let tl = composer_timeline_rect(
                                    ws.fb_w, ws.fb_h, ws.scale_factor,
                                    &self.composer);
                                // Drop target outside the timeline panel →
                                // remove this keyframe.
                                if !tl.contains(d.cursor_x as i32, d.cursor_y as i32) {
                                    self.composer.remove_from_timeline(d.from_idx);
                                    return;
                                }
                                let (slots, _) = timeline_slot_rects(
                                    &self.composer, tl, ws.scale_factor);
                                let insert =
                                    timeline_insert_index_for_x(&slots, d.cursor_x as i32);
                                let from = d.from_idx;
                                let insert = insert.min(self.composer.timeline.len());
                                if insert != from && insert != from + 1 {
                                    let item = self.composer.timeline.remove(from);
                                    let ins = if insert > from { insert - 1 } else { insert };
                                    let ins = ins.min(self.composer.timeline.len());
                                    self.composer.timeline.insert(ins, item);
                                    self.composer.mark_preview_dirty();
                                    self.composer.mark_save_dirty();
                                }
                                return;
                            }
                        }
                        // If a dialog is open, its own handler runs below in
                        // the shared path (we fall through to it via the
                        // live-mode Released block). Otherwise, route to the
                        // composer release handler.
                        if self.render_dialog.is_none() && button == MouseButton::Left {
                            // Queue-row × buttons: need render_progress + queue
                            // which live on App, so hit-test before delegating.
                            let queue_click = {
                                let cx = self.cursor_pos.x as i32;
                                let cy = self.cursor_pos.y as i32;
                                let l = composer_layout(
                                    ws.fb_w, ws.fb_h, ws.scale_factor, &self.composer);
                                if l.render_panel.contains(cx, cy) {
                                    let rl = render_panel_layout(
                                        l.render_panel, ws.scale_factor);
                                    let active = self.render_progress.read().is_some();
                                    let (active_row, queued_rows) =
                                        render_queue_row_rects(
                                            &rl, active,
                                            self.render_queue.len(),
                                            l.render_panel,
                                            ws.scale_factor);
                                    let mut hit: Option<ComposerAction> = None;
                                    if let Some(row) = active_row.as_ref() {
                                        if row.x_btn.contains(cx, cy) {
                                            hit = Some(
                                                ComposerAction::CancelActiveRender);
                                        }
                                    }
                                    if hit.is_none() {
                                        for (i, row) in queued_rows.iter().enumerate() {
                                            if row.x_btn.contains(cx, cy) {
                                                hit = Some(
                                                    ComposerAction::RemoveQueued(i));
                                                break;
                                            }
                                        }
                                    }
                                    hit
                                } else { None }
                            };
                            if let Some(act) = queue_click {
                                match act {
                                    ComposerAction::CancelActiveRender => {
                                        if let Some(p) =
                                            self.render_progress.read().as_ref()
                                        {
                                            let _ = std::fs::write(
                                                p.session_dir.join(".cancel"), b"");
                                        }
                                    }
                                    ComposerAction::RemoveQueued(i) => {
                                        if i < self.render_queue.len() {
                                            self.render_queue.remove(i);
                                        }
                                    }
                                    _ => {}
                                }
                                return;
                            }
                            let action = composer_handle_release(
                                &mut self.composer,
                                self.cursor_pos,
                                ws.fb_w, ws.fb_h, ws.scale_factor,
                                &self.view,
                                &self.sampler,
                                self.gamma,
                            );
                            match action {
                                Some(ComposerAction::OpenRenderDialog) => {
                                    if let Some(msg) = missing_blocker_msg(&self.composer) {
                                        eprintln!("{msg}");
                                        *self.last_render.write() =
                                            RenderResult::Failed(msg);
                                    } else if self.composer.timeline.len() >= 2 {
                                        let keyframes: Vec<PathBuf> = self.composer.timeline
                                            .iter().map(|t| t.source.clone()).collect();
                                        let tensions: Vec<f64> = self.composer.timeline.iter()
                                            .map(|t| t.tension).collect();
                                        let segments: Vec<f64> = self.composer.timeline.iter()
                                            .map(|t| t.segment_seconds.max(0.05)).collect();
                                        self.render_dialog = Some(RenderDialogState {
                                            native_w: ws.fb_w,
                                            native_h: ws.fb_h,
                                            width_text: ws.fb_w.to_string(),
                                            height_text: ws.fb_h.to_string(),
                                            scale_pct: 100,
                                            spp: SPP_DEFAULT,
                                            focus: DialogFocus::None,
                                            test: Arc::new(RwLock::new(TestState::NotRun)),
                                            keyframes,
                                            tensions,
                                            segments,
                                        });
                                    }
                                }
                                Some(ComposerAction::RenderFromPanel) => {
                                    if let Some(msg) = missing_blocker_msg(&self.composer) {
                                        eprintln!("{msg}");
                                        *self.last_render.write() =
                                            RenderResult::Failed(msg);
                                    } else if let Some(job) = build_render_job_from_composer(&self.composer) {
                                        if self.render_progress.read().is_some() {
                                            self.render_queue.push(job);
                                        } else if let Err(e) = spawn_render_job_free(
                                            &job, &self.render_progress, &self.last_render,
                                        ) {
                                            eprintln!("error: render spawn (composer) failed: {e}");
                                            *self.last_render.write() =
                                                RenderResult::Failed(e.to_string());
                                        }
                                    }
                                }
                                Some(ComposerAction::CancelActiveRender)
                                | Some(ComposerAction::RemoveQueued(_)) => {
                                    // Handled by the queue_click hit-test above.
                                }
                                None => {}
                            }
                            return;
                        }
                        // Dialog open in composer mode: fall through to the
                        // shared dialog click handler below.
                    }
                    // Finalize a top-bar button press (save / pause / resume).
                    // Fires the click iff the cursor is still inside the same
                    // button at release. No `DragState` was armed for this
                    // press, so the rotation/pan path was never reachable.
                    if let Some(b) = self.live_button_press.take() {
                        if button == MouseButton::Left {
                            let cx = self.cursor_pos.x as i32;
                            let cy = self.cursor_pos.y as i32;
                            let still_over = match b {
                                LiveButton::Save => save_button_rect(
                                    ws.fb_w, ws.scale_factor).contains(cx, cy),
                                LiveButton::Pause => pause_button_rect(
                                    ws.fb_w, ws.scale_factor).contains(cx, cy),
                                LiveButton::Resume => resume_button_rect(
                                    ws.fb_w, ws.scale_factor).contains(cx, cy),
                            };
                            if still_over {
                                match b {
                                    LiveButton::Save => {
                                        self.save_pending = true;
                                        self.last_save = SaveResult::None;
                                    }
                                    LiveButton::Pause => {
                                        let p = &self.sampler.paused;
                                        let was = p.load(Ordering::Relaxed);
                                        p.store(!was, Ordering::Relaxed);
                                    }
                                    LiveButton::Resume => {
                                        if let Some(dir) = self.resume_candidate.clone() {
                                            if let Err(e) = self.spawn_resume(dir) {
                                                eprintln!("error: render resume failed: {e}");
                                                *self.last_render.write() =
                                                    RenderResult::Failed(e.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return;
                    }
                    // Finalize a live-HUD widget drag (palette stop, gamma
                    // slider, RGB channel). Promote a no-move stop press to
                    // "open color popover" — mirrors the composer's pattern.
                    if self.live_widget_drag != LiveWidgetDrag::None {
                        let drag = self.live_widget_drag;
                        self.live_widget_drag = LiveWidgetDrag::None;
                        let origin = self.live_widget_press_origin.take();
                        if let LiveWidgetDrag::Stop(stop_idx) = drag {
                            let did_move = origin.map(|o| {
                                let dx = self.cursor_pos.x - o.x;
                                let dy = self.cursor_pos.y - o.y;
                                dx * dx + dy * dy > 9.0
                            }).unwrap_or(false);
                            if !did_move && button == MouseButton::Left {
                                self.palette_color_edit = Some(stop_idx);
                            }
                        }
                        // Drop any DragState we created (none, in this path)
                        // so we don't fall through to the rotation undo push.
                        self.drag = None;
                        return;
                    }
                    if let Some(d) = self.drag.take() {
                        if self.dialog_drag != DialogDrag::None {
                            self.dialog_drag = DialogDrag::None;
                            return;
                        }
                        if self.mode == Mode::Live
                            && d.moved
                            && matches!(d.button, MouseButton::Left | MouseButton::Middle | MouseButton::Right)
                            && self.render_dialog.is_none()
                        {
                            push_view_snapshot_fn(
                                &self.view,
                                &mut self.undo_stack,
                                &mut self.undo_cursor,
                                &mut self.last_undo_push,
                            );
                        }
                        if !d.moved && button == MouseButton::Left {
                            let cx = self.cursor_pos.x as i32;
                            let cy = self.cursor_pos.y as i32;

                            if self.render_dialog.is_some() {
                                let (fbw, fbh, sf) = (ws.fb_w, ws.fb_h, ws.scale_factor);
                                let l = dialog_layout(fbw, fbh, sf);
                                let n_workers = self.sampler.n_workers;

                                // Compute action in a block so the dlg borrow
                                // ends before we mutate self.render_dialog.
                                let mut dismiss = false;
                                let mut render_job: Option<RenderJob> = None;
                                {
                                    let dlg = self.render_dialog.as_mut().unwrap();
                                    let kf0 = dlg.keyframes.first().cloned();
                                    if l.width_field.contains(cx, cy) {
                                        dlg.focus = DialogFocus::Width;
                                    } else if l.height_field.contains(cx, cy) {
                                        dlg.focus = DialogFocus::Height;
                                    } else if l.test_btn.contains(cx, cy) {
                                        let running = matches!(
                                            &*dlg.test.read(), TestState::Running);
                                        if !running {
                                            if let Some(kf) = kf0 {
                                                let (w, h) = (dlg.width(), dlg.height());
                                                let spf = dlg.samples_per_frame();
                                                let st = dlg.test.clone();
                                                *st.write() = TestState::Running;
                                                thread::spawn(move || {
                                                    run_test_frame(kf, w, h, spf, n_workers, st);
                                                });
                                            }
                                        }
                                    } else if l.cancel_btn.contains(cx, cy) {
                                        dismiss = true;
                                    } else if l.render_btn.contains(cx, cy) {
                                        let (w, h) = (dlg.width(), dlg.height());
                                        let spf = dlg.samples_per_frame();
                                        let kfs = dlg.keyframes.clone();
                                        let tensions = dlg.tensions.clone();
                                        let segments = dlg.segments.clone();
                                        dismiss = true;
                                        render_job = Some(
                                            RenderJob::new(kfs, w, h, VIDEO_FPS, spf)
                                                .with_schedule(tensions, segments)
                                        );
                                    } else if !l.outer.contains(cx, cy) {
                                        dismiss = true;
                                    } else {
                                        dlg.focus = DialogFocus::None;
                                    }
                                }

                                if dismiss {
                                    self.render_dialog = None;
                                }
                                if let Some(job) = render_job {
                                    if self.render_progress.read().is_some() {
                                        self.render_queue.push(job);
                                    } else if let Err(e) = spawn_render_job_free(
                                        &job, &self.render_progress, &self.last_render,
                                    ) {
                                        eprintln!("error: render spawn (dialog) failed: {e}");
                                        *self.last_render.write() =
                                            RenderResult::Failed(e.to_string());
                                    }
                                }
                            } else {
                                let v = self.view.read();
                                if let Some(target_r) = ws.gizmo.pick(
                                    self.cursor_pos.x as i32,
                                    self.cursor_pos.y as i32,
                                    &v,
                                ) {
                                    let from = v.rotation;
                                    drop(v);
                                    let already_there = (0..4).all(|i| {
                                        (0..4).all(|j| (from[i][j] - target_r[i][j]).abs() < 1e-9)
                                    });
                                    if !already_there {
                                        self.snap = Some(SnapAnim {
                                            from,
                                            to: target_r,
                                            start: Instant::now(),
                                            duration_ms: 250,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            },

            WindowEvent::MouseWheel { delta, .. } => {
                if self.mode == Mode::Composer {
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) =>
                            (x as f64 * 40.0, y as f64 * 40.0),
                        MouseScrollDelta::PixelDelta(p) => (p.x, p.y),
                    };
                    composer_handle_wheel(
                        &mut self.composer, ws.fb_w, ws.fb_h, ws.scale_factor,
                        self.cursor_pos, dx, dy);
                    return;
                }
                if self.render_dialog.is_some() { return; }
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                let factor = (-dy * 0.15).exp();
                self.view
                    .write()
                    .zoom_at(factor, (self.cursor_pos.x, self.cursor_pos.y));
                self.sampler.invalidate(InvalidationReason::ViewChange);
                self.push_view_snapshot();
            }

            WindowEvent::KeyboardInput { event: k, .. } => {
                if k.state != ElementState::Pressed {
                    return;
                }

                // Route input to focused text field.
                if let Some(dlg) = self.render_dialog.as_mut() {
                    if dlg.focus != DialogFocus::None {
                        match k.logical_key {
                            Key::Named(NamedKey::Escape) => dlg.focus = DialogFocus::None,
                            Key::Named(NamedKey::Backspace) => {
                                let buf = match dlg.focus {
                                    DialogFocus::Width => &mut dlg.width_text,
                                    DialogFocus::Height => &mut dlg.height_text,
                                    _ => return,
                                };
                                buf.pop();
                            }
                            Key::Named(NamedKey::Tab) => {
                                // Cycle between Width and Height; never drop focus.
                                dlg.focus = match dlg.focus {
                                    DialogFocus::Width => DialogFocus::Height,
                                    _ => DialogFocus::Width,
                                };
                                let pct = ((dlg.width() as f64 / dlg.native_w.max(1) as f64)
                                    * 100.0).round() as u32;
                                dlg.scale_pct = pct.clamp(SCALE_MIN, SCALE_MAX);
                            }
                            Key::Named(NamedKey::Enter) => {
                                dlg.focus = DialogFocus::None;
                                let pct = ((dlg.width() as f64 / dlg.native_w.max(1) as f64)
                                    * 100.0).round() as u32;
                                dlg.scale_pct = pct.clamp(SCALE_MIN, SCALE_MAX);
                            }
                            Key::Character(ref s) => {
                                let buf = match dlg.focus {
                                    DialogFocus::Width => &mut dlg.width_text,
                                    DialogFocus::Height => &mut dlg.height_text,
                                    _ => return,
                                };
                                for ch in s.chars() {
                                    if ch.is_ascii_digit() && buf.len() < 5 {
                                        buf.push(ch);
                                    }
                                }
                            }
                            _ => {}
                        }
                        return;
                    }
                }

                // Composer render-panel text-field input.
                if self.mode == Mode::Composer
                    && self.composer.rp_focus != composer::RenderFieldFocus::None
                {
                    let c = &mut self.composer;
                    match k.logical_key {
                        Key::Named(NamedKey::Escape) =>
                            c.rp_focus = composer::RenderFieldFocus::None,
                        Key::Named(NamedKey::Backspace) => {
                            let buf = match c.rp_focus {
                                composer::RenderFieldFocus::Width => &mut c.rp_width_text,
                                composer::RenderFieldFocus::Height => &mut c.rp_height_text,
                                _ => return,
                            };
                            buf.pop();
                            c.schedule_render_test();
                            c.mark_save_dirty();
                        }
                        Key::Named(NamedKey::Tab) => {
                            // Tab cycles between width and height; never drops
                            // focus.
                            c.rp_focus = match c.rp_focus {
                                composer::RenderFieldFocus::Width =>
                                    composer::RenderFieldFocus::Height,
                                _ =>
                                    composer::RenderFieldFocus::Width,
                            };
                            c.schedule_render_test();
                        }
                        Key::Named(NamedKey::Enter) => {
                            c.rp_focus = composer::RenderFieldFocus::None;
                            c.schedule_render_test();
                        }
                        Key::Character(ref s) => {
                            let buf = match c.rp_focus {
                                composer::RenderFieldFocus::Width => &mut c.rp_width_text,
                                composer::RenderFieldFocus::Height => &mut c.rp_height_text,
                                _ => return,
                            };
                            for ch in s.chars() {
                                if ch.is_ascii_digit() && buf.len() < 5 {
                                    buf.push(ch);
                                }
                            }
                            c.schedule_render_test();
                            c.mark_save_dirty();
                        }
                        _ => {}
                    }
                    return;
                }

                match k.logical_key {
                    Key::Named(NamedKey::Escape) => {
                        if self.render_dialog.is_some() {
                            self.render_dialog = None;
                        } else if self.composer.load_dialog.is_some() {
                            self.composer.close_load_dialog();
                        } else if self.composer.context_menu.is_some() {
                            self.composer.context_menu = None;
                        } else {
                            event_loop.exit();
                        }
                    }
                    Key::Named(NamedKey::Space) => {
                        let p = &self.sampler.paused;
                        let was = p.load(Ordering::Relaxed);
                        p.store(!was, Ordering::Relaxed);
                    }
                    // Undo/redo: meta+z / meta+shift+z. Key on the physical
                    // Z keycode so shift-shifted logical chars ("Z" vs "z")
                    // don't matter — some compositors don't apply shift to
                    // the logical key when Super is held.
                    _ if self.mods_super
                        && k.physical_key == PhysicalKey::Code(KeyCode::KeyZ) =>
                    {
                        if self.mods_shift {
                            self.redo();
                        } else {
                            self.undo();
                        }
                    }
                    Key::Character(ref s) => match s.as_str() {
                        "q" | "Q" => event_loop.exit(),
                        "s" | "S" if self.mods_super => {
                            self.save_pending = true;
                            self.last_save = SaveResult::None;
                        }
                        "h" | "H" => self.help_visible = !self.help_visible,
                        "o" | "O" => self.onion_enabled = !self.onion_enabled,
                        "a" | "A" => self.show_axes = !self.show_axes,
                        "p" => {
                            let n = palette::builtin_stops().len();
                            self.load_palette_preset((self.palette_preset_idx + 1) % n);
                        }
                        "P" => {
                            let n = palette::builtin_stops().len();
                            self.load_palette_preset(
                                (self.palette_preset_idx + n - 1) % n);
                        }
                        "g" => {
                            self.gamma = (self.gamma - 0.05).max(0.1);
                        }
                        "G" => {
                            self.gamma = (self.gamma + 0.05).min(2.5);
                        }
                        "r" | "R" => {
                            self.view.write().reset();
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                            self.push_view_snapshot();
                        }
                        "[" | "{" => {
                            let mut v = self.view.write();
                            let delta = if self.mods_shift {
                                -100
                            } else {
                                -(v.max_iter as i32 / 2)
                            };
                            let mi = (v.max_iter as i32 + delta).max(50) as u32;
                            v.set_max_iter(mi);
                            drop(v);
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                            self.push_view_snapshot();
                        }
                        "]" | "}" => {
                            let mut v = self.view.write();
                            let delta = if self.mods_shift {
                                100
                            } else {
                                v.max_iter as i32
                            };
                            let mi = (v.max_iter as i32 + delta).min(8000) as u32;
                            v.set_max_iter(mi);
                            drop(v);
                            self.sampler.invalidate(InvalidationReason::ViewChange);
                            self.push_view_snapshot();
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                self.redraw(event_loop);
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ws) = &self.ws {
            ws.window.request_redraw();
        }
    }
}

impl App {
    /// Record the current view in the undo stack. Uses `push_view_snapshot_fn`
    /// so this works from contexts that hold a disjoint `&mut` on another
    /// field of `self` (e.g. `self.ws`).
    fn push_view_snapshot(&mut self) {
        push_view_snapshot_fn(
            &self.view,
            &mut self.undo_stack,
            &mut self.undo_cursor,
            &mut self.last_undo_push,
        );
    }

    /// Rebuild `self.palette` (1024-LUT) from the current `palette_stops`.
    /// Call after any stop edit (move, add, delete, color change).
    fn rebuild_palette(&mut self) {
        if self.palette_stops.len() < 2 {
            self.palette_stops = composer::default_palette_stops();
        }
        self.palette_stops.sort_by(|a, b|
            a.pos.partial_cmp(&b.pos).unwrap_or(std::cmp::Ordering::Equal));
        let stops: Vec<(f32, [u8; 3])> = self.palette_stops.iter()
            .map(|s| (s.pos, s.rgb)).collect();
        let name = self.palette.name;
        self.palette = Palette::from_stops_dyn(name, &stops);
    }

    /// Replace the live palette stops with preset `idx` from
    /// `palette::builtin_stops()` (cycled by the `p` / shift-`p` keys).
    fn load_palette_preset(&mut self, idx: usize) {
        let presets = palette::builtin_stops();
        if presets.is_empty() { return; }
        let idx = idx % presets.len();
        let (name, stops) = presets[idx].clone();
        self.palette_preset_idx = idx;
        self.palette_stops = stops.iter()
            .map(|(p, rgb)| composer::PaletteStop { pos: *p, rgb: *rgb })
            .collect();
        self.palette = Palette::from_stops_dyn(name, &stops);
        self.palette_color_edit = None;
    }

    /// Load the camera state stored in `path`'s PNG metadata into the live
    /// view, switch to Live mode, and invalidate the sampler. Width/height
    /// keep the current framebuffer dims (the screenshot's dims aren't
    /// relevant to the live preview window). No-op if the spec can't be
    /// parsed.
    fn use_screenshot_in_live(&mut self, path: &std::path::Path) {
        let cached = self.composer.thumbs.read().get(path)
            .and_then(|t| t.spec.clone());
        let spec = cached.or_else(|| crate::metadata::read_spec(path, 0, 1).ok());
        let Some(spec) = spec else { return };
        self.push_view_snapshot();
        {
            let mut v = self.view.write();
            v.center = spec.view.center;
            v.half_width = spec.view.half_width;
            v.set_max_iter(spec.view.max_iter);
            // `set_rotation` always bumps `generation`, so this last call
            // ensures workers preempt their current batch cleanly.
            v.set_rotation(spec.view.rotation);
        }
        self.sampler.invalidate(InvalidationReason::ViewChange);
        self.push_view_snapshot();
        self.mode = Mode::Live;
    }

    fn undo(&mut self) {
        if self.undo_cursor == 0 { return; }
        self.undo_cursor -= 1;
        self.apply_snapshot_at_cursor();
    }

    fn redo(&mut self) {
        if self.undo_cursor + 1 >= self.undo_stack.len() { return; }
        self.undo_cursor += 1;
        self.apply_snapshot_at_cursor();
    }

    fn apply_snapshot_at_cursor(&mut self) {
        let snap = self.undo_stack[self.undo_cursor].clone();
        let mut v = self.view.write();
        v.center = snap.center;
        v.half_width = snap.half_width;
        v.set_max_iter(snap.max_iter);
        v.set_rotation(snap.rotation);
        drop(v);
        // Reset the coalesce timer so the next recorded move starts fresh.
        self.last_undo_push = None;
        self.sampler.invalidate(InvalidationReason::ViewChange);
    }

    /// Look for unfinished render sessions on disk. Adopt a live one as our
    /// current render (start polling its session.json), or stash a dead one
    /// as a resume candidate so the user can pick it up.
    fn scan_for_sessions(&mut self) {
        let sessions = session::scan_sessions();
        for (dir, sess) in sessions {
            if sess.cur_frame >= sess.total_frames {
                // Frames finished; ffmpeg assembly may still be running or
                // the session dir may be in teardown — ignore either way.
                continue;
            }
            let alive = sess.pid.map_or(false, session::pid_alive);
            if alive {
                eprintln!("adopting live render session: {}", dir.display());
                start_session_watcher(
                    dir.clone(),
                    self.render_progress.clone(),
                    self.last_render.clone(),
                );
                return;
            } else {
                eprintln!("found resumable render session: {}", dir.display());
                self.resume_candidate = Some(dir);
                return;
            }
        }
    }

    /// If a render-panel debounce has elapsed and no test is already in
    /// flight, spawn a one-frame test render on a worker thread and update
    /// `composer.rp_test` with the measurement. Quick (scale 15 %, spp 1.0)
    /// so the estimate can be extrapolated cheaply.
    fn maybe_run_render_test(&mut self) {
        let Some(due) = self.composer.rp_test_due else { return };
        if Instant::now() < due { return; }
        // Drop the pending flag first so we don't re-enter on subsequent
        // ticks.
        self.composer.rp_test_due = None;
        // Already running? Skip.
        {
            let t = self.composer.rp_test.read();
            if matches!(&*t, composer::TestStatus::Running) { return; }
        }
        if self.composer.timeline.len() < 2 { return; }
        // Use a small scale + spp to keep the test itself snappy; the ratio
        // is then extrapolated to total-frame time.
        let w_full = self.composer.rp_width();
        let h_full = self.composer.rp_height();
        let spp_full = self.composer.rp_spp.max(0.1);
        let test_scale = 0.15_f64;
        let test_spp = 1.0_f64.min(spp_full);
        let tw = ((w_full as f64 * test_scale).round() as u32).max(64) & !1;
        let th = ((h_full as f64 * test_scale).round() as u32).max(64) & !1;
        let test_samples = ((tw as f64 * th as f64 * test_spp).round() as u64)
            .max(20_000);
        let pixel_ratio =
            (w_full as f64 * h_full as f64 * spp_full)
            / (tw as f64 * th as f64 * test_spp).max(1.0);
        // Build a FrameSpec from the first keyframe — good enough for a
        // representative measurement.
        let first = &self.composer.timeline[0];
        let mut view = first.spec.view.clone();
        view.width = tw;
        view.height = th;
        let spec = crate::render::FrameSpec {
            view,
            palette: self.composer.palette.clone(),
            gamma: first.spec.gamma,
            samples_target: test_samples,
            n_workers: (self.sampler.n_workers / 2).max(1),
        };
        let out = self.composer.rp_test.clone();
        let preview_active = self.composer.preview_active.clone();
        *out.write() = composer::TestStatus::Running;
        // Suspend the preview worker so the test gets the full machine to
        // itself; this is what makes the estimate stable across clicks.
        preview_active.store(false, Ordering::Relaxed);
        thread::spawn(move || {
            let start = Instant::now();
            let _rgba = crate::render::render_frame(&spec, None);
            let elapsed = start.elapsed();
            let scaled = elapsed.mul_f64(pixel_ratio);
            *out.write() = composer::TestStatus::Done { frame_time: scaled };
            preview_active.store(true, Ordering::Relaxed);
        });
    }

    /// Called each redraw: if nothing is running, start the next queued job.
    fn drain_queue_if_idle(&mut self) {
        if self.render_progress.read().is_some() { return; }
        if self.render_queue.is_empty() { return; }
        let job = self.render_queue.remove(0);
        if let Err(e) = spawn_render_job_free(
            &job, &self.render_progress, &self.last_render,
        ) {
            eprintln!("error: render spawn (queue drain) failed: {e}");
            *self.last_render.write() = RenderResult::Failed(e.to_string());
        }
    }

    fn spawn_resume(&mut self, session_dir: PathBuf) -> std::io::Result<()> {
        let exe = std::env::current_exe()?;
        let mut cmd = Command::new(&exe);
        cmd.arg("render-video")
            .arg("--resume").arg(&session_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let _child = cmd.spawn()?;

        // Seed progress from the on-disk session state so the HUD doesn't
        // flash 0/N.
        if let Ok(s) = SessionFile::read(&session_dir) {
            let sys_now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(s.started_at);
            let ago = sys_now.saturating_sub(s.started_at);
            *self.render_progress.write() = Some(RenderProgress {
                cur: s.cur_frame,
                total: s.total_frames,
                started: Instant::now() - Duration::from_secs(ago),
                fb_w: s.width,
                fb_h: s.height,
                fps: s.fps,
                samples_per_frame: s.samples_per_frame,
                session_dir: session_dir.clone(),
                output: s.output.clone(),
            });
            *self.last_render.write() = RenderResult::None;
        }

        self.resume_candidate = None;
        start_session_watcher(
            session_dir,
            self.render_progress.clone(),
            self.last_render.clone(),
        );
        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        // Start any queued renders that can begin now.
        self.drain_queue_if_idle();

        // Composer: advance playhead and kick a preview render if needed.
        if self.mode == Mode::Composer {
            self.composer.tick(Instant::now());
            // Detect Running→Done on the render-time-estimate test and dirty
            // the preview so it re-renders the frame the test cancelled.
            let test_running = matches!(
                *self.composer.rp_test.read(),
                composer::TestStatus::Running);
            if self.composer_test_was_running && !test_running {
                self.composer.mark_preview_dirty();
            }
            self.composer_test_was_running = test_running;
            if let Some(ws_ref) = self.ws.as_ref() {
                let l = composer_layout(ws_ref.fb_w, ws_ref.fb_h,
                    ws_ref.scale_factor, &self.composer);
                let n = self.sampler.n_workers;
                self.composer.request_preview_if_dirty(l.preview.w, l.preview.h, n);
            }
            // Render-panel auto-test: debounced measurement of one frame to
            // drive the estimate line.
            self.maybe_run_render_test();
        }

        let Some(ws) = self.ws.as_mut() else { return };

        // Advance snap animation.
        if let Some(anim) = self.snap.as_ref() {
            let elapsed = anim.start.elapsed().as_millis() as f64;
            let t = (elapsed / anim.duration_ms as f64).min(1.0);
            let eased = ease_in_out(t);
            let r = crate::gizmo::interp_rotation(&anim.from, &anim.to, eased);
            self.view.write().set_rotation(r);
            self.sampler.invalidate(InvalidationReason::ViewChange);
            if t >= 1.0 {
                self.view.write().set_rotation(anim.to);
                self.snap = None;
                self.sampler.invalidate(InvalidationReason::ViewChange);
                push_view_snapshot_fn(
                    &self.view,
                    &mut self.undo_stack,
                    &mut self.undo_cursor,
                    &mut self.last_undo_push,
                );
            }
        }

        // Samples/sec rolling.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().max(1e-6);
        self.last_frame = now;
        let cur_samples = self.sampler.samples.load(Ordering::Relaxed);
        let ds = cur_samples.saturating_sub(self.last_sample_count);
        self.last_sample_count = cur_samples;
        let inst = ds as f64 / dt;
        self.samples_per_sec = self.samples_per_sec * 0.8 + inst * 0.2;

        // Palette → RGBA work buffer (live mode). In composer mode we
        // overwrite with composer UI below, so skip live drawing entirely.
        if self.mode == Mode::Live {
            let hist = self.sampler.hist_snapshot();
            let max = self.sampler.max_count.load(Ordering::Relaxed).max(1);
            palette::apply(
                &hist,
                max,
                self.gamma,
                &self.palette,
                &mut ws.rgba,
            );
        } else {
            // Paint composer background.
            for px in ws.rgba.chunks_exact_mut(4) {
                px[0] = 18; px[1] = 22; px[2] = 30; px[3] = 255;
            }
        }

        // Save PNG now (clean frame, no overlays).
        if self.save_pending {
            self.save_pending = false;
            let v = self.view.read();
            match savepng::save_png(
                &ws.rgba,
                ws.fb_w,
                ws.fb_h,
                savepng::SaveMeta {
                    view: &v,
                    palette_name: self.palette.name,
                    gamma: self.gamma,
                    samples: cur_samples,
                },
            ) {
                Ok(p) => {
                    eprintln!("saved {}", p.display());
                    self.last_save = SaveResult::Ok;
                    self.keyframes.push(p);
                }
                Err(e) => {
                    eprintln!("save failed: {}", e);
                    self.last_save = SaveResult::Failed;
                }
            }
        }

        // Onion skin during active drag (under overlays).
        let dragging_active = self
            .drag
            .as_ref()
            .map(|d| {
                d.moved && matches!(d.button, MouseButton::Left | MouseButton::Middle | MouseButton::Right)
            })
            .unwrap_or(false);
        if self.onion_enabled && dragging_active {
            if let Some(o) = &self.onion {
                o.blit_over(&mut ws.rgba, ws.fb_w, ws.fb_h, &self.view.read(), ONION_FADE);
            }
        }

        // Composer mode: draw the composer UI in place of live overlays.
        if self.mode == Mode::Composer {
            let progress_snap = self.render_progress.read().clone();
            draw_composer(
                &mut self.text,
                &mut ws.rgba,
                ws.fb_w,
                ws.fb_h,
                ws.scale_factor,
                &self.composer,
                progress_snap,
                &self.render_queue,
                self.cursor_pos,
            );
            if let Some(dlg) = &self.render_dialog {
                draw_render_dialog(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                    dlg.keyframes.len(), dlg, ws.scale_factor,
                );
            }
            draw_tab_bar(
                &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                ws.scale_factor, self.mode,
            );
            {
                let progress = self.render_progress.read();
                let last = self.last_render.read();
                draw_render_status(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h, ws.scale_factor,
                    progress.as_ref(), self.render_queue.len(), &last,
                );
            }
            // Copy RGBA → softbuffer and return.
            let mut sb = match ws.surface.buffer_mut() {
                Ok(b) => b,
                Err(_) => { event_loop.exit(); return; }
            };
            let n = (ws.fb_w as usize) * (ws.fb_h as usize);
            for i in 0..n {
                let r = ws.rgba[i * 4] as u32;
                let g = ws.rgba[i * 4 + 1] as u32;
                let b = ws.rgba[i * 4 + 2] as u32;
                sb[i] = (r << 16) | (g << 8) | b;
            }
            let _ = sb.present();
            return;
        }

        // Overlays.
        {
            let v = self.view.read();
            if self.show_axes {
                draw_axes_overlay(&v, &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h, ws.scale_factor);
            }
            ws.gizmo.draw(&v, &mut ws.rgba, ws.fb_w, ws.fb_h);

            let mut lines = hud_lines(
                &v, &self.palette, self.gamma,
                cur_samples, self.samples_per_sec, self.sampler.n_workers,
            );
            if !self.keyframes.is_empty() {
                lines.push(format!("keyframes={}", self.keyframes.len()));
            }
            let progress_snap = self.render_progress.read().clone();
            if let Some(p) = progress_snap {
                let pct = if p.total > 0 { p.cur as f64 * 100.0 / p.total as f64 } else { 0.0 };
                let elapsed = p.started.elapsed();
                let rate = if elapsed.as_secs_f64() > 0.1 {
                    p.cur as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let eta_str = if p.cur > 0 {
                    let eta_s = elapsed.as_secs_f64() * (p.total - p.cur) as f64 / p.cur as f64;
                    fmt_dur(Duration::from_secs_f64(eta_s.max(0.0)))
                } else {
                    String::from("…")
                };
                lines.push(format!(
                    "rendering {}/{} ({:.1}%)  —  {} elapsed, eta {}",
                    p.cur, p.total, pct, fmt_dur(elapsed), eta_str
                ));
                lines.push(format!(
                    "  {}×{} @ {}fps  •  {} samples/frame  •  {:.2} fps",
                    p.fb_w, p.fb_h, p.fps, human_count(p.samples_per_frame), rate
                ));
            } else {
                let lr = self.last_render.read().clone();
                match lr {
                    RenderResult::Ok(d) => {
                        let avg = if d.total_frames > 0 {
                            d.total_frames as f64 / d.elapsed.as_secs_f64().max(1e-6)
                        } else {
                            0.0
                        };
                        let size_str = d.file_size.map(fmt_bytes)
                            .unwrap_or_else(|| "?".to_string());
                        lines.push(format!("video: {}", d.path.display()));
                        lines.push(format!(
                            "  {} frames in {}  •  render {:.2} fps  •  {}",
                            d.total_frames, fmt_dur(d.elapsed), avg, size_str
                        ));
                        lines.push(format!(
                            "  {}×{} @ {}fps  •  {} samples/frame",
                            d.fb_w, d.fb_h, d.fps, human_count(d.samples_per_frame)
                        ));
                    }
                    RenderResult::Failed(e) => {
                        lines.push(format!("render failed: {}", e));
                    }
                    RenderResult::None => {}
                }
            }
            lines.push(String::from("(press 'h' for help)"));

            draw_hud(&mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h, &lines, ws.scale_factor);
            {
                let layout = live_palette_hud_layout(ws.fb_w, ws.fb_h, ws.scale_factor);
                draw_live_palette_hud(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                    ws.scale_factor, &layout,
                    &self.palette, &self.palette_stops,
                    self.palette_color_edit, self.gamma,
                );
            }
            draw_save_button(
                &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                ws.scale_factor, self.last_save,
            );
            {
                let paused = self.sampler.paused.load(Ordering::Relaxed);
                let r = pause_button_rect(ws.fb_w, ws.scale_factor);
                let (fill, border, label) = if paused {
                    ([60u8, 40, 16, 230], [220u8, 170, 90, 240], "paused")
                } else {
                    ([24u8, 30, 40, 220], [120u8, 140, 170, 230], "pause")
                };
                draw_labeled_button(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                    r, label, ws.scale_factor, fill, border, [230, 235, 245],
                );
            }
            if self.resume_candidate.is_some() && self.render_progress.read().is_none() {
                let r = resume_button_rect(ws.fb_w, ws.scale_factor);
                draw_labeled_button(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                    r, "resume", ws.scale_factor,
                    [20, 40, 52, 230], [120, 180, 200, 240], [220, 240, 245],
                );
            }
            if let Some(dlg) = &self.render_dialog {
                draw_render_dialog(
                    &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
                    self.keyframes.len(), dlg, ws.scale_factor,
                );
            }
            if self.help_visible {
                draw_help(&mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h, ws.scale_factor);
            }
        }

        draw_tab_bar(
            &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h,
            ws.scale_factor, self.mode,
        );
        {
            let progress = self.render_progress.read();
            let last = self.last_render.read();
            draw_render_status(
                &mut self.text, &mut ws.rgba, ws.fb_w, ws.fb_h, ws.scale_factor,
                progress.as_ref(), self.render_queue.len(), &last,
            );
        }

        // Copy RGBA → softbuffer native u32 (0x00RRGGBB).
        let mut sb = match ws.surface.buffer_mut() {
            Ok(b) => b,
            Err(_) => {
                event_loop.exit();
                return;
            }
        };
        let n = (ws.fb_w as usize) * (ws.fb_h as usize);
        for i in 0..n {
            let r = ws.rgba[i * 4] as u32;
            let g = ws.rgba[i * 4 + 1] as u32;
            let b = ws.rgba[i * 4 + 2] as u32;
            sb[i] = (r << 16) | (g << 8) | b;
        }
        let _ = sb.present();
    }
}

fn gizmo_rect_for(w: u32, h: u32, scale: f64) -> Rect {
    let size = ((GIZMO_BASE_PX as f64) * scale).round() as u32;
    let size = size.min(w / 4).min(h / 4).max(60);
    let margin = ((MARGIN_BASE as f64) * scale).round() as i32;
    Rect {
        x: (w as i32 - size as i32 - margin),
        y: (h as i32 - size as i32 - margin),
        w: size,
        h: size,
    }
}

const TAB_BAR_H: u32 = 30;

fn tab_bar_rects(fb_w: u32, scale: f64) -> [Rect; 2] {
    let bh = ((TAB_BAR_H as f64) * scale).round() as u32;
    let bw = (110.0 * scale).round() as u32;
    let margin = (6.0 * scale).round() as i32;
    let total = bw as i32 * 2 + margin;
    let x0 = (fb_w as i32 - total) / 2;
    [
        Rect { x: x0, y: (4.0 * scale).round() as i32, w: bw, h: bh },
        Rect { x: x0 + bw as i32 + margin, y: (4.0 * scale).round() as i32, w: bw, h: bh },
    ]
}

fn save_button_rect(fb_w: u32, scale: f64) -> Rect {
    let bw = ((BUTTON_BASE_W as f64) * scale).round() as u32;
    let bh = ((BUTTON_BASE_H as f64) * scale).round() as u32;
    let margin = ((MARGIN_BASE as f64) * scale).round() as i32;
    // Start toolbar below the tab bar.
    let tab_bottom = ((4.0 + TAB_BAR_H as f64) * scale).round() as i32;
    Rect {
        x: fb_w as i32 - bw as i32 - margin,
        y: tab_bottom + (6.0 * scale).round() as i32,
        w: bw,
        h: bh,
    }
}

/// Small persistent render status widget in the top-right, visually a sibling
/// of the tab bar. Returns the widget rect (always drawn; callers don't need
/// it for hit-testing — the widget is non-interactive).
fn render_status_rect(fb_w: u32, scale: f64) -> Rect {
    let h = ((TAB_BAR_H as f64) * scale).round() as u32;
    let w = (420.0 * scale).round() as u32;
    let margin = (6.0 * scale).round() as i32;
    Rect {
        x: fb_w as i32 - w as i32 - margin,
        y: (4.0 * scale).round() as i32,
        w, h,
    }
}

fn draw_render_status(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32, scale: f64,
    progress: Option<&RenderProgress>,
    queue_len: usize,
    last: &RenderResult,
) {
    // If nothing to show, skip drawing entirely.
    let has_progress = progress.is_some();
    let has_queue = queue_len > 0;
    let has_last = !matches!(last, RenderResult::None);
    if !has_progress && !has_queue && !has_last { return; }

    let r = render_status_rect(fb_w, scale);
    overlay::fill_rect(frame, fb_w, fb_h, r.x, r.y, r.w, r.h, [8, 12, 22, 230]);
    draw_button_frame(frame, fb_w, fb_h, r,
        [8, 12, 22, 230], [80, 100, 130, 220]);

    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let pad = (8.0 * scale).round() as i32;
    let line_h = text.line_height(px) as i32;

    if let Some(p) = progress {
        // Line 1: counter + queue suffix.
        let pct = if p.total > 0 { p.cur as f64 * 100.0 / p.total as f64 } else { 0.0 };
        let line = if has_queue {
            format!("{}/{} ({:.0}%)  •  +{} queued", p.cur, p.total, pct, queue_len)
        } else {
            format!("{}/{}  ({:.0}%)", p.cur, p.total, pct)
        };
        text.draw_sized(frame, fb_w, fb_h, &line,
            r.x + pad, r.y + line_h, px, [220, 230, 245]);

        // Progress bar at the bottom.
        let bar_h = (4.0 * scale).round() as u32;
        let bar = Rect {
            x: r.x + pad,
            y: r.y + r.h as i32 - bar_h as i32 - 4,
            w: r.w.saturating_sub(pad as u32 * 2),
            h: bar_h,
        };
        overlay::fill_rect(frame, fb_w, fb_h, bar.x, bar.y, bar.w, bar.h,
            [25, 35, 50, 255]);
        let frac = (p.cur as f64 / p.total.max(1) as f64).clamp(0.0, 1.0);
        let filled = ((bar.w as f64) * frac).round() as u32;
        if filled > 0 {
            overlay::fill_rect(frame, fb_w, fb_h, bar.x, bar.y, filled, bar.h,
                [90, 210, 130, 255]);
        }
    } else if has_queue {
        let s = format!("{} render{} queued",
            queue_len, if queue_len == 1 { "" } else { "s" });
        text.draw_sized(frame, fb_w, fb_h, &s,
            r.x + pad, r.y + r.h as i32 / 2 + line_h / 2 - 4,
            px, [200, 210, 235]);
    } else {
        // Last render summary. Middle-truncate the variable portion so the
        // "last:"/"failed:" prefix stays visible while long filenames or
        // error messages collapse to fit the interior width.
        let (prefix, tail) = match last {
            RenderResult::Ok(d) => ("last: ", d.path.file_name()
                .and_then(|n| n.to_str()).unwrap_or("render").to_string()),
            RenderResult::Failed(e) => ("failed: ", e.clone()),
            RenderResult::None => ("", String::new()),
        };
        let interior = r.w.saturating_sub(pad as u32 * 2);
        let prefix_w = text.measure(prefix, px);
        let tail_max = interior.saturating_sub(prefix_w);
        let tail_fit = truncate_middle_to_fit(text, &tail, tail_max, px);
        let s = format!("{}{}", prefix, tail_fit);
        text.draw_sized(frame, fb_w, fb_h, &s,
            r.x + pad, r.y + r.h as i32 / 2 + line_h / 2 - 4,
            px, [170, 180, 200]);
    }
}

struct DialogLayout {
    outer: Rect,
    title_y: i32,
    info_y: i32,
    width_label_y: i32,
    width_field: Rect,
    height_label_y: i32,
    height_field: Rect,
    scale_label_y: i32,
    scale_slider: Rect,
    spp_label_y: i32,
    spp_slider: Rect,
    eta_y: i32,
    test_btn: Rect,
    cancel_btn: Rect,
    render_btn: Rect,
}

fn dialog_layout(fb_w: u32, fb_h: u32, scale: f64) -> DialogLayout {
    let w = ((540.0 * scale).round() as u32).min(fb_w.saturating_sub(40));
    let h = ((400.0 * scale).round() as u32).min(fb_h.saturating_sub(40));
    let outer = Rect {
        x: (fb_w as i32 - w as i32) / 2,
        y: (fb_h as i32 - h as i32) / 2,
        w,
        h,
    };
    let pad = (14.0 * scale).round() as i32;
    let row_h = (26.0 * scale).round() as i32;
    let label_w = (100.0 * scale).round() as i32;
    let field_w = (100.0 * scale).round() as u32;
    let slider_w = outer.w.saturating_sub(pad as u32 * 2 + label_w as u32 + (70.0 * scale).round() as u32);
    let btn_h = (32.0 * scale).round() as u32;
    let btn_w = (110.0 * scale).round() as u32;

    let mut y = outer.y + pad;
    let title_y = y + row_h - 6;
    y += row_h;
    let info_y = y + row_h - 6;
    y += (row_h as f64 * 1.4).round() as i32;

    let width_label_y = y + row_h - 6;
    let width_field = Rect {
        x: outer.x + pad + label_w,
        y,
        w: field_w,
        h: row_h as u32 - 2,
    };
    y += row_h;
    let height_label_y = y + row_h - 6;
    let height_field = Rect {
        x: outer.x + pad + label_w,
        y,
        w: field_w,
        h: row_h as u32 - 2,
    };
    y += row_h;
    let scale_label_y = y + row_h - 6;
    let scale_slider = Rect {
        x: outer.x + pad + label_w,
        y: y + 4,
        w: slider_w,
        h: (row_h as u32).saturating_sub(8),
    };
    y += (row_h as f64 * 1.5).round() as i32;

    let spp_label_y = y + row_h - 6;
    y += row_h;
    let spp_slider = Rect {
        x: outer.x + pad,
        y: y + 4,
        w: outer.w.saturating_sub(pad as u32 * 2 + (90.0 * scale).round() as u32),
        h: (row_h as u32).saturating_sub(8),
    };
    y += (row_h as f64 * 1.4).round() as i32;

    let eta_y = y + row_h - 6;
    let _ = y;

    let btn_gap = (12.0 * scale).round() as i32;
    let btns_total = (btn_w as i32) * 3 + btn_gap * 2;
    let btn_y = outer.y + outer.h as i32 - btn_h as i32 - pad;
    let btn_x0 = outer.x + (outer.w as i32 - btns_total) / 2;
    let test_btn   = Rect { x: btn_x0, y: btn_y, w: btn_w, h: btn_h };
    let cancel_btn = Rect { x: btn_x0 + btn_w as i32 + btn_gap, y: btn_y, w: btn_w, h: btn_h };
    let render_btn = Rect { x: btn_x0 + 2 * (btn_w as i32 + btn_gap), y: btn_y, w: btn_w, h: btn_h };

    DialogLayout {
        outer,
        title_y, info_y,
        width_label_y, width_field,
        height_label_y, height_field,
        scale_label_y, scale_slider,
        spp_label_y, spp_slider,
        eta_y,
        test_btn, cancel_btn, render_btn,
    }
}

fn pause_button_rect(fb_w: u32, scale: f64) -> Rect {
    let save = save_button_rect(fb_w, scale);
    let bw = (72.0 * scale).round() as u32;
    let gap = (6.0 * scale).round() as i32;
    Rect { x: save.x - gap - bw as i32, y: save.y, w: bw, h: save.h }
}

/// Left of the pause button; visible only when a resumable session exists.
fn resume_button_rect(fb_w: u32, scale: f64) -> Rect {
    let pause = pause_button_rect(fb_w, scale);
    let bw = (90.0 * scale).round() as u32;
    let gap = (6.0 * scale).round() as i32;
    Rect { x: pause.x - gap - bw as i32, y: pause.y, w: bw, h: pause.h }
}

fn draw_render_dialog(
    text: &mut TextRenderer,
    frame: &mut [u8],
    fb_w: u32, fb_h: u32,
    kf_count: usize,
    dlg: &RenderDialogState,
    scale: f64,
) {
    let l = dialog_layout(fb_w, fb_h, scale);
    // Dim background.
    overlay::fill_rect(frame, fb_w, fb_h, 0, 0, fb_w, fb_h, [0, 0, 0, 110]);
    // Dialog panel + border.
    overlay::fill_rect(frame, fb_w, fb_h, l.outer.x, l.outer.y, l.outer.w, l.outer.h,
        [14, 18, 26, 240]);
    draw_button_frame(frame, fb_w, fb_h, l.outer,
        [14, 18, 26, 240], [180, 190, 210, 230]);

    let title_px = ((HELP_BASE_PX as f64) * scale).round() as u32;
    let body_px  = ((HUD_BASE_PX  as f64) * scale).round() as u32;
    let pad = (14.0 * scale).round() as i32;
    let col2 = l.outer.x + pad + (100.0 * scale).round() as i32;

    // Title + info.
    let title = format!("render {} frames to video", kf_count);
    text.draw_sized(frame, fb_w, fb_h, &title, l.outer.x + pad, l.title_y,
                    title_px, [235, 238, 245]);
    let duration_s = 2.0 * (kf_count.saturating_sub(1)) as f64;
    let total_frames = (duration_s * VIDEO_FPS as f64).round().max(2.0) as u32;
    let info = format!(
        "@ {} fps  •  {:.1}s  •  {} frames",
        VIDEO_FPS, duration_s, total_frames
    );
    text.draw_sized(frame, fb_w, fb_h, &info, l.outer.x + pad, l.info_y,
                    body_px, [175, 190, 210]);

    // Width/Height labels + fields.
    text.draw_sized(frame, fb_w, fb_h, "width",
        l.outer.x + pad, l.width_label_y, body_px, [205, 215, 230]);
    draw_text_field(text, frame, fb_w, fb_h, l.width_field,
        &dlg.width_text, dlg.focus == DialogFocus::Width, body_px);
    text.draw_sized(frame, fb_w, fb_h, "px",
        col2 + l.width_field.w as i32 + (8.0 * scale).round() as i32,
        l.width_label_y, body_px, [140, 150, 170]);

    text.draw_sized(frame, fb_w, fb_h, "height",
        l.outer.x + pad, l.height_label_y, body_px, [205, 215, 230]);
    draw_text_field(text, frame, fb_w, fb_h, l.height_field,
        &dlg.height_text, dlg.focus == DialogFocus::Height, body_px);
    text.draw_sized(frame, fb_w, fb_h, "px",
        col2 + l.height_field.w as i32 + (8.0 * scale).round() as i32,
        l.height_label_y, body_px, [140, 150, 170]);

    // Scale slider.
    text.draw_sized(frame, fb_w, fb_h, "scale",
        l.outer.x + pad, l.scale_label_y, body_px, [205, 215, 230]);
    let scale_t = (dlg.scale_pct as f64 - SCALE_MIN as f64) / (SCALE_MAX - SCALE_MIN) as f64;
    draw_slider(frame, fb_w, fb_h, l.scale_slider, scale_t);
    let scale_val = format!("{}%", dlg.scale_pct);
    text.draw_sized(frame, fb_w, fb_h, &scale_val,
        l.scale_slider.x + l.scale_slider.w as i32 + (10.0 * scale).round() as i32,
        l.scale_label_y, body_px, [205, 215, 230]);

    // SPP slider (log-scale on value display).
    let spp_label = format!("samples/pixel/frame  •  {:.2}", dlg.spp);
    text.draw_sized(frame, fb_w, fb_h, &spp_label,
        l.outer.x + pad, l.spp_label_y, body_px, [205, 215, 230]);
    let total = dlg.samples_per_frame();
    let total_str = format!(" ({} samples/frame)", human_count(total));
    let total_w = text.measure(&spp_label, body_px) as i32;
    text.draw_sized(frame, fb_w, fb_h, &total_str,
        l.outer.x + pad + total_w, l.spp_label_y, body_px, [140, 150, 170]);
    let spp_t = ((dlg.spp.ln() - SPP_MIN.ln()) / (SPP_MAX.ln() - SPP_MIN.ln()))
        .clamp(0.0, 1.0);
    draw_slider(frame, fb_w, fb_h, l.spp_slider, spp_t);

    // ETA section.
    let eta_text = match &*dlg.test.read() {
        TestState::NotRun => {
            format!("estimated render time: unknown (tap 'test' to measure)")
        }
        TestState::Running => {
            format!("estimated render time: testing…")
        }
        TestState::Done { frame_time } => {
            let eta = Duration::from_secs_f64(
                frame_time.as_secs_f64() * total_frames as f64,
            );
            format!(
                "estimated render time: {}   ({:.2}s per frame × {} frames)",
                fmt_dur(eta), frame_time.as_secs_f64(), total_frames,
            )
        }
        TestState::Failed(e) => {
            format!("test failed: {}", e)
        }
    };
    text.draw_sized(frame, fb_w, fb_h, &eta_text,
        l.outer.x + pad, l.eta_y, body_px, [200, 210, 225]);

    // Buttons.
    let test_running = matches!(&*dlg.test.read(), TestState::Running);
    let test_fill = if test_running { [30, 30, 30, 230] } else { [24, 36, 60, 230] };
    let test_border = if test_running { [80, 80, 80, 240] } else { [120, 150, 200, 240] };
    draw_labeled_button(text, frame, fb_w, fb_h, l.test_btn, "test",
        scale, test_fill, test_border, [230, 235, 245]);
    draw_labeled_button(text, frame, fb_w, fb_h, l.cancel_btn, "cancel",
        scale, [40, 28, 32, 230], [170, 130, 140, 240], [240, 220, 225]);
    draw_labeled_button(text, frame, fb_w, fb_h, l.render_btn, "render",
        scale, [28, 60, 36, 230], [120, 200, 140, 240], [220, 240, 225]);
}

fn draw_text_field(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    r: Rect, s: &str, focused: bool, px: u32,
) {
    let border = if focused { [150, 200, 250, 240] } else { [90, 100, 120, 220] };
    overlay::fill_rect(frame, fb_w, fb_h, r.x, r.y, r.w, r.h, [6, 10, 16, 220]);
    draw_button_frame(frame, fb_w, fb_h, r, [6, 10, 16, 220], border);
    let y = r.y + r.h as i32 - ((r.h as i32 - text.line_height(px) as i32) / 2) - 4;
    text.draw_sized(frame, fb_w, fb_h, s, r.x + 6, y, px, [230, 235, 245]);
    if focused {
        // caret at end of string
        let tw = text.measure(s, px) as i32;
        let cx = r.x + 6 + tw + 1;
        for dy in 2..(r.h as i32 - 4) {
            put_pixel(frame, fb_w, fb_h, cx, r.y + dy, [220, 230, 245, 240]);
        }
    }
}

/// Draw a small radio-button row: a circle on the left + a label. The circle
/// is filled when `active`. Hit-testing uses the full `r` rect.
fn draw_radio_row(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    r: Rect, scale: f64, label: &str, active: bool,
) {
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let radius = ((r.h as i32 - 4) / 2).max(4);
    let cx = r.x + radius + 2;
    let cy = r.y + r.h as i32 / 2;
    let (fill, border) = if active {
        ([90u8, 170, 240, 255], [200u8, 225, 250, 255])
    } else {
        ([18u8, 22, 32, 230],   [110u8, 125, 150, 230])
    };
    draw_filled_circle(frame, fb_w, fb_h, cx, cy, radius, fill, border);
    let text_x = cx + radius + (6.0 * scale).round() as i32;
    let line_h = text.line_height(px) as i32;
    let text_y = r.y + (r.h as i32 + line_h) / 2 - 4;
    let color = if active { [225u8, 240, 255] } else { [200, 210, 225] };
    text.draw_sized(frame, fb_w, fb_h, label, text_x, text_y, px, color);
}

fn draw_slider(frame: &mut [u8], fb_w: u32, fb_h: u32, r: Rect, t: f64) {
    let t = t.clamp(0.0, 1.0);
    // Track.
    overlay::fill_rect(frame, fb_w, fb_h, r.x, r.y, r.w, r.h, [10, 14, 22, 220]);
    draw_button_frame(frame, fb_w, fb_h, r,
        [10, 14, 22, 220], [80, 90, 110, 220]);
    // Filled portion.
    let filled = (t * r.w as f64).round() as u32;
    overlay::fill_rect(frame, fb_w, fb_h, r.x, r.y, filled, r.h, [60, 110, 180, 230]);
    // Handle.
    let hx = r.x + filled as i32 - 4;
    overlay::fill_rect(frame, fb_w, fb_h, hx, r.y - 3, 8, r.h + 6,
        [200, 220, 240, 240]);
}

fn draw_labeled_button(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    r: Rect, label: &str, scale: f64,
    fill: [u8; 4], border: [u8; 4], label_color: [u8; 3],
) {
    draw_button_frame(frame, fb_w, fb_h, r, fill, border);
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let tw = text.measure(label, px) as i32;
    let tx = r.x + (r.w as i32 - tw) / 2;
    let ty = r.y + r.h as i32 - ((r.h as i32 - text.line_height(px) as i32) / 2) - 4;
    text.draw_sized(frame, fb_w, fb_h, label, tx, ty, px, label_color);
}

/// Labeled button with hover tint. Used for the composer's save/load strip.
fn draw_text_button(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    r: Rect, label: &str,
    cursor: PhysicalPosition<f64>,
) {
    let hot = r.contains(cursor.x as i32, cursor.y as i32);
    let (fill, border, color) = if hot {
        ([30, 46, 72, 235], [140, 190, 240, 240], [230, 240, 250])
    } else {
        ([22, 28, 40, 225], [100, 120, 160, 220], [210, 220, 235])
    };
    draw_labeled_button(text, frame, fb_w, fb_h, r, label, scale, fill, border, color);
}

fn draw_button_frame(
    frame: &mut [u8], fb_w: u32, fb_h: u32, r: Rect, fill: [u8; 4], border: [u8; 4],
) {
    overlay::fill_rect(frame, fb_w, fb_h, r.x, r.y, r.w, r.h, fill);
    for x in r.x..(r.x + r.w as i32) {
        for y in [r.y, r.y + r.h as i32 - 1] {
            put_pixel(frame, fb_w, fb_h, x, y, border);
        }
    }
    for y in r.y..(r.y + r.h as i32) {
        for x in [r.x, r.x + r.w as i32 - 1] {
            put_pixel(frame, fb_w, fb_h, x, y, border);
        }
    }
}

fn draw_save_button(
    text: &mut TextRenderer,
    frame: &mut [u8],
    fb_w: u32,
    fb_h: u32,
    scale: f64,
    state: SaveResult,
) {
    let r = save_button_rect(fb_w, scale);
    let (fill, border, marker) = match state {
        SaveResult::None   => ([24u8,30,40,220], [120u8,140,170,230], None),
        SaveResult::Ok     => ([18u8,40,24,230], [130u8,210,140,240], Some(("✓", [180u8, 240, 180]))),
        SaveResult::Failed => ([50u8,22,22,230], [230u8,120,120,240], Some(("×", [240u8, 180, 180]))),
    };
    draw_button_frame(frame, fb_w, fb_h, r, fill, border);

    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let label = "save frame";
    let label_w = text.measure(label, px) as i32;
    let marker_w = if let Some((m, _)) = marker {
        (text.measure(m, px) as i32) + (6.0 * scale).round() as i32
    } else {
        0
    };
    let total_w = label_w + marker_w;
    let tx = r.x + (r.w as i32 - total_w) / 2;
    let ty = r.y + r.h as i32 - ((r.h as i32 - text.line_height(px) as i32) / 2) - 4;
    if let Some((m, color)) = marker {
        text.draw_sized(frame, fb_w, fb_h, m, tx, ty, px, color);
        text.draw_sized(frame, fb_w, fb_h, label, tx + marker_w, ty, px, [230, 235, 245]);
    } else {
        text.draw_sized(frame, fb_w, fb_h, label, tx, ty, px, [230, 235, 245]);
    }
}

fn put_pixel(frame: &mut [u8], fw: u32, fh: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x as u32 >= fw || y as u32 >= fh {
        return;
    }
    let idx = ((y as u32 * fw + x as u32) * 4) as usize;
    let a = color[3] as f32 / 255.0;
    for c in 0..3 {
        let dst = frame[idx + c] as f32;
        let src = color[c] as f32;
        frame[idx + c] = (dst + (src - dst) * a).clamp(0.0, 255.0) as u8;
    }
    frame[idx + 3] = 255;
}

/// Free-function form of push_view_snapshot: used at call sites that already
/// hold a disjoint `&mut` on another `App` field (e.g. `ws`), where calling
/// `&mut self.*` via a method would trip the borrow checker.
fn push_view_snapshot_fn(
    view: &Arc<RwLock<View>>,
    stack: &mut Vec<ViewSnapshot>,
    cursor: &mut usize,
    last_push: &mut Option<Instant>,
) {
    const UNDO_COALESCE_MS: u128 = 350;
    const UNDO_CAP: usize = 256;
    let snap = ViewSnapshot::of(&view.read());
    if let Some(cur) = stack.get(*cursor) {
        if cur.matches(&snap) { return; }
    }
    let now = Instant::now();
    let coalesce = last_push
        .map_or(false, |t| now.duration_since(t).as_millis() < UNDO_COALESCE_MS)
        && *cursor > 0
        && *cursor + 1 == stack.len();
    if coalesce {
        stack[*cursor] = snap;
    } else {
        stack.truncate(*cursor + 1);
        stack.push(snap);
        *cursor = stack.len() - 1;
    }
    *last_push = Some(now);
    if stack.len() > UNDO_CAP {
        let drop = stack.len() - UNDO_CAP;
        stack.drain(0..drop);
        *cursor = stack.len().saturating_sub(1);
    }
}

/// Spawn a detached `render-video` subprocess for the given job and start a
/// watcher thread that mirrors session.json into `progress`. Does NOT touch
/// `self` directly, so it can be called from inside a `&mut self.ws` scope.
fn spawn_render_job_free(
    job: &RenderJob,
    progress: &Arc<RwLock<Option<RenderProgress>>>,
    last_render: &Arc<RwLock<RenderResult>>,
) -> std::io::Result<()> {
    if job.keyframes.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "need at least 2 keyframes",
        ));
    }
    if let Some(parent) = job.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            eprintln!("error: render spawn: create_dir_all({}) failed: {e}",
                parent.display());
            e
        })?;
    }
    let exe = std::env::current_exe().map_err(|e| {
        eprintln!("error: render spawn: current_exe() failed: {e}");
        e
    })?;
    let mut cmd = Command::new(&exe);
    cmd.arg("render-video")
        .arg("--out").arg(&job.output)
        .arg("--fps").arg(job.fps.to_string())
        .arg("--width").arg(job.fb_w.to_string())
        .arg("--height").arg(job.fb_h.to_string())
        .arg("--samples").arg(job.samples_per_frame.to_string())
        .arg("--frames").arg(job.total_frames.to_string())
        .arg("--session-dir").arg(&job.session_dir);
    if !job.tensions.is_empty() {
        let csv = job.tensions.iter()
            .map(|v| format!("{v}")).collect::<Vec<_>>().join(",");
        cmd.arg("--tensions").arg(csv);
    }
    if !job.segments.is_empty() {
        let csv = job.segments.iter()
            .map(|v| format!("{v}")).collect::<Vec<_>>().join(",");
        cmd.arg("--segments").arg(csv);
    }
    if let Some(g) = job.gamma_override {
        cmd.arg("--gamma").arg(format!("{g}"));
    }
    cmd.arg("--rho").arg(format!("{}", job.rho));
    if let Some(stops) = &job.palette_stops_override {
        // Flat CSV: pos,r,g,b per stop, all joined.
        let mut parts: Vec<String> = Vec::with_capacity(stops.len() * 4);
        for (pos, rgb) in stops {
            parts.push(format!("{pos}"));
            parts.push(format!("{}", rgb[0]));
            parts.push(format!("{}", rgb[1]));
            parts.push(format!("{}", rgb[2]));
        }
        cmd.arg("--palette-stops").arg(parts.join(","));
    }
    for kf in &job.keyframes {
        cmd.arg(kf);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let _child = cmd.spawn().map_err(|e| {
        eprintln!("error: render spawn: cmd.spawn({}) failed: {e}", exe.display());
        e
    })?;

    *progress.write() = Some(RenderProgress {
        cur: 0,
        total: job.total_frames,
        started: Instant::now(),
        fb_w: job.fb_w,
        fb_h: job.fb_h,
        fps: job.fps,
        samples_per_frame: job.samples_per_frame,
        session_dir: job.session_dir.clone(),
        output: job.output.clone(),
    });
    *last_render.write() = RenderResult::None;

    start_session_watcher(
        job.session_dir.clone(),
        progress.clone(),
        last_render.clone(),
    );
    Ok(())
}

/// Polls `session_dir/session.json` every ~500ms, mirroring its state into
/// `progress`. When the session dir disappears *after having existed*, sets
/// `last_render` based on whether `output` exists.
///
/// The watcher tolerates a startup grace period during which the subprocess
/// hasn't yet `mkdir`-ed the session dir — without this, a watcher spawned
/// just after a subprocess spawn would immediately conclude "session ended"
/// because its first poll hits a race window of ~tens-of-ms.
fn start_session_watcher(
    session_dir: PathBuf,
    progress: Arc<RwLock<Option<RenderProgress>>>,
    last_render: Arc<RwLock<RenderResult>>,
) {
    thread::spawn(move || {
        const STARTUP_GRACE: Duration = Duration::from_secs(15);
        let mut last_output: Option<PathBuf> = None;
        let mut last_sess: Option<SessionFile> = None;
        let mut ever_seen = false;
        let watch_started = Instant::now();
        let start_instant = Instant::now();
        loop {
            let exists = session_dir.exists();
            if exists {
                ever_seen = true;
                if let Ok(s) = SessionFile::read(&session_dir) {
                    let sys_now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(s.started_at);
                    let ago = sys_now.saturating_sub(s.started_at);
                    let started = Instant::now() - Duration::from_secs(ago);
                    *progress.write() = Some(RenderProgress {
                        cur: s.cur_frame,
                        total: s.total_frames,
                        started,
                        fb_w: s.width,
                        fb_h: s.height,
                        fps: s.fps,
                        samples_per_frame: s.samples_per_frame,
                        session_dir: session_dir.clone(),
                        output: s.output.clone(),
                    });
                    last_output = Some(s.output.clone());
                    last_sess = Some(s);
                }
            } else if ever_seen {
                // Session vanished after having existed — render ended.
                *progress.write() = None;
                if let (Some(sess), Some(path)) = (last_sess.as_ref(), last_output.clone()) {
                    if path.exists() {
                        let file_size = std::fs::metadata(&path).ok().map(|m| m.len());
                        *last_render.write() = RenderResult::Ok(RenderDone {
                            path,
                            elapsed: start_instant.elapsed(),
                            total_frames: sess.total_frames,
                            fb_w: sess.width,
                            fb_h: sess.height,
                            fps: sess.fps,
                            samples_per_frame: sess.samples_per_frame,
                            file_size,
                        });
                    } else {
                        let msg = format!(
                            "session ended without output (expected {})",
                            path.display());
                        eprintln!("error: render watcher: {msg}");
                        *last_render.write() = RenderResult::Failed(msg);
                    }
                } else {
                    let msg = "session ended before any progress".to_string();
                    eprintln!("error: render watcher: {msg}");
                    *last_render.write() = RenderResult::Failed(msg);
                }
                break;
            } else if watch_started.elapsed() > STARTUP_GRACE {
                // Grace expired without ever seeing the session.
                *progress.write() = None;
                let msg =
                    "render subprocess didn't create its session directory".to_string();
                eprintln!("error: render watcher ({}): {msg}", session_dir.display());
                *last_render.write() = RenderResult::Failed(msg);
                break;
            }
            // else: still in grace period, keep waiting.
            thread::sleep(Duration::from_millis(500));
        }
    });
}

fn run_test_frame(
    kf_path: PathBuf,
    w: u32, h: u32,
    samples_target: u64,
    n_workers: usize,
    out: Arc<RwLock<TestState>>,
) {
    match crate::metadata::read_spec(&kf_path, samples_target, n_workers) {
        Ok(mut spec) => {
            spec.view.width = w;
            spec.view.height = h;
            let t0 = Instant::now();
            let _rgba = crate::render::render_frame(&spec, None);
            let elapsed = t0.elapsed();
            *out.write() = TestState::Done { frame_time: elapsed };
        }
        Err(e) => {
            eprintln!("error: render-test read_spec({}) failed: {e}",
                kf_path.display());
            *out.write() = TestState::Failed(e.to_string());
        }
    }
}

fn ease_in_out(t: f64) -> f64 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

fn draw_hud(
    text: &mut TextRenderer,
    frame: &mut [u8],
    fw: u32,
    fh: u32,
    lines: &[String],
    scale: f64,
) {
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let line_h = text.line_height(px) as i32;
    let padding = ((6.0 * scale).round() as i32).max(3);
    let x = padding + (4.0 * scale).round() as i32;

    let mut max_w = 0u32;
    for line in lines {
        let w = text.measure(line, px);
        if w > max_w {
            max_w = w;
        }
    }
    let box_w = max_w + (16.0 * scale).round() as u32;
    let box_h = (lines.len() as i32 * line_h + (10.0 * scale).round() as i32) as u32;
    overlay::fill_rect(frame, fw, fh, padding, padding, box_w, box_h, [5, 8, 14, 180]);

    let mut y = padding + line_h - 2;
    for line in lines {
        text.draw_sized(frame, fw, fh, line, x, y, px, [220, 230, 240]);
        y += line_h;
    }
}

fn hud_lines(
    v: &View,
    palette: &Palette,
    gamma: f32,
    samples: u64,
    sps: f64,
    workers: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!(
        "iter={}  zoom={:.3}  ctr=({:+.4},{:+.4})",
        v.max_iter,
        1.8 / v.half_width,
        v.center[0],
        v.center[1]
    ));
    out.push(format!(
        "samples={}  ({:.0}/s ×{}w)  palette={}  γ={:.2}",
        human_count(samples),
        sps,
        workers,
        palette.name,
        gamma
    ));
    let deg = |k: usize| v.rotation[k][k].clamp(-1.0, 1.0).acos().to_degrees();
    out.push(format!(
        "tilt: x={:5.1}° y={:5.1}° z={:5.1}° w={:5.1}°",
        deg(0), deg(1), deg(2), deg(3)
    ));
    out.push(String::from("R ="));
    for i in 0..4 {
        out.push(format!(
            "  [{:+.2} {:+.2} {:+.2} {:+.2}]",
            v.rotation[i][0], v.rotation[i][1], v.rotation[i][2], v.rotation[i][3]
        ));
    }
    out
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m{:02}s", s / 3600, (s % 3600) / 60, s % 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else if s >= 10 {
        format!("{}s", s)
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

fn fmt_bytes(n: u64) -> String {
    let gb = 1u64 << 30;
    let mb = 1u64 << 20;
    let kb = 1u64 << 10;
    if n >= gb      { format!("{:.2} GB", n as f64 / gb as f64) }
    else if n >= mb { format!("{:.1} MB", n as f64 / mb as f64) }
    else if n >= kb { format!("{:.0} KB", n as f64 / kb as f64) }
    else            { format!("{} B", n) }
}

fn draw_axes_overlay(
    v: &View,
    text: &mut TextRenderer,
    frame: &mut [u8],
    fw: u32,
    fh: u32,
    scale: f64,
) {
    const L: f64 = 2.0;
    const BASE_ALPHA: f64 = 220.0;
    let axes: [(usize, &str, [u8; 3]); 4] = [
        (0, "X", [235,  90,  90]),
        (1, "Y", [235, 180,  70]),
        (2, "Z", [ 90, 160, 235]),
        (3, "W", [140, 200, 110]),
    ];
    let r = &v.rotation;
    let hw = v.half_width;
    let hh = v.half_height();
    let to_pixel = |wx: f64, wy: f64| -> (f64, f64) {
        let px = (wx - v.center[0] + hw) / (2.0 * hw) * fw as f64;
        let py = (hh - (wy - v.center[1])) / (2.0 * hh) * fh as f64;
        (px, py)
    };
    for &(i, label, col) in &axes {
        let dir_x = r[0][i];
        let dir_y = r[1][i];
        let depth = (r[2][i] * r[2][i] + r[3][i] * r[3][i]).sqrt().min(1.0);
        let (ox, oy) = to_pixel(0.0, 0.0);
        let (px_pos, py_pos) = to_pixel(L * dir_x, L * dir_y);
        let (px_neg, py_neg) = to_pixel(-L * dir_x, -L * dir_y);
        let end_alpha_f = (BASE_ALPHA * (1.0 - depth)).max(0.0);
        let center_alpha = BASE_ALPHA as u8;
        let end_alpha = end_alpha_f as u8;
        draw_line_alpha_grad(frame, fw, fh, ox, oy, px_pos, py_pos, col, center_alpha, end_alpha);
        draw_line_alpha_grad(frame, fw, fh, ox, oy, px_neg, py_neg, col, center_alpha, end_alpha);
        if end_alpha > 12 {
            let lpx = ((11.0 * scale).round() as u32).max(10);
            let lx = (px_pos + 4.0).round() as i32;
            let ly = (py_pos - 4.0).round() as i32;
            text.draw_sized(frame, fw, fh, label, lx, ly, lpx, fade_color(col, end_alpha));
        }
    }
}

#[inline]
fn fade_color(rgb: [u8; 3], alpha: u8) -> [u8; 3] {
    let a = alpha as u32;
    [
        (rgb[0] as u32 * a / 255) as u8,
        (rgb[1] as u32 * a / 255) as u8,
        (rgb[2] as u32 * a / 255) as u8,
    ]
}

fn draw_line_alpha_grad(
    frame: &mut [u8], fw: u32, fh: u32,
    x0: f64, y0: f64, x1: f64, y1: f64,
    rgb: [u8; 3], a0: u8, a1: u8,
) {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let steps = dx.max(dy).ceil().max(1.0) as i32;
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let x = (x0 + (x1 - x0) * t).round() as i32;
        let y = (y0 + (y1 - y0) * t).round() as i32;
        if x < 0 || y < 0 || x as u32 >= fw || y as u32 >= fh { continue; }
        let alpha = (a0 as f64 * (1.0 - t) + a1 as f64 * t) as u8;
        let idx = ((y as u32 * fw + x as u32) * 4) as usize;
        let a = alpha as f32 / 255.0;
        for c in 0..3 {
            let dst = frame[idx + c] as f32;
            let src = rgb[c] as f32;
            frame[idx + c] = (dst + (src - dst) * a).clamp(0.0, 255.0) as u8;
        }
        frame[idx + 3] = 255;
    }
}

fn draw_help(text: &mut TextRenderer, frame: &mut [u8], fw: u32, fh: u32, scale: f64) {
    let lines: &[&str] = &[
        "buddhabrot — controls",
        "",
        "drag                    rotate XZ / YW  (mix c↔z)",
        "shift+drag              rotate XY / ZW  (c-plane / z-plane spin)",
        "ctrl+drag               rotate XW / YZ  (cross-couples)",
        "right-drag              rotate XZ / YZ  (orbit with Z as depth)",
        "shift+right-drag        rotate XW / YW  (orbit with W as depth)",
        "middle-drag             pan",
        "scroll                  zoom (mouse-anchored)",
        "click gizmo vertex      snap to corner view at that vertex",
        "click gizmo face center snap axis-aligned to that plane",
        "click gizmo face corner snap to 45° corner view of that face",
        "click 'save frame'      save PNG to ~/buddhabrot/ + add as video keyframe",
        "",
        "[ / ]                   max-iter  ÷2 / ×2   (shift → ±100)",
        "p / shift-p             cycle palette preset forward / back",
        "g / shift-g             gamma  -  / +",
        "  (lower-left HUD also lets you edit palette stops + gamma directly)",
        "r                       reset view",
        "space                   pause/resume UI sampler",
        "meta-z / meta-shift-z   undo / redo camera move",
        "meta-s                  save PNG with view metadata",
        "o                       toggle onion skin during drag",
        "a                       toggle coordinate axes overlay",
        "h                       toggle this help",
        "q / esc                 quit",
        "",
        "axes: X=c_re  Y=c_im  Z=z_re  W=z_im",
    ];
    let px = (HELP_BASE_PX as f64 * scale).round() as u32;
    let line_h = text.line_height(px) as i32;
    let mut max_w = 0u32;
    for l in lines {
        let w = text.measure(l, px);
        if w > max_w {
            max_w = w;
        }
    }
    let box_w = max_w + (40.0 * scale).round() as u32;
    let box_h = (lines.len() as i32 * line_h + (30.0 * scale).round() as i32) as u32;
    // Anchor the help panel at top-center, just below the tab bar, so it
    // doesn't cover the lower-left palette/gamma HUD.
    let tab_bottom = ((4.0 + TAB_BAR_H as f64) * scale).round() as i32;
    let gap = (8.0 * scale).round() as i32;
    let x = ((fw as i32 - box_w as i32) / 2).max(0);
    let y = tab_bottom + gap;
    overlay::fill_rect(frame, fw, fh, x, y, box_w, box_h, [5, 8, 14, 220]);

    let mut yy = y + line_h + (8.0 * scale).round() as i32;
    for l in lines {
        text.draw_sized(
            frame,
            fw,
            fh,
            l,
            x + (20.0 * scale).round() as i32,
            yy,
            px,
            [230, 230, 240],
        );
        yy += line_h;
    }
}

// ========================================================================
// Tab bar
// ========================================================================

fn draw_tab_bar(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64, mode: Mode,
) {
    let tabs = tab_bar_rects(fb_w, scale);
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let active = [24u8, 50, 80, 230];
    let inactive = [12u8, 18, 28, 220];
    let border = [110u8, 130, 160, 220];
    let labels = ["live", "composer"];
    let modes = [Mode::Live, Mode::Composer];
    for (i, r) in tabs.iter().enumerate() {
        let fill = if mode == modes[i] { active } else { inactive };
        draw_button_frame(frame, fb_w, fb_h, *r, fill, border);
        let tw = text.measure(labels[i], px) as i32;
        let tx = r.x + (r.w as i32 - tw) / 2;
        let ty = r.y + r.h as i32 - ((r.h as i32 - text.line_height(px) as i32) / 2) - 4;
        text.draw_sized(frame, fb_w, fb_h, labels[i], tx, ty, px, [230, 235, 245]);
    }
}

// ========================================================================
// Composer drawing
// ========================================================================

fn draw_composer(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    render_progress: Option<RenderProgress>,
    render_queue: &[RenderJob],
    cursor: PhysicalPosition<f64>,
) {
    let l = composer_layout(fb_w, fb_h, scale, composer);

    // Thumbnail grid.
    draw_button_frame(frame, fb_w, fb_h, l.grid,
        [10, 14, 22, 200], [70, 80, 100, 220]);
    draw_thumb_grid(text, frame, fb_w, fb_h, scale, composer, l.grid);

    // Top-left strip: save/load buttons + thumb-size label + slider.
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    draw_text_button(text, frame, fb_w, fb_h, scale,
        l.save_btn, "save", cursor);
    draw_text_button(text, frame, fb_w, fb_h, scale,
        l.load_btn, "load", cursor);
    // Vertically center "thumb size" on the slider — baseline at the slider's
    // center plus half an ascender so the glyph optical center lands on
    // slider center.
    let slider_mid = l.thumb_slider.y + l.thumb_slider.h as i32 / 2;
    let label_baseline = slider_mid + (px as i32) / 2 - 1;
    text.draw_sized(frame, fb_w, fb_h, "thumb size",
        l.thumb_label.x, label_baseline, px, [200, 210, 225]);
    let t = (composer.thumb_size - composer::THUMB_SIZE_MIN) as f64
        / (composer::THUMB_SIZE_MAX - composer::THUMB_SIZE_MIN) as f64;
    draw_slider(frame, fb_w, fb_h, l.thumb_slider, t);

    // Preview pane — no frame, blends with the background.
    draw_preview_pane(text, frame, fb_w, fb_h, scale, composer, l.preview);

    // Motion panel (framed): path-ρ slider.
    draw_button_frame(frame, fb_w, fb_h, l.motion_panel,
        [10, 14, 22, 200], [70, 80, 100, 220]);
    draw_motion_panel(text, frame, fb_w, fb_h, scale, composer, l.motion_panel);

    // Render-task panel (framed).
    draw_button_frame(frame, fb_w, fb_h, l.render_panel,
        [10, 14, 22, 200], [70, 80, 100, 220]);
    draw_render_panel(text, frame, fb_w, fb_h, scale, composer,
        render_progress, render_queue, l.render_panel);

    // Timeline (bottom).
    draw_button_frame(frame, fb_w, fb_h, l.timeline,
        [10, 14, 22, 210], [80, 90, 110, 220]);
    draw_timeline(text, frame, fb_w, fb_h, scale, composer, l.timeline);

    // Pane splitter: 2-px visible line (6-px hit region). Brightened while
    // dragging so the user sees the grabbed affordance.
    let hot = composer.splitter_drag.is_some();
    let fill = if hot { [80, 130, 200, 230] } else { [55, 65, 85, 200] };
    overlay::fill_rect(frame, fb_w, fb_h,
        l.split_x - 1, l.grid.y, 2, l.grid.h, fill);

    // Grid right-click context menu — drawn last so it overlays everything.
    if let Some(menu) = composer.context_menu.as_ref() {
        let ml = grid_context_menu_layout(menu, fb_w, fb_h, scale);
        draw_grid_context_menu(text, frame, fb_w, fb_h, scale, cursor, &ml);
    }

    // Load-timeline dialog overlay. Anchored under the load button.
    if let Some(dlg) = composer.load_dialog.as_ref() {
        let ll = load_dialog_layout(l.load_btn, dlg, fb_w, fb_h, scale);
        draw_load_dialog(text, frame, fb_w, fb_h, scale, cursor, dlg, &ll);
    }
}

/// Simple view of a preview image passed to `blit_preview`.
struct PreviewImage<'a> {
    rgba: &'a [u8],
    w: u32,
    h: u32,
}

/// Scale-blit a preview RGBA image to `dst`, preserving aspect ratio.
fn blit_preview(
    frame: &mut [u8], fw: u32, fh: u32,
    src: &PreviewImage<'_>, dst: Rect,
) {
    if src.w == 0 || src.h == 0 { return; }
    let src_aspect = src.w as f64 / src.h as f64;
    let dst_aspect = dst.w as f64 / dst.h as f64;
    let (tw, th) = if src_aspect > dst_aspect {
        (dst.w, ((dst.w as f64 / src_aspect).round() as u32).max(1))
    } else {
        (((dst.h as f64 * src_aspect).round() as u32).max(1), dst.h)
    };
    let tx = dst.x + (dst.w as i32 - tw as i32) / 2;
    let ty = dst.y + (dst.h as i32 - th as i32) / 2;
    for py in 0..th {
        let sy = ((py as f64 + 0.5) * src.h as f64 / th as f64) as u32;
        let sy = sy.min(src.h - 1);
        let dy = ty + py as i32;
        if dy < 0 || dy as u32 >= fh { continue; }
        for px in 0..tw {
            let sx = ((px as f64 + 0.5) * src.w as f64 / tw as f64) as u32;
            let sx = sx.min(src.w - 1);
            let dx = tx + px as i32;
            if dx < 0 || dx as u32 >= fw { continue; }
            let si = ((sy * src.w + sx) * 4) as usize;
            let di = ((dy as u32 * fw + dx as u32) * 4) as usize;
            frame[di]     = src.rgba[si];
            frame[di + 1] = src.rgba[si + 1];
            frame[di + 2] = src.rgba[si + 2];
            frame[di + 3] = 255;
        }
    }
}

/// Layout of everything inside the composer's preview pane: image, playback
/// progress bar, circular play button, preview-quality sliders, palette
/// editor. Shared between draw and hit-test paths.
struct PreviewPaneLayout {
    image: Rect,
    /// Playback progress bar, directly under the image.
    progress: Rect,
    /// Circular play/pause button. Use the inscribed-square rect for
    /// hit-testing; draw code treats it as a circle.
    play_btn: Rect,
    /// Where the "playhead=X • WxH" label sits (left of play button).
    left_label_x: i32,
    right_label_x: i32,
    labels_y: i32,
    /// Preview render-size slider.
    size_slider: Rect,
    size_val_x: i32,
    size_row_text_y: i32,
    /// Preview samples/pixel slider.
    spp_slider: Rect,
    spp_val_x: i32,
    spp_row_text_y: i32,
    /// Palette gradient bar.
    palette_bar: Rect,
    /// Small button at the right end of the palette row that restores the
    /// default stops.
    palette_reset_btn: Rect,
    /// Y-position of the stop-marker row (just below palette_bar).
    palette_stops_y: i32,
    /// Height of the stop-marker row.
    palette_stops_h: u32,
    /// Preview gamma slider (under the palette area).
    gamma_slider: Rect,
    gamma_val_x: i32,
    gamma_row_text_y: i32,
    /// The color-edit popover rect (only meaningful if palette_color_edit is Some).
    color_popover: Rect,
}

fn preview_pane_layout(pane: Rect, scale: f64) -> PreviewPaneLayout {
    let pad = (10.0 * scale).round() as i32;
    let x = pane.x + pad;
    let right = pane.x + pane.w as i32 - pad;
    let inner_w = (right - x).max(10) as u32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;

    // Bottom-up reservations (so the image takes whatever's left at top).
    let palette_stops_h = (12.0 * scale).round() as u32;
    let palette_bar_h = (18.0 * scale).round() as u32;
    let slider_h = (10.0 * scale).round() as u32;
    let label_col_w = (56.0 * scale).round() as i32;
    // Room for the right-side value label. Long enough to fit
    // "WWWW×HHHH (100%)" and "10.0 (1234s/p)" without clipping.
    let val_w = (130.0 * scale).round() as i32;
    let slider_w = (inner_w as i32 - label_col_w - val_w).max(80) as u32;
    let slider_x = x + label_col_w;
    let val_x = slider_x + slider_w as i32 + (8.0 * scale).round() as i32;
    let play_btn_r = (44.0 * scale).round() as i32; // inscribed square side
    let progress_h = (8.0 * scale).round() as u32;

    // Bottom element: gamma slider row (under the palette area).
    let gamma_row_y = pane.y + pane.h as i32 - pad - line_h;
    let gamma_row_text_y = gamma_row_y + line_h - 2;
    let gamma_slider = Rect {
        x: slider_x,
        y: gamma_row_y + (line_h - slider_h as i32) / 2,
        w: slider_w, h: slider_h,
    };
    // Above gamma: palette stops row.
    let palette_stops_y = gamma_row_y - (4.0 * scale).round() as i32 - palette_stops_h as i32;
    // Above that: palette gradient bar.
    let palette_bar_y = palette_stops_y - 2 - palette_bar_h as i32;
    // Above that: spp slider row.
    let spp_row_y = palette_bar_y - (8.0 * scale).round() as i32 - line_h;
    let spp_row_text_y = spp_row_y + line_h - 2;
    let spp_slider = Rect {
        x: slider_x,
        y: spp_row_y + (line_h - slider_h as i32) / 2,
        w: slider_w, h: slider_h,
    };
    // Size slider row above spp.
    let size_row_y = spp_row_y - (4.0 * scale).round() as i32 - line_h;
    let size_row_text_y = size_row_y + line_h - 2;
    let size_slider = Rect {
        x: slider_x,
        y: size_row_y + (line_h - slider_h as i32) / 2,
        w: slider_w, h: slider_h,
    };
    // Play-button row: circular button centered, labels tucked L/R.
    let play_row_y = size_row_y - (12.0 * scale).round() as i32 - play_btn_r;
    let play_center_x = pane.x + pane.w as i32 / 2;
    let play_btn = Rect {
        x: play_center_x - play_btn_r / 2,
        y: play_row_y,
        w: play_btn_r as u32,
        h: play_btn_r as u32,
    };
    let left_label_x = x;
    let right_label_x = right - (90.0 * scale).round() as i32;
    let labels_y = play_row_y + play_btn_r / 2 + line_h / 3;
    // Progress bar immediately above play-button row.
    let progress_y = play_row_y - (6.0 * scale).round() as i32 - progress_h as i32;
    let progress = Rect { x, y: progress_y, w: inner_w, h: progress_h };
    // Image fills everything above the progress bar.
    let image = Rect {
        x, y: pane.y + pad,
        w: inner_w,
        h: (progress_y - (4.0 * scale).round() as i32 - (pane.y + pad)).max(80) as u32,
    };

    // Reserve a small "reset" button at the right end of the palette row;
    // shrink the gradient bar to fit it.
    let reset_w = (46.0 * scale).round() as u32;
    let reset_gap = (4.0 * scale).round() as i32;
    let bar_w = (inner_w as i32 - reset_w as i32 - reset_gap).max(20) as u32;
    let palette_bar = Rect {
        x, y: palette_bar_y, w: bar_w, h: palette_bar_h,
    };
    let palette_reset_btn = Rect {
        x: x + bar_w as i32 + reset_gap,
        y: palette_bar_y,
        w: reset_w,
        h: palette_bar_h,
    };

    // Color popover: overlays near palette (shown only when a stop is
    // selected for color editing). Size fits 3 RGB rows + padding.
    let pop_w = ((220.0 * scale).round() as u32).min(inner_w);
    let pop_h = (line_h * 4 + (20.0 * scale).round() as i32) as u32;
    let color_popover = Rect {
        x: (x + inner_w as i32 - pop_w as i32).max(x),
        y: palette_bar_y - pop_h as i32 - 4,
        w: pop_w, h: pop_h,
    };

    PreviewPaneLayout {
        image, progress, play_btn,
        left_label_x, right_label_x, labels_y,
        size_slider, size_val_x: val_x, size_row_text_y,
        spp_slider,  spp_val_x: val_x,  spp_row_text_y,
        palette_bar, palette_reset_btn, palette_stops_y, palette_stops_h,
        gamma_slider, gamma_val_x: val_x, gamma_row_text_y,
        color_popover,
    }
}

/// Layout for the live-mode palette + gamma HUD anchored in the lower-left
/// of the framebuffer. Mirrors the composer's preview-pane palette area but
/// stacked into a corner block. `color_popover` is meaningful only when the
/// user has clicked a stop to open it.
struct LivePaletteHudLayout {
    /// Bounding rect of the whole HUD (used to dim the area behind it).
    panel: Rect,
    palette_bar: Rect,
    palette_reset_btn: Rect,
    palette_stops_y: i32,
    palette_stops_h: u32,
    gamma_slider: Rect,
    gamma_label_x: i32,
    gamma_val_x: i32,
    gamma_row_text_y: i32,
    color_popover: Rect,
}

fn live_palette_hud_layout(_fb_w: u32, fb_h: u32, scale: f64) -> LivePaletteHudLayout {
    let pad = (8.0 * scale).round() as i32;
    let margin = (MARGIN_BASE as f64 * scale).round() as i32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;

    // Panel sized to fit a horizontal palette bar + stop markers + gamma row.
    let panel_w = (300.0 * scale).round() as u32;
    let palette_bar_h = (18.0 * scale).round() as u32;
    let palette_stops_h = (12.0 * scale).round() as u32;
    let slider_h = (10.0 * scale).round() as u32;
    let panel_h = (pad as u32)
        + palette_bar_h
        + 2
        + palette_stops_h
        + (6.0 * scale).round() as u32
        + line_h as u32
        + (pad as u32);

    let panel = Rect {
        x: margin,
        y: fb_h as i32 - margin - panel_h as i32,
        w: panel_w,
        h: panel_h,
    };

    let inner_x = panel.x + pad;
    let inner_w = (panel_w as i32 - pad * 2).max(40) as u32;

    // Top of the inner stack: palette gradient bar, with reset button to its
    // right (mirrors the composer's preview pane).
    let bar_y = panel.y + pad;
    let reset_w = (46.0 * scale).round() as u32;
    let reset_gap = (4.0 * scale).round() as i32;
    let bar_w = (inner_w as i32 - reset_w as i32 - reset_gap).max(20) as u32;
    let palette_bar = Rect { x: inner_x, y: bar_y, w: bar_w, h: palette_bar_h };
    let palette_reset_btn = Rect {
        x: inner_x + bar_w as i32 + reset_gap,
        y: bar_y, w: reset_w, h: palette_bar_h,
    };

    // Stop markers under the bar.
    let palette_stops_y = bar_y + palette_bar_h as i32 + 2;

    // Gamma row at the bottom: "gamma" label, slider, value column.
    let gamma_row_y = palette_stops_y + palette_stops_h as i32
        + (6.0 * scale).round() as i32;
    let gamma_row_text_y = gamma_row_y + line_h - 2;
    let label_col_w = (52.0 * scale).round() as i32;
    let val_w = (44.0 * scale).round() as i32;
    let slider_x = inner_x + label_col_w;
    let slider_w = (inner_w as i32 - label_col_w - val_w
        - (8.0 * scale).round() as i32).max(40) as u32;
    let gamma_slider = Rect {
        x: slider_x,
        y: gamma_row_y + (line_h - slider_h as i32) / 2,
        w: slider_w, h: slider_h,
    };
    let gamma_val_x = slider_x + slider_w as i32 + (8.0 * scale).round() as i32;

    // Color popover floats above the panel so an open editor can't push the
    // bar off-screen. Width matches the composer's popover; height fits 3
    // RGB rows + hint.
    let pop_w = ((220.0 * scale).round() as u32).min(inner_w);
    let pop_h = (line_h * 4 + (20.0 * scale).round() as i32) as u32;
    let pop_x = inner_x;
    // Anchor above panel, but never above the top edge of the framebuffer.
    let pop_y = (panel.y - pop_h as i32 - 4).max(margin);
    let color_popover = Rect {
        x: pop_x, y: pop_y, w: pop_w, h: pop_h,
    };

    LivePaletteHudLayout {
        panel,
        palette_bar, palette_reset_btn, palette_stops_y, palette_stops_h,
        gamma_slider, gamma_label_x: inner_x, gamma_val_x, gamma_row_text_y,
        color_popover,
    }
}

/// Draw the live-mode palette/gamma HUD. The popover (if `palette_color_edit`
/// is `Some`) is drawn last so it overlays everything.
fn draw_live_palette_hud(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32, scale: f64,
    layout: &LivePaletteHudLayout,
    palette: &Palette,
    palette_stops: &[composer::PaletteStop],
    palette_color_edit: Option<usize>,
    gamma: f32,
) {
    // Dim panel background so the controls stay legible over the fractal.
    overlay::fill_rect(frame, fb_w, fb_h,
        layout.panel.x, layout.panel.y, layout.panel.w, layout.panel.h,
        [10, 14, 22, 200]);
    draw_button_frame(frame, fb_w, fb_h, layout.panel,
        [0, 0, 0, 0], [70, 80, 100, 220]);

    draw_palette_bar(frame, fb_w, fb_h, layout.palette_bar, palette);

    draw_labeled_button(text, frame, fb_w, fb_h,
        layout.palette_reset_btn, "reset", scale,
        [32, 38, 52, 230], [120, 140, 170, 230], [220, 225, 235]);

    draw_palette_stops(frame, fb_w, fb_h, layout.palette_bar,
        layout.palette_stops_y, layout.palette_stops_h,
        palette_stops, palette_color_edit);

    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    text.draw_sized(frame, fb_w, fb_h, "gamma",
        layout.gamma_label_x, layout.gamma_row_text_y, px, [200, 210, 225]);
    let t_g = ((gamma - composer::GAMMA_MIN)
        / (composer::GAMMA_MAX - composer::GAMMA_MIN)).clamp(0.0, 1.0) as f64;
    draw_slider(frame, fb_w, fb_h, layout.gamma_slider, t_g);
    text.draw_sized(frame, fb_w, fb_h,
        &format!("{:.2}", gamma),
        layout.gamma_val_x, layout.gamma_row_text_y, px, [210, 220, 235]);

    if let Some(stop_idx) = palette_color_edit {
        if let Some(stop) = palette_stops.get(stop_idx) {
            draw_color_popover(text, frame, fb_w, fb_h,
                layout.color_popover, scale, stop_idx, stop);
        }
    }
}

/// Hit-test variant of `palette_stop_hit` for arbitrary stop slices (the
/// existing helper is wired against `Composer`). Returns the closest marker
/// within ≤6px horizontally, else `None`.
fn live_palette_stop_hit(
    stops: &[composer::PaletteStop],
    bar: Rect, stops_y: i32, stops_h: u32,
    cx: i32, cy: i32,
) -> Option<usize> {
    if cy < stops_y || cy >= stops_y + stops_h as i32 + 2 {
        return None;
    }
    let mut best: Option<(usize, i32)> = None;
    for (i, s) in stops.iter().enumerate() {
        let sx = bar.x + (s.pos.clamp(0.0, 1.0) * bar.w as f32).round() as i32;
        let d = (cx - sx).abs();
        if d <= 6 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// Hit-test the three RGB sliders inside the live HUD's color popover.
/// Returns rects in R/G/B order (matches `draw_color_popover`).
fn live_color_popover_sliders(rect: Rect, scale: f64) -> [Rect; 3] {
    color_popover_sliders(rect, scale)
}

/// Apply the live-HUD gamma slider's value from the cursor's x-position.
fn apply_live_gamma_slider(gamma: &mut f32, slider: Rect, cx: i32) {
    let t = ((cx - slider.x) as f64 / slider.w.max(1) as f64).clamp(0.0, 1.0);
    let lo = composer::GAMMA_MIN as f64;
    let hi = composer::GAMMA_MAX as f64;
    *gamma = (lo + t * (hi - lo)) as f32;
}

/// Apply the cursor x-position to one channel (R/G/B) of an editable stop in
/// the live HUD's color popover.
fn apply_live_color_channel(
    palette_stops: &mut [composer::PaletteStop],
    stop_idx: usize, drag: LiveWidgetDrag, slider: Rect, cx: i32,
) {
    let Some(stop) = palette_stops.get_mut(stop_idx) else { return };
    let t = ((cx - slider.x) as f64 / slider.w.max(1) as f64).clamp(0.0, 1.0);
    let v = (t * 255.0).round() as u8;
    match drag {
        LiveWidgetDrag::ColorR => stop.rgb[0] = v,
        LiveWidgetDrag::ColorG => stop.rgb[1] = v,
        LiveWidgetDrag::ColorB => stop.rgb[2] = v,
        _ => {}
    }
}

/// Draw a filled circle with a border at `(cx, cy, r)`.
fn draw_filled_circle(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    cx: i32, cy: i32, r: i32,
    fill: [u8; 4], border: [u8; 4],
) {
    let r2 = r * r;
    let r2_inner = (r - 1).max(0).pow(2);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 {
                let c = if d2 >= r2_inner { border } else { fill };
                put_pixel(frame, fb_w, fb_h, cx + dx, cy + dy, c);
            }
        }
    }
}

/// Draw the play triangle (▶) glyph centered at (cx, cy) with outer size r.
fn draw_play_triangle(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    cx: i32, cy: i32, r: i32, color: [u8; 4],
) {
    // Isoceles triangle pointing right; height = 2r * 0.7, width = 2r * 0.6.
    let w = (r as f64 * 0.9) as i32;
    let h = (r as f64 * 1.05) as i32;
    let apex_x = cx + (w * 5 / 8); // visual balance: shift slightly right
    let base_x = apex_x - w;
    let top_y = cy - h / 2;
    let bot_y = cy + h / 2;
    // Fill row by row.
    for y in top_y..=bot_y {
        let t = (y - top_y) as f64 / (bot_y - top_y).max(1) as f64;
        // Width at this row: full at center, 0 at top/bottom.
        let row_half = ((1.0 - (2.0 * t - 1.0).abs()) * w as f64 / 2.0) as i32;
        // Triangle pointing right: left edge at base_x, right edge at apex_x
        // but pinched at top/bottom.
        let xl = base_x;
        let xr = base_x + w - (w / 2 - row_half);
        for x in xl..=xr.min(apex_x) {
            put_pixel(frame, fb_w, fb_h, x, y, color);
        }
    }
}

/// Draw two vertical bars (pause glyph) centered at (cx, cy).
fn draw_pause_bars(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    cx: i32, cy: i32, r: i32, color: [u8; 4],
) {
    let bar_w = (r as f64 * 0.28).max(3.0) as u32;
    let bar_h = (r as f64 * 1.1) as u32;
    let gap = (r as f64 * 0.28) as i32;
    let y = cy - bar_h as i32 / 2;
    overlay::fill_rect(frame, fb_w, fb_h,
        cx - gap / 2 - bar_w as i32, y, bar_w, bar_h, color);
    overlay::fill_rect(frame, fb_w, fb_h,
        cx + gap / 2, y, bar_w, bar_h, color);
}

fn draw_preview_pane(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    rect: Rect,
) {
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let pp = preview_pane_layout(rect, scale);

    // Image backdrop.
    overlay::fill_rect(frame, fb_w, fb_h, pp.image.x, pp.image.y,
        pp.image.w, pp.image.h, [4, 6, 10, 255]);

    if composer.timeline.is_empty() {
        let msg = "preview — add keyframes to the timeline to compose";
        let tw = text.measure(msg, px) as i32;
        text.draw_sized(frame, fb_w, fb_h, msg,
            pp.image.x + (pp.image.w as i32 - tw) / 2,
            pp.image.y + pp.image.h as i32 / 2,
            px, [130, 140, 160]);
    } else {
        let ps = composer.preview_state.read();
        let drew_rendered = if let Some(rgba) = ps.rgba.as_ref() {
            let src = PreviewImage { rgba, w: ps.w, h: ps.h };
            blit_preview(frame, fb_w, fb_h, &src, pp.image);
            true
        } else { false };
        drop(ps);
        if !drew_rendered {
            let n = composer.timeline.len();
            let t = composer.playhead.clamp(0.0, 1.0) * (n - 1).max(0) as f64;
            let idx = (t.floor() as usize).min(n.saturating_sub(1));
            let thumbs = composer.thumbs.read();
            if let Some(tb) = thumbs.get(&composer.timeline[idx].source) {
                composer::blit_thumb(frame, fb_w, fb_h, tb, pp.image);
            }
        }
    }

    // Progress (playhead) bar directly under the image.
    draw_slider(frame, fb_w, fb_h, pp.progress, composer.playhead.clamp(0.0, 1.0));

    // Circular play/pause button.
    let btn_cx = pp.play_btn.x + pp.play_btn.w as i32 / 2;
    let btn_cy = pp.play_btn.y + pp.play_btn.h as i32 / 2;
    let btn_r = pp.play_btn.w as i32 / 2;
    let hot = composer.playing;
    let fill = if hot {
        [60u8, 110, 180, 255]
    } else {
        [32u8, 42, 64, 255]
    };
    let border = [180u8, 200, 230, 255];
    draw_filled_circle(frame, fb_w, fb_h, btn_cx, btn_cy, btn_r, fill, border);
    if composer.playing {
        draw_pause_bars(frame, fb_w, fb_h, btn_cx, btn_cy,
            (btn_r as f64 * 0.55) as i32, [240, 245, 255, 255]);
    } else {
        draw_play_triangle(frame, fb_w, fb_h, btn_cx, btn_cy,
            (btn_r as f64 * 0.55) as i32, [240, 245, 255, 255]);
    }

    // Status labels tucked on either side of the button.
    let ps = composer.preview_state.read();
    let elapsed_ms = ps.elapsed.map(|d| format!("{}ms", d.as_millis()))
        .unwrap_or_else(|| "—".into());
    let res = if ps.w > 0 { format!("{}×{}", ps.w, ps.h) } else { "—".into() };
    drop(ps);
    let left_label = format!("playhead {:.3}", composer.playhead);
    let right_label = format!("{}  •  {}", res, elapsed_ms);
    let rw = text.measure(&right_label, px) as i32;
    text.draw_sized(frame, fb_w, fb_h, &left_label,
        pp.left_label_x, pp.labels_y, px, [160, 175, 200]);
    text.draw_sized(frame, fb_w, fb_h, &right_label,
        pp.right_label_x + ((90.0 * scale).round() as i32 - rw).max(0),
        pp.labels_y, px, [160, 175, 200]);

    // Preview size slider. Value label shows the effective preview
    // pixel dimensions (from the last completed preview) alongside the %.
    text.draw_sized(frame, fb_w, fb_h, "size",
        pp.image.x, pp.size_row_text_y, px, [200, 210, 225]);
    let t_q = ((composer.preview_quality_pct as f64 - 10.0) / 90.0).clamp(0.0, 1.0);
    draw_slider(frame, fb_w, fb_h, pp.size_slider, t_q);
    let size_val = {
        let ps = composer.preview_state.read();
        if ps.w > 0 {
            format!("{}×{} ({}%)", ps.w, ps.h, composer.preview_quality_pct)
        } else {
            format!("({}%)", composer.preview_quality_pct)
        }
    };
    text.draw_sized(frame, fb_w, fb_h, &size_val,
        pp.size_val_x, pp.size_row_text_y, px, [210, 220, 235]);

    // Preview spp slider (log). Value label shows the effective samples
    // per pixel (which can exceed the slider value when the preview's
    // 20k-sample floor kicks in at small sizes).
    text.draw_sized(frame, fb_w, fb_h, "spp",
        pp.image.x, pp.spp_row_text_y, px, [200, 210, 225]);
    let lo = composer::PREVIEW_SPP_MIN.ln();
    let hi = composer::PREVIEW_SPP_MAX.ln();
    let t_s = ((composer.spp.max(composer::PREVIEW_SPP_MIN).ln() - lo) / (hi - lo))
        .clamp(0.0, 1.0);
    draw_slider(frame, fb_w, fb_h, pp.spp_slider, t_s);
    let spp_val = {
        let ps = composer.preview_state.read();
        if ps.w > 0 && ps.h > 0 {
            let pix = (ps.w as u64) * (ps.h as u64);
            let samples = (((pix as f64) * composer.spp).round() as u64)
                .max(20_000);
            let eff = samples as f64 / pix as f64;
            format!("{:.1} ({:.0}s/p)", composer.spp, eff)
        } else {
            format!("{:.1}", composer.spp)
        }
    };
    text.draw_sized(frame, fb_w, fb_h, &spp_val,
        pp.spp_val_x, pp.spp_row_text_y, px, [210, 220, 235]);

    // Palette gradient bar.
    draw_palette_bar(frame, fb_w, fb_h, pp.palette_bar, &composer.palette);

    // Reset button (restores default palette stops).
    draw_labeled_button(text, frame, fb_w, fb_h,
        pp.palette_reset_btn, "reset", scale,
        [32, 38, 52, 230], [120, 140, 170, 230], [220, 225, 235]);

    // Stop markers below the bar.
    draw_palette_stops(frame, fb_w, fb_h, pp.palette_bar, pp.palette_stops_y,
        pp.palette_stops_h, &composer.palette_stops, composer.palette_color_edit);

    // Gamma slider — applied between histogram and palette during preview.
    text.draw_sized(frame, fb_w, fb_h, "gamma",
        pp.image.x, pp.gamma_row_text_y, px, [200, 210, 225]);
    let t_g = ((composer.gamma - composer::GAMMA_MIN)
        / (composer::GAMMA_MAX - composer::GAMMA_MIN)).clamp(0.0, 1.0) as f64;
    draw_slider(frame, fb_w, fb_h, pp.gamma_slider, t_g);
    text.draw_sized(frame, fb_w, fb_h,
        &format!("{:.2}", composer.gamma),
        pp.gamma_val_x, pp.gamma_row_text_y, px, [210, 220, 235]);

    // Color popover if a stop is being color-edited.
    if let Some(stop_idx) = composer.palette_color_edit {
        if let Some(stop) = composer.palette_stops.get(stop_idx) {
            draw_color_popover(text, frame, fb_w, fb_h, pp.color_popover,
                scale, stop_idx, stop);
        }
    }
}

/// Draw the palette's 1024-LUT as a horizontal gradient inside `rect`.
fn draw_palette_bar(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    rect: Rect, palette: &palette::Palette,
) {
    let n = rect.w as usize;
    for i in 0..n {
        let t = i as f64 / (n - 1).max(1) as f64;
        let idx = ((t * (palette::LUT_SIZE - 1) as f64) as usize)
            .min(palette::LUT_SIZE - 1);
        let c = palette.lut[idx];
        overlay::fill_rect(frame, fb_w, fb_h,
            rect.x + i as i32, rect.y, 1, rect.h,
            [c[0], c[1], c[2], 255]);
    }
    draw_button_frame(frame, fb_w, fb_h, rect,
        [0, 0, 0, 0], [90, 100, 120, 220]);
}

/// Draw stop markers as small upward triangles under the palette bar.
fn draw_palette_stops(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    bar: Rect, stops_y: i32, stops_h: u32,
    stops: &[composer::PaletteStop],
    selected: Option<usize>,
) {
    for (i, s) in stops.iter().enumerate() {
        let sx = bar.x + (s.pos.clamp(0.0, 1.0) * bar.w as f32).round() as i32;
        let size = stops_h as i32;
        // Border color — brighter if selected.
        let border = if selected == Some(i) { [240, 240, 100, 255] }
                     else { [210, 220, 240, 255] };
        // Fill background of marker with the stop's color so it's identifiable.
        for dy in 0..size {
            let half = (size - dy) / 2 + 1;
            for dx in -half..=half {
                put_pixel(frame, fb_w, fb_h, sx + dx, stops_y + dy,
                    [s.rgb[0], s.rgb[1], s.rgb[2], 255]);
            }
        }
        // Border outline on the triangle edges.
        for dy in 0..size {
            let half = (size - dy) / 2 + 1;
            put_pixel(frame, fb_w, fb_h, sx - half, stops_y + dy, border);
            put_pixel(frame, fb_w, fb_h, sx + half, stops_y + dy, border);
        }
    }
}

/// Draw the RGB-slider popover for editing a stop's color.
fn draw_color_popover(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    rect: Rect, scale: f64,
    _stop_idx: usize,
    stop: &composer::PaletteStop,
) {
    draw_button_frame(frame, fb_w, fb_h, rect,
        [18, 22, 32, 240], [120, 140, 180, 240]);
    let pad = (8.0 * scale).round() as i32;
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;
    let label_w = (22.0 * scale).round() as i32;
    let val_w = (36.0 * scale).round() as i32;
    let slider_h = (10.0 * scale).round() as u32;
    let slider_w = (rect.w as i32 - pad * 2 - label_w - val_w).max(40) as u32;
    let slider_x = rect.x + pad + label_w;

    let labels = [("R", stop.rgb[0]), ("G", stop.rgb[1]), ("B", stop.rgb[2])];
    for (i, (name, v)) in labels.iter().enumerate() {
        let row_y = rect.y + pad + line_h * i as i32;
        text.draw_sized(frame, fb_w, fb_h, name,
            rect.x + pad, row_y + line_h - 2, px, [210, 220, 235]);
        let sl = Rect {
            x: slider_x,
            y: row_y + (line_h - slider_h as i32) / 2,
            w: slider_w, h: slider_h,
        };
        draw_slider(frame, fb_w, fb_h, sl, *v as f64 / 255.0);
        text.draw_sized(frame, fb_w, fb_h, &v.to_string(),
            slider_x + slider_w as i32 + (6.0 * scale).round() as i32,
            row_y + line_h - 2, px, [210, 220, 235]);
    }

    // Hint line at the bottom.
    let hint_y = rect.y + rect.h as i32 - line_h + 2;
    text.draw_sized(frame, fb_w, fb_h,
        "click outside to close  •  right-click marker to delete",
        rect.x + pad, hint_y, px.saturating_sub(1), [140, 150, 170]);
}

/// Layout for the grid right-click context menu. Two rows:
/// "Use in Live" (load coords into the live view) and "Trash" (move to trash).
/// Pre-clamped against the window so it never clips offscreen.
struct GridMenuLayout {
    frame: Rect,
    use_in_live: Rect,
    trash: Rect,
}

/// Rects for the load-timeline dialog: outer frame, plus one row rect per
/// saved entry (index-aligned with `dlg.entries`). Anchored under the load
/// button. When there are no entries a single "no saves yet" row is returned
/// so the user gets visible feedback rather than an empty void.
struct LoadDialogLayout {
    frame: Rect,
    rows: Vec<Rect>,
    /// Row height; also used for the synthetic "empty" row.
    row_h: i32,
    /// True when `dlg.entries` is empty — draw callers render a placeholder
    /// row in `rows[0]` and ignore click routing.
    is_empty: bool,
}

fn load_dialog_layout(
    anchor: Rect, dlg: &composer::LoadDialog,
    fb_w: u32, fb_h: u32, scale: f64,
) -> LoadDialogLayout {
    let w = (280.0 * scale).round() as i32;
    let row_h = (22.0 * scale).round() as i32;
    let pad = (4.0 * scale).round() as i32;
    let is_empty = dlg.entries.is_empty();
    let row_count = if is_empty { 1 } else { dlg.entries.len().min(16) };
    let h = row_h * row_count as i32 + pad * 2;
    // Anchor under the load button; flip upward if we'd spill offscreen.
    let mut x = anchor.x;
    let mut y = anchor.y + anchor.h as i32 + (2.0 * scale).round() as i32;
    if x + w > fb_w as i32 { x = (fb_w as i32 - w).max(0); }
    if y + h > fb_h as i32 {
        y = (anchor.y - h - (2.0 * scale).round() as i32).max(0);
    }
    let frame_r = Rect { x, y, w: w as u32, h: h as u32 };
    let mut rows = Vec::with_capacity(row_count);
    for i in 0..row_count {
        rows.push(Rect {
            x: x + pad,
            y: y + pad + row_h * i as i32,
            w: (w - pad * 2) as u32,
            h: row_h as u32,
        });
    }
    LoadDialogLayout { frame: frame_r, rows, row_h, is_empty }
}

fn draw_load_dialog(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    cursor: PhysicalPosition<f64>,
    dlg: &composer::LoadDialog,
    ll: &LoadDialogLayout,
) {
    draw_button_frame(frame, fb_w, fb_h, ll.frame,
        [18, 22, 32, 245], [120, 140, 180, 240]);
    let cx = cursor.x as i32;
    let cy = cursor.y as i32;
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let pad_x = (8.0 * scale).round() as i32;
    let text_y_off = ll.row_h / 2 + 4;

    if ll.is_empty {
        let r = ll.rows[0];
        text.draw_sized(frame, fb_w, fb_h, "no saves yet",
            r.x + pad_x, r.y + text_y_off, px, [170, 180, 200]);
        return;
    }

    for (i, r) in ll.rows.iter().enumerate() {
        let Some(entry) = dlg.entries.get(i) else { continue };
        let hot = r.contains(cx, cy);
        if hot {
            overlay::fill_rect(frame, fb_w, fb_h,
                r.x, r.y, r.w, r.h, [22, 50, 70, 220]);
        }
        let color = if hot { [220, 235, 250] } else { [200, 215, 235] };
        let right = format!("{} items", entry.item_count);
        let right_w = text.measure(&right, px) as i32;
        text.draw_sized(frame, fb_w, fb_h, &entry.display,
            r.x + pad_x, r.y + text_y_off, px, color);
        text.draw_sized(frame, fb_w, fb_h, &right,
            r.x + r.w as i32 - right_w - pad_x,
            r.y + text_y_off, px, [150, 170, 200]);
    }
}

fn grid_context_menu_layout(
    menu: &composer::GridContextMenu,
    fb_w: u32, fb_h: u32, scale: f64,
) -> GridMenuLayout {
    let w = (150.0 * scale).round() as i32;
    let row_h = (26.0 * scale).round() as i32;
    let pad = (4.0 * scale).round() as i32;
    let h = row_h * 2 + pad * 2;
    // Anchor at cursor; flip to the left / above if we'd spill offscreen.
    let mut x = menu.anchor_x as i32;
    let mut y = menu.anchor_y as i32;
    if x + w > fb_w as i32 { x = (fb_w as i32 - w).max(0); }
    if y + h > fb_h as i32 { y = (fb_h as i32 - h).max(0); }
    let frame_r = Rect { x, y, w: w as u32, h: h as u32 };
    let use_in_live = Rect {
        x: x + pad,
        y: y + pad,
        w: (w - pad * 2) as u32,
        h: row_h as u32,
    };
    let trash = Rect {
        x: x + pad,
        y: y + pad + row_h,
        w: (w - pad * 2) as u32,
        h: row_h as u32,
    };
    GridMenuLayout { frame: frame_r, use_in_live, trash }
}

fn draw_grid_context_menu(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    cursor: PhysicalPosition<f64>,
    ml: &GridMenuLayout,
) {
    // Panel.
    draw_button_frame(frame, fb_w, fb_h, ml.frame,
        [18, 22, 32, 245], [120, 140, 180, 240]);
    let cx = cursor.x as i32;
    let cy = cursor.y as i32;
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;
    let label_x = (8.0 * scale).round() as i32;

    // "Use in Live" row.
    let hot_live = ml.use_in_live.contains(cx, cy);
    if hot_live {
        overlay::fill_rect(frame, fb_w, fb_h,
            ml.use_in_live.x, ml.use_in_live.y,
            ml.use_in_live.w, ml.use_in_live.h,
            [22, 50, 70, 220]);
    }
    let live_color = if hot_live { [220, 235, 250] } else { [200, 215, 235] };
    let live_text_y = ml.use_in_live.y + (ml.use_in_live.h as i32 + line_h) / 2 - 4;
    text.draw_sized(frame, fb_w, fb_h, "Use in Live",
        ml.use_in_live.x + label_x, live_text_y, px, live_color);

    // "Trash" row.
    let hot_trash = ml.trash.contains(cx, cy);
    if hot_trash {
        overlay::fill_rect(frame, fb_w, fb_h,
            ml.trash.x, ml.trash.y, ml.trash.w, ml.trash.h,
            [60, 30, 36, 220]);
    }
    let trash_color = if hot_trash { [250, 210, 215] } else { [220, 200, 210] };
    let trash_text_y = ml.trash.y + (ml.trash.h as i32 + line_h) / 2 - 4;
    text.draw_sized(frame, fb_w, fb_h, "Trash",
        ml.trash.x + label_x, trash_text_y, px, trash_color);
}

enum StopHit {
    /// Cursor is within a stop marker's hit area → drag-to-reposition.
    Grab,
    /// Cursor is on the marker but drag hasn't started yet — interpreted as
    /// a click-to-open-color-popover (currently the same as Grab until the
    /// mouse moves). We emit Grab on press; the release handler promotes
    /// "no-move" to Click.
    #[allow(dead_code)]
    Click,
}

/// Hit-test the palette stop markers. Returns `Some((idx, StopHit::Grab))`
/// for the closest marker within ≤6px horizontally, else None.
fn palette_stop_hit(
    composer: &Composer,
    bar: Rect,
    stops_y: i32,
    stops_h: u32,
    cx: i32,
    cy: i32,
) -> Option<(usize, StopHit)> {
    if cy < stops_y || cy >= stops_y + stops_h as i32 + 2 {
        return None;
    }
    let mut best: Option<(usize, i32)> = None;
    for (i, s) in composer.palette_stops.iter().enumerate() {
        let sx = bar.x + (s.pos.clamp(0.0, 1.0) * bar.w as f32).round() as i32;
        let d = (cx - sx).abs();
        if d <= 6 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| (i, StopHit::Grab))
}

fn apply_color_channel(
    composer: &mut Composer,
    stop_idx: usize,
    ch: composer::ColorChannelDrag,
    rect: Rect,
    cx: i32,
) {
    let t = ((cx - rect.x) as f64 / rect.w.max(1) as f64).clamp(0.0, 1.0);
    let v = (t * 255.0).round() as u8;
    if let Some(s) = composer.palette_stops.get_mut(stop_idx) {
        match ch {
            composer::ColorChannelDrag::R => s.rgb[0] = v,
            composer::ColorChannelDrag::G => s.rgb[1] = v,
            composer::ColorChannelDrag::B => s.rgb[2] = v,
            composer::ColorChannelDrag::None => return,
        }
    }
    composer.rebuild_palette();
}

/// Rects of the three RGB sliders inside a color popover rect.
fn color_popover_sliders(rect: Rect, scale: f64) -> [Rect; 3] {
    let pad = (8.0 * scale).round() as i32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;
    let label_w = (22.0 * scale).round() as i32;
    let val_w = (36.0 * scale).round() as i32;
    let slider_h = (10.0 * scale).round() as u32;
    let slider_w = (rect.w as i32 - pad * 2 - label_w - val_w).max(40) as u32;
    let slider_x = rect.x + pad + label_w;
    let mk = |i: i32| {
        let row_y = rect.y + pad + line_h * i;
        Rect {
            x: slider_x,
            y: row_y + (line_h - slider_h as i32) / 2,
            w: slider_w, h: slider_h,
        }
    };
    [mk(0), mk(1), mk(2)]
}

/// Hit-test the thumbnail grid. Returns:
///  * `Some(None)` — cursor is on the always-first "+" (generate rotations) tile
///  * `Some(Some(idx))` — cursor is on `composer.png_paths[idx]`
///  * `None` — cursor is outside the grid cells (gap, below, or not in grid rect)
fn grid_thumb_hit(
    composer: &Composer, grid: Rect, scale: f64, cx: i32, cy: i32,
) -> Option<Option<usize>> {
    if !grid.contains(cx, cy) { return None; }
    let ts = ((composer.thumb_size as f64) * scale).round() as i32;
    let gap = (8.0 * scale).round() as i32;
    let pad = (8.0 * scale).round() as i32;
    let col_w = ts + gap;
    let row_h = ts + gap + (14.0 * scale).round() as i32;
    let cols = ((grid.w as i32 - pad * 2 + gap) / col_w).max(1);
    let local_x = cx - grid.x - pad;
    let local_y = cy - grid.y - pad + composer.grid_scroll.round() as i32;
    if local_x < 0 || local_y < 0 { return None; }
    let col = local_x / col_w;
    let row = local_y / row_h;
    if col >= cols { return None; }
    let inner_y = local_y % row_h;
    if inner_y >= ts { return None; }
    let flat = (row * cols + col) as usize;
    if flat == 0 { return Some(None); }
    let idx = flat - 1;
    if idx >= composer.png_paths.len() { return None; }
    Some(Some(idx))
}

fn draw_thumb_grid(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    rect: Rect,
) {
    // Compute thumbnail geometry.
    let ts = ((composer.thumb_size as f64) * scale).round() as u32;
    let gap = (8.0 * scale).round() as i32;
    let pad = (8.0 * scale).round() as i32;
    let col_w = ts as i32 + gap;
    let row_h = ts as i32 + gap + (14.0 * scale).round() as i32;
    let cols = ((rect.w as i32 - pad * 2 + gap) / col_w).max(1);

    let thumbs_map = composer.thumbs.read();
    let scroll = composer.grid_scroll.round() as i32;

    let px = (HUD_BASE_PX as f64 * scale).round() as u32;

    // Per-path occurrence count: how many times this screenshot appears on
    // the timeline. Used to stamp "★" badges on the thumb corner.
    let mut counts: std::collections::HashMap<&std::path::Path, usize>
        = std::collections::HashMap::new();
    for it in &composer.timeline {
        *counts.entry(it.source.as_path()).or_insert(0) += 1;
    }

    // Slot 0 is the always-present "+" tile (generate rotation keyframes).
    {
        let cx = rect.x + pad;
        let cy = rect.y + pad - scroll;
        if cy + row_h as i32 > rect.y && cy < rect.y + rect.h as i32 {
            let tile = Rect { x: cx, y: cy, w: ts, h: ts };
            overlay::fill_rect(frame, fb_w, fb_h,
                tile.x, tile.y, tile.w, tile.h, [40, 44, 54, 255]);
            draw_button_frame(frame, fb_w, fb_h, tile,
                [0, 0, 0, 0], [90, 100, 120, 220]);
            // Thick plus.
            let plus_r = (ts as i32 / 3).max(10);
            let thickness = (plus_r / 3).max(3) as u32;
            let ccx = tile.x + ts as i32 / 2;
            let ccy = tile.y + ts as i32 / 2;
            overlay::fill_rect(frame, fb_w, fb_h,
                ccx - plus_r, ccy - thickness as i32 / 2,
                (plus_r * 2) as u32, thickness, [140, 150, 170, 255]);
            overlay::fill_rect(frame, fb_w, fb_h,
                ccx - thickness as i32 / 2, ccy - plus_r,
                thickness, (plus_r * 2) as u32, [140, 150, 170, 255]);
            text.draw_sized(frame, fb_w, fb_h, "generate rotation",
                tile.x, tile.y + ts as i32 + (12.0 * scale).round() as i32,
                px, [170, 180, 200]);
        }
    }

    for (idx, path) in composer.png_paths.iter().enumerate() {
        // Grid index: shift everything by 1 for the "+" tile at slot 0.
        let flat = idx as i32 + 1;
        let row = flat / cols;
        let col = flat % cols;
        let cx = rect.x + pad + col * col_w;
        let cy = rect.y + pad + row * row_h - scroll;
        if cy + row_h < rect.y { continue; }
        if cy > rect.y + rect.h as i32 { break; }
        let thumb_rect = Rect {
            x: cx, y: cy, w: ts, h: ts,
        };

        // Clip test: only draw if visible within the grid's rect.
        if thumb_rect.y + thumb_rect.h as i32 <= rect.y { continue; }

        // Thumbnail or placeholder.
        overlay::fill_rect(frame, fb_w, fb_h, cx, cy, ts, ts, [25, 30, 40, 255]);
        if let Some(t) = thumbs_map.get(path) {
            composer::blit_thumb(frame, fb_w, fb_h, t, thumb_rect);
        } else {
            // Placeholder text
            text.draw_sized(frame, fb_w, fb_h, "…",
                cx + ts as i32 / 2 - 4, cy + ts as i32 / 2 + 4,
                px, [120, 130, 150]);
        }

        // Filename — middle-truncate with ellipsis so the string never
        // exceeds the thumbnail width.
        let full = path.file_name()
            .and_then(|n| n.to_str()).unwrap_or_default();
        let fname = truncate_middle_to_fit(text, full, ts, px);
        text.draw_sized(frame, fb_w, fb_h, &fname,
            cx, cy + ts as i32 + (12.0 * scale).round() as i32,
            px, [180, 190, 210]);

        // "On timeline" badge: one gold star per occurrence (up to 4), plus
        // "+N" for the overflow. Drawn top-right inside the thumbnail.
        let n_stars = counts.get(path.as_path()).copied().unwrap_or(0);
        if n_stars > 0 {
            draw_thumb_stars(text, frame, fb_w, fb_h, scale, thumb_rect, n_stars);
        }
    }
}

/// Draw up to 4 gold ★ glyphs in the top-right corner of `thumb`; when the
/// count exceeds 4, draw "+N" to the left of the 4 stars. Uses a 1-px black
/// outline so stars stay readable on bright thumbnails.
fn draw_thumb_stars(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64, thumb: Rect, n: usize,
) {
    let px = ((HUD_BASE_PX as f64 * 1.15 * scale).round() as u32).max(11);
    let pad = (4.0 * scale).round() as i32;
    let glyph = "★";
    let gw = text.measure(glyph, px) as i32;
    // If fontdue has no ★ (width 0), fall back to a filled diamond of the
    // same size — still legible, matches badge semantics.
    let has_star = gw > 0;
    let eff_w = if has_star { gw } else { ((12.0 * scale).round() as i32).max(8) };
    let base_y = thumb.y + pad + eff_w; // approximate baseline
    // Right-edge anchor; draw RTL.
    let mut anchor_x = thumb.x + thumb.w as i32 - pad;
    let visible = n.min(4);
    // If overflow, reserve space for the "+N" label first.
    let overflow_label = if n > 4 { Some(format!("+{}", n - 4)) } else { None };
    // Draw stars.
    for _k in 0..visible {
        let x = anchor_x - eff_w;
        if has_star {
            // Black outline via offset copies.
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                text.draw_sized(frame, fb_w, fb_h, glyph,
                    x + dx, base_y + dy, px, [0, 0, 0]);
            }
            text.draw_sized(frame, fb_w, fb_h, glyph, x, base_y, px,
                [255, 220, 80]);
        } else {
            // Diamond fallback.
            let cx = x + eff_w / 2;
            let cy = base_y - eff_w / 2;
            let r = eff_w / 2 - 1;
            for dy in -r..=r {
                let half = r - dy.abs();
                overlay::fill_rect(frame, fb_w, fb_h,
                    cx - half, cy + dy, (half * 2 + 1) as u32, 1,
                    [255, 220, 80, 255]);
            }
            // 1-px outline via fill_rect strips around it
            for dy in -(r + 1)..=(r + 1) {
                let halfo = (r + 1) - dy.abs();
                if halfo >= 0 {
                    put_pixel(frame, fb_w, fb_h, cx - halfo, cy + dy, [0, 0, 0, 255]);
                    put_pixel(frame, fb_w, fb_h, cx + halfo, cy + dy, [0, 0, 0, 255]);
                }
            }
        }
        anchor_x = x - 1;
    }
    if let Some(label) = overflow_label.as_ref() {
        let lw = text.measure(label, px) as i32;
        let x = anchor_x - lw - 2;
        // Black outline.
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            text.draw_sized(frame, fb_w, fb_h, label,
                x + dx, base_y + dy, px, [0, 0, 0]);
        }
        text.draw_sized(frame, fb_w, fb_h, label, x, base_y, px,
            [255, 240, 180]);
    }
}

/// Truncate `s` with a middle-ellipsis so its rendered width does not exceed
/// `max_w`. Keeps a prefix + suffix around the ellipsis; suffix is biased one
/// character longer so file extensions remain visible (".png", ".mp4").
fn truncate_middle_to_fit(
    text: &mut TextRenderer, s: &str, max_w: u32, px: u32,
) -> String {
    if max_w == 0 { return String::new(); }
    if text.measure(s, px) <= max_w {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 2 {
        return "…".to_string();
    }
    // Try keeping progressively fewer visible characters. First fit wins.
    for keep in (2..chars.len()).rev() {
        // Bias suffix 1 longer so the extension survives.
        let back = keep / 2 + keep % 2;
        let front = keep - back;
        if front == 0 { continue; }
        let front_s: String = chars[..front].iter().collect();
        let back_s: String = chars[chars.len() - back..].iter().collect();
        let candidate = format!("{}…{}", front_s, back_s);
        if text.measure(&candidate, px) <= max_w {
            return candidate;
        }
    }
    "…".to_string()
}

/// Layout of the embedded render-task panel at the bottom-right.
struct RenderPanelLayout {
    pad: i32,
    header_y: i32,
    width_label_y: i32,
    width_field: Rect,
    height_label_y: i32,
    height_field: Rect,
    /// Resolution preset radio rects. Same length / order as
    /// `composer::RESOLUTION_PRESETS`.
    res_radios: Vec<Rect>,
    spp_label_y: i32,
    spp_slider: Rect,
    spp_val_x: i32,
    estimate_y: i32,
    clear_btn: Rect,
    render_btn: Rect,
    /// Top y of the queue list region (may fit 0..N rows depending on
    /// available vertical space).
    queue_list_y: i32,
    /// Bottom y of the queue list region (clip boundary).
    queue_list_bottom: i32,
    /// Height of each row in the queue list.
    queue_row_h: u32,
}

fn render_panel_layout(rect: Rect, scale: f64) -> RenderPanelLayout {
    let pad = (10.0 * scale).round() as i32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;
    let x = rect.x + pad;
    let inner_w = (rect.w as i32 - pad * 2).max(40) as u32;
    let field_w = ((110.0 * scale).round() as u32).min(inner_w / 2);
    let field_h = (24.0 * scale).round() as u32;
    let label_col_w = (70.0 * scale).round() as i32;
    let val_w = (60.0 * scale).round() as i32;
    let slider_w = (inner_w as i32 - label_col_w - val_w).max(60) as u32;
    let slider_h = (10.0 * scale).round() as u32;
    let slider_x = x + label_col_w;
    let val_x = slider_x + slider_w as i32 + (8.0 * scale).round() as i32;
    let btn_h = (28.0 * scale).round() as u32;

    let mut y = rect.y + pad + line_h;
    let header_y = y;
    y += line_h + 6;

    let width_label_y = y + line_h - 4;
    let width_y = y;
    let width_field = Rect { x: slider_x, y: y, w: field_w, h: field_h };
    y += field_h as i32 + 4;

    let height_label_y = y + line_h - 4;
    let height_field = Rect { x: slider_x, y: y, w: field_w, h: field_h };
    let height_bottom = height_field.y + height_field.h as i32;
    y += field_h as i32 + 6;

    // Resolution preset radios — laid out column-major in a 3-row × 2-column
    // grid sitting to the right of the width/height fields with a wide gap so
    // the buttons aren't crowded against the field's "px" suffix.
    let presets = composer::RESOLUTION_PRESETS;
    let n_rows = composer::RESOLUTION_GRID_ROWS as i32;
    let n_cols = ((presets.len() as i32 + n_rows - 1) / n_rows).max(1);
    let radio_left_gap = (60.0 * scale).round() as i32;
    let radio_x = width_field.x + width_field.w as i32 + radio_left_gap;
    let radios_w_total = ((rect.x + inner_w as i32) - radio_x).max(40);
    let col_gap = (10.0 * scale).round() as i32;
    let col_w = ((radios_w_total - col_gap * (n_cols - 1).max(0)) / n_cols).max(40);
    let radios_total_h = height_bottom - width_y;
    let radio_row_h = (radios_total_h / n_rows).max(14);
    let mut res_radios: Vec<Rect> = Vec::with_capacity(presets.len());
    for i in 0..presets.len() as i32 {
        let col = i / n_rows;
        let row = i % n_rows;
        res_radios.push(Rect {
            x: radio_x + col * (col_w + col_gap),
            y: width_y + row * radio_row_h,
            w: col_w as u32,
            h: radio_row_h as u32,
        });
    }

    let spp_label_y = y + line_h - 2;
    let spp_slider = Rect {
        x: slider_x,
        y: y + (line_h - slider_h as i32) / 2,
        w: slider_w, h: slider_h,
    };
    // Extra room before the estimate line so its ascent doesn't overlap the
    // spp slider row's baseline.
    y += line_h + 20;

    let estimate_y = y;
    y += line_h + 8;

    let clear_w = (80.0 * scale).round() as u32;
    let clear_btn = Rect { x, y, w: clear_w, h: btn_h };
    let render_w = inner_w - clear_w - (8.0 * scale).round() as u32;
    let render_btn = Rect {
        x: x + clear_w as i32 + (8.0 * scale).round() as i32,
        y, w: render_w, h: btn_h,
    };
    y += btn_h as i32 + 10;

    let queue_list_y = y;
    let queue_list_bottom = rect.y + rect.h as i32 - pad;
    let queue_row_h = (22.0 * scale).round() as u32;

    RenderPanelLayout {
        pad, header_y,
        width_label_y, width_field,
        height_label_y, height_field,
        res_radios,
        spp_label_y, spp_slider, spp_val_x: val_x,
        estimate_y, clear_btn, render_btn,
        queue_list_y, queue_list_bottom, queue_row_h,
    }
}

/// One row in the queue list. `x_btn` is the [×] button on the far right.
struct QueueRowLayout {
    rect: Rect,
    x_btn: Rect,
}

/// Enumerate queue-list row rects for an active render (if any, shown first)
/// followed by each queued job. Rows are clipped to `queue_list_bottom`.
fn render_queue_row_rects(
    rl: &RenderPanelLayout,
    active: bool,
    queued: usize,
    panel_rect: Rect,
    scale: f64,
) -> (Option<QueueRowLayout>, Vec<QueueRowLayout>) {
    let pad = (10.0 * scale).round() as i32;
    let x = panel_rect.x + pad;
    let w = (panel_rect.w as i32 - pad * 2).max(40) as u32;
    let btn_size = rl.queue_row_h - 4;
    let mut y = rl.queue_list_y;
    let mk = |y: i32| {
        let rect = Rect { x, y, w, h: rl.queue_row_h };
        let x_btn = Rect {
            x: x + w as i32 - btn_size as i32 - 2,
            y: y + 2,
            w: btn_size,
            h: btn_size,
        };
        QueueRowLayout { rect, x_btn }
    };

    let mut out_active = None;
    if active {
        if y + rl.queue_row_h as i32 <= rl.queue_list_bottom {
            out_active = Some(mk(y));
            y += rl.queue_row_h as i32 + 2;
        }
    }
    let mut rows = Vec::new();
    for _ in 0..queued {
        if y + rl.queue_row_h as i32 > rl.queue_list_bottom { break; }
        rows.push(mk(y));
        y += rl.queue_row_h as i32 + 2;
    }
    (out_active, rows)
}

fn draw_render_panel(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    render_progress: Option<RenderProgress>,
    render_queue: &[RenderJob],
    rect: Rect,
) {
    let rl = render_panel_layout(rect, scale);
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let x = rect.x + rl.pad;

    // Header: "render" + timeline info.
    let dur_s = 2.0 * (composer.timeline.len().saturating_sub(1)) as f64;
    let total_frames = (composer.fps as f64 * dur_s).round() as u64;
    let header = if composer.timeline.len() < 2 {
        "render (add ≥2 keyframes)".to_string()
    } else {
        format!("render  •  {} fps  •  {:.1}s  •  {} frames",
            composer.fps, dur_s, total_frames)
    };
    text.draw_sized(frame, fb_w, fb_h, &header,
        x, rl.header_y, px, [200, 215, 240]);

    // Width + Height text fields.
    text.draw_sized(frame, fb_w, fb_h, "width",
        x, rl.width_label_y, px, [200, 210, 225]);
    draw_text_field(text, frame, fb_w, fb_h, rl.width_field,
        &composer.rp_width_text,
        composer.rp_focus == composer::RenderFieldFocus::Width, px);
    text.draw_sized(frame, fb_w, fb_h, "px",
        rl.width_field.x + rl.width_field.w as i32 + 8,
        rl.width_label_y, px, [160, 170, 190]);

    text.draw_sized(frame, fb_w, fb_h, "height",
        x, rl.height_label_y, px, [200, 210, 225]);
    draw_text_field(text, frame, fb_w, fb_h, rl.height_field,
        &composer.rp_height_text,
        composer.rp_focus == composer::RenderFieldFocus::Height, px);
    text.draw_sized(frame, fb_w, fb_h, "px",
        rl.height_field.x + rl.height_field.w as i32 + 8,
        rl.height_label_y, px, [160, 170, 190]);

    // Resolution preset radios. The selected radio is whichever preset's
    // (w, h) matches the current text fields exactly; if a user typed a
    // custom size, no radio is selected.
    let cur_w = composer.rp_width();
    let cur_h = composer.rp_height();
    for (rect, &(label, pw, ph)) in
        rl.res_radios.iter().zip(composer::RESOLUTION_PRESETS.iter())
    {
        let active = pw == cur_w && ph == cur_h;
        draw_radio_row(text, frame, fb_w, fb_h, *rect, scale, label, active);
    }

    // SPP slider (log).
    text.draw_sized(frame, fb_w, fb_h, "spp",
        x, rl.spp_label_y, px, [200, 210, 225]);
    let t_sp = (composer.rp_spp.ln() - SPP_MIN.ln())
        / (SPP_MAX.ln() - SPP_MIN.ln());
    draw_slider(frame, fb_w, fb_h, rl.spp_slider, t_sp.clamp(0.0, 1.0));
    text.draw_sized(frame, fb_w, fb_h, &format!("{:.1}", composer.rp_spp),
        rl.spp_val_x, rl.spp_label_y, px, [210, 220, 235]);

    // Estimate line — predictor for the next render based on the test
    // measurement against the current width/height/spp. Stays the same
    // shape regardless of whether something is currently rendering.
    let estimate = {
        use composer::TestStatus;
        let t = composer.rp_test.read();
        match &*t {
            TestStatus::NotRun => "estimate: —".to_string(),
            TestStatus::Running => "estimate: measuring…".to_string(),
            TestStatus::Done { frame_time } => {
                let per = frame_time.as_secs_f64();
                let total = Duration::from_secs_f64(per * total_frames as f64);
                format!("estimate: {}  •  {:.2}s/frame", fmt_dur(total), per)
            }
            TestStatus::Failed(e) => format!("test failed: {}", e),
        }
    };
    text.draw_sized(frame, fb_w, fb_h, &estimate,
        x, rl.estimate_y, px, [180, 195, 220]);

    // Clear + render buttons.
    draw_labeled_button(text, frame, fb_w, fb_h, rl.clear_btn,
        "clear", scale,
        [40, 24, 28, 230], [170, 120, 130, 240], [240, 220, 225]);

    let can_render = composer.timeline.len() >= 2;
    let (rfill, rborder, rtext) = if can_render {
        ([24, 50, 30, 230], [120, 200, 140, 240], [225, 245, 225])
    } else {
        ([28, 32, 40, 230], [80, 90, 100, 240], [140, 150, 170])
    };
    draw_labeled_button(text, frame, fb_w, fb_h, rl.render_btn,
        "render", scale, rfill, rborder, rtext);

    // Queue list.
    let (active_row, queued_rows) = render_queue_row_rects(
        &rl, render_progress.is_some(), render_queue.len(), rect, scale);

    if let (Some(row), Some(prog)) = (active_row, render_progress.as_ref()) {
        // Row background.
        overlay::fill_rect(frame, fb_w, fb_h,
            row.rect.x, row.rect.y, row.rect.w, row.rect.h, [18, 26, 42, 220]);
        draw_button_frame(frame, fb_w, fb_h, row.rect,
            [0, 0, 0, 0], [70, 100, 150, 220]);
        // Progress fill.
        let pct = prog.cur as f64 / prog.total.max(1) as f64;
        let fill_w = (row.rect.w as f64 * pct).round() as u32;
        overlay::fill_rect(frame, fb_w, fb_h,
            row.rect.x, row.rect.y + row.rect.h as i32 - 3,
            fill_w, 3, [80, 150, 220, 240]);
        // Label: progress + live ETA derived from observed frame timing
        // (elapsed / cur). The filename gets the leftover width.
        let elapsed = prog.started.elapsed().as_secs_f64();
        let timing = if prog.cur == 0 || elapsed <= 0.0 {
            "warming up…".to_string()
        } else {
            let per = elapsed / prog.cur as f64;
            let remaining = (prog.total.saturating_sub(prog.cur)) as f64;
            let eta = Duration::from_secs_f64((per * remaining).max(0.0));
            format!("eta {}  •  {:.2}s/frame", fmt_dur(eta), per)
        };
        let fname = prog.output.file_name()
            .and_then(|n| n.to_str()).unwrap_or("—");
        let prefix = format!("▶ {}/{}  •  {}  •  ",
            prog.cur, prog.total, timing);
        let avail_w = (row.x_btn.x - (row.rect.x + 6) - 6).max(0) as u32;
        let prefix_w = text.measure(&prefix, px);
        let fname_max_w = avail_w.saturating_sub(prefix_w);
        let fname_disp = truncate_middle_to_fit(text, fname, fname_max_w, px);
        let label = format!("{prefix}{fname_disp}");
        text.draw_sized(frame, fb_w, fb_h, &label,
            row.rect.x + 6, row.rect.y + row.rect.h as i32 - 6, px,
            [220, 230, 245]);
        // Cancel [×] button.
        draw_button_frame(frame, fb_w, fb_h, row.x_btn,
            [60, 30, 36, 230], [180, 100, 120, 240]);
        let tw = text.measure("×", px) as i32;
        text.draw_sized(frame, fb_w, fb_h, "×",
            row.x_btn.x + (row.x_btn.w as i32 - tw) / 2,
            row.x_btn.y + row.x_btn.h as i32 - 4,
            px, [240, 220, 225]);
    }

    for (i, row) in queued_rows.iter().enumerate() {
        overlay::fill_rect(frame, fb_w, fb_h,
            row.rect.x, row.rect.y, row.rect.w, row.rect.h, [14, 20, 32, 220]);
        draw_button_frame(frame, fb_w, fb_h, row.rect,
            [0, 0, 0, 0], [60, 70, 90, 200]);
        let job = &render_queue[i];
        let fname = job.output.file_name()
            .and_then(|n| n.to_str()).unwrap_or("—");
        let label = format!("● queued  •  {}", fname);
        text.draw_sized(frame, fb_w, fb_h, &label,
            row.rect.x + 6, row.rect.y + row.rect.h as i32 - 6, px,
            [180, 195, 220]);
        // Remove [×] button.
        draw_button_frame(frame, fb_w, fb_h, row.x_btn,
            [40, 30, 36, 220], [140, 100, 120, 230]);
        let tw = text.measure("×", px) as i32;
        text.draw_sized(frame, fb_w, fb_h, "×",
            row.x_btn.x + (row.x_btn.w as i32 - tw) / 2,
            row.x_btn.y + row.x_btn.h as i32 - 4,
            px, [230, 210, 215]);
    }
}

/// A +/- stepper widget: two small square buttons and a numeric label.
#[derive(Clone, Copy)]
pub struct StepperRect {
    pub inc: Rect,
    pub dec: Rect,
    /// Top-left of the numeric label text-anchor point (draw baseline).
    pub label_x: i32,
    pub label_y: i32,
    pub label_w: u32,
}

/// Full layout of the timeline panel.
///
/// - `slots[i]` — the i-th keyframe's thumbnail rect.
/// - `gaps[i]` — the rect between slot i and slot i+1 (hosts the segment
///   stepper).
/// - `tension_steppers[i]` — below slot i (per-keyframe tangent multiplier).
/// - `segment_steppers[i]` — inside gap i.
/// - `content_w` — total width of the timeline content (for scroll
///   clamping).
/// - `playhead_y{0,1}` — vertical range covered by the playhead line.
pub struct TimelineLayout {
    #[allow(dead_code)]
    pub panel: Rect,
    pub slots: Vec<Rect>,
    pub gaps: Vec<Rect>,
    pub tension_steppers: Vec<StepperRect>,
    pub segment_steppers: Vec<StepperRect>,
    pub content_w: i32,
    pub playhead_y0: i32,
    pub playhead_y1: i32,
}

fn timeline_layout(
    composer: &Composer, panel: Rect, scale: f64,
) -> TimelineLayout {
    let n = composer.timeline.len();
    let pad_x = (10.0 * scale).round() as i32;
    let pad_top = (8.0 * scale).round() as i32;
    let slot_h = (110.0 * scale).round() as i32;
    let slot_w = ((slot_h as f64 * 4.0 / 3.0).round()) as i32;
    let gap_w = (56.0 * scale).round() as i32;
    let btn = (18.0 * scale).round() as i32;
    let step_label_w = (56.0 * scale).round() as u32;
    let step_label_h = (14.0 * scale).round() as i32;

    let content_w = if n == 0 {
        0
    } else {
        (n as i32) * slot_w + (n.saturating_sub(1) as i32) * gap_w
    };

    // Center when content fits the panel; otherwise left-align with scroll.
    let x_origin = if content_w <= panel.w as i32 {
        panel.x + ((panel.w as i32 - content_w).max(pad_x) / 2)
    } else {
        panel.x + pad_x - composer.timeline_scroll_x.round() as i32
    };

    let slot_y = panel.y + pad_top;
    let slot_bottom = slot_y + slot_h;
    // Tension stepper stack (under each slot): [+]  label  [-]
    let tens_y = slot_bottom + (8.0 * scale).round() as i32;
    let tens_inc_y = tens_y;
    let tens_label_y = tens_inc_y + btn + (2.0 * scale).round() as i32;
    let tens_dec_y = tens_label_y + step_label_h + (2.0 * scale).round() as i32;

    // Segment stepper stack (in each gap, vertically centered on slot_y).
    let seg_center_y = slot_y + slot_h / 2;
    let seg_inc_y = seg_center_y - btn - (step_label_h + 4) / 2;
    let seg_label_y = seg_inc_y + btn + (2.0 * scale).round() as i32;
    let seg_dec_y = seg_label_y + step_label_h + (2.0 * scale).round() as i32;

    let mut slots: Vec<Rect> = Vec::with_capacity(n);
    let mut gaps: Vec<Rect> = Vec::with_capacity(n.saturating_sub(1));
    let mut tension_steppers: Vec<StepperRect> = Vec::with_capacity(n);
    let mut segment_steppers: Vec<StepperRect> = Vec::with_capacity(n.saturating_sub(1));

    let mut x = x_origin;
    for i in 0..n {
        let slot = Rect { x, y: slot_y, w: slot_w as u32, h: slot_h as u32 };
        slots.push(slot);

        // Tension stepper: centered under the thumb.
        let cxm = slot.x + slot_w / 2;
        let tens = StepperRect {
            inc: Rect {
                x: cxm - btn / 2, y: tens_inc_y,
                w: btn as u32, h: btn as u32,
            },
            dec: Rect {
                x: cxm - btn / 2, y: tens_dec_y,
                w: btn as u32, h: btn as u32,
            },
            label_x: cxm - step_label_w as i32 / 2,
            label_y: tens_label_y + step_label_h - 2,
            label_w: step_label_w,
        };
        tension_steppers.push(tens);

        x += slot_w;
        if i + 1 < n {
            let g = Rect { x, y: slot_y, w: gap_w as u32, h: slot_h as u32 };
            gaps.push(g);
            let gcx = g.x + gap_w / 2;
            let seg = StepperRect {
                inc: Rect {
                    x: gcx - btn / 2, y: seg_inc_y,
                    w: btn as u32, h: btn as u32,
                },
                dec: Rect {
                    x: gcx - btn / 2, y: seg_dec_y,
                    w: btn as u32, h: btn as u32,
                },
                label_x: gcx - step_label_w as i32 / 2,
                label_y: seg_label_y + step_label_h - 2,
                label_w: step_label_w,
            };
            segment_steppers.push(seg);
            x += gap_w;
        }
    }

    TimelineLayout {
        panel,
        slots,
        gaps,
        tension_steppers,
        segment_steppers,
        content_w,
        playhead_y0: slot_y,
        playhead_y1: tens_dec_y + btn,
    }
}

/// Legacy shim: return just the slot + gap rects so older call sites (drag
/// reorder, release handler) keep working while we incrementally migrate
/// them to `timeline_layout`.
fn timeline_slot_rects(
    composer: &Composer, rect: Rect, scale: f64,
) -> (Vec<Rect>, Vec<Rect>) {
    let l = timeline_layout(composer, rect, scale);
    (l.slots, l.gaps)
}

fn draw_timeline(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    rect: Rect,
) {
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    if composer.timeline.is_empty() {
        let msg = "timeline is empty — click screenshots to add keyframes";
        let tw = text.measure(msg, px) as i32;
        text.draw_sized(frame, fb_w, fb_h, msg,
            rect.x + (rect.w as i32 - tw) / 2,
            rect.y + rect.h as i32 / 2 + text.line_height(px) as i32 / 2 - 4,
            px, [130, 140, 160]);
        return;
    }

    let px_text_h = text.line_height(px) as i32;
    let layout = timeline_layout(composer, rect, scale);
    let slots = &layout.slots;
    let gaps = &layout.gaps;
    let dragging = composer.timeline_drag.as_ref().filter(|d| d.moved);
    let ghost_idx = dragging.map(|d| d.from_idx);
    let insert_idx = dragging.map(|d| timeline_insert_index_for_x(slots, d.cursor_x as i32));

    let thumbs = composer.thumbs.read();
    for (i, item) in composer.timeline.iter().enumerate() {
        let slot = slots[i];
        let is_ghost = ghost_idx == Some(i);
        let fill = if is_ghost {
            [20u8, 25, 35, 90]
        } else {
            [20u8, 25, 35, 230]
        };
        overlay::fill_rect(frame, fb_w, fb_h, slot.x, slot.y, slot.w, slot.h, fill);
        let border = if item.missing {
            [200, 90, 100, 240]
        } else {
            [100, 110, 130, 220]
        };
        draw_button_frame(frame, fb_w, fb_h, slot, fill, border);
        if !is_ghost {
            if let Some(t) = thumbs.get(&item.source) {
                composer::blit_thumb(frame, fb_w, fb_h, t, slot);
            }
            if item.missing {
                // Dim + red-tinted scrim, then an X across the thumb region.
                overlay::fill_rect(frame, fb_w, fb_h,
                    slot.x + 1, slot.y + 1,
                    slot.w.saturating_sub(2), slot.h.saturating_sub(2),
                    [20, 10, 14, 170]);
                draw_missing_x(frame, fb_w, fb_h, slot, scale);
                let label = "missing";
                let lw = text.measure(label, px) as i32;
                text.draw_sized(frame, fb_w, fb_h, label,
                    slot.x + (slot.w as i32 - lw) / 2,
                    slot.y + slot.h as i32 - 4,
                    px, [240, 180, 185]);
            }
            text.draw_sized(frame, fb_w, fb_h, &format!("{}", i + 1),
                slot.x + 4, slot.y + px_text_h - 2,
                px, [230, 235, 245]);
        }

        // Tension stepper under this thumb (per-keyframe tangent
        // multiplier; 1.0 = no slowdown).
        let stepper = &layout.tension_steppers[i];
        draw_stepper(text, frame, fb_w, fb_h, scale, stepper,
            &format!("{:.1}x", item.tension),
            [200, 210, 225]);

        // Delta arrow + segment stepper in the gap between this and the next.
        if i + 1 < composer.timeline.len() {
            let gap_r = gaps[i];
            // Thin delta-arrow band at the very top of the gap (above the
            // segment stepper).
            let arrow_rect = Rect {
                x: gap_r.x,
                y: gap_r.y,
                w: gap_r.w,
                h: (10.0 * scale).round() as u32,
            };
            draw_delta_arrow(frame, fb_w, fb_h,
                arrow_rect,
                &composer.timeline[i].spec,
                &composer.timeline[i + 1].spec);

            let seg = &layout.segment_steppers[i];
            draw_stepper(text, frame, fb_w, fb_h, scale, seg,
                &format!("{:.1}s", item.segment_seconds),
                [210, 220, 240]);
        }
    }

    // Insertion indicator while reordering.
    if let Some(ins) = insert_idx {
        let x = insertion_marker_x(slots, ins, rect);
        overlay::fill_rect(frame, fb_w, fb_h,
            x - 2, slots.first().map(|s| s.y).unwrap_or(rect.y + 4),
            4,
            slots.first().map(|s| s.h).unwrap_or(rect.h.saturating_sub(20)),
            [140, 200, 255, 240]);
    }

    // Floating ghost of the dragged item.
    if let (Some(d), Some(from)) = (dragging, ghost_idx) {
        let slot = slots[from];
        let dx = d.cursor_x as i32 - (slot.x + slot.w as i32 / 2);
        let dy = d.cursor_y as i32 - (slot.y + slot.h as i32 / 2);
        let ghost = Rect {
            x: slot.x + dx,
            y: slot.y + dy,
            w: slot.w,
            h: slot.h,
        };
        // If the ghost is outside the panel, tint it red to signal "release
        // here = remove".
        let outside = !rect.contains(d.cursor_x as i32, d.cursor_y as i32);
        let (fill, border) = if outside {
            ([40, 18, 22, 220], [220, 120, 130, 240])
        } else {
            ([20, 25, 35, 230], [150, 200, 255, 240])
        };
        if let Some(t) = thumbs.get(&composer.timeline[from].source) {
            overlay::fill_rect(frame, fb_w, fb_h, ghost.x, ghost.y, ghost.w, ghost.h,
                fill);
            draw_button_frame(frame, fb_w, fb_h, ghost, fill, border);
            composer::blit_thumb(frame, fb_w, fb_h, t, ghost);
        }
    }

    // Playhead line across the timeline.
    if composer.timeline.len() >= 2 {
        let x = playhead_x_on_timeline(&layout, composer);
        let y = layout.playhead_y0;
        let h = (layout.playhead_y1 - layout.playhead_y0).max(4) as u32;
        overlay::fill_rect(frame, fb_w, fb_h, x - 1, y, 2, h,
            [90, 170, 255, 235]);
        // Small triangle at top for affordance.
        for dy in 0..5 {
            let half = 5 - dy;
            overlay::fill_rect(frame, fb_w, fb_h,
                x - half, y - dy - 1, (half * 2 + 1) as u32, 1,
                [90, 170, 255, 235]);
        }
    }
}

/// Compute the pixel x-coordinate of the current playhead along the
/// timeline by linearly interpolating between adjacent slot centers
/// using the segment scheduler's `(from, local_u)`.
fn playhead_x_on_timeline(l: &TimelineLayout, c: &Composer) -> i32 {
    let n = c.timeline.len();
    if n == 0 { return l.panel.x; }
    let segments: Vec<f64> = c.timeline.iter().map(|t| t.segment_seconds).collect();
    let times = crate::videorender::cumulative_times(&segments, n);
    let total = *times.last().unwrap_or(&0.0);
    let (from, local_u) = crate::videorender::segment_for_playhead(
        &times, total, c.playhead);
    let a = l.slots[from];
    let b = l.slots[(from + 1).min(l.slots.len() - 1)];
    let ax = a.x + a.w as i32 / 2;
    let bx = b.x + b.w as i32 / 2;
    (ax as f64 + (bx - ax) as f64 * local_u).round() as i32
}

/// Draw a +/- stepper. `label` is centered between the two buttons.
fn draw_stepper(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    s: &StepperRect,
    label: &str,
    color: [u8; 3],
) {
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    let btn_fill = [24, 36, 60, 230];
    let btn_border = [120, 150, 200, 240];
    draw_button_frame(frame, fb_w, fb_h, s.inc, btn_fill, btn_border);
    draw_button_frame(frame, fb_w, fb_h, s.dec, btn_fill, btn_border);
    // + glyph: two centered bars
    let t = ((2.0 * scale).round() as u32).max(2);
    overlay::fill_rect(frame, fb_w, fb_h,
        s.inc.x + s.inc.w as i32 / 2 - (s.inc.w as i32 / 3),
        s.inc.y + s.inc.h as i32 / 2 - (t as i32 / 2),
        (s.inc.w * 2 / 3).max(4),
        t,
        [230, 235, 245, 240]);
    overlay::fill_rect(frame, fb_w, fb_h,
        s.inc.x + s.inc.w as i32 / 2 - (t as i32 / 2),
        s.inc.y + s.inc.h as i32 / 2 - (s.inc.h as i32 / 3),
        t,
        (s.inc.h * 2 / 3).max(4),
        [230, 235, 245, 240]);
    // - glyph
    overlay::fill_rect(frame, fb_w, fb_h,
        s.dec.x + s.dec.w as i32 / 2 - (s.dec.w as i32 / 3),
        s.dec.y + s.dec.h as i32 / 2 - (t as i32 / 2),
        (s.dec.w * 2 / 3).max(4),
        t,
        [230, 235, 245, 240]);
    // Numeric label — centered in its reserved width.
    let tw = text.measure(label, px) as i32;
    let lx = s.label_x + (s.label_w as i32 - tw) / 2;
    text.draw_sized(frame, fb_w, fb_h, label, lx, s.label_y, px, color);
}

/// Given the slot rects and a cursor x, compute where the dragged item would
/// be inserted (0..=slots.len()).
fn timeline_insert_index_for_x(slots: &[Rect], cx: i32) -> usize {
    for (i, s) in slots.iter().enumerate() {
        let mid = s.x + s.w as i32 / 2;
        if cx < mid { return i; }
    }
    slots.len()
}

fn insertion_marker_x(slots: &[Rect], insert: usize, rect: Rect) -> i32 {
    if slots.is_empty() { return rect.x + rect.w as i32 / 2; }
    if insert == 0 { return slots[0].x - 10; }
    if insert >= slots.len() {
        let s = slots[slots.len() - 1];
        return s.x + s.w as i32 + 10;
    }
    let prev = slots[insert - 1];
    let next = slots[insert];
    (prev.x + prev.w as i32 + next.x) / 2
}

/// Draw a red X across a rect (two diagonal strokes). Used on timeline slots
/// whose backing PNG is no longer on disk.
fn draw_missing_x(frame: &mut [u8], fb_w: u32, fb_h: u32, r: Rect, scale: f64) {
    let pad = ((r.w.min(r.h) as f64) * 0.15).round() as i32;
    let x0 = r.x + pad;
    let y0 = r.y + pad;
    let x1 = r.x + r.w as i32 - pad;
    let y1 = r.y + r.h as i32 - pad;
    let thick = ((3.0 * scale).round() as i32).max(2);
    draw_thick_line(frame, fb_w, fb_h, x0, y0, x1, y1, thick, [230, 90, 100, 240]);
    draw_thick_line(frame, fb_w, fb_h, x0, y1, x1, y0, thick, [230, 90, 100, 240]);
}

/// Rasterize a line of the given thickness by stamping a square brush along
/// Bresenham's algorithm. Ugly but good enough for an on-screen X.
fn draw_thick_line(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    x0: i32, y0: i32, x1: i32, y1: i32,
    thickness: i32, color: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let half = thickness / 2;
    loop {
        overlay::fill_rect(frame, fb_w, fb_h,
            x - half, y - half,
            thickness as u32, thickness as u32, color);
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}

fn draw_delta_arrow(
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    rect: Rect,
    a: &crate::render::FrameSpec,
    b: &crate::render::FrameSpec,
) {
    // Compute rough magnitudes of per-axis deltas.
    let dx = b.view.center[0] - a.view.center[0];
    let dy = b.view.center[1] - a.view.center[1];
    let dz = (b.view.half_width / a.view.half_width).ln().abs();
    let pan_mag = (dx * dx + dy * dy).sqrt();
    // Rotation magnitude: Frobenius norm of R_b - R_a (max ~ 2*sqrt(2)).
    let mut rot_mag = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = b.view.rotation[i][j] - a.view.rotation[i][j];
            rot_mag += d * d;
        }
    }
    let rot_mag = rot_mag.sqrt();

    // Background.
    overlay::fill_rect(frame, fb_w, fb_h, rect.x, rect.y, rect.w, rect.h,
        [24, 30, 40, 200]);

    // Stack three colored bars: rotation (magenta), pan (blue), zoom (yellow).
    let bar_h = (rect.h / 3).max(1);
    let magnitudes = [
        (rot_mag / 4.0, [200u8, 110, 220]),
        (pan_mag.min(4.0) / 4.0, [110u8, 170, 240]),
        (dz.min(2.0) / 2.0, [240u8, 220, 110]),
    ];
    for (i, (m, color)) in magnitudes.iter().enumerate() {
        let len = ((rect.w as f64) * m.clamp(0.0, 1.0)).round() as u32;
        let y = rect.y + (i as i32) * bar_h as i32 + 2;
        overlay::fill_rect(frame, fb_w, fb_h,
            rect.x + 1, y, len.max(1), bar_h.saturating_sub(2),
            [color[0], color[1], color[2], 230]);
    }
}

// ========================================================================
// Composer event handlers (free fns so they don't fight the ws borrow)
// ========================================================================

#[derive(Debug)]
enum ComposerAction {
    /// Open the popup render dialog (deprecated for composer; retained for
    /// any code paths that still call it).
    #[allow(dead_code)]
    OpenRenderDialog,
    /// Spawn a render directly using the embedded panel's current params
    /// and the composer's timeline keyframes.
    RenderFromPanel,
    /// Cancel the currently-running render (writes a `.cancel` marker into
    /// its session dir; the subprocess polls and exits cleanly).
    CancelActiveRender,
    /// Remove a queued (not yet spawned) render at index `i` from the queue.
    RemoveQueued(usize),
}

/// Shared geometry helpers for the composer's 4-pane layout.
struct ComposerLayout {
    grid: Rect,
    /// The preview pane: image, playback controls, preview-quality sliders,
    /// and the palette editor. No frame around it — blends with the
    /// background.
    preview: Rect,
    /// Motion panel between the preview and the render panel. Holds
    /// the V-N ρ (path pullback) slider.
    motion_panel: Rect,
    /// The render-task panel at the bottom-right: width/height/scale/spp
    /// render-dialog fields and the render button. Framed.
    render_panel: Rect,
    timeline: Rect,
    /// Manual save button in the top-left, above the grid.
    save_btn: Rect,
    /// Manual load button, right of `save_btn`.
    load_btn: Rect,
    /// The thumb-size slider shares the top-left strip with the save/load
    /// buttons. Shrunk relative to its original full-width layout so the
    /// buttons have room.
    thumb_slider: Rect,
    /// Rect reserved for the "thumb size" label. Positioned between the load
    /// button and the slider so the label can't overlap the button.
    thumb_label: Rect,
    /// Split x between left (grid) and right (preview + render panel).
    split_x: i32,
}

fn composer_layout(fb_w: u32, fb_h: u32, scale: f64, composer: &Composer)
    -> ComposerLayout
{
    let margin = (MARGIN_BASE as f64 * scale).round() as i32;
    let tab_bottom = ((4.0 + TAB_BAR_H as f64) * scale).round() as i32;
    // Top-left strip (save/load + thumb-size) sits in its own band between
    // the tab bar and the grid, so the grid shrinks to give the strip
    // breathing room instead of the strip intruding into the margin.
    let strip_pad = (6.0 * scale).round() as i32;
    let strip_h = (22.0 * scale).round() as u32;
    let strip_y = tab_bottom + strip_pad;
    let top = strip_y + strip_h as i32 + strip_pad;

    // Ideal split from the user-controlled fraction.
    let frac = composer.split_frac
        .clamp(composer::SPLIT_FRAC_MIN, composer::SPLIT_FRAC_MAX);
    let mut split_x = ((fb_w as f64) * frac).round() as i32;

    // Clamp so the grid keeps at least 2 thumb columns and the preview keeps
    // at least 200 px. These override the fraction in degenerate aspect
    // ratios (tall-narrow window).
    let ts = ((composer.thumb_size as f64) * scale).round() as i32;
    let gap = (8.0 * scale).round() as i32;
    let pad = (8.0 * scale).round() as i32;
    let min_grid_w = 2 * ts + gap + 2 * pad;
    let lo = margin + min_grid_w + 4;
    let hi = fb_w as i32 - 200 - margin;
    if hi > lo {
        split_x = split_x.clamp(lo, hi);
    } else {
        split_x = split_x.max(lo);
    }

    let timeline_h = ((260.0 * scale).round() as i32).max(180);
    let upper_bottom = fb_h as i32 - timeline_h - margin;

    let grid = Rect {
        x: margin,
        y: top,
        w: (split_x - margin - 4).max(100) as u32,
        h: (upper_bottom - top).max(100) as u32,
    };
    // Render panel gets a fixed minimum height that fits its content
    // (width/height fields, scale slider, spp slider, estimate, button).
    // Preview absorbs everything else in the upper area.
    let right_h_total = upper_bottom - top;
    let rp_target = ((260.0 * scale).round() as i32)
        .min((right_h_total - 200).max(160));
    let render_panel_h = rp_target.max(160);
    let motion_panel_h = ((40.0 * scale).round() as i32).max(32);
    let preview_h = (right_h_total - render_panel_h - motion_panel_h - 8).max(160);
    let right_w = (fb_w as i32 - split_x - margin).max(100) as u32;
    let preview = Rect {
        x: split_x,
        y: top,
        w: right_w,
        h: preview_h as u32,
    };
    let motion_panel = Rect {
        x: split_x,
        y: top + preview_h + 4,
        w: right_w,
        h: motion_panel_h as u32,
    };
    let render_panel = Rect {
        x: split_x,
        y: top + preview_h + 4 + motion_panel_h + 4,
        w: right_w,
        h: render_panel_h as u32,
    };
    let timeline = Rect {
        x: margin,
        y: upper_bottom + margin,
        w: (fb_w as i32 - 2 * margin) as u32,
        h: (timeline_h - 2) as u32,
    };
    // Top-left strip: [save] [gap] [load] [gap] [thumb size] [gap] [slider].
    // Laid out strictly left-to-right with hard-reserved widths for every
    // element so the label can't overlap a button regardless of fb_w.
    let btn_w = (52.0 * scale).round() as i32;
    let btn_gap = (6.0 * scale).round() as i32;
    let save_btn = Rect {
        x: grid.x,
        y: strip_y,
        w: btn_w as u32,
        h: strip_h,
    };
    let load_btn = Rect {
        x: save_btn.x + btn_w + btn_gap,
        y: strip_y,
        w: btn_w as u32,
        h: strip_h,
    };
    // Visual gap between the load button and the thumb-size label — bigger
    // than btn_gap so the label clearly separates from the button group.
    let post_btn_gap = (18.0 * scale).round() as i32;
    let label_w = (80.0 * scale).round() as i32;
    let label_slider_gap = (8.0 * scale).round() as i32;
    let thumb_label = Rect {
        x: load_btn.x + btn_w + post_btn_gap,
        y: strip_y,
        w: label_w as u32,
        h: strip_h,
    };
    let slider_h = (10.0 * scale).round() as i32;
    let slider_x = thumb_label.x + label_w + label_slider_gap;
    let slider_cap = (180.0 * scale).round() as i32;
    let slider_end_max = grid.x + grid.w as i32;
    let slider_w = (slider_end_max - slider_x).clamp(40, slider_cap);
    let thumb_slider = Rect {
        x: slider_x,
        y: strip_y + (strip_h as i32 - slider_h) / 2,
        w: slider_w as u32,
        h: slider_h as u32,
    };

    ComposerLayout {
        grid, preview, motion_panel, render_panel, timeline,
        save_btn, load_btn, thumb_slider, thumb_label, split_x,
    }
}

/// Layout for the motion panel's single row: label on the left, slider
/// in the middle, value on the right — mirrors the render panel's
/// row-plus-slider style.
struct MotionPanelLayout {
    slider: Rect,
    label_x: i32,
    val_x: i32,
    text_y: i32,
}

fn motion_panel_layout(rect: Rect, scale: f64) -> MotionPanelLayout {
    let pad = (10.0 * scale).round() as i32;
    let line_h = ((HUD_BASE_PX as f64) * 1.3 * scale).round() as i32;
    let x = rect.x + pad;
    let inner_w = (rect.w as i32 - pad * 2).max(40) as u32;
    let label_col_w = (90.0 * scale).round() as i32;
    let val_w = (54.0 * scale).round() as i32;
    let slider_w = (inner_w as i32 - label_col_w - val_w).max(60) as u32;
    let slider_h = (10.0 * scale).round() as u32;
    let slider_x = x + label_col_w;
    let val_x = slider_x + slider_w as i32 + (8.0 * scale).round() as i32;

    let row_y = rect.y + (rect.h as i32 - slider_h as i32) / 2;
    let slider = Rect { x: slider_x, y: row_y, w: slider_w, h: slider_h };
    let text_y = rect.y + (rect.h as i32 + line_h) / 2 - 4;
    MotionPanelLayout { slider, label_x: x, val_x, text_y }
}

fn draw_motion_panel(
    text: &mut TextRenderer,
    frame: &mut [u8], fb_w: u32, fb_h: u32,
    scale: f64,
    composer: &Composer,
    rect: Rect,
) {
    let ml = motion_panel_layout(rect, scale);
    let px = (HUD_BASE_PX as f64 * scale).round() as u32;
    text.draw_sized(frame, fb_w, fb_h, "path ρ",
        ml.label_x, ml.text_y, px, [200, 210, 225]);
    let t = ((composer.rho - composer::RHO_MIN)
        / (composer::RHO_MAX - composer::RHO_MIN)).clamp(0.0, 1.0);
    draw_slider(frame, fb_w, fb_h, ml.slider, t);
    text.draw_sized(frame, fb_w, fb_h,
        &format!("{:.2}", composer.rho),
        ml.val_x, ml.text_y, px, [210, 220, 235]);
}

/// Rect of the draggable vertical splitter between grid and preview/controls.
/// 6-px hit region centered on `split_x`, spanning the grid's vertical extent.
fn composer_splitter_rect(l: &ComposerLayout) -> Rect {
    Rect {
        x: l.split_x - 3,
        y: l.grid.y,
        w: 6,
        h: l.grid.h,
    }
}

fn composer_timeline_rect(fb_w: u32, fb_h: u32, scale: f64, composer: &Composer)
    -> Rect
{
    composer_layout(fb_w, fb_h, scale, composer).timeline
}

/// Layout of the composer controls pane. Single source of truth shared by
/// the draw and hit-test code so slider rects, button rects, and group-header
/// y-coordinates can't drift out of sync.
/// Enumerate the six composer sliders (thumb size, playhead/progress,
/// preview size/spp, render scale/spp) with their drag-state kind.
fn composer_sliders(l: &ComposerLayout, scale: f64, _composer: &Composer)
    -> [(Rect, composer::ComposerSliderDrag); 7]
{
    let pp = preview_pane_layout(l.preview, scale);
    let rl = render_panel_layout(l.render_panel, scale);
    let ml = motion_panel_layout(l.motion_panel, scale);
    [
        (l.thumb_slider,  composer::ComposerSliderDrag::ThumbSize),
        (pp.progress,     composer::ComposerSliderDrag::Playhead),
        (pp.size_slider,  composer::ComposerSliderDrag::PreviewQuality),
        (pp.spp_slider,   composer::ComposerSliderDrag::PreviewSpp),
        (pp.gamma_slider, composer::ComposerSliderDrag::PreviewGamma),
        (ml.slider,       composer::ComposerSliderDrag::PathRho),
        (rl.spp_slider,   composer::ComposerSliderDrag::RenderSpp),
    ]
}

/// Update the composer field behind `kind` from a cursor position within
/// the slider's rect.
fn apply_composer_slider(
    composer: &mut Composer,
    kind: composer::ComposerSliderDrag,
    r: Rect,
    cx: i32,
) {
    let t = ((cx - r.x) as f64 / r.w.max(1) as f64).clamp(0.0, 1.0);
    use composer::ComposerSliderDrag as K;
    match kind {
        K::None => {}
        K::ThumbSize => {
            composer.thumb_size = (composer::THUMB_SIZE_MIN as f64
                + t * (composer::THUMB_SIZE_MAX - composer::THUMB_SIZE_MIN) as f64)
                .round() as u32;
        }
        K::Playhead => {
            composer.playhead = t;
            composer.mark_preview_dirty();
            composer.mark_save_dirty();
        }
        K::PreviewQuality => {
            composer.preview_quality_pct = (10.0 + t * 90.0).round() as u32;
            composer.mark_preview_dirty();
        }
        K::PreviewSpp => {
            let lo = composer::PREVIEW_SPP_MIN.ln();
            let hi = composer::PREVIEW_SPP_MAX.ln();
            composer.spp = (lo + t * (hi - lo)).exp();
            composer.mark_preview_dirty();
        }
        K::PreviewGamma => {
            let lo = composer::GAMMA_MIN as f64;
            let hi = composer::GAMMA_MAX as f64;
            composer.gamma = (lo + t * (hi - lo)) as f32;
            composer.mark_preview_dirty();
            composer.mark_save_dirty();
        }
        K::RenderSpp => {
            composer.rp_spp = (SPP_MIN.ln()
                + t * (SPP_MAX.ln() - SPP_MIN.ln())).exp();
            composer.schedule_render_test();
            composer.mark_save_dirty();
        }
        K::PathRho => {
            composer.rho = composer::RHO_MIN
                + t * (composer::RHO_MAX - composer::RHO_MIN);
            composer.mark_preview_dirty();
            composer.mark_save_dirty();
        }
    }
}

/// Handle mouse-press-only bits in composer mode (slider drag start,
/// timeline item drag-grab, playhead drag start). Returns `true` if the press
/// was consumed.
fn composer_handle_press(
    composer: &mut Composer,
    cursor: PhysicalPosition<f64>,
    fb_w: u32, fb_h: u32, scale: f64,
) -> bool {
    let cx = cursor.x as i32;
    let cy = cursor.y as i32;
    let l = composer_layout(fb_w, fb_h, scale, composer);

    // Pane splitter (grab to resize grid vs preview/render_panel).
    if composer_splitter_rect(&l).contains(cx, cy) {
        composer.splitter_drag = Some((cursor.x, composer.split_frac));
        return true;
    }

    let pp = preview_pane_layout(l.preview, scale);

    // Palette color popover: press on one of the RGB sliders.
    if let Some(stop_idx) = composer.palette_color_edit {
        if pp.color_popover.contains(cx, cy) {
            let sliders = color_popover_sliders(pp.color_popover, scale);
            for (ch, rect) in [
                (composer::ColorChannelDrag::R, sliders[0]),
                (composer::ColorChannelDrag::G, sliders[1]),
                (composer::ColorChannelDrag::B, sliders[2]),
            ] {
                if rect.contains(cx, cy) {
                    composer.color_channel_drag = ch;
                    apply_color_channel(composer, stop_idx, ch, rect, cx);
                    return true;
                }
            }
            // Click inside popover but not on a slider — consume but no-op.
            return true;
        }
    }

    // Reset button: restore default palette stops.
    if pp.palette_reset_btn.contains(cx, cy) {
        composer.palette_stops = composer::default_palette_stops();
        composer.palette_color_edit = None;
        composer.rebuild_palette();
        return true;
    }

    // Palette stop markers: hit-test before the bar itself (markers overlap
    // the bar's bottom edge slightly).
    if let Some((i, hit)) = palette_stop_hit(composer, pp.palette_bar,
        pp.palette_stops_y, pp.palette_stops_h, cx, cy)
    {
        match hit {
            StopHit::Grab => {
                composer.palette_drag_stop = Some(i);
                return true;
            }
            StopHit::Click => {
                // Open color popover for this stop.
                composer.palette_color_edit = Some(i);
                return true;
            }
        }
    }
    // Palette bar (not on a marker): click adds a new stop at that position.
    if pp.palette_bar.contains(cx, cy) {
        let t = ((cx - pp.palette_bar.x) as f64
            / pp.palette_bar.w.max(1) as f64).clamp(0.0, 1.0) as f32;
        let rgb = composer.palette.sample_rgb(t);
        composer.palette_stops.push(composer::PaletteStop { pos: t, rgb });
        composer.rebuild_palette();
        return true;
    }

    // Render-panel resolution preset radios.
    if l.render_panel.contains(cx, cy) {
        let rl = render_panel_layout(l.render_panel, scale);
        for (rect, &(_label, pw, ph)) in
            rl.res_radios.iter().zip(composer::RESOLUTION_PRESETS.iter())
        {
            if rect.contains(cx, cy) {
                composer.rp_apply_resolution(pw, ph);
                return true;
            }
        }
    }

    // Any slider (thumb size / progress / preview size / preview spp / render spp).
    for (r, kind) in composer_sliders(&l, scale, composer) {
        if r.contains(cx, cy) {
            composer.slider_drag = kind;
            apply_composer_slider(composer, kind, r, cx);
            return true;
        }
    }

    // Timeline panel: steppers take priority over drag grab so a stepper
    // press never starts a reorder drag. The release handler actually fires
    // the +/- action; here we just consume the press.
    if l.timeline.contains(cx, cy) {
        let tl = timeline_layout(composer, l.timeline, scale);
        for s in tl.segment_steppers.iter().chain(tl.tension_steppers.iter()) {
            if s.inc.contains(cx, cy) || s.dec.contains(cx, cy) {
                return true;
            }
        }
        for (idx, slot) in tl.slots.iter().enumerate() {
            if slot.contains(cx, cy) {
                composer.timeline_drag = Some(composer::TimelineDrag {
                    from_idx: idx,
                    cursor_x: cursor.x,
                    cursor_y: cursor.y,
                    moved: false,
                });
                return true;
            }
        }
    }
    false
}

/// Click-action handler (runs on mouse release, like the live-mode buttons).
fn composer_handle_release(
    composer: &mut Composer,
    cursor: PhysicalPosition<f64>,
    fb_w: u32, fb_h: u32, scale: f64,
    view: &Arc<RwLock<View>>,
    _sampler: &SamplerHandle,
    gamma: f32,
) -> Option<ComposerAction> {
    let cx = cursor.x as i32;
    let cy = cursor.y as i32;
    let l = composer_layout(fb_w, fb_h, scale, composer);

    // Load dialog (if open) gets first shot at the click. Click on a row
    // loads that save; click anywhere else inside the frame is consumed but
    // no-op; click outside closes the dialog and falls through so e.g. a
    // click back on the load button reads as "toggle closed".
    if let Some(dlg) = composer.load_dialog.clone() {
        let ll = load_dialog_layout(l.load_btn, &dlg, fb_w, fb_h, scale);
        if ll.frame.contains(cx, cy) {
            if !ll.is_empty {
                for (i, r) in ll.rows.iter().enumerate() {
                    if r.contains(cx, cy) {
                        if let Some(entry) = dlg.entries.get(i) {
                            composer.load_from_path(&entry.path);
                        }
                        composer.close_load_dialog();
                        return None;
                    }
                }
            }
            return None;
        }
        composer.close_load_dialog();
        // A click on the load button with the dialog already open is a
        // toggle-close; don't re-open on the same release.
        if l.load_btn.contains(cx, cy) {
            return None;
        }
    }

    // Save / load buttons in the top-left strip.
    if l.save_btn.contains(cx, cy) {
        if let Some(p) = composer.save_manual() {
            eprintln!("saved timeline → {}", p.display());
        }
        return None;
    }
    if l.load_btn.contains(cx, cy) {
        composer.open_load_dialog();
        return None;
    }

    // Timeline panel: steppers, double-click to remove, single-click to
    // mark for double-click detection.
    if l.timeline.contains(cx, cy) {
        let tl = timeline_layout(composer, l.timeline, scale);

        // Segment steppers (inc 0.5s, dec 0.5s; clamp 0.1..=30.0).
        for (i, s) in tl.segment_steppers.iter().enumerate() {
            if s.inc.contains(cx, cy) {
                if let Some(it) = composer.timeline.get_mut(i) {
                    it.segment_seconds = (it.segment_seconds + 0.5).clamp(0.1, 30.0);
                    composer.mark_preview_dirty();
                    composer.schedule_render_test();
                    composer.mark_save_dirty();
                }
                return None;
            }
            if s.dec.contains(cx, cy) {
                if let Some(it) = composer.timeline.get_mut(i) {
                    it.segment_seconds = (it.segment_seconds - 0.5).clamp(0.1, 30.0);
                    composer.mark_preview_dirty();
                    composer.schedule_render_test();
                    composer.mark_save_dirty();
                }
                return None;
            }
        }

        // Tension steppers (inc/dec 0.1; clamp 0.0..=3.0).
        // 1.0 = classical centered-difference Hermite tangent (no
        // slowdown). 0.0 stops the camera at the keyframe. >1 pushes
        // it through faster, with overshoot.
        for (i, s) in tl.tension_steppers.iter().enumerate() {
            if s.inc.contains(cx, cy) {
                if let Some(it) = composer.timeline.get_mut(i) {
                    it.tension = (it.tension + 0.1).clamp(0.0, 3.0);
                    composer.mark_preview_dirty();
                    composer.schedule_render_test();
                    composer.mark_save_dirty();
                }
                return None;
            }
            if s.dec.contains(cx, cy) {
                if let Some(it) = composer.timeline.get_mut(i) {
                    it.tension = (it.tension - 0.1).clamp(0.0, 3.0);
                    composer.mark_preview_dirty();
                    composer.schedule_render_test();
                    composer.mark_save_dirty();
                }
                return None;
            }
        }

        // Slot click: single → record for double-click; double → remove.
        for (i, slot) in tl.slots.iter().enumerate() {
            if slot.contains(cx, cy) {
                let now = Instant::now();
                let is_dbl = composer.last_timeline_click
                    .map(|(t, idx)| idx == i
                        && now.duration_since(t) < Duration::from_millis(500))
                    .unwrap_or(false);
                if is_dbl {
                    composer.remove_from_timeline(i);
                    composer.last_timeline_click = None;
                } else {
                    composer.last_timeline_click = Some((now, i));
                }
                return None;
            }
        }
        return None;
    }

    // Preview pane: play/pause button click + palette editor popover close.
    if l.preview.contains(cx, cy) {
        let pp = preview_pane_layout(l.preview, scale);

        // Play button (circular — use center+radius check so the corners of
        // the bounding square aren't click-hot).
        let bx = pp.play_btn.x + pp.play_btn.w as i32 / 2;
        let by = pp.play_btn.y + pp.play_btn.h as i32 / 2;
        let br = pp.play_btn.w as i32 / 2;
        let dx = cx - bx;
        let dy = cy - by;
        if dx * dx + dy * dy <= br * br {
            composer.playing = !composer.playing;
            return None;
        }

        // Color-popover: clicking outside closes it. "Outside" excludes the
        // palette interaction strip (bar + stop markers + reset button) so a
        // marker click that just opened the popover, or a bar click adding a
        // new stop, doesn't immediately re-close it.
        let palette_zone_top = pp.palette_bar.y;
        let palette_zone_bottom = pp.palette_stops_y + pp.palette_stops_h as i32 + 2;
        let on_palette_strip = cy >= palette_zone_top && cy < palette_zone_bottom;
        if composer.palette_color_edit.is_some()
            && !pp.color_popover.contains(cx, cy)
            && !on_palette_strip
        {
            composer.palette_color_edit = None;
        }

        return None;
    }

    if l.grid.contains(cx, cy) {
        let ts = ((composer.thumb_size as f64) * scale).round() as i32;
        let gap = (8.0 * scale).round() as i32;
        let pad = (8.0 * scale).round() as i32;
        let col_w = ts + gap;
        let row_h = ts + gap + (14.0 * scale).round() as i32;
        let cols = ((l.grid.w as i32 - pad * 2 + gap) / col_w).max(1);
        let local_x = cx - l.grid.x - pad;
        let local_y = cy - l.grid.y - pad + composer.grid_scroll.round() as i32;
        if local_x >= 0 && local_y >= 0 {
            let col = local_x / col_w;
            let row = local_y / row_h;
            if col < cols {
                let flat = (row * cols + col) as usize;
                let inner_y = local_y % row_h;
                if inner_y < ts {
                    // Slot 0 is the always-present "+" generate-rotations tile.
                    if flat == 0 {
                        let base_view = view.read().clone();
                        let palette_for_gen = composer.palette.clone();
                        thread::spawn(move || {
                            match composer::generate_auto_rotation(
                                &base_view, &palette_for_gen, gamma, 500_000, 8,
                            ) {
                                Ok(paths) => eprintln!("generated {} keyframes", paths.len()),
                                Err(e) => eprintln!("auto-rotate failed: {}", e),
                            }
                        });
                        return None;
                    }
                    let idx = flat - 1;
                    if idx < composer.png_paths.len() {
                        let path = composer.png_paths[idx].clone();
                        // Every click appends a new keyframe; duplicates
                        // allowed. Removal happens from the timeline
                        // itself (double-click or drag-off).
                        composer.add_to_timeline(path);
                    }
                }
            }
        }
        return None;
    }

    if l.render_panel.contains(cx, cy) {
        let rl = render_panel_layout(l.render_panel, scale);
        if rl.clear_btn.contains(cx, cy) {
            composer.clear_timeline();
            composer.schedule_render_test();
            return None;
        }
        if rl.render_btn.contains(cx, cy) {
            if composer.timeline.len() >= 2 {
                return Some(ComposerAction::RenderFromPanel);
            }
            return None;
        }
        // Text-field focus.
        if rl.width_field.contains(cx, cy) {
            composer.rp_focus = composer::RenderFieldFocus::Width;
            return None;
        }
        if rl.height_field.contains(cx, cy) {
            composer.rp_focus = composer::RenderFieldFocus::Height;
            return None;
        }
        // (Queue-row × buttons are hit-tested by the caller since the row
        // geometry needs the live render_progress/render_queue.)
        // Click elsewhere in panel clears text focus.
        composer.rp_focus = composer::RenderFieldFocus::None;
    }
    None
}


fn composer_handle_wheel(
    composer: &mut Composer,
    fb_w: u32, fb_h: u32, scale: f64,
    cursor: PhysicalPosition<f64>,
    dx: f64, dy: f64,
) {
    let l = composer_layout(fb_w, fb_h, scale, composer);
    let cx = cursor.x as i32;
    let cy = cursor.y as i32;
    if l.timeline.contains(cx, cy) {
        // Pan. Prefer horizontal wheel axis; fall back to vertical so a
        // standard mouse wheel also pans.
        let pan = if dx.abs() > 1e-3 { -dx } else { -dy };
        let tl = timeline_layout(composer, l.timeline, scale);
        let max = (tl.content_w - l.timeline.w as i32).max(0) as f64;
        composer.timeline_scroll_x =
            (composer.timeline_scroll_x + pan).clamp(0.0, max);
        return;
    }
    if l.grid.contains(cx, cy) {
        composer.grid_scroll = (composer.grid_scroll - dy).max(0.0);
    }
}

fn composer_handle_cursor_moved(
    composer: &mut Composer,
    position: PhysicalPosition<f64>,
    fb_w: u32, fb_h: u32, scale: f64,
) {
    let l = composer_layout(fb_w, fb_h, scale, composer);

    // Splitter drag: recompute split_frac from cursor delta, preview pane
    // changes size → mark preview dirty so the next tick re-renders.
    if let Some((start_x, start_frac)) = composer.splitter_drag {
        let denom = (fb_w as f64).max(1.0);
        let new_frac = (start_frac + (position.x - start_x) / denom)
            .clamp(composer::SPLIT_FRAC_MIN, composer::SPLIT_FRAC_MAX);
        composer.split_frac = new_frac;
        composer.mark_preview_dirty();
    }

    // Timeline middle-click pan.
    if let Some((start_x, start_scroll)) = composer.timeline_pan_drag {
        let tl = timeline_layout(composer, l.timeline, scale);
        let max = (tl.content_w - l.timeline.w as i32).max(0) as f64;
        composer.timeline_scroll_x =
            (start_scroll - (position.x - start_x)).clamp(0.0, max);
    }

    // Slider drag (whichever kind is active).
    if composer.slider_drag != composer::ComposerSliderDrag::None {
        let kind = composer.slider_drag;
        for (r, k) in composer_sliders(&l, scale, composer) {
            if k == kind {
                apply_composer_slider(composer, kind, r, position.x as i32);
                break;
            }
        }
    }

    // Palette stop drag (reposition).
    if let Some(idx) = composer.palette_drag_stop {
        let pp = preview_pane_layout(l.preview, scale);
        let t = ((position.x as i32 - pp.palette_bar.x) as f64
            / pp.palette_bar.w.max(1) as f64)
            .clamp(0.0, 1.0) as f32;
        if let Some(s) = composer.palette_stops.get_mut(idx) {
            s.pos = t;
        }
        composer.rebuild_palette();
    }

    // Color channel drag (inside popover).
    if composer.color_channel_drag != composer::ColorChannelDrag::None {
        if let Some(stop_idx) = composer.palette_color_edit {
            let pp = preview_pane_layout(l.preview, scale);
            let sliders = color_popover_sliders(pp.color_popover, scale);
            let rect = match composer.color_channel_drag {
                composer::ColorChannelDrag::R => sliders[0],
                composer::ColorChannelDrag::G => sliders[1],
                composer::ColorChannelDrag::B => sliders[2],
                composer::ColorChannelDrag::None => Rect { x: 0, y: 0, w: 0, h: 0 },
            };
            apply_color_channel(composer, stop_idx,
                composer.color_channel_drag, rect, position.x as i32);
        }
    }

    if let Some(d) = composer.timeline_drag.as_mut() {
        let dx = position.x - d.cursor_x;
        let dy = position.y - d.cursor_y;
        if dx * dx + dy * dy > 9.0 { d.moved = true; }
        d.cursor_x = position.x;
        d.cursor_y = position.y;
    }
}
