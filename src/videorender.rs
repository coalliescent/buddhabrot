//! Video render: interpolate a camera path through keyframe PNGs, write each
//! frame as a numbered PNG to a session directory, and have ffmpeg assemble
//! them at the end.
//!
//! Writing per-frame PNGs instead of piping raw RGBA to ffmpeg gives three
//! properties that a single-pipeline render can't match:
//!
//!   * The render can **resume from where it left off** after a crash. On
//!     restart, frames already present on disk are skipped.
//!   * ffmpeg is only invoked once all frames exist, so a partially-written
//!     MP4 is never left behind.
//!   * Frame-level inspection / debugging is trivial — the PNGs are in the
//!     session directory.
//!
//! Interpolation is split by parameter kind:
//!
//!   * **Pan + zoom (`center`, `half_width`)** — Van Wijk–Nuij
//!     geodesic per segment (see `zoompan`). Constant apparent screen
//!     velocity and no overshoot, even when a zoomed-in keyframe sits
//!     between far-away wide shots. C⁰ at seams (each segment is its
//!     own geodesic); tension reparameterizes `u` so tension = 0 still
//!     parks the camera and > 1 still accelerates through.
//!   * **Rotation** — bi-quaternion log-space cubic Hermite (see
//!     `rotation_interp`), C¹ at seams via wall-time-scaled tangents.
//!   * **Gamma** — time-aware cubic Hermite over the 4-keyframe
//!     neighborhood, same scheme as rotation.
//!   * **`max_iter`** — linear between `v1` and `v2`.
//!   * **Palette LUTs** — linearly blended between adjacent keyframes.
//!
//! Progress: `session.json` is rewritten atomically after each frame. Progress
//! is also echoed to stdout as `frame X/Y` lines for anyone tailing it; those
//! writes use `writeln!` + ignored result so a closed pipe doesn't panic.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::metadata;
use crate::palette::{Palette, LUT_SIZE};
use crate::render::{self, FrameSpec};
use crate::session::SessionFile;
use crate::view::View;

pub struct VideoRenderArgs {
    pub keyframes: Vec<PathBuf>,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub total_frames: u32,
    pub samples_per_frame: u64,
    pub n_workers: usize,
    /// Optional override for the session directory. If None, a timestamped one
    /// is generated under `~/buddhabrot/.render-<ts>/`.
    pub session_dir: Option<PathBuf>,
    /// Per-keyframe tension (Hermite tangent multiplier). Length ==
    /// keyframes.len(). Empty = uniform 1.0 (classical centered-
    /// difference). 0 stops the camera at the keyframe; >1 accelerates
    /// through.
    pub tensions: Vec<f64>,
    /// Per-segment travel time (seconds). Length == keyframes.len(). Last
    /// entry is ignored. Empty = uniform 2.0 s per segment.
    pub segments: Vec<f64>,
    /// Composer gamma override: applied to every frame, replacing the
    /// interpolated per-keyframe gamma. Matches the composer's preview.
    pub gamma_override: Option<f32>,
    /// Composer palette override as stop list (pos, rgb). Applied to every
    /// frame, replacing per-keyframe palette interpolation. Matches preview.
    pub palette_stops_override: Option<Vec<(f32, [u8; 3])>>,
    /// V-N path shape parameter for pan/zoom interpolation. √2 is the
    /// paper default; 0 collapses to log-linear (no pullback).
    pub rho: f64,
}

/// Render a brand-new video. Creates a session directory under
/// `~/buddhabrot/.render-<unix>/`.
pub fn render_video(args: &VideoRenderArgs) -> io::Result<()> {
    validate(&args.keyframes, args.total_frames)?;

    let ts = SessionFile::now();
    let session_dir = args.session_dir.clone().unwrap_or_else(|| SessionFile::new_dir(ts));
    fs::create_dir_all(SessionFile::frames_dir(&session_dir))?;

    let mut sess = SessionFile {
        output: args.output.clone(),
        fps: args.fps,
        width: args.width & !1,
        height: args.height & !1,
        samples_per_frame: args.samples_per_frame,
        total_frames: args.total_frames,
        keyframes: args.keyframes.clone(),
        tensions: args.tensions.clone(),
        segments: args.segments.clone(),
        pid: Some(std::process::id()),
        started_at: ts,
        last_progress_at: ts,
        cur_frame: 0,
        gamma_override: args.gamma_override,
        palette_stops_override: args.palette_stops_override.clone(),
        rho: args.rho,
    };
    sess.write(&session_dir)?;

    run_session(&session_dir, &mut sess, args.n_workers)
}

/// Resume an interrupted render. `session_dir` must contain a readable
/// `session.json`. Any frames already present on disk are skipped.
pub fn resume_video(session_dir: &Path, n_workers: usize) -> io::Result<()> {
    let mut sess = SessionFile::read(session_dir)?;
    fs::create_dir_all(SessionFile::frames_dir(session_dir))?;
    sess.pid = Some(std::process::id());
    sess.last_progress_at = SessionFile::now();
    sess.write(session_dir)?;
    eprintln!(
        "resuming session {}: {} / {} frames already rendered",
        session_dir.display(),
        count_existing_frames(session_dir, sess.total_frames),
        sess.total_frames,
    );
    run_session(session_dir, &mut sess, n_workers)
}

fn run_session(
    session_dir: &Path,
    sess: &mut SessionFile,
    n_workers: usize,
) -> io::Result<()> {
    validate(&sess.keyframes, sess.total_frames)?;

    // Load keyframe specs.
    let kfs: Vec<FrameSpec> = sess
        .keyframes
        .iter()
        .map(|p| metadata::read_spec(p, sess.samples_per_frame, n_workers))
        .collect::<io::Result<Vec<_>>>()?;
    let k = kfs.len();
    // A timeline whose first and last keyframe reference the same PNG is
    // the user's loop convention; pick cyclic tangent neighbors so the
    // wrap-around is C¹-continuous.
    let cyclic = k >= 3 && sess.keyframes.first() == sess.keyframes.last();
    // Chain-align bi-quaternion signs across the whole timeline so each
    // keyframe has a single canonical representative that adjacent
    // segments agree on.
    let rotations: Vec<crate::view::Mat4> = kfs.iter().map(|f| f.view.rotation).collect();
    let (l_path, r_path) = crate::rotation_interp::build_aligned_biquats(&rotations);
    // Cumulative wall times per keyframe; total = times[k-1].
    let times = cumulative_times(&sess.segments, k);
    let total = *times.last().unwrap_or(&0.0);

    // Build the palette override (once) if present. Matches the composer
    // preview path: a single fixed palette used for every frame, rather than
    // interpolation between per-keyframe palettes.
    let palette_override: Option<Palette> = sess.palette_stops_override
        .as_ref()
        .map(|stops| {
            let s: Vec<(f32, [u8; 3])> = stops.iter().copied().collect();
            Palette::from_stops_dyn("composer", &s)
        });

    for i in 0..sess.total_frames {
        // Cancel signal: UI drops a `.cancel` file into the session dir.
        // Exit cleanly so the UI watcher sees the dir disappear.
        if session_dir.join(".cancel").exists() {
            eprintln!("render cancelled by UI");
            let _ = fs::remove_dir_all(session_dir);
            return Err(io::Error::new(
                io::ErrorKind::Interrupted, "cancelled"));
        }
        let frame_path = SessionFile::frame_path(session_dir, i);
        if frame_path.exists() {
            sess.cur_frame = i + 1;
            sess.last_progress_at = SessionFile::now();
            sess.write(session_dir).ok();
            print_progress(i + 1, sess.total_frames);
            continue;
        }

        let t_global = i as f64 / (sess.total_frames - 1) as f64;
        let (from, local_u) = segment_for_playhead(&times, total, t_global);
        let indices = pick_neighbors(k, from, cyclic);
        let nt = neighbor_times(&times, total, indices, from, cyclic);
        let ntens = neighbor_tensions(&sess.tensions, indices);
        let rotation = crate::rotation_interp::eval_hermite(
            &l_path, &r_path, indices, nt, ntens, local_u,
        );
        let p0 = &kfs[indices[0]];
        let p1 = &kfs[indices[1]];
        let p2 = &kfs[indices[2]];
        let p3 = &kfs[indices[3]];
        let view = interp_view(
            &p1.view, &p2.view, &p0.view, &p3.view,
            rotation, nt, ntens, sess.rho, local_u,
            sess.width, sess.height,
        );
        let gamma = hermite_time_aware(
            [p0.gamma as f64, p1.gamma as f64, p2.gamma as f64, p3.gamma as f64],
            nt, ntens, local_u,
        ) as f32;
        let palette = interp_palette(&p1.palette, &p2.palette, local_u);
        let mut spec = FrameSpec {
            view, palette, gamma,
            samples_target: sess.samples_per_frame,
            n_workers,
        };
        // Composer overrides (gamma/palette) win over whatever interpolation
        // produced, so the final video matches the preview.
        if let Some(g) = sess.gamma_override {
            spec.gamma = g;
        }
        if let Some(p) = &palette_override {
            spec.palette = p.clone();
        }
        let rgba = render::render_frame(&spec, None);
        write_frame_png(&frame_path, &rgba, sess.width, sess.height)?;

        sess.cur_frame = i + 1;
        sess.last_progress_at = SessionFile::now();
        sess.write(session_dir).ok();

        print_progress(i + 1, sess.total_frames);
    }

    // All frames are on disk; assemble with ffmpeg.
    eprintln!("assembling video → {}", sess.output.display());
    assemble_video(session_dir, &sess.output, sess.fps)?;

    // Clean up session dir on success.
    let _ = fs::remove_dir_all(session_dir);
    Ok(())
}

fn print_progress(cur: u32, total: u32) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "frame {}/{}", cur, total);
    let _ = out.flush();
}

fn validate(keyframes: &[PathBuf], total_frames: u32) -> io::Result<()> {
    if keyframes.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "need at least 2 keyframes",
        ));
    }
    if total_frames < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "need at least 2 output frames",
        ));
    }
    Ok(())
}

fn count_existing_frames(session_dir: &Path, total: u32) -> u32 {
    (0..total)
        .filter(|i| SessionFile::frame_path(session_dir, *i).exists())
        .count() as u32
}

fn write_frame_png(path: &Path, rgba: &[u8], w: u32, h: u32) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // Fast compression — these are intermediates.
    encoder.set_compression(png::Compression::Fast);
    let mut w = encoder
        .write_header()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    w.write_image_data(rgba)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(())
}

fn assemble_video(session_dir: &Path, output: &Path, fps: u32) -> io::Result<()> {
    let pattern = SessionFile::frames_dir(session_dir).join("%05d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel", "error",
            "-framerate", &fps.to_string(),
            "-start_number", "0",
            "-i",
        ])
        .arg(&pattern)
        .args([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "20",
        ])
        .arg(output)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("ffmpeg assembly exited with status {status}"),
        ));
    }
    Ok(())
}

/// Cumulative wall times per keyframe: `times[i] = Σ segments[0..i]`.
/// Length is `n`. Defaults missing segment entries to 2.0 s, with a
/// 0.05 s floor so a degenerate zero-length segment doesn't divide by
/// zero downstream.
pub fn cumulative_times(segments: &[f64], n: usize) -> Vec<f64> {
    let mut t = Vec::with_capacity(n);
    let mut acc = 0.0;
    t.push(acc);
    for i in 0..n.saturating_sub(1) {
        acc += segments.get(i).copied().unwrap_or(2.0).max(0.05);
        t.push(acc);
    }
    t
}

/// Map normalized playhead `t ∈ [0, 1]` to a `(segment, local_u)` pair
/// using the precomputed cumulative times. `local_u` is `0` at the
/// segment's start and `1` at its end.
pub fn segment_for_playhead(times: &[f64], total: f64, t: f64) -> (usize, f64) {
    let n = times.len();
    if n < 2 || total <= 1e-9 { return (0, 0.0); }
    let want = t.clamp(0.0, 1.0) * total;
    for i in 0..n - 1 {
        if want <= times[i + 1] {
            let h = (times[i + 1] - times[i]).max(1e-9);
            let u = ((want - times[i]) / h).clamp(0.0, 1.0);
            return (i, u);
        }
    }
    (n - 2, 1.0)
}

/// Pick the 4-keyframe Hermite neighborhood for segment `from`. With
/// `cyclic = true` the missing neighbor at each end of the timeline
/// wraps around, so a timeline whose first and last keyframe match
/// forms a smooth loop at the seam.
pub fn pick_neighbors(n: usize, from: usize, cyclic: bool) -> [usize; 4] {
    debug_assert!(n >= 2 && from + 1 < n);
    let p0 = if from == 0 {
        if cyclic && n >= 3 { n - 2 } else { 0 }
    } else {
        from - 1
    };
    let p3 = if from + 2 >= n {
        if cyclic && n >= 3 { 1 } else { n - 1 }
    } else {
        from + 2
    };
    [p0, from, from + 1, p3]
}

/// Build the `[t_prev, t_curr, t_next, t_after]` wall-time array for
/// a Hermite evaluation, applying cyclic wrap when the neighborhood
/// reaches across the seam (subtracting / adding the total duration so
/// the centered-difference tangent stays well-defined).
pub fn neighbor_times(
    times: &[f64], total: f64, indices: [usize; 4], from: usize, cyclic: bool,
) -> [f64; 4] {
    let n = times.len();
    let mut t = [0.0; 4];
    for k in 0..4 { t[k] = times[indices[k]]; }
    if cyclic && n >= 3 {
        if from == 0 { t[0] -= total; }
        if from + 2 >= n { t[3] += total; }
    }
    t
}

/// Look up tensions for the 4-keyframe neighborhood, defaulting to
/// 1.0 (classical centered difference) when the input array is short
/// or empty.
pub fn neighbor_tensions(tensions: &[f64], indices: [usize; 4]) -> [f64; 4] {
    let mut t = [1.0; 4];
    for k in 0..4 { t[k] = tensions.get(indices[k]).copied().unwrap_or(1.0); }
    t
}

/// Time-aware cubic Hermite interpolation between `values[1]` (at
/// `times[1]`) and `values[2]` (at `times[2]`). The four `values`,
/// `times`, and `tensions` come from the 4-keyframe neighborhood
/// `[i_prev, i_curr, i_next, i_after]`.
///
/// The tangent at `K_curr` is the wall-time-scaled centered difference
/// `tension_curr * (K_next − K_prev) / (t_next − t_prev)`, and likewise
/// at `K_next`. Because that tangent depends only on the keyframe and
/// its two neighbors — not on the segment we're evaluating — every
/// segment passing through `K_curr` agrees on the velocity there. C¹
/// in wall time at every seam, regardless of how segment_seconds vary.
///
/// `tension = 1` is the classical centered-difference tangent and
/// imposes no slowdown. `tension = 0` zeros the tangent (camera comes
/// to rest at the keyframe). `tension > 1` exaggerates the tangent —
/// the camera barrels through with overshoot. `u ∈ [0, 1]` is the
/// local parameter inside the segment.
pub fn hermite_time_aware(
    values: [f64; 4], times: [f64; 4], tensions: [f64; 4], u: f64,
) -> f64 {
    let m1 = tensions[1] * (values[2] - values[0])
        / (times[2] - times[0]).max(1e-9);
    let m2 = tensions[2] * (values[3] - values[1])
        / (times[3] - times[1]).max(1e-9);
    let h = (times[2] - times[1]).max(1e-9);
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 =  2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 =        u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 =        u3 -       u2;
    h00 * values[1] + h10 * h * m1 + h01 * values[2] + h11 * h * m2
}

/// Build a View by interpolating a rotation (pre-computed geodesic),
/// pan + half_width (Van Wijk–Nuij in `zoompan`), and per-keyframe
/// `max_iter` (linear in `u`). The `v0` / `v3` neighbors and the
/// `times` array are unused for pan/zoom — V-N is a two-keyframe
/// geodesic — but `tensions[1]` / `tensions[2]` still shape the
/// traversal via an `u → u_eff` Hermite reparameterization, so
/// tension = 0 still parks the camera at a keyframe and tension > 1
/// still accelerates through it.
pub fn interp_view(
    v1: &View, v2: &View, _v0: &View, _v3: &View,
    rotation: crate::view::Mat4,
    _times: [f64; 4], tensions: [f64; 4], rho: f64, u: f64,
    ow: u32, oh: u32,
) -> View {
    let u_eff = tension_reparam(u, tensions[1], tensions[2]);
    let (center, half_width) = crate::zoompan::interp(
        v1.center, v1.half_width,
        v2.center, v2.half_width,
        rho,
        u_eff,
    );
    let max_iter = ((1.0 - u) * v1.max_iter as f64
                  + u * v2.max_iter as f64).round() as u32;

    let mut out = View::new(ow, oh);
    out.rotation = rotation;
    out.center = center;
    out.half_width = half_width;
    out.max_iter = max_iter.max(10);
    out
}

/// Cubic Hermite from 0→1 with endpoint slopes `tension_curr` /
/// `tension_next`. Identity when both tensions are 1 (V-N's natural
/// constant-screen-velocity parameterization); smoothstep when both
/// are 0 (ease-in-out, zero velocity at both ends).
fn tension_reparam(u: f64, tension_curr: f64, tension_next: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    let h10 =        u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 =        u3 -       u2;
    h10 * tension_curr + h01 + h11 * tension_next
}

pub fn interp_palette(a: &Palette, b: &Palette, t: f64) -> Palette {
    if std::ptr::eq(a, b) || a.name == b.name {
        return a.clone();
    }
    let t = t.clamp(0.0, 1.0) as f32;
    let mut lut = vec![[0u8; 4]; LUT_SIZE];
    for i in 0..LUT_SIZE {
        let ac = a.lut[i];
        let bc = b.lut[i];
        for c in 0..4 {
            lut[i][c] = ((1.0 - t) * ac[c] as f32 + t * bc[c] as f32)
                .clamp(0.0, 255.0) as u8;
        }
    }
    Palette { name: "interp", lut }
}
