//! Command-line subcommand handlers.
//!
//! `render-frame` and `render-video` dispatch here from `main()`. Arg parsing
//! is hand-rolled (no clap) to keep dependencies minimal; the arg set is small.

use std::error::Error;
use std::path::PathBuf;

use crate::metadata;
use crate::render::{self, FrameSpec};
use crate::videorender::{self, VideoRenderArgs};

pub fn print_help() {
    eprintln!(
        "\
buddhabrot — interactive 4D Mandelbrot orbit visualizer

USAGE:
  buddhabrot                           open interactive window (default)
  buddhabrot render-frame [OPTS]       render one frame to PNG
  buddhabrot render-video [OPTS] KFS   render a video through keyframe PNGs

render-frame OPTIONS:
  --in  PATH       keyframe PNG to take view from (required)
  --out PATH       output PNG path (required)
  --samples N      target sample count (default 2000000)
  --width  W       override output width  (default: from input)
  --height H       override output height (default: from input)
  --threads K      worker threads (default: available - 1; use 1 for inline)

render-video OPTIONS:
  --out PATH       output video path (e.g. buddhabrot.mp4) (required)
  --fps N          video frame rate (default 30)
  --duration SEC   video duration in seconds (default 2 * (#keyframes - 1))
  --frames N       override total frame count (takes precedence over duration)
  --width W        output width  (default 1024, forced even)
  --height H       output height (default 768,  forced even)
  --samples N      target samples per frame (default 1000000)
  --threads K      worker threads (default: available - 1)
  --resume PATH    resume a previous render from a session dir
                   (~/buddhabrot/.render-<ts>/); other args are ignored
  --segments CSV   per-segment travel seconds (e.g. 1,2,0.5). One entry per
                   keyframe; last is ignored. Default: 2.0 s each.
  --tensions CSV   per-keyframe Hermite tangent multiplier (e.g.
                   1,0.5,1). Default: 1.0 (classical centered-difference
                   tangent — no slowdown). 0 stops the camera at that
                   keyframe; >1 accelerates through.
  --spacings CSV   deprecated alias for --segments (silent)
  --gamma F        override gamma for every frame (replaces per-keyframe
                   interpolation — matches the composer preview)
  --palette-stops CSV
                   override palette for every frame. CSV is groups of 4:
                   pos,r,g,b repeated per stop. Replaces per-keyframe
                   palette interpolation (matches the composer preview).
  --rho F          V-N path shape parameter (default √2). Controls how
                   far the camera pulls back mid-segment to pan at a
                   wider view. 0 = pure log-linear (no pullback).
  KFS              one or more keyframe PNG paths (positional, >=2 required)
"
    );
}

pub fn render_frame(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut samples: u64 = 2_000_000;
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut threads: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in"      => { in_path = Some(args[i+1].clone().into()); i += 2; }
            "--out"     => { out_path = Some(args[i+1].clone().into()); i += 2; }
            "--samples" => { samples = args[i+1].parse()?; i += 2; }
            "--width"   => { width = Some(args[i+1].parse()?); i += 2; }
            "--height"  => { height = Some(args[i+1].parse()?); i += 2; }
            "--threads" => { threads = Some(args[i+1].parse()?); i += 2; }
            other => return Err(format!("render-frame: unknown arg `{other}`").into()),
        }
    }
    let in_path = in_path.ok_or("render-frame: --in is required")?;
    let out_path = out_path.ok_or("render-frame: --out is required")?;
    let n_workers = threads.unwrap_or_else(default_threads);

    let mut spec = metadata::read_spec(&in_path, samples, n_workers)?;
    if let Some(w) = width { spec.view.width = w; }
    if let Some(h) = height { spec.view.height = h; }

    eprintln!(
        "rendering {}x{}, samples={}, threads={}",
        spec.view.width, spec.view.height, samples, n_workers
    );
    let rgba = render::render_frame(&spec, None);
    write_png(&out_path, &rgba, spec.view.width, spec.view.height, &spec)?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}

pub fn render_video(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut out_path: Option<PathBuf> = None;
    let mut fps: u32 = 30;
    let mut duration: Option<f64> = None;
    let mut frames: Option<u32> = None;
    let mut width: u32 = 1024;
    let mut height: u32 = 768;
    let mut samples: u64 = 1_000_000;
    let mut threads: Option<usize> = None;
    let mut keyframes: Vec<PathBuf> = Vec::new();
    let mut resume: Option<PathBuf> = None;
    let mut session_dir: Option<PathBuf> = None;
    let mut segments: Vec<f64> = Vec::new();
    let mut tensions: Vec<f64> = Vec::new();
    let mut gamma_override: Option<f32> = None;
    let mut palette_stops_override: Option<Vec<(f32, [u8; 3])>> = None;
    let mut rho: f64 = crate::zoompan::RHO_DEFAULT;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out"         => { out_path = Some(args[i+1].clone().into()); i += 2; }
            "--fps"         => { fps = args[i+1].parse()?; i += 2; }
            "--duration"    => { duration = Some(args[i+1].parse()?); i += 2; }
            "--frames"      => { frames = Some(args[i+1].parse()?); i += 2; }
            "--width"       => { width = args[i+1].parse()?; i += 2; }
            "--height"      => { height = args[i+1].parse()?; i += 2; }
            "--samples"     => { samples = args[i+1].parse()?; i += 2; }
            "--threads"     => { threads = Some(args[i+1].parse()?); i += 2; }
            "--resume"      => { resume = Some(args[i+1].clone().into()); i += 2; }
            "--session-dir" => { session_dir = Some(args[i+1].clone().into()); i += 2; }
            "--segments" | "--spacings" => {
                segments = args[i+1].split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
                i += 2;
            }
            "--tensions"    => {
                tensions = args[i+1].split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
                i += 2;
            }
            // v1 CLI flag — silently dropped; the dwell concept is gone.
            "--dwells"      => { i += 2; }
            "--gamma"         => { gamma_override = Some(args[i+1].parse()?); i += 2; }
            "--rho"           => { rho = args[i+1].parse()?; i += 2; }
            "--palette-stops" => {
                let nums: Vec<f64> = args[i+1].split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
                if nums.len() % 4 != 0 {
                    return Err("render-video: --palette-stops must be groups of 4: pos,r,g,b".into());
                }
                let stops: Vec<(f32, [u8; 3])> = nums.chunks_exact(4)
                    .map(|c| (c[0] as f32,
                              [c[1].clamp(0.0, 255.0) as u8,
                               c[2].clamp(0.0, 255.0) as u8,
                               c[3].clamp(0.0, 255.0) as u8]))
                    .collect();
                palette_stops_override = Some(stops);
                i += 2;
            }
            other if !other.starts_with("--") => {
                keyframes.push(PathBuf::from(other));
                i += 1;
            }
            other => return Err(format!("render-video: unknown arg `{other}`").into()),
        }
    }
    let n_workers = threads.unwrap_or_else(default_threads);

    if let Some(session_dir) = resume {
        eprintln!("resuming render from {}", session_dir.display());
        videorender::resume_video(&session_dir, n_workers)?;
        return Ok(());
    }

    let out_path = out_path.ok_or("render-video: --out is required")?;
    if keyframes.len() < 2 {
        return Err("render-video: need at least 2 keyframe PNGs".into());
    }

    let total_frames = frames.unwrap_or_else(|| {
        // Prefer computed duration from segments when given; otherwise
        // fall back to the classic 2 s × (n-1) model.
        let computed: f64 = segments.iter()
            .take(keyframes.len().saturating_sub(1)).sum::<f64>();
        let dur = duration.unwrap_or(if computed > 0.0 {
            computed
        } else {
            2.0 * (keyframes.len() - 1) as f64
        });
        (dur * fps as f64).round() as u32
    });

    let vargs = VideoRenderArgs {
        keyframes,
        output: out_path.clone(),
        width,
        height,
        fps,
        total_frames,
        samples_per_frame: samples,
        n_workers,
        session_dir,
        tensions,
        segments,
        gamma_override,
        palette_stops_override,
        rho,
    };
    eprintln!(
        "rendering {} frames at {} fps, {}x{}, samples/frame={}, threads={}",
        vargs.total_frames, vargs.fps, vargs.width, vargs.height,
        vargs.samples_per_frame, vargs.n_workers,
    );
    videorender::render_video(&vargs)?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(1)
        .max(1)
}

fn write_png(
    path: &std::path::Path,
    rgba: &[u8],
    w: u32,
    h: u32,
    spec: &FrameSpec,
) -> Result<(), Box<dyn Error>> {
    use crate::savepng;
    // Use the shared save_png path so CLI output has the same metadata schema
    // as UI output — future CLI runs can feed their own output back as input.
    let meta = savepng::SaveMeta {
        view: &spec.view,
        palette_name: spec.palette.name,
        gamma: spec.gamma,
        samples: spec.samples_target,
    };
    savepng::save_png_to(path, rgba, w, h, meta)?;
    Ok(())
}
