//! Persistent render-session state: a directory per in-progress video render
//! holding a session.json (simple key=value lines) plus numbered PNG frames.
//!
//! Sessions let the renderer survive UI death, and let a newly-started UI
//! pick up where it left off.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Everything needed to render (or resume) a video.
#[derive(Clone, Debug)]
pub struct SessionFile {
    pub output: PathBuf,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub samples_per_frame: u64,
    pub total_frames: u32,
    pub keyframes: Vec<PathBuf>,
    /// Per-keyframe Hermite tangent multiplier (length == keyframes.len(),
    /// may be empty for uniform 1.0 = no slowdown). Replaces the old
    /// `dwells` field.
    pub tensions: Vec<f64>,
    /// Per-keyframe segment-travel seconds (length == keyframes.len(), last
    /// entry ignored; may be empty for uniform 2.0 s).
    pub segments: Vec<f64>,
    pub pid: Option<u32>,
    pub started_at: u64,
    pub last_progress_at: u64,
    pub cur_frame: u32,
    /// Composer gamma override applied to every frame. When present, takes
    /// precedence over per-keyframe interpolation.
    pub gamma_override: Option<f32>,
    /// Composer palette override as stop list (pos, rgb). When present, takes
    /// precedence over per-keyframe palette interpolation.
    pub palette_stops_override: Option<Vec<(f32, [u8; 3])>>,
    /// V-N path shape parameter for intra-segment pan/zoom. `√2` is the
    /// paper default; `0` flattens to log-linear (no pullback).
    pub rho: f64,
}

impl SessionFile {
    pub fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Standard location `~/buddhabrot/`.
    pub fn root() -> PathBuf {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join("buddhabrot")
    }

    /// Create a new, unique session directory.
    pub fn new_dir(ts: u64) -> PathBuf {
        Self::root().join(format!(".render-{ts}"))
    }

    pub fn frames_dir(session_dir: &Path) -> PathBuf {
        session_dir.join("frames")
    }

    pub fn frame_path(session_dir: &Path, n: u32) -> PathBuf {
        Self::frames_dir(session_dir).join(format!("{n:05}.png"))
    }

    pub fn json_path(session_dir: &Path) -> PathBuf {
        session_dir.join("session.json")
    }

    /// Atomically rewrite the session file (tmp + rename).
    pub fn write(&self, session_dir: &Path) -> io::Result<()> {
        fs::create_dir_all(session_dir)?;
        let path = Self::json_path(session_dir);
        let tmp = session_dir.join("session.json.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            writeln!(f, "output={}", self.output.display())?;
            writeln!(f, "fps={}", self.fps)?;
            writeln!(f, "width={}", self.width)?;
            writeln!(f, "height={}", self.height)?;
            writeln!(f, "samples_per_frame={}", self.samples_per_frame)?;
            writeln!(f, "total_frames={}", self.total_frames)?;
            writeln!(f, "started_at={}", self.started_at)?;
            writeln!(f, "last_progress_at={}", self.last_progress_at)?;
            writeln!(f, "cur_frame={}", self.cur_frame)?;
            if let Some(pid) = self.pid {
                writeln!(f, "pid={pid}")?;
            }
            writeln!(f, "keyframe_count={}", self.keyframes.len())?;
            for (i, kf) in self.keyframes.iter().enumerate() {
                writeln!(f, "keyframe_{i}={}", kf.display())?;
            }
            writeln!(f, "tension_count={}", self.tensions.len())?;
            for (i, s) in self.tensions.iter().enumerate() {
                writeln!(f, "tension_{i}={s}")?;
            }
            writeln!(f, "segment_count={}", self.segments.len())?;
            for (i, s) in self.segments.iter().enumerate() {
                writeln!(f, "segment_{i}={s}")?;
            }
            if let Some(g) = self.gamma_override {
                writeln!(f, "gamma_override={g}")?;
            }
            writeln!(f, "rho={}", self.rho)?;
            if let Some(stops) = &self.palette_stops_override {
                writeln!(f, "palette_override_count={}", stops.len())?;
                for (i, (pos, rgb)) in stops.iter().enumerate() {
                    writeln!(f, "palette_override_{i}={},{},{},{}",
                        pos, rgb[0], rgb[1], rgb[2])?;
                }
            }
            f.sync_all()?;
        }
        fs::rename(tmp, path)
    }

    pub fn read(session_dir: &Path) -> io::Result<Self> {
        let path = Self::json_path(session_dir);
        let content = fs::read_to_string(&path)?;
        let mut output: Option<PathBuf> = None;
        let mut fps: u32 = 30;
        let mut width: u32 = 0;
        let mut height: u32 = 0;
        let mut samples_per_frame: u64 = 0;
        let mut total_frames: u32 = 0;
        let mut started_at: u64 = 0;
        let mut last_progress_at: u64 = 0;
        let mut cur_frame: u32 = 0;
        let mut pid: Option<u32> = None;
        let mut keyframe_count: usize = 0;
        let mut kf_lines: Vec<(usize, String)> = Vec::new();
        let mut tension_lines: Vec<(usize, f64)> = Vec::new();
        let mut seg_lines: Vec<(usize, f64)> = Vec::new();
        // Backward-compat: old sessions stored `spacing_i=<weight>` as
        // per-segment relative weights. Read into `legacy_spacings` and
        // treat them as absolute segment-seconds if no new-format entries
        // are present.
        let mut legacy_spacings: Vec<(usize, f64)> = Vec::new();
        let mut gamma_override: Option<f32> = None;
        let mut palette_override_lines: Vec<(usize, (f32, [u8; 3]))> = Vec::new();
        let mut rho: f64 = crate::zoompan::RHO_DEFAULT;

        for line in content.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            match k {
                "output" => output = Some(PathBuf::from(v)),
                "fps" => fps = v.parse().unwrap_or(30),
                "width" => width = v.parse().unwrap_or(0),
                "height" => height = v.parse().unwrap_or(0),
                "samples_per_frame" => samples_per_frame = v.parse().unwrap_or(0),
                "total_frames" => total_frames = v.parse().unwrap_or(0),
                "started_at" => started_at = v.parse().unwrap_or(0),
                "last_progress_at" => last_progress_at = v.parse().unwrap_or(0),
                "cur_frame" => cur_frame = v.parse().unwrap_or(0),
                "pid" => pid = v.parse().ok(),
                "keyframe_count" => keyframe_count = v.parse().unwrap_or(0),
                "spacing_count" | "dwell_count" | "tension_count"
                | "segment_count" | "palette_override_count" => {
                    /* informational */
                }
                "gamma_override" => { gamma_override = v.parse().ok(); }
                "rho" => {
                    if let Ok(r) = v.parse::<f64>() { rho = r; }
                }
                _ => {
                    if let Some(idx_str) = k.strip_prefix("palette_override_") {
                        if let Ok(i) = idx_str.parse::<usize>() {
                            let parts: Vec<&str> = v.split(',').collect();
                            if parts.len() == 4 {
                                let pos = parts[0].parse::<f32>().ok();
                                let r = parts[1].parse::<u16>().ok();
                                let g = parts[2].parse::<u16>().ok();
                                let b = parts[3].parse::<u16>().ok();
                                if let (Some(p), Some(r), Some(g), Some(b)) = (pos, r, g, b) {
                                    palette_override_lines.push((i,
                                        (p, [r.min(255) as u8,
                                             g.min(255) as u8,
                                             b.min(255) as u8])));
                                }
                            }
                        }
                    } else if let Some(idx_str) = k.strip_prefix("keyframe_") {
                        if let Ok(i) = idx_str.parse::<usize>() {
                            kf_lines.push((i, v.to_string()));
                        }
                    } else if let Some(idx_str) = k.strip_prefix("tension_") {
                        if let Ok(i) = idx_str.parse::<usize>() {
                            if let Ok(s) = v.parse::<f64>() {
                                tension_lines.push((i, s));
                            }
                        }
                    } else if k.starts_with("dwell_") {
                        // v1 sessions wrote `dwell_i=…`; silently dropped
                        // in v2 (the user deprioritized dwell, and we no
                        // longer have the holds-as-pause concept anyway).
                    } else if let Some(idx_str) = k.strip_prefix("segment_") {
                        if let Ok(i) = idx_str.parse::<usize>() {
                            if let Ok(s) = v.parse::<f64>() {
                                seg_lines.push((i, s));
                            }
                        }
                    } else if let Some(idx_str) = k.strip_prefix("spacing_") {
                        if let Ok(i) = idx_str.parse::<usize>() {
                            if let Ok(s) = v.parse::<f64>() {
                                legacy_spacings.push((i, s));
                            }
                        }
                    }
                }
            }
        }
        kf_lines.sort_by_key(|(i, _)| *i);
        tension_lines.sort_by_key(|(i, _)| *i);
        seg_lines.sort_by_key(|(i, _)| *i);
        legacy_spacings.sort_by_key(|(i, _)| *i);
        palette_override_lines.sort_by_key(|(i, _)| *i);
        let palette_stops_override: Option<Vec<(f32, [u8; 3])>> =
            if palette_override_lines.is_empty() {
                None
            } else {
                Some(palette_override_lines.into_iter().map(|(_, v)| v).collect())
            };
        let keyframes: Vec<PathBuf> =
            kf_lines.into_iter().map(|(_, v)| PathBuf::from(v)).collect();
        let tensions: Vec<f64> = tension_lines.into_iter().map(|(_, v)| v).collect();
        let mut segments: Vec<f64> = seg_lines.into_iter().map(|(_, v)| v).collect();
        if segments.is_empty() && !legacy_spacings.is_empty() {
            // Legacy sessions: treat each spacing_i as the travel time for
            // segment i. Uniform 2.0 s applies only when both are absent.
            segments = legacy_spacings.into_iter().map(|(_, v)| v).collect();
        }
        let _ = keyframe_count;

        let output = output.ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidData, "session.json missing 'output'"))?;

        Ok(Self {
            output,
            fps,
            width,
            height,
            samples_per_frame,
            total_frames,
            keyframes,
            tensions,
            segments,
            pid,
            started_at,
            last_progress_at,
            cur_frame,
            gamma_override,
            palette_stops_override,
            rho,
        })
    }
}

/// Returns all non-empty session dirs under `~/buddhabrot/` matching `.render-*`,
/// sorted by `started_at` descending.
pub fn scan_sessions() -> Vec<(PathBuf, SessionFile)> {
    let root = SessionFile::root();
    let mut out: Vec<(PathBuf, SessionFile)> = Vec::new();
    let Ok(rd) = fs::read_dir(&root) else { return out };
    for ent in rd.flatten() {
        let p = ent.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        if !name.starts_with(".render-") { continue }
        if !p.is_dir() { continue }
        if let Ok(s) = SessionFile::read(&p) {
            out.push((p, s));
        }
    }
    out.sort_by_key(|(_, s)| std::cmp::Reverse(s.started_at));
    out
}

/// Best-effort check whether a process with `pid` is still alive.
#[cfg(unix)]
pub fn pid_alive(pid: u32) -> bool {
    // kill(pid, 0) returns 0 if signal can be delivered; ESRCH if not.
    unsafe { libc_kill_zero(pid as i32) == 0 }
}

#[cfg(not(unix))]
pub fn pid_alive(_pid: u32) -> bool {
    false
}

#[cfg(unix)]
unsafe fn libc_kill_zero(pid: i32) -> i32 {
    extern "C" { fn kill(pid: i32, sig: i32) -> i32; }
    unsafe { kill(pid, 0) }
}
