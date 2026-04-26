//! Persist the composer's timeline + render-panel settings to
//! `~/buddhabrot/timeline.state` so a crash (or a restart after an exe-was-
//! replaced render failure) doesn't lose the user's carefully-sequenced
//! keyframes.
//!
//! Format is the same simple `key=value\n` scheme as `session.rs` — one line
//! per field, hand-rolled so we don't pull in serde for a dozen fields.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug)]
pub struct TimelineItemSave {
    pub source: PathBuf,
    pub segment_seconds: f64,
    /// Per-keyframe Hermite tangent multiplier (1.0 = no slowdown).
    /// Older v1 saves (which had `dwell_seconds` instead) load with
    /// tension defaulted to 1.0 and the dwell silently dropped.
    pub tension: f64,
}

#[derive(Clone, Debug)]
pub struct PaletteStopSave {
    pub pos: f32,
    pub rgb: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct TimelineSave {
    pub fps: u32,
    pub rp_width: u32,
    pub rp_height: u32,
    pub rp_spp: f64,
    pub gamma: f32,
    /// V-N path shape parameter for intra-segment pan/zoom.
    pub rho: f64,
    pub playhead: f64,
    pub palette_stops: Vec<PaletteStopSave>,
    pub items: Vec<TimelineItemSave>,
}

/// Autosave file: one slot at `~/buddhabrot/timeline.state`, overwritten on
/// every edit. Separate from the manual-save snapshots so an autosave can't
/// clobber a named save the user made intentionally.
pub fn path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("buddhabrot").join("timeline.state")
}

/// Directory holding manual, timestamped save snapshots.
pub fn saves_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("buddhabrot").join("timelines")
}

/// Atomic write to the default autosave slot. Creates the parent dir if
/// needed.
pub fn write(state: &TimelineSave) -> io::Result<()> {
    write_to(&path(), state)
}

/// Atomic write to an arbitrary path. Used by both the autosave slot and the
/// timestamped manual snapshots.
pub fn write_to(target: &Path, state: &TimelineSave) -> io::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = target.with_extension("state.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        writeln!(f, "version={}", SCHEMA_VERSION)?;
        writeln!(f, "fps={}", state.fps)?;
        writeln!(f, "rp_width={}", state.rp_width)?;
        writeln!(f, "rp_height={}", state.rp_height)?;
        writeln!(f, "rp_spp={}", state.rp_spp)?;
        writeln!(f, "gamma={}", state.gamma)?;
        writeln!(f, "rho={}", state.rho)?;
        writeln!(f, "playhead={}", state.playhead)?;
        writeln!(f, "palette_count={}", state.palette_stops.len())?;
        for (i, s) in state.palette_stops.iter().enumerate() {
            writeln!(f, "palette_{i}_pos={}", s.pos)?;
            writeln!(f, "palette_{i}_rgb={},{},{}",
                s.rgb[0], s.rgb[1], s.rgb[2])?;
        }
        writeln!(f, "item_count={}", state.items.len())?;
        for (i, it) in state.items.iter().enumerate() {
            writeln!(f, "item_{i}_source={}", it.source.display())?;
            writeln!(f, "item_{i}_segment={}", it.segment_seconds)?;
            writeln!(f, "item_{i}_tension={}", it.tension)?;
        }
    }
    fs::rename(tmp, target)
}

/// Write a timestamped snapshot to `~/buddhabrot/timelines/timeline-<ts>.state`
/// and return the resulting path. Distinct from the single-slot autosave:
/// manual saves accumulate so the user can flip between named versions.
pub fn save_manual(state: &TimelineSave) -> io::Result<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let target = saves_dir().join(format!("timeline-{ts}.state"));
    write_to(&target, state)?;
    Ok(target)
}

/// List manual-save files in `saves_dir()`, newest first (by mtime; falls
/// back to filename when mtime isn't available).
pub fn list_saves() -> Vec<PathBuf> {
    let dir = saves_dir();
    let rd = match fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("state") { continue; }
        let mtime = e.metadata().and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        entries.push((p, mtime));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

/// Read the autosave slot.
pub fn read() -> io::Result<Option<TimelineSave>> {
    read_from(&path())
}

/// Read an arbitrary saved-state file. Returns `Ok(None)` only when the file
/// doesn't exist (lets callers treat "first run" as a non-error).
pub fn read_from(p: &Path) -> io::Result<Option<TimelineSave>> {
    let content = match fs::read_to_string(p) {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut kv: HashMap<String, String> = HashMap::new();
    for line in content.lines() {
        if let Some((k, v)) = line.split_once('=') {
            kv.insert(k.to_string(), v.to_string());
        }
    }
    let get = |k: &str| kv.get(k).map(String::as_str);
    let parse_err = |field: &str| io::Error::new(
        io::ErrorKind::InvalidData,
        format!("timeline.state: bad {field}"),
    );

    // Version gate — accept current; reject futures so we don't eat data.
    let v: u32 = get("version").and_then(|s| s.parse().ok())
        .ok_or_else(|| parse_err("version"))?;
    if v > SCHEMA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("timeline.state: unknown version {v} (this build is {SCHEMA_VERSION})"),
        ));
    }

    let fps: u32 = get("fps").and_then(|s| s.parse().ok()).unwrap_or(30);
    let rp_width: u32 = get("rp_width").and_then(|s| s.parse().ok()).unwrap_or(1920);
    let rp_height: u32 = get("rp_height").and_then(|s| s.parse().ok()).unwrap_or(1080);
    let rp_spp: f64 = get("rp_spp").and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let gamma: f32 = get("gamma").and_then(|s| s.parse().ok()).unwrap_or(0.75);
    let rho: f64 = get("rho").and_then(|s| s.parse().ok())
        .unwrap_or(crate::zoompan::RHO_DEFAULT);
    let playhead: f64 = get("playhead").and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let palette_count: usize = get("palette_count")
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut palette_stops = Vec::with_capacity(palette_count);
    for i in 0..palette_count {
        let pos: f32 = get(&format!("palette_{i}_pos"))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| parse_err(&format!("palette_{i}_pos")))?;
        let rgb_str = get(&format!("palette_{i}_rgb"))
            .ok_or_else(|| parse_err(&format!("palette_{i}_rgb")))?;
        let parts: Vec<u8> = rgb_str.split(',')
            .filter_map(|s| s.trim().parse::<u8>().ok())
            .collect();
        if parts.len() != 3 {
            return Err(parse_err(&format!("palette_{i}_rgb")));
        }
        palette_stops.push(PaletteStopSave {
            pos,
            rgb: [parts[0], parts[1], parts[2]],
        });
    }

    let item_count: usize = get("item_count")
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut items = Vec::with_capacity(item_count);
    for i in 0..item_count {
        let source = get(&format!("item_{i}_source"))
            .map(PathBuf::from)
            .ok_or_else(|| parse_err(&format!("item_{i}_source")))?;
        let segment_seconds: f64 = get(&format!("item_{i}_segment"))
            .and_then(|s| s.parse().ok()).unwrap_or(2.0);
        // v2 writes `tension`; v1 wrote `dwell` (silently dropped on
        // load). Default tension = 1.0 means classical centered-
        // difference tangent — no slowdown.
        let tension: f64 = get(&format!("item_{i}_tension"))
            .and_then(|s| s.parse().ok()).unwrap_or(1.0);
        items.push(TimelineItemSave { source, segment_seconds, tension });
    }

    Ok(Some(TimelineSave {
        fps, rp_width, rp_height, rp_spp, gamma, rho, playhead,
        palette_stops, items,
    }))
}

/// Best-effort save: logs errors to stderr and swallows them. Call sites don't
/// need to care — losing a save is already the failure state we're preventing.
pub fn write_best_effort(state: &TimelineSave) {
    if let Err(e) = write(state) {
        eprintln!("timeline.state: save failed: {e}");
    }
}

/// True if the referenced PNG exists on disk right now.
pub fn source_exists(p: &Path) -> bool {
    p.is_file()
}
