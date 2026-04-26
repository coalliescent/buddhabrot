//! Read the `Buddhabrot.*` tEXt chunks written by `savepng` back into a
//! fully-formed `FrameSpec`. Used by the video renderer to ingest keyframe
//! PNGs produced by the UI.

use std::fs::File;
use std::io;
use std::path::Path;

use crate::palette;
use crate::render::FrameSpec;
use crate::view::{Mat4, View};

pub fn read_spec(path: &Path, samples_target: u64, n_workers: usize) -> io::Result<FrameSpec> {
    let file = File::open(path)?;
    let decoder = png::Decoder::new(file);
    let reader = decoder
        .read_info()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let info = reader.info();

    let mut rotation: Option<Mat4> = None;
    let mut center: Option<[f64; 2]> = None;
    let mut half_width: Option<f64> = None;
    let mut max_iter: Option<u32> = None;
    let mut palette_name: Option<String> = None;
    let mut gamma: Option<f32> = None;

    for t in &info.uncompressed_latin1_text {
        match t.keyword.as_str() {
            "Buddhabrot.Rotation" => rotation = parse_rotation(&t.text),
            "Buddhabrot.Center"   => center = parse_vec2(&t.text),
            "Buddhabrot.HalfWidth" => half_width = parse_f64(&t.text),
            "Buddhabrot.MaxIter"  => max_iter = t.text.trim().parse().ok(),
            "Buddhabrot.Palette"  => palette_name = Some(t.text.trim().to_string()),
            "Buddhabrot.Gamma"    => gamma = parse_f64(&t.text).map(|x| x as f32),
            _ => {}
        }
    }

    let (w, h) = (info.width, info.height);
    let mut view = View::new(w, h);
    if let Some(r) = rotation { view.rotation = r; }
    if let Some(c) = center { view.center = c; }
    if let Some(hw) = half_width { view.half_width = hw; }
    if let Some(mi) = max_iter { view.max_iter = mi; }

    let palettes = palette::builtin();
    let palette = palette_name
        .and_then(|n| palettes.iter().find(|p| p.name == n).cloned())
        .unwrap_or_else(|| palettes[0].clone());

    Ok(FrameSpec {
        view,
        palette,
        gamma: gamma.unwrap_or(0.75),
        samples_target,
        n_workers,
    })
}

fn parse_rotation(s: &str) -> Option<Mat4> {
    let rows: Vec<&str> = s.split(';').collect();
    if rows.len() != 4 { return None; }
    let mut r = [[0.0; 4]; 4];
    for (i, row) in rows.iter().enumerate() {
        let nums: Vec<&str> = row.split_whitespace().collect();
        if nums.len() != 4 { return None; }
        for (j, n) in nums.iter().enumerate() {
            r[i][j] = n.parse().ok()?;
        }
    }
    Some(r)
}

fn parse_vec2(s: &str) -> Option<[f64; 2]> {
    let nums: Vec<&str> = s.split_whitespace().collect();
    if nums.len() != 2 { return None; }
    Some([nums[0].parse().ok()?, nums[1].parse().ok()?])
}

fn parse_f64(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}
