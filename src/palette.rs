//! Color palettes for the histogram. Repaletting is O(pixels) — no sampling.

use std::sync::atomic::Ordering;

use crate::sampler::Histogram;

pub const LUT_SIZE: usize = 1024;

#[derive(Clone)]
pub struct Palette {
    pub name: &'static str,
    pub lut: Vec<[u8; 4]>, // length LUT_SIZE
}

impl Palette {
    fn from_stops(name: &'static str, stops: &[(f32, [u8; 3])]) -> Self {
        let mut lut = vec![[0u8; 4]; LUT_SIZE];
        for (i, entry) in lut.iter_mut().enumerate() {
            let t = i as f32 / (LUT_SIZE - 1) as f32;
            let [r, g, b] = interp_stops(stops, t);
            *entry = [r, g, b, 255];
        }
        Self { name, lut }
    }

    /// Build a palette from a runtime-editable stop list. Stops will be
    /// sorted by position internally.
    pub fn from_stops_dyn(name: &'static str, stops: &[(f32, [u8; 3])]) -> Self {
        Self::from_stops(name, stops)
    }

    /// Sample the palette's current gradient at `t ∈ [0,1]`, returning an
    /// RGB triple. Useful when inserting a new control point and the caller
    /// wants the inherited color.
    pub fn sample_rgb(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let idx = (t * (LUT_SIZE - 1) as f32).round() as usize;
        let c = self.lut[idx.min(LUT_SIZE - 1)];
        [c[0], c[1], c[2]]
    }
}

fn interp_stops(stops: &[(f32, [u8; 3])], t: f32) -> [u8; 3] {
    if t <= stops[0].0 { return stops[0].1; }
    if t >= stops[stops.len() - 1].0 { return stops[stops.len() - 1].1; }
    for i in 0..stops.len() - 1 {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t >= t0 && t <= t1 {
            let u = (t - t0) / (t1 - t0).max(1e-9);
            return [
                lerp_u8(c0[0], c1[0], u),
                lerp_u8(c0[1], c1[1], u),
                lerp_u8(c0[2], c1[2], u),
            ];
        }
    }
    stops[stops.len() - 1].1
}

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).clamp(0.0, 255.0) as u8
}

pub fn builtin() -> Vec<Palette> {
    builtin_stops()
        .into_iter()
        .map(|(name, stops)| Palette::from_stops(name, &stops))
        .collect()
}

/// Same set of palettes as `builtin()` but exposed as their underlying stop
/// data so callers (the live-mode editable palette HUD, the composer) can
/// seed an editable stop list from a preset.
pub fn builtin_stops() -> Vec<(&'static str, Vec<(f32, [u8; 3])>)> {
    vec![
        ("fire", vec![
            (0.0, [0, 0, 0]),
            (0.15, [40, 8, 60]),
            (0.35, [140, 20, 30]),
            (0.6, [230, 110, 20]),
            (0.85, [250, 220, 110]),
            (1.0, [255, 255, 240]),
        ]),
        ("ice", vec![
            (0.0, [0, 0, 0]),
            (0.2, [10, 20, 50]),
            (0.5, [30, 110, 200]),
            (0.8, [140, 220, 240]),
            (1.0, [255, 255, 255]),
        ]),
        ("nebula", vec![
            (0.0, [0, 0, 0]),
            (0.15, [20, 0, 40]),
            (0.4, [110, 20, 170]),
            (0.65, [220, 50, 120]),
            (0.85, [255, 180, 100]),
            (1.0, [255, 255, 230]),
        ]),
        ("green", vec![
            (0.0, [0, 0, 0]),
            (0.3, [0, 40, 20]),
            (0.65, [50, 200, 110]),
            (1.0, [220, 255, 200]),
        ]),
        ("gray", vec![
            (0.0, [0, 0, 0]),
            (1.0, [255, 255, 255]),
        ]),
    ]
}

/// Apply the palette to the histogram and write RGBA into `out`.
/// Uses log(1+count) / log(1+max) normalization with a gamma curve.
pub fn apply(hist: &Histogram, max: u32, gamma: f32, palette: &Palette, out: &mut [u8]) {
    let n = hist.data.len();
    debug_assert_eq!(out.len(), n * 4);
    let denom = (1.0 + max as f32).ln().max(1e-9);
    let inv_denom = 1.0 / denom;
    let lut = &palette.lut;
    for i in 0..n {
        let c = hist.data[i].load(Ordering::Relaxed);
        let t = if c == 0 { 0.0 } else {
            let ln = (1.0 + c as f32).ln() * inv_denom;
            ln.powf(gamma).clamp(0.0, 1.0)
        };
        let idx = (t * (LUT_SIZE - 1) as f32) as usize;
        let color = lut[idx];
        let o = i * 4;
        out[o] = color[0];
        out[o + 1] = color[1];
        out[o + 2] = color[2];
        out[o + 3] = 255;
    }
}
