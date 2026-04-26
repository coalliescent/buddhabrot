//! Mandelbrot orbit iteration with cheap interior rejection.

use crate::view::{View, Vec4};

/// True if `c` is inside the main cardioid or period-2 bulb (never escapes).
#[inline]
pub fn in_main_bulb(cr: f64, ci: f64) -> bool {
    // period-2 bulb
    let dx = cr + 1.0;
    if dx * dx + ci * ci < 0.0625 { return true; }
    // main cardioid
    let x = cr - 0.25;
    let q = x * x + ci * ci;
    q * (q + x) < 0.25 * ci * ci
}

/// Iterate z² + c from z=0, writing the orbit into `out[0..len]`.
///
/// Returns `(escaped, len)` where `escaped` means |z| > 2 within `max_iter`
/// steps and `len` is how many orbit points were recorded.
#[inline]
pub fn iterate(cr: f64, ci: f64, max_iter: u32, out: &mut [[f64; 2]]) -> (bool, usize) {
    let cap = out.len().min(max_iter as usize);
    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;
    let mut escaped = false;
    let mut len = 0usize;
    for _ in 0..cap {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 4.0 {
            escaped = true;
            break;
        }
        let new_zr = zr2 - zi2 + cr;
        let new_zi = 2.0 * zr * zi + ci;
        zr = new_zr;
        zi = new_zi;
        out[len] = [zr, zi];
        len += 1;
    }
    // Final escape check (caller only counts orbits that actually escaped).
    if !escaped && zr * zr + zi * zi > 4.0 {
        escaped = true;
    }
    (escaped, len)
}

/// Count how many orbit points (reconstructed into 4D with c fixed) project
/// into the current viewport, and return the corresponding pixel indices.
///
/// `pixels_out` must have length >= `len`. Returns the number of contributing
/// points; the first `k` entries of `pixels_out` are valid (as linear indices).
#[inline]
pub fn pixel_contributions(
    view: &View,
    cr: f64,
    ci: f64,
    orbit: &[[f64; 2]],
    pixels_out: &mut [u32],
) -> u32 {
    let mut k = 0u32;
    for &[zr, zi] in orbit {
        let p: Vec4 = [cr, ci, zr, zi];
        if let Some((px, py)) = view.world_to_pixel(p) {
            let idx = py * view.width + px;
            pixels_out[k as usize] = idx;
            k += 1;
        }
    }
    k
}
