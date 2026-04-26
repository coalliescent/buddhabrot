//! Van Wijk–Nuij smooth zoom-and-pan interpolation between two
//! `(center, half_width)` keyframes.
//!
//! Replaces per-dimension Hermite on `center` + `log(half_width)` for
//! intra-segment pan/zoom. Hermite treats pan in world units, so a
//! tangent inherited from wide-shot neighbors overshoots by many
//! screen-widths when the current keyframe is zoomed in — a visible
//! swing in the preview. Van Wijk–Nuij traces a hyperbolic geodesic in
//! `(x, y, log w)` space — a constant-apparent-screen-velocity path.
//! The camera pulls back, pans at that width, then zooms in.
//!
//! `ρ` controls how much pullback happens. √2 (≈1.414) is the paper's
//! recommendation for a single transition. Larger ρ → more mid-segment
//! pullback. ρ → 0 degenerates to log-linear: pure exponential zoom +
//! straight-line pan in world coords, no pullback at all — useful
//! when chained segments read as "bouncy" because every segment
//! independently pulls back.
//!
//! Reference: Jarke J. van Wijk, Wim A. A. Nuij, *Smooth and Efficient
//! Zooming and Panning*, IEEE InfoVis 2003.

/// Paper-recommended default for a single transition.
pub const RHO_DEFAULT: f64 = std::f64::consts::SQRT_2;

/// Interpolate `(center, half_width)` from `(c0, w0)` → `(c1, w1)` at
/// local `u ∈ [0, 1]`, with V-N pullback shape parameter `rho`.
pub fn interp(
    c0: [f64; 2], w0: f64,
    c1: [f64; 2], w1: f64,
    rho: f64,
    u: f64,
) -> ([f64; 2], f64) {
    let dx = c1[0] - c0[0];
    let dy = c1[1] - c0[1];
    let du = (dx * dx + dy * dy).sqrt();

    let w0 = w0.max(1e-300);
    let w1 = w1.max(1e-300);

    // Two degenerate cases share the same limit:
    //   - pan is negligible vs. zoom scale (V-N coefficients blow up);
    //   - ρ → 0 (V-N formulas divide by ρ² → 0).
    // The limit is exponential zoom in `u` plus whatever straight-line
    // pan remains — i.e. no pullback.
    let eps = w0.min(w1) * 1e-9;
    if du <= eps || rho.abs() < 1e-6 {
        let w = w0 * (w1 / w0).powf(u);
        return ([c0[0] + dx * u, c0[1] + dy * u], w);
    }

    let rho_sq = rho * rho;
    let rho_4 = rho_sq * rho_sq;

    let dw_sq = w1 * w1 - w0 * w0;
    let b0 = (dw_sq + rho_4 * du * du) / (2.0 * w0 * rho_sq * du);
    let b1 = (dw_sq - rho_4 * du * du) / (2.0 * w1 * rho_sq * du);

    let r0 = (-b0).asinh();
    let r1 = (-b1).asinh();
    let s_total = (r1 - r0) / rho;
    let z = rho * u * s_total + r0;

    let along = (w0 / rho_sq) * (r0.cosh() * z.tanh() - r0.sinh());
    let w_t = w0 * r0.cosh() / z.cosh();

    let inv_du = 1.0 / du;
    let cx = c0[0] + dx * inv_du * along;
    let cy = c0[1] + dy * inv_du * along;
    ([cx, cy], w_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_match() {
        let (c, w) = interp([0.0, 0.0], 1.0, [3.0, 4.0], 0.01, RHO_DEFAULT, 0.0);
        assert!((c[0]).abs() < 1e-12);
        assert!((c[1]).abs() < 1e-12);
        assert!((w - 1.0).abs() < 1e-12);

        let (c, w) = interp([0.0, 0.0], 1.0, [3.0, 4.0], 0.01, RHO_DEFAULT, 1.0);
        assert!((c[0] - 3.0).abs() < 1e-9);
        assert!((c[1] - 4.0).abs() < 1e-9);
        assert!((w - 0.01).abs() < 1e-9);
    }

    #[test]
    fn pure_zoom_no_pan() {
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let (c, w) = interp(
                [1.5, -2.0], 0.1, [1.5, -2.0], 0.001, RHO_DEFAULT, u);
            assert!((c[0] - 1.5).abs() < 1e-12);
            assert!((c[1] - (-2.0)).abs() < 1e-12);
            let expected = 0.1 * 0.01_f64.powf(u);
            assert!((w - expected).abs() < 1e-12,
                "u={u}: w={w} expected={expected}");
        }
    }

    #[test]
    fn peak_zooms_out_when_both_ends_zoomed_in() {
        // Two zoomed-in keyframes far apart in world coords: path
        // should pull back (peak half_width ≫ endpoint half_width).
        let w_end = 1e-3;
        let (_, w_mid) = interp(
            [0.0, 0.0], w_end, [1.0, 0.0], w_end, RHO_DEFAULT, 0.5);
        assert!(w_mid > w_end * 100.0,
            "expected pullback; w_mid={w_mid}");
    }

    #[test]
    fn rho_zero_suppresses_pullback() {
        // At ρ = 0 the path is pure log-linear zoom + straight-line
        // pan; no mid-segment zoom change when endpoints agree on w.
        let w_end = 1e-3;
        let (_, w_mid) = interp(
            [0.0, 0.0], w_end, [1.0, 0.0], w_end, 0.0, 0.5);
        assert!((w_mid - w_end).abs() < 1e-12,
            "expected no pullback at ρ=0; w_mid={w_mid}");
    }

    #[test]
    fn monotonic_along_pan_axis() {
        let samples: Vec<_> = (0..=20)
            .map(|i| i as f64 / 20.0)
            .map(|u| interp(
                [0.0, 0.0], 0.5, [2.0, 0.0], 0.002, RHO_DEFAULT, u).0[0])
            .collect();
        for w in samples.windows(2) {
            assert!(w[1] >= w[0] - 1e-12,
                "pan should be monotonic; got {} -> {}", w[0], w[1]);
        }
    }
}
