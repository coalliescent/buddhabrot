//! Input → view delta mapping. Kept tiny and pure.

use crate::view::Plane;

/// For a drag of (dx, dy) pixels with the given modifier state, what
/// rotations to apply? Horizontal and vertical can be different planes.
pub fn drag_planes(shift: bool, ctrl: bool) -> (Plane, Plane) {
    match (shift, ctrl) {
        (false, false) => (Plane::XZ, Plane::YW),  // mix c_re↔z_re, c_im↔z_im
        (true, false)  => (Plane::XY, Plane::ZW),  // spin c-plane / z-plane
        (false, true)  => (Plane::XW, Plane::YZ),  // cross-couples
        (true, true)   => (Plane::XZ, Plane::YW),  // fallback
    }
}

/// Right-drag is a "3D-camera-orbit" style rotation: horizontal yaw and
/// vertical pitch share one depth axis (Z or W), so the motion feels like
/// rotating a rigid object rather than shuffling through mixed 4D axes.
/// Shift picks W as depth instead of Z.
pub fn right_drag_planes(shift: bool) -> (Plane, Plane) {
    if shift {
        (Plane::XW, Plane::YW)  // W-depth orbit
    } else {
        (Plane::XZ, Plane::YZ)  // Z-depth orbit
    }
}

/// Convert a pixel drag delta into a rotation angle (radians). `zoom_scale`
/// attenuates the sensitivity so a given mouse motion rotates less when the
/// view is zoomed in (see `zoom_rotation_scale`).
pub fn drag_to_angle(d_px: f64, zoom_scale: f64) -> f64 {
    d_px * 0.005 * zoom_scale
}

/// Rotation-sensitivity multiplier derived from the current `half_width`.
/// Returns 1.0 at the default framing (half_width = 1.8), drops smoothly as
/// the user zooms in (≈0.5 at 10× deeper, ≈0.25 at 100×), and is clamped so
/// it neither speeds up when zoomed out nor falls below a usable floor.
pub fn zoom_rotation_scale(half_width: f64) -> f64 {
    const REF_HW: f64 = 1.8;
    const MIN: f64 = 0.15;
    const EXP: f64 = 0.3;
    (half_width / REF_HW).powf(EXP).clamp(MIN, 1.0)
}
