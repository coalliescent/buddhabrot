//! Onion-skin of the pre-drag frame, shown under the live sparse frame during
//! drag interactions as spatial feedback for rotation/pan/zoom.
//!
//! The snapshot is a palette-applied RGBA image plus the view parameters at
//! capture time. To draw it under the current view, we sample old pixels per
//! new pixel via an affine screen-space map derived from:
//!
//!   For each new-screen pixel `(u, v)`, find the world point `w_n = c_n + off`,
//!   lift to 4D assuming it lies in the new screen plane: `p4 = R_n^T [w_n;0;0]`,
//!   project to the old screen: `w_o = (R_o · p4)[..2] = (R_o R_n^T)[..2,..2] · w_n`,
//!   then old screen pixel `(u', v') = (w_o - c_o) / pitch_o + (cx_o, cy_o)`.
//!
//! Only the top-left 2×2 of `Q = R_o · R_n^T` matters. Rotations that mix in
//! axes outside the old screen plane (XZ, XW, YZ, YW) manifest as anisotropic
//! scaling of the onion — the x- or y-axis "tilts away" from the viewer. ZW
//! rotations leave Q=I, so the onion stays put (correct: ZW doesn't change
//! what's visible at identity).

use crate::view::{Mat4, View};

pub struct Onion {
    pub image: Vec<u8>,  // RGBA, tightly packed
    pub w: u32,
    pub h: u32,
    pub rotation: Mat4,
    pub center: [f64; 2],
    pub half_width: f64,
}

impl Onion {
    pub fn capture(rgba: &[u8], w: u32, h: u32, view: &View) -> Self {
        Self {
            image: rgba.to_vec(),
            w,
            h,
            rotation: view.rotation,
            center: view.center,
            half_width: view.half_width,
        }
    }

    /// Max-blend the onion (scaled by `fade`) into `dst`.
    pub fn blit_over(
        &self,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        view: &View,
        fade: f32,
    ) {
        if self.w == 0 || self.h == 0 { return; }

        let q = q_top_2x2(&self.rotation, &view.rotation);

        let pitch_n = 2.0 * view.half_width / dst_w.max(1) as f64;
        let pitch_o = 2.0 * self.half_width / self.w.max(1) as f64;
        let inv_po = 1.0 / pitch_o;
        let cx_n = dst_w as f64 * 0.5;
        let cy_n = dst_h as f64 * 0.5;
        let cx_o = self.w as f64 * 0.5;
        let cy_o = self.h as f64 * 0.5;

        // (Q · c_new) − c_old: world-space offset between the old and new frames.
        let qcn_x = q[0][0] * view.center[0] + q[0][1] * view.center[1];
        let qcn_y = q[1][0] * view.center[0] + q[1][1] * view.center[1];
        let d_x = qcn_x - self.center[0];
        let d_y = qcn_y - self.center[1];

        // DDA coefficients: px_o = a_xu*u + a_xv*v + b_x,  same for py_o.
        let a_xu = q[0][0] * pitch_n * inv_po;
        let a_xv = -q[0][1] * pitch_n * inv_po;
        let a_yu = -q[1][0] * pitch_n * inv_po;
        let a_yv = q[1][1] * pitch_n * inv_po;

        // Constants at (u,v)=(0,0):
        // offset_new at (0,0): ((0.5 - cx_n)*pitch_n, -(0.5 - cy_n)*pitch_n)
        // offset_old = Q · offset_new + (d_x, d_y)
        // px_o = offset_old.x / pitch_o + cx_o
        // py_o = -offset_old.y / pitch_o + cy_o
        let ox0 = q[0][0] * (0.5 - cx_n) * pitch_n
                + q[0][1] * (-(0.5 - cy_n) * pitch_n)
                + d_x;
        let oy0 = q[1][0] * (0.5 - cx_n) * pitch_n
                + q[1][1] * (-(0.5 - cy_n) * pitch_n)
                + d_y;
        let b_x = ox0 * inv_po + cx_o;
        let b_y = -oy0 * inv_po + cy_o;

        let fade = fade.clamp(0.0, 1.0);
        let src_img = &self.image;
        let dst_w_i = dst_w as i32;
        let ow_i = self.w as i32;
        let oh_i = self.h as i32;

        for v in 0..dst_h as i32 {
            let mut xo = a_xv * v as f64 + b_x;
            let mut yo = a_yv * v as f64 + b_y;
            let row_base = (v * dst_w_i) as usize * 4;
            for u in 0..dst_w_i {
                let xi = xo as i32;
                let yi = yo as i32;
                if xi >= 0 && yi >= 0 && xi < ow_i && yi < oh_i {
                    let src = ((yi as u32 * self.w + xi as u32) * 4) as usize;
                    let dst_i = row_base + (u as usize) * 4;
                    // Max blend with fade. Using integer math for speed.
                    let r = ((src_img[src] as f32) * fade) as u8;
                    let g = ((src_img[src + 1] as f32) * fade) as u8;
                    let b = ((src_img[src + 2] as f32) * fade) as u8;
                    if r > dst[dst_i]     { dst[dst_i]     = r; }
                    if g > dst[dst_i + 1] { dst[dst_i + 1] = g; }
                    if b > dst[dst_i + 2] { dst[dst_i + 2] = b; }
                    dst[dst_i + 3] = 255;
                }
                xo += a_xu;
                yo += a_yu;
            }
        }
    }
}

/// Top-left 2×2 of `R_old · R_new^T`.
fn q_top_2x2(r_old: &Mat4, r_new: &Mat4) -> [[f64; 2]; 2] {
    let mut q = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0.0;
            // R_new^T[k][j] = R_new[j][k]
            for k in 0..4 { s += r_old[i][k] * r_new[j][k]; }
            q[i][j] = s;
        }
    }
    q
}
