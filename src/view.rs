//! 4D viewing transform: SO(4) rotation, 2D pan, zoom, viewport projection.
//!
//! Axes are (X, Y, Z, W) = (c_re, c_im, z_re, z_im). With `rotation = I` the
//! view is the classic c-plane Buddhabrot. Any rotation into Z/W mixes the z
//! state into the visible plane.

pub type Mat4 = [[f64; 4]; 4];
pub type Vec4 = [f64; 4];

/// The six rotation planes of SO(4), indexed by the two axes they span.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Plane {
    XY, XZ, XW, YZ, YW, ZW,
}

impl Plane {
    pub fn axes(self) -> (usize, usize) {
        match self {
            Plane::XY => (0, 1),
            Plane::XZ => (0, 2),
            Plane::XW => (0, 3),
            Plane::YZ => (1, 2),
            Plane::YW => (1, 3),
            Plane::ZW => (2, 3),
        }
    }

    pub const ALL: [Plane; 6] = [
        Plane::XY, Plane::ZW, Plane::XZ, Plane::YW, Plane::XW, Plane::YZ,
    ];
}

#[derive(Clone, Debug)]
pub struct View {
    pub rotation: Mat4,
    /// 2D world-space center of the viewport (after rotation/projection).
    pub center: [f64; 2],
    /// Half-width of the viewport in world units (height derived from aspect).
    pub half_width: f64,
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    /// Monotonic counter bumped on any change. Workers poll to preempt.
    pub generation: u64,
    /// Counts updates since last reorthonormalization, to control drift.
    update_count: u32,
}

impl View {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            rotation: identity(),
            center: [-0.5, 0.0],
            half_width: 1.8,
            width,
            height,
            max_iter: 2000,
            generation: 1,
            update_count: 0,
        }
    }

    pub fn aspect(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    pub fn half_height(&self) -> f64 {
        self.half_width / self.aspect()
    }

    /// Rotate by `angle` in the given plane: R ← G(plane, angle) · R.
    pub fn apply_plane_rotation(&mut self, plane: Plane, angle: f64) {
        let (i, j) = plane.axes();
        let c = angle.cos();
        let s = angle.sin();
        // Build G in-place-acting on the left of R. Only rows i and j change.
        for col in 0..4 {
            let ri = self.rotation[i][col];
            let rj = self.rotation[j][col];
            self.rotation[i][col] = c * ri - s * rj;
            self.rotation[j][col] = s * ri + c * rj;
        }
        self.bump();
    }

    /// Replace the rotation wholesale (e.g. for snap targets).
    pub fn set_rotation(&mut self, r: Mat4) {
        self.rotation = r;
        self.bump();
    }

    pub fn pan_pixels(&mut self, dx_px: f64, dy_px: f64) {
        let sx = 2.0 * self.half_width / self.width as f64;
        let sy = 2.0 * self.half_height() / self.height as f64;
        self.center[0] -= dx_px * sx;
        // Screen y grows downward but world y grows upward, so a drag-down
        // should move the view center *up* (natural grab-and-drag feel).
        self.center[1] += dy_px * sy;
        self.bump();
    }

    /// Zoom around a pixel anchor. `factor < 1` zooms in.
    pub fn zoom_at(&mut self, factor: f64, anchor_px: (f64, f64)) {
        let (wx, wy) = self.pixel_to_world(anchor_px.0, anchor_px.1);
        self.half_width *= factor;
        // Keep the anchor stationary on screen after zoom.
        let (wx2, wy2) = self.pixel_to_world(anchor_px.0, anchor_px.1);
        self.center[0] += wx - wx2;
        self.center[1] += wy - wy2;
        self.bump();
    }

    pub fn reset(&mut self) {
        self.rotation = identity();
        self.center = [-0.5, 0.0];
        self.half_width = 1.8;
        self.max_iter = 2000;
        self.bump();
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.bump();
        }
    }

    pub fn set_max_iter(&mut self, mi: u32) {
        if mi != self.max_iter {
            self.max_iter = mi.max(10);
            self.bump();
        }
    }

    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.update_count += 1;
        if self.update_count >= 100 {
            self.update_count = 0;
            self.rotation = reorthonormalize(&self.rotation);
        }
    }

    /// 4D → 2D: take the first two components of R·p.
    #[inline]
    pub fn project(&self, p: Vec4) -> [f64; 2] {
        let r = &self.rotation;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + r[0][3] * p[3],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + r[1][3] * p[3],
        ]
    }

    #[inline]
    pub fn world_to_pixel(&self, p: Vec4) -> Option<(u32, u32)> {
        let [wx, wy] = self.project(p);
        let dx = wx - self.center[0];
        let dy = wy - self.center[1];
        let hw = self.half_width;
        let hh = self.half_height();
        if dx < -hw || dx >= hw || dy < -hh || dy >= hh {
            return None;
        }
        let px = ((dx + hw) / (2.0 * hw) * self.width as f64) as i32;
        let py = ((hh - dy) / (2.0 * hh) * self.height as f64) as i32;
        if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
            None
        } else {
            Some((px as u32, py as u32))
        }
    }

    pub fn pixel_to_world(&self, px: f64, py: f64) -> (f64, f64) {
        let hw = self.half_width;
        let hh = self.half_height();
        let wx = self.center[0] + (px / self.width as f64 * 2.0 - 1.0) * hw;
        let wy = self.center[1] + (1.0 - py / self.height as f64 * 2.0) * hh;
        (wx, wy)
    }
}

pub fn identity() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 { m[i][i] = 1.0; }
    m
}

/// Modified Gram-Schmidt on rows (rotation rows form an orthonormal basis).
pub fn reorthonormalize(m: &Mat4) -> Mat4 {
    let mut r = *m;
    for i in 0..4 {
        // Normalize row i.
        let mut n2 = 0.0;
        for k in 0..4 { n2 += r[i][k] * r[i][k]; }
        let n = n2.sqrt().max(1e-300);
        for k in 0..4 { r[i][k] /= n; }
        // Subtract projection of row i from later rows.
        for j in (i + 1)..4 {
            let mut dot = 0.0;
            for k in 0..4 { dot += r[i][k] * r[j][k]; }
            for k in 0..4 { r[j][k] -= dot * r[i][k]; }
        }
    }
    r
}
