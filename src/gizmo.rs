//! Tesseract orientation gizmo with face picking + snap-to-plane.

use crate::view::{self, Mat4, Plane, Vec4, View};

pub struct Gizmo {
    pub rect: Rect,          // screen-space bounds
    pub radius: f64,         // projected half-extent
    vertices_4d: [Vec4; 16],
    edges: Vec<(usize, usize)>,
    faces: Vec<Face4>,
}

#[derive(Copy, Clone)]
pub struct Rect { pub x: i32, pub y: i32, pub w: u32, pub h: u32 }

impl Rect {
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && py >= self.y
            && px < self.x + self.w as i32
            && py < self.y + self.h as i32
    }
}

#[derive(Clone, Debug)]
struct Face4 {
    plane: Plane,
    vertex_indices: [usize; 4],  // CCW in the face's own 2D
    fixed: [(usize, f64); 2],    // (axis, ±1) of the two axes held constant
}

/// Identity of a clicked face, enough to produce its snap target.
#[derive(Copy, Clone, Debug)]
pub struct FaceInfo {
    pub plane: Plane,
    pub fixed_axes: [usize; 2],
    pub fixed_signs: [f64; 2],
}

/// Which snap the caller wants for a clicked face.
///
/// Selected by click-zone in `Gizmo::pick`: center of a face → `Cardinal`
/// (axis-aligned view of that plane); near a corner → `Corner` (45° tilt
/// that pulls the face's two fixed axes into the view plane).
#[derive(Copy, Clone, Debug)]
pub enum SnapKind {
    Cardinal,
    Corner,
}

impl Gizmo {
    pub fn new(rect: Rect) -> Self {
        let radius = (rect.w.min(rect.h) as f64) * 0.36;
        let mut vertices_4d = [[0.0; 4]; 16];
        for i in 0..16 {
            let x = if i & 1 != 0 { 1.0 } else { -1.0 };
            let y = if i & 2 != 0 { 1.0 } else { -1.0 };
            let z = if i & 4 != 0 { 1.0 } else { -1.0 };
            let w = if i & 8 != 0 { 1.0 } else { -1.0 };
            vertices_4d[i] = [x, y, z, w];
        }
        let mut edges = Vec::with_capacity(32);
        for a in 0..16 {
            for b in (a + 1)..16 {
                let diff = (a ^ b) as u32;
                if diff.count_ones() == 1 {
                    edges.push((a, b));
                }
            }
        }
        let faces = build_faces();
        Self { rect, radius, vertices_4d, edges, faces }
    }

    pub fn set_rect(&mut self, rect: Rect) {
        self.rect = rect;
        self.radius = (rect.w.min(rect.h) as f64) * 0.36;
    }

    /// Project a tesseract vertex via the view rotation; returns (sx, sy, depth).
    /// sx,sy are screen-space pixels; depth is mean of the non-projected rows.
    #[inline]
    fn project(&self, v: Vec4, r: &Mat4) -> (f64, f64, f64) {
        let rx = r[0][0]*v[0] + r[0][1]*v[1] + r[0][2]*v[2] + r[0][3]*v[3];
        let ry = r[1][0]*v[0] + r[1][1]*v[1] + r[1][2]*v[2] + r[1][3]*v[3];
        let rz = r[2][0]*v[0] + r[2][1]*v[1] + r[2][2]*v[2] + r[2][3]*v[3];
        let rw = r[3][0]*v[0] + r[3][1]*v[1] + r[3][2]*v[2] + r[3][3]*v[3];
        let cx = self.rect.x as f64 + self.rect.w as f64 * 0.5;
        let cy = self.rect.y as f64 + self.rect.h as f64 * 0.5;
        let scale = self.radius * 0.5; // verts have range ±2 after rotation
        (cx + rx * scale, cy - ry * scale, rz + rw)
    }

    pub fn draw(&self, view: &View, frame: &mut [u8], fw: u32, fh: u32) {
        // Panel background.
        crate::overlay::fill_rect(
            frame, fw, fh,
            self.rect.x, self.rect.y, self.rect.w, self.rect.h,
            [10, 12, 18, 180],
        );
        // Thin border.
        draw_rect_outline(frame, fw, fh, self.rect, [90, 100, 120]);

        let r = &view.rotation;

        // Project all vertices once.
        let projected: Vec<(f64, f64, f64)> = self.vertices_4d.iter()
            .map(|&v| self.project(v, r)).collect();

        // Draw faces back-to-front.
        let mut face_order: Vec<usize> = (0..self.faces.len()).collect();
        face_order.sort_by(|&a, &b| {
            let da: f64 = self.faces[a].vertex_indices.iter()
                .map(|&vi| projected[vi].2).sum::<f64>();
            let db: f64 = self.faces[b].vertex_indices.iter()
                .map(|&vi| projected[vi].2).sum::<f64>();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        for fi in face_order {
            let f = &self.faces[fi];
            let pts = [
                (projected[f.vertex_indices[0]].0, projected[f.vertex_indices[0]].1),
                (projected[f.vertex_indices[1]].0, projected[f.vertex_indices[1]].1),
                (projected[f.vertex_indices[2]].0, projected[f.vertex_indices[2]].1),
                (projected[f.vertex_indices[3]].0, projected[f.vertex_indices[3]].1),
            ];
            let color = plane_color(f.plane);
            fill_convex_quad(frame, fw, fh, &pts, self.rect, [color[0], color[1], color[2], 70]);
        }

        // Draw edges on top.
        for &(a, b) in &self.edges {
            let (ax, ay, _) = projected[a];
            let (bx, by, _) = projected[b];
            draw_line_clipped(frame, fw, fh, ax, ay, bx, by, self.rect, [230, 230, 230, 220]);
        }
    }

    /// Hit-test a click at (px, py) and return the rotation to snap to, or
    /// `None` if the click missed the gizmo.
    ///
    /// Two hit modes:
    /// - **Vertex hit**: click within `0.15 * self.radius` of any projected
    ///   tesseract vertex. Candidates = the 6 faces adjacent to the chosen
    ///   vertex (closest-to-camera among colocated ones), kind = Corner,
    ///   restricted to sign-mirror variants that keep that vertex forward.
    ///   This makes repeat clicks at a shared corner idempotent even as the
    ///   gizmo re-renders under the cursor.
    /// - **Face hit**: fall back to strict point-in-quad. Kind is Cardinal
    ///   (near the face's centroid) or Corner (outer band), any of 8
    ///   sign-mirror variants.
    ///
    /// In both modes, the current rotation's XY + ZW roll is extracted,
    /// subtracted from the sign-variant search, then re-applied to the
    /// chosen target — so shift+drag "in-place" orientation survives the
    /// snap.
    pub fn pick(&self, px: i32, py: i32, view: &View) -> Option<Mat4> {
        if !self.rect.contains(px, py) { return None; }
        let r = &view.rotation;
        let projected: Vec<(f64, f64, f64)> = self.vertices_4d.iter()
            .map(|&v| self.project(v, r)).collect();
        let p = (px as f64, py as f64);

        let vertex_snap_r = 0.15 * self.radius;
        let near_vertex: Option<usize> = projected.iter().enumerate()
            .filter(|(_, pp)| {
                (p.0 - pp.0).powi(2) + (p.1 - pp.1).powi(2) < vertex_snap_r.powi(2)
            })
            .max_by(|a, b| a.1.2.partial_cmp(&b.1.2)
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        let (candidate_indices, kind): (Vec<usize>, SnapKind) = if near_vertex.is_some() {
            // Sign-mirror variants can yield a V-forward rotation even when
            // V isn't a vertex of the nominal face (e.g. near2.png is a
            // variant of XY(+1,+1) but its forward vertex has v[3]=-1).
            // Consider all 24 faces; forward_vertex_is filters the pool.
            ((0..self.faces.len()).collect(), SnapKind::Corner)
        } else {
            let mut best: Option<(usize, f64)> = None;
            for (fi, f) in self.faces.iter().enumerate() {
                let pts: [(f64, f64); 4] = std::array::from_fn(|n| {
                    (projected[f.vertex_indices[n]].0,
                     projected[f.vertex_indices[n]].1)
                });
                if !point_in_convex_quad(p, &pts) { continue; }
                let depth: f64 = f.vertex_indices.iter()
                    .map(|&vi| projected[vi].2).sum::<f64>() / 4.0;
                if best.map_or(true, |(_, d)| depth > d) {
                    best = Some((fi, depth));
                }
            }
            let (fi, _) = best?;
            let f = &self.faces[fi];
            let pts: [(f64, f64); 4] = std::array::from_fn(|n| {
                (projected[f.vertex_indices[n]].0,
                 projected[f.vertex_indices[n]].1)
            });
            let cx = (pts[0].0 + pts[1].0 + pts[2].0 + pts[3].0) * 0.25;
            let cy = (pts[0].1 + pts[1].1 + pts[2].1 + pts[3].1) * 0.25;
            let mean_r = pts.iter()
                .map(|&(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
                .sum::<f64>() * 0.25;
            let d = ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt();
            let kind = if d < 0.5 * mean_r { SnapKind::Cardinal } else { SnapKind::Corner };
            (vec![fi], kind)
        };

        if candidate_indices.is_empty() { return None; }

        // Extract roll so we can re-apply it after the canonical/sign-variant
        // is chosen. The atan2 is of current row-1 col-0 vs row-0 col-0 (XY
        // roll component) and row-3 col-2 vs row-2 col-2 (ZW roll component).
        let theta = r[1][0].atan2(r[0][0]);
        let phi = r[3][2].atan2(r[2][2]);
        let roll_then = mat_mul4(&gxy(theta), &gzw(phi));

        // Search directly for the target rotation = roll · variant that
        // minimizes L1 distance to the current rotation. For vertex hits,
        // require V forward in the *target* (not in the variant), since
        // G_ZW(φ) rotates rows 2, 3 and changes the forward vertex.
        let mut best: Option<(Mat4, f64)> = None;
        for fi in candidate_indices {
            let f = &self.faces[fi];
            let info = FaceInfo {
                plane: f.plane,
                fixed_axes: [f.fixed[0].0, f.fixed[1].0],
                fixed_signs: [f.fixed[0].1, f.fixed[1].1],
            };
            let canon = snap_rotation_for_face(&info, kind);
            for signs in &SIGN_VARIANTS {
                let variant = scale_rows(&canon, signs);
                let target = mat_mul4(&roll_then, &variant);
                if let Some(vi) = near_vertex {
                    if !forward_vertex_is(&target, vi, &self.vertices_4d) { continue; }
                }
                let d = mat_l1_dist(&target, r);
                if best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((target, d));
                }
            }
        }

        best.map(|(target, _)| target)
    }
}

fn mat_l1_dist(a: &Mat4, b: &Mat4) -> f64 {
    let mut d = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            d += (a[i][j] - b[i][j]).abs();
        }
    }
    d
}

fn mat_mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Elementary rotation in plane (0, 1). Left-multiplying by `gxy(θ)` rolls
/// the screen frame by θ.
fn gxy(theta: f64) -> Mat4 {
    let c = theta.cos();
    let s = theta.sin();
    let mut m = [[0.0; 4]; 4];
    m[0][0] =  c; m[0][1] = -s;
    m[1][0] =  s; m[1][1] =  c;
    m[2][2] = 1.0;
    m[3][3] = 1.0;
    m
}

/// Elementary rotation in plane (2, 3). Left-multiplying by `gzw(φ)` rolls
/// the depth frame by φ.
fn gzw(phi: f64) -> Mat4 {
    let c = phi.cos();
    let s = phi.sin();
    let mut m = [[0.0; 4]; 4];
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    m[2][2] =  c; m[2][3] = -s;
    m[3][2] =  s; m[3][3] =  c;
    m
}

fn scale_rows(mat: &Mat4, signs: &[f64; 4]) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = mat[i][j] * signs[i];
        }
    }
    m
}

/// Valid row-sign flips for a rotation matrix: even count of `-1`s so the
/// determinant stays `+1`. Each variant is a distinct snap target.
const SIGN_VARIANTS: [[f64; 4]; 8] = [
    [ 1.0,  1.0,  1.0,  1.0],
    [ 1.0,  1.0, -1.0, -1.0],
    [ 1.0, -1.0,  1.0, -1.0],
    [ 1.0, -1.0, -1.0,  1.0],
    [-1.0,  1.0,  1.0, -1.0],
    [-1.0,  1.0, -1.0,  1.0],
    [-1.0, -1.0,  1.0,  1.0],
    [-1.0, -1.0, -1.0, -1.0],
];

/// True iff tesseract vertex `vi` is the (strict) forward vertex under the
/// rotation `m` — i.e. it maximizes `row_2·v + row_3·v` over the 16 verts.
fn forward_vertex_is(m: &Mat4, vi: usize, verts: &[Vec4; 16]) -> bool {
    let depth = |v: &Vec4| -> f64 {
        (0..4).map(|c| (m[2][c] + m[3][c]) * v[c]).sum()
    };
    let target = depth(&verts[vi]);
    for (i, v) in verts.iter().enumerate() {
        if i == vi { continue; }
        if depth(v) > target + 1e-9 { return false; }
    }
    true
}

/// Target rotation for a clicked face.
///
/// - `SnapKind::Cardinal` (tilt = 0): axis-aligned snap
///   (`row 0 = e_i, row 1 = e_j, row 2 = sk·e_k, row 3 = sl·e_l`). The clicked
///   plane fills the screen; the two fixed axes stay perpendicular to it. All
///   4 faces of a given plane collapse to the same visible orientation.
/// - `SnapKind::Corner` (tilt = 45°): starts from the cardinal snap, then
///   rotates 45° in the (row 0, row 2) plane and 45° in the (row 1, row 3)
///   plane. That pulls the fixed axes into the view plane (so you see their
///   signed extents) while keeping the clicked plane dominant. Each of the
///   24 tesseract faces yields a distinct view.
pub fn snap_rotation_for_face(info: &FaceInfo, kind: SnapKind) -> Mat4 {
    let angle = match kind {
        SnapKind::Cardinal => 0.0,
        SnapKind::Corner => std::f64::consts::FRAC_PI_4,
    };
    let (i, j) = info.plane.axes();
    let k = info.fixed_axes[0];
    let l = info.fixed_axes[1];
    let sk = info.fixed_signs[0];
    let sl = info.fixed_signs[1];
    let ca = angle.cos();
    let sa = angle.sin();

    let mut r = [[0.0; 4]; 4];
    r[0][i] = ca;
    r[0][k] = sa * sk;
    r[1][j] = ca;
    r[1][l] = sa * sl;
    r[2][i] = -sa;
    r[2][k] = ca * sk;
    r[3][j] = -sa;
    r[3][l] = ca * sl;

    if det4(&r) < 0.0 {
        for c in 0..4 { r[3][c] = -r[3][c]; }
    }
    r
}

/// Matrix interpolation via componentwise lerp + Gram-Schmidt reorthonormalize.
/// Not a true SO(4) geodesic but smooth enough for a ~250 ms snap animation.
pub fn interp_rotation(a: &Mat4, b: &Mat4, t: f64) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[i][j] * (1.0 - t) + b[i][j] * t;
        }
    }
    view::reorthonormalize(&m)
}

pub fn plane_color(p: Plane) -> [u8; 3] {
    match p {
        Plane::XY => [230, 120, 120], // c-plane: warm
        Plane::ZW => [120, 180, 230], // z-plane: cool
        Plane::XZ => [220, 200, 110],
        Plane::YW => [150, 210, 150],
        Plane::XW => [200, 140, 210],
        Plane::YZ => [120, 210, 200],
    }
}

fn build_faces() -> Vec<Face4> {
    let mut faces = Vec::with_capacity(24);
    // For each plane (varying axes i, j), iterate the 4 sign combinations of
    // the other two (fixed) axes.
    for &plane in &Plane::ALL {
        let (i, j) = plane.axes();
        let mut fixed_axes = Vec::with_capacity(2);
        for k in 0..4 {
            if k != i && k != j { fixed_axes.push(k); }
        }
        let ka = fixed_axes[0];
        let kb = fixed_axes[1];
        for &sa in &[-1.0, 1.0] {
            for &sb in &[-1.0, 1.0] {
                // 4 corners in CCW order in the (i, j) plane.
                let corners_ij: [(f64, f64); 4] = [
                    (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
                ];
                let vs: [usize; 4] = std::array::from_fn(|n| {
                    let (vi, vj) = corners_ij[n];
                    let mut c = [0.0; 4];
                    c[i] = vi; c[j] = vj; c[ka] = sa; c[kb] = sb;
                    vec4_to_index(c)
                });
                faces.push(Face4 {
                    plane,
                    vertex_indices: vs,
                    fixed: [(ka, sa), (kb, sb)],
                });
            }
        }
    }
    faces
}

fn vec4_to_index(c: [f64; 4]) -> usize {
    let mut idx = 0;
    for k in 0..4 {
        if c[k] > 0.0 { idx |= 1 << k; }
    }
    idx
}

fn det4(m: &Mat4) -> f64 {
    // Direct 4x4 determinant by cofactor expansion along row 0.
    let mut d = 0.0;
    for j in 0..4 {
        let mut minor = [[0.0; 3]; 3];
        for ii in 1..4 {
            let mut jj2 = 0;
            for jj in 0..4 {
                if jj == j { continue; }
                minor[ii - 1][jj2] = m[ii][jj];
                jj2 += 1;
            }
        }
        let sign = if j & 1 == 0 { 1.0 } else { -1.0 };
        d += sign * m[0][j] * det3(&minor);
    }
    d
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn point_in_convex_quad(p: (f64, f64), q: &[(f64, f64); 4]) -> bool {
    let mut sign: f64 = 0.0;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
        if cross.abs() < 1e-9 { continue; }
        if sign == 0.0 { sign = cross.signum(); }
        else if sign * cross < 0.0 { return false; }
    }
    true
}

// --- Drawing primitives into the framebuffer, clipped to a rect. ---

fn clip_to_rect(fw: u32, fh: u32, clip: Rect, px: i32, py: i32) -> Option<(u32, u32)> {
    if px < clip.x || py < clip.y { return None; }
    if px >= clip.x + clip.w as i32 || py >= clip.y + clip.h as i32 { return None; }
    if px < 0 || py < 0 || px as u32 >= fw || py as u32 >= fh { return None; }
    Some((px as u32, py as u32))
}

fn plot(frame: &mut [u8], fw: u32, x: u32, y: u32, color: [u8; 4]) {
    let idx = ((y * fw + x) * 4) as usize;
    let a = color[3] as f32 / 255.0;
    for c in 0..3 {
        let dst = frame[idx + c] as f32;
        let src = color[c] as f32;
        frame[idx + c] = (dst + (src - dst) * a).clamp(0.0, 255.0) as u8;
    }
    frame[idx + 3] = 255;
}

fn draw_line_clipped(
    frame: &mut [u8], fw: u32, fh: u32,
    x0: f64, y0: f64, x1: f64, y1: f64,
    clip: Rect, color: [u8; 4],
) {
    // Simple Bresenham with alpha dots; good enough at 140px.
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let steps = dx.max(dy).ceil().max(1.0) as i32;
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let x = (x0 + (x1 - x0) * t).round() as i32;
        let y = (y0 + (y1 - y0) * t).round() as i32;
        if let Some((px, py)) = clip_to_rect(fw, fh, clip, x, y) {
            plot(frame, fw, px, py, color);
        }
    }
}

fn draw_rect_outline(frame: &mut [u8], fw: u32, fh: u32, r: Rect, color: [u8; 3]) {
    let c = [color[0], color[1], color[2], 200];
    for x in r.x..(r.x + r.w as i32) {
        for y in [r.y, r.y + r.h as i32 - 1] {
            if let Some((px, py)) = clip_to_rect(fw, fh, r, x, y) {
                plot(frame, fw, px, py, c);
            }
        }
    }
    for y in r.y..(r.y + r.h as i32) {
        for x in [r.x, r.x + r.w as i32 - 1] {
            if let Some((px, py)) = clip_to_rect(fw, fh, r, x, y) {
                plot(frame, fw, px, py, c);
            }
        }
    }
}

fn fill_convex_quad(
    frame: &mut [u8], fw: u32, fh: u32,
    pts: &[(f64, f64); 4],
    clip: Rect, color: [u8; 4],
) {
    let mut y_min = pts[0].1;
    let mut y_max = pts[0].1;
    for p in pts.iter() {
        if p.1 < y_min { y_min = p.1; }
        if p.1 > y_max { y_max = p.1; }
    }
    let y_min = y_min.floor() as i32;
    let y_max = y_max.ceil() as i32;

    for y in y_min..=y_max {
        // Find intersections of horizontal line y with the 4 edges.
        let mut x_in = [f64::INFINITY; 2];
        let mut count = 0;
        for i in 0..4 {
            let a = pts[i];
            let b = pts[(i + 1) % 4];
            let ya = a.1;
            let yb = b.1;
            let yf = y as f64 + 0.5;
            if (ya <= yf && yb > yf) || (yb <= yf && ya > yf) {
                let t = (yf - ya) / (yb - ya);
                let x = a.0 + (b.0 - a.0) * t;
                if count < 2 {
                    x_in[count] = x;
                    count += 1;
                }
            }
        }
        if count < 2 { continue; }
        let (x0, x1) = if x_in[0] <= x_in[1] { (x_in[0], x_in[1]) } else { (x_in[1], x_in[0]) };
        let ix0 = x0.ceil() as i32;
        let ix1 = x1.floor() as i32;
        for x in ix0..=ix1 {
            if let Some((px, py)) = clip_to_rect(fw, fh, clip, x, y) {
                plot(frame, fw, px, py, color);
            }
        }
    }
}
