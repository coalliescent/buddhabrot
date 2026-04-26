//! Geodesic interpolation between 4×4 rotation matrices via the
//! bi-quaternion representation of SO(4), with a time-aware cubic
//! Hermite curve so the camera's angular velocity is C¹ at keyframes
//! regardless of how the user spaces segment durations.
//!
//! Every `R ∈ SO(4)` can be written as `R(v) = l · v · conj(r)` where
//! `l` and `r` are unit quaternions (a 2:1 cover: `(l, r)` and `(-l,
//! -r)` give the same rotation). We extract both, sign-align them
//! globally so consecutive keyframes share representatives
//! (`build_aligned_biquats`), then interpolate each quaternion
//! independently with a cubic Hermite in its Lie algebra. Reconstruction
//! lands in SO(4) exactly — no Gram-Schmidt, no element-wise
//! interpolation passing through degenerate midpoints when adjacent
//! keyframes differ by ~180°, no velocity flip across seams from
//! antipodal sign choices.
//!
//! The Hermite tangent at each keyframe is the centered difference of
//! the relative-log neighbors, scaled by the user's per-keyframe
//! `tension` (default 1.0) and divided by the wall-time gap of the
//! flanking keyframes so segment-duration changes don't add velocity
//! kinks. See `eval_hermite`.
//!
//! Convention: quaternion components are `[w, x, y, z]`; the 4-vector
//! `v` being rotated is interpreted as the quaternion `w + xi + yj + zk`
//! using the same column order.

use crate::view::Mat4;

pub type Quat = [f64; 4];
/// Pure-imaginary log of a unit quaternion / body-frame angular velocity
/// vector — three components, scalar part is implicit zero.
pub type LogVec = [f64; 3];

/// Extract `(l, r)` unit quaternions such that `m(v) = l · v · conj(r)`.
/// The sign of the pair is ambiguous (any `(l, r)` equals `(-l, -r)`);
/// we pick a consistent one by choosing the rank-1 table's largest
/// column. Works for any `m ∈ SO(4)`, including isoclinic and 180°
/// rotations — these correspond to the table becoming low-rank in a
/// single column, and the largest-column rule still picks a valid
/// representative.
pub fn biquat_extract(m: &Mat4) -> (Quat, Quat) {
    // Table of 4 l_i r_j from M entries. Derivation: M = M_L(l) * M_R(conj(r))
    // where M_L and M_R are the left-/right-quaternion-multiplication 4×4
    // matrices. Expanding and grouping symmetric/antisymmetric combinations
    // of M[i][j] yields these closed-form products.
    let d00 = m[0][0] + m[1][1] + m[2][2] + m[3][3];
    let d11 = m[0][0] + m[1][1] - m[2][2] - m[3][3];
    let d22 = m[0][0] - m[1][1] + m[2][2] - m[3][3];
    let d33 = m[0][0] - m[1][1] - m[2][2] + m[3][3];

    let s01 = m[0][1] - m[1][0];
    let s10 = m[3][2] - m[2][3];
    let s02 = m[0][2] - m[2][0];
    let s20 = m[1][3] - m[3][1];
    let s03 = m[0][3] - m[3][0];
    let s30 = m[2][1] - m[1][2];

    let p01 = m[0][1] + m[1][0];
    let p10 = m[3][2] + m[2][3];
    let p02 = m[0][2] + m[2][0];
    let p20 = m[1][3] + m[3][1];
    let p03 = m[0][3] + m[3][0];
    let p30 = m[2][1] + m[1][2];

    // 4 * A[i][j] = 4 * l_i * r_j
    let a = [
        [d00,          s01 + s10,   s02 + s20,   s03 + s30],
        [s10 - s01,    d11,         p30 - p03,   p02 + p20],
        [s20 - s02,    p30 + p03,   d22,         p10 - p01],
        [s30 - s03,    p20 - p02,   p10 + p01,   d33],
    ];

    let mut best_col = 0;
    let mut best_norm_sq = -1.0;
    for j in 0..4 {
        let n = a[0][j] * a[0][j] + a[1][j] * a[1][j]
              + a[2][j] * a[2][j] + a[3][j] * a[3][j];
        if n > best_norm_sq { best_norm_sq = n; best_col = j; }
    }

    if best_norm_sq < 1e-30 {
        return ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    }

    let scale = best_norm_sq.sqrt();
    let l = [
        a[0][best_col] / scale,
        a[1][best_col] / scale,
        a[2][best_col] / scale,
        a[3][best_col] / scale,
    ];
    let mut r = [0.0f64; 4];
    for j in 0..4 {
        r[j] = (l[0] * a[0][j] + l[1] * a[1][j]
              + l[2] * a[2][j] + l[3] * a[3][j]) * 0.25;
    }
    let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();
    if rn > 1e-300 {
        for i in 0..4 { r[i] /= rn; }
    }
    (l, r)
}

/// Reconstruct the 4×4 rotation matrix from a bi-quaternion `(l, r)`.
/// Inverse of `biquat_extract`.
pub fn biquat_reconstruct(l: &Quat, r: &Quat) -> Mat4 {
    let ml: Mat4 = [
        [l[0], -l[1], -l[2], -l[3]],
        [l[1],  l[0], -l[3],  l[2]],
        [l[2],  l[3],  l[0], -l[1]],
        [l[3], -l[2],  l[1],  l[0]],
    ];
    let mr: Mat4 = [
        [ r[0],  r[1],  r[2],  r[3]],
        [-r[1],  r[0], -r[3],  r[2]],
        [-r[2],  r[3],  r[0], -r[1]],
        [-r[3], -r[2],  r[1],  r[0]],
    ];
    let mut m: Mat4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 { s += ml[i][k] * mr[k][j]; }
            m[i][j] = s;
        }
    }
    m
}

/// Quaternion product `a · b` (Hamilton convention, `[w, x, y, z]`).
fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ]
}

/// Conjugate of a unit quaternion (== inverse for unit quats).
fn quat_conj(q: &Quat) -> Quat { [q[0], -q[1], -q[2], -q[3]] }

/// Imaginary part of `log(q)` for a unit `q`. Returns the rotation
/// vector — direction is the rotation axis, magnitude is the half-angle.
/// Falls back to a Taylor expansion when `q` is near identity, where
/// `acos(w) / sin(acos(w))` is `0/0`.
fn quat_log_imag(q: &Quat) -> LogVec {
    let v_norm_sq = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if v_norm_sq < 1e-12 {
        // Near identity; log(q) ≈ (0, q.xyz). Use linear approximation
        // (sign of w accounts for rare antipodal-near-identity cases —
        // shouldn't happen with chain-aligned input but cheap to guard).
        let s = if q[0] >= 0.0 { 1.0 } else { -1.0 };
        return [s * q[1], s * q[2], s * q[3]];
    }
    let v_norm = v_norm_sq.sqrt();
    let theta = q[0].clamp(-1.0, 1.0).acos();
    let k = theta / v_norm;
    [k * q[1], k * q[2], k * q[3]]
}

/// Inverse of `quat_log_imag`: `exp((0, v))` for a pure-imaginary `v`.
fn quat_exp_imag(v: &LogVec) -> Quat {
    let n_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if n_sq < 1e-12 {
        // Near identity; cos(n) ≈ 1 - n²/2, sin(n)/n ≈ 1 - n²/6. The
        // n² corrections are below f64 precision here, so just lerp.
        let mut q = [1.0, v[0], v[1], v[2]];
        let m = (1.0 + n_sq).sqrt();
        for i in 0..4 { q[i] /= m; }
        return q;
    }
    let n = n_sq.sqrt();
    let s = n.sin() / n;
    [n.cos(), v[0] * s, v[1] * s, v[2] * s]
}

/// Spherical linear interpolation between unit quaternions, honoring
/// the caller's sign choice (no internal flip — flipping one quaternion
/// while leaving its bi-quaternion partner alone would land at a
/// different SO(4) rotation). Used as a building block; for our
/// camera-path interpolation, prefer `eval_hermite` which delivers C¹
/// in wall time by combining slerp arcs with cubic Hermite tangents.
#[allow(dead_code)]
fn slerp(a: &Quat, b: &Quat, t: f64) -> Quat {
    let dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    if dot.abs() > 0.9995 {
        let mut q = [
            a[0] * (1.0 - t) + b[0] * t,
            a[1] * (1.0 - t) + b[1] * t,
            a[2] * (1.0 - t) + b[2] * t,
            a[3] * (1.0 - t) + b[3] * t,
        ];
        let n = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt()
            .max(1e-300);
        for i in 0..4 { q[i] /= n; }
        return q;
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let s = theta.sin();
    let ca = ((1.0 - t) * theta).sin() / s;
    let cb = (t * theta).sin() / s;
    [
        a[0] * ca + b[0] * cb,
        a[1] * ca + b[1] * cb,
        a[2] * ca + b[2] * cb,
        a[3] * ca + b[3] * cb,
    ]
}

/// Extract bi-quaternion representatives for every rotation in the
/// sequence, then chain-align signs so consecutive keyframes share the
/// same S³ × S³ representative. Critical for velocity continuity at
/// segment seams: a keyframe whose extractor happens to return `(l, r)`
/// in one context and `(-l, -r)` after pair-alignment elsewhere
/// reconstructs to the same matrix, but the slerp tangent at the seam
/// points the other way on the sphere — producing a visible corner.
pub fn build_aligned_biquats(rotations: &[Mat4]) -> (Vec<Quat>, Vec<Quat>) {
    let n = rotations.len();
    let mut ls = Vec::with_capacity(n);
    let mut rs = Vec::with_capacity(n);
    for m in rotations {
        let (l, r) = biquat_extract(m);
        ls.push(l);
        rs.push(r);
    }
    for i in 1..n {
        let combined =
            ls[i - 1][0] * ls[i][0] + ls[i - 1][1] * ls[i][1]
          + ls[i - 1][2] * ls[i][2] + ls[i - 1][3] * ls[i][3]
          + rs[i - 1][0] * rs[i][0] + rs[i - 1][1] * rs[i][1]
          + rs[i - 1][2] * rs[i][2] + rs[i - 1][3] * rs[i][3];
        if combined < 0.0 {
            for k in 0..4 {
                ls[i][k] = -ls[i][k];
                rs[i][k] = -rs[i][k];
            }
        }
    }
    (ls, rs)
}

/// Centered-difference body-frame tangent at `q[1]`, scaled by tension
/// and divided by the wall-time gap of the flanking keyframes. Returns
/// a 3-vector (pure-imaginary log) — this is the angular velocity at
/// `q[1]` such that, evaluated in the segment ending or starting at
/// `q[1]`, the wall-time derivative there matches.
fn hermite_tangent(prev: &Quat, curr: &Quat, next: &Quat,
                    t_prev: f64, t_next: f64,
                    tension: f64) -> LogVec {
    // Both logs are in `curr`'s tangent space. Forward log points toward
    // `next`; the negative of the backward log points toward "future"
    // from `prev` through `curr`. Their sum is the "through-curr" direction.
    let curr_inv = quat_conj(curr);
    let fwd = quat_log_imag(&quat_mul(&curr_inv, next));
    let bwd_neg = {
        let l = quat_log_imag(&quat_mul(&curr_inv, prev));
        [-l[0], -l[1], -l[2]]
    };
    let dt = (t_next - t_prev).max(1e-9);
    let k = tension / dt;
    [
        k * (fwd[0] + bwd_neg[0]),
        k * (fwd[1] + bwd_neg[1]),
        k * (fwd[2] + bwd_neg[2]),
    ]
}

/// Cubic Hermite blend in `q1`'s log-space. The four neighbors map to
/// `[i_prev, i_curr, i_next, i_after]` (caller picks via
/// `videorender::pick_neighbors`); `times` and `tensions` are 4-element
/// arrays in the same index order. `u ∈ [0, 1]` is the local parameter
/// inside the segment from `i_curr` to `i_next`. The result is exactly
/// in SO(4).
///
/// At keyframes the body-frame angular velocity matches the tangent
/// computed by `hermite_tangent` on both sides of the seam, so wall-time
/// rotation is C¹ at every join — independently of segment durations.
/// The match at `u = 1` is exact in `q1`'s log-space; in `q2`'s log-space
/// it's only exact up to the nonlinearity of the exponential map, which
/// is below visual threshold for inter-keyframe rotations under ~60°.
pub fn eval_hermite(
    l_path: &[Quat], r_path: &[Quat],
    indices: [usize; 4],
    times: [f64; 4],
    tensions: [f64; 4],
    u: f64,
) -> Mat4 {
    let l0 = &l_path[indices[0]]; let l1 = &l_path[indices[1]];
    let l2 = &l_path[indices[2]]; let l3 = &l_path[indices[3]];
    let r0 = &r_path[indices[0]]; let r1 = &r_path[indices[1]];
    let r2 = &r_path[indices[2]]; let r3 = &r_path[indices[3]];

    let v_l_curr = hermite_tangent(l0, l1, l2, times[0], times[2], tensions[1]);
    let v_l_next = hermite_tangent(l1, l2, l3, times[1], times[3], tensions[2]);
    let v_r_curr = hermite_tangent(r0, r1, r2, times[0], times[2], tensions[1]);
    let v_r_next = hermite_tangent(r1, r2, r3, times[1], times[3], tensions[2]);

    let h = (times[2] - times[1]).max(1e-9);
    // Cubic Hermite basis on [0, 1].
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 =  2.0 * u3 - 3.0 * u2 + 1.0;  // unused (multiplies 0)
    let _ = h00;
    let h10 =        u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 =        u3 -       u2;

    // Anchored at l1: cubic in log-space with f(0)=0, f(1)=ω_f, f'(0)=v_curr*h, f'(1)=v_next*h.
    let omega_f_l = quat_log_imag(&quat_mul(&quat_conj(l1), l2));
    let omega_f_r = quat_log_imag(&quat_mul(&quat_conj(r1), r2));
    let phi_l: LogVec = [
        h10 * v_l_curr[0] * h + h01 * omega_f_l[0] + h11 * v_l_next[0] * h,
        h10 * v_l_curr[1] * h + h01 * omega_f_l[1] + h11 * v_l_next[1] * h,
        h10 * v_l_curr[2] * h + h01 * omega_f_l[2] + h11 * v_l_next[2] * h,
    ];
    let phi_r: LogVec = [
        h10 * v_r_curr[0] * h + h01 * omega_f_r[0] + h11 * v_r_next[0] * h,
        h10 * v_r_curr[1] * h + h01 * omega_f_r[1] + h11 * v_r_next[1] * h,
        h10 * v_r_curr[2] * h + h01 * omega_f_r[2] + h11 * v_r_next[2] * h,
    ];

    let l = quat_mul(l1, &quat_exp_imag(&phi_l));
    let r = quat_mul(r1, &quat_exp_imag(&phi_r));
    biquat_reconstruct(&l, &r)
}
