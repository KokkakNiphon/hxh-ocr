//! Points, 3x3 homographies and polygon helpers. Image coordinates are pixels with pixel
//! centres at integers (the OpenCV convention), y pointing down.

use crate::rng::Rng;

pub type P = [f64; 2];
pub type H3 = [[f64; 3]; 3];

pub const I3: H3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mul(a: &H3, b: &H3) -> H3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub fn tr(dx: f64, dy: f64) -> H3 {
    [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
}

pub fn rot(deg: f64) -> H3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

pub fn apply(m: &H3, p: P) -> P {
    let w = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
    [(m[0][0] * p[0] + m[0][1] * p[1] + m[0][2]) / w, (m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]) / w]
}

pub fn apply_all(m: &H3, pts: &[P]) -> Vec<P> {
    pts.iter().map(|&p| apply(m, p)).collect()
}

/// Local isotropic scale (sqrt |det J|) and rotation (degrees) of `m` at `p`.
pub fn local_frame(m: &H3, p: P) -> (f64, f64) {
    let a = apply(m, p);
    let bx = apply(m, [p[0] + 1.0, p[1]]);
    let by = apply(m, [p[0], p[1] + 1.0]);
    let j = [[bx[0] - a[0], by[0] - a[0]], [bx[1] - a[1], by[1] - a[1]]];
    let det = (j[0][0] * j[1][1] - j[0][1] * j[1][0]).abs();
    (det.sqrt(), j[1][0].atan2(j[0][0]).to_degrees())
}

/// Homography mapping the 4 `src` points onto `dst` (h33 = 1), by Gaussian elimination.
pub fn homography(src: &[P; 4], dst: &[P; 4]) -> H3 {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for c in 0..8 {
        let piv = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, piv);
        let d = a[c][c];
        if d.abs() < 1e-12 {
            return I3;
        }
        for k in c..9 {
            a[c][k] /= d;
        }
        for r in 0..8 {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    for k in c..9 {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8]).collect();
    [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
}

pub fn bbox(pts: &[P]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in pts {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

pub fn ellipse(cx: f64, cy: f64, a: f64, b: f64, n: usize) -> Vec<P> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            [cx + a * t.cos(), cy + b * t.sin()]
        })
        .collect()
}

pub fn round_rect(x0: f64, y0: f64, x1: f64, y1: f64, r: f64, n: usize) -> Vec<P> {
    let r = r.min((x1 - x0) / 2.0).min((y1 - y0) / 2.0);
    if r < 0.75 {
        return vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    }
    let mut pts = Vec::with_capacity(4 * n);
    for (cx, cy, a0) in [(x1 - r, y0 + r, -90.0), (x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0)] {
        for i in 0..n {
            let t = (a0 + 90.0 * i as f64 / (n - 1) as f64).to_radians();
            pts.push([cx + r * t.cos(), cy + r * t.sin()]);
        }
    }
    pts
}

pub fn bezier(c: &[P; 4], n: usize) -> Vec<P> {
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let (a, b, cc, d) = ((1.0 - t).powi(3), 3.0 * (1.0 - t).powi(2) * t, 3.0 * (1.0 - t) * t * t, t.powi(3));
            [a * c[0][0] + b * c[1][0] + cc * c[2][0] + d * c[3][0], a * c[0][1] + b * c[1][1] + cc * c[2][1] + d * c[3][1]]
        })
        .collect()
}

/// Smooth random closed shape around (cx, cy).
pub fn blob(rng: &mut Rng, cx: f64, cy: f64, r: f64, n: usize) -> Vec<P> {
    let mut amp = [0.0; 3];
    let mut ph = [0.0; 3];
    for i in 0..3 {
        amp[i] = rng.uniform(0.0, 0.25);
        ph[i] = rng.uniform(0.0, std::f64::consts::TAU);
    }
    let asp = rng.uniform(0.5, 1.5);
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            let rr = r * (1.0 + amp[0] * (2.0 * t + ph[0]).sin() + amp[1] * (3.0 * t + ph[1]).sin() + amp[2] * (5.0 * t + ph[2]).sin());
            [cx + rr * t.cos() * asp, cy + rr * t.sin() / asp]
        })
        .collect()
}

/// Corners of box `b` = (x0, y0, x1, y1) grown by `grow`, clockwise from the top-left in reading
/// orientation: horizontal TL, TR, BR, BL; vertical TR, BR, BL, TL (the box seen after turning
/// the column 90 degrees counter-clockwise).
pub fn box_quad(b: [f64; 4], vertical: bool, grow: f64) -> [P; 4] {
    let (x0, y0, x1, y1) = (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow);
    if vertical {
        [[x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    } else {
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }
}

pub fn dist(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Mean length of the two cross edges (q0-q3, q1-q2): line height ("h") or width ("v").
pub fn line_thickness(q: &[P]) -> f64 {
    0.5 * (dist(q[3], q[0]) + dist(q[2], q[1]))
}

/// Twice the signed area (shoelace); positive for clockwise polygons in y-down coordinates.
pub fn area2(q: &[P]) -> f64 {
    let n = q.len();
    (0..n).map(|i| q[i][0] * q[(i + 1) % n][1] - q[(i + 1) % n][0] * q[i][1]).sum()
}

pub fn perimeter(q: &[P]) -> f64 {
    let n = q.len();
    (0..n).map(|i| dist(q[i], q[(i + 1) % n])).sum()
}

fn seg_point_dist(p: P, a: P, b: P) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 1e-12 { (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p[0] - (a[0] + t * d[0])).hypot(p[1] - (a[1] + t * d[1]))
}

/// Separating-axis test for two convex polygons (touching counts as overlapping).
fn convex_overlap(a: &[P], b: &[P]) -> bool {
    for poly in [a, b] {
        let n = poly.len();
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let ax = [q[1] - p[1], p[0] - q[0]];
            if ax[0] == 0.0 && ax[1] == 0.0 {
                continue;
            }
            let proj = |s: &[P]| s.iter().map(|v| v[0] * ax[0] + v[1] * ax[1]).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)));
            let (a0, a1) = proj(a);
            let (b0, b1) = proj(b);
            if a1 < b0 || b1 < a0 {
                return false;
            }
        }
    }
    true
}

/// Euclidean distance between two convex polygons; 0 if they overlap or touch.
pub fn convex_dist(a: &[P], b: &[P]) -> f64 {
    if convex_overlap(a, b) {
        return 0.0;
    }
    let mut d = f64::INFINITY;
    for i in 0..a.len() {
        for j in 0..b.len() {
            d = d.min(seg_point_dist(a[i], b[j], b[(j + 1) % b.len()]));
            d = d.min(seg_point_dist(b[j], a[i], a[(i + 1) % a.len()]));
        }
    }
    d
}

/// Gap between two axis-aligned boxes (0 if they overlap).
pub fn bbox_gap(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = (b[0] - a[2]).max(a[0] - b[2]).max(0.0);
    let dy = (b[1] - a[3]).max(a[1] - b[3]).max(0.0);
    dx.hypot(dy)
}
