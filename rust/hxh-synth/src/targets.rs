//! DBNet-style training targets from line quads.
//!
//! Exact rule (reproducible with numpy + OpenCV in the trainer, after the quads have been
//! scaled / cropped / rotated together with the image; pixel centres at integers):
//!
//! * A = |shoelace area| of the quad, L = its perimeter, D = A (1 - r^2) / L with r = 0.4
//!   (DBNet's shrink ratio).
//! * Kernel: move every edge of the (convex) quad inward by D and intersect neighbouring moved
//!   edges (miter). For convex quads this is exactly the pyclipper/Vatti inward offset.
//!   The line is *ignored* if the kernel collapses (some edge flips direction) or if the
//!   quad is thinner than `min_size` px (thickness = mean length of edges q0-q3 and q1-q2).
//! * `prob` = 1 on pixels whose centre lies inside (or on) a kernel: for a clockwise polygon,
//!   cross(b - a, p - a) >= 0 for every edge a -> b. A numpy half-plane test reproduces this
//!   exactly; cv2.fillPoly fills a slightly fatter superset (about 0.5 px per side).
//! * `prob_mask` = 0 on pixels inside ignored quads (same centre rule), else 1 (the prob /
//!   binary loss mask).
//! * `thresh`: for each quad, over its bounding box grown by D, dist = distance from the pixel
//!   centre to the nearest quad edge segment (inside or outside); canvas = max over quads of
//!   1 - clip(dist / D, 0, 1); thresh = 0.3 + 0.4 * canvas.
//! * `thresh_mask` = 1 where some quad has dist <= D or contains the pixel (the quad dilated
//!   by D with round joins), else 0.
//! * `vert` = 1 on the kernels of vertical lines (optional orientation target).

use crate::geom::{area2, bbox, line_thickness, perimeter, P};
use crate::raster::{fill_convex_u8, IRect};

pub struct Targets {
    pub w: usize,
    pub h: usize,
    pub prob: Vec<f32>,
    pub prob_mask: Vec<f32>,
    pub thresh: Vec<f32>,
    pub thresh_mask: Vec<f32>,
    pub vert: Vec<f32>,
}

pub struct TargetParams {
    pub shrink: f64,
    pub thresh_min: f32,
    pub thresh_max: f32,
    pub min_size: f64,
}

impl Default for TargetParams {
    fn default() -> Self {
        TargetParams { shrink: 0.4, thresh_min: 0.3, thresh_max: 0.7, min_size: 3.0 }
    }
}

/// DBNet shrink / dilation distance D = A (1 - r^2) / L.
pub fn offset_distance(q: &[P], shrink: f64) -> f64 {
    let l = perimeter(q);
    if l <= 1e-9 { 0.0 } else { area2(q).abs() * 0.5 * (1.0 - shrink * shrink) / l }
}

/// Move each edge of a convex polygon inward by d (outward if d < 0), intersecting neighbouring
/// edges. None if it collapses.
pub fn shrink_quad(q: &[P], d: f64) -> Option<Vec<P>> {
    let n = q.len();
    let a2 = area2(q);
    if a2.abs() < 1e-9 {
        return None;
    }
    let sgn = a2.signum();
    let e: Vec<P> = (0..n).map(|i| [q[(i + 1) % n][0] - q[i][0], q[(i + 1) % n][1] - q[i][1]]).collect();
    let mut p: Vec<P> = Vec::with_capacity(n);
    for i in 0..n {
        let l = e[i][0].hypot(e[i][1]);
        if l < 1e-9 {
            return None;
        }
        let inward = [sgn * -e[i][1] / l, sgn * e[i][0] / l];
        p.push([q[i][0] + inward[0] * d, q[i][1] + inward[1] * d]);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + n - 1) % n;
        let det = e[j][0] * (-e[i][1]) - (-e[i][0]) * e[j][1];
        if det.abs() < 1e-12 {
            out.push(p[i]);
            continue;
        }
        let r = [p[i][0] - p[j][0], p[i][1] - p[j][1]];
        let t = (r[0] * (-e[i][1]) - (-e[i][0]) * r[1]) / det;
        out.push([p[j][0] + t * e[j][0], p[j][1] + t * e[j][1]]);
    }
    if d > 0.0 {
        for i in 0..n {
            let e2 = [out[(i + 1) % n][0] - out[i][0], out[(i + 1) % n][1] - out[i][1]];
            if e2[0] * e[i][0] + e2[1] * e[i][1] <= 0.0 {
                return None;
            }
        }
    }
    Some(out)
}

fn seg_dist(px: f64, py: f64, a: P, b: P) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-9);
    let t = (((px - a[0]) * d[0] + (py - a[1]) * d[1]) / l2).clamp(0.0, 1.0);
    (px - (a[0] + t * d[0])).hypot(py - (a[1] + t * d[1]))
}

/// `lines`: (quad, vertical) in target pixel coordinates.
pub fn rasterize_targets(lines: &[([P; 4], bool)], w: usize, h: usize, prm: &TargetParams) -> Targets {
    let full = IRect { x0: 0, y0: 0, x1: w as i32, y1: h as i32 };
    let mut prob = vec![0u8; w * h];
    let mut ignore = vec![0u8; w * h];
    let mut vert = vec![0u8; w * h];
    let mut canvas = vec![0.0f32; w * h];
    let mut tmask = vec![0u8; w * h];
    for (q, vertical) in lines {
        let d = offset_distance(q, prm.shrink);
        if d <= 1e-6 {
            continue;
        }
        let kernel = if line_thickness(q) >= prm.min_size { shrink_quad(q, d) } else { None };
        match kernel {
            Some(k) => {
                fill_convex_u8(&mut prob, full, &k, 1);
                if *vertical {
                    fill_convex_u8(&mut vert, full, &k, 1);
                }
            }
            None => fill_convex_u8(&mut ignore, full, q, 1),
        }
        let b = bbox(q);
        let r = IRect { x0: (b[0] - d).floor() as i32, y0: (b[1] - d).floor() as i32, x1: (b[2] + d).ceil() as i32 + 1, y1: (b[3] + d).ceil() as i32 + 1 }
            .intersect(full);
        if r.is_empty() {
            continue;
        }
        let mut inside = vec![0u8; r.w() * r.h()];
        fill_convex_u8(&mut inside, r, q, 1);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let (px, py) = (x as f64, y as f64);
                let dd = (0..4).map(|i| seg_dist(px, py, q[i], q[(i + 1) % 4])).fold(f64::INFINITY, f64::min);
                let i = y as usize * w + x as usize;
                let v = (1.0 - (dd / d).clamp(0.0, 1.0)) as f32;
                if v > canvas[i] {
                    canvas[i] = v;
                }
                if dd <= d || inside[(y - r.y0) as usize * r.w() + (x - r.x0) as usize] != 0 {
                    tmask[i] = 1;
                }
            }
        }
    }
    let span = prm.thresh_max - prm.thresh_min;
    Targets {
        w,
        h,
        prob: prob.iter().map(|&v| v as f32).collect(),
        prob_mask: ignore.iter().map(|&v| 1.0 - v as f32).collect(),
        thresh: canvas.iter().map(|&c| prm.thresh_min + span * c).collect(),
        thresh_mask: tmask.iter().map(|&v| v as f32).collect(),
        vert: vert.iter().map(|&v| v as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_rectangle() {
        // 100 x 20 box: A = 2000, L = 240, D = 2000 * 0.84 / 240 = 7
        let q = [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]];
        let d = offset_distance(&q, 0.4);
        assert!((d - 7.0).abs() < 1e-9);
        let k = shrink_quad(&q, d).unwrap();
        assert!((k[0][0] - 7.0).abs() < 1e-9 && (k[0][1] - 7.0).abs() < 1e-9);
        assert!((k[2][0] - 93.0).abs() < 1e-9 && (k[2][1] - 13.0).abs() < 1e-9);
        assert!(shrink_quad(&q, 10.5).is_none());
        let t = rasterize_targets(&[(q, false)], 120, 40, &TargetParams::default());
        assert_eq!(t.prob[10 * 120 + 50], 1.0);
        assert_eq!(t.prob[3 * 120 + 50], 0.0);
        assert!((t.thresh[50] - 0.7).abs() < 1e-6); // on the top edge
        assert_eq!(t.thresh_mask[(20 + 7) * 120 + 50], 1.0);
        assert_eq!(t.thresh_mask[(20 + 8) * 120 + 50], 0.0);
    }
}
