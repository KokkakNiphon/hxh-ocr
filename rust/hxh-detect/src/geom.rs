//! Geometry: convex hull, minimum-area rectangle (rotating calipers), unclip, quads.
//!
//! Coordinates are `[x, y]` in pixels, y down, with pixel centres at integers (pixel `i`
//! covers `[i - 0.5, i + 0.5]`), as in `rust/hxh-synth`.

use crate::math::sqrt;

/// A point `[x, y]`.
pub type P = [f64; 2];

#[inline]
fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// `(a - o) x (b - o)`: positive when `o -> a -> b` turns counter-clockwise in a y-up frame
/// (clockwise on screen, y down).
#[inline]
pub fn cross(o: P, a: P, b: P) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

#[inline]
pub fn dist(a: P, b: P) -> f64 {
    let d = sub(a, b);
    sqrt(dot(d, d))
}

/// Convex hull (Andrew's monotone chain). Sorts `pts` in place and writes the hull vertices
/// to `hull` (capacity `pts.len() + 1`) with positive orientation (`cross > 0` at every
/// vertex), without repeated or collinear vertices. Returns the vertex count (< 3 only for
/// degenerate input).
pub fn convex_hull(pts: &mut [P], hull: &mut [P]) -> usize {
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    chain(pts, hull, 1.0)
}

/// [`convex_hull`] of points already sorted by `(y, x)` (no sort in the binary): the chain
/// runs in the axis-swapped frame, then the result is reversed to positive orientation.
pub fn convex_hull_sorted_yx(pts: &[P], hull: &mut [P]) -> usize {
    let m = chain(pts, hull, -1.0);
    hull[..m].reverse();
    m
}

/// Monotone chain over lexicographically sorted points; `sign` = -1 when the sort order is
/// `(y, x)` (a reflection of the plane, which flips every cross product).
fn chain(pts: &[P], hull: &mut [P], sign: f64) -> usize {
    let n = pts.len();
    assert!(hull.len() > n, "hull needs pts.len() + 1 entries");
    if n < 3 {
        hull[..n].copy_from_slice(pts);
        return if n == 2 && pts[0] == pts[1] { 1 } else { n };
    }
    let mut k = 0;
    for &p in pts.iter() {
        while k >= 2 && sign * cross(hull[k - 2], hull[k - 1], p) <= 0.0 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }
    let lower = k + 1;
    for &p in pts.iter().rev().skip(1) {
        while k >= lower && sign * cross(hull[k - 2], hull[k - 1], p) <= 0.0 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }
    k - 1 // the last point repeats the first
}

/// A rotated rectangle: centre `c`, unit axis `u`, extent `w` along `u` and `h` along
/// `n = (-u.y, u.x)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotRect {
    pub c: P,
    pub u: P,
    pub w: f64,
    pub h: f64,
}

impl RotRect {
    #[inline]
    pub fn n(&self) -> P {
        [-self.u[1], self.u[0]]
    }

    #[inline]
    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// Corners `c -+ u w/2 -+ n h/2` in the order (-,-), (+,-), (+,+), (-,+).
    pub fn corners(&self) -> [P; 4] {
        let (u, n) = (self.u, self.n());
        let (a, b) = (self.w / 2.0, self.h / 2.0);
        let at = |s: f64, t: f64| [self.c[0] + u[0] * s + n[0] * t, self.c[1] + u[1] * s + n[1] * t];
        [at(-a, -b), at(a, -b), at(a, b), at(-a, b)]
    }
}

/// Rectangle aligned with hull edge `i` given the extreme vertices along the edge direction
/// (`jmin`, `jmax`) and along its inward normal (`jfar`).
#[inline]
fn edge_rect(hull: &[P], i: usize, jmin: usize, jmax: usize, jfar: usize) -> RotRect {
    let m = hull.len();
    let a = hull[i];
    let e = sub(hull[(i + 1) % m], a);
    let len = sqrt(dot(e, e));
    let u = [e[0] / len, e[1] / len];
    let n = [-u[1], u[0]];
    let (lo, hi) = (dot(sub(hull[jmin], a), u), dot(sub(hull[jmax], a), u));
    let h = dot(sub(hull[jfar], a), n);
    let (s, t) = ((lo + hi) / 2.0, h / 2.0);
    RotRect { c: [a[0] + u[0] * s + n[0] * t, a[1] + u[1] * s + n[1] * t], u, w: hi - lo, h }
}

/// Minimum-area enclosing rectangle of a convex polygon (output of [`convex_hull`]) by
/// rotating calipers: one side of the optimum lies on a hull edge (Freeman & Shapira), and
/// the three other extreme vertices only move forward as the edge advances, so the whole
/// scan is O(m). Ties keep the first edge. Degenerate hulls (< 3 vertices) give a
/// zero-height rectangle along the segment (or a point).
pub fn min_area_rect(hull: &[P]) -> RotRect {
    let m = hull.len();
    match m {
        0 => return RotRect { c: [0.0, 0.0], u: [1.0, 0.0], w: 0.0, h: 0.0 },
        1 => return RotRect { c: hull[0], u: [1.0, 0.0], w: 0.0, h: 0.0 },
        2 => {
            let len = dist(hull[0], hull[1]);
            if len == 0.0 {
                return RotRect { c: hull[0], u: [1.0, 0.0], w: 0.0, h: 0.0 };
            }
            let e = sub(hull[1], hull[0]);
            let c = [(hull[0][0] + hull[1][0]) / 2.0, (hull[0][1] + hull[1][1]) / 2.0];
            return RotRect { c, u: [e[0] / len, e[1] / len], w: len, h: 0.0 };
        }
        _ => {}
    }
    let nxt = |j: usize| (j + 1) % m;
    // extreme vertices for edge 0 by a full scan
    let e0 = sub(hull[1], hull[0]);
    let n0 = [-e0[1], e0[0]];
    let (mut jmin, mut jmax, mut jfar) = (0, 0, 0);
    for j in 1..m {
        if dot(hull[j], e0) < dot(hull[jmin], e0) {
            jmin = j;
        }
        if dot(hull[j], e0) > dot(hull[jmax], e0) {
            jmax = j;
        }
        if dot(hull[j], n0) > dot(hull[jfar], n0) {
            jfar = j;
        }
    }
    let mut best = edge_rect(hull, 0, jmin, jmax, jfar);
    for i in 1..m {
        let e = sub(hull[nxt(i)], hull[i]);
        let n = [-e[1], e[0]];
        // the projections are unimodal around a convex polygon: advance while they improve
        // (an m-step cap guards against float ties on degenerate input)
        for _ in 0..m {
            if dot(sub(hull[nxt(jmax)], hull[jmax]), e) > 0.0 {
                jmax = nxt(jmax);
            } else {
                break;
            }
        }
        for _ in 0..m {
            if dot(sub(hull[nxt(jfar)], hull[jfar]), n) > 0.0 {
                jfar = nxt(jfar);
            } else {
                break;
            }
        }
        for _ in 0..m {
            if dot(sub(hull[nxt(jmin)], hull[jmin]), e) < 0.0 {
                jmin = nxt(jmin);
            } else {
                break;
            }
        }
        let r = edge_rect(hull, i, jmin, jmax, jfar);
        if r.area() < best.area() {
            best = r;
        }
    }
    best
}

/// Unclip a (kernel) rectangle: grow every side by `D`.
///
/// * `shrink_ratio <= 0`: DBNet's rule `D = area * unclip_ratio / perimeter`.
/// * `shrink_ratio = r > 0`: the exact inverse of the training shrink of a rectangle
///   (`D = A (1 - r^2) / L` of the *original* box, as in `hxh-synth` targets): the positive
///   root of `(8 - 4a) D^2 + 2 (1 - a)(w + h) D - a w h = 0` with `a = 1 - r^2`, so that
///   shrinking the result again gives back `w x h`.
pub fn unclip(r: &RotRect, unclip_ratio: f64, shrink_ratio: f64) -> RotRect {
    let (w, h) = (r.w, r.h);
    let d = if shrink_ratio > 0.0 {
        let a = 1.0 - shrink_ratio * shrink_ratio;
        let (s, b) = (w + h, 1.0 - a);
        (-b * s + sqrt(b * b * s * s + (8.0 - 4.0 * a) * a * w * h)) / (8.0 - 4.0 * a)
    } else {
        let per = 2.0 * (w + h);
        if per > 0.0 { w * h * unclip_ratio / per } else { 0.0 }
    };
    RotRect { c: r.c, u: r.u, w: w + 2.0 * d, h: h + 2.0 * d }
}

/// The corners of `r` clockwise (on screen) from the top-left in reading orientation, with
/// the extents along and across the reading direction: `(quad, length, thickness)`.
///
/// The reading axis is the rectangle axis closest to +x for a horizontal line and to +y
/// (downwards) for a vertical one, so a single glyph keeps its orientation even when its box
/// is taller than wide. Horizontal: TL, TR, BR, BL. Vertical: the column's TR, BR, BL, TL,
/// i.e. the TL, TR, BR, BL of the strip after turning the column 90 degrees counter-clockwise
/// (the `rust/hxh-synth` convention). In both cases `q0 -> q1` is the reading direction.
pub fn rect_to_quad(r: &RotRect, vertical: bool) -> ([P; 4], f64, f64) {
    let (u, n) = (r.u, r.n());
    let cands = [(u, r.w, r.h), (n, r.h, r.w), ([-u[0], -u[1]], r.w, r.h), ([-n[0], -n[1]], r.h, r.w)];
    let k = if vertical { 1 } else { 0 };
    let mut best = 0;
    for i in 1..4 {
        if cands[i].0[k] > cands[best].0[k] {
            best = i;
        }
    }
    let (a, len, thick) = cands[best];
    let na = [-a[1], a[0]];
    let at = |s: f64, t: f64| [r.c[0] + a[0] * s + na[0] * t, r.c[1] + a[1] * s + na[1] * t];
    let (l2, t2) = (len / 2.0, thick / 2.0);
    ([at(-l2, -t2), at(l2, -t2), at(l2, t2), at(-l2, t2)], len, thick)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O(m^2) reference: the best rectangle over all hull edges.
    fn min_area_rect_brute(hull: &[P]) -> RotRect {
        let m = hull.len();
        let mut best: Option<RotRect> = None;
        for i in 0..m {
            let e = sub(hull[(i + 1) % m], hull[i]);
            let n = [-e[1], e[0]];
            let (mut jmin, mut jmax, mut jfar) = (0, 0, 0);
            for j in 0..m {
                if dot(hull[j], e) < dot(hull[jmin], e) {
                    jmin = j;
                }
                if dot(hull[j], e) > dot(hull[jmax], e) {
                    jmax = j;
                }
                if dot(hull[j], n) > dot(hull[jfar], n) {
                    jfar = j;
                }
            }
            let r = edge_rect(hull, i, jmin, jmax, jfar);
            if best.is_none_or(|b| r.area() < b.area()) {
                best = Some(r);
            }
        }
        best.unwrap()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn hull_of(pts: &[P]) -> Vec<P> {
        let mut p = pts.to_vec();
        let mut h = vec![[0.0; 2]; p.len() + 1];
        let m = convex_hull(&mut p, &mut h);
        h.truncate(m);
        h
    }

    #[test]
    fn hull_removes_interior_duplicate_and_collinear_points() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0], [0.0, 1.0]];
        let h = hull_of(&pts);
        assert_eq!(h.len(), 4);
        for i in 0..4 {
            assert!(cross(h[i], h[(i + 1) % 4], h[(i + 2) % 4]) > 0.0);
        }
    }

    #[test]
    fn hull_of_yx_sorted_points_equals_the_sorting_hull() {
        let mut rng = Rng(42);
        for _ in 0..500 {
            let mut pts: Vec<P> = (0..3 + (rng.next() * 40.0) as usize)
                .map(|_| [(rng.next() * 20.0).floor() + 0.5, (rng.next() * 20.0).floor() - 0.5])
                .collect();
            pts.sort_by(|a, b| a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0])));
            let mut h = vec![[0.0; 2]; pts.len() + 1];
            let m = convex_hull_sorted_yx(&pts, &mut h);
            let want = hull_of(&pts);
            assert_eq!(m, want.len());
            if m < 3 {
                continue;
            }
            // the same cyclic sequence with the same (positive) orientation
            let s = want.iter().position(|p| *p == h[0]).unwrap();
            for i in 0..m {
                assert_eq!(h[i], want[(s + i) % m]);
            }
        }
    }

    #[test]
    fn min_area_rect_of_axis_aligned_and_rotated_rectangles() {
        // axis-aligned 10 x 4
        let r = min_area_rect(&hull_of(&[[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0], [3.0, 2.0]]));
        assert!((r.area() - 40.0).abs() < 1e-12 && (r.c[0] - 5.0).abs() < 1e-12 && (r.c[1] - 2.0).abs() < 1e-12);
        // rotated by 30 degrees, 20 x 6 around (50, 40), plus interior points
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let rect = RotRect { c: [50.0, 40.0], u: [c, s], w: 20.0, h: 6.0 };
        let mut pts = rect.corners().to_vec();
        pts.push([50.0, 40.0]);
        pts.push([52.0, 41.0]);
        let r = min_area_rect(&hull_of(&pts));
        assert!((r.area() - 120.0).abs() < 1e-9, "{r:?}");
        assert!((r.c[0] - 50.0).abs() < 1e-9 && (r.c[1] - 40.0).abs() < 1e-9);
        assert!((r.w.max(r.h) - 20.0).abs() < 1e-9 && (r.w.min(r.h) - 6.0).abs() < 1e-9);
        // a right triangle: the optimum lies on a leg or the hypotenuse, area = legs product
        let r = min_area_rect(&hull_of(&[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]));
        assert!((r.area() - 12.0).abs() < 1e-9, "{r:?}");
        // degenerate inputs
        assert_eq!(min_area_rect(&hull_of(&[[1.0, 1.0], [1.0, 1.0]])).area(), 0.0);
        assert_eq!(min_area_rect(&hull_of(&[[0.0, 0.0], [3.0, 4.0]])).w, 5.0);
    }

    #[test]
    fn rotating_calipers_equals_brute_force() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..3000 {
            let n = 3 + (rng.next() * 60.0) as usize;
            let pts: Vec<P> = if case % 2 == 0 {
                // random points in a random ellipse
                let (a, b, t) = (1.0 + rng.next() * 50.0, 1.0 + rng.next() * 10.0, rng.next() * 6.3);
                (0..n)
                    .map(|_| {
                        let (r, ph) = (rng.next().sqrt(), rng.next() * 6.3);
                        let (x, y) = (a * r * ph.cos(), b * r * ph.sin());
                        [x * t.cos() - y * t.sin(), x * t.sin() + y * t.cos()]
                    })
                    .collect()
            } else {
                // pixel-square corners of a random blob on a grid (many exact ties)
                let mut v = Vec::new();
                for y in 0..(1 + (rng.next() * 8.0) as i32) {
                    let x0 = (rng.next() * 6.0) as i32;
                    let x1 = x0 + (rng.next() * 12.0) as i32;
                    for (x, d) in [(x0 as f64 - 0.5, 0.0), (x1 as f64 + 0.5, 0.0)] {
                        v.push([x + d, y as f64 - 0.5]);
                        v.push([x + d, y as f64 + 0.5]);
                    }
                }
                v
            };
            let h = hull_of(&pts);
            if h.len() < 3 {
                continue;
            }
            let (a, b) = (min_area_rect(&h), min_area_rect_brute(&h));
            assert!((a.area() - b.area()).abs() <= 1e-9 * b.area().max(1.0), "case {case}: {a:?} vs {b:?}");
            // every point inside the rectangle (with a little slack)
            for p in &pts {
                let d = sub(*p, a.c);
                assert!(dot(d, a.u).abs() <= a.w / 2.0 + 1e-9 && dot(d, a.n()).abs() <= a.h / 2.0 + 1e-9);
            }
        }
    }

    #[test]
    fn unclip_ratio_and_exact_inverse_of_the_shrink() {
        let r = RotRect { c: [0.0, 0.0], u: [1.0, 0.0], w: 10.0, h: 2.0 };
        // DBNet: D = 20 * 1.5 / 24 = 1.25
        let g = unclip(&r, 1.5, 0.0);
        assert!((g.w - 12.5).abs() < 1e-12 && (g.h - 4.5).abs() < 1e-12);
        // exact inverse: shrink (hxh-synth rule) the result by its own D and get 10 x 2 back
        for &ratio in &[0.4, 0.6, 0.7] {
            let g = unclip(&r, 0.0, ratio);
            let d = g.w * g.h * (1.0 - ratio * ratio) / (2.0 * (g.w + g.h));
            assert!((g.w - 2.0 * d - 10.0).abs() < 1e-9 && (g.h - 2.0 * d - 2.0).abs() < 1e-9, "r {ratio}: {g:?}");
        }
        // the numbers from the journal: a 100 x 10 line shrunk with r = 0.4 -> 92.36 x 2.36
        let k = RotRect { c: [0.0, 0.0], u: [1.0, 0.0], w: 100.0 - 2.0 * 3.818_181_818, h: 10.0 - 2.0 * 3.818_181_818 };
        let g = unclip(&k, 0.0, 0.4);
        assert!((g.w - 100.0).abs() < 1e-6 && (g.h - 10.0).abs() < 1e-6, "{g:?}");
    }

    #[test]
    fn quads_are_clockwise_from_the_reading_top_left() {
        // horizontal, slightly rotated, axis given as the short side: still read along x
        let r = RotRect { c: [50.0, 20.0], u: [0.1f64.sin(), -0.1f64.cos()], w: 8.0, h: 40.0 };
        let (q, len, thick) = rect_to_quad(&r, false);
        assert!((len - 40.0).abs() < 1e-9 && (thick - 8.0).abs() < 1e-9);
        assert!(q[0][0] < q[1][0] && q[3][1] > q[0][1], "{q:?}");
        // vertical column: the first corner is the top-right, then down the column
        let r = RotRect { c: [50.0, 50.0], u: [1.0, 0.0], w: 10.0, h: 60.0 };
        let (q, len, thick) = rect_to_quad(&r, true);
        assert!((len - 60.0).abs() < 1e-9 && (thick - 10.0).abs() < 1e-9);
        assert_eq!(q, [[55.0, 20.0], [55.0, 80.0], [45.0, 80.0], [45.0, 20.0]]);
        // a single square glyph: horizontal reads along +x, vertical along +y
        let r = RotRect { c: [0.0, 0.0], u: [0.0, 1.0], w: 6.0, h: 6.0 };
        assert_eq!(rect_to_quad(&r, false).0, [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]]);
        assert_eq!(rect_to_quad(&r, true).0, [[3.0, -3.0], [3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0]]);
        // clockwise on screen (y down) = positive shoelace sum
        for v in [false, true] {
            let (q, _, _) = rect_to_quad(&RotRect { c: [5.0, 5.0], u: [0.6, 0.8], w: 3.0, h: 7.0 }, v);
            let a2: f64 = (0..4).map(|i| q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]).sum();
            assert!(a2 > 0.0);
        }
    }
}
