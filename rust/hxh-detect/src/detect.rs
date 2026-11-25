//! From the detector's int8 maps to text lines: threshold, connected components,
//! minimum-area rectangles, unclip, mapping to the original image, reading order.

use crate::geom::{P, RotRect, convex_hull_sorted_yx, dist, min_area_rect, rect_to_quad, unclip};
use crate::math::{pow2, sigmoid};

/// Post-processing parameters (the defaults are DBNet's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Text-kernel probability threshold: a pixel is text when `sigmoid(logit) > bin_thresh`,
    /// evaluated as an integer comparison on the int8 logit (see [`kernel_threshold`]).
    pub bin_thresh: f64,
    /// Drop lines whose mean kernel probability over the component is below this.
    pub min_score: f64,
    /// Drop lines whose final (unclipped) box has a short side below this, in letterbox px.
    pub min_side: f64,
    /// DBNet unclip: grow the kernel rectangle by `D = area * unclip_ratio / perimeter`.
    pub unclip_ratio: f64,
    /// If > 0, replaces `unclip_ratio` by the exact inverse of the training shrink
    /// `D = A (1 - r^2) / L` with `r = shrink_ratio` (see [`crate::geom::unclip`]).
    pub shrink_ratio: f64,
    /// Reading order: two lines of the same orientation join one block when the gap between
    /// their axis-aligned boxes is at most `block_gap` x the thicker line's thickness in both
    /// x and y.
    pub block_gap: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { bin_thresh: 0.3, min_score: 0.5, min_side: 3.0, unclip_ratio: 1.5, shrink_ratio: 0.0, block_gap: 1.0 }
    }
}

/// The detector output: int8 NHWC `h x w x c` at exponent `exp` (real = q * 2^exp).
/// Channel 0 is the text-kernel logit, channel 1 (if `c >= 2`) the vertical-orientation logit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapInfo {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub exp: i8,
}

/// Where the map came from: the model input (`in_w x in_h`, the letterbox), the resized
/// image inside it (`content_w x content_h`, top-left aligned) and the original image
/// (`orig_w x orig_h`). Grid -> letterbox: `x_lb = (u + 0.5) * in_w / w - 0.5`; letterbox ->
/// original: `x = (x_lb + 0.5) * orig_w / content_w - 0.5` (pixel centres at integers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub in_w: usize,
    pub in_h: usize,
    pub content_w: usize,
    pub content_h: usize,
    pub orig_w: usize,
    pub orig_h: usize,
}

impl Frame {
    /// The map covers the image directly (no letterbox): content = input = original.
    pub fn identity(w: usize, h: usize) -> Self {
        Frame { in_w: w, in_h: h, content_w: w, content_h: h, orig_w: w, orig_h: h }
    }
}

/// One detected text line.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Line {
    /// Corners in original-image pixels (pixel centres at integers), clockwise from the
    /// top-left in reading orientation (`q0 -> q1` is the reading direction; for a vertical
    /// line `q0` is the column's top-right corner).
    pub quad: [P; 4],
    /// Mean text-kernel probability over the component.
    pub score: f64,
    /// Mean vertical-orientation logit over the component > 0.
    pub vertical: bool,
    /// `|q0 q1|` and `|q1 q2|`: extent along and across the reading direction (original px).
    pub length: f64,
    pub thickness: f64,
    /// Rank in reading order (0 = first) and block id (blocks numbered in reading order).
    pub order: u32,
    pub block: u32,
    /// Component size in map pixels.
    pub pixels: u32,
    /// sort key used by [`reading_order`]
    key: [f64; 5],
}

impl Line {
    /// A line from a quad (corners as described on [`Line::quad`]).
    pub fn new(quad: [P; 4], vertical: bool, score: f64) -> Self {
        Line {
            quad,
            score,
            vertical,
            length: dist(quad[0], quad[1]),
            thickness: dist(quad[1], quad[2]),
            ..Default::default()
        }
    }

    /// Axis-aligned bounds `[x0, y0, x1, y1]` of the quad.
    pub fn bounds(&self) -> [f64; 4] {
        let q = &self.quad;
        let (mut b0, mut b1, mut b2, mut b3) = (q[0][0], q[0][1], q[0][0], q[0][1]);
        for p in &q[1..] {
            b0 = b0.min(p[0]);
            b1 = b1.min(p[1]);
            b2 = b2.max(p[0]);
            b3 = b3.max(p[1]);
        }
        [b0, b1, b2, b3]
    }

    pub fn center(&self) -> P {
        let q = &self.quad;
        [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0]
    }

    /// The 16-float record of the JS API (`hxh_det_lines_ptr`, rust/DETECT.md):
    /// `x0 y0 x1 y1 x2 y2 x3 y3, score, vertical (0/1), order, block, length, thickness,
    /// crop width, crop height` (crop size = [`crate::image::crop_size`]).
    pub fn record(&self) -> [f32; 16] {
        let (cw, ch) = crate::image::crop_size(&self.quad);
        let q = &self.quad;
        [
            q[0][0] as f32,
            q[0][1] as f32,
            q[1][0] as f32,
            q[1][1] as f32,
            q[2][0] as f32,
            q[2][1] as f32,
            q[3][0] as f32,
            q[3][1] as f32,
            self.score as f32,
            self.vertical as u8 as f32,
            self.order as f32,
            self.block as f32,
            self.length as f32,
            self.thickness as f32,
            cw as f32,
            ch as f32,
        ]
    }
}

/// Working memory of [`detect`] for an `h x w` map, carved out of one caller buffer
/// ([`Scratch::bytes_needed`] bytes; any alignment).
pub struct Scratch<'a> {
    /// per pixel: 0 background, 1 text not yet visited, 2 visited
    mask: &'a mut [u8],
    /// BFS queue (pixel indices); also the union-find parents of [`reading_order`]
    queue: &'a mut [u32],
    /// per row: min and max x of the current component
    rows: &'a mut [[i32; 2]],
    pts: &'a mut [P],
    hull: &'a mut [P],
}

/// Split `n` items of `T` (a plain-old-data type) off the front of `buf`, aligned.
fn take<'a, T: Copy>(buf: &mut &'a mut [u8], n: usize) -> Option<&'a mut [T]> {
    let b = core::mem::take(buf);
    let off = b.as_ptr().align_offset(core::mem::align_of::<T>());
    let bytes = n.checked_mul(core::mem::size_of::<T>())?;
    if off.checked_add(bytes)? > b.len() {
        return None;
    }
    let (_, rest) = b.split_at_mut(off);
    let (mine, rest) = rest.split_at_mut(bytes);
    *buf = rest;
    // SAFETY: `mine` is aligned for T, holds exactly n * size_of::<T>() bytes, and is borrowed
    // exclusively for 'a. T is only instantiated with u8, u32, [i32; 2] and [f64; 2], for which
    // every bit pattern is valid.
    Some(unsafe { core::slice::from_raw_parts_mut(mine.as_mut_ptr().cast::<T>(), n) })
}

impl<'a> Scratch<'a> {
    /// Bytes [`Scratch::new`] needs for an `h x w` map (including alignment slack).
    pub fn bytes_needed(h: usize, w: usize) -> usize {
        let np = 2 * h + 3;
        h * w + 4 * h * w + 8 * h + 16 * np + 16 * (np + 1) + 32
    }

    /// Carve the scratch for an `h x w` map out of `buf`; `None` if it is too small.
    pub fn new(buf: &'a mut [u8], h: usize, w: usize) -> Option<Self> {
        let mut b = buf;
        let np = 2 * h + 3;
        Some(Scratch {
            mask: take::<u8>(&mut b, h * w)?,
            queue: take::<u32>(&mut b, h * w)?,
            rows: take::<[i32; 2]>(&mut b, h)?,
            pts: take::<P>(&mut b, np)?,
            hull: take::<P>(&mut b, np + 1)?,
        })
    }
}

/// `sigmoid(q * 2^exp)` for q = -128..=127 (index q + 128).
pub fn prob_table(exp: i8) -> [f64; 256] {
    let s = pow2(exp as i32);
    core::array::from_fn(|i| sigmoid((i as f64 - 128.0) * s))
}

/// The int8 decision threshold of channel 0: a pixel is text iff `q > kernel_threshold(..)`,
/// which is exactly `sigmoid(q * 2^exp) > bin_thresh`, i.e. `q * 2^exp > logit(bin_thresh)`
/// (for 0.3: `> -0.8473`). Returns -129 (everything passes) to 127 (nothing passes).
pub fn kernel_threshold(exp: i8, bin_thresh: f64) -> i32 {
    let p = prob_table(exp);
    (-128..=127).rev().find(|&q| p[(q + 128) as usize] <= bin_thresh).unwrap_or(-129)
}

/// A connected component of text pixels (8-connectivity), as passed to the callback of
/// [`for_each_component`].
#[derive(Debug)]
pub struct Component<'s> {
    /// Index `y * w + x` of its first pixel in raster order.
    pub first: usize,
    pub pixels: usize,
    /// Sum of `sigmoid(kernel logit)` over its pixels.
    pub prob_sum: f64,
    /// Sum of the int8 vertical-orientation logits (channel 1; 0 if the map has one channel).
    pub vert_sum: i64,
    /// Bounding box in map pixels, inclusive: x0, y0, x1, y1.
    pub bbox: [usize; 4],
    /// Convex hull of its pixels taken as unit squares (map coordinates, pixel centres at
    /// integers), positively oriented.
    pub hull: &'s [P],
    /// Minimum-area rectangle of `hull`.
    pub rect: RotRect,
}

/// Threshold channel 0 (`q > q_thr`) and call `f` for every 8-connected component, in raster
/// order of their first pixel.
///
/// The component's shape is the union of its pixel squares: per row, the squares of the
/// leftmost and rightmost pixel contribute their 4 corners (every other square of the row
/// lies between them), so the hull and the rectangle cover the pixels completely. A kernel
/// `k` pixels thick measures `k`, not `k - 1` as with pixel centres (OpenCV contours). On
/// each horizontal grid line only the outermost of those corners can be a hull vertex, so 2
/// points per line are generated, already in `(y, x)` order (no sort).
pub fn for_each_component<F: FnMut(&Component<'_>)>(map: &[i8], mi: &MapInfo, q_thr: i32, s: &mut Scratch<'_>, mut f: F) {
    let (h, w, c) = (mi.h, mi.w, mi.c);
    assert!(c >= 1 && map.len() >= h * w * c && s.mask.len() >= h * w && s.rows.len() >= h && s.pts.len() >= 2 * h + 2);
    let prob = prob_table(mi.exp);
    for (m, px) in s.mask[..h * w].iter_mut().zip(map.chunks_exact(c)) {
        *m = (px[0] as i32 > q_thr) as u8;
    }
    for r in s.rows[..h].iter_mut() {
        *r = [i32::MAX, i32::MIN];
    }
    for start in 0..h * w {
        if s.mask[start] != 1 {
            continue;
        }
        s.mask[start] = 2;
        s.queue[0] = start as u32;
        let (mut head, mut tail) = (0usize, 1usize);
        let (mut prob_sum, mut vert_sum) = (0f64, 0i64);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        while head < tail {
            let p = s.queue[head] as usize;
            head += 1;
            let (y, x) = (p / w, p % w);
            prob_sum += prob[(map[p * c] as i32 + 128) as usize];
            if c >= 2 {
                vert_sum += map[p * c + 1] as i64;
            }
            let r = &mut s.rows[y];
            r[0] = r[0].min(x as i32);
            r[1] = r[1].max(x as i32);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let q = ny * w + nx;
                    if s.mask[q] == 1 {
                        s.mask[q] = 2;
                        s.queue[tail] = q as u32;
                        tail += 1;
                    }
                }
            }
        }
        // grid line k - 0.5 (k = y0..=y1 + 1) touches rows k - 1 and k (8-connected: rows
        // y0..=y1 all occur): its outermost square corners, left then right
        let mut np = 0;
        for k in y0..=y1 + 1 {
            let (mut a, mut b) = (i32::MAX, i32::MIN);
            for y in [k.wrapping_sub(1), k] {
                if (y0..=y1).contains(&y) {
                    (a, b) = (a.min(s.rows[y][0]), b.max(s.rows[y][1]));
                }
            }
            let yf = k as f64 - 0.5;
            s.pts[np] = [a as f64 - 0.5, yf];
            s.pts[np + 1] = [b as f64 + 0.5, yf];
            np += 2;
        }
        for r in s.rows[y0..=y1].iter_mut() {
            *r = [i32::MAX, i32::MIN];
        }
        let m = convex_hull_sorted_yx(&s.pts[..np], s.hull);
        let rect = min_area_rect(&s.hull[..m]);
        f(&Component {
            first: start,
            pixels: tail,
            prob_sum,
            vert_sum,
            bbox: [x0, y0, x1, y1],
            hull: &s.hull[..m],
            rect,
        });
    }
}

/// Detect text lines in `map` (the detector output for the letterbox described by `fr`):
/// threshold -> 8-connected components -> per component: score = mean probability,
/// orientation = mean vertical logit > 0, hull of the pixel squares -> minimum-area rectangle
/// -> unclip -> filters (score, short side in letterbox px, centre inside the image content)
/// -> quad in original-image coordinates. The lines are written to `out` (at most
/// `out.len()`, in component order) and then sorted into reading order ([`reading_order`]).
/// Returns the number of lines.
pub fn detect(map: &[i8], mi: &MapInfo, fr: &Frame, p: &Params, s: &mut Scratch<'_>, out: &mut [Line]) -> usize {
    let q_thr = kernel_threshold(mi.exp, p.bin_thresh);
    let (sx, sy) = (fr.in_w as f64 / mi.w as f64, fr.in_h as f64 / mi.h as f64);
    let (kx, ky) = (sx * fr.orig_w as f64 / fr.content_w as f64, sy * fr.orig_h as f64 / fr.content_h as f64);
    let (cw, ch) = (fr.content_w as f64, fr.content_h as f64);
    let mut n = 0;
    for_each_component(map, mi, q_thr, s, |comp| {
        if n == out.len() {
            return;
        }
        let score = comp.prob_sum / comp.pixels as f64;
        if score < p.min_score {
            return;
        }
        let r = unclip(&comp.rect, p.unclip_ratio, p.shrink_ratio);
        if (r.w * sx).min(r.h * sy) < p.min_side || (r.w * sy).min(r.h * sx) < p.min_side {
            return;
        }
        // the centre must lie on the image, not in the letterbox padding
        let (cx, cy) = ((r.c[0] + 0.5) * sx - 0.5, (r.c[1] + 0.5) * sy - 0.5);
        if cx > cw - 0.5 || cy > ch - 0.5 {
            return;
        }
        let vertical = mi.c >= 2 && comp.vert_sum > 0;
        let (q, _, _) = rect_to_quad(&r, vertical);
        let q = q.map(|v| [(v[0] + 0.5) * kx - 0.5, (v[1] + 0.5) * ky - 0.5]);
        out[n] = Line { pixels: comp.pixels as u32, ..Line::new(q, vertical, score) };
        n += 1;
    });
    reading_order(&mut out[..n], p.block_gap, s.queue);
    n
}

fn find(parent: &mut [u32], mut i: usize) -> usize {
    while parent[i] as usize != i {
        let g = parent[parent[i] as usize];
        parent[i] = g;
        i = g as usize;
    }
    i
}

/// Sort `lines` into reading order and set their `order` and `block`.
///
/// Blocks: lines of the same orientation whose axis-aligned boxes are at most
/// `block_gap * max(thickness)` apart in both x and y are joined (union-find, transitively).
/// Blocks are ordered top to bottom by the top of their box (ties: the one further right
/// first). Inside a block, horizontal lines go top to bottom (then left to right), vertical
/// columns right to left (then top to bottom). `parent` is scratch for `lines.len()` u32.
pub fn reading_order(lines: &mut [Line], block_gap: f64, parent: &mut [u32]) {
    let n = lines.len();
    assert!(parent.len() >= n);
    for (i, p) in parent[..n].iter_mut().enumerate() {
        *p = i as u32;
    }
    for i in 0..n {
        let bi = lines[i].bounds();
        for j in i + 1..n {
            if lines[i].vertical != lines[j].vertical {
                continue;
            }
            let bj = lines[j].bounds();
            let g = block_gap * lines[i].thickness.max(lines[j].thickness);
            if bi[0] - g <= bj[2] && bj[0] - g <= bi[2] && bi[1] - g <= bj[3] && bj[1] - g <= bi[3] {
                let (a, b) = (find(parent, i), find(parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b) as u32;
                }
            }
        }
    }
    for i in 0..n {
        let r = find(parent, i);
        let (mut top, mut right) = (f64::INFINITY, f64::NEG_INFINITY);
        for (j, l) in lines.iter().enumerate() {
            if find(parent, j) == r {
                let b = l.bounds();
                top = top.min(b[1]);
                right = right.max(b[2]);
            }
        }
        let c = lines[i].center();
        let within = if lines[i].vertical { [-c[0], c[1]] } else { [c[1], c[0]] };
        lines[i].key = [top, -right, r as f64, within[0], within[1]];
    }
    // insertion sort (stable; a page has tens of lines, and core's sort is ~10 KB of wasm)
    let after = |a: &Line, b: &Line| a.key.iter().zip(&b.key).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).is_some_and(|o| o.is_gt());
    for i in 1..n {
        let mut j = i;
        while j > 0 && after(&lines[j - 1], &lines[j]) {
            lines.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut block = 0u32;
    for i in 0..n {
        if i > 0 && lines[i].key[2] != lines[i - 1].key[2] {
            block += 1;
        }
        lines[i].order = i as u32;
        lines[i].block = block;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(h: usize, w: usize) -> Vec<u8> {
        vec![0u8; Scratch::bytes_needed(h, w)]
    }

    fn comps(mask: &[&str]) -> Vec<(usize, usize, [usize; 4], f64)> {
        let (h, w) = (mask.len(), mask[0].len());
        let map: Vec<i8> = mask.iter().flat_map(|r| r.bytes().map(|b| if b == b'#' { 100 } else { -100 })).collect();
        let mut buf = scratch(h, w);
        let mut s = Scratch::new(&mut buf, h, w).unwrap();
        let mut out = Vec::new();
        for_each_component(&map, &MapInfo { h, w, c: 1, exp: -4 }, kernel_threshold(-4, 0.3), &mut s, |c| {
            out.push((c.first, c.pixels, c.bbox, c.rect.area()))
        });
        out
    }

    #[test]
    fn components_are_8_connected_and_cover_whole_pixels() {
        // the X arms touch only diagonally: one component of 5 pixels
        let c = comps(&["#...#", ".#.#.", "..#..", ".....", "##..#"]);
        assert_eq!(c.len(), 3, "{c:?}");
        assert_eq!((c[0].0, c[0].1), (0, 5));
        assert_eq!(c[0].2, [0, 0, 4, 2]);
        assert_eq!((c[1].0, c[1].1, c[1].3), (20, 2, 2.0)); // a 2 x 1 run: area 2 (pixel squares)
        assert_eq!((c[2].0, c[2].1, c[2].3), (24, 1, 1.0));
        // a 3-row bar measures 3 rows, not 2
        let c = comps(&["......", ".####.", ".####.", ".####.", "......"]);
        assert_eq!(c.len(), 1);
        assert!((c[0].3 - 12.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_is_exact_in_the_integer_domain() {
        for exp in -7..=2i8 {
            for &t in &[0.05, 0.3, 0.5, 0.7, 0.95] {
                let q = kernel_threshold(exp, t);
                for v in -128..=127 {
                    let prob = 1.0 / (1.0 + (-(v as f64) * 2f64.powi(exp as i32)).exp());
                    assert_eq!(v > q, prob > t, "exp {exp} thresh {t} q {v}");
                }
            }
        }
        // 0.3 <=> logit > -0.8473: at exp -3, q > -6.78 -> q >= -6
        assert_eq!(kernel_threshold(-3, 0.3), -7);
    }

    /// Draw rotated rectangles (as kernels, exactly) into a map and get them back.
    #[test]
    fn detect_recovers_rotated_kernels_in_original_coordinates() {
        let (h, w) = (80usize, 80usize);
        let mut map = vec![-100i8; h * w * 2];
        // (centre, axis angle, length, thickness, vertical)
        let lines = [([20.0, 15.0], 0.15f64, 24.0, 4.0, false), ([60.0, 40.0], 1.5f64, 30.0, 5.0, true)];
        for &(c, ang, len, th, vert) in &lines {
            let (u, n) = ([ang.cos(), ang.sin()], [-ang.sin(), ang.cos()]);
            for y in 0..h {
                for x in 0..w {
                    let d = [x as f64 - c[0], y as f64 - c[1]];
                    let (a, b) = (d[0] * u[0] + d[1] * u[1], d[0] * n[0] + d[1] * n[1]);
                    if a.abs() <= len / 2.0 && b.abs() <= th / 2.0 {
                        map[(y * w + x) * 2] = 64;
                        map[(y * w + x) * 2 + 1] = if vert { 40 } else { -40 };
                    }
                }
            }
        }
        let mi = MapInfo { h, w, c: 2, exp: -4 };
        // letterbox 160 x 160 (stride 2) holding a 160 x 120 resized image of a 640 x 480 original
        let fr = Frame { in_w: 160, in_h: 160, content_w: 160, content_h: 120, orig_w: 640, orig_h: 480 };
        let p = Params { unclip_ratio: 0.0, ..Default::default() };
        let mut buf = scratch(h, w);
        let mut s = Scratch::new(&mut buf, h, w).unwrap();
        let mut out = [Line::default(); 8];
        let n = detect(&map, &mi, &fr, &p, &mut s, &mut out);
        assert_eq!(n, 2);
        let k = 2.0 * 4.0; // grid -> original scale (x and y: 160/80 * 640/160, 160/80 * 480/120)
        for (l, &(c, _, len, th, vert)) in out[..n].iter().zip(&lines) {
            assert_eq!(l.vertical, vert);
            let cc = l.center();
            let want = [(c[0] + 0.5) * k - 0.5, (c[1] + 0.5) * k - 0.5];
            assert!((cc[0] - want[0]).abs() < 0.6 * k && (cc[1] - want[1]).abs() < 0.6 * k, "{cc:?} vs {want:?}");
            // pixel-square hulls overestimate a rotated kernel by up to ~1 grid px per side
            assert!((l.length / k - len).abs() < 2.0 && (l.thickness / k - th).abs() < 2.0, "{l:?}");
            assert!(l.score > 0.98);
        }
        // reading order: the horizontal line (higher on the page) first
        assert_eq!((out[0].order, out[1].order), (0, 1));
        assert!(!out[0].vertical && out[1].vertical);
        // the vertical column's first corner is its top-right
        let q = out[1].quad;
        assert!(q[0][0] > q[3][0] && q[0][1] < q[1][1], "{q:?}");
        // a stricter score threshold or a big min_side drops lines
        let n = detect(&map, &mi, &fr, &Params { min_side: 1e4, ..p }, &mut s, &mut out);
        assert_eq!(n, 0);
        // content region: a map of a 160 x 80 image keeps only lines whose centre is on it
        let fr2 = Frame { content_h: 80, orig_h: 320, ..fr };
        let n = detect(&map, &mi, &fr2, &p, &mut s, &mut out);
        assert_eq!(n, 1);
    }

    fn hline(x0: f64, y0: f64, len: f64, th: f64) -> Line {
        Line::new([[x0, y0], [x0 + len, y0], [x0 + len, y0 + th], [x0, y0 + th]], false, 1.0)
    }

    fn vline(x0: f64, y0: f64, len: f64, th: f64) -> Line {
        // column from (x0, y0) to (x0 + th, y0 + len); first corner = top-right
        Line::new([[x0 + th, y0], [x0 + th, y0 + len], [x0, y0 + len], [x0, y0]], true, 1.0)
    }

    #[test]
    fn reading_order_blocks_then_lines() {
        let mut lines = vec![
            // block A (top): three horizontal lines, given out of order
            hline(10.0, 30.0, 100.0, 10.0),
            hline(10.0, 10.0, 120.0, 10.0),
            hline(10.0, 50.0, 80.0, 10.0),
            // block B (lower): three vertical columns, right to left = 3 then 2 then 1
            vline(200.0, 120.0, 90.0, 12.0),
            vline(240.0, 115.0, 90.0, 12.0),
            vline(220.0, 118.0, 60.0, 12.0),
            // block C (middle, far left): a single line
            hline(10.0, 100.0, 50.0, 10.0),
        ];
        let mut parent = vec![0u32; lines.len()];
        reading_order(&mut lines, 1.0, &mut parent);
        let tops: Vec<(u32, f64, f64)> = lines.iter().map(|l| (l.block, l.quad[0][0], l.quad[0][1])).collect();
        assert_eq!(
            tops,
            vec![
                (0, 10.0, 10.0),
                (0, 10.0, 30.0),
                (0, 10.0, 50.0),
                (1, 10.0, 100.0),
                (2, 252.0, 115.0),
                (2, 232.0, 118.0),
                (2, 212.0, 120.0)
            ]
        );
        assert!(lines.iter().enumerate().all(|(i, l)| l.order == i as u32));
        // with no gap tolerance the columns (8 px apart) split into separate blocks
        reading_order(&mut lines, 0.1, &mut parent);
        assert_eq!(lines.iter().map(|l| l.block).max(), Some(6));
    }
}
