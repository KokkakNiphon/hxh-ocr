//! Grayscale buffers and rasterisation helpers. Shapes are rasterised with tiny-skia into
//! region-of-interest coverage masks and blended into an f32 page; geometry uses pixel centres
//! at integers, so paths are shifted by +0.5 for tiny-skia (which centres pixels at i + 0.5).

use tiny_skia::{FillRule, LineCap, LineJoin, Mask, Path, PathBuilder, Rect, Transform};

use crate::geom::P;
use crate::rng::Rng;

/// f32 grayscale image, values in [0, 1].
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Gray {
    pub fn new(w: usize, h: usize, v: f32) -> Self {
        Gray { w, h, d: vec![v; w * h] }
    }

    pub fn rect(&self) -> IRect {
        IRect { x0: 0, y0: 0, x1: self.w as i32, y1: self.h as i32 }
    }

    pub fn fill(&mut self, r: IRect, v: f32) {
        let r = r.intersect(self.rect());
        if r.is_empty() {
            return; // also an inverted rect (x1 < x0), which would panic below
        }
        for y in r.y0..r.y1 {
            let row = y as usize * self.w;
            self.d[row + r.x0 as usize..row + r.x1 as usize].fill(v);
        }
    }

    /// Mean of the rect.
    pub fn mean(&self, r: IRect) -> f32 {
        let r = r.intersect(self.rect());
        if r.is_empty() {
            return 0.5;
        }
        let mut s = 0.0f64;
        for y in r.y0..r.y1 {
            let row = y as usize * self.w;
            s += self.d[row + r.x0 as usize..row + r.x1 as usize].iter().map(|&v| v as f64).sum::<f64>();
        }
        (s / (r.w() * r.h()) as f64) as f32
    }
}

/// Integer rectangle, half-open [x0, x1) x [y0, y1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl IRect {
    pub fn w(&self) -> usize {
        (self.x1 - self.x0).max(0) as usize
    }
    pub fn h(&self) -> usize {
        (self.y1 - self.y0).max(0) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
    pub fn intersect(&self, o: IRect) -> IRect {
        IRect { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) }
    }
    /// Pixels touched by the continuous box b (pixel centres at integers), grown by pad.
    pub fn around(b: [f64; 4], pad: f64) -> IRect {
        IRect {
            x0: (b[0] - pad).floor() as i32,
            y0: (b[1] - pad).floor() as i32,
            x1: (b[2] + pad).ceil() as i32 + 1,
            y1: (b[3] + pad).ceil() as i32 + 1,
        }
    }
}

/// Coverage (alpha) over a rect of the page.
pub struct Cov {
    pub r: IRect,
    pub a: Vec<f32>,
}

#[derive(Clone, Copy)]
pub enum Draw {
    Fill,
    Stroke(f32),   // round caps and joins
    StrokeSq(f32), // square caps, mitred joins (font-like stroke ends)
}

pub fn poly_path(pts: &[P], closed: bool) -> Option<Path> {
    if pts.is_empty() {
        return None;
    }
    let mut pb = PathBuilder::new();
    pb.move_to(pts[0][0] as f32, pts[0][1] as f32);
    for p in &pts[1..] {
        pb.line_to(p[0] as f32, p[1] as f32);
    }
    if pts.len() == 1 {
        pb.line_to(pts[0][0] as f32 + 1e-3, pts[0][1] as f32);
    }
    if closed {
        pb.close();
    }
    pb.finish()
}

pub fn ellipse_path(cx: f64, cy: f64, rx: f64, ry: f64, ang_deg: f64) -> Option<Path> {
    let r = Rect::from_ltrb((cx - rx) as f32, (cy - ry) as f32, (cx + rx) as f32, (cy + ry) as f32)?;
    let p = PathBuilder::from_oval(r)?;
    if ang_deg == 0.0 { Some(p) } else { p.transform(Transform::from_rotate_at(ang_deg as f32, cx as f32, cy as f32)) }
}

fn stroked(path: &Path, how: Draw) -> Option<Path> {
    match how {
        Draw::Fill => Some(path.clone()),
        Draw::Stroke(w) => {
            let st = tiny_skia::Stroke { width: w.max(0.05), line_cap: LineCap::Round, line_join: LineJoin::Round, ..Default::default() };
            path.stroke(&st, 1.0)
        }
        Draw::StrokeSq(w) => {
            let st = tiny_skia::Stroke { width: w.max(0.05), line_cap: LineCap::Square, line_join: LineJoin::Miter, miter_limit: 2.0, ..Default::default() };
            path.stroke(&st, 1.0)
        }
    }
}

/// Rasterise the union of `items` into a mask over `clip` (cropped to the items' bounds).
/// Returns the rect and the raw u8 coverage.
pub fn rasterize(items: &[(Path, Draw)], clip: IRect, aa: bool) -> Option<(IRect, Vec<u8>)> {
    let mut outlines: Vec<Path> = Vec::with_capacity(items.len());
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for (p, how) in items {
        if let Some(o) = stroked(p, *how) {
            let r = o.bounds();
            b = [b[0].min(r.left() as f64), b[1].min(r.top() as f64), b[2].max(r.right() as f64), b[3].max(r.bottom() as f64)];
            outlines.push(o);
        }
    }
    if outlines.is_empty() {
        return None;
    }
    let roi = IRect::around(b, 1.0).intersect(clip);
    if roi.is_empty() {
        return None;
    }
    let mut m = Mask::new(roi.w() as u32, roi.h() as u32)?;
    let ts = Transform::from_translate(0.5 - roi.x0 as f32, 0.5 - roi.y0 as f32);
    for o in &outlines {
        m.fill_path(o, FillRule::Winding, aa, ts);
    }
    Some((roi, m.data().to_vec()))
}

pub fn coverage(items: &[(Path, Draw)], clip: IRect) -> Option<Cov> {
    let (r, d) = rasterize(items, clip, true)?;
    Some(Cov { r, a: d.iter().map(|&v| v as f32 * (1.0 / 255.0)).collect() })
}

/// page <- page * (1 - k a) + lum * k a over the coverage rect.
pub fn blend(page: &mut Gray, c: &Cov, lum: f32, k: f32) {
    let r = c.r.intersect(page.rect());
    let cw = c.r.w();
    for y in r.y0..r.y1 {
        let prow = y as usize * page.w;
        let crow = (y - c.r.y0) as usize * cw;
        for x in r.x0..r.x1 {
            let a = c.a[crow + (x - c.r.x0) as usize] * k;
            if a > 0.0 {
                let p = &mut page.d[prow + x as usize];
                *p += (lum - *p) * a;
            }
        }
    }
}

/// Zero the coverage wherever `occ` (page-sized u8) is set.
pub fn clip_by(c: &mut Cov, occ: &[u8], page_w: usize) {
    let cw = c.r.w();
    for y in 0..c.r.h() {
        let py = y as i32 + c.r.y0;
        for x in 0..cw {
            let px = x as i32 + c.r.x0;
            if occ[py as usize * page_w + px as usize] != 0 {
                c.a[y * cw + x] = 0.0;
            }
        }
    }
}

/// Fill a convex polygon (pixels whose centre is inside or on the boundary) into a u8 mask
/// that covers `mr` of the page.
pub fn fill_convex_u8(mask: &mut [u8], mr: IRect, q: &[P], v: u8) {
    let b = crate::geom::bbox(q);
    let r = IRect { x0: b[0].ceil() as i32, y0: b[1].ceil() as i32, x1: b[2].floor() as i32 + 1, y1: b[3].floor() as i32 + 1 }.intersect(mr);
    let sgn = if crate::geom::area2(q) >= 0.0 { 1.0 } else { -1.0 };
    let n = q.len();
    let mw = mr.w();
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let (px, py) = (x as f64, y as f64);
            let inside = (0..n).all(|i| {
                let (a, bb) = (q[i], q[(i + 1) % n]);
                sgn * ((bb[0] - a[0]) * (py - a[1]) - (bb[1] - a[1]) * (px - a[0])) >= -1e-9
            });
            if inside {
                mask[(y - mr.y0) as usize * mw + (x - mr.x0) as usize] = v;
            }
        }
    }
}

fn gauss_kernel(sigma: f64) -> Vec<f32> {
    let r = (3.0 * sigma).ceil().max(1.0) as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur in place; edges clamp (`clamp`) or read as zero.
pub fn blur(d: &mut [f32], w: usize, h: usize, sigma: f64, clamp: bool) {
    if sigma < 0.05 || w == 0 || h == 0 {
        return;
    }
    let k = gauss_kernel(sigma);
    let r = (k.len() / 2) as i64;
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = &d[y * w..(y + 1) * w];
        for x in 0..w {
            let mut s = 0.0;
            for (j, &kv) in k.iter().enumerate() {
                let xx = x as i64 + j as i64 - r;
                if xx >= 0 && xx < w as i64 {
                    s += kv * row[xx as usize];
                } else if clamp {
                    s += kv * row[xx.clamp(0, w as i64 - 1) as usize];
                }
            }
            tmp[y * w + x] = s;
        }
    }
    for x in 0..w {
        for y in 0..h {
            let mut s = 0.0;
            for (j, &kv) in k.iter().enumerate() {
                let yy = y as i64 + j as i64 - r;
                if yy >= 0 && yy < h as i64 {
                    s += kv * tmp[yy as usize * w + x];
                } else if clamp {
                    s += kv * tmp[yy.clamp(0, h as i64 - 1) as usize * w + x];
                }
            }
            d[y * w + x] = s;
        }
    }
}

fn cubic_w(t: f64) -> [f64; 4] {
    // Keys cubic, a = -0.75 (as OpenCV's INTER_CUBIC)
    let a = -0.75;
    let f = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        } else {
            0.0
        }
    };
    [f(1.0 + t), f(t), f(1.0 - t), f(2.0 - t)]
}

fn cubic_resize_1d(src: &[f64], n: usize) -> Vec<(usize, [f64; 4])> {
    // returns (base index, weights) for each output sample; base index is clamped later
    let scale = src.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let fx = (i as f64 + 0.5) * scale - 0.5;
            let x0 = fx.floor();
            ((x0 as i64 + 1).max(0) as usize, cubic_w(fx - x0))
        })
        .collect()
}

/// Smooth value noise (coarse random grid, cubic upsampling), normalised to mean 0, std 1.
pub fn value_noise(w: usize, h: usize, rng: &mut Rng, cell: f64) -> Vec<f32> {
    let gh = ((h as f64 / cell) as usize + 2).max(2);
    let gw = ((w as f64 / cell) as usize + 2).max(2);
    let g: Vec<f64> = (0..gh * gw).map(|_| rng.f64()).collect();
    let at = |v: &[f64], len: usize, i: i64| v[i.clamp(0, len as i64 - 1) as usize];
    let wx = cubic_resize_1d(&vec![0.0; gw], w);
    let wy = cubic_resize_1d(&vec![0.0; gh], h);
    let mut rows = vec![0.0f64; gh * w];
    for gy in 0..gh {
        let row = &g[gy * gw..(gy + 1) * gw];
        for (x, (b, k)) in wx.iter().enumerate() {
            let b = *b as i64;
            rows[gy * w + x] = (0..4).map(|j| k[j] * at(row, gw, b - 2 + j as i64)).sum();
        }
    }
    let mut out = vec![0.0f32; w * h];
    let mut col = vec![0.0f64; gh];
    for x in 0..w {
        for gy in 0..gh {
            col[gy] = rows[gy * w + x];
        }
        for (y, (b, k)) in wy.iter().enumerate() {
            let b = *b as i64;
            out[y * w + x] = (0..4).map(|j| k[j] * at(&col, gh, b - 2 + j as i64)).sum::<f64>() as f32;
        }
    }
    let n = out.len() as f64;
    let mean = out.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = out.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt() + 1e-6;
    out.iter_mut().for_each(|v| *v = ((*v as f64 - mean) / sd) as f32);
    out
}

/// Bilinear resize (pixel-centre aligned).
pub fn resize_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh];
    let (sx, sy) = (sw as f64 / dw as f64, sh as f64 / dh as f64);
    for y in 0..dh {
        let fy = ((y as f64 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = (fy - y0 as f64) as f32;
        for x in 0..dw {
            let fx = ((x as f64 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = (fx - x0 as f64) as f32;
            let a = src[y0 * sw + x0] * (1.0 - tx) + src[y0 * sw + x1] * tx;
            let b = src[y1 * sw + x0] * (1.0 - tx) + src[y1 * sw + x1] * tx;
            out[y * dw + x] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

/// Area-average downscale by an arbitrary factor (box filter over source pixels).
pub fn resize_area(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        let ya = y * sh / dh;
        let yb = ((y + 1) * sh / dh).max(ya + 1).min(sh);
        for x in 0..dw {
            let xa = x * sw / dw;
            let xb = ((x + 1) * sw / dw).max(xa + 1).min(sw);
            let mut s = 0.0f32;
            for yy in ya..yb {
                s += src[yy * sw + xa..yy * sw + xb].iter().sum::<f32>();
            }
            out[y * dw + x] = s / ((yb - ya) * (xb - xa)) as f32;
        }
    }
    out
}
