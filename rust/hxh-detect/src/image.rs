//! Image helpers: the detector's letterbox input and rectified line crops (u8 grayscale,
//! row-major, pixel centres at integers).

use crate::geom::{P, dist};
use crate::math::{floor, round};

/// Size of the image inside an `dst_w x dst_h` letterbox: scaled by
/// `s = min(dst_w / w, dst_h / h)`, rounded half up, at least 1 px (the limiting side gets
/// exactly the letterbox size).
pub fn letterbox_dims(w: usize, h: usize, dst_w: usize, dst_h: usize) -> (usize, usize) {
    let s = (dst_w as f64 / w as f64).min(dst_h as f64 / h as f64);
    let nw = (round(w as f64 * s) as usize).clamp(1, dst_w);
    let nh = (round(h as f64 * s) as usize).clamp(1, dst_h);
    (nw, nh)
}

/// The detector input: `src` (`w x h`) resized to [`letterbox_dims`] and placed at the
/// top-left of `dst` (`dst_w x dst_h`), the rest filled with 255. Shrinking averages areas
/// (each output pixel is the mean of the source area it covers, partial pixels weighted by
/// their overlap, like OpenCV's INTER_AREA); growing is bilinear with half-pixel centres
/// (OpenCV's INTER_LINEAR). `tmp` needs `2 * dst_w` entries. Returns the content size
/// `(new_w, new_h)`.
pub fn letterbox(src: &[u8], w: usize, h: usize, dst: &mut [u8], dst_w: usize, dst_h: usize, tmp: &mut [f64]) -> (usize, usize) {
    assert!(w > 0 && h > 0 && src.len() >= w * h && dst.len() >= dst_w * dst_h && tmp.len() >= 2 * dst_w);
    let (nw, nh) = letterbox_dims(w, h, dst_w, dst_h);
    dst[..dst_w * dst_h].fill(255);
    if (nw, nh) == (w, h) {
        for y in 0..h {
            dst[y * dst_w..y * dst_w + w].copy_from_slice(&src[y * w..(y + 1) * w]);
        }
    } else if nw <= w && nh <= h {
        area_resize(src, w, h, dst, dst_w, nw, nh, tmp);
    } else {
        bilinear_resize(src, w, h, dst, dst_w, nw, nh);
    }
    (nw, nh)
}

#[inline]
fn to_u8(v: f64) -> u8 {
    round(v).clamp(0.0, 255.0) as u8
}

/// Overlap of source cells with the output cell `[i * scale, (i + 1) * scale)`: calls
/// `f(source index, weight)`.
#[inline]
fn cover(i: usize, scale: f64, n: usize, mut f: impl FnMut(usize, f64)) {
    let (a, b) = (i as f64 * scale, (i + 1) as f64 * scale);
    let (s0, s1) = (floor(a) as usize, (b.min(n as f64) - 1e-9).max(a) as usize);
    for s in s0..=s1.min(n - 1) {
        let wt = b.min(s as f64 + 1.0) - a.max(s as f64);
        if wt > 0.0 {
            f(s, wt);
        }
    }
}

/// Area-weighted sum of `line` over `[ox * sx, (ox + 1) * sx)`: the two partial end pixels
/// weighted by their overlap, the pixels in between summed as integers.
#[inline]
fn area_sum(line: &[u8], ox: usize, sx: f64) -> f64 {
    let w = line.len();
    let (a, b) = (ox as f64 * sx, ((ox + 1) as f64 * sx).min(w as f64));
    let i0 = (floor(a) as usize).min(w - 1);
    let i1 = ((-floor(-b)) as usize).clamp(i0 + 1, w); // exclusive
    if i1 == i0 + 1 {
        return line[i0] as f64 * (b - a);
    }
    let mid: u32 = line[i0 + 1..i1 - 1].iter().map(|&v| v as u32).sum();
    line[i0] as f64 * ((i0 + 1) as f64 - a) + mid as f64 + line[i1 - 1] as f64 * (b - (i1 - 1) as f64)
}

#[allow(clippy::too_many_arguments)]
fn area_resize(src: &[u8], w: usize, h: usize, dst: &mut [u8], stride: usize, nw: usize, nh: usize, tmp: &mut [f64]) {
    let (sx, sy) = (w as f64 / nw as f64, h as f64 / nh as f64);
    let norm = 1.0 / (sx * sy);
    let (row, acc) = tmp[..2 * nw].split_at_mut(nw);
    let mut row_y = usize::MAX; // the source row currently in `row`
    for oy in 0..nh {
        acc.fill(0.0);
        cover(oy, sy, h, |y, wy| {
            if y != row_y {
                let line = &src[y * w..(y + 1) * w];
                for (ox, r) in row.iter_mut().enumerate() {
                    *r = area_sum(line, ox, sx);
                }
                row_y = y;
            }
            for (a, &r) in acc.iter_mut().zip(row.iter()) {
                *a += r * wy;
            }
        });
        for (d, &a) in dst[oy * stride..oy * stride + nw].iter_mut().zip(acc.iter()) {
            *d = to_u8(a * norm);
        }
    }
}

/// Source index pair and weight of the second for output coordinate `o` (half-pixel centres).
#[inline]
fn lin(o: usize, scale: f64, n: usize) -> (usize, usize, f64) {
    let s = ((o as f64 + 0.5) * scale - 0.5).clamp(0.0, (n - 1) as f64);
    let i = floor(s) as usize;
    (i, (i + 1).min(n - 1), s - i as f64)
}

fn bilinear_resize(src: &[u8], w: usize, h: usize, dst: &mut [u8], stride: usize, nw: usize, nh: usize) {
    let (sx, sy) = (w as f64 / nw as f64, h as f64 / nh as f64);
    for oy in 0..nh {
        let (y0, y1, fy) = lin(oy, sy, h);
        for ox in 0..nw {
            let (x0, x1, fx) = lin(ox, sx, w);
            let p = |y: usize, x: usize| src[y * w + x] as f64;
            let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
            let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
            dst[oy * stride + ox] = to_u8(top * (1.0 - fy) + bot * fy);
        }
    }
}

/// Size of the rectified crop of `quad` (corners as in [`crate::Line::quad`]):
/// `(round(|q0 q1|), round(|q1 q2|))`, each at least 1: native resolution, the reading
/// direction horizontal.
pub fn crop_size(quad: &[P; 4]) -> (usize, usize) {
    let w = round(dist(quad[0], quad[1])).max(1.0);
    let h = round(dist(quad[1], quad[2])).max(1.0);
    (w as usize, h as usize)
}

/// Projective map of the unit square onto `q`: (0,0) -> q0, (1,0) -> q1, (1,1) -> q2,
/// (0,1) -> q3 (Heckbert's square-to-quad). Exactly affine for parallelograms.
#[derive(Clone, Copy, Debug)]
struct Homography([f64; 8]);

impl Homography {
    fn square_to_quad(q: &[P; 4]) -> Self {
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *q;
        let (dx3, dy3) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
        let (dx1, dy1, dx2, dy2) = (x1 - x2, y1 - y2, x3 - x2, y3 - y2);
        let den = dx1 * dy2 - dx2 * dy1;
        let (g, hh) = if (dx3 == 0.0 && dy3 == 0.0) || den.abs() < 1e-12 {
            (0.0, 0.0)
        } else {
            ((dx3 * dy2 - dx2 * dy3) / den, (dx1 * dy3 - dx3 * dy1) / den)
        };
        Homography([x1 - x0 + g * x1, x3 - x0 + hh * x3, x0, y1 - y0 + g * y1, y3 - y0 + hh * y3, y0, g, hh])
    }

    #[inline]
    fn map(&self, s: f64, t: f64) -> P {
        let [a, b, c, d, e, f, g, h] = self.0;
        let z = g * s + h * t + 1.0;
        [(a * s + b * t + c) / z, (d * s + e * t + f) / z]
    }
}

/// Bilinear sample at `(x, y)` with edge replication.
#[inline]
fn sample(img: &[u8], w: usize, h: usize, x: f64, y: f64) -> u8 {
    let (xf, yf) = (floor(x), floor(y));
    let (fx, fy) = (x - xf, y - yf);
    let cl = |v: f64, n: usize| (v.clamp(0.0, (n - 1) as f64)) as usize;
    let (x0, x1, y0, y1) = (cl(xf, w), cl(xf + 1.0, w), cl(yf, h), cl(yf + 1.0, h));
    let p = |y: usize, x: usize| img[y * w + x] as f64;
    let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
    let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
    to_u8(top * (1.0 - fy) + bot * fy)
}

/// Rectify `quad` of `img` (`w x h`) into `out` (`ow x oh`, row-major): output pixel
/// `(i, j)` samples the image at the projective image of `((i + 0.5) / ow, (j + 0.5) / oh)`
/// (bilinear, edges replicated). `q0 -> q1` becomes the top edge read left to right, so a
/// vertical line (whose `q0` is the column's top-right corner) comes out rotated 90 degrees
/// counter-clockwise with its text running left to right. With [`crop_size`] and an
/// axis-aligned quad on pixel boundaries the crop is an exact copy.
pub fn crop(img: &[u8], w: usize, h: usize, quad: &[P; 4], out: &mut [u8], ow: usize, oh: usize) {
    assert!(w > 0 && h > 0 && img.len() >= w * h && out.len() >= ow * oh);
    let hm = Homography::square_to_quad(quad);
    for j in 0..oh {
        let t = (j as f64 + 0.5) / oh as f64;
        for i in 0..ow {
            let p = hm.map((i as f64 + 0.5) / ow as f64, t);
            out[j * ow + i] = sample(img, w, h, p[0], p[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_dims_and_padding() {
        assert_eq!(letterbox_dims(1000, 2000, 640, 640), (320, 640));
        assert_eq!(letterbox_dims(1280, 720, 640, 640), (640, 360));
        assert_eq!(letterbox_dims(100, 50, 640, 640), (640, 320));
        let src: Vec<u8> = (0..400 * 200).map(|i| (i % 7 * 30) as u8).collect();
        let mut dst = vec![0u8; 64 * 64];
        let mut tmp = vec![0f64; 128];
        assert_eq!(letterbox(&src, 400, 200, &mut dst, 64, 64, &mut tmp), (64, 32));
        assert!(dst[32 * 64..].iter().all(|&v| v == 255));
        // area average of a constant image is the constant; of a 2x2 checker its mean
        let src = vec![77u8; 300 * 120];
        letterbox(&src, 300, 120, &mut dst, 64, 64, &mut tmp);
        assert!(dst[..64 * 26].iter().all(|&v| v == 77) && dst[64 * 26..].iter().all(|&v| v == 255));
        let src: Vec<u8> = (0..128 * 128).map(|i| if (i / 128 + i % 128) % 2 == 0 { 0 } else { 200 }).collect();
        letterbox(&src, 128, 128, &mut dst, 64, 64, &mut tmp);
        assert!(dst.iter().all(|&v| v == 100));
        // exact copy when the image already fits; bilinear when it is smaller
        let src: Vec<u8> = (0..64 * 10).map(|i| (i % 64 * 4) as u8).collect();
        letterbox(&src, 64, 10, &mut dst, 64, 64, &mut tmp);
        assert_eq!(&dst[..640], &src[..]);
        let src = [0u8, 100, 200, 100];
        letterbox(&src, 2, 2, &mut dst, 64, 64, &mut tmp);
        assert_eq!((dst[0], dst[63], dst[63 * 64]), (0, 100, 200));
    }

    #[test]
    fn crops_copy_axis_aligned_boxes_and_turn_vertical_lines() {
        let (w, h) = (40, 30);
        let img: Vec<u8> = (0..w * h).map(|i| ((i % w) * 5 + (i / w) * 3) as u8).collect();
        // box over pixels x 5..=14, y 7..=10 (corners on pixel boundaries)
        let q = [[4.5, 6.5], [14.5, 6.5], [14.5, 10.5], [4.5, 10.5]];
        assert_eq!(crop_size(&q), (10, 4));
        let mut out = vec![0u8; 40];
        crop(&img, w, h, &q, &mut out, 10, 4);
        for j in 0..4 {
            for i in 0..10 {
                assert_eq!(out[j * 10 + i], img[(7 + j) * w + 5 + i]);
            }
        }
        // the same pixels as a vertical column x 5..=8, y 3..=12: q0 = top-right, reading down
        let q = [[8.5, 2.5], [8.5, 12.5], [4.5, 12.5], [4.5, 2.5]];
        assert_eq!(crop_size(&q), (10, 4));
        crop(&img, w, h, &q, &mut out, 10, 4);
        // rotated 90 degrees counter-clockwise: crop (i, j) = image (x = 8 - j, y = 3 + i)
        for j in 0..4 {
            for i in 0..10 {
                assert_eq!(out[j * 10 + i], img[(3 + i) * w + 8 - j], "i {i} j {j}");
            }
        }
        // outside the image the border is replicated; a rotated quad samples in between
        let q = [[-10.5, -10.5], [-0.5, -10.5], [-0.5, -0.5], [-10.5, -0.5]];
        crop(&img, w, h, &q, &mut out, 10, 4);
        assert!(out.iter().all(|&v| v == img[0]));
        let hm = Homography::square_to_quad(&[[0.0, 0.0], [4.0, 1.0], [5.0, 5.0], [0.5, 3.0]]);
        for (s, t, want) in [(0.0, 0.0, [0.0, 0.0]), (1.0, 0.0, [4.0, 1.0]), (1.0, 1.0, [5.0, 5.0]), (0.0, 1.0, [0.5, 3.0])] {
            let p = hm.map(s, t);
            assert!((p[0] - want[0]).abs() < 1e-12 && (p[1] - want[1]).abs() < 1e-12, "{p:?}");
        }
    }
}
