//! v5: a page photographed off a screen with a phone, like a phone photo of a
//! monitor. The composed page becomes the screen content inside a
//! bigger photo:
//!
//! * screen tone: black is never black and white rarely white (black level 0.02-0.2, white
//!   0.7-1.0);
//! * the screen sits anywhere in the frame at 35-95% of it, seen at an angle (keystone and corner
//!   jitter, a few degrees of rotation), inside a dark bezel, in front of a wall / desk made of
//!   gradients, noise and blurred blobs;
//! * moire bands, glare (a soft bright blob and sometimes a reflection gradient), vignetting and
//!   uneven exposure, defocus or motion blur, gain / gamma (a colour cast is only a brightness
//!   change in grayscale), sensor noise and JPEG.
//!
//! Every labelled quad and foreign quad goes through the same page -> photo homography, so the
//! ground truth stays exact (a projective map keeps quads quads).

use crate::codec::{decode_jpeg_gray, encode_jpeg};
use crate::geom::{apply, homography, H3, P};
use crate::page::{ForeignOut, LineOut};
use crate::raster::{blur, resize_area, value_noise, Gray};
use crate::rng::Rng;

const MAX_SIDE: f64 = 2000.0;

fn bilinear(src: &[f32], w: usize, h: usize, x: f64, y: f64) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let a = src[y0 * w + x0] * (1.0 - fx) + src[y0 * w + x1] * fx;
    let b = src[y1 * w + x0] * (1.0 - fx) + src[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

fn inside_convex(q: &[P; 4], x: f64, y: f64) -> bool {
    let mut sign = 0.0;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        if c != 0.0 {
            if sign == 0.0 {
                sign = c.signum();
            } else if c.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Grow a convex quad about its centre by `d` px (roughly: scale each corner away from the centre).
fn grow_quad(q: &[P; 4], d: f64) -> [P; 4] {
    let c = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0];
    q.map(|p| {
        let v = [p[0] - c[0], p[1] - c[1]];
        let l = v[0].hypot(v[1]).max(1e-6);
        [p[0] + v[0] / l * d * 1.41, p[1] + v[1] / l * d * 1.41]
    })
}

/// The environment around the screen: wall / desk gradients, low-frequency noise, blurred blobs.
fn environment(w: usize, h: usize, rng: &mut Rng) -> Vec<f32> {
    let (l1, l2) = (rng.uniform(0.15, 0.8) as f32, rng.uniform(0.05, 0.7) as f32);
    let vert = rng.chance(0.7);
    let mut img = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let t = if vert { y as f32 / h as f32 } else { x as f32 / w as f32 };
            img[y * w + x] = l1 + (l2 - l1) * t;
        }
    }
    let __r0 = rng.uniform(60.0, 400.0);
    let n = value_noise(w, h, rng, __r0);
    let amp = rng.uniform(0.03, 0.15) as f32;
    // blobs (objects on a desk, a lamp, a window) on a quarter-size layer, then blurred up
    let (sw, sh) = ((w / 4).max(4), (h / 4).max(4));
    let mut layer = Gray::new(sw, sh, 0.0);
    let mut mask = Gray::new(sw, sh, 0.0);
    for _ in 0..rng.int(1, 8) {
        let (cx, cy) = (rng.uniform(0.0, sw as f64), rng.uniform(0.0, sh as f64));
        let (rx, ry) = (rng.uniform(4.0, sw as f64 / 3.0), rng.uniform(4.0, sh as f64 / 3.0));
        let lum = rng.f64() as f32;
        let x0 = (cx - rx).max(0.0) as usize;
        let x1 = ((cx + rx) as usize).min(sw);
        let y0 = (cy - ry).max(0.0) as usize;
        let y1 = ((cy + ry) as usize).min(sh);
        let rect = rng.chance(0.5);
        for y in y0..y1 {
            for x in x0..x1 {
                let (u, v) = ((x as f64 - cx) / rx, (y as f64 - cy) / ry);
                if rect || u * u + v * v <= 1.0 {
                    layer.d[y * sw + x] = lum;
                    mask.d[y * sw + x] = 1.0;
                }
            }
        }
    }
    let sig = rng.uniform(0.5, 3.0);
    blur(&mut layer.d, sw, sh, sig, true);
    blur(&mut mask.d, sw, sh, sig, true);
    let lu = crate::raster::resize_bilinear(&layer.d, sw, sh, w, h);
    let mu = crate::raster::resize_bilinear(&mask.d, sw, sh, w, h);
    for i in 0..w * h {
        let m = mu[i].clamp(0.0, 1.0) * 0.9;
        img[i] = ((img[i] + amp * n[i]) * (1.0 - m) + lu[i] * m).clamp(0.0, 1.0);
    }
    img
}

/// Photograph `page` (the screen content) and move every quad into the photo. Returns the
/// photo's u8 pixels and size.
pub(crate) fn photograph(page: Gray, lines: &mut [LineOut], foreign: &mut [ForeignOut], rng: &mut Rng) -> (Vec<u8>, usize, usize) {
    let (w, h) = (page.w, page.h);
    // --- screen tone
    let (b0, b1) = (rng.uniform(0.02, 0.2) as f32, rng.uniform(0.7, 1.0) as f32);
    let mut src: Vec<f32> = page.d.iter().map(|&v| b0 + (b1 - b0) * v.clamp(0.0, 1.0)).collect();

    // --- photo size and where the screen goes
    let aspect = match rng.below(4) {
        0 | 1 => 0.75, // portrait phone photo
        2 => 4.0 / 3.0,
        _ => if rng.chance(0.5) { 9.0 / 16.0 } else { 16.0 / 9.0 },
    };
    let frac = rng.uniform(0.35, 0.95); // screen size over the photo's limiting side
    let mut s = rng.log_uniform(0.7, 1.6); // page px -> photo px
    let (sw, sh) = (w as f64 * s, h as f64 * s);
    // photo dims with W / H = aspect, screen = frac of the limiting dimension
    let mut pw = (sw / frac).max(sh / frac * aspect);
    let mut ph = pw / aspect;
    let big = pw.max(ph);
    if big > MAX_SIDE {
        let k = MAX_SIDE / big;
        pw *= k;
        ph *= k;
        s *= k;
    }
    let (pw_u, ph_u) = (pw.round().max(64.0) as usize, ph.round().max(64.0) as usize);
    let (pw, ph) = (pw_u as f64, ph_u as f64);
    let (sw, sh) = (w as f64 * s, h as f64 * s);

    // screen corners: centred somewhere, keystone, corner jitter, rotation
    let cx = rng.uniform(sw / 2.0 + 0.02 * pw, (pw - sw / 2.0 - 0.02 * pw).max(sw / 2.0 + 0.02 * pw));
    let cy = rng.uniform(sh / 2.0 + 0.02 * ph, (ph - sh / 2.0 - 0.02 * ph).max(sh / 2.0 + 0.02 * ph));
    let mut q: [P; 4] = [[-sw / 2.0, -sh / 2.0], [sw / 2.0, -sh / 2.0], [sw / 2.0, sh / 2.0], [-sw / 2.0, sh / 2.0]];
    let key = rng.uniform(-0.12, 0.12);
    if rng.chance(0.6) {
        // camera tilted up / down: one horizontal edge shorter
        q[0][0] *= 1.0 + key;
        q[1][0] *= 1.0 + key;
        q[2][0] *= 1.0 - key;
        q[3][0] *= 1.0 - key;
    } else {
        q[0][1] *= 1.0 + key;
        q[3][1] *= 1.0 + key;
        q[1][1] *= 1.0 - key;
        q[2][1] *= 1.0 - key;
    }
    let jit = rng.uniform(0.0, 0.05);
    for p in q.iter_mut() {
        p[0] += rng.uniform(-1.0, 1.0) * jit * sw;
        p[1] += rng.uniform(-1.0, 1.0) * jit * sh;
    }
    let ang = rng.normal(0.0, 3.0).clamp(-10.0, 10.0).to_radians();
    let (sa, ca) = ang.sin_cos();
    let mut q = q.map(|p| [cx + ca * p[0] - sa * p[1], cy + sa * p[0] + ca * p[1]]);
    // keep the screen in the frame
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in &q {
        minx = minx.min(p[0]);
        miny = miny.min(p[1]);
        maxx = maxx.max(p[0]);
        maxy = maxy.max(p[1]);
    }
    let fit = ((pw - 4.0) / (maxx - minx)).min((ph - 4.0) / (maxy - miny)).min(1.0);
    let (mcx, mcy) = ((minx + maxx) / 2.0, (miny + maxy) / 2.0);
    for p in q.iter_mut() {
        p[0] = mcx + (p[0] - mcx) * fit;
        p[1] = mcy + (p[1] - mcy) * fit;
    }
    let (mut minx, mut miny, mut maxx, mut maxy) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in &q {
        minx = minx.min(p[0]);
        miny = miny.min(p[1]);
        maxx = maxx.max(p[0]);
        maxy = maxy.max(p[1]);
    }
    let (dx, dy) = ((2.0 - minx).max(0.0) + (pw - 2.0 - maxx).min(0.0), (2.0 - miny).max(0.0) + (ph - 2.0 - maxy).min(0.0));
    for p in q.iter_mut() {
        p[0] += dx;
        p[1] += dy;
    }
    let s_eff = s * fit;

    // shrink the page first when the photo is smaller (area average, no aliasing)
    let (mut sw_px, mut sh_px) = (w, h);
    if s_eff < 0.9 {
        let (nw, nh) = (((w as f64 * s_eff * 1.05).round() as usize).max(8), ((h as f64 * s_eff * 1.05).round() as usize).max(8));
        src = resize_area(&src, w, h, nw, nh);
        (sw_px, sh_px) = (nw, nh);
    }
    let kx = sw_px as f64 / w as f64;
    let ky = sh_px as f64 / h as f64;
    let rect: [P; 4] = [[-0.5, -0.5], [sw_px as f64 - 0.5, -0.5], [sw_px as f64 - 0.5, sh_px as f64 - 0.5], [-0.5, sh_px as f64 - 0.5]];
    let hm: H3 = homography(&rect, &q); // (shrunk) page -> photo
    let hinv: H3 = homography(&q, &rect);
    let to_photo = |p: [f64; 2]| -> [f64; 2] {
        let r = apply(&hm, [(p[0] + 0.5) * kx - 0.5, (p[1] + 0.5) * ky - 0.5]);
        [(r[0] * 10.0).round() / 10.0, (r[1] * 10.0).round() / 10.0]
    };
    for l in lines.iter_mut() {
        l.quad = l.quad.map(to_photo);
    }
    for f in foreign.iter_mut() {
        f.quad = f.quad.map(to_photo);
    }

    // --- the photo
    let mut img = environment(pw_u, ph_u, rng);
    // bezel: a dark frame around the screen (sometimes thin, sometimes none: a laptop / phone edge)
    let bez = if rng.chance(0.85) { (sw.min(sh) * fit * rng.uniform(0.008, 0.05)).max(2.0) } else { 0.0 };
    let bez_lum = rng.uniform(0.02, 0.18) as f32;
    let outer = grow_quad(&q, bez);
    let x0 = outer.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
    let x1 = (outer.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max).ceil() as usize + 1).min(pw_u);
    let y0 = outer.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
    let y1 = (outer.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max).ceil() as usize + 1).min(ph_u);
    let mut on_screen = vec![false; pw_u * ph_u];
    for y in y0..y1 {
        for x in x0..x1 {
            let (fx, fy) = (x as f64, y as f64);
            if inside_convex(&q, fx, fy) {
                let p = apply(&hinv, [fx, fy]);
                img[y * pw_u + x] = bilinear(&src, sw_px, sh_px, p[0], p[1]);
                on_screen[y * pw_u + x] = true;
            } else if bez > 0.0 && inside_convex(&outer, fx, fy) {
                img[y * pw_u + x] = bez_lum;
            }
        }
    }

    // moire: low-frequency beating bands over the screen
    if rng.chance(0.5) {
        let period = rng.log_uniform(3.0, 30.0);
        let th = rng.uniform(0.0, std::f64::consts::PI);
        let (st, ct) = th.sin_cos();
        let k = rng.uniform(0.02, 0.1) as f32;
        let (wob_a, wob_f) = (rng.uniform(0.0, 3.0), rng.uniform(0.002, 0.02));
        let ph0 = rng.uniform(0.0, std::f64::consts::TAU);
        for y in y0..y1 {
            for x in x0..x1 {
                if on_screen[y * pw_u + x] {
                    let u = (x as f64 * ct + y as f64 * st) / period + wob_a * (wob_f * (y as f64 * ct - x as f64 * st)).sin();
                    let m = 0.5 + 0.5 * (std::f64::consts::TAU * u + ph0).cos() as f32;
                    img[y * pw_u + x] *= 1.0 - k * m;
                }
            }
        }
    }
    // glare: a soft bright blob, sometimes a reflection gradient
    if rng.chance(0.5) {
        let (gx, gy) = (rng.uniform(minx, maxx), rng.uniform(miny, maxy));
        let (rx, ry) = (sw * rng.uniform(0.08, 0.5), sh * rng.uniform(0.08, 0.5));
        let k = rng.uniform(0.1, 0.55) as f32;
        for y in 0..ph_u {
            for x in 0..pw_u {
                let (u, v) = ((x as f64 - gx) / rx, (y as f64 - gy) / ry);
                let g = (-(u * u + v * v)).exp() as f32;
                if g > 1e-3 {
                    let p = &mut img[y * pw_u + x];
                    *p += k * g * (1.0 - *p);
                }
            }
        }
    }
    if rng.chance(0.25) {
        let (sa, ca) = rng.uniform(0.0, std::f64::consts::TAU).sin_cos();
        let k = rng.uniform(0.05, 0.25) as f32;
        let span = pw.hypot(ph);
        for y in 0..ph_u {
            for x in 0..pw_u {
                let t = (((x as f64 * ca + y as f64 * sa) / span) + 0.5).clamp(0.0, 1.0) as f32;
                let p = &mut img[y * pw_u + x];
                *p += k * t * (1.0 - *p);
            }
        }
    }
    // vignetting and uneven exposure
    if rng.chance(0.65) {
        let k = rng.uniform(0.1, 0.5) as f32;
        let (vx, vy) = (pw / 2.0 + rng.normal(0.0, 0.1) * pw, ph / 2.0 + rng.normal(0.0, 0.1) * ph);
        let r2 = (pw * pw + ph * ph) / 4.0;
        for y in 0..ph_u {
            for x in 0..pw_u {
                let d = (((x as f64 - vx).powi(2) + (y as f64 - vy).powi(2)) / r2) as f32;
                img[y * pw_u + x] *= 1.0 - k * d.min(1.5);
            }
        }
    }
    if rng.chance(0.3) {
        let __r1 = rng.uniform(150.0, 600.0);
        let n = value_noise(pw_u, ph_u, rng, __r1);
        let k = rng.uniform(0.05, 0.2) as f32;
        img.iter_mut().zip(&n).for_each(|(v, nv)| *v *= 1.0 + k * nv);
    }
    // focus and shake
    if rng.chance(0.7) {
        blur(&mut img, pw_u, ph_u, rng.uniform(0.3, 1.8) * (pw.max(ph) / 1200.0).max(0.6), true);
    }
    if rng.chance(0.12) {
        let (sa, ca) = rng.uniform(0.0, std::f64::consts::PI).sin_cos();
        let len = rng.uniform(2.0, 7.0);
        let n = (len.ceil() as usize).max(2);
        let orig = img.clone();
        for y in 0..ph_u {
            for x in 0..pw_u {
                let mut acc = 0.0;
                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64 * len - len / 2.0;
                    acc += bilinear(&orig, pw_u, ph_u, x as f64 + t * ca, y as f64 + t * sa);
                }
                img[y * pw_u + x] = acc / n as f32;
            }
        }
    }
    // exposure: gain / gamma (white balance shifts are brightness only in grayscale)
    let (g, gam) = (rng.uniform(0.75, 1.2) as f32, rng.uniform(0.8, 1.25) as f32);
    let sd = rng.uniform(0.004, 0.03);
    for v in img.iter_mut() {
        *v = ((*v * g).clamp(0.0, 1.0)).powf(gam) + rng.normal(0.0, sd) as f32;
    }
    let mut out: Vec<u8> = img.iter().map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8).collect();
    if rng.chance(0.85) {
        let qv = rng.int(35, 92) as u8;
        if let Some((px, _, _)) = decode_jpeg_gray(&encode_jpeg(&out, pw_u, ph_u, qv))
            && px.len() == out.len() {
                out = px;
            }
    }
    (out, pw_u, ph_u)
}
