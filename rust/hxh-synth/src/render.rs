//! Port of `hxh_ocr/render.py` (generator v4): text styles, per-glyph transform and hand-drawn
//! jitter, line layout (horizontal / vertical, small kana, spaces, glyph variants with weights,
//! dakuten / handakuten placement), then drawing primitives in pixels.

use tiny_skia::{Path, PathBuilder};

use crate::geom::{apply, apply_all, local_frame, H3, P};
use crate::glyphs::{glyph_set, strokes_bbox, Mark, Stroke};
use crate::raster::{coverage, ellipse_path, poly_path, Cov, Draw, IRect};
use crate::rng::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StyleKind {
    Clean,
    Hand,
    Brush,
}

#[derive(Clone, Debug)]
pub struct TextStyle {
    pub kind: StyleKind,
    pub sw: f64, // stroke width (em)
    pub brush: bool,
    pub mono: bool,          // monospaced cells vs proportional
    pub cell: f64,           // em, mono advance
    pub spacing: f64,        // em, proportional gap
    pub wobble: f64,         // em, smooth path displacement
    pub stroke_jitter: f64,  // em
    pub rot_std: f64,        // deg, per-glyph rotation
    pub scale_std: f64,
    pub aspect_std: f64,
    pub shear: f64,
    pub cross_jitter: f64,   // em, per-glyph baseline wobble
    pub curve_amp: f64,      // em, baseline curve
    pub curve_freq: f64,     // rad per em
    pub dot_scale: f64,
    pub mark_scale: f64,
    // v5 (photo-style pages, model family): all off by default
    pub square: bool,    // square stroke ends and mitred joins instead of round
    pub serif: f64,      // em, ticks across the ends of open strokes (0 = none)
    pub wave_amp: f64,   // em, wavy warp across the line (whole glyph shapes bend)
    pub wave_len: f64,   // em, its wavelength along the line
    pub wave_ph: f64,
    pub wave_along: f64, // em, a second, smaller warp along the line
    // v6
    pub mark_small: f64, // > 0: dakuten drawn as a small dot this many stroke widths across, tucked at the corner
}

impl Default for TextStyle {
    fn default() -> Self {
        TextStyle {
            kind: StyleKind::Clean,
            sw: glyph_set().ref_stroke_em,
            brush: false,
            mono: false,
            cell: 1.25,
            spacing: 0.25,
            wobble: 0.0,
            stroke_jitter: 0.0,
            rot_std: 0.5,
            scale_std: 0.02,
            aspect_std: 0.02,
            shear: 0.0,
            cross_jitter: 0.0,
            curve_amp: 0.0,
            curve_freq: 0.0,
            dot_scale: 1.0,
            mark_scale: 1.0,
            square: false,
            serif: 0.0,
            wave_amp: 0.0,
            wave_len: 1.0,
            wave_ph: 0.0,
            wave_along: 0.0,
            mark_small: 0.0,
        }
    }
}

pub fn sample_style(rng: &mut Rng) -> TextStyle {
    let u = rng.f64();
    let mut st = if u < 0.4 {
        TextStyle { kind: StyleKind::Clean, sw: rng.uniform(0.045, 0.13), rot_std: 0.4, scale_std: 0.01, aspect_std: 0.01, ..Default::default() }
    } else if u < 0.8 {
        TextStyle {
            kind: StyleKind::Hand,
            sw: rng.uniform(0.04, 0.15),
            wobble: rng.uniform(0.003, 0.025),
            stroke_jitter: rng.uniform(0.005, 0.035),
            rot_std: rng.uniform(1.0, 6.0),
            scale_std: rng.uniform(0.03, 0.1),
            aspect_std: rng.uniform(0.03, 0.1),
            cross_jitter: rng.uniform(0.0, 0.08),
            ..Default::default()
        }
    } else {
        TextStyle {
            kind: StyleKind::Brush,
            brush: true,
            sw: rng.uniform(0.07, 0.2),
            wobble: rng.uniform(0.0, 0.02),
            stroke_jitter: rng.uniform(0.0, 0.03),
            rot_std: rng.uniform(0.5, 5.0),
            scale_std: rng.uniform(0.02, 0.08),
            aspect_std: rng.uniform(0.02, 0.08),
            cross_jitter: rng.uniform(0.0, 0.06),
            ..Default::default()
        }
    };
    if rng.chance(0.18) {
        st.sw = rng.uniform(0.15, 0.30); // v4: very bold, like the anime title-card font
    }
    st.mono = rng.chance(0.5);
    st.cell = rng.uniform(1.18, 1.55);
    st.spacing = rng.uniform(0.06, 0.5);
    st.shear = if rng.chance(0.3) { rng.normal(0.0, 0.12) } else { 0.0 };
    if rng.chance(0.08) {
        st.curve_amp = rng.uniform(0.05, 0.3);
        st.curve_freq = rng.uniform(0.15, 0.6);
    }
    let r = st.sw / glyph_set().ref_stroke_em;
    st.dot_scale = rng.uniform(0.85, 1.15) * r.powf(0.25);
    st.mark_scale = rng.uniform(0.8, 1.3) * r.powf(0.45);
    st
}

/// v5 styles: the v4 mix plus, with small probabilities, wide letter spacing (up to about 3
/// glyph widths: lyric videos), square stroke ends, serif ticks and a wavy warp (glowing
/// "Nen"-style titles). `wide` forces the wide spacing.
pub fn sample_style_v5(rng: &mut Rng, wide: bool) -> TextStyle {
    let mut st = sample_style(rng);
    if wide || rng.chance(0.07) {
        // gaps of 0.6-3 em between glyphs (a glyph is about 0.9 em wide)
        if st.mono && rng.chance(0.5) {
            st.cell = rng.uniform(1.6, 4.0);
        } else {
            st.mono = false;
            st.spacing = rng.log_uniform(0.6, 3.0);
        }
    }
    if rng.chance(0.15) {
        st.square = true;
    }
    if rng.chance(0.07) {
        st.serif = rng.uniform(0.08, 0.2);
    }
    if rng.chance(0.06) {
        st.wave_amp = rng.uniform(0.04, 0.16);
        st.wave_len = rng.uniform(1.2, 5.0);
        st.wave_ph = rng.uniform(0.0, std::f64::consts::TAU);
        st.wave_along = if rng.chance(0.5) { rng.uniform(0.0, 0.08) } else { 0.0 };
    }
    st
}

// --- geometry helpers ------------------------------------------------------------------------

pub fn densify(pts: &[P], closed: bool, step: f64) -> Vec<P> {
    let mut src: Vec<P> = pts.to_vec();
    if closed && !pts.is_empty() {
        src.push(pts[0]);
    }
    let mut out = vec![src[0]];
    for w in src.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = (b[0] - a[0]).hypot(b[1] - a[1]);
        if d <= 1e-9 {
            continue;
        }
        let n = ((d / step).ceil() as usize).max(1);
        for k in 1..=n {
            let t = k as f64 / n as f64;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    out
}

fn wobble(pts: Vec<P>, amp: f64, rng: &mut Rng) -> Vec<P> {
    if amp <= 0.0 || pts.len() < 2 {
        return pts;
    }
    let mut s = vec![0.0; pts.len()];
    for i in 1..pts.len() {
        s[i] = s[i - 1] + (pts[i][0] - pts[i - 1][0]).hypot(pts[i][1] - pts[i - 1][1]);
    }
    let mut out = pts;
    for _ in 0..2 {
        let f = rng.uniform(2.0, 12.0);
        let a = amp * rng.uniform(0.3, 1.0);
        let (p0, p1) = (rng.uniform(0.0, std::f64::consts::TAU), rng.uniform(0.0, std::f64::consts::TAU));
        for (p, &si) in out.iter_mut().zip(&s) {
            p[0] += a * (f * si + p0).sin();
            p[1] += a * (f * 1.3 * si + p1).sin();
        }
    }
    out
}

type M2 = [[f64; 2]; 2];

fn glyph_matrix(st: &TextStyle, rng: &mut Rng) -> M2 {
    let s = rng.normal(0.0, st.scale_std).exp();
    let a = rng.normal(0.0, st.aspect_std).exp();
    let th = rng.normal(0.0, st.rot_std).to_radians();
    let sh = st.shear + rng.normal(0.0, st.aspect_std);
    let (si, c) = th.sin_cos();
    // rot @ shear @ scale
    let rs = [[c, c * sh - si], [si, si * sh + c]];
    let (sx, sy) = (s * a, s / a);
    [[rs[0][0] * sx, rs[0][1] * sy], [rs[1][0] * sx, rs[1][1] * sy]]
}

fn transform_stroke(s: &Stroke, m: &M2) -> Stroke {
    let mut o = s.clone();
    o.pts = s.pts.iter().map(|p| [m[0][0] * p[0] + m[0][1] * p[1], m[1][0] * p[0] + m[1][1] * p[1]]).collect();
    if s.dot {
        o.r *= (m[0][0] * m[1][1] - m[0][1] * m[1][0]).abs().sqrt();
    }
    o
}

fn jitter_stroke(mut s: Stroke, st: &TextStyle, rng: &mut Rng) -> Stroke {
    if s.dot {
        let (dx, dy) = (rng.normal(0.0, st.stroke_jitter * 0.5), rng.normal(0.0, st.stroke_jitter * 0.5));
        s.pts[0][0] += dx;
        s.pts[0][1] += dy;
        let e = if st.kind != StyleKind::Clean { rng.normal(0.0, 0.06) } else { rng.normal(0.0, 0.0) };
        s.r *= st.dot_scale * e.exp();
        return s;
    }
    let mut pts = std::mem::take(&mut s.pts);
    if st.stroke_jitter > 0.0 {
        let j = st.stroke_jitter;
        let n = pts.len() as f64;
        let c = [pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n];
        let th = rng.normal(0.0, j * 1.5);
        let (si, co) = th.sin_cos();
        let (ox, oy) = (rng.normal(0.0, j), rng.normal(0.0, j));
        for p in pts.iter_mut() {
            let (x, y) = (p[0] - c[0], p[1] - c[1]);
            *p = [co * x - si * y + c[0] + ox, si * x + co * y + c[1] + oy];
        }
        if !s.closed && pts.len() >= 2 {
            let l = pts.len();
            for (end, nb) in [(0usize, 1usize), (l - 1, l - 2)] {
                let v = [pts[end][0] - pts[nb][0], pts[end][1] - pts[nb][1]];
                let nrm = v[0].hypot(v[1]);
                if nrm > 1e-6 {
                    let k = rng.normal(0.005, j) / nrm;
                    pts[end][0] += v[0] * k;
                    pts[end][1] += v[1] * k;
                }
            }
        }
    }
    if st.wobble > 0.0 {
        let mut w = wobble(densify(&pts, s.closed, 0.03), st.wobble, rng);
        if s.closed {
            w.pop();
        }
        pts = w;
    }
    s.pts = pts;
    s
}

fn mark_stroke(mark: Mark, bb: [f64; 4], scale: f64) -> Stroke {
    let g = glyph_set().mark_geom(mark);
    Stroke {
        dot: true,
        pts: vec![[bb[2] + g.dx * scale, bb[3] + g.dy * scale]],
        closed: false,
        r: g.r * scale,
        width_scale: g.width_scale,
        hollow: g.hollow,
    }
}

/// Place glyphs along a line in em units; glyph centres sit on the line's axis (cross = 0).
pub fn layout(text: &str, st: &TextStyle, rng: &mut Rng, vertical: bool) -> Vec<Stroke> {
    layout_ex(text, st, rng, vertical, None, None)
}

/// `layout`, plus (v6) digits and `! ? : / . -` drawn as strokes in the digit style `ds`
/// (horizontal lines; without `ds` they are skipped like any unknown character), and the extent
/// of every glyph along the line (char, start, end in em) pushed to `cells` (furigana). With
/// `ds` None and no digits in `text` the result (and the random draws) equal `layout`'s.
pub fn layout_ex(text: &str, st: &TextStyle, rng: &mut Rng, vertical: bool, ds: Option<&crate::digits::DigitStyle>, mut cells: Option<&mut Vec<(char, f64, f64)>>) -> Vec<Stroke> {
    let gs = glyph_set();
    let mut placed = Vec::new();
    let mut pen = 0.0;
    let (adv, crs) = if vertical { (1, 0) } else { (0, 1) };
    let small_scale = if rng.chance(0.65) { rng.uniform(0.55, 0.78) } else { 1.0 };
    let mut last_space = 0.0; // spacing added after the previous glyph
    let mut prev_extra = false;
    let mut prev_cross = false; // v7: a cross keeps clear of the digits next to it
    for ch in text.chars() {
        if ch == ' ' {
            pen += (if st.mono { st.cell } else { 0.0 }) + rng.uniform(0.55, 1.1);
            prev_extra = false;
            prev_cross = false;
            last_space = 0.0;
            continue;
        }
        if let Some(ds) = ds
            && !vertical
            && crate::digits::is_extra(ch) {
                // same per-glyph transform and hand jitter as the glyphs, the line's stroke width
                let m = glyph_matrix(st, rng);
                let mut strokes: Vec<Stroke> = crate::digits::strokes(ch, ds, st.sw).iter().map(|s| jitter_stroke(transform_stroke(s, &m), st, rng)).collect();
                if strokes.is_empty() {
                    continue;
                }
                let ub = strokes_bbox(&strokes);
                let is_x = ch == crate::digits::CROSS;
                if prev_extra && (is_x || prev_cross) {
                    pen += ds.gap.max(0.04) - last_space; // "3×3": tight, never touching
                } else if prev_extra {
                    pen += ds.gap - last_space; // digit pairs sit tight, often touching
                } else if st.mono {
                    pen += (st.cell - 0.95).max(0.0) / 2.0;
                }
                let shift = pen - ub[0];
                let cross = ds.y_off + rng.normal(0.0, st.cross_jitter * 0.5) + st.curve_amp * (st.curve_freq * pen).sin();
                for s in strokes.iter_mut() {
                    for p in s.pts.iter_mut() {
                        p[0] += shift;
                        p[1] += cross;
                    }
                }
                if let Some(c) = cells.as_deref_mut() {
                    c.push((ch, pen, pen + ub[2] - ub[0]));
                }
                let sp = if st.mono { (st.cell - 0.95).max(0.05) / 2.0 } else { st.spacing.min(0.6) * rng.normal(0.0, 0.15).exp() };
                pen += ub[2] - ub[0] + sp;
                last_space = sp;
                prev_extra = true;
                prev_cross = is_x;
                placed.extend(strokes);
                continue;
            }
        let small = gs.small_to_full.contains_key(&ch);
        let full = *gs.small_to_full.get(&ch).unwrap_or(&ch);
        let Some(g) = gs.glyphs.get(&full) else { continue };
        let mut m = glyph_matrix(st, rng);
        if small && small_scale < 1.0 {
            for r in m.iter_mut() {
                r[0] *= small_scale;
                r[1] *= small_scale;
            }
        }
        let k = match &g.weights {
            Some(w) if g.forms.len() > 1 => rng.weighted(w),
            _ => rng.below(g.forms.len()),
        };
        let mut strokes: Vec<Stroke> = g.forms[k].iter().map(|s| jitter_stroke(transform_stroke(s, &m), st, rng)).collect();
        let bb = strokes_bbox(&strokes);
        if let Some(mark) = g.mark {
            // v4: anywhere from tucked against the glyph to well clear of it, always with a gap
            let mut mk = mark_stroke(mark, bb, st.mark_scale);
            let base_dx = gs.mark_geom(mark).dx * st.mark_scale;
            let r_eff = mk.r.max(st.sw * if mk.hollow { 0.65 } else { 0.5 });
            let dx = (base_dx * rng.uniform(0.55, 1.35)).max(st.sw / 2.0 + r_eff + 0.03);
            mk.pts[0][0] += dx - base_dx;
            mk.pts[0][1] += rng.uniform(-0.06, 0.03) * st.mark_scale;
            let sd = 0.02 + st.stroke_jitter * 0.5;
            mk.pts[0][0] += rng.normal(0.0, sd);
            mk.pts[0][1] += rng.normal(0.0, sd);
            if st.mark_small > 0.0 && !mk.hollow {
                // v6: printed Hunter lettering often draws the dakuten as a small dot, at most a
                // stroke wide, close to the glyph's bottom-right corner on the baseline
                mk.r = st.sw * 0.5 * st.mark_small;
                mk.width_scale = st.mark_small;
                mk.pts[0] = [bb[2] + st.sw / 2.0 + rng.uniform(0.02, 0.12) + mk.r, bb[3] + st.sw / 2.0 - mk.r + rng.uniform(-0.05, 0.02)];
            }
            strokes.push(mk);
        }
        if st.serif > 0.0 {
            add_serifs(&mut strokes, st.serif);
        }
        let ub = strokes_bbox(&strokes);
        let (lo, hi) = if adv == 0 { (ub[0], ub[2]) } else { (ub[1], ub[3]) };
        let base_c = if crs == 0 { (bb[0] + bb[2]) / 2.0 } else { (bb[1] + bb[3]) / 2.0 };
        let shift;
        if st.mono {
            shift = pen + st.cell / 2.0 - (lo + hi) / 2.0;
            pen += st.cell;
            last_space = 0.0;
        } else {
            shift = pen - lo;
            let sp = st.spacing * rng.normal(0.0, 0.15).exp();
            pen += (hi - lo) + sp;
            last_space = sp;
        }
        prev_extra = false;
        if let Some(c) = cells.as_deref_mut() {
            c.push((ch, shift + lo, shift + hi));
        }
        let along = if st.mono { pen - st.cell / 2.0 } else { pen };
        let mut cross = -base_c + rng.normal(0.0, st.cross_jitter) + st.curve_amp * (st.curve_freq * along).sin();
        if small && small_scale < 1.0 {
            cross += 0.45 * (1.0 - small_scale); // sit low (horizontal) / to the right (vertical)
        }
        let mut d = [0.0; 2];
        d[adv] = shift;
        d[crs] = cross;
        for s in strokes.iter_mut() {
            for p in s.pts.iter_mut() {
                p[0] += d[0];
                p[1] += d[1];
            }
        }
        placed.extend(strokes);
    }
    if st.wave_amp > 0.0 || st.wave_along > 0.0 {
        warp_wave(&mut placed, st, adv, crs);
    }
    placed
}

/// Short ticks across both ends of every open stroke (serif-like stroke ends).
fn add_serifs(strokes: &mut Vec<Stroke>, len: f64) {
    let mut extra = Vec::new();
    for s in strokes.iter() {
        if s.dot || s.closed || s.pts.len() < 2 {
            continue;
        }
        let l = s.pts.len();
        for (end, nb) in [(0usize, 1usize), (l - 1, l - 2)] {
            let (e, n) = (s.pts[end], s.pts[nb]);
            let d = [e[0] - n[0], e[1] - n[1]];
            let m = d[0].hypot(d[1]);
            if m < 1e-6 {
                continue;
            }
            let v = [-d[1] / m * len / 2.0, d[0] / m * len / 2.0];
            extra.push(Stroke { dot: false, pts: vec![[e[0] - v[0], e[1] - v[1]], [e[0] + v[0], e[1] + v[1]]], closed: false, r: 0.0, width_scale: 0.8 * s.width_scale, hollow: false });
        }
    }
    strokes.extend(extra);
}

/// Bend the whole line: a sine displacement across it (and optionally along it) as a function
/// of the position along the line. Strokes are densified first so straight segments curve.
fn warp_wave(strokes: &mut [Stroke], st: &TextStyle, adv: usize, crs: usize) {
    let k = std::f64::consts::TAU / st.wave_len.max(0.3);
    let f = |p: P| -> P {
        let t = p[adv];
        let mut q = p;
        q[crs] += st.wave_amp * (k * t + st.wave_ph).sin();
        q[adv] += st.wave_along * (1.7 * k * t + 0.5 * st.wave_ph).sin();
        q
    };
    for s in strokes.iter_mut() {
        if s.dot {
            s.pts[0] = f(s.pts[0]);
            continue;
        }
        let mut d = densify(&s.pts, s.closed, 0.04);
        if s.closed {
            d.pop();
        }
        s.pts = d.into_iter().map(f).collect();
    }
}

// --- drawing primitives (pixels) -------------------------------------------------------------

/// What a stroke turns into on paper, in pixels (line-local frame).
#[derive(Clone, Debug)]
pub enum Prim {
    Poly { pts: Vec<P>, closed: bool, w: f64, sq: bool }, // round caps / joins (sq: square, mitred)
    Disk { c: P, rx: f64, ry: f64, ang: f64 },           // filled dot
    Ring { c: P, rx: f64, ry: f64, ang: f64, w: f64 },   // hollow dot (handakuten)
    Stamps { c: Vec<P>, r: Vec<f64> },                   // brush: union of pressure-sized disks
    Fill { polys: Vec<Vec<P>>, bold: f64 },               // font outlines (digits / punctuation inline), dilated by bold / 2
}

fn pressure(n: usize, rng: &mut Rng) -> Vec<f64> {
    let (a, b) = (rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.7));
    let (f, ph) = (rng.uniform(3.0, 9.0), rng.uniform(0.0, 6.0));
    (0..n)
        .map(|i| {
            let t = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
            let p = (a + (1.0 - a) * (t / 0.25).clamp(0.0, 1.0)).min(b + (1.0 - b) * ((1.0 - t) / 0.3).clamp(0.0, 1.0)).min(1.0);
            p * (1.0 + 0.08 * (t * f + ph).sin())
        })
        .collect()
}

/// Strokes already in pixels -> primitives. `sw_px` is the nominal stroke width.
pub fn strokes_to_prims(strokes: &[Stroke], sw_px: f64, brush: bool, rng: &mut Rng) -> Vec<Prim> {
    strokes_to_prims_sq(strokes, sw_px, brush, false, rng)
}

/// As `strokes_to_prims`, with square stroke ends (`sq`, not for brush strokes).
pub fn strokes_to_prims_sq(strokes: &[Stroke], sw_px: f64, brush: bool, sq: bool, rng: &mut Rng) -> Vec<Prim> {
    let mut out = Vec::with_capacity(strokes.len());
    for s in strokes {
        let w = (sw_px * s.width_scale).max(0.25);
        if s.dot {
            let r = s.r.max(if s.hollow { w } else { w * 0.5 });
            let e = rng.uniform(-0.08, 0.08);
            let ang = rng.uniform(0.0, 180.0);
            let (rx, ry) = (r * (1.0 + e), r * (1.0 - e));
            out.push(if s.hollow { Prim::Ring { c: s.pts[0], rx, ry, ang, w } } else { Prim::Disk { c: s.pts[0], rx, ry, ang } });
        } else if !brush {
            out.push(Prim::Poly { pts: s.pts.clone(), closed: s.closed, w, sq });
        } else {
            let dense = densify(&s.pts, s.closed, (w * 0.28).max(0.2));
            let prof = if s.closed { vec![1.0; dense.len()] } else { pressure(dense.len(), rng) };
            let r = prof.iter().map(|&p| (w * 0.5 * p).max(0.125)).collect();
            out.push(Prim::Stamps { c: dense, r });
        }
    }
    out
}

fn ellipse_half(rx: f64, ry: f64, ang: f64) -> (f64, f64) {
    let (s, c) = ang.to_radians().sin_cos();
    (((rx * c).powi(2) + (ry * s).powi(2)).sqrt(), ((rx * s).powi(2) + (ry * c).powi(2)).sqrt())
}

/// Tight box of the ink (stroke geometry including widths), grown by `grow` px.
pub fn prims_bbox(prims: &[Prim], grow: f64) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut add = |x: f64, y: f64, hx: f64, hy: f64| {
        b[0] = b[0].min(x - hx);
        b[1] = b[1].min(y - hy);
        b[2] = b[2].max(x + hx);
        b[3] = b[3].max(y + hy);
    };
    for p in prims {
        match p {
            Prim::Poly { pts, w, sq, .. } => {
                // square ends reach up to w / sqrt(2) from the end point on diagonals
                let h = if *sq { w * 0.62 } else { w / 2.0 } + grow;
                pts.iter().for_each(|q| add(q[0], q[1], h, h))
            }
            Prim::Disk { c, rx, ry, ang } => {
                let (hx, hy) = ellipse_half(rx + grow, ry + grow, *ang);
                add(c[0], c[1], hx, hy)
            }
            Prim::Ring { c, rx, ry, ang, w } => {
                let (hx, hy) = ellipse_half(rx + w / 2.0 + grow, ry + w / 2.0 + grow, *ang);
                add(c[0], c[1], hx, hy)
            }
            Prim::Stamps { c, r } => c.iter().zip(r).for_each(|(q, &rr)| add(q[0], q[1], rr + grow, rr + grow)),
            Prim::Fill { polys, bold } => polys.iter().flatten().for_each(|q| add(q[0], q[1], bold / 2.0 + grow, bold / 2.0 + grow)),
        }
    }
    b
}

/// Coverage of the primitives mapped through the homography `m` (line frame -> page), each
/// grown by `grow` px (outlines), clipped to `clip`. Stroke widths and dot radii use the local
/// scale of `m` at `centre` (good for the mild perspective used here).
pub fn prims_coverage(prims: &[Prim], m: &H3, centre: P, grow: f64, clip: IRect) -> Option<Cov> {
    let (s, rot) = local_frame(m, centre);
    let mut items: Vec<(Path, Draw)> = Vec::with_capacity(prims.len());
    for p in prims {
        match p {
            Prim::Poly { pts, closed, w, sq } => {
                if let Some(path) = poly_path(&apply_all(m, pts), *closed) {
                    let wd = ((w + 2.0 * grow) * s) as f32;
                    items.push((path, if *sq { Draw::StrokeSq(wd) } else { Draw::Stroke(wd) }));
                }
            }
            Prim::Disk { c, rx, ry, ang } => {
                let q = apply(m, *c);
                if let Some(path) = ellipse_path(q[0], q[1], (rx + grow) * s, (ry + grow) * s, ang + rot) {
                    items.push((path, Draw::Fill));
                }
            }
            Prim::Ring { c, rx, ry, ang, w } => {
                let q = apply(m, *c);
                if let Some(path) = ellipse_path(q[0], q[1], rx * s, ry * s, ang + rot) {
                    items.push((path, Draw::Stroke(((w + 2.0 * grow) * s) as f32)));
                }
            }
            Prim::Fill { polys, bold } => {
                let mut pb = PathBuilder::new();
                for poly in polys {
                    let q = apply(m, poly[0]);
                    pb.move_to(q[0] as f32, q[1] as f32);
                    for p in &poly[1..] {
                        let q = apply(m, *p);
                        pb.line_to(q[0] as f32, q[1] as f32);
                    }
                    pb.close();
                }
                if let Some(path) = pb.finish() {
                    let d = bold / 2.0 + grow;
                    if d > 0.01 {
                        items.push((path.clone(), Draw::Stroke((2.0 * d * s) as f32)));
                    }
                    items.push((path, Draw::Fill));
                }
            }
            Prim::Stamps { c, r } => {
                let mut pb = PathBuilder::new();
                for (q, &rr) in c.iter().zip(r) {
                    let q = apply(m, *q);
                    pb.push_circle(q[0] as f32, q[1] as f32, ((rr + grow) * s) as f32);
                }
                if let Some(path) = pb.finish() {
                    items.push((path, Draw::Fill));
                }
            }
        }
    }
    coverage(&items, clip)
}
