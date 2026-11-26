//! Generator v6: digits and `! ? : / . -` drawn as strokes, like the Hunter glyphs around them.
//!
//! Real lettering that mixes Hunter glyphs with numbers (merchandise promos, title cards, model
//! numbers, dates) draws the digits in the same hand as the glyphs: the same ink, the same
//! stroke weight and stroke ends, often tall and condensed, two digits touching. The v5 inline
//! digits came from text fonts in their own weight. Here every digit and punctuation mark is a
//! monoline skeleton (polylines and dots in em, centred on the line axis), so `render::layout`
//! puts it through the same per-glyph transform, hand jitter, stroke width, square / round
//! ends, brush, wave warp and effects as the glyphs.
//!
//! Frame: em units, x along the line, y down, the digit box centred on (0, 0): x in
//! [-w/2, w/2], y in [-h/2, h/2]. `.` sits on the baseline (y = h/2), `:` around the middle.
//!
//! Generator v7 adds the cross `×` (U+00D7), the separator of the 2011 anime's episode titles
//! ("A×ト×B") and of the series name. In Hunter lettering it is two crossing strokes in the
//! line's own weight, most of a glyph high and a little narrower than high: next to `ー` (a
//! stretched X, `>-<`), `め` (an X closed on one side) and `ね` (a bow tie) it is the one to tell
//! apart. Only `--v7` lines contain it, so v6 output is unchanged.

use crate::geom::{bezier, P};
use crate::glyphs::Stroke;
use crate::rng::Rng;

/// The tokens the extended reader vocab adds after the 74 kana + space tokens.
pub const EXTRA: &str = "0123456789!?:/.-";

/// v7: the cross, the vocab's 91st token (after `EXTRA`).
pub const CROSS: char = '\u{d7}';

/// Digits, `! ? : / . -` and (v7) the cross: drawn by `strokes`, never by the glyph traces.
pub fn is_extra(c: char) -> bool {
    EXTRA.contains(c) || c == CROSS
}

/// How the digits of one line are drawn (shared by every digit and mark of the line).
#[derive(Clone, Copy, Debug)]
pub struct DigitStyle {
    pub h: f64,          // digit height, em
    pub w: f64,          // width of the wide digits (0, 2-9), em
    pub gap: f64,        // ink gap between two digits / marks, em (< 0: touching, overlapping)
    pub slant: f64,      // shear of the digits (x += slant * height above the centre)
    pub pslant: f64,     // shear of ! and ?
    pub oval0: bool,     // 0 as an ellipse (else a stadium: straight sides)
    pub one_flag: bool,  // 1 with a flag
    pub one_base: bool,  // 1 with a base bar
    pub four_open: bool, // 4 with an open top
    pub flat3: bool,     // 3 with a flat top
    pub seven_bend: f64, // 0: straight stem of the 7, > 0: curved
    pub wedge: bool,     // ! as a tapered wedge (display lettering)
    pub dot_k: f64,      // dot radius relative to half the stroke width
    pub y_off: f64,      // em, the digits' centre relative to the glyphs' axis
    // v7, the cross (set by `sample_cross`; `sample_style` fills fixed values without drawing)
    pub xh: f64, // cross height, em
    pub xw: f64, // cross width, em
    pub xk: f64, // cross stroke width relative to the line's
    pub xy: f64, // cross centre relative to the digits' centre, em
    pub xr: f64, // where the second stroke crosses the first (0.5: the middle)
}

/// A digit style for a line whose glyph strokes are `sw` em wide.
pub fn sample_style(rng: &mut Rng, sw: f64) -> DigitStyle {
    let mut h = rng.uniform(0.78, 1.1);
    if rng.chance(0.25) {
        h *= rng.uniform(1.05, 1.2); // tall display digits
    }
    let wr = match rng.weighted(&[0.4, 0.38, 0.22]) {
        0 => rng.uniform(0.58, 0.72),
        1 => rng.uniform(0.46, 0.58),
        _ => rng.uniform(0.36, 0.46), // very condensed
    };
    // counters must stay open: a loop needs room inside the stroke
    let h = h.max(3.6 * sw);
    let w = (h * wr).max(2.3 * sw);
    let gap = match rng.weighted(&[0.35, 0.4, 0.25]) {
        0 => -sw * rng.uniform(0.0, 0.5), // touching / overlapping
        1 => rng.uniform(0.0, 0.1),
        _ => rng.uniform(0.1, 0.3),
    };
    let slant = if rng.chance(0.2) { rng.uniform(0.06, 0.22) } else { 0.0 };
    let pslant = if rng.chance(0.55) { rng.uniform(0.15, 0.35) } else { slant };
    DigitStyle {
        h,
        w,
        gap,
        slant,
        pslant,
        oval0: rng.chance(0.4),
        one_flag: rng.chance(0.85),
        one_base: rng.chance(0.3),
        four_open: rng.chance(0.4),
        flat3: rng.chance(0.2),
        seven_bend: if rng.chance(0.4) { rng.uniform(0.05, 0.2) } else { 0.0 },
        wedge: rng.chance(0.4),
        dot_k: rng.uniform(1.0, 1.6),
        y_off: rng.normal(0.0, 0.03),
        xh: 0.75 * h,
        xw: 0.65 * h,
        xk: 1.0,
        xy: 0.0,
        xr: 0.5,
    }
}

/// v7: how the line's crosses are drawn. Real title lettering draws `×` most of a glyph high
/// (0.55-0.95 em; a quarter of the lines smaller, like a font's `×`), 0.6-1.05 of its height
/// wide, in the glyphs' stroke weight (a fifth heavier, like a display `✕`), on the line's axis.
pub fn sample_cross(rng: &mut Rng, ds: &mut DigitStyle, sw: f64) {
    let xh = if rng.chance(0.25) { rng.uniform(0.36, 0.55) } else { rng.uniform(0.55, 0.95) };
    ds.xh = xh.max(2.4 * sw);
    ds.xw = (ds.xh * rng.uniform(0.6, 1.05)).max(2.0 * sw);
    ds.xk = if rng.chance(0.2) { rng.uniform(1.15, 1.6) } else { rng.uniform(0.85, 1.1) };
    ds.xy = rng.normal(0.0, 0.025) - ds.y_off; // on the glyphs' axis rather than the digits'
    ds.xr = if rng.chance(0.2) { rng.uniform(0.4, 0.6) } else { 0.5 };
}

fn line(pts: Vec<P>) -> Stroke {
    Stroke { dot: false, pts, closed: false, r: 0.0, width_scale: 1.0, hollow: false }
}

fn closed(pts: Vec<P>) -> Stroke {
    Stroke { dot: false, pts, closed: true, r: 0.0, width_scale: 1.0, hollow: false }
}

fn dot(c: P, r: f64) -> Stroke {
    Stroke { dot: true, pts: vec![c], closed: false, r, width_scale: 1.0, hollow: false }
}

/// Points on an ellipse arc, angles in degrees (y down: 90 is the bottom, 270 the top).
fn arc(cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64, n: usize) -> Vec<P> {
    (0..=n)
        .map(|k| {
            let a = (a0 + (a1 - a0) * k as f64 / n as f64).to_radians();
            [cx + rx * a.cos(), cy + ry * a.sin()]
        })
        .collect()
}

/// Skeleton of `c` in em (see the module docs); `sw` is the line's stroke width in em.
pub fn strokes(c: char, ds: &DigitStyle, sw: f64) -> Vec<Stroke> {
    let (h, w) = (ds.h, ds.w);
    let (l, r, t, b) = (-w / 2.0, w / 2.0, -h / 2.0, h / 2.0);
    let rr = w / 2.0;
    let dot_r = (0.5 * sw * ds.dot_k).max(0.04);
    let mut out: Vec<Stroke> = match c {
        '0' => {
            if ds.oval0 || h <= w {
                vec![closed(arc(0.0, 0.0, rr, h / 2.0, 0.0, 360.0, 40)[..40].to_vec())]
            } else {
                let mut p = arc(0.0, t + rr, rr, rr, 180.0, 360.0, 16);
                p.extend(arc(0.0, b - rr, rr, rr, 0.0, 180.0, 16));
                vec![closed(p)]
            }
        }
        '1' => {
            let xs = 0.12 * w;
            let mut p = Vec::new();
            if ds.one_flag {
                p.push([xs - 0.42 * w, t + 0.26 * h]);
            }
            p.push([xs, t]);
            p.push([xs, b]);
            let mut v = vec![line(p)];
            if ds.one_base {
                v.push(line(vec![[xs - 0.42 * w, b], [xs + 0.42 * w, b]]));
            }
            v
        }
        '2' => {
            let ry = rr.min(0.28 * h);
            let mut p = arc(0.0, t + ry, rr, ry, 195.0, 390.0, 18);
            p.push([l, b]);
            p.push([r, b]);
            vec![line(p)]
        }
        '3' => {
            let lower = arc(0.0, b - h / 4.0, rr, h / 4.0, if ds.flat3 { 250.0 } else { 270.0 }, 520.0, 20);
            if ds.flat3 {
                let mut p = vec![[l * 0.85, t], [r * 0.85, t]];
                p.push(lower[0]);
                p.extend(&lower[1..]);
                vec![line(p)]
            } else {
                vec![line(arc(0.0, t + h / 4.0, 0.9 * rr, h / 4.0, 200.0, 450.0, 18)), line(lower)]
            }
        }
        '4' => {
            let (xs, yc) = (0.22 * w, 0.18 * h);
            if ds.four_open {
                vec![line(vec![[-0.05 * w, t], [l, yc], [r + 0.04 * w, yc]]), line(vec![[xs, t + 0.3 * h], [xs, b]])]
            } else {
                vec![line(vec![[xs, b], [xs, t], [l, yc], [r + 0.04 * w, yc]])]
            }
        }
        '5' => {
            let (cy, ry) = (b - 0.3 * h, 0.3 * h);
            let mut p = vec![[r * 0.85, t], [l * 0.75, t], [l * 0.85, -0.05 * h]];
            p.extend(arc(0.0, cy, rr, ry, 215.0, 520.0, 20));
            vec![line(p)]
        }
        '6' | '9' => {
            let (cy, ry) = (b - 0.3 * h, 0.3 * h);
            let mut v = vec![closed(arc(0.0, cy, rr, ry, 0.0, 360.0, 32)[..32].to_vec())];
            v.push(line(bezier(&[[r * 0.7, t], [l * 0.2, t - 0.02 * h], [l * 1.05, -0.15 * h], [l, cy]], 14)));
            if c == '9' {
                for s in v.iter_mut() {
                    s.pts.iter_mut().for_each(|p| *p = [-p[0], -p[1]]);
                }
            }
            v
        }
        '7' => {
            let mut p = vec![[l, t], [r, t]];
            if ds.seven_bend > 0.0 {
                let k = ds.seven_bend;
                p.extend(&bezier(&[[r, t], [r - k * w, t + 0.35 * h], [-0.1 * w + k * w, 0.15 * h], [-0.12 * w, b]], 14)[1..]);
            } else {
                p.push([-0.15 * w, b]);
            }
            vec![line(p)]
        }
        '8' => {
            let (ru, rl) = (0.23 * h, 0.27 * h);
            vec![
                closed(arc(0.0, t + ru, 0.86 * rr, ru, 0.0, 360.0, 28)[..28].to_vec()),
                closed(arc(0.0, b - rl, rr, rl, 0.0, 360.0, 32)[..32].to_vec()),
            ]
        }
        '!' => {
            let y1 = b - 2.0 * dot_r - 0.16 * h;
            let mut v = if ds.wedge {
                // a tapered bar: strokes fanning out towards the top
                let s = 0.45 * sw;
                vec![line(vec![[-s, t], [0.0, y1]]), line(vec![[0.0, t], [0.0, y1]]), line(vec![[s, t], [0.0, y1]])]
            } else {
                vec![line(vec![[0.0, t], [0.0, y1]])]
            };
            v.push(dot([0.0, b - dot_r], dot_r));
            v
        }
        '?' => {
            let ry = 0.25 * h;
            let mut p = arc(0.0, t + ry, 0.85 * rr, ry, 200.0, 450.0, 18);
            let y1 = b - 2.0 * dot_r - 0.16 * h;
            p.push([0.0, y1.max(t + 2.0 * ry + 0.05 * h)]);
            vec![line(p), dot([0.0, b - dot_r], dot_r)]
        }
        ':' => {
            let d = 0.2 * h;
            vec![dot([0.0, -d], dot_r), dot([0.0, d], dot_r)]
        }
        '/' => vec![line(vec![[0.38 * w, t], [-0.38 * w, b]])],
        '.' => vec![dot([0.0, b - dot_r], dot_r)],
        '-' => vec![line(vec![[-0.3 * w, 0.02 * h], [0.3 * w, 0.02 * h]])],
        CROSS => {
            // two strokes; hand lettering now and then crosses them off-centre (`xr`)
            let (hw, hh, y) = (ds.xw / 2.0, ds.xh / 2.0, ds.xy);
            let d = (ds.xr - 0.5) * 2.0 * hw;
            let mut v = vec![line(vec![[-hw, y - hh], [hw, y + hh]]), line(vec![[hw + d, y - hh], [-hw + d, y + hh]])];
            for s in v.iter_mut() {
                s.width_scale = ds.xk;
            }
            v
        }
        _ => vec![],
    };
    let k = if c == '!' || c == '?' { ds.pslant } else { ds.slant };
    if k != 0.0 {
        for s in out.iter_mut() {
            s.pts.iter_mut().for_each(|p| p[0] -= k * p[1]);
        }
    }
    out
}

// --- inline content ---------------------------------------------------------------------------

/// A number the way real pictures carry them: model numbers (07, 001), counts, years, dates,
/// times, prices, ranges, decimals, fractions. Only vocab characters (no Latin letters, commas or
/// currency signs: those cannot be labelled).
pub fn number_token(rng: &mut Rng) -> String {
    match rng.weighted(&[0.16, 0.1, 0.07, 0.1, 0.11, 0.12, 0.1, 0.07, 0.06, 0.05, 0.06]) {
        0 => format!("{:02}", rng.int(0, 100)),
        1 => rng.int(0, 10).to_string(),
        2 => format!("{:03}", rng.int(0, 1000)),
        3 => rng.int(10, 1000).to_string(),
        4 => rng.int(1950, 2031).to_string(),
        5 => {
            let (m, d) = (rng.int(1, 13), rng.int(1, 32));
            match rng.below(4) {
                0 | 1 => format!("{m}/{d}"),
                2 => format!("{m}.{d}"),
                _ => format!("{}/{m}/{d}", rng.int(1990, 2031)),
            }
        }
        6 => format!("{}:{:02}", rng.int(0, 24), rng.int(0, 60)),
        7 => (rng.int(1, 1000) * [1, 10, 100][rng.below(3)]).to_string(),
        8 => format!("{}-{}", rng.int(1, 100), rng.int(1, 100)),
        9 => format!("{}.{}", rng.int(0, 100), rng.int(0, 10)),
        _ => format!("{}/{}", rng.int(1, 10), rng.int(2, 13)),
    }
}

/// Sentence-end punctuation: ! !! !? ?! ? !!! ?? ...
pub fn punct_token(rng: &mut Rng) -> String {
    ["!", "!!", "!?", "?!", "?", "!!!", "??", "..."][rng.weighted(&[0.25, 0.25, 0.12, 0.12, 0.14, 0.06, 0.03, 0.03])].to_string()
}

/// The parts of a Hunter line with numbers and / or punctuation in it, in order: (is kana, text).
/// Numbers go at the start, in the middle or at the end, punctuation mostly at the end; spaces
/// around a number now and then. `kana` is the line's kana text (no digits).
pub fn inline_parts(rng: &mut Rng, kana: &str) -> Vec<(bool, String)> {
    let k: Vec<char> = kana.chars().collect();
    let (num, punct) = match rng.weighted(&[0.45, 0.3, 0.25]) {
        0 => (Some(number_token(rng)), None),
        1 => (None, Some(punct_token(rng))),
        _ => (Some(number_token(rng)), Some(punct_token(rng))),
    };
    let sp = |rng: &mut Rng| if rng.chance(0.25) { " " } else { "" };
    let mut parts: Vec<(bool, String)> = Vec::new();
    let mut end_kana = String::new();
    if let Some(n) = num {
        let wh = rng.weighted(&[0.25, 0.4, 0.35]);
        if wh == 0 || k.is_empty() {
            parts.push((false, format!("{n}{}", sp(rng))));
            end_kana = kana.to_string();
        } else if wh == 1 && k.len() >= 2 {
            let cut = rng.int(1, k.len() as i64) as usize;
            let (s1, s2) = (sp(rng), sp(rng));
            parts.push((true, k[..cut].iter().collect()));
            parts.push((false, format!("{s1}{n}{s2}")));
            end_kana = k[cut..].iter().collect();
        } else {
            parts.push((true, kana.to_string()));
            parts.push((false, format!("{}{n}", sp(rng))));
        }
    } else {
        end_kana = kana.to_string();
    }
    if !end_kana.is_empty() {
        parts.push((true, end_kana));
    }
    if let Some(p) = punct {
        // mostly at the very end; now and then after a word in the middle ("はい!いこう")
        let last = parts.len() - 1;
        if parts[last].0 && rng.chance(0.15) {
            let kk: Vec<char> = parts[last].1.chars().collect();
            if kk.len() >= 3 {
                let cut = rng.int(1, kk.len() as i64 - 1) as usize;
                parts[last] = (true, kk[..cut].iter().collect());
                parts.push((false, p));
                parts.push((true, kk[cut..].iter().collect()));
                return parts;
            }
        }
        parts.push((false, p));
    }
    parts
}

/// The parts of a v7 title line, in order: (is kana, text). `words` are 1-4 kana words; the
/// crosses go between them the way the anime's titles do it: "A×と×B" (a particle between two
/// crosses, about half the lines), "A×B", "A×B×C", "A×B×C×D", now and then counts ("A×3",
/// "3×3") and a leading or trailing cross. A word before a cross ends in `ー` a third of the time
/// and `め` / `ね` sit next to one now and then, so the reader learns the cross against the glyphs
/// that look like it. Spaces around the crosses 8% of the time (labelled: a wide gap).
pub fn cross_parts(rng: &mut Rng, words: &[String]) -> Vec<(bool, String)> {
    let x = CROSS.to_string();
    let sep = if rng.chance(0.08) { format!(" {x} ") } else { x.clone() };
    let mut w: Vec<String> = words.iter().filter(|s| !s.is_empty()).cloned().collect();
    if w.is_empty() {
        w.push("はんたー".into());
    }
    // the glyphs that look like the cross, right before it
    let tweak = |rng: &mut Rng, s: &mut String| {
        let u = rng.f64();
        if u < 0.33 {
            if !s.ends_with('ー') {
                s.push('ー');
            }
        } else if u < 0.4 {
            s.push('め');
        } else if u < 0.45 {
            s.push('ね');
        }
    };
    let mut parts: Vec<(bool, String)> = Vec::new();
    match rng.weighted(&[0.48, 0.24, 0.13, 0.04, 0.05, 0.03, 0.03]) {
        0 => {
            // A×と×B
            let p = ["と", "の", "も", "や", "で", "が", "に", "を", "へ", "は"][rng.weighted(&[0.45, 0.15, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.05])];
            let mut a = w[0].clone();
            tweak(rng, &mut a);
            let b = w.get(1).cloned().unwrap_or_else(|| w[0].clone());
            parts.push((true, a));
            parts.push((false, sep.clone()));
            parts.push((true, p.to_string()));
            parts.push((false, sep));
            parts.push((true, b));
        }
        k @ 1..=3 => {
            // A×B, A×B×C, A×B×C×D
            let n = k + 1;
            for i in 0..n {
                let mut s = w[i % w.len()].clone();
                if i + 1 < n {
                    tweak(rng, &mut s);
                }
                parts.push((true, s));
                if i + 1 < n {
                    parts.push((false, sep.clone()));
                }
            }
        }
        4 => {
            // a count after a word: A×3, A×10
            let mut a = w[0].clone();
            tweak(rng, &mut a);
            let hi = if rng.chance(0.7) { 10 } else { 100 };
            let n = rng.int(1, hi);
            parts.push((true, a));
            parts.push((false, format!("{x}{n}")));
        }
        5 => {
            // a product of numbers, after or before a word: 3×3, 10×10
            let (a, b) = (rng.int(1, 13), rng.int(1, 13));
            if rng.chance(0.5) {
                parts.push((true, w[0].clone()));
                parts.push((false, format!(" {a}{x}{b}")));
            } else {
                parts.push((false, format!("{a}{x}{b}")));
                parts.push((true, w[0].clone()));
            }
        }
        _ => {
            // a leading or trailing cross: ×A, A×
            if rng.chance(0.5) {
                parts.push((false, x.clone()));
                parts.push((true, w[0].clone()));
            } else {
                let mut a = w[0].clone();
                tweak(rng, &mut a);
                parts.push((true, a));
                parts.push((false, x.clone()));
            }
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyphs::strokes_bbox;

    #[test]
    fn every_extra_char_has_a_skeleton_inside_its_box() {
        let mut rng = Rng::from_key(&[1, 2, 3]);
        for _ in 0..200 {
            let sw = rng.uniform(0.04, 0.3);
            let mut ds = sample_style(&mut rng, sw);
            sample_cross(&mut rng, &mut ds, sw);
            for c in EXTRA.chars().chain([CROSS]) {
                let s = strokes(c, &ds, sw);
                assert!(!s.is_empty(), "{c}");
                let b = strokes_bbox(&s);
                let slack = 0.05 + ds.slant.abs().max(ds.pslant.abs()) * ds.h + sw;
                // the cross has its own box: xw x xh around (0, xy)
                let (w, h0, h1) = if c == CROSS { (ds.xw, ds.xy - ds.xh / 2.0, ds.xy + ds.xh / 2.0) } else { (ds.w, -ds.h / 2.0, ds.h / 2.0) };
                assert!(b[1] >= h0 - slack && b[3] <= h1 + slack, "{c} {b:?} h {}", ds.h);
                assert!(b[0] >= -w / 2.0 - slack && b[2] <= w / 2.0 + slack, "{c} {b:?} w {w}");
            }
        }
    }

    #[test]
    fn inline_parts_keep_the_kana_and_use_only_vocab_tokens() {
        let mut rng = Rng::from_key(&[4, 5]);
        for _ in 0..2000 {
            let parts = inline_parts(&mut rng, "びーとるがた");
            let kana: String = parts.iter().filter(|p| p.0).map(|p| p.1.as_str()).collect();
            assert_eq!(kana, "びーとるがた");
            for (is_kana, t) in &parts {
                if !is_kana {
                    assert!(t.chars().all(|c| is_extra(c) || c == ' '), "{t}");
                    assert!(t.chars().any(is_extra));
                }
            }
        }
    }

    #[test]
    fn cross_parts_keep_the_words_and_put_a_cross_in_every_line() {
        let mut rng = Rng::from_key(&[6, 7]);
        let words: Vec<String> = ["はんたー", "と", "めね"].iter().map(|s| s.to_string()).collect();
        let mut beside_long = 0;
        for _ in 0..3000 {
            let parts = cross_parts(&mut rng, &words);
            let text: String = parts.iter().map(|p| p.1.as_str()).collect();
            assert!(text.contains(CROSS), "{text}");
            assert!(!text.starts_with(' ') && !text.ends_with(' ') && !text.contains("  "), "{text:?}");
            for (is_kana, t) in &parts {
                if *is_kana {
                    assert!(!t.is_empty() && !t.chars().any(is_extra) && !t.contains(' '), "{t}");
                } else {
                    assert!(t.chars().all(|c| is_extra(c) || c == ' ') && t.chars().any(is_extra), "{t}");
                }
            }
            beside_long += text.contains("ー\u{d7}") as usize;
        }
        assert!(beside_long > 300, "{beside_long}");
    }
}
