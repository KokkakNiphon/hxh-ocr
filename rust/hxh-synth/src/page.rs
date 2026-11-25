//! Page composer: manga / anime-style grayscale pages full of Hunter-alphabet text lines, with
//! one tight quad per line.
//!
//! A page (every choice random per page):
//! * layout: "manga" (paper page cut into bordered panels with gutters, each panel with its own
//!   background: white, flat grey, screentone, gradient tone, value noise, speed lines,
//!   hatching, black, blurred scene), "frame" (anime still / poster: one full-bleed scene,
//!   mostly landscape) or "doc" (paper, sometimes ruled, with paragraphs, labels, tables);
//! * text groups sharing one style and one transform (rotation within +-12 degrees, occasional
//!   perspective), ~10% vertical (top to bottom, glyphs upright): speech bubbles (ellipse /
//!   rounded box / cloud / burst, mostly with a tail, 1-3 short lines, some black with white
//!   text), captions, signs / cards / labels (border, drop shadow), paragraphs, free lines on
//!   the art (titles, sound effects; outlined when the background is busy or mid-grey),
//!   small filler lines, table cells;
//! * non-text art: curves, hair strands, filled blobs, rings, polygons, grids, glyph-sized
//!   scribbles, sparkles, rays. Art drawn before the text may sit under free text; art drawn
//!   after the text starts right at line ends (blobs, spiky hair, strokes: the line-end
//!   confusion seen on the real title cards) but is clipped out of every line's quad;
//! * page degradations: blur, resampling, uneven light, contrast / gamma, inversion, noise, JPEG.
//!
//! Quads come from the stroke geometry: each line's tight box (strokes including their width,
//! dot radii, brush stamps) in its own frame, pushed through the line's homography, so they hug
//! the glyph ink (outlines and shadows are decoration, outside). Corner order: clockwise from
//! the top-left in reading orientation (`geom::box_quad`). Lines never overlap: placed lines and
//! containers are stamped into an occupancy mask and new groups only go where their footprint
//! (container, or quads grown by a margin) is free.
//!
//! Foreign text and icons (unlabelled distractors, `FOREIGN.md`): real system fonts and icon
//! fonts rendered through the same groups, containers, transforms and degradations, as whole
//! groups, mixed groups (Hunter and foreign lines in one bubble), extra signage / labels /
//! subtitles / icon rows, and the "web" layout (`web.rs`). Every foreign run keeps
//! `keep_margin` line heights (the larger of the two) away from every labelled quad, checked
//! exactly on the quads (`clear_of`), so the targets never contradict the image. Foreign runs
//! are reported in `Page::foreign`, never in `lines`.

use serde::Serialize;
use std::f64::consts::{PI, TAU};
use tiny_skia::{Path, PathBuilder, Transform};

use crate::codec::{decode_jpeg_gray, encode_jpeg};
use crate::fonts::{Class, FontBank};
use crate::foreign::{Role, Script, TextGen, SCRIPTS};
use crate::geom::*;
use crate::glyphs::strokes_bbox;
use crate::raster::*;
use crate::render::{layout, prims_bbox, prims_coverage, sample_style, sample_style_v5, strokes_to_prims_sq, Prim, TextStyle};
use crate::rng::Rng;
use crate::text::{label_of, TextSampler};

#[derive(Clone, Debug, Serialize)]
pub struct PageConfig {
    pub long_min: usize,
    pub long_max: usize,
    pub portrait_prob: f64,
    pub mode_p: [f64; 8], // manga, frame, doc, web, card, lyric, tv (v5), promo (v6)
    pub vertical_prob: f64, // base share of vertical groups (scaled per element kind)
    pub em_min: f64,        // nominal glyph size, px per em (a line's tight height is ~0.9-1.1 em)
    pub em_max: f64,
    pub empty_prob: f64,    // negatives: no Hunter text (foreign text / icons / art only)
    pub min_stroke_px: f64, // thinnest stroke drawn
    pub foreign_prob: f64,  // share of text groups written in foreign text instead
    pub mixed_prob: f64,    // share of bubbles / captions / signs / blocks mixing both
    pub foreign_extra: f64, // multiplier on the extra foreign-only elements (labels, signage, ...)
    pub icon_extra: f64,    // multiplier on the icon elements
    pub ui_prob: f64,       // share of web pages that are app / UI screenshots (icon-heavy)
    pub foreign_em_min: f64, // smallest foreign font size, px per em
    pub keep_margin: f64,   // min gap foreign <-> labelled quad, in line heights (larger of the two)
    // v5: photo-style pages for the model family (all off by default, so v4 settings stay as they were)
    pub v5: bool,           // v5 text styles and effects (wide spacing, square ends, serifs, waves, glow, bevel, hollow)
    pub photo_prob: f64,    // share of pages photographed off a screen (perspective, moire, glare, blur, ...)
    pub ui_neg_prob: f64,   // share of pages with avatars, round icons and toolbar strips as hard negatives
    pub caption_prob: f64,  // share of free / sign Hunter groups with a Japanese or Latin caption right under or above
    pub inline_prob: f64,   // share of horizontal Hunter lines with digits / ! ? : / . - written inline (extended vocab)
    // v6 (digits and punctuation as part of the lettering, reading guides, sticker lettering): all off by default
    pub v6: bool,           // inline digits drawn like the glyphs (strokes) or weight-matched display fonts; realistic numbers
    pub ruby_prob: f64,     // share of horizontal free / sign / caption Hunter groups with a kana reading guide (furigana) above
    pub sticker_prob: f64,  // share of free-standing Hunter text with a thick outline (white around dark ink, or the inverse) and its soft shadow
    pub small_mark_prob: f64, // share of Hunter text styles with small dakuten dots (0.55-1 stroke widths) tucked at the corner
}

impl Default for PageConfig {
    fn default() -> Self {
        PageConfig {
            long_min: 512,
            long_max: 1024,
            portrait_prob: 0.6,
            mode_p: [0.5, 0.21, 0.14, 0.15, 0.0, 0.0, 0.0, 0.0],
            vertical_prob: 0.11,
            em_min: 11.0,
            em_max: 80.0,
            empty_prob: 0.03,
            min_stroke_px: 0.8,
            foreign_prob: 0.15,
            mixed_prob: 0.08,
            foreign_extra: 1.0,
            icon_extra: 1.0,
            ui_prob: 0.25,
            foreign_em_min: 6.0,
            keep_margin: 0.3,
            v5: false,
            photo_prob: 0.0,
            ui_neg_prob: 0.0,
            caption_prob: 0.0,
            inline_prob: 0.0,
            v6: false,
            ruby_prob: 0.0,
            sticker_prob: 0.0,
            small_mark_prob: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LineOut {
    pub quad: [[f64; 2]; 4],
    pub text: String,
    pub orient: &'static str,
    pub kind: &'static str,
}

/// An unlabelled foreign-text or icon run.
#[derive(Clone, Debug, Serialize)]
pub struct ForeignOut {
    pub quad: [[f64; 2]; 4],
    pub text: String,
    pub script: &'static str,
    pub font: String,
    pub orient: &'static str,
    pub kind: &'static str,
}

pub struct Page {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u8>,
    pub lines: Vec<LineOut>,
    pub foreign: Vec<ForeignOut>,
    pub layout: &'static str,
    pub photo: bool, // photographed off a screen (v5)
}

pub const LAYOUTS: [&str; 8] = ["manga", "frame", "doc", "web", "card", "lyric", "tv", "promo"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Bubble,
    Caption,
    Sign,
    Block,
    Free,
    Filler,
    Cell,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Bubble => "bubble",
            Kind::Caption => "caption",
            Kind::Sign => "sign",
            Kind::Block => "block",
            Kind::Free => "free",
            Kind::Filler => "filler",
            Kind::Cell => "cell",
        }
    }
    fn order(self) -> u8 {
        match self {
            Kind::Free => 0,
            Kind::Block => 1,
            Kind::Bubble => 2,
            Kind::Sign => 3,
            Kind::Caption => 4,
            Kind::Cell => 5,
            Kind::Filler => 6,
        }
    }
}

#[derive(Clone, Copy)]
enum Rot {
    Flat,
    Small,
    Wide,
}

/// em: (lo, hi cap, hi as a fraction of the region's short side); n / np: line counts and
/// probabilities; len: max text length range; ext: max line extent (fraction of the region);
/// gap: line gap (em); align: l / c / r (top / centre / bottom for vertical); rot: rotation
/// profile; persp: perspective probability; vmul: vertical probability multiplier.
struct Spec {
    em: (f64, f64, f64),
    n: &'static [usize],
    np: &'static [f64],
    len: (i64, i64),
    ext: f64,
    gap: (f64, f64),
    align: &'static [u8],
    rot: Rot,
    persp: f64,
    vmul: f64,
}

fn spec(k: Kind) -> Spec {
    match k {
        Kind::Bubble => Spec { em: (11.0, 44.0, 0.2), n: &[1, 2, 3], np: &[0.35, 0.4, 0.25], len: (2, 10), ext: 0.6, gap: (0.1, 0.5), align: b"c", rot: Rot::Small, persp: 0.03, vmul: 1.4 },
        Kind::Caption => Spec { em: (11.0, 30.0, 1.0 / 6.0), n: &[1, 2, 3], np: &[0.45, 0.35, 0.2], len: (4, 13), ext: 0.8, gap: (0.15, 0.5), align: b"lc", rot: Rot::Flat, persp: 0.0, vmul: 1.4 },
        Kind::Sign => Spec { em: (12.0, 64.0, 1.0 / 3.0), n: &[1, 2, 3], np: &[0.7, 0.25, 0.05], len: (1, 11), ext: 0.8, gap: (0.15, 0.5), align: b"cl", rot: Rot::Wide, persp: 0.3, vmul: 0.5 },
        Kind::Block => Spec { em: (11.0, 28.0, 0.125), n: &[2, 3, 4, 5, 6, 8], np: &[0.2, 0.25, 0.2, 0.15, 0.12, 0.08], len: (6, 21), ext: 0.9, gap: (0.15, 0.9), align: b"llc", rot: Rot::Small, persp: 0.1, vmul: 1.0 },
        Kind::Free => Spec { em: (14.0, 80.0, 0.4), n: &[1], np: &[1.0], len: (2, 13), ext: 0.9, gap: (0.2, 0.5), align: b"c", rot: Rot::Wide, persp: 0.15, vmul: 0.6 },
        Kind::Filler => Spec { em: (11.0, 26.0, 0.25), n: &[1], np: &[1.0], len: (1, 11), ext: 0.6, gap: (0.2, 0.5), align: b"c", rot: Rot::Wide, persp: 0.1, vmul: 0.6 },
        Kind::Cell => Spec { em: (11.0, 30.0, 1.0 / 1.4), n: &[1], np: &[1.0], len: (1, 9), ext: 0.92, gap: (0.2, 0.5), align: b"c", rot: Rot::Flat, persp: 0.0, vmul: 0.3 },
    }
}

// --- lines and groups ------------------------------------------------------------------------

/// One foreign run of a line: its own record and quad (a line can hold an icon and a label).
pub(crate) struct Run {
    pub face: usize,
    pub script: Script,
    pub text: String,
    pub b: [f64; 4], // tight box, line frame (px), including dilation
}

pub(crate) enum Ink {
    Hxh(Vec<Prim>),
    /// Font outlines (line frame, px); `bold` dilates by bold / 2, `hollow` > 0 draws only the
    /// contours with that width (outlined icons).
    Font { contours: Vec<Vec<P>>, runs: Vec<Run>, bold: f64, hollow: f64 },
}

pub(crate) struct LineG {
    pub raw: String,
    pub vertical: bool,
    pub ink: Ink,
    pub b: [f64; 4], // tight ink box, line frame (px)
    pub cells: Vec<(char, f64, f64)>, // v6: each glyph's extent along the line (px, line frame), for furigana
    pub ruby: bool, // v6: a reading guide drawn in its Hunter line's frame (keep-clear rule against that line waived)
}

impl LineG {
    pub fn hxh(&self) -> bool {
        matches!(self.ink, Ink::Hxh(_))
    }
}

pub(crate) struct Group {
    pub lines: Vec<LineG>,
    pub offs: Vec<P>, // line frame -> group coords translation
    pub w: f64,       // text block size
    pub h: f64,
    pub vertical: bool,
    pub em: f64,
}

#[allow(clippy::too_many_arguments)]
fn make_line(rng: &mut Rng, sampler: &TextSampler, st: &TextStyle, em: f64, vertical: bool, max_len: usize, max_extent: f64, min_px: f64) -> Option<LineG> {
    let mut max_len = max_len.max(1);
    let mut found = None;
    for _ in 0..5 {
        let raw = sampler.sample(rng, 1, max_len);
        let mut cells = Vec::new();
        let strokes = crate::render::layout_ex(&raw, st, rng, vertical, None, Some(&mut cells));
        if strokes.is_empty() {
            continue;
        }
        let bb = strokes_bbox(&strokes);
        let along = (if vertical { bb[3] - bb[1] } else { bb[2] - bb[0] }) * em + st.sw * em;
        if along <= max_extent {
            found = Some((raw, strokes, cells));
            break;
        }
        if max_len == 1 {
            return None;
        }
        max_len = (max_len - 1).min((raw.chars().count() as f64 * max_extent / along * 0.9) as usize).max(1);
    }
    let (raw, strokes, cells) = found?;
    let mut l = line_from_strokes(rng, raw, strokes, st, em, vertical, min_px);
    l.cells = cells.into_iter().map(|(c, a, b)| (c, a * em, b * em)).collect();
    Some(l)
}

fn line_from_strokes(rng: &mut Rng, raw: String, mut strokes: Vec<crate::glyphs::Stroke>, st: &TextStyle, em: f64, vertical: bool, min_px: f64) -> LineG {
    for s in strokes.iter_mut() {
        for p in s.pts.iter_mut() {
            p[0] *= em;
            p[1] *= em;
        }
        s.r *= em;
    }
    let prims = strokes_to_prims_sq(&strokes, (st.sw * em).max(min_px), st.brush, st.square && !st.brush, rng);
    let b = prims_bbox(&prims, 0.0);
    LineG { raw, vertical, ink: Ink::Hxh(prims), b, cells: Vec::new(), ruby: false }
}

/// A Hunter line with the given text (scenes: names on cards, lyrics); None if nothing renders.
pub(crate) fn make_line_text(rng: &mut Rng, text: &str, st: &TextStyle, em: f64, vertical: bool, min_px: f64) -> Option<LineG> {
    let strokes = layout(text, st, rng, vertical);
    if strokes.is_empty() {
        return None;
    }
    Some(line_from_strokes(rng, text.to_string(), strokes, st, em, vertical, min_px))
}

/// A Latin token written inside a Hunter line (extended vocab: digits and ! ? : / . -).
fn inline_token(rng: &mut Rng) -> String {
    let d = |rng: &mut Rng, lo: i64, hi: i64| rng.int(lo, hi).to_string();
    match rng.below(14) {
        0 | 1 => "!".into(),
        2 => "?".into(),
        3 => "!?".into(),
        4 => ["!!", "?!", "!!!", "??"][rng.below(4)].into(),
        5 => "...".into(),
        6 => d(rng, 0, 10),
        7 => d(rng, 10, 1000),
        8 => format!("{}:{:02}", rng.int(0, 24), rng.int(0, 60)),
        9 => format!("{}/{}", rng.int(1, 13), rng.int(1, 32)),
        10 => format!("{}-{}", rng.int(1, 100), rng.int(1, 100)),
        11 => format!("{}.{}", rng.int(0, 100), rng.int(0, 10)),
        12 => d(rng, 1990, 2030),
        _ => "-".into(),
    }
}

/// A horizontal Hunter line with digits / punctuation inline (font glyphs as `Prim::Fill`), sized
/// so the Latin cap height matches the Hunter glyphs. Tokens go at the end (punctuation mostly),
/// the start or the middle, with or without a word space.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_inline_line(rng: &mut Rng, sampler: &TextSampler, fonts: &FontBank, st: &TextStyle, em: f64, max_len: usize, max_extent: f64, min_px: f64) -> Option<LineG> {
    let face_i = pick_face(rng, fonts, Script::Digits, &[Class::Sans, Class::Display, Class::Serif])?;
    let face = &fonts.faces[face_i];
    let bold = if !face.bold && rng.chance(0.4) { em * rng.uniform(0.02, 0.07) } else { 0.0 };
    let kana = sampler.sample(rng, 1, max_len.clamp(1, 10));
    let kana = kana.trim().to_string();
    if kana.is_empty() {
        return None;
    }
    let tok = inline_token(rng);
    let where_ = rng.below(10); // 0-5 end, 6-7 start, 8-9 middle
    let sp = |rng: &mut Rng| if rng.chance(0.35) { " " } else { "" };
    let parts: Vec<(bool, String)> = match where_ {
        0..=5 => vec![(true, kana.clone()), (false, format!("{}{tok}", sp(rng)))],
        6 | 7 => vec![(false, format!("{tok}{}", sp(rng))), (true, kana.clone())],
        _ => {
            let k: Vec<char> = kana.chars().collect();
            if k.len() < 2 {
                vec![(true, kana.clone()), (false, format!("{}{tok}", sp(rng)))]
            } else {
                let cut = rng.int(1, k.len() as i64) as usize;
                let (a, b): (String, String) = (k[..cut].iter().collect(), k[cut..].iter().collect());
                let (s1, s2) = (sp(rng), sp(rng));
                vec![(true, a), (false, format!("{s1}{tok}{s2}")), (true, b)]
            }
        }
    };
    // cap height of the font ~ 0.72 px per px of size; Hunter glyphs are ~0.8-0.95 em tall
    let px = em * rng.uniform(0.8, 1.15) / 0.72;
    let mut prims: Vec<Prim> = Vec::new();
    let mut pen = 0.0;
    let mut raw = String::new();
    for (hxh, text) in &parts {
        if *hxh {
            let strokes = layout(text, st, rng, false);
            if strokes.is_empty() {
                continue;
            }
            let l = line_from_strokes(rng, text.clone(), strokes, st, em, false, min_px);
            let Ink::Hxh(ps) = l.ink else { continue };
            let dx = pen - l.b[0];
            for p in ps {
                prims.push(shift_prim(p, dx));
            }
            pen += l.b[2] - l.b[0] + em * st.spacing.clamp(0.06, 0.5);
            raw.push_str(text);
        } else {
            let lead = text.len() - text.trim_start().len();
            let trail = text.len() - text.trim_end().len();
            pen += em * 0.6 * lead as f64;
            let t = text.trim();
            let sh = face.shape(t, px, false, rng.uniform(0.0, 0.08), 0.0);
            if sh.contours.is_empty() || !sh.b[0].is_finite() {
                return None;
            }
            // centre the caps on the line axis (baseline at cap height / 2)
            let (dx, dy) = (pen - sh.b[0] + bold / 2.0, 0.36 * px);
            let polys: Vec<Vec<P>> = sh.contours.iter().map(|c| c.iter().map(|p| [p[0] + dx, p[1] + dy]).collect()).collect();
            prims.push(Prim::Fill { polys, bold });
            pen += sh.b[2] - sh.b[0] + bold + em * (0.12 + 0.6 * trail as f64) + em * rng.uniform(0.0, 0.2);
            raw.push_str(text);
        }
    }
    if prims.is_empty() {
        return None;
    }
    let b = prims_bbox(&prims, 0.0);
    if b[2] - b[0] > max_extent {
        return None;
    }
    Some(LineG { raw, vertical: false, ink: Ink::Hxh(prims), b, cells: Vec::new(), ruby: false })
}

fn shift_prim(p: Prim, dx: f64) -> Prim {
    let mv = |q: P| [q[0] + dx, q[1]];
    match p {
        Prim::Poly { pts, closed, w, sq } => Prim::Poly { pts: pts.into_iter().map(mv).collect(), closed, w, sq },
        Prim::Disk { c, rx, ry, ang } => Prim::Disk { c: mv(c), rx, ry, ang },
        Prim::Ring { c, rx, ry, ang, w } => Prim::Ring { c: mv(c), rx, ry, ang, w },
        Prim::Stamps { c, r } => Prim::Stamps { c: c.into_iter().map(mv).collect(), r },
        Prim::Fill { polys, bold } => Prim::Fill { polys: polys.into_iter().map(|v| v.into_iter().map(mv).collect()).collect(), bold },
    }
}

// --- v6: numbers and punctuation as part of the lettering, reading guides ----------------------

/// A face for digits written into Hunter lettering: bold, black and condensed display faces
/// first (italic ones for ! and ?), text faces less often, CJK faces rarely.
fn pick_digit_face(rng: &mut Rng, fonts: &FontBank) -> Option<usize> {
    let cands: Vec<usize> = (0..fonts.faces.len()).filter(|&i| fonts.faces[i].class != Class::Icon && fonts.faces[i].covers(crate::digits::EXTRA)).collect();
    if cands.is_empty() {
        return None;
    }
    let w: Vec<f64> = cands
        .iter()
        .map(|&i| {
            let f = &fonts.faces[i];
            let mut w = 1.0;
            if f.bold {
                w *= 3.0;
            }
            if f.class == Class::Display {
                w *= 4.0;
            }
            if f.cjk {
                w *= 0.4;
            }
            if matches!(f.class, Class::Mono | Class::Serif) {
                w *= 0.5;
            }
            w
        })
        .collect();
    Some(cands[rng.weighted(&w)])
}

/// Mean stroke thickness of filled outlines: 2 x area / perimeter (exact for a ring of constant
/// width, close for any monoline shape).
fn outline_stem(contours: &[Vec<P>]) -> f64 {
    let (mut a, mut p) = (0.0, 0.0);
    for c in contours {
        let n = c.len();
        for i in 0..n {
            let (u, v) = (c[i], c[(i + 1) % n]);
            a += u[0] * v[1] - v[0] * u[1];
            p += (v[0] - u[0]).hypot(v[1] - u[1]);
        }
    }
    if p > 0.0 { (a / 2.0).abs() * 2.0 / p } else { 0.0 }
}

/// Font digits / punctuation sized and weighted like the Hunter lettering around them: the digit
/// height of the line, the stroke thickness of the glyphs (outline dilation makes up the
/// difference), condensed now and then, tracking down to touching, italic ! and ?.
struct FontDigits {
    face: usize,
    px: f64,
    sx: f64,    // horizontal squeeze (condensed)
    bold: f64,  // dilation stroke, px
    track: f64, // em of px
    shear: f64, // digits
    pshear: f64, // ! ?
    h: f64,     // digit height, px
}

fn sample_font_digits(rng: &mut Rng, fonts: &FontBank, st: &TextStyle, em: f64) -> Option<FontDigits> {
    let face = pick_digit_face(rng, fonts)?;
    let f = &fonts.faces[face];
    let r = f.shape("0", 100.0, false, 0.0, 0.0);
    if r.contours.is_empty() || !r.b[0].is_finite() {
        return None;
    }
    let h0 = r.b[3] - r.b[1];
    let stem0 = outline_stem(&r.contours);
    let h = em * rng.uniform(0.78, 1.1) * if rng.chance(0.25) { rng.uniform(1.05, 1.2) } else { 1.0 };
    let px = 100.0 * h / h0.max(1.0);
    let sx = if rng.chance(0.4) { rng.uniform(0.68, 0.92) } else { 1.0 };
    let target = st.sw * em * rng.uniform(0.85, 1.25);
    let stem = stem0 * px / 100.0 * sx.sqrt();
    let bold = if rng.chance(0.85) { (target - stem).max(0.0) } else { 0.0 };
    let track = if rng.chance(0.45) { rng.uniform(-0.07, -0.01) } else { rng.uniform(0.0, 0.08) };
    let shear = if !f.italic && rng.chance(0.15) { rng.uniform(0.08, 0.2) } else { 0.0 };
    let pshear = if !f.italic && rng.chance(0.55) { rng.uniform(0.15, 0.32) } else { shear };
    Some(FontDigits { face, px, sx, bold, track, shear, pshear, h })
}

/// One token (digits / punctuation, no spaces) as filled outlines, its left ink edge at x = 0
/// and the digits centred on the line axis (baseline at h / 2). Returns the prims, the ink
/// width and the per-character extents.
fn font_token(rng: &mut Rng, fonts: &FontBank, fd: &FontDigits, tok: &str) -> Option<(Vec<Prim>, f64, Vec<(char, f64, f64)>)> {
    let f = &fonts.faces[fd.face];
    let punct = tok.chars().all(|c| c == '!' || c == '?');
    let shear = if punct { fd.pshear } else { fd.shear };
    let sh = f.shape(tok, fd.px, false, fd.track, shear);
    if sh.contours.is_empty() || !sh.b[0].is_finite() {
        return None;
    }
    let dy = fd.h / 2.0 + rng.normal(0.0, 0.02) * fd.h;
    let x0 = sh.b[0] * fd.sx;
    let polys: Vec<Vec<P>> = sh.contours.iter().map(|c| c.iter().map(|p| [p[0] * fd.sx - x0 + fd.bold / 2.0, p[1] + dy]).collect()).collect();
    let wid = (sh.b[2] - sh.b[0]) * fd.sx + fd.bold;
    let n = tok.chars().count().max(1) as f64;
    let cells = tok.chars().enumerate().map(|(i, c)| (c, wid * i as f64 / n, wid * (i + 1) as f64 / n)).collect();
    Some((vec![Prim::Fill { polys, bold: fd.bold }], wid, cells))
}

/// v6 inline line: Hunter kana with numbers and / or punctuation written into it
/// (`digits::inline_parts`), the digits drawn as strokes in the glyphs' own style (55%) or set in
/// a display font matched to the lettering (45%). Records each glyph's extent (`cells`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_inline_line_v6(rng: &mut Rng, sampler: &TextSampler, fonts: &FontBank, st: &TextStyle, em: f64, max_len: usize, max_extent: f64, min_px: f64) -> Option<LineG> {
    let kana = sampler.sample(rng, 1, max_len.clamp(1, 12)).trim().to_string();
    if kana.is_empty() {
        return None;
    }
    let parts = crate::digits::inline_parts(rng, &kana);
    let raw: String = parts.iter().map(|p| p.1.as_str()).collect();
    let fd = if fonts.faces.is_empty() || rng.chance(0.55) { None } else { sample_font_digits(rng, fonts, st, em) };
    let l = match fd {
        None => {
            // strokes: one layout pass over the whole text
            let ds = crate::digits::sample_style(rng, st.sw);
            let mut cells = Vec::new();
            let strokes = crate::render::layout_ex(&raw, st, rng, false, Some(&ds), Some(&mut cells));
            if strokes.is_empty() {
                return None;
            }
            let mut l = line_from_strokes(rng, raw, strokes, st, em, false, min_px);
            l.cells = cells.into_iter().map(|(c, a, b)| (c, a * em, b * em)).collect();
            l
        }
        Some(fd) => {
            let mut prims: Vec<Prim> = Vec::new();
            let mut cells: Vec<(char, f64, f64)> = Vec::new();
            let mut pen = 0.0;
            let mut raw = String::new();
            for (hxh, text) in &parts {
                if *hxh {
                    let mut kc = Vec::new();
                    let strokes = crate::render::layout_ex(text, st, rng, false, None, Some(&mut kc));
                    if strokes.is_empty() {
                        continue;
                    }
                    let l = line_from_strokes(rng, text.clone(), strokes, st, em, false, min_px);
                    let Ink::Hxh(ps) = l.ink else { continue };
                    let dx = pen - l.b[0];
                    prims.extend(ps.into_iter().map(|p| shift_prim(p, dx)));
                    cells.extend(kc.into_iter().map(|(c, a, b)| (c, a * em + dx, b * em + dx)));
                    pen += l.b[2] - l.b[0] + em * st.spacing.clamp(0.06, 0.6) * rng.normal(0.0, 0.15).exp();
                    raw.push_str(text);
                } else {
                    let lead = text.len() - text.trim_start().len();
                    let trail = text.len() - text.trim_end().len();
                    pen += em * 0.6 * lead as f64;
                    let (ps, wid, tc) = font_token(rng, fonts, &fd, text.trim())?;
                    prims.extend(ps.into_iter().map(|p| shift_prim(p, pen)));
                    cells.extend(tc.into_iter().map(|(c, a, b)| (c, a + pen, b + pen)));
                    pen += wid + em * (rng.uniform(0.04, 0.2) + 0.6 * trail as f64);
                    raw.push_str(text);
                }
            }
            if prims.is_empty() {
                return None;
            }
            let b = prims_bbox(&prims, 0.0);
            LineG { raw, vertical: false, ink: Ink::Hxh(prims), b, cells, ruby: false }
        }
    };
    if l.b[2] - l.b[0] > max_extent || !l.b[0].is_finite() {
        return None;
    }
    Some(l)
}

/// A horizontal Hunter line with digits inline: v6's builder when the config asks, else v5's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inline_line(rng: &mut Rng, sampler: &TextSampler, fonts: &FontBank, cfg: &PageConfig, st: &TextStyle, em: f64, max_len: usize, max_extent: f64) -> Option<LineG> {
    if cfg.v6 {
        make_inline_line_v6(rng, sampler, fonts, st, em, max_len, max_extent, cfg.min_stroke_px)
    } else {
        make_inline_line(rng, sampler, fonts, st, em, max_len, max_extent, cfg.min_stroke_px)
    }
}

/// A plain Hunter line with its glyph extents recorded (`cells`), for a reading guide.
pub(crate) fn make_line_cells(rng: &mut Rng, text: &str, st: &TextStyle, em: f64, min_px: f64) -> Option<LineG> {
    let mut cells = Vec::new();
    let strokes = crate::render::layout_ex(text, st, rng, false, None, Some(&mut cells));
    if strokes.is_empty() {
        return None;
    }
    let mut l = line_from_strokes(rng, text.to_string(), strokes, st, em, false, min_px);
    l.cells = cells.into_iter().map(|(c, a, b)| (c, a * em, b * em)).collect();
    Some(l)
}

/// Glyph extents for a line built without `cells` (v4 / v5 builders): its width split evenly.
fn even_cells(l: &LineG) -> Vec<(char, f64, f64)> {
    let chars: Vec<char> = l.raw.chars().filter(|&c| c != ' ').collect();
    let n = chars.len().max(1) as f64;
    let w = l.b[2] - l.b[0];
    chars.iter().enumerate().map(|(i, &c)| (c, l.b[0] + w * i as f64 / n, l.b[0] + w * (i + 1) as f64 / n)).collect()
}

/// A reading guide (furigana) for horizontal Hunter line `l`: small kana (hiragana, or katakana
/// for a whole line) centred over each glyph on one baseline just above the line's ink box, in
/// `l`'s own frame; numbers get their digits as the reading (often), punctuation none. An
/// unlabelled foreign run (script kana) that must never be read; the Hunter line's quad stays the
/// glyphs' own box.
pub(crate) fn ruby_line(rng: &mut Rng, fonts: &FontBank, l: &LineG, em: f64) -> Option<LineG> {
    if l.vertical || !l.hxh() {
        return None;
    }
    let face = pick_face(rng, fonts, crate::foreign::Script::Kana, &[Class::Sans])?;
    let f = &fonts.faces[face];
    let px = em * rng.uniform(0.24, 0.46);
    let kata_all = rng.chance(0.35);
    let digits_too = rng.chance(0.6);
    let gap = em * rng.uniform(0.04, 0.32);
    let bold = if rng.chance(0.4) { px * rng.uniform(0.03, 0.08) } else { 0.0 };
    let cells = if l.cells.is_empty() { even_cells(l) } else { l.cells.clone() };
    // readings: one per kana glyph, one per run of digits
    let mut items: Vec<(String, f64)> = Vec::new();
    let mut i = 0;
    while i < cells.len() {
        let (c, a, b) = cells[i];
        if c.is_ascii_digit() {
            let mut j = i;
            while j + 1 < cells.len() && (cells[j + 1].0.is_ascii_digit() || ".:/-".contains(cells[j + 1].0)) {
                j += 1;
            }
            if digits_too {
                let s: String = cells[i..=j].iter().map(|x| x.0).collect();
                items.push((s, (a + cells[j].2) / 2.0));
            }
            i = j + 1;
            continue;
        }
        if !crate::digits::is_extra(c) {
            let s = if kata_all { crate::foreign::kata(&c.to_string()) } else { c.to_string() };
            items.push((s, (a + b) / 2.0));
        }
        i += 1;
    }
    let base = l.b[1] - gap - 0.1 * px; // kana dip ~0.1 px under the baseline
    let mut contours: Vec<Vec<P>> = Vec::new();
    let mut text = String::new();
    for (s, cx) in items {
        if !f.covers(&s) {
            continue;
        }
        let sh = f.shape(&s, px, false, 0.0, 0.0);
        if sh.contours.is_empty() || !sh.b[0].is_finite() {
            continue;
        }
        let dx = cx - (sh.b[0] + sh.b[2]) / 2.0;
        contours.extend(sh.contours.into_iter().map(|c| c.into_iter().map(|p| [p[0] + dx, p[1] + base]).collect::<Vec<P>>()));
        text.push_str(&s);
    }
    if contours.is_empty() {
        return None;
    }
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in contours.iter().flatten() {
        b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    }
    let b = grow4(b, bold / 2.0);
    if b[3] >= l.b[1] - 0.5 {
        return None;
    }
    let run = Run { face, script: crate::foreign::Script::Kana, text: text.clone(), b };
    Some(LineG { raw: text, vertical: false, ink: Ink::Font { contours, runs: vec![run], bold, hollow: 0.0 }, b, cells: Vec::new(), ruby: true })
}

/// Put reading guides (`rubies[i]` for line i, in line i's frame) into a stacked group and
/// re-origin it so the group box covers them.
pub(crate) fn attach_ruby(g: &mut Group, rubies: Vec<Option<LineG>>) {
    let n = g.lines.len();
    for (i, r) in rubies.into_iter().enumerate().take(n) {
        if let Some(r) = r {
            g.lines.push(r);
            let o = g.offs[i];
            g.offs.push(o);
        }
    }
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for (l, o) in g.lines.iter().zip(&g.offs) {
        b = [b[0].min(l.b[0] + o[0]), b[1].min(l.b[1] + o[1]), b[2].max(l.b[2] + o[0]), b[3].max(l.b[3] + o[1])];
    }
    for o in g.offs.iter_mut() {
        *o = [o[0] - b[0], o[1] - b[1]];
    }
    g.w = b[2] - b[0];
    g.h = b[3] - b[1];
}

/// The text style for a new Hunter group: v4's mix, or v5's (more styles) when the page config asks.
pub(crate) fn style_for(rng: &mut Rng, cfg: &PageConfig) -> TextStyle {
    let mut st = if cfg.v5 { sample_style_v5(rng, false) } else { sample_style(rng) };
    small_marks(rng, cfg, &mut st);
    st
}

/// v6: small dakuten dots for a share of the styles (`small_mark_prob`).
pub(crate) fn small_marks(rng: &mut Rng, cfg: &PageConfig, st: &mut TextStyle) {
    if cfg.small_mark_prob > 0.0 && rng.chance(cfg.small_mark_prob) {
        st.mark_small = rng.uniform(0.55, 1.0);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hxh_lines(rng: &mut Rng, sampler: &TextSampler, cfg: &PageConfig, st: &TextStyle, em: f64, vertical: bool, n: usize, max_len: usize, max_extent: f64) -> Vec<LineG> {
    let mut lines = Vec::new();
    for _ in 0..n {
        let ml = if rng.chance(0.7) { max_len } else { ((max_len as f64 * rng.uniform(0.3, 1.0)) as usize).max(1) };
        if let Some(l) = make_line(rng, sampler, st, em, vertical, ml, max_extent, cfg.min_stroke_px) {
            lines.push(l);
        }
    }
    lines
}

fn grow4(b: [f64; 4], g: f64) -> [f64; 4] {
    [b[0] - g, b[1] - g, b[2] + g, b[3] + g]
}

/// Style shared by the foreign lines of one group.
#[derive(Clone)]
pub(crate) struct FStyle {
    pub face: usize,
    pub script: Script,
    pub bold: f64, // dilation stroke, em
    pub track: f64,
    pub shear: f64,
    pub caps: bool,
}

/// A text face for `script`, preferring `prefer`; Latin text rarely uses CJK faces.
pub(crate) fn pick_face(rng: &mut Rng, fonts: &FontBank, script: Script, prefer: &[Class]) -> Option<usize> {
    let cands: Vec<usize> = fonts.with(script.need()).into_iter().filter(|&i| fonts.faces[i].class != Class::Icon).collect();
    if cands.is_empty() {
        return None;
    }
    let latin_like = matches!(script, Script::Latin | Script::Romaji | Script::Url | Script::Digits | Script::Greek | Script::Cyrillic);
    let w: Vec<f64> = cands
        .iter()
        .map(|&i| {
            let f = &fonts.faces[i];
            let mut w = 1.0;
            if latin_like && f.cjk {
                w *= 0.12;
            }
            if matches!(f.class, Class::Hand | Class::Display) {
                w *= 0.5;
            }
            if prefer.contains(&f.class) {
                w *= 5.0;
            }
            w
        })
        .collect();
    Some(cands[rng.weighted(&w)])
}

pub(crate) fn sample_fstyle(rng: &mut Rng, fonts: &FontBank, script: Script, kind: Kind) -> Option<FStyle> {
    let prefer: &[Class] = match kind {
        Kind::Bubble if rng.chance(0.5) => &[Class::Hand],
        Kind::Free if rng.chance(0.5) => &[Class::Display, Class::Hand],
        Kind::Sign if rng.chance(0.3) => &[Class::Display],
        Kind::Block | Kind::Caption if rng.chance(0.4) => &[Class::Serif],
        _ => &[],
    };
    let face = pick_face(rng, fonts, script, prefer)?;
    let f = &fonts.faces[face];
    let heavy = matches!(kind, Kind::Free | Kind::Sign);
    let bold = if !f.bold && rng.chance(if heavy { 0.4 } else { 0.15 }) { rng.uniform(0.02, 0.07) } else { 0.0 };
    let track = if script.by_char() { rng.uniform(0.0, 0.15) } else if rng.chance(0.15) { rng.uniform(-0.02, 0.12) } else { 0.0 };
    let shear = if !f.italic && rng.chance(0.08) { rng.uniform(0.12, 0.25) } else { 0.0 };
    let latin = script == Script::Latin || script == Script::Romaji;
    let caps = latin
        && rng.chance(match kind {
            Kind::Bubble => 0.5,
            Kind::Free => 0.6,
            Kind::Sign => 0.4,
            Kind::Caption => 0.15,
            _ => 0.05,
        });
    Some(FStyle { face, script, bold, track, shear, caps })
}

/// One foreign line: `text` if given, else generated for `role`; cut to `max_extent` px.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_font_line(rng: &mut Rng, tg: &TextGen, fonts: &FontBank, fs: &FStyle, em: f64, vertical: bool, role: Role, max_extent: f64, max_chars: usize, text: Option<&str>) -> Option<LineG> {
    let face = &fonts.faces[fs.face];
    let bold_px = if fs.bold > 0.0 { (fs.bold * em).max(0.5) } else { 0.0 };
    for _ in 0..4 {
        let mut t = match text {
            Some(t) => t.to_string(),
            None => tg.text(rng, fs.script, role, max_chars),
        };
        if fs.caps {
            t = t.to_uppercase();
        }
        if !face.covers(&t) {
            t = face.filter(&t);
        }
        t = if vertical {
            crate::foreign::clip(&t, ((max_extent - bold_px) / (em * (1.0 + fs.track))).floor().max(0.0) as usize, true)
        } else {
            face.fit(&t, em, max_extent - bold_px - em * fs.shear, fs.track, fs.script.by_char())
        };
        if t.trim().is_empty() {
            if text.is_some() {
                return None;
            }
            continue;
        }
        let sh = face.shape(&t, em, vertical, fs.track, fs.shear);
        if sh.contours.is_empty() || !sh.b[0].is_finite() {
            continue;
        }
        let b = grow4(sh.b, bold_px / 2.0);
        let run = Run { face: fs.face, script: fs.script, text: t.clone(), b };
        return Some(LineG { raw: t, vertical, ink: Ink::Font { contours: sh.contours, runs: vec![run], bold: bold_px, hollow: 0.0 }, b, cells: Vec::new(), ruby: false });
    }
    None
}

/// Text recorded for icons: the character when it is a normal Unicode symbol, else `#<glyph id>`.
fn icon_text(ids: &[(ab_glyph::GlyphId, Option<char>)]) -> String {
    ids.iter()
        .map(|(g, c)| match c {
            Some(c) if !('\u{E000}'..='\u{F8FF}').contains(c) => c.to_string(),
            _ => format!("#{}", g.0),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A row of `n` icons from `face` (fewer if they do not fit), spaced `track` em; sometimes a
/// rating row (one glyph repeated, then another).
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_icon_line(rng: &mut Rng, fonts: &FontBank, face: usize, em: f64, n: usize, track: f64, hollow: bool, max_extent: f64) -> Option<LineG> {
    let f = &fonts.faces[face];
    if f.icons.is_empty() || n == 0 {
        return None;
    }
    let mut ids: Vec<(ab_glyph::GlyphId, Option<char>)> = if n >= 3 && rng.chance(0.15) {
        let a = f.icons[rng.below(f.icons.len())];
        let b = f.icons[rng.below(f.icons.len())];
        let k = rng.int(1, n as i64 + 1) as usize;
        (0..n).map(|i| if i < k { a } else { b }).collect()
    } else {
        (0..n).map(|_| f.icons[rng.below(f.icons.len())]).collect()
    };
    let hol = if hollow { (0.05 * em).max(1.0) } else { 0.0 };
    while !ids.is_empty() {
        let g: Vec<ab_glyph::GlyphId> = ids.iter().map(|x| x.0).collect();
        let sh = f.shape_ids(&g, em, track);
        if sh.contours.is_empty() || !sh.b[0].is_finite() {
            return None;
        }
        let b = grow4(sh.b, hol / 2.0);
        if b[2] - b[0] <= max_extent {
            let text = icon_text(&ids);
            let run = Run { face, script: Script::Icon, text: text.clone(), b };
            return Some(LineG { raw: text, vertical: false, ink: Ink::Font { contours: sh.contours, runs: vec![run], bold: 0.0, hollow: hol }, b, cells: Vec::new(), ruby: false });
        }
        ids.pop();
    }
    None
}

/// Join horizontal single-run foreign lines on one baseline, `gap` px apart; icon runs are
/// centred on the first text run. Dilation / hollow of the parts is dropped (filled, regular).
pub(crate) fn compose(parts: Vec<LineG>, gap: f64) -> Option<LineG> {
    let text_mid = parts.iter().find(|l| matches!(&l.ink, Ink::Font { runs, .. } if runs[0].script != Script::Icon)).map(|l| (l.b[1] + l.b[3]) / 2.0);
    let mut contours = Vec::new();
    let mut runs = Vec::new();
    let mut raw = Vec::new();
    let mut pen = 0.0;
    for l in parts {
        let Ink::Font { contours: cs, runs: rs, .. } = l.ink else { continue };
        let dx = pen - l.b[0];
        let dy = match (text_mid, rs[0].script) {
            (Some(m), Script::Icon) => m - (l.b[1] + l.b[3]) / 2.0,
            _ => 0.0,
        };
        contours.extend(cs.into_iter().map(|c| c.into_iter().map(|p| [p[0] + dx, p[1] + dy]).collect::<Vec<P>>()));
        for mut r in rs {
            r.b = [r.b[0] + dx, r.b[1] + dy, r.b[2] + dx, r.b[3] + dy];
            runs.push(r);
        }
        raw.push(l.raw);
        pen += l.b[2] - l.b[0] + gap;
    }
    if runs.is_empty() {
        return None;
    }
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for r in &runs {
        b = [b[0].min(r.b[0]), b[1].min(r.b[1]), b[2].max(r.b[2]), b[3].max(r.b[3])];
    }
    Some(LineG { raw: raw.join(" "), vertical: false, ink: Ink::Font { contours, runs, bold: 0.0, hollow: 0.0 }, b, cells: Vec::new(), ruby: false })
}

/// Stack lines into a group: rows top to bottom, or columns right to left. Neighbours of
/// different classes (Hunter / foreign) get at least (keep + 0.15) line heights between them.
pub(crate) fn stack(lines: Vec<LineG>, vertical: bool, em: f64, gap: f64, align: u8, keep: f64) -> Option<Group> {
    if lines.is_empty() {
        return None;
    }
    let gap = gap.max(2.0);
    let sizes: Vec<(f64, f64)> = lines.iter().map(|l| (l.b[2] - l.b[0], l.b[3] - l.b[1])).collect();
    let th = |k: usize| if vertical { sizes[k].0 } else { sizes[k].1 };
    let gaps: Vec<f64> = (0..lines.len())
        .map(|i| if i == 0 || lines[i].hxh() == lines[i - 1].hxh() { gap } else { gap.max((keep + 0.15) * th(i).max(th(i - 1)) + 2.0) })
        .collect();
    let mut offs = Vec::with_capacity(lines.len());
    let (gw, gh);
    if !vertical {
        gw = sizes.iter().map(|s| s.0).fold(0.0, f64::max);
        let mut y = 0.0;
        for (i, (l, &(w, h))) in lines.iter().zip(&sizes).enumerate() {
            if i > 0 {
                y += gaps[i];
            }
            let x = match align {
                b'l' => 0.0,
                b'r' => gw - w,
                _ => (gw - w) / 2.0,
            };
            offs.push([x - l.b[0], y - l.b[1]]);
            y += h;
        }
        gh = y;
    } else {
        gh = sizes.iter().map(|s| s.1).fold(0.0, f64::max);
        gw = sizes.iter().map(|s| s.0).sum::<f64>() + gaps[1..].iter().sum::<f64>();
        let mut x = gw; // columns right to left
        for (i, (l, &(w, h))) in lines.iter().zip(&sizes).enumerate() {
            if i > 0 {
                x -= gaps[i];
            }
            x -= w;
            let y = match align {
                b'l' => 0.0,
                b'r' => gh - h,
                _ => (gh - h) / 2.0,
            };
            offs.push([x - l.b[0], y - l.b[1]]);
        }
    }
    Some(Group { lines, offs, w: gw, h: gh, vertical, em })
}

/// Group coords -> coords centred on 0: rotation about the centre plus optional perspective.
fn group_matrix(rng: &mut Rng, ext: [f64; 4], angle: f64, persp: f64) -> H3 {
    let (w, h) = (ext[2] - ext[0], ext[3] - ext[1]);
    let mut m = tr(-(ext[0] + ext[2]) / 2.0, -(ext[1] + ext[3]) / 2.0);
    if persp > 0.0 {
        let src = [[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]];
        let mut dst = src;
        for p in dst.iter_mut() {
            p[0] += rng.uniform(-1.0, 1.0) * w * persp;
            p[1] += rng.uniform(-1.0, 1.0) * h * persp;
        }
        m = mul(&homography(&src, &dst), &m);
    }
    mul(&rot(angle), &m)
}

fn angle(rng: &mut Rng, r: Rot) -> f64 {
    match r {
        Rot::Flat => rng.normal(0.0, 0.4),
        Rot::Small => {
            if rng.chance(0.6) { rng.normal(0.0, 0.8) } else { rng.normal(0.0, 4.0).clamp(-12.0, 12.0) }
        }
        Rot::Wide => {
            let u = rng.f64();
            if u < 0.4 {
                rng.normal(0.0, 0.8)
            } else if u < 0.8 {
                rng.normal(0.0, 5.0).clamp(-12.0, 12.0)
            } else {
                rng.uniform(-12.0, 12.0)
            }
        }
    }
}

// --- containers (group coords) -----------------------------------------------------------------

fn tail(rng: &mut Rng, cx: f64, cy: f64, a: f64, b: f64, em: f64) -> Vec<P> {
    let phi = if rng.chance(0.7) { rng.uniform(0.1, 0.9) * PI } else { rng.uniform(0.0, TAU) };
    let d = rng.uniform(0.1, 0.25);
    let p1 = [cx + 0.85 * a * (phi - d).cos(), cy + 0.85 * b * (phi - d).sin()];
    let p2 = [cx + 0.85 * a * (phi + d).cos(), cy + 0.85 * b * (phi + d).sin()];
    let re = (a * phi.cos()).hypot(b * phi.sin());
    let l = re + (re * rng.uniform(0.3, 0.9)).min(em * rng.uniform(1.2, 3.5));
    let ang = phi + rng.uniform(-0.4, 0.4);
    vec![p1, [cx + l * ang.cos(), cy + l * ang.sin()], p2]
}

fn bubble_polys(rng: &mut Rng, g: &Group) -> Vec<Vec<P>> {
    let (cx, cy, em) = (g.w / 2.0, g.h / 2.0, g.em);
    let pad = em * rng.uniform(0.25, 0.7);
    let shape = rng.weighted(&[0.55, 0.22, 0.1, 0.13]); // ellipse, round box, cloud, burst
    let mut polys = Vec::new();
    let (a, b);
    if shape == 1 {
        a = g.w / 2.0 + pad * 1.3;
        b = g.h / 2.0 + pad * 1.3;
        polys.push(round_rect(cx - a, cy - b, cx + a, cy + b, a.min(b) * rng.uniform(0.35, 1.0), 7));
    } else {
        let k = rng.uniform(1.45, 1.65); // >= sqrt(2): the ellipse contains the whole text block
        a = g.w / 2.0 * k + pad;
        b = g.h / 2.0 * k + pad;
        if shape == 3 {
            let n = rng.int(12, 26) as usize;
            let body = (0..2 * n)
                .map(|i| {
                    let t = (i as f64 + rng.uniform(-0.25, 0.25)) * (PI / n as f64);
                    let r = if i % 2 == 0 { 1.0 } else { rng.uniform(1.15, 1.45) };
                    [cx + a * r * t.cos(), cy + b * r * t.sin()]
                })
                .collect();
            polys.push(body);
        } else {
            polys.push(ellipse(cx, cy, a, b, 72));
        }
        if shape == 2 {
            let n = ((TAU * ((a * a + b * b) / 2.0).sqrt() / (em * rng.uniform(1.0, 2.0))) as usize).max(6);
            let rb = (em * rng.uniform(0.45, 0.9)).min(0.5 * a.min(b));
            let t0 = rng.f64();
            for i in 0..n {
                let t = t0 + i as f64 / n as f64 * TAU;
                let r = rb * rng.uniform(0.8, 1.2);
                polys.push(ellipse(cx + a * t.cos(), cy + b * t.sin(), r, r, 20));
            }
        }
    }
    if rng.chance(if shape != 3 { 0.85 } else { 0.4 }) {
        polys.push(tail(rng, cx, cy, a, b, em));
    }
    polys
}

fn box_polys(rng: &mut Rng, g: &Group, padx: (f64, f64), pady: (f64, f64), round_p: f64) -> Vec<Vec<P>> {
    let (mut px, mut py) = (g.em * rng.uniform(padx.0, padx.1), g.em * rng.uniform(pady.0, pady.1));
    if g.vertical {
        std::mem::swap(&mut px, &mut py);
    }
    let (x0, y0, x1, y1) = (-px, -py, g.w + px, g.h + py);
    let r = if rng.chance(round_p) { (x1 - x0).min(y1 - y0) * rng.uniform(0.1, 0.5) } else { 0.0 };
    vec![round_rect(x0, y0, x1, y1, r, 7)]
}

// --- composer ----------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct Shadow {
    pub dx: f64,
    pub dy: f64,
    pub sigma: f64,
    pub strength: f32,
    pub lum: f32,
}

/// A blurred, grown copy of the ink under it: a glow (light) or a soft outer stroke (dark).
#[derive(Clone, Copy)]
pub(crate) struct Halo {
    pub grow: f64,
    pub sigma: f64,
    pub lum: f32,
    pub strength: f32,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct Fx {
    pub outline: Option<(f64, f32)>, // radius px, lum
    pub shadow: Option<Shadow>,
    pub patchy: bool,
    pub halo: Option<Halo>,
    pub bevel: Option<(f64, f32, f32)>, // offset px, highlight lum (up-left), shade lum (down-right)
    pub sticker: bool,    // v6: thick outline kept whatever the background (sticker lettering)
    pub shadow_grow: f64, // v6: the shadow is cast by the ink grown by this (the outline's shadow)
}

impl Fx {
    /// Decoration px around the ink (keep-clear and occupancy margins).
    pub(crate) fn extent(&self) -> f64 {
        let mut e: f64 = self.outline.map_or(0.0, |o| o.0);
        if let Some(s) = self.shadow {
            e = e.max((s.dx.abs() + s.dy.abs() + 2.0 * s.sigma + self.shadow_grow).ceil());
        }
        if let Some(h) = self.halo {
            e = e.max((h.grow + 2.0 * h.sigma).ceil());
        }
        if let Some(b) = self.bevel {
            e = e.max(b.0.ceil() + 1.0);
        }
        e
    }
}

struct Meta {
    quad: [P; 4],
    ink: f32,
    kind: Kind,
}

/// A placed quad for the keep-clear rule.
struct Keep {
    q: [P; 4],
    bb: [f64; 4],
    h: f64,
    halo: f64,
}

/// What a planned element is written in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Want {
    Auto,
    Foreign,
    Icons,
}

pub(crate) struct Composer<'a> {
    pub rng: &'a mut Rng,
    pub sampler: &'a TextSampler,
    pub tg: TextGen<'a>,
    pub fonts: &'a FontBank,
    pub cfg: &'a PageConfig,
    pub w: usize,
    pub h: usize,
    pub page: Gray,
    pub occ: Vec<u8>,
    pub lines: Vec<LineOut>,
    pub foreign: Vec<ForeignOut>,
    meta: Vec<Meta>,
    hq: Vec<Keep>,
    fq: Vec<Keep>,
    pub neg: bool, // negative page: no Hunter text at all
    fboost: f64,   // per-layout multiplier on foreign_prob
}

/// Quads of a line under `mi`: the whole line for Hunter text, one per run for foreign text.
fn line_quads(l: &LineG, mi: &H3) -> Vec<[P; 4]> {
    match &l.ink {
        Ink::Hxh(_) => vec![box_quad(l.b, l.vertical, 0.0).map(|p| apply(mi, p))],
        Ink::Font { runs, .. } => runs.iter().map(|r| box_quad(r.b, l.vertical, 0.0).map(|p| apply(mi, p))).collect(),
    }
}

fn font_coverage(contours: &[Vec<P>], m: &H3, centre: P, dil: f64, hollow: f64, clip: IRect) -> Option<Cov> {
    let (s, _) = local_frame(m, centre);
    let mut pb = PathBuilder::new();
    for c in contours {
        let q = apply(m, c[0]);
        pb.move_to(q[0] as f32, q[1] as f32);
        for p in &c[1..] {
            let q = apply(m, *p);
            pb.line_to(q[0] as f32, q[1] as f32);
        }
        pb.close();
    }
    let path = pb.finish()?;
    let mut items = Vec::with_capacity(2);
    if hollow > 0.0 {
        items.push((path, Draw::Stroke(((hollow + 2.0 * dil) * s) as f32)));
    } else {
        if dil > 0.01 {
            items.push((path.clone(), Draw::Stroke((2.0 * dil * s) as f32)));
        }
        items.push((path, Draw::Fill));
    }
    coverage(&items, clip)
}

/// Coverage of a line's ink through `m`, grown by `grow` px (outlines).
fn ink_coverage(ink: &Ink, m: &H3, centre: P, grow: f64, clip: IRect) -> Option<Cov> {
    match ink {
        Ink::Hxh(prims) => prims_coverage(prims, m, centre, grow, clip),
        Ink::Font { contours, bold, hollow, .. } => font_coverage(contours, m, centre, grow + bold / 2.0, *hollow, clip),
    }
}

fn paths_of(polys: &[Vec<P>]) -> Vec<Path> {
    polys.iter().filter_map(|p| poly_path(p, true)).collect()
}

fn round1(q: [P; 4]) -> [[f64; 2]; 4] {
    q.map(|p| [(p[0] * 10.0).round() / 10.0, (p[1] * 10.0).round() / 10.0])
}

impl Composer<'_> {
    pub fn page_rect(&self) -> IRect {
        IRect { x0: 0, y0: 0, x1: self.w as i32, y1: self.h as i32 }
    }

    /// Keep-clear rule: a foreign quad (with `halo` px of decoration) and a labelled quad must be
    /// at least keep_margin x the larger line height (+ the foreign halo) apart, exactly.
    fn clear_of(&self, q: &[P; 4], hxh: bool, halo: f64) -> bool {
        self.clear_of_first(q, hxh, halo, usize::MAX)
    }

    /// `clear_of` against the first `n` placed quads of the other class only (a reading guide is
    /// checked against everything but its own group's Hunter lines).
    fn clear_of_first(&self, q: &[P; 4], hxh: bool, halo: f64, n: usize) -> bool {
        let h = line_thickness(q);
        let bb = bbox(q);
        let others = if hxh { &self.fq } else { &self.hq };
        others.iter().take(n).all(|o| {
            let need = self.cfg.keep_margin * h.max(o.h) + if hxh { o.halo } else { halo } + 0.5;
            bbox_gap(bb, o.bb) >= need || convex_dist(q, &o.q) >= need
        })
    }

    fn group_clear(&self, g: &Group, m: &H3, halo: f64) -> bool {
        g.lines.iter().zip(&g.offs).all(|(l, o)| {
            let mi = mul(m, &tr(o[0], o[1]));
            line_quads(l, &mi).iter().all(|q| self.clear_of(q, l.hxh(), halo))
        })
    }

    /// Find a free spot for the group in `region`. Returns the group -> page homography, the
    /// footprint rect and the footprint mask.
    #[allow(clippy::too_many_arguments)]
    fn try_place(&mut self, g: &Group, m0: &H3, polys: &[Vec<P>], margin: f64, border: f64, region: [f64; 4], tries: usize, halo: f64) -> Option<(H3, IRect, Vec<u8>)> {
        let grown: Vec<[P; 4]> = g.lines.iter().zip(&g.offs).map(|(l, o)| box_quad(l.b, false, margin).map(|p| [p[0] + o[0], p[1] + o[1]])).collect();
        let mut all: Vec<P> = grown.iter().flatten().copied().collect();
        for p in polys {
            all.extend(p);
        }
        let rb = bbox(&apply_all(m0, &all));
        let (rx0, ry0, rx1, ry1) = (rb[0] - border - 2.0, rb[1] - border - 2.0, rb[2] + border + 2.0, rb[3] + border + 2.0);
        let [gx0, gy0, gx1, gy1] = region;
        if gx1 - gx0 < rx1 - rx0 || gy1 - gy0 < ry1 - ry0 {
            return None;
        }
        for _ in 0..tries {
            let cx = self.rng.uniform(gx0 - rx0, gx1 - rx1);
            let cy = self.rng.uniform(gy0 - ry0, gy1 - ry1);
            let roi = IRect { x0: (cx + rx0).floor() as i32, y0: (cy + ry0).floor() as i32, x1: (cx + rx1).ceil() as i32, y1: (cy + ry1).ceil() as i32 }
                .intersect(self.page_rect());
            if roi.is_empty() {
                continue;
            }
            let m = mul(&tr(cx, cy), m0);
            let mut fp = vec![0u8; roi.w() * roi.h()];
            if !polys.is_empty() {
                let mut items = Vec::new();
                for p in paths_of(&polys.iter().map(|p| apply_all(&m, p)).collect::<Vec<_>>()) {
                    if border > 0.0 {
                        items.push((p.clone(), Draw::Stroke(2.0 * border as f32)));
                    }
                    items.push((p, Draw::Fill));
                }
                if let Some((r, d)) = rasterize(&items, roi, false) {
                    for y in 0..r.h() {
                        for x in 0..r.w() {
                            if d[y * r.w() + x] != 0 {
                                fp[(y as i32 + r.y0 - roi.y0) as usize * roi.w() + (x as i32 + r.x0 - roi.x0) as usize] = 255;
                            }
                        }
                    }
                }
            }
            for q in &grown {
                fill_convex_u8(&mut fp, roi, &q.map(|p| apply(&m, p)), 255);
            }
            let mut clash = false;
            'rows: for y in roi.y0..roi.y1 {
                let o = y as usize * self.w;
                let f = (y - roi.y0) as usize * roi.w();
                for x in 0..roi.w() {
                    if fp[f + x] != 0 && self.occ[o + roi.x0 as usize + x] != 0 {
                        clash = true;
                        break 'rows;
                    }
                }
            }
            if !clash && self.group_clear(g, &m, halo) {
                return Some((m, roi, fp));
            }
        }
        None
    }

    fn draw_container(&mut self, m: &H3, polys: &[Vec<P>], fill: f32, border: f64, border_lum: f32, shadow: Option<Shadow>) {
        let pts: Vec<Vec<P>> = polys.iter().map(|p| apply_all(m, p)).collect();
        let paths = paths_of(&pts);
        let fills: Vec<(Path, Draw)> = paths.iter().map(|p| (p.clone(), Draw::Fill)).collect();
        let mut outer = fills.clone();
        if border > 0.0 {
            outer.extend(paths.iter().map(|p| (p.clone(), Draw::Stroke(2.0 * border as f32))));
        }
        let clip = self.page_rect();
        if let Some(sh) = shadow {
            let ts = Transform::from_translate(sh.dx as f32, sh.dy as f32);
            let moved: Vec<(Path, Draw)> = outer.iter().filter_map(|(p, d)| p.clone().transform(ts).map(|p| (p, *d))).collect();
            let pad = (3.0 * sh.sigma).ceil() as i32 + 1;
            if let Some(mut c) = coverage(&moved, clip) {
                c = pad_cov(c, pad, clip);
                let (w, h) = (c.r.w(), c.r.h());
                blur(&mut c.a, w, h, sh.sigma, false);
                blend(&mut self.page, &c, sh.lum, sh.strength);
            }
        }
        if border > 0.0
            && let Some(c) = coverage(&outer, clip) {
                blend(&mut self.page, &c, border_lum, 1.0);
            }
        if let Some(c) = coverage(&fills, clip) {
            blend(&mut self.page, &c, fill, 1.0);
        }
    }

    /// Draw the group's lines through `m` (Hunter lines first) and record them: Hunter lines in
    /// `lines` (kind `kind`), foreign runs in `foreign` (kind `fkind`). A line that would break
    /// the keep-clear rule is skipped. Returns the number of lines drawn.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_lines(&mut self, g: &Group, m: &H3, ink: f32, fx: &Fx, kind: Kind, halo: f64, fkind: &'static str) -> usize {
        let clip = self.page_rect();
        let mut order: Vec<usize> = (0..g.lines.len()).filter(|&i| g.lines[i].hxh()).collect();
        order.extend((0..g.lines.len()).filter(|&i| !g.lines[i].hxh()));
        let mut drawn = 0;
        let hq0 = self.hq.len();
        for i in order {
            let (l, o) = (&g.lines[i], g.offs[i]);
            let mi = mul(m, &tr(o[0], o[1]));
            let quads = line_quads(l, &mi);
            let n_keep = if l.ruby { hq0 } else { usize::MAX };
            if !quads.iter().all(|q| self.clear_of_first(q, l.hxh(), halo, n_keep)) {
                continue;
            }
            let centre = [(l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0];
            let roi = IRect::around(bbox(&box_quad(l.b, false, halo + 3.0).map(|p| apply(&mi, p))), 1.0).intersect(clip);
            if let Some(sh) = fx.shadow {
                let ms = mul(&tr(sh.dx, sh.dy), &mi);
                if let Some(c) = ink_coverage(&l.ink, &ms, centre, fx.shadow_grow, roi) {
                    let mut c = pad_cov(c, (3.0 * sh.sigma).ceil() as i32 + 1, roi);
                    let (w, h) = (c.r.w(), c.r.h());
                    blur(&mut c.a, w, h, sh.sigma, false);
                    blend(&mut self.page, &c, sh.lum, sh.strength);
                }
            }
            if let Some(hl) = fx.halo
                && let Some(c) = ink_coverage(&l.ink, &mi, centre, hl.grow, roi) {
                    let mut c = pad_cov(c, (3.0 * hl.sigma).ceil() as i32 + 1, roi);
                    let (w, h) = (c.r.w(), c.r.h());
                    blur(&mut c.a, w, h, hl.sigma, false);
                    blend(&mut self.page, &c, hl.lum, hl.strength);
                }
            if let Some((r, lum)) = fx.outline
                && let Some(c) = ink_coverage(&l.ink, &mi, centre, r, roi) {
                    blend(&mut self.page, &c, lum, 1.0);
                }
            if let Some((d, hi, lo)) = fx.bevel {
                for (dx, lum) in [(-d, hi), (d, lo)] {
                    if let Some(c) = ink_coverage(&l.ink, &mul(&tr(dx, dx), &mi), centre, 0.0, roi) {
                        blend(&mut self.page, &c, lum, 1.0);
                    }
                }
            }
            if let Some(mut c) = ink_coverage(&l.ink, &mi, centre, 0.0, roi) {
                if fx.patchy {
                    let k = self.rng.uniform(0.1, 0.45) as f32;
                    let cell = self.rng.uniform(2.0, 8.0);
                    let n = value_noise(c.r.w(), c.r.h(), self.rng, cell);
                    c.a.iter_mut().zip(&n).for_each(|(a, v)| *a *= (1.0 - k * v.abs()).clamp(0.3, 1.0));
                }
                blend(&mut self.page, &c, ink, 1.0);
            }
            let occ_q = box_quad(l.b, false, halo + 1.0).map(|p| apply(&mi, p));
            fill_convex_u8(&mut self.occ, clip, &occ_q, 255);
            let orient = if l.vertical { "v" } else { "h" };
            match &l.ink {
                Ink::Hxh(_) => {
                    let quad = quads[0];
                    self.lines.push(LineOut { quad: round1(quad), text: label_of(&l.raw), orient, kind: kind.name() });
                    self.meta.push(Meta { quad, ink, kind });
                    self.hq.push(Keep { q: quad, bb: bbox(&quad), h: line_thickness(&quad), halo: 0.0 });
                }
                Ink::Font { runs, .. } => {
                    for (r, q) in runs.iter().zip(quads) {
                        let font = self.fonts.faces[r.face].name.clone();
                        let kind = if l.ruby { "ruby" } else { fkind };
                        self.foreign.push(ForeignOut { quad: round1(q), text: r.text.clone(), script: r.script.name(), font, orient, kind });
                        self.fq.push(Keep { q, bb: bbox(&q), h: line_thickness(&q), halo });
                    }
                }
            }
            drawn += 1;
        }
        drawn
    }

    /// Draw one line axis-aligned with its frame origin at (x, y) (horizontal: start of the
    /// baseline). Returns its page box if drawn (inside the page, keep-clear rule satisfied).
    pub(crate) fn put_line(&mut self, l: LineG, x: f64, y: f64, lum: f32, kind: Kind, fkind: &'static str) -> Option<[f64; 4]> {
        let b = [l.b[0] + x, l.b[1] + y, l.b[2] + x, l.b[3] + y];
        if b[0] < 1.0 || b[1] < 1.0 || b[2] > self.w as f64 - 2.0 || b[3] > self.h as f64 - 2.0 {
            return None;
        }
        let em = (l.b[3] - l.b[1]).max(1.0);
        let g = Group { lines: vec![l], offs: vec![[0.0, 0.0]], w: b[2] - b[0], h: b[3] - b[1], vertical: false, em };
        (self.draw_lines(&g, &tr(x, y), lum, &Fx::default(), kind, 0.0, fkind) == 1).then_some(b)
    }

    /// Draw one line through the frame -> page map `m` with effects `fx`. Returns its page quad
    /// (first run) if drawn (keep-clear rule satisfied, inside the page).
    pub(crate) fn put_line_fx(&mut self, l: LineG, m: &H3, lum: f32, fx: &Fx, kind: Kind, fkind: &'static str) -> Option<[P; 4]> {
        let q = box_quad(l.b, l.vertical, 0.0).map(|p| apply(m, p));
        let bb = bbox(&q);
        let e = fx.extent();
        if bb[0] < 1.0 + e || bb[1] < 1.0 + e || bb[2] > self.w as f64 - 2.0 - e || bb[3] > self.h as f64 - 2.0 - e {
            return None;
        }
        if l.hxh() && self.overlaps_hunter(&q) {
            return None;
        }
        let em = (l.b[3] - l.b[1]).max(1.0);
        let g = Group { lines: vec![l], offs: vec![[0.0, 0.0]], w: bb[2] - bb[0], h: bb[3] - bb[1], vertical: false, em };
        (self.draw_lines(&g, m, lum, fx, kind, e, fkind) == 1).then_some(q)
    }

    /// v6: would a Hunter quad placed directly (scenes) overlap a labelled line already there?
    /// (Scenes space their lines by the requested text; v6's taller digits and longer numbers can
    /// make a line outgrow that.) Always false before v6, so older sets reproduce exactly.
    fn overlaps_hunter(&self, q: &[P; 4]) -> bool {
        let bb = bbox(q);
        self.cfg.v6 && self.hq.iter().any(|o| bbox_gap(bb, o.bb) < 1.0 && convex_dist(q, &o.q) < 1.0)
    }

    /// Draw lines that share one frame (a Hunter line and its reading guide) through `m` with
    /// effects `fx`. Returns the first Hunter line's page quad if it was drawn.
    pub(crate) fn put_lines_fx(&mut self, lines: Vec<LineG>, m: &H3, lum: f32, fx: &Fx, kind: Kind, fkind: &'static str) -> Option<[P; 4]> {
        let first = lines.iter().position(|l| l.hxh())?;
        let q = box_quad(lines[first].b, false, 0.0).map(|p| apply(m, p));
        let e = fx.extent();
        for l in &lines {
            let bb = bbox(&box_quad(l.b, false, 0.0).map(|p| apply(m, p)));
            if bb[0] < 1.0 + e || bb[1] < 1.0 + e || bb[2] > self.w as f64 - 2.0 - e || bb[3] > self.h as f64 - 2.0 - e {
                return None;
            }
        }
        if self.overlaps_hunter(&q) {
            return None;
        }
        let n = lines.len();
        let em = (lines[first].b[3] - lines[first].b[1]).max(1.0);
        let g = Group { lines, offs: vec![[0.0, 0.0]; n], w: 0.0, h: 0.0, vertical: false, em };
        let h0 = self.lines.len();
        self.draw_lines(&g, m, lum, fx, kind, e, fkind);
        (self.lines.len() > h0).then_some(q)
    }

    /// Is the page rect free of placed text and containers?
    pub(crate) fn free(&self, r: IRect) -> bool {
        let r = r.intersect(self.page_rect());
        if r.is_empty() {
            return false;
        }
        (r.y0..r.y1).all(|y| self.occ[y as usize * self.w + r.x0 as usize..y as usize * self.w + r.x1 as usize].iter().all(|&v| v == 0))
    }

    pub(crate) fn mark(&mut self, r: IRect) {
        let r = r.intersect(self.page_rect());
        if r.is_empty() {
            return;
        }
        for y in r.y0..r.y1 {
            self.occ[y as usize * self.w + r.x0 as usize..y as usize * self.w + r.x1 as usize].fill(255);
        }
    }

    /// Would an unlabelled run with this quad keep clear of every labelled line?
    pub(crate) fn foreign_clear(&self, q: &[P; 4], halo: f64) -> bool {
        self.clear_of(q, false, halo)
    }

    /// Record an unlabelled distractor drawn by hand (avatars, round icons, big background titles).
    pub(crate) fn record_foreign(&mut self, q: [P; 4], text: &str, script: &'static str, kind: &'static str, halo: f64) {
        self.foreign.push(ForeignOut { quad: round1(q), text: text.to_string(), script, font: String::new(), orient: "h", kind });
        self.fq.push(Keep { q, bb: bbox(&q), h: line_thickness(&q), halo });
    }

    /// Labelled quads placed so far.
    pub(crate) fn hunter_quads(&self) -> Vec<[P; 4]> {
        self.hq.iter().map(|k| k.q).collect()
    }

    /// Draw a group axis-aligned with its top-left at (x, y). Returns lines drawn.
    pub(crate) fn put_group(&mut self, g: &Group, x: f64, y: f64, lum: f32, kind: Kind, fkind: &'static str) -> usize {
        if x < 1.0 || y < 1.0 || x + g.w > self.w as f64 - 2.0 || y + g.h > self.h as f64 - 2.0 {
            return 0;
        }
        self.draw_lines(g, &tr(x, y), lum, &Fx::default(), kind, 0.0, fkind)
    }

    /// Script for a foreign element of `kind` (vertical: Japanese only).
    fn foreign_script(&mut self, kind: Kind, vertical: bool) -> Script {
        let w: [f64; 9] = match kind {
            // latin romaji url digits japanese kana korean greek cyrillic
            Kind::Bubble => [0.45, 0.06, 0.0, 0.01, 0.28, 0.12, 0.03, 0.02, 0.03],
            Kind::Caption => [0.45, 0.08, 0.02, 0.03, 0.25, 0.08, 0.03, 0.03, 0.03],
            Kind::Sign => [0.35, 0.06, 0.04, 0.1, 0.25, 0.1, 0.04, 0.03, 0.03],
            Kind::Block => [0.45, 0.08, 0.02, 0.02, 0.28, 0.06, 0.03, 0.03, 0.03],
            Kind::Free => [0.45, 0.08, 0.0, 0.04, 0.15, 0.2, 0.03, 0.02, 0.03],
            Kind::Filler => [0.3, 0.08, 0.1, 0.22, 0.15, 0.08, 0.03, 0.02, 0.02],
            Kind::Cell => [0.25, 0.15, 0.02, 0.25, 0.12, 0.15, 0.02, 0.02, 0.02],
        };
        if vertical {
            return if self.rng.chance(0.7) { Script::Japanese } else { Script::Kana };
        }
        let mut w = w;
        for (i, s) in SCRIPTS.iter().enumerate() {
            if !self.fonts.has_script(s.need()) {
                w[i] = 0.0;
            }
        }
        if w.iter().sum::<f64>() <= 0.0 {
            return Script::Latin;
        }
        SCRIPTS[self.rng.weighted(&w)]
    }

    /// Lines of an icon element: one row (single icon or 3-12), a grid of rows, or (signs,
    /// buttons) icon + label.
    fn icon_lines(&mut self, kind: Kind, em: f64, max_extent: f64) -> Vec<LineG> {
        let fonts = self.fonts;
        let (icon_faces, sym_faces) = fonts.icon_faces();
        let rng = &mut *self.rng;
        let face = if !icon_faces.is_empty() && (sym_faces.is_empty() || rng.chance(0.75)) {
            let w: Vec<f64> = icon_faces.iter().map(|&i| if fonts.faces[i].name.to_lowercase().starts_with("seg") { 3.0 } else { 1.0 }).collect();
            icon_faces[rng.weighted(&w)]
        } else if !sym_faces.is_empty() {
            sym_faces[rng.below(sym_faces.len())]
        } else {
            return vec![];
        };
        let hollow = rng.chance(0.2);
        let single = rng.chance(if kind == Kind::Filler { 0.45 } else { 0.2 });
        let n = if single { 1 } else { rng.int(3, 13) as usize };
        let track = rng.uniform(0.15, 1.2);
        if kind == Kind::Sign && rng.chance(0.4) {
            // a button: icon + label
            let script = if rng.chance(0.8) { Script::Latin } else { Script::Japanese };
            let Some(fs) = sample_fstyle(rng, fonts, script, Kind::Cell) else { return vec![] };
            let iem = em * rng.uniform(0.9, 1.3);
            let icon = make_icon_line(rng, fonts, face, iem, 1, 0.0, false, max_extent / 3.0);
            let label = make_font_line(rng, &self.tg, fonts, &fs, em, false, Role::Phrase, max_extent * 0.6, 16, None);
            return match (icon, label) {
                (Some(i), Some(l)) => compose(vec![i, l], em * rng.uniform(0.3, 0.7)).into_iter().collect(),
                _ => vec![],
            };
        }
        let rows = if !single && rng.chance(0.2) { rng.int(2, 5) as usize } else { 1 };
        (0..rows).filter_map(|_| make_icon_line(rng, fonts, face, em, n, track, hollow, max_extent)).collect()
    }

    /// Add one element of `kind` inside `region` (x0, y0, x1, y1): Hunter text, foreign text,
    /// both, or icons (`want`). Returns lines placed.
    pub(crate) fn add(&mut self, kind: Kind, region: [f64; 4], want: Want) -> usize {
        let sp = spec(kind);
        let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
        let rmin = rw.min(rh);
        let cfg = self.cfg;
        let has_fonts = !self.fonts.faces.is_empty();
        #[derive(PartialEq)]
        enum C {
            Hxh,
            Foreign,
            Mixed,
            Icons,
        }
        let content = match want {
            _ if !has_fonts && (want != Want::Auto || self.neg) => return 0,
            Want::Icons => C::Icons,
            Want::Foreign => C::Foreign,
            Want::Auto if self.neg => C::Foreign,
            Want::Auto if has_fonts && self.rng.chance(cfg.foreign_prob * self.fboost) => C::Foreign,
            Want::Auto if has_fonts && matches!(kind, Kind::Bubble | Kind::Caption | Kind::Sign | Kind::Block) && self.rng.chance(cfg.mixed_prob) => C::Mixed,
            Want::Auto => C::Hxh,
        };
        let small_ok = matches!(kind, Kind::Filler | Kind::Cell | Kind::Block | Kind::Caption);
        let em_lo = match content {
            C::Hxh | C::Mixed => cfg.em_min,
            C::Icons => 10.0,
            C::Foreign if small_ok && self.rng.chance(0.3) => cfg.foreign_em_min,
            C::Foreign => cfg.em_min,
        };
        if rmin < 2.0 * em_lo {
            return 0;
        }
        // Japanese manga dialogue is mostly vertical: foreign bubbles / captions go vertical more often
        let vp = if content == C::Foreign && matches!(kind, Kind::Bubble | Kind::Caption) { 0.4 } else { cfg.vertical_prob * sp.vmul };
        let vertical = content != C::Icons && self.rng.chance(vp) && (content == C::Hxh || self.fonts.has_script(crate::fonts::JAPANESE));
        let lo = match content {
            C::Icons => 10.0,
            C::Foreign if em_lo < cfg.em_min => em_lo,
            _ => sp.em.0.max(cfg.em_min),
        };
        let (hi_cap, frac) = if content == C::Icons { (64.0, sp.em.2.max(0.25)) } else { (sp.em.1.min(cfg.em_max), sp.em.2) };
        let hi = hi_cap.min(rmin * frac).max(lo);
        let mut em = self.rng.log_uniform(lo, hi);
        let mut n = sp.n[self.rng.weighted(sp.np)];
        if content == C::Mixed {
            n = n.max(2);
        }
        let fs = match content {
            C::Foreign | C::Mixed => {
                let script = self.foreign_script(kind, vertical);
                match sample_fstyle(self.rng, self.fonts, script, kind) {
                    Some(f) => Some(f),
                    None => return 0,
                }
            }
            _ => None,
        };
        let role = match kind {
            Kind::Bubble | Kind::Caption => Role::Sentence,
            Kind::Sign => {
                if self.rng.chance(0.7) {
                    Role::Phrase
                } else {
                    Role::Label
                }
            }
            Kind::Block => Role::Paragraph,
            Kind::Free => {
                if self.rng.chance(0.6) {
                    Role::Sfx
                } else {
                    Role::Heading
                }
            }
            Kind::Filler => Role::Label,
            Kind::Cell => Role::Word,
        };
        // v6: a reading guide (furigana) above each line of a horizontal Hunter group
        let ruby_k = match kind {
            Kind::Free | Kind::Sign => 1.0,
            Kind::Caption | Kind::Filler => 0.5,
            Kind::Bubble | Kind::Block => 0.25,
            Kind::Cell => 0.0,
        };
        let ruby = cfg.ruby_prob > 0.0 && content == C::Hxh && !vertical && self.fonts.has_script(crate::fonts::JAPANESE) && self.rng.chance(cfg.ruby_prob * ruby_k);
        for _ in 0..2 {
            let max_len = self.rng.int(sp.len.0, sp.len.1) as usize;
            let ext = (if vertical { rh } else { rw }) * sp.ext;
            let mut align = sp.align[self.rng.below(sp.align.len())];
            if vertical && matches!(kind, Kind::Bubble | Kind::Caption | Kind::Block) && self.rng.chance(0.75) {
                align = b'l'; // vertical columns hang from the top
            }
            let gap = self.rng.uniform(sp.gap.0, sp.gap.1) * em;
            let gap = if ruby { gap.max(em * 0.9) } else { gap };
            let lines: Vec<LineG> = match content {
                C::Hxh => {
                    let st = style_for(self.rng, cfg);
                    let mut ls = hxh_lines(self.rng, self.sampler, cfg, &st, em, vertical, n, max_len, ext);
                    if cfg.inline_prob > 0.0 && !vertical && has_fonts {
                        for l in ls.iter_mut() {
                            if self.rng.chance(cfg.inline_prob)
                                && let Some(x) = inline_line(self.rng, self.sampler, self.fonts, cfg, &st, em, max_len, ext) {
                                    *l = x;
                                }
                        }
                    }
                    ls
                }
                C::Icons => self.icon_lines(kind, em, ext),
                C::Foreign | C::Mixed => {
                    let fs = fs.as_ref().unwrap();
                    let max_chars = if fs.script.by_char() { max_len + 2 } else { 2 * max_len + 4 };
                    let slots: Vec<bool> = if content == C::Mixed {
                        let k = self.rng.below(n);
                        (0..n).map(|i| if i == k { false } else if i == (k + 1) % n { true } else { self.rng.chance(0.5) }).collect()
                    } else {
                        vec![false; n]
                    };
                    let st = style_for(self.rng, cfg);
                    let mut out = Vec::new();
                    if content == C::Foreign && kind == Kind::Block {
                        // a paragraph wrapped to the column
                        let face = &self.fonts.faces[fs.face];
                        let per = if fs.script.by_char() { ext / em } else { ext / (0.5 * em) };
                        let text = self.tg.text(self.rng, fs.script, Role::Paragraph, (per * n as f64) as usize + 8);
                        let text = if fs.caps { text.to_uppercase() } else { text };
                        let rows: Vec<String> = if vertical {
                            let k = (ext / (em * (1.0 + fs.track))).floor().max(1.0) as usize;
                            text.chars().collect::<Vec<_>>().chunks(k).map(|c| c.iter().collect()).collect()
                        } else {
                            face.wrap(&face.filter(&text), em, ext - em * (fs.shear + fs.bold + 0.1), fs.script.by_char())
                        };
                        for r in rows.iter().take(n) {
                            if let Some(l) = make_font_line(self.rng, &self.tg, self.fonts, fs, em, vertical, role, ext, 400, Some(r)) {
                                out.push(l);
                            }
                        }
                    } else {
                        for &h in &slots {
                            let l = if h {
                                make_line(self.rng, self.sampler, &st, em, vertical, max_len, ext, cfg.min_stroke_px)
                            } else {
                                let ml = if self.rng.chance(0.7) { max_chars } else { ((max_chars as f64 * self.rng.uniform(0.3, 1.0)) as usize).max(1) };
                                make_font_line(self.rng, &self.tg, self.fonts, fs, em, vertical, role, ext, ml, None)
                            };
                            out.extend(l);
                        }
                    }
                    out
                }
            };
            let rubies: Vec<Option<LineG>> = if ruby { lines.iter().map(|l| ruby_line(self.rng, self.fonts, l, em)).collect() } else { Vec::new() };
            let Some(mut g) = stack(lines, vertical, em, gap, align, cfg.keep_margin) else { return 0 };
            if ruby {
                attach_ruby(&mut g, rubies);
            }
            if self.place_group(kind, &g, region, &sp) {
                return g.lines.len();
            }
            em = (em * 0.7).max(em_lo);
            n = (n - 1).max(if content == C::Mixed { 2 } else { 1 });
        }
        0
    }

    fn place_group(&mut self, kind: Kind, g: &Group, region: [f64; 4], sp: &Spec) -> bool {
        let em = g.em;
        let mut polys: Vec<Vec<P>> = Vec::new();
        let (mut border, mut border_lum, mut fill, mut ink) = (0.0f64, 0.0f32, 1.0f32, 0.0f32);
        let mut shadow: Option<Shadow> = None;
        let mut fx = Fx::default();
        let rng = &mut *self.rng;
        let framed_block = kind == Kind::Block && rng.chance(0.2);
        match kind {
            Kind::Bubble => {
                polys = bubble_polys(rng, g);
                border = rng.pick(&[1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
                if rng.chance(0.08) {
                    (fill, ink, border_lum) = (rng.uniform(0.0, 0.15) as f32, rng.uniform(0.85, 1.0) as f32, rng.f64() as f32);
                } else {
                    (fill, ink, border_lum) = (rng.uniform(0.92, 1.0) as f32, rng.uniform(0.0, 0.18) as f32, rng.uniform(0.0, 0.15) as f32);
                }
            }
            Kind::Caption => {
                polys = box_polys(rng, g, (0.3, 0.8), (0.25, 0.6), 0.1);
                border = rng.pick(&[1.0, 1.0, 2.0, 2.0, 3.0]);
                if rng.chance(0.15) {
                    (fill, ink, border_lum) = (rng.uniform(0.0, 0.15) as f32, rng.uniform(0.85, 1.0) as f32, rng.uniform(0.0, 0.2) as f32);
                } else {
                    (fill, ink, border_lum) = (rng.uniform(0.9, 1.0) as f32, rng.uniform(0.0, 0.2) as f32, rng.uniform(0.0, 0.15) as f32);
                }
            }
            Kind::Sign | Kind::Block if kind == Kind::Sign || framed_block => {
                polys = box_polys(rng, g, (0.3, 1.2), (0.2, 0.8), 0.4);
                let f = if kind == Kind::Sign { rng.f64() } else { rng.uniform(0.75, 1.0) };
                let i = if f > 0.5 { rng.uniform(0.0, (f - 0.45).max(0.0)) } else { rng.uniform((f + 0.45).min(1.0), 1.0) };
                (fill, ink) = (f as f32, i as f32);
                if rng.chance(0.6) {
                    border = rng.pick(&[1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);
                    border_lum = if rng.chance(0.6) { ink } else { rng.f64() as f32 };
                }
                if rng.chance(0.25) {
                    let (dx, dy) = if rng.chance(0.6) {
                        (rng.uniform(0.04, 0.25) * em, rng.uniform(0.04, 0.25) * em) // soft down-right, like the anime cards
                    } else {
                        (rng.uniform(-0.25, 0.25) * em, rng.uniform(-0.25, 0.25) * em)
                    };
                    shadow = Some(Shadow { dx, dy, sigma: rng.uniform(0.5, 3.0), strength: rng.uniform(0.3, 0.8) as f32, lum: rng.uniform(0.0, 0.25) as f32 });
                }
            }
            Kind::Cell => {}
            _ if self.cfg.sticker_prob > 0.0 && rng.chance(self.cfg.sticker_prob) => {
                // v6 sticker lettering: a thick outline (white around dark ink, or the inverse)
                // around every stroke, often with the outline's own soft shadow
                let r = (rng.uniform(0.1, 0.34) * em).round().max(1.5);
                fx.outline = Some((r, 0.0));
                fx.sticker = true;
                if rng.chance(0.75) {
                    let (dx, dy) = (rng.uniform(-0.04, 0.1) * em, rng.uniform(0.0, 0.12) * em);
                    fx.shadow = Some(Shadow { dx, dy, sigma: rng.uniform(0.03, 0.12) * em, strength: rng.uniform(0.25, 0.7) as f32, lum: 0.0 });
                    fx.shadow_grow = r;
                }
            }
            _ => {
                // free-standing text: outline / shadow sized now, ink chosen once we see the background
                fx.outline = Some((((rng.uniform(0.05, 0.14) * em).round()).max(1.0), 0.0));
                if rng.chance(0.18) {
                    let (dx, dy) = if rng.chance(0.6) {
                        (rng.uniform(0.04, 0.12) * em, rng.uniform(0.04, 0.12) * em)
                    } else {
                        (rng.uniform(-0.12, 0.12) * em, rng.uniform(-0.12, 0.12) * em)
                    };
                    fx.shadow = Some(Shadow { dx, dy, sigma: rng.uniform(0.5, 2.5), strength: rng.uniform(0.4, 1.0) as f32, lum: 0.0 });
                }
            }
        }
        fx.patchy = rng.chance(0.08);
        if self.cfg.v5 && polys.is_empty() && rng.chance(0.3) {
            // room for a glow / soft stroke / bevel chosen once the background is known
            fx.halo = Some(Halo { grow: em * 0.25, sigma: em * 0.15, lum: 0.0, strength: 0.0 });
        }
        let halo: f64 = fx.extent();
        fx.halo = None;
        let margin = if kind == Kind::Cell { 1.0 } else { (0.15 * em).max(2.0) } + halo;
        let mut ext_pts: Vec<P> = vec![[0.0, 0.0], [g.w, g.h]];
        for p in &polys {
            ext_pts.extend(p);
        }
        let ext = bbox(&ext_pts);
        let persp = if rng.chance(sp.persp) { rng.uniform(0.03, 0.12) } else { 0.0 };
        let ang = angle(rng, sp.rot);
        let m0 = group_matrix(rng, ext, ang, persp);
        let mut region = region;
        if kind == Kind::Bubble {
            // bubbles may cross panel borders a little
            let e = 0.08 * (region[2] - region[0]).min(region[3] - region[1]);
            region = [(region[0] - e).max(2.0), (region[1] - e).max(2.0), (region[2] + e).min(self.w as f64 - 2.0), (region[3] + e).min(self.h as f64 - 2.0)];
        }
        let sh_pad = shadow.map_or(0.0, |s| (s.dx.abs() + s.dy.abs() + 3.0 * s.sigma).ceil() + 1.0);
        let Some((m, roi, fp)) = self.try_place(g, &m0, &polys, margin, border + sh_pad, region, 10, halo) else { return false };
        if !polys.is_empty() {
            self.draw_container(&m, &polys, fill, border, border_lum, shadow);
        } else {
            // free-standing: read the background under the footprint
            let (mut s, mut s2, mut cnt) = (0.0f64, 0.0f64, 0usize);
            for y in 0..roi.h() {
                for x in 0..roi.w() {
                    if fp[y * roi.w() + x] != 0 {
                        let v = self.page.d[(y as i32 + roi.y0) as usize * self.w + (x as i32 + roi.x0) as usize] as f64;
                        s += v;
                        s2 += v * v;
                        cnt += 1;
                    }
                }
            }
            let mu = if cnt > 0 { s / cnt as f64 } else { 1.0 };
            let sd = if cnt > 0 { (s2 / cnt as f64 - mu * mu).max(0.0).sqrt() } else { 0.0 };
            let rng = &mut *self.rng;
            let dark = if (mu - 0.5).abs() > 0.08 { mu > 0.5 } else { rng.chance(0.5) };
            let i = if dark { rng.uniform(0.0, 0.18) } else { rng.uniform(0.82, 1.0) };
            let olum = if dark { rng.uniform(0.85, 1.0) } else { rng.uniform(0.0, 0.15) };
            ink = i as f32;
            let busy = sd > 0.15 || (i - mu).abs() < 0.5; // mid-grey or textured: outline, as titles / SFX do
            match fx.outline {
                Some((r, _)) if fx.sticker || rng.chance(if busy { 0.92 } else { 0.2 }) => fx.outline = Some((r, olum as f32)),
                _ => fx.outline = None,
            }
            if let Some(s) = fx.shadow.as_mut() {
                s.lum = (olum * 0.5 + rng.uniform(-0.1, 0.1)).clamp(0.0, 1.0) as f32;
                if dark {
                    s.lum = rng.uniform(0.3, 0.6) as f32; // a grey shadow under dark ink on a light page
                }
            }
            if self.cfg.v5 && !fx.sticker {
                v5_effects(rng, &mut fx, &mut ink, mu as f32, olum as f32, em, halo);
            }
        }
        let drawn_before = self.lines.len();
        self.draw_lines(g, &m, ink, &fx, kind, halo, kind.name());
        if self.cfg.v5 && self.lines.len() > drawn_before && matches!(kind, Kind::Free | Kind::Sign | Kind::Filler) && self.rng.chance(self.cfg.caption_prob) {
            let q = self.meta[self.meta.len() - 1].quad;
            crate::scenes::caption_near(self, &q, ink, g.vertical);
        }
        if !polys.is_empty() {
            for y in 0..roi.h() {
                let o = (y as i32 + roi.y0) as usize * self.w + roi.x0 as usize;
                for x in 0..roi.w() {
                    self.occ[o + x] |= fp[y * roi.w() + x];
                }
            }
        }
        true
    }
}

/// Grow a coverage rect by `pad` pixels (zeros), within `clip`, so a blur can spread.
fn pad_cov(c: Cov, pad: i32, clip: IRect) -> Cov {
    let r = IRect { x0: c.r.x0 - pad, y0: c.r.y0 - pad, x1: c.r.x1 + pad, y1: c.r.y1 + pad }.intersect(clip);
    let mut a = vec![0.0f32; r.w() * r.h()];
    for y in 0..c.r.h() {
        let dy = (y as i32 + c.r.y0 - r.y0) as usize;
        let dx = (c.r.x0 - r.x0) as usize;
        a[dy * r.w() + dx..dy * r.w() + dx + c.r.w()].copy_from_slice(&c.a[y * c.r.w()..(y + 1) * c.r.w()]);
    }
    Cov { r, a }
}

// --- backgrounds and art ----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Bg {
    White,
    Tone,
    Gradient,
    Speed,
    Hatch,
    Noise,
    Dark,
    Scene,
    Flat,
}

const PANEL_BG: [Bg; 9] = [Bg::White, Bg::Tone, Bg::Gradient, Bg::Speed, Bg::Hatch, Bg::Noise, Bg::Dark, Bg::Scene, Bg::Flat];
const PANEL_P: [f64; 9] = [0.34, 0.2, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06];

fn draw(page: &mut Gray, items: &[(Path, Draw)], clip: IRect, lum: f32) {
    if let Some(c) = coverage(items, clip) {
        blend(page, &c, lum, 1.0);
    }
}

fn lines_path(segs: &[(P, P)]) -> Option<Path> {
    let mut pb = PathBuilder::new();
    for (a, b) in segs {
        pb.move_to(a[0] as f32, a[1] as f32);
        pb.line_to(b[0] as f32, b[1] as f32);
    }
    pb.finish()
}

fn screentone(page: &mut Gray, r: IRect, rng: &mut Rng, base: f32, dot: f32) {
    let s = rng.uniform(3.0, 8.0);
    let a = rng.uniform(0.0, PI / 2.0);
    let (sa, ca) = a.sin_cos();
    let r0 = s * rng.uniform(0.15, 0.45);
    let grad = if rng.chance(0.4) { Some((rng.chance(0.5), rng.chance(0.5))) } else { None };
    let (w, h) = (r.w().max(2), r.h().max(2));
    for y in 0..r.h() {
        let row = (y as i32 + r.y0) as usize * page.w + r.x0 as usize;
        for x in 0..r.w() {
            let (fx, fy) = (x as f64, y as f64);
            let u = fx * ca + fy * sa;
            let v = -fx * sa + fy * ca;
            let du = u.rem_euclid(s) - s / 2.0;
            let dv = v.rem_euclid(s) - s / 2.0;
            let mut rr = r0;
            if let Some((along_x, flip)) = grad {
                let mut t = if along_x { fx / (w - 1) as f64 } else { fy / (h - 1) as f64 };
                if flip {
                    t = 1.0 - t;
                }
                rr *= 0.15 + t;
            }
            let f = (rr - du.hypot(dv) + 0.5).clamp(0.0, 1.0) as f32;
            page.d[row + x] = base + (dot - base) * f;
        }
    }
}

fn scene(w: usize, h: usize, rng: &mut Rng) -> Vec<f32> {
    let (l1, l2) = (rng.f64() as f32, rng.f64() as f32);
    let (sa, ca) = rng.uniform(0.0, TAU).sin_cos();
    let ts = [0.0, w as f64 * ca, h as f64 * sa, w as f64 * ca + h as f64 * sa];
    let (tmin, tmax) = (ts.iter().copied().fold(f64::INFINITY, f64::min), ts.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let mut img = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let t = ((x as f64 * ca + y as f64 * sa - tmin) / (tmax - tmin + 1e-6)) as f32;
            img[y * w + x] = l1 + (l2 - l1) * t;
        }
    }
    let (sw, sh) = ((w / 4).max(4), (h / 4).max(4));
    let mut shapes = Gray::new(sw, sh, 0.0);
    let mut mask = Gray::new(sw, sh, 0.0);
    let big = sw.max(sh) as f64;
    for _ in 0..rng.int(3, 12) {
        let col = rng.f64() as f32;
        let path = if rng.chance(0.5) {
            PathBuilder::from_circle(rng.uniform(0.0, sw as f64) as f32, rng.uniform(0.0, sh as f64) as f32, rng.uniform(3.0, big / 2.0) as f32)
        } else {
            let c = [rng.uniform(0.0, sw as f64), rng.uniform(0.0, sh as f64)];
            let pts: Vec<P> = (0..4).map(|_| [c[0] + rng.uniform(-1.0, 1.0) * big / 3.0, c[1] + rng.uniform(-1.0, 1.0) * big / 3.0]).collect();
            poly_path(&pts, true)
        };
        if let Some(p) = path
            && let Some(c) = coverage(&[(p, Draw::Fill)], shapes.rect()) {
                blend(&mut shapes, &c, col, 1.0);
                blend(&mut mask, &c, 1.0, 1.0);
            }
    }
    let k = rng.uniform(0.2, 0.7) as f32;
    blur(&mut shapes.d, sw, sh, rng.uniform(0.5, 4.0) / 4.0, true);
    let sh_up = resize_bilinear(&shapes.d, sw, sh, w, h);
    let m_up = resize_bilinear(&mask.d, sw, sh, w, h);
    let amp = rng.uniform(0.02, 0.1) as f32;
    let cell = rng.uniform(20.0, 120.0);
    let n = value_noise(w, h, rng, cell);
    for i in 0..w * h {
        let m = k * m_up[i];
        img[i] = (img[i] * (1.0 - m) + sh_up[i] * m + amp * n[i]).clamp(0.0, 1.0);
    }
    img
}

/// Fill rect `r` with a background; returns an ink lum that reads on it.
fn fill_bg(page: &mut Gray, r: IRect, rng: &mut Rng, kind: Bg) -> f32 {
    let (w, h) = (r.w(), r.h());
    match kind {
        Bg::White => page.fill(r, rng.uniform(0.9, 1.0) as f32),
        Bg::Flat => page.fill(r, rng.uniform(0.25, 1.0) as f32),
        Bg::Dark => page.fill(r, rng.uniform(0.0, 0.18) as f32),
        Bg::Tone => {
            let (base, dot) = (rng.uniform(0.85, 1.0) as f32, rng.uniform(0.0, 0.35) as f32);
            screentone(page, r, rng, base, dot);
        }
        Bg::Gradient => {
            let (l1, l2) = (rng.uniform(0.3, 1.0) as f32, rng.f64() as f32);
            let vert = rng.chance(0.6);
            for y in 0..h {
                let row = (y as i32 + r.y0) as usize * page.w + r.x0 as usize;
                for x in 0..w {
                    let t = if vert { y as f32 / (h.max(2) - 1) as f32 } else { x as f32 / (w.max(2) - 1) as f32 };
                    page.d[row + x] = l1 + (l2 - l1) * t;
                }
            }
        }
        Bg::Noise => {
            let base = rng.uniform(0.3, 1.0) as f32;
            let cell = rng.uniform(8.0, 80.0);
            let amp = rng.uniform(0.04, 0.18) as f32;
            let n = value_noise(w, h, rng, cell);
            for y in 0..h {
                let row = (y as i32 + r.y0) as usize * page.w + r.x0 as usize;
                for x in 0..w {
                    page.d[row + x] = (base + amp * n[y * w + x]).clamp(0.0, 1.0);
                }
            }
        }
        Bg::Speed | Bg::Hatch => {
            page.fill(r, rng.uniform(if kind == Bg::Speed { 0.9 } else { 0.8 }, 1.0) as f32);
            let col = rng.uniform(0.0, if kind == Bg::Speed { 0.3 } else { 0.4 }) as f32;
            let (cx, cy) = (r.x0 as f64 + w as f64 / 2.0, r.y0 as f64 + h as f64 / 2.0);
            let big = (w as f64).hypot(h as f64);
            let mut thin: Vec<(P, P)> = Vec::new();
            let mut thick: Vec<(P, P)> = Vec::new();
            if kind == Bg::Hatch {
                let sp = rng.uniform(3.0, 10.0);
                for _ in 0..(if rng.chance(0.6) { 1 } else { 2 }) {
                    let (sa, ca) = rng.uniform(0.0, PI).sin_cos();
                    let mut o = -big / 2.0;
                    while o < big / 2.0 {
                        let p0 = [cx - sa * o - ca * big, cy + ca * o - sa * big];
                        thin.push((p0, [p0[0] + 2.0 * ca * big, p0[1] + 2.0 * sa * big]));
                        o += sp;
                    }
                }
            } else if rng.chance(0.5) {
                // radial burst around a focus point
                let (fx, fy) = (r.x0 as f64 + rng.uniform(0.2, 0.8) * w as f64, r.y0 as f64 + rng.uniform(0.2, 0.8) * h as f64);
                let r0 = rng.uniform(0.1, 0.4) * w.min(h) as f64;
                for _ in 0..rng.int(60, 220) {
                    let t = rng.uniform(0.0, TAU);
                    let ri = r0 * rng.uniform(0.8, 1.8);
                    let seg = ([fx + ri * t.cos(), fy + ri * t.sin()], [fx + big * t.cos(), fy + big * t.sin()]);
                    if rng.chance(0.5) { thin.push(seg) } else { thick.push(seg) }
                }
            } else {
                // parallel motion lines
                let (sa, ca) = rng.uniform(0.0, PI).sin_cos();
                for _ in 0..rng.int(30, 120) {
                    let o = rng.uniform(-big / 2.0, big / 2.0);
                    let s0 = rng.uniform(-big / 2.0, big / 4.0);
                    let l = rng.uniform(0.2, 1.0) * big;
                    let p0 = [cx - sa * o + ca * s0, cy + ca * o + sa * s0];
                    let seg = (p0, [p0[0] + ca * l, p0[1] + sa * l]);
                    if rng.chance(0.5) { thin.push(seg) } else { thick.push(seg) }
                }
            }
            for (segs, wd) in [(thin, 1.0f32), (thick, 2.0f32)] {
                if let Some(p) = lines_path(&segs) {
                    draw(page, &[(p, Draw::Stroke(wd))], r, col);
                }
            }
        }
        Bg::Scene => {
            let s = scene(w, h, rng);
            for y in 0..h {
                let row = (y as i32 + r.y0) as usize * page.w + r.x0 as usize;
                page.d[row..row + w].copy_from_slice(&s[y * w..(y + 1) * w]);
            }
        }
    }
    if page.mean(r) > 0.45 { rng.uniform(0.0, 0.2) as f32 } else { rng.uniform(0.8, 1.0) as f32 }
}

#[derive(Clone, Copy, PartialEq)]
enum Art {
    Curve,
    Hair,
    Blob,
    Ring,
    Poly,
    Grid,
    Scribble,
    Spark,
    Rays,
}

const ART: [Art; 9] = [Art::Curve, Art::Hair, Art::Blob, Art::Ring, Art::Poly, Art::Grid, Art::Scribble, Art::Spark, Art::Rays];
const ART_P: [f64; 9] = [0.22, 0.1, 0.17, 0.1, 0.1, 0.05, 0.12, 0.06, 0.08];

/// Non-text drawing inside `clip`: `ink` is the line lum, `scale` a typical size in px.
fn art(page: &mut Gray, clip: IRect, rng: &mut Rng, n: usize, ink: f32, scale: f64, kinds: Option<&[Art]>) {
    let (w, h) = (clip.w() as f64, clip.h() as f64);
    for _ in 0..n {
        let k = match kinds {
            Some(ks) => rng.pick(ks),
            None => ART[rng.weighted(&ART_P)],
        };
        let s = scale * rng.log_uniform(0.4, 4.0);
        let (cx, cy) = (clip.x0 as f64 + rng.uniform(-0.1, 1.1) * w, clip.y0 as f64 + rng.uniform(-0.1, 1.1) * h);
        let th = rng.pick(&[1.0f32, 1.0, 2.0, 2.0, 3.0, 4.0, 6.0]);
        let col = (ink + rng.normal(0.0, 0.08) as f32).clamp(0.0, 1.0);
        let fill = rng.f64() as f32;
        let rnd = |rng: &mut Rng| [cx + rng.uniform(-1.0, 1.0) * s, cy + rng.uniform(-1.0, 1.0) * s];
        match k {
            Art::Curve | Art::Hair => {
                let c = [rnd(rng), rnd(rng), rnd(rng), rnd(rng)];
                if k == Art::Curve {
                    if let Some(p) = poly_path(&bezier(&c, 32), false) {
                        draw(page, &[(p, Draw::Stroke(th))], clip, col);
                    }
                } else {
                    let d = [c[3][0] - c[0][0], c[3][1] - c[0][1]];
                    let l = d[0].hypot(d[1]) + 1e-6;
                    let nv = [-d[1] / l, d[0] / l];
                    let sp = rng.uniform(2.0, 8.0);
                    let mut items = Vec::new();
                    for j in 0..rng.int(3, 9) {
                        let q = c.map(|p| [p[0] + nv[0] * j as f64 * sp + rng.normal(0.0, sp * 0.3), p[1] + nv[1] * j as f64 * sp + rng.normal(0.0, sp * 0.3)]);
                        if let Some(p) = poly_path(&bezier(&q, 32), false) {
                            items.push((p, Draw::Stroke((th / 2.0).max(1.0))));
                        }
                    }
                    draw(page, &items, clip, col);
                }
            }
            Art::Blob => {
                let poly = blob(rng, cx, cy, s / 2.0, 48);
                if let Some(p) = poly_path(&poly, true) {
                    draw(page, &[(p.clone(), Draw::Fill)], clip, fill);
                    if rng.chance(0.8) {
                        draw(page, &[(p, Draw::Stroke(th))], clip, col);
                    }
                }
            }
            Art::Ring => {
                let (rx, ry) = (s / 2.0, s / 2.0 * rng.uniform(0.4, 1.0));
                let ang = rng.uniform(0.0, 180.0);
                if let Some(p) = ellipse_path(cx, cy, rx, ry, ang) {
                    draw(page, &[(p, Draw::Stroke(th))], clip, col);
                }
                if rng.chance(0.4)
                    && let Some(p) = PathBuilder::from_circle(cx as f32, cy as f32, (s * rng.uniform(0.08, 0.2)) as f32) {
                        draw(page, &[(p, Draw::Fill)], clip, col);
                    }
            }
            Art::Poly => {
                let poly: Vec<P> = if rng.chance(0.5) {
                    let (a, b) = (s * rng.uniform(0.3, 2.0), s * rng.uniform(0.3, 2.0));
                    vec![[cx, cy], [cx + s, cy], [cx + s, cy + a], [cx, cy + b]]
                } else {
                    (0..rng.int(3, 6)).map(|_| rnd(rng)).collect()
                };
                if let Some(p) = poly_path(&poly, true) {
                    draw(page, &[(p.clone(), Draw::Fill)], clip, fill);
                    draw(page, &[(p, Draw::Stroke(th))], clip, col);
                }
            }
            Art::Grid => {
                let (gw, gh) = (s * rng.uniform(1.0, 3.0), s * rng.uniform(1.0, 3.0));
                let (nx, ny) = (rng.int(2, 6), rng.int(2, 6));
                let mut segs = Vec::new();
                for i in 0..=nx {
                    let x = cx + gw * i as f64 / nx as f64;
                    segs.push(([x, cy], [x, cy + gh]));
                }
                for j in 0..=ny {
                    let y = cy + gh * j as f64 / ny as f64;
                    segs.push(([cx, y], [cx + gw, y]));
                }
                if let Some(p) = lines_path(&segs) {
                    draw(page, &[(p, Draw::Stroke((th / 2.0).max(1.0)))], clip, col);
                }
            }
            Art::Scribble => {
                let ss = s.min(scale * 1.5);
                let mut p = [cx, cy];
                let mut pts = Vec::new();
                for _ in 0..rng.int(3, 10) {
                    p = [p[0] + rng.normal(0.0, ss / 3.0), p[1] + rng.normal(0.0, ss / 3.0)];
                    pts.push(p);
                }
                if let Some(path) = poly_path(&pts, false) {
                    draw(page, &[(path, Draw::Stroke(th.min(3.0)))], clip, col);
                }
            }
            Art::Spark => {
                let mut segs = Vec::new();
                for _ in 0..rng.int(4, 10) {
                    let t = rng.uniform(0.0, TAU);
                    let l = s * rng.uniform(0.3, 1.0);
                    segs.push(([cx, cy], [cx + l * t.cos(), cy + l * t.sin()]));
                }
                if let Some(p) = lines_path(&segs) {
                    draw(page, &[(p, Draw::Stroke((th / 2.0).max(1.0)))], clip, col);
                }
            }
            Art::Rays => {
                let mut segs = Vec::new();
                let l = w.hypot(h);
                for _ in 0..rng.int(2, 6) {
                    let p0 = [clip.x0 as f64 + rng.uniform(0.0, w), clip.y0 as f64 + rng.uniform(0.0, h)];
                    let (st, ct) = rng.uniform(0.0, PI).sin_cos();
                    segs.push(([p0[0] - l * ct, p0[1] - l * st], [p0[0] + l * ct, p0[1] + l * st]));
                }
                if let Some(p) = lines_path(&segs) {
                    draw(page, &[(p, Draw::Stroke(th))], clip, col);
                }
            }
        }
    }
}

/// `art` for the v5 scenes (any kind).
pub(crate) fn art_pub(page: &mut Gray, clip: IRect, rng: &mut Rng, n: usize, ink: f32, scale: f64) {
    art(page, clip, rng, n, ink, scale, None);
}

/// A scene background (`scene`) or a random panel background for the v5 scenes.
pub(crate) fn fill_bg_pub(page: &mut Gray, r: IRect, rng: &mut Rng, scene: bool) -> f32 {
    let kind = if scene && rng.chance(0.75) { Bg::Scene } else { [Bg::Gradient, Bg::Noise, Bg::Tone, Bg::Flat, Bg::Dark][rng.below(5)] };
    fill_bg(page, r, rng, kind)
}

/// Distractors drawn after the text, clipped out of every quad and container: v4-style art
/// fragments right at line ends (blobs with outlines, spiky hair, strokes, scribbles, bars) and a
/// few loose strokes anywhere.
fn post_art(c: &mut Composer) {
    let clip = c.page_rect();
    let free: Vec<usize> = (0..c.meta.len()).filter(|&i| matches!(c.meta[i].kind, Kind::Free | Kind::Filler | Kind::Block)).collect();
    let mut jobs: Vec<usize> = Vec::new();
    if !free.is_empty() && c.rng.chance(0.45) {
        let mut pool = free.clone();
        for _ in 0..c.rng.int(1, 4).min(pool.len() as i64) {
            let i = c.rng.below(pool.len());
            jobs.push(pool.swap_remove(i));
        }
    }
    for j in jobs {
        let (q, ink) = (c.meta[j].quad, c.meta[j].ink);
        let rng = &mut *c.rng;
        let mut d = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
        let l = d[0].hypot(d[1]) + 1e-6;
        d = [d[0] / l, d[1] / l];
        let t = line_thickness(&q).max(4.0);
        let (anchor, dir) = if rng.chance(0.6) {
            ([(q[1][0] + q[2][0]) / 2.0, (q[1][1] + q[2][1]) / 2.0], d)
        } else {
            ([(q[0][0] + q[3][0]) / 2.0, (q[0][1] + q[3][1]) / 2.0], [-d[0], -d[1]])
        };
        let g = t * rng.uniform(0.12, 0.5);
        let a = [anchor[0] + dir[0] * g, anchor[1] + dir[1] * g];
        let s = t * rng.uniform(0.6, 2.5);
        let th = rng.int(1, ((t * 0.15) as i64 + 2).max(2)) as f32;
        let u = rng.f64();
        let mut items: Vec<(Path, Draw)> = Vec::new();
        let mut fill_lum = ink;
        if u < 0.35 {
            // strokes heading away from the line
            for _ in 0..rng.int(1, 4) {
                let ang = dir[1].atan2(dir[0]) + rng.uniform(-1.0, 1.0);
                let e = [a[0] + s * ang.cos(), a[1] + s * ang.sin()];
                let c4 = [a, [a[0] + (e[0] - a[0]) * 0.33 + rng.normal(0.0, s * 0.3), a[1] + (e[1] - a[1]) * 0.33 + rng.normal(0.0, s * 0.3)], [a[0] + (e[0] - a[0]) * 0.66 + rng.normal(0.0, s * 0.3), a[1] + (e[1] - a[1]) * 0.66 + rng.normal(0.0, s * 0.3)], e];
                if let Some(p) = poly_path(&bezier(&c4, 24), false) {
                    items.push((p, Draw::Stroke(th)));
                }
            }
        } else if u < 0.6 {
            // blob (filled, outlined) or spiky hair next to the line end
            let centre = [a[0] + dir[0] * s * 0.6, a[1] + dir[1] * s * 0.6];
            let spiky = rng.chance(0.4);
            let n = if spiky { 2 * rng.int(5, 10) as usize } else { rng.int(7, 15) as usize };
            let mut angs: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, TAU)).collect();
            angs.sort_by(f64::total_cmp);
            let pts: Vec<P> = angs
                .iter()
                .enumerate()
                .map(|(i, &tt)| {
                    let r = s * 0.6 * if spiky { if i % 2 == 0 { 1.0 } else { rng.uniform(0.3, 0.6) } } else { rng.uniform(0.55, 1.0) };
                    [centre[0] + r * tt.cos(), centre[1] + r * tt.sin()]
                })
                .collect();
            if let Some(p) = poly_path(&pts, true) {
                fill_lum = if spiky { rng.uniform(0.03, 0.3) as f32 } else { rng.uniform(0.2, 0.95) as f32 };
                let mut fc = coverage(&[(p.clone(), Draw::Fill)], clip);
                if let Some(fc) = fc.as_mut() {
                    clip_by(fc, &c.occ, c.w);
                    blend(&mut c.page, fc, fill_lum, 1.0);
                }
                items.push((p, Draw::Stroke(th)));
                fill_lum = ink.clamp(0.0, 0.2);
            }
        } else if u < 0.85 {
            let mut p = [a[0] + dir[0] * t * 0.5, a[1] + dir[1] * t * 0.5];
            let mut pts = vec![p];
            for _ in 0..rng.int(3, 8) {
                p = [p[0] + rng.normal(0.0, t / 3.0), p[1] + rng.normal(0.0, t / 3.0)];
                pts.push(p);
            }
            if let Some(path) = poly_path(&pts, false) {
                items.push((path, Draw::Stroke(th)));
            }
        } else {
            let nv = [-dir[1], dir[0]];
            if let Some(p) = lines_path(&[([a[0] - nv[0] * s, a[1] - nv[1] * s], [a[0] + nv[0] * s, a[1] + nv[1] * s])]) {
                items.push((p, Draw::Stroke(th)));
            }
        }
        if let Some(mut cv) = coverage(&items, clip) {
            clip_by(&mut cv, &c.occ, c.w);
            blend(&mut c.page, &cv, fill_lum, 1.0);
        }
    }
    if c.rng.chance(0.3) {
        // loose strokes anywhere, around the text
        let mut layer = Gray::new(c.w, c.h, 0.0);
        let n = c.rng.int(1, 6) as usize;
        let scale = c.w.min(c.h) as f64 / 12.0;
        art(&mut layer, clip, c.rng, n, 1.0, scale, Some(&[Art::Curve, Art::Hair, Art::Scribble, Art::Spark, Art::Ring]));
        let mut cv = Cov { r: clip, a: layer.d };
        clip_by(&mut cv, &c.occ, c.w);
        let lum = if c.rng.chance(0.8) { c.rng.uniform(0.0, 0.2) as f32 } else { 1.0 };
        blend(&mut c.page, &cv, lum, 1.0);
    }
}

fn degrade(mut g: Gray, rng: &mut Rng) -> Vec<u8> {
    let (w, h) = (g.w, g.h);
    if rng.chance(0.3) {
        blur(&mut g.d, w, h, rng.uniform(0.3, 1.1), true);
    }
    if rng.chance(0.2) {
        let s = rng.uniform(1.2, 2.0);
        let (sw, sh) = (((w as f64 / s) as usize).max(8), ((h as f64 / s) as usize).max(8));
        let small = resize_area(&g.d, w, h, sw, sh);
        g.d = resize_bilinear(&small, sw, sh, w, h);
    }
    if rng.chance(0.25) {
        let cell = rng.uniform(80.0, 400.0);
        let k = rng.uniform(0.05, 0.3) as f32;
        let n = value_noise(w, h, rng, cell);
        g.d.iter_mut().zip(&n).for_each(|(v, nv)| *v *= 1.0 + k * nv);
    }
    if rng.chance(0.5) {
        let (c, b, gam) = (rng.uniform(0.6, 1.2) as f32, rng.uniform(-0.1, 0.1) as f32, rng.uniform(0.7, 1.4) as f32);
        g.d.iter_mut().for_each(|v| *v = ((*v - 0.5) * c + 0.5 + b).clamp(0.0, 1.0).powf(gam));
    }
    if rng.chance(0.05) {
        g.d.iter_mut().for_each(|v| *v = 1.0 - *v);
    }
    if rng.chance(0.5) {
        let sd = rng.uniform(0.005, 0.05);
        g.d.iter_mut().for_each(|v| *v += rng.normal(0.0, sd) as f32);
    }
    let mut out: Vec<u8> = g.d.iter().map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8).collect();
    if rng.chance(0.4) {
        let q = rng.int(30, 90) as u8;
        if let Some((px, _, _)) = decode_jpeg_gray(&encode_jpeg(&out, w, h, q))
            && px.len() == out.len() {
                out = px;
            }
    }
    out
}

// --- page layouts ------------------------------------------------------------------------------

fn panel_grid(rng: &mut Rng, h: usize, w: usize) -> Vec<IRect> {
    let m = if rng.chance(0.15) { 0.0 } else { rng.uniform(0.02, 0.07) };
    let (mx, my) = ((w as f64 * m) as i32, (h as f64 * m) as i32);
    let g = ((w.min(h) as f64 * rng.uniform(0.008, 0.03)).round() as i32).max(3);
    let (x0, y0, x1, y1) = (mx, my, w as i32 - mx, h as i32 - my);
    let portrait = h >= w;
    let rp: [f64; 5] = if portrait { [0.05, 0.2, 0.4, 0.28, 0.07] } else { [0.25, 0.45, 0.25, 0.05, 0.0] };
    let n_rows = rng.weighted(&rp) + 1;
    let mut hts: Vec<f64> = (0..n_rows).map(|_| rng.uniform(0.6, 1.4)).collect();
    let hs: f64 = hts.iter().sum();
    hts.iter_mut().for_each(|v| *v = *v / hs * (y1 - y0 - g * (n_rows as i32 - 1)) as f64);
    let mut rects = Vec::new();
    let mut y = y0 as f64;
    for &ht in &hts {
        let cp: [f64; 3] = if portrait { [0.3, 0.5, 0.2] } else { [0.35, 0.4, 0.25] };
        let n_cols = rng.weighted(&cp) + 1;
        let mut wds: Vec<f64> = (0..n_cols).map(|_| rng.uniform(0.6, 1.4)).collect();
        let ws: f64 = wds.iter().sum();
        wds.iter_mut().for_each(|v| *v = *v / ws * (x1 - x0 - g * (n_cols as i32 - 1)) as f64);
        let mut x = x0 as f64;
        for &cw in &wds {
            rects.push(IRect { x0: x.round() as i32, y0: y.round() as i32, x1: (x + cw).round() as i32, y1: (y + ht).round() as i32 });
            x += cw + g as f64;
        }
        y += ht + g as f64;
    }
    rects
}

/// Rule a table (forms, price lists). Returns the cell regions and the table's bounding rect.
fn table(page: &mut Gray, rng: &mut Rng, h: usize, w: usize) -> (Vec<[f64; 4]>, Option<IRect>) {
    let (rows, cols) = (rng.int(2, 7) as usize, rng.int(1, 5) as usize);
    let ch = rng.uniform(20.0, 48.0);
    let cws: Vec<f64> = (0..cols).map(|_| rng.uniform(40.0, 160.0)).collect();
    let (tw, th) = (cws.iter().sum::<f64>(), ch * rows as f64);
    if tw > w as f64 - 8.0 || th > h as f64 - 8.0 {
        return (vec![], None);
    }
    let (x0, y0) = (rng.uniform(4.0, w as f64 - 4.0 - tw), rng.uniform(4.0, h as f64 - 4.0 - th));
    let col = rng.uniform(0.0, 0.4) as f32;
    let lw = rng.pick(&[1.0f64, 1.0, 2.0, 3.0]);
    let mut xs = vec![x0];
    for c in &cws {
        xs.push(xs.last().unwrap() + c);
    }
    let ys: Vec<f64> = (0..=rows).map(|r| y0 + ch * r as f64).collect();
    let mut segs = Vec::new();
    for &x in &xs {
        segs.push(([x, y0], [x, y0 + th]));
    }
    for &y in &ys {
        segs.push(([x0, y], [x0 + tw, y]));
    }
    if let Some(p) = lines_path(&segs) {
        let r = page.rect();
        draw(page, &[(p, Draw::Stroke(lw as f32))], r, col);
    }
    let gp = lw + 1.0;
    let mut cells = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if rng.chance(0.8) {
                cells.push([xs[c] + gp, ys[r] + gp, xs[c + 1] - gp, ys[r + 1] - gp]);
            }
        }
    }
    let bx = IRect { x0: (x0 - lw - 1.0) as i32, y0: (y0 - lw - 1.0) as i32, x1: (x0 + tw + lw + 2.0) as i32, y1: (y0 + th + lw + 2.0) as i32 };
    (cells, Some(bx))
}


/// Lighter degradations for screenshots: occasional rescale, mild blur / contrast / noise,
/// usually JPEG.
fn degrade_web(mut g: Gray, rng: &mut Rng) -> Vec<u8> {
    let (w, h) = (g.w, g.h);
    if rng.chance(0.25) {
        let s = rng.uniform(1.1, 1.8);
        let (sw, sh) = (((w as f64 / s) as usize).max(8), ((h as f64 / s) as usize).max(8));
        let small = resize_area(&g.d, w, h, sw, sh);
        g.d = resize_bilinear(&small, sw, sh, w, h);
    }
    if rng.chance(0.08) {
        blur(&mut g.d, w, h, rng.uniform(0.3, 0.7), true);
    }
    if rng.chance(0.15) {
        let (c, b, gam) = (rng.uniform(0.8, 1.1) as f32, rng.uniform(-0.05, 0.05) as f32, rng.uniform(0.85, 1.2) as f32);
        g.d.iter_mut().for_each(|v| *v = ((*v - 0.5) * c + 0.5 + b).clamp(0.0, 1.0).powf(gam));
    }
    if rng.chance(0.15) {
        let sd = rng.uniform(0.003, 0.015);
        g.d.iter_mut().for_each(|v| *v += rng.normal(0.0, sd) as f32);
    }
    let mut out: Vec<u8> = g.d.iter().map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8).collect();
    if rng.chance(0.6) {
        let q = rng.int(50, 96) as u8;
        if let Some((px, _, _)) = decode_jpeg_gray(&encode_jpeg(&out, w, h, q))
            && px.len() == out.len() {
                out = px;
            }
    }
    out
}

/// Compose one page.
pub fn make_page(rng: &mut Rng, sampler: &TextSampler, fonts: &FontBank, cfg: &PageConfig) -> Page {
    let long = rng.int(cfg.long_min as i64, cfg.long_max as i64 + 1) as usize;
    let portrait = rng.chance(cfg.portrait_prob);
    let short = (long as f64 / if portrait { rng.uniform(1.25, 1.55) } else { rng.uniform(1.3, 1.8) }) as usize;
    let (h, w) = if portrait { (long, short) } else { (short, long) };
    let mode = if fonts.faces.is_empty() {
        rng.weighted(&cfg.mode_p[..3]) // the web layout (and the v5 scenes) are all fonts
    } else {
        rng.weighted(&cfg.mode_p)
    };
    let empty = rng.chance(cfg.empty_prob);
    let mut c = Composer {
        rng,
        sampler,
        tg: TextGen::new(sampler),
        fonts,
        cfg,
        w,
        h,
        page: Gray::new(w, h, 1.0),
        occ: vec![0u8; w * h],
        lines: Vec::new(),
        foreign: Vec::new(),
        meta: Vec::new(),
        hq: Vec::new(),
        fq: Vec::new(),
        neg: empty,
        fboost: if mode == 0 { 1.6 } else { 1.0 },
    };
    let mut plan: Vec<(Kind, [f64; 4], Want)> = Vec::new();
    let inset = 4.0;
    let full = [inset, inset, w as f64 - inset, h as f64 - inset];
    let pr = c.page_rect();
    let (fe, ie) = (cfg.foreign_extra, cfg.icon_extra);
    let rep = |plan: &mut Vec<(Kind, [f64; 4], Want)>, k: Kind, r: [f64; 4], want: Want, n: usize| plan.extend(std::iter::repeat_n((k, r, want), n));
    match mode {
        0 => {
            // manga: panels
            let v = c.rng.uniform(0.88, 1.0) as f32;
            c.page.fill(pr, v);
            for r in panel_grid(c.rng, h, w) {
                if r.w() < 24 || r.h() < 24 {
                    continue;
                }
                let kind = PANEL_BG[c.rng.weighted(&PANEL_P)];
                let ink = fill_bg(&mut c.page, r, c.rng, kind);
                let area = (r.w() * r.h()) as f64 / (w * h) as f64;
                if c.rng.chance(0.75) {
                    let n = c.rng.int(1, 3 + (14.0 * area) as i64) as usize;
                    art(&mut c.page, r, c.rng, n, ink, r.w().min(r.h()) as f64 / 6.0, None);
                }
                let bw = if c.rng.chance(0.08) { 0.0 } else { c.rng.pick(&[1.0, 2.0, 2.0, 3.0, 3.0, 4.0]) };
                if bw > 0.0 {
                    let col = c.rng.uniform(0.0, 0.12) as f32;
                    let (x0, y0, x1, y1) = (r.x0 as f64 + bw / 2.0 - 0.5, r.y0 as f64 + bw / 2.0 - 0.5, r.x1 as f64 - bw / 2.0 - 0.5, r.y1 as f64 - bw / 2.0 - 0.5);
                    if let Some(p) = poly_path(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], true) {
                        let st = tiny_skia::Stroke { width: bw as f32, line_join: tiny_skia::LineJoin::Miter, ..Default::default() };
                        if let Some(o) = p.stroke(&st, 1.0) {
                            draw(&mut c.page, &[(o, Draw::Fill)], pr, col);
                        }
                    }
                }
                let reg = [r.x0 as f64 + bw + 2.0, r.y0 as f64 + bw + 2.0, r.x1 as f64 - bw - 2.0, r.y1 as f64 - bw - 2.0];
                let rng = &mut *c.rng;
                rep(&mut plan, Kind::Bubble, reg, Want::Auto, rng.poisson(0.8 + 7.0 * area).min(5));
                rep(&mut plan, Kind::Caption, reg, Want::Auto, rng.chance(0.15) as usize);
                rep(&mut plan, Kind::Free, reg, Want::Auto, rng.poisson(0.35).min(2));
                rep(&mut plan, Kind::Sign, reg, Want::Auto, rng.chance(0.12) as usize);
                rep(&mut plan, Kind::Block, reg, Want::Auto, rng.chance(0.06) as usize);
                rep(&mut plan, Kind::Filler, reg, Want::Auto, rng.poisson(0.8).min(4));
                // translated SFX, small labels, symbols (hearts, notes, stars) in the art
                rep(&mut plan, Kind::Filler, reg, Want::Foreign, rng.poisson(1.5 * fe).min(4));
                rep(&mut plan, Kind::Bubble, reg, Want::Foreign, rng.poisson(1.0 * fe).min(3));
                rep(&mut plan, Kind::Free, reg, Want::Foreign, rng.poisson(0.3 * fe).min(2));
                rep(&mut plan, Kind::Filler, reg, Want::Icons, rng.poisson(0.25 * ie).min(2));
            }
            if c.rng.chance(0.5 * fe) {
                plan.push((Kind::Filler, [inset, h as f64 * 0.93, w as f64 - inset, h as f64 - inset], Want::Foreign)); // page number / credit
            }
        }
        1 => {
            // frame: anime still / poster / video screenshot
            let kind = if c.rng.chance(0.7) { Bg::Scene } else { [Bg::Gradient, Bg::Noise, Bg::Tone, Bg::Flat][c.rng.below(4)] };
            fill_bg(&mut c.page, pr, c.rng, kind);
            let mu = c.page.mean(pr);
            let ink = if mu > 0.5 { c.rng.uniform(0.0, 0.25) } else { c.rng.uniform(0.7, 1.0) } as f32;
            let n = c.rng.int(2, 12) as usize;
            art(&mut c.page, pr, c.rng, n, ink, w.min(h) as f64 / 5.0, None);
            let rng = &mut *c.rng;
            rep(&mut plan, Kind::Free, full, Want::Auto, 1 + rng.poisson(1.2).min(4));
            rep(&mut plan, Kind::Sign, full, Want::Auto, rng.poisson(1.8).min(6));
            rep(&mut plan, Kind::Bubble, full, Want::Auto, rng.poisson(0.8).min(3));
            rep(&mut plan, Kind::Block, full, Want::Auto, rng.chance(0.3) as usize);
            rep(&mut plan, Kind::Filler, full, Want::Auto, rng.poisson(4.0).min(12));
            let strip = [inset, h as f64 * 0.78, w as f64 - inset, h as f64 - inset];
            if rng.chance(0.3) {
                plan.push((Kind::Free, strip, Want::Auto)); // subtitle strip
            }
            // background signage, credits / channel logos / timestamps, subtitles, player icons
            rep(&mut plan, Kind::Sign, full, Want::Foreign, rng.poisson(3.0 * fe).min(8));
            rep(&mut plan, Kind::Block, full, Want::Foreign, rng.chance(0.5 * fe) as usize);
            rep(&mut plan, Kind::Filler, full, Want::Foreign, rng.poisson(3.5 * fe).min(10));
            rep(&mut plan, Kind::Free, strip, Want::Foreign, rng.chance(0.35 * fe) as usize);
            rep(&mut plan, Kind::Filler, full, Want::Icons, rng.poisson(0.6 * ie).min(3));
            rep(&mut plan, Kind::Sign, strip, Want::Icons, rng.chance(0.2 * ie) as usize);
        }
        2 => {
            // doc: paper with paragraphs, labels, tables
            let base = c.rng.uniform(0.78, 1.0) as f32;
            let amp = c.rng.uniform(0.01, 0.05) as f32;
            let cell = c.rng.uniform(10.0, 60.0);
            let n = value_noise(w, h, c.rng, cell);
            c.page.d.iter_mut().zip(&n).for_each(|(v, nv)| *v = (base + amp * nv).clamp(0.0, 1.0));
            if c.rng.chance(0.2) {
                // ruled paper
                let sp = c.rng.uniform(16.0, 40.0);
                let col = c.rng.uniform(0.3, 0.75) as f32;
                let mut segs = Vec::new();
                let mut y = c.rng.uniform(0.0, sp);
                while y < h as f64 {
                    segs.push(([0.0, y], [w as f64, y]));
                    y += sp;
                }
                if let Some(p) = lines_path(&segs) {
                    draw(&mut c.page, &[(p, Draw::Stroke(1.0))], pr, col);
                }
            }
            let n = c.rng.int(0, 6) as usize;
            let ink = c.rng.uniform(0.0, 0.3) as f32;
            art(&mut c.page, pr, c.rng, n, ink, w.min(h) as f64 / 8.0, None);
            if c.rng.chance(0.35) {
                let (cells, bx) = table(&mut c.page, c.rng, h, w);
                if !empty || !fonts.faces.is_empty() {
                    for reg in cells {
                        c.add(Kind::Cell, reg, Want::Auto); // fill the cells now, then keep everything else off the table
                    }
                }
                if let Some(b) = bx {
                    let b = b.intersect(pr);
                    for y in b.y0..b.y1 {
                        c.occ[y as usize * w + b.x0 as usize..y as usize * w + b.x1 as usize].fill(255);
                    }
                }
            }
            let rng = &mut *c.rng;
            let n_cols = rng.weighted(&[0.5, 0.35, 0.15]) + 1;
            let cw = (w as f64 - 2.0 * inset) / n_cols as f64;
            for i in 0..n_cols {
                let reg = [inset + i as f64 * cw, inset, inset + (i + 1) as f64 * cw, h as f64 - inset];
                rep(&mut plan, Kind::Block, reg, Want::Auto, 1 + rng.poisson(0.8).min(3));
                rep(&mut plan, Kind::Block, reg, Want::Foreign, rng.poisson(2.0 * fe).min(4));
            }
            rep(&mut plan, Kind::Sign, full, Want::Auto, rng.poisson(1.5).min(5));
            rep(&mut plan, Kind::Free, full, Want::Auto, rng.poisson(1.0).min(3));
            rep(&mut plan, Kind::Bubble, full, Want::Auto, rng.poisson(0.4).min(2));
            rep(&mut plan, Kind::Filler, full, Want::Auto, rng.poisson(3.0).min(10));
            rep(&mut plan, Kind::Filler, full, Want::Foreign, rng.poisson(3.5 * fe).min(10));
            rep(&mut plan, Kind::Sign, full, Want::Foreign, rng.poisson(0.75 * fe).min(3));
            rep(&mut plan, Kind::Filler, full, Want::Icons, rng.poisson(0.5 * ie).min(3));
            rep(&mut plan, Kind::Sign, full, Want::Icons, rng.chance(0.2 * ie) as usize);
        }
        3 => crate::web::web_page(&mut c),
        4 => crate::scenes::card_page(&mut c),
        5 => crate::scenes::lyric_page(&mut c),
        6 => crate::scenes::tv_page(&mut c),
        _ => crate::scenes::promo_page(&mut c),
    }
    if !empty || !fonts.faces.is_empty() {
        // Hunter-capable elements first (in kind order), then the foreign-only / icon extras, which
        // fill the space left over instead of crowding out labelled lines
        let mut keyed: Vec<(u8, f64, Kind, [f64; 4], Want)> = plan.into_iter().map(|(k, r, want)| (k.order() + if want == Want::Auto { 0 } else { 10 }, c.rng.f64(), k, r, want)).collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for (_, _, k, r, want) in keyed {
            c.add(k, r, want);
        }
    }
    let web = mode == 3;
    if !web && mode != 5 && mode != 7 {
        post_art(&mut c);
    }
    if cfg.v5 && !fonts.faces.is_empty() && c.rng.chance(cfg.ui_neg_prob) {
        crate::scenes::ui_negatives(&mut c);
    }
    let photo = cfg.photo_prob > 0.0 && c.rng.chance(cfg.photo_prob);
    let Composer { rng, page, mut lines, mut foreign, .. } = c;
    if photo {
        let (pixels, pw, ph) = crate::photo::photograph(page, &mut lines, &mut foreign, rng);
        return Page { w: pw, h: ph, pixels, lines, foreign, layout: LAYOUTS[mode], photo: true };
    }
    let pixels = if (web || mode == 5) && rng.chance(0.7) { degrade_web(page, rng) } else { degrade(page, rng) };
    Page { w, h, pixels, lines, foreign, layout: LAYOUTS[mode], photo: false }
}

/// v5 decoration for free-standing text, chosen once the background (`bg`) is known: a glow
/// (light ink on a dark ground), a soft dark outer stroke, a bevel, or hollow glyphs (only the
/// outline contrasts, the strokes take the background's value).
fn v5_effects(rng: &mut Rng, fx: &mut Fx, ink: &mut f32, bg: f32, contrast: f32, em: f64, room: f64) {
    let light = *ink > 0.5;
    let u = rng.f64();
    if u < 0.14 && light {
        let grow = em * rng.uniform(0.04, 0.12);
        let sigma = em * rng.uniform(0.04, 0.1);
        if grow + 2.0 * sigma <= room {
            fx.halo = Some(Halo { grow, sigma, lum: rng.uniform(0.8, 1.0) as f32, strength: rng.uniform(0.4, 0.9) as f32 });
        }
    } else if u < 0.24 {
        let grow = em * rng.uniform(0.03, 0.1);
        let sigma = em * rng.uniform(0.02, 0.07);
        if grow + 2.0 * sigma <= room {
            fx.halo = Some(Halo { grow, sigma, lum: contrast, strength: rng.uniform(0.6, 1.0) as f32 });
        }
    } else if u < 0.29 {
        let d = (em * rng.uniform(0.02, 0.05)).max(1.0);
        if d + 1.0 <= room.max(fx.outline.map_or(0.0, |o| o.0)) {
            fx.bevel = Some(((d).min(room), (*ink + 0.45).min(1.0), (*ink - 0.45).max(0.0)));
        }
    } else if u < 0.37 {
        // hollow: the outline takes the ink's value (it carries the contrast), the strokes are
        // filled near the background or with a fill on the far side from the ink
        let r = (em * rng.uniform(0.04, 0.08)).max(1.2);
        let line = *ink;
        fx.outline = Some((r, line));
        *ink = if rng.chance(0.6) {
            (bg + rng.uniform(-0.08, 0.08) as f32).clamp(0.0, 1.0)
        } else if line < 0.5 {
            rng.uniform(0.7, 1.0) as f32
        } else {
            rng.uniform(0.0, 0.3) as f32
        };
        let _ = contrast;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// System fonts (default dir or $HXH_FONT_DIR); empty on machines without them.
    fn test_fonts() -> &'static FontBank {
        static F: OnceLock<FontBank> = OnceLock::new();
        F.get_or_init(|| FontBank::load(&[std::path::PathBuf::from(FontBank::default_dir())]))
    }

    /// v5 (scenes, photos, effects, inline digits): reproducible, labelled quads disjoint and
    /// inside the image, every foreign run keep_margin away from every labelled line.
    #[test]
    fn v5_pages_keep_the_rules() {
        let fonts = test_fonts();
        let sampler = TextSampler::new(&[]);
        let cfg = PageConfig {
            v5: true, mode_p: [0.2, 0.1, 0.05, 0.05, 0.2, 0.2, 0.2, 0.0], photo_prob: 0.4, ui_neg_prob: 0.6, caption_prob: 0.4, inline_prob: 0.2,
            long_min: 384, long_max: 800, ..Default::default()
        };
        let (mut nl, mut nf, mut photos) = (0, 0, 0);
        for i in 0..80u64 {
            let a = make_page(&mut Rng::from_key(&[5, 1, i]), &sampler, fonts, &cfg);
            let b = make_page(&mut Rng::from_key(&[5, 1, i]), &sampler, fonts, &cfg);
            assert_eq!(a.pixels, b.pixels, "page {i} not reproducible");
            assert_eq!(a.pixels.len(), a.w * a.h);
            photos += a.photo as usize;
            let full = IRect { x0: 0, y0: 0, x1: a.w as i32, y1: a.h as i32 };
            let mut acc = vec![0u8; a.w * a.h];
            for l in &a.lines {
                assert!(l.quad.iter().all(|p| p[0] >= -0.5 && p[1] >= -0.5 && p[0] <= a.w as f64 && p[1] <= a.h as f64), "page {i}: quad off the image");
                let mut m = vec![0u8; a.w * a.h];
                fill_convex_u8(&mut m, full, &l.quad, 1);
                acc.iter_mut().zip(&m).for_each(|(x, y)| *x += y);
            }
            assert!(acc.iter().all(|&v| v <= 1), "overlapping quads on page {i} ({})", a.layout);
            for f in &a.foreign {
                for l in &a.lines {
                    // photos scale quads by a projective map: allow for the shrink
                    let need = cfg.keep_margin * line_thickness(&f.quad).max(line_thickness(&l.quad)) * if a.photo { 0.5 } else { 1.0 } - 0.3;
                    assert!(convex_dist(&f.quad, &l.quad) >= need, "page {i} ({}): {} '{}' too close to '{}'", a.layout, f.script, f.text, l.text);
                }
            }
            nl += a.lines.len();
            nf += a.foreign.len();
        }
        assert!(nl > 100 && photos > 10, "lines {nl} foreign {nf} photos {photos}");
    }

    /// v6 (stroke / display-font digits, reading guides, sticker lettering, promo scene):
    /// reproducible, labelled quads disjoint and inside the image, labels in the extended vocab,
    /// every foreign run but the reading guides keep_margin away from every labelled line, and the
    /// reading guides clear of every labelled quad.
    #[test]
    fn v6_pages_keep_the_rules() {
        let fonts = test_fonts();
        if fonts.faces.is_empty() {
            eprintln!("no fonts found: skipped");
            return;
        }
        let sampler = TextSampler::new(&[]);
        let cfg = PageConfig {
            v5: true, v6: true, mode_p: [0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.4], photo_prob: 0.3, ui_neg_prob: 0.4, caption_prob: 0.3,
            inline_prob: 0.5, ruby_prob: 0.5, sticker_prob: 0.4, small_mark_prob: 0.5, long_min: 384, long_max: 800, ..Default::default()
        };
        let ok: Vec<char> = crate::glyphs::glyph_set().vocab.iter().copied().chain(crate::digits::EXTRA.chars()).collect();
        let (mut nl, mut nd, mut nr, mut promo) = (0, 0, 0, 0);
        for i in 0..80u64 {
            let a = make_page(&mut Rng::from_key(&[6, 1, i]), &sampler, fonts, &cfg);
            let b = make_page(&mut Rng::from_key(&[6, 1, i]), &sampler, fonts, &cfg);
            assert_eq!(a.pixels, b.pixels, "page {i} not reproducible");
            promo += (a.layout == "promo") as usize;
            let full = IRect { x0: 0, y0: 0, x1: a.w as i32, y1: a.h as i32 };
            let mut acc = vec![0u8; a.w * a.h];
            for l in &a.lines {
                assert!(l.quad.iter().all(|p| p[0] >= -0.5 && p[1] >= -0.5 && p[0] <= a.w as f64 && p[1] <= a.h as f64), "page {i}: quad off the image");
                assert!(l.text.chars().all(|c| c == ' ' || ok.contains(&c)), "page {i}: label {} outside the vocab", l.text);
                nd += l.text.chars().any(crate::digits::is_extra) as usize;
                let mut m = vec![0u8; a.w * a.h];
                fill_convex_u8(&mut m, full, &l.quad, 1);
                acc.iter_mut().zip(&m).for_each(|(x, y)| *x += y);
            }
            assert!(acc.iter().all(|&v| v <= 1), "overlapping quads on page {i} ({})", a.layout);
            for f in &a.foreign {
                nr += (f.kind == "ruby") as usize;
                for l in &a.lines {
                    let k = if f.kind == "ruby" { 0.0 } else { cfg.keep_margin };
                    let need = k * line_thickness(&f.quad).max(line_thickness(&l.quad)) * if a.photo { 0.5 } else { 1.0 } - 0.3;
                    assert!(convex_dist(&f.quad, &l.quad) >= need, "page {i} ({}): {} {} '{}' too close to '{}'", a.layout, f.kind, f.script, f.text, l.text);
                }
            }
            nl += a.lines.len();
        }
        eprintln!("v6: {nl} lines, {nd} with digits / punctuation, {nr} reading guides, {promo} promo pages");
        assert!(nl > 100 && nd > 20 && nr > 10 && promo > 10, "lines {nl} digits {nd} ruby {nr} promo {promo}");
    }

    #[test]
    fn pages_are_reproducible_and_lines_disjoint() {
        let sampler = TextSampler::new(&[]);
        let cfg = PageConfig::default();
        let mut total = 0;
        for i in 0..12u64 {
            let a = make_page(&mut Rng::from_key(&[7, 9, i]), &sampler, test_fonts(), &cfg);
            let b = make_page(&mut Rng::from_key(&[7, 9, i]), &sampler, test_fonts(), &cfg);
            assert_eq!(a.pixels, b.pixels);
            assert_eq!(a.foreign.len(), b.foreign.len());
            assert_eq!(a.lines.len(), b.lines.len());
            let mut acc = vec![0u8; a.w * a.h];
            let full = IRect { x0: 0, y0: 0, x1: a.w as i32, y1: a.h as i32 };
            for l in &a.lines {
                let mut m = vec![0u8; a.w * a.h];
                fill_convex_u8(&mut m, full, &l.quad, 1);
                acc.iter_mut().zip(&m).for_each(|(x, y)| *x += y);
                assert!(!l.text.is_empty());
                assert!(l.quad.iter().all(|p| p[0] >= -0.5 && p[1] >= -0.5 && p[0] <= a.w as f64 && p[1] <= a.h as f64));
            }
            assert!(acc.iter().all(|&v| v <= 1), "overlapping quads on page {i}");
            total += a.lines.len();
        }
        assert!(total > 50, "too few lines: {total}");
    }

    /// Every foreign-text / icon run stays keep_margin x (the larger line height) away from every
    /// labelled quad, exactly (quads are rounded to 0.1 px in the output, hence the slack).
    #[test]
    fn foreign_keeps_clear_of_labelled_lines() {
        let fonts = test_fonts();
        if fonts.faces.is_empty() {
            eprintln!("no fonts found: skipped");
            return;
        }
        let sampler = TextSampler::new(&[]);
        // more mixing than the defaults, to stress the rule
        let cfg = PageConfig { long_min: 384, long_max: 800, foreign_prob: 0.35, mixed_prob: 0.3, ..Default::default() };
        let (mut nf, mut nl, mut ni, mut mixed_pages) = (0, 0, 0, 0);
        let mut layouts = [0usize; 8];
        for i in 0..320u64 {
            let p = make_page(&mut Rng::from_key(&[11, 3, i]), &sampler, fonts, &cfg);
            layouts[LAYOUTS.iter().position(|&l| l == p.layout).unwrap()] += 1;
            for f in &p.foreign {
                assert!(!f.text.is_empty());
                assert!(f.quad.iter().all(|q| q[0] >= -0.5 && q[1] >= -0.5 && q[0] <= p.w as f64 && q[1] <= p.h as f64), "page {i}: foreign quad off the page");
                let hf = line_thickness(&f.quad);
                for l in &p.lines {
                    let need = cfg.keep_margin * hf.max(line_thickness(&l.quad)) - 0.2;
                    let d = convex_dist(&f.quad, &l.quad);
                    assert!(d >= need, "page {i} ({}): {} '{}' {:?} is {d:.2} px from '{}' {:?}, needs {need:.2}", p.layout, f.script, f.text, f.quad, l.text, l.quad);
                }
                ni += (f.script == "icon") as usize;
            }
            mixed_pages += (!p.foreign.is_empty() && !p.lines.is_empty()) as usize;
            nf += p.foreign.len();
            nl += p.lines.len();
        }
        eprintln!("{nl} labelled lines, {nf} foreign runs ({ni} icon runs), {mixed_pages} pages with both, layouts {layouts:?}");
        assert!(nl > 2000 && nf > 2000 && ni > 100 && mixed_pages > 150);
        assert!(layouts[..4].iter().all(|&n| n > 10));
    }
}
