//! System fonts for the foreign-text distractors (see `FOREIGN.md`), read at run time from one
//! or more font directories (default `C:\Windows\Fonts`). Nothing is copied or embedded: the
//! binary only knows file names. `ab_glyph` (pure Rust, `ttf-parser` underneath; handles .ttf,
//! CFF .otf and .ttc collections) gives metrics and glyph outlines; the outlines are flattened to
//! polygons in pixels and rasterised with tiny-skia exactly like the Hunter strokes, so rotation,
//! perspective, fake bold (outline dilation), outlines and shadows all take the same path.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ab_glyph::{Font, FontRef, GlyphId, OutlineCurve};

use crate::geom::P;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Sans,
    Serif,
    Mono,
    Display,
    Hand,
    Icon, // icon / dingbat font: glyphs used by id (symbol cmaps are not Unicode)
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Sans => "sans",
            Class::Serif => "serif",
            Class::Mono => "mono",
            Class::Display => "display",
            Class::Hand => "hand",
            Class::Icon => "icon",
        }
    }
}

/// Script coverage bits.
pub const LATIN: u8 = 1;
pub const LATIN_EXT: u8 = 2; // Hepburn macrons, en dash
pub const JAPANESE: u8 = 4;
pub const KOREAN: u8 = 8;
pub const GREEK: u8 = 16;
pub const CYRILLIC: u8 = 32;

const PROBES: [(u8, &str); 6] = [
    (LATIN, "AZaz09.,:;!?()-/\"'"),
    (LATIN_EXT, "āīūēōĀŌ–"),
    (JAPANESE, "あいうアイウー日本語文字、。「」"),
    (KOREAN, "한국어글자"),
    (GREEK, "ΑΒΓαβγω"),
    (CYRILLIC, "АБВабвя"),
];

/// (file name, collection index, class, bold, italic, CJK design). Only files present are used.
const KNOWN: &[(&str, u32, Class, bool, bool, bool)] = &[
    // Latin sans
    ("arial.ttf", 0, Class::Sans, false, false, false),
    ("arialbd.ttf", 0, Class::Sans, true, false, false),
    ("ariali.ttf", 0, Class::Sans, false, true, false),
    ("arialn.ttf", 0, Class::Sans, false, false, false),
    ("arialnb.ttf", 0, Class::Sans, true, false, false),
    ("verdana.ttf", 0, Class::Sans, false, false, false),
    ("verdanab.ttf", 0, Class::Sans, true, false, false),
    ("tahoma.ttf", 0, Class::Sans, false, false, false),
    ("tahomabd.ttf", 0, Class::Sans, true, false, false),
    ("segoeui.ttf", 0, Class::Sans, false, false, false),
    ("segoeuib.ttf", 0, Class::Sans, true, false, false),
    ("segoeuil.ttf", 0, Class::Sans, false, false, false),
    ("segoeuii.ttf", 0, Class::Sans, false, true, false),
    ("calibri.ttf", 0, Class::Sans, false, false, false),
    ("calibrib.ttf", 0, Class::Sans, true, false, false),
    ("calibrii.ttf", 0, Class::Sans, false, true, false),
    ("trebuc.ttf", 0, Class::Sans, false, false, false),
    ("trebucbd.ttf", 0, Class::Sans, true, false, false),
    ("candara.ttf", 0, Class::Sans, false, false, false),
    ("corbel.ttf", 0, Class::Sans, false, false, false),
    ("bahnschrift.ttf", 0, Class::Sans, false, false, false),
    ("framd.ttf", 0, Class::Sans, false, false, false),
    ("micross.ttf", 0, Class::Sans, false, false, false),
    ("ebrima.ttf", 0, Class::Sans, false, false, false),
    ("gadugi.ttf", 0, Class::Sans, false, false, false),
    // Latin serif
    ("times.ttf", 0, Class::Serif, false, false, false),
    ("timesbd.ttf", 0, Class::Serif, true, false, false),
    ("timesi.ttf", 0, Class::Serif, false, true, false),
    ("georgia.ttf", 0, Class::Serif, false, false, false),
    ("georgiab.ttf", 0, Class::Serif, true, false, false),
    ("georgiai.ttf", 0, Class::Serif, false, true, false),
    ("cambria.ttc", 0, Class::Serif, false, false, false),
    ("cambriab.ttf", 0, Class::Serif, true, false, false),
    ("pala.ttf", 0, Class::Serif, false, false, false),
    ("palab.ttf", 0, Class::Serif, true, false, false),
    ("palai.ttf", 0, Class::Serif, false, true, false),
    ("gara.ttf", 0, Class::Serif, false, false, false),
    ("garabd.ttf", 0, Class::Serif, true, false, false),
    ("bookos.ttf", 0, Class::Serif, false, false, false),
    ("century.ttf", 0, Class::Serif, false, false, false),
    ("constan.ttf", 0, Class::Serif, false, false, false),
    ("sylfaen.ttf", 0, Class::Serif, false, false, false),
    // monospaced
    ("consola.ttf", 0, Class::Mono, false, false, false),
    ("consolab.ttf", 0, Class::Mono, true, false, false),
    ("cour.ttf", 0, Class::Mono, false, false, false),
    ("courbd.ttf", 0, Class::Mono, true, false, false),
    ("lucon.ttf", 0, Class::Mono, false, false, false),
    // display / handwriting / comic lettering
    ("impact.ttf", 0, Class::Display, true, false, false),
    ("framdcn.ttf", 0, Class::Display, false, false, false),
    ("comic.ttf", 0, Class::Hand, false, false, false),
    ("comicbd.ttf", 0, Class::Hand, true, false, false),
    ("inkfree.ttf", 0, Class::Hand, false, false, false),
    ("segoepr.ttf", 0, Class::Hand, false, false, false),
    ("segoeprb.ttf", 0, Class::Hand, true, false, false),
    ("segoesc.ttf", 0, Class::Hand, false, false, false),
    ("mvboli.ttf", 0, Class::Hand, false, false, false),
    ("gabriola.ttf", 0, Class::Hand, false, false, false),
    // Japanese
    ("msgothic.ttc", 0, Class::Sans, false, false, true),
    ("msgothic.ttc", 1, Class::Sans, false, false, true),
    ("msgothic.ttc", 2, Class::Sans, false, false, true),
    ("yugothr.ttc", 0, Class::Sans, false, false, true),
    ("yugothm.ttc", 0, Class::Sans, false, false, true),
    ("yugothb.ttc", 0, Class::Sans, true, false, true),
    ("yugothl.ttc", 0, Class::Sans, false, false, true),
    ("notosansjp-regular.otf", 0, Class::Sans, false, false, true),
    ("notosansjp-bold.otf", 0, Class::Sans, true, false, true),
    ("meiryo.ttc", 0, Class::Sans, false, false, true),
    ("meiryob.ttc", 0, Class::Sans, true, false, true),
    ("msmincho.ttc", 0, Class::Serif, false, false, true),
    ("msmincho.ttc", 1, Class::Serif, false, false, true),
    ("yumin.ttf", 0, Class::Serif, false, false, true),
    ("yumindb.ttf", 0, Class::Serif, true, false, true),
    ("yuminl.ttf", 0, Class::Serif, false, false, true),
    ("biz-udgothicr.ttc", 0, Class::Sans, false, false, true),
    ("biz-udgothicb.ttc", 0, Class::Sans, true, false, true),
    ("biz-udminchom.ttc", 0, Class::Serif, false, false, true),
    // Chinese fonts with kana: Ming / Song and Hei shapes when no Japanese mincho is installed
    ("simsun.ttc", 0, Class::Serif, false, false, true),
    ("msyh.ttc", 0, Class::Sans, false, false, true),
    ("msyhbd.ttc", 0, Class::Sans, true, false, true),
    // icons, dingbats, emoji (outlines of the colour glyphs' base layer)
    ("segmdl2.ttf", 0, Class::Icon, false, false, false),
    ("segoeicons.ttf", 0, Class::Icon, false, false, false),
    ("seguiemj.ttf", 0, Class::Icon, false, false, false),
    ("webdings.ttf", 0, Class::Icon, false, false, false),
    ("wingding.ttf", 0, Class::Icon, false, false, false),
    ("wingdng2.ttf", 0, Class::Icon, false, false, false),
    ("wingdng3.ttf", 0, Class::Icon, false, false, false),
    ("marlett.ttf", 0, Class::Icon, false, false, false),
    ("symbol.ttf", 0, Class::Icon, false, false, false),
    ("holomdl2.ttf", 0, Class::Icon, false, false, false),
    ("seguisym.ttf", 0, Class::Sans, false, false, false), // Segoe UI Symbol: text face with a large symbol repertoire
    // ---- openly licensed fonts (fonts/open, scripts/fetch_fonts.py; OFL-1.1 or Apache-2.0), the
    // only faces used for v5 data so the rendered datasets can be published under CC BY 4.0
    ("notosans-regular.ttf", 0, Class::Sans, false, false, false),
    ("inter.ttf", 0, Class::Sans, false, false, false),
    ("intertight-light.ttf", 0, Class::Sans, false, false, false),
    ("roboto-light.ttf", 0, Class::Sans, false, false, false),
    ("roboto-regular.ttf", 0, Class::Sans, false, false, false),
    ("roboto-italic.ttf", 0, Class::Sans, false, true, false),
    ("roboto-medium.ttf", 0, Class::Sans, false, false, false),
    ("roboto-bold.ttf", 0, Class::Sans, true, false, false),
    ("roboto-bolditalic.ttf", 0, Class::Sans, true, true, false),
    ("roboto-black.ttf", 0, Class::Display, true, false, false),
    ("roboto-boldcondensed.ttf", 0, Class::Display, true, false, false),
    ("lato-light.ttf", 0, Class::Sans, false, false, false),
    ("lato-regular.ttf", 0, Class::Sans, false, false, false),
    ("lato-italic.ttf", 0, Class::Sans, false, true, false),
    ("lato-bold.ttf", 0, Class::Sans, true, false, false),
    ("lato-black.ttf", 0, Class::Display, true, false, false),
    ("opensans-light.ttf", 0, Class::Sans, false, false, false),
    ("opensans-regular.ttf", 0, Class::Sans, false, false, false),
    ("opensans-semibold.ttf", 0, Class::Sans, false, false, false),
    ("opensans-bold.ttf", 0, Class::Sans, true, false, false),
    ("robotoslab-regular.ttf", 0, Class::Serif, false, false, false),
    ("robotoslab-bold.ttf", 0, Class::Serif, true, false, false),
    ("robotomono-regular.ttf", 0, Class::Mono, false, false, false),
    ("gonotokurrent-regular.ttf", 0, Class::Sans, false, false, false),
    ("gonotokurrent-bold.ttf", 0, Class::Sans, true, false, false),
    ("notosanskr-regular.otf", 0, Class::Sans, false, false, true),
    ("notosanstc-light.otf", 0, Class::Sans, false, false, true),
    ("notosanstc-regular.otf", 0, Class::Sans, false, false, true),
    ("notosanstc-bold.otf", 0, Class::Sans, true, false, true),
    ("notosanstc-black.otf", 0, Class::Display, true, false, true),
    ("notosanscjksc-regular.otf", 0, Class::Sans, false, false, true),
    ("geneigothicpro-light.otf", 0, Class::Sans, false, false, true),
    ("geneigothicpro-regular.otf", 0, Class::Sans, false, false, true),
    ("geneigothicpro-semibold.otf", 0, Class::Sans, false, false, true),
    ("geneigothicpro-bold.otf", 0, Class::Sans, true, false, true),
    ("geneigothicpro-heavy.otf", 0, Class::Display, true, false, true),
    ("droidsansfallback.ttf", 0, Class::Sans, false, false, true),
    ("fontawesome-webfont.ttf", 0, Class::Icon, false, false, false),
    ("notoemoji-regular.ttf", 0, Class::Icon, false, false, false),
    ("notosanssymbols2-regular.ttf", 0, Class::Icon, false, false, false),
    // Korean
    ("malgun.ttf", 0, Class::Sans, false, false, true),
    ("malgunbd.ttf", 0, Class::Sans, true, false, true),
    ("gulim.ttc", 0, Class::Sans, false, false, true),
    ("batang.ttc", 0, Class::Serif, false, false, true),
];

pub struct Face {
    pub name: String,
    pub class: Class,
    pub bold: bool,
    pub italic: bool,
    pub cjk: bool,
    pub scripts: u8,
    /// Icon glyphs (glyph id, Unicode char when the cmap has one): every outlined glyph of an
    /// icon font, or the `SYMBOLS` a text face covers.
    pub icons: Vec<(GlyphId, Option<char>)>,
    font: FontRef<'static>,
    upem: f64,
}

/// Unicode symbols looked up in ordinary fonts: arrows, stars, checks, shapes, music, card suits,
/// circled numbers, legal / math signs, box drawing, misc pictographs, currency.
pub const SYMBOLS: &str = concat!(
    "←↑→↓↔↕↖↗↘↙⇐⇑⇒⇓⇔➔➜➤➡⬅⬆⬇★☆✓✔✗✘✕✖✚✦✧✩✪✰●○◎◉■□▪▫▲△▼▽◆◇◊►◄▶◀▸◂▴▾⬛⬜⬤◯",
    "♠♣♥♦♡♢♤♧♩♪♫♬①②③④⑤⑥⑦⑧⑨⑩❶❷❸©®™℠±×÷≠≈≤≥∞∑∏√∫∂∆∇∀∃∈∉∩∪⊂⊃⊕⊗⊥∠°′″‰§¶†‡•‣◦※",
    "─│┌┐└┘├┤┬┴┼═║╔╗╚╝▀▄█▌▐░▒▓☀☁☂☃☎☏☐☑☒☺☻☹♀♂⚑⚐⚙⚠⚡⌂⌘⌚⌛⏏⏩⏪⏯⏸⏹⏺✂✈✉✎✏✒❄❤➕➖➗☰☷⊞⊟⊠€£¥¢₩₽"
);

/// One shaped line in its own frame (px): baseline at y = 0 starting at x = 0 (horizontal), or
/// column axis at x = 0 starting at y = 0 (vertical).
pub struct Shaped {
    pub contours: Vec<Vec<P>>,
    pub b: [f64; 4], // tight ink box
    pub adv: f64,    // pen advance (along the line)
}

/// Vertical text: glyphs turned 90 degrees (long-vowel bar, brackets, dashes).
const V_ROT: &str = "ー－—―‐-~〜～…‥（）()「」『』【】〈〉《》［］[]＝=";
/// Vertical text: punctuation moved to the top-right of its cell.
const V_PUNCT: &str = "、。，．,.";
const SMALL_KANA: &str = "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ";

impl Face {
    pub fn has(&self, c: char) -> bool {
        c == ' ' || self.font.glyph_id(c).0 != 0
    }

    pub fn covers(&self, s: &str) -> bool {
        s.chars().all(|c| self.has(c))
    }

    /// Drop characters the face has no glyph for (and collapse the spaces left behind).
    pub fn filter(&self, s: &str) -> String {
        let t: String = s.chars().filter(|&c| self.has(c)).collect();
        t.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
    }

    fn gid(&self, c: char) -> GlyphId {
        self.font.glyph_id(if c == '\u{3000}' { ' ' } else { c })
    }

    /// Horizontal advance of `s` at `px` pixels per em with `track` em of extra letter spacing.
    pub fn measure(&self, s: &str, px: f64, track: f64) -> f64 {
        let k = px / self.upem;
        let mut w = 0.0;
        let mut prev: Option<GlyphId> = None;
        let mut n = 0;
        for c in s.chars() {
            let id = self.gid(c);
            if let Some(p) = prev {
                w += self.font.kern_unscaled(p, id) as f64 * k;
            }
            w += self.font.h_advance_unscaled(id) as f64 * k + track * px;
            prev = Some(id);
            n += 1;
        }
        if n > 0 { w - track * px } else { 0.0 }
    }

    /// Greedy wrap to `max_w` px: at spaces, or anywhere for CJK text (`by_char`).
    pub fn wrap(&self, s: &str, px: f64, max_w: f64, by_char: bool) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        let units: Vec<String> = if by_char { s.chars().map(String::from).collect() } else { s.split(' ').filter(|w| !w.is_empty()).map(String::from).collect() };
        for u in units {
            let cand = if cur.is_empty() || by_char { format!("{cur}{u}") } else { format!("{cur} {u}") };
            if self.measure(&cand, px, 0.0) <= max_w || cur.is_empty() {
                if cur.is_empty() && self.measure(&cand, px, 0.0) > max_w {
                    // a single word wider than the column: hard break
                    let mut piece = String::new();
                    for ch in u.chars() {
                        let t = format!("{piece}{ch}");
                        if self.measure(&t, px, 0.0) > max_w && !piece.is_empty() {
                            out.push(std::mem::take(&mut piece));
                            piece.push(ch);
                        } else {
                            piece = t;
                        }
                    }
                    cur = piece;
                } else {
                    cur = cand;
                }
            } else {
                out.push(std::mem::replace(&mut cur, u));
            }
        }
        if !cur.trim().is_empty() {
            out.push(cur);
        }
        out.into_iter().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
    }

    /// Longest prefix of `s` (cut at a space unless `by_char`) whose advance fits `max_w`.
    pub fn fit(&self, s: &str, px: f64, max_w: f64, track: f64, by_char: bool) -> String {
        if self.measure(s, px, track) <= max_w {
            return s.to_string();
        }
        let chars: Vec<char> = s.chars().collect();
        let mut best = String::new();
        let mut cur = String::new();
        for (i, &c) in chars.iter().enumerate() {
            cur.push(c);
            if self.measure(&cur, px, track) > max_w {
                break;
            }
            if by_char || chars.get(i + 1).is_none_or(|&n| n == ' ') {
                best = cur.clone();
            }
        }
        best.trim().to_string()
    }

    /// Glyph outlines of `s`, flattened to polygons in the line frame (see `Shaped`).
    /// `track`: extra spacing (em); `shear`: synthetic italic (x += shear * height above baseline).
    pub fn shape(&self, s: &str, px: f64, vertical: bool, track: f64, shear: f64) -> Shaped {
        let seq: Vec<(GlyphId, char)> = s.chars().map(|c| (self.gid(c), c)).collect();
        self.shape_seq(&seq, px, vertical, track, shear)
    }

    /// Like `shape`, for glyph ids (icons); horizontal.
    pub fn shape_ids(&self, ids: &[GlyphId], px: f64, track: f64) -> Shaped {
        let seq: Vec<(GlyphId, char)> = ids.iter().map(|&g| (g, '\u{FFFC}')).collect();
        self.shape_seq(&seq, px, false, track, 0.0)
    }

    /// Outline extent of a glyph in em (width, height); None for empty glyphs.
    pub fn glyph_extent(&self, id: GlyphId) -> Option<(f64, f64)> {
        let o = self.font.outline(id)?;
        let (w, h) = (((o.bounds.max.x - o.bounds.min.x) as f64).abs(), ((o.bounds.max.y - o.bounds.min.y) as f64).abs());
        if o.curves.len() < 2 || w <= 0.0 || h <= 0.0 { None } else { Some((w / self.upem, h / self.upem)) }
    }

    fn icon_glyphs(&self) -> Vec<(GlyphId, Option<char>)> {
        let ok = |id: GlyphId| self.glyph_extent(id).is_some_and(|(w, h)| w.max(h) >= 0.25 && w.min(h) >= 0.04);
        if self.class != Class::Icon {
            return SYMBOLS.chars().map(|c| (self.font.glyph_id(c), c)).filter(|&(g, _)| g.0 != 0 && ok(g)).map(|(g, c)| (g, Some(c))).collect();
        }
        let mut uni: BTreeMap<u16, char> = BTreeMap::new();
        for (g, c) in self.font.codepoint_ids() {
            uni.entry(g.0).or_insert(c);
        }
        let n = self.font.glyph_count().min(u16::MAX as usize) as u16;
        if n > 4000 {
            // emoji: only the base glyphs the cmap maps (the rest are colour-layer pieces)
            return uni.iter().map(|(&g, &c)| (GlyphId(g), Some(c))).filter(|&(g, _)| ok(g)).collect();
        }
        (1..n).map(GlyphId).filter(|&g| ok(g)).map(|g| (g, uni.get(&g.0).copied())).collect()
    }

    fn shape_seq(&self, seq: &[(GlyphId, char)], px: f64, vertical: bool, track: f64, shear: f64) -> Shaped {
        let k = px / self.upem;
        let tol = 0.15 / k; // flattening tolerance, font units (0.15 px)
        let mut contours: Vec<Vec<P>> = Vec::new();
        let mut pen = 0.0f64;
        let mut prev: Option<GlyphId> = None;
        let cy = 0.38 * self.upem; // centre of the ideographic em box above the baseline
        for &(id, c) in seq {
            let adv = self.font.h_advance_unscaled(id) as f64;
            if !vertical {
                if let Some(p) = prev {
                    pen += self.font.kern_unscaled(p, id) as f64 * k;
                }
                let x0 = pen;
                self.push_outline(id, tol, &mut contours, |gx, gy| [x0 + gx * k + shear * gy * k, -gy * k]);
                pen += adv * k + track * px;
            } else {
                let yc = pen + 0.5 * px;
                let (mut ox, mut oy) = (0.0, 0.0);
                if V_PUNCT.contains(c) {
                    (ox, oy) = (0.6 * px, -0.6 * px);
                } else if SMALL_KANA.contains(c) {
                    (ox, oy) = (0.1 * px, -0.1 * px);
                }
                let rotate = V_ROT.contains(c);
                self.push_outline(id, tol, &mut contours, |gx, gy| {
                    let (dx, dy) = ((gx - adv / 2.0) * k, (gy - cy) * k);
                    if rotate { [dy + ox, yc + dx + oy] } else { [dx + ox, yc - dy + oy] }
                });
                pen += px * (1.0 + track);
            }
            prev = Some(id);
        }
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in contours.iter().flatten() {
            b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        }
        let adv = if seq.is_empty() { 0.0 } else { pen - track * px };
        Shaped { contours, b, adv }
    }

    fn push_outline(&self, id: GlyphId, tol: f64, out: &mut Vec<Vec<P>>, f: impl Fn(f64, f64) -> P) {
        let Some(o) = self.font.outline(id) else { return };
        let mut cur: Vec<P> = Vec::new();
        let mut last: Option<(f32, f32)> = None;
        let segs = |dd: f64, k: f64| ((k * dd / tol).sqrt().ceil() as usize).clamp(1, 16);
        for c in &o.curves {
            let (p0, end) = match c {
                OutlineCurve::Line(a, b) => (*a, *b),
                OutlineCurve::Quad(a, _, b) => (*a, *b),
                OutlineCurve::Cubic(a, _, _, b) => (*a, *b),
            };
            if last.is_none_or(|(x, y)| (x - p0.x).abs() > 1e-3 || (y - p0.y).abs() > 1e-3) {
                if cur.len() >= 3 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
                cur.push(f(p0.x as f64, p0.y as f64));
            }
            match c {
                OutlineCurve::Line(_, b) => cur.push(f(b.x as f64, b.y as f64)),
                OutlineCurve::Quad(a, b, e) => {
                    let (a, b, e) = ([a.x as f64, a.y as f64], [b.x as f64, b.y as f64], [e.x as f64, e.y as f64]);
                    let dd = (a[0] - 2.0 * b[0] + e[0]).hypot(a[1] - 2.0 * b[1] + e[1]);
                    let n = segs(dd, 0.125);
                    for i in 1..=n {
                        let t = i as f64 / n as f64;
                        let (u, v, w) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
                        cur.push(f(u * a[0] + v * b[0] + w * e[0], u * a[1] + v * b[1] + w * e[1]));
                    }
                }
                OutlineCurve::Cubic(a, b, cc, e) => {
                    let (a, b, cc, e) = ([a.x as f64, a.y as f64], [b.x as f64, b.y as f64], [cc.x as f64, cc.y as f64], [e.x as f64, e.y as f64]);
                    let dd = (a[0] - 2.0 * b[0] + cc[0]).hypot(a[1] - 2.0 * b[1] + cc[1]).max((b[0] - 2.0 * cc[0] + e[0]).hypot(b[1] - 2.0 * cc[1] + e[1]));
                    let n = segs(dd, 0.75);
                    for i in 1..=n {
                        let t = i as f64 / n as f64;
                        let s = 1.0 - t;
                        let (k0, k1, k2, k3) = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
                        cur.push(f(k0 * a[0] + k1 * b[0] + k2 * cc[0] + k3 * e[0], k0 * a[1] + k1 * b[1] + k2 * cc[1] + k3 * e[1]));
                    }
                }
            }
            last = Some((end.x, end.y));
        }
        if cur.len() >= 3 {
            out.push(cur);
        }
    }
}

pub struct FontBank {
    pub faces: Vec<Face>,
    pub dirs: Vec<String>,
}

fn guess(name: &str) -> (Class, bool, bool) {
    let n = name.to_lowercase();
    let bold = n.contains("bold") || n.ends_with("bd.ttf") || n.ends_with("b.ttf");
    let italic = n.contains("italic") || n.ends_with("i.ttf");
    let class = if n.contains("mono") || n.contains("cour") || n.contains("consol") {
        Class::Mono
    } else if n.contains("serif") && !n.contains("sans") || n.contains("mincho") || n.contains("times") {
        Class::Serif
    } else {
        Class::Sans
    };
    (class, bold, italic)
}

impl FontBank {
    pub fn empty() -> Self {
        FontBank { faces: Vec::new(), dirs: Vec::new() }
    }

    /// Default font directory for this platform.
    pub fn default_dir() -> String {
        std::env::var("HXH_FONT_DIR").unwrap_or_else(|_| if cfg!(windows) { r"C:\Windows\Fonts".to_string() } else { "/usr/share/fonts".to_string() })
    }

    /// Load the known faces present in `dirs` (`;`-separated in the CLI). A directory with none of
    /// the known files contributes every .ttf / .otf / .ttc in it (first face, at most 64 files).
    /// File data is leaked on purpose: the bank lives for the whole run.
    pub fn load(dirs: &[PathBuf]) -> Self {
        let mut faces = Vec::new();
        let mut data: BTreeMap<PathBuf, &'static [u8]> = BTreeMap::new();
        let mut read = |p: &Path| -> Option<&'static [u8]> {
            if let Some(d) = data.get(p) {
                return Some(d);
            }
            let bytes = fs::read(p).ok()?;
            let d: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            data.insert(p.to_path_buf(), d);
            Some(d)
        };
        for dir in dirs {
            let Ok(rd) = fs::read_dir(dir) else { continue };
            let mut files: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                files.insert(name.to_lowercase(), (name, e.path()));
            }
            let mut wanted: Vec<(String, PathBuf, u32, Class, bool, bool, bool)> = KNOWN
                .iter()
                .filter_map(|&(f, idx, class, bold, italic, cjk)| files.get(f).map(|(n, p)| (n.clone(), p.clone(), idx, class, bold, italic, cjk)))
                .collect();
            if wanted.is_empty() {
                wanted = files
                    .iter()
                    .filter(|(k, _)| k.ends_with(".ttf") || k.ends_with(".otf") || k.ends_with(".ttc"))
                    .take(64)
                    .map(|(_, (n, p))| {
                        let (class, bold, italic) = guess(n);
                        (n.clone(), p.clone(), 0, class, bold, italic, false)
                    })
                    .collect();
            }
            for (name, path, idx, class, bold, italic, cjk) in wanted {
                let Some(d) = read(&path) else { continue };
                let Ok(font) = FontRef::try_from_slice_and_index(d, idx) else { continue };
                let Some(upem) = font.units_per_em() else { continue };
                let mut scripts = 0u8;
                for (bit, probe) in PROBES {
                    if probe.chars().all(|c| font.glyph_id(c).0 != 0) {
                        scripts |= bit;
                    }
                }
                let cjk = cjk || scripts & (JAPANESE | KOREAN) != 0;
                let name = if idx > 0 { format!("{name}#{idx}") } else { name };
                let mut face = Face { name, class, bold, italic, cjk, scripts, icons: Vec::new(), font, upem: upem as f64 };
                face.icons = face.icon_glyphs();
                if class == Class::Icon {
                    face.scripts = 0; // symbol-encoded: never used for text
                    if face.icons.len() < 16 {
                        continue;
                    }
                } else if scripts == 0 {
                    continue;
                }
                faces.push(face);
            }
        }
        FontBank { faces, dirs: dirs.iter().map(|d| d.display().to_string()).collect() }
    }

    /// Faces that cover all of `need`.
    pub fn with(&self, need: u8) -> Vec<usize> {
        (0..self.faces.len()).filter(|&i| self.faces[i].scripts & need == need).collect()
    }

    pub fn has_script(&self, need: u8) -> bool {
        self.faces.iter().any(|f| f.scripts & need == need)
    }

    /// Faces with icon glyphs: (icon fonts, text faces with enough `SYMBOLS`).
    pub fn icon_faces(&self) -> (Vec<usize>, Vec<usize>) {
        let icon = (0..self.faces.len()).filter(|&i| self.faces[i].class == Class::Icon).collect();
        let sym = (0..self.faces.len()).filter(|&i| self.faces[i].class != Class::Icon && self.faces[i].icons.len() >= 24).collect();
        (icon, sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn list_fonts() {
        let t = std::time::Instant::now();
        let b = FontBank::load(&[PathBuf::from(FontBank::default_dir())]);
        eprintln!("loaded {} faces in {:.2}s", b.faces.len(), t.elapsed().as_secs_f64());
        for f in &b.faces {
            eprintln!("{:24} {:8} bold={} it={} cjk={} scripts={:06b} icons={}", f.name, f.class.name(), f.bold, f.italic, f.cjk, f.scripts, f.icons.len());
        }
    }
}

