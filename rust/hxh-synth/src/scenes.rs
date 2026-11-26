//! v5 scenes for photo-style pages:
//!
//! * `card_page`: "Hunter Vocabulary" name cards. Saturated colour panels (in grayscale: mid
//!   greys from purple, pink, blue, yellow ...) with huge outlined Latin letters ("HUNTER")
//!   behind, a bold Hunter name, its katakana right under it (a hard negative), the romaji under that, a small header, character art on one side.
//! * `lyric_page`: a lyric video (TikTok style): white glyphs on near-black with letters spaced
//!   up to about three glyph widths apart, a kanji / kana caption right under each line, app UI
//!   around (round icon buttons with counts, an avatar with a badge, user name, hashtags, tabs,
//!   progress bar).
//! * `tv_page`: an anime frame with screens in the scene: outlined white glyphs on a blue-ish
//!   panel, a bubble-font title with hollow glyphs, a news ticker bar, white glyphs on a dark
//!   shirt, times and numbers ("3/11 15:00", "225") next to the Hunter text.
//! * `ui_negatives`: avatars, round icon buttons and toolbar / browser-chrome strips placed where
//!   the page is still free (round icons were read as ほ, which is a circled cross).
//! * `caption_near`: a Japanese or Latin caption right under / above a Hunter line.
//! * `promo_page` (v6): a merchandise / product promo picture: sticker lettering (dark glyphs in
//!   a thick light outline with its soft shadow, or the inverse) on a light or dark textured
//!   ground, numbers and `!!` written into the Hunter lines, a kana reading guide above them,
//!   the product on one side, a logo and fine print.
//! * `title_page` (v7): an anime episode title card: a big bold katakana title with crosses
//!   between its words ("A×ト×B"), the same title in Hunter lettering with its crosses under (or
//!   over) it, both centred, on a plain, dark or blurred ground.
//!
//! Everything unlabelled is recorded in `Page::foreign` (icons and avatars as script "icon"),
//! and every labelled line keeps the keep-clear rule of `page.rs`.

use std::f64::consts::TAU;

use tiny_skia::PathBuilder;

use crate::fonts::Class;
use crate::foreign::{kata, romaji, Role, Script};
use crate::geom::*;
use crate::page::{compose, inline_line, make_cross_line_v7, make_font_line, make_icon_line, make_line_cells, make_line_text, pick_face, ruby_line, sample_fstyle, art_pub, fill_bg_pub, Composer, FStyle, Fx, Halo, Ink, Kind, LineG, Shadow};
use crate::raster::*;
use crate::render::{sample_style_v5, TextStyle};
use crate::text::label_of;

/// Grayscale values of saturated panel colours (luminance of purple, pink, magenta, blue, teal,
/// orange, yellow, lime, cyan, red ...).
const PANEL_LUM: [f32; 12] = [0.42, 0.52, 0.47, 0.36, 0.55, 0.66, 0.86, 0.78, 0.74, 0.35, 0.6, 0.7];

fn quad_of(b: [f64; 4]) -> [P; 4] {
    [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

fn irect(b: [f64; 4]) -> IRect {
    IRect { x0: b[0].floor() as i32, y0: b[1].floor() as i32, x1: b[2].ceil() as i32, y1: b[3].ceil() as i32 }
}

/// Font outlines through `m`, filled (and/or stroked), blended into the page inside `clip`.
fn draw_contours(c: &mut Composer, contours: &[Vec<P>], m: &H3, clip: IRect, lum: f32, how: Draw, k: f32) {
    let mut pb = PathBuilder::new();
    for ct in contours {
        let q = apply(m, ct[0]);
        pb.move_to(q[0] as f32, q[1] as f32);
        for p in &ct[1..] {
            let q = apply(m, *p);
            pb.line_to(q[0] as f32, q[1] as f32);
        }
        pb.close();
    }
    if let Some(path) = pb.finish()
        && let Some(cv) = coverage(&[(path, how)], clip.intersect(c.page_rect())) {
            blend(&mut c.page, &cv, lum, k);
        }
}

fn disk(c: &mut Composer, cx: f64, cy: f64, r: f64, lum: f32, how: Draw) {
    if let Some(p) = PathBuilder::from_circle(cx as f32, cy as f32, r as f32)
        && let Some(cv) = coverage(&[(p, how)], c.page_rect()) {
            blend(&mut c.page, &cv, lum, 1.0);
        }
}

fn hxh_style(c: &mut Composer, wide: bool) -> TextStyle {
    let mut st = if c.cfg.v5 { sample_style_v5(c.rng, wide) } else { crate::render::sample_style(c.rng) };
    crate::page::small_marks(c.rng, c.cfg, &mut st);
    st
}

/// A short Hunter word (names, titles): the sampler's text, 1-8 kana.
fn word(c: &mut Composer, lo: usize, hi: usize) -> String {
    for _ in 0..8 {
        let s = c.sampler.sample(c.rng, lo, hi);
        if !s.trim().is_empty() {
            return s.trim().to_string();
        }
    }
    "はんたー".to_string()
}

/// Place a horizontal Hunter line centred at (cx, cy), rotated by `deg`. Returns its quad.
#[allow(clippy::too_many_arguments)]
fn hunter_at(c: &mut Composer, text: &str, st: &TextStyle, em: f64, cx: f64, cy: f64, deg: f64, ink: f32, fx: &Fx, kind: Kind) -> Option<[P; 4]> {
    let inline = c.cfg.inline_prob > 0.0 && c.rng.chance((2.0 * c.cfg.inline_prob).min(0.5));
    let l = if inline {
        match inline_line(c.rng, c.sampler, c.fonts, c.cfg, st, em, text.chars().count().max(2), c.w as f64 * 0.9) {
            Some(l) => l,
            None => make_line_text(c.rng, text, st, em, false, c.cfg.min_stroke_px)?,
        }
    } else {
        make_line_text(c.rng, text, st, em, false, c.cfg.min_stroke_px)?
    };
    let (mx, my) = ((l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0);
    let m = mul(&tr(cx, cy), &mul(&rot(deg), &tr(-mx, -my)));
    c.put_line_fx(l, &m, ink, fx, kind, kind.name())
}

/// Place a foreign line centred at (cx, cy), rotated by `deg`.
fn foreign_at(c: &mut Composer, l: LineG, cx: f64, cy: f64, deg: f64, lum: f32, fx: &Fx, fkind: &'static str) -> Option<[P; 4]> {
    let (mx, my) = ((l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0);
    let m = mul(&tr(cx, cy), &mul(&rot(deg), &tr(-mx, -my)));
    c.put_line_fx(l, &m, lum, fx, Kind::Caption, fkind)
}

fn font_line(c: &mut Composer, script: Script, prefer: &[Class], text: &str, px: f64, bold: f64) -> Option<LineG> {
    let face = pick_face(c.rng, c.fonts, script, prefer)?;
    let fs = FStyle { face, script, bold, track: 0.0, shear: 0.0, caps: false };
    make_font_line(c.rng, &c.tg, c.fonts, &fs, px, false, Role::Word, 1e9, 400, Some(text))
}

/// The caption under (or above) a Hunter line: its katakana reading, a Japanese phrase, or Latin.
fn caption_text(c: &mut Composer, label: &str) -> (Script, String) {
    let u = c.rng.f64();
    if u < 0.45 && c.fonts.has_script(crate::fonts::JAPANESE) {
        (Script::Kana, kata(label).replace(' ', "・"))
    } else if u < 0.8 && c.fonts.has_script(crate::fonts::JAPANESE) {
        let __r0 = c.rng.chance(0.5);
        (Script::Japanese, c.tg.text(c.rng, Script::Japanese, if __r0 { Role::Sentence } else { Role::Heading }, 16))
    } else if u < 0.9 {
        let r = romaji(label);
        let mut ch = r.chars();
        (Script::Romaji, match ch.next() { Some(f) => f.to_uppercase().collect::<String>() + ch.as_str(), None => r })
    } else {
        (Script::Latin, c.tg.text(c.rng, Script::Latin, Role::Phrase, 24))
    }
}

/// A caption right under (75%) or above a horizontal Hunter line with quad `q`, in the line's own
/// frame, `ink`-coloured (or close). Lyric videos and name cards put the Japanese reading there.
pub(crate) fn caption_near(c: &mut Composer, q: &[P; 4], ink: f32, vertical: bool) -> bool {
    if vertical {
        return false;
    }
    let label = c.lines.last().map(|l| l.text.clone()).unwrap_or_default();
    let th = line_thickness(q);
    let u = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
    let ul = u[0].hypot(u[1]).max(1e-6);
    let deg = u[1].atan2(u[0]).to_degrees();
    let v = [-u[1] / ul, u[0] / ul]; // down, in the line frame
    let centre = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0];
    let (script, text) = caption_text(c, &label);
    let px = th * c.rng.uniform(0.45, 0.95);
    let bold = if c.rng.chance(0.4) { c.rng.uniform(0.02, 0.06) } else { 0.0 };
    let prefer: &[Class] = if c.rng.chance(0.5) { &[Class::Sans] } else { &[] };
    let Some(l) = font_line(c, script, prefer, &text, px, bold) else { return false };
    let ch = l.b[3] - l.b[1];
    let below = c.rng.chance(0.75);
    let lum = (ink + c.rng.normal(0.0, 0.05) as f32).clamp(0.0, 1.0);
    for k in 0..3 {
        let gap = (0.31 + 0.12 * k as f64) * th.max(ch) + c.rng.uniform(0.0, 0.4) * th + 1.0;
        let d = th / 2.0 + gap + ch / 2.0;
        let s = if below { 1.0 } else { -1.0 };
        let (cx, cy) = (centre[0] + s * v[0] * d + c.rng.normal(0.0, 0.05) * ul, centre[1] + s * v[1] * d);
        let lc = LineG { raw: l.raw.clone(), vertical: false, ink: clone_ink(&l.ink), b: l.b, cells: Vec::new(), ruby: false };
        if foreign_at(c, lc, cx, cy, deg, lum, &Fx::default(), "caption_pair").is_some() {
            return true;
        }
    }
    false
}

fn clone_ink(i: &Ink) -> Ink {
    match i {
        Ink::Hxh(p) => Ink::Hxh(p.clone()),
        Ink::Font { contours, runs, bold, hollow } => Ink::Font {
            contours: contours.clone(),
            runs: runs.iter().map(|r| crate::page::Run { face: r.face, script: r.script, text: r.text.clone(), b: r.b }).collect(),
            bold: *bold,
            hollow: *hollow,
        },
    }
}

// --- name cards --------------------------------------------------------------------------------

/// Huge outlined Latin letters behind the card content, clipped to the card.
fn big_letters(c: &mut Composer, card: [f64; 4], bg: f32) {
    const WORDS: [&str; 8] = ["HUNTER", "HUNTER", "HUNTER", "HxH", "HUNTERS", "LICENSE", "NEN", "HUNTER×HUNTER"];
    let w = WORDS[c.rng.below(WORDS.len())];
    let (cw, chh) = (card[2] - card[0], card[3] - card[1]);
    let px = chh * c.rng.uniform(0.45, 0.95);
    let Some(l) = font_line(c, Script::Latin, &[Class::Display, Class::Sans], w, px, 0.0) else { return };
    let Ink::Font { contours, .. } = &l.ink else { return };
    let lw = l.b[2] - l.b[0];
    let k = (cw * c.rng.uniform(0.9, 1.6) / lw).min(2.5);
    let m = mul(&tr(card[0] + cw * c.rng.uniform(-0.15, 0.1), card[1] + chh * c.rng.uniform(0.75, 1.05)), &mul(&rot(c.rng.normal(0.0, 1.5)), &[[k, 0.0, -l.b[0] * k], [0.0, k, 0.0], [0.0, 0.0, 1.0]]));
    let clip = irect(card);
    let lighter = (bg + c.rng.uniform(0.12, 0.45) as f32).min(1.0);
    if c.rng.chance(0.5) {
        let fill = (bg + c.rng.uniform(0.03, 0.15) as f32).min(1.0);
        draw_contours(c, contours, &m, clip, fill, Draw::Fill, 1.0);
    }
    let sw = (px * k * c.rng.uniform(0.015, 0.05)).max(1.5) as f32;
    let __r1 = c.rng.uniform(0.6, 1.0);
    draw_contours(c, contours, &m, clip, lighter, Draw::Stroke(sw), __r1 as f32);
}

fn card(c: &mut Composer, r: [f64; 4]) {
    let (cw, ch) = (r[2] - r[0], r[3] - r[1]);
    let bg = PANEL_LUM[c.rng.below(PANEL_LUM.len())] + c.rng.normal(0.0, 0.04) as f32;
    let bg = bg.clamp(0.25, 0.95);
    let ir = irect(r);
    // flat or gradient panel
    if c.rng.chance(0.5) {
        let b2 = (bg + c.rng.uniform(-0.2, 0.2) as f32).clamp(0.2, 1.0);
        for y in ir.y0.max(0)..ir.y1.min(c.h as i32) {
            let t = (y - ir.y0) as f32 / ch.max(1.0) as f32;
            for x in ir.x0.max(0)..ir.x1.min(c.w as i32) {
                c.page.d[y as usize * c.w + x as usize] = bg + (b2 - bg) * t;
            }
        }
    } else {
        c.page.fill(ir.intersect(c.page_rect()), bg);
    }
    if c.rng.chance(0.85) {
        big_letters(c, r, bg);
    }
    // character art on one side (drawn first: text never goes on it)
    let right = c.rng.chance(0.75);
    let art_r = if right { [r[0] + cw * c.rng.uniform(0.58, 0.7), r[1], r[2], r[3]] } else { [r[0], r[1], r[0] + cw * c.rng.uniform(0.3, 0.42), r[3]] };
    let ar = irect(art_r).intersect(c.page_rect());
    if !ar.is_empty() && c.rng.chance(0.85) {
        let n = c.rng.int(3, 12) as usize;
        let ink = c.rng.uniform(0.0, 0.3) as f32;
        art_pub(&mut c.page, ar, c.rng, n, ink, ar.w().min(ar.h()) as f64 / 3.0);
        c.mark(ar);
    }
    let text_r = if right { [r[0], r[1], art_r[0], r[3]] } else { [art_r[2], r[1], r[2], r[3]] };
    let tw = text_r[2] - text_r[0];
    let dark_ink = bg > 0.42 || c.rng.chance(0.3);
    let ink = if dark_ink { c.rng.uniform(0.0, 0.15) as f32 } else { c.rng.uniform(0.88, 1.0) as f32 };
    let contrast = if dark_ink { c.rng.uniform(0.85, 1.0) as f32 } else { c.rng.uniform(0.0, 0.15) as f32 };
    // header: "Hunter Vocabulary" + "ハンター語講座"
    let hx = text_r[0] + tw * c.rng.uniform(0.05, 0.2);
    let mut y = r[1] + ch * c.rng.uniform(0.04, 0.1);
    if c.rng.chance(0.8) {
        let px = ch * c.rng.uniform(0.035, 0.06);
        let __r2 = c.rng.chance(0.7);
        if let Some(l) = font_line(c, Script::Latin, &[Class::Serif, Class::Sans], if __r2 { "Hunter Vocabulary" } else { "Hunter Language" }, px.max(6.0), 0.0) {
            let w = l.b[2] - l.b[0];
            if foreign_at(c, l, hx + w / 2.0, y + px * 0.4, 0.0, ink, &Fx::default(), "card_header").is_some() {
                y += px * 1.3;
            }
        }
        let px = ch * c.rng.uniform(0.05, 0.09);
        let __r3 = c.rng.chance(0.7);
        let __r4 = c.rng.uniform(0.0, 0.05);
        if c.fonts.has_script(crate::fonts::JAPANESE)
            && let Some(l) = font_line(c, Script::Japanese, &[Class::Sans], if __r3 { "ハンター語講座" } else { "ハンター文字" }, px.max(7.0), __r4) {
                let w = l.b[2] - l.b[0];
                if foreign_at(c, l, hx + w / 2.0, y + px * 0.5, 0.0, ink, &Fx::default(), "card_header").is_some() {
                    y += px * 1.3;
                }
            }
    }
    // the name, its katakana reading, its romaji
    let name = word(c, 2, 6);
    let mut st = hxh_style(c, false);
    st.sw = st.sw.max(c.rng.uniform(0.1, 0.26));
    st.mono = false;
    st.spacing = st.spacing.min(0.6);
    let em = (ch * c.rng.uniform(0.13, 0.26)).min(tw / (name.chars().count() as f64 * 1.3 + 0.5)).max(c.cfg.em_min);
    let ny = y + em * c.rng.uniform(0.8, 1.8);
    let mut fx = Fx::default();
    if c.rng.chance(0.35) {
        fx.outline = Some(((em * c.rng.uniform(0.03, 0.08)).max(1.0), contrast));
    }
    if c.rng.chance(0.2) {
        fx.shadow = Some(Shadow { dx: em * 0.06, dy: em * 0.06, sigma: em * 0.05, strength: 0.6, lum: (bg * 0.5).min(0.4) });
    }
    let l = make_line_text(c.rng, &name, &st, em, false, c.cfg.min_stroke_px);
    let Some(l) = l else { return };
    let lw = l.b[2] - l.b[0];
    let nx = text_r[0] + (tw - lw).max(0.0) * c.rng.uniform(0.1, 0.5) + lw / 2.0;
    let (mx, my) = ((l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0);
    let m = mul(&tr(nx, ny), &mul(&rot(c.rng.normal(0.0, 0.6)), &tr(-mx, -my)));
    let Some(q) = c.put_line_fx(l, &m, ink, &fx, Kind::Sign, "sign") else { return };
    let th = line_thickness(&q);
    let label = label_of(&name);
    // katakana right under (the hard negative), then romaji
    if c.fonts.has_script(crate::fonts::JAPANESE) && c.rng.chance(0.9) {
        let px = th * c.rng.uniform(0.55, 1.0);
        let __r5 = c.rng.uniform(0.0, 0.06);
        if let Some(kl) = font_line(c, Script::Kana, &[Class::Sans], &kata(&label), px, __r5) {
            let kh = kl.b[3] - kl.b[1];
            let mut placed = None;
            for k in 0..3 {
                let gap = (0.32 + 0.15 * k as f64) * th.max(kh) + c.rng.uniform(0.0, 0.3) * th + 1.0;
                let cy = ny + th / 2.0 + gap + kh / 2.0;
                let kc = LineG { raw: kl.raw.clone(), vertical: false, ink: clone_ink(&kl.ink), b: kl.b, cells: Vec::new(), ruby: false };
                let __r6 = c.rng.normal(0.0, 0.1);
                if let Some(qq) = foreign_at(c, kc, nx + __r6 * lw, cy, 0.0, ink, &fx, "card_kana") {
                    placed = Some(qq);
                    break;
                }
            }
            if let Some(qq) = placed {
                let px = th * c.rng.uniform(0.25, 0.4);
                let rj = romaji(&label);
                let mut chs = rj.chars();
                let rj = match chs.next() { Some(f) => f.to_uppercase().collect::<String>() + chs.as_str(), None => rj };
                if let Some(rl) = font_line(c, Script::Romaji, &[Class::Sans], &rj, px.max(6.0), 0.02) {
                    let rh = rl.b[3] - rl.b[1];
                    let bot = qq.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
                    foreign_at(c, rl, nx, bot + rh * 0.9 + 2.0, 0.0, ink, &Fx::default(), "card_romaji");
                }
            }
        }
    }
}

pub(crate) fn card_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let pr = c.page_rect();
    let bg = if c.rng.chance(0.6) { c.rng.uniform(0.08, 0.25) } else { c.rng.uniform(0.85, 1.0) } as f32;
    c.page.fill(pr, bg);
    let n = c.rng.weighted(&[0.35, 0.35, 0.3]) + 1;
    let cols = if w > h * 1.1 && c.rng.chance(0.5) { 2 } else { 1 };
    let margin = w.min(h) * c.rng.uniform(0.0, 0.08);
    let gap = if c.rng.chance(0.5) { 0.0 } else { h * 0.01 };
    let cw = (w - 2.0 * margin - gap * (cols - 1) as f64) / cols as f64;
    let chh = ((h - 2.0 * margin - gap * (n - 1) as f64) / n as f64).min(cw * c.rng.uniform(0.45, 0.9));
    let total = chh * n as f64 + gap * (n - 1) as f64;
    let y0 = margin + (h - 2.0 * margin - total) * c.rng.uniform(0.0, 1.0);
    for col in 0..cols {
        for i in 0..n {
            let x = margin + col as f64 * (cw + gap);
            let y = y0 + i as f64 * (chh + gap);
            card(c, [x, y, x + cw, y + chh]);
        }
    }
}

// --- lyric videos ------------------------------------------------------------------------------

fn round_button(c: &mut Composer, cx: f64, cy: f64, r: f64, lum_bg: f32, icon_lum: f32) -> bool {
    let b = [cx - r - 2.0, cy - r - 2.0, cx + r + 2.0, cy + r + 2.0];
    if !c.free(irect(b)) || !c.foreign_clear(&quad_of(b), 2.0) {
        return false;
    }
    let style = c.rng.below(3);
    match style {
        0 => disk(c, cx, cy, r, lum_bg, Draw::Fill),
        1 => { let sw = (r * c.rng.uniform(0.06, 0.15)).max(1.0) as f32; disk(c, cx, cy, r * 0.94, lum_bg, Draw::Stroke(sw)) }
        _ => {}
    }
    // the glyph inside: an icon-font glyph or a drawn symbol (+, x, ...), the ほ look-alikes
    let (icons, syms) = c.fonts.icon_faces();
    let drawn = if !icons.is_empty() && c.rng.chance(0.6) {
        let f = icons[c.rng.below(icons.len())];
        let __r7 = c.rng.uniform(0.9, 1.4);
        let __r8 = c.rng.chance(0.3);
        match make_icon_line(c.rng, c.fonts, f, r * __r7, 1, 0.0, __r8, r * 3.0) {
            Some(l) => draw_icon_raw(c, &l, cx, cy, icon_lum),
            None => false,
        }
    } else if !syms.is_empty() && c.rng.chance(0.3) {
        let f = syms[c.rng.below(syms.len())];
        match make_icon_line(c.rng, c.fonts, f, r * 1.2, 1, 0.0, false, r * 3.0) {
            Some(l) => draw_icon_raw(c, &l, cx, cy, icon_lum),
            None => false,
        }
    } else {
        false
    };
    if !drawn {
        let a = r * c.rng.uniform(0.35, 0.6);
        let lw = (r * c.rng.uniform(0.08, 0.18)).max(1.0) as f32;
        let mut pb = PathBuilder::new();
        match c.rng.below(5) {
            0 => { pb.move_to((cx - a) as f32, cy as f32); pb.line_to((cx + a) as f32, cy as f32); pb.move_to(cx as f32, (cy - a) as f32); pb.line_to(cx as f32, (cy + a) as f32); }
            1 => { pb.move_to((cx - a) as f32, (cy - a) as f32); pb.line_to((cx + a) as f32, (cy + a) as f32); pb.move_to((cx + a) as f32, (cy - a) as f32); pb.line_to((cx - a) as f32, (cy + a) as f32); }
            2 => { for k in -1..=1 { pb.move_to((cx - a) as f32, (cy + k as f64 * a * 0.6) as f32); pb.line_to((cx + a) as f32, (cy + k as f64 * a * 0.6) as f32); } }
            3 => { pb.move_to((cx - a * 0.5) as f32, (cy - a) as f32); pb.line_to((cx + a * 0.6) as f32, cy as f32); pb.line_to((cx - a * 0.5) as f32, (cy + a) as f32); pb.close(); }
            _ => { for k in -1..=1 { pb.push_circle((cx + k as f64 * a * 0.7) as f32, cy as f32, (a * 0.18).max(1.0) as f32); } }
        }
        if let Some(p) = pb.finish()
            && let Some(cv) = coverage(&[(p, Draw::Stroke(lw))], c.page_rect()) {
                blend(&mut c.page, &cv, icon_lum, 1.0);
            }
    }
    c.mark(irect(b));
    c.record_foreign(quad_of([cx - r, cy - r, cx + r, cy + r]), "round icon", "icon", "round_icon", 2.0);
    true
}

fn draw_icon_raw(c: &mut Composer, l: &LineG, cx: f64, cy: f64, lum: f32) -> bool {
    let Ink::Font { contours, hollow, .. } = &l.ink else { return false };
    let m = tr(cx - (l.b[0] + l.b[2]) / 2.0, cy - (l.b[1] + l.b[3]) / 2.0);
    let how = if *hollow > 0.0 { Draw::Stroke(*hollow as f32) } else { Draw::Fill };
    let clip = c.page_rect();
    draw_contours(c, contours, &m, clip, lum, how, 1.0);
    true
}

/// An avatar: a disc with a tiny scene inside (face-like blobs), optional ring and badge.
fn avatar(c: &mut Composer, cx: f64, cy: f64, r: f64) -> bool {
    let b = [cx - r - 3.0, cy - r - 3.0, cx + r + 3.0, cy + r + 3.0];
    if !c.free(irect(b)) || !c.foreign_clear(&quad_of(b), 3.0) {
        return false;
    }
    let base = c.rng.uniform(0.2, 0.9) as f32;
    disk(c, cx, cy, r, base, Draw::Fill);
    // blobs clipped to the disc
    let n = c.rng.int(2, 6);
    for _ in 0..n {
        let (a, d) = (c.rng.uniform(0.0, TAU), r * c.rng.uniform(0.0, 0.7));
        let rr = r * c.rng.uniform(0.15, 0.55);
        let (bx, by) = (cx + d * a.cos(), cy + d * a.sin());
        let lum = c.rng.f64() as f32;
        let k = rr.min(r - d.min(r - 1.0)).max(1.0);
        disk(c, bx, by, k, lum, Draw::Fill);
    }
    if c.rng.chance(0.5) {
        // a face: two eyes
        let e = r * 0.12;
        let lum = c.rng.uniform(0.0, 0.2) as f32;
        disk(c, cx - r * 0.3, cy - r * 0.1, e, lum, Draw::Fill);
        disk(c, cx + r * 0.3, cy - r * 0.1, e, lum, Draw::Fill);
    }
    if c.rng.chance(0.5) {
        let lum = c.rng.f64() as f32;
        let __r9 = c.rng.uniform(0.05, 0.14);
        disk(c, cx, cy, r, lum, Draw::Stroke((r * __r9).max(1.0) as f32));
    }
    if c.rng.chance(0.3) {
        let (bx, by, br) = (cx + r * 0.7, cy + r * 0.7, r * 0.3);
        let __r10 = c.rng.uniform(0.3, 0.6);
        disk(c, bx, by, br, __r10 as f32, Draw::Fill);
        let mut pb = PathBuilder::new();
        pb.move_to((bx - br * 0.5) as f32, by as f32);
        pb.line_to((bx + br * 0.5) as f32, by as f32);
        pb.move_to(bx as f32, (by - br * 0.5) as f32);
        pb.line_to(bx as f32, (by + br * 0.5) as f32);
        if let Some(p) = pb.finish()
            && let Some(cv) = coverage(&[(p, Draw::Stroke((br * 0.25).max(1.0) as f32))], c.page_rect()) {
                blend(&mut c.page, &cv, 1.0, 1.0);
            }
    }
    c.mark(irect(b));
    c.record_foreign(quad_of([cx - r, cy - r, cx + r, cy + r]), "avatar", "icon", "avatar", 3.0);
    true
}

fn small_text(c: &mut Composer, script: Script, role: Role, px: f64, x: f64, y: f64, lum: f32, kind: &'static str) -> Option<[P; 4]> {
    let fonts = c.fonts;
    if !fonts.has_script(script.need()) {
        return None;
    }
    let fs = sample_fstyle(c.rng, fonts, script, Kind::Caption)?;
    let fs = FStyle { bold: if c.rng.chance(0.5) { 0.04 } else { 0.0 }, ..fs };
    let l = make_font_line(c.rng, &c.tg, fonts, &fs, px, false, role, c.w as f64 * 0.6, 40, None)?;
    let (w, h) = (l.b[2] - l.b[0], l.b[3] - l.b[1]);
    foreign_at(c, l, x + w / 2.0, y + h / 2.0, 0.0, lum, &Fx::default(), kind)
}

pub(crate) fn lyric_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let pr = c.page_rect();
    let bg = c.rng.uniform(0.0, 0.12) as f32;
    c.page.fill(pr, bg);
    // the video: a darkened scene, sometimes a portrait column with darker side bars
    let (vx0, vx1) = if w > h && c.rng.chance(0.6) {
        let vw = h * c.rng.uniform(0.5, 0.65);
        let x0 = (w - vw) / 2.0 + c.rng.normal(0.0, 0.05) * w;
        (x0.max(0.0), (x0 + vw).min(w))
    } else {
        (0.0, w)
    };
    let vr = IRect { x0: vx0 as i32, y0: 0, x1: vx1 as i32, y1: h as i32 }.intersect(pr);
    if c.rng.chance(0.6) {
        fill_bg_pub(&mut c.page, vr, c.rng, true);
        let k = c.rng.uniform(0.1, 0.35) as f32;
        for y in vr.y0..vr.y1 {
            for x in vr.x0..vr.x1 {
                let v = &mut c.page.d[y as usize * c.w + x as usize];
                *v = bg + (*v - bg) * k;
            }
        }
    }
    let vw = vx1 - vx0;
    let ink = c.rng.uniform(0.85, 1.0) as f32;
    // lyric lines with captions
    let n = c.rng.weighted(&[0.5, 0.35, 0.15]) + 1;
    let em = (vw * c.rng.uniform(0.035, 0.085)).max(c.cfg.em_min);
    let mut y = h * c.rng.uniform(0.2, 0.45);
    let mut fx = Fx::default();
    if c.rng.chance(0.35) {
        fx.halo = Some(Halo { grow: em * 0.06, sigma: em * c.rng.uniform(0.05, 0.12), lum: 1.0, strength: c.rng.uniform(0.3, 0.8) as f32 });
    } else if c.rng.chance(0.3) {
        fx.shadow = Some(Shadow { dx: em * 0.05, dy: em * 0.05, sigma: em * 0.05, strength: 0.8, lum: 0.0 });
    }
    for _ in 0..n {
        let __r11 = c.rng.chance(0.75);
        let mut st = hxh_style(c, __r11);
        st.wave_amp = 0.0;
        // as many glyphs as fit in 90% of the video width at this spacing
        let per = if st.mono { st.cell } else { 0.95 + st.spacing };
        let fit = ((0.9 * vw / em - 0.95) / per).floor() as usize + 1;
        if fit < 2 {
            break;
        }
        let text: String = word(c, 2, fit.min(10)).chars().take(fit).collect();
        let cx = vx0 + vw / 2.0 + c.rng.normal(0.0, 0.03) * vw;
        let __r12 = c.rng.normal(0.0, 0.5);
        if let Some(q) = hunter_at(c, &text, &st, em, cx, y, __r12, ink, &fx, Kind::Free) {
            let th = line_thickness(&q);
            if c.rng.chance(0.85) {
                caption_near(c, &q, ink, false);
            }
            y += th * c.rng.uniform(2.8, 4.5);
        } else {
            y += em * 2.0;
        }
        if y > h * 0.8 {
            break;
        }
    }
    // app UI: right-hand column of round buttons with counts, avatar on top
    let r = (vw * c.rng.uniform(0.035, 0.06)).max(8.0);
    let colx = vx1 - r * c.rng.uniform(1.6, 2.4);
    let mut yy = h * c.rng.uniform(0.35, 0.5);
    if c.rng.chance(0.8) {
        avatar(c, colx, yy, r * 1.1);
        yy += r * 3.2;
    }
    for _ in 0..c.rng.int(2, 5) {
        if yy > h - r * 3.0 {
            break;
        }
        let __r13 = c.rng.uniform(0.15, 0.4);
        let __r14 = c.rng.uniform(0.85, 1.0);
        round_button(c, colx, yy, r, __r13 as f32, __r14 as f32);
        let px = (r * 0.5).max(6.0);
        let count = format!("{}{}", c.rng.int(1, 999), if c.rng.chance(0.6) { ".".to_string() + &c.rng.int(1, 9).to_string() + "K" } else { String::new() });
        if let Some(l) = font_line(c, Script::Digits, &[Class::Sans], &count, px, 0.03) {
            foreign_at(c, l, colx, yy + r * 1.45, 0.0, 0.95, &Fx::default(), "ui_count");
        }
        yy += r * 3.4;
    }
    // bottom-left: @user, description with hashtags, sound line; top tabs; progress bar
    let px = (vw * c.rng.uniform(0.022, 0.035)).max(6.0);
    let x = vx0 + vw * c.rng.uniform(0.03, 0.06);
    let mut by = h * c.rng.uniform(0.8, 0.86);
    let user = format!("@{}", c.tg.text(c.rng, Script::Romaji, Role::Word, 14).to_lowercase().replace(' ', "_"));
    if let Some(l) = font_line(c, Script::Latin, &[Class::Sans], &user, px * 1.1, 0.04) {
        let (lw, lh) = (l.b[2] - l.b[0], l.b[3] - l.b[1]);
        foreign_at(c, l, x + lw / 2.0, by + lh / 2.0, 0.0, 0.95, &Fx::default(), "ui_user");
        by += lh * 1.8;
    }
    let script = if c.rng.chance(0.5) && c.fonts.has_script(crate::fonts::JAPANESE) { Script::Japanese } else { Script::Latin };
    if let Some(q) = small_text(c, script, Role::Phrase, px, x, by, 0.9, "ui_text") {
        by = q[2][1] + px * 0.7;
    }
    small_text(c, Script::Latin, Role::Label, px * 0.9, x, by, 0.85, "ui_text");
    let tabs = ["Following", "For You", "LIVE", "Explore", "Friends"];
    let mut tx = vx0 + vw * c.rng.uniform(0.15, 0.3);
    for t in tabs.iter().take(c.rng.int(2, 4) as usize) {
        if let Some(l) = font_line(c, Script::Latin, &[Class::Sans], t, px * 1.1, 0.03) {
            let lw = l.b[2] - l.b[0];
            let __r15 = c.rng.uniform(0.6, 1.0);
            foreign_at(c, l, tx + lw / 2.0, h * 0.05, 0.0, __r15 as f32, &Fx::default(), "ui_tab");
            tx += lw + px * 1.5;
        }
    }
    let bar_y = (h - h * c.rng.uniform(0.005, 0.03)) as i32;
    let bar = IRect { x0: vx0 as i32, y0: bar_y, x1: vx1 as i32, y1: bar_y + 2 }.intersect(pr);
    c.page.fill(bar, 0.45);
    let done = IRect { x0: vx0 as i32, y0: bar_y, x1: (vx0 + vw * c.rng.f64()) as i32, y1: bar_y + 2 }.intersect(pr);
    c.page.fill(done, 0.95);
}

// --- anime frames with screens -----------------------------------------------------------------

pub(crate) fn tv_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let pr = c.page_rect();
    fill_bg_pub(&mut c.page, pr, c.rng, true);
    let n = c.rng.int(2, 8) as usize;
    let __r16 = c.rng.uniform(0.0, 0.3);
    art_pub(&mut c.page, pr, c.rng, n, __r16 as f32, w.min(h) / 5.0);
    let mut placed_any = false;
    // 1-2 screens / panels in the scene
    for _ in 0..c.rng.int(1, 3) {
        let sw = w * c.rng.uniform(0.3, 0.7);
        let sh = sw * c.rng.uniform(0.35, 0.7);
        let (x0, y0) = (c.rng.uniform(0.0, (w - sw).max(1.0)), c.rng.uniform(0.0, (h - sh).max(1.0)));
        let r = [x0, y0, x0 + sw, y0 + sh];
        if !c.free(irect(r)) {
            continue;
        }
        let bg = c.rng.uniform(0.2, 0.55) as f32; // blue / teal screens
        let ir = irect(r).intersect(pr);
        c.page.fill(ir, bg);
        let bw = c.rng.uniform(1.0, 6.0);
        let frame_lum = c.rng.uniform(0.0, 0.25) as f32;
        for (a, b) in [((ir.x0, ir.y0), (ir.x1, ir.y0 + bw as i32)), ((ir.x0, ir.y1 - bw as i32), (ir.x1, ir.y1)), ((ir.x0, ir.y0), (ir.x0 + bw as i32, ir.y1)), ((ir.x1 - bw as i32, ir.y0), (ir.x1, ir.y1))] {
            c.page.fill(IRect { x0: a.0, y0: a.1, x1: b.0, y1: b.1 }.intersect(pr), frame_lum);
        }
        let mut st = hxh_style(c, false);
        let em = (sh * c.rng.uniform(0.15, 0.3)).max(c.cfg.em_min);
        let mut fx = Fx::default();
        let bubble = c.rng.chance(0.35);
        let ink;
        if bubble {
            // bubble-font title: thick hollow glyphs, light fill, darker outline
            st.sw = c.rng.uniform(0.2, 0.32);
            fx.outline = Some(((em * c.rng.uniform(0.04, 0.08)).max(1.0), c.rng.uniform(0.2, 0.5) as f32));
            ink = c.rng.uniform(0.7, 0.95) as f32;
        } else {
            ink = c.rng.uniform(0.85, 1.0) as f32;
            if c.rng.chance(0.6) {
                fx.outline = Some(((em * c.rng.uniform(0.03, 0.07)).max(1.0), c.rng.uniform(0.0, 0.2) as f32));
            }
        }
        let text = word(c, 2, 8);
        let lines_n = c.rng.int(1, 3);
        for k in 0..lines_n {
            let cy = y0 + sh * (k as f64 + 1.0) / (lines_n as f64 + 1.0);
            let __w = if k == 0 { text.clone() } else { word(c, 2, 8) };
            let __r17 = c.rng.normal(0.0, 0.5);
            if let Some(q) = hunter_at(c, &__w, &st, em, x0 + sw / 2.0, cy, __r17, ink, &fx, Kind::Sign) {
                placed_any = true;
                if c.rng.chance(0.35) {
                    // a time / number next to the title ("3/11 15:00", "225")
                    let t = match c.rng.below(3) {
                        0 => format!("{}/{} {}:{:02}", c.rng.int(1, 13), c.rng.int(1, 29), c.rng.int(0, 24), c.rng.int(0, 60) / 5 * 5),
                        1 => format!("{}", c.rng.int(1, 999)),
                        _ => format!("{}:{:02}", c.rng.int(0, 24), c.rng.int(0, 60)),
                    };
                    let th = line_thickness(&q);
                    let __r18 = c.rng.uniform(0.6, 1.0);
                    if let Some(l) = font_line(c, Script::Digits, &[Class::Sans, Class::Display], &t, th * __r18, 0.03) {
                        let lw = l.b[2] - l.b[0];
                        let left = q.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
                        let right = q.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
                        let gx = th * c.rng.uniform(0.45, 1.2);
                        let x = if c.rng.chance(0.5) { left - gx - lw / 2.0 } else { right + gx + lw / 2.0 };
                        foreign_at(c, l, x, (q[0][1] + q[3][1]) / 2.0, 0.0, ink, &fx, "tv_digits");
                    }
                }
            }
        }
        c.mark(ir);
    }
    // a news ticker bar
    if c.rng.chance(0.6) {
        let bh = h * c.rng.uniform(0.06, 0.12);
        let by = h * c.rng.uniform(0.7, 0.9);
        let bar = IRect { x0: 0, y0: by as i32, x1: c.w as i32, y1: (by + bh) as i32 }.intersect(pr);
        if c.free(bar) {
            let bg = c.rng.uniform(0.18, 0.45) as f32;
            c.page.fill(bar, bg);
            let lab_w = w * c.rng.uniform(0.1, 0.2);
            c.page.fill(IRect { x0: 0, y0: bar.y0, x1: lab_w as i32, y1: bar.y1 }.intersect(pr), c.rng.uniform(0.5, 0.9) as f32);
            let __r19 = c.rng.below(3);
            if let Some(l) = font_line(c, Script::Latin, &[Class::Sans, Class::Display], ["NEWS", "LIVE", "BREAKING", "速報"][__r19], bh * 0.45, 0.04) {
                foreign_at(c, l, lab_w / 2.0, by + bh / 2.0, 0.0, 0.05, &Fx::default(), "ticker_label");
            }
            let st = hxh_style(c, false);
            let em = bh * c.rng.uniform(0.45, 0.65);
            let text = c.sampler.sample(c.rng, 4, 14);
            let ink = c.rng.uniform(0.88, 1.0) as f32;
            let l = make_line_text(c.rng, text.trim(), &st, em, false, c.cfg.min_stroke_px);
            if let Some(l) = l {
                let lw = l.b[2] - l.b[0];
                let x = lab_w + em * 0.8 + lw / 2.0;
                let (mx, my) = ((l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0);
                if x + lw / 2.0 < w - 4.0 {
                    let m = mul(&tr(x, by + bh / 2.0), &tr(-mx, -my));
                    placed_any |= c.put_line_fx(l, &m, ink, &Fx::default(), Kind::Caption, "caption").is_some();
                }
            }
            c.mark(bar);
        }
    }
    // white glyphs on a dark shirt
    if c.rng.chance(0.4) {
        let r = w.min(h) * c.rng.uniform(0.12, 0.25);
        let (cx, cy) = (c.rng.uniform(r, w - r), c.rng.uniform(r, h - r));
        let b = [cx - r, cy - r, cx + r, cy + r];
        if c.free(irect(b)) {
            let poly = blob(c.rng, cx, cy, r, 40);
            if let Some(p) = poly_path(&poly, true)
                && let Some(cv) = coverage(&[(p, Draw::Fill)], pr) {
                    blend(&mut c.page, &cv, c.rng.uniform(0.02, 0.15) as f32, 1.0);
                }
            let st = hxh_style(c, false);
            let text = word(c, 2, 4);
            let em = (r * c.rng.uniform(0.25, 0.45)).max(c.cfg.em_min);
            let __r20 = c.rng.normal(0.0, 4.0);
            let __r21 = c.rng.uniform(0.85, 1.0);
            placed_any |= hunter_at(c, &text, &st, em, cx, cy, __r20, __r21 as f32, &Fx::default(), Kind::Free).is_some();
            c.mark(irect(b));
        }
    }
    // the usual frame elements fill the rest
    let full = [4.0, 4.0, w - 4.0, h - 4.0];
    for _ in 0..c.rng.int(if placed_any { 0 } else { 1 }, 3) {
        c.add(Kind::Free, full, crate::page::Want::Auto);
    }
    for _ in 0..c.rng.int(1, 5) {
        c.add(Kind::Filler, full, crate::page::Want::Foreign);
    }
}

// --- hard negatives on any page ------------------------------------------------------------------

/// A toolbar / browser-chrome strip along the top or bottom edge with icons and an address,
/// unless a labelled line is near it.
fn chrome_strip(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let sh = (h * c.rng.uniform(0.03, 0.06)).max(14.0);
    let top = c.rng.chance(0.7);
    let y0 = if top { 0.0 } else { h - sh };
    let strip = [0.0, y0, w, y0 + sh];
    let sq = quad_of([0.0, y0 - sh * 0.5, w, y0 + sh * 1.5]);
    if c.hunter_quads().iter().any(|q| convex_dist(q, &sq) < 2.0) {
        return;
    }
    let lum = if c.rng.chance(0.5) { c.rng.uniform(0.1, 0.3) } else { c.rng.uniform(0.85, 0.97) } as f32;
    let fg = if lum < 0.5 { c.rng.uniform(0.75, 0.95) } else { c.rng.uniform(0.1, 0.4) } as f32;
    c.page.fill(irect(strip).intersect(c.page_rect()), lum);
    let px = sh * 0.38;
    let url = c.tg.text(c.rng, Script::Url, Role::Link, 60);
    if let Some(l) = font_line(c, Script::Latin, &[Class::Sans], &url, px, 0.0) {
        let lw = l.b[2] - l.b[0];
        foreign_at(c, l, w * 0.15 + lw / 2.0, y0 + sh / 2.0, 0.0, fg, &Fx::default(), "chrome_url");
    }
    let (icons, _) = c.fonts.icon_faces();
    if !icons.is_empty() {
        let f = icons[c.rng.below(icons.len())];
        let n = c.rng.int(3, 10) as usize;
        let __r22 = c.rng.uniform(0.4, 1.0);
        let __r23 = c.rng.chance(0.3);
        if let Some(l) = make_icon_line(c.rng, c.fonts, f, sh * 0.5, n, __r22, __r23, w * 0.35) {
            let lw = l.b[2] - l.b[0];
            foreign_at(c, l, w - lw / 2.0 - sh * 0.5, y0 + sh / 2.0, 0.0, fg, &Fx::default(), "chrome_icons");
        }
    }
    c.mark(irect(strip));
}

/// Avatars, round icon buttons and a toolbar strip wherever the page is still free.
pub(crate) fn ui_negatives(c: &mut Composer) {
    if c.rng.chance(0.35) {
        chrome_strip(c);
    }
    let n = c.rng.int(1, 7);
    let s = c.w.min(c.h) as f64;
    for _ in 0..n {
        let r = c.rng.log_uniform(6.0, (s * 0.08).max(8.0));
        for _ in 0..6 {
            let (cx, cy) = (c.rng.uniform(r + 3.0, c.w as f64 - r - 3.0), c.rng.uniform(r + 3.0, c.h as f64 - r - 3.0));
            let ok = if c.rng.chance(0.45) {
                avatar(c, cx, cy, r)
            } else {
                let bgl = c.page.d[(cy as usize).min(c.h - 1) * c.w + (cx as usize).min(c.w - 1)];
                let icon_lum = if bgl > 0.5 { c.rng.uniform(0.0, 0.3) } else { c.rng.uniform(0.75, 1.0) } as f32;
                let __r24 = c.rng.uniform(0.1, 0.9);
                round_button(c, cx, cy, r, __r24 as f32, icon_lum)
            };
            if ok {
                break;
            }
        }
    }
    // a row of round buttons (toolbar / player controls)
    if c.rng.chance(0.3) {
        let r = c.rng.uniform(6.0, (s * 0.03).max(7.0));
        let k = c.rng.int(3, 7);
        let (x0, y) = (c.rng.uniform(r + 3.0, (c.w as f64 * 0.5).max(r + 4.0)), c.rng.uniform(r + 3.0, c.h as f64 - r - 3.0));
        let lum = c.rng.uniform(0.1, 0.9) as f32;
        let il = if lum > 0.5 { 0.1 } else { 0.95 };
        for i in 0..k {
            round_button(c, x0 + i as f64 * r * 2.8, y, r, lum, il);
        }
    }
}

// --- product promos (v6) ----------------------------------------------------------------------

fn polyline(c: &mut Composer, pts: &[P], width: f64, lum: f32, k: f32) {
    if let Some(p) = poly_path(pts, false)
        && let Some(cv) = coverage(&[(p, Draw::Stroke(width as f32))], c.page_rect()) {
            blend(&mut c.page, &cv, lum, k);
        }
}

/// The product: a glossy body with a rim, a highlight, seams and buttons, a sparkle.
fn product(c: &mut Composer, r: [f64; 4]) {
    let (cx, cy) = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0);
    let (a, b) = ((r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0);
    if a < 8.0 || b < 8.0 {
        return;
    }
    let body = c.rng.uniform(0.12, 0.6) as f32;
    let pts: Vec<P> = if c.rng.chance(0.5) {
        ellipse(cx, cy, a, b, 64)
    } else {
        let k = a / a.min(b);
        blob(c.rng, cx, cy, a.min(b), 24).into_iter().map(|p| [cx + (p[0] - cx) * k, p[1]]).collect()
    };
    let rim = (a.min(b) * c.rng.uniform(0.02, 0.06)).max(1.0);
    if let Some(p) = poly_path(&pts, true) {
        let dark = (body - c.rng.uniform(0.05, 0.12) as f32).max(0.0);
        if let Some(cv) = coverage(&[(p.clone(), Draw::Fill), (p, Draw::Stroke(2.0 * rim as f32))], c.page_rect()) {
            blend(&mut c.page, &cv, dark, 1.0);
        }
    }
    if let Some(p) = poly_path(&ellipse(cx, cy, (a - 2.0 * rim).max(1.0), (b - 2.0 * rim).max(1.0), 64), true)
        && let Some(cv) = coverage(&[(p, Draw::Fill)], c.page_rect()) {
            let k = c.rng.uniform(0.3, 0.9) as f32;
            blend(&mut c.page, &cv, body, k);
        }
    // highlight and details
    let hl = (body + c.rng.uniform(0.25, 0.45) as f32).min(1.0);
    let (hx, hy) = (cx - a * c.rng.uniform(0.1, 0.4), cy - b * c.rng.uniform(0.2, 0.5));
    let (hrx, hry, hang) = (a * c.rng.uniform(0.15, 0.4), b * c.rng.uniform(0.05, 0.15), c.rng.uniform(-30.0, 30.0));
    if let Some(p) = ellipse_path(hx, hy, hrx, hry, hang)
        && let Some(cv) = coverage(&[(p, Draw::Fill)], c.page_rect()) {
            let mut cv = cv;
            let (cw, ch) = (cv.r.w(), cv.r.h());
            blur(&mut cv.a, cw, ch, (a.min(b) * 0.03).max(0.8), false);
            let k = c.rng.uniform(0.5, 0.9) as f32;
            blend(&mut c.page, &cv, hl, k);
        }
    let clip = IRect { x0: (cx - a * 0.8) as i32, y0: (cy - b * 0.8) as i32, x1: (cx + a * 0.8) as i32, y1: (cy + b * 0.8) as i32 }.intersect(c.page_rect());
    let n = c.rng.int(1, 5) as usize;
    let ink = (body - 0.2).max(0.0);
    if !clip.is_empty() {
        art_pub(&mut c.page, clip, c.rng, n, ink, a.min(b) / 3.0);
    }
    for _ in 0..c.rng.int(0, 4) {
        let (bx, by) = (cx + a * c.rng.uniform(-0.6, 0.6), cy + b * c.rng.uniform(-0.5, 0.5));
        let br = a.min(b) * c.rng.uniform(0.04, 0.1);
        let l = c.rng.uniform(0.1, 0.5) as f32;
        disk(c, bx, by, br, l, Draw::Fill);
    }
    if c.rng.chance(0.6) {
        // a sparkle: rays from a point
        let (sx, sy) = (cx + a * c.rng.uniform(-0.3, 0.8), cy - b * c.rng.uniform(0.0, 0.7));
        let len = a.min(b) * c.rng.uniform(0.2, 0.6);
        let k = c.rng.int(4, 9);
        let ph = c.rng.uniform(0.0, TAU);
        for i in 0..k {
            let t = ph + TAU * i as f64 / k as f64;
            let l = len * if i % 2 == 0 { 1.0 } else { 0.45 };
            polyline(c, &[[sx, sy], [sx + l * t.cos(), sy + l * t.sin()]], (len * 0.03).max(1.0), 1.0, 0.9);
        }
        disk(c, sx, sy, (len * 0.08).max(1.5), 1.0, Draw::Fill);
    }
}

/// A product promo picture (v6). See the module docs.
pub(crate) fn promo_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let dark = c.rng.chance(0.28);
    // ground: grey with a marble / paper texture
    let base = if dark { c.rng.uniform(0.05, 0.3) } else { c.rng.uniform(0.72, 0.97) } as f32;
    let (cell1, cell2) = (c.rng.uniform(25.0, 140.0), c.rng.uniform(3.0, 14.0));
    let n1 = value_noise(c.w, c.h, c.rng, cell1);
    let n2 = value_noise(c.w, c.h, c.rng, cell2);
    let (a1, a2) = (c.rng.uniform(0.02, 0.1) as f32, c.rng.uniform(0.0, 0.05) as f32);
    for (i, v) in c.page.d.iter_mut().enumerate() {
        *v = (base + a1 * n1[i] + a2 * n2[i]).clamp(0.0, 1.0);
    }
    if c.rng.chance(0.6) {
        // marble veins
        for _ in 0..c.rng.int(2, 9) {
            let p0 = [c.rng.uniform(-0.1, 1.1) * w, c.rng.uniform(-0.1, 1.1) * h];
            let p3 = [c.rng.uniform(-0.1, 1.1) * w, c.rng.uniform(-0.1, 1.1) * h];
            let p1 = [c.rng.uniform(0.0, 1.0) * w, c.rng.uniform(0.0, 1.0) * h];
            let p2 = [c.rng.uniform(0.0, 1.0) * w, c.rng.uniform(0.0, 1.0) * h];
            let lum = if dark { base + c.rng.uniform(0.05, 0.25) as f32 } else { base - c.rng.uniform(0.05, 0.3) as f32 }.clamp(0.0, 1.0);
            let wd = c.rng.uniform(0.5, 2.5);
            let k = c.rng.uniform(0.3, 0.8) as f32;
            polyline(c, &bezier(&[p0, p1, p2, p3], 40), wd, lum, k);
        }
    }
    // the product on one side (landscape) or under the lettering (portrait)
    let portrait = h > w * 1.1;
    let right = c.rng.chance(0.7);
    let (tx0, tx1, prod) = if portrait {
        let (x0, x1) = (w * c.rng.uniform(0.03, 0.08), w * c.rng.uniform(0.9, 0.97));
        (x0, x1, [w * c.rng.uniform(0.0, 0.3), h * c.rng.uniform(0.4, 0.55), w * c.rng.uniform(0.7, 1.05), h * c.rng.uniform(0.85, 1.02)])
    } else if right {
        let (x0, x1) = (w * c.rng.uniform(0.02, 0.07), w * c.rng.uniform(0.5, 0.62));
        (x0, x1, [w * c.rng.uniform(0.5, 0.62), h * c.rng.uniform(0.1, 0.35), w * c.rng.uniform(0.9, 1.05), h * c.rng.uniform(0.7, 1.0)])
    } else {
        let (x0, x1) = (w * c.rng.uniform(0.45, 0.55), w * 0.97);
        (x0, x1, [w * c.rng.uniform(-0.05, 0.08), h * c.rng.uniform(0.1, 0.35), w * c.rng.uniform(0.4, 0.5), h * c.rng.uniform(0.7, 1.0)])
    };
    product(c, prod);
    // the lettering: 1-3 lines of sticker Hunter text, left-aligned, numbers and !! in them
    let tw = tx1 - tx0;
    let n = c.rng.weighted(&[0.3, 0.5, 0.2]) + 1;
    let em0 = (h.min(w * 0.75) * c.rng.uniform(0.05, 0.12)).min(tw / 3.5).max(c.cfg.em_min);
    let mut st = hxh_style(c, false);
    st.wave_amp = 0.0;
    st.wave_along = 0.0;
    if c.rng.chance(0.6) {
        st.sw = c.rng.uniform(0.1, 0.22);
    }
    if !st.mono {
        st.spacing = st.spacing.min(0.6);
    }
    let ink = if dark { c.rng.uniform(0.85, 1.0) } else { c.rng.uniform(0.0, 0.15) } as f32;
    let olum = if dark { c.rng.uniform(0.0, 0.15) } else { c.rng.uniform(0.9, 1.0) } as f32;
    let sticker = c.rng.chance(0.85);
    let (out_k, shadow) = (c.rng.uniform(0.12, 0.32), c.rng.chance(0.7));
    let (sdx, sdy, ssig, sstr) = (c.rng.uniform(-0.03, 0.08), c.rng.uniform(0.02, 0.12), c.rng.uniform(0.04, 0.12), c.rng.uniform(0.3, 0.7) as f32);
    let slum = if dark { 0.0 } else { c.rng.uniform(0.35, 0.65) as f32 };
    let ruby_on = c.fonts.has_script(crate::fonts::JAPANESE) && c.rng.chance(0.7);
    let digits_p = (c.cfg.inline_prob * 1.5).min(0.8);
    let mut y = h * c.rng.uniform(0.04, 0.12);
    for _ in 0..n {
        let with_digits = c.cfg.inline_prob > 0.0 && c.rng.chance(digits_p);
        let mut em = em0 * c.rng.uniform(0.85, 1.1);
        let mut built = None;
        for _ in 0..4 {
            let r_out = if sticker { (em * out_k).round().max(1.5) } else { 0.0 };
            let room = tw - 2.0 * r_out;
            let max_len = ((room / (em * (1.0 + st.spacing.min(0.6)))).floor() as usize).clamp(1, 9);
            let l = if with_digits {
                inline_line(c.rng, c.sampler, c.fonts, c.cfg, &st, em, max_len, room)
            } else {
                let t = word(c, 1, max_len.max(2));
                make_line_cells(c.rng, &t, &st, em, c.cfg.min_stroke_px)
            };
            if let Some(l) = l
                && l.b[2] - l.b[0] <= room {
                    built = Some((l, r_out));
                    break;
                }
            em = (em * 0.8).max(c.cfg.em_min);
        }
        let Some((l, r_out)) = built else { continue };
        let mut fx = Fx::default();
        if sticker {
            fx.outline = Some((r_out, olum));
            fx.sticker = true;
            if shadow {
                fx.shadow = Some(Shadow { dx: em * sdx, dy: em * sdy, sigma: em * ssig, strength: sstr, lum: slum });
                fx.shadow_grow = r_out;
            }
        }
        let ruby = if ruby_on { ruby_line(c.rng, c.fonts, &l, em) } else { None };
        let top = ruby.as_ref().map_or(l.b[1], |r| r.b[1].min(l.b[1]));
        let lh = l.b[3] - top;
        let e = fx.extent();
        if y + lh + 2.0 * e > h * 0.95 {
            break;
        }
        let ang = c.rng.normal(0.0, 0.8);
        let m = mul(&tr(tx0 + e, y + e), &mul(&rot(ang), &tr(-l.b[0], -top)));
        let mut lines = vec![l];
        lines.extend(ruby);
        c.put_lines_fx(lines, &m, ink, &fx, Kind::Free, "free");
        y += lh + 2.0 * e + em * c.rng.uniform(0.15, 0.6);
    }
    // logo: a big Latin word in a display face, outlined, in a corner (unlabelled)
    if c.rng.chance(0.75) {
        let text = c.tg.text(c.rng, Script::Latin, Role::Heading, 12).to_uppercase();
        let px = h * c.rng.uniform(0.07, 0.14);
        let bold = c.rng.uniform(0.0, 0.06); // em
        if let Some(l) = font_line(c, Script::Latin, &[Class::Display], &text, px, bold) {
            let lw = l.b[2] - l.b[0];
            let cx = if right { w * 0.03 + lw / 2.0 } else { w * 0.97 - lw / 2.0 };
            let cy = h * c.rng.uniform(0.72, 0.86);
            let lum = c.rng.f64() as f32;
            let lfx = Fx { outline: Some(((px * 0.06).max(1.0), if lum > 0.5 { 0.05 } else { 0.95 })), ..Default::default() };
            let ang = c.rng.normal(0.0, 1.0);
            foreign_at(c, l, cx, cy, ang, lum, &lfx, "logo");
        }
    }
    // fine print at the bottom (notes, copyright lines: unlabelled)
    let px = (h * c.rng.uniform(0.018, 0.032)).max(6.0);
    let mut fy = h - px * c.rng.uniform(1.6, 2.2) * 3.0;
    let fl = if dark { c.rng.uniform(0.7, 0.95) } else { c.rng.uniform(0.05, 0.35) } as f32;
    for _ in 0..c.rng.int(1, 4) {
        let script = if c.rng.chance(0.6) && c.fonts.has_script(crate::fonts::JAPANESE) { Script::Japanese } else { Script::Latin };
        let x = w * c.rng.uniform(0.01, 0.05);
        if let Some(q) = small_text(c, script, Role::Sentence, px, x, fy, fl, "fine_print") {
            fy = q[2][1] + px * 0.4;
        } else {
            fy += px * 1.4;
        }
        if fy > h - px * 1.2 {
            break;
        }
    }
    // a price tag now and then (unlabelled: currency signs and commas are not Hunter)
    if c.rng.chance(0.3) {
        let t = format!("\u{a5}{},{:03}", c.rng.int(1, 30), c.rng.int(0, 1000));
        let px = h * c.rng.uniform(0.04, 0.08);
        if let Some(l) = font_line(c, Script::Latin, &[Class::Display, Class::Sans], &t, px, 0.0) {
            let (cx, cy) = (w * c.rng.uniform(0.6, 0.9), h * c.rng.uniform(0.75, 0.92));
            let lum = if dark { 0.95 } else { 0.08 };
            foreign_at(c, l, cx, cy, 0.0, lum, &Fx::default(), "price");
        }
    }
}

/// v7: an anime episode title card, after a title card the owner keeps as a private test
/// picture (never trained on): the title in big bold katakana with crosses between its words
/// ("A×ト×B", unlabelled), the same title in Hunter lettering with its crosses under it (80%) or
/// over it, both centred, on a plain light ground with a soft gradient or vignette (55%), a dark
/// one (20%) or a blurred scene (25%). The Hunter line takes the crosses' accent colour (a mid
/// grey in grayscale) or the title's ink; an episode number above and a Latin title with crosses
/// below now and then (both unlabelled).
pub(crate) fn title_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let pr = c.page_rect();
    let u = c.rng.f64();
    let (dark, scene) = ((0.55..0.75).contains(&u), u >= 0.75);
    let light = if scene {
        fill_bg_pub(&mut c.page, pr, c.rng, true);
        let n = c.rng.int(2, 8) as usize;
        let k = c.rng.uniform(0.0, 0.3) as f32;
        art_pub(&mut c.page, pr, c.rng, n, k, w.min(h) / 5.0);
        let s = c.rng.uniform(2.0, 9.0);
        blur(&mut c.page.d, c.w, c.h, s, true);
        c.page.d.iter().sum::<f32>() / c.page.d.len().max(1) as f32 > 0.5
    } else {
        let base = if dark { c.rng.uniform(0.02, 0.25) } else { c.rng.uniform(0.82, 1.0) } as f32;
        let cell = c.rng.uniform(150.0, 500.0);
        let n1 = value_noise(c.w, c.h, c.rng, cell);
        let a1 = c.rng.uniform(0.0, 0.04) as f32;
        let vig = if c.rng.chance(0.5) { c.rng.uniform(0.03, 0.2) } else { 0.0 } as f32;
        let (cw, ch) = (c.w, c.h);
        for y in 0..ch {
            let dy = (y as f64 / h - 0.5) * 2.0;
            for x in 0..cw {
                let dx = (x as f64 / w - 0.5) * 2.0;
                let r2 = ((dx * dx + dy * dy) / 2.0) as f32;
                let i = y * cw + x;
                let v = base + a1 * n1[i] + if dark { 0.3 * vig * r2 } else { -vig * r2 };
                c.page.d[i] = v.clamp(0.0, 1.0);
            }
        }
        !dark
    };
    // the Hunter title, neat lettering in a medium-heavy weight
    let mut st = hxh_style(c, false);
    st.wave_amp = 0.0;
    st.wave_along = 0.0;
    if c.rng.chance(0.6) {
        st.sw = c.rng.uniform(0.09, 0.17);
    }
    if !st.mono {
        st.spacing = st.spacing.min(0.7);
    }
    let title_px = (h.min(w * 0.6) * c.rng.uniform(0.09, 0.2)).max(14.0);
    let mut em = (title_px * c.rng.uniform(0.32, 0.62)).max(c.cfg.em_min);
    let room = w * c.rng.uniform(0.8, 0.94);
    let mut built = None;
    for _ in 0..5 {
        let max_len = ((room / (em * (1.0 + st.spacing.min(0.6)))).floor() as usize).clamp(3, 14);
        let l = if c.rng.chance(0.92) {
            make_cross_line_v7(c.rng, c.sampler, c.fonts, &st, em, max_len, room, c.cfg.min_stroke_px)
        } else {
            let t = word(c, 2, max_len.max(2));
            make_line_cells(c.rng, &t, &st, em, c.cfg.min_stroke_px)
        };
        if let Some(l) = l
            && l.b[2] - l.b[0] <= room
        {
            built = Some(l);
            break;
        }
        em = (em * 0.85).max(c.cfg.em_min);
    }
    let Some(l) = built else { return };
    // inks: the title's, and the Hunter line in the crosses' accent colour (55%) or the same ink
    let accent = c.rng.chance(0.55);
    let (tlum, ink) = if light {
        let t = c.rng.uniform(0.0, 0.2);
        (t, if accent { c.rng.uniform(0.3, 0.7) } else { (t + c.rng.uniform(-0.05, 0.1)).clamp(0.0, 0.3) })
    } else {
        let t = c.rng.uniform(0.82, 1.0);
        (t, if accent { c.rng.uniform(0.45, 0.8) } else { (t + c.rng.uniform(-0.1, 0.05)).clamp(0.7, 1.0) })
    };
    let (mut fx, mut tfx) = (Fx::default(), Fx::default());
    if scene && c.rng.chance(0.7) {
        let olum = if light { c.rng.uniform(0.85, 1.0) } else { c.rng.uniform(0.0, 0.2) } as f32;
        tfx.outline = Some(((title_px * c.rng.uniform(0.03, 0.08)).max(1.0), olum));
        if c.rng.chance(0.7) {
            fx.outline = Some(((em * c.rng.uniform(0.04, 0.12)).max(1.0), olum));
        }
    }
    // centred; the Hunter line under the title (or over it)
    let below = c.rng.chance(0.8);
    let cx = w / 2.0 + c.rng.normal(0.0, w * 0.015);
    let lh = l.b[3] - l.b[1];
    let gap = title_px * c.rng.uniform(0.45, 1.0) + fx.extent() + tfx.extent();
    let top = ((h - (title_px + gap + lh)) * c.rng.uniform(0.25, 0.6)).max(h * 0.04);
    let (ty, hy) = if below { (top + title_px / 2.0, top + title_px + gap + lh / 2.0) } else { (top + lh + gap + title_px / 2.0, top + lh / 2.0) };
    let label = l.raw.clone();
    let deg = c.rng.normal(0.0, 0.3);
    let (mx, my) = ((l.b[0] + l.b[2]) / 2.0, (l.b[1] + l.b[3]) / 2.0);
    let m = mul(&tr(cx, hy), &mul(&rot(deg), &tr(-mx, -my)));
    if c.put_line_fx(l, &m, ink as f32, &fx, Kind::Free, Kind::Free.name()).is_none() {
        return;
    }
    // the katakana title (unlabelled), shrunk to the page width if need be
    if c.fonts.has_script(crate::fonts::JAPANESE) {
        let text: String = kata(&label).chars().filter(|&ch| ch != ' ').collect();
        let bold = c.rng.uniform(0.0, 0.05); // em (font_line dilates by bold x px)
        let mut px = title_px;
        for _ in 0..2 {
            let Some(tl) = font_line(c, Script::Kana, &[Class::Display, Class::Sans], &text, px, bold) else { break };
            let tw = tl.b[2] - tl.b[0];
            if tw <= w * 0.95 {
                foreign_at(c, tl, cx, ty, deg, tlum as f32, &tfx, "title");
                break;
            }
            px *= w * 0.92 / tw;
        }
    }
    // an episode number above the block, a Latin title with crosses below it (unlabelled)
    let block_top = top - title_px * 0.2;
    if c.rng.chance(0.35) {
        let n = c.rng.int(1, 149);
        let (script, text) = match c.rng.below(3) {
            0 if c.fonts.has_script(crate::fonts::JAPANESE) => (Script::Japanese, format!("\u{7b2c}{n}\u{8a71}")),
            1 => (Script::Latin, format!("EPISODE {n}")),
            _ => (Script::Latin, format!("#{n:02}")),
        };
        let px = title_px * c.rng.uniform(0.25, 0.4);
        if let Some(el) = font_line(c, script, &[Class::Sans, Class::Display], &text, px, 0.0) {
            let y = block_top - px * c.rng.uniform(0.9, 1.6);
            if y > px {
                foreign_at(c, el, cx, y, deg, tlum as f32, &tfx, "episode");
            }
        }
    }
    if c.rng.chance(0.2) {
        let words = c.tg.text(c.rng, Script::Latin, Role::Heading, 28);
        let text = words.split_whitespace().take(3).collect::<Vec<_>>().join(" \u{d7} ");
        let px = title_px * c.rng.uniform(0.2, 0.32);
        if !text.is_empty()
            && let Some(el) = font_line(c, Script::Latin, &[Class::Sans, Class::Serif], &text, px, 0.0)
        {
            let y = top + title_px + gap + lh + title_px * c.rng.uniform(0.5, 0.9) + px / 2.0;
            if y < h - px && el.b[2] - el.b[0] < w * 0.9 {
                foreign_at(c, el, cx, y, deg, tlum as f32, &tfx, "subtitle");
            }
        }
    }
}

#[allow(dead_code)]
fn unused(_: &Shadow, _: &str) {
    let _ = compose;
}
