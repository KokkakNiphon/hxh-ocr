//! The "web" layout: a screenshot of a web page or document, mostly about the Hunter alphabet,
//! like the Omniglot page the detector failed on. Plain light (sometimes dark) background,
//! sharp axis-aligned text: nav bar with links and icons, breadcrumb, headings, paragraphs of
//! English / Japanese / romaji prose, link lines, lists with bullets or icons, ruled tables,
//! embedded images, footers with social-icon rows, and (unless the page is a negative) a sample
//! block of 2-6 labelled Hunter lines or a chart of Hunter glyphs with kana / romaji labels.
//! A quarter of the pages are app / UI screenshots instead: toolbars, icon + label lists and
//! icon grids. Everything goes through `Composer::put_line` / `put_group`, so the keep-clear
//! rule and the records are the same as elsewhere.

use crate::fonts::{Class, JAPANESE};
use crate::foreign::{romaji, Role, Script};
use crate::page::{compose, hxh_lines, make_font_line, make_icon_line, pick_face, stack, Composer, FStyle, Ink, Kind, LineG};
use crate::raster::IRect;
use crate::page::style_for;
use crate::text::label_of;

struct Theme {
    bg: f32,
    ink: f32,
    muted: f32,
    link: f32,
    rule: f32,
    dark: bool,
}

struct Ctx {
    t: Theme,
    body: usize,
    head: usize,
    jp: Option<usize>,
    lang: Script,
    px: f64, // body text size
    lh: f64, // body line height
    icons: Vec<usize>,
    syms: Vec<usize>,
    ymax: f64, // content bottom (the footer band, if any, is below)
}

fn rect(c: &mut Composer, x0: f64, y0: f64, x1: f64, y1: f64, lum: f32) {
    let r = IRect { x0: x0.round() as i32, y0: y0.round() as i32, x1: x1.round() as i32, y1: y1.round() as i32 };
    c.page.fill(r, lum);
}

fn frame(c: &mut Composer, x0: f64, y0: f64, x1: f64, y1: f64, lw: f64, lum: f32) {
    rect(c, x0, y0, x1, y0 + lw, lum);
    rect(c, x0, y1 - lw, x1, y1, lum);
    rect(c, x0, y0, x0 + lw, y1, lum);
    rect(c, x1 - lw, y0, x1, y1, lum);
}

/// A text line (frame: baseline at 0), optionally underlined below its lowest ink.
fn tline(c: &mut Composer, face: usize, script: Script, s: &str, px: f64, bold: f64, underline: bool) -> Option<LineG> {
    let fs = FStyle { face, script, bold, track: 0.0, shear: 0.0, caps: false };
    let mut l = make_font_line(c.rng, &c.tg, c.fonts, &fs, px, false, Role::Word, 1e9, 100_000, Some(s))?;
    if underline {
        let (y0, x0, x1) = (l.b[3] + (0.06 * px).max(1.0), l.b[0], l.b[2]);
        let y1 = y0 + (0.07 * px).max(1.0);
        if let Ink::Font { contours, runs, .. } = &mut l.ink {
            contours.push(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
            runs[0].b[3] = y1;
        }
        l.b[3] = y1;
    }
    Some(l)
}

fn width(c: &Composer, face: usize, s: &str, px: f64) -> f64 {
    c.fonts.faces[face].measure(s, px, 0.0)
}

/// Text at baseline-left (x, y); returns its box.
#[allow(clippy::too_many_arguments)]
fn text(c: &mut Composer, face: usize, script: Script, s: &str, px: f64, x: f64, y: f64, lum: f32, bold: f64, underline: bool, kind: &'static str) -> Option<[f64; 4]> {
    let l = tline(c, face, script, s, px, bold, underline)?;
    c.put_line(l, x, y, lum, Kind::Block, kind)
}

fn face_for(c: &mut Composer, cx: &Ctx, script: Script) -> Option<usize> {
    match script {
        Script::Japanese | Script::Kana => cx.jp.or_else(|| pick_face(c.rng, c.fonts, script, &[])),
        Script::Latin | Script::Url | Script::Digits if c.fonts.faces[cx.body].scripts & script.need() == script.need() => Some(cx.body),
        _ => pick_face(c.rng, c.fonts, script, &[]),
    }
}

/// Wrapped paragraph from (x0, *y) to x1; at most `n` lines.
#[allow(clippy::too_many_arguments)]
fn para(c: &mut Composer, cx: &Ctx, script: Script, px: f64, lh: f64, x0: f64, x1: f64, y: &mut f64, n: usize, lum: f32, kind: &'static str) {
    let Some(face) = face_for(c, cx, script) else { return };
    let per = if script.by_char() { (x1 - x0) / px } else { (x1 - x0) / (0.48 * px) };
    let txt = c.tg.text(c.rng, script, Role::Paragraph, (per * n as f64) as usize + 4);
    let f = &c.fonts.faces[face];
    let rows = f.wrap(&f.filter(&txt), px, x1 - x0, script.by_char());
    let indent = if c.rng.chance(0.2) { 1.5 * px } else { 0.0 };
    for (i, r) in rows.iter().take(n).enumerate() {
        if *y + lh > cx.ymax - 3.0 {
            break;
        }
        let x = if i == 0 { x0 + indent } else { x0 };
        let base = *y + 0.5 * lh + 0.32 * px;
        let r = if i == 0 && indent > 0.0 { f.fit(r, px, x1 - x, 0.0, script.by_char()) } else { r.clone() };
        text(c, face, script, &r, px, x, base, lum, 0.0, false, kind);
        *y += lh;
    }
}

fn heading(c: &mut Composer, cx: &Ctx, px: f64, x0: f64, x1: f64, y: &mut f64, kind: &'static str) {
    let script = if c.rng.chance(0.75) { cx.lang } else if cx.lang == Script::Latin { Script::Japanese } else { Script::Latin };
    let script = if matches!(script, Script::Japanese | Script::Kana) && cx.jp.is_none() { Script::Latin } else { script };
    let face = if script == Script::Latin || script == Script::Romaji { cx.head } else { face_for(c, cx, script).unwrap_or(cx.head) };
    let bold = if c.fonts.faces[face].bold { 0.0 } else { c.rng.uniform(0.0, 0.05) };
    let s = c.tg.text(c.rng, script, Role::Heading, 60);
    let s = c.fonts.faces[face].fit(&c.fonts.faces[face].filter(&s), px, x1 - x0 - 2.0 * px, 0.0, script.by_char());
    *y += px * c.rng.uniform(0.3, 0.8);
    let base = *y + px;
    let mut x = x0;
    if c.rng.chance(0.2) && !cx.icons.is_empty() {
        // a small icon before the heading
        let f = cx.icons[c.rng.below(cx.icons.len())];
        if let Some(l) = make_icon_line(c.rng, c.fonts, f, px * 0.9, 1, 0.0, false, px * 3.0) {
            let (b0, b1, w) = (l.b[1], l.b[3], l.b[2] - l.b[0]);
            let dy = base - 0.35 * px - (b0 + b1) / 2.0;
            let lx = x - l.b[0];
            if c.put_line(l, lx, dy, cx.t.ink, Kind::Block, "heading").is_some() {
                x += w + 0.5 * px;
            }
        }
    }
    text(c, face, script, &s, px, x, base, cx.t.ink, bold, false, kind);
    *y += px * 1.35;
    if c.rng.chance(0.4) {
        rect(c, x0, *y, x1, *y + 1.0, cx.t.rule);
        *y += 3.0;
    }
    *y += px * c.rng.uniform(0.2, 0.6);
}

/// A row of icons centred vertically on `mid`, starting at x; returns its right edge.
#[allow(clippy::too_many_arguments)]
fn icon_row(c: &mut Composer, face: usize, em: f64, n: usize, track: f64, x: f64, mid: f64, max_w: f64, lum: f32, kind: &'static str) -> Option<f64> {
    let hollow = c.rng.chance(0.2);
    let l = make_icon_line(c.rng, c.fonts, face, em, n, track, hollow, max_w)?;
    let dx = x - l.b[0];
    let dy = mid - (l.b[1] + l.b[3]) / 2.0;
    let w = l.b[2] - l.b[0];
    c.put_line(l, dx, dy, lum, Kind::Block, kind).map(|_| x + w)
}

fn pick_icon_face(c: &mut Composer, cx: &Ctx) -> Option<usize> {
    if !cx.icons.is_empty() && (cx.syms.is_empty() || c.rng.chance(0.75)) {
        Some(cx.icons[c.rng.below(cx.icons.len())])
    } else if !cx.syms.is_empty() {
        Some(cx.syms[c.rng.below(cx.syms.len())])
    } else {
        None
    }
}

/// Top bar: logo / site name, links, icons.
fn nav(c: &mut Composer, cx: &Ctx, y0: f64) -> f64 {
    let w = c.w as f64;
    let px = (cx.px * c.rng.uniform(1.0, 1.3)).max(8.0);
    let bar_h = (px * c.rng.uniform(2.2, 3.6)).round();
    let style = c.rng.below(3);
    let (fill, ink) = match style {
        0 => {
            let f = c.rng.uniform(0.08, 0.35) as f32;
            (f, c.rng.uniform(0.85, 1.0) as f32)
        }
        1 => (c.rng.uniform(0.86, 0.97) as f32, c.rng.uniform(0.0, 0.25) as f32),
        _ => (cx.t.bg, cx.t.ink),
    };
    rect(c, 0.0, y0, w, y0 + bar_h, fill);
    if style == 2 {
        rect(c, 0.0, y0 + bar_h - 1.0, w, y0 + bar_h, cx.t.rule);
    }
    let mid = y0 + bar_h / 2.0;
    let base = mid + 0.35 * px;
    let mut x = px;
    if c.rng.chance(0.4)
        && let Some(f) = pick_icon_face(c, cx)
            && let Some(r) = icon_row(c, f, px * 1.5, 1, 0.0, x, mid, px * 4.0, ink, "nav") {
                x = r + 0.4 * px;
            }
    let tscript = if cx.lang == Script::Japanese && cx.jp.is_some() && c.rng.chance(0.5) { Script::Japanese } else { Script::Latin };
    let title = c.tg.text(c.rng, tscript, Role::Phrase, 24);
    let tface = if tscript == Script::Japanese { cx.jp.unwrap_or(cx.head) } else { cx.head };
    let tpx = px * c.rng.uniform(1.1, 1.6);
    if let Some(b) = text(c, tface, tscript, &title, tpx, x, mid + 0.35 * tpx, ink, 0.03, false, "nav") {
        x = b[2] + 2.0 * px;
    }
    // links, some with icons; then an icon cluster at the right
    let n = c.rng.int(2, 8);
    let icon_links = c.rng.chance(0.3) && !cx.icons.is_empty();
    for _ in 0..n {
        let s = c.tg.text(c.rng, Script::Latin, Role::Word, 14);
        if x + width(c, cx.body, &s, px) + 2.0 * px > w * 0.8 {
            break;
        }
        if icon_links {
            let f = cx.icons[c.rng.below(cx.icons.len())];
            if let (Some(i), Some(t)) = (make_icon_line(c.rng, c.fonts, f, px * 1.1, 1, 0.0, false, px * 3.0), tline(c, cx.body, Script::Latin, &s, px, 0.0, false))
                && let Some(l) = compose(vec![i, t], 0.35 * px) {
                    let lx = x - l.b[0];
                    let wl = l.b[2] - l.b[0];
                    if c.put_line(l, lx, base, ink, Kind::Block, "nav").is_some() {
                        x += wl + 1.5 * px;
                    }
                    continue;
                }
        }
        if let Some(b) = text(c, cx.body, Script::Latin, &s, px, x, base, ink, 0.0, false, "nav") {
            x = b[2] + 1.5 * px;
        }
    }
    if c.rng.chance(0.7)
        && let Some(f) = pick_icon_face(c, cx) {
            let k = c.rng.int(2, 6) as usize;
            let em = px * c.rng.uniform(1.1, 1.6);
            let wmax = w - x - px;
            if wmax > em * 3.0 {
                let start = (w - px - em * 1.8 * k as f64).max(x);
                let tr = c.rng.uniform(0.4, 1.0);
                icon_row(c, f, em, k, tr, start, mid, w - start - px, ink, "nav");
            }
        }
    bar_h
}

fn list(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let n = c.rng.int(2, 7);
    let bullet = c.rng.below(5); // 0 dot, 1 dash, 2 numbers, 3 icon, 4 links (no bullet)
    let links = bullet == 4 || c.rng.chance(0.3);
    let ifont = if bullet == 3 { pick_icon_face(c, cx) } else { None };
    for i in 0..n {
        if *y + cx.lh > cx.ymax - 3.0 {
            break;
        }
        let base = *y + 0.5 * cx.lh + 0.32 * cx.px;
        let script = if c.rng.chance(0.8) { cx.lang } else { Script::Romaji };
        let Some(face) = face_for(c, cx, script) else { return };
        let role = if c.rng.chance(0.5) { Role::Link } else { Role::Phrase };
        let s = if links { c.tg.text(c.rng, Script::Latin, role, 40) } else { c.tg.text(c.rng, script, Role::Sentence, 80) };
        let s = match bullet {
            0 => format!("• {s}"),
            1 => format!("– {s}"),
            2 => format!("{}. {s}", i + 1),
            _ => s,
        };
        let s = c.fonts.faces[face].fit(&c.fonts.faces[face].filter(&s), cx.px, x1 - x0 - 3.0 * cx.px, 0.0, script.by_char());
        let lum = if links { cx.t.link } else { cx.t.ink };
        let ul = links && c.rng.chance(0.5);
        match ifont {
            Some(f) => {
                let parts = (make_icon_line(c.rng, c.fonts, f, cx.px * 1.1, 1, 0.0, false, cx.px * 3.0), tline(c, face, script, &s, cx.px, 0.0, ul));
                if let (Some(a), Some(b)) = parts
                    && let Some(l) = compose(vec![a, b], 0.5 * cx.px) {
                        let lx = x0 + cx.px - l.b[0];
                        c.put_line(l, lx, base, lum, Kind::Block, "list");
                    }
            }
            None => {
                text(c, face, script, &s, cx.px, x0 + cx.px, base, lum, 0.0, ul, "list");
            }
        }
        *y += cx.lh;
    }
    *y += cx.lh * 0.4;
}

fn links(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    for _ in 0..c.rng.int(1, 4) {
        if *y + cx.lh > cx.ymax - 3.0 {
            break;
        }
        let s = if c.rng.chance(0.6) { c.tg.text(c.rng, Script::Url, Role::Link, 80) } else { c.tg.text(c.rng, Script::Latin, Role::Link, 40) };
        let s = c.fonts.faces[cx.body].fit(&s, cx.px, x1 - x0, 0.0, false);
        let ul = c.rng.chance(0.6);
        text(c, cx.body, Script::Url, &s, cx.px, x0, *y + 0.5 * cx.lh + 0.32 * cx.px, cx.t.link, 0.0, ul, "link");
        *y += cx.lh;
    }
    *y += cx.lh * 0.3;
}

fn table(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let cols = c.rng.int(2, 6) as usize;
    let rows = c.rng.int(2, 8) as usize;
    let mut ws: Vec<f64> = (0..cols).map(|_| c.rng.uniform(0.6, 1.6)).collect();
    let tw = (x1 - x0) * c.rng.uniform(0.6, 1.0);
    let s: f64 = ws.iter().sum();
    ws.iter_mut().for_each(|v| *v *= tw / s);
    let rh = (cx.px * c.rng.uniform(1.6, 2.4)).round();
    if *y + rh * rows as f64 > cx.ymax - 4.0 {
        return;
    }
    let style = c.rng.below(3); // grid, rules, zebra
    let pad = 0.4 * cx.px;
    let scripts = [cx.lang, Script::Digits, Script::Romaji, Script::Kana, Script::Url, Script::Latin];
    let sw = [0.45, 0.2, 0.1, if cx.jp.is_some() { 0.1 } else { 0.0 }, 0.03, 0.12];
    let col_script: Vec<Script> = (0..cols).map(|_| scripts[c.rng.weighted(&sw)]).collect();
    for r in 0..rows {
        let ry = *y + r as f64 * rh;
        if style == 2 && r % 2 == 1 {
            let z = if cx.t.dark { cx.t.bg + 0.06 } else { cx.t.bg - 0.05 };
            rect(c, x0, ry, x0 + tw, ry + rh, z);
        }
        if style == 1 || style == 0 {
            rect(c, x0, ry, x0 + tw, ry + 1.0, cx.t.rule);
        }
        let mut x = x0;
        for (k, &cw) in ws.iter().enumerate() {
            let script = if r == 0 { cx.lang } else { col_script[k] };
            if let Some(face) = face_for(c, cx, script) {
                let role = if r == 0 || c.rng.chance(0.5) { Role::Word } else { Role::Label };
                let s = c.tg.text(c.rng, script, role, 30);
                let f = &c.fonts.faces[face];
                let s = f.fit(&f.filter(&s), cx.px, cw - 2.0 * pad, 0.0, script.by_char());
                if !s.is_empty() {
                    let bold = if r == 0 { 0.04 } else { 0.0 };
                    text(c, face, script, &s, cx.px, x + pad, ry + 0.5 * rh + 0.32 * cx.px, cx.t.ink, bold, false, "table");
                }
            }
            x += cw;
        }
    }
    let th = rh * rows as f64;
    if style == 0 {
        let mut x = x0;
        rect(c, x0, *y + th, x0 + tw, *y + th + 1.0, cx.t.rule);
        for &cw in ws.iter() {
            rect(c, x, *y, x + 1.0, *y + th, cx.t.rule);
            x += cw;
        }
        rect(c, x0 + tw, *y, x0 + tw + 1.0, *y + th + 1.0, cx.t.rule);
    }
    *y += th + cx.lh * 0.6;
}

/// Labelled Hunter sample: 2-6 lines (sometimes in a framed image), then a transliteration.
fn sample(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let cfg = c.cfg;
    let em = c.rng.log_uniform(cfg.em_min.max(12.0), 30.0).min((x1 - x0) / 5.0).max(cfg.em_min);
    let vertical = c.rng.chance(0.08);
    let n = c.rng.int(2, 7) as usize;
    let boxed = c.rng.chance(0.6);
    let pad = if boxed { em * c.rng.uniform(0.4, 1.0) } else { 0.0 };
    let ext = if vertical { em * c.rng.uniform(5.0, 12.0) } else { (x1 - x0 - 2.0 * pad) * c.rng.uniform(0.5, 1.0) };
    let st = style_for(c.rng, c.cfg);
    let max_len = c.rng.int(3, 17) as usize;
    let lines = hxh_lines(c.rng, c.sampler, cfg, &st, em, vertical, n, max_len, ext);
    let align = if c.rng.chance(0.7) { b'l' } else { b'c' };
    let gap = em * c.rng.uniform(0.2, 0.8);
    let Some(g) = stack(lines, vertical, em, gap, align, cfg.keep_margin) else { return };
    let raws: Vec<String> = g.lines.iter().map(|l| label_of(&l.raw)).collect();
    *y += 0.45 * em.max(cx.px) + 2.0;
    if *y + g.h + 2.0 * pad > cx.ymax - 4.0 {
        return;
    }
    let gx = if align == b'c' { (x0 + x1) / 2.0 - g.w / 2.0 } else { x0 + pad };
    let mut ink = cx.t.ink;
    if boxed {
        let light = c.rng.chance(if cx.t.dark { 0.4 } else { 0.85 });
        let fill = if light { c.rng.uniform(0.85, 1.0) as f32 } else { c.rng.uniform(0.0, 0.25) as f32 };
        ink = if light { c.rng.uniform(0.0, 0.2) as f32 } else { c.rng.uniform(0.85, 1.0) as f32 };
        rect(c, gx - pad, *y, gx + g.w + pad, *y + g.h + 2.0 * pad, fill);
        if c.rng.chance(0.6) {
            frame(c, gx - pad, *y, gx + g.w + pad, *y + g.h + 2.0 * pad, 1.0, cx.t.rule);
        }
    }
    c.put_group(&g, gx, *y + pad, ink, Kind::Block, "sample");
    *y += g.h + 2.0 * pad + 0.45 * em.max(cx.px) + 2.0;
    if c.rng.chance(0.7) {
        // transliteration / translation under the sample
        let lat = cx.body;
        let px = cx.px * c.rng.uniform(0.85, 1.0);
        let lh = px * 1.4;
        let rom = raws.iter().map(|r| romaji(r)).collect::<Vec<_>>().join(" ");
        let kana = raws.join(" ");
        for (s, script) in [(rom, Script::Romaji), (kana, Script::Kana)] {
            if *y + lh > cx.ymax - 3.0 || (script == Script::Kana && (cx.jp.is_none() || c.rng.chance(0.5))) {
                continue;
            }
            let face = if script == Script::Kana { cx.jp.unwrap_or(lat) } else if c.fonts.faces[lat].scripts & script.need() == script.need() { lat } else { face_for(c, cx, script).unwrap_or(lat) };
            let f = &c.fonts.faces[face];
            for r in f.wrap(&f.filter(&s), px, x1 - x0, script.by_char()).into_iter().take(3) {
                if *y + lh > cx.ymax - 3.0 {
                    break;
                }
                text(c, face, script, &r, px, x0, *y + 0.5 * lh + 0.32 * px, cx.t.muted, 0.0, false, "caption");
                *y += lh;
            }
        }
        *y += lh * 0.4;
    }
}

/// Chart of Hunter glyphs (labelled cells) with kana and / or romaji labels under each.
fn chart(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let cfg = c.cfg;
    let cols = c.rng.int(4, 11) as usize;
    let rows = c.rng.int(1, 5) as usize;
    let cw = (x1 - x0) / cols as f64;
    let em = (cw * c.rng.uniform(0.35, 0.55)).clamp(cfg.em_min, 34.0);
    if cw < em * 1.5 {
        return;
    }
    let lpx = (em * c.rng.uniform(0.45, 0.7)).max(cfg.foreign_em_min);
    let show_kana = cx.jp.is_some() && c.rng.chance(0.6);
    let show_rom = !show_kana || c.rng.chance(0.6);
    let pad = 0.3 * em;
    let lab_h = lpx * 1.35 * (show_kana as usize + show_rom as usize) as f64;
    let ch = pad + em * 1.2 + 0.55 * em.max(lpx) + 2.0 + lab_h + pad;
    *y += 0.45 * em + 2.0;
    let grid = c.rng.chance(0.6);
    let st = style_for(c.rng, c.cfg);
    let mut rows_drawn = 0;
    for r in 0..rows {
        let cy = *y + r as f64 * ch;
        if cy + ch > cx.ymax - 4.0 {
            break;
        }
        rows_drawn += 1;
        if grid {
            rect(c, x0, cy, x1, cy + 1.0, cx.t.rule);
        }
        for k in 0..cols {
            let cx0 = x0 + k as f64 * cw;
            if grid {
                rect(c, cx0, cy, cx0 + 1.0, cy + ch, cx.t.rule);
            }
            let ml = if c.rng.chance(0.8) { 1 } else { 2 };
            let lines = hxh_lines(c.rng, c.sampler, cfg, &st, em, false, 1, ml, cw - 0.4 * em);
            let Some(g) = stack(lines, false, em, 2.0, b'c', cfg.keep_margin) else { continue };
            let raw = label_of(&g.lines[0].raw);
            let gx = cx0 + (cw - g.w) / 2.0;
            if c.put_group(&g, gx, cy + pad, cx.t.ink, Kind::Cell, "chart") == 0 {
                continue;
            }
            let mut ly = cy + pad + g.h + 0.55 * em.max(lpx) + 2.0;
            for (s, script, show) in [(raw.clone(), Script::Kana, show_kana), (romaji(&raw), Script::Romaji, show_rom)] {
                if !show {
                    continue;
                }
                let face = if script == Script::Kana { cx.jp.unwrap_or(cx.body) } else { cx.body };
                let wdt = width(c, face, &s, lpx);
                text(c, face, script, &s, lpx, cx0 + (cw - wdt) / 2.0, ly + lpx * 0.95, cx.t.muted, 0.0, false, "chart");
                ly += lpx * 1.35;
            }
        }
        if grid {
            rect(c, x1 - 1.0, cy, x1, cy + ch, cx.t.rule);
        }
    }
    let used = rows_drawn as f64 * ch;
    if grid {
        rect(c, x0, *y + used, x1, *y + used + 1.0, cx.t.rule);
    }
    *y += used + 0.5 * em + 2.0;
}

fn image(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let iw = (x1 - x0) * c.rng.uniform(0.3, 0.9);
    let ih = iw * c.rng.uniform(0.3, 0.8);
    if *y + ih + cx.lh * 2.0 > cx.ymax - 4.0 || iw < 24.0 || ih < 16.0 {
        return;
    }
    let ix = if c.rng.chance(0.5) { x0 } else { (x0 + x1 - iw) / 2.0 };
    let r = IRect { x0: ix as i32, y0: *y as i32, x1: (ix + iw) as i32, y1: (*y + ih) as i32 };
    // a picture: smooth tones and shapes, no text
    let l1 = c.rng.f64() as f32;
    let l2 = c.rng.f64() as f32;
    for yy in r.y0..r.y1 {
        let t = (yy - r.y0) as f32 / r.h().max(1) as f32;
        c.page.fill(IRect { x0: r.x0, y0: yy, x1: r.x1, y1: yy + 1 }, l1 + (l2 - l1) * t);
    }
    for _ in 0..c.rng.int(2, 7) {
        let (a, b) = (c.rng.uniform(r.x0 as f64, r.x1 as f64), c.rng.uniform(r.y0 as f64, r.y1 as f64));
        let s = c.rng.uniform(0.1, 0.4) * ih;
        let col = c.rng.f64() as f32;
        rect(c, (a - s).max(r.x0 as f64), (b - s).max(r.y0 as f64), (a + s).min(r.x1 as f64), (b + s).min(r.y1 as f64), col);
    }
    *y += ih + 0.3 * cx.lh;
    if c.rng.chance(0.6) {
        let px = cx.px * 0.85;
        let s = c.tg.text(c.rng, Script::Latin, Role::Sentence, 70);
        let s = c.fonts.faces[cx.body].fit(&s, px, iw, 0.0, false);
        text(c, cx.body, Script::Latin, &s, px, ix, *y + px, cx.t.muted, 0.0, false, "caption");
        *y += px * 1.5;
    }
    *y += 0.3 * cx.lh;
}

/// Grid of icon tiles with labels (features, app launcher, file view).
fn icon_grid(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let Some(f) = pick_icon_face(c, cx) else { return };
    let cols = c.rng.int(3, 9) as usize;
    let cw = (x1 - x0) / cols as f64;
    let em = (cw * c.rng.uniform(0.25, 0.55)).clamp(12.0, 64.0);
    let lpx = (cx.px * c.rng.uniform(0.8, 1.0)).max(c.cfg.foreign_em_min);
    let ch = em * 1.3 + lpx * 2.2;
    let rows = c.rng.int(1, 4) as usize;
    let labels = c.rng.chance(0.8);
    for r in 0..rows {
        let ry = *y + r as f64 * ch;
        if ry + ch > cx.ymax - 4.0 {
            break;
        }
        for k in 0..cols {
            let tx = x0 + k as f64 * cw;
            let hollow = c.rng.chance(0.2);
            let Some(l) = make_icon_line(c.rng, c.fonts, f, em, 1, 0.0, hollow, cw) else { continue };
            let lw = l.b[2] - l.b[0];
            let (dx, dy) = (tx + (cw - lw) / 2.0 - l.b[0], ry + 0.15 * em - l.b[1]);
            let iy = l.b[3] + dy;
            c.put_line(l, dx, dy, cx.t.ink, Kind::Block, "icon_grid");
            if labels {
                let ls = if cx.jp.is_some() && cx.lang == Script::Japanese && c.rng.chance(0.3) { Script::Japanese } else { Script::Latin };
                let s = c.tg.text(c.rng, ls, Role::Word, 14);
                let face = if ls == Script::Japanese { cx.jp.unwrap_or(cx.body) } else { cx.body };
                let s = c.fonts.faces[face].fit(&c.fonts.faces[face].filter(&s), lpx, cw - 4.0, 0.0, false);
                let wdt = width(c, face, &s, lpx);
                text(c, face, ls, &s, lpx, tx + (cw - wdt) / 2.0, iy + lpx * 1.3, cx.t.ink, 0.0, false, "icon_grid");
            }
        }
    }
    *y += rows as f64 * ch + cx.lh * 0.5;
}

/// Toolbar: a row of 3-16 icons, sometimes in buttons, sometimes with a search box.
fn toolbar(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: &mut f64) {
    let Some(f) = pick_icon_face(c, cx) else { return };
    let em = c.rng.log_uniform(10.0, 28.0);
    let bh = em * c.rng.uniform(1.6, 2.4);
    if *y + bh > cx.ymax - 4.0 {
        return;
    }
    if c.rng.chance(0.5) {
        let z = if cx.t.dark { cx.t.bg + 0.08 } else { cx.t.bg - 0.06 };
        rect(c, x0, *y, x1, *y + bh, z);
    }
    let mid = *y + bh / 2.0;
    let n = c.rng.int(3, 17) as usize;
    let track = c.rng.uniform(0.4, 1.4);
    let ink = if c.rng.chance(0.8) { cx.t.ink } else { cx.t.muted };
    let mut x = x0 + 0.5 * em;
    if c.rng.chance(0.3) {
        // a button with icon + label
        let s = c.tg.text(c.rng, Script::Latin, Role::Word, 12);
        let parts = (make_icon_line(c.rng, c.fonts, f, em, 1, 0.0, false, em * 3.0), tline(c, cx.body, Script::Latin, &s, em * 0.75, 0.0, false));
        if let (Some(a), Some(b)) = parts
            && let Some(l) = compose(vec![a, b], 0.4 * em) {
                let (lw, lb1, lb3, lb0) = (l.b[2] - l.b[0], l.b[1], l.b[3], l.b[0]);
                let bx1 = x + lw + em;
                frame(c, x - 0.3 * em, *y + 0.15 * bh, bx1, *y + 0.85 * bh, 1.0, cx.t.rule);
                if c.put_line(l, x + 0.2 * em - lb0, mid - (lb1 + lb3) / 2.0, ink, Kind::Block, "toolbar").is_some() {
                    x = bx1 + em;
                }
            }
    }
    icon_row(c, f, em, n, track, x, mid, x1 - x - 0.5 * em, ink, "toolbar");
    *y += bh + cx.lh * 0.4;
}

/// Rating row: stars + score text.
fn rating(c: &mut Composer, cx: &Ctx, x0: f64, y: &mut f64) {
    let Some(&f) = cx.syms.iter().find(|&&i| c.fonts.faces[i].icons.iter().any(|g| g.1 == Some('★'))) else { return };
    let face = &c.fonts.faces[f];
    let s: String = {
        let k = c.rng.int(1, 6) as usize;
        (0..5).map(|i| if i < k { '★' } else { '☆' }).collect()
    };
    if !face.covers(&s) || *y + cx.lh > cx.ymax - 3.0 {
        return;
    }
    let fs = FStyle { face: f, script: Script::Icon, bold: 0.0, track: 0.1, shear: 0.0, caps: false };
    let stars = make_font_line(c.rng, &c.tg, c.fonts, &fs, cx.px * 1.2, false, Role::Word, 1e9, 100, Some(&s));
    let score = format!("{}.{} ({})", c.rng.int(1, 5), c.rng.int(0, 10), c.rng.int(3, 9999));
    let t = tline(c, cx.body, Script::Digits, &score, cx.px, 0.0, false);
    if let (Some(a), Some(b)) = (stars, t)
        && let Some(l) = compose(vec![a, b], 0.5 * cx.px) {
            let lx = x0 - l.b[0];
            c.put_line(l, lx, *y + 0.5 * cx.lh + 0.32 * cx.px, cx.t.ink, Kind::Block, "rating");
        }
    *y += cx.lh * 1.3;
}

fn footer(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y: f64) {
    let h = c.h as f64;
    if y > h - 3.0 * cx.lh {
        return;
    }
    let mut y = y + cx.lh * 0.5;
    if c.rng.chance(0.5) {
        let z = if cx.t.dark { cx.t.bg + 0.06 } else { cx.t.bg - 0.07 };
        rect(c, 0.0, y - 0.3 * cx.lh, c.w as f64, h, z);
    } else {
        rect(c, x0, y - 0.3 * cx.lh, x1, y - 0.3 * cx.lh + 1.0, cx.t.rule);
    }
    let px = (cx.px * 0.85).max(c.cfg.foreign_em_min);
    let lh = px * 1.6;
    // social icons
    if c.rng.chance(0.6)
        && let Some(f) = pick_icon_face(c, cx) {
            let em = c.rng.log_uniform(12.0, 26.0);
            if y + em * 1.5 < h - 3.0 {
                let (k, tr) = (c.rng.int(3, 7) as usize, c.rng.uniform(0.5, 1.2));
                icon_row(c, f, em, k, tr, x0, y + 0.75 * em, x1 - x0, cx.t.muted, "footer");
                y += em * 1.6;
            }
        }
    while y + lh < h - 3.0 {
        let s = match c.rng.below(3) {
            0 => c.tg.text(c.rng, Script::Latin, Role::Label, 60),
            1 => (0..c.rng.int(2, 6)).map(|_| c.tg.text(c.rng, Script::Latin, Role::Link, 20)).collect::<Vec<_>>().join(if c.rng.chance(0.5) { " · " } else { " | " }),
            _ => format!("© {} {}. All rights reserved.", c.rng.int(2000, 2027), c.tg.text(c.rng, Script::Latin, Role::Phrase, 20)),
        };
        let s = c.fonts.faces[cx.body].fit(&c.fonts.faces[cx.body].filter(&s), px, x1 - x0, 0.0, false);
        text(c, cx.body, Script::Latin, &s, px, x0, y + 0.5 * lh + 0.32 * px, cx.t.muted, 0.0, false, "footer");
        y += lh;
        if c.rng.chance(0.4) {
            break;
        }
    }
}

fn sidebar(c: &mut Composer, cx: &Ctx, x0: f64, x1: f64, y0: f64) {
    let z = if cx.t.dark { cx.t.bg + 0.05 } else { cx.t.bg - 0.04 };
    rect(c, x0, y0, x1, cx.ymax, z);
    let px = cx.px * c.rng.uniform(0.85, 1.0);
    let lh = px * c.rng.uniform(1.5, 2.2);
    let pad = px;
    let mut y = y0 + pad;
    let title = c.tg.text(c.rng, if cx.lang == Script::Japanese { Script::Japanese } else { Script::Latin }, Role::Heading, 20);
    let tface = if title.chars().any(|ch| ch as u32 > 0x2FFF) { cx.jp.unwrap_or(cx.head) } else { cx.head };
    let t = c.fonts.faces[tface].fit(&title, px * 1.1, x1 - x0 - 2.0 * pad, 0.0, true);
    text(c, tface, cx.lang, &t, px * 1.1, x0 + pad, y + px, cx.t.ink, 0.03, false, "sidebar");
    y += lh * 1.2;
    let icons = c.rng.chance(0.4);
    let numbered = !icons && c.rng.chance(0.4);
    let ifont = if icons { pick_icon_face(c, cx) } else { None };
    let mut i = 0;
    while y + lh < cx.ymax - 3.0 && i < 30 {
        i += 1;
        let script = if c.rng.chance(0.85) { Script::Latin } else { cx.lang };
        let Some(face) = face_for(c, cx, script) else { break };
        let role = if c.rng.chance(0.5) { Role::Phrase } else { Role::Link };
        let s = c.tg.text(c.rng, script, role, 30);
        let s = if numbered { format!("{}{} {s}", i, if c.rng.chance(0.5) { "." } else { "" }) } else { s };
        let f = &c.fonts.faces[face];
        let s = f.fit(&f.filter(&s), px, x1 - x0 - 2.0 * pad - if icons { 1.6 * px } else { 0.0 }, 0.0, script.by_char());
        let base = y + 0.5 * lh + 0.32 * px;
        match ifont {
            Some(ff) => {
                let parts = (make_icon_line(c.rng, c.fonts, ff, px * 1.15, 1, 0.0, false, px * 3.0), tline(c, face, script, &s, px, 0.0, false));
                if let (Some(a), Some(b)) = parts
                    && let Some(l) = compose(vec![a, b], 0.5 * px) {
                        let lx = x0 + pad - l.b[0];
                        c.put_line(l, lx, base, cx.t.link, Kind::Block, "sidebar");
                    }
            }
            None => {
                text(c, face, script, &s, px, x0 + pad, base, cx.t.link, 0.0, false, "sidebar");
            }
        }
        y += lh;
    }
}

/// Compose a web / document / app screenshot.
pub(crate) fn web_page(c: &mut Composer) {
    let (w, h) = (c.w as f64, c.h as f64);
    let dark = c.rng.chance(0.12);
    let t = if dark {
        let bg = c.rng.uniform(0.05, 0.18) as f32;
        Theme { bg, ink: c.rng.uniform(0.8, 0.96) as f32, muted: c.rng.uniform(0.55, 0.75) as f32, link: c.rng.uniform(0.6, 0.85) as f32, rule: bg + 0.2, dark }
    } else {
        let bg = c.rng.uniform(0.93, 1.0) as f32;
        Theme { bg, ink: c.rng.uniform(0.0, 0.2) as f32, muted: c.rng.uniform(0.3, 0.5) as f32, link: c.rng.uniform(0.2, 0.5) as f32, rule: c.rng.uniform(0.6, 0.85) as f32, dark }
    };
    let pr = c.page_rect();
    c.page.fill(pr, t.bg);
    let fonts = c.fonts;
    let has_jp = fonts.has_script(JAPANESE);
    let lang = match c.rng.weighted(&[0.65, if has_jp { 0.3 } else { 0.0 }, 0.05]) {
        0 => Script::Latin,
        1 => Script::Japanese,
        _ => [Script::Korean, Script::Greek, Script::Cyrillic][c.rng.below(3)],
    };
    let lang = if fonts.has_script(lang.need()) { lang } else { Script::Latin };
    let pref: &[Class] = if c.rng.chance(0.3) { &[Class::Serif] } else { &[Class::Sans] };
    let Some(body) = pick_face(c.rng, fonts, Script::Latin, pref) else { return };
    let head = if c.rng.chance(0.5) { body } else { pick_face(c.rng, fonts, Script::Latin, &[Class::Sans, Class::Serif]).unwrap_or(body) };
    let jp = if has_jp { pick_face(c.rng, fonts, Script::Japanese, &[]) } else { None };
    let px = match c.rng.weighted(&[0.25, 0.35, 0.3, 0.1]) {
        0 => c.rng.uniform(6.0, 8.0),
        1 => c.rng.uniform(8.0, 11.0),
        2 => c.rng.uniform(11.0, 15.0),
        _ => c.rng.uniform(15.0, 19.0),
    };
    let lh = px * c.rng.uniform(1.25, 1.6);
    let (icons, syms) = fonts.icon_faces();
    let foot = c.rng.chance(0.5);
    let ymax = if foot { h - (lh * 4.5 + 36.0).min(h * 0.3) } else { h };
    let cx = Ctx { t, body, head, jp, lang, px, lh, icons, syms, ymax };
    let ui = c.rng.chance(c.cfg.ui_prob);
    let mut y = 0.0;
    if c.rng.chance(if ui { 0.8 } else { 0.55 }) {
        y += nav(c, &cx, 0.0);
    }
    // columns: optional sidebar
    let (mut x0, mut x1) = (w * c.rng.uniform(0.03, 0.15), w * (1.0 - c.rng.uniform(0.03, 0.15)));
    if c.rng.chance(if ui { 0.6 } else { 0.3 }) && w > 400.0 {
        let sw = w * c.rng.uniform(0.18, 0.28);
        if c.rng.chance(0.7) {
            sidebar(c, &cx, 0.0, sw, y);
            x0 = sw + px * 1.5;
        } else {
            sidebar(c, &cx, w - sw, w, y);
            x1 = w - sw - px * 1.5;
        }
    }
    if x1 - x0 < 8.0 * px {
        return;
    }
    y += px * c.rng.uniform(0.8, 2.0);
    if c.rng.chance(0.4) {
        // breadcrumb
        let s = (0..c.rng.int(2, 5)).map(|_| c.tg.text(c.rng, Script::Latin, Role::Word, 14)).collect::<Vec<_>>().join(c.rng.pick(&[" › ", " > ", " / ", " » "]));
        let s = fonts.faces[body].fit(&s, px * 0.9, x1 - x0, 0.0, false);
        text(c, body, Script::Latin, &s, px * 0.9, x0, y + px, cx.t.link, 0.0, false, "breadcrumb");
        y += px * 1.8;
    }
    let hpx = px * c.rng.uniform(1.6, 2.8);
    heading(c, &cx, hpx, x0, x1, &mut y, "heading");
    if c.rng.chance(0.4) {
        let s = format!("{} {} · {}", c.tg.text(c.rng, Script::Digits, Role::Label, 20), c.tg.text(c.rng, Script::Latin, Role::Phrase, 20), c.tg.text(c.rng, Script::Latin, Role::Link, 20));
        let s = fonts.faces[body].fit(&s, px * 0.85, x1 - x0, 0.0, false);
        text(c, body, Script::Latin, &s, px * 0.85, x0, y + px, cx.t.muted, 0.0, false, "meta");
        y += px * 1.8;
    }
    // the Hunter content goes somewhere in the first few elements
    let mut hxh_at = if c.neg { usize::MAX } else { c.rng.int(0, 4) as usize };
    let mut k = 0;
    while y < ymax - 2.0 * lh && k < 60 {
        if k == hxh_at {
            if c.rng.chance(0.65) { sample(c, &cx, x0, x1, &mut y) } else { chart(c, &cx, x0, x1, &mut y) }
            if c.rng.chance(0.3) {
                hxh_at = k + c.rng.int(2, 6) as usize; // a second sample further down
            }
            k += 1;
            continue;
        }
        let wts = if ui { [0.12, 0.05, 0.08, 0.04, 0.06, 0.02, 0.3, 0.25, 0.05, 0.03] } else { [0.38, 0.12, 0.1, 0.06, 0.08, 0.05, 0.06, 0.08, 0.04, 0.03] };
        match c.rng.weighted(&wts) {
            0 => {
                let script = match c.rng.weighted(&[0.75, 0.1, 0.1, 0.05]) {
                    0 => lang,
                    1 => Script::Romaji,
                    2 if jp.is_some() => Script::Japanese,
                    _ => Script::Latin,
                };
                let n = c.rng.int(2, 9) as usize;
                para(c, &cx, script, px, lh, x0, x1, &mut y, n, cx.t.ink, "para");
                y += lh * c.rng.uniform(0.3, 1.0);
            }
            1 => {
                let hpx = px * c.rng.uniform(1.15, 1.7);
                heading(c, &cx, hpx, x0, x1, &mut y, "heading")
            }
            2 => list(c, &cx, x0, x1, &mut y),
            3 => links(c, &cx, x0, x1, &mut y),
            4 => table(c, &cx, x0, x1, &mut y),
            5 => image(c, &cx, x0, x1, &mut y),
            6 => icon_grid(c, &cx, x0, x1, &mut y),
            7 => toolbar(c, &cx, x0, x1, &mut y),
            8 => rating(c, &cx, x0, &mut y),
            _ => {
                rect(c, x0, y + lh * 0.5, x1, y + lh * 0.5 + 1.0, cx.t.rule);
                y += lh;
            }
        }
        k += 1;
    }
    if foot {
        footer(c, &cx, x0, x1, ymax);
    }
}
