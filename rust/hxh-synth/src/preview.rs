//! Contact sheets for a generated split: pages with their quads drawn (green = horizontal,
//! magenta = vertical, red dot = first corner, i.e. reading-orientation top-left; unlabelled
//! foreign text orange, icons cyan, thinner), and DBNet targets for a few pages.

use std::path::Path as FsPath;

use serde::Deserialize;
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::codec::decode_jpeg_gray;
use crate::raster::resize_area;
use crate::targets::{rasterize_targets, TargetParams};

#[derive(Deserialize, Clone)]
pub struct LineRec {
    pub quad: [[f64; 2]; 4],
    pub text: String,
    pub orient: String,
    #[serde(default)]
    pub kind: String,
}

#[derive(Deserialize, Clone)]
pub struct ForeignRec {
    pub quad: [[f64; 2]; 4],
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub script: String,
}

#[derive(Deserialize, Clone)]
pub struct PageRec {
    pub image: String,
    pub width: usize,
    pub height: usize,
    pub lines: Vec<LineRec>,
    #[serde(default)]
    pub foreign: Vec<ForeignRec>,
}

pub fn read_labels(split: &FsPath, limit: usize) -> Result<Vec<PageRec>, String> {
    let p = split.join("labels.jsonl");
    let s = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    s.lines().filter(|l| !l.trim().is_empty()).take(limit).map(|l| serde_json::from_str(l).map_err(|e| e.to_string())).collect()
}

pub fn load_gray(split: &FsPath, rec: &PageRec) -> Result<Vec<u8>, String> {
    let p = split.join(&rec.image);
    let data = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let (px, w, h) = decode_jpeg_gray(&data).ok_or_else(|| format!("{}: bad jpeg", p.display()))?;
    if w != rec.width || h != rec.height {
        return Err(format!("{}: size mismatch", p.display()));
    }
    Ok(px)
}

fn scaled(px: &[u8], w: usize, h: usize, s: f64) -> (Vec<u8>, usize, usize) {
    let (dw, dh) = (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1));
    let f: Vec<f32> = px.iter().map(|&v| v as f32).collect();
    let r = resize_area(&f, w, h, dw, dh);
    (r.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect(), dw, dh)
}

fn put_rgb(pm: &mut Pixmap, x0: usize, y0: usize, w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 3]) {
    let pw = pm.width() as usize;
    let ph = pm.height() as usize;
    let d = pm.data_mut();
    for y in 0..h.min(ph.saturating_sub(y0)) {
        for x in 0..w.min(pw.saturating_sub(x0)) {
            let c = f(x, y);
            let i = ((y0 + y) * pw + x0 + x) * 4;
            d[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
}

fn draw_foreign(pm: &mut Pixmap, ox: f64, oy: f64, s: f64, items: &[ForeignRec]) {
    let mut paint = Paint { anti_alias: true, ..Default::default() };
    let st = Stroke { width: 1.0, ..Default::default() };
    for f in items {
        let mut pb = PathBuilder::new();
        for (i, p) in f.quad.iter().enumerate() {
            let (x, y) = ((ox + (p[0] + 0.5) * s) as f32, (oy + (p[1] + 0.5) * s) as f32);
            if i == 0 { pb.move_to(x, y) } else { pb.line_to(x, y) }
        }
        pb.close();
        if let Some(path) = pb.finish() {
            if f.script == "icon" {
                paint.set_color_rgba8(0, 170, 230, 255);
            } else {
                paint.set_color_rgba8(255, 130, 0, 255);
            }
            pm.stroke_path(&path, &paint, &st, Transform::identity(), None);
        }
    }
}

fn draw_quads(pm: &mut Pixmap, ox: f64, oy: f64, s: f64, lines: &[LineRec]) {
    let mut paint = Paint { anti_alias: true, ..Default::default() };
    let st = Stroke { width: 1.5, ..Default::default() };
    for l in lines {
        let q: Vec<[f32; 2]> = l.quad.iter().map(|p| [(ox + (p[0] + 0.5) * s) as f32, (oy + (p[1] + 0.5) * s) as f32]).collect();
        let mut pb = PathBuilder::new();
        pb.move_to(q[0][0], q[0][1]);
        for p in &q[1..] {
            pb.line_to(p[0], p[1]);
        }
        pb.close();
        if let Some(path) = pb.finish() {
            if l.orient == "v" {
                paint.set_color_rgba8(220, 0, 220, 255);
            } else {
                paint.set_color_rgba8(0, 190, 0, 255);
            }
            pm.stroke_path(&path, &paint, &st, Transform::identity(), None);
        }
        if let Some(dot) = PathBuilder::from_circle(q[0][0], q[0][1], 2.5) {
            paint.set_color_rgba8(230, 0, 0, 255);
            pm.fill_path(&dot, &paint, FillRule::Winding, Transform::identity(), None);
        }
    }
}

/// Grid of `n` pages from `split`, each fitted into a `cell` x `cell` box, quads drawn.
pub fn contact_sheet(split: &FsPath, out: &FsPath, n: usize, cols: usize, cell: usize) -> Result<(), String> {
    let recs = read_labels(split, n)?;
    if recs.is_empty() {
        return Err("no pages".into());
    }
    let gap = 8;
    let rows = recs.len().div_ceil(cols);
    let mut pm = Pixmap::new(((cell + gap) * cols + gap) as u32, ((cell + gap) * rows + gap) as u32).ok_or("sheet too large")?;
    pm.fill(tiny_skia::Color::from_rgba8(235, 235, 235, 255));
    for (i, rec) in recs.iter().enumerate() {
        let px = load_gray(split, rec)?;
        let s = (cell as f64 / rec.width as f64).min(cell as f64 / rec.height as f64);
        let (t, tw, th) = scaled(&px, rec.width, rec.height, s);
        let (x0, y0) = (gap + (i % cols) * (cell + gap), gap + (i / cols) * (cell + gap));
        put_rgb(&mut pm, x0, y0, tw, th, |x, y| {
            let v = t[y * tw + x];
            [v, v, v]
        });
        draw_foreign(&mut pm, x0 as f64, y0 as f64, tw as f64 / rec.width as f64, &rec.foreign);
        draw_quads(&mut pm, x0 as f64, y0 as f64, tw as f64 / rec.width as f64, &rec.lines);
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    pm.save_png(out).map_err(|e| e.to_string())
}

/// For `n` pages: image with quads | prob kernels (ignored lines red) | threshold map |
/// threshold mask (vertical kernels magenta), all at `long` px on the long side.
pub fn targets_sheet(split: &FsPath, out: &FsPath, n: usize, long: usize) -> Result<(), String> {
    let recs = read_labels(split, n)?;
    let gap = 8;
    let mut rows = Vec::new();
    for rec in &recs {
        let px = load_gray(split, rec)?;
        let s = long as f64 / rec.width.max(rec.height) as f64;
        let (img, w, h) = scaled(&px, rec.width, rec.height, s);
        let lines: Vec<([[f64; 2]; 4], bool)> = rec
            .lines
            .iter()
            .map(|l| (l.quad.map(|p| [(p[0] + 0.5) * s - 0.5, (p[1] + 0.5) * s - 0.5]), l.orient == "v"))
            .collect();
        let t = rasterize_targets(&lines, w, h, &TargetParams::default());
        let mut scaled_lines = rec.lines.clone();
        for (l, (q, _)) in scaled_lines.iter_mut().zip(&lines) {
            l.quad = *q;
        }
        rows.push((img, w, h, t, scaled_lines));
    }
    let width = 4 * (long + gap) + gap;
    let height: usize = rows.iter().map(|r| r.2 + gap).sum::<usize>() + gap;
    let mut pm = Pixmap::new(width as u32, height as u32).ok_or("sheet too large")?;
    pm.fill(tiny_skia::Color::from_rgba8(235, 235, 235, 255));
    let mut y0 = gap;
    for (img, w, h, t, lines) in &rows {
        let (w, h) = (*w, *h);
        let x = |k: usize| gap + k * (long + gap);
        put_rgb(&mut pm, x(0), y0, w, h, |xx, yy| {
            let v = img[yy * w + xx];
            [v, v, v]
        });
        draw_quads(&mut pm, x(0) as f64, y0 as f64, 1.0, lines);
        put_rgb(&mut pm, x(1), y0, w, h, |xx, yy| {
            let i = yy * w + xx;
            let base = (img[i] as f32 * 0.3) as u8;
            if t.prob[i] > 0.0 {
                [255, 255, 255]
            } else if t.prob_mask[i] == 0.0 {
                [200, 30, 30]
            } else {
                [base, base, base]
            }
        });
        put_rgb(&mut pm, x(2), y0, w, h, |xx, yy| {
            let v = (((t.thresh[yy * w + xx] - 0.3) / 0.4).clamp(0.0, 1.0) * 255.0) as u8;
            [v, v, v]
        });
        put_rgb(&mut pm, x(3), y0, w, h, |xx, yy| {
            let i = yy * w + xx;
            if t.vert[i] > 0.0 {
                [220, 0, 220]
            } else if t.thresh_mask[i] > 0.0 {
                [255, 255, 255]
            } else {
                [40, 40, 40]
            }
        });
        y0 += h + gap;
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    pm.save_png(out).map_err(|e| e.to_string())
}
