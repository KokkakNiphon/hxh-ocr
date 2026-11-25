//! hxh-synth CLI: page-mode synthetic data for Hunter-alphabet text-line detection.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};

use hxh_synth::fonts::FontBank;
use hxh_synth::geom::{area2, line_thickness};
use hxh_synth::page::{make_page, ForeignOut, LineOut, PageConfig, LAYOUTS};
use hxh_synth::rng::Rng;
use hxh_synth::targets::{rasterize_targets, TargetParams};
use hxh_synth::text::TextSampler;
use hxh_synth::{codec, preview};

const USAGE: &str = "\
hxh-synth: synthetic Hunter-alphabet pages for text-line detection

USAGE:
  hxh-synth pages --out DIR [--train N] [--val N] [--test N] [--seed S] [--threads T]
                  [--quality Q] [--long-min PX] [--long-max PX] [--vertical-prob P]
                  [--corpus FILE] [--preview PNG] [--preview-n N]
                  [--font-dir DIR[;DIR...]] [--mode-p MANGA,FRAME,DOC,WEB] [--empty-prob P]
                  [--foreign-prob P] [--mixed-prob P] [--foreign-extra X] [--icon-extra X]
                  [--ui-prob P] [--foreign-em-min PX] [--keep-margin K]
                  [--v5 1] [--photo-prob P] [--ui-neg-prob P] [--caption-prob P] [--inline-prob P] [--resume 1]
                  [--v6 1] [--ruby-prob P] [--sticker-prob P] [--small-mark-prob P]
      Writes DIR/<split>/labels.jsonl (one JSON object per page:
      {\"image\", \"width\", \"height\", \"layout\", \"lines\": [{\"quad\": [[x,y] x4], \"text\", \"orient\": \"h\"|\"v\", \"kind\"}],
       \"foreign\": [{\"quad\", \"text\", \"script\", \"font\", \"orient\", \"kind\"}]}),
      DIR/<split>/images/NNNN/XXXXXXXX.jpg (grayscale) and DIR/meta.json (stats, timing, fonts).
      Quads: pixel centres at integers, clockwise from the top-left in reading orientation
      ('v' lines start at the top-right of the upright column). Each page is seeded from
      (seed, split, index). --preview writes a contact sheet with quads (and <name>_targets.png).
      `lines` are the labelled Hunter lines; `foreign` lists the unlabelled distractor runs
      (foreign text, and icons with script \"icon\"); they are never targets. Fonts are read at
      run time from --font-dir (default C:\\Windows\\Fonts or $HXH_FONT_DIR); with no fonts the
      pages have no distractors and no web layout.
      Defaults: train 2000, val 200, test 0, seed 0, threads 6, quality 90,
      mode-p 0.5,0.21,0.14,0.15, empty-prob 0.03, foreign-prob 0.2, mixed-prob 0.08,
      foreign-extra 1, icon-extra 1, ui-prob 0.25, foreign-em-min 6, keep-margin 0.3.
      v5 (photo-style pages, model family): --v5 1 turns on the v5 text styles and effects;
      --mode-p takes 7 weights (adding card, lyric and tv scenes); --photo-prob photographs
      pages off a screen (\"photo\": true in labels.jsonl); --ui-neg-prob adds avatars, round
      icons and toolbar strips; --caption-prob puts a caption right under / above free Hunter text;
      --inline-prob writes digits and ! ? : / . - inside horizontal Hunter lines (labels keep them:
      the extended reader vocab, see `lines`, which then writes the extended vocab.json).
      All default to off / 0, which reproduces the v4 settings.
      v6 (numbers as part of the lettering): --v6 1 writes the inline digits / punctuation like the
      glyphs (strokes in the line's own weight and ends, or a bold / condensed display font matched
      to the stroke weight; touching digit pairs, italic ! ?) with realistic content (model
      numbers, years, dates, times, prices, ranges, !! !? ?!), adds the promo scene (8th --mode-p
      weight: light or dark textured ground, sticker lettering with furigana and numbers, product
      art, logo, fine print) and, in `lines`, drops negative crops whose text has digits or ! ?;
      --ruby-prob puts a kana reading guide (furigana) just above horizontal Hunter groups
      (unlabelled, script kana, kind \"ruby\"; the line quad stays the glyphs' box); --sticker-prob
      draws free-standing Hunter text with a thick outline and the outline's soft shadow;
      --small-mark-prob gives that share of Hunter text styles small dakuten dots (0.55-1 stroke
      widths across, close to the glyph's bottom-right corner) instead of the v4 marks.

  hxh-synth lines --out DIR [--train N] [--val N] [--seed S] [--threads T] [--jitter 0.04]
                  [--neg-prob 0.25] [--min-height 9] [--quality 92] [page flags as for pages]
      Reader training lines cut from generated pages the way the browser cuts them (hxh-detect
      crop of the line quad grown like web/ocr.js expandQuad, the quad jittered like a detector
      box). Writes DIR/<split>/labels.tsv (path, label, h|v, layout; vertical columns upright,
      empty label = a negative crop of non-Hunter writing or icons), DIR/vocab.json, DIR/meta.json.

  hxh-synth preview --data DIR/<split> --out PNG [--n 12] [--targets PNG] [--long 512]

  hxh-synth fonts [--font-dir DIR[;DIR...]]
      List the faces the generator would use (class, scripts, icon glyph count).

  hxh-synth targets --data DIR/<split> --out DIR [--long PX] [--limit N] [--shrink 0.4] [--min-size 3]
      DBNet targets per page as grayscale PNGs <index>.{prob,prob_mask,thresh,thresh_mask,vert}.png
      (thresh stored as round(255 * thresh)); --long rescales pages so the long side is PX.
      Rule: D = A (1 - shrink^2) / L; kernel = quad edges moved inward by D (miter), filled where
      the pixel centre is inside (cv2.fillPoly is ~0.5 px fatter); lines
      thinner than --min-size or whose kernel collapses are ignored (prob_mask 0); thresh =
      0.3 + 0.4 (1 - min(dist to quad border / D, 1)); thresh_mask = quad dilated by D.
";

struct Args {
    cmd: String,
    kv: BTreeMap<String, String>,
}

impl Args {
    fn parse() -> Result<Args, String> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().ok_or("missing command")?;
        let mut kv = BTreeMap::new();
        while let Some(k) = it.next() {
            let key = k.strip_prefix("--").ok_or_else(|| format!("unexpected argument {k}"))?.to_string();
            let v = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            kv.insert(key, v);
        }
        Ok(Args { cmd, kv })
    }
    fn get<T: FromStr>(&self, k: &str, default: T) -> Result<T, String> {
        match self.kv.get(k) {
            Some(v) => v.parse().map_err(|_| format!("bad value for --{k}: {v}")),
            None => Ok(default),
        }
    }
    fn req(&self, k: &str) -> Result<String, String> {
        self.kv.get(k).cloned().ok_or_else(|| format!("--{k} is required"))
    }
}

#[derive(Serialize)]
struct PageJson<'a> {
    image: String,
    width: usize,
    height: usize,
    layout: &'a str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    photo: bool,
    lines: &'a [LineOut],
    foreign: &'a [ForeignOut],
}

struct PageSummary {
    w: usize,
    h: usize,
    layout: &'static str,
    lines: Vec<LineOut>,
    foreign: Vec<ForeignOut>,
}

fn load_fonts(a: &Args) -> FontBank {
    let dirs: Vec<PathBuf> = a.kv.get("font-dir").cloned().unwrap_or_else(FontBank::default_dir).split(';').filter(|d| !d.is_empty()).map(PathBuf::from).collect();
    FontBank::load(&dirs)
}

fn font_list(fonts: &FontBank) -> Value {
    json!(fonts
        .faces
        .iter()
        .map(|f| json!({"name": f.name, "class": f.class.name(), "bold": f.bold, "italic": f.italic, "cjk": f.cjk, "scripts": f.scripts, "icons": f.icons.len()}))
        .collect::<Vec<_>>())
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = q / 100.0 * (sorted.len() - 1) as f64;
    let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
    let v = sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo as f64);
    (v * 10.0).round() / 10.0
}

fn split_stats(pages: &[PageSummary]) -> Value {
    let mut n: Vec<usize> = pages.iter().map(|p| p.lines.len()).collect();
    let total: usize = n.iter().sum();
    let mut th: Vec<f64> = pages.iter().flat_map(|p| p.lines.iter().map(|l| line_thickness(&l.quad))).collect();
    th.sort_by(f64::total_cmp);
    let nv = pages.iter().flat_map(|p| &p.lines).filter(|l| l.orient == "v").count();
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for l in pages.iter().flat_map(|p| &p.lines) {
        *kinds.entry(l.kind).or_default() += 1;
    }
    n.sort();
    let np = pages.len().max(1) as f64;
    let area = |q: &[[f64; 2]; 4]| area2(q).abs() * 0.5;
    let hxh_area: f64 = pages.iter().flat_map(|p| &p.lines).map(|l| area(&l.quad)).sum();
    let mut fn_: Vec<usize> = pages.iter().map(|p| p.foreign.len()).collect();
    fn_.sort();
    let nf: usize = fn_.iter().sum();
    let mut scripts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut fonts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut fkinds: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut f_area, mut i_area) = (0.0, 0.0);
    let mut fth: Vec<f64> = Vec::new();
    for f in pages.iter().flat_map(|p| &p.foreign) {
        *scripts.entry(f.script).or_default() += 1;
        *fonts.entry(f.font.as_str()).or_default() += 1;
        *fkinds.entry(f.kind).or_default() += 1;
        if f.script == "icon" {
            i_area += area(&f.quad);
        } else {
            f_area += area(&f.quad);
            fth.push(line_thickness(&f.quad));
        }
    }
    fth.sort_by(f64::total_cmp);
    let mut layouts: BTreeMap<&str, usize> = LAYOUTS.iter().map(|&l| (l, 0)).collect();
    for p in pages {
        *layouts.entry(p.layout).or_default() += 1;
    }
    let negatives = pages.iter().filter(|p| p.lines.is_empty()).count();
    let r2 = |x: f64| (x * 100.0).round() / 100.0;
    json!({
        "pages": pages.len(),
        "lines": total,
        "lines_per_page": {"mean": ((total as f64 / np) * 100.0).round() / 100.0, "median": n.get(n.len() / 2).copied().unwrap_or(0),
                           "min": n.first().copied().unwrap_or(0), "max": n.last().copied().unwrap_or(0),
                           "empty_pages": n.iter().filter(|&&v| v == 0).count()},
        "line_height_px": {"p1": pct(&th, 1.0), "p10": pct(&th, 10.0), "p25": pct(&th, 25.0), "p50": pct(&th, 50.0),
                           "p75": pct(&th, 75.0), "p90": pct(&th, 90.0), "p99": pct(&th, 99.0)},
        "vertical_pct": ((1e4 * nv as f64 / total.max(1) as f64).round()) / 100.0,
        "kinds": kinds,
        "portrait_pct": ((1e3 * pages.iter().filter(|p| p.h > p.w).count() as f64 / np).round()) / 10.0,
        "layouts": layouts,
        "pages_without_hunter_lines": negatives,
        "foreign": {
            "items": nf,
            "per_page": {"mean": r2(nf as f64 / np), "median": fn_.get(fn_.len() / 2).copied().unwrap_or(0), "max": fn_.last().copied().unwrap_or(0)},
            "scripts": scripts,
            "kinds": fkinds,
            "fonts": fonts,
            "text_height_px": {"p1": pct(&fth, 1.0), "p10": pct(&fth, 10.0), "p50": pct(&fth, 50.0), "p90": pct(&fth, 90.0), "p99": pct(&fth, 99.0)},
        },
        "area_px": {"hunter": hxh_area.round(), "foreign_text": f_area.round(), "icons": i_area.round(),
                    "foreign_text_over_hunter": r2(f_area / hxh_area.max(1.0)), "foreign_all_over_hunter": r2((f_area + i_area) / hxh_area.max(1.0))},
    })
}

fn page_cfg(a: &Args) -> Result<PageConfig, String> {
    let d = PageConfig::default();
    let mode_p = match a.kv.get("mode-p") {
        Some(v) => {
            let p: Vec<f64> = v.split(',').map(|x| x.trim().parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("bad --mode-p {v}"))?;
            if !(p.len() == 4 || p.len() == 7 || p.len() == 8) || p.iter().any(|&x| x < 0.0) || p.iter().sum::<f64>() <= 0.0 {
                return Err(format!("--mode-p needs 4, 7 or 8 non-negative weights (manga,frame,doc,web[,card,lyric,tv[,promo]]), got {v}"));
            }
            let mut m = [0.0; 8];
            m[..p.len()].copy_from_slice(&p);
            m
        }
        None => d.mode_p,
    };
    let cfg = PageConfig {
        long_min: a.get("long-min", 512)?,
        long_max: a.get("long-max", 1024)?,
        vertical_prob: a.get("vertical-prob", 0.11)?,
        mode_p,
        empty_prob: a.get("empty-prob", d.empty_prob)?,
        foreign_prob: a.get("foreign-prob", d.foreign_prob)?,
        mixed_prob: a.get("mixed-prob", d.mixed_prob)?,
        foreign_extra: a.get("foreign-extra", d.foreign_extra)?,
        icon_extra: a.get("icon-extra", d.icon_extra)?,
        ui_prob: a.get("ui-prob", d.ui_prob)?,
        foreign_em_min: a.get("foreign-em-min", d.foreign_em_min)?,
        keep_margin: a.get("keep-margin", d.keep_margin)?,
        v5: a.get::<u8>("v5", 0)? != 0,
        photo_prob: a.get("photo-prob", d.photo_prob)?,
        ui_neg_prob: a.get("ui-neg-prob", d.ui_neg_prob)?,
        caption_prob: a.get("caption-prob", d.caption_prob)?,
        inline_prob: a.get("inline-prob", d.inline_prob)?,
        v6: a.get::<u8>("v6", 0)? != 0,
        ruby_prob: a.get("ruby-prob", d.ruby_prob)?,
        sticker_prob: a.get("sticker-prob", d.sticker_prob)?,
        small_mark_prob: a.get("small-mark-prob", d.small_mark_prob)?,
        ..d
    };
    Ok(cfg)
}

fn cmd_pages(a: &Args) -> Result<(), String> {
    let out = PathBuf::from(a.req("out")?);
    let (train, val, test): (usize, usize, usize) = (a.get("train", 2000)?, a.get("val", 200)?, a.get("test", 0)?);
    let seed: u64 = a.get("seed", 0)?;
    let threads: usize = a.get("threads", 6)?;
    let quality: u8 = a.get("quality", 90)?;
    let cfg = page_cfg(a)?;
    let t_fonts = Instant::now();
    let fonts = load_fonts(a);
    let n_icon = fonts.faces.iter().filter(|f| f.class == hxh_synth::fonts::Class::Icon).count();
    eprintln!("fonts: {} faces ({} icon fonts) from {:?} in {:.2}s", fonts.faces.len(), n_icon, fonts.dirs, t_fonts.elapsed().as_secs_f64());
    if fonts.faces.is_empty() {
        eprintln!("warning: no fonts found: pages get no foreign text, icons or web layout");
    }
    let corpus: Vec<String> = match a.kv.get("corpus") {
        Some(p) => fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect(),
        None => vec![],
    };
    let sampler = TextSampler::new(&corpus);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?;
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut splits = serde_json::Map::new();
    for (split, n, sid) in [("train", train, 0u64), ("val", val, 1), ("test", test, 2)] {
        if n == 0 {
            continue;
        }
        let dir = out.join(split);
        for k in 0..n.div_ceil(1000) {
            fs::create_dir_all(dir.join(format!("images/{k:04}"))).map_err(|e| e.to_string())?;
        }
        // pages go to labels.jsonl.partial in chunks of 2000 (in index order), so a run that dies
        // loses at most one chunk; --resume 1 continues after the pages already written
        let partial = dir.join("labels.jsonl.partial");
        let start = if a.get::<u8>("resume", 0)? != 0 && partial.exists() {
            fs::read_to_string(&partial).map_err(|e| e.to_string())?.lines().count()
        } else {
            let _ = fs::remove_file(&partial);
            0
        };
        if start > 0 {
            eprintln!("{split}: resuming after {start} pages");
        }
        let done = AtomicUsize::new(start);
        let t0 = Instant::now();
        let mut res: Vec<Result<(String, PageSummary), String>> = Vec::with_capacity(n);
        let mut c0 = start;
        while c0 < n {
            let c1 = (c0 + 2000).min(n);
            let chunk: Vec<Result<(String, PageSummary), String>> = pool.install(|| {
            (c0..c1)
                .into_par_iter()
                .map(|i| {
                    let mut rng = Rng::from_key(&[seed, sid, i as u64]);
                    let p = make_page(&mut rng, &sampler, &fonts, &cfg);
                    let rel = format!("images/{:04}/{:08}.jpg", i / 1000, i);
                    fs::write(dir.join(&rel), codec::encode_jpeg(&p.pixels, p.w, p.h, quality)).map_err(|e| e.to_string())?;
                    let pj = PageJson { image: rel, width: p.w, height: p.h, layout: p.layout, photo: p.photo, lines: &p.lines, foreign: &p.foreign };
                    let line = serde_json::to_string(&pj).map_err(|e| e.to_string())?;
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_multiple_of(100) {
                        eprint!("\r{split}: {k}/{n}  {:.1} pages/s ", k as f64 / t0.elapsed().as_secs_f64());
                    }
                    Ok((line, PageSummary { w: p.w, h: p.h, layout: p.layout, lines: p.lines, foreign: p.foreign }))
                })
                .collect()
            });
            let mut block = String::new();
            for r in &chunk {
                let (line, _) = r.as_ref().map_err(|e| e.clone())?;
                block.push_str(line);
                block.push('\n');
            }
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&partial).map_err(|e| e.to_string())?;
            std::io::Write::write_all(&mut f, block.as_bytes()).map_err(|e| e.to_string())?;
            res.extend(chunk);
            c0 = c1;
        }
        let el = t0.elapsed().as_secs_f64();
        let mut pages = Vec::with_capacity(n);
        for r in res {
            let (_, s) = r?;
            pages.push(s);
        }
        fs::rename(&partial, dir.join("labels.jsonl")).map_err(|e| e.to_string())?;
        let n = n - start; // stats below cover this run's pages
        let mut st = split_stats(&pages);
        st["seconds"] = json!((el * 10.0).round() / 10.0);
        st["pages_per_s"] = json!((n as f64 / el * 10.0).round() / 10.0);
        st["threads"] = json!(threads);
        eprintln!("\r{split}: {n} pages in {el:.1}s ({:.1} pages/s, {threads} threads)", n as f64 / el);
        println!("{split}: {st}");
        splits.insert(split.to_string(), st);
    }
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let meta = json!({
        "generator": "hxh-synth pages",
        "args": a.kv,
        "config": cfg,
        "created_unix": created,
        "font_dirs": fonts.dirs,
        "fonts": font_list(&fonts),
        "format": "labels.jsonl: one page per line; quad = 4 corners, pixel centres at integers, clockwise from the top-left in reading orientation ('v': top-right of the upright column first); text = normalised label; foreign = unlabelled distractor runs (foreign text; script icon = icons), never in the targets",
        "splits": splits,
    });
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
    if let Some(pv) = a.kv.get("preview") {
        let src = if val > 0 { out.join("val") } else { out.join("train") };
        let pv = PathBuf::from(pv);
        preview::contact_sheet(&src, &pv, a.get("preview-n", 12)?, 4, 480)?;
        let tp = pv.with_file_name(format!("{}_targets.png", pv.file_stem().unwrap().to_string_lossy()));
        preview::targets_sheet(&src, &tp, 3, 512)?;
        println!("preview: {} and {}", pv.display(), tp.display());
    }
    println!("done: {}", out.display());
    Ok(())
}

/// Grow a reading-oriented quad (q0 -> q1 along the text, q0 -> q3 across) by `along` / `across`
/// times its thickness at both ends / both sides: web/ocr.js expandQuad.
fn expand_quad(q: &[[f64; 2]; 4], along: f64, across: f64) -> [[f64; 2]; 4] {
    let len = (q[1][0] - q[0][0]).hypot(q[1][1] - q[0][1]).max(1e-9);
    let th = (q[3][0] - q[0][0]).hypot(q[3][1] - q[0][1]).max(1e-9);
    let u = [(q[1][0] - q[0][0]) / len, (q[1][1] - q[0][1]) / len];
    let v = [(q[3][0] - q[0][0]) / th, (q[3][1] - q[0][1]) / th];
    let (a, c) = (along * th, across * th);
    [
        [q[0][0] - u[0] * a - v[0] * c, q[0][1] - u[1] * a - v[1] * c],
        [q[1][0] + u[0] * a - v[0] * c, q[1][1] + u[1] * a - v[1] * c],
        [q[2][0] + u[0] * a + v[0] * c, q[2][1] + u[1] * a + v[1] * c],
        [q[3][0] - u[0] * a + v[0] * c, q[3][1] - u[1] * a + v[1] * c],
    ]
}

/// One line crop the way the browser pipeline cuts it (hxh-detect `crop` of the expanded quad, at
/// native resolution), from a quad that imitates a detector box: corners jittered, thickness and
/// length off by a little. Vertical crops are turned back upright (the line datasets store
/// columns upright; the trainer turns them 90 degrees counter-clockwise, as the browser does).
fn cut_line(px: &[u8], w: usize, h: usize, quad: &[[f64; 2]; 4], vertical: bool, rng: &mut Rng, jit: f64) -> Option<(Vec<u8>, usize, usize)> {
    let th = line_thickness(quad);
    let mut q = *quad;
    for p in q.iter_mut() {
        p[0] += rng.normal(0.0, jit * th);
        p[1] += rng.normal(0.0, jit * th);
    }
    // a detector box is usually a little fatter and longer than the tight ink box
    let q = expand_quad(&q, rng.uniform(-0.05, 0.25), rng.uniform(-0.04, 0.15));
    let q = expand_quad(&q, rng.uniform(0.15, 0.45), rng.uniform(0.1, 0.3));
    if q.iter().any(|p| p[0] < -0.5 * th || p[1] < -0.5 * th || p[0] > w as f64 + 0.5 * th || p[1] > h as f64 + 0.5 * th) {
        return None;
    }
    let (cw, ch) = hxh_detect::crop_size(&q);
    if cw < 4 || ch < 4 || cw > 8000 || ch > 2000 {
        return None;
    }
    let mut out = vec![0u8; cw * ch];
    hxh_detect::crop(px, w, h, &q, &mut out, cw, ch);
    // legibility: a photo blurred past reading, or an effect that hid the strokes, leaves a crop
    // with almost no contrast; such a line is not a training example (5th-95th percentile spread)
    let mut hist = [0usize; 256];
    out.iter().for_each(|&v| hist[v as usize] += 1);
    let pct = |f: f64| {
        let k = (f * out.len() as f64) as usize;
        let mut acc = 0;
        (0..256).find(|&i| {
            acc += hist[i];
            acc > k
        }).unwrap_or(255)
    };
    if pct(0.95) < pct(0.05) + MIN_SPREAD {
        return None;
    }
    if !vertical {
        return Some((out, cw, ch));
    }
    // upright column U[r][c] = C[ch - 1 - c][r]
    let mut u = vec![0u8; cw * ch];
    for r in 0..cw {
        for c in 0..ch {
            u[r * ch + c] = out[(ch - 1 - c) * cw + r];
        }
    }
    Some((u, ch, cw))
}

/// Least 5th-95th percentile grey spread of a line crop worth training on.
const MIN_SPREAD: usize = 28;

/// Tokens the extended reader vocab adds after the 74 kana + space tokens.
const EXTRA_VOCAB: &str = "0123456789!?:/.-";

/// `lines`: reader training lines cut from v5 pages (Rust line generation: every page style,
/// photo effect and neighbour / caption context comes with the crops for free).
fn cmd_lines(a: &Args) -> Result<(), String> {
    let out = PathBuf::from(a.req("out")?);
    let (train, val): (usize, usize) = (a.get("train", 2000)?, a.get("val", 200)?);
    let seed: u64 = a.get("seed", 0)?;
    let threads: usize = a.get("threads", 6)?;
    let quality: u8 = a.get("quality", 92)?;
    let jit: f64 = a.get("jitter", 0.04)?;
    let neg: f64 = a.get("neg-prob", 0.25)?; // pages that give one unlabelled crop (empty label)
    let min_th: f64 = a.get("min-height", 9.0)?;
    let cfg = page_cfg(a)?;
    let fonts = load_fonts(a);
    eprintln!("fonts: {} faces from {:?}", fonts.faces.len(), fonts.dirs);
    let sampler = TextSampler::new(&[]);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| e.to_string())?;
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut vocab: Vec<String> = hxh_synth::glyphs::glyph_set().vocab.iter().map(|c| c.to_string()).collect();
    if cfg.inline_prob > 0.0 {
        // extended vocab: the 74 kana tokens first (same ids as before), then digits and punctuation
        vocab.extend(EXTRA_VOCAB.chars().map(|c| c.to_string()));
    }
    fs::write(out.join("vocab.json"), serde_json::to_string(&vocab).unwrap()).map_err(|e| e.to_string())?;
    let mut splits = serde_json::Map::new();
    for (split, n, sid) in [("train", train, 0u64), ("val", val, 1)] {
        if n == 0 {
            continue;
        }
        let dir = out.join(split);
        for k in 0..n.div_ceil(1000) {
            fs::create_dir_all(dir.join(format!("images/{k:04}"))).map_err(|e| e.to_string())?;
        }
        // chunks of 2000 pages appended to labels.tsv.partial (progress.txt = next page), so a
        // run that dies loses at most one chunk; --resume 1 continues from progress.txt
        let partial = dir.join("labels.tsv.partial");
        let progress = dir.join("progress.txt");
        let start: usize = if a.get::<u8>("resume", 0)? != 0 && partial.exists() && progress.exists() {
            fs::read_to_string(&progress).map_err(|e| e.to_string())?.trim().parse().unwrap_or(0)
        } else {
            let _ = fs::remove_file(&partial);
            0
        };
        if start > 0 {
            eprintln!("{split}: resuming at page {start}");
        }
        let done = AtomicUsize::new(start);
        let t0 = Instant::now();
        let mut c0 = start;
        while c0 < n {
            let c1 = (c0 + 2000).min(n);
            let rows: Vec<Vec<String>> = pool.install(|| {
            (c0..c1)
                .into_par_iter()
                .map(|i| {
                    let mut rng = Rng::from_key(&[seed, sid, i as u64]);
                    let p = make_page(&mut rng, &sampler, &fonts, &cfg);
                    let mut rr = Rng::from_key(&[seed, sid, i as u64, 7]);
                    let mut rows = Vec::new();
                    let style = if p.photo { format!("{}+photo", p.layout) } else { p.layout.to_string() };
                    for (k, l) in p.lines.iter().enumerate() {
                        if line_thickness(&l.quad) < min_th {
                            continue;
                        }
                        let v = l.orient == "v";
                        if let Some((px, w, h)) = cut_line(&p.pixels, p.w, p.h, &l.quad, v, &mut rr, jit) {
                            let rel = format!("images/{:04}/{:08}_{:02}.jpg", i / 1000, i, k);
                            if fs::write(dir.join(&rel), codec::encode_jpeg(&px, w, h, quality)).is_ok() {
                                rows.push(format!("{rel}\t{}\t{}\t{style}", l.text, l.orient));
                            }
                        }
                    }
                    // a negative: non-Hunter writing or an icon cut the same way, empty label
                    if !p.foreign.is_empty() && rr.chance(neg) {
                        let f = &p.foreign[rr.below(p.foreign.len())];
                        let v = f.orient == "v";
                        // v6: digits are read wherever they are written, so a run with digits (or
                        // ! ?) is no "read nothing" example
                        let digits = cfg.v6 && f.text.chars().any(|c| c.is_ascii_digit() || c == '!' || c == '?');
                        if line_thickness(&f.quad) >= min_th && !digits
                            && let Some((px, w, h)) = cut_line(&p.pixels, p.w, p.h, &f.quad, v, &mut rr, jit) {
                                let rel = format!("images/{:04}/{:08}_n.jpg", i / 1000, i);
                                if fs::write(dir.join(&rel), codec::encode_jpeg(&px, w, h, quality)).is_ok() {
                                    rows.push(format!("{rel}\t\t{}\tneg-{}", f.orient, f.script));
                                }
                            }
                    }
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_multiple_of(200) {
                        eprint!("\r{split}: {k}/{n} pages  {:.1} pages/s ", k as f64 / t0.elapsed().as_secs_f64());
                    }
                    rows
                })
                .collect()
            });
            let block: String = rows.into_iter().flatten().map(|r| r + "\n").collect();
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&partial).map_err(|e| e.to_string())?;
            std::io::Write::write_all(&mut f, block.as_bytes()).map_err(|e| e.to_string())?;
            fs::write(&progress, c1.to_string()).map_err(|e| e.to_string())?;
            c0 = c1;
        }
        fs::rename(&partial, dir.join("labels.tsv")).map_err(|e| e.to_string())?;
        let _ = fs::remove_file(&progress);
        let text = fs::read_to_string(dir.join("labels.tsv")).map_err(|e| e.to_string())?;
        let all: Vec<&str> = text.lines().collect();
        let nv = all.iter().filter(|r| r.split('\t').nth(2) == Some("v")).count();
        let nn = all.iter().filter(|r| r.split('\t').nth(1) == Some("")).count();
        let el = t0.elapsed().as_secs_f64();
        eprintln!("\r{split}: {n} pages -> {} lines ({nv} vertical, {nn} negatives) in {el:.1}s", all.len());
        splits.insert(split.into(), json!({"pages": n, "lines": all.len(), "vertical": nv, "negatives": nn, "seconds": (el * 10.0).round() / 10.0}));
    }
    let meta = json!({"generator": "hxh-synth lines", "args": a.kv, "config": cfg, "font_dirs": fonts.dirs, "fonts": font_list(&fonts), "splits": splits,
        "format": "labels.tsv: path, label (empty = negative: non-Hunter writing or icons), h|v, page layout; vertical columns stored upright"});
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
    println!("done: {}", out.display());
    Ok(())
}

fn cmd_fonts(a: &Args) -> Result<(), String> {
    let fonts = load_fonts(a);
    println!("{} faces from {:?}", fonts.faces.len(), fonts.dirs);
    for f in &fonts.faces {
        let sc: Vec<&str> = [(1u8, "latin"), (2, "latin-ext"), (4, "japanese"), (8, "korean"), (16, "greek"), (32, "cyrillic")]
            .iter()
            .filter(|(b, _)| f.scripts & b != 0)
            .map(|x| x.1)
            .collect();
        let style = format!("{}{}", if f.bold { "bold " } else { "" }, if f.italic { "italic" } else { "" });
        println!("{:26} {:8} {:12} icons={:5} {}", f.name, f.class.name(), style, f.icons.len(), sc.join(","));
    }
    Ok(())
}

fn cmd_preview(a: &Args) -> Result<(), String> {
    let src = PathBuf::from(a.req("data")?);
    let out = PathBuf::from(a.req("out")?);
    preview::contact_sheet(&src, &out, a.get("n", 12)?, 4, 480)?;
    if let Some(t) = a.kv.get("targets") {
        preview::targets_sheet(&src, Path::new(t), 3, a.get("long", 512)?)?;
    }
    Ok(())
}

fn save_gray_png(path: &Path, px: &[u8], w: usize, h: usize) -> Result<(), String> {
    let f = fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(px).map_err(|e| e.to_string())
}

fn cmd_targets(a: &Args) -> Result<(), String> {
    let src = PathBuf::from(a.req("data")?);
    let out = PathBuf::from(a.req("out")?);
    let long: usize = a.get("long", 0)?;
    let prm = TargetParams { shrink: a.get("shrink", 0.4)?, min_size: a.get("min-size", 3.0)?, ..Default::default() };
    let recs = preview::read_labels(&src, a.get("limit", usize::MAX)?)?;
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    for (i, rec) in recs.iter().enumerate() {
        let s = if long > 0 { long as f64 / rec.width.max(rec.height) as f64 } else { 1.0 };
        let (w, h) = (((rec.width as f64 * s).round() as usize).max(1), ((rec.height as f64 * s).round() as usize).max(1));
        let lines: Vec<([[f64; 2]; 4], bool)> =
            rec.lines.iter().map(|l| (l.quad.map(|p| [(p[0] + 0.5) * s - 0.5, (p[1] + 0.5) * s - 0.5]), l.orient == "v")).collect();
        let t = rasterize_targets(&lines, w, h, &prm);
        let stem = Path::new(&rec.image).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| format!("{i:08}"));
        for (name, m) in [("prob", &t.prob), ("prob_mask", &t.prob_mask), ("thresh", &t.thresh), ("thresh_mask", &t.thresh_mask), ("vert", &t.vert)] {
            let px: Vec<u8> = m.iter().map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect();
            save_gray_png(&out.join(format!("{stem}.{name}.png")), &px, w, h)?;
        }
    }
    println!("wrote targets for {} pages to {}", recs.len(), out.display());
    Ok(())
}

fn main() {
    let a = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let r = match a.cmd.as_str() {
        "pages" => cmd_pages(&a),
        "lines" => cmd_lines(&a),
        "preview" => cmd_preview(&a),
        "fonts" => cmd_fonts(&a),
        "targets" => cmd_targets(&a),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        c => Err(format!("unknown command {c}")),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
