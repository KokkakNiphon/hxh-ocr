//! Glyph traces and text data, embedded from `assets/glyphs.json`, which
//! `scripts/export_glyphs.py` writes from `hxh_ocr/glyphs.py` and `hxh_ocr/text.py`
//! (the single source of truth). Also label normalisation (port of `glyphs.normalize`).

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::geom::P;

const DATA: &str = include_str!("../assets/glyphs.json");

#[derive(Clone, Debug)]
pub struct Stroke {
    pub dot: bool,
    pub pts: Vec<P>, // polyline, or the centre of a dot
    pub closed: bool,
    pub r: f64, // dot radius
    pub width_scale: f64,
    pub hollow: bool, // dot drawn as a ring (handakuten)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    Dakuten,
    Handakuten,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct MarkGeom {
    pub dx: f64,
    pub dy: f64,
    pub r: f64,
    pub hollow: bool,
    pub width_scale: f64,
}

pub struct Glyph {
    pub mark: Option<Mark>,
    pub weights: Option<Vec<f64>>,
    pub forms: Vec<Vec<Stroke>>,
}

pub struct SamplerData {
    pub chars: Vec<char>,
    pub freq: Vec<f64>,
    pub words: Vec<Vec<char>>,
    pub i_row: Vec<char>,
    pub smallable: HashMap<char, char>,
}

pub struct GlyphSet {
    pub vocab: Vec<char>,
    pub glyphs: HashMap<char, Glyph>,
    pub small_to_full: HashMap<char, char>,
    pub old: HashMap<char, char>,
    pub dash_chars: Vec<char>,
    pub ref_stroke_em: f64,
    pub dakuten: MarkGeom,
    pub handakuten: MarkGeom,
    pub sampler: SamplerData,
}

#[derive(Deserialize)]
struct StrokeDef {
    kind: String,
    pts: Vec<[f64; 2]>,
    closed: bool,
    r: f64,
    width_scale: f64,
    hollow: bool,
}

#[derive(Deserialize)]
struct GlyphDef {
    mark: Option<String>,
    weights: Option<Vec<f64>>,
    forms: Vec<Vec<StrokeDef>>,
}

#[derive(Deserialize)]
struct SamplerDef {
    chars: Vec<String>,
    freq: Vec<f64>,
    words: Vec<String>,
    i_row: String,
    smallable: HashMap<String, String>,
}

#[derive(Deserialize)]
struct Data {
    vocab: Vec<String>,
    small_to_full: HashMap<String, String>,
    old: HashMap<String, String>,
    dash_chars: String,
    ref_stroke_em: f64,
    marks: HashMap<String, MarkGeom>,
    glyphs: HashMap<String, GlyphDef>,
    sampler: SamplerDef,
}

fn ch(s: &str) -> char {
    s.chars().next().expect("empty glyph key")
}

fn cmap(m: &HashMap<String, String>) -> HashMap<char, char> {
    m.iter().map(|(k, v)| (ch(k), ch(v))).collect()
}

fn load() -> GlyphSet {
    let d: Data = serde_json::from_str(DATA).expect("assets/glyphs.json does not parse");
    let glyphs = d
        .glyphs
        .iter()
        .map(|(k, g)| {
            let mark = match g.mark.as_deref() {
                Some("dakuten") => Some(Mark::Dakuten),
                Some("handakuten") => Some(Mark::Handakuten),
                _ => None,
            };
            let forms = g
                .forms
                .iter()
                .map(|f| {
                    f.iter()
                        .map(|s| Stroke {
                            dot: s.kind == "dot",
                            pts: s.pts.clone(),
                            closed: s.closed,
                            r: s.r,
                            width_scale: s.width_scale,
                            hollow: s.hollow,
                        })
                        .collect()
                })
                .collect();
            (ch(k), Glyph { mark, weights: g.weights.clone(), forms })
        })
        .collect();
    GlyphSet {
        vocab: d.vocab.iter().map(|s| ch(s)).collect(),
        glyphs,
        small_to_full: cmap(&d.small_to_full),
        old: cmap(&d.old),
        dash_chars: d.dash_chars.chars().collect(),
        ref_stroke_em: d.ref_stroke_em,
        dakuten: d.marks["dakuten"],
        handakuten: d.marks["handakuten"],
        sampler: SamplerData {
            chars: d.sampler.chars.iter().map(|s| ch(s)).collect(),
            freq: d.sampler.freq.clone(),
            words: d.sampler.words.iter().map(|w| w.chars().collect()).collect(),
            i_row: d.sampler.i_row.chars().collect(),
            smallable: cmap(&d.sampler.smallable),
        },
    }
}

pub fn glyph_set() -> &'static GlyphSet {
    static G: OnceLock<GlyphSet> = OnceLock::new();
    G.get_or_init(load)
}

impl GlyphSet {
    pub fn mark_geom(&self, m: Mark) -> &MarkGeom {
        match m {
            Mark::Dakuten => &self.dakuten,
            Mark::Handakuten => &self.handakuten,
        }
    }
}

pub fn strokes_bbox(strokes: &[Stroke]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for s in strokes {
        let r = if s.dot { s.r } else { 0.0 };
        for p in &s.pts {
            b[0] = b[0].min(p[0] - r);
            b[1] = b[1].min(p[1] - r);
            b[2] = b[2].max(p[0] + r);
            b[3] = b[3].max(p[1] + r);
        }
    }
    b
}

/// Katakana -> hiragana, small kana -> full size (unless keep_small), whitespace -> one space.
pub fn normalize(text: &str, keep_small: bool) -> String {
    let g = glyph_set();
    let mut out: Vec<char> = Vec::with_capacity(text.len());
    for c0 in text.chars() {
        let mut c = c0;
        let o = c as u32;
        if (0x30A1..=0x30F6).contains(&o) {
            c = char::from_u32(o - 0x60).unwrap_or(c);
        } else if c == 'ヷ' {
            c = 'わ';
        }
        if !keep_small {
            if let Some(&f) = g.small_to_full.get(&c) {
                c = f;
            }
        }
        if let Some(&f) = g.old.get(&c) {
            c = f;
        }
        if g.dash_chars.contains(&c) {
            c = 'ー';
        }
        if c.is_whitespace() {
            c = ' ';
            if out.last().is_none_or(|&l| l == ' ') {
                continue;
            }
        }
        out.push(c);
    }
    out.into_iter().collect::<String>().trim().to_string()
}
