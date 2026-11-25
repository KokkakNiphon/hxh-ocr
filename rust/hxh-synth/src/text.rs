//! Text sampler (port of `hxh_ocr/text.py`): uniform kana, frequency-weighted kana with
//! small kana and word breaks, and real words / names; labels are `normalize(raw)`.

use std::collections::{HashMap, HashSet};

use crate::glyphs::{glyph_set, normalize};
use crate::rng::Rng;

pub struct TextSampler {
    chars: Vec<char>,
    freq: Vec<f64>,
    words: Vec<Vec<char>>,
    non_bar: Vec<char>,
    i_row: HashSet<char>,
    smallable: HashMap<char, char>,
}

/// Training label for a raw sampled string (small kana folded to full size).
pub fn label_of(raw: &str) -> String {
    normalize(raw, false)
}

impl TextSampler {
    /// `extra_corpus`: extra kana/katakana entries (normalised, entries with other characters dropped).
    pub fn new(extra_corpus: &[String]) -> Self {
        let g = glyph_set();
        let d = &g.sampler;
        let mut ok: HashSet<char> = d.chars.iter().copied().collect();
        ok.extend(g.small_to_full.keys().copied());
        ok.insert(' ');
        let mut words = d.words.clone();
        for x in extra_corpus {
            let n: Vec<char> = normalize(x, true).chars().collect();
            if !n.is_empty() && n.iter().all(|c| ok.contains(c)) {
                words.push(n);
            }
        }
        TextSampler {
            chars: d.chars.clone(),
            freq: d.freq.clone(),
            words,
            non_bar: d.chars.iter().copied().filter(|&c| c != 'ー').collect(),
            i_row: d.i_row.iter().copied().collect(),
            smallable: d.smallable.clone(),
        }
    }

    fn length(&self, rng: &mut Rng, lo: usize, hi: usize) -> usize {
        (rng.gamma(2.2, 3.0).round() as i64).clamp(lo as i64, hi.max(lo) as i64) as usize
    }

    pub fn sample(&self, rng: &mut Rng, min_len: usize, max_len: usize) -> String {
        let u = rng.f64();
        let n = self.length(rng, min_len, max_len);
        let s: Vec<char> = if u < 0.6 {
            let mut s: Vec<char> = if u < 0.35 {
                (0..n).map(|_| self.chars[rng.below(self.chars.len())]).collect()
            } else {
                (0..n).map(|_| self.chars[rng.weighted(&self.freq)]).collect()
            };
            for i in 1..s.len() {
                // sprinkle small kana where Japanese would have them
                if let Some(&sm) = self.smallable.get(&s[i]) {
                    if self.i_row.contains(&s[i - 1]) && rng.chance(0.4) {
                        s[i] = sm;
                    }
                } else if s[i] == 'つ' && rng.chance(0.2) {
                    s[i] = 'っ';
                }
            }
            if n >= 4 && rng.chance(0.15) {
                for _ in 0..rng.int(1, (n as i64 / 4).max(2)) {
                    let pos = rng.int(1, s.len() as i64) as usize;
                    s.insert(pos, ' ');
                }
            }
            s.truncate(max_len);
            s
        } else {
            let sep = rng.chance(0.35);
            let mut s: Vec<char> = Vec::new();
            let mut total = 0;
            while total < n {
                let w = &self.words[rng.below(self.words.len())];
                if sep && !s.is_empty() {
                    s.push(' ');
                }
                s.extend(w);
                total += w.len() + sep as usize;
            }
            if s.len() > max_len {
                let start = if rng.chance(0.3) { rng.int(0, (s.len() - max_len + 1) as i64) as usize } else { 0 };
                s = s[start..start + max_len].to_vec();
            }
            s
        };
        self.fix(s, rng)
    }

    fn fix(&self, s: Vec<char>, rng: &mut Rng) -> String {
        let mut out: Vec<char> = Vec::with_capacity(s.len());
        for mut c in s {
            if c == ' ' && out.last().is_none_or(|&l| l == ' ') {
                continue;
            }
            if c == 'ー' && out.last().is_none_or(|&l| l == 'ー' || l == ' ') {
                c = self.non_bar[rng.below(self.non_bar.len())];
            }
            out.push(c);
        }
        let s = out.into_iter().collect::<String>().trim().to_string();
        if s.is_empty() { self.non_bar[rng.below(self.non_bar.len())].to_string() } else { s }
    }
}
