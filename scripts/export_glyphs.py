"""Export the glyph and text-sampler data to rust/hxh-synth/assets/glyphs.json.

    python scripts/export_glyphs.py

hxh_ocr/glyphs.py and hxh_ocr/text.py stay the single source of truth for the glyph
traces and the text data; the Rust page generator (rust/hxh-synth) embeds this file.
Re-run after editing either module. Only data is exported, no behaviour:
  vocab, small_to_full, old (obsolete kana), dash_chars   label normalisation tables
  ref_stroke_em                                          chart stroke width (em)
  marks                                                  dakuten / handakuten geometry
  glyphs[char]                                           mark, per-form sampling weights and every
                                                         form as strokes in em units, exactly as
                                                         glyphs.py builds them
  sampler                                                kana frequencies, word list (normalised,
                                                         small kana kept), i-row kana, smallable kana
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr import glyphs as G  # noqa: E402
from hxh_ocr import text as T  # noqa: E402


def _stroke(s: G.Stroke) -> dict:
    return {"kind": s.kind, "pts": [[round(float(x), 6), round(float(y), 6)] for x, y in s.pts],
            "closed": bool(s.closed), "r": round(float(s.r), 6), "width_scale": float(s.width_scale),
            "hollow": bool(s.hollow)}


def main():
    sampler = T.TextSampler()
    glyphs = {}
    for ch in G.GLYPH_CHARS:
        g = G.GLYPHS[ch]
        forms = g.variants or [g.strokes]
        glyphs[ch] = {"base": g.base, "mark": g.mark,
                      "weights": [float(w) for w in g.weights] if g.weights else None,
                      "forms": [[_stroke(s) for s in f] for f in forms]}
    data = {
        "source": "hxh_ocr/glyphs.py + hxh_ocr/text.py via scripts/export_glyphs.py",
        "vocab": G.VOCAB,
        "small_to_full": G.SMALL_TO_FULL,
        "old": G._OLD,
        "dash_chars": "‐－−～〜",
        "ref_stroke_em": G.REF_STROKE_EM,
        "marks": {
            "dakuten": {"dx": G.DAKUTEN_DX, "dy": G.DAKUTEN_DY, "r": G.DAKUTEN_R, "hollow": False, "width_scale": 1.0},
            "handakuten": {"dx": G.HANDAKUTEN_DX, "dy": G.HANDAKUTEN_DY, "r": G.HANDAKUTEN_R, "hollow": True,
                           "width_scale": 0.65},
        },
        "glyphs": glyphs,
        "sampler": {
            "chars": sampler.chars,
            "freq": [float(T._FREQ[c]) for c in sampler.chars],
            "words": sampler.words,
            "i_row": "".join(sorted(T._I_ROW)),
            "smallable": T._SMALLABLE,
        },
    }
    # sanity: mark geometry must match glyphs.mark_stroke
    for mark, m in data["marks"].items():
        s = G.mark_stroke(mark, (0.0, 0.0, 0.0, 0.0), 1.0)
        assert abs(s.pts[0, 0] - m["dx"]) < 1e-9 and abs(s.pts[0, 1] - m["dy"]) < 1e-9 and abs(s.r - m["r"]) < 1e-9
        assert s.hollow == m["hollow"] and abs(s.width_scale - m["width_scale"]) < 1e-9
    out = ROOT / "rust" / "hxh-synth" / "assets" / "glyphs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_forms = sum(len(g["forms"]) for g in glyphs.values())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(glyphs)} glyphs, {n_forms} forms, "
          f"{len(sampler.words)} words")


if __name__ == "__main__":
    main()
