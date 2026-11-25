"""Render every glyph in a labelled grid, to check the traced shapes against the chart."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.glyphs import GLYPH_CHARS, GLYPHS, to_romaji  # noqa: E402
from hxh_ocr.render import TextStyle, render_text_mask  # noqa: E402
from hxh_ocr.fonts import label_font  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="previews/glyph_sheet.png")
    ap.add_argument("--em", type=float, default=90)
    ap.add_argument("--cols", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    st = TextStyle()
    # one cell per (kana, form); alternative forms are labelled "alt"
    cells = []
    for k in GLYPH_CHARS:
        g = GLYPHS[k]
        for j, form in enumerate(g.variants or [g.strokes]):
            cells.append((k, j, form))
    cw, ch = int(args.em * 1.7), int(args.em * 1.85)
    rows = (len(cells) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (cw * args.cols, ch * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = label_font(int(args.em * 0.22))
    for i, (k, j, form) in enumerate(cells):
        g = GLYPHS[k]
        saved, g.variants = g.variants, [form]
        m = render_text_mask(k, st, args.em, rng, pad=(6, 6, 6, 6))
        g.variants = saved
        g = Image.fromarray((255 - m * 255).astype(np.uint8)).convert("RGB")
        cx, cy = (i % args.cols) * cw, (i // args.cols) * ch
        sheet.paste(g, (cx + (cw - g.width) // 2, cy + int(args.em * 0.1) + (int(args.em * 1.3) - g.height) // 2))
        tag = f"{k} {to_romaji(k)}" + (" alt" if j else "")
        draw.text((cx + 8, cy + ch - int(args.em * 0.32)), tag, fill=(200, 30, 30), font=font)
        draw.rectangle([cx, cy, cx + cw - 1, cy + ch - 1], outline=(225, 225, 225))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out, sheet.size)


if __name__ == "__main__":
    main()
