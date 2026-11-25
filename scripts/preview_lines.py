"""Contact sheet of a line dataset (labels.tsv): crops as the reader sees them (vertical columns
turned 90 degrees counter-clockwise), label and page layout under each.

    python scripts/preview_lines.py data/lines_v5/val previews/lines_v5.png [--n 48] [--seed 0]
        [--filter inline|neg|v|photo]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.fonts import label_font  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()
    d = Path(a.split)
    rows = [l.split("\t") for l in (d / "labels.tsv").read_text(encoding="utf-8").splitlines() if l]
    f = a.filter
    if f == "inline":
        rows = [r for r in rows if re.search(r"[0-9!?:/.\-]", r[1])]
    elif f == "neg":
        rows = [r for r in rows if r[1] == ""]
    elif f == "v":
        rows = [r for r in rows if r[2] == "v"]
    elif f == "photo":
        rows = [r for r in rows if "photo" in r[3]]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    H, cw = 44, 560
    font = label_font(15)
    cells = []
    for r in rows:
        im = cv2.imread(str(d / r[0]), cv2.IMREAD_GRAYSCALE)
        if r[2] == "v":
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        w = max(8, min(cw - 8, round(im.shape[1] * H / im.shape[0])))
        im = cv2.resize(im, (w, H), interpolation=cv2.INTER_AREA)
        c = Image.new("RGB", (cw, H + 24), "white")
        c.paste(Image.fromarray(im).convert("RGB"), (4, 2))
        txt = (r[1] if r[1] else "(negative: empty label)") + f"   [{r[2]} {r[3]}]"
        ImageDraw.Draw(c).text((4, H + 4), txt, font=font, fill=(0, 110, 0) if r[1] else (170, 0, 0))
        cells.append(c)
    rows_n = (len(cells) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (cw * a.cols, (H + 24) * rows_n), (235, 235, 235))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % a.cols) * cw, (i // a.cols) * (H + 24)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"{len(cells)} lines -> {a.out}")


if __name__ == "__main__":
    main()
