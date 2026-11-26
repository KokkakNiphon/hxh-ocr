"""Figures for the report's addendum on readers v7 (the cross of the episode titles).

- web/blog/img/deep/v7_pages.webp: two generated title-card pages (generator v7) from
  previews/pages_v7.png, with the contact sheet's boxes: Hunter lines green, the katakana title
  and other foreign text orange.
- web/blog/img/deep/v7_lines_{en,th,ja}.webp: ten held-out lines with a cross, drawn at random
  (seed 7) from data/lines_v7_cross/val, with what they say and what the v6 Accurate reader and
  the v7 one read, int8, through the app's own readCrop (scripts/lines_int8.mjs). Wrong characters
  in red. One image per language (only the column heads differ). The samples and every reading
  (the deployed readers and the tiny ones too) go to v7_lines.json.

    .venv/Scripts/python.exe scripts/blog/v7_images.py [--cells r,c;r,c]
"""
from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hxh_ocr.metrics import align  # noqa: E402

OUT = ROOT / "web" / "blog" / "img" / "deep"
SET = ROOT / "data" / "lines_v7_cross" / "val"
XS = set("\u00d7")
READERS = {
    "fast": ("web/models/model.hxhm", "web/models/model.json"),
    "tiny_v6": ("experiments/family/tiny_v6/model.hxhm", "experiments/family/tiny_v6/model.json"),
    "tiny_v7": ("experiments/family/tiny_v7/model.hxhm", "experiments/family/tiny_v7/model.json"),
    "accurate": ("web/models/accurate/model.hxhm", "web/models/accurate/model.json"),
    "base_v6": ("experiments/family/base_v6/model.hxhm", "experiments/family/base_v6/model.json"),
    "base_v7": ("experiments/family/base_v7/model.hxhm", "experiments/family/base_v7/model.json"),
}
COLS = ("base_v6", "base_v7")  # the two reading columns
HEADS = {
    "en": ("written", "Accurate v6", "Accurate v7"),
    "th": ("ข้อความจริง", "แม่นยำ v6", "แม่นยำ v7"),
    "ja": ("正解", "高精度 v6", "高精度 v7"),
}
INK, MUTED, RED, GREEN, RULE = (24, 26, 32), (110, 114, 124), (200, 40, 40), (30, 140, 70), (225, 227, 232)
FAINT = (240, 170, 170)
MAX_BYTES = 150 * 1024


def font(size, thai=False, bold=False):
    names = (["C:/Windows/Fonts/LeelUIsl.ttf", "C:/Windows/Fonts/LeelawUI.ttf", "C:/Windows/Fonts/tahoma.ttf"] if thai else
             ["C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"])
    for p in names:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise SystemExit("no font for the figure")


def save_webp(im: Image.Image, name: str, lossless=False) -> dict:
    path = OUT / name
    q = 86
    while True:
        im.save(path, "WEBP", quality=q, method=6, lossless=lossless)
        if lossless or path.stat().st_size <= MAX_BYTES or q <= 60:
            break
        q -= 6
    return {"file": name, "width": im.width, "height": im.height, "kb": round(path.stat().st_size / 1024)}


def pages(cells) -> dict:
    sheet = np.array(Image.open(ROOT / "previews" / "pages_v7.png").convert("RGB"))
    cell, gap = 480, 8
    bg = sheet[2, 2].astype(int)
    crops = []
    for r, c in cells:
        x, y = gap + c * (cell + gap), gap + r * (cell + gap)
        a = sheet[y:y + cell, x:x + cell]
        ys, xs = np.nonzero(np.abs(a.astype(int) - bg).sum(2) > 6)
        crops.append(a[:ys.max() + 1, :xs.max() + 1])
    h = max(cr.shape[0] for cr in crops)
    w = sum(cr.shape[1] for cr in crops) + gap * (len(crops) - 1)
    out = np.full((h, w, 3), 255, np.uint8)
    x = 0
    for cr in crops:
        out[:cr.shape[0], x:x + cr.shape[1]] = cr
        x += cr.shape[1] + gap
    im = Image.fromarray(out)
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)  # shown at about half its width
    return {"source": f"previews/pages_v7.png, cells {cells}", **save_webp(im, "v7_pages.webp")}


def sample(n=10, seed=7):
    rows = []
    for line in (SET / "labels.tsv").read_text(encoding="utf-8").splitlines():
        path, text, orient, kind = line.split("\t")
        if orient == "h" and set(text) & XS:
            rows.append((path, text, kind))
    return random.Random(seed).sample(rows, n)


def read(rows) -> dict:
    crops = [cv2.imread(str(SET / p), cv2.IMREAD_GRAYSCALE) for p, _, _ in rows]
    with tempfile.TemporaryDirectory() as td:
        b = Path(td) / "crops.bin"
        with b.open("wb") as f:
            f.write(struct.pack("<I", len(crops)))
            for g in crops:
                f.write(struct.pack("<II", g.shape[1], g.shape[0]))
                f.write(np.ascontiguousarray(g).tobytes())
        out = {}
        for name, (m, j) in READERS.items():
            res = subprocess.run(["node", str(ROOT / "scripts" / "lines_int8.mjs"), str(b), str(ROOT / m), str(ROOT / j)],
                                 capture_output=True, text=True, encoding="utf-8", check=True)
            out[name] = json.loads(res.stdout)
    return out


def draw_reading(d: ImageDraw.ImageDraw, x, y, truth, hyp, f):
    """The reading at (x, y): characters that do not match the truth in red, characters the reader
    dropped faint red and struck through; a check if exact."""
    if hyp == truth:
        d.text((x, y), hyp, font=f, fill=INK)
        d.text((x + d.textlength(hyp, font=f) + 8, y), "✓", font=f, fill=GREEN)
        return
    _, ops = align(truth, hyp)
    for a, b in ops:
        if b is None:
            w = d.textlength(a, font=f)
            d.text((x, y), a, font=f, fill=FAINT)
            mid = y + f.size * 0.62
            d.line((x, mid, x + w, mid), fill=RED, width=max(2, f.size // 12))
            x += w
            continue
        d.text((x, y), b, font=f, fill=INK if a == b else RED)
        w = d.textlength(b, font=f)
        if b == " " and a != b:  # a space the reader made up: underline it
            base = y + f.size * 1.18
            d.line((x + 2, base, x + w - 2, base), fill=RED, width=max(2, f.size // 12))
        x += w


def sheet(rows, readings, lang) -> Image.Image:
    S = 2  # drawn at 2x; shown at about 700 CSS px
    row_h, img_h, pad = 30 * S, 20 * S, 8 * S
    col_img, col_w = 160 * S, 208 * S  # v7 lines are longer than v6's
    f = font(12 * S)
    fh = font(11 * S, thai=lang == "th", bold=lang != "th")
    W = pad + col_img + 3 * col_w + pad
    H = pad + 22 * S + len(rows) * row_h + pad
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    x0 = pad + col_img
    for k, head in enumerate(HEADS[lang]):
        d.text((x0 + k * col_w, pad), head, font=fh, fill=MUTED)
    y = pad + 22 * S
    for i, (path, truth, _) in enumerate(rows):
        d.line((pad, y, W - pad, y), fill=RULE, width=S)
        g = Image.open(SET / path).convert("RGB")
        k = img_h / g.height
        w = min(int(g.width * k), col_img - 12 * S)
        g = g.resize((w, img_h), Image.LANCZOS) if g.width * k <= col_img - 12 * S else \
            g.resize((w, int(g.height * w / g.width)), Image.LANCZOS)
        im.paste(g, (pad, y + (row_h - g.height) // 2))
        ty = y + (row_h - 12 * S) // 2 - 3 * S
        d.text((x0, ty), truth, font=f, fill=INK)
        draw_reading(d, x0 + col_w, ty, truth, readings[COLS[0]][i][0], f)
        draw_reading(d, x0 + 2 * col_w, ty, truth, readings[COLS[1]][i][0], f)
        y += row_h
    return im


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="0,0;0,1", help="row,col of the two title pages in previews/pages_v7.png")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    made = [pages([tuple(int(v) for v in c.split(",")) for c in a.cells.split(";")])]
    rows = sample()
    readings = read(rows)
    for lang in HEADS:
        made.append(save_webp(sheet(rows, readings, lang), f"v7_lines_{lang}.webp", lossless=False))
    data = {"set": "data/lines_v7_cross/val (held out, generator v7, seed 63)", "pick": "random.Random(7).sample of the "
            "horizontal lines with a cross, 10 lines", "readers": {k: v[0] for k, v in READERS.items()},
            "lines": [{"image": p, "kind": kind, "text": t, **{r: readings[r][i] for r in READERS}}
                      for i, (p, t, kind) in enumerate(rows)], "images": made}
    (OUT / "v7_lines.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for m in made:
        print(m)
    for line in data["lines"]:
        print(" | ".join([line["text"]] + [line[r][0] for r in READERS]))


if __name__ == "__main__":
    main()
