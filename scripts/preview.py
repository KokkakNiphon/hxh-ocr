"""Contact sheet of random synthetic samples with their labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.fonts import label_font  # noqa: E402
from hxh_ocr.glyphs import to_romaji  # noqa: E402
from hxh_ocr.synth import SynthConfig, load_bg_images, make_sample  # noqa: E402
from hxh_ocr.text import TextSampler  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="horizontal samples")
    ap.add_argument("--nv", type=int, default=8, help="vertical samples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--out", default="previews/samples.png")
    args = ap.parse_args()

    cfg = SynthConfig()
    if args.bg_dir:
        cfg.bg_images = load_bg_images(args.bg_dir)
    sampler = TextSampler()
    rng = np.random.default_rng(args.seed)
    font = label_font(18)

    hs, vs = [], []
    while len(hs) < args.n or len(vs) < args.nv:
        img, text, meta = make_sample(rng, cfg, sampler)
        (vs if meta["vertical"] else hs).append((img, text, meta))
    hs, vs = hs[: args.n], vs[: args.nv]

    row_h = cfg.size + 30
    col_w = args.width // 2
    rows = (len(hs) + 1) // 2
    v_h = max((im.shape[0] for im, _, _ in vs), default=0)
    sheet = Image.new("RGB", (args.width, rows * row_h + v_h + 60), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (im, text, meta) in enumerate(hs):
        x, y = (i % 2) * col_w + 10, (i // 2) * row_h + 4
        pim = Image.fromarray(im)
        if pim.width > col_w - 20:
            pim = pim.resize((col_w - 20, int(pim.height * (col_w - 20) / pim.width)))
        sheet.paste(pim, (x, y))
        raw = meta["raw"]
        shown = text if raw == text else f"{raw} → {text}"
        d.text((x, y + cfg.size + 2), f"{shown}  [{to_romaji(text)}]  {meta['style']}", fill=(190, 20, 20), font=font)
    x, y0 = 10, rows * row_h + 10
    for im, text, meta in vs:
        sheet.paste(Image.fromarray(im), (x, y0))
        d.text((x, y0 + im.shape[0] + 2), str(len(text)), fill=(190, 20, 20), font=font)
        x += cfg.size + 24
        if x > args.width - cfg.size:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out, sheet.size)
    print("vertical labels:", " | ".join(t for _, t, _ in vs))


if __name__ == "__main__":
    main()
