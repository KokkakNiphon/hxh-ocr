"""Write a synthetic Hunter-alphabet line dataset to disk.

    python scripts/generate.py --out data/synth_v1 --train 200000 --val 5000 --workers 12

Layout:
    <out>/meta.json, <out>/vocab.json
    <out>/<split>/labels.tsv        path<TAB>label<TAB>h|v<TAB>style
    <out>/<split>/images/NNNN/XXXXXXXX.jpg

Every sample is seeded from (seed, split, index), so a dataset can be regenerated
or extended exactly, independent of the worker count.
"""
from __future__ import annotations

import os

# one BLAS/OpenMP thread per worker process; set before numpy is imported (inherited by workers)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.glyphs import VOCAB  # noqa: E402
from hxh_ocr.synth import SynthConfig, load_bg_images, make_sample  # noqa: E402
from hxh_ocr.text import TextSampler, load_corpus  # noqa: E402

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
_W: dict = {}


def _init(size, max_len, vertical_prob, bg_dir, corpus):
    cv2.setNumThreads(1)
    cfg = SynthConfig(size=size, max_len=max_len, vertical_prob=vertical_prob)
    if bg_dir:
        cfg.bg_images = load_bg_images(bg_dir)
    _W["cfg"] = cfg
    _W["sampler"] = TextSampler(load_corpus(corpus) if corpus else None)


def _work(job):
    out_dir, split, seed, start, end, quality, gray = job
    lines = []
    for i in range(start, end):
        rng = np.random.default_rng([seed, SPLIT_IDS[split], i])
        img, label, meta = make_sample(rng, _W["cfg"], _W["sampler"])
        rel = f"images/{i // 10000:04d}/{i:08d}.jpg"
        p = Path(out_dir) / split / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if gray else img[..., ::-1]
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        p.write_bytes(enc.tobytes())
        lines.append((i, f"{rel}\t{label}\t{'v' if meta['vertical'] else 'h'}\t{meta['style']}"))
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=200_000)
    ap.add_argument("--val", type=int, default=5_000)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--size", type=int, default=64, help="line thickness in px")
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--vertical-prob", type=float, default=0.3)
    ap.add_argument("--bg-dir", default=None, help="folder of real background images (optional)")
    ap.add_argument("--corpus", default=None, help="extra kana/katakana text, one entry per line (optional)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--gray", action="store_true", help="save single-channel grayscale JPEGs (~3x smaller)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(VOCAB, ensure_ascii=False), encoding="utf-8")
    meta = {"args": vars(args), "vocab_size": len(VOCAB), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "glyph_source": "Hunterpedia File:Hunterjap.png traced; も ring variant from omniglot.com chart",
            "splits": {}}

    with Pool(args.workers, initializer=_init,
              initargs=(args.size, args.max_len, args.vertical_prob, args.bg_dir, args.corpus)) as pool:
        for split in ("train", "val", "test"):
            n = getattr(args, split)
            if n <= 0:
                continue
            jobs = [(str(out), split, args.seed, a, min(n, a + args.chunk), args.quality, args.gray)
                    for a in range(0, n, args.chunk)]
            t0, done, rows = time.time(), 0, []
            for lines in pool.imap_unordered(_work, jobs):
                rows.extend(lines)
                done += len(lines)
                el = time.time() - t0
                print(f"\r{split}: {done}/{n}  {done / max(el, 1e-6):.0f}/s", end="", flush=True)
            rows.sort()
            (out / split / "labels.tsv").write_text("\n".join(r for _, r in rows) + "\n", encoding="utf-8")
            el = time.time() - t0
            print(f"\r{split}: {n} samples in {el:.1f}s ({n / el:.0f}/s)")
            meta["splits"][split] = {"n": n, "seconds": round(el, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done:", out.resolve())


if __name__ == "__main__":
    main()
