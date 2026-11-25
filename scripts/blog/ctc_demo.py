"""Build the blog's CTC animation data from one example line (see ctc_demo.mjs).

    .venv\\Scripts\\python.exe scripts\\blog\\ctc_demo.py [--example ex6.png]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="ex6.png")
    args = ap.parse_args()
    ex = json.loads((ROOT / "web/examples/examples.json").read_text(encoding="utf-8"))
    label = next(e["text"] for e in ex if e["file"] == args.example)
    img = cv2.imread(str(ROOT / "web/examples" / args.example), cv2.IMREAD_GRAYSCALE)
    with tempfile.TemporaryDirectory() as td:
        pgm = Path(td) / args.example.replace(".png", ".pgm")
        h, w = img.shape
        pgm.write_bytes(f"P5\n{w} {h}\n255\n".encode() + img.tobytes())
        subprocess.run(["node", str(ROOT / "scripts/blog/ctc_demo.mjs"), str(pgm), label], check=True)
    strip_pgm = ROOT / "web/blog/img/ctc_strip.pgm"
    strip = cv2.imread(str(strip_pgm), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(ROOT / "web/blog/img/ctc_strip.png"), strip, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    strip_pgm.unlink()
    print("wrote web/blog/data/ctc.json and web/blog/img/ctc_strip.png", strip.shape)


if __name__ == "__main__":
    main()
