"""Write the demo's built-in example images: real lines from the synth_v3 val split,
decoded like training does (cv2.IMREAD_GRAYSCALE) and saved as lossless grayscale PNG, so
the browser gets exactly the pixels the Python pipeline sees. Also writes examples.json.

Usage (from the repo root): .venv\\Scripts\\python.exe web\\test\\make_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hxh_ocr.glyphs import to_romaji  # noqa: E402

VAL = ROOT / "data" / "synth_v3" / "val"
OUT = ROOT / "web" / "examples"
PICK = [  # val rows, chosen for variety (style, background, polarity, length)
    "images/0000/00000007.jpg",  # clean, noisy paper
    "images/0000/00000033.jpg",  # brush
    "images/0000/00000091.jpg",  # light text on dark
    "images/0000/00000192.jpg",  # hand, cloudy background
    "images/0000/00000212.jpg",  # clean thin strokes
    "images/0000/00000099.jpg",  # long line: wider than 256 px at height 32 (squeezed)
]


def main():
    labels = {}
    with open(VAL / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            labels[p[0]] = p[1]
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for i, rel in enumerate(PICK, 1):
        img = cv2.imread(str(VAL / rel), cv2.IMREAD_GRAYSCALE)
        name = f"ex{i}.png"
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        items.append({"file": name, "text": labels[rel], "romaji": to_romaji(labels[rel]),
                      "source": f"data/synth_v3/val/{rel}", "size": [img.shape[1], img.shape[0]]})
        print(name, img.shape[::-1], (OUT / name).stat().st_size, "B")
    (OUT / "examples.json").write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
