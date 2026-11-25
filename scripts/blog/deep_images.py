"""Images for the research deep dive (web/blog/deep-dive.html).

Crops the generator / evaluation contact sheets in previews/ (multi-MB PNGs, not shipped) into
small WebP files under web/blog/img/deep/, each at most 150 KB. Re-run after regenerating the
previews:

    .venv/Scripts/python.exe scripts/blog/deep_images.py

Sheet geometry (rust/hxh-synth/src/preview.rs): `contact_sheet` puts pages top-left in
480 x 480 cells with an 8 px gap (4 columns -> 1960 px); `targets_sheet` draws rows of
4 panels, 512 px on the long side, 8 px gap (2088 px wide).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "previews"
OUT = ROOT / "web" / "blog" / "img" / "deep"
MAX_BYTES = 150 * 1024

CELL, GAP = 480, 8


def rows_of_contact_sheet(im: Image.Image, rows: int) -> Image.Image:
    return im.crop((0, 0, im.width, GAP + rows * (CELL + GAP)))


def save_webp(im: Image.Image, name: str, width: int) -> dict:
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    for q in range(84, 30, -4):
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=q, method=6)
        if buf.tell() <= MAX_BYTES:
            break
    (OUT / name).write_bytes(buf.getvalue())
    return {"file": name, "width": im.width, "height": im.height, "bytes": buf.tell(), "quality": q}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    # pages_v3 and pages_foreign: the first two rows (8 pages) of the 16-page preview
    for name in ("pages_v3", "pages_foreign"):
        im = Image.open(SRC / f"{name}.png")
        made.append({"source": f"previews/{name}.png (rows 1-2)",
                     **save_webp(rows_of_contact_sheet(im, 2), f"{name}.webp", 1400)})
    # DBNet targets: first row of pages_v2_targets (image + quads | kernel | threshold map | mask)
    im = Image.open(SRC / "pages_v2_targets.png")
    made.append({"source": "previews/pages_v2_targets.png (row 1)",
                 **save_webp(im.crop((0, 0, im.width, GAP + 512 + GAP)), "targets.webp", 1400)})
    # detector on validation pages: first row (two pages with their readings) of det_val.png
    im = Image.open(SRC / "det_val.png")
    made.append({"source": "previews/det_val.png (row 1)",
                 **save_webp(im.crop((0, 0, im.width, 800)), "det_val.webp", 1400)})
    (OUT / "images.json").write_text(json.dumps({"about": __doc__.strip().splitlines()[0], "images": made},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    for m in made:
        print(m)


if __name__ == "__main__":
    main()
