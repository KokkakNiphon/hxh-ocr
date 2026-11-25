"""Python reference data for web/test/verify.mjs (run with the project .venv: cv2 + numpy).

Writes <out>/reference.json (metadata) and <out>/reference.bin (raw u8 blobs):
- resize: random u8 images and target sizes (down, integer, fractional, up, mixed)
  with cv2.resize(..., INTER_AREA) outputs, to check web/ocr.js resizeArea bit for bit;
- gray: random RGB images with cv2.cvtColor(RGB2GRAY), to check rgbaToGray;
- lines: horizontal val lines (grayscale JPEG decoded by cv2.IMREAD_GRAYSCALE, i.e. what
  training saw) with the calibration preprocessing scripts/espdl_to_hxhm.resize_pad_u8
  (= experiments/esp_ppq_spike/spike_common.preprocess before normalisation) and
  hxh_ocr.glyphs.to_romaji of the label;
- pages: two "photos of a sign" (several lines rendered with the training glyph renderer
  on paper, slightly rotated, blurred, JPEG), with the glyph boxes of each line, for the
  hand-drawn-box test in verify.mjs --robust. Also saved as out/page*.jpg.

Usage (from the repo root):
  .venv\\Scripts\\python.exe web\\test\\make_reference.py [--data data\\synth_v3\\val] [--n 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from espdl_to_hxhm import resize_pad_u8  # noqa: E402
from hxh_ocr.glyphs import VOCAB, strokes_bbox, to_romaji  # noqa: E402
from hxh_ocr.render import SS, _translate, draw_strokes, layout, sample_style  # noqa: E402
from hxh_ocr.synth import _to_px  # noqa: E402

# strip size of the demo model (web/models/model.json), e.g. 48x768
_MJ = json.loads((ROOT / "web" / "models" / "model.json").read_text(encoding="utf-8-sig"))
H, W = int(_MJ["input"]["height"]), int(_MJ["input"]["width"])


class Blob:
    def __init__(self):
        self.parts, self.size = [], 0

    def add(self, arr: np.ndarray) -> list[int]:
        b = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
        off = self.size
        self.parts.append(b)
        self.size += len(b)
        return [off, len(b)]


def resize_cases(rng: np.random.Generator, blob: Blob) -> list[dict]:
    sizes = []
    for _ in range(120):  # arbitrary down/up/mixed
        sizes.append(tuple(int(v) for v in rng.integers(1, 160, 4)))
    for _ in range(80):  # line-like downscales to height 32
        sh = int(rng.integers(33, 400)); sw = int(rng.integers(8, 1500))
        sizes.append((sw, sh, int(min(256, max(1, round(sw * 32 / sh)))), 32))
    for f in [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (5, 5), (8, 8), (1, 2), (2, 1)]:
        for _ in range(6):  # exact integer factors (resizeAreaFast)
            dw, dh = int(rng.integers(1, 90)), int(rng.integers(1, 40))
            sizes.append((dw * f[0], dh * f[1], dw, dh))
    for _ in range(40):  # upscale to height 32 (small crops) and squeeze + upscale
        sh = int(rng.integers(4, 32)); sw = int(rng.integers(4, 700))
        sizes.append((sw, sh, int(min(256, max(1, round(sw * 32 / sh)))), 32))
    out = []
    for i, (sw, sh, dw, dh) in enumerate(sizes):
        if i % 3 == 0:  # smooth content + edges, like text
            y, x = np.mgrid[0:sh, 0:sw]
            img = 128 + 100 * np.sin(x / (1 + i % 7)) * np.cos(y / (2 + i % 5))
            img = np.where((x + y) % 11 < 3, 20, img)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = rng.integers(0, 256, (sh, sw), dtype=np.uint8)
        dst = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        out.append({"sw": sw, "sh": sh, "dw": dw, "dh": dh, "src": blob.add(img), "dst": blob.add(dst)})
    return out


def gray_cases(rng: np.random.Generator, blob: Blob) -> list[dict]:
    out = []
    for _ in range(8):
        h, w = (int(v) for v in rng.integers(1, 120, 2))
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        out.append({"w": w, "h": h, "rgb": blob.add(rgb), "gray": blob.add(gray)})
    return out


PAGE_TEXT = ["はんたー しけん", "くじらじま", "ぐりーど あいらんど", "ごん きるあ"]


def render_page(texts, seed, em, angle, blur, out: Path, W=1400, lh=1.9, x0=120, y0=160) -> dict:
    """Lines of HxH script on paper, like a phone photo of a sign."""
    rng = np.random.default_rng(seed)
    st = sample_style(rng)
    H = int(y0 * 2 + lh * em * len(texts))
    big = np.zeros((H * SS, W * SS), np.uint8)
    boxes = []
    for i, t in enumerate(texts):
        s = layout(t, st, rng, False)
        _to_px(s, em)
        bb = strokes_bbox(s)
        _translate(s, x0 - bb[0], y0 + i * lh * em - bb[1])
        boxes.append([float(v) for v in strokes_bbox(s)])
        draw_strokes(big, s, st.sw * em, st.brush, rng)
    alpha = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy, xx = np.mgrid[0:H, 0:W]
    paper = 0.90 - 0.10 * (xx / W) - 0.05 * (yy / H) + rng.normal(0, 0.015, (H, W))
    img = np.clip((paper * (1 - alpha) + 0.12 * alpha) * 255, 0, 255).astype(np.uint8)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bgr[..., 0] = np.clip(bgr[..., 0].astype(int) - 12, 0, 255)  # warm paper tint
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"file": out.name, "texts": texts, "boxes": boxes, "w": W, "h": H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth_v3" / "val"))
    ap.add_argument("--n", type=int, default=300, help="horizontal val lines")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    blob = Blob()

    rows = []
    with open(data / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[2] == "h":
                rows.append(p)
    rows = rows[args.offset: args.offset + args.n]
    lines = []
    for rel, label, _, style, *_ in rows:
        gray = cv2.imread(str(data / rel), cv2.IMREAD_GRAYSCALE)
        ref = resize_pad_u8(gray, H, W)
        h, w = gray.shape
        lines.append({"rel": rel, "label": label, "style": style, "w": w, "h": h,
                      "natural_w": max(1, round(w * H / h)), "gray": blob.add(gray), "ref": blob.add(ref),
                      "romaji": to_romaji(label)})

    pages = []
    for seed, em, angle, blur in [(3, 90, 1.5, 1.0), (11, 60, -2.5, 1.4)]:
        p = render_page(PAGE_TEXT, seed, em, angle, blur, out / f"page{len(pages) + 1}.jpg")
        bgr = cv2.imread(str(out / p["file"]), cv2.IMREAD_COLOR)  # decoded like a browser would
        p["gray"] = blob.add(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        pages.append(p)

    romaji_vocab = {t: to_romaji(t) for t in VOCAB}
    extra = ["せーぶ", "ーあ", "あ ー", "ーー", "じぢづ", "きゃ"]
    meta = {
        "cv2": cv2.__version__, "data": str(data), "height": H, "width": W,
        "resize": resize_cases(rng, blob), "gray": gray_cases(rng, blob), "lines": lines, "pages": pages,
        "romaji": {**romaji_vocab, **{s: to_romaji(s) for s in extra}},
    }
    (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (out / "reference.bin").write_bytes(b"".join(blob.parts))
    print(f"wrote {out / 'reference.json'} ({len(lines)} lines, {len(meta['resize'])} resize cases, "
          f"{blob.size / 1e6:.1f} MB blobs), OpenCV {cv2.__version__}")


if __name__ == "__main__":
    main()
