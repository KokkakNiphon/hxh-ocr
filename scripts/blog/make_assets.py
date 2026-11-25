"""Regenerate the blog article's image and data assets: web/blog/img/ and web/blog/data/.

    .venv\\Scripts\\python.exe scripts\\blog\\make_assets.py [--only pipeline,synth,glyphs,charts]
                                                       [--runs 7] [--conf-sweep | --conf-sweep-log FILE]

pipeline  web/examples/page1.jpg through the exact browser pipeline in node
          (scripts/blog/pipeline_assets.mjs: web/ocr.js readPage, web/hxh_infer.wasm, web/models/*):
          img/pipeline/01_page .. 05_boxes.webp, crop_N.webp, crops.json, pipeline.json
synth     img/synth/lines_v4, generator_evolution, pages_v2, vertical (.webp + .json) from data/
glyphs    data/glyphs.json (from rust/hxh-synth/assets/glyphs.json, checked against hxh_ocr/glyphs.py)
          and data/glyph_test.svg drawn from it
charts    data/charts.json: training curves from runs/*/log.csv and tables from docs/research/JOURNAL.md.
          The confidence-filter sweep is measured with web/test/page_e2e.mjs --conf-sweep
          (--conf-sweep runs it, about 2 min of one CPU core; --conf-sweep-log parses a saved run;
          without either the table already in charts.json is kept).

Needs numpy, opencv-python, pillow (with WebP) and node >= 20. Everything is deterministic except the
timings in pipeline.json and the sweep's ms columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hxh_ocr import glyphs as G  # noqa: E402

IMG = ROOT / "web" / "blog" / "img"
DATA = ROOT / "web" / "blog" / "data"
BLUE = (0x23, 0x40, 0xD8)    # horizontal lines
PURPLE = (0x7A, 0x4F, 0xD6)  # vertical columns
MAX_KB = 150
GAP = 8  # px between grid cells in the synth images (about 6 CSS px at their display size)


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def save_webp(im: Image.Image, path: Path, quality: int = 82, lossless: bool = False, max_kb: int = MAX_KB) -> Path:
    """WebP at `quality`, stepping the quality down (not below 60) if the file is over `max_kb`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    q = quality
    while True:
        im.save(path, "WEBP", quality=q, method=6, lossless=lossless)
        if lossless or path.stat().st_size <= max_kb * 1024 or q <= 60:
            break
        q -= 6
    if path.stat().st_size > max_kb * 1024:
        print(f"  warning: {rel(path)} is {path.stat().st_size / 1024:.0f} KB")
    return path


def write_json(obj, path: Path, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def font(size: int, bold: bool = True):
    for p in (["C:/Windows/Fonts/arialbd.ttf"] if bold else []) + ["C:/Windows/Fonts/arial.ttf",
                                                                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def tint(base_rgb: np.ndarray, color, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return (base_rgb * (1 - a) + np.array(color, np.float32) * a).round().clip(0, 255).astype(np.uint8)


def draw_quad(rgb: np.ndarray, quad, k: float, color, width: int):
    """Quad in page px (pixel centres at integers) onto an image scaled by k, anti-aliased."""
    q = (np.asarray(quad, np.float64).reshape(4, 2) + 0.5) * k - 0.5
    cv2.polylines(rgb, [np.round(q * 16).astype(np.int32)], True, color, width, cv2.LINE_AA, 4)


def quad_iou(a, b) -> float:
    A = np.asarray(a, np.float32).reshape(4, 2)
    B = np.asarray(b, np.float32).reshape(4, 2)
    # same winding for both (y down: shoelace > 0 is clockwise on screen)
    A = A if cv2.contourArea(A, oriented=True) > 0 else A[::-1].copy()
    B = B if cv2.contourArea(B, oriented=True) > 0 else B[::-1].copy()
    inter, _ = cv2.intersectConvexConvex(A, B)
    ua = cv2.contourArea(A) + cv2.contourArea(B) - inter
    return float(inter / ua) if ua > 0 else 0.0


# ================================================================== 1. pipeline stages
def pipeline(runs: int):
    out = IMG / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("crop_*.webp"):
        old.unlink()
    page_path = ROOT / "web" / "examples" / "page1.jpg"
    bgr = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    # canvas RGBA -> rgbaToGray in the browser is OpenCV's RGB2GRAY bit for bit (web/ocr.js)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "page.pgm").write_bytes(f"P5\n{W} {H}\n255\n".encode() + gray.tobytes())
        subprocess.run(["node", str(ROOT / "scripts" / "blog" / "pipeline_assets.mjs"), "--page", str(td / "page.pgm"),
                        "--out", str(td / "out"), "--runs", str(runs)], check=True, cwd=ROOT)
        o = td / "out"
        meta = json.loads((o / "meta.json").read_text(encoding="utf-8"))
        S = meta["letterbox"]["size"]
        mh, mw, mc, exp = (meta["maps"][k] for k in ("height", "width", "channels", "exp"))
        lb = np.fromfile(o / "letterbox.u8", np.uint8).reshape(S, S)
        maps = np.fromfile(o / "maps.i8", np.int8).reshape(mh, mw, mc)
        strips = {l["index"]: np.fromfile(o / l["strip"]["file"], np.uint8).reshape(l["strip"]["height"], l["strip"]["width"])
                  for l in meta["lines"]}

    # ---- 01 page, 02 letterbox
    save_webp(Image.open(page_path).convert("RGB"), out / "01_page.webp")
    save_webp(Image.fromarray(lb, "L"), out / "02_letterbox.webp")

    # ---- 03 kernel probability, 04 orientation (map stride 2 -> letterbox, half-pixel aligned)
    scale = 2.0 ** exp
    prob = sigmoid(maps[..., 0].astype(np.float32) * scale)
    vert = sigmoid(maps[..., 1].astype(np.float32) * scale)
    prob_up = cv2.resize(prob, (S, S), interpolation=cv2.INTER_LINEAR)
    vert_up = cv2.resize(vert, (S, S), interpolation=cv2.INTER_LINEAR)
    faded = 255.0 - (255.0 - lb.astype(np.float32)) * 0.55  # page faded so the overlay reads
    base = np.repeat(faded[..., None], 3, axis=2)
    a_kernel = np.clip((prob_up - 0.05) / 0.95, 0, 1) ** 0.75 * 0.92
    save_webp(Image.fromarray(tint(base, BLUE, a_kernel)), out / "03_kernel.webp")
    a_text = smoothstep(0.15, 0.45, prob_up) * 0.92
    col = (np.array(BLUE, np.float32) * (1 - vert_up[..., None]) + np.array(PURPLE, np.float32) * vert_up[..., None])
    a = a_text[..., None]
    orient_rgb = (base * (1 - a) + col * a).round().clip(0, 255).astype(np.uint8)
    save_webp(Image.fromarray(orient_rgb), out / "04_orient.webp")

    # ---- line status as in web/app.js: no glyph read -> dropped; conf < min_conf -> hidden
    min_conf = meta["detector"]["min_conf"]
    lines = meta["lines"]
    badge = 0
    for l in lines:
        if not l["text"].strip():
            l["status"] = "empty"
        elif l["confidence"] < min_conf:
            l["status"] = "low_confidence"
        else:
            l["status"] = "shown"
        if l["status"] == "shown":
            badge += 1
            l["badge"] = badge
        else:
            l["badge"] = None

    # ---- ground truth of the example page (generated page from pages_v2 val)
    ex = next(e for e in json.loads((ROOT / "web" / "examples" / "examples.json").read_text(encoding="utf-8"))
              if e["file"] == page_path.name)
    src = ex["source"]  # data/pages_v2/val/images/...
    split_dir = ROOT / src.split("/images/")[0]
    img_rel = "images/" + src.split("/images/")[1]
    gt = None
    for row in (split_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(row)
        if r["image"] == img_rel:
            gt = r
            break
    assert gt and gt["width"] == W and gt["height"] == H, "ground truth not found for the example page"
    gtq = [[c for pt in g["quad"] for c in pt] for g in gt["lines"]]
    pairs = sorted(((quad_iou(g, l["quad"]), gi, li) for gi, g in enumerate(gtq) for li, l in enumerate(lines)), reverse=True)
    g_used, l_used = {}, {}
    for v, gi, li in pairs:
        if v >= 0.5 and gi not in g_used and li not in l_used:
            g_used[gi], l_used[li] = li, (gi, v)
    for li, l in enumerate(lines):
        if li in l_used:
            gi, v = l_used[li]
            g = gt["lines"][gi]
            l["ground_truth"] = {"text": g["text"], "romaji": G.to_romaji(g["text"]), "orient": g["orient"],
                                 "iou": round(v, 3), "exact": l["text"] == g["text"]}
        else:
            l["ground_truth"] = None
    missed = [{"text": g["text"], "orient": g["orient"], "kind": g["kind"], "quad": gtq[gi]}
              for gi, g in enumerate(gt["lines"]) if gi not in g_used]

    # ---- 05 boxes on the page, at 2x (falls back to a smaller scale if the file gets too big)
    for k in (2.0, 1.5, 1.0):
        rgb = cv2.cvtColor(cv2.resize(bgr, (round(W * k), round(H * k)), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        shown = [l for l in lines if l["status"] == "shown"]
        for l in shown:
            draw_quad(rgb, l["quad"], k, PURPLE if l["vertical"] else BLUE, max(1, round(2 * k)))
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        r = 9 * k
        f = font(round(11.5 * k))
        for l in shown:
            q = (np.asarray(l["quad"]).reshape(4, 2) + 0.5) * k - 0.5
            u = q[1] - q[0]
            v = q[3] - q[0]
            u /= np.linalg.norm(u) or 1
            v /= np.linalg.norm(v) or 1
            c = q[0] - (u + v) / math.sqrt(2) * r * 0.75  # just outside the q0 corner
            cx = float(np.clip(c[0], r + 1, im.width - r - 2))
            cy = float(np.clip(c[1], r + 1, im.height - r - 2))
            colr = PURPLE if l["vertical"] else BLUE
            d.ellipse([cx - r - k, cy - r - k, cx + r + k, cy + r + k], fill=(255, 255, 255))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colr)
            d.text((cx, cy + 0.3 * k), str(l["badge"]), fill=(255, 255, 255), font=f, anchor="mm")
        p = save_webp(im, out / "05_boxes.webp", max_kb=10_000)
        if p.stat().st_size <= MAX_KB * 1024:
            break
    boxes_scale = k

    # ---- 06 reader inputs (prep.pixels): one lossless file per line the reader found glyphs in
    featured = pick_featured(lines)
    crops = []
    for l in lines:
        if l["status"] == "empty":
            l["crop_file"] = None
            continue
        name = f"crop_{l['index']}.webp"
        save_webp(Image.fromarray(strips[l["index"]], "L"), out / name, lossless=True)
        l["crop_file"] = name
        crops.append({"file": name, "line": l["index"], "badge": l["badge"], "status": l["status"],
                      "featured": l["index"] in featured, "vertical": l["vertical"], "text": l["text"],
                      "romaji": l["romaji"], "confidence": l["confidence"],
                      "ground_truth": l["ground_truth"]["text"] if l["ground_truth"] else None,
                      "width": l["strip"]["width"], "height": l["strip"]["height"],
                      "content_width": l["strip"]["content_width"], "squeezed": l["strip"]["squeezed"]})
    write_json({
        "about": "The reader's exact input for each page line: web/ocr.js prepareLine output (prep.pixels), "
                 "48 x 768 u8, lossless. Vertical columns come turned 90 degrees counter-clockwise (top glyph on "
                 "the left), as the vertical reader sees them. Columns right of content_width are padding "
                 "(the last column repeated). Lines in reading order; `featured` marks a mixed set of 6.",
        "source": "web/examples/page1.jpg via scripts/blog/pipeline_assets.mjs",
        "crops": crops,
    }, out / "crops.json")

    shown = [l for l in lines if l["status"] == "shown"]
    counts = {
        "detected": len(lines),
        "empty_dropped": sum(l["status"] == "empty" for l in lines),
        "low_confidence_hidden": sum(l["status"] == "low_confidence" for l in lines),
        "shown": len(shown),
        "ground_truth_lines": len(gt["lines"]),
        "ground_truth_vertical": sum(g["orient"] == "v" for g in gt["lines"]),
        "matched_ground_truth": len(g_used),
        "shown_matched": sum(l["ground_truth"] is not None for l in shown),
        "shown_exact": sum(bool(l["ground_truth"] and l["ground_truth"]["exact"]) for l in shown),
    }
    pipe = {
        "about": "One example page through exactly the browser pipeline (web/ocr.js readPage) in node: letterbox -> "
                 "int8 detector -> Rust post-processing -> padded crops -> int8 horizontal / vertical readers. "
                 "Quads are page px (pixel centres at integers), clockwise from the top-left in reading orientation "
                 "(vertical: q0 is the column's top-right). status: shown | low_confidence (hidden by det.json "
                 "min_conf) | empty (no glyph read; the app drops it). badge = number drawn in 05_boxes.webp. "
                 "ground_truth: the generated page's own label, matched at quad IoU >= 0.5.",
        "source": {"page": rel(page_path), "generated_from": src, "wasm": "web/hxh_infer.wasm",
                   "models": ["web/models/det.hxhm", "web/models/model.hxhm", "web/models/model_v.hxhm"],
                   "params": "web/models/det.json (params, pad, min_conf)", "script": "scripts/blog/pipeline_assets.mjs"},
        "images": {"05_boxes.webp": {"scale": boxes_scale,
                                     "note": "image px = (page px + 0.5) * scale - 0.5"},
                   "02_letterbox.webp": {"note": "page drawn at content_width x content_height in the top-left, rest 255"},
                   "03_kernel.webp": {"note": "sigmoid(channel 0 logit) as alpha of #2340d8 over the letterbox (faded)"},
                   "04_orient.webp": {"note": "where the kernel probability is high: #2340d8 horizontal, #7a4fd6 vertical, "
                                              "mixed by sigmoid(channel 1 logit)"}},
        **{k: meta[k] for k in ("page", "letterbox", "maps", "detector", "readers", "timings_ms", "machine")},
        "line_count": len(lines),
        "counts": counts,
        "lines": lines,
        "missed_ground_truth": missed,
    }
    write_json(pipe, out / "pipeline.json")
    print(f"pipeline: {counts}")


def pick_featured(lines, n=6):
    """Up to n shown lines, half vertical, read exactly if possible, longest first, in reading order."""
    ok = [l for l in lines if l["status"] == "shown"]
    exact = [l for l in ok if l["ground_truth"] and l["ground_truth"]["exact"]]
    pool = exact if len(exact) >= n else ok
    v = sorted((l for l in pool if l["vertical"]), key=lambda l: -len(l["text"]))[: n // 2]
    h = sorted((l for l in pool if not l["vertical"]), key=lambda l: -len(l["text"]))[: n - len(v)]
    return {l["index"] for l in v + h}


# ================================================================== 2. synthetic data grids
def read_rows(split_dir: Path):
    rows = []
    for line in (split_dir / "labels.tsv").read_text(encoding="utf-8").splitlines():
        if line:
            path, label, orient, style = line.split("\t")
            rows.append({"path": path, "label": label, "orient": orient, "style": style})
    return rows


def load_rgba(split_dir: Path, path: str) -> np.ndarray:
    im = cv2.imread(str(split_dir / path), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def candidates(split_dir: Path, rows, rng, orient, wmin, wmax, limit=400):
    """Shuffled rows of one orientation whose image width (horizontal) / height (vertical) is in range."""
    out = []
    for i in rng.permutation(len(rows)):
        r = rows[i]
        if r["orient"] != orient:
            continue
        im = load_rgba(split_dir, r["path"])
        size = im.shape[1] if orient == "h" else im.shape[0]
        if wmin <= size <= wmax:
            out.append({**r, "img": im, "index": int(i)})
            if len(out) >= limit:
                break
    return out


def interleave_styles(cands, styles=("clean", "hand", "brush")):
    by = {s: [c for c in cands if c["style"] == s] for s in styles}
    out = []
    while any(by.values()):
        for s in styles:
            if by[s]:
                out.append(by[s].pop(0))
    return out


def pack_rows(pool, row_w, n_total, per_row_max=None, gap=GAP):
    """Greedy best-fit rows: each row starts with the next sample, then adds the widest that still fits."""
    rows, used = [], set()
    pool = list(pool)
    while sum(len(r) for r in rows) < n_total:
        start = next((c for c in pool if id(c) not in used), None)
        if start is None:
            break
        row, w = [start], start["img"].shape[1]
        used.add(id(start))
        while per_row_max is None or len(row) < per_row_max:
            room = row_w - w - gap
            # look a little ahead in the (style-interleaved) pool, keep the mix, prefer the widest fit
            ahead = [c for c in pool if id(c) not in used][:12]
            fit = [c for c in ahead if c["img"].shape[1] <= room]
            if not fit:
                break
            c = max(fit, key=lambda c: c["img"].shape[1])
            row.append(c)
            used.add(id(c))
            w += gap + c["img"].shape[1]
        rows.append(row)
    return rows


def compose_rows(rows, row_w, row_h, gap=GAP):
    H = len(rows) * row_h + (len(rows) - 1) * gap
    canvas = np.zeros((H, row_w, 4), np.uint8)
    boxes = []
    for ri, row in enumerate(rows):
        x, y = 0, ri * (row_h + gap)
        for c in row:
            h, w = c["img"].shape[:2]
            canvas[y:y + h, x:x + w] = c["img"]
            boxes.append({"row": ri, "x": x, "y": y, "w": w, "h": h, "c": c})
            x += w + gap
    return canvas, boxes


def sample_entry(b, split_dir):
    c = b["c"]
    return {"label": c["label"], "romaji": G.to_romaji(c["label"]), "style": c["style"], "orient": c["orient"],
            "file": rel(split_dir / c["path"]), "box": [b["x"], b["y"], b["w"], b["h"]]}


def synth():
    out = IMG / "synth"
    out.mkdir(parents=True, exist_ok=True)

    # ---- lines_v4: ~24 horizontal val lines, styles interleaved, 64 px rows (shown at 48 CSS px)
    d4 = ROOT / "data" / "synth_v4" / "val"
    rng = np.random.default_rng(4)
    pool = interleave_styles(candidates(d4, read_rows(d4), rng, "h", 150, 600))
    row_w = 960
    rows = pack_rows(pool, row_w, 24)
    canvas, boxes = compose_rows(rows, row_w, 64)
    save_webp(Image.fromarray(canvas, "RGBA"), out / "lines_v4.webp")
    write_json({
        "about": "Validation lines of synth_v4 (generator v4) as the reader trains on them: horizontal, 64 px tall "
                 "(display the 960 px wide image at 720 CSS px for 48 px rows). Gaps are transparent. Samples in reading order (row by row); "
                 "box = [x, y, w, h] in image px.",
        "image": "lines_v4.webp", "width": canvas.shape[1], "height": canvas.shape[0], "row_height": 64,
        "source": rel(d4) + " (seed 3)",
        "samples": [sample_entry(b, d4) for b in boxes],
    }, out / "lines_v4.json")
    print(f"synth: lines_v4 {len(boxes)} samples in {len(rows)} rows")

    # ---- generator_evolution: one row of ~4 samples per dataset version
    versions = [
        ("v1", "synth_v1", "First generator: traced glyphs, clean/hand/brush styles, colour, 100k + 5k lines."),
        ("v2", "synth_v2", "Same generator, 500k + 10k lines, grayscale JPEG q90."),
        ("v3", "synth_v3", "Tighter crops (bigger glyphs), binarisation only when the text survives, "
                           "neighbour lines mostly outside the crop."),
        ("v4", "synth_v4", "Official glyph forms (left-pointing ま, narrow せ, ring も), art beside the line ends, "
                           "heavier strokes, more drop shadows, varied dakuten placement."),
    ]
    all_rows, meta_rows = [], []
    for tag, name, desc in versions:
        dd = ROOT / "data" / name / "val"
        if not dd.exists():
            continue
        rng = np.random.default_rng(11)
        pool = interleave_styles(candidates(dd, read_rows(dd), rng, "h", 150, 228, limit=60))
        row = pack_rows(pool, row_w, 4, per_row_max=4)[0]
        all_rows.append(row)
        meta_rows.append((tag, name, desc, dd))
    canvas, boxes = compose_rows(all_rows, row_w, 64)
    save_webp(Image.fromarray(canvas, "RGBA"), out / "generator_evolution.webp")
    write_json({
        "about": "One row per synthetic line-set version (validation split, horizontal lines, 64 px tall, shown "
                 "at 48 CSS px: 960 px image at 720 CSS px). Row labels are not drawn; descriptions follow docs/research/JOURNAL.md. "
                 "box = [x, y, w, h] in image px.",
        "image": "generator_evolution.webp", "width": canvas.shape[1], "height": canvas.shape[0], "row_height": 64,
        "rows": [{"name": tag, "dataset": rel(dd), "description": desc,
                  "samples": [sample_entry(b, dd) for b in boxes if b["row"] == ri]}
                 for ri, (tag, name, desc, dd) in enumerate(meta_rows)],
    }, out / "generator_evolution.json")
    print(f"synth: generator_evolution rows {[m[0] for m in meta_rows]}")

    pages_grid()
    vertical()


def pages_grid():
    out = IMG / "synth"
    dv = ROOT / "data" / "pages_v2" / "val"
    pages = [json.loads(l) for l in (dv / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rng = np.random.default_rng(5)
    order = rng.permutation(len(pages))
    ok = [int(i) for i in order
          if 1.25 <= pages[i]["width"] / pages[i]["height"] <= 1.8 and 10 <= len(pages[i]["lines"]) <= 30
          and any(l["orient"] == "v" for l in pages[i]["lines"])]
    chosen = []
    for kind in ("bubble", "block", "sign", "cell", "caption", "free"):  # one page featuring each kind
        i = next((i for i in ok if i not in chosen and any(l["kind"] == kind for l in pages[i]["lines"])), None)
        if i is not None:
            chosen.append(i)
    chosen = chosen[:6]
    cw, ch, gap, cols = 600, 420, 12, 3
    rows_n = math.ceil(len(chosen) / cols)
    canvas = np.zeros((rows_n * ch + (rows_n - 1) * gap, cols * cw + (cols - 1) * gap, 4), np.uint8)
    entries = []
    for n, i in enumerate(chosen):
        p = pages[i]
        im = cv2.cvtColor(cv2.imread(str(dv / p["image"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        k = min(cw / p["width"], ch / p["height"])
        w, h = round(p["width"] * k), round(p["height"] * k)
        im = np.ascontiguousarray(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
        for l in p["lines"]:
            draw_quad(im, [c for pt in l["quad"] for c in pt], k, PURPLE if l["orient"] == "v" else BLUE, 2)
        x0 = (n % cols) * (cw + gap) + (cw - w) // 2
        y0 = (n // cols) * (ch + gap) + (ch - h) // 2
        canvas[y0:y0 + h, x0:x0 + w, :3] = im
        canvas[y0:y0 + h, x0:x0 + w, 3] = 255
        entries.append({"file": rel(dv / p["image"]), "width": p["width"], "height": p["height"],
                        "box": [x0, y0, w, h], "scale": round(k, 5), "lines": len(p["lines"]),
                        "vertical": sum(l["orient"] == "v" for l in p["lines"]),
                        "kinds": sorted({l["kind"] for l in p["lines"]}),
                        "texts": [l["text"] for l in p["lines"]]})
    save_webp(Image.fromarray(canvas, "RGBA"), out / "pages_v2.webp")
    write_json({
        "about": "Generated detector training pages (pages_v2 val) with their ground-truth line quads: horizontal "
                 "#2340d8, vertical #7a4fd6. Each page is fitted into a 600 x 420 cell (transparent around it); "
                 "box = [x, y, w, h] of the page in the image, scale = image px per page px.",
        "image": "pages_v2.webp", "width": canvas.shape[1], "height": canvas.shape[0],
        "source": rel(dv) + " (hxh-synth pages, seed 1)", "pages": entries,
    }, out / "pages_v2.json")
    print(f"synth: pages_v2 pages {chosen}")


def vertical():
    """A few synth_v4 vertical columns upright (left) and the same columns turned 90 deg CCW (right),
    which is how the vertical reader sees them (LineDataset(orient='v'))."""
    out = IMG / "synth"
    d4 = ROOT / "data" / "synth_v4" / "val"
    rng = np.random.default_rng(6)
    pool = interleave_styles(candidates(d4, read_rows(d4), rng, "v", 230, 420, limit=60))[:5]
    colgap, rowgap, panel_gap = 16, GAP, 56
    left_w = sum(c["img"].shape[1] for c in pool) + colgap * (len(pool) - 1)
    rot = [np.ascontiguousarray(cv2.rotate(c["img"], cv2.ROTATE_90_COUNTERCLOCKWISE)) for c in pool]
    right_w = max(r.shape[1] for r in rot)
    right_h = sum(r.shape[0] for r in rot) + rowgap * (len(rot) - 1)
    H = max(max(c["img"].shape[0] for c in pool), right_h)
    canvas = np.zeros((H, left_w + panel_gap + right_w, 4), np.uint8)
    samples, x = [], 0
    for c in pool:
        h, w = c["img"].shape[:2]
        canvas[0:h, x:x + w] = c["img"]
        samples.append({"label": c["label"], "romaji": G.to_romaji(c["label"]), "style": c["style"],
                        "file": rel(d4 / c["path"]), "upright_box": [x, 0, w, h]})
        x += w + colgap
    x0, y = left_w + panel_gap, (H - right_h) // 2
    for s, r in zip(samples, rot):
        h, w = r.shape[:2]
        canvas[y:y + h, x0:x0 + w] = r
        s["rotated_box"] = [x0, y, w, h]
        y += h + rowgap
    save_webp(Image.fromarray(canvas, "RGBA"), out / "vertical.webp")
    write_json({
        "about": "synth_v4 vertical columns: upright on the left (read top to bottom), and the same columns turned "
                 "90 degrees counter-clockwise on the right (top glyph on the left), the vertical reader's input. "
                 "Columns are 64 px wide. Gaps are transparent; boxes = [x, y, w, h] in image px.",
        "image": "vertical.webp", "width": canvas.shape[1], "height": canvas.shape[0],
        "left_panel": [0, 0, left_w, H], "right_panel": [left_w + panel_gap, 0, right_w, H],
        "source": rel(d4), "samples": samples,
    }, out / "vertical.json")
    print(f"synth: vertical {[s['label'] for s in samples]}")


# ================================================================== 3. glyph data
GOJUON_ROWS = [("", "あいうえお"), ("k", "かきくけこ"), ("s", "さしすせそ"), ("t", "たちつてと"), ("n", "なにぬねの"),
               ("h", "はひふへほ"), ("m", "まみむめも"), ("y", "や・ゆ・よ"), ("r", "らりるれろ"), ("w", "わ・・・を"),
               ("n", "ん"), ("-", "ー")]
FORM_INFO = {
    "ま": [("left-pointing (Ƹ)", "2011 anime chart and printed guidebook chart (official)"),
           ("right-pointing (Σ)", "Hunterpedia fan chart; kept as a rare variant")],
    "せ": [("wide", "Hunterpedia trace; printed guidebook chart"),
           ("narrow (\"4\")", "2011 anime chart; Omniglot")],
    "も": [("filled dot", "Hunterpedia trace; printed guidebook chart"),
           ("hollow ring", "2011 anime chart; Omniglot")],
}
TRACED = ("traced", "Hunterpedia chart File:Hunterjap.png, traced by hand")


def r4(v: float) -> float:
    return round(float(v), 4)


def bbox_of(strokes):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for s in strokes:
        if s["type"] == "dot":
            (cx, cy), r = s["c"], s["r"]
            xs0.append(cx - r); xs1.append(cx + r); ys0.append(cy - r); ys1.append(cy + r)
        else:
            p = np.asarray(s["pts"])
            xs0.append(p[:, 0].min()); xs1.append(p[:, 0].max()); ys0.append(p[:, 1].min()); ys1.append(p[:, 1].max())
    return [r4(min(xs0)), r4(min(ys0)), r4(max(xs1)), r4(max(ys1))]


def mark_for(marks, mark, bbox, scale=1.0):
    m = marks[mark]
    return {"type": "dot", "c": [r4(bbox[2] + m["dx"] * scale), r4(bbox[3] + m["dy"] * scale)], "r": r4(m["r"] * scale),
            "hollow": m["hollow"], "width_scale": m["width_scale"]}


def glyphs():
    src = ROOT / "rust" / "hxh-synth" / "assets" / "glyphs.json"
    g = json.loads(src.read_text(encoding="utf-8"))
    order = [c for _, row in GOJUON_ROWS for c in row if c != "・"]
    assert sorted(order) == sorted(c for c in G.BASE_CHARS), "base glyph set changed"
    pos = {c: [ri, ci] for ri, (_, row) in enumerate(GOJUON_ROWS) for ci, c in enumerate(row) if c != "・"}
    out_glyphs = {}
    for ch in order:
        e = g["glyphs"][ch]
        assert e["base"] == ch and e["mark"] is None
        weights = e["weights"] or [1.0 / len(e["forms"])] * len(e["forms"])
        forms = []
        for fi, form in enumerate(e["forms"]):
            strokes = []
            for s in form:
                assert s["width_scale"] == 1.0 and not s["hollow"], ch
                if s["kind"] == "dot":
                    strokes.append({"type": "dot", "c": [r4(s["pts"][0][0]), r4(s["pts"][0][1])], "r": r4(s["r"])})
                else:
                    strokes.append({"type": "line", "pts": [[r4(x), r4(y)] for x, y in s["pts"]], "closed": s["closed"]})
            # check against the Python source of truth
            ref = G.GLYPHS[ch].variants[fi]
            for a, b in zip(strokes, ref):
                pa = np.asarray(a["c"] if a["type"] == "dot" else a["pts"], float).reshape(-1, 2)
                assert pa.shape == b.pts.shape and np.abs(pa - b.pts).max() < 1e-3, ch
            name, origin = FORM_INFO[ch][fi] if ch in FORM_INFO else TRACED
            forms.append({"name": name, "source": origin, "default": fi == 0, "weight": round(weights[fi], 4),
                          "bbox": bbox_of(strokes), "strokes": strokes})
        out_glyphs[ch] = {"romaji": G.to_romaji(ch), "pos": pos[ch], "forms": forms}
        if ch == "ー":
            out_glyphs[ch]["note"] = "long-vowel mark: lengthens the vowel before it (romaji puts a macron on it)"

    composed = {}
    for ch in G.GLYPH_CHARS:
        e = g["glyphs"][ch]
        if e["mark"]:
            composed[ch] = {"base": e["base"], "mark": e["mark"], "romaji": G.to_romaji(ch)}
            for fi, f in enumerate(out_glyphs[e["base"]]["forms"]):  # placement rule == glyphs.mark_stroke
                m = mark_for(g["marks"], e["mark"], f["bbox"])
                ref = G.mark_stroke(e["mark"], G.strokes_bbox(G.GLYPHS[e["base"]].variants[fi]), 1.0)
                assert abs(m["c"][0] - ref.pts[0, 0]) < 1e-3 and abs(m["c"][1] - ref.pts[0, 1]) < 1e-3, ch
    marks = {k: {**v, "drawn_as": "filled dot" if not v["hollow"] else "ring"} for k, v in g["marks"].items()}
    data = {
        "about": "Hunter x Hunter alphabet glyph geometry for drawing in JS. Exported from "
                 "rust/hxh-synth/assets/glyphs.json (itself exported from hxh_ocr/glyphs.py by "
                 "scripts/export_glyphs.py) by scripts/blog/make_assets.py. Format: web/blog/data/README.md.",
        "units": "em; each form is centred on its own bounding box; x to the right, y down",
        "stroke_width_em": g["ref_stroke_em"],
        "order": order,
        "rows": [{"consonant": c, "kana": [k if k != "・" else None for k in row]} for c, row in GOJUON_ROWS],
        "glyphs": out_glyphs,
        "marks": marks,
        "mark_rule": {
            "bbox": "bounding box of the base form's stroke geometry: line points and dot centre +- r "
                    "(stroke width not included); see each form's `bbox`",
            "center": "[bbox.x1 + dx * s, bbox.y1 + dy * s] with s = 1 for the chart geometry (bbox.x1 = right, "
                      "bbox.y1 = bottom since y points down)",
            "radius": "r * s",
            "draw": "dakuten: filled disc of radius max(r, stroke_width / 2); handakuten: ring of radius "
                    "max(r, stroke_width * width_scale), line width stroke_width * width_scale",
            "generator_jitter": "generator v4 moves the mark sideways to (0.55..1.35) x dx (keeping a gap of at least "
                                "stroke_width / 2 + mark radius + 0.03 em) and down/up by U(-0.06, 0.03) em",
        },
        "draw": {
            "line": "polyline through pts (closed: join the last point to the first), width stroke_width, round "
                    "caps and joins",
            "dot": "filled disc of radius max(r, stroke_width / 2)",
        },
        "composed": composed,
        "vocab": g["vocab"],
        "small_to_full": g["small_to_full"],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "glyphs.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    glyph_svg(data)
    n_forms = sum(len(v["forms"]) for v in out_glyphs.values())
    print(f"glyphs: {len(out_glyphs)} base glyphs ({n_forms} forms), {len(composed)} composed")


def glyph_svg(data):
    """Every vocabulary glyph (and every form) drawn from glyphs.json alone, laid out like
    previews/glyph_sheet.png (10 columns, VOCAB order, alternative forms labelled "alt")."""
    em, cols = 60.0, 10
    cw, chh = em * 1.7, em * 1.85
    sw = data["stroke_width_em"]
    cells = []
    for ch in G.GLYPH_CHARS:
        c = data["composed"].get(ch)
        base = c["base"] if c else ch
        for fi, f in enumerate(data["glyphs"][base]["forms"]):
            strokes = list(f["strokes"])
            if c:
                strokes.append(mark_for(data["marks"], c["mark"], f["bbox"]))
            cells.append((ch, fi, strokes))
    rows = math.ceil(len(cells) / cols)
    W, H = cols * cw, rows * chh
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
           f'font-family="Yu Gothic, Meiryo, sans-serif">',
           '<rect width="100%" height="100%" fill="#fff"/>']

    def f2(v):
        return f"{v:.2f}".rstrip("0").rstrip(".")

    for i, (ch, fi, strokes) in enumerate(cells):
        x0, y0 = (i % cols) * cw, (i // cols) * chh
        bb = bbox_of(strokes)
        cx, cy = x0 + cw / 2, y0 + em * 0.1 + em * 0.65
        ox, oy = cx - (bb[0] + bb[2]) / 2 * em, cy - (bb[1] + bb[3]) / 2 * em
        out.append(f'<rect x="{f2(x0 + .5)}" y="{f2(y0 + .5)}" width="{f2(cw - 1)}" height="{f2(chh - 1)}" fill="none" stroke="#e1e1e1"/>')
        out.append(f'<g transform="translate({f2(ox)} {f2(oy)})" fill="none" stroke="#000" stroke-width="{f2(sw * em)}" '
                   f'stroke-linecap="round" stroke-linejoin="round">')
        for s in strokes:
            if s["type"] == "line":
                pts = " ".join(f"{f2(x * em)},{f2(y * em)}" for x, y in s["pts"])
                out.append(f'<{"polygon" if s["closed"] else "polyline"} points="{pts}"/>')
            elif s.get("hollow"):
                w = sw * s["width_scale"]
                out.append(f'<circle cx="{f2(s["c"][0] * em)}" cy="{f2(s["c"][1] * em)}" r="{f2(max(s["r"], w) * em)}" '
                           f'stroke-width="{f2(w * em)}"/>')
            else:
                out.append(f'<circle cx="{f2(s["c"][0] * em)}" cy="{f2(s["c"][1] * em)}" r="{f2(max(s["r"], sw / 2) * em)}" '
                           f'fill="#000" stroke="none"/>')
        out.append("</g>")
        tag = f"{ch} {G.to_romaji(ch)}" + (" alt" if fi else "")
        out.append(f'<text x="{f2(x0 + 8)}" y="{f2(y0 + chh - 8)}" font-size="{f2(em * 0.22)}" fill="#c81e1e">{tag}</text>')
    out.append("</svg>")
    (DATA / "glyph_test.svg").write_text("\n".join(out) + "\n", encoding="utf-8")


# ================================================================== 4. chart data
J = "docs/research/JOURNAL.md"


def src(entry: str, section: str | None = None) -> dict:
    return {"source": section or entry, "entry": entry, "file": J}


def curve(run: str, label: str, y: str, scale: float = 100.0, extra=None, max_points=60):
    rows = list(csv.DictReader((ROOT / "runs" / run / "log.csv").open(encoding="utf-8")))
    step = max(1, math.ceil(len(rows) / max_points))
    keep = rows[::step]
    if keep[-1] is not rows[-1]:
        keep.append(rows[-1])
    s = {"id": run, "label": label, "file": f"runs/{run}/log.csv",
         "points": [[int(r["step"]), round(float(r[y]) * scale, 3)] for r in keep]}
    for key, col in (extra or {}).items():
        s[key] = [[int(r["step"]), round(float(r[col]) * scale, 3)] for r in keep]
    return s


def col(key, label, unit=None):
    return {"key": key, "label": label, **({"unit": unit} if unit else {})}


E_RUNS = "2026-11-24: first GPU training runs, and why they stalled"
E_DEMO = "2026-11-24: mobile browser demo (built by a parallel agent)"
E_QAT = "2026-11-24: quantisation-aware training (built by a parallel agent)"
E_REAL = "2026-11-24: first real test set (scraped by a cheap agent) and real-image CER"
E_SIMD = "2026-11-25: WASM SIMD kernels for hxh-infer (built by a parallel agent)"
E_PAGES = "2026-11-25: page-mode generator in Rust (`rust/hxh-synth`), milestone step 1"
E_VERT = "2026-11-25: detector training pages, the vertical recogniser, romaji macrons"
E_DET = "2026-11-25: the line detector `det_v1` (built by a parallel agent)"
E_INTERIM = "2026-11-25: full-page reading in the demo (interim detector)"
E_FINAL = "2026-11-25: final detector in the demo, live camera mode, milestone done"

SWEEP_RE = re.compile(r"min reading confidence ([\d.]+)\s+P ([\d.]+)% R ([\d.]+)% F1 ([\d.]+)% orient ([\d.]+)% \| "
                      r"CER h ([\d.]+)% v ([\d.]+)% matched ([\d.]+)% \| exact h ([\d.]+)% v ([\d.]+)% \| "
                      r"([\d.]+) ms/page")


def conf_sweep_table(text: str | None, previous: dict | None):
    det = json.loads((ROOT / "web" / "models" / "det.json").read_text(encoding="utf-8"))
    base = {
        "id": "confidence_filter",
        "title": "Hiding page lines the reader is unsure of (mean character confidence below a threshold)",
        "unit": "%",
        "source": "web/models/det.json (min_conf_note); full sweep measured with web/test/page_e2e.mjs --conf-sweep "
                  "(not in the journal)",
        "entry": None, "file": "web/models/det.json",
        "recorded_note": det.get("min_conf_note"),
        "chosen_threshold": det.get("min_conf"),
        "eval": "pages_v2 val[900:1000] (100 generated pages, held out from detector training), browser pipeline, "
                "every model int8; CER counts every ground-truth line (missed = deleted)",
        "columns": [col("min_conf", "min confidence"), col("precision", "precision", "%"), col("recall", "recall", "%"),
                    col("f1", "F1", "%"), col("cer_h", "CER h", "%"), col("cer_v", "CER v", "%"),
                    col("cer_matched", "CER of matched lines", "%"), col("exact_h", "exact lines h", "%"),
                    col("exact_v", "exact lines v", "%")],
    }
    if text:
        rows = []
        for m in SWEEP_RE.finditer(text):
            v = [float(x) for x in m.groups()]
            rows.append(dict(zip(["min_conf", "precision", "recall", "f1", "orient", "cer_h", "cer_v", "cer_matched",
                                  "exact_h", "exact_v", "ms_page"], v)))
        for r in rows:
            r.pop("orient"); r.pop("ms_page")
        assert rows, "no sweep rows found"
        return {**base, "rows": rows}
    if previous and previous.get("rows"):
        return {**base, "rows": previous["rows"]}
    return {**base, "rows": [], "note": "run make_assets.py --conf-sweep to fill the rows"}


def charts(conf_text: str | None):
    path = DATA / "charts.json"
    prev = None
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        prev = next((t for t in old.get("tables", []) if t["id"] == "confidence_filter"), None)

    cer_note = ("val CER on a fixed 2,000-line subset of the run's own val set, every 1,000 steps; EMA weights with "
                "BatchNorm statistics re-estimated before each eval")
    curves = [
        {"id": "recognizers_v4", "title": "Line readers on synth_v4: validation CER while training (fp32)",
         "x": "step", "y": "val CER", "unit": "%", "note": cer_note, **src(E_PAGES, "Round 3 results (synth_v4, 30k steps)"),
         "series": [curve("v4_mnv4s_h48", "MobileNetV4-small, 48 px, horizontal", "val_cer", extra={"acc": "val_acc"}),
                    curve("v4_mnv4s_h48_vert", "MobileNetV4-small, 48 px, vertical columns", "val_cer", extra={"acc": "val_acc"}),
                    curve("v4_mnv4s3_h32", "3-stage MobileNetV4-small, 32 px, horizontal", "val_cer", extra={"acc": "val_acc"})]},
        {"id": "detector_det_v1", "title": "Line detector det_v1: validation F1 while training (fp32)",
         "x": "step", "y": "val F1", "unit": "%",
         "note": "300 pages_v2 val pages, every 2,000 steps; a line counts as found at quad IoU >= 0.5", **src(E_DET),
         "series": [curve("det_v1", "det_v1 (DBNet-lite on MobileNetV4-small)", "val_f1",
                          extra={"precision": "val_p", "recall": "val_r"})]},
        {"id": "ablation_round1", "title": "Round 1 (synth_v1 + v2, 32 px): pretrained vs from scratch",
         "x": "step", "y": "val CER", "unit": "%", "note": cer_note + " (synth_v2 val)",
         **src(E_RUNS, "Ablation result (synth_v1 + v2, 32 px, 30k steps × batch 128, BN-fixed eval)"),
         "series": [curve("mnv4s_h32", "MobileNetV4-small, ImageNet-pretrained", "val_cer", extra={"acc": "val_acc"}),
                    curve("mnv4s_scratch_h32", "MobileNetV4-small, from scratch", "val_cer", extra={"acc": "val_acc"}),
                    curve("tiny_h32", "tiny CNN, from scratch", "val_cer", extra={"acc": "val_acc"})]},
        {"id": "round2_v3", "title": "Round 2 (synth_v3): 32 px vs 48 px vs 3 stages",
         "x": "step", "y": "val CER", "unit": "%", "note": cer_note + " (synth_v3 val)",
         **src(E_DEMO, "Round 2 summary (synth_v3, 30k steps, fp32, full v3 val, 6,998 lines)"),
         "series": [curve("v3_mnv4s_h48", "MobileNetV4-small, 48 px", "val_cer", extra={"acc": "val_acc"}),
                    curve("v3_mnv4s_h32", "MobileNetV4-small, 32 px", "val_cer", extra={"acc": "val_acc"}),
                    curve("v3_mnv4s3_h32", "3-stage MobileNetV4-small, 32 px", "val_cer", extra={"acc": "val_acc"})]},
    ]

    hxhm_kb = {f: round((ROOT / "web" / "models" / f).stat().st_size / 1024, 1)
               for f in ("model.hxhm", "model_v.hxhm", "det.hxhm")}
    tables = [
        {"id": "ablation_round1", "title": "Pretrained MobileNetV4-small vs from scratch vs a tiny CNN (round 1)",
         "unit": "%", **src(E_RUNS, "Ablation result (synth_v1 + v2, 32 px, 30k steps × batch 128, BN-fixed eval)"),
         "note": "all ~323k params, ~30M MACs at 32x256; 30k steps x batch 128",
         "columns": [col("model", "model"), col("wall_s", "wall time", "s"), col("best_val_cer_2k", "best val CER (2k subset)", "%"),
                     col("val_cer", "full synth_v2 val CER", "%"), col("exact", "exact lines", "%"),
                     col("cer_step2k", "CER at step 2k", "%")],
         "rows": [{"model": "mnv4s, ImageNet-pretrained", "wall_s": 1727, "best_val_cer_2k": 1.54, "val_cer": 1.16, "exact": 94.4, "cer_step2k": 3.0},
                  {"model": "mnv4s, from scratch", "wall_s": 1757, "best_val_cer_2k": 1.78, "val_cer": 1.47, "exact": 93.0, "cer_step2k": 7.5},
                  {"model": "tiny, from scratch", "wall_s": 1218, "best_val_cer_2k": 1.77, "val_cer": 1.27, "exact": 93.8, "cer_step2k": 5.1}]},
        {"id": "round2_input_size", "title": "32 px vs 48 px vs 3 stages (round 2, synth_v3)", "unit": "%",
         **src(E_DEMO, "Round 2 summary (synth_v3, 30k steps, fp32, full v3 val, 6,998 lines)"),
         "note": "fp32, full v3 val (6,998 lines); S3 estimate at the ~0.7 GMAC/s Espressif measured for ESP-DL (an "
                 "estimate, not a measurement)",
         "columns": [col("model", "model"), col("params", "params"), col("macs", "MACs (input)"), col("s3_estimate", "S3 estimate"),
                     col("train_s", "train time", "s"), col("val_cer", "val CER", "%"), col("exact", "exact lines", "%")],
         "rows": [{"model": "mnv4s 48 px", "params": "323k", "macs": "77M (48x384)", "s3_estimate": "~110 ms", "train_s": 1977, "val_cer": 0.64, "exact": 96.5},
                  {"model": "mnv4s 32 px", "params": "323k", "macs": "30M (32x256)", "s3_estimate": "~43 ms", "train_s": 1921, "val_cer": 0.84, "exact": 95.4},
                  {"model": "mnv4s3 32 px (3 stages)", "params": "96k", "macs": "15M (32x256)", "s3_estimate": "~22 ms", "train_s": 1342, "val_cer": 1.14, "exact": 94.0}]},
        {"id": "cross_eval_v3_v4", "title": "Generator v3 vs v4: the 48 px reader evaluated on both val sets", "unit": "%",
         **src(E_PAGES, "Round 3 results (synth_v4, 30k steps)"),
         "note": "fp32, 3,000 lines of each val set; v4 val has the official glyph forms, art beside lines and bold strokes",
         "columns": [col("model", "48 px model"), col("on_v3", "on v3 val (clean)", "%"), col("on_v4", "on v4 val", "%")],
         "rows": [{"model": "trained on v3", "on_v3": 0.63, "on_v4": 3.07},
                  {"model": "trained on v4", "on_v3": 0.69, "on_v4": 0.90}]},
        {"id": "round3", "title": "Round 3 (synth_v4, 30k steps): best validation CER", "unit": "%",
         **src(E_PAGES, "Round 3 results (synth_v4, 30k steps)"),
         "columns": [col("model", "model"), col("train_s", "train time", "s"), col("best_val_cer_2k", "best val CER (v4, 2k subset)", "%")],
         "rows": [{"model": "mnv4s 48 px", "train_s": 1963, "best_val_cer_2k": 0.95},
                  {"model": "mnv4s3 32 px (3-stage)", "train_s": 1354, "best_val_cer_2k": 1.61}]},
        {"id": "qat_first", "title": "First QAT test: round-1 mnv4s 32 px, plain int8 PTQ vs QAT", "unit": "%",
         **src(E_QAT), "note": "QAT for 4k steps on synth_v3 (611 s); PTQ = ESP-PPQ KL calibration",
         "columns": [col("model", "model"), col("fp32", "fp32 CER", "%"), col("int8_ptq", "int8 PTQ", "%"),
                     col("int8_ptq_tqt", "int8 PTQ + TQT", "%")],
         "rows": [{"model": "baseline, v1 holdout (512 lines, 32x256)", "fp32": 3.68, "int8_ptq": 5.88, "int8_ptq_tqt": 4.87},
                  {"model": "QAT, v1 holdout", "fp32": 3.92, "int8_ptq": 4.06, "int8_ptq_tqt": 5.28},
                  {"model": "baseline, v3 holdout (1,500 lines)", "fp32": 3.16, "int8_ptq": 8.25, "int8_ptq_tqt": None},
                  {"model": "QAT, v3 holdout", "fp32": 3.58, "int8_ptq": 3.68, "int8_ptq_tqt": None}]},
        {"id": "qat_int8", "title": "int8 vs float per model after QAT, with sizes", "unit": "%",
         **src(E_REAL, "QAT on the round-2 models (`scripts/qat_pipeline.sh`)"),
         "note": "CER on 512 held-out lines of the model's data. 'float' for QAT models = the QAT weights run without "
                 "fake quantisation (not what they were trained for); for the PTQ+TQT model it is plain fp32. "
                 "Round-3 rows: entry 'Round 3 results (synth_v4, 30k steps)'; vertical row: entry "
                 f"'{E_VERT}'. hxhm_kb_file = size of the deployed file in web/models/ (measured, not from the journal).",
         "columns": [col("model", "model (deploy input)"), col("data", "data"), col("float", "float CER", "%"),
                     col("int8", "int8 CER", "%"), col("bit_exact", "Rust bit-exact"), col("espdl_kb", ".espdl", "KB"),
                     col("hxhm_kb", ".hxhm", "KB"), col("hxhm_kb_file", ".hxhm file in web/models", "KB"),
                     col("qat_time", "QAT wall time")],
         "rows": [{"model": "v3 mnv4s 32 px, PTQ+TQT, no QAT (32x512)", "data": "v3", "float": 0.68, "int8": 1.16, "bit_exact": "512/512", "espdl_kb": 400, "hxhm_kb": 331, "qat_time": None},
                  {"model": "QAT mnv4s3 32 px (32x512)", "data": "v3", "float": 1.78, "int8": 0.82, "bit_exact": "512/512", "espdl_kb": 141, "qat_time": "8 min"},
                  {"model": "QAT mnv4s 32 px (32x512)", "data": "v3", "float": 1.69, "int8": 0.96, "bit_exact": "512/512", "espdl_kb": 403, "qat_time": "11 min"},
                  {"model": "QAT mnv4s 48 px (48x768)", "data": "v3", "float": 1.64, "int8": 0.51, "bit_exact": "512/512", "espdl_kb": 402, "qat_time": "13 min"},
                  {"model": "qat_v4_mnv4s3_h32 (32x512)", "data": "v4", "float": 2.06, "int8": 1.75, "bit_exact": "yes", "espdl_kb": 141},
                  {"model": "qat_v4_mnv4s_h48 (48x768), the demo's horizontal reader", "data": "v4", "float": 1.31, "int8": 0.84, "bit_exact": "yes", "espdl_kb": 402, "hxhm_kb": 338, "hxhm_kb_file": hxhm_kb["model.hxhm"]},
                  {"model": "qat_v4_mnv4s_h48_vert (48x768), the demo's vertical reader", "data": "v4 vertical columns", "int8": 0.71, "bit_exact": "512/512", "espdl_kb": 401, "hxhm_kb_file": hxhm_kb["model_v.hxhm"], "qat_time": "626 s"}]},
        {"id": "vertical_reader", "title": "Horizontal vs vertical reader (synth_v4)", "unit": "%", **src(E_VERT),
         "note": "the two val sets differ in content, so this is not a like-for-like comparison",
         "columns": [col("model", "model"), col("val_cer_fp32", "val CER (fp32, 2k subset)", "%"),
                     col("int8_holdout", "int8 hold-out CER (512)", "%"), col("espdl_kb", ".espdl", "KB")],
         "rows": [{"model": "horizontal qat_v4_mnv4s_h48", "val_cer_fp32": 0.95, "int8_holdout": 0.84, "espdl_kb": 402},
                  {"model": "vertical qat_v4_mnv4s_h48_vert", "val_cer_fp32": 0.55, "int8_holdout": 0.71, "espdl_kb": 401}]},
        {"id": "wasm_speed", "title": "Reader speed per line in wasm: old scalar demo build vs SIMD kernels", "unit": "ms",
         **src(E_SIMD), "note": "node 24, Ryzen 7 3700X, one thread, CPU shared; about 16 GMAC/s with SIMD",
         "columns": [col("model", "model (MACs)"), col("macs_m", "MACs", "M"), col("old_ms", "old demo build", "ms"),
                     col("simd_ms", "SIMD", "ms"), col("speedup", "speedup")],
         "rows": [{"model": "48 px, 48x768", "macs_m": 154, "old_ms": 65.5, "simd_ms": 9.5, "speedup": "~7x"},
                  {"model": "32 px, 32x512", "macs_m": 60.5, "old_ms": 25.6, "simd_ms": 3.7, "speedup": "~7x"},
                  {"model": "3-stage, 32x512", "macs_m": 30.5, "old_ms": 14.9, "simd_ms": 2.0, "speedup": "~7.6x"}]},
        {"id": "detector_int8", "title": "Detector det_v1: int8 PTQ vs PTQ + TQT vs fp32", "unit": "mixed", **src(E_DET),
         "note": "ESP-PPQ, esp32s3 rules, KL calibration on 64 pages; 100 held-out pages. TQT took 28 min on the CPU. "
                 f"Deployed .hxhm v2: 367 KB (journal; file in web/models: {hxhm_kb['det.hxhm']} KB), 62 ms per page in node "
                 "with SIMD (25 GMAC/s), scalar fallback 683 ms (entry: " + E_FINAL + ")",
         "columns": [col("version", "version"), col("f1", "F1", "fraction"), col("cer_h", "CER h", "%"), col("cer_v", "CER v", "%")],
         "rows": [{"version": "fp32", "f1": 0.916, "cer_h": 8.6, "cer_v": 16.4},
                  {"version": "PTQ", "f1": 0.904, "cer_h": 9.7, "cer_v": 16.5},
                  {"version": "PTQ + TQT (deployed)", "f1": 0.914, "cer_h": 9.1, "cer_v": 18.2, "note": "CER v on 222 lines, noisy"}]},
        {"id": "detector_full_val", "title": "Detector det_v1 on the full pages_v2 val set (fp32)", "unit": "mixed", **src(E_DET),
         "note": "1,000 pages, 16,748 lines, quad IoU >= 0.5",
         "columns": [col("metric", "metric"), col("value", "value")],
         "rows": [{"metric": "precision", "value": 0.935}, {"metric": "recall", "value": 0.909}, {"metric": "F1", "value": 0.922},
                  {"metric": "orientation accuracy (%)", "value": 98.9},
                  {"metric": "recall, horizontal", "value": 0.921}, {"metric": "recall, vertical", "value": 0.816},
                  {"metric": "end-to-end CER, horizontal (%)", "value": 9.1}, {"metric": "  matched lines only (%)", "value": 4.9},
                  {"metric": "end-to-end CER, vertical (%)", "value": 15.3}, {"metric": "  matched lines only (%)", "value": 4.7},
                  {"metric": "reader alone on ground-truth quads, horizontal (%)", "value": 3.8},
                  {"metric": "reader alone on ground-truth quads, vertical (%)", "value": 3.2}]},
        {"id": "recall_by_thickness", "title": "Detector recall by line thickness", "unit": "%", **src(E_DET),
         "note": "det_v1 fp32, full pages_v2 val (1,000 pages); thickness in px of the 640 x 640 letterbox",
         "columns": [col("thickness", "line thickness at 640", "px"), col("recall", "recall", "%")],
         "rows": [{"thickness": "6-9", "recall": 73}, {"thickness": "9-14", "recall": 91},
                  {"thickness": "14-25", "recall": 95}, {"thickness": ">= 25", "recall": 96}]},
        {"id": "unclip_margin_final", "title": "Post-processing and crop margins, final detector (browser pipeline)",
         "unit": "%", **src(E_FINAL),
         "note": "web/test/page_e2e.mjs --sweep, pages_v2 val[900:1000], every model int8; margins along / across in "
                 "units of the line's thickness",
         "columns": [col("post", "post-processing"), col("margins", "margins (along / across)"), col("f1", "F1", "%"),
                     col("cer_h", "CER h", "%"), col("cer_v", "CER v", "%"), col("cer_matched", "CER of matched lines", "%")],
         "rows": [{"post": "unclip 1.5", "margins": "0.3 / 0.2", "f1": 91.0, "cer_h": 9.2, "cer_v": 18.2, "cer_matched": 4.4, "chosen": True},
                  {"post": "unclip 1.0", "margins": "0.5 / 0.3", "f1": 90.3, "cer_h": 10.7, "cer_v": 17.7, "cer_matched": 4.4},
                  {"post": "exact inverse, r = 0.4", "margins": "0.3 / 0.2", "f1": 89.5, "cer_h": 11.0, "cer_v": 26.5, "cer_matched": 5.2}]},
        {"id": "unclip_margin_interim", "title": "Post-processing and crop margins, interim detector (step 14k, int8)",
         "unit": "%", **src(E_INTERIM),
         "note": "pages_v2 val[900:1000] (1,657 lines); the 0 / 0 row shows that tight crops triple the reader's error",
         "columns": [col("post", "post-processing"), col("margins", "margins (along / across)"), col("f1", "F1", "%"),
                     col("cer_h", "CER h", "%"), col("cer_v", "CER v", "%"), col("cer_matched", "CER of matched lines", "%")],
         "rows": [{"post": "exact inverse, r = 0.4", "margins": "0.3 / 0.2", "f1": 87.7, "cer_h": 14.0, "cer_v": 28.7, "cer_matched": 5.6},
                  {"post": "unclip 1.0", "margins": "0.3 / 0.3", "f1": 89.0, "cer_h": 11.6, "cer_v": 20.4, "cer_matched": 4.9},
                  {"post": "unclip 1.5", "margins": "0.3 / 0.2", "f1": 89.6, "cer_h": 10.3, "cer_v": 20.6, "cer_matched": 4.4, "chosen": True},
                  {"post": "unclip 1.5", "margins": "0 / 0", "f1": 89.6, "cer_h": 23.1, "cer_v": 26.2, "cer_matched": 17.3},
                  {"post": "unclip 2.0", "margins": "0.3 / 0.2", "f1": 86.6, "cer_h": 11.7, "cer_v": 28.2, "cer_matched": 4.9}]},
        conf_sweep_table(conf_text, prev),
    ]
    write_json({
        "about": "Chart data for the blog. Curves: from runs/<run>/log.csv (every logged eval; <= 60 points each). "
                 "Tables: numbers copied from docs/research/JOURNAL.md; `source` is the journal heading they sit under "
                 "(`entry` = the dated entry). Percentages are plain numbers (1.16 = 1.16%). null = not reported.",
        "curves": curves, "tables": tables,
    }, path)
    print(f"charts: {len(curves)} curve groups, {len(tables)} tables")


# ================================================================== main
def report():
    files = sorted(p for d in (IMG, DATA) for p in d.rglob("*") if p.is_file())
    total = 0
    for p in files:
        kb = p.stat().st_size / 1024
        total += kb
        flag = "  <-- over 150 KB" if kb > MAX_KB and p.suffix in (".webp", ".png") else ""
        print(f"  {kb:7.1f} KB  {rel(p)}{flag}")
    print(f"  {total:7.1f} KB  total ({len(files)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="pipeline,synth,glyphs,charts")
    ap.add_argument("--runs", type=int, default=7, help="readPage runs for the timings (median)")
    ap.add_argument("--conf-sweep", action="store_true", help="run web/test/page_e2e.mjs --conf-sweep (about 2 min)")
    ap.add_argument("--conf-sweep-log", help="parse a saved page_e2e.mjs --conf-sweep output instead")
    args = ap.parse_args()
    only = set(args.only.split(","))
    if "pipeline" in only:
        pipeline(args.runs)
    if "synth" in only:
        synth()
    if "glyphs" in only:
        glyphs()
    if "charts" in only:
        text = None
        if args.conf_sweep_log:
            text = Path(args.conf_sweep_log).read_text(encoding="utf-8", errors="replace")
        elif args.conf_sweep:
            if not (ROOT / "web" / "test" / "out" / "pages" / "pages.json").exists():
                subprocess.run([sys.executable, str(ROOT / "web" / "test" / "make_pages.py"), "--offset", "900", "--n", "100"],
                               check=True, cwd=ROOT)
            text = subprocess.run(["node", str(ROOT / "web" / "test" / "page_e2e.mjs"), "--conf-sweep"], check=True,
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
            print(text)
        charts(text)
    report()


if __name__ == "__main__":
    main()
