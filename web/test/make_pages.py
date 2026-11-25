"""Generated val pages for web/test/page_e2e.mjs: grayscale PGM files plus their ground truth.

Writes <out>/pages.json ({file, width, height, lines: [{quad (8 floats, page px, clockwise
from the reading top-left), text, orient, kind}]}) and <out>/pNNNN.pgm, decoded with
cv2.IMREAD_GRAYSCALE exactly like the Python detector eval (scripts/eval_det.py).

    .venv\\Scripts\\python.exe web\\test\\make_pages.py [--data data\\pages_v2\\val] [--offset 900] [--n 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "pages_v2" / "val"))
    ap.add_argument("--offset", type=int, default=900, help="first page (the detector's held-out pages start at 900)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "pages"))
    ap.add_argument("--drop-illegible", action="store_true",
                    help="leave out labelled lines with almost no grey spread in their box (v5 photos blurred past "
                         "reading; hxh_ocr/det_data._legible), so no model is asked to find invisible text")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (data / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l]
    pages = []
    dropped = 0
    if args.drop_illegible:
        import sys
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from hxh_ocr.det_data import _legible
    for i, r in enumerate(rows[args.offset : args.offset + args.n]):
        img = cv2.imread(str(data / r["image"]), cv2.IMREAD_GRAYSCALE)
        if args.drop_illegible:
            keep = [l for l in r["lines"] if _legible(img, np.asarray(l["quad"], np.float64))]
            dropped += len(r["lines"]) - len(keep)
            r = {**r, "lines": keep}
        name = f"p{args.offset + i:04d}.pgm"
        h, w = img.shape
        (out / name).write_bytes(f"P5\n{w} {h}\n255\n".encode() + img.tobytes())
        pages.append({"file": name, "width": w, "height": h, "source": r["image"],
                      "lines": [{"quad": [c for pt in l["quad"] for c in pt], "text": l["text"],
                                 "orient": l["orient"], "kind": l["kind"]} for l in r["lines"]],
                      # unlabelled non-Hunter text and icons (pages_v3 / pages_foreign), for false-positive counts
                      "foreign": [{"quad": [c for pt in f["quad"] for c in pt], "script": f.get("script", "")}
                                  for f in r.get("foreign", [])]})
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pages)} pages, {sum(len(p['lines']) for p in pages)} lines -> {out}"
          + (f" ({dropped} illegible lines left out)" if args.drop_illegible else ""))


if __name__ == "__main__":
    main()
