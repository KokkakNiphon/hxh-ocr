"""Detector check on the v6 test pages (data/pages_v6): recall of the unchanged detectors by line
type (numbers / punctuation in the line, a kana reading guide right above it, promo scene) and
what happens to the reading guides (boxed into a Hunter line, boxed on their own, left alone).

    python scripts/det_check_v6.py [--models v2tiny,base] [--out experiments/family/eval/det_check_v6.json]

Runs scripts/pages_dump.mjs (web/ocr.js readPage, int8) on data/pages_v6/e2e with each model set
under experiments/family/<name>/ and matches lines at IoU >= 0.5 like web/test/page_e2e.mjs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XS = set("0123456789!?:/.-")


def poly(q):
    q = np.asarray(q, np.float32).reshape(4, 2)
    return q if cv2.contourArea(q, oriented=True) > 0 else q[::-1].copy()


def inter(a, b):
    r, _ = cv2.intersectConvexConvex(a, b)
    return float(r)


def iou(a, b):
    i = inter(a, b)
    return i / (cv2.contourArea(a) + cv2.contourArea(b) - i + 1e-9)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="v2tiny,base")
    ap.add_argument("--pages", default="data/pages_v6/e2e")
    ap.add_argument("--labels", default="data/pages_v6/val/labels.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pages = json.loads((ROOT / a.pages / "pages.json").read_text(encoding="utf-8"))
    src = {}
    for l in (ROOT / a.labels).read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        src[r["image"]] = r
    res = {}
    for m in a.models.split(","):
        d = ROOT / "experiments" / "family" / m
        p = subprocess.run(["node", str(ROOT / "scripts" / "pages_dump.mjs"), str(ROOT / a.pages), str(d / "det.hxhm"),
                            str(d / "model.hxhm"), str(d / "model_v.hxhm")], capture_output=True, text=True, encoding="utf-8", check=True)
        dump = json.loads(p.stdout)
        min_score = json.loads((d / "det.json").read_text(encoding="utf-8")).get("min_show_score", 0)
        cats = {k: [0, 0] for k in ("all h", "h with numbers / punct.", "h without", "h with a reading guide", "h in promo scenes", "v")}
        misses = {k: {"partly boxed (>= 30% of it)": 0, "not found": 0} for k in ("numbers", "plain")}
        ruby = {"guides": 0, "inside a matched Hunter box (>= 30% of it)": 0, "boxed on its own": 0, "shown on its own": 0, "left alone": 0}
        for pg, dp in zip(pages, dump):
            s = src[pg["source"]]
            gq = [poly(l["quad"]) for l in pg["lines"]]
            dq = [poly(l["quad"]) for l in dp["lines"]]
            pairs = sorted(((iou(g, q), gi, li) for gi, g in enumerate(gq) for li, q in enumerate(dq)), reverse=True)
            gu, lu = {}, set()
            for v, gi, li in pairs:
                if v < 0.5:
                    break
                if gi not in gu and li not in lu:
                    gu[gi] = li
                    lu.add(li)
            rubies = [poly(f["quad"]) for f in s.get("foreign", []) if f.get("kind") == "ruby"]
            for gi, (g, gl) in enumerate(zip(gq, pg["lines"])):
                hit = gi in gu
                keys = []
                if gl["orient"] == "v":
                    keys.append("v")
                else:
                    keys += ["all h", "h with numbers / punct." if set(gl["text"]) & XS else "h without"]
                    th = float(np.linalg.norm(g[3] - g[0]))
                    x0, x1, top = g[:, 0].min(), g[:, 0].max(), g[:, 1].min()
                    if any(r[:, 1].max() <= top + 1 and top - r[:, 1].max() < 0.8 * th and min(x1, r[:, 0].max()) - max(x0, r[:, 0].min()) > 0
                           for r in rubies):
                        keys.append("h with a reading guide")
                    if s["layout"] == "promo":
                        keys.append("h in promo scenes")
                for k in keys:
                    cats[k][0] += hit
                    cats[k][1] += 1
                if not hit and gl["orient"] == "h":
                    # a miss: a box covering part of the line (split / cut short) or nothing there
                    ag = cv2.contourArea(g) + 1e-9
                    part = max((inter(g, q) / ag for q in dq), default=0.0)
                    k = "numbers" if set(gl["text"]) & XS else "plain"
                    misses[k]["partly boxed (>= 30% of it)" if part >= 0.3 else "not found"] += 1
            for r in rubies:
                ruby["guides"] += 1
                ar = cv2.contourArea(r) + 1e-9
                cov = [(inter(r, q) / ar, li) for li, q in enumerate(dq)]
                best = max(cov, default=(0.0, -1))
                if best[0] >= 0.3 and best[1] in lu:
                    ruby["inside a matched Hunter box (>= 30% of it)"] += 1
                elif any(iou(r, q) >= 0.3 and li not in lu for li, q in enumerate(dq)):
                    ruby["boxed on its own"] += 1
                    if any(iou(r, q) >= 0.3 and li not in lu and dp["lines"][li]["conf"] >= 0.8 and dp["lines"][li]["score"] >= min_score
                           and dp["lines"][li]["text"].strip() for li, q in enumerate(dq)):
                        ruby["shown on its own"] += 1
                else:
                    ruby["left alone"] += 1
        res[m] = {"recall": {k: {"found": v[0], "lines": v[1], "recall": v[0] / max(1, v[1])} for k, v in cats.items()}, "reading_guides": ruby,
                  "missed_h_lines": misses}
        print(f"== {m}")
        for k, v in cats.items():
            print(f"  recall {k:26s} {v[0]:5d}/{v[1]:5d} = {100 * v[0] / max(1, v[1]):5.1f}%")
        print("  reading guides: " + ", ".join(f"{k} {v}" for k, v in ruby.items()))
        print("  missed horizontal lines: " + "; ".join(f"{k}: " + ", ".join(f"{kk} {vv}" for kk, vv in v.items()) for k, v in misses.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
