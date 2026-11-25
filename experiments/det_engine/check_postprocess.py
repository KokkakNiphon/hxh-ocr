"""Check rust/hxh-detect's post-processing against OpenCV on random synthetic maps.

For every random int8 map (rotated bars, ellipses, smoothed-noise blobs, thin diagonals,
speckle; random exponents) the Rust example (`detect maps`) dumps its components, hulls,
minimum-area rectangles and final lines; this script recomputes them with OpenCV / numpy:

  threshold     Rust's integer threshold q > thr  ==  float sigmoid(q 2^exp) > 0.3, all q
  components    cv2.connectedComponentsWithStats(connectivity=8): same count, and per
                component the same pixel count, bounding box, sum of probabilities and of
                channel-1 logits
  hull          cv2.convexHull of the same point set (corners of the per-row extreme pixel
                squares): the same vertex set (exact, the coordinates are half-integers)
  min-area rect cv2.minAreaRect of the same points: area, centre, side lengths and corners
                (OpenCV computes in float32; tolerance below). Several rectangles can share
                the minimal area (symmetric hulls); those cases are counted as ties when the
                areas agree but the corners differ.
  lines         DBNet unclip (D = A * 1.5 / L on the cv2 rectangle) and the filters
                (score >= 0.5, short side >= 3): same lines, sizes within tolerance
  DBNet boxes   for reference only: cv2.minAreaRect of the cv2.findContours contour (pixel
                centres, as DBNet / PaddleOCR do) is systematically ~1 px smaller per side
                than the pixel-square rectangle used here (reported, not asserted)
  letterbox     the Rust area-average letterbox vs cv2.resize(INTER_AREA / INTER_LINEAR):
                max |difference| (expected <= 1 from rounding) on random images

Usage (from the repo root, .venv-ppq; build the example first:
  cargo build --release -p hxh-detect --example detect   (in rust/))
  python experiments/det_engine/check_postprocess.py [--n 300] [--seed 0]
Exit code 0 when every assertion holds.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXE = ROOT / "rust" / "target" / "release" / "examples" / ("detect.exe" if sys.platform == "win32" else "detect")
sys.path.insert(0, str(HERE))
from golden_det import letterbox_u8  # noqa: E402

TOL_RECT = 2e-3  # px: OpenCV's minAreaRect works in float32 (coordinates up to ~200 px)
TOL_LINE = 2e-3  # px: f32 line records


def random_map(rng: np.random.Generator):
    h, w = int(rng.integers(16, 200)), int(rng.integers(16, 200))
    prob = np.zeros((h, w), np.float32)
    vert = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 12))):
        kind = rng.integers(0, 5)
        m = np.zeros((h, w), np.uint8)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        if kind == 0:  # rotated bar
            box = ((cx, cy), (rng.uniform(1, w), rng.uniform(0.5, 12)), rng.uniform(-90, 90))
            cv2.fillPoly(m, [np.round(cv2.boxPoints(box) * 16).astype(np.int32)], 1, shift=4)
        elif kind == 1:  # ellipse
            cv2.ellipse(m, (int(cx), int(cy)), (int(rng.integers(1, 30)), int(rng.integers(1, 10))), rng.uniform(0, 180), 0, 360, 1, -1)
        elif kind == 2:  # smoothed-noise blob
            n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(1, 4))
            m = (n > np.quantile(n, rng.uniform(0.9, 0.99))).astype(np.uint8)
        elif kind == 3:  # thin diagonal (8-connected staircase)
            cv2.line(m, (int(cx), int(cy)), (int(rng.uniform(0, w)), int(rng.uniform(0, h))), 1, 1, cv2.LINE_8)
        else:  # speckle
            m = (rng.random((h, w)) < 0.01).astype(np.uint8)
        p = rng.uniform(0.35, 0.99)
        prob = np.maximum(prob, m * p)
        vert = np.where(m > 0, rng.uniform(-1, 1), vert)
    prob = np.where(prob > 0, prob, rng.uniform(0.01, 0.25, (h, w)))  # background below threshold
    exp = int(rng.integers(-5, 0))
    logit = np.log(prob / (1 - prob))
    q0 = np.clip(np.floor(logit / 2.0 ** exp + 0.5), -128, 127).astype(np.int8)
    q1 = np.clip(np.floor(vert * 3 / 2.0 ** exp + 0.5), -128, 127).astype(np.int8)
    return np.stack([q0, q1], -1), exp


def run_rust(maps, args_extra=()):
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "maps.bin", Path(td) / "out.txt"
        b = bytearray(struct.pack("<I", len(maps)))
        for m, e in maps:
            h, w, c = m.shape
            b += struct.pack("<HHHbB", h, w, c, e, 0) + m.astype(np.int8).tobytes()
        inp.write_bytes(bytes(b))
        subprocess.run([str(EXE), "maps", str(inp), str(out), *args_extra], check=True)
        return out.read_text()


def parse(txt):
    res, cur = [], None
    for line in txt.splitlines():
        t = line.split()
        if t[0] == "map":
            cur = {"thr": int(t[7]), "comps": [], "lines": []}
            res.append(cur)
        elif t[0] == "comp":
            first, pixels, vsum = int(t[1]), int(t[2]), int(t[3])
            psum = float(t[4])
            bbox = [int(v) for v in t[5:9]]
            r = [float(v) for v in t[10:16]]
            m = int(t[17])
            hull = np.array([float(v) for v in t[18 : 18 + 2 * m]]).reshape(m, 2)
            cur["comps"].append(dict(first=first, pixels=pixels, vsum=vsum, psum=psum, bbox=bbox, c=r[0:2], u=r[2:4],
                                     w=r[4], h=r[5], hull=hull))
        elif t[0] == "line":
            cur["lines"].append(np.array([float(v) for v in t[1:17]]))
    return res


def square_corners(mask):
    pts = []
    for y in np.flatnonzero(mask.any(1)):
        xs = np.flatnonzero(mask[y])
        a, b = xs[0] - 0.5, xs[-1] + 0.5
        pts += [(a, y - 0.5), (a, y + 0.5), (b, y - 0.5), (b, y + 0.5)]
    return np.array(pts, np.float32)


def rect_corners(c, u, w, h):
    c, u = np.array(c), np.array(u)
    n = np.array([-u[1], u[0]])
    return np.array([c - u * w / 2 - n * h / 2, c + u * w / 2 - n * h / 2, c + u * w / 2 + n * h / 2, c - u * w / 2 + n * h / 2])


def same_corner_set(a, b, tol):
    return all(np.min(np.linalg.norm(b - p, axis=1)) <= tol for p in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} missing: cargo build --release -p hxh-detect --example detect (in rust/)")
    rng = np.random.default_rng(args.seed)
    maps = [random_map(rng) for _ in range(args.n)]
    res = parse(run_rust(maps))
    assert len(res) == len(maps)
    stats = dict(maps=len(maps), comps=0, hull_ok=0, rect_ok=0, rect_tie=0, rect_bad=0, max_area_rel=0.0, max_corner=0.0,
                 lines=0, lines_ok=0, max_line_dev=0.0)
    dbnet_short, dbnet_long = [], []
    failures = []
    for (m, exp), r in zip(maps, res):
        h, w, _ = m.shape
        q = np.arange(-128, 128)
        prob_q = 1 / (1 + np.exp(-q * 2.0 ** exp))
        if not np.array_equal(q > r["thr"], prob_q > 0.3):
            failures.append(f"threshold exp {exp}")
        mask = (m[..., 0].astype(np.int32) > r["thr"]).astype(np.uint8)
        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab - 1 != len(r["comps"]):
            failures.append(f"component count {len(r['comps'])} vs cv2 {n_lab - 1}")
            continue
        want_lines = []
        for c in r["comps"]:
            stats["comps"] += 1
            y0, x0 = divmod(c["first"], w)
            k = lab[y0, x0]
            cm = lab == k
            sx, sy, sw, sh, area = st[k]
            prob = 1 / (1 + np.exp(-m[..., 0].astype(np.float64) * 2.0 ** exp))
            ok = (area == c["pixels"] and [sx, sy, sx + sw - 1, sy + sh - 1] == c["bbox"]
                  and int(m[..., 1][cm].astype(np.int64).sum()) == c["vsum"]
                  and abs(prob[cm].sum() - c["psum"]) <= 1e-9 * c["psum"])
            if not ok:
                failures.append(f"component stats at {c['first']}")
            pts = square_corners(cm)
            hull_cv = cv2.convexHull(pts).reshape(-1, 2)
            if len(hull_cv) == len(c["hull"]) and same_corner_set(c["hull"], hull_cv, 1e-9):
                stats["hull_ok"] += 1
            else:
                failures.append(f"hull at {c['first']}: {len(c['hull'])} vs cv2 {len(hull_cv)} vertices")
            (ccx, ccy), (cw_, ch_), ang = cv2.minAreaRect(pts)
            cv_corners = cv2.boxPoints(((ccx, ccy), (cw_, ch_), ang))
            ours = rect_corners(c["c"], c["u"], c["w"], c["h"])
            a_ours, a_cv = c["w"] * c["h"], cw_ * ch_
            rel = abs(a_ours - a_cv) / max(a_cv, 1e-9)
            stats["max_area_rel"] = max(stats["max_area_rel"], rel)
            exact = same_corner_set(ours, cv_corners, TOL_RECT)
            if exact:
                stats["rect_ok"] += 1
                stats["max_corner"] = max(stats["max_corner"], max(np.min(np.linalg.norm(cv_corners - p, axis=1)) for p in ours))
            elif rel <= 1e-5:
                stats["rect_tie"] += 1
            else:
                stats["rect_bad"] += 1
                failures.append(f"min-area rect at {c['first']}: area {a_ours:.4f} vs cv2 {a_cv:.4f}")
            # DBNet-style box (contour pixel centres) for reference
            cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (_, _), (dw, dh), _ = cv2.minAreaRect(np.concatenate(cnts))
            dbnet_short.append(min(c["w"], c["h"]) - min(dw, dh))
            dbnet_long.append(max(c["w"], c["h"]) - max(dw, dh))
            # expected line (DBNet unclip on the cv2 rectangle, default filters, identity frame)
            score = c["psum"] / c["pixels"]
            d = cw_ * ch_ * 1.5 / (2 * (cw_ + ch_)) if cw_ + ch_ > 0 else 0
            uw, uh = cw_ + 2 * d, ch_ + 2 * d
            # filter decisions within float32 noise of a threshold may go either way, and so may
            # those of equal-area ties (e.g. a diagonal pixel pair: 2 x 2 or 2.83 x 1.41)
            border = abs(min(uw, uh) - 3.0) < 1e-3 or abs(score - 0.5) < 1e-9 or not exact
            if score >= 0.5 and min(uw, uh) >= 3.0 or border:
                # (an equal-area tie may pick another rectangle: then only the presence is compared)
                want_lines.append((ccx, ccy, max(uw, uh), min(uw, uh), exact, border))
        got = list(r["lines"])
        # exact, unambiguous lines first, so a tie's wider match radius cannot take their line
        want_lines.sort(key=lambda wl: (wl[5], not wl[4]))
        for wl in want_lines:
            # match by centre
            if not got:
                if not wl[5]:
                    failures.append(f"line at ({wl[0]:.1f}, {wl[1]:.1f}) missing")
                continue
            k = int(np.argmin([np.hypot(g_[0:8:2].mean() - wl[0], g_[1:8:2].mean() - wl[1]) for g_ in got]))
            best = got[k]
            if np.hypot(best[0:8:2].mean() - wl[0], best[1:8:2].mean() - wl[1]) > (0.75 if wl[4] else 3.0):
                if not wl[5]:
                    failures.append(f"line at ({wl[0]:.1f}, {wl[1]:.1f}) missing")
                continue
            got.pop(k)
            if not wl[4] or wl[5]:
                continue
            stats["lines"] += 1
            dev = max(abs(best[0:8:2].mean() - wl[0]), abs(best[1:8:2].mean() - wl[1]),
                      abs(max(best[12], best[13]) - wl[2]), abs(min(best[12], best[13]) - wl[3]))
            stats["max_line_dev"] = max(stats["max_line_dev"], dev)
            if dev <= TOL_LINE:
                stats["lines_ok"] += 1
            else:
                failures.append(f"line deviation {dev:.4f}")
        if got:
            failures.append(f"{len(got)} unexpected lines, e.g. {got[0][:8]}")

    # letterbox vs OpenCV
    lb_rows = []
    with tempfile.TemporaryDirectory() as td:
        for w_, h_ in [(1000, 700), (640, 640), (2480, 3508), (333, 1201), (100, 60), (7, 640)]:
            img = cv2.GaussianBlur(rng.integers(0, 256, (h_, w_), dtype=np.uint8), (0, 0), 1.5)
            img = np.where(rng.random((h_, w_)) < 0.02, 0, img).astype(np.uint8)
            p_in, p_out = Path(td) / "in.pgm", Path(td) / "out.pgm"
            p_in.write_bytes(f"P5\n{w_} {h_}\n255\n".encode() + img.tobytes())
            o = subprocess.run([str(EXE), "letterbox", str(p_in), "640", str(p_out)], check=True, capture_output=True, text=True)
            nw, nh = map(int, o.stdout.split())
            raw = p_out.read_bytes()
            ours = np.frombuffer(raw[-640 * 640 :], np.uint8).reshape(640, 640)
            ref, (rw, rh) = letterbox_u8(img, 640)
            diff = np.abs(ours.astype(int) - ref.astype(int))
            lb_rows.append((f"{w_}x{h_}", (nw, nh) == (rw, rh), int(diff.max()), float((diff > 0).mean())))
            if (nw, nh) != (rw, rh) or diff.max() > 1:
                failures.append(f"letterbox {w_}x{h_}: dims {(nw, nh)} vs {(rw, rh)}, max diff {diff.max()}")

    print(f"maps {stats['maps']}, components {stats['comps']}: hull == cv2.convexHull {stats['hull_ok']}, "
          f"min-area rect == cv2.minAreaRect {stats['rect_ok']} (max corner deviation {stats['max_corner']:.2e} px), "
          f"equal-area ties {stats['rect_tie']}, wrong {stats['rect_bad']}; max relative area difference "
          f"{stats['max_area_rel']:.1e}")
    print(f"lines {stats['lines_ok']}/{stats['lines']} as expected from cv2 rect + DBNet unclip "
          f"(max deviation {stats['max_line_dev']:.2e} px)")
    ds, dl = np.array(dbnet_short), np.array(dbnet_long)
    print(f"pixel-square rect minus DBNet contour rect: short side {ds.mean():+.2f} px (median {np.median(ds):+.2f}), "
          f"long side {dl.mean():+.2f} px (median {np.median(dl):+.2f})")
    for name, dims_ok, mx, frac in lb_rows:
        print(f"letterbox {name:>9s} -> 640: dims equal {dims_ok}, max |rust - cv2| {mx}, pixels differing {100 * frac:.2f} %")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures[:30]:
            print("  ", f)
        sys.exit(1)
    print("all post-processing checks passed")


if __name__ == "__main__":
    main()
