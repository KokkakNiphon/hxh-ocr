"""Page dataset, augmentation and DBNet targets for the text-line detector.

Records come from `hxh-synth pages` (`<split>/labels.jsonl`): one page per line,
`{"image", "width", "height", "lines": [{"quad": [[x, y] x4], "text", "orient": "h"|"v", "kind"}]}`.
Quads are tight around the ink, clockwise from the top-left in reading orientation (vertical
lines start at the column's top-right), with pixel centres at integers (OpenCV convention).

Input convention (shared with the deployed model and the Rust engine):
  * grayscale u8 page, letterboxed to 640x640: scale s = 640 / max(w, h), resized size
    (round(w * s), round(h * s)) (INTER_AREA when shrinking, INTER_LINEAR when enlarging), placed
    at the top-left, the rest (right / bottom) filled with 255;
  * normalised like the recogniser: (x / 255 - 0.5) / 0.5.
Output: [1, 2, 320, 320] at stride 2 (channel 0 text-kernel logit, channel 1 vertical logit).
Output pixel u <-> input pixel 2u + 0.5 (pixel-centre convention: (p + 0.5) / 2 - 0.5).

Targets are an exact numpy port of `rust/hxh-synth/src/targets.rs` (shrink D = A(1-r^2)/L with
r = 0.4, mitred kernel, pixel-centre half-plane fill, threshold map 0.3..0.7 with its mask,
ignore mask for thin lines, vertical map on the kernels of vertical lines), computed after
augmentation at the output stride.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

LETTERBOX = int(os.environ.get("HXH_LETTERBOX", "640"))  # 960 for the base detector (experiments/family)
STRIDE = 2
PAD_VALUE = 255


# ----------------------------------------------------------------------------
# geometry (ports of rust/hxh-synth/src/geom.rs + targets.rs)
# ----------------------------------------------------------------------------
def area2(q) -> float:
    """Twice the signed shoelace area; positive for clockwise polygons in y-down coordinates."""
    q = q.tolist() if isinstance(q, np.ndarray) else q
    n = len(q)
    return sum(q[i][0] * q[(i + 1) % n][1] - q[(i + 1) % n][0] * q[i][1] for i in range(n))


def perimeter(q) -> float:
    q = q.tolist() if isinstance(q, np.ndarray) else q
    n = len(q)
    return sum(math.hypot(q[(i + 1) % n][0] - q[i][0], q[(i + 1) % n][1] - q[i][1]) for i in range(n))


def line_thickness(q) -> float:
    """Mean length of edges q0-q3 and q1-q2 (the line's height in reading orientation)."""
    q = q.tolist() if isinstance(q, np.ndarray) else q
    return 0.5 * (math.hypot(q[3][0] - q[0][0], q[3][1] - q[0][1]) + math.hypot(q[2][0] - q[1][0], q[2][1] - q[1][1]))


def offset_distance(q: np.ndarray, shrink: float = 0.4) -> float:
    """DBNet shrink / dilation distance D = A (1 - r^2) / L."""
    L = perimeter(q)
    return 0.0 if L <= 1e-9 else abs(area2(q)) * 0.5 * (1.0 - shrink * shrink) / L


def shrink_quad(q: np.ndarray, d: float):
    """Move each edge of a convex polygon inward by d (outward if d < 0) and intersect neighbouring
    edges (mitre). None if it collapses. Exact port of targets.rs::shrink_quad."""
    q = q.tolist() if isinstance(q, np.ndarray) else q
    n = len(q)
    a2 = area2(q)
    if abs(a2) < 1e-9:
        return None
    sgn = 1.0 if a2 > 0 else -1.0
    e = [(float(q[(i + 1) % n][0] - q[i][0]), float(q[(i + 1) % n][1] - q[i][1])) for i in range(n)]
    p = []
    for i in range(n):
        l = math.hypot(*e[i])
        if l < 1e-9:
            return None
        inward = (sgn * -e[i][1] / l, sgn * e[i][0] / l)
        p.append((q[i][0] + inward[0] * d, q[i][1] + inward[1] * d))
    out = []
    for i in range(n):
        j = (i + n - 1) % n
        det = e[j][0] * (-e[i][1]) - (-e[i][0]) * e[j][1]
        if abs(det) < 1e-12:
            out.append(p[i])
            continue
        r = (p[i][0] - p[j][0], p[i][1] - p[j][1])
        t = (r[0] * (-e[i][1]) - (-e[i][0]) * r[1]) / det
        out.append((p[j][0] + t * e[j][0], p[j][1] + t * e[j][1]))
    if d > 0.0:
        for i in range(n):
            e2 = (out[(i + 1) % n][0] - out[i][0], out[(i + 1) % n][1] - out[i][1])
            if e2[0] * e[i][0] + e2[1] * e[i][1] <= 0.0:
                return None
    return np.array(out, np.float64)


def _clip_rect(x0, y0, x1, y1, w, h):
    return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)


def fill_convex(mask: np.ndarray, q, v=1) -> None:
    """Set `mask` to v on pixels whose centre lies inside or on the convex polygon q
    (raster.rs::fill_convex_u8: half-plane test with a -1e-9 tolerance)."""
    q = q.tolist() if isinstance(q, np.ndarray) else q
    h, w = mask.shape
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    x0, y0 = math.ceil(min(xs)), math.ceil(min(ys))
    x1, y1 = math.floor(max(xs)) + 1, math.floor(max(ys)) + 1
    x0, y0, x1, y1 = _clip_rect(x0, y0, x1, y1, w, h)
    if x1 <= x0 or y1 <= y0:
        return
    inside = _inside(q, x0, y0, x1, y1)
    mask[y0:y1, x0:x1][inside] = v


def _inside(q, x0, y0, x1, y1) -> np.ndarray:
    q = q.tolist() if isinstance(q, np.ndarray) else q
    sgn = 1.0 if area2(q) >= 0.0 else -1.0
    px = np.arange(x0, x1, dtype=np.float64)[None, :]
    py = np.arange(y0, y1, dtype=np.float64)[:, None]
    ok = None
    n = len(q)
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        # sgn * cross(b - a, p - a) >= -1e-9, split into a row and a column term
        t = (sgn * (b[0] - a[0])) * (py - a[1]) - (sgn * (b[1] - a[1])) * (px - a[0]) >= -1e-9
        ok = t if ok is None else (ok & t)
    return ok


def rasterize_targets(quads: np.ndarray, vertical: np.ndarray, w: int, h: int, shrink: float = 0.4,
                      thresh_min: float = 0.3, thresh_max: float = 0.7, min_size: float = 3.0,
                      ignore: np.ndarray | None = None) -> dict:
    """quads [N, 4, 2] (target pixel coordinates), vertical [N] bool. `ignore` [N] bool forces a
    line into the ignore mask (used for lines mostly cut off by the crop). Port of targets.rs."""
    prob = np.zeros((h, w), np.uint8)
    ign = np.zeros((h, w), np.uint8)
    vert = np.zeros((h, w), np.uint8)
    canvas = np.zeros((h, w), np.float32)
    tmask = np.zeros((h, w), np.uint8)
    quads = np.asarray(quads, np.float64)
    for k in range(len(quads)):
        q = quads[k].tolist()
        d = offset_distance(q, shrink)
        if d <= 1e-6:
            continue
        forced = ignore is not None and bool(ignore[k])
        kernel = shrink_quad(q, d) if (not forced and line_thickness(q) >= min_size) else None
        if kernel is not None:
            fill_convex(prob, kernel, 1)
            if vertical[k]:
                fill_convex(vert, kernel, 1)
        else:
            fill_convex(ign, q, 1)
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        x0, y0 = math.floor(min(xs) - d), math.floor(min(ys) - d)
        x1, y1 = math.ceil(max(xs) + d) + 1, math.ceil(max(ys) + d) + 1
        x0, y0, x1, y1 = _clip_rect(x0, y0, x1, y1, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        px = np.arange(x0, x1, dtype=np.float64)[None, :]
        py = np.arange(y0, y1, dtype=np.float64)[:, None]
        dd = None
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            dx, dy = b[0] - a[0], b[1] - a[1]
            l2 = max(dx * dx + dy * dy, 1e-9)
            t = np.clip(((px - a[0]) * dx + (py - a[1]) * dy) / l2, 0.0, 1.0)
            di = np.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))
            dd = di if dd is None else np.minimum(dd, di)
        v = (1.0 - np.clip(dd / d, 0.0, 1.0)).astype(np.float32)
        sub = canvas[y0:y1, x0:x1]
        np.maximum(sub, v, out=sub)
        tm = (dd <= d) | _inside(q, x0, y0, x1, y1)
        tmask[y0:y1, x0:x1][tm] = 1
    span = thresh_max - thresh_min
    return {
        "prob": prob,
        "prob_mask": (1 - ign).astype(np.uint8),
        "thresh": (thresh_min + span * canvas).astype(np.float32),
        "thresh_mask": tmask,
        "vert": vert,
    }


# ----------------------------------------------------------------------------
# letterbox (inference / eval input)
# ----------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = LETTERBOX):
    """u8 grayscale page -> (u8 [size, size], sx, sy). Long side scaled to `size`, placed top-left,
    right / bottom padded with 255. Map an original point p to the input with
    (p + 0.5) * s - 0.5 (per axis: sx = new_w / w, sy = new_h / h)."""
    h, w = img.shape[:2]
    s = size / max(w, h)
    nw, nh = max(1, min(size, round(w * s))), max(1, min(size, round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else img
    out = np.full((size, size), PAD_VALUE, np.uint8)
    out[:nh, :nw] = r
    return out, nw / w, nh / h


_NORM_LUT = ((np.arange(256, dtype=np.float32) / 255.0 - 0.5) / 0.5).astype(np.float32)


def normalize_u8(u8: np.ndarray) -> np.ndarray:
    """(x / 255 - 0.5) / 0.5 as float32 (table lookup, same values)."""
    return _NORM_LUT[u8]


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------
class PageIndex:
    """labels.jsonl of one or more split dirs, stored as compact numpy arrays (on Windows every
    DataLoader worker receives its own pickled copy of the dataset)."""

    def __init__(self, roots, limit: int | None = None, offset: int = 0, keep_text: bool = False):
        paths, sizes, counts, quads, orient, texts, kinds = [], [], [], [], [], [], []
        for root in map(Path, roots):
            with open(root / "labels.jsonl", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if limit is not None and len(paths) >= limit:
                        break
                    r = json.loads(line)
                    paths.append(str(root / r["image"]))
                    sizes.append((r["width"], r["height"]))
                    counts.append(len(r["lines"]))
                    for ln in r["lines"]:
                        quads.append(ln["quad"])
                        orient.append(ln["orient"] == "v")
                        if keep_text:
                            texts.append(ln["text"])
                            kinds.append(ln.get("kind", ""))
        self.paths = np.array(paths)
        self.sizes = np.array(sizes, np.int32).reshape(-1, 2)
        self.line_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.quads = np.array(quads, np.float32).reshape(-1, 4, 2)
        self.vertical = np.array(orient, bool)
        self.texts = np.array(texts) if keep_text else None
        self.kinds = np.array(kinds) if keep_text else None

    def __len__(self):
        return len(self.paths)

    def lines(self, i: int):
        a, b = self.line_off[i], self.line_off[i + 1]
        return self.quads[a:b], self.vertical[a:b]

    def line_range(self, i: int):
        return int(self.line_off[i]), int(self.line_off[i + 1])

    def read(self, i: int) -> np.ndarray:
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cannot read {self.paths[i]}")
        return img


def split_dirs(data_dirs, split: str) -> list[Path]:
    out = []
    for d in data_dirs:
        p = Path(d) / split
        if (p / "labels.jsonl").exists():
            out.append(p)
        else:
            print(f"[det_data] skip {p} (no labels.jsonl)")
    return out


# ----------------------------------------------------------------------------
# training dataset
# ----------------------------------------------------------------------------
def _visible_fraction(q: np.ndarray, size: int) -> float:
    a = abs(area2(q)) * 0.5
    if a < 1e-6:
        return 0.0
    rect = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]],
                    np.float32)
    inter, _ = cv2.intersectConvexConvex(q.astype(np.float32), rect)
    return float(inter) / a


MIN_SPREAD = 28  # rust/hxh-synth `lines` uses the same rule


def _legible(img: np.ndarray, q: np.ndarray) -> bool:
    """Is there visible writing in the quad's box? A line a photo blurred past reading (or one an
    effect hid) has almost no grey spread there; it goes to the ignore mask instead of the targets
    (5th-95th percentile spread of the box, at least MIN_SPREAD grey levels)."""
    h, w = img.shape
    x0, y0 = max(0, int(q[:, 0].min())), max(0, int(q[:, 1].min()))
    x1, y1 = min(w, int(np.ceil(q[:, 0].max())) + 1), min(h, int(np.ceil(q[:, 1].max())) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    p5, p95 = np.percentile(img[y0:y1, x0:x1], (5, 95))
    return p95 - p5 >= MIN_SPREAD


class PageDataset(Dataset):
    """Random-scale / crop / small-rotation augmentation with DBNet targets at stride 2.

    train=True:  page long side log-uniform in `long_range`, a crop x crop window (padded with 255
                 if the page is smaller), biased to contain text, rotation U(-rot, rot) degrees,
                 mild brightness / contrast. No flips (glyphs are mirror-sensitive).
    train=False: the deterministic 640 letterbox (used for BN calibration-free checks)."""

    def __init__(self, roots, train: bool = True, crop: int = 512, long_range=(384, 1280),
                 rot: float = 5.0, text_bias: float = 0.8, limit: int | None = None, offset: int = 0):
        self.idx = PageIndex(roots, limit=limit, offset=offset)
        self.train = train
        self.crop = crop
        self.long_range = long_range
        self.rot = rot
        self.text_bias = text_bias

    def __len__(self):
        return len(self.idx)

    def _rng(self, i):
        if self.train:
            self._calls = getattr(self, "_calls", 0) + 1
            return random.Random(torch.initial_seed() * 1_000_003 + self._calls)
        return random.Random(i)

    def sample(self, i: int, rng: random.Random):
        """-> (u8 image [S, S], quads [N, 4, 2] in image coords, vertical [N], ignore [N])."""
        img = self.idx.read(i)
        quads, vert = self.idx.lines(i)
        quads = quads.astype(np.float64)
        h, w = img.shape
        if not self.train:
            u8, sx, sy = letterbox(img, LETTERBOX)
            q = np.empty_like(quads)
            q[..., 0] = (quads[..., 0] + 0.5) * sx - 0.5
            q[..., 1] = (quads[..., 1] + 0.5) * sy - 0.5
            return u8, q, vert, np.zeros(len(q), bool)

        S = self.crop
        # photometric first, so the padding stays exactly 255 as in the letterbox
        if rng.random() < 0.8:
            c = rng.uniform(0.75, 1.25)
            b = rng.uniform(-25, 25)
            m = float(img.mean())
            img = cv2.convertScaleAbs(img, alpha=c, beta=b + (1 - c) * m)
        if rng.random() < 0.05:
            img = 255 - img
        lo, hi = self.long_range
        long_new = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        s = long_new / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        interp = (cv2.INTER_AREA if rng.random() < 0.7 else cv2.INTER_LINEAR) if s < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        sx, sy = nw / w, nh / h
        q = np.empty_like(quads)
        q[..., 0] = (quads[..., 0] + 0.5) * sx - 0.5
        q[..., 1] = (quads[..., 1] + 0.5) * sy - 0.5

        # crop origin (top-left of the window in the scaled page), biased towards text
        def axis_origin(n, centre=None):
            if n <= S:  # page smaller than the window: random placement inside it
                return -rng.uniform(0, S - n)
            if centre is None:
                return rng.uniform(0, n - S)
            o = centre - rng.uniform(0.1, 0.9) * S
            return min(max(o, 0.0), n - S)

        if len(q) and rng.random() < self.text_bias:
            c = q[rng.randrange(len(q))].mean(axis=0)
            ox, oy = axis_origin(nw, c[0]), axis_origin(nh, c[1])
        else:
            ox, oy = axis_origin(nw), axis_origin(nh)
        ang = rng.uniform(-self.rot, self.rot) if rng.random() < 0.7 else 0.0
        # forward affine: scaled page -> crop; rotate about the crop centre
        cx, cy = (S - 1) / 2, (S - 1) / 2
        R = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)  # 2x3, about the crop centre
        T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], np.float64)
        M = R @ T
        out = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=PAD_VALUE)
        qq = q @ M[:, :2].T + M[:, 2]
        keep, ign = [], []
        for k in range(len(qq)):
            f = _visible_fraction(qq[k], S)
            if f <= 0.0:
                continue
            keep.append(k)
            ign.append(f < 0.6 or not _legible(out, qq[k]))
        keep = np.array(keep, np.int64)
        return out, qq[keep].reshape(-1, 4, 2), vert[keep], np.array(ign, bool)

    def __getitem__(self, i):
        rng = self._rng(i)
        u8, q, vert, ign = self.sample(i, rng)
        S = u8.shape[0]
        tw = S // STRIDE
        qt = (q + 0.5) / STRIDE - 0.5
        t = rasterize_targets(qt, vert, tw, tw, ignore=ign)
        return {
            "images": torch.from_numpy(normalize_u8(u8))[None],
            "prob": torch.from_numpy(t["prob"]),
            "prob_mask": torch.from_numpy(t["prob_mask"]),
            "thresh": torch.from_numpy(t["thresh"]),
            "thresh_mask": torch.from_numpy(t["thresh_mask"]),
            "vert": torch.from_numpy(t["vert"]),
        }


def _worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_train_loader(ds: PageDataset, batch_size: int, num_workers: int = 2, seed: int = 0, weights=None):
    """weights: optional per-page sampling weights (a weighted sampler with replacement, one
    'epoch' = len(ds) draws) to balance several page sets."""
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = None
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(ds),
                                                         replacement=True, generator=g)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True, num_workers=num_workers,
        generator=g, persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
    )
