"""Compose full synthetic line images: background + styled ink + geometric warp + camera/scan degradations.

make_sample() returns one text-line crop (horizontal: fixed height, variable width;
vertical: fixed width, variable height) and its kana label.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .glyphs import Stroke, strokes_bbox
from .render import SS, draw_strokes, layout, sample_style, _translate
from .text import TextSampler, label_of


@dataclass
class SynthConfig:
    size: int = 64               # line thickness in px (height of horizontal lines, width of vertical ones)
    min_len: int = 1
    max_len: int = 20
    vertical_prob: float = 0.3
    bg_images: list = field(default_factory=list)  # optional real background images (RGB float32)


# --- colour helpers ---------------------------------------------------------------

def lum(c) -> float:
    return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


def color_with_lum(rng, target: float, sat_max: float = 1.0) -> np.ndarray:
    h, s = rng.random(), (rng.random() ** 1.6) * sat_max
    c = np.array(colorsys.hsv_to_rgb(h, s, 1.0))
    c = c * (target / max(lum(c), 1e-3))
    if c.max() > 1:
        c = np.clip(c, 0, 1)
        l1 = lum(c)
        if l1 < target:
            c = c + (1 - c) * ((target - l1) / max(1 - l1, 1e-6))
    return np.clip(c, 0, 1).astype(np.float32)


def value_noise(h, w, rng, cell: float) -> np.ndarray:
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return (n - n.mean()) / (n.std() + 1e-6)


# --- backgrounds --------------------------------------------------------------------

def make_background(h, w, rng, bg_lum: float, cfg: SynthConfig) -> np.ndarray:
    kinds = ["flat", "gradient", "noise", "paper", "tone", "hatch", "shapes"]
    p = [0.2, 0.15, 0.2, 0.12, 0.13, 0.08, 0.12]
    if cfg.bg_images:
        kinds.append("image")
        p.append(0.25)
    p = np.array(p) / sum(p)
    kind = rng.choice(kinds, p=p)
    sat = rng.choice([0.0, 0.2, 0.6, 1.0], p=[0.35, 0.3, 0.25, 0.1])
    base = color_with_lum(rng, bg_lum, sat)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "gradient":
        c2 = color_with_lum(rng, np.clip(bg_lum + rng.uniform(-0.25, 0.25), 0, 1), sat)
        a = rng.uniform(0, 2 * np.pi)
        t = xx * math.cos(a) + yy * math.sin(a)
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        img = img * (1 - t[..., None]) + c2 * t[..., None]
    elif kind == "noise":
        amp = rng.uniform(0.02, 0.12)
        n = sum(value_noise(h, w, rng, c) * wgt for c, wgt in ((rng.uniform(20, 80), 1.0), (rng.uniform(4, 12), 0.5)))
        img = img + amp * n[..., None] * rng.uniform(0.5, 1.0, 3).astype(np.float32)
    elif kind == "paper":
        img = img + rng.uniform(0.01, 0.05) * rng.standard_normal((h, w, 1)).astype(np.float32)
        img = img + rng.uniform(0.02, 0.08) * value_noise(h, w, rng, rng.uniform(10, 40))[..., None]
    elif kind == "tone":  # manga screentone dots
        s = rng.uniform(3.0, 8.0)
        a = rng.uniform(0, np.pi / 2)
        u = xx * math.cos(a) + yy * math.sin(a)
        v = -xx * math.sin(a) + yy * math.cos(a)
        d = np.hypot((u % s) - s / 2, (v % s) - s / 2)
        r = s * rng.uniform(0.15, 0.45)
        if rng.random() < 0.4:  # gradient tone
            t = xx / max(w - 1, 1) if rng.random() < 0.5 else yy / max(h - 1, 1)
            r = r * (0.3 + 0.9 * t)
        dots = np.clip(r - d + 0.5, 0, 1)
        tone = color_with_lum(rng, np.clip(bg_lum + rng.choice([-1, 1]) * rng.uniform(0.15, 0.4), 0, 1), 0.2)
        k = rng.uniform(0.5, 1.0)
        img = img * (1 - k * dots[..., None]) + tone * k * dots[..., None]
    elif kind == "hatch":
        s = rng.uniform(3, 10)
        a = rng.uniform(0, np.pi)
        u = xx * math.cos(a) + yy * math.sin(a)
        lines = np.clip(rng.uniform(0.6, 1.6) - np.abs((u % s) - s / 2), 0, 1)
        tone = color_with_lum(rng, np.clip(bg_lum + rng.choice([-1, 1]) * rng.uniform(0.1, 0.35), 0, 1), 0.2)
        k = rng.uniform(0.3, 0.9)
        img = img * (1 - k * lines[..., None]) + tone * k * lines[..., None]
    elif kind == "shapes":
        canvas = (img * 255).astype(np.uint8)
        for _ in range(rng.integers(2, 9)):
            col = color_with_lum(rng, np.clip(bg_lum + rng.uniform(-0.2, 0.2), 0, 1), sat + 0.3)
            col = tuple(int(x * 255) for x in col)
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            if rng.random() < 0.5:
                cv2.circle(canvas, (cx, cy), int(rng.uniform(4, h * 1.5)), col, -1, cv2.LINE_AA)
            else:
                pts = (rng.uniform(-1, 1, (rng.integers(3, 6), 2)) * h * rng.uniform(0.5, 2) + [cx, cy]).astype(np.int32)
                cv2.fillPoly(canvas, [pts], col, cv2.LINE_AA)
        img = canvas.astype(np.float32) / 255
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 3))
    elif kind == "image":
        src = cfg.bg_images[rng.integers(len(cfg.bg_images))]
        sh, sw = src.shape[:2]
        scale = max(h / sh, w / sw) * rng.uniform(1.0, 3.0)
        rs = cv2.resize(src, (max(w, int(sw * scale)), max(h, int(sh * scale))), interpolation=cv2.INTER_AREA)
        y0, x0 = rng.integers(0, rs.shape[0] - h + 1), rng.integers(0, rs.shape[1] - w + 1)
        img = rs[y0:y0 + h, x0:x0 + w].copy()
        # pull towards the planned luminance so the ink keeps some contrast
        m = img.mean()
        img = np.clip(img + (bg_lum - m) * rng.uniform(0.5, 1.0), 0, 1)
    return np.clip(img, 0, 1).astype(np.float32)


# --- geometry -------------------------------------------------------------------------

def random_warp(h, w, rng, box, strength: float):
    """Small perspective + rotation keeping the content box inside the canvas. Returns 3x3 or None."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    x0, y0, x1, y1 = box
    corners = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
    for attempt in range(4):
        f = 0.5 ** attempt
        j = strength * min(h, w) * f
        dst = src + rng.uniform(-j, j, (4, 2)).astype(np.float32)
        ang = math.radians(rng.normal(0, 1.5) * f)
        c = np.float32([w / 2, h / 2])
        rot = np.float32([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        dst = (dst - c) @ rot.T + c
        m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        wc = cv2.perspectiveTransform(corners, m).reshape(-1, 2)
        if wc[:, 0].min() >= 1 and wc[:, 1].min() >= 1 and wc[:, 0].max() <= w - 1 and wc[:, 1].max() <= h - 1:
            return m
    return None


# --- degradations ----------------------------------------------------------------------

def degrade(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    if rng.random() < 0.07:
        k = int(rng.integers(3, 8))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, rot, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    if rng.random() < 0.3:
        f = rng.uniform(1.3, 3.2)
        small = cv2.resize(img, (max(4, int(w / f)), max(4, int(h / f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_NEAREST)
    if rng.random() < 0.3:  # uneven lighting
        g = value_noise(h, w, rng, rng.uniform(40, 200))
        img = img * (1 + rng.uniform(0.05, 0.3) * g[..., None])
    if rng.random() < 0.5:  # contrast / brightness / gamma
        img = (img - 0.5) * rng.uniform(0.6, 1.2) + 0.5 + rng.uniform(-0.12, 0.12)
        img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        img = img + rng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0.005, 0.06)
    img = np.clip(img, 0, 1)
    if rng.random() < 0.15:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.repeat(g[..., None], 3, 2)
    out = (img * 255 + 0.5).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 92))])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)


# --- composition -------------------------------------------------------------------------

def _to_px(strokes: list[Stroke], em_px: float):
    for s in strokes:
        s.pts = s.pts * em_px
        s.r *= em_px


def _clutter(img, rng, bg_lum, box):
    """Stray lines: panel borders / bubble edges in the margins, occasionally crossing the text."""
    h, w = img.shape[:2]
    canvas = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    for _ in range(rng.integers(1, 4)):
        if rng.random() < 0.75:  # in a margin, roughly parallel to the box edge
            gaps = [("t", y0), ("b", h - y1), ("l", x0), ("r", w - x1)]
            gaps = [g for g in gaps if g[1] >= 3]
            if not gaps:
                continue
            side, g = gaps[rng.integers(len(gaps))]
            off = rng.uniform(1, g - 1)
            j = rng.uniform(-1, 1, 2) * min(g - 1, off, 3)
            if side in "tb":
                yy = off if side == "t" else h - off
                pts = np.array([[-5, yy + j[0]], [w + 5, yy + j[1]]])
            else:
                xx = off if side == "l" else w - off
                pts = np.array([[xx + j[0], -5], [xx + j[1], h + 5]])
        else:
            pts = np.stack([rng.uniform(-0.2, 1.2, 4) * w, rng.uniform(-0.2, 1.2, 4) * h], 1)
            if rng.random() < 0.5:
                pts = pts[[0, 3]]
        t = np.linspace(0, 1, 40)[:, None]
        if len(pts) == 4:
            p0, p1, p2, p3 = pts
            pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        cv2.polylines(canvas, [np.round(pts * 16).astype(np.int32)], False, 255, int(rng.integers(1, 4)),
                      cv2.LINE_AA, 4)
    a = canvas.astype(np.float32)[..., None] / 255
    col = color_with_lum(rng, np.clip(1 - bg_lum + rng.uniform(-0.2, 0.2), 0, 1), 0.3)
    return img * (1 - a) + col * a


def _art_fragments(img, rng, box, sides, vertical, em_px):
    """v4: non-text 'character art' beside the line ends (blobs with dark outlines, spiky hair-like
    shapes, thick outline strokes), confined to the margin beyond the text box with a gap. On the
    anime title cards, art at the crop edge was read as extra glyphs (journal, real test set)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    out = img
    for side in sides:
        gap = rng.uniform(0.12, 0.4) * em_px
        lo_t, hi_t = (x0, x1) if not vertical else (y0, y1)
        n_along = w if not vertical else h
        lo, hi = (0.0, lo_t - gap) if side == 0 else (hi_t + gap, float(n_along))
        if hi - lo < 3:
            continue
        cross = h if not vertical else w

        def P(a, c):  # (along, cross) -> (x, y)
            return (a, c) if not vertical else (c, a)

        layer = np.zeros((h, w, 3), np.uint8)
        alpha = np.zeros((h, w), np.uint8)
        kind = rng.choice(["blob", "spikes", "strokes"], p=[0.45, 0.25, 0.3])
        outline = tuple(int(v * 255) for v in color_with_lum(rng, rng.uniform(0.0, 0.2), 0.3))
        thick = max(1, int(round(rng.uniform(0.03, 0.1) * em_px)))
        # anchor towards the outer edge so the shape is usually cut by the crop
        a_c = rng.uniform(lo + 0.3 * (hi - lo), hi + 0.6 * cross) if side == 1 else \
            rng.uniform(lo - 0.6 * cross, hi - 0.3 * (hi - lo))
        c_c = rng.uniform(-0.2, 1.2) * cross
        if kind in ("blob", "spikes"):
            n = int(rng.integers(7, 15)) if kind == "blob" else int(rng.integers(5, 10)) * 2
            R = rng.uniform(0.4, 1.4) * cross
            ang = np.sort(rng.uniform(0, 2 * np.pi, n))
            rad = R * (rng.uniform(0.55, 1.0, n) if kind == "blob" else np.where(np.arange(n) % 2 == 0, 1.0, rng.uniform(0.3, 0.6)))
            pts = np.array([P(a_c + r * np.cos(t), c_c + r * np.sin(t)) for r, t in zip(rad, ang)], np.float64)
            fill_l = rng.uniform(0.2, 0.95) if kind == "blob" else rng.uniform(0.03, 0.3)
            fill = tuple(int(v * 255) for v in color_with_lum(rng, fill_l, rng.uniform(0.2, 1.0)))
            p16 = np.round(pts * 16).astype(np.int32)
            cv2.fillPoly(layer, [p16], fill, cv2.LINE_AA, 4)
            cv2.fillPoly(alpha, [p16], 255, cv2.LINE_AA, 4)
            cv2.polylines(layer, [p16], True, outline, thick, cv2.LINE_AA, 4)
            cv2.polylines(alpha, [p16], True, 255, thick, cv2.LINE_AA, 4)
        else:
            for _ in range(int(rng.integers(1, 5))):
                t = np.linspace(0, 1, 30)[:, None]
                ctrl = [np.array(P(a_c + rng.uniform(-0.6, 0.6) * cross, rng.uniform(-0.2, 1.2) * cross)) for _ in range(4)]
                pts = ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1] + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])
                p16 = np.round(pts * 16).astype(np.int32)
                tk = max(1, int(round(rng.uniform(0.06, 0.2) * em_px)))
                cv2.polylines(layer, [p16], False, outline, tk, cv2.LINE_AA, 4)
                cv2.polylines(alpha, [p16], False, 255, tk, cv2.LINE_AA, 4)
        region = np.zeros((h, w), np.float32)  # keep the art strictly inside the margin
        a_lo, a_hi = int(max(0, np.floor(lo))), int(min(n_along, np.ceil(hi)))
        if not vertical:
            region[:, a_lo:a_hi] = 1
        else:
            region[a_lo:a_hi, :] = 1
        al = (alpha.astype(np.float32) / 255) * region
        out = out * (1 - al[..., None]) + (layer.astype(np.float32) / 255) * al[..., None]
    return out


def make_sample(rng: np.random.Generator, cfg: SynthConfig, sampler: TextSampler, text: str | None = None):
    text = text if text is not None else sampler.sample(rng, cfg.min_len, cfg.max_len)
    vertical = bool(rng.random() < cfg.vertical_prob)
    st = sample_style(rng)
    adv, crs = (1, 0) if vertical else (0, 1)
    T = cfg.size

    strokes = layout(text, st, rng, vertical)
    bb = strokes_bbox(strokes)
    half = st.sw / 2
    ext = [bb[2] - bb[0] + 2 * half, bb[3] - bb[1] + 2 * half]  # em extent (x, y)
    # v3: favour tighter crops (bigger glyphs) so dakuten/handakuten keep a few pixels at the
    # model's input height; v2's U(0.42, 0.74) made the marks 1-2 px at 32 px height.
    if rng.random() < 0.65:
        em_px = T * rng.uniform(0.58, 0.84)
    else:
        em_px = T * rng.uniform(0.40, 0.62)
    em_px = min(em_px, 0.94 * T / ext[crs])
    _to_px(strokes, em_px)
    bb = strokes_bbox(strokes)
    hp = half * em_px
    lo = [bb[0] - hp, bb[1] - hp]
    size = [(bb[2] - bb[0]) + 2 * hp, (bb[3] - bb[1]) + 2 * hp]

    # canvas along the reading direction: content + padding; across: fixed T
    if rng.random() < 0.1:
        pads = rng.uniform(0.0, 0.08, 2) * em_px
    else:
        pads = rng.uniform(0.05, 0.9, 2) * em_px
    art_sides = []  # v4: non-text art beside the line ends; make room for it in the crop
    if rng.random() < 0.25:
        art_sides = [int(rng.integers(2))] if rng.random() < 0.8 else [0, 1]
        for s_ in art_sides:
            pads[s_] = max(pads[s_], rng.uniform(0.5, 1.6) * em_px)
    L = int(math.ceil(size[adv] + pads.sum()))
    free = T - size[crs]
    off_c = rng.uniform(0.05, 0.95) * free
    shift = [0.0, 0.0]
    shift[adv] = pads[0] - lo[adv]
    shift[crs] = off_c - lo[crs]
    _translate(strokes, shift[0], shift[1])
    W, H = (T, L) if vertical else (L, T)
    box = (pads[0], off_c, pads[0] + size[adv], off_c + size[crs])
    if vertical:
        box = (box[1], box[0], box[3], box[2])

    # neighbouring lines peeking in at the edges (imperfect crops). Only a sliver of each
    # neighbour (<= 35% of its height) enters the crop, and never closer than a small gap to
    # the labelled line; in v2 a neighbour could be as visible as the main line, which made
    # the label ambiguous (see journal, tiny_h32 worst-32).
    extra: list[Stroke] = []
    if rng.random() < 0.15:
        gap = 0.12 * em_px
        for sign in ([-1, 1] if rng.random() < 0.3 else [rng.choice([-1, 1])]):
            other = layout(sampler.sample(rng, 3, cfg.max_len), st, rng, vertical)
            _to_px(other, em_px)
            obb = strokes_bbox(other)
            o_lo, o_hi = obb[crs] - hp, obb[crs + 2] + hp
            margin = off_c - gap if sign < 0 else (T - (off_c + size[crs])) - gap
            vis = min(rng.uniform(0.05, 0.35) * (o_hi - o_lo), margin)
            if vis < 1.0:
                continue
            d = [0.0, 0.0]
            d[adv] = rng.uniform(-0.3, 0.3) * L - obb[adv]
            d[crs] = (vis - o_hi) if sign < 0 else (T - vis - o_lo)
            _translate(other, d[0], d[1])
            extra.extend(other)

    big = np.zeros((H * SS, W * SS), np.uint8)
    draw_strokes(big, strokes + extra, st.sw * em_px, st.brush, rng)
    alpha = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    if rng.random() < 0.45:
        m = random_warp(H, W, rng, box, rng.uniform(0.02, 0.09))
        if m is not None:
            alpha = cv2.warpPerspective(alpha, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)

    # colours
    scheme = rng.choice(["dark_on_light", "light_on_dark", "color"], p=[0.55, 0.2, 0.25])
    if scheme == "dark_on_light":
        bg_l, ink_l = rng.uniform(0.6, 1.0), rng.uniform(0.0, 0.35)
    elif scheme == "light_on_dark":
        bg_l, ink_l = rng.uniform(0.0, 0.4), rng.uniform(0.65, 1.0)
    else:
        bg_l = rng.uniform(0, 1)
        gap = rng.uniform(0.25, 0.7)
        ink_l = bg_l + gap if bg_l + gap <= 1 else bg_l - gap
        ink_l = float(np.clip(ink_l, 0, 1))
    bg = make_background(H, W, rng, bg_l, cfg)
    if rng.random() < 0.12:
        bg = _clutter(bg, rng, bg_l, box)
    if art_sides:
        bg = _art_fragments(bg, rng, box, art_sides, vertical, em_px)
    ink = color_with_lum(rng, ink_l, rng.choice([0.0, 0.5, 1.0], p=[0.5, 0.3, 0.2]))
    img = bg

    if rng.random() < 0.18:  # drop shadow; v4: more often, and usually soft down-right like the anime cards
        if rng.random() < 0.6:
            dx, dy = rng.uniform(0.04, 0.12, 2) * em_px
        else:
            dx, dy = rng.uniform(-0.1, 0.1, 2) * em_px
        sm = cv2.warpAffine(alpha, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H))
        sm = cv2.GaussianBlur(sm, (0, 0), rng.uniform(0.5, 2.5)) * rng.uniform(0.4, 1.0)
        scol = color_with_lum(rng, float(np.clip(ink_l + (0.6 if ink_l < 0.5 else -0.6), 0, 1)) * 0.5, 0.3)
        img = img * (1 - sm[..., None]) + scol * sm[..., None]
    if rng.random() < 0.12:  # outline (common for titles and signs)
        r = max(1, int(round(rng.uniform(0.03, 0.12) * em_px)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        om = cv2.dilate(alpha, k)
        ocol = color_with_lum(rng, 1.0 - ink_l, 0.3)
        img = img * (1 - om[..., None]) + ocol * om[..., None]
    a = alpha
    if rng.random() < 0.2:  # patchy ink
        a = a * np.clip(1 - rng.uniform(0.1, 0.5) * np.abs(value_noise(H, W, rng, rng.uniform(2, 10))), 0.2, 1)
    if rng.random() < 0.1:  # gradient ink
        c2 = color_with_lum(rng, ink_l, 1.0)
        t = np.linspace(0, 1, W if not vertical else H, dtype=np.float32)
        t = t[None, :, None] if not vertical else t[:, None, None]
        inkmap = ink * (1 - t) + c2 * t
        img = img * (1 - a[..., None]) + inkmap * a[..., None]
    else:
        img = img * (1 - a[..., None]) + ink * a[..., None]

    if rng.random() < 0.05:  # hard binarisation, like a cheap scan
        # v2 ran Otsu on the finished image, which on dark or patterned backgrounds turned the
        # text into a solid blob. Now: threshold midway between the known ink and background
        # luminance, and only when the background is fairly uniform, so the text survives.
        lw = np.array([0.299, 0.587, 0.114], np.float32)
        bg_gray = bg @ lw
        if bg_gray.std() < 0.1:
            thr = (ink_l + float(bg_gray.mean())) / 2 + rng.normal(0, 0.03)
            b = ((img @ lw) > thr).astype(np.float32)
            img = np.repeat(b[..., None], 3, 2)
    out = degrade(np.clip(img, 0, 1).astype(np.float32), rng)
    meta = {"vertical": vertical, "style": st.kind, "scheme": str(scheme), "em_px": round(em_px, 2), "raw": text}
    return out, label_of(text), meta


def load_bg_images(folder: str | Path, max_side: int = 1024) -> list[np.ndarray]:
    imgs = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        s = max_side / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgs.append(im[..., ::-1].astype(np.float32) / 255)
    return imgs
