"""Turn kana strings into Hunter-alphabet ink masks.

Pipeline: glyph strokes (em units) -> per-glyph affine + hand-drawn jitter ->
line layout (horizontal or vertical) -> pixel coordinates -> anti-aliased mask
rendered at SS x supersampling and downsampled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .glyphs import (DAKUTEN_DX, GLYPHS, HANDAKUTEN_DX, REF_STROKE_EM, SMALL_TO_FULL, SPACE, Stroke,
                     mark_stroke, strokes_bbox)

SS = 4  # supersampling factor for drawing


@dataclass
class TextStyle:
    kind: str = "clean"         # clean | hand | brush
    sw: float = REF_STROKE_EM   # stroke width (em)
    brush: bool = False
    mono: bool = False          # monospaced cells (typical for Japanese) vs proportional
    cell: float = 1.25          # em, mono advance
    spacing: float = 0.25       # em, proportional gap
    wobble: float = 0.0         # em, smooth path displacement
    stroke_jitter: float = 0.0  # em, per-stroke offset / endpoint jitter
    rot_std: float = 0.5        # deg, per-glyph rotation
    scale_std: float = 0.02     # per-glyph scale jitter
    aspect_std: float = 0.02    # per-glyph log-aspect jitter
    shear: float = 0.0          # line-level italic shear
    cross_jitter: float = 0.0   # em, per-glyph baseline wobble
    curve_amp: float = 0.0      # em, baseline curve amplitude
    curve_freq: float = 0.0     # radians per em
    dot_scale: float = 1.0
    mark_scale: float = 1.0


def sample_style(rng: np.random.Generator) -> TextStyle:
    u = rng.random()
    if u < 0.4:
        st = TextStyle(kind="clean", sw=rng.uniform(0.045, 0.13), rot_std=0.4, scale_std=0.01, aspect_std=0.01)
    elif u < 0.8:
        st = TextStyle(kind="hand", sw=rng.uniform(0.04, 0.15), wobble=rng.uniform(0.003, 0.025),
                       stroke_jitter=rng.uniform(0.005, 0.035), rot_std=rng.uniform(1, 6),
                       scale_std=rng.uniform(0.03, 0.1), aspect_std=rng.uniform(0.03, 0.1),
                       cross_jitter=rng.uniform(0.0, 0.08))
    else:
        st = TextStyle(kind="brush", brush=True, sw=rng.uniform(0.07, 0.2), wobble=rng.uniform(0, 0.02),
                       stroke_jitter=rng.uniform(0, 0.03), rot_std=rng.uniform(0.5, 5),
                       scale_std=rng.uniform(0.02, 0.08), aspect_std=rng.uniform(0.02, 0.08),
                       cross_jitter=rng.uniform(0.0, 0.06))
    if rng.random() < 0.18:  # very bold text; v4: up to 0.30 em, the anime title-card font is that heavy
        st.sw = rng.uniform(0.15, 0.30)
    st.mono = rng.random() < 0.5
    st.cell = rng.uniform(1.18, 1.55)
    st.spacing = rng.uniform(0.06, 0.5)
    st.shear = rng.normal(0, 0.12) if rng.random() < 0.3 else 0.0
    if rng.random() < 0.08:
        st.curve_amp = rng.uniform(0.05, 0.3)
        st.curve_freq = rng.uniform(0.15, 0.6)
    st.dot_scale = rng.uniform(0.85, 1.15) * (st.sw / REF_STROKE_EM) ** 0.25
    st.mark_scale = rng.uniform(0.8, 1.3) * (st.sw / REF_STROKE_EM) ** 0.45
    return st


# --- geometry helpers ------------------------------------------------------------

def _densify(pts: np.ndarray, closed: bool, step: float) -> np.ndarray:
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.diff(pts, axis=0)
    d = np.hypot(seg[:, 0], seg[:, 1])
    keep = d > 1e-9
    seg, d, start = seg[keep], d[keep], pts[:-1][keep]
    if len(seg) == 0:
        return pts[:1].copy()
    n = np.maximum(1, np.ceil(d / step).astype(np.int64))
    idx = np.repeat(np.arange(len(seg)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1  # 1..n_i within each segment
    t = (k / np.repeat(n, n))[:, None]
    return np.vstack([pts[:1], start[idx] + seg[idx] * t])


def _wobble(pts: np.ndarray, amp: float, rng) -> np.ndarray:
    if amp <= 0 or len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    disp = np.zeros_like(pts)
    for _ in range(2):
        f = rng.uniform(2.0, 12.0)  # rad per em
        a = amp * rng.uniform(0.3, 1.0)
        ph = rng.uniform(0, 2 * np.pi, 2)
        disp[:, 0] += a * np.sin(f * s + ph[0])
        disp[:, 1] += a * np.sin(f * 1.3 * s + ph[1])
    return pts + disp


def _affine(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def _glyph_matrix(st: TextStyle, rng) -> np.ndarray:
    s = math.exp(rng.normal(0, st.scale_std))
    a = math.exp(rng.normal(0, st.aspect_std))
    th = math.radians(rng.normal(0, st.rot_std))
    sh = st.shear + rng.normal(0, st.aspect_std)
    c, si = math.cos(th), math.sin(th)
    rot = np.array([[c, -si], [si, c]])
    shear = np.array([[1, sh], [0, 1]])
    scale = np.diag([s * a, s / a])
    m2 = rot @ shear @ scale
    return np.hstack([m2, np.zeros((2, 1))])


def _jitter_stroke(s: Stroke, st: TextStyle, rng) -> Stroke:
    s = s.copy()
    if s.kind == "dot":
        s.pts = s.pts + rng.normal(0, st.stroke_jitter * 0.5, 2)
        s.r *= st.dot_scale * math.exp(rng.normal(0, 0.06 if st.kind != "clean" else 0.0))
        return s
    pts = s.pts
    if st.stroke_jitter > 0:
        j = st.stroke_jitter
        c = pts.mean(0)
        th = rng.normal(0, j * 1.5)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pts = (pts - c) @ rot.T + c + rng.normal(0, j, 2)
        if not s.closed and len(pts) >= 2:  # over/under-shoot at the ends
            for end, nb in ((0, 1), (-1, -2)):
                v = pts[end] - pts[nb]
                n = np.linalg.norm(v)
                if n > 1e-6:
                    pts[end] = pts[end] + v / n * rng.normal(0.005, j)
    if st.wobble > 0:
        pts = _wobble(_densify(pts, s.closed, 0.03), st.wobble, rng)
        if s.closed:
            pts = pts[:-1]
    s.pts = pts
    return s


def _transform_stroke(s: Stroke, m: np.ndarray) -> Stroke:
    s = s.copy()
    s.pts = _affine(s.pts, m)
    if s.kind == "dot":
        s.r *= math.sqrt(abs(np.linalg.det(m[:, :2])))
    return s


def _translate(strokes: list[Stroke], dx: float, dy: float):
    for s in strokes:
        s.pts = s.pts + np.array([dx, dy])


# --- layout ---------------------------------------------------------------------

def layout(text: str, st: TextStyle, rng, vertical: bool = False) -> list[Stroke]:
    """Place glyphs along a line in em units. Glyph centres sit on the line's axis (cross = 0).

    `text` may contain spaces and small kana (ゃゅょっ...): small kana use the full-size glyph,
    drawn smaller and shifted towards the bottom (or right, in vertical text) for most lines.
    """
    placed: list[Stroke] = []
    pen = 0.0
    adv = 1 if vertical else 0   # axis along which we advance
    crs = 1 - adv
    small_scale = rng.uniform(0.55, 0.78) if rng.random() < 0.65 else 1.0
    for ch in text:
        if ch == SPACE:
            pen += (st.cell if st.mono else 0) + rng.uniform(0.55, 1.1)
            continue
        small = ch in SMALL_TO_FULL
        g = GLYPHS[SMALL_TO_FULL.get(ch, ch)]
        m = _glyph_matrix(st, rng)
        if small and small_scale < 1:
            m[:, :2] *= small_scale
        forms = g.variants or [g.strokes]
        k = int(rng.choice(len(forms), p=g.weights)) if g.weights and len(forms) > 1 else int(rng.integers(len(forms)))
        base = forms[k]
        strokes = [_jitter_stroke(_transform_stroke(s, m), st, rng) for s in base]
        bb = strokes_bbox(strokes)
        if g.mark:
            mk = mark_stroke(g.mark, bb, st.mark_scale)
            # v4: marks sit anywhere from tucked against the glyph (the anime's "4." ぜ) to well
            # clear of it, but always with a visible gap so a bold stroke can't swallow the mark
            base_dx = (DAKUTEN_DX if g.mark == "dakuten" else HANDAKUTEN_DX) * st.mark_scale
            r_eff = max(mk.r, st.sw * (0.65 if mk.hollow else 0.5))
            dx = max(base_dx * rng.uniform(0.55, 1.35), st.sw / 2 + r_eff + 0.03)
            mk.pts[:, 0] += dx - base_dx
            mk.pts[:, 1] += rng.uniform(-0.06, 0.03) * st.mark_scale
            mk.pts = mk.pts + rng.normal(0, 0.02 + st.stroke_jitter * 0.5, 2)
            strokes.append(mk)
        ub = strokes_bbox(strokes)
        lo, hi = (ub[0], ub[2]) if adv == 0 else (ub[1], ub[3])
        base_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)[crs]
        if st.mono:
            shift = pen + st.cell / 2 - (lo + hi) / 2
            pen += st.cell
        else:
            shift = pen - lo
            pen += (hi - lo) + st.spacing * math.exp(rng.normal(0, 0.15))
        along = pen - st.cell / 2 if st.mono else pen
        cross = -base_c + rng.normal(0, st.cross_jitter) + st.curve_amp * math.sin(st.curve_freq * along)
        if small and small_scale < 1:  # sit low (horizontal) / to the right (vertical)
            cross += 0.45 * (1 - small_scale)
        d = [0.0, 0.0]
        d[adv], d[crs] = shift, cross
        _translate(strokes, d[0], d[1])
        placed.extend(strokes)
    return placed


# --- rasterisation ---------------------------------------------------------------

def _pressure(n: int, rng) -> np.ndarray:
    t = np.linspace(0, 1, n)
    a, b = rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.7)
    prof = np.minimum(1, np.minimum(a + (1 - a) * np.clip(t / 0.25, 0, 1), b + (1 - b) * np.clip((1 - t) / 0.3, 0, 1)))
    return prof * (1 + 0.08 * np.sin(t * rng.uniform(3, 9) + rng.uniform(0, 6)))


def draw_strokes(mask: np.ndarray, strokes: list[Stroke], sw_px: float, brush: bool, rng,
                 ss: int = SS) -> None:
    """Draw strokes (pixel coords at output scale) into a uint8 mask at ss x resolution."""
    sh = 4
    k = ss * (1 << sh)
    for s in strokes:
        w = max(1.0, sw_px * s.width_scale * ss)
        if s.kind == "dot":
            c = (int(round(s.pts[0, 0] * k)), int(round(s.pts[0, 1] * k)))
            r = max(s.r * ss, w * 0.5 if not s.hollow else w)
            e = rng.uniform(-0.08, 0.08)
            axes = (max(1, int(r * (1 + e) * (1 << sh))), max(1, int(r * (1 - e) * (1 << sh))))
            ang = float(rng.uniform(0, 180))
            if s.hollow:
                cv2.ellipse(mask, c, axes, ang, 0, 360, 255, max(1, int(round(w))), cv2.LINE_AA, sh)
            else:
                cv2.ellipse(mask, c, axes, ang, 0, 360, 255, -1, cv2.LINE_AA, sh)
            continue
        pts = s.pts
        if not brush:
            p = np.round(pts * k).astype(np.int32)
            cv2.polylines(mask, [p], s.closed, 255, max(1, int(round(w))), cv2.LINE_AA, sh)
            # round joins at sharp corners (dense wobbly paths don't need them)
            if w >= 3 and len(p) <= 16:
                for q in p:
                    cv2.circle(mask, (int(q[0]), int(q[1])), int(w / 2 * (1 << sh)), 255, -1, cv2.LINE_AA, sh)
            continue
        dense = _densify(pts * ss, s.closed, max(0.8, w * 0.28))
        prof = _pressure(len(dense), rng) if not s.closed else np.ones(len(dense))
        for q, pr in zip(dense, prof):
            rad = max(0.5, w * 0.5 * pr)
            cv2.circle(mask, (int(q[0] * (1 << sh)), int(q[1] * (1 << sh))), int(rad * (1 << sh)), 255, -1,
                       cv2.LINE_AA, sh)


def strokes_extent(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    return strokes_bbox(strokes)


def render_text_mask(text: str, st: TextStyle, em_px: float, rng, vertical: bool = False,
                     pad=(4, 4, 4, 4), ss: int = SS):
    """Simple helper: render a text line tightly with padding (l, t, r, b). Returns float mask in [0,1]."""
    strokes = layout(text, st, rng, vertical)
    for s in strokes:
        s.pts = s.pts * em_px
        s.r *= em_px
    x0, y0, x1, y1 = strokes_bbox(strokes)
    half = st.sw * em_px / 2
    x0, y0, x1, y1 = x0 - half, y0 - half, x1 + half, y1 + half
    W = int(math.ceil(x1 - x0 + pad[0] + pad[2]))
    H = int(math.ceil(y1 - y0 + pad[1] + pad[3]))
    _translate(strokes, pad[0] - x0, pad[1] - y0)
    big = np.zeros((H * ss, W * ss), np.uint8)
    draw_strokes(big, strokes, st.sw * em_px, st.brush, rng, ss)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
