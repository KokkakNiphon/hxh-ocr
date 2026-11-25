"""Reference post-processing for the DBNet-lite line detector (the Rust / browser side ports this).

Input: the model output for one page, [2, 320, 320] float (channel 0 = text-kernel logit,
channel 1 = vertical logit), plus the letterbox scales (see det_data.letterbox).

  1. Binarise without a sigmoid: prob > 0.3  <=>  logit > log(0.3 / 0.7) = -0.8473.
  2. Connected components (8-connectivity) of the binary map.
  3. Per component: drop it if it has fewer than `min_area` pixels or its score (mean
     sigmoid(logit) over its pixels) is below `box_thresh`. Orientation: vertical if the mean
     vertical logit over its pixels is > 0.
  4. cv2.minAreaRect of the pixel centres; the extent grows by 1 px per axis (pixels have unit
     size); then unclip: both sides grow by 2 * D (see `unclip_distance`; default D = 0.7 x the
     exact inverse of the training shrink).
  5. Map the corners back: output u -> input 2u + 0.5 -> original (x_in + 0.5) / sx - 0.5
     (per axis), then order them clockwise from the top-left in reading orientation (vertical
     lines start at the column's top-right, like the generator).
  6. Drop boxes whose short side is below `min_side` original pixels.

`reading_order` groups lines into blocks and orders them (blocks top-to-bottom; within a block,
horizontal lines top-to-bottom, vertical columns right-to-left).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

LOGIT_THRESH = math.log(0.3 / 0.7)  # -0.8473: sigmoid(x) > 0.3
SHRINK_R = 0.4                       # must match the targets' shrink ratio


@dataclass
class PostParams:
    logit_thresh: float = LOGIT_THRESH
    box_thresh: float = 0.5        # mean sigmoid(prob logit) over the component
    min_area: int = 3              # component pixels at stride 2
    unclip: str = "exact"          # "exact": inverse of the target shrink; "ratio": DBNet's A*r/L
    unclip_ratio: float = 1.5
    unclip_scale: float = 0.7      # multiplies D; the predicted region at prob > 0.3 is fatter than the
                                   # training kernel, so the exact inverse over-grows. 0.7 was tuned on
                                   # pages_v2 val[300:600] for det_v1 (F1 0.922 vs 0.909 at 1.0; DBNet's
                                   # ratio 1.5 gives 0.920)
    min_side: float = 3.0          # original px


@dataclass
class Det:
    quad: np.ndarray               # [4, 2] original image coords, clockwise from reading top-left
    vertical: bool
    score: float
    vert_logit: float
    length: float = 0.0            # along the reading direction (original px)
    thickness: float = 0.0         # across (original px)
    text: str = ""
    extra: dict = field(default_factory=dict)


def unclip_distance(wk: float, hk: float, p: PostParams) -> float:
    """Outward offset D for a kernel rectangle wk x hk (output px).

    exact: the D for which a (wk + 2D) x (hk + 2D) rectangle shrinks to the kernel under the
    training rule D = A (1 - r^2) / L, i.e. the positive root of
    (8 - 4a) D^2 + (2 - 2a)(wk + hk) D - a wk hk = 0 with a = 1 - r^2.
    ratio: DBNet's D = A * ratio / L of the kernel rectangle."""
    if p.unclip == "ratio":
        d = wk * hk * p.unclip_ratio / max(2.0 * (wk + hk), 1e-6)
    else:
        a = 1.0 - SHRINK_R * SHRINK_R
        A, B, C = 8.0 - 4.0 * a, (2.0 - 2.0 * a) * (wk + hk), -a * wk * hk
        d = (-B + math.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    return d * p.unclip_scale


def _rect_axes(pts: np.ndarray):
    """minAreaRect of points -> centre, unit axes (a, b), extents along them (pixel extent)."""
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float64)
    c = box.mean(axis=0)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 > 1e-6:
        a = e1 / l1
    elif l2 > 1e-6:
        a = np.array([e2[1], -e2[0]]) / l2
    else:
        a = np.array([1.0, 0.0])
    b = np.array([-a[1], a[0]])
    return c, a, b, l1 + 1.0, l2 + 1.0


def order_quad(c, a, la, b, lb, vertical: bool) -> np.ndarray:
    """Corners of the rectangle (centre c, unit axes a/b with full extents la/lb), clockwise from
    the top-left in reading orientation (vertical: the column's top-right first)."""
    cands = [(a, la, b, lb), (-a, la, b, lb), (b, lb, a, la), (-b, lb, a, la)]
    if vertical:  # reading direction = the axis closest to +y (down)
        u, lu, _, lv = max(cands, key=lambda t: t[0][1])
        r = np.array([u[1], -u[0]])  # u = (0, 1) -> r = (1, 0)
        return np.array([c - u * lu / 2 + r * lv / 2, c + u * lu / 2 + r * lv / 2,
                         c + u * lu / 2 - r * lv / 2, c - u * lu / 2 - r * lv / 2])
    u, lu, _, lv = max(cands, key=lambda t: t[0][0])  # reading direction closest to +x
    v = np.array([-u[1], u[0]])  # u = (1, 0) -> v = (0, 1) (down)
    return np.array([c - u * lu / 2 - v * lv / 2, c + u * lu / 2 - v * lv / 2,
                     c + u * lu / 2 + v * lv / 2, c - u * lu / 2 + v * lv / 2])


def decode(out: np.ndarray, sx: float, sy: float, p: PostParams | None = None,
           stride: int = 2) -> list[Det]:
    """out [2, h, w] logits for one page -> detections in original image coordinates."""
    p = p or PostParams()
    prob, vlog = out[0].astype(np.float32), out[1].astype(np.float32)
    mask = (prob > p.logit_thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < p.min_area:
            continue
        sub = labels[y:y + h, x:x + w] == i
        lg = prob[y:y + h, x:x + w][sub]
        score = float(np.mean(1.0 / (1.0 + np.exp(-lg))))
        if score < p.box_thresh:
            continue
        vmean = float(vlog[y:y + h, x:x + w][sub].mean())
        ys, xs = np.nonzero(sub)
        pts = np.stack([xs + x, ys + y], 1).astype(np.float32)
        c, a, b, la, lb = _rect_axes(pts)
        d = unclip_distance(la, lb, p)
        la, lb = la + 2 * d, lb + 2 * d
        vertical = vmean > 0
        q = order_quad(c, a, la, b, lb, vertical)
        q = q * stride + (stride - 1) / 2.0           # output px -> input px (2u + 0.5)
        q[:, 0] = (q[:, 0] + 0.5) / sx - 0.5          # input px -> original px
        q[:, 1] = (q[:, 1] + 0.5) / sy - 0.5
        length = 0.5 * (np.hypot(*(q[1] - q[0])) + np.hypot(*(q[2] - q[3])))
        thick = 0.5 * (np.hypot(*(q[3] - q[0])) + np.hypot(*(q[2] - q[1])))
        if min(length, thick) < p.min_side:
            continue
        dets.append(Det(q, vertical, score, vmean, float(length), float(thick)))
    return dets


# ----------------------------------------------------------------------------
# reading order
# ----------------------------------------------------------------------------
def _reading_dir(d: Det) -> np.ndarray:
    v = d.quad[1] - d.quad[0]
    return v / max(np.hypot(*v), 1e-9)


def reading_order(dets: list[Det], gap: float = 0.8, max_ratio: float = 2.0):
    """Group lines into blocks and order them. Returns (order, blocks): `order` is a list of
    indices into `dets`, `blocks` a list of index lists (each in reading order).

    Two lines join a block when they share the orientation, their thicknesses differ by at most
    `max_ratio`, they overlap along the reading direction (x for horizontal, y for vertical
    lines) and the gap across is at most `gap` x the thinner line's thickness. Blocks are read
    top-to-bottom (by their top edge); inside a block horizontal lines top-to-bottom and
    vertical columns right-to-left."""
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = np.array([[d.quad[:, 0].min(), d.quad[:, 1].min(), d.quad[:, 0].max(), d.quad[:, 1].max()]
                      for d in dets]).reshape(-1, 4)
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = dets[i], dets[j]
            if di.vertical != dj.vertical:
                continue
            ti, tj = max(di.thickness, 1e-6), max(dj.thickness, 1e-6)
            if max(ti, tj) / min(ti, tj) > max_ratio:
                continue
            bi, bj = boxes[i], boxes[j]
            gx = max(bi[0], bj[0]) - min(bi[2], bj[2])  # > 0: gap, < 0: overlap
            gy = max(bi[1], bj[1]) - min(bi[3], bj[3])
            t = min(ti, tj)
            if di.vertical:
                ok = gy < 0 and gx <= gap * t
            else:
                ok = gx < 0 and gy <= gap * t
            if ok:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    blocks = []
    for members in groups.values():
        if dets[members[0]].vertical:  # columns right-to-left
            members.sort(key=lambda k: -dets[k].quad[:, 0].mean())
        else:  # lines top-to-bottom along the block's mean "down" direction
            u = np.mean([_reading_dir(dets[k]) for k in members], axis=0)
            v = np.array([-u[1], u[0]])
            members.sort(key=lambda k: float(dets[k].quad.mean(axis=0) @ v))
        blocks.append(members)
    blocks.sort(key=lambda m: (min(boxes[k][1] for k in m), -max(boxes[k][2] for k in m)))
    order = [k for m in blocks for k in m]
    return order, blocks


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------
def poly_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    u = poly_area(a) + poly_area(b) - inter
    return float(inter) / u if u > 0 else 0.0


def match(gt: np.ndarray, pred: list[np.ndarray], thr: float = 0.5):
    """Greedy one-to-one matching by IoU (descending). Returns [(gt_i, pred_j, iou)]."""
    if len(gt) == 0 or len(pred) == 0:
        return []
    gb = np.stack([gt[:, :, 0].min(1), gt[:, :, 1].min(1), gt[:, :, 0].max(1), gt[:, :, 1].max(1)], 1)
    P = np.stack(pred)
    pb = np.stack([P[:, :, 0].min(1), P[:, :, 1].min(1), P[:, :, 0].max(1), P[:, :, 1].max(1)], 1)
    cand = []
    for i in range(len(gt)):
        ov = (np.minimum(gb[i, 2], pb[:, 2]) > np.maximum(gb[i, 0], pb[:, 0])) & \
             (np.minimum(gb[i, 3], pb[:, 3]) > np.maximum(gb[i, 1], pb[:, 1]))
        for j in np.nonzero(ov)[0]:
            iou = quad_iou(gt[i], P[j])
            if iou >= thr:
                cand.append((iou, i, int(j)))
    cand.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j)
        out.append((i, j, iou))
    return out
