"""Toy "detector-like" model for the hxh-infer Resize / Concat bit-exactness proof, plus a
synthetic page generator (dark glyph-like strokes in lines on a noisy light background).

The model mirrors the planned DBNet-lite text-line detector at a small size:
input 1x1x128x128 grayscale -> output [1, 2, 64, 64] (stride 2): channel 0 = text-kernel
logit, channel 1 = vertical-orientation logit. It uses every op the real detector will:

  stem   Conv 3x3 s2, Cin 1 -> 8, ReLU                         8 x 64 x 64   (c1)
  b1     depthwise 3x3 s2 + 1x1 8 -> 16, ReLU                  16 x 32 x 32  (c2)
  b2     depthwise 3x3 s2 + 1x1 16 -> 24, ReLU                 24 x 16 x 16  (c3)
  b3     depthwise 3x3 s2 + 1x1 24 -> 32, ReLU                 32 x 8 x 8    (c4)
  FPN    1x1 laterals to 16 ch; p3 = l3 + up2(l4); p2 = l2 + up2(p3);
         p1 = relu(l1 + up2(p2))  (Add, Add + ReLU -> LUT, Resize nearest x2 on 16 ch)
  fuse   o1 = conv3x3(p1) 8 ch | up2(conv3x3(p2)) 16 ch | up4(conv3x3(p3)) 8 ch |
         up8(conv1x1(l4)) 24 ch | p1 16 ch | c1 8 ch -> Concat (channels) = 80 ch at 64 x 64
         (Resize x2 / x4 / x8 with 16 / 8 / 24 channels: the SIMD fast path, the scalar
         fallback and the overlapping-tail path; Concat segments of 8, 16, 8, 24, 16, 8
         bytes; p1 and c1 also feed other ops, so ESP-PPQ cannot move their exponents and
         inserts RequantizeLinear in front of the Concat when they differ from its output's)
  head   conv3x3 80 -> 16, ReLU; conv1x1 16 -> 2

BatchNorm is folded before export. Training is a short CPU run on synthetic pages (about
a minute), only so that the maps look like a detector's (blobs on text lines) rather than
noise; the kernel target uses a shrink ratio of 0.7 (D = A (1 - 0.7^2) / L), larger than
DBNet's 0.4, because the toy lines are thin (5-14 px).

Usage (from the repo root, .venv-ppq):
  python experiments/det_engine/toy_det.py --train-seconds 60      # -> models/toy_det.onnx (+ .pt)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
SIZE = 128
STRIDE = 2
SHRINK = 0.7


# ------------------------------------------------------------------ synthetic pages


def _glyph_strokes(rng, cell_w, cell_h):
    """A few random strokes inside a cell (local coords, origin top-left)."""
    out = []
    for _ in range(rng.integers(2, 5)):
        p = rng.uniform([0.1 * cell_w, 0.1 * cell_h], [0.9 * cell_w, 0.9 * cell_h], (2, 2))
        out.append(p)
    return out


def make_page(rng: np.random.Generator, size: int = SIZE, max_lines: int = 5):
    """Return (u8 image [size, size], lines) with lines = [{"quad": 4x2 list, "vertical": bool}].

    Quads: pixel centres at integers, clockwise from the top-left in reading orientation
    (horizontal TL, TR, BR, BL; vertical TR, BR, BL, TL), like rust/hxh-synth."""
    bg = rng.uniform(170, 250)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = bg + rng.uniform(-30, 30) * (xx / size) + rng.uniform(-30, 30) * (yy / size)
    img += rng.normal(0, rng.uniform(2, 12), (size, size))
    img = np.clip(img, 0, 255).astype(np.float32)
    lines = []
    occupied = np.zeros((size, size), np.uint8)
    for _ in range(rng.integers(1, max_lines + 1)):
        for _attempt in range(20):
            vertical = rng.random() < 0.3
            h = rng.uniform(5, 14)  # line thickness (glyph height)
            n = int(rng.integers(1, 9))
            adv = h * rng.uniform(0.8, 1.1)
            length = n * adv
            if length > 0.8 * size:
                continue
            ang = math.radians(rng.uniform(-10, 10))
            cx, cy = rng.uniform(0.15 * size, 0.85 * size, 2)
            # reading axis a and cross axis nrm (= a rotated +90 deg in y-down coordinates)
            a = np.array([math.cos(ang), math.sin(ang)]) if not vertical else np.array([-math.sin(ang), math.cos(ang)])
            nrm = np.array([-a[1], a[0]])
            c = np.array([cx, cy])
            q = np.array([c - a * length / 2 - nrm * h / 2, c + a * length / 2 - nrm * h / 2,
                          c + a * length / 2 + nrm * h / 2, c - a * length / 2 + nrm * h / 2])
            if q.min() < 2 or q.max() > size - 3:
                continue
            m = np.zeros_like(occupied)
            grow = q + np.sign(q - c) * 3
            cv2.fillPoly(m, [np.round(grow * 16).astype(np.int32)], 1, shift=4)
            if (m & occupied).any():
                continue
            occupied |= m
            ink = rng.uniform(0, 80)
            thick = max(1, int(round(h / 6)))
            canvas = np.zeros((size, size), np.uint8)
            for k in range(n):
                for p in _glyph_strokes(rng, adv * 0.9, h):
                    # local (along, across) -> image; the cell starts at q0 + k * adv along a
                    pts = [q[0] + a * (k * adv + u) + nrm * v for u, v in p]
                    pts = np.round(np.array(pts) * 16).astype(np.int32)
                    cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), 255, thick, cv2.LINE_AA, shift=4)
            alpha = canvas.astype(np.float32) / 255
            img = img * (1 - alpha) + ink * alpha
            lines.append({"quad": q.tolist(), "vertical": bool(vertical)})
            break
    # a few distractor dots / scratches
    for _ in range(rng.integers(0, 4)):
        p = rng.uniform(0, size, 2)
        cv2.circle(img, tuple(int(v) for v in p), int(rng.integers(1, 3)), float(rng.uniform(0, 120)), -1)
    return np.clip(img, 0, 255).astype(np.uint8), lines


def shrink_quad(q: np.ndarray, r: float = SHRINK) -> np.ndarray | None:
    """DBNet shrink of a rectangle-like quad: every edge moves inward by D = A (1 - r^2) / L."""
    area = 0.5 * abs(np.cross(q[1] - q[0], q[3] - q[0])) + 0.5 * abs(np.cross(q[3] - q[2], q[1] - q[2]))
    per = sum(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    d = area * (1 - r * r) / per
    c = q.mean(0)
    a = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0])
    nrm = (q[3] - q[0]) / np.linalg.norm(q[3] - q[0])
    L = np.linalg.norm(q[1] - q[0]) - 2 * d
    T = np.linalg.norm(q[3] - q[0]) - 2 * d
    if L <= 0 or T <= 0:
        return None
    return np.array([c - a * L / 2 - nrm * T / 2, c + a * L / 2 - nrm * T / 2,
                     c + a * L / 2 + nrm * T / 2, c - a * L / 2 + nrm * T / 2])


def targets(lines, size: int = SIZE, stride: int = STRIDE):
    """Kernel and vertical maps at the output grid (pixel-centre convention:
    grid coordinate g = (x + 0.5) / stride - 0.5)."""
    g = size // stride
    kern = np.zeros((g, g), np.float32)
    vert = np.zeros((g, g), np.float32)
    for ln in lines:
        k = shrink_quad(np.array(ln["quad"]))
        if k is None:
            continue
        kg = (k + 0.5) / stride - 0.5
        m = np.zeros((g, g), np.uint8)
        cv2.fillPoly(m, [np.round(kg * 16).astype(np.int32)], 1, shift=4)
        if not m.any():  # thinner than a grid pixel: mark the nearest pixel of the centre
            cxy = np.clip(np.round(kg.mean(0)).astype(int), 0, g - 1)
            m[cxy[1], cxy[0]] = 1
        kern = np.maximum(kern, m)
        if ln["vertical"]:
            vert = np.maximum(vert, m)
    return kern, vert


def normalize(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255.0 - 0.5) / 0.5


# ------------------------------------------------------------------ model


def cbr(cin, cout, k, s=1, groups=1, relu=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def dw_pw(cin, cout, s):
    return nn.Sequential(cbr(cin, cin, 3, s, groups=cin), cbr(cin, cout, 1))


class ToyDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = cbr(1, 8, 3, 2)
        self.b1 = dw_pw(8, 16, 2)
        self.b2 = dw_pw(16, 24, 2)
        self.b3 = dw_pw(24, 32, 2)
        self.l1, self.l2, self.l3, self.l4 = cbr(8, 16, 1, relu=False), cbr(16, 16, 1, relu=False), \
            cbr(24, 16, 1, relu=False), cbr(32, 16, 1, relu=False)
        self.o1 = cbr(16, 8, 3)
        self.o2 = cbr(16, 16, 3)
        self.o3 = cbr(16, 8, 3)
        self.o4 = cbr(16, 24, 1)
        self.head = nn.Sequential(cbr(80, 16, 3), nn.Conv2d(16, 2, 1))

    @staticmethod
    def up(x, s):
        return F.interpolate(x, scale_factor=s, mode="nearest")

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.b1(c1)
        c3 = self.b2(c2)
        c4 = self.b3(c3)
        l4 = self.l4(c4)
        p3 = self.l3(c3) + self.up(l4, 2)
        p2 = self.l2(c2) + self.up(p3, 2)
        p1 = F.relu(self.l1(c1) + self.up(p2, 2))
        f = torch.cat([self.o1(p1), self.up(self.o2(p2), 2), self.up(self.o3(p3), 4), self.up(self.o4(l4), 8), p1, c1], 1)
        return self.head(f)


def fold_bn(model: nn.Module) -> nn.Module:
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model = copy.deepcopy(model).eval()
    for mod in list(model.modules()):
        if isinstance(mod, nn.Sequential):
            for i in range(len(mod) - 1):
                if isinstance(mod[i], nn.Conv2d) and isinstance(mod[i + 1], nn.BatchNorm2d):
                    mod[i] = fuse_conv_bn_eval(mod[i], mod[i + 1])
                    mod[i + 1] = nn.Identity()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return model


def batch(rng, n):
    xs, ks, vs = [], [], []
    for _ in range(n):
        img, lines = make_page(rng)
        k, v = targets(lines)
        xs.append(normalize(img)[None])
        ks.append(k)
        vs.append(v)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ks)), torch.from_numpy(np.stack(vs))


def train(model: ToyDet, seconds: float, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train()
    step, t0, losses = 0, time.time(), []
    while time.time() - t0 < seconds:
        x, k, v = batch(rng, 16)
        out = model(x)
        p = torch.sigmoid(out[:, 0])
        bce = F.binary_cross_entropy_with_logits(out[:, 0], k, pos_weight=torch.tensor(4.0))
        dice = 1 - (2 * (p * k).sum() + 1) / (p.sum() + k.sum() + 1)
        orient = (F.binary_cross_entropy_with_logits(out[:, 1], v, reduction="none") * k).sum() / (k.sum() + 1)
        loss = bce + dice + orient
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(losses[-50:]):.3f} ({time.time() - t0:.0f}s)")
    model.eval()
    return {"steps": step, "final_loss": float(np.mean(losses[-20:])) if losses else None, "seconds": time.time() - t0}


def export_onnx(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, SIZE, SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (x,), str(path), input_names=["input"], output_names=["maps"], opset_version=18,
                          do_constant_folding=True, dynamo=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = ToyDet()
    info = train(model, args.train_seconds, args.seed) if args.train_seconds > 0 else {}
    print("training:", info)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "toy_det.pt")
    fused = fold_bn(model)
    x = torch.from_numpy(np.stack([normalize(make_page(np.random.default_rng(99))[0])[None]]))
    with torch.no_grad():
        d = (fused(x) - model(x)).abs().max().item()
    print(f"BN fold max abs diff: {d:.2e}")
    export_onnx(fused, MODELS / "toy_det.onnx")
    n_params = sum(p.numel() for p in fused.parameters())
    (MODELS / "toy_det_info.json").write_text(json.dumps({**info, "params": n_params, "bn_fold_maxdiff": d}, indent=1))
    print(f"exported {MODELS / 'toy_det.onnx'} ({n_params} params)")


if __name__ == "__main__":
    main()
