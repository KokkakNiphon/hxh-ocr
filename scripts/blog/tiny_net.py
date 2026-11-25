"""A tiny, real neural network for the explainer's "neural network up close" figure.

Six symbols (あ い く ぐ と ど), drawn from the same glyph strokes the training generator uses
(web/blog/data/glyphs.json), rasterised the way the page rasterises them (64 px canvas, round caps,
area-averaged down to 16 x 16), with random wobble, size, position, slant and pen width.
A 256 -> 12 -> 8 -> 6 network (ReLU, softmax) is trained on them and written, weights and all,
to web/blog/data/tinynet.json; the page runs it live in JavaScript.

    .venv\\Scripts\\python.exe scripts\\blog\\tiny_net.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
G = json.loads((ROOT / "web" / "blog" / "data" / "glyphs.json").read_text(encoding="utf-8"))
OUT = ROOT / "web" / "blog" / "data" / "tinynet.json"
CLASSES = ["あ", "い", "く", "ぐ", "と", "ど"]
N, R = 16, 64          # input grid, render canvas
EM = 0.66              # em size as a fraction of the canvas (the page draws at this size)
SS = 4                 # supersampling for anti-aliased round strokes


def shape_of(ch: str):
    comp = G["composed"].get(ch)
    g = G["glyphs"][comp["base"] if comp else ch]
    return g, (comp["mark"] if comp else None)


def render(ch: str, rng: np.random.Generator | None = None, form: int = 0) -> np.ndarray:
    """Ink coverage on an N x N grid, 0..1. rng None = the page's clean drawing."""
    g, mark = shape_of(ch)
    f = g["forms"][form % len(g["forms"])]
    if rng is None:
        S, cx, cy, wscale, jit, rot = EM * R, R / 2, R / 2, 1.0, 0.0, 0.0
    else:
        S = R * rng.uniform(0.52, 0.8)
        cx = R / 2 + rng.uniform(-5, 5)
        cy = R / 2 + rng.uniform(-5, 5)
        wscale = rng.uniform(0.8, 1.9)
        jit = rng.uniform(0.0, 0.045)
        rot = math.radians(rng.uniform(-10, 10))
    cr, sr = math.cos(rot), math.sin(rot)
    j = (lambda: rng.uniform(-jit, jit)) if rng is not None and jit > 0 else (lambda: 0.0)

    def P(x, y):
        x, y = x + j(), y + j()
        return ((cx + (x * cr - y * sr) * S) * SS, (cy + (x * sr + y * cr) * S) * SS)

    img = np.zeros((R * SS, R * SS), np.uint8)
    sw = G["stroke_width_em"] * wscale * S * SS
    t = max(1, int(round(sw)))
    for st in f["strokes"]:
        if st["type"] == "dot":
            x, y = P(*st["c"])
            cv2.circle(img, (int(x), int(y)), int(round(max(st["r"] * S * SS, sw / 2))), 255, -1, cv2.LINE_AA)
        else:
            pts = [P(*p) for p in st["pts"]]
            arr = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], bool(st.get("closed")), 255, t, cv2.LINE_AA)
            for x, y in pts:  # round caps and joins
                cv2.circle(img, (int(x), int(y)), t // 2, 255, -1, cv2.LINE_AA)
    if mark:
        m = G["marks"][mark]
        x1, y1 = f["bbox"][2], f["bbox"][3]
        x, y = P(x1 + m["dx"], y1 + m["dy"])
        if m["hollow"]:
            w = G["stroke_width_em"] * m["width_scale"] * wscale * S * SS
            cv2.circle(img, (int(x), int(y)), int(round(max(m["r"], G["stroke_width_em"] * m["width_scale"]) * S * SS)),
                       255, max(1, int(round(w))), cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), int(round(max(m["r"] * S * SS, sw / 2))), 255, -1, cv2.LINE_AA)
    return cv2.resize(img, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def dataset(n_per: int, seed: int):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for k, ch in enumerate(CLASSES):
        nf = len(shape_of(ch)[0]["forms"])
        for i in range(n_per):
            a = render(ch, rng, form=int(rng.integers(nf)))
            if rng.random() < 0.3:  # a few missing or extra pixels, like a hand-edited grid
                m = rng.random(a.shape) < 0.03
                a = np.where(m, 1 - a.round(), a)
            X.append(a.ravel())
            Y.append(k)
    return torch.tensor(np.array(X)), torch.tensor(Y)


def main():
    torch.manual_seed(0)
    Xtr, Ytr = dataset(4000, 1)
    Xva, Yva = dataset(600, 2)
    net = nn.Sequential(nn.Linear(N * N, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, len(CLASSES)))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(60):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            with torch.no_grad():
                acc = (net(Xva).argmax(1) == Yva).float().mean().item()
            print(f"epoch {ep + 1}: loss {loss.item():.3f}, held-out accuracy {acc * 100:.1f}%")
    net.eval()
    with torch.no_grad():
        acc = (net(Xva).argmax(1) == Yva).float().mean().item()
        clean = torch.tensor(np.array([render(c).ravel() for c in CLASSES]))
        pc = torch.softmax(net(clean), 1)
        for c, p in zip(CLASSES, pc):
            print(c, CLASSES[int(p.argmax())], f"{float(p.max()) * 100:.1f}%")
        # the teaching moment: ぐ without its dot should read as く
        g, _ = shape_of("ぐ")
        nodot = torch.tensor(render("く").ravel()[None])
        print("く clean ->", CLASSES[int(net(nodot).argmax())])
    L = [m for m in net if isinstance(m, nn.Linear)]
    r = lambda t: [[round(float(v), 4) for v in row] for row in t.detach().numpy()]
    data = {
        "about": "A 256-12-8-6 ReLU network trained by scripts/blog/tiny_net.py on wobbly 16x16 renders of six symbols.",
        "classes": CLASSES,
        "grid": N, "canvas": R, "em": EM,
        "layers": [{"w": r(l.weight), "b": [round(float(v), 4) for v in l.bias.detach().numpy()]} for l in L],
        "heldout_accuracy": round(acc, 4),
        "train_examples": len(Xtr),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"held-out accuracy {acc * 100:.1f}% on {len(Xva)} renders; {OUT.stat().st_size / 1024:.0f} KB")
    # a look at the training renders
    rng = np.random.default_rng(5)
    tiles = [np.hstack([cv2.resize(render(c, rng), (64, 64), interpolation=cv2.INTER_NEAREST) for _ in range(8)]) for c in CLASSES]
    cv2.imwrite(str(Path(__file__).with_name("_tinynet_samples.png")), (np.vstack(tiles) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
