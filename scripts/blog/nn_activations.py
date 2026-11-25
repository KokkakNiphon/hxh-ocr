"""Real activations of the line reader on the blog's example line, for the "inside the reader" figure.

Runs the float reader runs/v4_mnv4s_h48/best.pt (the base of the deployed int8 model) on
web/blog/img/ctc_strip.png (the exact 48 x 768 strip the int8 reader saw) and writes:
  web/blog/img/nn/input.png, stage{0..4}_{k}.png   the 4 most active feature maps of each stage
  web/blog/img/nn/features.png                     the per-column feature vectors (C x 192)
  web/blog/data/nn.json                            shapes, channel counts, params per stage, the letter
                                                   probabilities of the most likely classes per column,
                                                   and for every column the span of input pixels that
                                                   move its answer (gradient of its best score)

    .venv\\Scripts\\python.exe scripts\\blog\\nn_activations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hxh_ocr.engine import load_checkpoint  # noqa: E402

OUT_IMG = ROOT / "web" / "blog" / "img" / "nn"
OUT_JSON = ROOT / "web" / "blog" / "data" / "nn.json"


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-9)


def save_gray(path: Path, a: np.ndarray):
    cv2.imwrite(str(path), (norm01(a) * 255).round().astype(np.uint8))


def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    model, cfg = load_checkpoint(ROOT / "runs" / "v4_mnv4s_h48" / "best.pt", "cpu")
    model.eval()
    tokens = cfg["tokens"]
    strip = cv2.imread(str(ROOT / "web" / "blog" / "img" / "ctc_strip.png"), cv2.IMREAD_GRAYSCALE)
    assert strip.shape == (48, 768), strip.shape
    x = torch.from_numpy(((strip.astype(np.float32) / 255.0) - 0.5) / 0.5)[None, None]
    cv2.imwrite(str(OUT_IMG / "input.png"), strip)

    acts = {}
    hooks = [model.bn1.register_forward_hook(lambda m, i, o: acts.__setitem__("stage0", o.detach()))]
    for si, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, si=si: acts.__setitem__(f"stage{si + 1}", o.detach())))
    hooks.append(model.head.blocks.register_forward_hook(lambda m, i, o: acts.__setitem__("features", o.detach())))
    with torch.no_grad():
        logits = model(x)[0, :, 0, :]  # [C, T]
    for h in hooks:
        h.remove()

    stages = []
    names = ["stage0", "stage1", "stage2", "stage3", "stage4"]
    params = {
        "stage0": sum(p.numel() for p in list(model.conv_stem.parameters()) + list(model.bn1.parameters())),
        **{f"stage{i + 1}": sum(p.numel() for p in b.parameters()) for i, b in enumerate(model.blocks)},
    }
    for n in names:
        a = acts[n][0].numpy()  # [C, h, w]
        # the 4 channels with the most spatial variation: the ones that "see" something here
        score = a.reshape(a.shape[0], -1).std(1)
        top = np.argsort(-score)[:4]
        files = []
        for k, c in enumerate(top):
            f = f"{n}_{k}.png"
            save_gray(OUT_IMG / f, a[c])
            files.append(f)
        stages.append({"name": n, "channels": int(a.shape[0]), "height": int(a.shape[1]), "width": int(a.shape[2]),
                       "params": int(params[n]), "maps": files, "shown_channels": [int(c) for c in top]})

    feats = acts["features"][0, :, 0, :].numpy()  # [C, T]
    save_gray(OUT_IMG / "features.png", feats)
    head_params = sum(p.numel() for p in model.head.parameters())

    probs = torch.softmax(logits, 0).numpy()  # [75, T]
    T = probs.shape[1]
    ids = probs.argmax(0)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(tokens[i - 1])
        prev = i
    text = "".join(out)
    # classes worth a row: blank, every class that is best somewhere, plus the strongest runners-up
    keep = [0] + sorted({int(i) for i in ids if i != 0}, key=lambda c: int(np.argmax(probs[c])))
    runners = np.argsort(-probs.max(1))
    for c in runners:
        if len(keep) >= 14:
            break
        if int(c) not in keep:
            keep.append(int(c))
    rows = [{"id": c, "label": "blank" if c == 0 else tokens[c - 1], "p": [round(float(v), 3) for v in probs[c]]} for c in keep]

    # what each column looks at: |d best-score / d input|, summed over the height, per input pixel column
    xg = x.clone().requires_grad_(True)
    lg = model(xg)[0, :, 0, :]
    spans = []
    for t in range(T):
        if xg.grad is not None:
            xg.grad = None
        lg[int(ids[t]), t].backward(retain_graph=True)
        prof = xg.grad[0, 0].abs().sum(0).numpy()  # [768]
        cs = np.cumsum(prof) / (prof.sum() + 1e-12)
        x0 = int(np.searchsorted(cs, 0.1))
        x1 = int(np.searchsorted(cs, 0.9))
        spans.append([x0, x1])

    data = {
        "source": "runs/v4_mnv4s_h48/best.pt (float) on web/blog/img/ctc_strip.png",
        "text": text,
        "input": {"height": 48, "width": 768, "file": "input.png"},
        "stages": stages,
        "features": {"channels": int(feats.shape[0]), "width": int(feats.shape[1]), "file": "features.png",
                     "head_params": int(head_params)},
        "classes": len(tokens) + 1,
        "timesteps": T,
        "rows": rows,
        "best": [int(i) for i in ids],
        "spans": spans,
        "total_params": int(sum(p.numel() for p in model.parameters())),
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    widths = [s[1] - s[0] for s in spans]
    print(f"text {text}; stages {[(s['channels'], s['height'], s['width']) for s in stages]}; features {feats.shape}; "
          f"params {data['total_params']:,}; column span median {int(np.median(widths))} px (10-90% of the gradient)")


if __name__ == "__main__":
    main()
