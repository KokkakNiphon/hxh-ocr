"""Export a checkpoint (EMA weights) to ONNX with BatchNorm folded away.

  python scripts/export_onnx.py --run runs/mnv4s_v1 --width 256 --dynamic

Writes <run>/model_{H}x{W}.onnx (static input 1x1xHxW) and, with --dynamic,
<run>/model_{H}xN_dyn.onnx (batch fixed at 1, width dynamic).  Fails if a BatchNormalization
node survives, and compares onnxruntime against the fused PyTorch model.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.engine import load_checkpoint  # noqa: E402
from hxh_ocr.model import count_macs, count_params, fuse_bn  # noqa: E402


def export(model, path, height, width, opset, dynamic, dynamo):
    x = torch.randn(1, 1, height, width)
    kw = {}
    if dynamic:
        kw["dynamic_axes"] = {"image": {3: "width"}, "logits": {3: "time"}}
    torch.onnx.export(model, (x,), str(path), input_names=["image"], output_names=["logits"],
                      opset_version=opset, dynamo=dynamo, **kw)


def check(model, path, height, widths, expect_dynamic_ok=True):
    import onnx
    import onnxruntime as ort
    g = onnx.load(str(path))
    onnx.checker.check_model(g)
    ops = Counter(n.op_type for n in g.graph.node)
    assert "BatchNormalization" not in ops, f"BatchNormalization left in graph: {dict(ops)}"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for w in widths:
        x = torch.randn(1, 1, height, w)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        assert ref.shape == got.shape, (ref.shape, got.shape)
        worst = max(worst, float(np.abs(ref - got).max()))
        same = (ref.argmax(1) == got.argmax(1)).all()
        print(f"  W={w}: out {tuple(got.shape)}  max|diff| {np.abs(ref - got).max():.2e}  argmax equal {same}")
    return ops, worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default=None, help="default <run>/best.pt")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--height", type=int, default=None, help="default: training height")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic", action="store_true", help="also export a dynamic-width variant")
    p.add_argument("--dynamo", action="store_true", help="use the torch.export based exporter")
    p.add_argument("--out", default=None, help="output dir (default: run dir)")
    args = p.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_checkpoint(args.ckpt or run / "best.pt", "cpu", args.weights)
    H = args.height or cfg["height"]
    fused = fuse_bn(model, (H, args.width))
    with torch.no_grad():
        x = torch.randn(2, 1, H, args.width)
        print(f"fuse_bn: max|diff| vs unfused (eval) = {(model(x) - fused(x)).abs().max():.2e}")
    print(f"fused model: {count_params(fused)/1e3:.1f}k params, "
          f"{count_macs(fused, H, args.width)/1e6:.1f}M MACs @ {H}x{args.width}")

    jobs = [(out / f"model_{H}x{args.width}.onnx", False, [args.width])]
    if args.dynamic:
        jobs.append((out / f"model_{H}xN_dyn.onnx", True, [args.width, 128, 384, 512]))
    for path, dyn, widths in jobs:
        export(fused, path, H, args.width, args.opset, dyn, args.dynamo)
        print(f"{path}  ({path.stat().st_size/1024:.1f} KiB, opset {args.opset}, {'dynamic width' if dyn else 'static'})")
        ops, worst = check(fused, path, H, widths)
        print(f"  ops: {dict(sorted(ops.items()))}")
        print(f"  onnxruntime vs PyTorch worst max|diff| = {worst:.2e}  ->  {'OK' if worst < 1e-3 else 'MISMATCH'}")
        if worst >= 1e-3:
            sys.exit(1)


if __name__ == "__main__":
    main()
