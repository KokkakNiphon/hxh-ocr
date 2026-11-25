"""Export the line detector (EMA weights) to a BN-folded ONNX for ESP-PPQ / the Rust engine.

  python scripts/export_det.py --run runs/det_v1 [--out exports/det_v1.onnx]

Interface: input "image" [1, 1, 640, 640] float, the 640 letterbox of a grayscale page normalised
to (x / 255 - 0.5) / 0.5 (see hxh_ocr/det_data.py); output "logits" [1, 2, 320, 320] (stride 2):
channel 0 = text-kernel logit, channel 1 = vertical-orientation logit. The threshold branch is
dropped. Fails unless the graph uses only Conv, Relu, Add, Resize (nearest, x2) and Concat, and
unless onnxruntime matches the fused PyTorch model.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr.det_data import LETTERBOX  # noqa: E402
from hxh_ocr.det_model import count_macs, load_det_checkpoint  # noqa: E402
from hxh_ocr.model import fuse_bn  # noqa: E402

ALLOWED = {"Conv", "Relu", "Add", "Resize", "Concat"}


def constants_to_initializers(model):
    """Move Constant nodes (Resize scales etc.) into initializers so the node list is ops only."""
    import onnx
    from onnx import numpy_helper
    g = model.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant":
            val = next(a for a in n.attribute if a.name == "value")
            t = numpy_helper.to_array(val.t)
            g.initializer.append(numpy_helper.from_array(t, n.output[0]))
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None, help="default <run>/best.pt")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out", default=None, help="default exports/<run name>.onnx")
    args = ap.parse_args()
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    run = Path(args.run)
    out = Path(args.out) if args.out else ROOT / "exports" / f"{run.name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_det_checkpoint(args.ckpt or run / "best.pt", "cpu", args.weights)
    model.thresh = None  # training-only branch
    fused = fuse_bn(model, (128, 128)).eval()
    x = torch.randn(2, 1, LETTERBOX, LETTERBOX)
    with torch.no_grad():
        print(f"fuse_bn: max|diff| vs unfused = {(model(x) - fused(x)).abs().max():.2e}")
    params = sum(p.numel() for p in fused.parameters())
    print(f"fused: {params/1e3:.1f}k params, {count_macs(fused, LETTERBOX)/1e6:.0f}M MACs @ {LETTERBOX}x{LETTERBOX}")

    torch.onnx.export(fused, (torch.zeros(1, 1, LETTERBOX, LETTERBOX),), str(out), input_names=["image"],
                      output_names=["logits"], opset_version=args.opset, dynamo=False, do_constant_folding=True)
    g = constants_to_initializers(onnx.load(str(out)))
    onnx.checker.check_model(g)
    ops = Counter(n.op_type for n in g.graph.node)
    bad = set(ops) - ALLOWED
    assert not bad, f"ops outside the allowed set: {bad} ({dict(ops)})"
    inits = {i.name: numpy_helper.to_array(i) for i in g.graph.initializer}
    shown = False
    for n in g.graph.node:
        if n.op_type == "Resize":
            at = {a.name: (a.s.decode() if a.type == onnx.AttributeProto.STRING else a) for a in n.attribute}
            assert at.get("mode", "nearest") == "nearest", at
            scales = inits.get(n.input[2]) if len(n.input) > 2 and n.input[2] else None
            sizes = inits.get(n.input[3]) if len(n.input) > 3 and n.input[3] else None
            assert (scales is not None and list(scales) == [1, 1, 2, 2]) or sizes is not None, (scales, sizes)
            if not shown:
                print(f"  Resize: mode={at.get('mode')} coord={at.get('coordinate_transformation_mode')} "
                      f"nearest_mode={at.get('nearest_mode')} scales={None if scales is None else list(scales)}")
                shown = True
        if n.op_type == "Concat":
            ax = next(a.i for a in n.attribute if a.name == "axis")
            assert ax == 1, ax
    shp = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
    assert shp(g.graph.input[0]) == [1, 1, LETTERBOX, LETTERBOX], shp(g.graph.input[0])
    assert shp(g.graph.output[0]) == [1, 2, LETTERBOX // 2, LETTERBOX // 2], shp(g.graph.output[0])
    onnx.save(g, str(out))
    print(f"{out}  ({out.stat().st_size/1024:.1f} KiB, opset {args.opset})")
    print(f"  ops: {dict(sorted(ops.items()))}")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(3):
        x = torch.randn(1, 1, LETTERBOX, LETTERBOX)
        with torch.no_grad():
            ref = fused(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"  onnxruntime vs PyTorch max|diff| = {worst:.2e} -> {'OK' if worst < 1e-3 else 'MISMATCH'}")
    if worst >= 1e-3:
        sys.exit(1)


if __name__ == "__main__":
    main()
