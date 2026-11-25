"""Per-tensor error of ESP-PPQ's simulation: quantize (no export), then compare
every intermediate tensor of the quantized graph against the same graph with
quantization switched off. Used to find where the w16a16 error comes from.

Usage: python diag_tensors.py --onnx models/toy_trained_avg_op18.onnx --quant-type w16a16
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from spike_common import SPIKE, load_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--quant-type", default="w8a8")
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.IR.quantize import QuantableOperation

    work = SPIKE / "out" / "_diag.onnx"
    shutil.copy(args.onnx, work)
    calib = load_lines("val", 64)
    loader = DataLoader(list(torch.from_numpy(calib)), batch_size=8)
    bits = 16 if args.quant_type == "w16a16" else 8
    graph = espdl_quantize_onnx(
        onnx_import_file=str(work), espdl_export_file=str(SPIKE / "out" / "_diag.espdl"),
        calib_dataloader=loader, calib_steps=len(loader), input_shape=[1, 1, 32, 256],
        target=args.target, quant_type=args.quant_type,
        setting=QuantizationSettingFactory.espdl_setting(num_of_bits=bits),
        collate_fn=lambda b: b.float(), device="cpu", error_report=False, skip_export=True,
    )
    x = torch.from_numpy(load_lines("val", args.n, offset=2000))
    ex = TorchExecutor(graph=graph, device="cpu")
    names = [v for op in ex._executing_order for v in (o.name for o in op.outputs)]
    with torch.no_grad():
        q = [t.numpy() for t in ex.forward(inputs=x, output_names=names)]
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.dequantize()
        f = [t.numpy() for t in ex.forward(inputs=x, output_names=names)]
    print(f"{'tensor':45s} {'op':12s} {'SNR dB':>8s}  fp32 range")
    for name, a, b in zip(names, f, q):
        op = graph.variables[name].source_op
        snr = 10 * np.log10((a**2).sum() / (((a - b) ** 2).sum() + 1e-20) + 1e-20)
        print(f"{name[:45]:45s} {op.type:12s} {snr:8.1f}  [{a.min():.3g}, {a.max():.3g}]")
    for p in (SPIKE / "out").glob("_diag.*"):
        p.unlink()


if __name__ == "__main__":
    main()
