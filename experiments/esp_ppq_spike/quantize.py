"""Quantize a toy ONNX model to .espdl (ESP32-S3, int8) with ESP-PPQ, optionally
with TQT, and measure the error of ESP-PPQ's simulated int8 output against the
fp32 ONNX output (onnxruntime).

Follows the official flow (esp-dl docs, how_to_quantize_model / TQT tutorial):
    # espdl_quantize_onnx only picks the 16-bit defaults (minmax calibration) when setting=None;
    # a hand-made 8-bit setting with quant_type="w16a16" keeps KL calibration and clips badly.
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=16 if args.quant_type == "w16a16" else 8)
    setting.tqt_optimization = True; setting.tqt_optimization_setting.* = ...
    graph = espdl_quantize_onnx(onnx_import_file=..., espdl_export_file=...,
                                calib_dataloader=..., calib_steps=..., input_shape=[1,1,32,256],
                                target="esp32s3", quant_type="w8a8", setting=setting, ...)
    TorchExecutor(graph)(x)  -> simulated int8 output (the docs say it matches the chip)

Also tries exporting the same quantized graph as a QDQ ONNX for onnxruntime.

Usage (from this folder, with .venv-ppq):
    python quantize.py --onnx models/toy_random_avg_op18.onnx --tag random_avg_op18
    python quantize.py --onnx models/toy_trained_avg_op18.onnx --tag trained_avg_op18_tqt --tqt
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from spike_common import SPIKE, H, W, cer, compare, load_lines


def ort_run(path: Path, xs: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: xs[i : i + 1]})[0] for i in range(len(xs))])


def ppq_run(graph, xs: np.ndarray) -> np.ndarray:
    from esp_ppq.executor import TorchExecutor

    ex = TorchExecutor(graph=graph, device="cpu")
    outs = []
    with torch.no_grad():
        for i in range(len(xs)):
            outs.append(ex.forward(inputs=torch.from_numpy(xs[i : i + 1]))[0].cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--quant-type", default="w8a8")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--calib-algo", default=None, help="activation calib: kl (espdl default), percentile, minmax, mse")
    ap.add_argument("--eval-n", type=int, default=256, help="held-out val lines (not used for calibration)")
    ap.add_argument("--tqt", action="store_true")
    ap.add_argument("--tqt-steps", type=int, default=500)
    ap.add_argument("--tqt-lr", type=float, default=1e-5)
    ap.add_argument("--tqt-block", type=int, default=4)
    ap.add_argument("--tqt-int-lambda", type=float, default=0.25)
    ap.add_argument("--test-values", action="store_true", help="embed test input/output in the .espdl")
    ap.add_argument("--no-qdq", action="store_true")
    ap.add_argument("--save-outputs", action="store_true", help="save simulated int8 logits as .npy")
    ap.add_argument("--debug", action="store_true", help="w16a16 only: ESP-PPQ high-precision int16 simulation")
    ap.add_argument(
        "--mixed",
        action="append",
        default=[],
        metavar="OP=QTYPE",
        help="per-op quant type, e.g. /head/Conv=w8a16 (repeatable). A Conv fused with a ReLU "
        "needs the ReLU dispatched too, or ESP-PPQ warns 'should be send to a same platform'",
    )
    args = ap.parse_args()

    from esp_ppq import QuantizationSettingFactory, TargetPlatform
    from esp_ppq.api import espdl_quantize_onnx, export_ppq_graph, get_target_platform

    torch.manual_seed(0)
    out = SPIKE / "out" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.onnx).resolve()
    # espdl_quantize_onnx overwrites its input file with the onnxsim-simplified model,
    # so hand it a copy and keep the original as the fp32 reference.
    work_onnx = out / "model_fp32_simplified.onnx"
    shutil.copy(src, work_onnx)

    calib = load_lines("val", args.calib_n, offset=0)
    hold, hold_labels = load_lines("val", args.eval_n, offset=2000, with_labels=True)
    loader = DataLoader(list(torch.from_numpy(calib)), batch_size=args.calib_bs, shuffle=False)

    # espdl_quantize_onnx only picks the 16-bit defaults (minmax calibration) when setting=None;
    # a hand-made 8-bit setting with quant_type="w16a16" keeps KL calibration and clips badly.
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=16 if args.quant_type == "w16a16" else 8)
    if args.calib_algo:
        setting.quantize_activation_setting.calib_algorithm = args.calib_algo
    for spec in args.mixed:
        op, qtype = spec.split("=")
        setting.dispatching_table.append(op, get_target_platform(args.target, quant_type=qtype))
        if "16" in qtype:  # KL calibration is poor at 16 bit; ESP-PPQ's own 16-bit default is minmax
            setting.quantize_activation_setting.calib_algorithm_per_layer[op] = "minmax"
    if args.tqt:
        setting.tqt_optimization = True
        t = setting.tqt_optimization_setting
        t.steps, t.lr, t.block_size = args.tqt_steps, args.tqt_lr, args.tqt_block
        t.int_lambda, t.collecting_device = args.tqt_int_lambda, "cpu"

    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(work_onnx),
        espdl_export_file=str(out / "model.espdl"),
        calib_dataloader=loader,
        calib_steps=len(loader),
        input_shape=[1, 1, H, W],
        inputs=[torch.from_numpy(calib[:1])],
        target=args.target,
        quant_type=args.quant_type,
        debug=args.debug,
        setting=setting,
        collate_fn=lambda b: b.float(),
        device="cpu",
        error_report=True,
        export_test_values=args.test_values,
        verbose=1,
    )
    t_quant = time.time() - t0

    # ---------------- error: fp32 onnxruntime vs ESP-PPQ simulated int8
    res = {"args": vars(args), "quantize_seconds": t_quant}
    sim = {}
    for name, xs in (("calib", calib), ("holdout", hold)):
        ref = ort_run(src, xs)
        q = ppq_run(graph, xs)
        res[name] = compare(ref, q)
        sim[name] = q
        if args.save_outputs:
            np.save(out / f"{name}_ppq_int8.npy", q)
        if name == "holdout":
            res["holdout_cer"] = {"fp32": cer(ref, hold_labels), "ppq_int8": cer(q, hold_labels)}
        if name == "calib":
            ref_simpl = ort_run(work_onnx, xs)
            res["simplified_vs_original_maxabs"] = float(np.abs(ref - ref_simpl).max())

    # ---------------- QDQ ONNX export of the same quantized graph
    if not args.no_qdq:
        try:
            qdq = out / "model_qdq.onnx"
            export_ppq_graph(
                graph=graph,
                platform=TargetPlatform.ONNXRUNTIME,
                graph_save_to=str(qdq),
                config_save_to=str(out / "model_qdq.json"),
            )
            for name, xs in (("calib", calib), ("holdout", hold)):
                q_ppq = sim[name]
                q_ort = ort_run(qdq, xs)
                res[f"qdq_ort_vs_ppq_{name}"] = compare(q_ppq, q_ort)
                res[f"qdq_ort_vs_fp32_{name}"] = compare(ort_run(src, xs), q_ort)
                if name == "holdout":
                    res["holdout_cer"]["qdq_ort"] = cer(q_ort, hold_labels)
            res["qdq_bytes"] = qdq.stat().st_size
            # The S3 rounds half up; ONNX QuantizeLinear rounds half to even. Switch the
            # simulation to half-even (after the .espdl was written) to show that this
            # explains the QDQ-vs-simulation mismatch.
            from esp_ppq.core import RoundingPolicy
            from esp_ppq.IR.quantize import QuantableOperation

            for op in graph.operations.values():
                if isinstance(op, QuantableOperation):
                    for cfg, _ in op.config_with_variable:
                        cfg.rounding = RoundingPolicy.ROUND_HALF_EVEN
            q_even = ppq_run(graph, hold)
            res["qdq_ort_vs_ppq_half_even_holdout"] = compare(q_even, ort_run(qdq, hold))
        except Exception as e:  # noqa: BLE001
            res["qdq_error"] = f"{type(e).__name__}: {e}"

    res["sizes"] = {
        "fp32_onnx": src.stat().st_size,
        "espdl": (out / "model.espdl").stat().st_size,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n=== RESULT", args.tag)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
