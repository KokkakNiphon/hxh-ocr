"""Quantise the line detector with ESP-PPQ (esp32s3, int8, per-tensor power-of-two) and compare the
simulated int8 output with fp32 on held-out pages.

Run with the ESP-PPQ venv (no timm there; scoring calls scripts/eval_det.py in the main venv):
  .venv-ppq\\Scripts\\python.exe scripts/quantize_det.py --onnx exports/det_v1.onnx --tag det_v1_ptq
  .venv-ppq\\Scripts\\python.exe scripts/quantize_det.py --onnx exports/det_v1.onnx --tag det_v1_tqt --tqt

Pages (data/pages_v2/val, 640 letterbox, see hxh_ocr/det_data.py): calibration = val[0:64],
held-out = val[900:1000]. Writes experiments/det/<tag>/:
  model.espdl, model_fp32_simplified.onnx, metrics.json
  holdout_fp32.npy / holdout_int8.npy   [100, 2, 320, 320] float32 outputs (onnxruntime fp32 /
                                        ESP-PPQ TorchExecutor int8 simulation, values = q * 2^exp)
  golden/inputs_u8.npy                  [64, 640, 640] uint8 letterboxed pages (val[900:964])
  golden/outputs_int8.npy               [64, 2, 320, 320] int8 (simulated output / 2^exp)
  golden/README.md                      layout, exponents, the input LUT
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hxh_ocr.det_data import LETTERBOX, PageIndex, letterbox, normalize_u8  # noqa: E402

MAIN_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def load_pages(data, offset, n):
    idx = PageIndex([Path(data) / "val"], limit=n, offset=offset)
    return np.stack([letterbox(idx.read(i), LETTERBOX)[0] for i in range(len(idx))])


def ort_run(path, xs_u8):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: normalize_u8(xs_u8[i:i + 1])[:, None]})[0]
                           for i in range(len(xs_u8))])


def ppq_run(graph, xs_u8):
    from esp_ppq.executor import TorchExecutor
    ex = TorchExecutor(graph=graph, device="cpu")
    outs = []
    with torch.no_grad():
        for i in range(len(xs_u8)):
            x = torch.from_numpy(normalize_u8(xs_u8[i:i + 1])[:, None])
            outs.append(ex.forward(inputs=x)[0].cpu().numpy())
    return np.concatenate(outs)


def compare(ref, q, logit_thresh=-0.8473):
    ref = ref.astype(np.float64); q = q.astype(np.float64)
    out = {}
    for c, name in ((0, "prob"), (1, "vert")):
        a, b = ref[:, c].ravel(), q[:, c].ravel()
        out[f"{name}_snr_db"] = float(10 * np.log10((a @ a) / (((a - b) ** 2).sum() + 1e-12)))
        out[f"{name}_max_abs_err"] = float(np.abs(a - b).max())
    pa, pb = ref[:, 0] > logit_thresh, q[:, 0] > logit_thresh
    out["prob_binary_agree"] = float((pa == pb).mean())
    out["prob_binary_iou"] = float((pa & pb).sum() / max((pa | pb).sum(), 1))
    k = pa | pb
    out["vert_sign_agree_in_text"] = float(((ref[:, 1] > 0) == (q[:, 1] > 0))[k].mean()) if k.any() else 1.0
    return out


def score(npy, data, offset, n, out_json):
    cmd = [str(MAIN_PY), str(ROOT / "scripts" / "eval_det.py"), "--outputs", str(npy), "--data", str(data),
           "--offset", str(offset), "--n", str(n), "--json", str(out_json), "--no-oracle"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("detection", "e2e"))))
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return None
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


def output_exponent(espdl_path):
    import espdl_to_hxhm as cv
    m = cv.read_espdl(espdl_path)
    return int(m.graph.input[0].exponents[0]), int(m.graph.output[0].exponents[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "pages_v2"))
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--calib-algo", default=None, help="kl (ESP-PPQ default) | minmax | percentile | mse")
    ap.add_argument("--hold-offset", type=int, default=900)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--save-n", type=int, default=64)
    ap.add_argument("--tqt", action="store_true")
    ap.add_argument("--tqt-steps", type=int, default=500)
    ap.add_argument("--tqt-lr", type=float, default=1e-5)
    ap.add_argument("--tqt-block", type=int, default=4)
    ap.add_argument("--tqt-int-lambda", type=float, default=0.25)
    ap.add_argument("--fp32-from", default=None, help="reuse holdout_fp32.npy from another tag dir")
    ap.add_argument("--equalize", action="store_true", help="ESP-PPQ layerwise (cross-layer) equalization before calibration")
    ap.add_argument("--eq-iterations", type=int, default=10)
    ap.add_argument("--eq-level", type=int, default=1)
    ap.add_argument("--error-report", action="store_true", help="ESP-PPQ's per-layer error analysis (slow: ~20 s a layer at 960)")
    args = ap.parse_args()

    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx

    torch.manual_seed(0)
    out = ROOT / "experiments" / "det" / args.tag
    (out / "golden").mkdir(parents=True, exist_ok=True)
    src = Path(args.onnx).resolve()
    work = out / "model_fp32_simplified.onnx"
    shutil.copy(src, work)  # espdl_quantize_onnx overwrites its input with a simplified copy

    t0 = time.time()
    calib = load_pages(args.data, 0, args.calib_n)
    hold = load_pages(args.data, args.hold_offset, args.eval_n)
    print(f"pages loaded: calib {len(calib)}, holdout {len(hold)} ({time.time()-t0:.0f}s)", flush=True)
    calib_x = torch.from_numpy(normalize_u8(calib)[:, None])
    loader = DataLoader(list(calib_x), batch_size=args.calib_bs, shuffle=False)

    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
    if args.calib_algo:
        setting.quantize_activation_setting.calib_algorithm = args.calib_algo
    if args.equalize:
        # cross-layer equalization (conv -> relu -> conv pairs, exact in fp32): balances the
        # per-channel weight ranges of the depthwise convs, which per-tensor power-of-two int8
        # otherwise flattens (the width-1.0 MobileNetV4 has dw layers with 190 of 192 channels
        # under 4 quantisation levels after BN folding)
        setting.equalization = True
        setting.equalization_setting.iterations = args.eq_iterations
        setting.equalization_setting.opt_level = args.eq_level
    if args.tqt:
        setting.tqt_optimization = True
        t = setting.tqt_optimization_setting
        t.steps, t.lr, t.block_size = args.tqt_steps, args.tqt_lr, args.tqt_block
        t.int_lambda, t.collecting_device = args.tqt_int_lambda, "cpu"

    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(work), espdl_export_file=str(out / "model.espdl"),
        calib_dataloader=loader, calib_steps=len(loader), input_shape=[1, 1, LETTERBOX, LETTERBOX],
        inputs=[calib_x[:1]], target=args.target, quant_type="w8a8", setting=setting,
        collate_fn=lambda b: b.float(), device="cpu", error_report=args.error_report, export_test_values=False, verbose=1,
    )
    t_q = time.time() - t0
    in_exp, out_exp = output_exponent(out / "model.espdl")
    print(f"quantised in {t_q:.0f}s; input exponent {in_exp}, output exponent {out_exp}", flush=True)

    t0 = time.time()
    if args.fp32_from:
        ref = np.load(Path(args.fp32_from) / "holdout_fp32.npy")
    else:
        ref = ort_run(src, hold).astype(np.float32)
    q = ppq_run(graph, hold).astype(np.float32)
    print(f"holdout inference {time.time()-t0:.0f}s", flush=True)
    np.save(out / "holdout_fp32.npy", ref)
    np.save(out / "holdout_int8.npy", q)
    qi = q / 2.0 ** out_exp
    assert np.array_equal(qi, np.round(qi)), "simulated outputs are not on the int8 grid"
    qi = np.clip(np.round(qi), -128, 127).astype(np.int8)

    # golden vectors for the Rust engine
    ns = min(args.save_n, len(hold))
    np.save(out / "golden" / "inputs_u8.npy", hold[:ns])
    np.save(out / "golden" / "outputs_int8.npy", qi[:ns])
    import espdl_to_hxhm as cv
    lut = cv.input_lut_from_formula(in_exp)
    np.save(out / "golden" / "input_lut_int8.npy", lut)
    (out / "golden" / "README.md").write_text(f"""# Detector golden vectors ({args.tag})

Model: `../model.espdl` (ESP-PPQ, target {args.target}, int8 per-tensor power-of-two,
{'PTQ + TQT' if args.tqt else 'PTQ'}, calibration {args.calib_algo or 'kl'} on val[0:{args.calib_n}]).
Source ONNX: `{src.name}`.

- `inputs_u8.npy`: uint8 [{ns}, 640, 640], pages val[{args.hold_offset}:{args.hold_offset + ns}] of
  `data/pages_v2`, letterboxed (long side scaled to 640 with cv2 INTER_AREA when shrinking /
  INTER_LINEAR when enlarging, placed top-left, right/bottom padded with 255).
- `input_lut_int8.npy`: int8 [256], u8 pixel -> int8 model input. The float input is
  (x / 255 - 0.5) / 0.5, quantised with exponent {in_exp} (round half up, clip to [-128, 127]).
- `outputs_int8.npy`: int8 [{ns}, 2, 320, 320] (NCHW), ESP-PPQ TorchExecutor simulated output
  (the .espdl semantics), stored as integers; float value = int * 2^{out_exp}.
  Channel 0 = text-kernel logit, channel 1 = vertical-orientation logit.
- Output exponent {out_exp}: the prob threshold logit > -0.8473 (prob 0.3) becomes
  int > {-0.8473 / 2 ** out_exp:.3f}, i.e. int >= {int(np.floor(-0.8473 / 2 ** out_exp)) + 1}.
""", encoding="utf-8")

    res = {"args": vars(args), "quantize_seconds": t_q, "input_exponent": in_exp, "output_exponent": out_exp,
           "compare_holdout": compare(ref, q),
           "sizes": {"fp32_onnx": src.stat().st_size, "espdl": (out / "model.espdl").stat().st_size}}
    print(json.dumps(res["compare_holdout"], indent=1), flush=True)
    for name in ("fp32", "int8"):
        print(f"--- {name}")
        res[f"eval_{name}"] = score(out / f"holdout_{name}.npy", args.data, args.hold_offset, len(hold),
                                    out / f"eval_{name}.json")
    (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"=== {args.tag}: espdl {res['sizes']['espdl']} B")


if __name__ == "__main__":
    main()
