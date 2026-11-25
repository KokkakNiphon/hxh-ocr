"""Check hxh_ocr.qat against ESP-PPQ: exponents and simulated int8 logits.

Builds the QAT fake-quant view of a checkpoint, calibrates it on the same 64 lines that
experiments/esp_ppq_spike/quantize.py calibrates on, then compares
  - per conv: weight / input / output exponents vs ESP-PPQ's model.json (same calibration),
  - the fake-quant logits on quantize.py's 512 held-out lines vs ESP-PPQ's TorchExecutor
    output (holdout_ppq_int8.npy, written by quantize.py --save-outputs), and the CERs.

  .venv\\Scripts\\python.exe experiments\\qat\\compare_ppq.py --ckpt runs\\mnv4s_h32\\best.pt `
      --ppq-out experiments\\esp_ppq_spike\\out\\mnv4s_h32_ptq

CPU only (keeps the GPU free and matches ESP-PPQ's float path). Uses quantize.py's data and
static size, including its HXH_DATA / HXH_H / HXH_W overrides (e.g. 48x384 models).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "esp_ppq_spike"))
from hxh_ocr.engine import load_checkpoint  # noqa: E402
from hxh_ocr.model import fuse_bn  # noqa: E402
from hxh_ocr.qat import QATModel  # noqa: E402
from spike_common import H, W, cer, compare, load_lines  # noqa: E402  (HXH_DATA / HXH_H / HXH_W)


def ppq_conv_exponents(model_json: Path) -> dict:
    """{conv module path: {w, in, out}} from ESP-PPQ's model.json (effective = dominator scale)."""
    d = json.loads(model_json.read_text())
    vals = d["values"]
    exp = lambda cfg: int(round(math.log2(vals[str(cfg["dominator"])]["scale"])))  # noqa: E731
    out = {}
    for op, cfgs in d["configs"].items():
        keys = list(cfgs)
        wk = [k for k in keys if k.endswith(".weight")]
        if not wk or not op.endswith("Conv"):
            continue
        out[wk[0][: -len(".weight")]] = {"w": exp(cfgs[wk[0]]), "in": exp(cfgs[keys[0]]), "out": exp(cfgs[keys[-1]])}
    return out


def compare_exponents(mine: dict, ppq: dict) -> dict:
    rows, diff = [], {"w": 0, "in": 0, "out": 0}
    for name, e in mine.items():
        p = ppq.get(name)
        if p is None:
            rows.append(f"  {name}: not in ESP-PPQ graph")
            continue
        bad = [k for k in ("w", "in", "out") if e[k] != p[k]]
        for k in bad:
            diff[k] += 1
        if bad:
            rows.append(f"  {name:32s} " + "  ".join(f"{k}: qat {e[k]} ppq {p[k]}" for k in bad))
    return {"n_convs": len(mine), "mismatch": diff, "rows": rows}


@torch.no_grad()
def run(model, xs, bs=8):
    return np.concatenate([model(torch.from_numpy(xs[i:i + bs])).numpy() for i in range(0, len(xs), bs)])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ppq-out", default=None, help="quantize.py output dir (model.json, holdout_ppq_int8.npy)")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--json", default=None, help="write the comparison here")
    ap.add_argument("--jitter", type=int, default=0,
                    help="N: also recalibrate N copies with 1e-5 relative weight noise (a stand-in for "
                         "the tiny differences ESP-PPQ's KL pick is sensitive to), apply each exponent set "
                         "to the real weights and report the holdout int8 CER spread")
    ap.add_argument("--calib-algo", default="kl", choices=["kl", "minmax"], help="as quantize.py --calib-algo")
    args = ap.parse_args()
    torch.set_num_threads(8)

    model, cfg = load_checkpoint(args.ckpt, "cpu")
    fused = fuse_bn(model, (H, W))
    qat = QATModel(fused, input_exp=-7 if args.calib_algo == "kl" else None, calib_algo=args.calib_algo)
    print(qat.summary())
    calib = load_lines("val", args.calib_n, offset=0)
    qat.calibrate([torch.from_numpy(calib[i:i + 8]) for i in range(0, len(calib), 8)])
    ex = qat.exponents()
    res = {"ckpt": args.ckpt, "summary": qat.summary(), "exponents": ex}

    hold, labels = load_lines("val", args.eval_n, offset=2000, with_labels=True)
    ref = run(fused.eval(), hold)
    fq = run(qat.eval(), hold)
    res["holdout_cer"] = {"fp32": cer(ref, labels), "qat_fakequant": cer(fq, labels)}
    res["fakequant_vs_fp32"] = compare(ref, fq)

    if args.ppq_out:
        out = Path(args.ppq_out)
        ppq = ppq_conv_exponents(out / "model.json")
        c = compare_exponents(ex["convs"], ppq)
        res["exponents_vs_ppq"] = {k: v for k, v in c.items() if k != "rows"}
        print(f"exponents vs ESP-PPQ ({c['n_convs']} convs), mismatches: {c['mismatch']}")
        print("\n".join(c["rows"]) or "  all equal")
        npy = out / "holdout_ppq_int8.npy"
        if npy.exists():
            q = np.load(npy)[: len(hold)]
            res["holdout_cer"]["ppq_int8"] = cer(q, labels)
            res["fakequant_vs_ppq"] = compare(q, fq)
            res["fakequant_vs_ppq"]["values_identical"] = float((q == fq).mean())
            res["fakequant_vs_ppq"]["lines_identical"] = float(np.mean([(q[i] == fq[i]).all() for i in range(len(q))]))
    if args.jitter:
        cb = [torch.from_numpy(calib[i:i + 8]) for i in range(0, len(calib), 8)]
        base_exp = {n: float(qat.gm.get_submodule(n).exp) for n in qat.aq_names}
        cers, flips = [], []
        for seed in range(args.jitter):
            g = torch.Generator().manual_seed(seed)
            q2 = QATModel(fused, input_exp=-7 if args.calib_algo == "kl" else None, calib_algo=args.calib_algo)
            with torch.no_grad():
                for p in q2.parameters():
                    p.mul_(1 + 1e-5 * torch.randn(p.shape, generator=g))
            q2.calibrate(cb)
            e2 = {n: float(q2.gm.get_submodule(n).exp) for n in q2.aq_names}
            flips.append(sum(e2[n] != base_exp[n] for n in base_exp))
            for n in qat.aq_names:
                qat.gm.get_submodule(n).exp.fill_(e2[n])
            cers.append(cer(run(qat.eval(), hold), labels))
        for n in qat.aq_names:
            qat.gm.get_submodule(n).exp.fill_(base_exp[n])
        res["jitter"] = {"exponent_flips": flips, "holdout_int8_cer": cers, "cer_mean": float(np.mean(cers)),
                         "cer_min": float(np.min(cers)), "cer_max": float(np.max(cers))}
    print(json.dumps({k: v for k, v in res.items() if k not in ("exponents",)}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
