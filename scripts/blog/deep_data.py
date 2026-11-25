"""Data for the research deep dive's own figures (web/blog/deep.js).

Reads training logs and quantisation records that the page cannot fetch (runs/ is not served)
and writes web/blog/img/deep/data.json. Nothing is computed beyond unit changes (fractions
to percent) and a label for each conv's kind (from its name in the timm / hxh_ocr model).

    .venv/Scripts/python.exe scripts/blog/deep_data.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "blog" / "img" / "deep" / "data.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(v: str) -> float:
    return round(float(v) * 100, 3)


def qat_curve(run: str, label: str) -> dict:
    rows = read_csv(ROOT / "runs" / run / "log.csv")
    return {
        "id": run, "label": label, "file": f"runs/{run}/log.csv",
        "int8": [[int(r["step"]), pct(r["val_cer"])] for r in rows],
        "float": [[int(r["step"]), pct(r["val_cer_float"])] for r in rows],
        "exp_changes": [[int(r["step"]), int(r["exp_changes"])] for r in rows],
    }


def conv_kind(name: str) -> str:
    # MobileNetV4-conv-small: blocks.0.0 / blocks.1.0 are full 3x3 convs, blocks.0.1 / blocks.1.1
    # are 1x1; UIB blocks have dw_start / dw_mid (depthwise) and pw_exp / pw_proj (1x1);
    # the sequence head has dw (1x5 depthwise), pw and cls (1x1). See hxh_ocr/model.py.
    if name == "conv_stem" or name in ("blocks.0.0.conv", "blocks.1.0.conv"):
        return "full"
    if ".dw" in name:
        return "depthwise"
    return "pointwise"


def main() -> None:
    det = read_csv(ROOT / "runs" / "det_v1" / "log.csv")
    ex = json.loads((ROOT / "runs" / "qat_v4_mnv4s_h48" / "exponents.json").read_text(encoding="utf-8"))
    data = {
        "about": "Written by scripts/blog/deep_data.py from runs/*/log.csv and "
                 "runs/qat_v4_mnv4s_h48/exponents.json; percentages are the logged fractions x 100.",
        "qat_curves": {
            "note": "QAT fine-tunes (synth_v4, 4k steps + exponent settling): val CER on the run's "
                    "2,000-line val subset every 500 steps, with ESP-DL-exact fake quantisation "
                    "(int8) and without it (float); step 0 = the starting fp32 checkpoint.",
            "series": [
                qat_curve("qat_v4_mnv4s_h48", "48 px horizontal"),
                qat_curve("qat_v4_mnv4s_h48_vert", "48 px vertical"),
                qat_curve("qat_v4_mnv4s3_h32", "32 px three-stage"),
            ],
        },
        "det_loss": {
            "file": "runs/det_v1/log.csv",
            "note": "det_v1 training-loss terms (unweighted), mean over the last 2,000 steps at each eval",
            "steps": [int(r["step"]) for r in det],
            "terms": {k: [round(float(r[k]), 5) for r in det] for k in ("l_prob", "l_thr", "l_dice", "l_vert")},
            "total": [round(float(r["train_loss"]), 5) for r in det],
        },
        "exponents": {
            "file": "runs/qat_v4_mnv4s_h48/exponents.json",
            "note": "power-of-two exponents of the deployed horizontal reader after QAT: real = q * 2^exp",
            "convs": [{"name": k, "kind": conv_kind(k), "w": v["w"], "in": v["in"], "out": v["out"]}
                      for k, v in ex["convs"].items()],
            "sites": [{"node": s["node"], "kind": s["kind"], "exp": s["exp"], "clip": round(s["clip"], 4)}
                      for s in ex["sites"]],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
