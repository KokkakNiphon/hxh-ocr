"""Evaluate models on the real-image test lines (data/real/lines/labels.tsv).

fp32 checkpoints run through the training-time predictor (natural width). int8 models run
through the Rust engine itself (rust/target/release/hxh-cli), on the exact u8 strip the
calibration pipeline and the browser demo build (resize to H keeping aspect with INTER_AREA,
squeeze to W if longer, else pad right with the replicated edge), so the numbers are what the
demo and the ESP32-S3 compute.

    python scripts/eval_real.py            # prints a table, writes data/real/lines/eval_real.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr.data import LineDataset, find_vocab, make_loader  # noqa: E402
from hxh_ocr.engine import load_checkpoint, predict  # noqa: E402
from hxh_ocr.metrics import score  # noqa: E402

LINES = ROOT / "data" / "real" / "lines"
CLI = ROOT / "rust" / "target" / "release" / "hxh-cli.exe"

FP32 = ["mnv4s_h32", "v3_mnv4s_h32", "v3_mnv4s_h48", "v3_mnv4s3_h32"]
INT8 = [  # (name, .hxhm, H, W)
    ("demo: v3_mnv4s_h32 PTQ+TQT", "experiments/rust_engine/v3_mnv4s_h32_w512_tqt/model.hxhm", 32, 512),
    ("QAT v3_mnv4s3_h32", "experiments/rust_engine/qat_v3_mnv4s3_h32_ptq/model.hxhm", 32, 512),
    ("QAT v3_mnv4s_h32", "experiments/rust_engine/qat_v3_mnv4s_h32_ptq/model.hxhm", 32, 512),
    ("QAT v3_mnv4s_h48", "experiments/rust_engine/qat_v3_mnv4s_h48_ptq/model.hxhm", 48, 768),
]


def resize_pad_u8(img, H, W):
    h, w = img.shape
    new_w = max(1, round(w * H / h))
    img = cv2.resize(img, (min(new_w, W), H), interpolation=cv2.INTER_AREA)
    if img.shape[1] < W:
        img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_REPLICATE)
    return img


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [l.split("\t") for l in (LINES / "labels.tsv").read_text(encoding="utf-8").splitlines() if l]
    refs = [r[1] for r in rows]
    vocab = find_vocab([ROOT / "data" / "synth_v3"]).tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, per_line = {}, {r[0]: {"label": r[1]} for r in rows}

    for run in FP32:
        ck = ROOT / "runs" / run / "best.pt"
        if not ck.exists():
            continue
        model, cfg = load_checkpoint(ck, device)
        v = find_vocab(cfg["data"])
        ds = LineDataset([LINES], v, cfg["height"], cfg.get("max_width"), train=False)
        dl, _ = make_loader(ds, 32, False, 0)
        r, hyps, idx = predict(model, dl, device, v.tokens, device.type == "cuda")
        cer, acc = score(r, hyps)
        results[f"fp32 {run}"] = (cer, acc)
        for i, h in zip(idx, hyps):
            per_line[rows[i][0]][f"fp32 {run}"] = h

    for name, path, H, W in INT8:
        p = ROOT / path
        if not p.exists():
            continue
        hyps = []
        with tempfile.TemporaryDirectory() as td:
            for rel, *_ in rows:
                img = cv2.imread(str(LINES / rel), cv2.IMREAD_GRAYSCALE)
                strip = resize_pad_u8(img, H, W)
                pgm = Path(td) / "x.pgm"
                pgm.write_bytes(f"P5 {W} {H} 255\n".encode() + strip.tobytes())
                out = subprocess.run([str(CLI), str(p), str(pgm), "--iters", "1"], capture_output=True, text=True).stdout
                m = re.search(r"ids \(\d+\): \[([^\]]*)\]", out)
                ids = [int(t) for t in m.group(1).split(",") if t.strip()] if m else []
                text = "".join(vocab[i - 1] if 1 <= i <= len(vocab) else "?" for i in ids)
                hyps.append(text)
                per_line[rel][f"int8 {name}"] = text
        results[f"int8 {name}"] = score(refs, hyps)

    print(f"{len(rows)} real lines (2011 anime 'Hunter Language Course' title cards)\n")
    print(f"{'model':<34} {'CER':>7} {'exact lines':>12}")
    for k, (cer, acc) in results.items():
        print(f"{k:<34} {cer * 100:6.2f}% {acc * 100:10.1f}%")
    print("\nper line (label | predictions):")
    keys = list(results)
    for rel, d in per_line.items():
        preds = " | ".join(f"{d.get(k, '-')}" for k in keys)
        print(f"{rel:<18} {d['label']:<8} | {preds}")
    (LINES / "eval_real.json").write_text(json.dumps({"results": {k: list(v) for k, v in results.items()},
                                                      "lines": per_line}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
