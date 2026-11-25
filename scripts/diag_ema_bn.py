"""Diagnose an eval/train gap: score a checkpoint's EMA weights vs raw weights, and EMA
weights after re-estimating BatchNorm statistics on training batches.

    python scripts/diag_ema_bn.py --run runs/mnv4s_h32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.data import LineDataset, find_vocab, make_loader, split_dirs  # noqa: E402
from hxh_ocr.engine import load_checkpoint, predict  # noqa: E402
from hxh_ocr.metrics import score  # noqa: E402


@torch.no_grad()
def recalibrate_bn(model, loader, device, batches=100):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative average
    model.train()
    for i, b in enumerate(loader):
        if i >= batches:
            break
        model(b["images"].to(device))
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--val-n", type=int, default=1000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(args.run) / args.ckpt
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
    vocab = find_vocab(cfg["data"])
    val = LineDataset(split_dirs(cfg["data"], "val"), vocab, cfg["height"], cfg.get("max_width"), train=False,
                      limit=args.val_n)
    vl, _ = make_loader(val, 128, False, 0)
    tr = LineDataset(split_dirs(cfg["data"], "train"), vocab, cfg["height"], cfg.get("max_width"), train=True)
    tl, _ = make_loader(tr, 128, True, 0, seed=123)
    for weights in ("ema", "model"):
        m, _ = load_checkpoint(ck, device, weights)
        cer, acc = score(*predict(m, vl, device, vocab.tokens, True)[:2])
        print(f"{weights:5s} weights, stored BN stats : CER {cer:.4f} acc {acc:.4f}", flush=True)
        recalibrate_bn(m, tl, device)
        cer, acc = score(*predict(m, vl, device, vocab.tokens, True)[:2])
        print(f"{weights:5s} weights, recalibrated BN : CER {cer:.4f} acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
