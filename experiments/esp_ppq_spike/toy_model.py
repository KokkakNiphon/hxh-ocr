"""Toy CRNN-style CTC line recogniser that exercises every op planned for the
real model, plus BN folding and ONNX export variants.

Ops covered (NCHW, input 1x1x32x256):
  stem   Conv 3x3 stride 2, Cin=1 -> 16              16x16x128
  blk1   depthwise 3x3 + 1x1, residual Add, ReLU      16x16x128
  down1  regular Conv 3x3 stride 2, 16 -> 32          32x8x64
  blk2   depthwise 5x5 + 1x1, residual Add, ReLU      32x8x64
  down2  depthwise 3x3 stride (2,1) + 1x1 32 -> 64    64x4x64
  blk3   depthwise (1,5) + 1x1, residual Add, ReLU    64x4x64
  pool   AveragePool kernel (4,1)  or  ReduceMean over H (keepdim)  -> 64x1x64
  head   1x1 Conv 64 -> 75                            [1,75,1,64]

BatchNorm layers exist in the trainable model and are folded into the convs
(torch.nn.utils.fusion.fuse_conv_bn_eval) before export, so the ONNX graph has
no BatchNormalization nodes.

Usage (from the repo root, with .venv-ppq):
  python experiments/esp_ppq_spike/toy_model.py                    # random weights
  python experiments/esp_ppq_spike/toy_model.py --train-seconds 300  # + short CPU CTC training
"""
from __future__ import annotations

import argparse
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from spike_common import NUM_CLASSES, SPIKE, cer, load_lines, load_vocab

OUT = SPIKE / "models"


def cbr(cin, cout, k, s=1, p=0, groups=1, relu=True):
    layers = [nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DWRes(nn.Module):
    """depthwise kxk (+BN+ReLU) -> pointwise 1x1 (+BN) -> Add(skip) -> ReLU"""

    def __init__(self, c, k):
        super().__init__()
        pad = (k[0] // 2, k[1] // 2)
        self.dw = cbr(c, c, k, 1, pad, groups=c)
        self.pw = cbr(c, c, 1, relu=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.pw(self.dw(x)))


class ToyCRNN(nn.Module):
    def __init__(self, pool: str = "avg"):
        super().__init__()
        self.pool_kind = pool
        self.stem = cbr(1, 16, 3, 2, 1)  # Cin=1, stride 2
        self.blk1 = DWRes(16, (3, 3))
        self.down1 = cbr(16, 32, 3, 2, 1)  # regular conv
        self.blk2 = DWRes(32, (5, 5))
        self.down2 = nn.Sequential(
            cbr(32, 32, 3, (2, 1), 1, groups=32),  # depthwise, asymmetric stride (2,1)
            cbr(32, 64, 1),
        )
        self.blk3 = DWRes(64, (1, 5))  # depthwise kernel (1,5)
        self.pool = nn.AvgPool2d((4, 1))  # height collapse, kernel (h,1)
        self.head = nn.Conv2d(64, NUM_CLASSES, 1)  # classifier -> [N,75,1,64]

    def forward(self, x):
        x = self.stem(x)
        x = self.blk1(x)
        x = self.down1(x)
        x = self.blk2(x)
        x = self.down2(x)
        x = self.blk3(x)
        if self.pool_kind == "avg":
            x = self.pool(x)
        else:
            x = x.mean(dim=2, keepdim=True)  # ReduceMean over H
        return self.head(x)


def randomise_bn(model: nn.Module, seed: int = 0) -> None:
    """Give BN layers non-trivial stats so folding actually changes weights."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) * 0.5 + 0.75)
            m.weight.data.copy_(torch.rand(n, generator=g) * 0.5 + 0.75)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


def fold_bn(model: nn.Module) -> nn.Module:
    """Return an eval-mode copy with every Conv+BN pair fused (BN -> Identity)."""
    model = copy.deepcopy(model).eval()
    for mod in list(model.modules()):
        if isinstance(mod, nn.Sequential):
            for i in range(len(mod) - 1):
                if isinstance(mod[i], nn.Conv2d) and isinstance(mod[i + 1], nn.BatchNorm2d):
                    mod[i] = fuse_conv_bn_eval(mod[i], mod[i + 1])
                    mod[i + 1] = nn.Identity()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return model


def export_onnx(model: nn.Module, path: Path, opset: int, dynamo: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 1, 32, 256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (x,),
            str(path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=dynamo,
            **({"external_data": False} if dynamo else {}),
        )


# ---------------------------------------------------------------- quick CTC training


def train_ctc(model: ToyCRNN, seconds: float, n_train: int = 16000, seed: int = 0) -> dict:
    """Short CPU CTC run so the logits look like a (weakly) trained recogniser:
    blank-dominated, peaky. Not meant to be a good model."""
    torch.manual_seed(seed)
    vocab = load_vocab()
    idx = {c: i + 1 for i, c in enumerate(vocab)}  # 0 = blank
    t0 = time.time()
    xs, labels = load_lines("train", n_train, with_labels=True)
    print(f"loaded {len(xs)} train lines in {time.time() - t0:.1f}s")
    xs = torch.from_numpy(xs)
    targets = [torch.tensor([idx[c] for c in s], dtype=torch.long) for s in labels]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    step, bs, t0, losses = 0, 64, time.time(), []
    rng = np.random.default_rng(seed)
    while time.time() - t0 < seconds:
        b = rng.choice(len(xs), bs, replace=False)
        x = xs[b]
        tg = [targets[i] for i in b]
        logits = model(x).squeeze(2)  # [N, C, T]
        logp = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # [T, N, C]
        loss = ctc(
            logp,
            torch.cat(tg),
            torch.full((bs,), logp.shape[0], dtype=torch.long),
            torch.tensor([len(t) for t in tg]),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {np.mean(losses[-100:]):.3f} ({time.time() - t0:.0f}s)")
    model.eval()
    return {"steps": step, "final_loss": float(np.mean(losses[-50:])), "seconds": time.time() - t0}


def greedy_cer(model: nn.Module, n: int = 256) -> float:
    """Character error rate of greedy CTC decoding on val horizontal lines."""
    xs, labels = load_lines("val", n, offset=1000, with_labels=True)
    with torch.no_grad():
        out = model(torch.from_numpy(xs)).numpy()
    return cer(out, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seconds", type=float, default=0, help="0 = random weights only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    base = ToyCRNN("avg")
    randomise_bn(base, args.seed)
    base.eval()
    kinds = {"random": base}

    if args.train_seconds > 0:
        trained = copy.deepcopy(base)
        info = train_ctc(trained, args.train_seconds, seed=args.seed)
        info["val_cer_greedy_fp32"] = greedy_cer(trained)
        print("training:", info)
        (OUT / "trained_info.txt").parent.mkdir(parents=True, exist_ok=True)
        (OUT / "trained_info.txt").write_text(repr(info))
        kinds["trained"] = trained

    for kind, m in kinds.items():
        torch.save(m.state_dict(), OUT / f"toy_{kind}.pt")
        for pool in ("avg", "mean"):
            m.pool_kind = pool
            fused = fold_bn(m)
            x = torch.randn(1, 1, 32, 256)
            with torch.no_grad():
                d = (fused(x) - m.eval()(x)).abs().max().item()
            print(f"[{kind}/{pool}] BN fold max abs diff: {d:.2e}")
            for opset in (13, 17, 18):
                p = OUT / f"toy_{kind}_{pool}_op{opset}.onnx"
                export_onnx(fused, p, opset)
                print("exported", p.name, p.stat().st_size, "bytes")
            # torch>=2.9 default exporter (dynamo) for comparison
            p = OUT / f"toy_{kind}_{pool}_op18_dynamo.onnx"
            try:
                export_onnx(fused, p, 18, dynamo=True)
                print("exported", p.name, p.stat().st_size, "bytes")
            except Exception as e:  # noqa: BLE001
                print("dynamo export failed:", type(e).__name__, str(e)[:300])
        m.pool_kind = "avg"
    n_params = sum(p.numel() for p in fold_bn(base).parameters())
    print("params (BN folded):", n_params)


if __name__ == "__main__":
    main()
