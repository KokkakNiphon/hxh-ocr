"""Quantization-aware fine-tuning for ESP-DL int8 (ESP32-S3), see hxh_ocr/qat.py.

  python scripts/train_qat.py --ckpt runs/mnv4s_h32/best.pt --data data/synth_v3 --name qat_mnv4s_h32

Loads a train.py checkpoint (EMA weights with recalibrated BN), folds BN, wraps the model in
ESP-DL fake quantization, calibrates the activation exponents exactly as ESP-PPQ will (KL on
the same calibration lines as experiments/esp_ppq_spike/quantize.py by default), and fine-tunes
with CTC. Exponents are re-calibrated every --recalib-every steps until --freeze-after of the
run, then frozen. Writes runs/<name>/{config.json,log.csv,best.pt,last.pt,exponents.json}.
Checkpoints hold the stripped float model (BN folded, `folded: True`) under 'ema' and 'model',
so scripts/export_onnx.py and scripts/eval.py take them unchanged.

log.csv: step, train_loss, lr, val_cer / val_acc (fake-quant int8), seconds,
val_cer_float / val_acc_float (same weights without fake quant), exp_changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.data import LineDataset, find_vocab, make_loader, split_dirs  # noqa: E402
from hxh_ocr.engine import ctc_lengths, load_checkpoint, pad_like_app, predict  # noqa: E402
from hxh_ocr.metrics import score  # noqa: E402
from hxh_ocr.model import fuse_bn  # noqa: E402
from hxh_ocr.qat import QATModel, equalize  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train.py checkpoint (e.g. runs/mnv4s_h32/best.pt)")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--data", nargs="+", default=["data/synth_v3"])
    p.add_argument("--name", required=True, help="-> runs/<name>/")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--input-exp", type=int, default=None,
                   help="fixed input exponent; default -7 for kl (what ESP-PPQ picks), calibrated for minmax")
    p.add_argument("--calib-algo", default="kl", choices=["kl", "minmax"],
                   help="activation calibration ESP-PPQ will use (quantize.py --calib-algo); kl is its default")
    p.add_argument("--calib-dir", default="data/synth_v1/val",
                   help="lines for the KL calibration; default = what esp_ppq_spike/quantize.py uses")
    p.add_argument("--calib-n", type=int, default=64)
    p.add_argument("--calib-width", type=int, default=None, help="static width, default 8*height")
    p.add_argument("--recalib-every", type=int, default=100,
                   help="re-run the KL calibration every N steps (ESP-PPQ's KL pick flips for ~10%% of "
                        "tensors under tiny weight changes, so keep following it)")
    p.add_argument("--freeze-after", type=float, default=1.0,
                   help="fraction of the run after which activation exponents stay fixed")
    p.add_argument("--settle-rounds", type=int, default=5,
                   help="after the run: recalibrate, and if an exponent moved train --settle-steps more "
                        "steps at the final lr under the new exponents; repeat until stable")
    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--exp-sampling", type=int, default=0,
                   help="N > 0: each training step draws every activation exponent from its last N "
                        "calibrations, so the network tolerates whichever near-tied pick ESP-PPQ makes")
    p.add_argument("--no-fakequant", action="store_true",
                   help="control run: same fine-tune in float (int8 is still evaluated)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--pad-blank", type=float, default=0.0,
                   help="v7: > 0 supervises every frame with CTC (the padding after a short line must read as blank) and pads "
                        "this share of the batches to the static width like the app (default 0: frames over the text only)")
    p.add_argument("--equalize", action="store_true",
                   help="cross-layer equalization of the folded model before QAT (width-1.0 MobileNetV4)")
    p.add_argument("--tf32", action="store_true", help="allow TF32 convs (inexact float-mode / calibration)")
    return p.parse_args(argv)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def calib_batches(args, vocab, height, device, bs=8, orient="h"):
    """The calibration lines as quantize.py feeds them: height H, squashed or right-padded
    (replicated edge) to the static width."""
    W = args.calib_width or 8 * height
    ds = LineDataset([Path(args.calib_dir)], vocab, height, W, train=False, limit=args.calib_n, orient=orient)
    xs = []
    for i in range(len(ds)):
        x = ds[i][0]
        if x.shape[-1] < W:
            x = torch.cat([x, x[..., -1:].expand(-1, -1, W - x.shape[-1])], dim=-1)
        xs.append(x)
    xs = torch.stack(xs).to(device)
    return [xs[i:i + bs] for i in range(0, len(xs), bs)]


def main():
    args = get_args()
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # fake-quant tensors are exact in TF32, but float-mode passes (KL calibration) are not
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    base, cfg = load_checkpoint(args.ckpt, "cpu", args.weights)
    H = cfg["height"]
    max_width = cfg.get("max_width") or 16 * H
    vocab = find_vocab(args.data)
    assert vocab.tokens == cfg["tokens"], "dataset vocab differs from the checkpoint's"
    fused = fuse_bn(base, (H, 8 * H))
    if args.equalize:
        print(f"cross-layer equalization: {equalize(fused)} conv pairs")
    input_exp = args.input_exp if args.input_exp is not None else (-7 if args.calib_algo == "kl" else None)
    qat = QATModel(fused, input_exp=input_exp, calib_algo=args.calib_algo).to(device)
    qat.exp_sampling = args.exp_sampling
    print(f"{cfg['arch']} h{H} from {args.ckpt}: {qat.summary()}")

    tr_roots, va_roots = split_dirs(args.data, "train"), split_dirs(args.data, "val")
    orient = cfg.get("orient", "h")
    train_ds = LineDataset(tr_roots, vocab, H, max_width, train=True, orient=orient)
    val_ds = LineDataset(va_roots, vocab, H, max_width, train=False, limit=args.val_n, orient=orient)
    train_loader, sampler = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader, _ = make_loader(val_ds, 128, False, 0)
    calib = calib_batches(args, vocab, H, device, orient=orient)
    print(f"train {len(train_ds)} lines, val {len(val_ds)}, calib {sum(len(b) for b in calib)} lines "
          f"({args.calib_dir}, static {calib[0].shape[-1]} px); device {device}")

    t0 = time.time()
    qat.calibrate(calib)
    print(f"calibrated in {time.time() - t0:.1f}s")

    config = {**cfg, "name": args.name, "qat": {**vars(args), "base_config": cfg.get("name")},
              "stats": cfg.get("stats"), "train_lines": len(train_ds)}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    def evaluate():
        res = {}
        for mode in ("quant", "float"):
            qat.set_mode(mode)
            refs, hyps, _ = predict(qat, val_loader, device, vocab.tokens, amp=False)
            res[mode] = score(refs, hyps)
        qat.set_mode("float" if args.no_fakequant else "quant")
        return res

    decay = [p for p in qat.parameters() if p.ndim >= 2]
    no_decay = [p for p in qat.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    log_path = out / "log.csv"
    log_path.write_text("step,train_loss,lr,val_cer,val_acc,seconds,val_cer_float,val_acc_float,exp_changes\n")
    step, best_cer, exp_changes, run_loss, run_n = 0, float("inf"), 0, 0.0, 0

    def log_eval(loss, lr):
        nonlocal best_cer
        r = evaluate()
        (cer, acc), (fcer, facc) = r["quant"], r["float"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{loss:.5f}", f"{lr:.3e}", f"{cer:.5f}", f"{acc:.5f}",
                                    f"{time.time() - t0:.1f}", f"{fcer:.5f}", f"{facc:.5f}", exp_changes])
        print(f"== eval step {step}: int8 CER {cer:.4f} acc {acc:.4f} | float CER {fcer:.4f} acc {facc:.4f} "
              f"| exponent changes {exp_changes}", flush=True)
        if step > 0:
            if cer < best_cer:
                best_cer = cer
                save(out / "best.pt")
            save(out / "last.pt")

    def save(path):
        sd = qat.strip_qat().state_dict()
        torch.save({"config": config, "folded": True, "step": step, "ema": sd, "model": sd,
                    "exponents": qat.exponents(),
                    "best_cer": best_cer, "seconds": time.time() - t0}, path)

    t0 = time.time()
    log_eval(float("nan"), 0.0)
    freeze_step = int(args.freeze_after * args.steps)
    qat.set_mode("float" if args.no_fakequant else "quant")
    batches = iter(())
    epoch = 0

    def next_batch():
        nonlocal batches, epoch
        while True:
            try:
                return next(batches)
            except StopIteration:
                sampler.set_epoch(epoch + args.seed * 1000 + 777); epoch += 1
                batches = iter(train_loader)

    pad_rng = random.Random(args.seed + 12345)

    def train_step(lr):
        nonlocal step, run_loss, run_n
        for g in opt.param_groups:
            g["lr"] = lr
        qat.train()
        batch = next_batch()
        x = batch["images"].to(device, non_blocking=True)
        if args.pad_blank > 0 and pad_rng.random() < args.pad_blank:
            x = pad_like_app(x, max_width)  # the app's padding to the static width
        logits = qat(x)  # [B, C, 1, T], fp32
        logp = F.log_softmax(logits[:, :, 0, :], dim=1).permute(2, 0, 1)
        in_len = ctc_lengths(batch["widths"], logp.shape[0], args.pad_blank)
        loss = F.ctc_loss(logp, batch["targets"], in_len, batch["target_lengths"],
                          blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qat.parameters(), 5.0)
        opt.step()
        step += 1
        run_loss += loss.item(); run_n += 1

    def recalibrate(tag):
        nonlocal exp_changes
        changed = qat.calibrate(calib)
        exp_changes += len(changed)
        if changed:
            print(f"recalibrated ({tag}, step {step}): {len(changed)} exponents changed {changed}", flush=True)
        qat.set_mode("float" if args.no_fakequant else "quant")
        return changed

    while step < args.steps:
        lr = lr_at(step, args)
        train_step(lr)
        if step % args.log_every == 0:
            print(f"step {step}/{args.steps} loss {run_loss / run_n:.4f} lr {lr:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % args.recalib_every == 0 and step <= freeze_step and step < args.steps:
            recalibrate("periodic")
        if step % args.eval_every == 0 or step == args.steps:
            log_eval(run_loss / max(1, run_n), lr)
            run_loss, run_n = 0.0, 0

    # ESP-PPQ calibrates the exported weights once more. Settle: if that moves an exponent,
    # train a few steps at the final lr under the new exponents and check again.
    final = {"steps_main": step, "settle": []}
    for _ in range(args.settle_rounds + 1):
        changed = recalibrate("settle check")
        final["settle"].append({"step": step, "changed": changed})
        if not changed or args.no_fakequant or len(final["settle"]) > args.settle_rounds:
            break
        for _ in range(args.settle_steps):
            train_step(lr_at(args.steps - 1, args))
    final["stable"] = not final["settle"][-1]["changed"]
    final["steps"], final["seconds"] = step, time.time() - t0
    if step != args.steps or final["settle"][-1]["changed"]:
        log_eval(run_loss / max(1, run_n), lr_at(args.steps - 1, args))
    (out / "exponents.json").write_text(json.dumps(qat.exponents(), indent=1))
    (out / "qat_result.json").write_text(json.dumps(final, indent=1))
    print(f"done in {time.time() - t0:.0f}s ({step} steps, exponents stable at the end: {final['stable']}), "
          f"best int8 val CER {best_cer:.4f}")


if __name__ == "__main__":
    main()
