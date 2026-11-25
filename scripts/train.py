"""Train the CTC line recogniser.

  python scripts/train.py --data data/synth_v1 --arch mnv4s --name mnv4s_v1 --steps 20000

Outputs runs/<name>/{config.json,log.csv,best.pt,last.pt}.  best.pt / last.pt hold the
EMA weights under 'ema' (used by eval / export) and raw weights + optimiser under
'model' / 'opt'.
"""
from __future__ import annotations

import argparse
import copy
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
from hxh_ocr.engine import EMA, input_lengths, predict, recalibrate_bn  # noqa: E402
from hxh_ocr.metrics import score  # noqa: E402
from hxh_ocr.model import build_model, report  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data/synth_v1"], help="dataset dirs (each with train/ val/)")
    p.add_argument("--arch", choices=["mnv4s", "mnv4s1", "mnv4s3", "tiny"], default="mnv4s")
    p.add_argument("--init", default=None, help="start from this checkpoint's EMA weights (fine-tuning)")
    p.add_argument("--height", type=int, default=32, choices=[32, 48])
    p.add_argument("--max-width", type=int, default=None, help="default 16*height")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--bn-calib-batches", type=int, default=50,
                   help="training batches used to re-estimate the EMA model's BN stats before eval/save")
    p.add_argument("--orient", default="h", choices=["h", "v"], help="h: horizontal lines, v: vertical columns (rotated 90 deg CCW)")
    p.add_argument("--name", default=None, help="run name -> runs/<name>/ (default: <arch>)")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-n", type=int, default=1000, help="val subset size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    return p.parse_args(argv)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def param_groups(model, wd):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    args = get_args()
    name = args.name or args.arch
    out = Path(args.out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"  # bf16 autocast on CUDA, fp32 on CPU

    vocab = find_vocab(args.data)
    tr_roots, va_roots = split_dirs(args.data, "train"), split_dirs(args.data, "val")
    assert tr_roots and va_roots, "need at least one finished train and val split"
    train_ds = LineDataset(tr_roots, vocab, args.height, args.max_width, train=True, orient=args.orient)
    val_ds = LineDataset(va_roots, vocab, args.height, args.max_width, train=False, orient=args.orient,
                         limit=args.val_n)
    train_loader, sampler = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader, _ = make_loader(val_ds, 128, False, 0)
    calib_loader, calib_sampler = make_loader(train_ds, args.batch_size, True, 0, args.seed + 7)
    calib_state = {"it": iter(calib_loader), "epoch": 0}

    def calib_batches(n):
        out = []
        while len(out) < n:
            try:
                out.append(next(calib_state["it"]))
            except StopIteration:
                calib_state["epoch"] += 1
                calib_sampler.set_epoch(calib_state["epoch"])
                calib_state["it"] = iter(calib_loader)
        return out
    print(f"train {len(train_ds)} lines from {len(tr_roots)} dirs, val {len(val_ds)}; device {device}")

    model = build_model(args.arch, vocab.num_classes, pretrained=args.pretrained, dropout=args.dropout)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck["config"].get("folded"):
            raise SystemExit("--init needs an unfolded train.py checkpoint, not a QAT one")
        sd, own = dict(ck["ema"]), model.state_dict()
        for k, v in list(sd.items()):
            if k in own and own[k].shape != v.shape:
                # a bigger vocab (the extended classifier): keep the old classes' rows, new rows start fresh
                if v.dim() >= 1 and v.shape[1:] == own[k].shape[1:] and v.shape[0] < own[k].shape[0]:
                    w = own[k].clone()
                    w[: v.shape[0]] = v
                    if k.endswith("bias"):
                        w[v.shape[0]:] = v.min()  # new classes start unlikely
                    sd[k] = w
                    print(f"  {k}: {tuple(v.shape)} -> {tuple(w.shape)} (old rows kept)")
                else:
                    del sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"init from {args.init} (step {ck.get('step')}): {len(missing)} missing, {len(unexpected)} unexpected keys")
    rep = report(model)
    print(f"{args.arch}: {rep['params']/1e3:.0f}k params, {rep['macs_32x256']/1e6:.1f}M MACs @32x256")
    model.to(device).train()
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))

    config = {**vars(args), "name": name, "num_classes": vocab.num_classes, "tokens": vocab.tokens,
              "stats": rep, "train_lines": len(train_ds)}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    step, best_cer, seconds_before = 0, float("inf"), 0.0
    log_path = out / "log.csv"
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ema.updates = ck["step"]
        best_cer = ck.get("best_cer", best_cer)
        seconds_before = ck.get("seconds", 0.0)
        print(f"resumed at step {step}")
    elif log_path.exists():
        log_path.unlink()
    if not log_path.exists():
        log_path.write_text("step,train_loss,lr,val_cer,val_acc,seconds\n")

    t0 = time.time()

    eval_model = ema.module  # replaced by a BN-recalibrated copy of the EMA at each eval

    def save(path):
        torch.save({"config": config, "step": step, "ema": eval_model.state_dict(),
                    "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_cer": best_cer, "seconds": time.time() - t0 + seconds_before}, path)

    run_loss, run_n, epoch = 0.0, 0, 0
    model.train()
    while step < args.steps:
        sampler.set_epoch(epoch + args.seed * 1000); epoch += 1
        for batch in train_loader:
            if step >= args.steps:
                break
            lr = lr_at(step, args)
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["images"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)  # [B, C, 1, T]
            logp = F.log_softmax(logits.float()[:, :, 0, :], dim=1).permute(2, 0, 1)  # [T, B, C], fp32
            in_len = input_lengths(batch["widths"], logp.shape[0])
            loss = F.ctc_loss(logp, batch["targets"], in_len, batch["target_lengths"],
                              blank=0, reduction="mean", zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            step += 1
            run_loss += loss.item(); run_n += 1
            if step % args.log_every == 0:
                print(f"step {step}/{args.steps} loss {run_loss/run_n:.4f} lr {lr:.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                eval_model = copy.deepcopy(ema.module)
                if args.bn_calib_batches > 0:
                    recalibrate_bn(eval_model, calib_batches(args.bn_calib_batches), device, amp)
                refs, hyps, _ = predict(eval_model, val_loader, device, vocab.tokens, amp)
                cer, acc = score(refs, hyps)
                secs = time.time() - t0 + seconds_before
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{run_loss/max(1,run_n):.5f}", f"{lr:.3e}",
                                            f"{cer:.5f}", f"{acc:.5f}", f"{secs:.1f}"])
                print(f"== eval step {step}: val CER {cer:.4f} acc {acc:.4f} (best {min(best_cer, cer):.4f})",
                      flush=True)
                if cer < best_cer:
                    best_cer = cer
                    save(out / "best.pt")
                save(out / "last.pt")
                run_loss, run_n = 0.0, 0
                model.train()
    print(f"done in {time.time()-t0:.0f}s, best val CER {best_cer:.4f}")


if __name__ == "__main__":
    main()
