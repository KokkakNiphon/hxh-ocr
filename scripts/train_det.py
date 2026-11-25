"""Train the DBNet-lite text-line detector on hxh-synth pages.

  python scripts/train_det.py --data data/pages_v2 --name det_v1 --steps 30000

Outputs runs/<name>/{config.json,log.csv,best.pt,last.pt}. best.pt / last.pt hold the EMA weights
(BatchNorm statistics re-estimated on training crops, see engine.recalibrate_bn) under 'ema' and
the raw weights + optimiser under 'model' / 'opt'. best.pt = best val F1 (quad IoU >= 0.5, 640
letterbox, first --val-n val pages).

Loss: BCE with OHEM 3:1 on the text-kernel logit (masked by prob_mask) x alpha, L1 on the
threshold map inside thresh_mask x beta, Dice on the approximate binary map
sigmoid(k (P - T)) with k = 50, BCE on the vertical logit inside text kernels x gamma.
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
from hxh_ocr.det_data import (LETTERBOX, PageDataset, PageIndex, letterbox,  # noqa: E402
                              make_train_loader, normalize_u8, split_dirs)
from hxh_ocr.det_model import build_det_model, count_macs, count_params  # noqa: E402
from hxh_ocr.det_post import PostParams, decode, match  # noqa: E402
from hxh_ocr.engine import EMA, recalibrate_bn  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data/pages_v2"])
    p.add_argument("--val-data", nargs="+", default=None, help="default: --data")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--long-range", type=float, nargs=2, default=[384, 1280],
                   help="training page long side, log-uniform (scale with the letterbox: x1.5 for 960)")
    p.add_argument("--width", type=float, default=0.5, help="backbone width: 0.5 (tiny) or 1.0 (base)")
    p.add_argument("--fpn", type=int, default=48)
    p.add_argument("--level", type=int, default=16)
    p.add_argument("--mid", type=int, default=64)
    p.add_argument("--head-dw", action="store_true", help="depthwise-separable head convs")
    p.add_argument("--data-weights", type=float, nargs="+", default=None,
                   help="sampling weight per --data dir (default: proportional to its page count)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--bn-calib-batches", type=int, default=40)
    p.add_argument("--alpha", type=float, default=5.0, help="prob BCE weight")
    p.add_argument("--beta", type=float, default=10.0, help="threshold L1 weight")
    p.add_argument("--gamma", type=float, default=1.0, help="vertical BCE weight")
    p.add_argument("--ohem", type=float, default=3.0)
    p.add_argument("--k", type=float, default=50.0)
    p.add_argument("--name", default="det")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-n", type=int, default=300)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default=None, help="start from this checkpoint's (EMA) weights, e.g. to fine-tune")
    p.add_argument("--max-minutes", type=float, default=0, help="stop (eval + save) after this wall time")
    return p.parse_args(argv)


def root_weights(ds, roots, weights):
    """Per-page sampling weights so each data dir gets its share (None: uniform over pages)."""
    if not weights:
        return None
    assert len(weights) == len(roots), "--data-weights needs one weight per --data dir"
    counts = []
    for r in roots:
        with open(Path(r) / "labels.jsonl", encoding="utf-8") as f:
            counts.append(sum(1 for _ in f))
    w = np.concatenate([np.full(n, wt / n) for n, wt in zip(counts, weights)])
    assert len(w) == len(ds), (len(w), len(ds))
    return w


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


def det_loss(out, thr, b, args):
    """out [B, 2, h, w] logits, thr [B, 1, h, w] logits; targets from the batch dict."""
    prob_l, vert_l, thr_l = out[:, 0].float(), out[:, 1].float(), thr[:, 0].float()
    gt = b["prob"].float()
    mask = b["prob_mask"].float()
    # BCE with OHEM (all positives + the hardest 3x as many negatives)
    bce = F.binary_cross_entropy_with_logits(prob_l, gt, reduction="none")
    pos = gt * mask
    neg = (1 - gt) * mask
    n_pos = int(pos.sum().item())
    n_neg = int(min(neg.sum().item(), max(n_pos * args.ohem, 1000 if n_pos == 0 else 0)))
    pos_loss = (bce * pos).sum()
    neg_loss = torch.topk((bce * neg).flatten(), n_neg).values.sum() if n_neg > 0 else bce.sum() * 0
    l_prob = (pos_loss + neg_loss) / max(n_pos + n_neg, 1)
    # threshold map L1 inside thresh_mask
    T = torch.sigmoid(thr_l)
    tm = b["thresh_mask"].float()
    l_thr = ((T - b["thresh"]).abs() * tm).sum() / (tm.sum() + 1e-6)
    # Dice on the approximate binary map
    P = torch.sigmoid(prob_l)
    B = torch.sigmoid(args.k * (P - T))
    inter = (B * gt * mask).sum()
    l_dice = 1 - 2 * inter / ((B * mask).sum() + (gt * mask).sum() + 1e-6)
    # vertical orientation inside text kernels
    kern = pos
    vb = F.binary_cross_entropy_with_logits(vert_l, b["vert"].float(), reduction="none")
    l_vert = (vb * kern).sum() / (kern.sum() + 1e-6)
    total = args.alpha * l_prob + args.beta * l_thr + l_dice + args.gamma * l_vert
    return total, {"prob": l_prob.item(), "thr": l_thr.item(), "dice": l_dice.item(), "vert": l_vert.item()}


class ValSet:
    """Letterboxed val pages kept in memory as u8, with GT quads in original coordinates."""

    def __init__(self, roots, n):
        self.idx = PageIndex(roots, limit=n)
        self.x, self.scales = [], []
        for i in range(len(self.idx)):
            u8, sx, sy = letterbox(self.idx.read(i), LETTERBOX)
            self.x.append(u8)
            self.scales.append((sx, sy))
        self.x = np.stack(self.x)

    def __len__(self):
        return len(self.x)


@torch.no_grad()
def run_outputs(model, xs_u8, device, amp, bs=16):
    """u8 [N, 640, 640] -> float32 logits [N, 2, 320, 320] (numpy)."""
    model.eval()
    outs = []
    for s in range(0, len(xs_u8), bs):
        x = torch.from_numpy(normalize_u8(xs_u8[s:s + bs]))[:, None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            o = model(x)
        outs.append(o.float().cpu().numpy())
    return np.concatenate(outs)


def score_outputs(outs, val: ValSet, post: PostParams | None = None, iou=0.5):
    tp = fp = fn = 0
    o_ok = o_n = 0
    ious = []
    for i in range(len(outs)):
        sx, sy = val.scales[i]
        dets = decode(outs[i], sx, sy, post)
        gq, gv = val.idx.lines(i)
        m = match(gq.astype(np.float64), [d.quad for d in dets], iou)
        tp += len(m); fp += len(dets) - len(m); fn += len(gq) - len(m)
        for g, j, v in m:
            ious.append(v)
            o_ok += int(dets[j].vertical == bool(gv[g])); o_n += 1
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"p": P, "r": R, "f1": 2 * P * R / max(P + R, 1e-9), "orient": o_ok / max(o_n, 1),
            "iou": float(np.mean(ious)) if ious else 0.0, "tp": tp, "fp": fp, "fn": fn}


def main():
    args = get_args()
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    tr_roots = split_dirs(args.data, "train")
    va_roots = split_dirs(args.val_data or args.data, "val")
    assert tr_roots and va_roots
    train_ds = PageDataset(tr_roots, train=True, crop=args.crop, long_range=tuple(args.long_range))
    loader = make_train_loader(train_ds, args.batch_size, args.num_workers, args.seed,
                               weights=root_weights(train_ds, tr_roots, args.data_weights))
    calib_loader = make_train_loader(train_ds, args.batch_size, 0, args.seed + 7)
    calib_state = {"it": iter(calib_loader)}

    def calib_batches(n):
        res = []
        while len(res) < n:
            try:
                res.append(next(calib_state["it"]))
            except StopIteration:
                calib_state["it"] = iter(calib_loader)
        return res

    t_load = time.time()
    val = ValSet(va_roots, args.val_n)
    print(f"train {len(train_ds)} pages from {len(tr_roots)} dirs, val {len(val)} pages "
          f"(loaded in {time.time()-t_load:.0f}s); device {device}", flush=True)

    model = build_det_model(pretrained=args.pretrained, width=args.width, fpn=args.fpn, level=args.level,
                            mid=args.mid, head_dw=args.head_dw)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck.get("ema") or ck["model"], strict=False)
        print(f"init from {args.init} (step {ck.get('step')}): {len(missing)} missing, {len(unexpected)} unexpected keys",
              flush=True)
    params, macs = count_params(model), count_macs(model, LETTERBOX)
    print(f"dbnet_lite: {params/1e3:.1f}k deploy params ({count_params(model, False)/1e3:.1f}k with the "
          f"threshold head), {macs/1e6:.0f}M MACs @ {LETTERBOX}x{LETTERBOX}", flush=True)
    model.to(device).train()
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))
    config = {**vars(args), "model_cfg": model.cfg, "params": params, "macs_640": macs, "letterbox": LETTERBOX,
              "train_pages": len(train_ds)}
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    step, best_f1, seconds_before = 0, -1.0, 0.0
    log_path = out / "log.csv"
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ema.updates = ck["step"]
        best_f1 = ck.get("best_f1", best_f1)
        seconds_before = ck.get("seconds", 0.0)
        print(f"resumed at step {step}")
    elif log_path.exists():
        log_path.unlink()
    if not log_path.exists():
        log_path.write_text("step,train_loss,l_prob,l_thr,l_dice,l_vert,lr,val_p,val_r,val_f1,"
                            "val_orient,val_iou,seconds\n")

    t0 = time.time()
    eval_model = ema.module

    def save(path):
        torch.save({"config": config, "step": step, "ema": eval_model.state_dict(),
                    "model": model.state_dict(), "opt": opt.state_dict(), "best_f1": best_f1,
                    "seconds": time.time() - t0 + seconds_before}, path)

    run = {"loss": 0.0, "prob": 0.0, "thr": 0.0, "dice": 0.0, "vert": 0.0}
    run_n = 0
    t_data = t_step = 0.0
    stop = False
    model.train()
    while step < args.steps and not stop:
        t_a = time.time()
        for batch in loader:
            if step >= args.steps:
                break
            t_b = time.time(); t_data += t_b - t_a
            lr = lr_at(step, args)
            for g in opt.param_groups:
                g["lr"] = lr
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                o, thr = model(b["images"])
            loss, parts = det_loss(o, thr, b, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            step += 1
            run["loss"] += loss.item(); run_n += 1
            for k, v in parts.items():
                run[k] += v
            t_a = time.time(); t_step += t_a - t_b
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"step {step}/{args.steps} loss {run['loss']/run_n:.4f} (prob {run['prob']/run_n:.4f} "
                      f"thr {run['thr']/run_n:.4f} dice {run['dice']/run_n:.4f} vert {run['vert']/run_n:.4f}) "
                      f"lr {lr:.2e} {el:.0f}s data {t_data/step*1000:.0f}ms/it step {t_step/step*1000:.0f}ms/it",
                      flush=True)
            timeout = args.max_minutes > 0 and (time.time() - t0) / 60 > args.max_minutes
            if step % args.eval_every == 0 or step == args.steps or timeout:
                eval_model = copy.deepcopy(ema.module)
                if args.bn_calib_batches > 0:
                    recalibrate_bn(eval_model, calib_batches(args.bn_calib_batches), device, amp)
                outs = run_outputs(eval_model, val.x, device, amp)
                m = score_outputs(outs, val)
                secs = time.time() - t0 + seconds_before
                n = max(run_n, 1)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{run['loss']/n:.5f}", f"{run['prob']/n:.5f}",
                                            f"{run['thr']/n:.5f}", f"{run['dice']/n:.5f}", f"{run['vert']/n:.5f}",
                                            f"{lr:.3e}", f"{m['p']:.5f}", f"{m['r']:.5f}", f"{m['f1']:.5f}",
                                            f"{m['orient']:.5f}", f"{m['iou']:.5f}", f"{secs:.1f}"])
                print(f"== eval step {step}: P {m['p']:.4f} R {m['r']:.4f} F1 {m['f1']:.4f} orient "
                      f"{m['orient']:.4f} IoU {m['iou']:.3f} (best F1 {max(best_f1, m['f1']):.4f})", flush=True)
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    save(out / "best.pt")
                save(out / "last.pt")
                run = {k: 0.0 for k in run}
                run_n = 0
                model.train()
                if timeout:
                    print(f"stopping: --max-minutes {args.max_minutes} reached")
                    stop = True
                    break
    print(f"done in {time.time()-t0:.0f}s, best val F1 {best_f1:.4f}")


if __name__ == "__main__":
    main()
