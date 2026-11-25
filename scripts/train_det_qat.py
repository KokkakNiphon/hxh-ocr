"""Quantization-aware fine-tuning of the line detector (ESP-DL int8, as hxh_ocr/qat.py simulates it),
then export of the folded float model for plain ESP-PPQ PTQ (the recogniser recipe: QAT -> PTQ KL).

  [HXH_LETTERBOX=960] python scripts/train_det_qat.py --ckpt runs/det_base_v1/best.pt --name det_base_v1_qat \
      --data data/pages_v2 data/pages_v3o data/pages_v5 --data-weights 0.25 0.25 0.5 --calib-data data/pages_v5

Calibration = the pages ESP-PPQ will calibrate on (`quantize_det.py`: <calib-data>/val[0:64],
letterboxed). Loss: BCE with OHEM on the kernel logit, Dice on sigmoid(kernel), BCE on the
vertical logit inside kernels (the threshold branch is gone after folding). Selection: int8
(fake-quant) F1 on <val-data>/val[:val-n]. Writes runs/<name>/{log.csv, best_folded.pt} and
exports/<name>.onnx (same interface and checks as export_det.py).
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hxh_ocr.det_data import LETTERBOX, PageDataset, PageIndex, letterbox, make_train_loader, normalize_u8, split_dirs  # noqa: E402
from hxh_ocr.det_model import load_det_checkpoint  # noqa: E402
from hxh_ocr.model import fuse_bn  # noqa: E402
from hxh_ocr.qat import QATModel, equalize  # noqa: E402
from train_det import ValSet, root_weights, score_outputs  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--data-weights", type=float, nargs="+", default=None)
    p.add_argument("--val-data", default="data/pages_v5")
    p.add_argument("--calib-data", default="data/pages_v5")
    p.add_argument("--calib-n", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--long-range", type=float, nargs=2, default=[384, 1280])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--recalib-every", type=int, default=250)
    p.add_argument("--freeze-after", type=float, default=0.75)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--ohem", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equalize", action="store_true", help="cross-layer equalization of the folded model first")
    return p.parse_args()


def qat_loss(out, b, ohem):
    prob_l, vert_l = out[:, 0].float(), out[:, 1].float()
    gt, mask = b["prob"].float(), b["prob_mask"].float()
    bce = F.binary_cross_entropy_with_logits(prob_l, gt, reduction="none")
    pos, neg = gt * mask, (1 - gt) * mask
    n_pos = int(pos.sum().item())
    n_neg = int(min(neg.sum().item(), max(n_pos * ohem, 1000 if n_pos == 0 else 0)))
    neg_loss = torch.topk((bce * neg).flatten(), n_neg).values.sum() if n_neg > 0 else bce.sum() * 0
    l_prob = ((bce * pos).sum() + neg_loss) / max(n_pos + n_neg, 1)
    P = torch.sigmoid(prob_l)
    l_dice = 1 - 2 * (P * gt * mask).sum() / ((P * mask).sum() + (gt * mask).sum() + 1e-6)
    vb = F.binary_cross_entropy_with_logits(vert_l, b["vert"].float(), reduction="none")
    l_vert = (vb * pos).sum() / (pos.sum() + 1e-6)
    return 5 * l_prob + l_dice + l_vert


@torch.no_grad()
def outputs(model, xs_u8, device, bs=8):
    model.eval()
    res = []
    for s in range(0, len(xs_u8), bs):
        x = torch.from_numpy(normalize_u8(xs_u8[s:s + bs]))[:, None].to(device)
        res.append(model(x).float().cpu().numpy())
    return np.concatenate(res)


def export_onnx(model, out):
    import onnx
    import onnxruntime as ort
    from export_det import ALLOWED, constants_to_initializers
    model = model.eval().cpu()
    torch.onnx.export(model, (torch.zeros(1, 1, LETTERBOX, LETTERBOX),), str(out), input_names=["image"],
                      output_names=["logits"], opset_version=18, dynamo=False, do_constant_folding=True)
    g = constants_to_initializers(onnx.load(str(out)))
    onnx.checker.check_model(g)
    ops = {n.op_type for n in g.graph.node}
    assert ops <= ALLOWED, ops
    onnx.save(g, str(out))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 1, LETTERBOX, LETTERBOX)
    with torch.no_grad():
        d = float(np.abs(model(x).numpy() - sess.run(None, {"image": x.numpy()})[0]).max())
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KiB), onnxruntime max|diff| {d:.2e}", flush=True)


def main():
    args = get_args()
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model, cfg = load_det_checkpoint(args.ckpt, "cpu")
    model.thresh = None
    fused = fuse_bn(model, (128, 128))
    if args.equalize:
        print(f"cross-layer equalization: {equalize(fused)} conv pairs", flush=True)
    qat = QATModel(fused, input_exp=-7).to(device)
    print(f"{args.ckpt} @ {LETTERBOX}: {qat.summary()}", flush=True)

    idx = PageIndex([Path(args.calib_data) / "val"], limit=args.calib_n)
    cal = np.stack([letterbox(idx.read(i), LETTERBOX)[0] for i in range(len(idx))])
    calib = [torch.from_numpy(normalize_u8(cal[i:i + 8]))[:, None].to(device) for i in range(0, len(cal), 8)]
    val = ValSet([Path(args.val_data) / "val"], args.val_n)
    tr_roots = split_dirs(args.data, "train")
    ds = PageDataset(tr_roots, train=True, crop=args.crop, long_range=tuple(args.long_range))
    loader = make_train_loader(ds, args.batch_size, args.num_workers, args.seed, weights=root_weights(ds, tr_roots, args.data_weights))

    t0 = time.time()
    qat.calibrate(calib)
    m0 = score_outputs(outputs(qat, val.x, device), val)
    qat.set_mode("float")
    mf = score_outputs(outputs(qat, val.x, device), val)
    qat.set_mode("quant")
    print(f"start: float F1 {mf['f1']:.4f}, int8 (fake-quant) F1 {m0['f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "loss", "lr", "int8_p", "int8_r", "int8_f1", "seconds"])
    wr.writerow([0, "", "", f"{m0['p']:.5f}", f"{m0['r']:.5f}", f"{m0['f1']:.5f}", 0])
    best = m0["f1"]
    torch.save({"config": {**cfg, "folded": True, "qat": vars(args)}, "folded": qat.strip_qat().state_dict(), "step": 0, "int8_f1": best}, out / "best_folded.pt")
    opt = torch.optim.AdamW([p for p in qat.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    step, run = 0, 0.0
    qat.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = qat_loss(qat(b["images"]), b, args.ohem)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 5.0)
            opt.step()
            step += 1
            run += loss.item()
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {run / 50:.4f} lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
                run = 0.0
            if step % args.recalib_every == 0 and step < args.freeze_after * args.steps:
                ch = qat.calibrate(calib)
                if ch:
                    print(f"  recalibrated: {len(ch)} exponents moved", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                qat.calibrate(calib)  # the exponents ESP-PPQ will pick for these weights
                m = score_outputs(outputs(qat, val.x, device), val)
                wr.writerow([step, f"{loss.item():.5f}", f"{lr:.3e}", f"{m['p']:.5f}", f"{m['r']:.5f}", f"{m['f1']:.5f}", f"{time.time() - t0:.0f}"])
                log.flush()
                print(f"== step {step}: int8 P {m['p']:.4f} R {m['r']:.4f} F1 {m['f1']:.4f} (best {max(best, m['f1']):.4f})", flush=True)
                if m["f1"] > best:
                    best = m["f1"]
                    torch.save({"config": {**cfg, "folded": True, "qat": vars(args)}, "folded": qat.strip_qat().state_dict(), "step": step, "int8_f1": best}, out / "best_folded.pt")
                qat.train()
    ck = torch.load(out / "best_folded.pt", map_location="cpu", weights_only=False)
    m = copy.deepcopy(fused).cpu()
    m.load_state_dict(ck["folded"])
    export_onnx(m, ROOT / "exports" / f"{args.name}.onnx")
    print(f"done: best int8 F1 {best:.4f} at step {ck['step']}")


if __name__ == "__main__":
    main()
