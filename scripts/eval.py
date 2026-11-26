"""Evaluate a checkpoint on a labels.tsv: CER, line accuracy, top confusions, worst-32 grid.

  python scripts/eval.py --run runs/mnv4s_v1 --tsv data/synth_v1/val/labels.tsv --n 5000

Writes <run>/worst32.png and <run>/eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.data import LineDataset, Vocab, make_loader  # noqa: E402
from hxh_ocr.engine import load_checkpoint, predict  # noqa: E402
from hxh_ocr.fonts import label_font  # noqa: E402
from hxh_ocr.metrics import align, confusions, cross_metrics, edit_distance, score  # noqa: E402


def worst_grid(ds, refs, hyps, idxs, path, k=32, cols=2, scale=2):
    """PNG grid of the k worst predictions: image on the left, GT (green) / prediction (red)."""
    errs = sorted(range(len(refs)), key=lambda i: (-edit_distance(refs[i], hyps[i]) / max(1, len(refs[i])),
                                                   -len(refs[i])))[:k]
    H = ds.height * scale
    font = label_font(18)
    cell_w, cell_h = ds.max_width * scale // 2 + 20, H + 52
    rows = (len(errs) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    dr = ImageDraw.Draw(canvas)
    for n, i in enumerate(errs):
        x0, y0 = (n % cols) * cell_w, (n // cols) * cell_h
        img = cv2.imread(ds.paths[idxs[i]], cv2.IMREAD_GRAYSCALE)
        w = min(int(img.shape[1] * ds.height / img.shape[0]), ds.max_width)
        img = cv2.resize(img, (w, ds.height), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (min(w * scale, cell_w - 10), H), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(img).convert("RGB"), (x0 + 5, y0 + 4))
        dr.text((x0 + 5, y0 + H + 6), "GT   " + refs[i], font=font, fill=(0, 120, 0))
        dr.text((x0 + 5, y0 + H + 28), "PRED " + (hyps[i] or "(empty)"), font=font, fill=(190, 0, 0))
        dr.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(200, 200, 200))
    canvas.save(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # kana on a cp1252 Windows console
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run dir (uses best.pt)")
    p.add_argument("--ckpt", default=None, help="checkpoint (default <run>/best.pt)")
    p.add_argument("--tsv", default="data/synth_v1/val/labels.tsv")
    p.add_argument("--n", type=int, default=2000, help="use the first n lines (0 = all)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--height", type=int, default=None, help="default: training height")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="", help="suffix for eval_<tag>.json / worst32_<tag>.png")
    args = p.parse_args()

    run = Path(args.run)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_checkpoint(args.ckpt or run / "best.pt", device, args.weights)
    vocab = Vocab(cfg["tokens"])
    root = Path(args.tsv).parent
    # labels may hold tokens the model does not have (digits / punctuation of the extended vocab
    # read by an older model): encode them with a superset, they simply count as errors
    extra = sorted({c for l in Path(args.tsv).read_text(encoding="utf-8").splitlines() if l.count("\t") >= 2
                    for c in l.split("\t")[1]} - set(cfg["tokens"]))
    ds = LineDataset([root], Vocab(list(cfg["tokens"]) + extra), args.height or cfg["height"], cfg.get("max_width"),
                     train=False, orient=cfg.get("orient", "h"), limit=args.n or None)
    loader, _ = make_loader(ds, args.batch_size, False, 0)
    refs, hyps, idxs = predict(model, loader, device, vocab.tokens, amp=device.type == "cuda")
    # empty labels are negatives (non-Hunter crops, lines_v5): scored apart
    pos = [i for i, r in enumerate(refs) if r]
    neg = [i for i, r in enumerate(refs) if not r]
    cer, acc = score([refs[i] for i in pos], [hyps[i] for i in pos])
    false_read = sum(1 for i in neg if hyps[i].strip()) / max(1, len(neg))
    # extended vocab: digits / punctuation read on lines that have none (pure Hunter lines)
    xs = set("0123456789!?:/.-×")  # with the cross (v7)
    plain = [i for i in pos if not set(refs[i]) & xs]
    x_chars = sum(sum(c in xs for c in hyps[i]) for i in plain)
    x_rate = x_chars / max(1, sum(len(refs[i]) for i in plain))
    x_lines = sum(1 for i in plain if set(hyps[i]) & xs)
    if set(cfg["tokens"]) & xs:
        print(f"digits / punctuation read on lines without any: {x_chars} chars ({1000 * x_rate:.2f} per 1000 chars), "
              f"on {x_lines} of {len(plain)} lines")
    # lines with digits / punctuation: their CER and line accuracy, and the error rate on the digit /
    # punctuation characters alone (substituted or dropped ones, plus inserted ones, over their count)
    xl = [i for i in pos if set(refs[i]) & xs]
    x_cer, x_acc = score([refs[i] for i in xl], [hyps[i] for i in xl]) if xl else (0.0, 0.0)
    x_ref = x_err = 0
    for i in xl:
        for a, b in align(refs[i], hyps[i])[1]:
            if a is not None and a in xs:
                x_ref += 1
                x_err += a != b
            elif a is None and b in xs:
                x_err += 1
    x_char_err = x_err / max(1, x_ref)
    if xl:
        print(f"{len(xl)} lines with digits / punctuation: CER {x_cer:.4f}  line accuracy {x_acc:.4f} | "
              f"digit / punctuation characters: {x_ref}, error rate {x_char_err:.4f}")
    xm = cross_metrics([refs[i] for i in pos], [hyps[i] for i in pos])
    if xm["cross_ref"] or xm["cross_false"]:
        print(f"crosses: {xm['cross_ref']} in the labels, recall {xm['cross_recall']:.4f}, precision {xm['cross_precision']:.4f}, "
              f"missed ones read as {xm['cross_read_as']}; {xm['cross_lines']} lines with crosses, line accuracy "
              f"{xm['cross_lines_acc']:.4f}; crosses on {xm['cross_on_plain_lines']} lines without any; "
              f"look-alikes read as crosses {xm['lookalikes_read_as_cross']}")
    conf = confusions([refs[i] for i in pos], [hyps[i] for i in pos], 20)
    print(f"{len(pos)} lines  CER {cer:.4f}  line accuracy {acc:.4f}" +
          (f" | {len(neg)} negatives, read as text: {false_read:.4f}" if neg else ""))
    print("top confusions (GT -> pred, count; <none> = missing / extra):")
    for (a, b), c in conf:
        print(f"  {a} -> {b}  {c}")
    tag = f"_{args.tag}" if args.tag else ""
    worst_grid(ds, [refs[i] for i in pos], [hyps[i] for i in pos], [idxs[i] for i in pos], run / f"worst32{tag}.png")
    (run / f"eval{tag}.json").write_text(json.dumps(
        {"tsv": args.tsv, "n": len(pos), "cer": cer, "acc": acc, "negatives": len(neg), "negatives_read": false_read,
         "extra_tokens_on_plain_lines": x_chars, "extra_tokens_per_char": x_rate, "plain_lines": len(plain), "plain_lines_with_extra": x_lines,
         "plain_chars": sum(len(refs[i]) for i in plain),
         "digit_lines": len(xl), "digit_lines_cer": x_cer, "digit_lines_acc": x_acc,
         "digit_chars": x_ref, "digit_char_err": x_char_err, **xm,
         "confusions": [[a, b, c] for (a, b), c in conf]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {run / f'worst32{tag}.png'}")


if __name__ == "__main__":
    main()
