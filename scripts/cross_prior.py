"""Readers v7: how much to lower the cross's logit (a prior shift) before int8 export.

The training lines carry crosses far more often, relative to the long-vowel mark ー, than real
Hunter text does, and a third of their crosses follow a ー; on blurry photo-style lines the v7
readers read about 5% of the ー as × (int8, lines_v5b val). Lowering the × logit by a constant
trades cross recall for fewer of those. This sweeps the shift on a checkpoint (the QAT one,
BN-folded, run in float) over two tuning sets that are not reported as results:

  * lines_v7x val (generator v7, cross-heavy): cross recall and precision;
  * lines_v5 val (photo-style, no crosses): ー read as ×, and crosses read on plain lines.

    python scripts/cross_prior.py --ckpt runs/qat_v7b_base_h48/best.pt [--shifts 0,0.5,1,1.5,2,3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr.data import LineDataset, Vocab, make_loader  # noqa: E402
from hxh_ocr.decode import decode_text  # noqa: E402
from hxh_ocr.engine import input_lengths, load_checkpoint  # noqa: E402
from hxh_ocr.metrics import CROSS, align, cross_metrics  # noqa: E402


@torch.no_grad()
def logits_of(model, tsv: Path, vocab: Vocab, height: int, device, limit=None):
    ds = LineDataset([tsv.parent], vocab, height, 16 * height, train=False, orient="h", limit=limit)
    loader, _ = make_loader(ds, 64, False, 0)
    out = []
    for b in loader:
        lo = model(b["images"].to(device)).float()[:, :, 0, :].cpu()  # [B, C, T]
        lens = input_lengths(b["widths"], lo.shape[-1])
        for row, n, ref in zip(lo, lens.tolist(), b["texts"]):
            out.append((ref, row[:, :n]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shifts", default="0,0.5,1,1.5,2,2.5,3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.ckpt, device)
    tokens = cfg["tokens"]
    vocab = Vocab(tokens)
    xid = tokens.index(CROSS) + 1
    sets = {"lines_v7x val": ROOT / "data/lines_v7x/val/labels.tsv", "lines_v5 val": ROOT / "data/lines_v5/val/labels.tsv"}
    cache = {k: [(r, lo) for r, lo in logits_of(model, p, vocab, cfg["height"], device) if r] for k, p in sets.items()}
    res = []
    for s in [float(v) for v in a.shifts.split(",")]:
        row = {"shift": s}
        for k, items in cache.items():
            refs, hyps = [], []
            for ref, lo in items:
                lo = lo.clone()
                lo[xid] -= s
                hyps.append(decode_text(lo.argmax(0).tolist(), tokens))
                refs.append(ref)
            m = cross_metrics(refs, hyps)
            edits = sum(align(r, h)[0] for r, h in zip(refs, hyps))
            row[k] = {"cer": edits / max(1, sum(map(len, refs))), "recall": m["cross_recall"], "precision": m["cross_precision"],
                      "long_as_cross": m["lookalikes_read_as_cross"]["ー"], "cross_on_plain": m["cross_on_plain_lines"],
                      "lines": len(refs)}
        res.append(row)
        x, v = row["lines_v7x val"], row["lines_v5 val"]
        print(f"shift {s:4.1f} | v7x: CER {100 * x['cer']:5.2f}% recall {100 * x['recall']:5.1f}% precision {100 * x['precision']:5.1f}% "
              f"ー->x {x['long_as_cross']['as_cross']}/{x['long_as_cross']['ref']} | v5: CER {100 * v['cer']:5.2f}% "
              f"ー->x {v['long_as_cross']['as_cross']}/{v['long_as_cross']['ref']} crosses on {v['cross_on_plain']} lines", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "sweep": res}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
