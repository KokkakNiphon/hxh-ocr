"""Measure data-loader throughput (images/s) for a few worker counts, without the GPU.

    python scripts/bench_loader.py --data data/synth_v1 data/synth_v2 --workers 2 4 6 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.data import LineDataset, find_vocab, make_loader, split_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/synth_v1"])
    ap.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4, 6, 8])
    ap.add_argument("--batches", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    vocab = find_vocab(args.data)
    ds = LineDataset(split_dirs(args.data, "train"), vocab, 32, train=True)
    for nw in args.workers:
        dl, sampler = make_loader(ds, args.batch_size, True, nw, seed=nw)
        it = iter(dl)
        for _ in range(3):  # warm up workers
            next(it)
        t, n = time.time(), 0
        for _ in range(args.batches):
            n += next(it)["images"].shape[0]
        el = time.time() - t
        print(f"workers={nw}: {n / el:.0f} img/s ({args.batches / el:.1f} batches/s)", flush=True)
        del it, dl


if __name__ == "__main__":
    main()
