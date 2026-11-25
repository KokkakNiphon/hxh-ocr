"""Check the dataset's pickled size (what each Windows DataLoader worker receives) and
loader throughput for a given worker count.

    python scripts/bench_memory.py --data data/synth_v3 --workers 2
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hxh_ocr.data import LineDataset, find_vocab, make_loader, split_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/synth_v3"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    t = time.time()
    ds = LineDataset(split_dirs(args.data, "train"), find_vocab(args.data), 32, train=True)
    print(f"dataset: {len(ds)} lines, built in {time.time() - t:.1f}s, "
          f"pickled {len(pickle.dumps(ds)) / 1e6:.1f} MB", flush=True)
    x, y, text, i = ds[5]
    print("item:", tuple(x.shape), y.tolist()[:8], repr(text), flush=True)
    dl, _ = make_loader(ds, 128, True, args.workers)
    it = iter(dl)
    for _ in range(3):
        next(it)
    t, n = time.time(), 0
    for _ in range(60):
        n += next(it)["images"].shape[0]
    print(f"workers={args.workers}: {n / (time.time() - t):.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
