"""Diagnose slow DataLoader workers on Windows: time the same per-image work in the
main process, in a spawned child, and the cost of handing a batch across processes."""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _work(n):
    import cv2
    import torch
    from hxh_ocr.data import LineDataset, collate, find_vocab
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    ds = LineDataset(["data/synth_v1/train"], find_vocab(["data/synth_v1"]), 32, train=True)
    t = time.time()
    for i in range(n):
        ds[(i * 7919) % len(ds)]
    per_img = (time.time() - t) / n
    batch = collate([ds[i] for i in range(128)])
    t = time.time()
    for _ in range(20):
        collate([ds[i] for i in range(128)])
    return per_img, (time.time() - t) / 20, tuple(batch["images"].shape)


def main():
    print("main process : %.2f ms/img, %.0f ms/batch, batch %s" % ((lambda r: (r[0] * 1e3, r[1] * 1e3, r[2]))(_work(600))), flush=True)
    with mp.get_context("spawn").Pool(1) as pool:
        r = pool.apply(_work, (600,))
    print("spawned child: %.2f ms/img, %.0f ms/batch" % (r[0] * 1e3, r[1] * 1e3), flush=True)


if __name__ == "__main__":
    main()
