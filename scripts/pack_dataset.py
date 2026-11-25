"""Pack a split's JPEG files into one blob so training reads a memory map instead of
opening hundreds of thousands of small files (slow on Windows: every first open costs
a few ms of file-system / antivirus overhead).

    python scripts/pack_dataset.py data/synth_v1 data/synth_v2

Writes <split>/pack.bin (concatenated JPEG bytes, in labels.tsv row order) and
<split>/pack_index.npy (int64 [N, 2] = offset, length). The individual files are kept.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def pack_split(split: Path, threads: int = 16):
    rows = [l.split("\t")[0] for l in (split / "labels.tsv").read_text(encoding="utf-8").splitlines() if l]
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        blobs = list(ex.map(lambda r: (split / r).read_bytes(), rows, chunksize=256))
    def size_of(b):  # header-only parse from memory: no second pass over the files
        with Image.open(BytesIO(b)) as im:
            return im.size  # (w, h)

    with ThreadPoolExecutor(threads) as ex:
        sizes = np.array(list(ex.map(size_of, blobs, chunksize=512)), dtype=np.int32)
    np.save(split / "pack_sizes.npy", sizes)  # (N, 2) w, h for every labels.tsv row
    lens = np.array([len(b) for b in blobs], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    tmp = split / "pack.bin.tmp"
    with open(tmp, "wb") as f:
        for b in blobs:
            f.write(b)
    tmp.replace(split / "pack.bin")
    np.save(split / "pack_index.npy", np.stack([offs, lens], 1))
    print(f"{split}: {len(rows)} images, {lens.sum() / 1e6:.0f} MB, {time.time() - t:.0f}s", flush=True)


def main():
    for d in sys.argv[1:]:
        for split in ("train", "val", "test"):
            s = Path(d) / split
            if (s / "labels.tsv").exists():
                pack_split(s)


if __name__ == "__main__":
    main()
