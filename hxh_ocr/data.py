"""Dataset, collate and width-bucketing sampler for the line recogniser.

Labels are hiragana strings; CTC classes are 75 = blank (0) + 74 vocab tokens
(token i -> class i + 1).  Images are grayscale line crops resized to a fixed
height H, width follows the aspect ratio (capped).
"""
from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

cv2.setNumThreads(0)

BLANK = 0


# ----------------------------------------------------------------------------
# vocabulary
# ----------------------------------------------------------------------------
class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.index = {t: i + 1 for i, t in enumerate(self.tokens)}  # class ids, 0 = blank

    @property
    def num_classes(self) -> int:
        return len(self.tokens) + 1

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in text]

    def decode_ids(self, ids) -> str:
        """Map class ids (no blanks / repeats handling) to a string."""
        return "".join(self.tokens[i - 1] for i in ids if i != BLANK)


def find_vocab(data_dirs) -> Vocab:
    for d in data_dirs:
        p = Path(d) / "vocab.json"
        if p.exists():
            return Vocab.load(p)
    raise FileNotFoundError(f"no vocab.json in {list(data_dirs)}")


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
def split_dirs(data_dirs, split: str) -> list[Path]:
    """Existing `<data_dir>/<split>` dirs that have a finished labels.tsv."""
    out = []
    for d in data_dirs:
        p = Path(d) / split
        if (p / "labels.tsv").exists():
            out.append(p)
        else:
            print(f"[data] skip {p} (no labels.tsv yet)")
    return out


def _read_header_size(path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def _load_sizes(root: Path, relpaths: list[str], cache_name: str = "sizes.npy") -> np.ndarray:
    """(N, 2) array of (w, h) per image, cached in <root>/<cache_name> (keyed by length)."""
    cache = root / cache_name
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(relpaths):
            return arr
    with ThreadPoolExecutor(16) as ex:
        sizes = list(ex.map(_read_header_size, [str(root / r) for r in relpaths], chunksize=256))
    arr = np.asarray(sizes, dtype=np.int32)
    try:
        np.save(cache, arr)
    except OSError:
        pass
    return arr


class LineDataset(Dataset):
    """Reads `labels.tsv` (relpath, label, h|v, style) from one or more split dirs.

    orient: 'h' reads the horizontal lines, 'v' the vertical ones.  Vertical columns are
    rotated 90 deg counter-clockwise on load, so the first glyph (top) ends up on the left
    and a vertical model sees text running left to right like the horizontal one (with
    every glyph turned on its side, which is why it is a separate model).
    """

    def __init__(
        self,
        roots,
        vocab: Vocab,
        height: int = 32,
        max_width: int | None = None,
        train: bool = True,
        orient: str = "h",
        limit: int | None = None,
    ):
        if orient not in ("h", "v"):
            raise ValueError(f"orient must be 'h' or 'v', not {orient!r}")
        self.vocab = vocab
        self.height = height
        self.max_width = max_width or 16 * height
        self.train = train
        self.orient = orient
        paths, labels, widths = [], [], []
        packs, item_pack, item_off, item_len = [], [], [], []
        roots = list(roots)
        per_root = math.ceil(limit / len(roots)) if limit else None  # spread a subset over all roots
        for root in map(Path, roots):
            rel, lab, rows = [], [], []
            with open(root / "labels.tsv", encoding="utf-8") as f:
                for row, line in enumerate(f):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3 or parts[2] != orient:
                        continue
                    rel.append(parts[0]); lab.append(parts[1]); rows.append(row)
            if not rel:
                continue
            if (root / "pack_sizes.npy").exists():  # written by pack_dataset.py for every row
                sizes = np.load(root / "pack_sizes.npy")[np.asarray(rows)]
            else:
                sizes = _load_sizes(root, rel, "sizes.npy" if orient == "h" else f"sizes_{orient}.npy")
            if orient == "v":  # (w, h) of the column -> (w, h) after the 90 deg turn
                sizes = sizes[:, ::-1]
            if per_root:
                rel, lab, sizes, rows = rel[:per_root], lab[:per_root], sizes[:per_root], rows[:per_root]
            paths += [str(root / r) for r in rel]
            labels += lab
            w = np.minimum(np.round(sizes[:, 0] * height / sizes[:, 1]), self.max_width)
            widths.append(w)
            # packed split (scripts/pack_dataset.py): one memory-mapped blob instead of many small files
            if (root / "pack.bin").exists() and (root / "pack_index.npy").exists():
                index = np.load(root / "pack_index.npy")[np.asarray(rows)]
                packs.append(str(root / "pack.bin"))
                item_pack.append(np.full(len(rows), len(packs) - 1, np.int32))
                item_off.append(index[:, 0]); item_len.append(index[:, 1])
            else:
                item_pack.append(np.full(len(rel), -1, np.int32))
                item_off.append(np.zeros(len(rel), np.int64)); item_len.append(np.zeros(len(rel), np.int64))
        # Compact numpy storage, not Python lists: on Windows every DataLoader worker gets its
        # own pickled copy of the dataset, and 3 runs x 4 workers x 350k-item lists exhausted
        # the 72 GB commit limit (error 1455, see journal).
        self.paths = np.array(paths) if paths else np.zeros(0, "<U1")
        self.labels = np.array(labels) if labels else np.zeros(0, "<U1")
        self.packs = packs
        self.item_pack = np.concatenate(item_pack) if item_pack else np.zeros(0, np.int32)
        self.item_off = np.concatenate(item_off) if item_off else np.zeros(0, np.int64)
        self.item_len = np.concatenate(item_len) if item_len else np.zeros(0, np.int64)
        self._mm: dict = {}
        enc = [vocab.encode(s) for s in labels]
        self.tgt_len = np.array([len(t) for t in enc], np.int32)
        self.tgt_off = np.concatenate([[0], np.cumsum(self.tgt_len)[:-1]]).astype(np.int64)
        self.tgt_flat = np.array([c for t in enc for c in t], np.int16)
        del enc
        self.widths = (np.concatenate(widths).astype(np.int32) if widths
                       else np.zeros(0, np.int32))

    def __len__(self):
        return len(self.paths)

    # -- image ---------------------------------------------------------------
    def __getstate__(self):  # don't pickle open memory maps into DataLoader workers
        state = self.__dict__.copy()
        state["_mm"] = {}
        return state

    def _read(self, idx: int) -> np.ndarray | None:
        p = int(self.item_pack[idx])
        if p < 0:
            return cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        mm = self._mm.get(p)
        if mm is None:
            mm = self._mm[p] = np.memmap(self.packs[p], dtype=np.uint8, mode="r")
        off, n = int(self.item_off[idx]), int(self.item_len[idx])
        return cv2.imdecode(np.asarray(mm[off:off + n]), cv2.IMREAD_GRAYSCALE)

    def _load(self, idx: int, rng: random.Random) -> np.ndarray:
        img = self._read(idx)
        if img is None:
            raise RuntimeError(f"cannot read {self.paths[idx]}")
        if self.orient == "v":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        H = self.height
        h, w = img.shape
        if self.train:
            # small random padding in source space (replicated edge)
            pl, pr = (int(rng.uniform(0, 0.25) * h) for _ in range(2))
            pt, pb = (int(rng.uniform(0, 0.06) * h) for _ in range(2))
            if pl or pr or pt or pb:
                img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
                h, w = img.shape
        new_w = w * H / h
        if self.train:
            new_w *= rng.uniform(0.85, 1.15)
        new_w = int(max(8, min(round(new_w), self.max_width)))
        img = cv2.resize(img, (new_w, H), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        if self.train:
            if rng.random() < 0.10:
                x = 1.0 - x
            if rng.random() < 0.5:
                sigma = rng.uniform(0.0, 0.05)
                x = x + np.random.default_rng(rng.getrandbits(32)).normal(0, sigma, x.shape).astype(np.float32)
                x = np.clip(x, 0.0, 1.0)
        return (x - 0.5) / 0.5

    def __getitem__(self, idx):
        if self.train:  # per-worker counter -> different aug each time, reproducible per worker seed
            self._calls = getattr(self, "_calls", 0) + 1
            rng = random.Random(torch.initial_seed() * 1_000_003 + self._calls)
        else:
            rng = random.Random(idx)
        x = torch.from_numpy(self._load(idx, rng))[None]  # [1, H, W]
        o, n = int(self.tgt_off[idx]), int(self.tgt_len[idx])
        y = torch.from_numpy(self.tgt_flat[o:o + n].astype(np.int64))
        return x, y, str(self.labels[idx]), idx


def collate(batch, multiple: int = 8):
    """Pad images to the batch max width (rounded up to `multiple`) with replicated edge."""
    xs, ys, texts, indices = zip(*batch)
    widths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    W = int(math.ceil(int(widths.max()) / multiple) * multiple)
    out = torch.empty(len(xs), 1, xs[0].shape[1], W)
    for i, x in enumerate(xs):
        w = x.shape[-1]
        out[i, :, :, :w] = x
        if w < W:
            out[i, :, :, w:] = x[:, :, -1:]
    tgt_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return {
        "images": out,
        "widths": widths,
        "targets": torch.cat(ys) if len(ys) else torch.zeros(0, dtype=torch.long),
        "target_lengths": tgt_lens,
        "texts": list(texts),
        "indices": list(indices),
    }


# ----------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------
class BucketBatchSampler(Sampler):
    """Batches of similar width: shuffle, cut into mega-chunks, sort each by width,
    cut into batches, shuffle the batches.  Call set_epoch() per epoch."""

    def __init__(self, widths, batch_size: int, shuffle: bool = True, seed: int = 0,
                 chunk_mult: int = 50, drop_last: bool = False):
        self.widths = np.asarray(widths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.chunk = batch_size * chunk_mult
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        n = len(self.widths)
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(n) if self.shuffle else np.arange(n)
        batches = []
        for s in range(0, n, self.chunk):
            c = idx[s:s + self.chunk]
            c = c[np.argsort(self.widths[c], kind="stable")]
            for b in range(0, len(c), self.batch_size):
                batches.append(c[b:b + self.batch_size].tolist())
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        n = len(self.widths)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


def _worker_init(_):
    # one thread per worker: several runs x several workers x 16 OpenCV/torch threads
    # oversubscribe the CPU on Windows
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_loader(dataset: LineDataset, batch_size: int, train: bool, num_workers: int = 0,
                seed: int = 0):
    sampler = BucketBatchSampler(dataset.widths, batch_size, shuffle=train, seed=seed,
                                 drop_last=train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, sampler
