"""Shared helpers for the ESP-PPQ spike: line loading, preprocessing, metrics.

Input convention (matches the planned recogniser): grayscale, 1x1x32x256 NCHW,
normalised to (x/255 - 0.5) / 0.5. HXH_ORIENT=v selects the vertical columns instead of the
horizontal lines; they are rotated 90 deg counter-clockwise like hxh_ocr/data.py does.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the repo root
# Overridable for real models: HXH_DATA (dataset dir with val/), HXH_H / HXH_W (static input size).
DATA = Path(os.environ.get("HXH_DATA", ROOT / "data" / "synth_v1"))
SPIKE = Path(__file__).resolve().parent
H, W = int(os.environ.get("HXH_H", 32)), int(os.environ.get("HXH_W", 256))
ORIENT = os.environ.get("HXH_ORIENT", "h")  # 'h' lines or 'v' columns
NUM_CLASSES = 75  # 74 tokens + CTC blank (index 0)


def load_vocab() -> list[str]:
    """HXH_VOCAB overrides the dataset's vocab.json (extended-vocab readers evaluated on v4 lines)."""
    path = Path(os.environ["HXH_VOCAB"]) if os.environ.get("HXH_VOCAB") else DATA / "vocab.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_labels(split: str) -> list[tuple[str, str, str]]:
    rows = []
    with open(DATA / split / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


def read_gray(split: str, rel: str) -> np.ndarray:
    """One sample as the model sees it before resizing: grayscale, vertical columns turned
    90 deg counter-clockwise so the text runs left to right."""
    img = cv2.imread(str(DATA / split / rel), cv2.IMREAD_GRAYSCALE)
    if ORIENT == "v":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def oriented_rows(split: str) -> list[tuple[str, str, str]]:
    return [r for r in read_labels(split) if r[2] == ORIENT]


def preprocess(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale line (64 px high) -> float32 [1, 32, 256] in [-1, 1].

    Resize to 32 px height keeping aspect; squash to 256 if wider, otherwise pad
    on the right by replicating the last column.
    """
    h, w = img.shape
    new_w = max(1, round(w * H / h))
    img = cv2.resize(img, (min(new_w, W), H), interpolation=cv2.INTER_AREA)
    if img.shape[1] < W:
        img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_REPLICATE)
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    return x[None]


def load_lines(split: str, n: int, offset: int = 0, with_labels: bool = False):
    """First n lines of a split in the ORIENT orientation (after skipping `offset` of them)."""
    rows = oriented_rows(split)[offset : offset + n]
    xs = []
    for rel, _, _ in rows:
        xs.append(preprocess(read_gray(split, rel)))
    xs = np.stack(xs).astype(np.float32)  # [n, 1, 32, 256]
    if with_labels:
        return xs, [r[1] for r in rows]
    return xs


# ---------------------------------------------------------------- metrics


def ctc_greedy(logits_ct: np.ndarray) -> tuple[int, ...]:
    """logits [C, T] -> collapsed label ids (blank = 0)."""
    ids = logits_ct.argmax(axis=0)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(int(i))
        prev = i
    return tuple(out)


def _edit(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def cer(logits: np.ndarray, labels: list[str]) -> float:
    """Greedy-CTC character error rate of logits [N, 75, 1, T] against labels."""
    vocab = load_vocab()
    lg = logits.reshape(logits.shape[0], logits.shape[1], -1)
    errs = tot = 0
    for o, lab in zip(lg, labels):
        hyp = "".join(vocab[i - 1] for i in ctc_greedy(o))
        errs += _edit(hyp, lab)
        tot += len(lab)
    return errs / max(tot, 1)


def compare(ref: np.ndarray, q: np.ndarray) -> dict:
    """ref, q: [N, 75, 1, 64] logits. Returns aggregate metrics."""
    ref = ref.reshape(ref.shape[0], ref.shape[1], -1).astype(np.float64)  # [N, C, T]
    q = q.reshape(q.shape[0], q.shape[1], -1).astype(np.float64)
    n = ref.shape[0]
    cos, snr = [], []
    for i in range(n):
        a, b = ref[i].ravel(), q[i].ravel()
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        snr.append(float(10 * np.log10((a @ a) / (((a - b) ** 2).sum() + 1e-12))))
    am_ref, am_q = ref.argmax(1), q.argmax(1)  # [N, T]
    agree = (am_ref == am_q).mean(1)
    seq_same = [ctc_greedy(ref[i]) == ctc_greedy(q[i]) for i in range(n)]
    # top-1 margin of the fp32 model, to tell confident from near-tie timesteps
    srt = np.sort(ref, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return {
        "n": n,
        "cos_mean": float(np.mean(cos)),
        "cos_min": float(np.min(cos)),
        "snr_db_mean": float(np.mean(snr)),
        "snr_db_min": float(np.min(snr)),
        "argmax_agree_mean": float(agree.mean()),
        "argmax_agree_min": float(agree.min()),
        "timesteps_disagree": int((am_ref != am_q).sum()),
        "timesteps_total": int(am_ref.size),
        "greedy_seq_identical": float(np.mean(seq_same)),
        "fp32_top1_margin_median": float(np.median(margin)),
        "max_abs_err": float(np.abs(ref - q).max()),
    }
