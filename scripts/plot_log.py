"""Plot training curves (train loss, val CER, val accuracy) from one or more runs' log.csv.

  python scripts/plot_log.py runs/mnv4s_v1 runs/tiny_v1 --out runs/compare.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run dirs (each containing log.csv)")
    p.add_argument("--out", default=None, help="PNG path (default <first run>/curves.png)")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in args.runs:
        d = read(Path(run) / "log.csv")
        if not d:
            continue
        label = Path(run).name
        axes[0].plot(d["step"], d["train_loss"], marker="o", ms=3, label=label)
        axes[1].plot(d["step"], d["val_cer"], marker="o", ms=3, label=label)
        axes[2].plot(d["step"], d["val_acc"], marker="o", ms=3, label=label)
    for ax, title in zip(axes, ["train CTC loss", "val CER (lower is better)", "val line accuracy"]):
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend()
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.runs[0]) / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
