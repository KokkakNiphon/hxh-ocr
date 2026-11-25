"""Build a golden.bin for a model quantized by experiments/esp_ppq_spike/quantize.py with
--save-outputs, without re-running ESP-PPQ: convert its model.espdl to .hxhm and pair the
saved simulated logits (holdout_ppq_int8.npy, float, [N, C, 1, T]) with the same val lines
(offset 2000), preprocessed to u8 exactly like the spike. HXH_ORIENT=v (as for quantize.py)
pairs them with the vertical columns, turned 90 deg counter-clockwise.

Usage (from the repo root, .venv-ppq):
  python experiments/rust_engine/golden_from_saved.py experiments/esp_ppq_spike/out/mnv4s_h32_ptq \
      experiments/rust_engine/mnv4s_h32_ptq [--n 64] [--num-classes 75]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "esp_ppq_spike"))
import espdl_to_hxhm as cv  # noqa: E402
import spike_common as sc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--split", choices=["holdout", "calib"], default="holdout",
                    help="holdout_ppq_int8.npy (val offset 2000) or calib_ppq_int8.npy (val offset 0)")
    ap.add_argument("--num-classes", type=int)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    m = cv.read_espdl(src / "model.espdl")
    in_exp = int(m.graph.input[0].exponents[0])
    data, info = cv.convert(src / "model.espdl", cv.input_lut_from_formula(in_exp), args.num_classes)
    (dst / "model.hxhm").write_bytes(data)
    print(f"wrote {dst / 'model.hxhm'}")
    cv.print_info(info)
    T = info["tensors"]
    out = T[info["output"]]
    ncls = args.num_classes or out["c"]

    offset = 2000 if args.split == "holdout" else 0
    q = np.load(src / f"{args.split}_ppq_int8.npy")[: args.n]  # [N, C, 1, T] float
    qi = q / 2.0 ** out["exp"]
    assert np.array_equal(qi, np.round(qi)), "saved logits are not on the int8 grid"
    qi = np.clip(qi, -128, 127).astype(np.int8)

    # HXH_ORIENT=v: vertical columns, turned 90 deg counter-clockwise (spike_common)
    rows = sc.oriented_rows("val")[offset : offset + len(q)]
    lut = cv.input_lut_from_formula(in_exp)
    H, W = T[info["input"]]["h"], T[info["input"]]["w"]
    Tt, C = out["w"], out["c"]
    rec = 2 * H * W + Tt * C + 2 + 2 * Tt
    gb = bytearray(struct.pack("<4sIIHHHHH", cv.GOLDEN_MAGIC, 1, len(rows), H, W, Tt, C, Tt).ljust(32, b"\0"))
    for (rel, _, _), lg in zip(rows, qi):
        img = sc.read_gray("val", rel)
        u8 = cv.resize_pad_u8(img, H, W)
        assert np.array_equal(cv.normalize_u8(u8)[None], sc.preprocess(img))
        ltc = np.ascontiguousarray(lg[:, 0, :].T)  # [T, C]
        ids = sc.ctc_greedy(ltc[:, :ncls].T.astype(np.int32))
        r = u8.tobytes() + lut[u8].tobytes() + ltc.tobytes() + struct.pack("<H", len(ids))
        r += np.array(list(ids) + [0] * (Tt - len(ids)), "<u2").tobytes()
        assert len(r) == rec
        gb += r
    (dst / "golden.bin").write_bytes(bytes(gb))
    with open(dst / "sample.pgm", "wb") as f:
        img = sc.read_gray("val", rows[0][0])
        f.write(f"P5\n{W} {H}\n255\n".encode() + cv.resize_pad_u8(img, H, W).tobytes())
    print(f"golden: {len(rows)} vectors -> {dst / 'golden.bin'}")


if __name__ == "__main__":
    main()
