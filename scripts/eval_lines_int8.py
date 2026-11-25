"""Int8 readers on line sets, through the app's own code path (scripts/lines_int8.mjs -> web/ocr.js
readCrop, WASM SIMD): CER and line accuracy, the digit / punctuation metrics of scripts/eval.py,
digits or punctuation read on lines without any, and negatives read as text.

    python scripts/eval_lines_int8.py [--models name=model.hxhm,...] [--sets name=labels.tsv:n,...] [--out x.json]

Defaults: the deployed Fast and Accurate horizontal readers and the v6 ones on lines_v6_digits val,
lines_v6 val, lines_v5b val and the first 3,000 lines of synth_v4 val (horizontal lines only).
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr.metrics import align, score  # noqa: E402

XS = set("0123456789!?:/.-")
MODELS = {
    "fast (qat_fam_tiny_h48)": "web/models/model.hxhm",
    "tiny_v6 (qat_v6c_tiny_h48)": "experiments/family/tiny_v6/model.hxhm",
    "accurate (qat_fam_base_h48)": "web/models/accurate/model.hxhm",
    "base_v6 (qat_v6b_base_h48)": "experiments/family/base_v6/model.hxhm",
}
SETS = {
    "lines_v6_digits val": "data/lines_v6_digits/val/labels.tsv:0",
    "lines_v6 val": "data/lines_v6/val/labels.tsv:0",
    "lines_v5b val": "data/lines_v5b/val/labels.tsv:0",
    "synth_v4 val (3000)": "data/synth_v4/val/labels.tsv:3000",
}


def load(tsv: Path, n: int):
    rows = [l.split("\t") for l in tsv.read_text(encoding="utf-8").splitlines() if l.count("\t") >= 2]
    rows = [r for r in rows if r[2] == "h"]
    return rows[:n] if n else rows


def write_bin(tsv: Path, rows, path: Path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(rows)))
        for r in rows:
            img = cv2.imread(str(tsv.parent / r[0]), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            f.write(struct.pack("<II", w, h))
            f.write(img.tobytes())


def metrics(refs, hyps, confs, min_conf=0.8):
    pos = [i for i, r in enumerate(refs) if r]
    neg = [i for i, r in enumerate(refs) if not r]
    cer, acc = score([refs[i] for i in pos], [hyps[i] for i in pos])
    plain = [i for i in pos if not set(refs[i]) & XS]
    x_chars = sum(sum(c in XS for c in hyps[i]) for i in plain)
    x_lines = sum(1 for i in plain if set(hyps[i]) & XS)
    xl = [i for i in pos if set(refs[i]) & XS]
    x_cer, x_acc = score([refs[i] for i in xl], [hyps[i] for i in xl]) if xl else (0.0, 0.0)
    x_ref = x_err = 0
    for i in xl:
        for a, b in align(refs[i], hyps[i])[1]:
            if a is not None and a in XS:
                x_ref += 1
                x_err += a != b
            elif a is None and b in XS:
                x_err += 1
    neg_read = sum(1 for i in neg if hyps[i].strip()) / max(1, len(neg))
    neg_shown = sum(1 for i in neg if hyps[i].strip() and confs[i] >= min_conf) / max(1, len(neg))
    return {"lines": len(pos), "cer": cer, "acc": acc, "digit_lines": len(xl), "digit_lines_cer": x_cer,
            "digit_lines_acc": x_acc, "digit_chars": x_ref, "digit_char_err": x_err / max(1, x_ref),
            "plain_lines": len(plain), "plain_chars": sum(len(refs[i]) for i in plain),
            "extra_on_plain_chars": x_chars, "extra_on_plain_lines": x_lines,
            "negatives": len(neg), "negatives_read": neg_read, "negatives_shown_at_0.8": neg_shown}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="name=path.hxhm,... (json next to it, else web/models/model.json)")
    ap.add_argument("--sets", default=None, help="name=labels.tsv:n,...")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    models = dict(x.split("=", 1) for x in a.models.split(",")) if a.models else MODELS
    sets = dict(x.split("=", 1) for x in a.sets.split(",")) if a.sets else SETS
    res: dict = {}
    with tempfile.TemporaryDirectory() as td:
        jobs = []
        for sname, spec in sets.items():
            tsv, n = spec.rsplit(":", 1)
            tsv = ROOT / tsv
            rows = load(tsv, int(n))
            binp = Path(td) / f"{len(jobs)}.bin"
            write_bin(tsv, rows, binp)
            for mname, mpath in models.items():
                mp = ROOT / mpath
                js = mp.with_suffix(".json") if mp.with_suffix(".json").exists() else ROOT / "web/models/model.json"
                jobs.append((sname, mname, rows, binp, mp, js))

        def run(job):
            sname, mname, rows, binp, mp, js = job
            p = subprocess.run(["node", str(ROOT / "scripts" / "lines_int8.mjs"), str(binp), str(mp), str(js)],
                               capture_output=True, text=True, encoding="utf-8", check=True)
            out = json.loads(p.stdout)
            m = metrics([r[1] for r in rows], [o[0] for o in out], [o[1] for o in out])
            return sname, mname, m

        with ThreadPoolExecutor(a.jobs) as ex:
            for sname, mname, m in ex.map(run, jobs):
                res.setdefault(sname, {})[mname] = m
                print(f"{sname:22s} {mname:28s} CER {100 * m['cer']:5.2f}%  acc {100 * m['acc']:5.1f}%  | digit lines "
                      f"{m['digit_lines']:5d} CER {100 * m['digit_lines_cer']:5.2f}% acc {100 * m['digit_lines_acc']:5.1f}% "
                      f"char err {100 * m['digit_char_err']:5.2f}% | false digits {m['extra_on_plain_chars']} chars on "
                      f"{m['extra_on_plain_lines']}/{m['plain_lines']} plain lines | neg read {100 * m['negatives_read']:.0f}%",
                      flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
