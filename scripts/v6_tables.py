"""Markdown tables for the v6 readers (docs/research/JOURNAL.md, "readers v6"): fp32 line sets
(runs/<run>/eval_<tag>.json from scripts/eval.py), int8 line sets (scripts/eval_lines_int8.py
json files) and the browser pipeline on the page sets (experiments/family/eval/<models>_<set>.txt
from web/test/page_e2e.mjs).

    python scripts/v6_tables.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "family" / "eval"
FP32 = [  # (label, run, tag prefix)
    ("tiny (family, fam_tiny_h48b)", "fam_tiny_h48b", "v6base"),
    ("tiny v6 stage 1", "v6_tiny_h48", "v6"),
    ("tiny v6 stage 2 without lines_v6m (v6b)", "v6b_tiny_h48", "v6"),
    ("**tiny v6** (v6c)", "v6c_tiny_h48", "v6"),
    ("base (family, fam_base_h48)", "fam_base_h48", "v6base"),
    ("base v6 stage 1", "v6_base_h48", "v6"),
    ("base v6 stage 2 without lines_v6m (v6b)", "v6b_base_h48", "v6"),
    ("**base v6** (v6c)", "v6c_base_h48", "v6"),
]
PAGE_SETS = [("pages", "pages_v2 val"), ("pages_v3o", "pages_v3o val"), ("pages_foreign_open", "pages_foreign_open"),
             ("pages_v5", "pages_v5 val"), ("pages_v6", "pages_v6 val (new)")]
PAT = re.compile(r"P ([\d.]+)% R ([\d.]+)% F1 ([\d.]+)%.*CER h ([\d.]+)% v ([\d.]+)% matched ([\d.]+)%.*?exact h ([\d.]+)%.*?"
                 r"([\d.]+) ms/page.*on text ([\d.]+), on icons ([\d.]+), other ([\d.]+)")


def pc(x):
    return f"{100 * x:.2f}%"


def fp32_table():
    print("fp32 checkpoints (scripts/eval.py; synth_v4 = first 3,000 lines of val)\n")
    print("| reader | digit test CER | digit test: digit / punct. char error | digit lines exact | lines_v6 val CER | "
          "lines_v5b val CER | synth_v4 CER | stray digits / punct. on plain lines (v5b; v4) |")
    print("|---|---|---|---|---|---|---|---|")
    for label, run, pre in FP32:
        def j(tag):
            p = ROOT / "runs" / run / f"eval_{pre}_{tag}.json"
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        d, v, b, s = j("digits"), j("val"), j("v5b"), j("v4")
        if not all((d, v, b, s)):
            continue
        print(f"| {label} | {pc(d['cer'])} | {pc(d['digit_char_err'])} | {pc(d['digit_lines_acc'])} | {pc(v['cer'])} | "
              f"{pc(b['cer'])} | {pc(s['cer'])} | {b['plain_lines_with_extra']}/{b['plain_lines']}; "
              f"{s['plain_lines_with_extra']}/{s['plain_lines']} |")


def int8_table(files):
    res = {}
    for f in files:
        p = EVAL / f
        if p.exists():
            for sname, ms in json.loads(p.read_text(encoding="utf-8")).items():
                res.setdefault(sname, {}).update(ms)
    print("\nint8 readers, the app's readCrop (WASM SIMD; scripts/eval_lines_int8.py)\n")
    print("| set | reader | CER | lines exact | lines with digits / punct.: CER | their digit / punct. chars wrong | "
          "stray digits / punct. on plain lines | negatives read |")
    print("|---|---|---|---|---|---|---|---|")
    for sname, ms in res.items():
        for m, x in ms.items():
            dl = f"{pc(x['digit_lines_cer'])} | {pc(x['digit_char_err'])}" if x["digit_lines"] else "- | -"
            neg = f"{100 * x['negatives_read']:.0f}%" if x["negatives"] else "-"
            print(f"| {sname} | {m} | {pc(x['cer'])} | {100 * x['acc']:.1f}% | {dl} | {x['extra_on_plain_lines']}/{x['plain_lines']} | {neg} |")


def page_table(models):
    print("\nbrowser pipeline (web/ocr.js readPage in node, int8), reading confidence 0.8\n")
    print("| set | models | P | R | F1 | CER h | CER v | CER of matched | exact h | false lines / page (text, icons, other) | F1 at score 0.55 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, name in PAGE_SETS:
        for m in models:
            p = EVAL / f"{m}_{s}.txt"
            if not p.exists():
                continue
            row = None
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("min reading confidence 0.8 "):
                    row = PAT.search(line)
            if not row:
                continue
            P, R, F1, ch, cv, cm, eh, ms, ft, fi, fo = row.groups()
            sc = "-"
            q = EVAL / f"{m}_{s}_score.txt"
            if q.exists():
                for line in q.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("conf 0.8, min detector score 0.55"):
                        mm = re.search(r"F1 ([\d.]+)%", line)
                        sc = f"{mm.group(1)}%" if mm else "-"
            print(f"| {name} | {m} | {P}% | {R}% | **{F1}%** | {ch}% | {cv}% | {cm}% | {eh}% | {ft}, {fi}, {fo} | {sc} |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    fp32_table()
    int8_table(["lines_int8_v6.json", "lines_int8_v6b_tiny.json"])
    page_table(["v2tiny", "tiny_v6", "base", "base_v6"])


if __name__ == "__main__":
    main()
