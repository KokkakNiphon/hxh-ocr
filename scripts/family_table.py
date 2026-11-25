"""Summarise experiments/family/eval/*.txt (scripts/family_eval.sh) as markdown tables.

    python scripts/family_table.py [--conf 0.8] [--models current,tiny,base]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments" / "family" / "eval"
SETS = ["pages", "pages_v3", "pages_foreign", "pages_v3o", "pages_foreign_open", "pages_v5"]
NAMES = {"pages": "pages_v2 val", "pages_v3": "pages_v3 val", "pages_foreign": "pages_foreign",
         "pages_v3o": "pages_v3o val (open fonts)", "pages_foreign_open": "pages_foreign_open", "pages_v5": "pages_v5 val (photo-style)"}
PAT = re.compile(r"P ([\d.]+)% R ([\d.]+)% F1 ([\d.]+)%.*CER h ([\d.]+)% v ([\d.]+)% matched ([\d.]+)%.*?"
                 r"([\d.]+) ms/page \(detect ([\d.]+), ([\d.]+) ms/line\).*on text ([\d.]+), on icons ([\d.]+), other ([\d.]+)")


def row(path: Path, key: str):
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(key):
            m = PAT.search(line)
            if m:
                return [float(x) for x in m.groups()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="0.8")
    ap.add_argument("--models", default="current,tiny,base")
    a = ap.parse_args()
    models = a.models.split(",")
    print(f"browser pipeline, int8, reading confidence >= {a.conf} (page_e2e.mjs --conf-sweep)\n")
    print("| set | models | P | R | F1 | CER h | CER v | CER of matched | false lines / page (text, icons, other) | ms / page |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in SETS:
        for m in models:
            r = row(EVAL / f"{m}_{s}.txt", f"min reading confidence {a.conf} ")
            if r is None:
                continue
            P, R, F1, ch, cv, cm, ms, msd, msl, ft, fi, fo = r
            print(f"| {NAMES[s]} | {m} | {P:.1f}% | {R:.1f}% | **{F1:.1f}%** | {ch:.1f}% | {cv:.1f}% | {cm:.1f}% | "
                  f"{ft:.2f}, {fi:.2f}, {fo:.2f} | {ms:.0f} |")
    print("\nwith the detector-score filter (reading confidence 0.8 and detector score >= 0.55, as the app):\n")
    print("| set | models | F1 at score 0 | F1 at 0.55 | F1 at 0.6 |")
    print("|---|---|---|---|---|")
    for s in SETS:
        for m in models:
            p = EVAL / f"{m}_{s}_score.txt"
            vals = [row(p, f"conf 0.8, min detector score {t}") for t in ("0 ", "0.55", "0.6")]
            if all(v is None for v in vals):
                continue
            print(f"| {NAMES[s]} | {m} | " + " | ".join(f"{v[2]:.1f}%" if v else "-" for v in vals) + " |")


if __name__ == "__main__":
    main()
