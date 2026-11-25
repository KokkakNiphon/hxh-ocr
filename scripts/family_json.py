"""Write the model jsons of a model-family member (experiments/family/<size>/) in the format of
web/models/*.json, next to the .hxhm files, reading sizes from the .hxhm headers.

    python scripts/family_json.py --size tiny --det-name det_tiny_v3 --rec-name ... --rec-v-name ... \
        --vocab data/lines_v5/vocab.json --min-conf 0.8 [--notes notes.json]

notes.json (optional) holds per-file "description" and "tuning" strings.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hxhm_header(path: Path) -> dict:
    b = path.read_bytes()
    assert b[:4] == b"HXHM", path
    n_t, n_o, t_in, t_out = struct.unpack_from("<HHHH", b, 12)
    t_off = struct.unpack_from("<I", b, 24)[0]
    ncls, blank = struct.unpack_from("<HH", b, 36)
    macs = struct.unpack_from("<Q", b, 40)[0]
    flags = struct.unpack_from("<I", b, 48)[0]

    def tensor(i):
        h, w, c = struct.unpack_from("<HHH", b, t_off + 16 * i)
        return {"h": h, "w": w, "c": c}

    return {"input": tensor(t_in), "output": tensor(t_out), "num_classes": ncls, "blank": blank, "macs": macs,
            "raw_output": bool(flags & 2), "bytes": len(b)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True)
    ap.add_argument("--det-name", required=True)
    ap.add_argument("--rec-name", required=True)
    ap.add_argument("--rec-v-name", required=True)
    ap.add_argument("--vocab", required=True, help="vocab.json of the readers' training data")
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--min-show-score", type=float, default=None, help="default: web/models/det.json's")
    ap.add_argument("--params", default=None, help="json of post-processing params (default: det_v2's)")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()
    d = ROOT / "experiments" / "family" / args.size
    notes = json.loads(Path(args.notes).read_text(encoding="utf-8")) if args.notes else {}
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    base = json.loads((ROOT / "web" / "models" / "det.json").read_text(encoding="utf-8"))
    dh = hxhm_header(d / "det.hxhm")
    assert dh["raw_output"], "det.hxhm is not a detector"
    S = dh["input"]["w"]
    det = {
        "name": args.det_name,
        "description": notes.get("det", {}).get("description", ""),
        "file": "det.hxhm",
        "input": {"size": S, "letterbox": f"long side to {S}, top-left, pad 255"},
        "params": json.loads(Path(args.params).read_text()) if args.params else base["params"],
        "pad": base["pad"],
        "tuning": notes.get("det", {}).get("tuning", ""),
        "min_conf": args.min_conf,
        "min_conf_note": "page lines whose mean character confidence is below this are hidden as probably not Hunter script (web/test/page_e2e.mjs --conf-sweep)",
        "min_show_score": args.min_show_score if args.min_show_score is not None else base.get("min_show_score", 0),
        "min_show_score_note": "page lines whose detector score (mean text-kernel probability) is below this are hidden too (web/test/page_e2e.mjs --score-sweep)",
        "macs": dh["macs"],
    }
    (d / "det.json").write_text(json.dumps(det, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, fname, name, vert in (("rec", "model", args.rec_name, False), ("rec_v", "model_v", args.rec_v_name, True)):
        h = hxhm_header(d / f"{fname}.hxhm")
        assert h["num_classes"] == len(vocab) + 1, (fname, h["num_classes"], len(vocab))
        inp = {"height": h["input"]["h"], "width": h["input"]["w"], "channels": 1, "resize": "area",
               "pad": "replicate-right", "overflow": "squeeze"}
        if vert:
            inp["orientation"] = "vertical column rotated 90 deg CCW"
        m = {"name": name, "description": notes.get(key, {}).get("description", ""), "file": f"{fname}.hxhm",
             "input": inp, "classes": h["num_classes"], "blank": h["blank"], "macs": h["macs"], "vocab": vocab}
        (d / f"{fname}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {d}/det.json, model.json, model_v.json (det {S}x{S}, {dh['macs'] / 1e9:.2f} G MACs; vocab {len(vocab)})")


if __name__ == "__main__":
    main()
