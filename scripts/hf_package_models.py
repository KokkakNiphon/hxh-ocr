"""Collect the released model family into release/hf-model/ for the Hugging Face Hub (nothing is uploaded).

Three sets, each a line detector plus a horizontal and a vertical CTC line reader:

    v4    the first models (75 classes): det_v2, qat_v4_mnv4s_h48, qat_v4_mnv4s_h48_vert.
          The blog and the parked ESP32 work describe these.
    tiny  model family "tiny" (91 classes): det_tiny_v3_qat, qat_fam_tiny_h48, _vert.
    base  model family "base" (91 classes): det_base_v1_qat (960 px), qat_fam_base_h48, _vert.

The browser demo runs two presets of them (presets.json):
    Fast      = v4/det (det_v2) + tiny readers   -> web/models/            (live camera, tiles)
    Accurate  = base detector + base readers     -> web/models/accurate/   (still images, no tiles)

    release/hf-model/
        README.md, LICENSE, NOTICE      model card (oss/hf/model/README.md), Apache-2.0 text and the
                                        NOTICE with the credit every redistribution must keep
        presets.json                    which int8 files the browser runs as Fast and Accurate
        vocab/v4.json, v4.txt           75 CTC classes: 0 = blank, class i = tokens[i - 1]
        vocab/family.json, family.txt   91 classes: the same 74 tokens first, then 0-9 ! ? : / . -
        int8/<set>/                     det.hxhm, det.json, model.hxhm, model.json, model_v.hxhm,
                                        model_v.json (the .json files are the web/models format)
        espdl/<set>/<model>.espdl       the same int8 models as ESP-PPQ wrote them (ESP-DL, esp32s3)
        onnx/<set>/<model>_<HxW>.onnx   fp32, BN-folded, static input: the graphs ESP-PPQ quantised
        pytorch/<set>/<run>.pt          fp32 checkpoints (before QAT) and QAT checkpoints, stripped

Checks (all must pass, or nothing is written in the end):
  * every int8 file equals its experiment output (the run it came from), and the browser's copies
    in web/models and web/models/accurate equal the preset files byte for byte;
  * each reader .espdl converts (scripts/espdl_to_hxhm.py, needs .venv-ppq) to exactly the shipped
    .hxhm; each detector .espdl equals the one its .hxhm was converted from;
  * stripped checkpoints load with torch.load(weights_only=True) and with the repo's loaders and
    give outputs identical to the original checkpoint; QAT readers, QAT detectors and det_v2
    match the shipped ONNX in onnxruntime;
  * reader vocab: checkpoint tokens == model json vocab == vocab/<v4|family>.json, classes match
    the ONNX output;
  * no absolute or private path in any config, and no private path, user name or e-mail address
    in the bytes of any written file.

    python scripts/hf_package_models.py [--out release/hf-model] [--namespace <HF_NAMESPACE>] [--no-check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The v4 readers were deployed in web/models until the Fast / Accurate switch (e211e35); their
# web jsons are taken from the last commit that still had them, after checking that commit's
# .hxhm equals the experiment output.
V4_WEB_REF = "9d33d0d"

# int8 files per set: name in int8/<set>/ -> (source, experiment output it must equal or None)
#   a source "git:<path>" is read from V4_WEB_REF
INT8 = {
    "v4": {
        "det.hxhm": ("web/models/det.hxhm", "experiments/det_engine/det_v2/model.hxhm"),
        "det.json": ("web/models/det.json", None),
        "model.hxhm": ("experiments/rust_engine/qat_v4_mnv4s_h48_ptq/model.hxhm", "git:web/models/model.hxhm"),
        "model.json": ("git:web/models/model.json", None),
        "model_v.hxhm": ("experiments/rust_engine/qat_v4_mnv4s_h48_vert_ptq/model.hxhm", "git:web/models/model_v.hxhm"),
        "model_v.json": ("git:web/models/model_v.json", None),
    },
    "tiny": {
        "det.hxhm": ("experiments/family/tiny/det.hxhm", "experiments/det_engine/det_tiny_v3_qat_ptq/model.hxhm"),
        "det.json": ("experiments/family/tiny/det.json", None),
        "model.hxhm": ("experiments/family/tiny/model.hxhm", "experiments/rust_engine/qat_fam_tiny_h48_ptq/model.hxhm"),
        "model.json": ("experiments/family/tiny/model.json", None),
        "model_v.hxhm": ("experiments/family/tiny/model_v.hxhm", "experiments/rust_engine/qat_fam_tiny_h48_vert_ptq/model.hxhm"),
        "model_v.json": ("experiments/family/tiny/model_v.json", None),
    },
    "base": {
        "det.hxhm": ("experiments/family/base/det.hxhm", "experiments/det_engine/det_base_v1_qat_ptq/model.hxhm"),
        "det.json": ("experiments/family/base/det.json", None),
        "model.hxhm": ("experiments/family/base/model.hxhm", "experiments/rust_engine/qat_fam_base_h48_ptq/model.hxhm"),
        "model.json": ("experiments/family/base/model.json", None),
        "model_v.hxhm": ("experiments/family/base/model_v.hxhm", "experiments/rust_engine/qat_fam_base_h48_vert_ptq/model.hxhm"),
        "model_v.json": ("experiments/family/base/model_v.json", None),
    },
}
VOCAB_OF = {"v4": "v4", "tiny": "family", "base": "family"}

# What the browser runs: preset -> (web folder, {file in that folder: file in the package})
PRESETS = {
    "fast": ("web/models", {f: f"int8/{'v4' if f.startswith('det') else 'tiny'}/{f}" for f in INT8["v4"]}),
    "accurate": ("web/models/accurate", {f: f"int8/base/{f}" for f in INT8["base"]}),
}
PRESET_NOTES = {
    "fast": {"label": "Fast", "detector": "det_v2 (640)", "readers": "qat_fam_tiny_h48, qat_fam_tiny_h48_vert",
             "tiles": True, "use": "the live camera, phones, and still images when speed matters; "
                                   "full-page pass plus tiles and a second look in the region of the text"},
    "accurate": {"label": "Accurate", "detector": "det_base_v1_qat (960)", "readers": "qat_fam_base_h48, qat_fam_base_h48_vert",
                 "tiles": False, "use": "still images (photos, screenshots, scans): one full-page pass at 960 px, no tiles"},
}

# ESP-PPQ outputs: set -> {packaged name: (.espdl, fp32 ONNX, int8 file it belongs to, kind)}
_PPQ = "experiments/esp_ppq_spike/out"
_DET = "experiments/det"
MODELS = {
    "v4": {
        "det_v2": (f"{_DET}/det_v2_tqt", "det.hxhm", "experiments/det_engine/det_v2/model.espdl"),
        "qat_v4_mnv4s_h48": (f"{_PPQ}/qat_v4_mnv4s_h48_ptq", "model.hxhm", None),
        "qat_v4_mnv4s_h48_vert": (f"{_PPQ}/qat_v4_mnv4s_h48_vert_ptq", "model_v.hxhm", None),
    },
    "tiny": {
        "det_tiny_v3_qat": (f"{_DET}/det_tiny_v3_qat_ptq", "det.hxhm", "experiments/det_engine/det_tiny_v3_qat_ptq/model.espdl"),
        "qat_fam_tiny_h48": (f"{_PPQ}/qat_fam_tiny_h48_ptq", "model.hxhm", None),
        "qat_fam_tiny_h48_vert": (f"{_PPQ}/qat_fam_tiny_h48_vert_ptq", "model_v.hxhm", None),
    },
    "base": {
        "det_base_v1_qat": (f"{_DET}/det_base_v1_qat_ptq", "det.hxhm", "experiments/det_engine/det_base_v1_qat_ptq/model.espdl"),
        "qat_fam_base_h48": (f"{_PPQ}/qat_fam_base_h48_ptq", "model.hxhm", None),
        "qat_fam_base_h48_vert": (f"{_PPQ}/qat_fam_base_h48_vert_ptq", "model_v.hxhm", None),
    },
}

# checkpoints: set -> {name: (run checkpoint, kind, ONNX model name to compare with or None, json)}
#   kind: reader | det (fp32, BN unfolded) | det_qat (BN folded, "folded" state dict)
CKPTS = {
    "v4": {
        "det_v2.pt": ("runs/det_v2/best.pt", "det", "det_v2", "det.json"),
        "v4_mnv4s_h48.pt": ("runs/v4_mnv4s_h48/best.pt", "reader", None, "model.json"),
        "qat_v4_mnv4s_h48.pt": ("runs/qat_v4_mnv4s_h48/best.pt", "reader", "qat_v4_mnv4s_h48", "model.json"),
        "v4_mnv4s_h48_vert.pt": ("runs/v4_mnv4s_h48_vert/best.pt", "reader", None, "model_v.json"),
        "qat_v4_mnv4s_h48_vert.pt": ("runs/qat_v4_mnv4s_h48_vert/best.pt", "reader", "qat_v4_mnv4s_h48_vert", "model_v.json"),
    },
    "tiny": {
        "det_tiny_v3.pt": ("runs/det_tiny_v3/best.pt", "det", None, "det.json"),
        "det_tiny_v3_qat.pt": ("runs/det_tiny_v3_qat/best_folded.pt", "det_qat", "det_tiny_v3_qat", "det.json"),
        "fam_tiny_h48b.pt": ("runs/fam_tiny_h48b/best.pt", "reader", None, "model.json"),
        "qat_fam_tiny_h48.pt": ("runs/qat_fam_tiny_h48/best.pt", "reader", "qat_fam_tiny_h48", "model.json"),
        "fam_tiny_h48b_vert.pt": ("runs/fam_tiny_h48b_vert/best.pt", "reader", None, "model_v.json"),
        "qat_fam_tiny_h48_vert.pt": ("runs/qat_fam_tiny_h48_vert/best.pt", "reader", "qat_fam_tiny_h48_vert", "model_v.json"),
    },
    "base": {
        "det_base_v1.pt": ("runs/det_base_v1/best.pt", "det", None, "det.json"),
        "det_base_v1_qat.pt": ("runs/det_base_v1_qat/best_folded.pt", "det_qat", "det_base_v1_qat", "det.json"),
        "fam_base_h48.pt": ("runs/fam_base_h48/best.pt", "reader", None, "model.json"),
        "qat_fam_base_h48.pt": ("runs/qat_fam_base_h48/best.pt", "reader", "qat_fam_base_h48", "model.json"),
        "fam_base_h48_vert.pt": ("runs/fam_base_h48_vert/best.pt", "reader", None, "model_v.json"),
        "qat_fam_base_h48_vert.pt": ("runs/qat_fam_base_h48_vert/best.pt", "reader", "qat_fam_base_h48_vert", "model_v.json"),
    },
}
KEEP = ("config", "step", "folded", "ema", "exponents", "best_cer", "best_f1", "int8_f1", "seconds")
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp|root)/|\\\\)|AppData|\.claude")
NOTE = ("HxH OCR (https://github.com/KokkakNiphon/hxh-ocr), Apache-2.0. {what} Optimizer state and raw "
        "weights removed. Load with {loader}.")
LOADERS = {
    "reader": "hxh_ocr.engine.load_checkpoint (BN-folded QAT checkpoints included)",
    "det": "hxh_ocr.det_model.load_det_checkpoint",
    "det_qat": "the BN-folded detector: build_det_model(**config['model_cfg']), thresh = None, "
               "hxh_ocr.model.fuse_bn(model, (128, 128)), load_state_dict(ckpt['folded']) "
               "(scripts/hf_package_models.py load_det_folded)",
}


def git_bytes(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{V4_WEB_REF}:{path}"], check=True,
                          capture_output=True).stdout


def src_bytes(spec: str) -> bytes:
    return git_bytes(spec[4:]) if spec.startswith("git:") else (ROOT / spec).read_bytes()


def check_paths(obj, where="config"):
    """Raise on any absolute / machine-specific path in a nested config."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_paths(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str) and ABS_PATH.search(obj):
        raise ValueError(f"absolute or private path in {where}: {obj!r}")


def private_needles() -> list[tuple[str, re.Pattern]]:
    user = re.escape(Path.home().name)
    sep = r"(?:\\\\|\\|/)"
    return [
        ("repo path", re.compile(re.escape(str(ROOT).replace("\\", "/")).encode(), re.I)),
        ("repo path", re.compile(re.escape(str(ROOT)).encode(), re.I)),
        ("drive path", re.compile(rb"\b[A-Za-z]:(?:\\\\|\\|/)(?:Users|HxH-OCR|Windows)\b", re.I)),
        ("home folder", re.compile(("Users" + sep + user + r"\b").encode(), re.I)),
        ("home folder", re.compile((r"/home/" + user + r"\b").encode(), re.I)),
        ("AppData / scratch", re.compile(rb"AppData|scratch[p]ad|\.claude[\\/]", re.I)),  # [p]: keeps the word out of the source
        ("e-mail", re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|io|ai|dev)\b")),
    ]


def scan_private(out: Path) -> list[str]:
    needles = private_needles()
    hits = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            for name, rx in needles:
                m = rx.search(data)
                if m:
                    hits.append(f"{p.relative_to(out).as_posix()}: {name} ({m.group(0)[:60]!r})")
    return hits


def strip_checkpoint(src: Path, dst: Path, kind: str) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    dropped = sorted(set(ck) - set(KEEP))
    out = {k: ck[k] for k in KEEP if k in ck}
    for k in ("ema", "folded"):  # the weights: EMA state dict, or the folded detector's state dict
        if isinstance(ck.get(k), dict):
            out[k] = {n: v.detach().clone().contiguous() for n, v in ck[k].items()}
    assert "ema" in out or isinstance(out.get("folded"), dict), f"{src}: no weights found"
    out["config"] = json.loads(json.dumps(out["config"], default=str))  # builtins only (weights_only)
    check_paths(out["config"])
    what = "BN-folded weights of the QAT detector." if kind == "det_qat" else "EMA weights only."
    out["note"] = NOTE.format(what=what, loader=LOADERS[kind])
    torch.save(out, dst)
    torch.load(dst, map_location="cpu", weights_only=True)  # plain tensors and builtins only
    return {"source": src.relative_to(ROOT).as_posix(), "dropped": dropped,
            "bytes_before": src.stat().st_size, "bytes_after": dst.stat().st_size}


def load_det_folded(path, device="cpu"):
    """The QAT detector: the BN-folded graph of build_det_model with the threshold branch removed."""
    from hxh_ocr.det_model import build_det_model
    from hxh_ocr.model import fuse_bn

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = cfg.get("model_cfg", {})
    model = build_det_model(pretrained=False, fpn=m.get("fpn", 48), level=m.get("level", 16), mid=m.get("mid", 64),
                            head=tuple(m.get("head", (32, 16))), width=m.get("width", 0.5),
                            head_dw=m.get("head_dw", False))
    model.thresh = None
    fused = fuse_bn(model.eval(), (128, 128)).eval()
    fused.load_state_dict(ck["folded"])
    return fused.to(device).eval(), cfg


@torch.no_grad()
def check_checkpoint(src: Path, dst: Path, kind: str, onnx_path: Path | None) -> dict:
    from hxh_ocr.det_model import load_det_checkpoint
    from hxh_ocr.engine import load_checkpoint

    load = {"reader": load_checkpoint, "det": load_det_checkpoint, "det_qat": load_det_folded}[kind]
    a, cfg = load(src)
    b, _ = load(dst)
    if kind == "reader":
        shape = (1, 1, 48, 768)
    else:
        lb = int(cfg.get("letterbox") or 640)
        shape = (1, 1, lb, lb)
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, shape).astype(np.float32))
    ya, yb = a(x), b(x)
    ya = ya[0] if isinstance(ya, (tuple, list)) else ya
    yb = yb[0] if isinstance(yb, (tuple, list)) else yb
    res = {"identical_to_original": bool(torch.equal(ya, yb)), "output_shape": list(yb.shape)}
    assert res["identical_to_original"], f"{dst.name}: outputs differ from the original checkpoint"
    if onnx_path is not None:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        assert list(inp.shape) == list(shape), f"{onnx_path.name}: input {inp.shape} != {shape}"
        if kind == "det":  # the ONNX is the BN-folded export; compare with the exported channels
            from hxh_ocr.model import fuse_bn

            b.thresh = None  # training-only branch, not exported (as in scripts/export_det.py)
            b = fuse_bn(b, (128, 128)).eval()
        yo = sess.run(None, {inp.name: x.numpy()})[0]
        yt = b(x)
        yt = (yt[0] if isinstance(yt, (tuple, list)) else yt).numpy()
        if yt.shape != yo.shape:  # detector training output may carry the threshold branch
            yt = yt[:, : yo.shape[1]]
        diff = float(np.abs(yt - yo).max())
        res["onnx_max_abs_diff"] = diff
        res["onnx_output_shape"] = list(yo.shape)
        assert diff < 1e-3, f"{dst.name}: ONNX differs from the checkpoint by {diff}"
    return res


def check_espdl_reader(espdl: Path, hxhm: Path, num_classes: int) -> bool:
    ppq = ROOT / ".venv-ppq" / "Scripts" / "python.exe"
    if not ppq.exists():
        ppq = ROOT / ".venv-ppq" / "bin" / "python"
    if not ppq.exists():
        print(f"  (no .venv-ppq: skipped the .espdl -> .hxhm reconversion of {espdl.name})")
        return False
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "model.hxhm"
        subprocess.run([str(ppq), str(ROOT / "scripts" / "espdl_to_hxhm.py"), "convert", str(espdl), "-o", str(t),
                        "--num-classes", str(num_classes)], check=True, capture_output=True,
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
        same = t.read_bytes() == hxhm.read_bytes()
    assert same, f"{espdl}: converts to a .hxhm that differs from {hxhm}"
    return True


def onnx_hw(path: Path) -> tuple[int, int]:
    import onnx

    g = onnx.load(str(path), load_external_data=False)
    d = [x.dim_value for x in g.graph.input[0].type.tensor_type.shape.dim]
    return d[2], d[3]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_vocab(out: Path, name: str, tokens: list[str], note: str):
    (out / "vocab").mkdir(exist_ok=True)
    (out / "vocab" / f"{name}.json").write_text(json.dumps(
        {"blank": 0, "num_classes": len(tokens) + 1, "note": note, "tokens": tokens},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # one class per line (line 0 = blank), as rust/hxh-cli --vocab reads it
    (out / "vocab" / f"{name}.txt").write_text("\n".join(["<blank>"] + tokens) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "release" / "hf-model"))
    ap.add_argument("--namespace", default=None, help="Hugging Face user or org; fills <HF_NAMESPACE> in the card")
    ap.add_argument("--no-check", action="store_true", help="skip the load/forward/ONNX/.espdl checks")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = {"sets": {}, "presets": {}, "files": {}, "checkpoints": {}, "checks": {}}

    # ---- int8 files, each against its experiment output
    for s, files in INT8.items():
        d = out / "int8" / s
        d.mkdir(parents=True)
        for name, (src, exp) in files.items():
            data = src_bytes(src)
            if exp is not None:
                assert data == src_bytes(exp), f"int8/{s}/{name}: {src} != {exp}"
            (d / name).write_bytes(data)
        manifest["sets"][s] = {n: {"source": (f"{V4_WEB_REF}:{src[4:]}" if src.startswith("git:") else src),
                                   "equals": (f"{V4_WEB_REF}:{e[4:]}" if e and e.startswith("git:") else e)}
                               for n, (src, e) in files.items()}

    # ---- vocabularies: identical in every reader json of a set, and per vocab family
    vocabs: dict[str, list[str]] = {}
    for s in INT8:
        vs = [json.loads((out / "int8" / s / f).read_text(encoding="utf-8")) for f in ("model.json", "model_v.json")]
        assert vs[0]["vocab"] == vs[1]["vocab"], f"{s}: horizontal and vertical vocab differ"
        for v in vs:
            assert v["classes"] == len(v["vocab"]) + 1 and v["blank"] == 0, f"{s}: classes / blank"
        key = VOCAB_OF[s]
        if key in vocabs:
            assert vocabs[key] == vs[0]["vocab"], f"{s}: vocab differs from the other {key} set"
        vocabs[key] = vs[0]["vocab"]
    assert vocabs["family"][: len(vocabs["v4"])] == vocabs["v4"], "family vocab must extend the v4 one"
    write_vocab(out, "v4", vocabs["v4"],
                "v4 readers (int8/v4): CTC class 0 is the blank; class i (1..74) is tokens[i - 1]. Labels are "
                "full-size hiragana, ー and space: katakana and small kana are folded in, since they share glyphs.")
    write_vocab(out, "family", vocabs["family"],
                "model family readers (int8/tiny, int8/base): CTC class 0 is the blank; class i (1..90) is "
                "tokens[i - 1]. Classes 1..74 are the v4 tokens with the same ids, then 0-9 ! ? : / . - "
                "(digits and punctuation as they appear inline in Hunter lines).")

    # ---- presets: the browser's copies must be exactly these files
    presets = {}
    for p, (web, files) in PRESETS.items():
        for f, pkg in files.items():
            w = ROOT / web / f
            assert w.exists(), f"{web}/{f} missing (preset {p})"
            assert w.read_bytes() == (out / pkg).read_bytes(), f"{web}/{f} != {pkg} (preset {p})"
        presets[p] = {**PRESET_NOTES[p], "web_folder": web + "/",
                      "files": {f"{web}/{f}": pkg for f, pkg in files.items()},
                      "vocab": "vocab/family.json"}
    (out / "presets.json").write_text(json.dumps(
        {"note": "The two model sets the HxH OCR Reader runs (https://hxh-ocr.kokkakniphon.workers.dev). Copy "
                 "each preset's files to the web folder named here, under the left-hand file names.",
         **presets}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest["presets"] = presets

    # ---- ESP-PPQ outputs and fp32 ONNX
    onnx_names = {}
    for s, models in MODELS.items():
        (out / "espdl" / s).mkdir(parents=True)
        (out / "onnx" / s).mkdir(parents=True)
        for name, (d, int8_name, engine_espdl) in models.items():
            espdl = ROOT / d / "model.espdl"
            onnx_src = ROOT / d / "model_fp32_simplified.onnx"
            h, w = onnx_hw(onnx_src)
            onnx_dst = out / "onnx" / s / f"{name}_{h}x{w}.onnx"
            shutil.copy2(espdl, out / "espdl" / s / f"{name}.espdl")
            shutil.copy2(onnx_src, onnx_dst)
            onnx_names[name] = onnx_dst
            chk = {}
            if engine_espdl is not None:  # detector: the .hxhm was converted from this .espdl
                assert espdl.read_bytes() == (ROOT / engine_espdl).read_bytes(), f"{d}: .espdl != {engine_espdl}"
                chk["espdl_equals"] = engine_espdl
            elif not args.no_check:
                nc = json.loads((out / "int8" / s / int8_name.replace(".hxhm", ".json")).read_text(encoding="utf-8"))["classes"]
                chk["espdl_converts_to_hxhm"] = check_espdl_reader(espdl, out / "int8" / s / int8_name, nc)
            manifest["checks"][f"{s}/{name}"] = chk
            print(f"{s}/{name}: espdl {espdl.stat().st_size / 1e3:.0f} KB, onnx {h}x{w} "
                  f"{onnx_src.stat().st_size / 1e6:.2f} MB {chk}", flush=True)

    # ---- checkpoints
    for s, ckpts in CKPTS.items():
        (out / "pytorch" / s).mkdir(parents=True)
        for name, (rel, kind, onnx_name, json_name) in ckpts.items():
            src, dst = ROOT / rel, out / "pytorch" / s / name
            info = strip_checkpoint(src, dst, kind)
            cfg = torch.load(dst, map_location="cpu", weights_only=True)["config"]
            if kind == "reader":
                tokens = json.loads((out / "int8" / s / json_name).read_text(encoding="utf-8"))["vocab"]
                assert cfg.get("tokens") == tokens, f"{s}/{name}: checkpoint tokens differ from int8/{s}/{json_name}"
                assert cfg["num_classes"] == len(tokens) + 1, f"{s}/{name}: num_classes"
            if not args.no_check:
                info.update(check_checkpoint(src, dst, kind, onnx_names[onnx_name] if onnx_name else None))
                if kind == "reader" and "onnx_output_shape" in info:
                    assert info["onnx_output_shape"][1] == cfg["num_classes"], f"{s}/{name}: ONNX classes"
            manifest["checkpoints"][f"{s}/{name}"] = info
            print(f"{s}/{name}: {info['bytes_before'] / 1e6:.1f} MB -> {info['bytes_after'] / 1e6:.1f} MB, "
                  f"dropped {info['dropped']}" + (f", ONNX diff {info['onnx_max_abs_diff']:.2e}"
                                                   if "onnx_max_abs_diff" in info else ""), flush=True)

    # ---- card and licence
    card = ROOT / "oss" / "hf" / "model" / "README.md"
    if card.exists():
        text = card.read_text(encoding="utf-8")
        if args.namespace:
            text = text.replace("<HF_NAMESPACE>", args.namespace)
        (out / "README.md").write_text(text, encoding="utf-8")
    else:
        print("(no model card yet: oss/hf/model/README.md)")
    lic = ROOT / "oss" / "LICENSE"
    if lic.exists():
        shutil.copy2(lic, out / "LICENSE")
    # Apache-2.0 4(d) binds the credit in NOTICE only if the NOTICE ships with the weights
    notice = ROOT / "oss" / "NOTICE"
    assert notice.exists(), "oss/NOTICE is missing: the model package must carry it"
    shutil.copy2(notice, out / "NOTICE")

    # ---- no private strings anywhere in the package
    hits = scan_private(out)
    assert not hits, "private strings in the package:\n  " + "\n  ".join(hits)
    manifest["checks"]["private_strings"] = "none"

    for p in sorted(out.rglob("*")):
        if p.is_file():
            manifest["files"][p.relative_to(out).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    total = sum(f["bytes"] for f in manifest["files"].values())
    manifest["total_bytes"] = total
    log = ROOT / "release" / "logs" / "hf_model_manifest.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(manifest['files'])} files, {total / 1e6:.1f} MB -> {out.relative_to(ROOT)}; manifest {log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
