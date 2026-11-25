"""Quantize a text-line detector ONNX with ESP-PPQ (ESP32-S3, int8, exactly like the
recogniser), convert the .espdl to .hxhm (raw output, format version 2) and dump golden
vectors from ESP-PPQ's TorchExecutor int8 simulation, for the bit-exact tests of
rust/hxh-infer (host) and rust/hxh-infer-wasm (node).

Detector interface (agreed with the detector side): input 1x1xSxS grayscale u8, letterboxed
(long side scaled to S with area interpolation, placed top-left, padded with 255), normalised
(x/255 - 0.5)/0.5 (folded into the .hxhm input LUT); output ONE int8 tensor [1, C, S/2, S/2]
(channel 0 = text-kernel logit, channel 1 = vertical-orientation logit).

Inputs for calibration and golden vectors: the toy's synthetic pages (default, input 128),
or the images of a directory (--images, e.g. hxh-synth pages), letterboxed by `letterbox_u8`.

Usage (from the repo root, .venv-ppq):
  # toy detector (experiments/det_engine/toy_det.py) -> experiments/det_engine/toy_det + rust test data
  python experiments/det_engine/golden_det.py quantize --onnx experiments/det_engine/models/toy_det.onnx \
      --out-dir experiments/det_engine/toy_det --test-data rust/hxh-infer/tests/data/toy_det
  # the real detector (640 input), calibrated and tested on generated pages
  python experiments/det_engine/golden_det.py quantize --onnx path/det.onnx --images data/pages_v1/val \
      --out-dir experiments/det_engine/det_v1 --calib-n 64 --n 16
  # a detector quantized elsewhere: its .espdl plus the simulated outputs it saved
  # (npz: images u8 [N, S, S] letterboxed inputs, outputs float [N, C, S/2, S/2] from TorchExecutor)
  python experiments/det_engine/golden_det.py from-saved --espdl path/model.espdl --saved path/sim.npz \
      --out-dir experiments/det_engine/det_v1

Output folder: model.espdl, model.hxhm, golden.bin (raw-output records, see rust/FORMAT.md),
golden_inter.bin (intermediate tensors of the first --n-inter vectors), tensors.txt,
espdl_nodes.txt (every .espdl node with its input/output exponents: shows where ESP-PPQ
inserted RequantizeLinear), meta.json (sources, ground-truth quads of synthetic pages).
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import espdl_to_hxhm as cv  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".webp"}


def letterbox_u8(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Long side -> size (INTER_AREA when shrinking, INTER_LINEAR when growing), top-left,
    pad 255. Returns (u8 [size, size], (new_w, new_h))."""
    h, w = img.shape
    s = size / max(h, w)
    # round half up, as rust/hxh-detect letterbox_dims (Python's round() rounds half to even)
    nw, nh = max(1, min(size, int(np.floor(w * s + 0.5)))), max(1, min(size, int(np.floor(h * s + 0.5))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else img
    out = np.full((size, size), 255, np.uint8)
    out[:nh, :nw] = r
    return out, (nw, nh)


def load_inputs(args, size: int, n: int, offset: int):
    """u8 letterboxed inputs + metadata (source, ground-truth lines when synthetic)."""
    imgs, meta = [], []
    if args.images:
        files = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT)[offset : offset + n]
        if len(files) < n:
            raise SystemExit(f"{args.images}: need {n} images after offset {offset}, found {len(files)}")
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            lb, (nw, nh) = letterbox_u8(img, size)
            imgs.append(lb)
            meta.append({"src": str(p), "orig_w": img.shape[1], "orig_h": img.shape[0], "new_w": nw, "new_h": nh})
    else:
        import toy_det

        if size != toy_det.SIZE:
            raise SystemExit(f"synthetic toy pages are {toy_det.SIZE} px; pass --images for a {size} px model")
        rng = np.random.default_rng(1000 + offset)
        for i in range(n):
            img, lines = toy_det.make_page(rng)
            imgs.append(img)
            meta.append({"src": f"toy_det.make_page(seed {1000 + offset}, #{i})", "orig_w": size, "orig_h": size,
                         "new_w": size, "new_h": size, "lines": lines})
    return imgs, meta


def synthetic_edge_cases(size: int):
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size]
    return {
        "black": np.zeros((size, size), np.uint8),
        "white": np.full((size, size), 255, np.uint8),
        "noise": rng.integers(0, 256, (size, size), dtype=np.uint8),
        "checker": (((yy + xx) % 2) * 255).astype(np.uint8),
    }


def espdl_nodes_txt(espdl: Path) -> str:
    """Every node of the .espdl with its activation inputs/outputs and their exponents."""
    m = cv.read_espdl(espdl)
    g = m.graph
    inits = {cv._s(t.name) for t in g.initializer}
    exps = {}
    for v in list(g.input) + list(g.output) + list(g.valueInfo or []):
        e = [int(x) for x in v.exponents] if v.exponents is not None else []
        dims = [int(d.value.dimValue) for d in v.valueInfoType.value.shape.dim]
        exps[cv._s(v.name)] = (dims, e)
    lines = []
    for node in g.node:
        ins = [cv._s(i) for i in node.input if cv._s(i) and cv._s(i) not in inits]
        outs = [cv._s(o) for o in node.output]
        attrs = {cv._s(a.name): cv._attr_val(a) for a in (node.attribute or [])}
        show = {k: v for k, v in attrs.items() if k in ("axis", "mode", "coordinate_transformation_mode", "nearest_mode",
                                                       "activation", "group", "strides")}
        fmt = lambda n: f"{n}{exps.get(n, ([], []))[0]} e{exps.get(n, ([], ['?']))[1]}"  # noqa: E731
        lines.append(f"{cv._s(node.opType):18s} {cv._s(node.name)[:36]:36s} {show}\n"
                     f"    in:  {', '.join(fmt(i) for i in ins)}\n    out: {', '.join(fmt(o) for o in outs)}")
    return "\n".join(lines) + "\n"


def write_golden(out: Path, imgs, q_ins, q_outs, inter_records, n_inter: int):
    H, W = imgs[0].shape
    oh, ow, C = q_outs[0].shape
    T = oh * ow
    rec = 2 * H * W + T * C + 2  # max_ids = 0: raw-output records carry no ids
    # T is u16 in the header; 0 there means "read the u32 at offset 24" (out_h*out_w may exceed 65535)
    hdr = struct.pack("<4sIIHHHHHHI", cv.GOLDEN_MAGIC, 1, len(imgs), H, W, T if T <= 0xFFFF else 0, C, 0, 0, T)
    gb = bytearray(hdr.ljust(32, b"\0"))
    for u8, qi, qo in zip(imgs, q_ins, q_outs):
        r = u8.astype(np.uint8).tobytes() + qi.astype(np.int8).tobytes() + qo.astype(np.int8).tobytes()
        r += struct.pack("<H", 0)
        assert len(r) == rec
        gb += r
    (out / "golden.bin").write_bytes(bytes(gb))
    ib = bytearray(struct.pack("<4sIII", cv.INTER_MAGIC, 1, n_inter, len(inter_records)))
    for vi, t_id, payload in inter_records:
        ib += struct.pack("<IHHI", vi, t_id, 0, len(payload)) + payload
    (out / "golden_inter.bin").write_bytes(bytes(ib))


def finish(args, out: Path, info, imgs, meta, q_ins, q_outs, inter_records):
    T = info["tensors"]
    write_golden(out, imgs, q_ins, q_outs, inter_records, args.n_inter)
    H, W = imgs[0].shape
    with open(out / "sample.pgm", "wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode() + imgs[0].tobytes())
    (out / "tensors.txt").write_text(
        "\n".join(f"t{i}\t{t['h']}x{t['w']}x{t['c']}\texp {t['exp']}\t{t['name']}" for i, t in enumerate(T)) + "\n",
        encoding="utf-8")
    (out / "espdl_nodes.txt").write_text(espdl_nodes_txt(out / "model.espdl"), encoding="utf-8")
    o = T[info["output"]]
    pos = [float((q[..., 0] > 0).mean()) for q in q_outs]
    full = {"input": [H, W], "output_hwc": [o["h"], o["w"], o["c"]], "output_exp": o["exp"],
            "input_exp": T[info["input"]]["exp"], "arena_size": info["arena_size"], "macs": info["macs"],
            "hxhm_bytes": info["file_size"], "n": len(imgs), "n_inter_vectors": args.n_inter,
            "ops": {cv.KIND_NAME[k]: sum(op["kind"] == k for op in info["ops"]) for k in cv.KIND_NAME},
            "share_of_positive_kernel_logits": pos, "vectors": meta}
    (out / "meta.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
    print(f"golden: {len(imgs)} vectors, {len(inter_records)} intermediate tensors -> {out}")
    print(f"ops: {full['ops']}; output {o['h']}x{o['w']}x{o['c']} exp {o['exp']}; "
          f"share of kernel logits > 0 per vector: {[round(p, 3) for p in pos]}")
    if args.test_data:
        td = Path(args.test_data)
        td.mkdir(parents=True, exist_ok=True)
        for f in ("model.hxhm", "golden.bin", "golden_inter.bin", "tensors.txt", "sample.pgm", "meta.json"):
            shutil.copy(out / f, td / f)
        print(f"copied test data to {td}")


def cmd_quantize(args):
    import torch
    from torch.utils.data import DataLoader
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "model_fp32_simplified.onnx"  # espdl_quantize_onnx overwrites its input
    shutil.copy(args.onnx, work)
    import onnx

    shp = [d.dim_value for d in onnx.load(str(work)).graph.input[0].type.tensor_type.shape.dim]
    if len(shp) != 4 or shp[0] != 1 or shp[1] != 1 or shp[2] != shp[3]:
        raise SystemExit(f"expected a 1x1xSxS input, got {shp}")
    S = shp[2]

    calib_u8, _ = load_inputs(args, S, args.calib_n, offset=0)
    calib = np.stack([cv.normalize_u8(u)[None] for u in calib_u8]).astype(np.float32)
    loader = DataLoader(list(torch.from_numpy(calib)), batch_size=args.calib_bs, shuffle=False)
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
    if args.tqt:
        setting.tqt_optimization = True
        t = setting.tqt_optimization_setting
        t.steps, t.lr, t.block_size = args.tqt_steps, args.tqt_lr, args.tqt_block
        t.int_lambda, t.collecting_device = args.tqt_int_lambda, "cpu"
    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(work), espdl_export_file=str(out / "model.espdl"), calib_dataloader=loader,
        calib_steps=len(loader), input_shape=[1, 1, S, S], inputs=[torch.from_numpy(calib[:1])],
        target="esp32s3", quant_type="w8a8", setting=setting, collate_fn=lambda b: b.float(), device="cpu",
        error_report=False, verbose=0,
    )
    print(f"quantized in {time.time() - t0:.1f}s -> {out / 'model.espdl'}")

    in_var = list(graph.inputs.values())[0]
    dest = in_var.dest_ops[0]
    in_cfg = dest.input_quant_config[dest.inputs.index(in_var)]
    in_exp = int(round(np.log2(float(in_cfg.scale))))
    assert float(in_cfg.scale) == 2.0 ** in_exp
    lut = PPQLinearQuant_toInt(torch.from_numpy(cv.normalize_u8(np.arange(256, dtype=np.uint8))), in_cfg).numpy()
    lut = lut.astype(np.int8)
    assert np.array_equal(lut, cv.input_lut_from_formula(in_exp)), "input LUT formula disagrees with PPQ"

    data, info = cv.convert(out / "model.espdl", lut, raw_output=True)
    (out / "model.hxhm").write_bytes(data)
    print(f"wrote {out / 'model.hxhm'}")
    cv.print_info(info)
    T, tid = info["tensors"], info["tid"]

    imgs, meta = load_inputs(args, S, args.n, offset=args.offset)
    if args.synthetic:
        for k, v in synthetic_edge_cases(S).items():
            imgs.append(v)
            meta.append({"src": f"synthetic:{k}", "orig_w": S, "orig_h": S, "new_w": S, "new_h": S})

    ex = TorchExecutor(graph=graph, device="cpu")
    out_var = graph.outputs[list(graph.outputs.keys())[0]]
    out_cfg = out_var.source_op.output_quant_config[out_var.source_op.outputs.index(out_var)]
    out_exp = T[info["output"]]["exp"]
    assert float(out_cfg.scale) == 2.0 ** out_exp, "output exponent mismatch between PPQ graph and .espdl"
    inter_names = [n for n in tid if n in graph.variables and n != list(graph.inputs.keys())[0]]
    q_ins, q_outs, inter_records, not_exact = [], [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for i, u8 in enumerate(imgs):
            x = torch.from_numpy(cv.normalize_u8(u8)[None, None])
            q_in = PPQLinearQuant_toInt(x, in_cfg).numpy().astype(np.int8)[0, 0]
            assert np.array_equal(q_in, lut[u8]), "per-image input quantisation != LUT"
            want = [out_var.name] + (inter_names if i < args.n_inter else [])
            res = ex.forward(inputs=x, output_names=want)
            y = res[0]
            yq = y / out_cfg.scale
            not_exact += int(not torch.equal(yq, torch.round(yq)))
            yi = PPQLinearQuant_toInt(y, out_cfg).numpy().astype(np.int8)  # [1, C, h, w]
            q_ins.append(q_in)
            q_outs.append(np.ascontiguousarray(yi[0].transpose(1, 2, 0)))  # NHWC
            if i < args.n_inter:
                for name, v in zip(want[1:], res[1:]):
                    if v is None:
                        continue
                    t_id = tid[name]
                    e = T[t_id]["exp"]
                    vv = v.numpy()[0].transpose(1, 2, 0)
                    qi = np.clip(np.floor(vv / 2.0 ** e + 0.5), -128, 127).astype(np.int8)
                    inter_records.append((i, t_id, qi.tobytes()))
    print(f"simulated {len(imgs)} inputs in {time.time() - t0:.1f}s; outputs not on the int8 grid: {not_exact}")
    finish(args, out, info, imgs, meta, q_ins, q_outs, inter_records)


def cmd_from_saved(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.espdl).resolve() != (out / "model.espdl").resolve():
        shutil.copy(args.espdl, out / "model.espdl")
    in_exp = int(cv.read_espdl(out / "model.espdl").graph.input[0].exponents[0])
    lut = cv.input_lut_from_formula(in_exp)
    data, info = cv.convert(out / "model.espdl", lut, raw_output=True)
    (out / "model.hxhm").write_bytes(data)
    cv.print_info(info)
    z = np.load(args.saved)
    imgs = [np.asarray(u, np.uint8) for u in z["images"]][: args.n]
    o = info["tensors"][info["output"]]
    q = np.asarray(z["outputs"])[: args.n] / 2.0 ** o["exp"]
    assert np.array_equal(q, np.round(q)), "saved outputs are not on the int8 grid of the .espdl output exponent"
    q_outs = [np.ascontiguousarray(np.clip(v, -128, 127).astype(np.int8).transpose(1, 2, 0)) for v in q]
    q_ins = [lut[u] for u in imgs]
    meta = [{"src": f"{args.saved}#{i}"} for i in range(len(imgs))]
    args.n_inter = 0
    finish(args, out, info, imgs, meta, q_ins, q_outs, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="ONNX -> ESP-PPQ .espdl -> .hxhm + golden vectors")
    q.add_argument("--onnx", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--images", help="directory of page images (default: the toy's synthetic pages)")
    q.add_argument("--calib-n", type=int, default=32)
    q.add_argument("--calib-bs", type=int, default=8)
    q.add_argument("--n", type=int, default=8, help="golden vectors (after --offset images)")
    q.add_argument("--offset", type=int, default=1000, help="skip this many images for the golden vectors")
    q.add_argument("--no-synthetic", dest="synthetic", action="store_false", help="no black/white/noise/checker")
    q.add_argument("--n-inter", type=int, default=2, help="vectors with intermediate tensors dumped")
    q.add_argument("--test-data", help="also copy model + golden files here (rust tests)")
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--tqt", action="store_true")
    q.add_argument("--tqt-steps", type=int, default=500)
    q.add_argument("--tqt-lr", type=float, default=1e-5)
    q.add_argument("--tqt-block", type=int, default=4)
    q.add_argument("--tqt-int-lambda", type=float, default=0.25)
    q.set_defaults(fn=cmd_quantize)
    s = sub.add_parser("from-saved", help=".espdl + saved simulated outputs (npz) -> .hxhm + golden vectors")
    s.add_argument("--espdl", required=True)
    s.add_argument("--saved", required=True, help="npz with images (u8 [N,S,S]) and outputs (float [N,C,S/2,S/2])")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--test-data")
    s.set_defaults(fn=cmd_from_saved)
    args = ap.parse_args()
    try:
        args.fn(args)
    except cv.ConvertError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
