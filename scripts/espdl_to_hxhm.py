"""Convert an ESP-PPQ `.espdl` (ESP32-S3, int8) into the lean `.hxhm` format read by
`rust/hxh-infer`, and dump golden test vectors from ESP-PPQ's int8 simulation.

The format is documented in `rust/FORMAT.md`. In short: a 64-byte header, a tensor
table (NHWC shapes, exponents, precomputed arena offsets), an op table (fixed 32-byte
records), then 16-byte-aligned blobs: int8 weights packed as (N/16)HWC16 with the output
channels zero-padded to a multiple of 16, int32 biases (padded the same way), 256-entry
LUTs and a u8-pixel -> int8-input table that folds the input normalisation in.

Run with .venv-ppq\\Scripts\\python.exe (needs esp-ppq for both commands).

  # parse only (no PPQ graph needed): .espdl -> .hxhm
  python scripts/espdl_to_hxhm.py convert experiments/esp_ppq_spike/out/trained_avg_op18_tqt/model.espdl \
      -o experiments/rust_engine/toy.hxhm

  # quantize an ONNX exactly like experiments/esp_ppq_spike/quantize.py, export the .espdl,
  # convert it, and dump golden vectors (real val lines + a few synthetic images) from
  # ESP-PPQ's TorchExecutor simulation, plus intermediate tensors for debugging
  python scripts/espdl_to_hxhm.py golden --onnx experiments/esp_ppq_spike/models/toy_trained_avg_op18.onnx \
      --tqt --out-dir experiments/rust_engine/toy_tqt --test-data rust/hxh-infer/tests/data/toy_tqt \
      --compare-espdl experiments/esp_ppq_spike/out/trained_avg_op18_tqt/model.espdl

  # print a .hxhm
  python scripts/espdl_to_hxhm.py dump experiments/rust_engine/toy_tqt/model.hxhm

ESP-DL semantics the converter enforces (see FORMAT.md, "Semantics"):
  * per-tensor power-of-two exponents, int8 activations/weights, int32 biases at
    exponent in_exp + w_exp (checked);
  * Add inputs and output must share one exponent: ESP-DL's int8 Add is a plain saturating
    add that ignores exponents. ESP-PPQ guarantees this with "Align to Output" plus
    RequantizeLinear nodes; anything else is rejected rather than guessed;
  * the same holds for Concat (ESP-DL copies bytes, `dl_module_concat.hpp`) and nearest
    Resize (ESP-DL computes round(x * 2^(in_exp - out_exp)), a copy when they are equal;
    ESP-PPQ's espdl setting aligns both to the output: align_concat_to / align_resize_to =
    "Align to Output"). Resize must be nearest with integer factors for which ESP-DL's
    float source index int(y * (1/s)) equals y // s;
  * only the S3/P4 packed filter layouts (N/16)HWC16[_UNALIGNED] are accepted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / "experiments" / "esp_ppq_spike"

MAGIC = b"HXHM"
VERSION = 2  # newest; files that use no version-2 feature are still written as version 1
VERSION_V1 = 1
HEADER_SIZE = 64
TENSOR_REC = 16
OP_REC = 40
HEADER_FMT = "<4sHHIHHHHIIIIHHQI"  # 52 bytes used, 12 reserved
HDR_INPUT_CLOBBERED = 1  # header flags bit: run() overwrites the input tensor
HDR_RAW_OUTPUT = 2  # header flags bit (v2): the output is a raw tensor (detector maps), not CTC logits
TENSOR_FMT = "<HHHbBII"  # 16 bytes
OP_FMT = "<BBHHHHHHHBBBBbbHIIII"  # 40 bytes
ALIGN = 16
NONE16 = 0xFFFF

K_CONV, K_DWCONV, K_ADD, K_LUT, K_REQUANT, K_AVGPOOL, K_RESIZE, K_CONCAT = 1, 2, 3, 4, 5, 6, 7, 8
KIND_NAME = {K_CONV: "Conv", K_DWCONV: "DwConv", K_ADD: "Add", K_LUT: "Lut", K_REQUANT: "Requant", K_AVGPOOL: "AvgPool",
             K_RESIZE: "Resize", K_CONCAT: "Concat"}
V2_KINDS = {K_RESIZE, K_CONCAT}
MAX_CONCAT_INPUTS = 64
FLAG_RELU = 1
FLAG_REDUCE_MEAN = 2

DT_FLOAT, DT_INT8, DT_INT32, DT_INT64 = 1, 3, 6, 7
NP_DT = {DT_FLOAT: "<f4", DT_INT8: "i1", DT_INT32: "<i4", DT_INT64: "<i8"}

GOLDEN_MAGIC = b"HXHG"
INTER_MAGIC = b"HXHI"


class ConvertError(Exception):
    pass


def _s(b) -> str:
    return b.decode() if isinstance(b, bytes) else str(b)


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# ------------------------------------------------------------------ .espdl reading


def read_espdl(path: Path):
    from esp_ppq.parser.espdl.FlatBuffers.Dl.Model import ModelT

    data = Path(path).read_bytes()
    magic = data[:4]
    enc, length = struct.unpack("<II", data[4:12])
    if magic != b"EDL2":
        raise ConvertError(f"{path}: not an .espdl (magic {magic!r})")
    if enc:
        raise ConvertError(f"{path}: encrypted .espdl is not supported")
    if length != len(data) - 16:
        raise ConvertError(f"{path}: payload length {length} != file size - 16")
    return ModelT.InitFromPackedBuf(data[16:], 0)


def _attr_val(a):
    t = a.attrType
    if t == 1:
        return float(a.f.f)
    if t == 2:
        return int(a.i.i)
    if t == 3:
        return bytes(a.s).decode(errors="replace")
    if t == 6:
        return [float(v) for v in a.floats]
    if t == 7:
        return [int(v) for v in a.ints]
    return None


def _init_array(t) -> np.ndarray:
    count = int(np.prod([int(d) for d in t.dims])) if t.dims is not None and len(t.dims) else 1
    if t.dataType not in NP_DT:
        raise ConvertError(f"initializer {_s(t.name)}: unsupported dtype {t.dataType}")
    dt = np.dtype(NP_DT[t.dataType])
    if t.rawData:
        raw = b"".join(np.asarray(a.bytes).astype(np.uint8).tobytes() for a in t.rawData)
        if len(raw) < count * dt.itemsize:
            raise ConvertError(f"initializer {_s(t.name)}: raw data too short")
        return np.frombuffer(raw[: count * dt.itemsize], dtype=dt).copy()
    for field in ("int32Data", "int64Data", "floatData"):
        v = getattr(t, field)
        if v is not None and len(v):
            return np.asarray(v).astype(dt)[:count]
    raise ConvertError(f"initializer {_s(t.name)}: no data")


def unpack_espdl_filter(raw: np.ndarray, n: int, kh: int, kw: int, c: int, depthwise: bool) -> np.ndarray:
    """Invert esp-ppq's ResetParamLayoutPattern for S8 on S3/P4.

    raw: flat int8. Returns W[n][kh][kw][c] (c = 1 for depthwise), i.e. the ONNX filter
    [n, c, kh, kw] permuted to NHWC.
    """
    a = n // 16 * 16
    per = kh * kw * c
    if raw.size < n * per:
        raise ConvertError("filter raw data too short")
    blk = raw[: a * per].reshape(n // 16, kh, kw, c, 16).transpose(0, 4, 1, 2, 3).reshape(a, kh, kw, c)
    rem = n - a
    if not rem:
        return blk
    tail = raw[a * per : n * per]
    if depthwise:  # NHWC -> CHWN for the unaligned depthwise tail
        tail = tail.reshape(kh, kw, rem).transpose(2, 0, 1).reshape(rem, kh, kw, 1)
    else:  # regular conv tail stays NHWC
        tail = tail.reshape(rem, kh, kw, c)
    return np.concatenate([blk, tail], 0)


def pack_filter(w: np.ndarray, depthwise: bool) -> tuple[bytes, int]:
    """W[n][kh][kw][c] -> hxhm layout: output channels zero-padded to a multiple of 16,
    then (N/16)[kh][kw][c][16] (regular) or (N/16)[kh][kw][16] (depthwise)."""
    n, kh, kw, c = w.shape
    cp = _align(n, 16)
    wp = np.zeros((cp, kh, kw, c), np.int8)
    wp[:n] = w
    packed = wp.reshape(cp // 16, 16, kh, kw, c).transpose(0, 2, 3, 4, 1)
    if depthwise:
        packed = packed.reshape(cp // 16, kh, kw, 16)
    return np.ascontiguousarray(packed).tobytes(), cp


# ------------------------------------------------------------------ conversion


def convert(espdl_path: Path, input_lut: np.ndarray | None = None, num_classes: int | None = None, blank: int = 0,
            reuse_input: bool = False, raw_output: bool = False):
    """Parse an .espdl and build the .hxhm bytes. Returns (bytes, info dict).

    num_classes: classes the decoder considers (default: all output channels; set it when
    the classifier is padded, e.g. 75 of 80). reuse_input: let the arena planner reuse the
    input tensor's memory after its last reader (saves input-size bytes, but every run then
    needs a fresh input; recorded as a header flag). raw_output: the output is not CTC logits
    (e.g. a text detector's maps); recorded as a header flag, decode() then returns nothing.
    The file is version 2 if it uses Resize / Concat or raw_output, else version 1."""
    m = read_espdl(espdl_path)
    g = m.graph

    inits = {_s(t.name): t for t in g.initializer}
    vinfo = {}
    for v in list(g.input) + list(g.output) + list(g.valueInfo or []):
        tt = v.valueInfoType.value
        dims = [int(d.value.dimValue) for d in tt.shape.dim]
        exps = [int(e) for e in v.exponents] if v.exponents is not None else []
        if int(tt.elemType) != DT_INT8:
            raise ConvertError(f"activation {_s(v.name)} is dtype {tt.elemType}, only INT8 is supported")
        if len(exps) != 1:
            raise ConvertError(f"activation {_s(v.name)}: exponents {exps} (need exactly one, per-tensor)")
        vinfo[_s(v.name)] = (dims, exps[0])
    if len(g.input) != 1 or len(g.output) != 1:
        raise ConvertError("exactly one graph input and one output are supported")
    in_name, out_name = _s(g.input[0].name), _s(g.output[0].name)

    tensors: list[dict] = []
    tid: dict[str, int] = {}

    def tensor(name: str) -> int:
        if name in tid:
            return tid[name]
        if name not in vinfo:
            raise ConvertError(f"no value_info for tensor {name}")
        dims, exp = vinfo[name]
        if len(dims) != 4 or dims[0] != 1:
            raise ConvertError(f"tensor {name}: shape {dims}, need NHWC with N=1")
        tid[name] = len(tensors)
        tensors.append({"name": name, "h": dims[1], "w": dims[2], "c": dims[3], "exp": exp})
        return tid[name]

    blobs: list[bytes] = []

    def blob(b: bytes) -> int:
        blobs.append(b)
        return len(blobs) - 1

    def init_exp(t) -> int:
        e = [int(x) for x in t.exponents] if t.exponents is not None else []
        if len(e) != 1:
            raise ConvertError(f"initializer {_s(t.name)}: exponents {e}, need per-tensor")
        return e[0]

    tensor(in_name)
    ops: list[dict] = []
    produced = {in_name}
    for node in g.node:
        typ, name = _s(node.opType), _s(node.name)
        ins, outs = [_s(i) for i in node.input], [_s(o) for o in node.output]
        attrs = {_s(a.name): _attr_val(a) for a in (node.attribute or [])}
        qt = attrs.get("quant_type", "S8")
        if qt != "S8":
            raise ConvertError(f"{name}: quant_type {qt}; only S8 (int8) is supported in v0")
        if len(outs) != 1:
            raise ConvertError(f"{name}: {len(outs)} outputs")
        act_ins = [i for i in ins if i and i not in inits]
        for i in act_ins:
            if i not in produced:
                raise ConvertError(f"{name}: input {i} used before it is produced (graph not topologically sorted?)")
        op = {"name": name, "onnx": typ, "flags": 0, "in0": NONE16, "in1": NONE16, "kh": 0, "kw": 0, "sh": 0, "sw": 0,
              "pads": [0, 0, 0, 0], "shift": 0, "w_exp": 0, "cp": 0, "data": None, "bias": None, "ins": None}

        if typ == "Conv":
            x, y = tensor(ins[0]), tensor(outs[0])
            X, Y = tensors[x], tensors[y]
            wt = inits[ins[1]]
            group = int(attrs.get("group", 1))
            kh, kw = attrs["kernel_shape"]
            sh, sw = attrs.get("strides", [1, 1])
            pads = attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]
            if attrs.get("dilations", [1, 1]) != [1, 1]:
                raise ConvertError(f"{name}: dilation not supported")
            act = attrs.get("activation", "Linear")
            if act not in ("Linear", "Relu"):
                raise ConvertError(f"{name}: fused activation {act} not supported")
            layout = _s(wt.docString or b"")
            if "(N/16)HWC16" not in layout:
                raise ConvertError(f"{name}: filter layout '{layout}', expected the S3 S8 (N/16)HWC16 packing")
            w_exp = init_exp(wt)
            depthwise = group != 1
            if depthwise and not (group == X["c"] == Y["c"]):
                raise ConvertError(f"{name}: group={group} with Cin={X['c']} Cout={Y['c']} (only 1 or depthwise)")
            cin = 1 if depthwise else X["c"]
            wdims = [int(d) for d in wt.dims]
            expect = [kh, kw, X["c"], 1] if depthwise else [kh, kw, cin, Y["c"]]
            if wdims != expect:
                raise ConvertError(f"{name}: filter dims {wdims}, expected {expect}")
            w = unpack_espdl_filter(_init_array(wt), Y["c"], kh, kw, cin, depthwise)
            if len(ins) > 2 and ins[2]:
                bt = inits[ins[2]]
                if bt.dataType != DT_INT32:
                    raise ConvertError(f"{name}: bias dtype {bt.dataType}, expected INT32")
                if init_exp(bt) != X["exp"] + w_exp:
                    raise ConvertError(f"{name}: bias exponent {init_exp(bt)} != in {X['exp']} + w {w_exp}")
                b = _init_array(bt).astype(np.int64)
            else:
                b = np.zeros(Y["c"], np.int64)
            if np.abs(b).max(initial=0) > 2**31 - 1:
                raise ConvertError(f"{name}: bias overflows int32")
            oh = (X["h"] + pads[0] + pads[2] - kh) // sh + 1
            ow = (X["w"] + pads[1] + pads[3] - kw) // sw + 1
            if (oh, ow) != (Y["h"], Y["w"]):
                raise ConvertError(f"{name}: output {Y['h']}x{Y['w']} != computed {oh}x{ow}")
            wbytes, cp = pack_filter(w, depthwise)
            bp = np.zeros(cp, "<i4")
            bp[: Y["c"]] = b
            shift = Y["exp"] - X["exp"] - w_exp
            op.update(kind=K_DWCONV if depthwise else K_CONV, in0=x, out=y, kh=kh, kw=kw, sh=sh, sw=sw, pads=pads,
                      shift=shift, w_exp=w_exp, cp=cp, data=blob(wbytes), bias=blob(bp.tobytes()),
                      flags=FLAG_RELU if act == "Relu" else 0,
                      macs=Y["h"] * Y["w"] * Y["c"] * kh * kw * cin,
                      _w=w, _b=b)

        elif typ == "Add":
            if len(act_ins) != 2 or len(ins) != 2:
                raise ConvertError(f"{name}: Add with constant input not supported")
            a, b_, y = tensor(ins[0]), tensor(ins[1]), tensor(outs[0])
            A, B, Y = tensors[a], tensors[b_], tensors[y]
            if any((T["h"], T["w"], T["c"]) != (Y["h"], Y["w"], Y["c"]) for T in (A, B)):
                raise ConvertError(f"{name}: broadcasting Add not supported")
            if not (A["exp"] == B["exp"] == Y["exp"]):
                raise ConvertError(
                    f"{name}: Add exponents in0={A['exp']} in1={B['exp']} out={Y['exp']} differ; ESP-DL's int8 Add "
                    "ignores exponents, so this graph would not run correctly on the device either")
            op.update(kind=K_ADD, in0=a, in1=b_, out=y)

        elif "lut" in attrs:
            x, y = tensor(ins[0]), tensor(outs[0])
            X, Y = tensors[x], tensors[y]
            if (X["h"], X["w"], X["c"]) != (Y["h"], Y["w"], Y["c"]):
                raise ConvertError(f"{name}: LUT op changes shape")
            lt = inits[attrs["lut"]]
            if lt.dataType != DT_INT8 or [int(d) for d in lt.dims] != [256]:
                raise ConvertError(f"{name}: LUT {attrs['lut']} is not a 256-entry int8 table")
            if init_exp(lt) != Y["exp"]:
                raise ConvertError(f"{name}: LUT exponent {init_exp(lt)} != output exponent {Y['exp']}")
            op.update(kind=K_LUT, in0=x, out=y, data=blob(_init_array(lt).tobytes()), _lut_of=typ)

        elif typ == "RequantizeLinear":
            x, y = tensor(ins[0]), tensor(outs[0])
            X, Y = tensors[x], tensors[y]
            if (X["h"], X["w"], X["c"]) != (Y["h"], Y["w"], Y["c"]):
                raise ConvertError(f"{name}: RequantizeLinear changes shape")
            shift = Y["exp"] - X["exp"]
            if len(ins) > 1 and ins[1] in inits:
                sc = float(_init_array(inits[ins[1]])[0])
                if sc != 2.0 ** shift:
                    raise ConvertError(f"{name}: scale {sc} != 2^(out_exp - in_exp) = {2.0 ** shift}")
            if len(ins) > 2 and ins[2] in inits and int(_init_array(inits[ins[2]])[0]) != 0:
                raise ConvertError(f"{name}: non-zero zero point")
            op.update(kind=K_REQUANT, in0=x, out=y, shift=shift)

        elif typ in ("AveragePool", "ReduceMean", "GlobalAveragePool"):
            x, y = tensor(ins[0]), tensor(outs[0])
            X, Y = tensors[x], tensors[y]
            flags = 0
            if typ == "AveragePool":
                kh, kw = attrs["kernel_shape"]
                sh, sw = attrs.get("strides", [1, 1])
                if any(attrs.get("pads", [0, 0, 0, 0])) or attrs.get("ceil_mode", 0):
                    raise ConvertError(f"{name}: AveragePool padding / ceil_mode not supported")
                if attrs.get("dilations", [1, 1]) != [1, 1]:
                    raise ConvertError(f"{name}: dilation not supported")
            else:
                if typ == "ReduceMean":
                    if len(ins) > 1 and ins[1]:
                        axes = sorted(int(v) for v in _init_array(inits[ins[1]]))
                    else:
                        axes = sorted(attrs.get("axes", []))
                    if attrs.get("keepdims", 1) != 1:
                        raise ConvertError(f"{name}: ReduceMean keepdims=0 not supported")
                    flags = FLAG_REDUCE_MEAN
                else:
                    axes = [1, 2]
                if axes == [1]:
                    kh, kw = X["h"], 1
                elif axes == [1, 2]:
                    kh, kw = X["h"], X["w"]
                else:
                    raise ConvertError(f"{name}: ReduceMean axes {axes} (NHWC) not supported; only H or H,W")
                sh, sw = kh, kw
            oh, ow = (X["h"] - kh) // sh + 1, (X["w"] - kw) // sw + 1
            if (oh, ow, X["c"]) != (Y["h"], Y["w"], Y["c"]):
                raise ConvertError(f"{name}: output {Y['h']}x{Y['w']}x{Y['c']} != computed {oh}x{ow}x{X['c']}")
            op.update(kind=K_AVGPOOL, in0=x, out=y, kh=kh, kw=kw, sh=sh, sw=sw, flags=flags)

        elif typ == "Resize":
            if len(act_ins) != 1:
                raise ConvertError(f"{name}: Resize with {len(act_ins)} activation inputs")
            x, y = tensor(act_ins[0]), tensor(outs[0])
            X, Y = tensors[x], tensors[y]
            mode = attrs.get("mode", "nearest")
            if mode != "nearest":
                raise ConvertError(f"{name}: Resize mode {mode!r}; only nearest is supported")
            if X["c"] != Y["c"] or Y["h"] % X["h"] or Y["w"] % X["w"]:
                raise ConvertError(f"{name}: Resize {X['h']}x{X['w']}x{X['c']} -> {Y['h']}x{Y['w']}x{Y['c']} "
                                   "is not an integer upscale")
            sh, sw = Y["h"] // X["h"], Y["w"] // X["w"]
            # ESP-DL: in_y = min(int(y * (1.f / scale)), in_h - 1) in float32 (PPQ's F.interpolate
            # does the same); accept only factors where that equals y // scale for every y
            for s_, n_out, n_in in ((sh, Y["h"], X["h"]), (sw, Y["w"], X["w"])):
                inv = np.float32(1.0) / np.float32(s_)
                inv2 = np.float32(n_in) / np.float32(n_out)
                idx = np.arange(n_out)
                for iv in (inv, inv2):
                    src = np.minimum((idx.astype(np.float32) * iv).astype(np.int64), n_in - 1)
                    if not np.array_equal(src, idx // s_):
                        raise ConvertError(f"{name}: scale {s_}: ESP-DL's float source index differs from y // {s_}")
            if X["exp"] != Y["exp"]:
                raise ConvertError(
                    f"{name}: Resize exponents in={X['exp']} out={Y['exp']} differ; ESP-PPQ aligns them "
                    "(align_resize_to='Align to Output' + RequantizeLinear), so this graph is unexpected")
            ctm = attrs.get("coordinate_transformation_mode", "half_pixel")
            nm = attrs.get("nearest_mode", "round_prefer_floor")
            if max(sh, sw) > 2 and ctm == "asymmetric" and nm != "floor":
                print(f"warning: {name}: ONNX nearest_mode {nm!r} with asymmetric coordinates differs from ESP-DL / "
                      f"ESP-PPQ (floor) at scale {max(sh, sw)}; the int8 engine follows ESP-DL")
            op.update(kind=K_RESIZE, in0=x, out=y, sh=sh, sw=sw)

        elif typ == "Concat":
            if len(act_ins) != len(ins) or not 2 <= len(ins) <= MAX_CONCAT_INPUTS:
                raise ConvertError(f"{name}: Concat needs 2..{MAX_CONCAT_INPUTS} activation inputs (got {ins})")
            xs, y = [tensor(i) for i in ins], tensor(outs[0])
            Y = tensors[y]
            axis = int(attrs.get("axis", 0))
            if axis < 0:
                axis += 4
            if axis not in (1, 2, 3):  # the .espdl's axis is NHWC (esp-ppq ResetConcatPattern)
                raise ConvertError(f"{name}: Concat axis {attrs.get('axis')} (NHWC) not supported; only H, W or C")

            def dims(T):
                return (T["h"], T["w"], T["c"])

            along = 0
            for i in xs:
                Xi = tensors[i]
                if any(dims(Xi)[d] != dims(Y)[d] for d in range(3) if d != axis - 1):
                    raise ConvertError(f"{name}: Concat input {Xi['name']} shape {dims(Xi)} does not fit {dims(Y)}")
                if Xi["exp"] != Y["exp"]:
                    raise ConvertError(
                        f"{name}: Concat input {Xi['name']} exponent {Xi['exp']} != output {Y['exp']}; ESP-DL's Concat "
                        "copies bytes and ignores exponents (ESP-PPQ aligns them with RequantizeLinear)")
                along += dims(Xi)[axis - 1]
            if along != dims(Y)[axis - 1]:
                raise ConvertError(f"{name}: Concat sizes along axis {axis} sum to {along}, output has {dims(Y)[axis - 1]}")
            op.update(kind=K_CONCAT, in0=xs[0], out=y, kh=len(xs), kw=axis, ins=xs,
                      data=blob(np.asarray(xs, "<u2").tobytes()))

        else:
            raise ConvertError(f"{name}: op type {typ} not supported by hxh-infer")

        ops.append(op)
        produced.add(outs[0])

    if out_name not in tid:
        raise ConvertError("graph output not produced")
    in_t, out_t = tid[in_name], tid[out_name]
    if num_classes is None:
        num_classes = tensors[out_t]["c"]
    if not 0 < num_classes <= tensors[out_t]["c"] or not 0 <= blank < num_classes:
        raise ConvertError(f"num_classes {num_classes} / blank {blank} do not fit {tensors[out_t]['c']} output channels")

    # ---------------- arena planning (first fit, 16-byte aligned, in-place for elementwise ops)
    n_ops = len(ops)
    first = {in_t: -1}
    last = {in_t: -1}
    def op_inputs(op):
        return op["ins"] if op["ins"] is not None else [i for i in (op["in0"], op["in1"]) if i != NONE16]

    for k, op in enumerate(ops):
        for i in op_inputs(op):
            last[i] = k
        first[op["out"]] = k
        last.setdefault(op["out"], k)
    last[out_t] = n_ops
    if not reuse_input:
        last[in_t] = n_ops  # keep the input intact so run() can be repeated on the same input
    for t in tensors:
        t["size"] = t["h"] * t["w"] * t["c"]
    live: dict[int, tuple[int, int]] = {}  # tensor -> (off, end)
    offset = {}

    def place(size: int) -> int:
        cur = 0
        for off, end in sorted(live.values()):
            if off - cur >= size:
                return cur
            cur = max(cur, _align(end))
        return cur

    offset[in_t] = 0
    live[in_t] = (0, tensors[in_t]["size"])
    for k, op in enumerate(ops):
        for t in [t for t in live if last[t] < k]:
            del live[t]
        o = op["out"]
        size = tensors[o]["size"]
        alias = None
        if op["kind"] in (K_ADD, K_LUT, K_REQUANT):
            for i in (op["in0"], op["in1"]):
                if i != NONE16 and last[i] == k and tensors[i]["size"] == size and i != out_t:
                    alias = i
                    break
        if alias is not None:
            offset[o] = offset[alias]
            del live[alias]
        else:
            offset[o] = place(size)
        live[o] = (offset[o], offset[o] + size)
        op["inplace"] = alias is not None
    arena_size = _align(max(offset[t] + tensors[t]["size"] for t in offset))

    # ---------------- serialize
    tensor_table_off = HEADER_SIZE
    op_table_off = _align(tensor_table_off + TENSOR_REC * len(tensors))
    cur = _align(op_table_off + OP_REC * n_ops)
    blob_off = []
    for b in blobs:
        blob_off.append(cur)
        cur = _align(cur + len(b))
    lut_off = 0
    if input_lut is not None:
        lut_off = cur
        cur = _align(cur + 256)
    file_size = cur
    macs = sum(op.get("macs", 0) for op in ops)
    version = VERSION if raw_output or any(op["kind"] in V2_KINDS for op in ops) else VERSION_V1
    hflags = (HDR_INPUT_CLOBBERED if reuse_input else 0) | (HDR_RAW_OUTPUT if raw_output else 0)

    buf = bytearray(file_size)
    struct.pack_into(HEADER_FMT, buf, 0, MAGIC, version, HEADER_SIZE, file_size, len(tensors), n_ops,
                     in_t, out_t, arena_size, tensor_table_off, op_table_off, lut_off, num_classes, blank, macs,
                     hflags)
    for i, t in enumerate(tensors):
        struct.pack_into(TENSOR_FMT, buf, tensor_table_off + TENSOR_REC * i, t["h"], t["w"], t["c"], t["exp"], 0,
                         offset[i], t["size"])
    for k, op in enumerate(ops):
        d_off = blob_off[op["data"]] if op["data"] is not None else 0
        d_len = len(blobs[op["data"]]) if op["data"] is not None else 0
        b_off = blob_off[op["bias"]] if op["bias"] is not None else 0
        p = op["pads"]
        struct.pack_into(OP_FMT, buf, op_table_off + OP_REC * k, op["kind"], op["flags"], op["in0"],
                         op["in1"], op["out"], op["kh"], op["kw"], op["sh"], op["sw"], p[0], p[1], p[2], p[3],
                         op["shift"], op["w_exp"], op["cp"], d_off, d_len, b_off, 0)
    for b, off in zip(blobs, blob_off):
        buf[off : off + len(b)] = b
    if input_lut is not None:
        buf[lut_off : lut_off + 256] = np.asarray(input_lut, np.int8).tobytes()

    info = {"tensors": tensors, "ops": ops, "offset": offset, "arena_size": arena_size, "file_size": file_size,
            "macs": macs, "input": in_t, "output": out_t, "tid": tid, "version": version, "raw_output": raw_output,
            "weight_bytes": sum(len(blobs[o["data"]]) for o in ops if o["kind"] in (K_CONV, K_DWCONV)),
            "bias_bytes": sum(len(blobs[o["bias"]]) for o in ops if o["bias"] is not None)}
    # round trip through the reader as a format self-check
    rt = read_hxhm(bytes(buf))
    assert rt["arena_size"] == arena_size and len(rt["ops"]) == n_ops and len(rt["tensors"]) == len(tensors)
    return bytes(buf), info


# ------------------------------------------------------------------ .hxhm reading (self-check / dump)


def read_hxhm(data: bytes) -> dict:
    (magic, ver, hsz, fsz, nt, no, it, ot, arena, tto, oto, lut, ncls, blank, macs, hflags) = struct.unpack_from(
        HEADER_FMT, data, 0)
    if magic != MAGIC or not VERSION_V1 <= ver <= VERSION or hsz != HEADER_SIZE or fsz != len(data):
        raise ConvertError("bad .hxhm header")
    tensors = [dict(zip(("h", "w", "c", "exp", "flags", "off", "size"), struct.unpack_from(TENSOR_FMT, data, tto + TENSOR_REC * i)))
               for i in range(nt)]
    keys = ("kind", "flags", "in0", "in1", "out", "kh", "kw", "sh", "sw", "pt", "pl", "pb", "pr", "shift", "w_exp", "cp",
            "data_off", "data_len", "bias_off", "reserved")
    ops = [dict(zip(keys, struct.unpack_from(OP_FMT, data, oto + OP_REC * k))) for k in range(no)]
    for o in ops:
        if o["kind"] == K_CONCAT:
            o["ins"] = list(struct.unpack_from(f"<{o['kh']}H", data, o["data_off"]))
    return {"version": ver, "file_size": fsz, "input": it, "output": ot, "arena_size": arena, "input_lut_off": lut,
            "num_classes": ncls, "blank": blank, "macs": macs, "flags": hflags, "tensors": tensors, "ops": ops}


def cmd_dump(args):
    data = Path(args.hxhm).read_bytes()
    h = read_hxhm(data)
    print(f"{args.hxhm}: version {h['version']}, {len(data)} B, arena {h['arena_size']} B, MACs {h['macs']:,}, "
          f"input t{h['input']} output t{h['output']}, classes {h['num_classes']} blank {h['blank']}, "
          f"input LUT @{h['input_lut_off']}, flags {h['flags']:#x}"
          + (" (raw output)" if h["flags"] & HDR_RAW_OUTPUT else ""))
    for i, t in enumerate(h["tensors"]):
        print(f"  t{i:<3d} {t['h']:4d}x{t['w']:<4d}x{t['c']:<4d} exp {t['exp']:3d}  arena [{t['off']:6d}, {t['off'] + t['size']:6d})")
    for k, o in enumerate(h["ops"]):
        extra = ""
        if o["kind"] in (K_CONV, K_DWCONV, K_AVGPOOL):
            extra = f"k{o['kh']}x{o['kw']} s{o['sh']}x{o['sw']} pad[{o['pt']},{o['pl']},{o['pb']},{o['pr']}]"
        if o["kind"] in (K_CONV, K_DWCONV):
            extra += f" shift {o['shift']} w_exp {o['w_exp']} cp {o['cp']} relu {o['flags'] & 1}"
        if o["kind"] == K_REQUANT:
            extra = f"shift {o['shift']}"
        if o["kind"] == K_RESIZE:
            extra = f"nearest x{o['sh']}x{o['sw']}"
        if o["kind"] == K_CONCAT:
            extra = "axis " + "?HWC"[o["kw"]]
            ins = ",".join(f"t{i}" for i in o["ins"])
        else:
            ins = f"t{o['in0']}" + (f",t{o['in1']}" if o["in1"] != NONE16 else "")
        print(f"  op{k:<3d} {KIND_NAME[o['kind']]:8s} {ins:>9s} -> t{o['out']:<3d} {extra}")


def print_info(info: dict) -> None:
    T = info["tensors"]
    print(f"  version {info['version']}{' (raw output)' if info['raw_output'] else ''}, "
          f"{len(info['ops'])} ops, {len(T)} tensors, file {info['file_size']} B "
          f"(weights {info['weight_bytes']} B, biases {info['bias_bytes']} B), arena {info['arena_size']} B, "
          f"MACs {info['macs']:,}")
    for k, op in enumerate(info["ops"]):
        o = T[op["out"]]
        print(f"  op{k:<3d} {KIND_NAME[op['kind']]:8s} {op['name'][:38]:38s} -> {o['h']}x{o['w']}x{o['c']} e{o['exp']}"
              f" @{info['offset'][op['out']]}{' (in place)' if op.get('inplace') else ''}")


def cmd_convert(args):
    lut = None
    if not args.no_input_lut:
        in_exp = int(read_espdl(Path(args.espdl)).graph.input[0].exponents[0])
        lut = input_lut_from_formula(in_exp, args.norm_mean, args.norm_std)
    data, info = convert(Path(args.espdl), lut, args.num_classes, reuse_input=args.reuse_input,
                         raw_output=args.raw_output)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"wrote {out}")
    print_info(info)


# ------------------------------------------------------------------ input normalisation


def normalize_u8(u8: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Exactly spike_common.preprocess's float conversion (numpy float32 throughout)."""
    return (u8.astype(np.float32) / 255.0 - mean) / std


def input_lut_from_formula(exp: int, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """u8 pixel -> int8 model input: PPQ's quantisation (x / 2^exp, round half up, clip)."""
    import torch

    x = torch.from_numpy(normalize_u8(np.arange(256, dtype=np.uint8), mean, std))
    q = torch.clamp(torch.floor(x / (2.0 ** exp) + 0.5), -128, 127)
    return q.numpy().astype(np.int8)


# ------------------------------------------------------------------ golden vectors


def resize_pad_u8(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """spike_common.preprocess without the float normalisation (uint8 in, uint8 out)."""
    import cv2

    h, w = img.shape
    new_w = max(1, round(w * H / h))
    img = cv2.resize(img, (min(new_w, W), H), interpolation=cv2.INTER_AREA)
    if img.shape[1] < W:
        img = cv2.copyMakeBorder(img, 0, 0, 0, W - img.shape[1], cv2.BORDER_REPLICATE)
    return img


def cmd_golden(args):
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(SPIKE))
    import spike_common as sc
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.IR.quantize import QuantableOperation
    from esp_ppq.quantization.qfunction.linear import PPQLinearQuant_toInt

    torch.manual_seed(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "model_fp32_simplified.onnx"  # espdl_quantize_onnx overwrites its input
    shutil.copy(args.onnx, work)

    # ---- quantize exactly like experiments/esp_ppq_spike/quantize.py
    calib = sc.load_lines("val", args.calib_n, offset=0)
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
        calib_steps=len(loader), input_shape=[1, 1, sc.H, sc.W], inputs=[torch.from_numpy(calib[:1])],
        target="esp32s3", quant_type="w8a8", setting=setting, collate_fn=lambda b: b.float(), device="cpu",
        error_report=False, verbose=0,
    )
    print(f"quantized in {time.time() - t0:.1f}s -> {out / 'model.espdl'}")
    if args.compare_espdl:
        same = Path(args.compare_espdl).read_bytes() == (out / "model.espdl").read_bytes()
        print(f"re-quantized .espdl byte-identical to {args.compare_espdl}: {same}")
        if not same:
            a = {_s(t.name): t for t in read_espdl(Path(args.compare_espdl)).graph.initializer}
            b = {_s(t.name): t for t in read_espdl(out / "model.espdl").graph.initializer}
            diffs = []
            for k in sorted(set(a) & set(b)):
                va, vb = _init_array(a[k]), _init_array(b[k])
                ea, eb = list(a[k].exponents), list(b[k].exponents)
                if ea != eb or va.shape != vb.shape or not np.array_equal(va, vb):
                    n = int((va != vb).sum()) if va.shape == vb.shape else -1
                    diffs.append(f"{k} (exp {ea}->{eb}, {n} values differ)")
            print(f"  differing initializers: {len(diffs)} of {len(a)}; names only in one file: "
                  f"{sorted(set(a) ^ set(b))}")
            for d in diffs[:20]:
                print("   ", d)

    # ---- input quantisation config from the PPQ graph -> u8 pixel LUT
    in_var = list(graph.inputs.values())[0]
    dest = in_var.dest_ops[0]
    in_cfg = dest.input_quant_config[dest.inputs.index(in_var)]
    in_exp = int(round(np.log2(float(in_cfg.scale))))
    assert float(in_cfg.scale) == 2.0 ** in_exp
    x_all = torch.from_numpy(normalize_u8(np.arange(256, dtype=np.uint8)))
    lut = PPQLinearQuant_toInt(x_all, in_cfg).numpy().astype(np.int8)
    assert np.array_equal(lut, input_lut_from_formula(in_exp)), "input LUT formula disagrees with PPQ"

    data, info = convert(out / "model.espdl", lut, args.num_classes, reuse_input=args.reuse_input)
    ncls = args.num_classes or info["tensors"][info["output"]]["c"]
    (out / "model.hxhm").write_bytes(data)
    print(f"wrote {out / 'model.hxhm'}")
    print_info(info)
    T, tid = info["tensors"], info["tid"]
    assert T[info["input"]]["exp"] == in_exp

    # ---- cross-check unpacked weights/biases against the PPQ graph's own quantisation
    n_checked = 0
    for op in info["ops"]:
        if op["kind"] not in (K_CONV, K_DWCONV):
            continue
        pop = graph.operations[op["name"]]
        assert isinstance(pop, QuantableOperation)
        wq = PPQLinearQuant_toInt(pop.inputs[1].value, pop.config.input_quantization_config[1]).numpy()  # [n,c,kh,kw]
        if not np.array_equal(wq.transpose(0, 2, 3, 1), op["_w"]):
            raise ConvertError(f"{op['name']}: unpacked .espdl filter != PPQ quantized weights")
        if len(pop.inputs) > 2:
            bq = PPQLinearQuant_toInt(pop.inputs[2].value, pop.config.input_quantization_config[2]).numpy()
            if not np.array_equal(bq.astype(np.int64), op["_b"]):
                raise ConvertError(f"{op['name']}: .espdl bias != PPQ quantized bias")
        n_checked += 1
    print(f"filters/biases of {n_checked} convs match the PPQ graph's quantized parameters")

    # ---- inputs: held-out real val lines + synthetic edge cases
    # HXH_ORIENT=v: vertical columns, turned 90 deg counter-clockwise (spike_common)
    rows = sc.oriented_rows("val")[args.offset : args.offset + args.n]
    imgs, labels, srcs = [], [], []
    for rel, label, _ in rows:
        img = sc.read_gray("val", rel)
        u8 = resize_pad_u8(img, sc.H, sc.W)
        assert np.array_equal(normalize_u8(u8)[None], sc.preprocess(img)), "u8 path differs from spike preprocess"
        imgs.append(u8)
        labels.append(label)
        srcs.append(f"val/{rel}")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0 : sc.H, 0 : sc.W]
    synth = {
        "black": np.zeros((sc.H, sc.W), np.uint8),
        "white": np.full((sc.H, sc.W), 255, np.uint8),
        "noise": rng.integers(0, 256, (sc.H, sc.W), dtype=np.uint8),
        "checker": (((yy + xx) % 2) * 255).astype(np.uint8),
        "stripes": (((xx // 3) % 2) * 255).astype(np.uint8),
    }
    if args.synthetic:
        for k, v in synth.items():
            imgs.append(v)
            labels.append("")
            srcs.append(f"synthetic:{k}")
    N = len(imgs)

    ex = TorchExecutor(graph=graph, device="cpu")
    out_var = graph.outputs[list(graph.outputs.keys())[0]]
    out_cfg = out_var.source_op.output_quant_config[out_var.source_op.outputs.index(out_var)]
    out_exp = T[info["output"]]["exp"]
    assert float(out_cfg.scale) == 2.0 ** out_exp, "logits exponent mismatch between PPQ graph and .espdl"
    Tt, C = T[info["output"]]["w"], T[info["output"]]["c"]
    inter_names = [n for n in tid if n in graph.variables and n != list(graph.inputs.keys())[0]]

    vocab = sc.load_vocab()
    golden_in, golden_logits, golden_ids, decoded = [], [], [], []
    inter_records = []
    missing_inter: set[str] = set()
    not_exact = 0
    t0 = time.time()
    with torch.no_grad():
        for i, u8 in enumerate(imgs):
            x = torch.from_numpy(normalize_u8(u8)[None, None])
            q_in = PPQLinearQuant_toInt(x, in_cfg).numpy().astype(np.int8)[0, 0]
            assert np.array_equal(q_in, lut[u8]), "per-image input quantisation != LUT"
            want = [out_var.name] + (inter_names if i < args.n_inter else [])
            res = ex.forward(inputs=x, output_names=want)
            y = res[0]
            yq = y / out_cfg.scale
            if not torch.equal(yq, torch.round(yq)):
                not_exact += 1
            yi = PPQLinearQuant_toInt(y, out_cfg).numpy().astype(np.int8)  # [1, C, 1, T]
            ltc = np.ascontiguousarray(yi[0, :, 0, :].T)  # [T, C] (NHWC order)
            ids = sc.ctc_greedy(ltc[:, :ncls].T.astype(np.int32))
            golden_in.append(q_in)
            golden_logits.append(ltc)
            golden_ids.append(ids)
            decoded.append("".join(vocab[j - 1] for j in ids if 0 < j <= len(vocab)))
            if i < args.n_inter:
                for name, v in zip(want[1:], res[1:]):
                    if v is None:  # not materialised by the executor
                        missing_inter.add(name)
                        continue
                    t_id = tid[name]
                    e = T[t_id]["exp"]
                    vv = v.numpy()[0].transpose(1, 2, 0)  # NCHW -> HWC
                    qi = np.clip(np.floor(vv / 2.0 ** e + 0.5), -128, 127).astype(np.int8)
                    exact = bool(np.array_equal(vv / 2.0 ** e, np.round(vv / 2.0 ** e)))
                    inter_records.append((i, t_id, qi.tobytes(), exact, name))
    print(f"simulated {N} inputs in {time.time() - t0:.1f}s; logits not on the int8 grid: {not_exact}")
    if missing_inter:
        print(f"intermediates not returned by TorchExecutor: {sorted(missing_inter)}")

    # ---- golden.bin
    rec = 2 * sc.H * sc.W + Tt * C + 2 + 2 * Tt
    gb = bytearray(struct.pack("<4sIIHHHHH", GOLDEN_MAGIC, 1, N, sc.H, sc.W, Tt, C, Tt).ljust(32, b"\0"))
    for u8, qi, lg, ids in zip(imgs, golden_in, golden_logits, golden_ids):
        r = bytearray()
        r += u8.astype(np.uint8).tobytes()
        r += qi.astype(np.int8).tobytes()
        r += lg.astype(np.int8).tobytes()
        r += struct.pack("<H", len(ids))
        r += np.array(list(ids) + [0] * (Tt - len(ids)), "<u2").tobytes()
        assert len(r) == rec
        gb += r
    (out / "golden.bin").write_bytes(bytes(gb))

    ib = bytearray(struct.pack("<4sIII", INTER_MAGIC, 1, args.n_inter, len(inter_records)))
    for vi, t_id, payload, _, _ in inter_records:
        ib += struct.pack("<IHHI", vi, t_id, 0, len(payload)) + payload
    (out / "golden_inter.bin").write_bytes(bytes(ib))

    # ---- metadata, vocab, a sample image for the CLI
    (out / "vocab.txt").write_text("\n".join(["<blank>"] + vocab) + "\n", encoding="utf-8")
    with open(out / "sample.pgm", "wb") as f:
        f.write(f"P5\n{sc.W} {sc.H}\n255\n".encode() + imgs[0].tobytes())
    (out / "tensors.txt").write_text(
        "\n".join(f"t{i}\t{t['h']}x{t['w']}x{t['c']}\texp {t['exp']}\t{t['name']}" for i, t in enumerate(T)) + "\n",
        encoding="utf-8")
    seq_ok = sum(d == l for d, l in zip(decoded, labels) if l)
    meta = {
        "onnx": str(Path(args.onnx).resolve().relative_to(ROOT)) if Path(args.onnx).resolve().is_relative_to(ROOT) else args.onnx,
        "tqt": args.tqt, "n": N, "n_real": len(rows), "val_offset": args.offset, "n_inter_vectors": args.n_inter,
        "input_exp": in_exp, "logits_exp": out_exp, "arena_size": info["arena_size"], "macs": info["macs"],
        "hxhm_bytes": info["file_size"], "espdl_bytes": (out / "model.espdl").stat().st_size,
        "inter_tensors_not_on_grid": sorted({n for _, _, _, ex_, n in inter_records if not ex_}),
        "exact_line_matches_ppq_int8": seq_ok,
        "vectors": [{"src": s, "label": l, "ppq_int8_decoded": d, "ids": list(map(int, ids))}
                    for s, l, d, ids in zip(srcs, labels, decoded, golden_ids)],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"golden: {N} vectors ({len(rows)} real), {len(inter_records)} intermediate tensors; "
          f"PPQ int8 decodes == label on {seq_ok}/{len(rows)} real lines")

    if args.test_data:
        td = Path(args.test_data)
        td.mkdir(parents=True, exist_ok=True)
        for f in ("model.hxhm", "golden.bin", "golden_inter.bin", "tensors.txt", "vocab.txt", "sample.pgm"):
            shutil.copy(out / f, td / f)
        print(f"copied test data to {td}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help=".espdl -> .hxhm")
    c.add_argument("espdl")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--norm-mean", type=float, default=0.5, help="training normalisation (x/255 - mean) / std")
    c.add_argument("--norm-std", type=float, default=0.5)
    c.add_argument("--no-input-lut", action="store_true")
    c.add_argument("--num-classes", type=int, help="classes to decode (default: all output channels)")
    c.add_argument("--reuse-input", action="store_true", help="let activations overwrite the input (saves arena)")
    c.add_argument("--raw-output", action="store_true",
                   help="the output is a raw tensor (e.g. text detector maps), not CTC logits (format version 2)")
    c.set_defaults(fn=cmd_convert)

    gld = sub.add_parser("golden", help="quantize ONNX like the spike, convert, dump golden vectors")
    gld.add_argument("--onnx", required=True)
    gld.add_argument("--out-dir", required=True)
    gld.add_argument("--test-data", help="also copy model + golden files here (rust tests)")
    gld.add_argument("--compare-espdl", help="report whether the re-quantized .espdl is byte-identical to this one")
    gld.add_argument("--tqt", action="store_true")
    gld.add_argument("--tqt-steps", type=int, default=500)
    gld.add_argument("--tqt-lr", type=float, default=1e-5)
    gld.add_argument("--tqt-block", type=int, default=4)
    gld.add_argument("--tqt-int-lambda", type=float, default=0.25)
    gld.add_argument("--calib-n", type=int, default=64)
    gld.add_argument("--calib-bs", type=int, default=8)
    gld.add_argument("--n", type=int, default=64, help="real val lines")
    gld.add_argument("--offset", type=int, default=2000, help="skip this many horizontal val lines (held out)")
    gld.add_argument("--no-synthetic", dest="synthetic", action="store_false")
    gld.add_argument("--n-inter", type=int, default=2, help="vectors with intermediate tensors dumped")
    gld.add_argument("--num-classes", type=int, help="classes to decode (default: all output channels)")
    gld.add_argument("--reuse-input", action="store_true", help="let activations overwrite the input (saves arena)")
    gld.set_defaults(fn=cmd_golden)

    d = sub.add_parser("dump", help="print a .hxhm")
    d.add_argument("hxhm")
    d.set_defaults(fn=cmd_dump)

    args = ap.parse_args()
    try:
        args.fn(args)
    except ConvertError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
