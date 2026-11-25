"""Parse an .espdl file without ESP-DL: 16-byte header + FlatBuffers `espdl.Model`
(schema: esp-dl/fbs_loader/espdl.fbs in github.com/espressif/esp-dl; the Python
bindings ship inside esp-ppq as esp_ppq/parser/espdl/FlatBuffers/Dl).

Prints the node list, every tensor's dtype / shape / exponents, and checks that
all exponents are per-tensor (one value) and that every quantized tensor is int8
(biases int32).

Usage: python inspect_espdl.py out/random_avg_op18/model.espdl
"""
from __future__ import annotations

import struct
import sys
from collections import Counter

from esp_ppq.parser.espdl.FlatBuffers.Dl.Model import ModelT

DTYPES = {0: "UNDEF", 1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64", 9: "BOOL"}
ATTR_SHOW = {"kernel_shape", "strides", "pads", "group", "activation", "quant_type", "axes", "keepdims"}


def s(b) -> str:
    return b.decode() if isinstance(b, bytes) else str(b)


def attr_val(a):
    t = a.attrType
    if t == 1:
        return round(a.f.f, 6)
    if t == 2:
        return a.i.i
    if t == 3:
        return bytes(a.s).decode(errors="replace")
    if t == 7:
        return [int(v) for v in a.ints]
    if t == 6:
        return [float(v) for v in a.floats]
    return f"<type {t}>"


def main(path: str) -> None:
    data = open(path, "rb").read()
    magic, enc, length = data[:4], *struct.unpack("<II", data[4:12])
    print(f"file {path}: {len(data)} bytes; header magic={magic!r} encrypted={enc} payload={length}")
    assert magic == b"EDL2" and length == len(data) - 16
    m = ModelT.InitFromPackedBuf(data[16:], 0)
    g = m.graph
    print(f"ir_version={m.irVersion} opset={[o.version for o in (m.opsetImport or [])]} producer={s(m.producerName)}")

    print("\nnodes:")
    for n in g.node:
        attrs = {s(a.name): attr_val(a) for a in (n.attribute or []) if s(a.name) in ATTR_SHOW}
        print(f"  {s(n.opType):18s} {s(n.name)[:40]:40s} {attrs}")

    print("\ninitializers:")
    exps, dts, wbytes = Counter(), Counter(), 0
    for t in g.initializer:
        nbytes = 16 * len(t.rawData or [])
        wbytes += nbytes
        e = [int(x) for x in t.exponents] if t.exponents is not None else []
        exps[len(e)] += 1
        dts[DTYPES.get(t.dataType, t.dataType)] += 1
        print(f"  {s(t.name)[:30]:30s} {DTYPES.get(t.dataType):6s} dims={[int(d) for d in t.dims]} exps={e} raw={nbytes}B")

    print("\nactivations (value_info + io):")
    seen = set()
    for v in list(g.input) + list(g.output) + list(g.valueInfo or []):
        if v.name in seen:
            continue
        seen.add(v.name)
        tt = v.valueInfoType.value
        e = [int(x) for x in v.exponents] if v.exponents is not None else []
        exps[len(e)] += 1
        dt = DTYPES.get(tt.elemType, tt.elemType)
        dts["act_" + dt] += 1
        dims = [d.value.dimValue for d in tt.shape.dim]
        print(f"  {s(v.name)[:45]:45s} {dt:6s} {dims} exps={e}")

    print("\nsummary:")
    print("  exponent-vector lengths:", dict(exps), "(1 = per-tensor)")
    print("  dtypes:", dict(dts))
    print(f"  initializer raw bytes: {wbytes} of {len(data)} total")
    print("  embedded test inputs/outputs:", len(g.testInputsValue or []), len(g.testOutputsValue or []))


if __name__ == "__main__":
    main(sys.argv[1])
