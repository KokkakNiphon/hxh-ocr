#!/usr/bin/env bash
# int8 path for one trained detector (model family): BN-folded ONNX -> ESP-PPQ PTQ (+TQT) for
# esp32s3 -> .hxhm (raw output, v2) -> golden vectors -> bit-exact checks in the Rust engine
# (host scalar + wasm SIMD / scalar).
#
#   [LB=960] [TQT=1] [DATA=data/pages_v5] bash scripts/det_int8_pipeline.sh <run name> [dest dir]
#   LB: letterbox / input size (640 tiny, 960 base); dest dir gets det.hxhm (e.g. experiments/family/base)
set -eu
cd "$(dirname "$0")/.."
NAME=$1; DEST=${2:-}; LB=${LB:-640}; DATA=${DATA:-data/pages_v5}; TQT=${TQT:-0}
PY=.venv/Scripts/python.exe
PPQ=.venv-ppq/Scripts/python.exe
export PYTHONUTF8=1 HXH_LETTERBOX=$LB
EQ=${EQ:-0}  # 1: ESP-PPQ cross-layer equalization before calibration (needed by the width-1.0 backbone)
TAG=${NAME}_$([ "$EQ" = 1 ] && echo eq_)$([ "$TQT" = 1 ] && echo tqt || echo ptq)
log() { echo "[$(date +%T)] $NAME: $*"; }

log "export ONNX ${LB}x${LB}"
$PY scripts/export_det.py --run runs/$NAME --out exports/$NAME.onnx 2>&1 | grep -E "fused|OK|MISMATCH|ops" || true

log "ESP-PPQ ($TAG, calibration $DATA/val[0:64], hold-out val[900:1000])"
$PPQ scripts/quantize_det.py --onnx exports/$NAME.onnx --tag $TAG --data $DATA $([ "$TQT" = 1 ] && echo --tqt) $([ "$EQ" = 1 ] && echo --equalize) \
  > runs/$NAME/quantize_$TAG.out 2>&1
grep -E "detection|e2e|===" runs/$NAME/quantize_$TAG.out || true

log "convert + golden vectors (16 hold-out pages)"
OUT=experiments/det_engine/$TAG
mkdir -p $OUT
$PY - "$TAG" "$OUT" <<'EOF'
import json, sys
import numpy as np
tag, out = sys.argv[1], sys.argv[2]
d = f"experiments/det/{tag}"
m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
x = np.load(f"{d}/golden/inputs_u8.npy")[:16]
q = np.load(f"{d}/golden/outputs_int8.npy")[:16].astype(np.float32) * 2.0 ** m["output_exponent"]
np.savez(f"{out}/saved.npz", images=x, outputs=q)
print("saved", x.shape, q.shape, "output exponent", m["output_exponent"])
EOF
$PPQ experiments/det_engine/golden_det.py from-saved --espdl experiments/det/$TAG/model.espdl --saved $OUT/saved.npz \
  --out-dir $OUT --n 16 > $OUT/convert.out 2>&1
tail -3 $OUT/convert.out

log "bit-exact: host (scalar reference)"
ABS=$(cd $OUT && pwd -W)
(cd rust && HXH_MODEL="$ABS/model.hxhm" HXH_GOLDEN="$ABS/golden.bin" \
  cargo test --release -p hxh-infer --test golden -- --ignored --nocapture external 2>&1 \
  | grep -E "bit-exact|test result|mismatch" ) || log "host check FAILED"
log "bit-exact: wasm SIMD and scalar"
node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $OUT/model.hxhm $OUT/golden.bin --bench-ms 4000 2>&1 | tail -3
node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $OUT/model.hxhm $OUT/golden.bin --scalar --bench-ms 0 2>&1 | tail -2
if [ -n "$DEST" ]; then
  mkdir -p $DEST && cp $OUT/model.hxhm $DEST/det.hxhm && log "copied to $DEST/det.hxhm"
fi
log "done"
