#!/usr/bin/env bash
# The steps of qat_pipeline.sh after QAT, for a QAT run that ended early but saved best.pt
# (e.g. stopped by the machine's commit limit): ONNX -> ESP-PPQ PTQ -> .hxhm -> bit-exact check.
#   [DATA=data/synth_v4] [NC=91] [HXH_VOCAB=...] [HXH_ORIENT=v] bash scripts/qat_finish.sh <name> 48 768
set -eu
cd "$(dirname "$0")/.."
NAME=$1; H=${2:-48}; W=${3:-768}; DATA=${DATA:-data/synth_v4}; NC=${NC:-75}
PY=.venv/Scripts/python.exe
PPQ=.venv-ppq/Scripts/python.exe
export PYTHONUTF8=1 HXH_ORIENT=${HXH_ORIENT:-h}
$PY scripts/export_onnx.py --run runs/$NAME --width $W --opset 18 2>&1 | grep -E "OK|ops:" || true
cp runs/$NAME/model_${H}x${W}.onnx exports/q_$NAME.onnx
HXH_DATA=$DATA HXH_H=$H HXH_W=$W $PPQ experiments/esp_ppq_spike/quantize.py \
  --onnx exports/q_$NAME.onnx --tag ${NAME}_ptq --eval-n 512 --save-outputs > runs/$NAME/quantize.out 2>&1
OUT=experiments/rust_engine/${NAME}_ptq
mkdir -p $OUT
$PPQ scripts/espdl_to_hxhm.py convert experiments/esp_ppq_spike/out/${NAME}_ptq/model.espdl \
  -o $OUT/model.hxhm --num-classes $NC > $OUT/convert.out 2>&1
HXH_DATA=$DATA HXH_H=$H HXH_W=$W $PPQ experiments/rust_engine/golden_from_saved.py \
  experiments/esp_ppq_spike/out/${NAME}_ptq $OUT --n 512 --num-classes $NC > $OUT/golden.out 2>&1
ABS=$(cd $OUT && pwd -W)
(cd rust && HXH_MODEL="$ABS/model.hxhm" HXH_GOLDEN="$ABS/golden.bin" \
  cargo test --release -p hxh-infer --test golden -- --ignored --nocapture external 2>&1 | grep -E "bit-exact|test result")
$PY - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
m = json.load(open(f"experiments/esp_ppq_spike/out/{name}_ptq/metrics.json", encoding="utf-8"))
c = m["holdout_cer"]; h = m["holdout"]
print(f"RESULT {name}: holdout CER fp32 {c['fp32']:.4f} int8 {c['ppq_int8']:.4f} | "
      f"greedy identical {h['greedy_seq_identical']:.3f} | espdl {m['sizes']['espdl']} B")
EOF
