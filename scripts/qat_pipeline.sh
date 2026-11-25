#!/usr/bin/env bash
# Full int8 path for one trained model, following the QAT recipe in the journal:
#   QAT (own checkpoint, own data, calibration lines = the ones ESP-PPQ will use)
#   -> ONNX (static HxW) -> ESP-PPQ plain PTQ (KL, no TQT) for esp32s3
#   -> .hxhm for the Rust engine -> bit-exact check against ESP-PPQ on 512 held-out lines.
#
#   [DATA=data/synth_v4] bash scripts/qat_pipeline.sh qat_v3_mnv4s_h32 v3_mnv4s_h32 32 512
#   HXH_ORIENT=v DATA=data/synth_v4 bash scripts/qat_pipeline.sh qat_v4_mnv4s_h48_vert v4_mnv4s_h48_vert 48 768
#     (vertical model: QAT reads the orientation from the checkpoint; ESP-PPQ calibration,
#      hold-out and golden vectors use the rotated columns via HXH_ORIENT)
set -eu
cd "$(dirname "$0")/.."
NAME=$1; SRC=$2; H=${3:-32}; W=${4:-512}; DATA=${DATA:-data/synth_v3}
# QAT_DATA: training lines for QAT (default DATA); NC: classes incl. blank (91 with the extended
# vocab; then also export HXH_VOCAB=<the extended vocab.json> so the hold-out decode knows them)
QAT_DATA=${QAT_DATA:-$DATA}; NC=${NC:-75}  # QAT_WORKERS: loader workers (default 2; 1 when memory is short)
PY=.venv/Scripts/python.exe
PPQ=.venv-ppq/Scripts/python.exe
export PYTHONUTF8=1
export HXH_ORIENT=${HXH_ORIENT:-h}
log() { echo "[$(date +%T)] $NAME: $*"; }

log "QAT from runs/$SRC/best.pt"
mkdir -p runs/$NAME
$PY scripts/train_qat.py --ckpt runs/$SRC/best.pt --data $QAT_DATA --name $NAME \
  --calib-dir $DATA/val --calib-width $W --num-workers ${QAT_WORKERS:-2} --lr ${QAT_LR:-1e-4} ${QAT_STEPS:+--steps $QAT_STEPS}   $([ "${EQ:-0}" = 1 ] && echo --equalize) > runs/$NAME/train_qat.out 2>&1
tail -2 runs/$NAME/train_qat.out

log "export ONNX ${H}x${W}"
$PY scripts/export_onnx.py --run runs/$NAME --width $W --opset 18 2>&1 | grep -E "OK|ops:" || true
cp runs/$NAME/model_${H}x${W}.onnx exports/q_$NAME.onnx   # quantize.py overwrites its input

log "ESP-PPQ PTQ (esp32s3, KL, no TQT)"
HXH_DATA=$DATA HXH_H=$H HXH_W=$W $PPQ experiments/esp_ppq_spike/quantize.py \
  --onnx exports/q_$NAME.onnx --tag ${NAME}_ptq --eval-n 512 --save-outputs > runs/$NAME/quantize.out 2>&1

log "convert to .hxhm and check the Rust engine"
OUT=experiments/rust_engine/${NAME}_ptq
mkdir -p $OUT
$PPQ scripts/espdl_to_hxhm.py convert experiments/esp_ppq_spike/out/${NAME}_ptq/model.espdl \
  -o $OUT/model.hxhm --num-classes $NC > $OUT/convert.out 2>&1
HXH_DATA=$DATA HXH_H=$H HXH_W=$W $PPQ experiments/rust_engine/golden_from_saved.py \
  experiments/esp_ppq_spike/out/${NAME}_ptq $OUT --n 512 --num-classes $NC > $OUT/golden.out 2>&1
ABS=$(cd $OUT && pwd -W)
(cd rust && HXH_MODEL="$ABS/model.hxhm" HXH_GOLDEN="$ABS/golden.bin" \
  cargo test --release -p hxh-infer --test golden -- --ignored --nocapture external 2>&1 \
  | grep -E "bit-exact|test result" ) || log "Rust check FAILED"

$PY - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
m = json.load(open(f"experiments/esp_ppq_spike/out/{name}_ptq/metrics.json", encoding="utf-8"))
c = m["holdout_cer"]; h = m["holdout"]
print(f"RESULT {name}: holdout CER fp32 {c['fp32']:.4f} int8 {c['ppq_int8']:.4f} | "
      f"greedy identical {h['greedy_seq_identical']:.3f} | espdl {m['sizes']['espdl']} B")
EOF
log "done"
