#!/usr/bin/env bash
# Round 3 on synth_v4 (official-font glyph variants, art beside lines, heavier strokes, shadows):
#   mnv4s 48 px (browser demo) and mnv4s3 32 px (ESP32-S3 candidate), trained on v4 only
#   (v3 drew ま only as Σ, so mixing it in would teach the wrong orientation).
# At most MAXPAR runs at once with 2 loader workers each: 3 runs x 4 workers exhausted the
# Windows commit limit (error 1455). Each run starts only after the previous one reaches
# step 100 (simultaneous CUDA start-up crashed runs on WDDM).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
MAXPAR=${MAXPAR:-2}
COMMON="--data data/synth_v4 --steps 30000 --batch-size 128 --lr 2e-3 --num-workers 2 --eval-every 1000 --val-n 2000"
runs=("v4_mnv4s_h48:--arch mnv4s --height 48 --max-width 768"
      "v4_mnv4s3_h32:--arch mnv4s3 --height 32")
running=0
for spec in "${runs[@]}"; do
  if [ "$running" -ge "$MAXPAR" ]; then wait -n; running=$((running - 1)); fi
  name=${spec%%:*}; extra=${spec#*:}
  rm -rf "runs/$name"; mkdir -p "runs/$name"
  PYTHONUTF8=1 $PY scripts/train.py $COMMON $extra --name "$name" > "runs/$name/train.log" 2>&1 &
  running=$((running + 1))
  echo "started $name (pid $!)"
  for i in $(seq 1 180); do
    grep -q "step 100/" "runs/$name/train.log" && break
    grep -qE "Traceback|Error" "runs/$name/train.log" && { echo "$name failed to start"; tail -5 "runs/$name/train.log"; break; }
    sleep 2
  done
done
wait
for spec in "${runs[@]}"; do name=${spec%%:*}; echo "== $name"; tail -1 "runs/$name/train.log"; done
