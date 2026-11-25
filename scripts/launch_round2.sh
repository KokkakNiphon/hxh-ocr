#!/usr/bin/env bash
# Round 2 on synth_v3 (tighter crops, fixed binarisation and neighbour lines):
#   mnv4s 32 px vs mnv4s 48 px (do dakuten/handakuten need more pixels?) vs mnv4s3 32 px
#   (3-stage, 96k params / 15M MACs: the ESP32-S3 speed option).
# At most MAXPAR runs at once with 2 loader workers each: 3 runs x 4 workers exhausted the
# Windows commit limit (error 1455). Each run starts only after the previous one reaches
# step 100 (simultaneous CUDA start-up crashed runs on WDDM).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
MAXPAR=${MAXPAR:-2}
COMMON="--data data/synth_v3 --steps 30000 --batch-size 128 --lr 2e-3 --num-workers 2 --eval-every 1000 --val-n 2000"
runs=("v3_mnv4s_h32:--arch mnv4s --height 32"
      "v3_mnv4s_h48:--arch mnv4s --height 48 --max-width 768"
      "v3_mnv4s3_h32:--arch mnv4s3 --height 32")
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
