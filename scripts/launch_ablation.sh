#!/usr/bin/env bash
# Launch the three h32 ablation runs on one GPU, staggered: each run starts only after the
# previous one has reached step 100 (starting several CUDA processes at the same instant
# crashed all of them with "CUDA error: unknown error" on Windows/WDDM).
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
COMMON="--data data/synth_v1 data/synth_v2 --steps 30000 --batch-size 128 --lr 2e-3 --num-workers 4 --eval-every 1000 --val-n 2000"
runs=("mnv4s_h32:--arch mnv4s" "mnv4s_scratch_h32:--arch mnv4s --no-pretrained" "tiny_h32:--arch tiny")
pids=()
for spec in "${runs[@]}"; do
  name=${spec%%:*}; extra=${spec#*:}
  rm -rf "runs/$name"; mkdir -p "runs/$name"
  PYTHONUTF8=1 $PY scripts/train.py $COMMON $extra --name "$name" > "runs/$name/train.log" 2>&1 &
  pids+=($!)
  echo "started $name (pid $!)"
  for i in $(seq 1 180); do
    grep -q "step 100/" "runs/$name/train.log" && break
    grep -qE "Traceback|Error" "runs/$name/train.log" && { echo "$name failed to start"; tail -5 "runs/$name/train.log"; break; }
    sleep 2
  done
done
for p in "${pids[@]}"; do wait "$p"; done
for spec in "${runs[@]}"; do name=${spec%%:*}; echo "== $name"; tail -2 "runs/$name/train.log"; done
