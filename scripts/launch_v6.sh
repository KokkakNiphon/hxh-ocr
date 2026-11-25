#!/usr/bin/env bash
# Readers v6: numbers and punctuation written into the lettering (generator v6, lines_v6), the
# horizontal readers of both family members fine-tuned from their family checkpoints, int8 and
# checked bit for bit like the family runs -> experiments/family/<tiny_v6|base_v6>/.
# What was run on 2026-11-25 (docs/research/JOURNAL.md, "readers v6: numbers in the lettering"),
# stage by stage. One stage at a time; two trainers with 2 loader workers each at most (the
# Windows commit limit, see the journal).
#
#   bash scripts/launch_v6.sh data | readers | int8 | json | eval
#
# The vertical readers and both detectors stay as they are (copied unchanged into the v6 folders).
set -eu
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SYNTH=rust/target/release/hxh-synth.exe
export PYTHONUTF8=1
V6="--font-dir fonts/open --v5 1 --v6 1 --mode-p 0.2,0.1,0.06,0.08,0.1,0.1,0.12,0.24 --ui-neg-prob 0.4 --photo-prob 0.3
    --caption-prob 0.25 --ruby-prob 0.35 --sticker-prob 0.35 --empty-prob 0.03"
# Stage 1 mix: lines_v6 twice (43% of the lines), lines_v5b 12%, synth_v4 45%. Stage 2 (rebalancing,
# with the small-dakuten supplement): lines_v6 + lines_v6m 23%, lines_v5b 9%, synth_v4 68% (twice),
# close to the family's 79% synth_v4: stage 1 alone doubled the synth_v4 hold-out error and read
# stray "!" at line ends (journal). MIX2 (no supplement) was the first stage 2, v6b_*.
MIX="data/lines_v6 data/lines_v6 data/lines_v5b data/synth_v4"
MIX2="data/lines_v6 data/lines_v5b data/synth_v4 data/synth_v4"
MIX3="data/lines_v6 data/lines_v6m data/lines_v5b data/synth_v4 data/synth_v4"

case "${1:-}" in
data)
  cargo build --release -p hxh-synth --manifest-path rust/Cargo.toml
  $SYNTH lines --out data/lines_v6 --train 30000 --val 1000 --seed 16 --threads 6 $V6 --inline-prob 0.5
  $PY scripts/pack_dataset.py data/lines_v6
  # supplement: the same recipe with small dakuten dots in 60% of the styles (added after the first
  # int8 readers still missed small dakuten in bold printed lettering)
  $SYNTH lines --out data/lines_v6m --train 12000 --val 500 --seed 17 --threads 4 $V6 --inline-prob 0.5 --small-mark-prob 0.6
  $PY scripts/pack_dataset.py data/lines_v6m
  # held-out digit-heavy test lines (new seed, 9 of 10 Hunter lines with numbers / punctuation)
  $SYNTH lines --out data/lines_v6_digits --train 0 --val 1500 --seed 61 --threads 4 $V6 --inline-prob 0.9
  # test pages in the v6 styles for the unchanged detectors (browser pipeline, page_e2e.mjs)
  $SYNTH pages --out data/pages_v6 --train 0 --val 200 --seed 62 --threads 6 $V6 --inline-prob 0.5 \
    --preview previews/pages_v6.png --preview-n 16
  $PY web/test/make_pages.py --data data/pages_v6/val --offset 0 --n 200 --out data/pages_v6/e2e --drop-illegible ;;
readers)
  # one trainer at a time, one loader worker (two trainers with two workers each hit the Windows
  # commit limit, error 1455, on 2026-11-25). The base stage 1 died at step 8000 of 10000 in that
  # crash; its best.pt (step 7000, lr already at 11% of peak) was used as is.
  $PY scripts/train.py --data $MIX --arch mnv4s --height 48 --init runs/fam_tiny_h48b/best.pt     --steps 8000 --lr 8e-4 --warmup 300 --name v6_tiny_h48 --num-workers 1 --eval-every 1000
  $PY scripts/train.py --data $MIX --arch mnv4s1 --height 48 --init runs/fam_base_h48/best.pt     --steps 10000 --lr 8e-4 --warmup 300 --name v6_base_h48 --num-workers 1 --eval-every 1000
  $PY scripts/train.py --data $MIX2 --arch mnv4s --height 48 --init runs/v6_tiny_h48/best.pt     --steps 3000 --lr 3e-4 --warmup 100 --name v6b_tiny_h48 --num-workers 1 --eval-every 500
  $PY scripts/train.py --data $MIX2 --arch mnv4s1 --height 48 --init runs/v6_base_h48/best.pt     --steps 3000 --lr 3e-4 --warmup 100 --name v6b_base_h48 --num-workers 1 --eval-every 500
  $PY scripts/train.py --data $MIX3 --arch mnv4s --height 48 --init runs/v6_tiny_h48/best.pt --steps 3000 --lr 3e-4 --warmup 100 --name v6c_tiny_h48 --num-workers 1 --eval-every 500
  $PY scripts/train.py --data $MIX3 --arch mnv4s1 --height 48 --init runs/v6_base_h48/best.pt --steps 3000 --lr 3e-4 --warmup 100 --name v6c_base_h48 --num-workers 1 --eval-every 500 ;;
int8)
  # the family recipe: QAT on the training mix, ESP-PPQ PTQ (KL) calibrated on synth_v4, .hxhm,
  # bit-exact against ESP-PPQ (512 hold-out lines, host engine and naive reference); the
  # width-1.0 reader with cross-layer equalization first (EQ=1); then wasm SIMD and scalar
  # delivered: tiny = v6c (stage 2 with lines_v6m), base = v6b (stage 2 without it: v6c was no
  # better for base, journal); qat_v6b_tiny_h48 was built too, for the comparison
  export HXH_VOCAB=data/lines_v6/vocab.json NC=91 DATA=data/synth_v4 QAT_WORKERS=1
  QAT_DATA="$MIX3" bash scripts/qat_pipeline.sh qat_v6c_tiny_h48 v6c_tiny_h48 48 768
  QAT_DATA="$MIX2" EQ=1 bash scripts/qat_pipeline.sh qat_v6b_base_h48 v6b_base_h48 48 768
  QAT_DATA="$MIX2" bash scripts/qat_pipeline.sh qat_v6b_tiny_h48 v6b_tiny_h48 48 768
  for n in qat_v6c_tiny_h48 qat_v6b_base_h48; do
    o=experiments/rust_engine/${n}_ptq
    node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $o/model.hxhm $o/golden.bin --bench-ms 3000 | tail -3
    node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $o/model.hxhm $o/golden.bin --scalar --bench-ms 0 | tail -2
  done ;;
json)
  # tiny_v6 = the Fast set (det_v2, deployed) with the new horizontal reader; base_v6 = the
  # Accurate set (det_base_v1_qat) with the new one; vertical readers unchanged
  for s in tiny base; do
    d=experiments/family/${s}_v6
    mkdir -p $d
    if [ $s = tiny ]; then src=web/models; det=det_v2; else src=experiments/family/base; det=det_base_v1_qat; fi
    cp $src/det.hxhm $src/det.json $d/
    cp experiments/family/$s/model_v.hxhm experiments/family/$s/model_v.json $d/
    r=qat_v6b_${s}_h48; [ $s = tiny ] && r=qat_v6c_tiny_h48
    cp experiments/rust_engine/${r}_ptq/model.hxhm $d/model.hxhm
    cp $d/det.json $d/det.json.keep
    $PY scripts/family_json.py --size ${s}_v6 --det-name $det --rec-name $r \
      --rec-v-name qat_fam_${s}_h48_vert --vocab data/lines_v6/vocab.json --notes $d/notes.json
    mv $d/det.json.keep $d/det.json   # the detector json stays the deployed one
    cp experiments/family/$s/model_v.json $d/model_v.json
  done ;;
eval)
  # fp32 checkpoints on the line sets (scripts/eval.py: CER, digit / punctuation metrics, false digits)
  for r in fam_tiny_h48b v6_tiny_h48 v6b_tiny_h48 v6c_tiny_h48 fam_base_h48 v6_base_h48 v6b_base_h48 v6c_base_h48; do
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v6_digits/val/labels.tsv --n 0 --tag v6_digits
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v6/val/labels.tsv --n 0 --tag v6_val
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v5b/val/labels.tsv --n 0 --tag v6_v5b
    $PY scripts/eval.py --run runs/$r --tsv data/synth_v4/val/labels.tsv --n 3000 --tag v6_v4
  done
  # int8 readers on the same line sets through the app's readCrop (WASM SIMD)
  $PY scripts/eval_lines_int8.py --out experiments/family/eval/lines_int8_v6.json   # deployed vs delivered
  $PY scripts/eval_lines_int8.py --models "tiny v6b (qat_v6b_tiny_h48)=experiments/rust_engine/qat_v6b_tiny_h48_ptq/model.hxhm"     --out experiments/family/eval/lines_int8_v6b_tiny.json
  # browser pipeline on the page sets: deployed Fast (det_v2 + tiny readers = v2tiny) and Accurate
  # (base) against the v6 sets with the same detectors
  mkdir -p experiments/family/v2tiny
  cp web/models/det.hxhm web/models/det.json experiments/family/tiny/model*.hxhm experiments/family/tiny/model*.json experiments/family/v2tiny/
  OUT=experiments/family/eval
  # (v2tiny and base on the older sets: their files from scripts/launch_family.sh eval)
  for m in v2tiny tiny_v6 base base_v6; do
    a="--det experiments/family/$m/det.hxhm --rec experiments/family/$m/model.hxhm --rec-v experiments/family/$m/model_v.hxhm"
    sets="pages pages_v3o pages_foreign_open pages_v5 pages_v6"
    case $m in v2tiny|base) sets=pages_v6 ;; esac
    for s in $sets; do
      dir=web/test/out/$s; [ $s = pages_v6 ] && dir=data/pages_v6/e2e
      node web/test/page_e2e.mjs --pages $dir $a --conf-sweep > $OUT/${m}_$s.txt 2>&1 &
      node web/test/page_e2e.mjs --pages $dir $a --score-sweep --scores 0,0.55,0.6 > $OUT/${m}_${s}_score.txt 2>&1 &
    done
    wait
  done
  $PY scripts/det_check_v6.py --models v2tiny,base --out $OUT/det_check_v6.json
  $PY scripts/v6_tables.py ;;
*) echo "usage: $0 data | readers | int8 | json | eval"; exit 2 ;;
esac
