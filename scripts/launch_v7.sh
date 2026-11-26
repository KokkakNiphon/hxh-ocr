#!/usr/bin/env bash
# Readers v7: the cross `×` (the separator of the anime's episode titles, "A×ト×B") as the 91st
# token (generator v7: --v7 1, cross lines and the title-card scene, lines_v7), the horizontal
# readers of both family members fine-tuned from their v6 checkpoints with the classifier grown
# from 91 to 92 classes, int8 and checked bit for bit like the v6 runs ->
# experiments/family/<tiny_v7|base_v7>/. See docs/research/JOURNAL.md, "readers v7".
# One stage at a time; one trainer with one loader worker (the Windows commit limit, v6 journal).
#
#   bash scripts/launch_v7.sh data | readers | int8 | json | eval
#
# The vertical readers and both detectors stay as they are (copied unchanged into the v7 folders).
set -eu
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SYNTH=rust/target/release/hxh-synth.exe
export PYTHONUTF8=1
# v6's recipe plus the title scene (9th weight, from the promo weight and a little of each other
# scene) and small dakuten dots in 30% of the styles
V7="--font-dir fonts/open --v5 1 --v6 1 --v7 1 --mode-p 0.18,0.09,0.05,0.07,0.09,0.09,0.11,0.12,0.2 --ui-neg-prob 0.4
    --photo-prob 0.3 --caption-prob 0.25 --ruby-prob 0.35 --sticker-prob 0.35 --empty-prob 0.03 --small-mark-prob 0.3"
# lines_v7 twice (15% of the lines), lines_v6 + v6m 16%, lines_v5b 6%, synth_v4 63% (twice): v6
# kept synth_v4 at about 70% after stage 1 alone (43% new lines) hurt the old distribution
MIX7="data/lines_v7 data/lines_v7 data/lines_v6 data/lines_v6m data/lines_v5b data/synth_v4 data/synth_v4"
# stage 2: the cross-heavy supplement lines_v7x twice on top (crosses in about 3% of the lines
# instead of 1.6%; stage 1 read 87-89% of the held-out crosses, stage 2 91-92%)
MIX7B="data/lines_v7x data/lines_v7x data/lines_v7 data/lines_v6 data/lines_v6m data/lines_v5b data/synth_v4 data/synth_v4"
# tiny stage 3 (rebalancing, like v6's stage 2): synth_v4 three times (about 70% of the lines); the
# int8 tiny reader of stage 2 dropped word spaces on synth_v4 (106 against 55 in 3,000 lines)
MIX7D="data/lines_v7x data/lines_v7 data/lines_v6 data/lines_v6m data/lines_v5b data/synth_v4 data/synth_v4 data/synth_v4"

case "${1:-}" in
data)
  cargo build --release -p hxh-synth --manifest-path rust/Cargo.toml
  # inline lines 50% of the horizontal Hunter lines, 40% of those titles with crosses
  $SYNTH lines --out data/lines_v7 --train 20000 --val 1000 --seed 18 --threads 6 $V7 --inline-prob 0.5 --cross-prob 0.4
  $PY scripts/pack_dataset.py data/lines_v7
  # held-out cross-heavy test lines (new seed; 9 of 10 horizontal lines inline, 70% of those with crosses)
  $SYNTH lines --out data/lines_v7_cross --train 0 --val 1500 --seed 63 --threads 6 $V7 --inline-prob 0.9 --cross-prob 0.7
  # test pages in the v7 styles for the unchanged detectors (browser pipeline, page_e2e.mjs)
  $SYNTH pages --out data/pages_v7 --train 0 --val 200 --seed 64 --threads 6 $V7 --inline-prob 0.5 --cross-prob 0.4 \
    --preview previews/pages_v7.png --preview-n 16
  $PY web/test/make_pages.py --data data/pages_v7/val --offset 0 --n 200 --out data/pages_v7/e2e --drop-illegible
  # cross-heavy supplement for stage 2 (added after stage 1): title scenes 43% of the pages, inline
  # lines 90% of the horizontal Hunter lines, 80% of those with crosses
  $SYNTH lines --out data/lines_v7x --train 10000 --val 300 --seed 19 --threads 5     ${V7/0.18,0.09,0.05,0.07,0.09,0.09,0.11,0.12,0.2/0.15,0.07,0.04,0.05,0.06,0.06,0.08,0.06,0.43} --inline-prob 0.9 --cross-prob 0.8
  $PY scripts/pack_dataset.py data/lines_v7x ;;
readers)
  # from the delivered v6 float checkpoints (tiny v6c, base v6b); train.py grows the classifier
  # (old rows kept, the new row's bias starts at the old minimum)
  $PY scripts/train.py --data $MIX7 --arch mnv4s --height 48 --init runs/v6c_tiny_h48/best.pt --steps 5000 --lr 4e-4 --warmup 200 --name v7_tiny_h48 --num-workers 1 --eval-every 1000
  $PY scripts/train.py --data $MIX7 --arch mnv4s1 --height 48 --init runs/v6b_base_h48/best.pt --steps 5000 --lr 4e-4 --warmup 200 --name v7_base_h48 --num-workers 1 --eval-every 1000
  # stage 2 with the supplement: base v7b (delivered); tiny v7b, then v7c with the padding after
  # short lines supervised as blank (--pad-blank: v7b's int8 build read runs of ! there)
  $PY scripts/train.py --data $MIX7B --arch mnv4s --height 48 --init runs/v7_tiny_h48/best.pt --steps 3000 --lr 2e-4 --warmup 100 --name v7b_tiny_h48 --num-workers 1 --eval-every 500
  $PY scripts/train.py --data $MIX7B --arch mnv4s1 --height 48 --init runs/v7_base_h48/best.pt --steps 3000 --lr 2e-4 --warmup 100 --name v7b_base_h48 --num-workers 1 --eval-every 500
  $PY scripts/train.py --data $MIX7B --arch mnv4s --height 48 --init runs/v7_tiny_h48/best.pt --steps 3000 --lr 2e-4 --warmup 100 --name v7c_tiny_h48 --num-workers 1 --eval-every 500 --pad-blank 0.3
  # tiny stage 3 (delivered: v7d)
  $PY scripts/train.py --data $MIX7D --arch mnv4s --height 48 --init runs/v7c_tiny_h48/best.pt --steps 2000 --lr 1.5e-4 --warmup 100 --name v7d_tiny_h48 --num-workers 1 --eval-every 500 --pad-blank 0.3
  # the cross prior: how far to lower the cross's logit (tuning sets lines_v7x val, lines_v5 val)
  $PY scripts/cross_prior.py --ckpt runs/qat_v7b_base_h48/best.pt --out runs/qat_v7b_base_h48/cross_prior.json ;;
int8)
  # the family recipe (QAT, ESP-PPQ PTQ calibrated on synth_v4, .hxhm, bit-exact checks), 92 classes
  export HXH_VOCAB=data/lines_v7/vocab.json NC=92 DATA=data/synth_v4 QAT_WORKERS=1
  QAT_DATA="$MIX7B" bash scripts/qat_pipeline.sh qat_v7b_tiny_h48 v7b_tiny_h48 48 768
  QAT_DATA="$MIX7B" EQ=1 bash scripts/qat_pipeline.sh qat_v7b_base_h48 v7b_base_h48 48 768
  QAT_DATA="$MIX7B" QAT_PAD_BLANK=0.3 bash scripts/qat_pipeline.sh qat_v7c_tiny_h48 v7c_tiny_h48 48 768
  QAT_DATA="$MIX7D" QAT_PAD_BLANK=0.3 bash scripts/qat_pipeline.sh qat_v7d_tiny_h48 v7d_tiny_h48 48 768
  # the same QAT checkpoints exported with the cross's logit lowered (`s`: 1.0, `h`: 0.5); delivered: base s, tiny h
  LOGIT_SHIFT=91=1.0 QAT_FROM=qat_v7b_base_h48 bash scripts/qat_pipeline.sh qat_v7bs_base_h48 v7b_base_h48 48 768
  LOGIT_SHIFT=91=1.0 QAT_FROM=qat_v7c_tiny_h48 bash scripts/qat_pipeline.sh qat_v7cs_tiny_h48 v7c_tiny_h48 48 768
  LOGIT_SHIFT=91=1.0 QAT_FROM=qat_v7d_tiny_h48 bash scripts/qat_pipeline.sh qat_v7ds_tiny_h48 v7d_tiny_h48 48 768
  # tiny delivered with half the shift (1.0 cost it 8 points of cross recall)
  LOGIT_SHIFT=91=0.5 QAT_FROM=qat_v7d_tiny_h48 bash scripts/qat_pipeline.sh qat_v7dh_tiny_h48 v7d_tiny_h48 48 768
  for n in qat_v7dh_tiny_h48 qat_v7bs_base_h48; do
    o=experiments/rust_engine/${n}_ptq
    node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $o/model.hxhm $o/golden.bin --bench-ms 3000 | tail -3
    node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm $o/model.hxhm $o/golden.bin --scalar --bench-ms 0 | tail -2
  done ;;
json)
  # tiny_v7 = the Fast set (det_v2, deployed) with the new horizontal reader; base_v7 = the
  # Accurate set (det_base_v1_qat); vertical readers unchanged
  for s in tiny base; do
    d=experiments/family/${s}_v7
    mkdir -p $d
    if [ $s = tiny ]; then src=web/models; det=det_v2; else src=experiments/family/base; det=det_base_v1_qat; fi
    cp $src/det.hxhm $src/det.json $d/
    cp experiments/family/$s/model_v.hxhm experiments/family/$s/model_v.json $d/
    r=qat_v7bs_base_h48; [ $s = tiny ] && r=qat_v7dh_tiny_h48
    cp experiments/rust_engine/${r}_ptq/model.hxhm $d/model.hxhm
    cp $d/det.json $d/det.json.keep
    $PY scripts/family_json.py --size ${s}_v7 --det-name $det --rec-name $r \
      --rec-v-name qat_fam_${s}_h48_vert --vocab data/lines_v7/vocab.json --vocab-v data/lines_v6/vocab.json --notes $d/notes.json
    mv $d/det.json.keep $d/det.json   # the detector json stays the deployed one
    cp experiments/family/$s/model_v.json $d/model_v.json
  done ;;
eval)
  # fp32 checkpoints on the line sets (scripts/eval.py: CER, digit / punctuation and cross metrics)
  for r in v6c_tiny_h48 v7_tiny_h48 v7b_tiny_h48 v7c_tiny_h48 v7d_tiny_h48 v6b_base_h48 v7_base_h48 v7b_base_h48; do
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v7_cross/val/labels.tsv --n 0 --tag v7_cross
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v7/val/labels.tsv --n 0 --tag v7_val
    $PY scripts/eval.py --run runs/$r --tsv data/lines_v6_digits/val/labels.tsv --n 0 --tag v7_v6digits
    $PY scripts/eval.py --run runs/$r --tsv data/synth_v4/val/labels.tsv --n 3000 --tag v7_v4
  done
  # int8 readers on the line sets through the app's readCrop (WASM SIMD): deployed, v6, v7
  OUT=experiments/family/eval
  $PY scripts/eval_lines_int8.py --out $OUT/lines_int8_v7.json
  # browser pipeline on the page sets (same detectors): v6 against v7, and v7 on the older sets
  for m in tiny_v6 tiny_v7 base_v6 base_v7; do
    a="--det experiments/family/$m/det.hxhm --rec experiments/family/$m/model.hxhm --rec-v experiments/family/$m/model_v.hxhm"
    sets="pages pages_v3o pages_foreign_open pages_v5 pages_v6 pages_v7"
    case $m in *_v6) sets=pages_v7 ;; esac
    for s in $sets; do
      dir=web/test/out/$s; [ $s = pages_v6 ] && dir=data/pages_v6/e2e; [ $s = pages_v7 ] && dir=data/pages_v7/e2e
      node web/test/page_e2e.mjs --pages $dir $a --conf-sweep > $OUT/${m}_$s.txt 2>&1 &
      node web/test/page_e2e.mjs --pages $dir $a --score-sweep --scores 0,0.55,0.6 > $OUT/${m}_${s}_score.txt 2>&1 &
    done
    wait
  done
  # the unchanged detectors on the v7 pages: lines with crosses and title scenes against the rest
  $PY scripts/det_check_v6.py --models tiny_v7,base_v7 --pages data/pages_v7/e2e --labels data/pages_v7/val/labels.jsonl \
    --out $OUT/det_check_v7.json ;;
*) echo "usage: $0 data | readers | int8 | json | eval"; exit 2 ;;
esac
