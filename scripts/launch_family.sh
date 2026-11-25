#!/usr/bin/env bash
# The model family (experiments/family/<tiny|base>/): data, training, QAT, int8, model jsons.
# What was run on 2026-11-25 (docs/research/JOURNAL.md), stage by stage. Run one stage at a
# time: on this 32 GB machine two trainers with 2-3 loader workers each is the most that fits
# the Windows commit limit next to everything else.
#
#   bash scripts/launch_family.sh data | det-tiny | det-base | readers-tiny | readers-base | int8 | json | eval
set -eu
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SYNTH=rust/target/release/hxh-synth.exe
export PYTHONUTF8=1
V5="--font-dir fonts/open --v5 1 --mode-p 0.25,0.12,0.08,0.1,0.15,0.15,0.15 --ui-neg-prob 0.5"
MIX="data/pages_v2 data/pages_v3o data/pages_v5"

case "${1:-}" in
data)
  # detector pages: v5 (photo-style), the v3 recipe re-rendered with open fonts, open-font foreign check set
  $SYNTH pages --out data/pages_v5 --train 30000 --val 1000 --seed 5 --threads 8 $V5 --photo-prob 0.4 \
    --caption-prob 0.25 --empty-prob 0.05 --preview previews/pages_v5.png --preview-n 16
  $SYNTH pages --out data/pages_v3o --train 30000 --val 1000 --seed 3 --threads 8 --font-dir fonts/open
  $SYNTH pages --out data/pages_foreign_open --train 0 --val 300 --seed 4 --threads 4 --font-dir fonts/open \
    --mode-p 0.1,0.1,0.25,0.55 --empty-prob 0.35 --foreign-prob 0.5 --mixed-prob 0.15 --foreign-extra 1.5 \
    --icon-extra 3 --ui-prob 0.45
  # reader lines cut from v5 pages (extended vocab: digits / punctuation inline). lines_v5b: after the
  # hollow-glyph fix and with the legibility filter (lines_v5, before them, was not used in the end)
  $SYNTH lines --out data/lines_v5b --train 14000 --val 1000 --seed 6 --threads 6 $V5 --photo-prob 0.35 \
    --caption-prob 0.3 --inline-prob 0.15 --empty-prob 0.03
  $PY scripts/pack_dataset.py data/lines_v5b
  for s in pages_v5:900 pages_v3o:900 pages_foreign_open:0; do
    $PY web/test/make_pages.py --data data/${s%%:*}/val --offset ${s##*:} --n $([ ${s%%:*} = pages_foreign_open ] && echo 150 || echo 100) \
      --out web/test/out/${s%%:*} $([ ${s%%:*} = pages_v5 ] && echo --drop-illegible)
  done ;;
det-tiny)
  $PY scripts/train_det.py --data $MIX --data-weights 0.25 0.25 0.5 --val-data data/pages_v5 \
    --init runs/det_v2/best.pt --name det_tiny_v3 --steps 8000 --lr 5e-4 --warmup 300 --batch-size 16 \
    --num-workers 3 --val-n 300 --eval-every 2000
  $PY scripts/train_det_qat.py --ckpt runs/det_tiny_v3/best.pt --name det_tiny_v3_qat --data $MIX \
    --data-weights 0.25 0.25 0.5 --steps 2000 --batch-size 12 ;;
det-base)
  export HXH_LETTERBOX=960
  # ImageNet -> old page sets (24k steps; best at 22k), then the v5 mix
  $PY scripts/train_det.py --data data/pages_v2 data/pages_v3 --name det_base_pre --steps 24000 --batch-size 12 \
    --crop 640 --long-range 576 1920 --width 1.0 --fpn 64 --level 24 --mid 96 --head-dw --num-workers 4 --val-n 200
  $PY scripts/train_det.py --data $MIX --data-weights 0.25 0.25 0.5 --val-data data/pages_v5 \
    --init runs/det_base_pre/best.pt --width 1.0 --fpn 64 --level 24 --mid 96 --head-dw --name det_base_v1 \
    --steps 7000 --lr 1e-3 --warmup 300 --batch-size 12 --crop 640 --long-range 576 1920 --num-workers 3 --val-n 300
  $PY scripts/train_det_qat.py --ckpt runs/det_base_v1/best.pt --name det_base_v1_qat --data $MIX \
    --data-weights 0.25 0.25 0.5 --steps 2000 --batch-size 8 --crop 640 --long-range 576 1920 ;;
readers-tiny)
  $PY scripts/train.py --data data/lines_v5b data/synth_v4 --arch mnv4s --height 48 --init runs/v4_mnv4s_h48/best.pt \
    --steps 6000 --lr 1e-3 --warmup 300 --name fam_tiny_h48b --num-workers 2 --eval-every 1000
  $PY scripts/train.py --data data/lines_v5b data/synth_v4 --arch mnv4s --height 48 --orient v \
    --init runs/v4_mnv4s_h48_vert/best.pt --steps 6000 --lr 1e-3 --warmup 300 --name fam_tiny_h48b_vert \
    --num-workers 2 --eval-every 1000 ;;
readers-base)
  $PY scripts/train.py --data data/synth_v4 --arch mnv4s1 --height 48 --steps 20000 --name fam_base_h48_pre \
    --num-workers 2   # stopped at 14k (val CER 0.99%), see the journal
  $PY scripts/train.py --data data/lines_v5b data/synth_v4 --arch mnv4s1 --height 48 \
    --init runs/fam_base_h48_pre/best.pt --steps 9000 --lr 1e-3 --warmup 300 --name fam_base_h48 --num-workers 2
  $PY scripts/train.py --data data/lines_v5b data/synth_v4 --arch mnv4s1 --height 48 --orient v \
    --init runs/fam_base_h48_pre/best.pt --steps 9000 --lr 1.5e-3 --warmup 300 --name fam_base_h48_vert --num-workers 2 ;;
int8)
  export HXH_VOCAB=data/lines_v5b/vocab.json NC=91 DATA=data/synth_v4 QAT_DATA="data/lines_v5b data/synth_v4"
  bash scripts/qat_pipeline.sh qat_fam_tiny_h48 fam_tiny_h48b 48 768
  HXH_ORIENT=v bash scripts/qat_pipeline.sh qat_fam_tiny_h48_vert fam_tiny_h48b_vert 48 768
  # the width-1.0 readers need cross-layer equalization before QAT (EQ=1); for the tiny ones it did not help
  EQ=1 bash scripts/qat_pipeline.sh qat_fam_base_h48 fam_base_h48 48 768
  EQ=1 HXH_ORIENT=v bash scripts/qat_pipeline.sh qat_fam_base_h48_vert fam_base_h48_vert 48 768
  LB=640 bash scripts/det_int8_pipeline.sh det_tiny_v3_qat experiments/family/tiny
  LB=960 bash scripts/det_int8_pipeline.sh det_base_v1_qat experiments/family/base ;;
json)
  for s in tiny base; do
    cp experiments/rust_engine/qat_fam_${s}_h48_ptq/model.hxhm experiments/family/$s/model.hxhm
    cp experiments/rust_engine/qat_fam_${s}_h48_vert_ptq/model.hxhm experiments/family/$s/model_v.hxhm
    $PY scripts/family_json.py --size $s --det-name det_${s}_v$([ $s = tiny ] && echo 3 || echo 1)_qat \
      --rec-name qat_fam_${s}_h48 --rec-v-name qat_fam_${s}_h48_vert --vocab data/lines_v5b/vocab.json --notes experiments/family/$s/notes.json
  done ;;
eval)
  # "v2tiny" = the deployed detector (web/models/det.*) with the tiny readers (not a deliverable, a comparison)
  mkdir -p experiments/family/v2tiny
  cp web/models/det.hxhm web/models/det.json experiments/family/tiny/model*.hxhm experiments/family/tiny/model*.json experiments/family/v2tiny/
  bash scripts/family_eval.sh "current tiny base v2tiny"
  $PY scripts/family_table.py --models current,tiny,v2tiny,base ;;
*) echo "usage: $0 data | det-tiny | det-base | readers-tiny | readers-base | int8 | json | eval"; exit 2 ;;
esac
