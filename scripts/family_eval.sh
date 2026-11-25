#!/usr/bin/env bash
# Browser-pipeline benchmarks for the model family (node, exactly web/ocr.js readPage):
# every model set on every synthetic test page set, with the reading-confidence sweep and the
# detector-score sweep. Results go to
# experiments/family/eval/<models>_<set>[_score].txt; scripts/family_table.py summarises them.
#
#   bash scripts/family_eval.sh "current tiny base" [sets...]
set -u
cd "$(dirname "$0")/.."
MODELS=${1:-"current tiny base"}; shift || true
SETS=${*:-"pages pages_v3 pages_foreign pages_v3o pages_foreign_open pages_v5"}
OUT=experiments/family/eval
mkdir -p $OUT
args() {
  [ "$1" = current ] && return
  echo --det experiments/family/$1/det.hxhm --rec experiments/family/$1/model.hxhm --rec-v experiments/family/$1/model_v.hxhm
}
for m in $MODELS; do
  for s in $SETS; do
    node web/test/page_e2e.mjs --pages web/test/out/$s $(args $m) --conf-sweep > $OUT/${m}_$s.txt 2>&1 &
  done
  wait
  for s in $SETS; do
    node web/test/page_e2e.mjs --pages web/test/out/$s $(args $m) --score-sweep --scores 0,0.55,0.6 > $OUT/${m}_${s}_score.txt 2>&1 &
  done
  wait
done
