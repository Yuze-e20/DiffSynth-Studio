#!/bin/bash
# Evaluate Ref2VA only on the 70-video set across 8 GPUs. Appends to full_*.jsonl.
set -euo pipefail
ROOT=/mnt/nas3/sunyuzework/Diffutoon-2
DS=$ROOT/DiffSynth-Studio
PY=/root/miniconda3/envs/syz-diffsynth/bin/python
OUT=$ROOT/data/test_refined/_metrics
LOG=$OUT/full_logs_ref2va
mkdir -p "$LOG"
export DIFFSYNTH_MODEL_BASE_PATH=/root/models
export PYTHONPATH=$DS:/mnt/nas3/sunyuzework/Diffutoon-2/data/pydeps
export TORCH_HOME=/root/models/metrics/torch_hub
METHODS=Ref2VA

run_metric_sharded () {
  local metric=$1
  local n=${2:-8}
  echo "$(date '+%F %T') start $metric x$n shards methods=$METHODS"
  local pids=()
  for i in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY -u -m metrics.compare.run_full_metrics \
      --metric "$metric" --shard "$i/$n" --methods "$METHODS" \
      > "$LOG/${metric}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  echo "$(date '+%F %T') done $metric fail=$fail"
  return $fail
}

cd "$DS"
run_metric_sharded ur2 8
run_metric_sharded ur_edit 8
run_metric_sharded clip 8
run_metric_sharded temporal 8
run_metric_sharded color 8
echo "$(date '+%F %T') ALL REF2VA METRICS FINISHED"

$PY - <<'PY'
import json, collections, os, numpy as np
OUT="/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined/_metrics"
print("Ref2VA rows in each metric:")
for met in ["color","clip","ur2","ur_edit","csd","smoothness"]:
    p=f"{OUT}/full_{met}.jsonl"
    rows=[json.loads(l) for l in open(p) if l.strip() and json.loads(l).get("method")=="Ref2VA"]
    if not rows:
        print(f"  {met}: 0"); continue
    if met=="color":
        print(f"  {met}: n={len(rows)} dE {np.mean([r['delta_e'] for r in rows]):.3f} hist {np.mean([r['ab_hist_intersection'] for r in rows]):.4f}")
    elif met=="clip":
        print(f"  {met}: n={len(rows)} clip {np.mean([r['clip'] for r in rows]):.3f}")
    elif met=="ur2":
        print(f"  {met}: n={len(rows)} a {np.mean([r['alignment'] for r in rows]):.3f} c {np.mean([r['coherence'] for r in rows]):.3f} s {np.mean([r['style'] for r in rows]):.3f} m {np.mean([r['score'] for r in rows]):.3f}")
    elif met=="ur_edit":
        print(f"  {met}: n={len(rows)} score {np.mean([r['score'] for r in rows]):.3f} suc {np.mean([r['editing_success'] for r in rows]):.3f} over {np.mean([r['overediting'] for r in rows]):.3f}")
    elif met=="csd":
        print(f"  {met}: n={len(rows)} csd {np.mean([r['csd_consistency'] for r in rows]):.4f}")
    elif met=="smoothness":
        print(f"  {met}: n={len(rows)} smooth {np.mean([r['motion_smoothness'] for r in rows]):.5f}")
PY
