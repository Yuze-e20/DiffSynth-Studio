#!/bin/bash
# Full-set metrics on 8 GPUs. Resumable via jsonl. Logs under _metrics/full_logs/
set -euo pipefail
ROOT=/mnt/nas3/sunyuzework/Diffutoon-2
DS=$ROOT/DiffSynth-Studio
PY=/root/miniconda3/envs/syz-diffsynth/bin/python
OUT=$ROOT/data/test_refined/_metrics
LOG=$OUT/full_logs
mkdir -p "$LOG"
export DIFFSYNTH_MODEL_BASE_PATH=/root/models
export PYTHONPATH=$DS:/mnt/nas3/sunyuzework/Diffutoon-2/data/pydeps
export TORCH_HOME=/root/models/metrics/torch_hub

run_metric_sharded () {
  local metric=$1
  local n=${2:-8}
  echo "$(date '+%F %T') start $metric x$n shards"
  local pids=()
  for i in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=$i $PY -u -m metrics.compare.run_full_metrics \
      --metric "$metric" --shard "$i/$n" \
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
# Phase order: heavy VLM first while GPUs free, then lighter
run_metric_sharded ur2 8
run_metric_sharded ur_edit 8
run_metric_sharded clip 8
run_metric_sharded temporal 8   # csd + smoothness together
run_metric_sharded color 8      # mostly CPU; still shard to parallelise

echo "$(date '+%F %T') ALL PHASES FINISHED"
$PY - <<'PY'
import json, collections, os, numpy as np
OUT="/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined/_metrics"
metrics=["color","clip","ur2","ur_edit","csd","smoothness"]
methods=["Source","Ours","EchoStyle","PickStyle","Kiwi-Edit","AnyV2V"]
for met in metrics:
    p=f"{OUT}/full_{met}.jsonl"
    if not os.path.exists(p):
        print(met, "MISSING"); continue
    rows=[json.loads(l) for l in open(p) if l.strip()]
    by=collections.defaultdict(list)
    for r in rows: by[r["method"]].append(r)
    print(f"\n== {met} n={len(rows)}")
    for m in methods:
        rs=by.get(m,[])
        if not rs: continue
        if met=="color":
            print(f"  {m:10s} dE {np.mean([r['delta_e'] for r in rs]):.3f} hist {np.mean([r['ab_hist_intersection'] for r in rs]):.4f} (n={len(rs)})")
        elif met=="clip":
            print(f"  {m:10s} clip {np.mean([r['clip'] for r in rs]):.3f} (n={len(rs)})")
        elif met=="ur2":
            print(f"  {m:10s} a {np.mean([r['alignment'] for r in rs]):.3f} c {np.mean([r['coherence'] for r in rs]):.3f} s {np.mean([r['style'] for r in rs]):.3f} m {np.mean([r['score'] for r in rs]):.3f} (n={len(rs)})")
        elif met=="ur_edit":
            print(f"  {m:10s} score {np.mean([r['score'] for r in rs]):.3f} suc {np.mean([r['editing_success'] for r in rs]):.3f} over {np.mean([r['overediting'] for r in rs]):.3f} (n={len(rs)})")
        elif met=="csd":
            print(f"  {m:10s} csd {np.mean([r['csd_consistency'] for r in rs]):.4f} (n={len(rs)})")
        elif met=="smoothness":
            print(f"  {m:10s} smooth {np.mean([r['motion_smoothness'] for r in rs]):.5f} (n={len(rs)})")
PY
