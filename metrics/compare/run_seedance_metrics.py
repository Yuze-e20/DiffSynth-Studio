"""Run full metrics + lineart F1 for Seedance25 on available videos only."""
import json
import os
import sys

import numpy as np

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
OUT = f"{ROOT}/_metrics"
METHOD = "Seedance25"


def available_videos():
    allv = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{ROOT}/all") if f.endswith(".mp4"))
    return [v for v in allv if os.path.exists(f"{ROOT}/{METHOD}/{v}.mp4") and os.path.getsize(f"{ROOT}/{METHOD}/{v}.mp4") > 0]


def main():
    vids = available_videos()
    print(f"Seedance25 available: {len(vids)}", flush=True)
    print(vids, flush=True)
    # monkeypatch all_videos for run_full_metrics
    import metrics.compare.registry as reg
    reg.all_videos = lambda: vids
    # ensure method registered
    if METHOD not in reg.METHODS:
        reg.METHODS = list(reg.METHODS) + [METHOD]
    if METHOD not in reg.GEN_METHODS:
        reg.GEN_METHODS = list(reg.GEN_METHODS) + [METHOD]

    from metrics.compare import run_full_metrics as rfm
    # also patch its imports of all_videos
    rfm.all_videos = lambda: vids

    metrics = ["ur2", "ur_edit", "clip", "temporal", "color"]
    # shard via env SHARD=i/n
    shard = os.environ.get("SHARD", "0/1")
    i, n = map(int, shard.split("/"))
    # only run requested metric
    metric = os.environ["METRIC"]
    print(f"metric={metric} shard={shard}", flush=True)
    videos = reg.shard(vids, i, n)
    methods = [METHOD]
    if metric == "color":
        rfm.run_color(videos, methods)
    elif metric == "clip":
        rfm.run_clip(videos, methods)
    elif metric == "ur2":
        rfm.run_ur2(videos, methods)
    elif metric == "ur_edit":
        rfm.run_ur_edit(videos, methods)
    elif metric == "temporal":
        rfm.run_temporal(videos, methods, do_csd=True, do_smooth=True)
    elif metric == "csd":
        rfm.run_temporal(videos, methods, do_csd=True, do_smooth=False)
    elif metric == "smoothness":
        rfm.run_temporal(videos, methods, do_csd=False, do_smooth=True)
    else:
        raise SystemExit(f"unknown metric {metric}")
    print("DONE", metric, shard, flush=True)


if __name__ == "__main__":
    main()
