"""UR2-A/C/S + UR-E only for scale66@sigma15 full 70."""
from __future__ import annotations
import argparse, json, os, itertools
from pathlib import Path

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
OUT_DIR = os.path.join(ROOT, "_metrics")
SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
METHODS = [f"s{u:.1f}_t{t:.1f}" for u, t in itertools.product(SCALES, SCALES)]
PATHS = {m: os.path.join(ROOT, "ablation_scale66_sigma15", m, "{v}.mp4") for m in METHODS}

from metrics.compare.lineart_compare import GEOMETRY, box_center_crop_diffsynth
from metrics.compare import run_full_metrics as rfm

def all_videos():
    d = os.path.join(ROOT, "DAVIS2017_origin")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".mp4"))

for m in METHODS:
    GEOMETRY[m] = box_center_crop_diffsynth


def path_of(method, video):
    if method == "Source":
        return os.path.join(ROOT, "DAVIS2017_origin", f"{video}.mp4")
    p = PATHS[method].format(v=video)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def out_path(metric):
    return os.path.join(OUT_DIR, f"scale66_s15_{metric}.jsonl")


def load_done(metric):
    done = set()
    p = out_path(metric)
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                done.add((r["video"], r["method"]))
    return done


def append(metric, row):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path(metric), "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def shard(items, i, n):
    items = list(items)
    a, b = i * len(items) // n, (i + 1) * len(items) // n
    return items[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=["ur2", "ur_edit"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(all_videos(), i, n)
    print(f"scale66_s15 metric={a.metric} shard={a.shard} nvid={len(videos)} nmethod={len(METHODS)}", flush=True)
    import metrics.compare.lineart_compare as lc
    for m in METHODS:
        lc.GEOMETRY[m] = box_center_crop_diffsynth
    rfm.path_of = path_of
    rfm.out_path = out_path
    rfm.load_done = load_done
    rfm.append_result = append
    if a.metric == "ur2":
        rfm.run_ur2(videos, METHODS)
    else:
        rfm.run_ur_edit(videos, METHODS)
    print("DONE", a.metric, a.shard, flush=True)


if __name__ == "__main__":
    main()
