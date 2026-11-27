"""CLIP with anime style + global caption, ALL frames. Ours vs PickStyle, 70 videos.

    DIFFSYNTH_MODEL_BASE_PATH=/root/models python -m metrics.compare.clip_caption_compare --shard 0/8
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from diffsynth.metrics import CLIPMetric, ModelConfig
from metrics.compare.lineart_compare import eval_size, read_all
from metrics.compare.registry import (
    ROOT, STYLE_PREFIX, all_videos, global_caption, path_of, shard,
)

OUT = os.path.join(ROOT, "_metrics", "clip_caption_allframes_ours_vs_pickstyle.jsonl")
METHODS = ["Ours", "PickStyle"]
BATCH = 32


def frames_all(path):
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    imgs = [Image.fromarray(f).resize((ew, eh), Image.BICUBIC) for f in frames]
    return imgs, fps, (ew, eh)


def load_done():
    done = set()
    if not os.path.exists(OUT):
        return done
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["video"], r["method"]))
    return done


def append(row):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(all_videos(), i, n)
    print(f"shard {a.shard}: {len(videos)} videos · ALL frames · caption prompt", flush=True)

    metric = CLIPMetric.from_pretrained(
        model_config=ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors",
        ),
        device="cuda",
    )
    done = load_done()
    for v in videos:
        prompt = f"{STYLE_PREFIX} {global_caption(v)}"
        for m in METHODS:
            if (v, m) in done:
                print(f"[skip] {v} {m}", flush=True)
                continue
            imgs, fps, size = frames_all(path_of(m, v))
            scores = []
            for j in range(0, len(imgs), BATCH):
                scores.extend(metric.compute(prompt, imgs[j:j + BATCH]))
            row = {
                "video": v, "method": m, "prompt": prompt,
                "eval_size": list(size), "fps": fps,
                "frames_scored": len(scores),
                "clip": float(np.mean(scores)), "clip_std": float(np.std(scores)),
            }
            append(row)
            print(f"{v:18s} {m:10s} {row['clip']:.3f}  ({len(scores)} frames)", flush=True)
    print("DONE", a.shard, flush=True)


if __name__ == "__main__":
    main()
