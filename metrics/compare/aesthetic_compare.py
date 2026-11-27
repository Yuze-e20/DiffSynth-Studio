"""Aesthetic score of every frame, per method.

    python -m metrics.compare.aesthetic_compare --out results.json

Model and scoring are examples/image_quality_metric/aesthetic.py's: diffsynth's
AestheticMetric, the LAION aesthetic predictor (CLIP ViT-L/14 image embedding
into an MLP, roughly a 1-10 scale). Frames are resized to short side 480 at
their own aspect, like the other metrics; the metric's own processor then does
short side 224 + center crop 224x224, so this matches feeding it raw frames.
A video's score is the mean over all its frames; a method's, the mean over
videos. The source video is scored too, as the reference point.
"""
import argparse
import json
import os
import random

import numpy as np

from diffsynth.metrics import AestheticMetric, ModelConfig
from metrics.compare.clip_compare import frames_480
from metrics.compare.lineart_compare import ROOT, path_of

BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/aesthetic_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Source,Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    metric = AestheticMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="Aesthetic/model.safetensors"),
        device="cuda")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        for m in methods:
            path = f"{ROOT}/all/{v}.mp4" if m == "Source" else path_of(m, v)
            frames, size = frames_480(path)
            s = []
            for i in range(0, len(frames), BATCH):
                s.extend(metric.compute(frames[i:i + BATCH]))
            r = {"video": v, "method": m, "frames": len(frames), "eval_size": list(size),
                 "aesthetic": float(np.mean(s)), "aesthetic_std": float(np.std(s))}
            results.append(r)
            print(f"{v:18s} {m:10s} aesthetic {r['aesthetic']:.3f}  ({r['frames']} frames)", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos (higher is better)")
    for m in methods:
        print(f"  {m:10s} {np.mean([r['aesthetic'] for r in results if r['method'] == m]):.3f}")


if __name__ == "__main__":
    main()
