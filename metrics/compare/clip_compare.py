"""CLIP Score of every frame against an anime-style text prompt, per method.

    python -m metrics.compare.clip_compare --out results.json

The model and scoring are examples/image_quality_metric/clipscore.py's:
diffsynth's CLIPMetric with CLIP ViT-H-14 (laion2B-s32B-b79K), whose score is
100 * cos(image, text). Every output frame is resized to short side 480 at its
own aspect ratio, as for the other metrics, then fed in; a video's score is the
mean over all its frames, and a method's is the mean over videos.

The source video is scored too, as the photographic reference point: a method
is only "more anime" by how far it rises above the real footage.

CLIP's own processor then resizes the short side to 224 and center-crops
224x224, so each frame is judged on its middle ~57% of width at 16:9.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from diffsynth.metrics import CLIPMetric, ModelConfig
from metrics.compare.lineart_compare import ROOT, eval_size, path_of, read_all

PROMPTS = [
    "A picture from Japanese anime-style cartoon video with clean line work and flat cel-shaded colors",
    "A Japanese anime-style cartoon video with clean line work and flat cel-shaded colors",
]
BATCH = 32


def frames_480(path):
    frames, _ = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    return [Image.fromarray(f).resize((ew, eh), Image.BICUBIC) for f in frames], (ew, eh)


def score_frames(metric, frames, prompt):
    out = []
    for i in range(0, len(frames), BATCH):
        out.extend(metric.compute(prompt, frames[i:i + BATCH]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/clip_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Source,Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    metric = CLIPMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                 origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"),
        device="cuda")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        for m in methods:
            path = f"{ROOT}/all/{v}.mp4" if m == "Source" else path_of(m, v)
            frames, size = frames_480(path)
            r = {"video": v, "method": m, "frames": len(frames), "eval_size": list(size)}
            for k, prompt in enumerate(PROMPTS):
                s = score_frames(metric, frames, prompt)
                r[f"clip_p{k}"] = float(np.mean(s))
                r[f"clip_p{k}_std"] = float(np.std(s))
            results.append(r)
            print(f"{v:18s} {m:10s} p0 {r['clip_p0']:.3f}  p1 {r['clip_p1']:.3f}  ({r['frames']} frames, {size[0]}x{size[1]})",
                  flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "prompts": PROMPTS, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos (higher = closer to the anime text)")
    for k, prompt in enumerate(PROMPTS):
        print(f"  prompt {k}: {prompt!r}")
        for m in methods:
            print(f"    {m:10s} {np.mean([r[f'clip_p{k}'] for r in results if r['method'] == m]):.3f}")


if __name__ == "__main__":
    main()
