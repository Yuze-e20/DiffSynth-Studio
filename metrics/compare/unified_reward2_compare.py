"""UnifiedReward-2.0 pointwise scores on generated frames.

    DIFFSYNTH_MODEL_BASE_PATH=/root/models python -m metrics.compare.unified_reward2_compare

Model/scoring are examples/image_quality_metric/unified_reward_2.py's:
UnifiedReward2Metric (Qwen3.5-9B) returning three 1..5 axes —
Alignment / Coherence / Style — plus their mean.

Text caption fed to the judge:
  STYLE_PREFIX + " " + global_caption
where global_caption is the same VLM caption our pipeline used
(prompt_segments_w328_c1g.json).
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

from diffsynth.metrics import ModelConfig, UnifiedReward2Metric
from metrics.compare.lineart_compare import ROOT, eval_size, path_of, read_all

CAPTION_DIR = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS-MiraData_nf328_ctx1g_s20_u05_t05/DAVIS2017/_work"
STYLE_PREFIX = (
    "A Japanese anime-style cartoon frame with clean line work and flat cel-shaded colors."
)
N_INSTANTS = 4
METHODS = "Source,Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V"


def global_caption(video):
    p = os.path.join(CAPTION_DIR, f"{video}_auto", "prompt_segments_w328_c1g.json")
    d = json.load(open(p))
    cap = d["global_caption"].strip()
    return f"{STYLE_PREFIX} {cap}"


def frames_at_instants(method, video, n):
    path = f"{ROOT}/all/{video}.mp4" if method == "Source" else path_of(method, video)
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    duration = len(frames) / fps
    out = []
    for s in (np.arange(n) + 0.5) * duration / n:
        i = min(len(frames) - 1, int(round(s * fps)))
        out.append(Image.fromarray(frames[i]).resize((ew, eh), Image.BICUBIC))
    return out, (ew, eh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/unified_reward2_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instants", type=int, default=N_INSTANTS)
    ap.add_argument("--methods", default=METHODS)
    a = ap.parse_args()

    davis = sorted(
        f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "")
        for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
        if f.endswith(".mp4")
    )
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    print("style prefix:", STYLE_PREFIX, flush=True)

    metric = UnifiedReward2Metric.from_pretrained(
        model_config=ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="UnifiedReward-2.0-qwen35-9b/model-*.safetensors",
        ),
        processor_config=ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="UnifiedReward-2.0-qwen35-9b/",
        ),
        device="cuda",
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        prompt = global_caption(v)
        print(f"\n=== {v} caption: {prompt[:160]}...", flush=True)
        for m in methods:
            imgs, size = frames_at_instants(m, v, a.instants)
            frames = []
            for img in imgs:
                d = metric.evaluate(prompt, img)[0]
                frames.append({k: d.get(k) for k in ("alignment", "coherence", "style", "score", "raw") if k in d or True})
                # keep raw text if present
                if "raw" not in d and isinstance(d, dict):
                    frames[-1] = {k: d.get(k) for k in ("alignment", "coherence", "style", "score")}
                    if "text" in d:
                        frames[-1]["text"] = d["text"]
            ok = [f for f in frames if f.get("score") is not None]
            def mean_key(k):
                vals = [f[k] for f in ok if f.get(k) is not None]
                return float(np.mean(vals)) if vals else None
            r = {
                "video": v, "method": m, "prompt": prompt, "eval_size": list(size),
                "instants": len(frames), "parsed": len(ok),
                "alignment": mean_key("alignment"),
                "coherence": mean_key("coherence"),
                "style": mean_key("style"),
                "score": mean_key("score"),
                "frames": frames,
            }
            results.append(r)
            print(
                f"{v:18s} {m:10s} align {r['alignment']}  coh {r['coherence']}  "
                f"style {r['style']}  mean {r['score']}  ({r['parsed']}/{r['instants']})",
                flush=True,
            )
            with open(a.out, "w") as fh:
                json.dump(
                    {"videos": videos, "style_prefix": STYLE_PREFIX, "results": results},
                    fh, indent=1, ensure_ascii=False,
                )

    print("\nmean over", len(videos), "videos (1..5, higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m and r["score"] is not None]
        if not rs:
            continue
        print(
            f"  {m:10s} align {np.mean([r['alignment'] for r in rs]):.3f}  "
            f"coh {np.mean([r['coherence'] for r in rs]):.3f}  "
            f"style {np.mean([r['style'] for r in rs]):.3f}  "
            f"mean {np.mean([r['score'] for r in rs]):.3f}"
        )


if __name__ == "__main__":
    main()
