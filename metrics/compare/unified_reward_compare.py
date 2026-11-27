"""UnifiedReward-Edit pointwise score: source frame vs generated frame, per method.

    python -m metrics.compare.unified_reward_compare --out results.json

Model, task and parsing are examples/image_quality_metric/unified_reward_edit.py's:
diffsynth's UnifiedRewardEditMetric (UnifiedReward-Edit, Qwen3-VL 8B) with its
built-in "edit_pointwise_score" template, which takes [source, edited] and an
editing instruction and returns two scores on 0..25 -- editing success (did the
instruction happen) and over-editing (25 = a minimal yet effective edit of the
source) -- plus their mean. Only the instruction text is ours; the template is
the model's own, so nothing about the judging is re-worded.

Each pair is the source frame and the method's frame at the same instant, both
in that method's geometry (lineart_compare's alignment: the source is cut the way
that method cut it) and at short side 480. Scored at N_INSTANTS evenly spaced
instants per video, since every call is a full VLM generation.
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

from diffsynth.metrics import ModelConfig, UnifiedRewardEditMetric
from metrics.compare.lineart_compare import GEOMETRY, ROOT, eval_size, path_of, read_all

INSTRUCTION = "Redraw this photo as a frame from a Japanese anime, with clean line art and flat cel-shaded colors."
# Always append the video's global caption after the style instruction (same VLM
# caption our pipeline used). Future runs must keep this.
CAPTION_DIR = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS-MiraData_nf328_ctx1g_s20_u05_t05/DAVIS2017/_work"
N_INSTANTS = 4
METHODS = "Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V"


def instruction_for(video):
    p = os.path.join(CAPTION_DIR, f"{video}_auto", "prompt_segments_w328_c1g.json")
    cap = json.load(open(p))["global_caption"].strip()
    return f"{INSTRUCTION} {cap}"


def pairs(method, v, src, src_fps, n):
    gen, gen_fps = read_all(path_of(method, v))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    out = []
    for s in (np.arange(n) + 0.5) * duration / n:          # centres of n equal spans
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
        out.append((ref.resize((ew, eh), Image.BICUBIC), Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/unified_reward_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instants", type=int, default=N_INSTANTS)
    ap.add_argument("--methods", default=METHODS)
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, "\nstyle instruction:", INSTRUCTION, "\n(+ per-video global_caption)", flush=True)
    metric = UnifiedRewardEditMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                 origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors"),
        processor_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                     origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/"),
        device="cuda")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            frames = []
            for ref, gen in pairs(m, v, src, src_fps, a.instants):
                d = metric.evaluate(instruction_for(v), [ref, gen], task="edit_pointwise_score")[0]
                frames.append({k: d.get(k) for k in ("editing_success", "overediting", "score", "reasoning")})
            ok = [f for f in frames if f["score"] is not None]
            r = {"video": v, "method": m, "instants": len(frames), "parsed": len(ok),
                 "editing_success": float(np.mean([f["editing_success"] for f in ok])) if ok else None,
                 "overediting": float(np.mean([f["overediting"] for f in ok])) if ok else None,
                 "score": float(np.mean([f["score"] for f in ok])) if ok else None,
                 "frames": frames}
            results.append(r)
            print(f"{v:18s} {m:10s} score {r['score']:6.2f}  success {r['editing_success']:6.2f}  "
                  f"overedit {r['overediting']:6.2f}  ({r['parsed']}/{r['instants']} parsed)", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "instruction": INSTRUCTION, "caption_appended": True, "results": results}, fh, indent=1, ensure_ascii=False)
    print("\nmean over", len(videos), "videos (0..25, higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m and r["score"] is not None]
        print(f"  {m:10s} score {np.mean([r['score'] for r in rs]):6.2f}  "
              f"success {np.mean([r['editing_success'] for r in rs]):6.2f}  overedit {np.mean([r['overediting'] for r in rs]):6.2f}")


if __name__ == "__main__":
    main()
