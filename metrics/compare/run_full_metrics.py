"""Full 70-video x 5-method metric runner, sharded across GPUs.

    DIFFSYNTH_MODEL_BASE_PATH=/root/models CUDA_VISIBLE_DEVICES=k \
    python -m metrics.compare.run_full_metrics --metric color --shard 0/8

Metrics: color, clip, ur2, ur_edit, csd, smoothness, temporal (=csd+smoothness)
Each writes/merges into data/test_refined/_metrics/full_<metric>.jsonl (one line per
video-method), so shards can run in parallel and a crashed shard is resumable.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from metrics.compare.lineart_compare import GEOMETRY, eval_size, read_all
from metrics.compare.registry import (
    EDIT_INSTRUCTION,
    GEN_METHODS,
    METHODS,
    ROOT,
    STYLE_PREFIX,
    all_videos,
    edit_prompt,
    path_of,
    shard,
    ur2_prompt,
)

OUT_DIR = os.path.join(ROOT, "_metrics")
N_INSTANTS = 4  # UR2 / UR-edit / (also used as stride-like sampling for light metrics)


def out_path(metric):
    return os.path.join(OUT_DIR, f"full_{metric}.jsonl")


def load_done(metric):
    p = out_path(metric)
    done = set()
    if not os.path.exists(p):
        return done
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["video"], r["method"]))
    return done


def append_result(metric, row):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path(metric), "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def pick_instants(frames, fps, n=N_INSTANTS):
    duration = len(frames) / max(fps, 1e-6)
    out = []
    for s in (np.arange(n) + 0.5) * duration / n:
        i = min(len(frames) - 1, int(round(s * fps)))
        out.append(i)
    return out


def frames_480(path):
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    imgs = [Image.fromarray(f).resize((ew, eh), Image.BICUBIC) for f in frames]
    return imgs, fps, (ew, eh)


def aligned_pair(method, video, n=N_INSTANTS):
    """Source cropped like method + gen frame, short-side 480, n instants."""
    src, src_fps = read_all(path_of("Source", video))
    gen, gen_fps = read_all(path_of(method, video))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    if method == "Source":
        box = (0, 0, sw, sh)
    else:
        box = GEOMETRY[method](sw, sh, ow, oh)
    l, t, r, b = box
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    pairs = []
    for s in (np.arange(n) + 0.5) * duration / n:
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
        pairs.append((
            ref.resize((ew, eh), Image.BICUBIC),
            Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC),
        ))
    return pairs, (ew, eh)


# -------------------- color --------------------
def run_color(videos, methods):
    from metrics.color_consistency import frame_delta_e, ab_histogram, histogram_intersection, SIGMA
    sigma = SIGMA * 480 / 768  # same scale as color_compare
    done = load_done("color")
    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            pairs, size = aligned_pair(m, v, n=N_INSTANTS)
            refs = [np.asarray(a).astype(np.float64) for a, _ in pairs]
            gens = [np.asarray(b).astype(np.float64) for _, b in pairs]
            de = [frame_delta_e(r, g, sigma) for r, g in zip(refs, gens)]
            hi = histogram_intersection(ab_histogram(refs), ab_histogram(gens))
            row = {"video": v, "method": m, "eval_size": list(size), "sigma": sigma,
                   "delta_e": float(np.mean(de)), "ab_hist_intersection": float(hi),
                   "instants": len(de)}
            append_result("color", row)
            print(f"[color] {v:18s} {m:10s} dE {row['delta_e']:.3f} hist {row['ab_hist_intersection']:.4f}", flush=True)


# -------------------- clip --------------------
def run_clip(videos, methods):
    from diffsynth.metrics import CLIPMetric, ModelConfig
    metric = CLIPMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                 origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"),
        device="cuda")
    prompt = "A Japanese anime-style cartoon video with clean line work and flat cel-shaded colors"
    done = load_done("clip")
    BATCH = 32
    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            imgs, fps, size = frames_480(path_of(m, v))
            # score every frame would be slow on long videos; use same 4 instants + mean of those
            idxs = pick_instants(imgs, fps, N_INSTANTS)
            picked = [imgs[i] for i in idxs]
            scores = []
            for i in range(0, len(picked), BATCH):
                scores.extend(metric.compute(prompt, picked[i:i + BATCH]))
            row = {"video": v, "method": m, "eval_size": list(size), "frames_scored": len(scores),
                   "clip": float(np.mean(scores)), "clip_std": float(np.std(scores)), "prompt": prompt}
            append_result("clip", row)
            print(f"[clip] {v:18s} {m:10s} {row['clip']:.3f}", flush=True)


# -------------------- ur2 --------------------
def run_ur2(videos, methods):
    from diffsynth.metrics import ModelConfig, UnifiedReward2Metric
    metric = UnifiedReward2Metric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                 origin_file_pattern="UnifiedReward-2.0-qwen35-9b/model-*.safetensors"),
        processor_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                     origin_file_pattern="UnifiedReward-2.0-qwen35-9b/"),
        device="cuda")
    done = load_done("ur2")
    for v in videos:
        prompt = ur2_prompt(v)
        for m in methods:
            if (v, m) in done:
                continue
            imgs, fps, size = frames_480(path_of(m, v))
            idxs = pick_instants(imgs, fps, N_INSTANTS)
            frames = []
            for i in idxs:
                d = metric.evaluate(prompt, imgs[i])[0]
                frames.append({k: d.get(k) for k in ("alignment", "coherence", "style", "score")})
            ok = [f for f in frames if f.get("score") is not None]
            def mk(k):
                vals = [f[k] for f in ok if f.get(k) is not None]
                return float(np.mean(vals)) if vals else None
            row = {"video": v, "method": m, "prompt": prompt, "eval_size": list(size),
                   "instants": len(frames), "parsed": len(ok),
                   "alignment": mk("alignment"), "coherence": mk("coherence"),
                   "style": mk("style"), "score": mk("score"), "frames": frames}
            append_result("ur2", row)
            print(f"[ur2] {v:18s} {m:10s} a {row['alignment']} c {row['coherence']} s {row['style']} m {row['score']}", flush=True)


# -------------------- ur_edit --------------------
def run_ur_edit(videos, methods):
    from diffsynth.metrics import ModelConfig, UnifiedRewardEditMetric
    metric = UnifiedRewardEditMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                 origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors"),
        processor_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
                                     origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/"),
        device="cuda")
    done = load_done("ur_edit")
    # Source vs Source is degenerate; still score gen methods only + optionally skip Source
    methods = [m for m in methods if m != "Source"]
    for v in videos:
        instr = edit_prompt(v)
        for m in methods:
            if (v, m) in done:
                continue
            pairs, size = aligned_pair(m, v, n=N_INSTANTS)
            frames = []
            for ref, gen in pairs:
                d = metric.evaluate(instr, [ref, gen], task="edit_pointwise_score")[0]
                frames.append({k: d.get(k) for k in ("editing_success", "overediting", "score", "reasoning")})
            ok = [f for f in frames if f.get("score") is not None]
            def mk(k):
                vals = [f[k] for f in ok if f.get(k) is not None]
                return float(np.mean(vals)) if vals else None
            row = {"video": v, "method": m, "instruction": instr, "eval_size": list(size),
                   "instants": len(frames), "parsed": len(ok),
                   "editing_success": mk("editing_success"), "overediting": mk("overediting"),
                   "score": mk("score"), "frames": frames}
            append_result("ur_edit", row)
            print(f"[ur_edit] {v:18s} {m:10s} score {row['score']} suc {row['editing_success']} over {row['overediting']}", flush=True)


# -------------------- temporal: csd + smoothness --------------------
def run_temporal(videos, methods, do_csd=True, do_smooth=True):
    from metrics.compare.temporal_compare import (
        AMT_CKPT, allow_amt_checkpoint, load_csd, csd_consistency, write_lossless,
    )
    csd_model = tf = ms = None
    if do_csd:
        csd_model, tf = load_csd()
    if do_smooth:
        from vbench.motion_smoothness import MotionSmoothness
        allow_amt_checkpoint()
        cfg = os.path.join(os.path.dirname(__import__("vbench").__file__), "third_party/amt/cfgs/AMT-S.yaml")
        ms = MotionSmoothness(cfg, AMT_CKPT, "cuda")

    done_csd = load_done("csd") if do_csd else set()
    done_sm = load_done("smoothness") if do_smooth else set()

    with tempfile.TemporaryDirectory() as tmp:
        for v in videos:
            for m in methods:
                need_csd = do_csd and (v, m) not in done_csd
                need_sm = do_smooth and (v, m) not in done_sm
                if not need_csd and not need_sm:
                    continue
                frames_u8, fps = _native_480(path_of(m, v))
                if need_csd:
                    cmean, cmin = csd_consistency(csd_model, tf, frames_u8)
                    row = {"video": v, "method": m, "fps": fps, "frames": len(frames_u8),
                           "csd_consistency": cmean, "csd_min": cmin}
                    append_result("csd", row)
                    print(f"[csd] {v:18s} {m:10s} {cmean:.4f}", flush=True)
                if need_sm:
                    mp4 = os.path.join(tmp, f"{v}__{m}.mp4")
                    write_lossless(frames_u8, mp4, fps)
                    sm = float(ms.motion_score(mp4))
                    row = {"video": v, "method": m, "fps": fps, "frames": len(frames_u8),
                           "motion_smoothness": sm}
                    append_result("smoothness", row)
                    print(f"[smooth] {v:18s} {m:10s} {sm:.5f}", flush=True)


def _native_480(path):
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    out = [np.asarray(Image.fromarray(f).resize((ew, eh), Image.BICUBIC)) for f in frames]
    return out, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True,
                    choices=["color", "clip", "ur2", "ur_edit", "csd", "smoothness", "temporal"])
    ap.add_argument("--shard", default="0/1", help="i/n zero-based shard of the video list")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(all_videos(), i, n)
    methods = a.methods.split(",")
    print(f"metric={a.metric} shard={a.shard} videos={len(videos)} methods={methods}", flush=True)
    print("video list:", videos, flush=True)

    if a.metric == "color":
        run_color(videos, methods)
    elif a.metric == "clip":
        run_clip(videos, methods)
    elif a.metric == "ur2":
        run_ur2(videos, methods)
    elif a.metric == "ur_edit":
        run_ur_edit(videos, methods)
    elif a.metric == "csd":
        run_temporal(videos, methods, do_csd=True, do_smooth=False)
    elif a.metric == "smoothness":
        run_temporal(videos, methods, do_csd=False, do_smooth=True)
    elif a.metric == "temporal":
        run_temporal(videos, methods, do_csd=True, do_smooth=True)
    print("DONE", a.metric, a.shard, flush=True)


if __name__ == "__main__":
    main()
