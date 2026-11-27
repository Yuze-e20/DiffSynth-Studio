"""9 metrics on 70 videos for blur-sigma ablation (sigma5/8/12/15/25)."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
OUT_DIR = os.path.join(ROOT, "_metrics")
SIGMAS = [5, 8, 12, 15, 25]
PATHS = {f"sigma{s}": os.path.join(ROOT, "ablation_sigma", f"sigma{s}", "{v}.mp4") for s in SIGMAS}
# optional: main 9blocks as sigma20 reference (already in full_*.jsonl as 9blocks)
METHODS = list(PATHS.keys())
BATCH = 32
N_INSTANTS = 4

from metrics.compare.lineart_compare import (
    DETECT_RESOLUTION, GEOMETRY, OTSU_SAMPLES, STRIDE_S,
    box_center_crop_diffsynth, eval_size, lineart, read_all,
)
from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray, otsu_threshold
from metrics.compare.registry import all_videos
from metrics.compare import run_full_metrics as rfm

for m in METHODS:
    GEOMETRY[m] = box_center_crop_diffsynth


def path_of(method, video):
    if method == "Source":
        return os.path.join(ROOT, "all", f"{video}.mp4")
    p = PATHS[method].format(v=video)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def out_path(metric):
    return os.path.join(OUT_DIR, f"sigma_ablation_{metric}.jsonl")


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


def pick_instants(frames, fps, n=N_INSTANTS):
    duration = len(frames) / max(fps, 1e-6)
    return [min(len(frames) - 1, int(round(s * fps)))
            for s in (np.arange(n) + 0.5) * duration / n]


def frames_480(path):
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    imgs = [Image.fromarray(f).resize((ew, eh), Image.BICUBIC) for f in frames]
    return imgs, fps, (ew, eh)


def aligned_pair(method, video, n=N_INSTANTS):
    src, src_fps = read_all(path_of("Source", video))
    gen, gen_fps = read_all(path_of(method, video))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
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


def patch_rfm():
    rfm.path_of = path_of
    rfm.GEOMETRY = GEOMETRY
    rfm.out_path = out_path
    rfm.load_done = load_done
    rfm.append_result = append
    rfm.aligned_pair = aligned_pair
    rfm.frames_480 = frames_480
    rfm.GEN_METHODS = METHODS
    rfm.METHODS = METHODS


def run_clip(videos, methods):
    from diffsynth.metrics import CLIPMetric, ModelConfig
    metric = CLIPMetric.from_pretrained(
        model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"),
        device="cuda")
    prompt = "A Japanese anime-style cartoon video with clean line work and flat cel-shaded colors"
    done = load_done("clip")
    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            imgs, fps, size = frames_480(path_of(m, v))
            idxs = pick_instants(imgs, fps)
            picked = [imgs[i] for i in idxs]
            scores = []
            for i in range(0, len(picked), BATCH):
                scores.extend(metric.compute(prompt, picked[i:i + BATCH]))
            row = {"video": v, "method": m, "clip": float(np.mean(scores)),
                   "clip_std": float(np.std(scores)), "frames_scored": len(scores),
                   "eval_size": list(size), "prompt": prompt}
            append("clip", row)
            print(f"[clip] {v:20s} {m:10s} {row['clip']:.3f}", flush=True)


def run_lineart(videos, methods):
    from diffsynth.utils.controlnet.annotator import Annotator
    det = Annotator("lineart", model_path="models/Annotators",
                    detect_resolution=DETECT_RESOLUTION, device="cuda")
    done = load_done("lineart")
    for v in videos:
        src, src_fps = read_all(path_of("Source", v))
        for m in methods:
            if (v, m) in done:
                continue
            gen, gen_fps = read_all(path_of(m, v))
            sh, sw = src[0].shape[:2]
            oh, ow = gen[0].shape[:2]
            ew, eh = eval_size(ow, oh)
            l, t, r, b = GEOMETRY[m](sw, sh, ow, oh)
            duration = min(len(src) / src_fps, len(gen) / gen_fps)
            instants = np.arange(0, duration - 1e-6, STRIDE_S)
            ref_maps, gen_maps = [], []
            for s in instants:
                si = min(len(src) - 1, int(round(s * src_fps)))
                gi = min(len(gen) - 1, int(round(s * gen_fps)))
                ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
                ref_maps.append(lineart(det, ref.resize((ew, eh), Image.BICUBIC)))
                gen_maps.append(lineart(det, Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)))
            t_ref = otsu_threshold([x[..., None] for x in ref_maps[:OTSU_SAMPLES]])
            t_gen = otsu_threshold([x[..., None] for x in gen_maps[:OTSU_SAMPLES]])
            rows = [boundary_f(gray(g) > t_gen, gray(rf) > t_ref, TOLERANCE)
                    for rf, g in zip(ref_maps, gen_maps)]
            p, rc, f = np.array(rows).mean(0)
            row = {"video": v, "method": m, "precision": float(p), "recall": float(rc),
                   "f1": float(f), "instants": len(instants), "eval_size": [ew, eh]}
            append("lineart", row)
            print(f"[lineart] {v:20s} {m:10s} F1={f:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True,
                    choices=["clip", "color", "ur2", "ur_edit", "temporal", "lineart"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(all_videos(), i, n)
    print(f"sigma_ablation metric={a.metric} shard={a.shard} nvid={len(videos)} methods={METHODS}", flush=True)
    patch_rfm = patch = True
    rfm.path_of = path_of
    # also patch GEOMETRY used inside full metrics aligned_pair if it imports from lineart_compare
    import metrics.compare.lineart_compare as lc
    for m in METHODS:
        lc.GEOMETRY[m] = box_center_crop_diffsynth
    rfm.out_path = out_path
    rfm.load_done = load_done
    rfm.append_result = append
    # override helpers that close over path_of
    rfm.aligned_pair = aligned_pair
    rfm.frames_480 = frames_480

    if a.metric == "clip":
        run_clip(videos, METHODS)
    elif a.metric == "color":
        rfm.run_color(videos, METHODS)
    elif a.metric == "ur2":
        rfm.run_ur2(videos, METHODS)
    elif a.metric == "ur_edit":
        rfm.run_ur_edit(videos, METHODS)
    elif a.metric == "temporal":
        rfm.run_temporal(videos, METHODS, do_csd=True, do_smooth=True)
    elif a.metric == "lineart":
        run_lineart(videos, METHODS)
    print("DONE", a.metric, a.shard, flush=True)


if __name__ == "__main__":
    main()
