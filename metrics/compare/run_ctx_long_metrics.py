"""9 metrics on 21 long videos for ctx1(9blocks)/ctx22/56/90/indep."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from PIL import Image

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
OUT_DIR = os.path.join(ROOT, "_metrics")
LONG = [
    "8","10","12","13","14","30","32","33","34","36",
    "47","61","66","73","78","85","91","92","98","100","105",
]
# method name -> path template
PATHS = {
    "ctx1": os.path.join(ROOT, "9blocks", "{v}.mp4"),
    "ctx22": os.path.join(ROOT, "ablation_ctx_long", "ctx22", "{v}.mp4"),
    "ctx56": os.path.join(ROOT, "ablation_ctx_long", "ctx56", "{v}.mp4"),
    "ctx90": os.path.join(ROOT, "ablation_ctx_long", "ctx90", "{v}.mp4"),
    "indep": os.path.join(ROOT, "ablation_ctx_long", "indep", "{v}.mp4"),
}
METHODS = list(PATHS.keys())
N_INSTANTS = 4
BATCH = 32

from metrics.compare.lineart_compare import (
    DETECT_RESOLUTION, GEOMETRY, OTSU_SAMPLES, STRIDE_S,
    box_center_crop_diffsynth, eval_size, lineart, read_all,
)
from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray, otsu_threshold
from metrics.compare.registry import STYLE_PREFIX, global_caption, ur2_prompt, edit_prompt
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
    return os.path.join(OUT_DIR, f"ctx_long_{metric}.jsonl")


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
        fh.write(json.dumps(row) + "\n")
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
                scores.extend(metric.compute(prompt, picked[i:i+BATCH]))
            row = {"video": v, "method": m, "clip": float(np.mean(scores)),
                   "clip_std": float(np.std(scores)), "frames_scored": len(scores)}
            append("clip", row)
            print(f"[clip] {v} {m} {row['clip']:.3f}", flush=True)


def run_color(videos, methods):
    from skimage.color import rgb2lab, deltaE_ciede2000
    done = load_done("color")
    def ab_hist(a, b, bins=32):
        def h(x):
            lab = rgb2lab(x.astype(np.float64) / 255.0)
            ab = lab[..., 1:3].reshape(-1, 2)
            hh, _ = np.histogramdd(ab, bins=bins, range=[[-128, 127], [-128, 127]])
            hh = hh.astype(np.float64); hh /= (hh.sum() + 1e-12)
            return hh
        return float(np.minimum(h(a), h(b)).sum())
    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            pairs, size = aligned_pair(m, v)
            des, hists = [], []
            for ref, gen in pairs:
                ra, ga = np.asarray(ref), np.asarray(gen)
                if ra.shape != ga.shape:
                    ga = np.asarray(Image.fromarray(ga).resize((ra.shape[1], ra.shape[0]), Image.BICUBIC))
                des.append(float(np.mean(deltaE_ciede2000(
                    rgb2lab(ra.astype(np.float64)/255.0), rgb2lab(ga.astype(np.float64)/255.0)))))
                hists.append(ab_hist(ra, ga))
            row = {"video": v, "method": m, "delta_e": float(np.mean(des)),
                   "ab_hist_intersection": float(np.mean(hists)), "instants": len(des)}
            append("color", row)
            print(f"[color] {v} {m} dE={row['delta_e']:.3f}", flush=True)


def run_ur2(videos, methods):
    # reuse rfm.run_ur2 with monkeypatch
    rfm.path_of = path_of
    rfm.GEN_METHODS = methods
    rfm.METHODS = methods
    rfm.out_path = lambda metric: out_path(metric)
    rfm.load_done = load_done
    rfm.append_result = append
    rfm.run_ur2(videos, methods)


def run_ur_edit(videos, methods):
    rfm.path_of = path_of
    rfm.out_path = lambda metric: out_path(metric)
    rfm.load_done = load_done
    rfm.append_result = append
    rfm.run_ur_edit(videos, methods)


def run_temporal(videos, methods):
    rfm.path_of = path_of
    rfm.out_path = lambda metric: out_path(metric)
    rfm.load_done = load_done
    rfm.append_result = append
    rfm.run_temporal(videos, methods, do_csd=True, do_smooth=True)


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
            duration = min(len(src)/src_fps, len(gen)/gen_fps)
            instants = np.arange(0, duration - 1e-6, STRIDE_S)
            ref_maps, gen_maps = [], []
            for s in instants:
                si = min(len(src)-1, int(round(s*src_fps)))
                gi = min(len(gen)-1, int(round(s*gen_fps)))
                ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
                ref_maps.append(lineart(det, ref.resize((ew, eh), Image.BICUBIC)))
                gen_maps.append(lineart(det, Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)))
            t_ref = otsu_threshold([x[..., None] for x in ref_maps[:OTSU_SAMPLES]])
            t_gen = otsu_threshold([x[..., None] for x in gen_maps[:OTSU_SAMPLES]])
            rows = [boundary_f(gray(g) > t_gen, gray(rf) > t_ref, TOLERANCE)
                    for rf, g in zip(ref_maps, gen_maps)]
            p, rc, f = np.array(rows).mean(0)
            row = {"video": v, "method": m, "precision": float(p), "recall": float(rc),
                   "f1": float(f), "instants": len(instants)}
            append("lineart", row)
            print(f"[lineart] {v} {m} F1={f:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True,
                    choices=["clip", "color", "ur2", "ur_edit", "temporal", "lineart"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(LONG, i, n)
    print(f"ctx_long metric={a.metric} shard={a.shard} n={len(videos)} methods={METHODS}", flush=True)
    # patch rfm path for aligned helpers
    rfm.path_of = path_of
    if a.metric == "clip":
        run_clip(videos, METHODS)
    elif a.metric == "color":
        run_color(videos, METHODS)
    elif a.metric == "ur2":
        run_ur2(videos, METHODS)
    elif a.metric == "ur_edit":
        run_ur_edit(videos, METHODS)
    elif a.metric == "temporal":
        run_temporal(videos, METHODS)
    elif a.metric == "lineart":
        run_lineart(videos, METHODS)
    print("DONE", a.metric, a.shard, flush=True)


if __name__ == "__main__":
    main()
