"""Ablation_davis40 metrics: lineart F1, CLIP, color (dE+hist), CSD on 7 configs × DAVIS-40."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from PIL import Image

from metrics.compare.lineart_compare import (
    DETECT_RESOLUTION, GEOMETRY, OTSU_SAMPLES, ROOT, STRIDE_S,
    box_center_crop_diffsynth, eval_size, lineart, read_all,
)
from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray, otsu_threshold

ABLATION_ROOT = f"{ROOT}/ablation_davis40"
METHODS = [
    "s1.0_0.0", "s0.0_1.0", "s0.5_0.0", "s0.0_0.5",
    "s0.5_1.0", "s1.0_0.5", "s0.5_0.5_blur-rgb",
]
N_INSTANTS = 4
BATCH = 32
for m in METHODS:
    GEOMETRY[m] = box_center_crop_diffsynth


def davis_videos():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(f"{ROOT}/DAVIS2017_origin") if f.endswith(".mp4"))


def shard(items, i, n):
    items = list(items)
    a, b = i * len(items) // n, (i + 1) * len(items) // n
    return items[a:b]


def path_of(method, v):
    return f"{ABLATION_ROOT}/{method}/{v}.mp4"


def src_path(v):
    return f"{ROOT}/DAVIS2017_origin/{v}.mp4"


def load_done(path):
    d = set()
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                d.add((r["video"], r["method"]))
    return d


def append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def pick_instants(frames, fps, n=N_INSTANTS):
    duration = len(frames) / max(fps, 1e-6)
    out = []
    for s in (np.arange(n) + 0.5) * duration / n:
        out.append(min(len(frames) - 1, int(round(s * fps))))
    return out


def frames_480(path):
    frames, fps = read_all(path)
    oh, ow = frames[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    imgs = [Image.fromarray(f).resize((ew, eh), Image.BICUBIC) for f in frames]
    return imgs, fps, (ew, eh)


def aligned_pair(method, video, n=N_INSTANTS):
    src, src_fps = read_all(src_path(video))
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


def run_lineart(videos, methods):
    from diffsynth.utils.controlnet.annotator import Annotator
    det = Annotator("lineart", model_path="models/Annotators",
                    detect_resolution=DETECT_RESOLUTION, device="cuda")
    out = f"{ROOT}/_metrics/ablation_davis40_lineart.jsonl"
    done = load_done(out)
    for v in videos:
        src, src_fps = read_all(src_path(v))
        for m in methods:
            if (v, m) in done:
                continue
            if not os.path.exists(path_of(m, v)):
                print(f"[skip] {v} {m}", flush=True)
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
            append(out, row)
            print(f"[lineart] {v:18s} {m:28s} F1 {row['f1']:.4f}", flush=True)


def run_clip(videos, methods):
    from diffsynth.metrics import CLIPMetric, ModelConfig
    out = f"{ROOT}/_metrics/ablation_davis40_clip.jsonl"
    done = load_done(out)
    metric = CLIPMetric.from_pretrained(
        model_config=ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"),
        device="cuda")
    prompt = "A Japanese anime-style cartoon video with clean line work and flat cel-shaded colors"
    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            if not os.path.exists(path_of(m, v)):
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
            append(out, row)
            print(f"[clip] {v:18s} {m:28s} {row['clip']:.3f}", flush=True)


def run_color(videos, methods):
    from skimage.color import rgb2lab, deltaE_ciede2000
    out = f"{ROOT}/_metrics/ablation_davis40_color.jsonl"
    done = load_done(out)

    def ab_hist_intersection(a, b, bins=32):
        def hist(x):
            lab = rgb2lab(x.astype(np.float64) / 255.0)
            ab = lab[..., 1:3].reshape(-1, 2)
            h, _ = np.histogramdd(ab, bins=bins, range=[[-128, 127], [-128, 127]])
            h = h.astype(np.float64)
            h /= (h.sum() + 1e-12)
            return h
        return float(np.minimum(hist(a), hist(b)).sum())

    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            if not os.path.exists(path_of(m, v)):
                continue
            pairs, size = aligned_pair(m, v)
            des, hists = [], []
            for ref, gen in pairs:
                ra, ga = np.asarray(ref), np.asarray(gen)
                if ra.shape != ga.shape:
                    ga = np.asarray(Image.fromarray(ga).resize((ra.shape[1], ra.shape[0]), Image.BICUBIC))
                lab_r = rgb2lab(ra.astype(np.float64) / 255.0)
                lab_g = rgb2lab(ga.astype(np.float64) / 255.0)
                des.append(float(np.mean(deltaE_ciede2000(lab_r, lab_g))))
                hists.append(ab_hist_intersection(ra, ga))
            row = {"video": v, "method": m, "eval_size": list(size), "instants": len(des),
                   "delta_e": float(np.mean(des)), "ab_hist_intersection": float(np.mean(hists))}
            append(out, row)
            print(f"[color] {v:18s} {m:28s} dE {row['delta_e']:.3f} hist {row['ab_hist_intersection']:.4f}", flush=True)


def run_csd(videos, methods):
    """CSD style consistency vs first frame (same as full metrics)."""
    import sys
    import torch
    import torchvision.transforms as T
    CSD_REPO = "/mnt/nas3/sunyuzework/Diffutoon-2/baselines/_metric_repos/CSD"
    CSD_CKPT = "/root/models/metrics/CSD-ViT-L/pytorch_model.bin"
    CLIP_DIR = "/root/models/metrics/clip"
    sys.path.insert(0, CSD_REPO)
    import models.clip as clip_pkg
    import models.clip.clip as clipmod
    sys.modules["clip"] = clip_pkg
    orig_load = clipmod.load
    clip_pkg.load = clipmod.load = lambda name, *a, **k: orig_load(name, *a, **{**k, "download_root": CLIP_DIR})
    from CSD.model import CSD_CLIP
    from CSD.utils import convert_state_dict
    model = CSD_CLIP("vit_large", "default")
    ck = torch.load(CSD_CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(convert_state_dict(sd), strict=False)
    model = model.to("cuda").eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    out = f"{ROOT}/_metrics/ablation_davis40_csd.jsonl"
    done = load_done(out)

    @torch.no_grad()
    def csd_consistency(frames):
        # subsample if very long
        if len(frames) > 64:
            idx = np.linspace(0, len(frames) - 1, 64).astype(int)
            frames = [frames[i] for i in idx]
        x = torch.stack([tf(Image.fromarray(f)) for f in frames]).cuda()
        with torch.cuda.amp.autocast():
            _, _, style = model(x)
        style = style.float()
        sims = (style[1:] @ style[0]).cpu().numpy()
        return float(sims.mean()), float(sims.min())

    for v in videos:
        for m in methods:
            if (v, m) in done:
                continue
            if not os.path.exists(path_of(m, v)):
                continue
            frames, fps = read_all(path_of(m, v))
            # short-side 480
            oh, ow = frames[0].shape[:2]
            ew, eh = eval_size(ow, oh)
            frames = [np.asarray(Image.fromarray(f).resize((ew, eh), Image.BICUBIC)) for f in frames]
            cmean, cmin = csd_consistency(frames)
            row = {"video": v, "method": m, "csd_consistency": cmean, "csd_min": cmin,
                   "frames": len(frames), "fps": fps}
            append(out, row)
            print(f"[csd] {v:18s} {m:28s} {cmean:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=["lineart", "clip", "color", "csd"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    videos = shard(davis_videos(), i, n)
    print(f"ablation metric={a.metric} shard={a.shard} nvid={len(videos)} methods={METHODS}", flush=True)
    if a.metric == "lineart":
        run_lineart(videos, METHODS)
    elif a.metric == "clip":
        run_clip(videos, METHODS)
    elif a.metric == "color":
        run_color(videos, METHODS)
    elif a.metric == "csd":
        run_csd(videos, METHODS)
    print("DONE", a.metric, a.shard, flush=True)


if __name__ == "__main__":
    main()
