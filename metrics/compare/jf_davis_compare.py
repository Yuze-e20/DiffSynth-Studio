"""DAVIS J&F structure consistency via SAM 2 semi-supervised tracking.

    python metrics/compare/jf_davis_compare.py --out results.json

Official pieces, called as-is:
  * First-frame GT mask prompt (DAVIS 2017 semi-supervised protocol).
  * SAM 2 video predictor (facebookresearch/sam2 tools/vos_inference.py flow:
    init_state -> add_new_mask per object on frame 0 -> propagate_in_video).
  * davis2017-evaluation metrics: db_eval_iou (J) and db_eval_boundary (F),
    excluding the first and last frames, per-object then mean (J&F = (J+F)/2).

What we add is only geometry/time alignment shared with the other metrics:
each method's output is scored on short-side-480 at its own aspect; the source
crop that method used is applied to the GT masks with NEAREST resize; gen frame
k is paired with source frame round(k * src_fps / gen_fps).

For each method we track SAM2 on the *generated* video with the first-frame GT
prompt (warped to that geometry). Reference is the full GT sequence warped the
same way. "Source" is also scored: SAM2 on the real video with the same prompt,
as an upper-bound of tracker quality on photos.

PixaBay has no GT, so this metric stays DAVIS-only unless first-frame masks are
produced another way.
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

# avoid importing the local repo named "sam2" as the package
sys.path = [p for p in sys.path if not p.rstrip("/").endswith("_metric_repos")
            and not p.rstrip("/").endswith("_metric_repos/sam2")]
sys.path.insert(0, "/mnt/nas3/sunyuzework/Diffutoon-2/baselines/_metric_repos/davis2017-evaluation")
sys.path.insert(0, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")

from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017.utils import db_statistics
from metrics.compare.lineart_compare import GEOMETRY, ROOT, eval_size, path_of, read_all
from sam2.build_sam import build_sam2_video_predictor

ANN = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS2017_official/DAVIS/Annotations/480p"
CKPT = "/root/models/metrics/sam2/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_gt_masks(video):
    d = os.path.join(ANN, video)
    names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    masks = [np.array(Image.open(os.path.join(d, n))) for n in names]
    return masks  # list of HxW uint8, object ids >= 1


def warp_mask(mask, box, ew, eh):
    """Apply the same center-crop geometry as the RGB path, NEAREST resize."""
    l, t, r, b = box
    img = Image.fromarray(mask)
    img = img.crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
    return np.array(img.resize((ew, eh), Image.NEAREST))


def write_jpg_seq(frames, directory):
    os.makedirs(directory, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(directory, f"{i:05d}.jpg"), quality=95)


@torch.no_grad()
def track(predictor, frames, first_mask):
    """SAM2 semi-supervised track: first-frame multi-object masks -> full video."""
    tmp = tempfile.mkdtemp(prefix="sam2_")
    try:
        write_jpg_seq(frames, tmp)
        state = predictor.init_state(video_path=tmp)
        obj_ids = sorted(int(i) for i in set(first_mask.ravel().tolist()) if i > 0)
        for oid in obj_ids:
            predictor.add_new_mask(state, frame_idx=0, obj_id=oid, mask=(first_mask == oid))
        # frame_idx -> {obj_id: HxW bool}
        out = {}
        for frame_idx, oids, logits in predictor.propagate_in_video(state):
            out[frame_idx] = {
                int(oid): (logits[i, 0].cpu().numpy() > 0.0)
                for i, oid in enumerate(oids)
            }
        predictor.reset_state(state)
        return out, obj_ids
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_track(pred_by_frame, gt_masks_aligned, obj_ids):
    """Official semi-supervised J/F: drop first and last, per-object mean, then J&F."""
    T = len(gt_masks_aligned)
    if T < 3:
        # too short to drop ends; score all non-first frames
        idxs = list(range(1, T))
    else:
        idxs = list(range(1, T - 1))  # official [1:-1]
    Js, Fs = [], []
    per_obj = {}
    for oid in obj_ids:
        gt = np.stack([(gt_masks_aligned[i] == oid) for i in idxs], 0).astype(np.uint8)
        pr = np.stack([
            pred_by_frame.get(i, {}).get(oid, np.zeros_like(gt_masks_aligned[0], dtype=bool))
            for i in idxs
        ], 0).astype(np.uint8)
        j = db_eval_iou(gt, pr)
        f = db_eval_boundary(gt, pr)
        j_m, _, _ = db_statistics(j)
        f_m, _, _ = db_statistics(f)
        Js.append(j_m); Fs.append(f_m)
        per_obj[int(oid)] = {"J": float(j_m), "F": float(f_m), "JF": float(0.5 * (j_m + f_m))}
    J = float(np.mean(Js)) if Js else 0.0
    F = float(np.mean(Fs)) if Fs else 0.0
    return {"J": J, "F": F, "JF": 0.5 * (J + F), "n_frames_scored": len(idxs),
            "n_objects": len(obj_ids), "per_object": per_obj}


def prepare_method(method, video, src, src_fps, gt_masks):
    if method == "Source":
        gen, gen_fps = src, src_fps
        path = f"{ROOT}/all/{video}.mp4"
    else:
        path = path_of(method, video)
        gen, gen_fps = read_all(path)
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    # for Source, use identity geometry matching the eval grid of the source aspect
    if method == "Source":
        box = (0, 0, sw, sh)
        # stretch whole frame to eval size (no crop), same spirit as EchoStyle/Kiwi stretch
        frames = [np.asarray(Image.fromarray(f).resize((ew, eh), Image.BICUBIC)) for f in gen]
        gt_al = [np.asarray(Image.fromarray(m).resize((ew, eh), Image.NEAREST)) for m in gt_masks]
        # source frames are 1:1 with gt
        gt_for_gen = gt_al[:len(frames)]
        if len(gt_for_gen) < len(frames):
            gt_for_gen = gt_for_gen + [gt_for_gen[-1]] * (len(frames) - len(gt_for_gen))
        return frames, gt_for_gen, ew, eh

    box = GEOMETRY[method](sw, sh, ow, oh)
    frames = [np.asarray(Image.fromarray(f).resize((ew, eh), Image.BICUBIC)) for f in gen]
    # pair each gen frame with a source-time GT mask, then warp
    gt_for_gen = []
    for k in range(len(frames)):
        si = min(len(gt_masks) - 1, int(round(k * src_fps / gen_fps)))
        gt_for_gen.append(warp_mask(gt_masks[si], box, ew, eh))
    return frames, gt_for_gen, ew, eh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/jf_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Source,Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "")
                   for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2") if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)

    predictor = build_sam2_video_predictor(CFG, CKPT, device="cuda")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        gt_masks = load_gt_masks(v)
        print(f"\n=== {v}: {len(src)} src frames @ {src_fps:g}fps, "
              f"gt {len(gt_masks)} masks, objs {sorted(set(gt_masks[0].ravel()) - {0})}", flush=True)
        for m in methods:
            frames, gt_al, ew, eh = prepare_method(m, v, src, src_fps, gt_masks)
            first = gt_al[0]
            obj_ids = sorted(int(i) for i in set(first.ravel().tolist()) if i > 0)
            if not obj_ids:
                print(f"{v:18s} {m:10s} SKIP no objects in first-frame GT", flush=True)
                continue
            pred, _ = track(predictor, frames, first)
            sc = score_track(pred, gt_al, obj_ids)
            r = {"video": v, "method": m, "eval_size": [ew, eh], "frames": len(frames), **sc}
            results.append(r)
            print(f"{v:18s} {m:10s} J {sc['J']:.4f}  F {sc['F']:.4f}  J&F {sc['JF']:.4f}  "
                  f"({sc['n_frames_scored']} frames, {sc['n_objects']} objs, {ew}x{eh})", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "protocol": "semi-supervised first-frame GT + SAM2.1-L + official J&F",
                           "results": results}, fh, indent=1)

    print("\nmean over", len(videos), "videos (higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        if not rs:
            continue
        print(f"  {m:10s} J {np.mean([r['J'] for r in rs]):.4f}  F {np.mean([r['F'] for r in rs]):.4f}  "
              f"J&F {np.mean([r['JF'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
