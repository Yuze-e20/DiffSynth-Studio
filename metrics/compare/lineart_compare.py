"""Line-art boundary F across methods whose outputs differ in size, crop and fps.

    python -m metrics.compare.lineart_compare --out results.json

The scoring is metrics.lineart_boundary_f unchanged -- same detector (lineart,
detect_resolution 512), same pooled Otsu cut per track, same Euclidean
tolerance -- imported, not reimplemented. What this adds is the alignment:

  * Space. Each method's output is put back onto the source's own pixels by
    undoing its geometry: the reference frame is cut and stretched exactly as
    that method's preprocessing did (center crop for ours / PickStyle / AnyV2V,
    a plain stretch for EchoStyle and Kiwi). Then both are
    resized to one evaluation grid, short side 480 at the method's own aspect
    ratio, BEFORE line art is extracted -- so every method is measured with the
    same detector input size and the 1.5 px tolerance means the same thing.
  * Time. Every method is scored at the same instants of the source: every
    STRIDE_S seconds, each method takes its output frame nearest that instant.
    Methods at 8, 16 and 24 fps therefore contribute the same number of frames.
"""
import argparse
import json
import os
import random

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray, otsu_threshold

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_refined"
SHORT = 480            # evaluation grid: short side 480, aspect kept
STRIDE_S = 10 / 24     # score one instant every 10 source frames (the metric's stride at 24 fps)
OTSU_SAMPLES = 24
DETECT_RESOLUTION = 512


def path_of(method, v):
    if method == "Ours":
        return f"{ROOT}/DAVIS2017_Diffutoon2/{v}_nf328_ctx1g_s20_u05_t05.mp4"
    return f"{ROOT}/{method}/{v}.mp4"


# --- each method's own geometry, as a source-pixel crop box (l, t, r, b) -------
def box_center_crop_diffsynth(sw, sh, ow, oh):
    """diffsynth crop_and_resize (ours): crop to the output aspect, int() sizes."""
    if sh / sw < oh / ow:
        cw = int(sh / oh * ow); l = (sw - cw) // 2
        return (l, 0, l + cw, sh)
    ch = int(sw / ow * oh); t = (sh - ch) // 2
    return (0, t, sw, t + ch)


def box_stretch(sw, sh, ow, oh):
    """EchoStyle eval_long: F.interpolate straight to 832x480, nothing cropped."""
    return (0, 0, sw, sh)


def box_scale_cover_crop(sw, sh, ow, oh):
    """Scale to cover ow x oh, then center crop -- VACE resize_crop (PickStyle) and
    AnyV2V prepare_video (center_crop=True). Mapped back to source pixels."""
    s = max(ow / sw, oh / sh)
    cw, ch = ow / s, oh / s
    l, t = (sw - cw) / 2, (sh - ch) / 2
    return (l, t, l + cw, t + ch)


def box_kiwi(sw, sh, ow, oh):
    """Kiwi wrapper: the whole frame resized to its 32-snapped size (854x480 ->
    832x480), a slight stretch with nothing cropped -- same kind as EchoStyle."""
    return (0, 0, sw, sh)


GEOMETRY = {"Ours": box_center_crop_diffsynth, "EchoStyle": box_stretch,
            "PickStyle": box_scale_cover_crop, "AnyV2V": box_scale_cover_crop,
            "Kiwi-Edit": box_kiwi, "Ref2VA": box_center_crop_diffsynth,
            "Diffutoon": box_center_crop_diffsynth, "Seedance25": box_center_crop_diffsynth,
            "9blocks": box_center_crop_diffsynth}


def eval_size(ow, oh):
    """Short side SHORT at the output's own aspect ratio."""
    if ow >= oh:
        return round(ow * SHORT / oh), SHORT
    return SHORT, round(oh * SHORT / ow)


def read_all(path):
    r = imageio.get_reader(path)
    fps = float(r.get_meta_data()["fps"])
    frames = [f for f in r]
    r.close()
    return frames, fps


def detector():
    from diffsynth.utils.controlnet.annotator import Annotator
    return Annotator("lineart", model_path="models/Annotators", detect_resolution=DETECT_RESOLUTION, device="cuda")


def lineart(det, img):
    return np.asarray(det(img).convert("L"))


def score_method(det, method, v, src, src_fps):
    gen, gen_fps = read_all(path_of(method, v))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    instants = np.arange(0, duration - 1e-6, STRIDE_S)
    ref_maps, gen_maps = [], []
    for s in instants:
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
        ref_maps.append(lineart(det, ref.resize((ew, eh), Image.BICUBIC)))
        gen_maps.append(lineart(det, Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)))
    t_ref = otsu_threshold([m[..., None] for m in ref_maps[:OTSU_SAMPLES]])
    t_gen = otsu_threshold([m[..., None] for m in gen_maps[:OTSU_SAMPLES]])
    rows = [boundary_f(gray(g) > t_gen, gray(rf) > t_ref, TOLERANCE) for rf, g in zip(ref_maps, gen_maps)]
    p, rc, f = np.array(rows).mean(0)
    return {"video": v, "method": method, "out_size": [ow, oh], "eval_size": [ew, eh], "fps": gen_fps,
            "crop_box_src": [round(x, 1) for x in (l, t, r, b)], "instants": len(instants),
            "precision": float(p), "recall": float(rc), "f1": float(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/lineart_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    det = detector()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            r = score_method(det, m, v, src, src_fps)
            results.append(r)
            print(f"{v:18s} {m:10s} F1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  "
                  f"({r['instants']} instants, eval {r['eval_size'][0]}x{r['eval_size'][1]})", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos")
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        print(f"  {m:10s} F1 {np.mean([r['f1'] for r in rs]):.4f}  P {np.mean([r['precision'] for r in rs]):.4f}  "
              f"R {np.mean([r['recall'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
