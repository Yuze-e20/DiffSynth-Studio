"""Line-art F1 for 9blocks vs prior baselines on the same DAVIS-10 set."""
import json
import os
import sys

import numpy as np
from PIL import Image

from metrics.compare.lineart_compare import (
    DETECT_RESOLUTION, GEOMETRY, OTSU_SAMPLES, ROOT, STRIDE_S,
    box_center_crop_diffsynth, eval_size, lineart, otsu_threshold, read_all,
)
from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray

# same 10 videos as lineart_davis10.json
VIDEOS = [
    "breakdance-flare", "gold-fish", "kite-walk", "lady-running",
    "motocross-bumps", "motorbike", "night-race", "rhino",
    "rollerblade", "scooter-gray",
]

# extend geometry
GEOMETRY = dict(GEOMETRY)
GEOMETRY["9blocks"] = box_center_crop_diffsynth
GEOMETRY["Ours"] = box_center_crop_diffsynth


def path_of(method, v):
    if method == "9blocks":
        return f"{ROOT}/9blocks/{v}.mp4"
    if method == "Ours":
        p = f"{ROOT}/DAVIS2017_Diffutoon2/{v}_nf328_ctx1g_s20_u05_t05.mp4"
        if os.path.exists(p):
            return p
    if method == "Ref2VA":
        return f"{ROOT}/Ref2VA/{v}_nf328_ctx1g.mp4"
    return f"{ROOT}/{method}/{v}.mp4"


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
    return {
        "video": v, "method": method, "out_size": [ow, oh], "eval_size": [ew, eh],
        "fps": gen_fps, "instants": len(instants),
        "precision": float(p), "recall": float(rc), "f1": float(f),
    }


def main():
    # load detector like lineart_compare
    os.chdir("/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")
    from diffsynth.utils.controlnet.annotator import Annotator
    # try common model paths
    for mp in ("models/Annotators", "/root/models/Annotators",
               "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/Annotators"):
        if os.path.isdir(mp):
            model_path = mp
            break
    else:
        model_path = "models/Annotators"
    print("annotator path", model_path, flush=True)
    det = Annotator("lineart", model_path=model_path, detect_resolution=DETECT_RESOLUTION, device="cuda")

    methods = ["9blocks", "Ours", "AnyV2V", "Kiwi-Edit", "EchoStyle", "PickStyle"]
    # only methods with files
    methods = [m for m in methods if all(os.path.exists(path_of(m, v)) for v in VIDEOS)]
    print("methods", methods, flush=True)
    print("videos", VIDEOS, flush=True)

    results = []
    out = f"{ROOT}/_metrics/lineart_9blocks_davis10.json"
    for v in VIDEOS:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            r = score_method(det, m, v, src, src_fps)
            results.append(r)
            print(f"{v:18s} {m:12s} F1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  "
                  f"({r['instants']} inst)", flush=True)
            with open(out, "w") as fh:
                json.dump({"videos": VIDEOS, "results": results, "note": "same 10 DAVIS as lineart_davis10"}, fh, indent=1)

    print("\n=== mean over", len(VIDEOS), "videos ===", flush=True)
    ranking = []
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        f1 = float(np.mean([r["f1"] for r in rs]))
        p = float(np.mean([r["precision"] for r in rs]))
        rc = float(np.mean([r["recall"] for r in rs]))
        ranking.append((f1, m, p, rc, len(rs)))
        print(f"  {m:12s} F1 {f1:.4f}  P {p:.4f}  R {rc:.4f}  n={len(rs)}", flush=True)
    ranking.sort(reverse=True)
    print("\nranking:", [(m, f"{f1:.4f}") for f1, m, *_ in ranking], flush=True)
    best = ranking[0]
    nine = next(x for x in ranking if x[1] == "9blocks")
    print(f"\n9blocks F1={nine[0]:.4f} vs previous-best-in-this-run {best[1]} F1={best[0]:.4f}  "
          f"delta={nine[0]-best[0]:+.4f}", flush=True)
    # also vs historical AnyV2V from old json
    old = json.load(open(f"{ROOT}/_metrics/lineart_davis10.json"))
    old_by = {}
    for r in old["results"]:
        old_by.setdefault(r["method"], []).append(r["f1"])
    print("historical davis10 means:", {m: float(np.mean(vs)) for m, vs in old_by.items()}, flush=True)
    hist_best_m = max(old_by, key=lambda m: np.mean(old_by[m]))
    hist_best = float(np.mean(old_by[hist_best_m]))
    print(f"vs historical best {hist_best_m}={hist_best:.4f}: 9blocks-hist_best={nine[0]-hist_best:+.4f}", flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
