"""Full-set lineart F1 for multiple methods. Resumable jsonl. Shard by video."""
import argparse, json, os
import numpy as np
from PIL import Image

from metrics.compare.lineart_compare import (
    DETECT_RESOLUTION, GEOMETRY, OTSU_SAMPLES, ROOT, STRIDE_S,
    box_center_crop_diffsynth, box_stretch, box_scale_cover_crop, box_kiwi,
    eval_size, lineart, read_all,
)
from metrics.lineart_boundary_f import TOLERANCE, boundary_f, gray, otsu_threshold
from metrics.compare.registry import all_videos, shard

GEOMETRY = dict(GEOMETRY)
GEOMETRY.update({
    "9blocks": box_center_crop_diffsynth,
    "Seedance25": box_center_crop_diffsynth,
    "Diffutoon": box_center_crop_diffsynth,
    "Ref2VA": box_center_crop_diffsynth,
    "Ours": box_center_crop_diffsynth,
})

OUT = f"{ROOT}/_metrics/full_lineart.jsonl"
METHODS_DEFAULT = "Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V,Ref2VA,Diffutoon,Seedance25,9blocks"


def path_of(method, v):
    if method == "Ours":
        p1 = f"{ROOT}/DAVIS2017_Diffutoon2/{v}_nf328_ctx1g_s20_u05_t05.mp4"
        if os.path.exists(p1):
            return p1
        for p in (f"{ROOT}/pixabay100_nf328_ctx1g_s20_u05_t05/{v}.mp4",
                  f"{ROOT}/pixabay100_nf328_ctx1g_s20_u05_t05/{v}_nf328_ctx1g_s20_u05_t05.mp4"):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Ours missing {v}")
    if method == "Ref2VA":
        return f"{ROOT}/Ref2VA/{v}_nf328_ctx1g.mp4"
    return f"{ROOT}/{method}/{v}.mp4"


def load_done():
    done = set()
    if not os.path.exists(OUT):
        return done
    for line in open(OUT):
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["video"], r["method"]))
    return done


def append(row):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def score_one(det, method, v, src, src_fps):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--methods", default=METHODS_DEFAULT)
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    methods = a.methods.split(",")
    videos = shard(all_videos(), i, n)
    print(f"lineart shard {a.shard}: {len(videos)} videos x {methods}", flush=True)

    from diffsynth.utils.controlnet.annotator import Annotator
    det = Annotator("lineart", model_path="models/Annotators",
                    detect_resolution=DETECT_RESOLUTION, device="cuda")
    done = load_done()
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            if (v, m) in done:
                continue
            p = path_of(m, v)
            if not os.path.exists(p):
                print(f"[skip missing] {v} {m}", flush=True)
                continue
            try:
                row = score_one(det, m, v, src, src_fps)
            except Exception as e:
                print(f"[FAIL] {v} {m}: {e}", flush=True)
                continue
            append(row)
            print(f"{v:18s} {m:12s} F1 {row['f1']:.4f}", flush=True)
    print("DONE", a.shard, flush=True)


if __name__ == "__main__":
    main()
