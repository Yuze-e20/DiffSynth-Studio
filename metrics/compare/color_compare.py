"""Chroma CIEDE2000 and ab-histogram intersection across methods.

    python -m metrics.compare.color_compare --out results.json

Scoring is metrics.color_consistency unchanged -- a/b only with L forced to a
constant, Gaussian on a/b, CIEDE2000, plus the unblurred ab histogram
intersection -- imported, not reimplemented.

Alignment is lineart_compare's, imported: each method's reference is cut from the
source exactly as that method's preprocessing cut it, both frames are resized to
the method's evaluation grid (short side 480, its own aspect), and every method
is scored at the same instants of the source.

Sigma is a pixel distance. The shipped SIGMA = 15 was chosen on the 1344x768
output grid; on a short-side-480 grid the same physical blur is 15 * 480/768 =
9.375. Using 15 here would blur 1.6x further in source terms, which pushes the
metric toward comparing whole-frame average colour (see the module docstring of
color_consistency).
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image

from metrics.color_consistency import SIGMA, ab_histogram, frame_delta_e, histogram_intersection
from metrics.compare.lineart_compare import GEOMETRY, ROOT, SHORT, STRIDE_S, eval_size, path_of, read_all

SIGMA_EVAL = SIGMA * SHORT / 768   # 9.375: the 1344x768 sigma carried to the short-side-480 grid


def score_method(method, v, src, src_fps, sigma):
    gen, gen_fps = read_all(path_of(method, v))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    instants = np.arange(0, duration - 1e-6, STRIDE_S)
    refs, gens = [], []
    for s in instants:
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b))))
        refs.append(np.asarray(ref.resize((ew, eh), Image.BICUBIC)).astype(np.float64))
        gens.append(np.asarray(Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)).astype(np.float64))
    de = [frame_delta_e(rf, g, sigma) for rf, g in zip(refs, gens)]
    hi = histogram_intersection(ab_histogram(refs), ab_histogram(gens))
    return {"video": v, "method": method, "eval_size": [ew, eh], "instants": len(instants), "sigma": sigma,
            "delta_e": float(np.mean(de)), "delta_e_std": float(np.std(de)), "ab_hist_intersection": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/color_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=SIGMA_EVAL)
    ap.add_argument("--methods", default="Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, f"· sigma {a.sigma:g} on the short-side-{SHORT} grid", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            r = score_method(m, v, src, src_fps, a.sigma)
            results.append(r)
            print(f"{v:18s} {m:10s} dE {r['delta_e']:6.3f}  hist {r['ab_hist_intersection']:.4f}  "
                  f"({r['instants']} instants)", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "sigma": a.sigma, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos  (dE lower is better, hist higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        print(f"  {m:10s} dE {np.mean([r['delta_e'] for r in rs]):6.3f}  "
              f"hist {np.mean([r['ab_hist_intersection'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
