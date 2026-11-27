"""ControlNet++'s lineart controllability metric, applied to our methods' videos.

    python -m metrics.compare.controlnetpp_lineart_compare --out results.json

Everything that decides the number is taken from liming-ai/ControlNet_Plus_Plus
(eval/eval.py + eval/eval_edge.py), not reimplemented:

  * detector: its LineDrawingModel (eval/utils.py) with its weights,
    awacke1/Image-to-Line-Drawings model.pth -- byte-identical to the
    informative-drawings sk_model.pth our pipeline's lineart annotator uses;
  * input: PIL image -> F.pil_to_tensor / 255, one forward pass, no extra
    normalisation, exactly as eval.py does for both the condition and the
    generated images;
  * GT side: condition = 1 - model(source); generated side: lineart =
    model(generated), saved as an image, then eval_edge.py's
    `pred = 1 - pred` on the uint8 PNG tensor;
  * score: torchmetrics MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
    on (pred/255).clip(0,1) vs (gt/255).clip(0,1); PSNR alongside.

Polarity. eval_edge.py inverts the generated lineart as `1 - pred` on the uint8
PNG tensor. That wraps rather than saturates: (1 - p) mod 256 = 257 - p for
p >= 2, i.e. the intended 255 - p plus 2/255, so it does give the white-on-black
polarity of the GT side's `1 - condition`. This script scores with 255 - p and
also records the literal uint8 result; the two agree to the third decimal.

Resolution. ControlNet++ scores square 512x512 images. Here every frame goes
through the same alignment as the other metrics -- the source cut the way each
method cut it, both at short side 480 at the method's own aspect -- and every
method is scored at the same instants of the source.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from metrics.compare.lineart_compare import GEOMETRY, ROOT, STRIDE_S, eval_size, path_of, read_all

CNPP = "/mnt/nas3/sunyuzework/Diffutoon-2/baselines/_metric_repos/ControlNet_Plus_Plus/eval"
WEIGHT = "/root/models/metrics/controlnetpp/lineart_model.pth"


def load_model(device="cuda"):
    sys.path.insert(0, CNPP)
    # eval/utils.py imports mmengine (segmentation) and canny_tools at module top;
    # only LineDrawingModel is needed, so read that class out without the rest.
    src = open(os.path.join(CNPP, "utils.py")).read()
    start = src.index("norm_layer = nn.InstanceNorm2d")   # ResidualBlock reads this module global
    end = src.index("class DoubleConvBlock")
    ns = {"torch": torch, "nn": torch.nn}
    exec(compile("import torch\nimport torch.nn as nn\n" + src[start:end], "controlnetpp_eval_utils", "exec"), ns)
    model = ns["LineDrawingModel"]()
    model.load_state_dict(torch.load(WEIGHT, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def lineart(model, img, device="cuda"):
    """eval.py: F.pil_to_tensor(img) / 255.0 -> model -> [1,H,W] in [0,1]."""
    x = (F.pil_to_tensor(img) / 255.0).unsqueeze(0).to(device)
    return model(x)[0]


def score_method(model, ssim, psnr, method, v, src, src_fps, official_uint8, device="cuda"):
    gen, gen_fps = read_all(path_of(method, v))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    gts, preds = [], []
    for s in np.arange(0, duration - 1e-6, STRIDE_S):
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b)))).resize((ew, eh), Image.BICUBIC)
        out = Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)
        # GT, as eval.py builds it: 1 - model(condition), to an 'L' image, saved as PNG
        gt = np.array(F.to_pil_image((1 - lineart(model, ref, device)).clamp(0, 1).cpu(), "L"))
        # generated, as eval.py saves it: model(image) -> PIL -> PNG
        pr = np.array(F.to_pil_image(lineart(model, out, device).clamp(0, 1).cpu()).convert("L"))
        gts.append(torch.from_numpy(gt)); preds.append(torch.from_numpy(pr))
    gt = torch.stack(gts)[:, None].to(device)            # uint8, as eval_edge.py loads the PNGs
    pr = torch.stack(preds)[:, None].to(device)
    pr = (1 - pr) if official_uint8 else (255 - pr)       # eval_edge.py's `1 - pred`, literal or as intended
    a, g = (pr / 255.0).clip(0, 1), (gt / 255.0).clip(0, 1)
    vals = [float(ssim(a[i:i + 1], g[i:i + 1])) for i in range(len(a))]
    ps = [float(psnr(a[i:i + 1], g[i:i + 1])) for i in range(len(a))]
    ps = [100.0 if not np.isfinite(x) else x for x in ps]    # eval_edge.py caps inf at 100
    return {"video": v, "method": method, "eval_size": [ew, eh], "instants": len(vals),
            "ms_ssim": float(np.mean(vals)), "psnr": float(np.mean(ps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/controlnetpp_lineart_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    model = load_model()
    ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            r = score_method(model, ssim, psnr, m, v, src, src_fps, official_uint8=False)
            lit = score_method(model, ssim, psnr, m, v, src, src_fps, official_uint8=True)
            r["ms_ssim_official_uint8"], r["psnr_official_uint8"] = lit["ms_ssim"], lit["psnr"]
            results.append(r)
            print(f"{v:18s} {m:10s} MS-SSIM {r['ms_ssim']:.4f}  PSNR {r['psnr']:6.2f}  | literal-uint8 MS-SSIM "
                  f"{r['ms_ssim_official_uint8']:.4f}  ({r['instants']} instants)", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos (higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        print(f"  {m:10s} MS-SSIM {np.mean([r['ms_ssim'] for r in rs]):.4f}  PSNR {np.mean([r['psnr'] for r in rs]):6.2f}  "
              f"| literal-uint8 MS-SSIM {np.mean([r['ms_ssim_official_uint8'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
