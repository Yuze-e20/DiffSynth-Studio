"""DINO-ViT self-similarity structure distance (Splice), per method.

    python -m metrics.compare.dino_structure_compare --out results.json

The distance is omerbt/Splice's own, imported from its repo and called as its
LossG.calculate_global_ssim_loss does:

  * extractor: models.extractor.VitExtractor('dino_vitb8') -- DINO ViT-B/8,
    the config default (conf/default/config.yaml: dino_model_name);
  * transform: Resize(224, max_size=480) then ImageNet normalisation, the
    global_transform built in util/losses.py (dino_global_patch_size = 224);
  * features: get_keys_self_sim_from_input(img, layer_num=11) -- keys of the
    last block, 12 heads concatenated, [CLS] included, pairwise cosine
    (attn_cosine_sim);
  * distance: F.mse_loss(self_sim(source), self_sim(generated)). Lower is
    closer structure. The paper writes a Frobenius norm; MSE is the same up to
    a constant for a fixed token count, which Resize(224) fixes per aspect.

Each pair is the source frame and the method's frame at the same instant, the
source cut the way that method cut it (lineart_compare's geometry), both
resized to the same evaluation grid first so the two go through Splice's
transform at an identical size and their token grids match.

DINO is loaded with torch.hub from a local hub dir (TORCH_HOME), so nothing is
fetched at run time.
"""
import argparse
import json
import os
import random
import sys

TORCH_HOME = "/root/models/metrics/torch_hub"
os.environ.setdefault("TORCH_HOME", TORCH_HOME)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Resize

from metrics.compare.lineart_compare import GEOMETRY, ROOT, STRIDE_S, eval_size, path_of, read_all

SPLICE = "/mnt/nas3/sunyuzework/Diffutoon-2/baselines/_metric_repos/Splice"
MODEL = "dino_vitb8"
PATCH = 224


def load_extractor(device="cuda"):
    sys.path.insert(0, SPLICE)
    # torch.hub resolves 'facebookresearch/dino:main' to TORCH_HOME/hub/facebookresearch_dino_main
    # and would try GitHub first; source='local' is not reachable through Splice's call,
    # so pre-empt the remote lookup by making hub.load use that directory.
    _load = torch.hub.load
    hub_dir = os.path.join(os.environ["TORCH_HOME"], "hub", "facebookresearch_dino_main")
    torch.hub.load = lambda repo, model, *a, **k: _load(hub_dir, model, *a, source="local", **k)
    from models.extractor import VitExtractor
    ex = VitExtractor(model_name=MODEL, device=device)
    torch.hub.load = _load
    tf = transforms.Compose([Resize(PATCH, max_size=480),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return ex, tf


@torch.no_grad()
def self_sim(ex, tf, img, device="cuda"):
    x = tf(TF.to_tensor(img).to(device))            # Splice feeds [0,1] tensors into global_transform
    return ex.get_keys_self_sim_from_input(x.unsqueeze(0), layer_num=11)


def score_method(ex, tf, method, v, src, src_fps):
    gen, gen_fps = read_all(path_of(method, v))
    sh, sw = src[0].shape[:2]
    oh, ow = gen[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    l, t, r, b = GEOMETRY[method](sw, sh, ow, oh)
    duration = min(len(src) / src_fps, len(gen) / gen_fps)
    d = []
    for s in np.arange(0, duration - 1e-6, STRIDE_S):
        si = min(len(src) - 1, int(round(s * src_fps)))
        gi = min(len(gen) - 1, int(round(s * gen_fps)))
        ref = Image.fromarray(src[si]).crop((int(round(l)), int(round(t)), int(round(r)), int(round(b)))).resize((ew, eh), Image.BICUBIC)
        out = Image.fromarray(gen[gi]).resize((ew, eh), Image.BICUBIC)
        a, g = self_sim(ex, tf, ref), self_sim(ex, tf, out)
        assert a.shape == g.shape, (a.shape, g.shape)
        d.append(float(F.mse_loss(g, a)))
    return {"video": v, "method": method, "eval_size": [ew, eh], "instants": len(d),
            "dino_structure_dist": float(np.mean(d)), "dino_structure_dist_std": float(np.std(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/dino_structure_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    a = ap.parse_args()
    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    ex, tf = load_extractor()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for v in videos:
        src, src_fps = read_all(f"{ROOT}/all/{v}.mp4")
        for m in methods:
            r = score_method(ex, tf, m, v, src, src_fps)
            results.append(r)
            print(f"{v:18s} {m:10s} dist {r['dino_structure_dist']:.5f}  ({r['instants']} instants)", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"videos": videos, "model": MODEL, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos (lower is better)")
    for m in methods:
        print(f"  {m:10s} {np.mean([r['dino_structure_dist'] for r in results if r['method'] == m]):.5f}")


if __name__ == "__main__":
    main()
