"""Temporal metrics across methods, every video brought to 8 fps first.

    python -m metrics.compare.temporal_compare --out results.json

Three official implementations, called unchanged:

  * VBench temporal_flickering: vbench.temporal_flickering.cal_score(mp4).
  * VBench motion_smoothness: vbench.motion_smoothness.MotionSmoothness with
    VBench's own AMT-S config and checkpoint, .motion_score(mp4).
  * CSD style consistency, as EchoStyle defines it ("CSD scores between each
    subsequent frame and the initial frame"), using learn2phoenix/CSD exactly as
    its main_sim.py does: CSD_CLIP('vit_large') loaded via convert_state_dict,
    the csd eval transform (Resize 256, CenterCrop 224, ImageNet normalise), the
    'head' embedding (style_output, already L2-normalised), inner product.

Frame rate. Outputs run at 24 (ours, Kiwi, source), 16 (EchoStyle, PickStyle)
and 8 fps (AnyV2V). Every adjacent-frame metric gets harsher as the frame gap
grows, so all are resampled to 8 fps, the lowest: frame k*fps/8, which is an
exact frame for 24, 16 and 8. Each is also resized to short side 480 at its own
aspect, like the other metrics. The VBench functions only take a file, so the
resampled frames are written losslessly (libx264 -qp 0, yuv444p) to a temp mp4.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from metrics.compare.lineart_compare import ROOT, eval_size, path_of, read_all

TARGET_FPS = 8
CSD_REPO = "/mnt/nas3/sunyuzework/Diffutoon-2/baselines/_metric_repos/CSD"
CSD_CKPT = "/root/models/metrics/CSD-ViT-L/pytorch_model.bin"
CLIP_DIR = "/root/models/metrics/clip"
AMT_CKPT = "/root/models/metrics/amt_model/amt-s.pth"


def resampled_480(path, native=False):
    """Frames at TARGET_FPS (or the file's own fps if native), short side 480."""
    frames, fps = read_all(path)
    if native:
        picked = frames
    else:
        step = fps / TARGET_FPS
        assert abs(step - round(step)) < 1e-6, f"{path}: {fps} fps is not a multiple of {TARGET_FPS}"
        picked = frames[::int(round(step))]
    oh, ow = picked[0].shape[:2]
    ew, eh = eval_size(ow, oh)
    return [np.asarray(Image.fromarray(f).resize((ew, eh), Image.BICUBIC)) for f in picked], (fps if native else TARGET_FPS)


def write_lossless(frames, path, fps=TARGET_FPS):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                          "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def allow_amt_checkpoint():
    """VBench 0.1.5 predates PyTorch 2.6's weights_only=True default, and its AMT-S
    checkpoint stores a typing.OrderedDict. Allowlist that one container type
    rather than turning the safe loader off."""
    import typing
    torch.serialization.add_safe_globals([typing.OrderedDict])


def load_csd(device="cuda"):
    sys.path.insert(0, CSD_REPO)
    # CSD/model.py does `import clip` (OpenAI's package); the repo vendors that
    # code in models/clip, so alias it rather than install a second copy, and
    # point its weight download at the local ViT-L-14.pt.
    import models.clip as clip_pkg
    import models.clip.clip as clipmod
    sys.modules["clip"] = clip_pkg
    orig_load = clipmod.load
    clip_pkg.load = clipmod.load = lambda name, *a, **k: orig_load(name, *a, **{**k, "download_root": CLIP_DIR})
    from CSD.model import CSD_CLIP
    from CSD.utils import convert_state_dict
    import torchvision.transforms as T
    model = CSD_CLIP("vit_large", "default")
    ck = torch.load(CSD_CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    msg = model.load_state_dict(convert_state_dict(sd), strict=False)
    print("CSD load:", msg, flush=True)
    model = model.to(device).eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return model, tf


@torch.no_grad()
def csd_consistency(model, tf, frames, device="cuda"):
    x = torch.stack([tf(Image.fromarray(f)) for f in frames]).to(device)
    with torch.cuda.amp.autocast():
        _, _, style = model(x)
    style = style.float()
    sims = (style[1:] @ style[0]).cpu().numpy()        # each later frame vs the initial one
    return float(sims.mean()), float(sims.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/_metrics/temporal_davis10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="Source,Ours,EchoStyle,PickStyle,Kiwi-Edit,AnyV2V")
    ap.add_argument("--native-fps", action="store_true", help="score each video at its own fps, no resampling")
    a = ap.parse_args()
    if a.native_fps and a.out.endswith("temporal_davis10.json"):
        a.out = a.out.replace("temporal_davis10.json", "temporal_davis10_nativefps.json")
    from vbench.motion_smoothness import MotionSmoothness
    from vbench.temporal_flickering import cal_score
    allow_amt_checkpoint()
    cfg = os.path.join(os.path.dirname(__import__("vbench").__file__), "third_party/amt/cfgs/AMT-S.yaml")
    ms = MotionSmoothness(cfg, AMT_CKPT, "cuda")
    csd, tf = load_csd()

    davis = sorted(f.replace("_nf328_ctx1g_s20_u05_t05.mp4", "") for f in os.listdir(f"{ROOT}/DAVIS2017_Diffutoon2")
                   if f.endswith(".mp4"))
    videos = sorted(random.Random(a.seed).sample(davis, a.n))
    methods = a.methods.split(",")
    print("videos:", videos, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for v in videos:
            for m in methods:
                path = f"{ROOT}/all/{v}.mp4" if m == "Source" else path_of(m, v)
                frames, fps = resampled_480(path, native=a.native_fps)
                mp4 = os.path.join(tmp, f"{v}__{m}.mp4")
                write_lossless(frames, mp4, fps)
                cmean, cmin = csd_consistency(csd, tf, frames)
                r = {"video": v, "method": m, "fps": fps, "frames": len(frames),
                     "temporal_flickering": float(cal_score(mp4)),
                     "motion_smoothness": float(ms.motion_score(mp4)),
                     "csd_consistency": cmean, "csd_min": cmin}
                results.append(r)
                print(f"{v:18s} {m:10s} flicker {r['temporal_flickering']:.5f}  smooth {r['motion_smoothness']:.5f}  "
                      f"csd {cmean:.4f} (min {cmin:.4f})  ({len(frames)} frames @{fps:g}fps)", flush=True)
                with open(a.out, "w") as fh:
                    json.dump({"videos": videos, "fps": "native" if a.native_fps else TARGET_FPS, "results": results}, fh, indent=1)
    print("\nmean over", len(videos), "videos (all higher is better)")
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        print(f"  {m:10s} flicker {np.mean([r['temporal_flickering'] for r in rs]):.5f}  "
              f"smooth {np.mean([r['motion_smoothness'] for r in rs]):.5f}  csd {np.mean([r['csd_consistency'] for r in rs]):.4f}")


if __name__ == "__main__":
    main()
