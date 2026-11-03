"""Flow-based Pixel-MSE (warp error) -- temporal smoothness of the generated video.

The smoothness metric that ships with the other two is `vbench_flicker.py`, which
calls VBench's API directly. This file is the flow-based alternative: it costs a
RAFT pass per triplet and it disagrees with VBench on the trained-branch scale
(t0.7 here against VBench's t0.9), because it normalises by local contrast and
VBench does not. Keep it for diagnosing a disagreement, not as the default number.

The question is whether the generated video flickers: does a pixel keep its
colour as the scene moves, or does it get re-invented every frame. The standard
answer is warp error -- take the motion from one frame to the next, carry frame
t-1 forward along it, and see how far the real frame t is from that prediction.

Plain warp error has two failure modes that matter here, both measured on the
u0.8 row of clip 13's scale grid (see the table at the bottom of this docstring).
The headline number is therefore the contrast-normalised second difference,
`flicker`, with the plain warp error reported next to it for reference.

**It reads blur as stability.** Squared residual scales with local contrast, so a
crisp cel edge off by half a pixel costs far more than a photographic gradient
off by the same amount. As the trained branch's scale rises the output gets
flatter -- gradient energy falls 7.07 -> 5.70 across t0.5..t1.0 -- so the plain
error falls for a reason that has nothing to do with temporal stability. Dividing
the residual by the local contrast (std in a 9x9 window, floored so flat regions
do not blow up) fixes it: the normalised error bottoms out at t0.6-0.7 and rises
again at t1.0, which is what the eye reports.

**It mixes smooth drift with flicker.** A single t -> t+1 residual cannot tell a
slow drift from frame-to-frame alternation, and alternation is what reads as
flicker. The second difference along the motion trajectory,
2*I_t - warp(I_t-1) - warp(I_t+1), cancels any motion the flow explains and any
linear change along it, leaving the alternating component.

Three further decisions worth stating, since they are what makes the number mean
something:

**The flow comes from the source video, not the generated one.** Estimating flow
on the generated frames would let a flickery result hide: the flow estimator
absorbs the flicker into a wrong motion field and the residual comes out small.
The source video is the motion ground truth, and it is the same motion the
generation was conditioned on, so the flow is shared across every scale
configuration being compared -- the only thing that varies is the pixels being
warped.

**Occlusions are excluded.** Where the motion field is unreliable (disocclusion,
out-of-frame, the forward and backward flows disagreeing) the residual says
nothing about flicker. We use the same forward/backward consistency test
Rerender uses (UnFlow's alpha=0.01, beta=0.5), plus the frame border that the
warp samples from outside. Reported alongside the error is the fraction of
pixels kept, because a configuration that occludes differently is not being
scored on the same pixels.

**Everything lands on the generation grid first.** Source is 1920x1080@30, output
is 1344x768@24; the flow has to be estimated on the same grid and the same frame
map the generation saw, so this reuses `read_aligned` from the lineart metric,
which in turn imports diffsynth's own `crop_and_resize`.

The flow model is torchvision's RAFT (C_T_SKHT_V2). Rerender's own
`flow/flow_utils.py` wants gmflow with `models/gmflow_sintel-0c07dcb3.pth`,
which is not downloaded here and not reachable; RAFT is equivalent for this
purpose and ships its weights through torchvision. pix2video has no evaluation
code at all -- its `code/common.py:8` `warp` is dead code (no call site, and the
author marks it "ignore" in three places), warps latents at 1/8 resolution with
nearest sampling, and the flow field it needs is not in the repository.

**Neither number can pick a configuration on its own.** The source video scores
better than every generated one -- warp error 1.42 against the best 6.83 --
because the optimum of any pure
temporal-stability measure is to output the input unchanged. Normalising fixes
the direction within the generated set but not this: read `flicker` as a ranking
among stylised outputs, never against the source, and pair it with a metric that
rewards looking stylised at all.

What this measures on clip 13, u0.8 row (stride 20, 30 triplets):

    config   warp_mse  norm_mse  raw2nd  flicker  gradient energy
    t0.5       10.755    0.0252  25.315   0.0539   7.314
    t0.6        8.865    0.0225  21.183   0.0507   7.222
    t0.7        7.795    0.0226  18.645   0.0501   6.885
    t0.8        7.193    0.0237  16.709   0.0520   6.546
    t0.9        6.868    0.0252  15.317   0.0547   6.134
    t1.0        6.831    0.0282  14.976   0.0597   5.804

Both raw columns say t1.0 is the most stable; both normalised columns put the
optimum at t0.6-0.7 and rank t1.0 worse than t0.5.

`pair_flow` must return the a->b field, and getting this backwards is the kind of
bug the numbers absorb quietly: measured on the source warped onto itself, the
correct field gives residual 1.42 and the reversed one 5.49, and the gap
correlates 0.83 with motion magnitude -- so a reversed field reads as "unstable
wherever anything moves", which looks like a plausible metric rather than a bug.

Usage:
    python -m metrics.flow_warp_error source.mp4 generated.mp4
    python -m metrics.flow_warp_error source.mp4 generated.mp4 --stride 10
    python -m metrics.flow_warp_error source.mp4 generated.mp4 --compare-dir metrics/compare/13_flow
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from metrics.lineart_boundary_f import read_aligned, video_size

FPS = 24
ALPHA = 0.01        # forward/backward consistency, following UnFlow (and Rerender)
BETA = 0.5
BORDER = 8          # pixels dropped at the frame edge, where the warp samples outside
CONTRAST_WINDOW = 9     # side of the window the local contrast is measured in
CONTRAST_FLOOR = 4.0    # minimum local contrast, so flat regions do not blow the ratio up
STRIDE = 20             # score every Nth triplet; the flow is the cost, so sample sparsely


def load_flow_model(device="cuda"):
    """RAFT-large, Sintel+KITTI+HD1K fine-tuned. Weights come from torchvision."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
    return model.eval().to(device)


def to_tensor(frame, device):
    """uint8 HWC -> float NCHW in [-1, 1], which is what RAFT's transform expects."""
    x = torch.from_numpy(frame).permute(2, 0, 1).float()[None] / 255.0
    return (x * 2 - 1).to(device)


def coords_grid(b, h, w, device):
    y, x = torch.meshgrid(torch.arange(h, device=device),
                          torch.arange(w, device=device), indexing="ij")
    return torch.stack([x, y], dim=0).float()[None].repeat(b, 1, 1, 1)


def flow_warp(img, flow):
    """Sample `img` at each pixel's position displaced by `flow`.

    Returns the warped image and the mask of pixels whose source fell inside the
    frame. Same construction as Rerender's `flow_warp` / `bilinear_sample`, with
    the validity mask kept rather than discarded.
    """
    b, _, h, w = img.shape
    grid = coords_grid(b, h, w, img.device) + flow
    gx = 2 * grid[:, 0] / (w - 1) - 1
    gy = 2 * grid[:, 1] / (h - 1) - 1
    g = torch.stack([gx, gy], dim=-1)
    out = F.grid_sample(img, g, mode="bilinear", padding_mode="zeros",
                        align_corners=True)
    inside = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    return out, inside


def consistency_occlusion(fwd, bwd, alpha=ALPHA, beta=BETA):
    """Pixels where forward and backward flow disagree -- occluded or unreliable.

    Exactly the test in Rerender's `forward_backward_consistency_check`: a pixel
    is occluded when |fwd + warp(bwd, fwd)| exceeds alpha*|flow| + beta.
    """
    mag = fwd.norm(dim=1) + bwd.norm(dim=1)
    warped_bwd, _ = flow_warp(bwd, fwd)
    diff = (fwd + warped_bwd).norm(dim=1)
    return diff > (alpha * mag + beta)


def local_contrast(frame, device, window=CONTRAST_WINDOW, floor=CONTRAST_FLOOR):
    """Local luma standard deviation, as the denominator for the residual.

    Computed on the frame being scored rather than on the source, because what
    we are correcting for is how much contrast *this* configuration put there:
    a flatter output earns a smaller denominator and so stops being rewarded for
    its flatness. `floor` keeps a genuinely flat region from dividing a small
    residual by something near zero.
    """
    x = torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(device)
    luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    pad = window // 2
    mean = F.avg_pool2d(F.pad(luma, (pad,) * 4, mode="reflect"), window, stride=1)
    sq = F.avg_pool2d(F.pad(luma * luma, (pad,) * 4, mode="reflect"), window, stride=1)
    return (sq - mean * mean).clamp(min=0).sqrt().clamp(min=floor)


def gradient_energy(frame, device):
    """Mean spatial gradient magnitude -- how much high-frequency detail is left.

    Reported so a drop in the raw error can be checked against the blur that
    probably caused it.
    """
    x = torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(device)
    luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    dx = (luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs().mean()
    dy = (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs().mean()
    return float(((dx + dy) / 2).item())


@torch.no_grad()
def pair_flow(model, frame_a, frame_b, device):
    """Flow a->b plus the occlusion mask, both on frame_a's grid.

    The returned field is what `flow_warp` needs to sample frame_b onto frame_a:
    `flow_warp` computes out[p] = img[p + flow[p]], so the displacement has to be
    indexed by frame_a's pixels and point at where that content sits in frame_b,
    which is model(a, b). Returning model(b, a) instead would displace in roughly
    the opposite direction -- for small motion the two are near negatives, so the
    residual grows with motion magnitude rather than measuring instability. The
    occlusion mask lives on frame_a's grid too, so the two agree.

    RAFT needs the spatial dims divisible by 8; 1344x768 already is, but pad
    defensively so an odd crop does not fail here.
    """
    a, b = to_tensor(frame_a, device), to_tensor(frame_b, device)
    h, w = a.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        a = F.pad(a, (0, pw, 0, ph), mode="replicate")
        b = F.pad(b, (0, pw, 0, ph), mode="replicate")
    fwd = model(a, b)[-1]
    bwd = model(b, a)[-1]
    occ = consistency_occlusion(fwd, bwd)
    return fwd[..., :h, :w], occ[..., :h, :w]


def mask_border(valid, border=BORDER):
    """Drop the frame edge, where a warp samples from outside the frame."""
    if border:
        valid[:, :border, :] = False
        valid[:, -border:, :] = False
        valid[:, :, :border] = False
        valid[:, :, -border:] = False
    return valid


def score_pair(source, generated, stride=STRIDE, device="cuda", compare_dir=None,
               fps=FPS, max_triplets=None, flow_cache=None):
    """Temporal stability of `generated`, with motion taken from `source`.

    Scores triplets (t-1, t, t+1) centred `stride` apart. Four numbers come back:
    `flicker` is the headline -- the contrast-normalised second difference along
    the motion trajectory -- and `pixel_mse`, `second_diff` and `gradient_energy`
    are kept beside it so a ranking can be checked against the raw quantities and
    against the blur that moves them. See the module docstring for why the raw
    warp error alone picks the wrong configuration.

    `flow_cache` lets a sweep pay for the flow once: the flow depends only on the
    source video, so pass the same dict across every configuration being
    compared. Keys are centre frame indices, so it is only valid for one
    (source, stride, grid) combination.
    """
    h, w, n, _ = video_size(generated)
    gen, _ = read_aligned(generated, n, h, w, fps=fps)
    src, src_fps = read_aligned(source, n, h, w, fps=fps)
    count = min(len(gen), len(src))
    model = None

    centres = list(range(1, count - 1, max(1, stride)))
    if max_triplets:
        centres = centres[:max_triplets]

    # the grid comes from `generated`, so a cache built on another resolution
    # would silently be warping with the wrong flow field -- say so instead
    if flow_cache:
        cached_w = next(iter(flow_cache.values()))[0].shape[-1]
        if cached_w != w:
            raise ValueError(
                f"flow_cache was built on width {cached_w} but {generated} grids "
                f"to width {w}; a cache is only valid for one source+grid+stride")

    writer = None
    if compare_dir:
        os.makedirs(compare_dir, exist_ok=True)
        writer = imageio.get_writer(os.path.join(compare_dir, "flicker_residual.mp4"),
                                    fps=fps, quality=9, macro_block_size=1)

    cache = flow_cache if flow_cache is not None else {}
    mses, norm_mses, raw2, flicks, kepts, grads = [], [], [], [], [], []
    for t in centres:
        if t not in cache:
            if model is None:
                model = load_flow_model(device)
            # each call returns the flow that samples the neighbour onto frame t
            back, occ_b = pair_flow(model, src[t], src[t - 1], device)
            fwd, occ_f = pair_flow(model, src[t], src[t + 1], device)
            cache[t] = (back, mask_border(~occ_b), fwd, mask_border(~occ_f))
        back, ok_b, fwd, ok_f = cache[t]

        cur = torch.from_numpy(gen[t]).permute(2, 0, 1).float()[None].to(device)
        prv = torch.from_numpy(gen[t - 1]).permute(2, 0, 1).float()[None].to(device)
        nxt = torch.from_numpy(gen[t + 1]).permute(2, 0, 1).float()[None].to(device)
        w_prev, in_b = flow_warp(prv, back)
        w_next, in_f = flow_warp(nxt, fwd)
        contrast = local_contrast(gen[t], device)

        # first order, against the previous frame only -- the classic warp error
        v1 = ok_b & in_b
        r1 = ((cur - w_prev) ** 2).mean(dim=1)
        mses.append(r1[v1].mean().item() if v1.any() else float("nan"))
        rn = (((cur - w_prev) / contrast) ** 2).mean(dim=1)
        norm_mses.append(rn[v1].mean().item() if v1.any() else float("nan"))

        # second order along the trajectory -- cancels motion and linear change
        v2 = v1 & ok_f & in_f
        d2 = 2 * cur - w_prev - w_next
        raw2.append((d2 ** 2).mean(dim=1)[v2].mean().item() if v2.any() else float("nan"))
        dn = ((d2 / contrast) ** 2).mean(dim=1)
        flicks.append(dn[v2].mean().item() if v2.any() else float("nan"))

        kepts.append(v2.float().mean().item())
        grads.append(gradient_energy(gen[t], device))

        if writer is not None:
            vis = np.sqrt(dn[0].clamp(min=0).cpu().numpy())
            vis = np.clip(vis * 255 * 2, 0, 255).astype(np.uint8)
            vis[~v2[0].cpu().numpy()] = 0
            writer.append_data(np.repeat(vis[..., None], 3, axis=2))

    if writer is not None:
        writer.close()

    arr = lambda xs: np.asarray(xs, dtype=np.float64)
    mses, norm_mses, raw2, flicks = arr(mses), arr(norm_mses), arr(raw2), arr(flicks)
    return {
        "triplets": len(flicks),
        "frames": count,
        "src_fps": src_fps,
        "flicker": float(np.nanmean(flicks)),
        "flicker_std": float(np.nanstd(flicks)),
        "second_diff": float(np.nanmean(raw2)),
        "pixel_mse": float(np.nanmean(mses)),
        "pixel_mse_std": float(np.nanstd(mses)),
        "norm_pixel_mse": float(np.nanmean(norm_mses)),
        "psnr": float(10 * np.log10(255.0 ** 2 / np.nanmean(mses))),
        "gradient_energy": float(np.mean(grads)),
        "valid_fraction": float(np.mean(kepts)),
        "per_triplet": flicks.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="the real video the generation was conditioned on")
    p.add_argument("generated", help="the generated anime video to score")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"score every n-th triplet (default {STRIDE}; 1 = all)")
    p.add_argument("--max-triplets", type=int, default=None)
    p.add_argument("--compare-dir", default=None,
                   help="write a residual heat-map video here")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    r = score_pair(args.source, args.generated,
                   stride=args.stride, device=args.device,
                   compare_dir=args.compare_dir, max_triplets=args.max_triplets)
    print(f"{os.path.basename(args.generated)}: {r['frames']} frames, "
          f"{r['triplets']} triplets scored (source {r['src_fps']:.2f}fps -> {FPS}fps)")
    print(f"  flicker {r['flicker']:.4f} (sd {r['flicker_std']:.4f})   <- headline")
    print(f"  raw second difference {r['second_diff']:.3f}")
    print(f"  pixel-MSE {r['pixel_mse']:.3f} (sd {r['pixel_mse_std']:.3f})   "
          f"normalised {r['norm_pixel_mse']:.4f}   warp PSNR {r['psnr']:.2f} dB")
    print(f"  gradient energy {r['gradient_energy']:.3f} "
          f"(a drop here can explain a drop in the raw numbers)")
    print(f"  valid pixels {r['valid_fraction']:.4f} "
          f"(occlusion + border excluded, both neighbours)")
    if args.compare_dir:
        print(f"  residual video -> {args.compare_dir}")


if __name__ == "__main__":
    main()
