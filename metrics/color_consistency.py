"""Colour consistency -- CIEDE2000 on the chroma plane, with the lightness dropped.

The generated anime should keep the source video's colour identity while being
free to change its tonality: cel shading flattens gradients and pushes contrast,
so a metric that looks at lightness would punish exactly the thing we asked for.
So we compare a/b only.

**Dropping L is implementation-independent.** With dL = 0 the CIEDE2000 lightness
term vanishes (S_L multiplies zero), so forcing both frames to L=50, both to
L=20, or copying the reference's L all give the same number. We force L=50 for
both and let the chroma and hue terms carry the whole difference.

**CIEDE2000 rather than MSE in Lab.** They are not two methods: Lab Euclidean
distance *is* dE76, and Lab 3-channel MSE times 3 equals mean(dE76^2). dE76 is
CIEDE2000's predecessor and over-penalises saturated regions -- for the same ab
shift it reports 8.49 everywhere, while CIEDE2000 reports 8.70 on neutral grey
and 3.75 on saturated yellow. For cel shading, which is mostly large saturated
flats, that difference matters.

**The Gaussian on a/b.** Black outlines are a = b = 0, i.e. neutral, so they sit
on top of grey in Lab and an outline drawn one pixel off registers as a full
colour error. Blurring the chroma planes lets the surrounding colour diffuse into
the outline and dilutes it -- the same reasoning as chroma subsampling, where the
eye tolerates exactly this. It is a frequency separation, not laxity: blurring
lowers the matched dE while leaving the mismatched dE flat, and injected
low-frequency faults (global tint, a locally recoloured patch, 30% desaturation)
are not masked by it.

Sigma is set to 15. Measured across sigma 0..64 on two clips, the ratio of
mismatched to matched dE rises monotonically for the dense clip but peaks around
sigma 12-14 for the sparse one and then falls back, so there is no common
plateau. Sigma 15 sits at that peak. The cost of going further is real and
measurable: the penalty for a localised 128x128 fault is flat through sigma 12
but decays past sigma 24 (0.48 -> 0.45 -> 0.35 at sigma 48), and the a-channel's
within-frame standard deviation falls 8.61 -> 6.81 -> 5.53 over the same range,
i.e. the metric degenerates toward comparing whole-frame average colour.

Alignment is shared with the lineart metric: `read_aligned` applies diffsynth's
own `crop_and_resize` and the inference frame map, so the reference lands on the
pixels the generation actually saw.

Usage:
    python -m metrics.color_consistency source.mp4 generated.mp4
    python -m metrics.color_consistency source.mp4 generated.mp4 --stride 20
    python -m metrics.color_consistency source.mp4 generated.mp4 --sigma 0
"""

import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab

from metrics.lineart_boundary_f import read_aligned, video_size

FPS = 24
SIGMA = 15      # Gaussian sigma on the a/b planes; 0 disables. See the docstring.
HIST_BINS = 32  # per axis, for the complementary ab histogram intersection
AB_RANGE = 110  # a and b are roughly -110..110 for in-gamut sRGB
STRIDE = 10     # score every Nth frame; every sweep run so far used 10


def chroma_lab(frame, sigma=SIGMA):
    """RGB frame -> Lab with a/b blurred and L flattened to a constant.

    Flattening L is what removes the lightness dimension; see the module
    docstring for why the constant's value does not matter.
    """
    lab = rgb2lab(frame / 255.0)
    if sigma > 0:
        lab[..., 1] = gaussian_filter(lab[..., 1], sigma)
        lab[..., 2] = gaussian_filter(lab[..., 2], sigma)
    lab[..., 0] = 50.0
    return lab


def frame_delta_e(ref, gen, sigma=SIGMA):
    """Mean CIEDE2000 between two RGB frames, chroma only."""
    return float(deltaE_ciede2000(chroma_lab(ref, sigma),
                                  chroma_lab(gen, sigma)).mean())


def ab_histogram(frames, bins=HIST_BINS, ab_range=AB_RANGE):
    """Joint a-b histogram over a set of frames, normalised to sum 1.

    Unblurred on purpose: this is the complementary view. dE2000 is per-pixel and
    so is sensitive to where the colour is; the histogram ignores position
    entirely and asks only whether the palette is the same. A result that keeps
    the right palette but puts it in the wrong places scores well here and badly
    on dE, and the two together say which failure happened.
    """
    hist = np.zeros((bins, bins), dtype=np.float64)
    edges = np.linspace(-ab_range, ab_range, bins + 1)
    for frame in frames:
        lab = rgb2lab(frame / 255.0)
        h, _, _ = np.histogram2d(lab[..., 1].ravel(), lab[..., 2].ravel(),
                                 bins=[edges, edges])
        hist += h
    total = hist.sum()
    return hist / total if total else hist


def histogram_intersection(h1, h2):
    """Sum of elementwise minima of two normalised histograms: 1 = identical."""
    return float(np.minimum(h1, h2).sum())


def score_pair(source, generated, sigma=SIGMA, stride=STRIDE, fps=FPS):
    """Chroma dE2000 and ab histogram intersection between source and generation."""
    h, w, n, _ = video_size(generated)
    gen, _ = read_aligned(generated, n, h, w, fps=fps)
    ref, src_fps = read_aligned(source, n, h, w, fps=fps)
    count = min(len(gen), len(ref))
    idx = list(range(0, count, max(1, stride)))

    per_frame = [frame_delta_e(ref[i], gen[i], sigma) for i in idx]
    hi = histogram_intersection(ab_histogram([ref[i] for i in idx]),
                                ab_histogram([gen[i] for i in idx]))

    values = np.asarray(per_frame, dtype=np.float64)
    return {
        "frames": count,
        "scored": len(values),
        "src_fps": src_fps,
        "sigma": sigma,
        "delta_e": float(values.mean()),
        "delta_e_std": float(values.std()),
        "ab_hist_intersection": hi,
        "per_frame": values.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="the real video whose colour should be kept")
    p.add_argument("generated", help="the generated anime video to score")
    p.add_argument("--sigma", type=float, default=SIGMA,
                   help=f"Gaussian sigma on a/b, 0 disables (default {SIGMA})")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"score every n-th frame (default {STRIDE}; 1 = all)")
    args = p.parse_args()

    r = score_pair(args.source, args.generated, sigma=args.sigma,
                   stride=args.stride)
    print(f"{os.path.basename(args.generated)}: {r['frames']} frames, "
          f"{r['scored']} scored (source {r['src_fps']:.2f}fps -> {FPS}fps)")
    print(f"  chroma dE2000 {r['delta_e']:.3f} (sd {r['delta_e_std']:.3f}), "
          f"sigma {r['sigma']:g}, lightness dropped   [lower is better]")
    print(f"  ab histogram intersection {r['ab_hist_intersection']:.4f}"
          f"   [higher is better, 1 = same palette]")


if __name__ == "__main__":
    main()
