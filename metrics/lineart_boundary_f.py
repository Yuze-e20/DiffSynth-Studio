"""Boundary F-score between the line art of a generated video and of its source.

    python -m metrics.lineart_boundary_f real.mp4 generated.mp4 --compare-dir compare/13

The generated video's line art is extracted with the same detector the pipeline
feeds to the ControlNet, so both sides come from one model and a difference is a
difference in drawing, not in extractor. The reference track is then put into the
generated video's own geometry and timing -- diffsynth's `crop_and_resize` and the
`round(k*src_fps/fps)` frame map that `read_video_audio` uses -- because a 1920x1080
source against a 1344x768 output is a 1.6% aspect mismatch, which at a two-pixel
tolerance would dominate the score.

Scoring is the usual spatial-tolerance boundary F: a predicted edge pixel counts as
matched if a reference edge pixel lies within `--tolerance` pixels, and vice versa,
so precision and recall are forgiving about where a line sits but not about whether
it exists. Distances come from a Euclidean distance transform, which is exact, not
from dilation.

Line art is grayscale, so it is thresholded first, with Otsu's own cut. `--strict`
scales that threshold up to keep only stronger lines; raising it was measured to
change the binarisation very little on this material, so the default leaves Otsu
alone. This binarisation is for scoring only -- the ControlNet is still fed the
original grayscale track.
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

from diffsynth.utils.data import crop_and_resize

FPS = 24
TOLERANCE = 1.5     # pixels of slack, on the 1344x768 output grid; Euclidean, so
                    # this is the 8-neighbourhood: a diagonal neighbour is 1.414
STRICT = 1.0        # Otsu threshold multiplier; 1.0 = Otsu's own cut, unmodified
OTSU_SAMPLES = 24   # frames sampled to fix one threshold for the whole video
STRIDE = 10         # score every Nth frame; every sweep run so far used 10


def gray(frame):
    """Line art is written as a grayscale image in three identical channels."""
    return frame[..., 0] if frame.ndim == 3 else frame


def otsu_threshold(frames, strict=STRICT):
    """One threshold for the whole video, from a sample of its frames.

    Per-frame Otsu would move the threshold with the content: a frame holding
    only faint lines gets its own low cut and reports as many edges as a busy
    one, which makes the score drift for reasons that have nothing to do with
    the generation. Pooling the samples fixes a single cut instead. The result
    is clipped to the valid range so a large `strict` cannot threshold
    everything away silently.
    """
    pooled = np.concatenate([gray(f).ravel() for f in frames])
    t = threshold_otsu(pooled) * strict
    return float(np.clip(t, 1, 254))


def boundary_f(pred, ref, tolerance=TOLERANCE):
    """Boundary precision/recall/F1 with `tolerance` pixels of spatial slack.

    `distance_transform_edt` of the complement gives, at every pixel, the exact
    Euclidean distance to the nearest edge pixel of that map, so a predicted
    pixel is matched when the reference distance map reads <= tolerance there.
    Precision and recall are counted against their own distance maps, which is
    what makes the measure asymmetric and the F1 meaningful: a generated frame
    that draws nothing scores recall 0, one that draws everywhere scores
    precision ~0.
    """
    if not pred.any() or not ref.any():
        # An empty map has no matches; F is 0 unless both are empty, where the
        # two frames do agree and the score is 1.
        both_empty = not pred.any() and not ref.any()
        return (1.0, 1.0, 1.0) if both_empty else (0.0, 0.0, 0.0)
    d_ref = distance_transform_edt(~ref)
    d_pred = distance_transform_edt(~pred)
    precision = (d_ref[pred] <= tolerance).mean()
    recall = (d_pred[ref] <= tolerance).mean()
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def read_aligned(path, count, height, width, fps=FPS):
    """Read `count` frames at `fps`, cropped and resized to height x width.

    This is the frame map `read_video_audio` uses at inference -- output frame k
    reads source frame round(k*src_fps/fps) -- and diffsynth's own
    `crop_and_resize`, imported rather than reimplemented so the reference lands
    on exactly the pixels the generation saw. A source that runs out early
    yields fewer frames, same as it does during inference.
    """
    reader = imageio.get_reader(path)
    src_fps = float(reader.get_meta_data()["fps"])
    out, cache, cached_index = [], None, -1
    for k in range(count):
        index = int(round(k * src_fps / fps))
        if cached_index != index:
            try:
                cache = reader.get_data(index)
            except (IndexError, RuntimeError):
                break
            cached_index = index
        img = Image.fromarray(cache)
        if img.size != (width, height):
            img = crop_and_resize(img, height, width)
        out.append(np.asarray(img))
    reader.close()
    return out, src_fps


def video_size(path):
    reader = imageio.get_reader(path)
    frame = reader.get_data(0)
    n = reader.count_frames()
    fps = float(reader.get_meta_data()["fps"])
    reader.close()
    return frame.shape[0], frame.shape[1], n, fps


def score_pair(real_lineart, gen_lineart, tolerance=TOLERANCE, strict=STRICT,
               compare_dir=None, stride=STRIDE):
    """Score one pair of line-art tracks, optionally writing the compare videos.

    Both thresholds are computed once, each from its own track: the two have
    different intensity distributions (a photographic frame is busier than a cel)
    and forcing one cut on both would measure that offset instead of the drawing.
    """
    height, width, gen_n, gen_fps = video_size(gen_lineart)
    ref_frames, ref_src_fps = read_aligned(real_lineart, gen_n, height, width)
    gen_frames, _ = read_aligned(gen_lineart, gen_n, height, width)
    n = min(len(ref_frames), len(gen_frames))
    print(f"{os.path.basename(gen_lineart)}: {width}x{height} @{gen_fps:g}fps, "
          f"scoring {n} frames (reference {ref_src_fps:g}fps -> {FPS}fps)", flush=True)

    picked = list(range(0, n, stride))
    sample = [gen_frames[i] for i in picked[:OTSU_SAMPLES]]
    ref_sample = [ref_frames[i] for i in picked[:OTSU_SAMPLES]]
    t_gen = otsu_threshold(sample, strict)
    t_ref = otsu_threshold(ref_sample, strict)
    print(f"  otsu x{strict}: reference {t_ref:.1f}, generated {t_gen:.1f}", flush=True)

    rows, writers = [], None
    if compare_dir:
        os.makedirs(compare_dir, exist_ok=True)
        writers = {k: imageio.get_writer(os.path.join(compare_dir, f"{k}.mp4"),
                                         fps=FPS, quality=9, macro_block_size=1)
                   for k in ("real_lineart", "anime_lineart",
                             "real_lineart_binary", "anime_lineart_binary")}
    for i in picked:
        ref_bin = gray(ref_frames[i]) > t_ref
        gen_bin = gray(gen_frames[i]) > t_gen
        p, r, f = boundary_f(gen_bin, ref_bin, tolerance)
        rows.append((i, p, r, f, float(ref_bin.mean()), float(gen_bin.mean())))
        if writers:
            writers["real_lineart"].append_data(ref_frames[i])
            writers["anime_lineart"].append_data(gen_frames[i])
            writers["real_lineart_binary"].append_data((ref_bin * 255).astype(np.uint8))
            writers["anime_lineart_binary"].append_data((gen_bin * 255).astype(np.uint8))
    if writers:
        for w in writers.values():
            w.close()      # without close() the mp4 trailer is missing

    arr = np.array([[p, r, f, rb, gb] for _, p, r, f, rb, gb in rows])
    mean = arr.mean(axis=0)
    return {"frames": len(rows), "precision": mean[0], "recall": mean[1], "f1": mean[2],
            "ref_edge_density": mean[3], "gen_edge_density": mean[4],
            "f1_std": float(arr[:, 2].std()), "threshold_ref": t_ref,
            "threshold_gen": t_gen, "per_frame": rows}


def ensure_lineart(video, cache_dir=None, force=False):
    """Extract line art for `video` unless it is already cached beside it.

    Runs out of process for the same reason the pipeline's preprocessing does:
    the detector sits on the GPU and only a process exit really frees it.
    """
    import subprocess
    import sys

    from diffsynth.utils.controlnet import video_lineart

    stem = os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(cache_dir or os.path.dirname(os.path.abspath(video)), f"{stem}_lineart.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0 and not force:
        print(f"[skip] lineart: {out} already there", flush=True)
        return out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"[run ] lineart -> {out}", flush=True)
    r = subprocess.run([sys.executable, "-m", video_lineart.__name__, video, out])
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"lineart failed (rc={r.returncode}); see the output above")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("real", help="the source video, or its line-art track with --lineart-given")
    p.add_argument("generated", help="the generated video, or its line-art track with --lineart-given")
    p.add_argument("--lineart-given", action="store_true",
                   help="both inputs are already line-art tracks; skip extraction")
    p.add_argument("--compare-dir", default=None,
                   help="write real/anime line art and their binarisations here")
    p.add_argument("--cache-dir", default=None, help="where extracted line art lands (default beside the input)")
    p.add_argument("--tolerance", type=float, default=TOLERANCE, help=f"pixels of slack (default {TOLERANCE})")
    p.add_argument("--strict", type=float, default=STRICT,
                   help=f"Otsu threshold multiplier, >1 keeps only stronger lines (default {STRICT})")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"score every Nth frame (default {STRIDE})")
    p.add_argument("--force", action="store_true", help="re-extract line art even if cached")
    args = p.parse_args()

    if args.lineart_given:
        real_la, gen_la = args.real, args.generated
    else:
        real_la = ensure_lineart(args.real, args.cache_dir, args.force)
        gen_la = ensure_lineart(args.generated, args.cache_dir, args.force)

    res = score_pair(real_la, gen_la, tolerance=args.tolerance, strict=args.strict,
                     compare_dir=args.compare_dir, stride=args.stride)
    print(f"\nboundary F @ {args.tolerance}px, otsu x{args.strict}, {res['frames']} frames")
    print(f"  precision {res['precision']:.4f}  recall {res['recall']:.4f}  "
          f"F1 {res['f1']:.4f} (sd {res['f1_std']:.4f})")
    print(f"  edge density: reference {res['ref_edge_density']:.4f}  generated {res['gen_edge_density']:.4f}")
    if args.compare_dir:
        print(f"  compare videos -> {args.compare_dir}")
    return res


if __name__ == "__main__":
    main()
