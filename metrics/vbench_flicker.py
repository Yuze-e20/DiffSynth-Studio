"""Temporal flickering, straight from VBench's API. The smoothness metric of the three.

VBench's `temporal_flickering` is the mean absolute difference between adjacent
frames, mapped to (255 - mean MAE) / 255 so higher is better. No optical flow, no
motion compensation, no occlusion handling -- which is why this file is short.

    python -m metrics.vbench_flicker video.mp4 [more.mp4 ...]
    python -m metrics.vbench_flicker --grid 13                 # a clip's whole u/t sweep
    python -m metrics.vbench_flicker --grid 13 --row u08       # one u row

`score()` is the number to use: VBench's own `cal_score`, its own reader, the file
as it sits on disk. `--aligned` additionally reports VBench's arithmetic over our
shared `read_aligned` frames, which only matters when comparing against the source
video -- the source is 1920x1080@30fps/753f while the generations are
1344x768@24fps/603f, so the raw path would score them on different grids. On the
generations the two agree to five decimals.

Two limits worth knowing before trusting a ranking from this:

- **It reads flatness as stability.** No contrast normalisation, so a flatter
  picture mechanically produces smaller frame differences. On clip 13's u0.8 row
  it correlates +0.92 with gradient energy and picks t0.9, where the picture is
  visibly flickering and barely stylised; a contrast-normalised trajectory second
  difference picks t0.7 instead (see the README for the full table). Use this to
  rank `union` scale, and treat its opinion on `trained` scale with suspicion.
- **The spread is tiny.** All six configs of a row land in 0.9961-0.9966; an 11.7%
  spread in the underlying MAE compressed into the fourth decimal. VBench is built
  to separate video generators, not two control_scale settings on one model. The
  `mean_abs_diff` column is the same quantity without that compression, so read it
  when configs are close.

Like the other two metrics this is a within-clip ranking only, and its optimum is
identity: the source video scores 0.99775 against the best generated 0.99655.

Install: `pip install --no-deps vbench` plus `decord`. VBench 0.1.5 pins
transformers==4.33.2 and timm<=1.0.12, which a plain install would downgrade.
"""
import argparse
import os
import re

import numpy as np

from vbench.temporal_flickering import cal_score, mae_seq

from metrics.lineart_boundary_f import read_aligned, video_size

FPS = 24
ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2/data/test_Pixabay100"


def score(path):
    """VBench's temporal_flickering score for one video. Higher is better."""
    return float(cal_score(path))


def score_aligned(path, grid_ref=None, fps=FPS):
    """VBench's arithmetic over our aligned frames, so grids and frame maps match.

    `grid_ref` is the video whose resolution and frame count define the grid --
    pass a generated video when scoring the source, so both land on the same
    basis the other two metrics use.
    """
    h, w, n, _ = video_size(grid_ref or path)
    frames, _ = read_aligned(path, n, h, w, fps=fps)
    seq = mae_seq(list(frames))
    mae = float(np.mean(seq))
    return {"vbench_aligned": (255.0 - mae) / 255.0,
            "mean_abs_diff": mae,
            "mad_std": float(np.std(seq)),
            "frames": len(frames)}


def grid_paths(clip, row=None):
    """The sweep videos for a clip, optionally just one u row (e.g. "u08")."""
    d = f"{ROOT}/{clip}"
    pat = re.compile(rf"{clip}_nf345_ctx1g_s20_u(\d+)_t(\d+)\.mp4")
    out = []
    for name in sorted(os.listdir(d)):
        m = pat.fullmatch(name)
        if m and (row is None or f"u{m.group(1)}" == row):
            out.append((int(m.group(1)) / 10, int(m.group(2)) / 10, f"{d}/{name}"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="videos to score")
    p.add_argument("--grid", default=None, help="score a clip's whole u/t sweep")
    p.add_argument("--row", default=None, help="limit --grid to one u row (e.g. u08)")
    p.add_argument("--aligned", action="store_true",
                   help="also score through our read_aligned frames (needed for the source)")
    p.add_argument("--source", default=None,
                   help="also score this source video, aligned onto the sweep's grid")
    args = p.parse_args()

    if args.grid:
        items = grid_paths(args.grid, args.row)
        if not items:
            raise SystemExit(f"no sweep videos for clip {args.grid} row {args.row}")
    else:
        items = [(None, None, v) for v in args.videos]
        if not items:
            raise SystemExit("give some videos, or --grid CLIP")
    ref = items[0][2]

    rows = []
    if args.source:
        r = {"label": "SOURCE", "vbench": score(args.source)}
        r.update(score_aligned(args.source, grid_ref=ref))
        rows.append(r)
    for u, t, path in items:
        label = os.path.basename(path) if u is None else f"u{u:.1f} t{t:.1f}"
        r = {"label": label, "vbench": score(path), "u": u, "t": t}
        if args.aligned or args.source:
            r.update(score_aligned(path, grid_ref=ref))
        rows.append(r)

    wide = "vbench_aligned" in rows[0]
    head = f"{'config':>18} {'vbench':>9}"
    if wide:
        head += f" {'aligned':>9} {'MAE':>7} {'sd':>6}"
    print(head)
    for r in rows:
        line = f"{r['label']:>18} {r['vbench']:9.5f}"
        if wide:
            line += (f" {r['vbench_aligned']:9.5f} {r['mean_abs_diff']:7.3f} "
                     f"{r['mad_std']:6.3f}")
        print(line)

    scored = [r for r in rows if r.get("u") is not None]
    if len(scored) > 1:
        best = max(scored, key=lambda r: r["vbench"])
        print(f"\nbest: u{best['u']:.1f} t{best['t']:.1f}  ({best['vbench']:.5f}, "
              f"higher is better)")
        spread = max(r["vbench"] for r in scored) - min(r["vbench"] for r in scored)
        print(f"spread across {len(scored)} configs: {spread:.5f} "
              f"-- see the docstring on reading small gaps")


if __name__ == "__main__":
    main()
