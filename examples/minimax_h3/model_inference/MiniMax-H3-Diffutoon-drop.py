"""One real-world video in, one anime-style long video out.

Everything the generator needs is derived from the input clip, so the only
argument is the video itself:

    python MiniMax-H3-Diffutoon-auto.py /path/to/input.mp4

Three preprocessing steps run first, each as its own subprocess so the models
they load are gone from the GPU before the 27GB generation pipeline comes up
(the captioner alone is a 27B model):

    data/script/video2lineart.py   input.mp4 -> <work>/lineart.mp4
    data/script/restorevideo.py    input.mp4 -> <work>/blur15.mp4   (sigma=15)
    data/script/video2prompt_segments.py
                                   input.mp4 -> <work>/prompt_segments_w56_c13.json

The scripts are invoked rather than re-implemented, so this file and the
standalone tools can never drift apart. Their outputs are cached: a rerun skips
any step whose output is already on disk unless --force is given.

The video is rendered as overlapping windows, and each window gets its OWN
caption, keyed by its start frame. A single whole-clip caption describes the
whole arc and so tells every window to perform all of it, which makes each
segment restart the arc and the seams visibly jump; per-window captions measured
roughly half the seam discontinuity on a 40s bloom. Each caption is dropped into
the `detailed_description` slot of the style prompt, which is otherwise shared:

    <STYLE_OPENING>
    detailed_description
    <caption text>

Generation itself is `MiniMax-H3-Controlnet-Multi-LongVideo.py` unchanged in
substance -- 124-frame windows chained by a 5-frame pinned context, two
ControlNet branches fed by the lineart and blur tracks. The one thing that is
no longer hardcoded is TOTAL_FRAMES: it defaults to every frame the input can
fill at 24fps, so the last window overhangs the end and is truncated.
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
import torch

ROOT = "/mnt/nas3/sunyuzework/Diffutoon-2"
SCRIPT_DIR = f"{ROOT}/data/script"
PYTHON = sys.executable

# `controlnet_aux` (the lineart detector) and its cv2/skimage/timm dependencies are
# not installed in the conda env -- they live here instead, so an install cannot
# disturb a env shared with running training/sweep jobs. Only the preprocessing
# subprocesses need them; this process never imports them.
PYDEPS = f"{ROOT}/data/pydeps"

# ---------------------------------------------------------------- configuration
# Output resolution. Left at None, it is derived from the blur control deck so the
# output keeps the source's aspect ratio, scaled to fit MAX_PIXELS; see
# resolve_height_width(). 1032192 is 768*1344, the resolution the ControlNet was
# trained at, so a 16:9 source lands exactly on it.
HEIGHT, WIDTH = None, None
MAX_PIXELS    = 1032192
NUM_FRAMES    = 124         # window size, must be 17n+5
CONTEXT_FRAMES = 22          # pinned tail from the previous segment; see PIN_TABLE
FPS           = 24
STEPS         = 4
SEED          = 43
BLUR_SIGMA    = 20
CONTROL_SCALE = [0.3, 0.7]   # branch 0 = public union (lineart), branch 1 = self-trained (blur)

# Also hand every continuation segment its first frame through FL2VA's native
# first-frame conditioning, on top of the retake context region. The two act on
# different channels: retake pins the first PIN *latents* of the noisy video
# stream, while a keyframe is encoded into its own condition rows that the FL2VA
# DiT was trained to continue from. False falls back to retake-only.
USE_FIRST_FRAME = True

# Carry the previous segment's audio tail into the next segment's audio latents the
# same way retake carries the video tail. Off: every segment generates its audio
# from pure noise, so the audio never inherits the previous segment's drift. The
# video tail is still pinned either way -- the two streams are separate latent
# spaces and the audio retake unit is skipped entirely when retake_audio is absent.
RETAKE_AUDIO = False

# Pin the previous segment's tail into this segment's video latents. Off: nothing
# is pinned and continuity rests entirely on FL2VA's own first-frame conditioning,
# which is what the model was trained for -- CONTEXT_FRAMES then only says how many
# frames the windows overlap, and USE_FIRST_FRAME feeds the first of them as the
# keyframe. Retake costs sharpness (ctx 17 measured at 33% against no-retake), so
# leaning on FL2VA alone is worth testing.
USE_RETAKE = True

# Make the final window end exactly at TOTAL_FRAMES by starting it further back,
# instead of letting it reach past the end and truncating. That window then
# overlaps its predecessor by more than CONTEXT_FRAMES -- handled per segment
# below -- but no control frame is padded and no generated frame is thrown away.
ALIGN_LAST = False

# Cross-fade the overlap instead of cutting at it. Segment k regenerates the
# CONTEXT_FRAMES it was pinned on and those re-renders are normally discarded
# (`video[new_from:]`), so the assembled video jumps from segment k-1's decode
# straight to segment k's at one frame. Both show the same content but differ by
# a VAE round-trip, which makes that jump visible; blending spreads it over the
# overlap. False restores the hard cut.
BLEND_CONTEXT = True

VAE_CLIP = 17
VRAM_HEADROOM = 24          # GiB kept free for activations; see vram_limit below

# The failing allocation was a 4.78GB contiguous request with 14.6GB reserved but
# unusable, i.e. fragmentation rather than a genuine shortfall. Expandable segments
# let the allocator grow a block instead of needing one contiguous run.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
assert NUM_FRAMES % VAE_CLIP == 5, "num_frames must be 17n+5"

# STYLE_OPENING = (
#     "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork "
#     "with clean line work, vibrant colors, and expressive character designs.")

TRAINED_CONTROLNET = f"{ROOT}/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors"
# TOON_LORA = f"{ROOT}/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split-480p-124_merged/step-2600.safetensors"


# ---------------------------------------------------------------------- the CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="a real-world video; everything else is derived from it")
    p.add_argument("-o", "--output", default=None,
                   help="output mp4 (default <input stem>_diffutoon.mp4 next to the input)")
    p.add_argument("--work-dir", default=None,
                   help="where the derived lineart/blur/prompt land "
                        "(default <input stem>_auto/ next to the input)")
    p.add_argument("--total-frames", type=int, default=0,
                   help="output length at 24fps; 0 = as much as the input covers")
    p.add_argument("--sigma", type=float, default=BLUR_SIGMA, help=f"blur sigma (default {BLUR_SIGMA})")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--control-scale", type=float, nargs=2, default=CONTROL_SCALE,
                   metavar=("UNION", "TRAINED"),
                   help="per-branch ControlNet scale: branch 0 is the public union "
                        "(lineart), branch 1 the self-trained checkpoint (blur)")
    p.add_argument("--prompt", default=None,
                   help="skip per-segment captioning and use this one text as "
                        "detailed_description for every segment")
    p.add_argument("--height", type=int, default=HEIGHT,
                   help="output height; by default it is derived from the control "
                        "deck's aspect ratio under --max-pixels")
    p.add_argument("--width", type=int, default=WIDTH,
                   help="output width; see --height")
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                   help=f"pixel budget for the derived resolution (default "
                        f"{MAX_PIXELS} = 768*1344); ignored when both --height "
                        f"and --width are given")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                   help=f"frames generated per window (default {NUM_FRAMES}); "
                        f"must be 17n+5")
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES,
                   help=f"pinned tail from the previous segment (default "
                        f"{CONTEXT_FRAMES}); must be a latent boundary, see PIN_TABLE")
    p.add_argument("--controlnet", default=TRAINED_CONTROLNET,
                   help="branch 1 checkpoint, i.e. the self-trained ControlNet "
                        "(default the step-1600 sigma20 run)")
    p.add_argument("--no-retake", dest="retake", action="store_false", default=USE_RETAKE,
                   help="do not pin the previous tail; rely on FL2VA first-frame "
                        "conditioning alone for continuity")
    p.add_argument("--align-last", action="store_true", default=ALIGN_LAST,
                   help="start the last window far enough back that it ends exactly "
                        "at the end, so no control frame is padded")
    p.add_argument("--force", action="store_true",
                   help="redo the preprocessing steps even if their outputs exist")
    return p.parse_args()


args = parse_args()
# Everything downstream reads the module-level name, including the PIN_TABLE
# assertion and the segment-caption cache key, so override it here, once.
CONTEXT_FRAMES = args.context_frames
NUM_FRAMES = args.num_frames
USE_RETAKE = args.retake
ALIGN_LAST = args.align_last
# The module-level assert above only saw the default; PIN_TABLE and the window
# arithmetic below are both derived from NUM_FRAMES, so check the real value here.
if NUM_FRAMES % VAE_CLIP != 5:
    sys.exit(f"--num-frames={NUM_FRAMES} is not 17n+5; try "
             + ", ".join(str(17 * n + 5) for n in range(1, 8)))
# HEIGHT/WIDTH are resolved after the blur deck exists; see resolve_height_width().
INPUT = os.path.abspath(args.input)
if not os.path.exists(INPUT):
    sys.exit(f"input not found: {INPUT}")

stem = os.path.splitext(os.path.basename(INPUT))[0]
WORK = os.path.abspath(args.work_dir or os.path.join(os.path.dirname(INPUT), f"{stem}_auto"))
OUTPUT = os.path.abspath(args.output or os.path.join(os.path.dirname(INPUT), f"{stem}_diffutoon.mp4"))
os.makedirs(WORK, exist_ok=True)

CONTROL_A = os.path.join(WORK, "lineart.mp4")      # branch 0, public union ControlNet
CONTROL_B = os.path.join(WORK, f"blur{args.sigma:g}.mp4")   # branch 1, self-trained


# ------------------------------------------------------------- preprocessing
def step(name, out_path, cmd):
    """Run one preprocessing script unless its output is already there.

    Each runs as a subprocess on purpose: the lineart annotator and the 27B
    captioner both sit on the GPU, and the generation pipeline wants nearly all
    of it. A subprocess guarantees the memory is actually released, which
    importing them into this process would not.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force:
        print(f"[skip] {name}: {out_path} already there", flush=True)
        return
    print(f"[run ] {name} -> {out_path}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = PYDEPS + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if r.returncode != 0 or not os.path.exists(out_path):
        sys.exit(f"{name} failed (rc={r.returncode}); see the output above")


step("lineart", CONTROL_A,
     [PYTHON, f"{SCRIPT_DIR}/video2lineart.py", INPUT, CONTROL_A])
step("blur", CONTROL_B,
     [PYTHON, f"{SCRIPT_DIR}/restorevideo.py", INPUT, CONTROL_B, str(args.sigma)])


# ------------------------------------------------------------ output resolution
def resolve_height_width():
    """Keep the control deck's aspect ratio, scaled down to fit --max-pixels.

    Hardcoding 768x1344 silently center-crops anything that is not 16:9, so the
    output framing stops matching the source. This is diffsynth's own
    ImageCropAndResize sizing formula -- scale by sqrt(w*h / max_pixels), snap
    to the pipeline's 32 grid -- with two deviations, both noted below: it
    scales up as well as down, so a small source is not generated far below the
    trained resolution, and it tries both 32-grid neighbours per side instead of
    flooring. Everything stays at or under the budget on purpose: the pipeline's
    check_resize_height_width() rounds *up* to 32, so an off-grid size would get
    pushed back over.

    --height/--width given together override this.
    """
    if args.height is not None and args.width is not None:
        return args.height, args.width
    if args.height is not None or args.width is not None:
        sys.exit("--height and --width must be given together, or neither")

    from diffsynth.utils.data import VideoData

    src_h, src_w = VideoData(CONTROL_B).shape()
    scale = (args.max_pixels / (src_w * src_h)) ** 0.5
    h0 = int(src_h * scale) // 32 * 32
    w0 = int(src_w * scale) // 32 * 32
    # Among the four 32-grid corners around the ideal size, take the one that
    # still fits the budget and whose aspect ratio is closest to the source,
    # breaking ties on area. Flooring both sides (what ImageCropAndResize does)
    # would give a 16:9 source 1344x736 at 1.83, so read_video_audio would crop
    # 5% of its height; the extra candidates recover 1344x768, which is both
    # inside the budget and the resolution the ControlNet was trained at.
    want = src_w / src_h
    cand = [(abs(w / h - want), -(w * h), h, w)
            for h in (h0, h0 + 32) for w in (w0, w0 + 32)
            if w * h <= args.max_pixels]
    if not cand:
        sys.exit(f"cannot fit {src_w}x{src_h} into {args.max_pixels} px on a 32 grid")
    _, _, h, w = min(cand)
    print(f"[res ] {os.path.basename(CONTROL_B)} is {src_w}x{src_h} -> "
          f"generating {w}x{h} ({w * h} px, budget {args.max_pixels})", flush=True)
    return h, w


HEIGHT, WIDTH = resolve_height_width()

# Captioning is deferred: one caption per window, and the window schedule is only
# known once the usable frame count is measured further down. See "per-segment
# captions" below the schedule.
#
# The caption goes in verbatim under `detailed_description`, which is the slot
# the ToonLoRA was trained with. Everything above it is identical for every
# segment; only `detailed_description` varies.

# SUBJECT_DEFINITIONS = (
#     "<Video Line-art> is the reference video providing the line-art contours, "
#     "character and object outlines, structural edges, and the exact framing and "
#     "camera trajectory.\n"
#     "<Video Blur> is the reference video providing the color palette, regional "
#     "color assignment, lighting direction, character motion, and temporal structure."
# )

# SUMMARY = (
#     "[3D-render-to-2D re-rendering] Please generate the target video, which is a high-quality Japanese "
#     "anime re-rendering of a live-action source, generated with reference to both "
#     "<Video Line-art> and <Video Blur>, featuring crisp clean linework, flat "
#     "cel-shaded color regions with sharp shadow boundaries, vivid saturated colors, "
#     "and sharp details, following the contours of <Video Line-art> and the colors, "
#     "composition, animation timing, and motion of <Video Blur>."
# )

# RETENTION_ANALYSIS = (
#     "Preserve the outlines and edge placement from <Video Line-art>, and redraw them "
#     "as clean hand-drawn anime linework.\n"
#     "Preserve the color palette, lighting, camera motion, character movements and "
#     "pacing from <Video Blur>, and replace photographic texture and gradients with "
#     "flat cel-shaded regions."
# )
prefix = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video."

def build_prompt(detailed_description):
    return (
        f"{prefix}\n"
        f"detailed_description:\n{detailed_description}"
    )


# ---------------------------------------------- the pinned-context arithmetic
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
import diffsynth.pipelines.minimax_h3_audio_video as mm
from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
from PIL import Image
from diffsynth.utils.data import LowMemoryVideo
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio


def latents_to_pin(frames, clip=VAE_CLIP, per_clip=5):
    """How many leading latents the first `frames` frames fully cover.

    Each 17-frame clip encodes to 5 latents as 1+4+4+4+4, measured by editing one
    frame at a time and watching which latents move:
        latent 0 <- frame 0        latent 5 <- frame 17
        latent 1 <- frames 1..4    latent 6 <- frames 18..21
        latent 2 <- frames 5..8    latent 7 <- frames 22..25
        latent 3 <- frames 9..12
        latent 4 <- frames 13..16
    So a context can pin 1, 5, 9, 13, 17, 18, 22, 26, 30, 34 ... frames. 17 is not
    a floor -- 5 is perfectly expressible, and measured best: it costs 12% sharpness
    against a no-retake baseline where 17 costs 33% and 102 costs 73%.
    """
    n = 0
    for L in range(10 ** 6):
        clip_idx, within = divmod(L, per_clip)
        end = clip_idx * clip + (0 if within == 0 else 4 * within)   # last frame latent L sees
        if end < frames:
            n = L + 1
        else:
            break
    return n


PIN = latents_to_pin(CONTEXT_FRAMES)
PIN_TABLE = [f for f in range(1, NUM_FRAMES) if latents_to_pin(f) != latents_to_pin(f - 1)]
# context 0 chains the windows instead of overlapping them: nothing is pinned, so
# retake has nothing to do, and the only tie between segments is the FL2VA keyframe
# -- the previous window's last frame handed in as this window's frame 0.
if CONTEXT_FRAMES == 0:
    if USE_RETAKE:
        print("context 0: nothing to pin, retake disabled", flush=True)
        USE_RETAKE = False
    if not USE_FIRST_FRAME:
        sys.exit("context 0 leaves no link between segments unless USE_FIRST_FRAME is on")
else:
    assert PIN > 0, "CONTEXT_FRAMES must be at least 1"
    assert CONTEXT_FRAMES in PIN_TABLE, (
        f"CONTEXT_FRAMES={CONTEXT_FRAMES} is not a latent boundary; pick one of {PIN_TABLE[:12]}")
STRIDE = NUM_FRAMES - CONTEXT_FRAMES

_orig_retake = mm.MiniMaxH3Unit_VideoRetakeEmbedder.process


def _patched_retake(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw):
    """The stock mask starts at `floor(start / 17) * 5`, which only recognises
    multiples of 17 and floors anything shorter to 0 -- a 5-frame context silently
    pins nothing and retake becomes a no-op. Recompute the start index from the
    real frame -> latent table instead."""
    out = _orig_retake(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw)
    if not out:
        return out
    latents = out["input_latents_video"]
    mask = torch.zeros(latents.shape[2], dtype=pipe.torch_dtype, device=pipe.device)
    mask[PIN:] = 1                                   # 1 = regenerate
    return {"input_latents_video": latents,
            "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *latents.shape[-2:])}


mm.MiniMaxH3Unit_VideoRetakeEmbedder.process = _patched_retake


# ------------------------------------------- ControlNet x first-frame condition
# Drop the condition rows from the ControlNet entirely.
#
# The auto script's variants all keep the keyframe condition rows inside the
# ControlNet's sequence and only vary what is written on them. That cannot undo the
# coupling: cu_seqlens spans [text | cond | audio | video] as ONE attention segment,
# so whatever sits on those rows attends with every control row at all 5 blocks, and
# zeroing the hint afterwards only suppresses the output.
#
# Here the rows are removed instead. Every per-row tensor the ControlNet reads is
# gathered through the same `keep` index, so the surviving rows carry their original
# position values -- rope is computed per row from img_position_ids, and the block's
# adaln lookup is per row through combined_indices, so nothing re-derives a position
# from an index. The text, audio and video rows therefore see exactly the positions
# the main branch gives them; only the cond rows are gone. cu_seqlens shrinks by the
# same count so the attention segment stays one piece, and the hints are scattered
# back onto the full-length sequence with zeros where the cond rows were, which is
# what the DiT adds as a residual.
#
# The main branch is untouched: it still gets its keyframe anchor. This is also the
# layout the ControlNets were trained in -- `--extra_inputs "input_audio,control_video"`,
# never `input_image`, so a cond row is something they have never seen.
_orig_cn_forward = MiniMaxH3ControlNet.forward


def _patched_cn_forward(self, embeddings, control_rows, img_pos, audio_pos, **kw):
    n_cond = img_pos.shape[0] - control_rows.shape[0]
    if n_cond <= 0:
        return _orig_cn_forward(self, embeddings, control_rows, img_pos, audio_pos, **kw)

    seq_len = embeddings.shape[0]
    device = embeddings.device
    cond_pos = img_pos[:n_cond]

    # Every row except the condition rows, in order.
    mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    mask[cond_pos] = False
    keep = mask.nonzero(as_tuple=True)[0]

    # The cond rows sit between the text and audio blocks, so every row after them
    # shifts down by exactly n_cond. Build the new indices by inverse mapping rather
    # than assuming that, so a different layout cannot silently mis-wire this.
    remap = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    remap[keep] = torch.arange(keep.shape[0], device=device)
    new_img_pos = remap[img_pos[n_cond:]]
    new_audio_pos = remap[audio_pos]
    assert int(new_img_pos.min()) >= 0 and int(new_audio_pos.min()) >= 0, "dropped a kept row"

    kw = dict(kw)
    kw["rope_freqs"] = kw["rope_freqs"][keep]
    kw["combined_indices"] = kw["combined_indices"][keep]
    cu = kw["cu_seqlens"]
    kw["cu_seqlens"] = torch.tensor(
        [0, int(cu[1]) - n_cond, keep.shape[0]], dtype=cu.dtype, device=cu.device)
    kw["max_seqlen"] = int(cu[1]) - n_cond

    hints = _orig_cn_forward(self, embeddings[keep], control_rows,
                             new_img_pos, new_audio_pos, **kw)

    # Back onto the full sequence; the cond rows get a zero residual.
    out = []
    for hint in hints:
        full = hint.new_zeros((seq_len, hint.shape[-1]))
        full.index_copy_(0, keep, hint)
        out.append(full)
    return tuple(out)


MiniMaxH3ControlNet.forward = _patched_cn_forward


# ------------------------------------------------------------- how long to make
def available_frames(path):
    """How many frames at FPS the source can actually fill.

    `read_video_audio` maps output frame k to source frame round(k * source_fps / FPS)
    and stops as soon as that index runs past the end -- quietly, returning fewer
    frames than asked for. So the real ceiling is the number of k whose index still
    lands inside the file, which for a 30fps source is ceil(T * 24/30), not T.
    """
    v = LowMemoryVideo(path)
    total = len(v)
    source_fps = float(v.reader.get_meta_data()["fps"])
    k = 0
    while int(round(k * source_fps / FPS)) < total:
        k += 1
    print(f"{os.path.basename(path)}: {total} frames @ {source_fps:g}fps "
          f"({total / source_fps:.2f}s) -> {k} frames @ {FPS}fps", flush=True)
    return k


def check_long_enough(frames):
    """One window is the hard minimum; anything shorter cannot be generated at all."""
    if frames < NUM_FRAMES:
        sys.exit(f"input covers only {frames} frames at {FPS}fps, but one window is "
                 f"{NUM_FRAMES}; the clip is too short ({NUM_FRAMES / FPS:.1f}s needed)")
    return frames


CAP = min(available_frames(CONTROL_A), available_frames(CONTROL_B))
if args.total_frames:
    if args.total_frames > CAP:
        sys.exit(f"--total-frames {args.total_frames} exceeds what the input covers ({CAP})")
    TOTAL_FRAMES = check_long_enough(args.total_frames)
else:
    TOTAL_FRAMES = check_long_enough(CAP)

segments = []
g = 0
while g < TOTAL_FRAMES:
    segments.append(g)
    if g + NUM_FRAMES >= TOTAL_FRAMES:
        break
    g += STRIDE

if ALIGN_LAST and len(segments) > 1 and segments[-1] + NUM_FRAMES > TOTAL_FRAMES:
    # Slide the last window back onto the end. It now overlaps its predecessor by
    # more than CONTEXT_FRAMES; the loop derives each segment's overlap from how
    # far all_frames already reaches, so that is handled without a special case.
    segments[-1] = TOTAL_FRAMES - NUM_FRAMES
    assert segments[-1] > segments[-2], (
        f"--align-last cannot fit: window {NUM_FRAMES} is longer than the span "
        f"from segment {len(segments) - 2} to the end")

# Lengths the schedule lands on exactly are NUM_FRAMES + n * STRIDE; TOTAL_FRAMES is
# usually between two of them, so the last window reaches past the end. Those extra
# frames are generated and then truncated, which is why asking for every frame the
# input covers costs nothing but a little compute on the final window -- rounding
# TOTAL_FRAMES down to the schedule instead would throw away up to STRIDE-1 real
# frames of input.
CONTROL_FRAMES = segments[-1] + NUM_FRAMES
DISCARDED = CONTROL_FRAMES - TOTAL_FRAMES
print(f"{len(segments)} segments · window {NUM_FRAMES} · context {CONTEXT_FRAMES} "
      f"({PIN} latents) · stride {STRIDE}"
      + (" · last window aligned to the end" if ALIGN_LAST and len(segments) > 1
         and segments[-1] != segments[0] + STRIDE * (len(segments) - 1) else ""),
      flush=True)
print(f"assembling {CONTROL_FRAMES} -> truncating to {TOTAL_FRAMES} "
      f"({TOTAL_FRAMES / FPS:.1f}s of {CAP} available)"
      + (f" · discarding {DISCARDED}" if DISCARDED else ""), flush=True)


# ------------------------------------------------------- per-segment captions
# One caption per window. A single whole-clip caption describes the entire arc,
# which tells every window to perform all of it, so each segment restarts the arc
# and the seams jump. Captions are keyed by window start g0 and cached on disk;
# the cache is keyed by window and context too, since changing either changes the
# schedule and would otherwise silently reuse captions for the wrong spans.
# The `g` suffix marks captions sampled from the generated part of each window
# only, i.e. past the pinned context; caches without it were sampled from the
# whole window and are not interchangeable.
SEGMENT_PROMPT_JSON = os.path.join(
    WORK, f"prompt_segments_w{NUM_FRAMES}_c{CONTEXT_FRAMES}g.json")


def resolve_segment_captions():
    """-> {g0: detailed_description} covering every window in `segments`."""
    if args.prompt is not None:
        print("[skip] caption: --prompt given, every segment uses it", flush=True)
        return {g0: args.prompt for g0 in segments}

    need = set(segments)
    if os.path.exists(SEGMENT_PROMPT_JSON) and not args.force:
        with open(SEGMENT_PROMPT_JSON) as fh:
            blob = json.load(fh)
        have = {int(k): v for k, v in blob.get("captions", {}).items()}
        if blob.get("window") == NUM_FRAMES and need <= set(have):
            print(f"[skip] caption: {len(need)} windows already in "
                  f"{os.path.basename(SEGMENT_PROMPT_JSON)}", flush=True)
            return {g0: have[g0] for g0 in segments}

    # The captioner is a 27B model and wants the GPU to itself, so it runs as a
    # subprocess -- the same reason lineart does, and it has to finish before the
    # pipeline below starts loading weights.
    step("caption", SEGMENT_PROMPT_JSON,
         [PYTHON, f"{SCRIPT_DIR}/video2prompt_segments.py", INPUT, SEGMENT_PROMPT_JSON,
          "--g0", ",".join(str(g) for g in segments),
          "--window", str(NUM_FRAMES),
          "--context", str(CONTEXT_FRAMES if USE_RETAKE else 0)])
    with open(SEGMENT_PROMPT_JSON) as fh:
        have = {int(k): v for k, v in json.load(fh)["captions"].items()}
    missing = sorted(need - set(have))
    if missing:
        sys.exit(f"caption missing for windows {missing}; see the output above")
    return {g0: have[g0] for g0 in segments}


SEGMENT_CAPTIONS = resolve_segment_captions()
SEGMENT_PROMPTS = [build_prompt(SEGMENT_CAPTIONS[g0]) for g0 in segments]
print(f"\nprompt for segment 0 of {len(segments)}:\n{SEGMENT_PROMPTS[0]}\n", flush=True)
for k, g0 in enumerate(segments):
    print(f"  seg {k:2d} g0={g0:4d} [{g0/FPS:5.1f}s] "
          f"{SEGMENT_CAPTIONS[g0][:88].replace(chr(10), ' ')}...", flush=True)


# ------------------------------------------------------------------- the model
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        # Branch 0: the public union ControlNet, fed the lineart track.
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
        # Branch 1: the self-trained checkpoint, fed the blurred track.
        ModelConfig(path=args.controlnet, **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    # `vram_limit` is a high-water mark, not a budget: each layer only onloads its
    # weights while *used* memory is still under it (core/vram/layers.py:74). So the
    # headroom left here is all the activations get, and at 1568x1568x124 a single
    # MLP allocation is ~5GB. `total - 5` left exactly that much and OOM'd on the
    # first segment as soon as the allocator fragmented. Measured peaks at this
    # resolution sit near 80GB of weights, so reserve 24GB and let the rest offload.
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
)
print(f"ControlNet branches: {len(pipe.controlnet)}")
# pipe.load_lora(pipe.dit, TOON_LORA)   # disabled along with the TOON_LORA path above
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))


# --------------------------------------------------------------- control tracks
def load_control(path, num_frames, max_padding):
    """Read `num_frames` frames resampled onto FPS.

    `read_video_audio` returns FEWER frames than asked for, without complaining,
    when the source runs out. The last window reaches past TOTAL_FRAMES and those
    frames are truncated at the end anyway, so a shortfall is padded with the last
    frame -- but only within the part that gets discarded. Padding beyond that
    would drive real output frames from a frozen control, so it asserts instead.
    """
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH,
                                    num_frames=num_frames, fps=FPS)
    short = num_frames - len(frames)
    print(f"{os.path.basename(path)}: {len(frames)} frames @ {FPS}fps "
          f"({len(frames) / FPS:.2f}s)"
          + (f" · padding {short} discarded frames" if short else ""), flush=True)
    assert short <= max_padding, (
        f"{os.path.basename(path)} is {short - max_padding} frames too short: it gives "
        f"{len(frames)} frames at {FPS}fps but {num_frames - max_padding} are needed "
        f"for {TOTAL_FRAMES} real output frames")
    return frames + [frames[-1]] * short


control_a = load_control(CONTROL_A, CONTROL_FRAMES, DISCARDED)
control_b = load_control(CONTROL_B, CONTROL_FRAMES, DISCARDED)


# ------------------------------------------------------------------------- loop
all_frames = []          # the assembled long video
audio_chunks = []        # per-segment newly generated audio
prev_audio = None
audio_sr = pipe.audio_vae.sample_rate
ctx_seconds = CONTEXT_FRAMES / FPS

# Regenerate everything after the pinned context. The region's `start` only has to
# land inside the first latent that should be regenerated -- the patched mask reads
# the boundary from PIN, not from this number.
video_regions = [(CONTEXT_FRAMES, NUM_FRAMES)]
audio_regions = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]
print(f"retake regions · frames {video_regions[0]} · seconds "
      f"({audio_regions[0][0]:.4f}, {audio_regions[0][1]:.4f}) · pinning {PIN} latents")
print(f"audio retake: {'on' if RETAKE_AUDIO else 'off (each segment from pure noise)'}", flush=True)
print(f"first-frame conditioning: {'on' if USE_FIRST_FRAME else 'off'}"
      + (" · ControlNet sees no cond row (dropped)" if USE_FIRST_FRAME else ""), flush=True)
print(f"retake video: {'on' if USE_RETAKE else 'off (FL2VA first frame only)'}", flush=True)
if BLEND_CONTEXT and CONTEXT_FRAMES == 1:
    print("context blend · single frame, even 0.5 average", flush=True)
elif BLEND_CONTEXT and CONTEXT_FRAMES > 1:
    _w = [i / (CONTEXT_FRAMES - 1) for i in range(CONTEXT_FRAMES)]
    _p = ", ".join(f"{1-w:.2f}" for w in _w[:3]) + " ... " + f"{1-_w[-1]:.2f}"
    _n = ", ".join(f"{w:.2f}" for w in _w[:3]) + " ... " + f"{_w[-1]:.2f}"
    print(f"context blend · prev [{_p}] · new [{_n}]", flush=True)
else:
    print("context blend: off (hard cut)", flush=True)

for k, g0 in enumerate(segments):
    kwargs = dict(
        prompt=SEGMENT_PROMPTS[k],
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        seed=args.seed + k,                  # vary per segment or every one repeats
        control_video=[control_a[g0:g0 + NUM_FRAMES],
                       control_b[g0:g0 + NUM_FRAMES]],
        control_scale=list(args.control_scale),
    )

    # How far the assembled video already reaches into this window. Normally this
    # is CONTEXT_FRAMES, but the --align-last window starts further back and so
    # overlaps more; everything below is written in terms of this number.
    overlap = len(all_frames) - g0 if k > 0 else 0
    if k > 0:
        assert 0 <= overlap <= NUM_FRAMES, f"segment {k}: overlap {overlap}"
        # With overlap 0 the windows are back to back, so the frame this window
        # starts on is the one just assembled -- all_frames[-1], not all_frames[0].
        context = all_frames[-overlap:] if overlap else all_frames[-1:]
        if USE_RETAKE:
            # Pin the previous tail, then hand the pipeline a full-length window: the
            # context frames followed by the control track for the span being generated.
            # Those padding frames are the latent initial value for the regenerated
            # region, so real motion there beats `overlap` copies of one still frame.
            # They cannot leak into the pinned latents -- the VAE's temporal conv is
            # causal, verified bit-identical under three different paddings.
            kwargs["retake_video"] = context + control_b[g0 + overlap:g0 + NUM_FRAMES]
            kwargs["frame_regions_to_retake"] = [(overlap, NUM_FRAMES)]
            assert len(kwargs["retake_video"]) == NUM_FRAMES, len(kwargs["retake_video"])
        if USE_FIRST_FRAME:
            # all_frames currently ends at global frame g0 + overlap, so context[0]
            # is global frame g0 -- window-local frame 0, exactly what
            # keyframe_indices=[0] conditions on.
            kwargs["keyframes"] = [context[0]]
            kwargs["keyframe_indices"] = [0]
        if RETAKE_AUDIO and prev_audio is not None:
            tail = int((overlap / FPS) * audio_sr)
            kwargs["retake_audio"] = prev_audio[:, -tail:]
            kwargs["retake_audio_sample_rate"] = audio_sr
            kwargs["seconds_regions_to_retake"] = [(overlap / FPS, NUM_FRAMES / FPS)]

    video, audio = pipe(**kwargs)

    new_from = overlap
    if k > 0 and BLEND_CONTEXT and overlap:
        # Linear cross-fade over the overlap. The two ramps must be complementary
        # (w_new = 1 - w_prev), not mirrored -- a mirrored pair like (1.0 .. 0.2)
        # against (0.2 .. 1.0) sums to 1.2 everywhere and brightens the overlap by
        # 20%. i/(n-1) reaches 1.0 on the last overlap frame, so it meets segment
        # k's own output with no step left over. A single overlapping frame has no
        # ramp to walk, so it gets the even 0.5 average of the two renders.
        n = overlap
        base = len(all_frames) - n          # global index of overlap frame 0
        for i in range(n):
            all_frames[base + i] = Image.blend(
                all_frames[base + i].convert("RGB"),
                video[i].convert("RGB"),
                0.5 if n == 1 else i / (n - 1),
            )
    all_frames.extend(video[new_from:])
    if audio is not None:
        cut = int((new_from / FPS) * audio_sr)
        audio_chunks.append(audio[:, cut:])
        prev_audio = audio

    print(f"  seg {k+1}/{len(segments)} · global [{g0},{g0+NUM_FRAMES}) "
          f"· +{len(video) - new_from} frames · total {len(all_frames)}", flush=True)

all_frames = all_frames[:TOTAL_FRAMES]
full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
if full_audio is not None:
    full_audio = full_audio[:, :int(len(all_frames) / FPS * audio_sr)]

write_video_audio(video=all_frames, audio=full_audio, output_path=OUTPUT,
                  fps=FPS, audio_sample_rate=audio_sr)
print(f"wrote {OUTPUT} · {len(all_frames)} frames · {len(all_frames)/FPS:.1f}s")