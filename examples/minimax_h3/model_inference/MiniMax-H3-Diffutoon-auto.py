"""One real-world video in, one anime-style long video out.

    python MiniMax-H3-Diffutoon-auto.py /path/to/input.mp4

Three preprocessing steps run first, each as its own subprocess so their models
leave the GPU before the generation pipeline comes up:

    diffsynth.utils.controlnet.video_lineart  input.mp4 -> <work>/lineart.mp4
    diffsynth.utils.controlnet.video_blur     input.mp4 -> <work>/blur15.mp4
    diffsynth.utils.controlnet.video_caption  input.mp4 -> <work>/prompt_segments_w<N>_c<C>g.json

Their outputs are cached; a rerun skips any step whose output exists unless
--force is given.

The video is rendered as overlapping windows chained by a pinned context, two
ControlNet branches fed by the lineart and blur tracks, and every window gets
its own caption keyed by its start frame. Windows are NUM_FRAMES frames except
the last, which shrinks to the nearest 17n+5 that still covers what is left --
so a clip shorter than one window renders as a single short window rather than
being padded out to NUM_FRAMES.
"""
import argparse
import json
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
import subprocess
import sys

import torch
from PIL import Image

from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, VAE_CLIP_FRAMES, latents_fully_covered)
from diffsynth.utils.controlnet import video_blur, video_caption, video_lineart
from diffsynth.utils.data import LowMemoryVideo
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

# ---------------------------------------------------------------- configuration
# HEIGHT/WIDTH default to the control deck's aspect ratio scaled to MAX_PIXELS;
# 1032192 = 768*1344 is the resolution the ControlNet was trained at.
HEIGHT, WIDTH = None, None
MAX_PIXELS     = 1032192
NUM_FRAMES     = 328         # window size, must be 17n+5
CONTEXT_FRAMES = 1          # pinned tail from the previous segment
FPS            = 24
STEPS          = 4
SEED           = 43
BLUR_SIGMA     = 15
CONTROL_SCALE  = [0.4, 0.4]  # branch 0 = public union (lineart), branch 1 = self-trained (blur)
VRAM_HEADROOM  = 24          # GiB kept free for activations

TRAINED_CONTROLNET = "models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors"
STYLE_PREFIX = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video."

# The failing allocation was a 4.78GB contiguous request with 14.6GB reserved but
# unusable, i.e. fragmentation rather than a genuine shortfall.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------- the CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="a real-world video; everything else is derived from it")
    p.add_argument("-o", "--output", default=None, help="output mp4 (default <stem>_diffutoon.mp4 next to the input)")
    p.add_argument("--work-dir", default=None, help="where the derived lineart/blur/prompt land (default <stem>_auto/)")
    p.add_argument("--total-frames", type=int, default=0, help="output length at 24fps; 0 = as much as the input covers")
    p.add_argument("--sigma", type=float, default=BLUR_SIGMA, help=f"blur sigma (default {BLUR_SIGMA})")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--control-scale", type=float, nargs=2, default=CONTROL_SCALE, metavar=("UNION", "TRAINED"),
                   help="per-branch ControlNet scale: branch 0 the public union (lineart), branch 1 the self-trained checkpoint (blur)")
    p.add_argument("--prompt", default=None, help="skip per-segment captioning and use this text for every segment")
    p.add_argument("--height", type=int, default=HEIGHT, help="output height; derived from the control deck under --max-pixels by default")
    p.add_argument("--width", type=int, default=WIDTH, help="output width; see --height")
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help=f"pixel budget for the derived resolution (default {MAX_PIXELS})")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES, help=f"frames per window (default {NUM_FRAMES}); must be 17n+5")
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES, help=f"pinned tail from the previous segment (default {CONTEXT_FRAMES}); must be a latent boundary")
    p.add_argument("--controlnet", default=TRAINED_CONTROLNET, help=f"branch 1 checkpoint (default {TRAINED_CONTROLNET})")
    p.add_argument("--structure-controlnet", default=None,
                   help="branch 0 checkpoint; default is PAI MiniMax-H3-Fun-Controlnet-Union")
    p.add_argument("--control-layers", default=None,
                   help="comma DiT block ids for BOTH ControlNet branches, e.g. 0,5,10,15,20,25,30,35,40")
    p.add_argument("--control-b", default=None,
                   help="override branch-1 control video path (e.g. the raw RGB input instead of Gaussian blur)")
    p.add_argument("--force", action="store_true", help="redo the preprocessing steps even if their outputs exist")
    return p.parse_args()


args = parse_args()
NUM_FRAMES = args.num_frames
CONTEXT_FRAMES = args.context_frames

PIN = latents_fully_covered(CONTEXT_FRAMES)
PIN_TABLE = [f for f in range(1, NUM_FRAMES) if latents_fully_covered(f) != latents_fully_covered(f - 1)]
if NUM_FRAMES % VAE_CLIP_FRAMES != 5:
    sys.exit(f"--num-frames={NUM_FRAMES} is not 17n+5; try " + ", ".join(str(17 * n + 5) for n in range(1, 8)))
if CONTEXT_FRAMES not in PIN_TABLE:
    sys.exit(f"--context-frames={CONTEXT_FRAMES} is not a latent boundary; pick one of {PIN_TABLE[:12]}")
STRIDE = NUM_FRAMES - CONTEXT_FRAMES

INPUT = os.path.abspath(args.input)
if not os.path.exists(INPUT):
    sys.exit(f"input not found: {INPUT}")
stem = os.path.splitext(os.path.basename(INPUT))[0]
WORK = os.path.abspath(args.work_dir or os.path.join(os.path.dirname(INPUT), f"{stem}_auto"))
OUTPUT = os.path.abspath(args.output or os.path.join(os.path.dirname(INPUT), f"{stem}_diffutoon.mp4"))
os.makedirs(WORK, exist_ok=True)
CONTROL_A = os.path.join(WORK, "lineart.mp4")               # branch 0, public union
CONTROL_B = (os.path.abspath(args.control_b) if args.control_b
             else os.path.join(WORK, f"blur{args.sigma:g}.mp4"))  # branch 1


# ---------------------------------------------------------------- preprocessing
def step(name, out_path, module, extra):
    """Run one preprocessing module unless its output is already there.

    `module` is the imported module itself, so a wrong name fails at import
    rather than as a subprocess returning nonzero. It still runs out of process:
    the lineart annotator and the 27B captioner both sit on the GPU and the
    generation pipeline wants nearly all of it, so the memory has to be actually
    released rather than merely freed.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force:
        print(f"[skip] {name}: {out_path} already there", flush=True)
        return
    print(f"[run ] {name} -> {out_path}", flush=True)
    r = subprocess.run([sys.executable, "-m", module.__name__, INPUT, out_path, *extra])
    if r.returncode != 0 or not os.path.exists(out_path):
        sys.exit(f"{name} failed (rc={r.returncode}); see the output above")


step("lineart", CONTROL_A, video_lineart, [])
if args.control_b:
    if not os.path.exists(CONTROL_B):
        sys.exit(f"--control-b not found: {CONTROL_B}")
    print(f"[ctrl] branch-1 control video override: {CONTROL_B}", flush=True)
else:
    step("blur", CONTROL_B, video_blur, [str(args.sigma)])


# ------------------------------------------------------------ output resolution
def resolve_height_width():
    """Keep the control deck's aspect ratio, scaled to fit --max-pixels.

    Hardcoding 768x1344 center-crops anything that is not 16:9. This is
    diffsynth's own ImageCropAndResize sizing -- scale by sqrt(w*h/max_pixels),
    snap to the 32 grid -- with two deviations: it scales up as well as down, so a
    small source is not generated far below the trained resolution, and it tries
    both 32-grid neighbours per side instead of flooring. Flooring both would give
    a 16:9 source 1344x736, and read_video_audio would then crop 5% of its height.
    Everything stays at or under the budget because the pipeline's
    check_resize_height_width() rounds *up* to 32.
    """
    if args.height is not None and args.width is not None:
        return args.height, args.width
    if args.height is not None or args.width is not None:
        sys.exit("--height and --width must be given together, or neither")

    from diffsynth.utils.data import VideoData

    src_h, src_w = VideoData(CONTROL_B).shape()
    scale = (args.max_pixels / (src_w * src_h)) ** 0.5
    h0, w0 = int(src_h * scale) // 32 * 32, int(src_w * scale) // 32 * 32
    want = src_w / src_h
    cand = [(abs(w / h - want), -(w * h), h, w)
            for h in (h0, h0 + 32) for w in (w0, w0 + 32) if w * h <= args.max_pixels]
    if not cand:
        sys.exit(f"cannot fit {src_w}x{src_h} into {args.max_pixels} px on a 32 grid")
    _, _, h, w = min(cand)
    print(f"[res ] {os.path.basename(CONTROL_B)} is {src_w}x{src_h} -> generating "
          f"{w}x{h} ({w * h} px, budget {args.max_pixels})", flush=True)
    return h, w


HEIGHT, WIDTH = resolve_height_width()


# ---------------------------------------------------------------- the schedule
def available_frames(path):
    """How many frames at FPS the source can actually fill.

    read_video_audio maps output frame k to source frame round(k*source_fps/FPS)
    and stops once that index runs past the end -- quietly, returning fewer frames
    than asked for. So the ceiling is the number of k still inside the file, which
    for a 30fps source is ceil(T * 24/30), not T.
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


def snap_window(frames):
    """Smallest legal window (17n+5) that still covers `frames`."""
    n = max(1, -(-(frames - 5) // VAE_CLIP_FRAMES))
    return VAE_CLIP_FRAMES * n + 5


CAP = min(available_frames(CONTROL_A), available_frames(CONTROL_B))
if args.total_frames > CAP:
    sys.exit(f"--total-frames {args.total_frames} exceeds what the input covers ({CAP})")
TOTAL_FRAMES = args.total_frames or CAP
if TOTAL_FRAMES < 1:
    sys.exit(f"{os.path.basename(INPUT)} covers no frames at {FPS}fps")

# Full windows while one does not reach the end; the last one shrinks to the
# smallest legal window covering what is left, rather than overhanging by up to
# STRIDE-1 frames. A clip shorter than NUM_FRAMES is a single short window. The
# last window still overshoots by under 17 frames and those are truncated.
segments = []
g = 0
while True:
    remaining = TOTAL_FRAMES - g
    if remaining <= NUM_FRAMES:
        segments.append((g, snap_window(remaining)))
        break
    segments.append((g, NUM_FRAMES))
    g += STRIDE

CONTROL_FRAMES = segments[-1][0] + segments[-1][1]
DISCARDED = CONTROL_FRAMES - TOTAL_FRAMES
print(f"{len(segments)} segments · window {NUM_FRAMES} · context {CONTEXT_FRAMES} "
      f"({PIN} latents) · stride {STRIDE} · last window {segments[-1][1]}", flush=True)
print(f"assembling {CONTROL_FRAMES} -> truncating to {TOTAL_FRAMES} "
      f"({TOTAL_FRAMES / FPS:.1f}s of {CAP} available)"
      + (f" · discarding {DISCARDED}" if DISCARDED else ""), flush=True)


# -------------------------------------------------------- per-segment captions
# One caption per window. A single whole-clip caption describes the entire arc,
# which tells every window to perform all of it, so each segment restarts the arc
# and the seams jump. Captions are keyed by window start g0 and cached on disk,
# keyed by window and context too since changing either changes the schedule. The
# `g` suffix marks captions sampled from the generated part of each window only,
# past the pinned context; caches without it are not interchangeable. The last
# window is shorter than NUM_FRAMES but the captioner clamps every span to what
# the file covers, so it samples real frames either way.
SEGMENT_PROMPT_JSON = os.path.join(WORK, f"prompt_segments_w{NUM_FRAMES}_c{CONTEXT_FRAMES}g.json")


def resolve_segment_captions():
    """-> {g0: detailed_description} covering every window in `segments`."""
    starts = [g0 for g0, _ in segments]
    if args.prompt is not None:
        print("[skip] caption: --prompt given, every segment uses it", flush=True)
        return {g0: args.prompt for g0 in starts}

    need = set(starts)
    if os.path.exists(SEGMENT_PROMPT_JSON) and not args.force:
        with open(SEGMENT_PROMPT_JSON) as fh:
            blob = json.load(fh)
        have = {int(k): v for k, v in blob.get("captions", {}).items()}
        if blob.get("window") == NUM_FRAMES and need <= set(have):
            print(f"[skip] caption: {len(need)} windows already in "
                  f"{os.path.basename(SEGMENT_PROMPT_JSON)}", flush=True)
            return {g0: have[g0] for g0 in starts}

    step("caption", SEGMENT_PROMPT_JSON, video_caption,
         ["--g0", ",".join(str(g0) for g0 in starts),
          "--window", str(NUM_FRAMES), "--context", str(CONTEXT_FRAMES)])
    with open(SEGMENT_PROMPT_JSON) as fh:
        have = {int(k): v for k, v in json.load(fh)["captions"].items()}
    missing = sorted(need - set(have))
    if missing:
        sys.exit(f"caption missing for windows {missing}; see the output above")
    return {g0: have[g0] for g0 in starts}


SEGMENT_CAPTIONS = resolve_segment_captions()
SEGMENT_PROMPTS = [f"{STYLE_PREFIX}\ndetailed_description:\n{SEGMENT_CAPTIONS[g0]}" for g0, _ in segments]
print(f"\nprompt for segment 0 of {len(segments)}:\n{SEGMENT_PROMPTS[0]}\n", flush=True)
for k, (g0, win) in enumerate(segments):
    print(f"  seg {k:2d} g0={g0:4d} [{g0/FPS:5.1f}s] win={win:4d} "
          f"{SEGMENT_CAPTIONS[g0][:80].replace(chr(10), ' ')}...", flush=True)

# ------------------------------------------------------------------- the model
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

# branch 0: structure CN override or public PAI union
if args.structure_controlnet:
    branch0 = ModelConfig(path=args.structure_controlnet, **vram_config)
    print(f"[cn0] structure={args.structure_controlnet}", flush=True)
else:
    branch0 = ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union",
                          origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config)
    print("[cn0] PAI MiniMax-H3-Fun-Controlnet-Union", flush=True)
print(f"[cn1] trained={args.controlnet}", flush=True)

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        branch0,
        ModelConfig(path=args.controlnet, **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    # `vram_limit` is a high-water mark, not a budget: a layer onloads only while
    # *used* memory is under it, so the headroom left here is all the activations
    # get. Measured weights near 80GB at this resolution, so reserve 24GB.
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
)
print(f"ControlNet branches: {len(pipe.controlnet)}")
if args.control_layers:
    from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3MultiControlNet
    branches = list(pipe.controlnet) if isinstance(pipe.controlnet, MiniMaxH3MultiControlNet) else [pipe.controlnet]
    for i, branch in enumerate(branches):
        if not isinstance(branch, MiniMaxH3ControlNet):
            raise TypeError(f"branch {i} is {type(branch)}")
        before = list(branch.control_layers)
        branch.set_control_layers(args.control_layers)
        print(f"[layers] branch {i}: {before} -> {list(branch.control_layers)} ({len(branch.blocks)} blocks)", flush=True)
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))


# --------------------------------------------------------------- control tracks
def load_control(path, num_frames, max_padding):
    """Read `num_frames` frames resampled onto FPS.

    read_video_audio quietly returns fewer frames when the source runs out. The
    last window reaches past TOTAL_FRAMES and that overhang is truncated at the
    end anyway, so a shortfall is padded with the last frame -- but only inside
    the discarded part, else frozen control would drive real output frames.
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
# Each segment past the first pins the previous tail (retake_video + the region
# starting at CONTEXT_FRAMES, whose first fully covered latent is PIN) and takes
# its own first frame as the FL2VA keyframe, then the overlap is cross-faded.
all_frames = []
audio_chunks = []
audio_sr = pipe.audio_vae.sample_rate
print(f"pinning {PIN} latents · {CONTEXT_FRAMES} frames", flush=True)

for k, (g0, win) in enumerate(segments):
    kwargs = dict(
        prompt=SEGMENT_PROMPTS[k],
        height=HEIGHT, width=WIDTH, num_frames=win,
        num_inference_steps=STEPS,
        seed=args.seed + k,                  # vary per segment or every one repeats
        control_video=[control_a[g0:g0 + win], control_b[g0:g0 + win]],
        control_scale=list(args.control_scale),
    )

    overlap = len(all_frames) - g0 if k > 0 else 0
    if k > 0:
        assert 0 < overlap <= win, f"segment {k}: overlap {overlap}"
        context = all_frames[-overlap:]
        # The pinned context followed by the control track for the span being
        # generated: real motion there beats `overlap` copies of one still frame,
        # and it cannot leak into the pinned latents (causal temporal conv).
        kwargs["retake_video"] = context + control_b[g0 + overlap:g0 + win]
        kwargs["frame_regions_to_retake"] = [(overlap, win)]
        kwargs["keyframes"] = [context[0]]
        kwargs["keyframe_indices"] = [0]

    video, audio = pipe(**kwargs)

    if overlap:
        # Linear cross-fade, complementary ramps (a mirrored pair sums above 1 and
        # brightens the overlap). One overlapping frame gets the even average.
        base = len(all_frames) - overlap
        for i in range(overlap):
            all_frames[base + i] = Image.blend(
                all_frames[base + i].convert("RGB"), video[i].convert("RGB"),
                0.5 if overlap == 1 else i / (overlap - 1))
    all_frames.extend(video[overlap:])
    if audio is not None:
        audio_chunks.append(audio[:, int((overlap / FPS) * audio_sr):])

    print(f"  seg {k+1}/{len(segments)} · global [{g0},{g0+win}) "
          f"· +{len(video) - overlap} frames · total {len(all_frames)}", flush=True)

all_frames = all_frames[:TOTAL_FRAMES]
full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
if full_audio is not None:
    full_audio = full_audio[:, :int(len(all_frames) / FPS * audio_sr)]

write_video_audio(video=all_frames, audio=full_audio, output_path=OUTPUT,
                  fps=FPS, audio_sample_rate=audio_sr)
