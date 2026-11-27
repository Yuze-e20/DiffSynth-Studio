"""9blocks long-video with latent-space pin (ctx=5 frames = 2 latents), no FL2VA.

Previous window's final 2 video latents (covering its last 5 frames, the 17n+5
tail) are copied into the next window's first 2 latent slots and held by the
retake denoise mask (mask=0). No pixel re-encode, no FL2VA keyframes.

    CUDA_VISIBLE_DEVICES=4 python MiniMax-H3-Diffutoon-latent-pin-ctx5.py \\
        /path/to/13.mp4 -o out.mp4 --work-dir work
"""
import argparse, json, os, math, sys
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/models")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from safetensors import safe_open

from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, VAE_CLIP_FRAMES, VAE_CLIP_LATENTS, VAE_RATIO_T,
    latents_fully_covered, MiniMaxH3Unit_VideoRetakeEmbedder,
)
from diffsynth.utils.controlnet import video_blur, video_caption, video_lineart
from diffsynth.utils.data import LowMemoryVideo, VideoData
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio
import subprocess

HEIGHT = WIDTH = None
MAX_PIXELS = 1032192
NUM_FRAMES = 328
CONTEXT_FRAMES_DEFAULT = 5   # 5 frames -> 2 latents; 22 frames -> 7 latents
FPS = 24
STEPS = 4
SEED = 43
VRAM_HEADROOM = 24
BLUR_SIGMA = 15
CONTROL_SCALE = [0.4, 0.4]
STRUCT = "models/train/blur-720p-39-sigma20-10block1-lineart/step-1800.safetensors"
TRAINED = "models/train/blur-720p-39-sigma20-10block1-blur/step-1800.safetensors"
LAYERS = "0,5,10,15,20,25,30,35,40"
STYLE_PREFIX = (
    "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, "
    "following the outlines of the line-art video and the colors, lighting and motion of the blurred video."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--total-frames", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sigma", type=float, default=BLUR_SIGMA)
    p.add_argument("--control-scale", type=float, nargs=2, default=CONTROL_SCALE)
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES_DEFAULT,
                   help="pixel context / frames to pin via leading latents (5->2, 22->7)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


args = parse_args()
CONTEXT_FRAMES = args.context_frames
PIN_LATENTS = latents_fully_covered(CONTEXT_FRAMES)
assert PIN_LATENTS >= 1, f"ctx={CONTEXT_FRAMES} pins 0 latents"
TAIL_LATENTS = PIN_LATENTS  # copy last N latents of prev window
STRIDE = NUM_FRAMES - CONTEXT_FRAMES
print(f"[cfg ] sigma={args.sigma} scale={args.control_scale} "
      f"ctx_frames={CONTEXT_FRAMES} pin_latents={PIN_LATENTS} stride={STRIDE}", flush=True)
INPUT = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(INPUT))[0]
WORK = os.path.abspath(args.work_dir or os.path.join(os.path.dirname(INPUT), f"{stem}_auto"))
OUTPUT = os.path.abspath(args.output or os.path.join(os.path.dirname(INPUT), f"{stem}_latent_pin_ctx{CONTEXT_FRAMES}.mp4"))
os.makedirs(WORK, exist_ok=True)

LINEART = os.path.join(WORK, "lineart.mp4")
BLUR = os.path.join(WORK, f"blur{int(args.sigma) if float(args.sigma)==int(args.sigma) else args.sigma}.mp4".replace(".0", ""))
if os.path.exists(os.path.join(WORK, f"blur{int(args.sigma)}.mp4")):
    BLUR = os.path.join(WORK, f"blur{int(args.sigma)}.mp4")


def run_pre(name, out_path, module, extra):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force:
        print(f"[skip] {name}: {out_path}", flush=True); return
    print(f"[run ] {name} -> {out_path}", flush=True)
    r = subprocess.run([sys.executable, "-m", module, INPUT, out_path, *extra],
                       cwd="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")
    if r.returncode != 0 or not os.path.exists(out_path):
        sys.exit(f"{name} failed rc={r.returncode}")


run_pre("lineart", LINEART, "diffsynth.utils.controlnet.video_lineart", [])
run_pre("blur", BLUR, "diffsynth.utils.controlnet.video_blur", [str(args.sigma)])


def resolve_hw():
    src_h, src_w = VideoData(INPUT).shape()
    scale = (MAX_PIXELS / (src_w * src_h)) ** 0.5
    h0, w0 = int(src_h * scale) // 32 * 32, int(src_w * scale) // 32 * 32
    want = src_w / src_h
    cand = [(abs(w / h - want), -(w * h), h, w)
            for h in (h0, h0 + 32) for w in (w0, w0 + 32) if w * h <= MAX_PIXELS]
    _, _, h, w = min(cand)
    print(f"[res ] {src_w}x{src_h} -> {w}x{h}", flush=True)
    return h, w


HEIGHT, WIDTH = resolve_hw()


def available_frames(path):
    v = LowMemoryVideo(path)
    total = len(v)
    source_fps = float(v.reader.get_meta_data()["fps"])
    k = 0
    while int(round(k * source_fps / FPS)) < total:
        k += 1
    return k


def snap_window(frames):
    n = max(1, -(-(frames - 5) // VAE_CLIP_FRAMES))
    return VAE_CLIP_FRAMES * n + 5


CAP = available_frames(INPUT)
TOTAL_FRAMES = args.total_frames or CAP
segments = []
g = 0
while True:
    remaining = TOTAL_FRAMES - g
    if remaining <= NUM_FRAMES:
        segments.append((g, snap_window(remaining))); break
    segments.append((g, NUM_FRAMES)); g += STRIDE

CONTROL_FRAMES = segments[-1][0] + segments[-1][1]
DISCARDED = CONTROL_FRAMES - TOTAL_FRAMES
print(f"{len(segments)} segs · win {NUM_FRAMES} · ctx {CONTEXT_FRAMES} "
      f"({PIN_LATENTS} latents pin) · stride {STRIDE}", flush=True)
print(f"latent pin: copy last {TAIL_LATENTS} latents of prev window -> first "
      f"{PIN_LATENTS} of next; NO FL2VA keyframes; NO pixel retake encode", flush=True)

# captions
SEGMENT_PROMPT_JSON = os.path.join(WORK, f"prompt_segments_w{NUM_FRAMES}_c{CONTEXT_FRAMES}g.json")
starts = [g0 for g0, _ in segments]
if os.path.exists(SEGMENT_PROMPT_JSON) and not args.force:
    blob = json.load(open(SEGMENT_PROMPT_JSON))
    have = {int(k): v for k, v in blob.get("captions", {}).items()}
    if blob.get("window") == NUM_FRAMES and set(starts) <= set(have):
        SEGMENT_CAPTIONS = {g0: have[g0] for g0 in starts}
        print(f"[skip] caption {len(starts)} windows", flush=True)
    else:
        have = None
else:
    have = None
if have is None:
    print(f"[run ] caption -> {SEGMENT_PROMPT_JSON}", flush=True)
    r = subprocess.run([
        sys.executable, "-m", "diffsynth.utils.controlnet.video_caption",
        INPUT, SEGMENT_PROMPT_JSON, "--g0", ",".join(map(str, starts)),
        "--window", str(NUM_FRAMES), "--context", str(CONTEXT_FRAMES),
    ], cwd="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")
    if r.returncode != 0:
        sys.exit("caption failed")
    have = {int(k): v for k, v in json.load(open(SEGMENT_PROMPT_JSON))["captions"].items()}
    SEGMENT_CAPTIONS = {g0: have[g0] for g0 in starts}

SEGMENT_PROMPTS = [f"{STYLE_PREFIX}\ndetailed_description:\n{SEGMENT_CAPTIONS[g0]}" for g0, _ in segments]

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3MultiControlNet

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ModelConfig(path=STRUCT if os.path.isabs(STRUCT) else os.path.join("/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio", STRUCT), **vram_config),
        ModelConfig(path=TRAINED if os.path.isabs(TRAINED) else os.path.join("/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio", TRAINED), **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
)
branches = list(pipe.controlnet) if isinstance(pipe.controlnet, MiniMaxH3MultiControlNet) else [pipe.controlnet]
for i, branch in enumerate(branches):
    branch.set_control_layers(LAYERS)
    print(f"[layers] branch {i}: {list(branch.control_layers)}", flush=True)
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo",
    origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

# ---- latent pin state shared with retake embedder patch ----
_PIN = {"latents": None}  # (1,C,T_pin,H,W) clean latents to hold

_orig_retake = MiniMaxH3Unit_VideoRetakeEmbedder.process


def _patched_retake(self, pipe, retake_video, frame_regions_to_retake, video_latents, height, width, num_frames, tiled, tile_size, tile_overlap):
    """If _PIN latents set: inject them at T=0..n-1 with mask=0; else no-op."""
    if _PIN["latents"] is None:
        return {}
    pin = _PIN["latents"].to(device=video_latents.device, dtype=video_latents.dtype)
    assert pin.shape[0] == video_latents.shape[0] and pin.shape[1] == video_latents.shape[1], (
        f"channel mismatch pin {pin.shape} vs noise {video_latents.shape}")
    assert pin.shape[-2:] == video_latents.shape[-2:], (
        f"spatial mismatch pin {pin.shape} vs noise {video_latents.shape}")
    n = pin.shape[2]
    assert n <= video_latents.shape[2], f"pin T={n} > window T={video_latents.shape[2]}"
    input_latents = torch.zeros_like(video_latents)
    input_latents[:, :, :n] = pin
    mask = torch.ones(video_latents.shape[2], dtype=video_latents.dtype, device=video_latents.device)
    mask[:n] = 0  # 0 = keep pinned
    print(f"[latent-pin] inject T=0..{n-1} shape={tuple(pin.shape)} into window T={video_latents.shape[2]} "
          f"H={video_latents.shape[3]} W={video_latents.shape[4]}", flush=True)
    return {
        "input_latents_video": input_latents,
        "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *video_latents.shape[-2:]),
    }


MiniMaxH3Unit_VideoRetakeEmbedder.process = _patched_retake

# capture final latents at decode time
_CAPTURE = {}
_orig_decode = pipe.video_vae.decode_video


def _capture_decode(latents, *a, **kw):
    _CAPTURE["latents"] = latents.detach().to("cpu")
    return _orig_decode(latents, *a, **kw)


pipe.video_vae.decode_video = _capture_decode


def load_control(path, num_frames, max_padding):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=num_frames, fps=FPS)
    short = num_frames - len(frames)
    print(f"{os.path.basename(path)}: {len(frames)} @ {FPS}fps pad={short}", flush=True)
    assert short <= max_padding
    return list(frames) + [frames[-1]] * short


control_a = load_control(LINEART, CONTROL_FRAMES, DISCARDED)
control_b = load_control(BLUR, CONTROL_FRAMES, DISCARDED)

all_frames = []
audio_chunks = []
audio_sr = pipe.audio_vae.sample_rate
prev_tail_latents = None  # last TAIL_LATENTS of previous full denoise

for k, (g0, win) in enumerate(segments):
    # expected latent T for this window
    exp_t = ((win - 5) // 17) * 5 + 2
    print(f"\n=== seg {k} g0={g0} win={win} exp_latent_T={exp_t} ===", flush=True)

    if k == 0:
        _PIN["latents"] = None
    else:
        assert prev_tail_latents is not None
        # pin first PIN_LATENTS (=2) slots
        _PIN["latents"] = prev_tail_latents
        assert prev_tail_latents.shape[2] == PIN_LATENTS, f"tail T={prev_tail_latents.shape[2]} != pin {PIN_LATENTS}"

    kwargs = dict(
        prompt=SEGMENT_PROMPTS[k],
        height=HEIGHT, width=WIDTH, num_frames=win,
        num_inference_steps=STEPS,
        seed=args.seed + k,
        control_video=[control_a[g0:g0 + win], control_b[g0:g0 + win]],
        control_scale=list(args.control_scale),
        # Force retake unit to run even without pixel video: pass a dummy flag
        # via empty retake_video of length win so unit is entered — our patch
        # ignores the pixels when _PIN is set. For seg0 _PIN is None -> {}.
        retake_video=[Image.new("RGB", (WIDTH, HEIGHT))] * win if k > 0 else None,
        frame_regions_to_retake=[(CONTEXT_FRAMES, win)] if k > 0 else None,
        # explicitly NO keyframes / FL2VA
    )

    _CAPTURE.clear()
    video, audio = pipe(**kwargs)
    assert "latents" in _CAPTURE, "failed to capture video latents at decode"
    latents = _CAPTURE["latents"]  # cpu
    print(f"  captured latents {tuple(latents.shape)} (expect T={exp_t})", flush=True)
    assert latents.shape[2] == exp_t, f"latent T {latents.shape[2]} != {exp_t}"

    # tail for next pin: last TAIL_LATENTS along T
    prev_tail_latents = latents[:, :, -TAIL_LATENTS:].contiguous()
    print(f"  saved tail latents {tuple(prev_tail_latents.shape)} for next pin", flush=True)

    overlap = CONTEXT_FRAMES if k > 0 else 0
    if overlap:
        # optional pixel cross-fade on the 5-frame overlap for display continuity
        base = len(all_frames) - overlap
        assert base == g0, f"base {base} != g0 {g0}"
        for i in range(overlap):
            all_frames[base + i] = Image.blend(
                all_frames[base + i].convert("RGB"), video[i].convert("RGB"),
                0.5 if overlap == 1 else i / (overlap - 1))
    all_frames.extend([f.convert("RGB") for f in video[overlap:]])
    if audio is not None:
        audio_chunks.append(audio[:, int((overlap / FPS) * audio_sr):])
    print(f"  +{len(video)-overlap} frames total={len(all_frames)}", flush=True)

all_frames = all_frames[:TOTAL_FRAMES]
full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
if full_audio is not None:
    full_audio = full_audio[:, :int(len(all_frames) / FPS * audio_sr)]
try:
    write_video_audio(video=all_frames, audio=full_audio, output_path=OUTPUT, fps=FPS, audio_sample_rate=audio_sr)
except Exception as e:
    print(f"[warn] {e}; imageio fallback", flush=True)
    import imageio.v2 as iio, numpy as np
    iio.mimwrite(OUTPUT, [np.asarray(f) for f in all_frames], fps=FPS, codec="libx264", quality=8)
print(f"WROTE {OUTPUT} n={len(all_frames)}", flush=True)

# restore
MiniMaxH3Unit_VideoRetakeEmbedder.process = _orig_retake
pipe.video_vae.decode_video = _orig_decode
