"""Ref2VA toon baseline with lineart + blur references (no RGB source as ref).

Same schedule/captions/retake (ctx=1) as Diffutoon-auto, Ref2VA turbo LoRA,
NO first-frame keyframes. Reference videos are lineart + blur20 from the work dir.
"""
import argparse, json, os, subprocess, sys
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, VAE_CLIP_FRAMES, latents_fully_covered)
from diffsynth.utils.controlnet import video_caption, video_lineart, video_blur
from diffsynth.utils.data import LowMemoryVideo, VideoData
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio

HEIGHT, WIDTH = None, None
MAX_PIXELS = 1032192
NUM_FRAMES = 328
CONTEXT_FRAMES = 1
FPS = 24
STEPS = 4
SEED = 43
VRAM_HEADROOM = 24
BLUR_SIGMA = 20
TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors"
# match Diffutoon-auto prompt intent
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
    p.add_argument("--prompt", default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    p.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--sigma", type=float, default=BLUR_SIGMA)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

args = parse_args()
NUM_FRAMES = args.num_frames
CONTEXT_FRAMES = args.context_frames
PIN = latents_fully_covered(CONTEXT_FRAMES)
PIN_TABLE = [f for f in range(1, NUM_FRAMES) if latents_fully_covered(f) != latents_fully_covered(f - 1)]
if NUM_FRAMES % VAE_CLIP_FRAMES != 5:
    sys.exit(f"bad num-frames {NUM_FRAMES}")
if CONTEXT_FRAMES not in PIN_TABLE:
    sys.exit(f"bad context-frames {CONTEXT_FRAMES}; try {PIN_TABLE[:12]}")
STRIDE = NUM_FRAMES - CONTEXT_FRAMES

INPUT = os.path.abspath(args.input)
stem = os.path.splitext(os.path.basename(INPUT))[0]
WORK = os.path.abspath(args.work_dir or os.path.join(os.path.dirname(INPUT), f"{stem}_auto"))
OUTPUT = os.path.abspath(args.output or os.path.join(os.path.dirname(INPUT), f"{stem}_ref2va_lb.mp4"))
os.makedirs(WORK, exist_ok=True)

LINEART = os.path.join(WORK, "lineart.mp4")
BLUR = os.path.join(WORK, f"blur{int(args.sigma) if float(args.sigma)==int(args.sigma) else args.sigma}.mp4".replace('.0',''))
# prefer blur20
if os.path.exists(os.path.join(WORK, "blur20.mp4")):
    BLUR = os.path.join(WORK, "blur20.mp4")

def run_pre(name, out_path, module_name, extra):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force:
        print(f"[skip] {name}: {out_path}", flush=True); return
    print(f"[run ] {name} -> {out_path}", flush=True)
    r = subprocess.run([sys.executable, "-m", module_name, INPUT, out_path, *extra], cwd="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")
    if r.returncode != 0 or not os.path.exists(out_path):
        sys.exit(f"{name} failed rc={r.returncode}")

run_pre("lineart", LINEART, "diffsynth.utils.controlnet.video_lineart", [])
run_pre("blur", BLUR, "diffsynth.utils.controlnet.video_blur", [str(args.sigma)])

def resolve_hw():
    if args.height and args.width:
        return args.height, args.width
    src_h, src_w = VideoData(INPUT).shape()
    scale = (args.max_pixels / (src_w * src_h)) ** 0.5
    h0, w0 = int(src_h * scale) // 32 * 32, int(src_w * scale) // 32 * 32
    want = src_w / src_h
    cand = [(abs(w/h - want), -(w*h), h, w)
            for h in (h0, h0+32) for w in (w0, w0+32) if w*h <= args.max_pixels]
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
if args.total_frames > CAP:
    sys.exit("total-frames exceeds input")
segments = []
g = 0
while True:
    remaining = TOTAL_FRAMES - g
    if remaining <= NUM_FRAMES:
        segments.append((g, snap_window(remaining))); break
    segments.append((g, NUM_FRAMES)); g += STRIDE

CONTROL_FRAMES = segments[-1][0] + segments[-1][1]
DISCARDED = CONTROL_FRAMES - TOTAL_FRAMES
print(f"{len(segments)} segs · win {NUM_FRAMES} · ctx {CONTEXT_FRAMES} · stride {STRIDE}", flush=True)

SEGMENT_PROMPT_JSON = os.path.join(WORK, f"prompt_segments_w{NUM_FRAMES}_c{CONTEXT_FRAMES}g.json")

def resolve_captions():
    starts = [g0 for g0, _ in segments]
    if args.prompt is not None:
        return {g0: args.prompt for g0 in starts}
    need = set(starts)
    if os.path.exists(SEGMENT_PROMPT_JSON) and not args.force:
        blob = json.load(open(SEGMENT_PROMPT_JSON))
        have = {int(k): v for k, v in blob.get("captions", {}).items()}
        if blob.get("window") == NUM_FRAMES and need <= set(have):
            print(f"[skip] caption {len(need)} windows", flush=True)
            return {g0: have[g0] for g0 in starts}
    print(f"[run ] caption -> {SEGMENT_PROMPT_JSON}", flush=True)
    r = subprocess.run([sys.executable, "-m", "diffsynth.utils.controlnet.video_caption",
        INPUT, SEGMENT_PROMPT_JSON, "--g0", ",".join(map(str, starts)),
        "--window", str(NUM_FRAMES), "--context", str(CONTEXT_FRAMES)],
        cwd="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio")
    if r.returncode != 0:
        sys.exit("caption failed")
    have = {int(k): v for k, v in json.load(open(SEGMENT_PROMPT_JSON))["captions"].items()}
    missing = sorted(need - set(have))
    if missing:
        sys.exit(f"caption missing {missing}")
    return {g0: have[g0] for g0 in starts}

SEGMENT_CAPTIONS = resolve_captions()
# dual-ref prompt: Video1=lineart, Video2=blur; same style intent as Diffutoon-auto
REF_TMPL = (
    "subject_definitions:\n"
    "<Video 1> is the line-art / outline video that defines structure, contours, silhouettes and edge layout.\n"
    "<Video 2> is the blurred color video that defines colors, lighting, shading and motion.\n\n"
    "summary:\n"
    "[reference generation] The target video is a Japanese anime-style cartoon rendering guided by "
    "<Video 1> (structure) and <Video 2> (color and motion). {prefix}\n\n"
    "retention_analysis:\n"
    "<Video 1> (structure/outlines): fully_preserved - contours, silhouettes and spatial layout from the line-art are retained.\n"
    "<Video 2> (color/motion): fully_preserved - colors, lighting and motion trajectories from the blur track are retained.\n\n"
    "detailed_description:\n"
    "The target video is in Japanese anime cel-shaded cartoon style.\n"
    "{caption}"
)
SEGMENT_PROMPTS = [REF_TMPL.format(prefix=STYLE_PREFIX, caption=SEGMENT_CAPTIONS[g0]) for g0, _ in segments]
print(f"prompt0:\n{SEGMENT_PROMPTS[0][:400]}...\n", flush=True)

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="Ref2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
)
turbo = ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern=TURBO_LORA)
turbo.download_if_necessary()
from safetensors import safe_open
with safe_open(turbo.path if isinstance(turbo.path, str) else turbo.path[0], "pt") as fh:
    lora_alpha = float((fh.metadata() or {}).get("alpha", 0)) or None
    rank = fh.get_slice(next(k for k in fh.keys() if "lora_A" in k)).get_shape()[0]
lora_scale = lora_alpha / rank if lora_alpha else 1.0
print(f"turbo LoRA alpha={lora_alpha} rank={rank} scale={lora_scale:g}", flush=True)
pipe.load_lora(pipe.dit, turbo, alpha=lora_scale)

def load_track(path, num_frames, max_padding):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=num_frames, fps=FPS)
    short = num_frames - len(frames)
    print(f"{os.path.basename(path)}: {len(frames)} @ {FPS}fps pad={short}", flush=True)
    assert short <= max_padding, f"{path} shortfall {short} > {max_padding}"
    return list(frames) + [frames[-1]] * short

lineart_v = load_track(LINEART, CONTROL_FRAMES, DISCARDED)
blur_v = load_track(BLUR, CONTROL_FRAMES, DISCARDED)

all_frames = []
audio_chunks = []
audio_sr = pipe.audio_vae.sample_rate
print(f"pinning {PIN} latents · ctx {CONTEXT_FRAMES} · NO first-frame keyframes", flush=True)

for k, (g0, win) in enumerate(segments):
    kwargs = dict(
        prompt=SEGMENT_PROMPTS[k],
        height=HEIGHT, width=WIDTH, num_frames=win,
        num_inference_steps=STEPS,
        seed=args.seed + k,
        references=[
            {"type": "video", "video": lineart_v[g0:g0 + win]},
            {"type": "video", "video": blur_v[g0:g0 + win]},
        ],
        # explicitly no keyframes / first frame
    )
    overlap = len(all_frames) - g0 if k > 0 else 0
    if k > 0:
        assert 0 < overlap <= win
        context = all_frames[-overlap:]
        # retake: pin context; fill rest with blur track (color/motion), same idea as Diffutoon-auto
        kwargs["retake_video"] = context + blur_v[g0 + overlap:g0 + win]
        kwargs["frame_regions_to_retake"] = [(overlap, win)]
        # NO keyframes

    video, audio = pipe(**kwargs)
    if overlap:
        base = len(all_frames) - overlap
        for i in range(overlap):
            all_frames[base + i] = Image.blend(
                all_frames[base + i].convert("RGB"), video[i].convert("RGB"),
                0.5 if overlap == 1 else i / (overlap - 1))
    all_frames.extend([f.convert("RGB") for f in video[overlap:]])
    if audio is not None:
        audio_chunks.append(audio[:, int((overlap / FPS) * audio_sr):])
    print(f"  seg {k+1}/{len(segments)} [{g0},{g0+win}) total={len(all_frames)}", flush=True)

all_frames = all_frames[:TOTAL_FRAMES]
full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
if full_audio is not None:
    full_audio = full_audio[:, :int(len(all_frames) / FPS * audio_sr)]
try:
    write_video_audio(video=all_frames, audio=full_audio, output_path=OUTPUT, fps=FPS, audio_sample_rate=audio_sr)
except Exception as e:
    print(f"[warn] write_video_audio: {e}; imageio fallback", flush=True)
    import imageio.v2 as iio, numpy as np
    iio.mimwrite(OUTPUT, [np.asarray(f) for f in all_frames], fps=FPS, codec="libx264", quality=8)
print(f"WROTE {OUTPUT} n={len(all_frames)}", flush=True)
