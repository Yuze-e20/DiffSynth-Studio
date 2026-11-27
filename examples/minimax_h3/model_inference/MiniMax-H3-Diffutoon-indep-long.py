"""Full-length independent windows (no ctx pin, no FF). Hard-cut abutting 328 windows."""
from __future__ import annotations
# Reuse auto by monkeypatching after parse — simplest: copy key loop from auto with CONTEXT=0.
# Instead invoke via environment and importlib is messy. Standalone thin wrapper:
import argparse, json, os, sys
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/models")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Patch sys.argv to force context then import auto? Auto validates ctx in PIN_TABLE.
# Implement minimal: exec auto source with CONTEXT_FRAMES override after validation skip.

import runpy
# We'll call the standard auto with a fake context of 1 but custom script is cleaner.

import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, VAE_CLIP_FRAMES, latents_fully_covered)
from diffsynth.utils.data.audio_video import read_video_audio, write_video_audio
from diffsynth.utils.controlnet import video_blur, video_lineart, video_caption
import subprocess

NUM_FRAMES = 328
FPS = 24
STEPS = 4
SEED = 43
MAX_PIXELS = 1032192
VRAM_HEADROOM = 24
STYLE_PREFIX = (
    "Generate a Japanese anime-style cartoon video with clean line work and flat "
    "cel-shaded colors, following the outlines of the line-art video and the colors, "
    "lighting and motion of the blurred video."
)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--control-scale", type=float, nargs=2, default=[0.5, 0.5])
    p.add_argument("--structure-controlnet", required=True)
    p.add_argument("--controlnet", required=True)
    p.add_argument("--control-layers", default="0,5,10,15,20,25,30,35,40")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()

def main():
    args = parse()
    src = os.path.abspath(args.input)
    work = os.path.abspath(args.work_dir)
    out = os.path.abspath(args.output)
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    lineart = os.path.join(work, "lineart.mp4")
    blur = os.path.join(work, "blur20.mp4")
    env = dict(os.environ, PYTHONPATH="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio:" + os.environ.get("PYTHONPATH",""))
    cwd = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio"
    if not os.path.exists(lineart):
        subprocess.check_call([sys.executable, "-m", "diffsynth.utils.controlnet.video_lineart", src, lineart], cwd=cwd, env=env)
    if not os.path.exists(blur):
        subprocess.check_call([sys.executable, "-m", "diffsynth.utils.controlnet.video_blur", src, blur, "20"], cwd=cwd, env=env)

    # total frames at 24fps from source duration via reading controls
    # first get duration
    import subprocess as sp
    r = sp.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0", src], capture_output=True, text=True)
    total = max(1, int(float(r.stdout.strip()) * FPS))
    # independent non-overlapping windows of 328, last snapped
    def snap(n):
        if n <= 5: return 5
        return VAE_CLIP_FRAMES * max(1, -(-(n - 5) // VAE_CLIP_FRAMES)) + 5
    segments = []
    g = 0
    while g < total:
        remaining = total - g
        if remaining >= NUM_FRAMES:
            win = NUM_FRAMES
        else:
            win = snap(remaining)
        segments.append((g, win))
        g += win  # no overlap
    print(f"indep segments={segments} total={total}", flush=True)

    # captions: reuse w328_c1g if has keys, else generate
    cap_path = os.path.join(work, "prompt_segments_w328_c1g.json")
    starts = [g0 for g0,_ in segments]
    if os.path.exists(cap_path):
        blob = json.load(open(cap_path))
        have = {int(k): v for k, v in blob.get("captions", {}).items()}
    else:
        have = {}
    missing = [g for g in starts if g not in have]
    if missing:
        g0s = ",".join(str(g) for g in starts)
        subprocess.check_call([
            sys.executable, "-m", "diffsynth.utils.controlnet.video_caption",
            src, cap_path, "--g0", g0s, "--window", str(NUM_FRAMES), "--context", "0",
        ], cwd=cwd, env={**env, "DIFFSYNTH_MODEL_BASE_PATH": "/root/models", "DIFFSYNTH_SKIP_DOWNLOAD": "True"})
        blob = json.load(open(cap_path))
        have = {int(k): v for k, v in blob.get("captions", {}).items()}
    prompts = [f"{STYLE_PREFIX}\ndetailed_description:\n{have[g0]}" for g0,_ in segments]

    # resolution from control
    import imageio.v2 as iio
    fr0 = Image.fromarray(iio.get_reader(lineart).get_data(0))
    w, h = fr0.size
    s = (MAX_PIXELS / float(w * h)) ** 0.5
    if s < 1:
        w, h = int(w * s), int(h * s)
    width, height = max(32, (w // 32) * 32), max(32, (h // 32) * 32)

    need = segments[-1][0] + segments[-1][1]
    ca, _, _ = read_video_audio(lineart, height=height, width=width, num_frames=need, fps=FPS)
    cb, _, _ = read_video_audio(blur, height=height, width=width, num_frames=need, fps=FPS)
    ca, cb = list(ca), list(cb)
    if len(ca) < need:
        ca += [ca[-1]] * (need - len(ca))
    if len(cb) < need:
        cb += [cb[-1]] * (need - len(cb))

    vram = dict(offload_dtype=torch.bfloat16, offload_device="cpu",
                onload_dtype=torch.bfloat16, onload_device="cpu",
                preparing_dtype=torch.bfloat16, preparing_device="cuda",
                computation_dtype=torch.bfloat16, computation_device="cuda")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram),
            ModelConfig(path=args.structure_controlnet, **vram),
            ModelConfig(path=args.controlnet, **vram),
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
    )
    from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3MultiControlNet
    branches = list(pipe.controlnet) if isinstance(pipe.controlnet, MiniMaxH3MultiControlNet) else [pipe.controlnet]
    for br in branches:
        try:
            br.set_control_layers(args.control_layers)
        except Exception:
            br.set_control_layers(args.control_layers, allow_resize=True)
    pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo",
        origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

    all_frames = []
    for k, (g0, win) in enumerate(segments):
        print(f"=== seg {k} g0={g0} win={win} ===", flush=True)
        video, _ = pipe(
            prompt=prompts[k], height=height, width=width, num_frames=win,
            num_inference_steps=STEPS, seed=args.seed + k,
            control_video=[ca[g0:g0+win], cb[g0:g0+win]],
            control_scale=list(args.control_scale),
        )
        all_frames.extend([f.convert("RGB") for f in video])
    all_frames = all_frames[:total]
    try:
        write_video_audio(video=all_frames, audio=None, output_path=out, fps=FPS)
    except Exception:
        import imageio.v2 as iio
        iio.mimwrite(out, [np.asarray(f) for f in all_frames], fps=FPS, codec="libx264", quality=8)
    print(f"WROTE {out} n={len(all_frames)}", flush=True)

if __name__ == "__main__":
    import numpy as np
    main()
