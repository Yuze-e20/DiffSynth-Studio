"""Two-window continuation ablation at NUM_FRAMES=124. Saves w1.mp4 + w2.mp4 (each 124 frames)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/models")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, latents_fully_covered)
from diffsynth.utils.data.audio_video import read_video_audio

NUM_FRAMES = 124
FPS = 24
STEPS = 4
SEED = 43
BLUR_SIGMA = 20
CONTROL_SCALE = [0.5, 0.5]
MAX_PIXELS = 1032192
VRAM_HEADROOM = 24
STYLE_PREFIX = (
    "Generate a Japanese anime-style cartoon video with clean line work and flat "
    "cel-shaded colors, following the outlines of the line-art video and the colors, "
    "lighting and motion of the blurred video."
)
CKPT_5B_BLUR = "models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors"
LAYERS_5B = "0,10,20,30,40"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--mode", choices=["pin", "indep_cut"], required=True)
    p.add_argument("--context-frames", type=int, default=22)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def ensure_controls(work, src, force):
    os.makedirs(work, exist_ok=True)
    lineart = os.path.join(work, "lineart.mp4")
    blur = os.path.join(work, "blur20.mp4")
    env = dict(os.environ, PYTHONPATH="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio:" + os.environ.get("PYTHONPATH", ""))
    cwd = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio"
    if not os.path.exists(lineart) or force:
        subprocess.check_call([sys.executable, "-m", "diffsynth.utils.controlnet.video_lineart", src, lineart], cwd=cwd, env=env)
    if not os.path.exists(blur) or force:
        subprocess.check_call([sys.executable, "-m", "diffsynth.utils.controlnet.video_blur", src, blur, str(BLUR_SIGMA)], cwd=cwd, env=env)
    return lineart, blur


def ensure_captions(work, src, g0_w2, ctx, force):
    """Load precomputed captions (prompt_segments_w124_seam_all.json) for g0=0 and g0_w2."""
    path = os.path.join(work, f"prompt_segments_w{NUM_FRAMES}_seam_all.json")
    if not os.path.exists(path):
        sys.exit(f"missing captions {path}; pre-generate with g0=0,34,68,102,124")
    caps = {str(k): v for k, v in json.load(open(path))["captions"].items()}
    if "0" not in caps or str(g0_w2) not in caps:
        sys.exit(f"need captions 0 and {g0_w2} in {path}, have {sorted(caps)}")
    print(f"[caption] using {path} keys 0,{g0_w2}", flush=True)
    return caps["0"], caps[str(g0_w2)]


def derive_hw(frame):
    w, h = frame.size
    s = (MAX_PIXELS / float(w * h)) ** 0.5
    if s < 1:
        w, h = int(w * s), int(h * s)
    return max(32, (w // 32) * 32), max(32, (h // 32) * 32)


def load_control(path, num_frames, height, width):
    frames, _, _ = read_video_audio(path, height=height, width=width, num_frames=num_frames, fps=FPS)
    frames = list(frames)
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    return frames[:num_frames]


def write_mp4(path, frames):
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    iio.mimwrite(path, [np.asarray(f.convert("RGB")) for f in frames], fps=FPS, codec="libx264", quality=8)
    print(f"WROTE {path} n={len(frames)}", flush=True)


def main():
    args = parse_args()
    src = os.path.abspath(args.input)
    work = os.path.abspath(args.work_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "pin":
        ctx = args.context_frames
        table = [f for f in range(1, NUM_FRAMES) if latents_fully_covered(f) != latents_fully_covered(f - 1)]
        if ctx not in table:
            sys.exit(f"ctx={ctx} not boundary; {table}")
        g0_w2 = NUM_FRAMES - ctx
        overlap = ctx
        use_retake = True
        use_ff = True
        do_blend = True
    else:
        ctx = 0
        g0_w2 = NUM_FRAMES
        overlap = 0
        use_retake = False
        use_ff = False
        do_blend = False

    # skip if both window mp4s exist
    w1_path = os.path.join(out_dir, "w1.mp4")
    w2_path = os.path.join(out_dir, "w2.mp4")
    if os.path.exists(w1_path) and os.path.exists(w2_path) and not args.force:
        print(f"[skip done] {out_dir}", flush=True)
        return

    total_need = g0_w2 + NUM_FRAMES
    print(f"mode={args.mode} ctx={ctx} g0_w2={g0_w2} overlap={overlap} need={total_need}", flush=True)

    lineart, blur = ensure_controls(work, src, args.force)
    cap0, cap1 = ensure_captions(work, src, g0_w2, ctx, args.force)
    prompts = [
        f"{STYLE_PREFIX}\ndetailed_description:\n{cap0}",
        f"{STYLE_PREFIX}\ndetailed_description:\n{cap1}",
    ]

    import imageio.v2 as iio
    r = iio.get_reader(lineart)
    fr0 = Image.fromarray(r.get_data(0)); r.close()
    width, height = derive_hw(fr0)
    print(f"res {width}x{height}", flush=True)

    control_a = load_control(lineart, total_need, height, width)
    control_b = load_control(blur, total_need, height, width)

    vram_config = {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    branch0 = ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union",
                          origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config)
    branch1 = ModelConfig(path=CKPT_5B_BLUR, **vram_config)

    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
            branch0, branch1,
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - VRAM_HEADROOM,
    )
    from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3MultiControlNet
    branches = list(pipe.controlnet) if isinstance(pipe.controlnet, MiniMaxH3MultiControlNet) else [pipe.controlnet]
    for i, branch in enumerate(branches):
        try:
            branch.set_control_layers(LAYERS_5B)
        except Exception:
            branch.set_control_layers(LAYERS_5B, allow_resize=True)
        print(f"[layers] b{i} n={len(branch.blocks)}", flush=True)
    pipe.load_lora(pipe.dit, ModelConfig(
        model_id="lightx2v/Minimax-h3-Turbo",
        origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

    print("=== window 1 ===", flush=True)
    video1, _ = pipe(
        prompt=prompts[0], height=height, width=width, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS, seed=args.seed,
        control_video=[control_a[0:NUM_FRAMES], control_b[0:NUM_FRAMES]],
        control_scale=list(CONTROL_SCALE),
    )
    w1 = [f.convert("RGB") for f in video1]
    write_mp4(w1_path, w1)

    print("=== window 2 ===", flush=True)
    kwargs = dict(
        prompt=prompts[1], height=height, width=width, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS, seed=args.seed + 1,
        control_video=[control_a[g0_w2:g0_w2 + NUM_FRAMES], control_b[g0_w2:g0_w2 + NUM_FRAMES]],
        control_scale=list(CONTROL_SCALE),
    )
    if use_retake:
        context = w1[-overlap:]
        kwargs["retake_video"] = context + control_b[g0_w2 + overlap:g0_w2 + NUM_FRAMES]
        kwargs["frame_regions_to_retake"] = [(overlap, NUM_FRAMES)]
        if use_ff:
            kwargs["keyframes"] = [context[0]]
            kwargs["keyframe_indices"] = [0]
    video2, _ = pipe(**kwargs)
    w2 = [f.convert("RGB") for f in video2]
    write_mp4(w2_path, w2)

    # optional stitched preview
    all_frames = list(w1)
    if overlap and do_blend:
        base = len(all_frames) - overlap
        for i in range(overlap):
            wt = 0.5 if overlap == 1 else i / (overlap - 1)
            all_frames[base + i] = Image.blend(all_frames[base + i], w2[i], wt)
        all_frames.extend(w2[overlap:])
    else:
        if g0_w2 < len(all_frames):
            all_frames = all_frames[:g0_w2]
        all_frames.extend(w2)
    write_mp4(os.path.join(out_dir, "stitched.mp4"), all_frames)

    meta = dict(mode=args.mode, context=ctx, g0_w2=g0_w2, overlap=overlap,
                num_frames=NUM_FRAMES, seed=args.seed, width=width, height=height)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"DONE {out_dir}", flush=True)


if __name__ == "__main__":
    main()
