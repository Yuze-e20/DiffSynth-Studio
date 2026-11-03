"""Pin a 5-frame context -- the shortest context the VAE can actually express.

Measured frame -> latent boundaries (probe in /tmp/map2.py): each 17-frame clip
encodes to 5 latents as 1+4+4+4+4, so the first latents cover

  latent 0 <- frame 0        latent 5 <- frame 17
  latent 1 <- frames 1..4    latent 6 <- frames 18..21
  latent 2 <- frames 5..8    latent 7 <- frames 22..25
  latent 3 <- frames 9..12
  latent 4 <- frames 13..16

Pinning the first K latents therefore pins 1, 5, 9, 13, 17, 18, 22, 26, 30, 34 ...
frames. 5 is legal; 17 is not special. The stock mask computes
`floor(start/17) * 5`, which only recognises multiples of 17 and silently floors
5 to 0 -- that is why CONTEXT=5 came out byte-identical to no retake at all.

latents_to_pin below inverts the real boundary table, and the monkeypatch feeds
it a frame count instead. Everything else -- seed, control slices, LoRAs,
control_scale -- matches test_retake_short_ctx.py so the numbers are comparable.

Cases:
  5     pin 5 frames (2 latents), 119 new frames   <- what we are after
  1     pin 1 frame  (1 latent),  123 new frames   <- the absolute minimum
  5base same seed, retake off                      <- the baseline for both
"""
import math, sys, torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
import diffsynth.pipelines.minimax_h3_audio_video as mm
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

CASE = sys.argv[1]                      # "5" | "1" | "5base"
HEIGHT, WIDTH = 1568, 1568
NUM_FRAMES, FPS, STEPS, SEED, VAE_CLIP = 124, 24, 4, 43, 17
CONTEXT_FRAMES = {"5": 5, "1": 1, "5base": 5}[CASE]


def latents_to_pin(frames, clip=VAE_CLIP, per_clip=5):
    """How many leading latents are fully covered by the first `frames` frames."""
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
NEW_FRAMES = NUM_FRAMES - CONTEXT_FRAMES
NEED = NEW_FRAMES + NUM_FRAMES
print(f"case {CASE}: context {CONTEXT_FRAMES} frames -> pin {PIN} latents, "
      f"{NEW_FRAMES} new frames, control slice [{NEW_FRAMES},{NEW_FRAMES + NUM_FRAMES})")

_orig = mm.MiniMaxH3Unit_VideoRetakeEmbedder.process


def patched(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw):
    """Same as the stock unit, but the mask's start index comes from the real
    frame -> latent table rather than from floor(start / 17) * 5."""
    out = _orig(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw)
    if not out:
        return out
    mask = torch.zeros(out["input_latents_video"].shape[2], dtype=pipe.torch_dtype, device=pipe.device)
    mask[PIN:] = 1                       # 1 = regenerate; pin exactly the leading PIN latents
    print(f"  mask: pinned latents [0,{PIN}) of {mask.shape[0]}, regenerating the rest")
    return {"input_latents_video": out["input_latents_video"],
            "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *out["input_latents_video"].shape[-2:])}


if CASE != "5base":
    mm.MiniMaxH3Unit_VideoRetakeEmbedder.process = patched

SEG0 = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/examples/minimax_h3/model_inference/test_D_seg0.mp4"
TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-2000.safetensors"
CONTROL_A = "/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_lineart.mp4"
CONTROL_B = "/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_blur20.mp4"
PROMPT = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA girl is dancing."

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
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
        ModelConfig(path=TRAINED_CONTROLNET, **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
pipe.load_lora(pipe.dit, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split-480p-124_merged/step-2600.safetensors")
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))


def load_video(path, n):
    frames, audio, sr = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=n, fps=FPS)
    assert len(frames) == n, f"{path} gave {len(frames)}, need {n}"
    return frames, audio, sr


control_a, _, _ = load_video(CONTROL_A, NEED)
control_b, _, _ = load_video(CONTROL_B, NEED)
seg0, seg0_audio, _ = load_video(SEG0, NUM_FRAMES)

kwargs = dict(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED + 1, control_scale=[0.2, 0.8],
    control_video=[control_a[NEW_FRAMES:NEW_FRAMES + NUM_FRAMES],
                   control_b[NEW_FRAMES:NEW_FRAMES + NUM_FRAMES]],
)
if CASE != "5base":
    sr = pipe.audio_vae.sample_rate
    kwargs["retake_video"] = seg0[-CONTEXT_FRAMES:]
    kwargs["frame_regions_to_retake"] = [(CONTEXT_FRAMES, NUM_FRAMES)]  # patched mask ignores the rounding
    if seg0_audio is not None:
        kwargs["retake_audio"] = seg0_audio[:, -int(CONTEXT_FRAMES / FPS * sr):]
        kwargs["retake_audio_sample_rate"] = sr
        kwargs["seconds_regions_to_retake"] = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]

video, audio = pipe(**kwargs)
sr = pipe.audio_vae.sample_rate
write_video_audio(video=video, audio=audio, output_path=f"test_G{CASE}.mp4", fps=FPS, audio_sample_rate=sr)
write_video_audio(video=seg0 + video[CONTEXT_FRAMES:], audio=None,
                  output_path=f"test_G{CASE}_joined.mp4", fps=FPS, audio_sample_rate=sr)
print("done")
