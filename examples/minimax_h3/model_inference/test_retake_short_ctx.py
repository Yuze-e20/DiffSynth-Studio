"""Short pinned contexts: is a smaller context cheaper, and what do 5 and 0 actually do?

Measured so far, retake's sharpness cost against a same-seed no-retake baseline:
  CONTEXT=34 (90 new frames)  ratio 0.51
  CONTEXT=102 (22 new frames) ratio 0.27
So the cost grows as the new span shrinks. This run goes the other way.

The mask is mask[floor(ctx/17)*5 : ceil(124/17)*5] = 1, and 1 means regenerate.
  CONTEXT=17 -> mask[5:40], 5 latents pinned, 107 new frames. The most aggressive
                tier that actually pins anything.
  CONTEXT=5  -> floor(5/17) = 0, so mask[0:40] is entirely 1. Nothing is pinned.
  CONTEXT=0  -> same all-1 mask, and `video[-0:]` is the whole list in Python, so
                the full 124 frames are VAE-encoded for nothing.

With an all-1 mask both of the places the mask is used are identity ops:
  latents * mask + input * (1 - mask)   and   timesteps[mask == 0] = 1.0
so 5 and 0 should match a plain no-retake call numerically. F17_BASE tests that
claim too: it shares its seed with F17, so any difference is retake's alone.
"""
import sys, torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

CASE = sys.argv[1]            # "17" | "17base" | "5" | "0"
HEIGHT, WIDTH = 1568, 1568
NUM_FRAMES, FPS, STEPS, SEED, VAE_CLIP = 124, 24, 4, 43, 17
CONTEXT_FRAMES = {"17": 17, "17base": 17, "5": 5, "0": 0}[CASE]
G1 = NUM_FRAMES - (CONTEXT_FRAMES // VAE_CLIP) * VAE_CLIP   # frames the new span advances
NEED = G1 + NUM_FRAMES

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


def align_to_clips(start, end, total_frames, clip_frames=VAE_CLIP):
    """Widen the half-open frame range [start, end) to whole clips. Frames count from 0."""
    first_clip, last_clip = start // clip_frames, (end - 1) // clip_frames
    return first_clip * clip_frames, min((last_clip + 1) * clip_frames, total_frames)


def load_video(path, n):
    frames, audio, sr = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=n, fps=FPS)
    assert len(frames) == n, f"{path} gave {len(frames)}, need {n}"
    return frames, audio, sr


control_a, _, _ = load_video(CONTROL_A, NEED)
control_b, _, _ = load_video(CONTROL_B, NEED)
# seg0 is reused from the previous run instead of regenerating it; it comes back
# through H.264, a small fidelity loss that is identical across all four cases.
seg0, seg0_audio, seg0_sr = load_video(SEG0, NUM_FRAMES)

kwargs = dict(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED + 1, control_scale=[0.2, 0.8],
    control_video=[control_a[G1:G1 + NUM_FRAMES], control_b[G1:G1 + NUM_FRAMES]],
)
print(f"case {CASE}: CONTEXT={CONTEXT_FRAMES}, new span advances {G1} frames, "
      f"control slice [{G1},{G1 + NUM_FRAMES})")

if CASE != "17base":
    sr = pipe.audio_vae.sample_rate
    # Deliberately written the naive way: at CONTEXT_FRAMES=0 these slices are
    # `[-0:]`, which Python reads as the whole list.
    context = seg0[-CONTEXT_FRAMES:]
    kwargs["retake_video"] = context
    kwargs["frame_regions_to_retake"] = [align_to_clips(CONTEXT_FRAMES, NUM_FRAMES, NUM_FRAMES)]
    if seg0_audio is not None:
        kwargs["retake_audio"] = seg0_audio[:, -int(CONTEXT_FRAMES / FPS * sr):]
        kwargs["retake_audio_sample_rate"] = sr
        kwargs["seconds_regions_to_retake"] = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]
    print(f"  retake_video {len(context)} frames · regions {kwargs['frame_regions_to_retake']}"
          + ("  <- 0 asked for, whole list given" if CONTEXT_FRAMES == 0 else ""))

video, audio = pipe(**kwargs)
sr = pipe.audio_vae.sample_rate
write_video_audio(video=video, audio=audio, output_path=f"test_F{CASE}.mp4",
                  fps=FPS, audio_sample_rate=sr)
pinned = (CONTEXT_FRAMES // VAE_CLIP) * VAE_CLIP
if pinned:
    write_video_audio(video=seg0 + video[pinned:], audio=None,
                      output_path=f"test_F{CASE}_joined.mp4", fps=FPS, audio_sample_rate=sr)
print("done")
