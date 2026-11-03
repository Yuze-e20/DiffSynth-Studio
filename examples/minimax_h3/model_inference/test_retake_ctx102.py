"""Does a much larger pinned context reduce the retake sharpness penalty?

Previous measurement at CONTEXT=34 (90 new frames): retake cost 49% of sharpness
against a same-seed, same-control no-retake baseline.

Here CONTEXT=102, so only 22 frames are generated. Two readings are possible:
  - the penalty comes from generating a long span away from the anchor, so a short
    span should cost less; or
  - it comes from the clean/pure-noise juxtaposition itself, which is present at any
    span length, so the penalty stays.

D  retake, CONTEXT=102, global [102, 226) -- 22 new frames, global [204, 226)
E  fresh,  same span and seed, no retake  -- the baseline those 22 frames need

107 is NOT usable: 107 % 17 == 5, so the mask floors it to 102 and the extra 5
frames of context are silently discarded.
"""
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

HEIGHT, WIDTH = 1568, 1568
NUM_FRAMES = 124
CONTEXT_FRAMES = 102          # must be a multiple of 17; 107 would collapse to this
FPS, STEPS, SEED, VAE_CLIP = 24, 4, 43, 17
assert CONTEXT_FRAMES % VAE_CLIP == 0
G1 = NUM_FRAMES - CONTEXT_FRAMES          # 22
NEED = G1 + NUM_FRAMES                    # 146

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


def load_control(path):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=NEED, fps=FPS)
    assert len(frames) == NEED, f"{path} gave {len(frames)}, need {NEED}"
    return frames


control_a, control_b = load_control(CONTROL_A), load_control(CONTROL_B)
controls = lambda g0: [control_a[g0:g0 + NUM_FRAMES], control_b[g0:g0 + NUM_FRAMES]]
common = dict(prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
              num_inference_steps=STEPS, control_scale=[0.2, 0.8])

# A: the first 124 frames, no retake
video_a, audio_a = pipe(**common, seed=SEED, control_video=controls(0))
sr = pipe.audio_vae.sample_rate
write_video_audio(video=video_a, audio=audio_a, output_path="test_D_seg0.mp4",
                  fps=FPS, audio_sample_rate=sr)

# D: retake with a 102-frame pinned context -> 22 new frames, global [204, 226)
video_regions = [align_to_clips(CONTEXT_FRAMES, NUM_FRAMES, NUM_FRAMES)]
audio_regions = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]
print(f"regions · frames {video_regions[0]} · seconds "
      f"({audio_regions[0][0]:.4f}, {audio_regions[0][1]:.4f}) · new {G1}")
video_d, audio_d = pipe(
    **common, seed=SEED + 1, control_video=controls(G1),
    retake_video=video_a[-CONTEXT_FRAMES:], frame_regions_to_retake=video_regions,
    retake_audio=audio_a[:, -int(CONTEXT_FRAMES / FPS * sr):] if audio_a is not None else None,
    retake_audio_sample_rate=sr, seconds_regions_to_retake=audio_regions,
)
write_video_audio(video=video_d, audio=audio_d, output_path="test_D_retake.mp4",
                  fps=FPS, audio_sample_rate=sr)
write_video_audio(video=video_d[CONTEXT_FRAMES:], audio=None,
                  output_path=f"test_D_new{G1}.mp4", fps=FPS, audio_sample_rate=sr)
write_video_audio(video=video_a + video_d[CONTEXT_FRAMES:], audio=None,
                  output_path="test_D_joined.mp4", fps=FPS, audio_sample_rate=sr)

# E: control group -- same span, same seed, retake disabled
video_e, audio_e = pipe(**common, seed=SEED + 1, control_video=controls(G1))
write_video_audio(video=video_e, audio=audio_e, output_path="test_E_fresh.mp4",
                  fps=FPS, audio_sample_rate=sr)
print("done")
