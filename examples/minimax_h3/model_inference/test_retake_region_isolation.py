"""Isolation test for the under-moving second segment.

Three outputs over the SAME global frame span [90, 214), so motion can be compared
directly against each other and against the control track:

  A  seg0 alone, global [0, 124)          -- the reference, known good
  B  retake-by-region, global [90, 214)   -- pins A's last 34 frames, generates 90.
                                             Written in the style of the stock
                                             `MiniMax-H3-Retake.py` region example.
  C  fresh, global [90, 214), NO retake   -- the control group. Same control slices,
                                             same seed as B, but nothing pinned.

If C moves normally and B does not, retake itself is suppressing motion and the bug
is not in the control slicing (which C shares verbatim). If C is also sluggish, the
control track for that span simply has less motion and the pipeline is faithful.
"""
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

HEIGHT, WIDTH = 1568, 1568
NUM_FRAMES = 124
CONTEXT_FRAMES = 34
FPS = 24
STEPS = 4
SEED = 43
VAE_CLIP = 17
G1 = NUM_FRAMES - CONTEXT_FRAMES          # 90: where segment B starts globally

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

# Both control tracks cover global [0, 214) so every stage slices the same list.
NEED = G1 + NUM_FRAMES
def load_control(path):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=NEED, fps=FPS)
    print(f"{path.split('/')[-1]}: {len(frames)} frames @ {FPS}fps ({len(frames)/FPS:.2f}s)")
    assert len(frames) == NEED, f"exhausted: got {len(frames)}, need {NEED}"
    return frames
control_a, control_b = load_control(CONTROL_A), load_control(CONTROL_B)

def controls(g0):
    return [control_a[g0:g0 + NUM_FRAMES], control_b[g0:g0 + NUM_FRAMES]]

def save(name, video, audio):
    write_video_audio(video=video, audio=audio, output_path=name,
                      fps=FPS, audio_sample_rate=pipe.audio_vae.sample_rate)
    print(f"  -> {name} · {len(video)} frames")

# ---------------------------------------------------------- A: seg0, global [0,124)
print(f"\n=== A: fresh, global [0,{NUM_FRAMES}) ===")
video_a, audio_a = pipe(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED,
    control_video=controls(0), control_scale=[0.2, 0.8],
)
save("test_A_seg0.mp4", video_a, audio_a)

# ------------------------------- B: retake by region, global [90,214), pins A's tail
# Exactly the stock region example's call shape: a full-length retake_video plus
# explicit frame/second regions. The source video here is A's last 34 frames
# followed by the control track for the span being generated -- the stock example
# has real frames throughout because it retakes a span in the MIDDLE; we only have
# real frames on the left, so the right side is the control signal.
video_regions = [align_to_clips(CONTEXT_FRAMES, NUM_FRAMES, NUM_FRAMES)]
audio_regions = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]
print(f"\n=== B: retake by region, global [{G1},{G1+NUM_FRAMES}) ===")
print(f"  regions · frames {video_regions[0]} · seconds "
      f"({audio_regions[0][0]:.4f}, {audio_regions[0][1]:.4f})")

sr = pipe.audio_vae.sample_rate
source_video = video_a[-CONTEXT_FRAMES:] + control_b[G1 + CONTEXT_FRAMES:G1 + NUM_FRAMES]
assert len(source_video) == NUM_FRAMES, len(source_video)
source_audio = audio_a[:, -int(CONTEXT_FRAMES / FPS * sr):] if audio_a is not None else None

video_b, audio_b = pipe(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED + 1,
    control_video=controls(G1), control_scale=[0.2, 0.8],
    retake_video=source_video, frame_regions_to_retake=video_regions,
    retake_audio=source_audio, retake_audio_sample_rate=sr,
    seconds_regions_to_retake=audio_regions,
)
save("test_B_retake.mp4", video_b, audio_b)
save("test_B_new90.mp4", video_b[CONTEXT_FRAMES:],
     audio_b[:, int(CONTEXT_FRAMES / FPS * sr):] if audio_b is not None else None)

# ------------------------- C: control group, same span and seed, retake disabled
print(f"\n=== C: fresh (no retake), global [{G1},{G1+NUM_FRAMES}) ===")
video_c, audio_c = pipe(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED + 1,
    control_video=controls(G1), control_scale=[0.2, 0.8],
)
save("test_C_fresh.mp4", video_c, audio_c)
print("\nB 的新增 90 帧 (test_B_new90.mp4) 与 C 的后 90 帧覆盖同一全局区间 "
      f"[{G1+CONTEXT_FRAMES},{G1+NUM_FRAMES})，可直接比运动幅度")
