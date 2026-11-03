"""Two ControlNets driving one MiniMax-H3 generation.

The DiT injects ControlNet output as a plain additive residual per block, so
several ControlNets compose by summing their per-layer hints. Loading two
`minimax_h3_controlnet` checkpoints makes the pipeline wrap them in a
`MiniMaxH3MultiControlNet`, and `control_video` / `control_scale` then take one
entry per branch.

Scales matter: each branch was trained to act alone, so its hint magnitude is
calibrated as the only residual on the backbone. Summing two branches at 1.0
roughly doubles the residual and tends to over-drive the DiT (washed-out or
over-saturated frames). Start near 1/N per branch and tune from there.

Both control videos must land on the same latent grid as the target -- same
height, width and frame count -- because the ControlNet asserts one control row
per video row.
"""
import torch
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Path to the checkpoint trained by
# examples/minimax_h3/model_training/full/MiniMax-H3-Fun-Controlnet-Union.sh
# TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-124-sigma20-newprompt/step-1700.safetensors"
TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors"
# TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-124-sigma20/step-1800.safetensors"

# TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma15/step-3000.safetensors"

HEIGHT, WIDTH, NUM_FRAMES = 768, 1344, 141
# HEIGHT, WIDTH, NUM_FRAMES = 896, 1568, 124
FPS = 24          # control decks are resampled to this; the model always generates at 24fps

def load_control_video(path):
    # `read_video_audio` resamples to FPS by nearest-source-frame lookup
    # (index = round(k * source_fps / FPS)), so a 30fps deck keeps its real motion
    # speed instead of being stretched by 30/24. `VideoData` does NO fps conversion --
    # it hands back raw decoded frame i, which is why it ran 1.25x slow here.
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, fps=FPS)
    print(f"{path.split('/')[-1]}: {len(frames)} frames @ {FPS}fps, {frames[0].size}")
    assert len(frames) == NUM_FRAMES, f"deck exhausted: got {len(frames)}, need {NUM_FRAMES}"
    return frames


# control_video_union = load_control_video("/mnt/nas3/sunyuzework/Diffutoon-2/data/test_Pixabay100/103/103_auto/lineart.mp4")
# control_video_trained = load_control_video("/mnt/nas3/sunyuzework/Diffutoon-2/data/test_Pixabay100/103/103_auto/blur30.mp4")
# control_video_union = load_control_video("/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_lineart.mp4")
# control_video_trained = load_control_video("/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_blur15.mp4")
control_video_union = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4")
control_video_trained = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_gaussian20.mp4")

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        # Branch 0: the public union ControlNet.
        # ModelConfig(path="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/lineart-720p-39/step-1700.safetensors", **vram_config),
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
        # Branch 1: your own checkpoint. Branch order here is the order used by
        # `control_video` and `control_scale` below.
        ModelConfig(path=TRAINED_CONTROLNET, **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
print(f"ControlNet branches loaded: {len(pipe.controlnet)}")


# pipe.load_lora(pipe.dit, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split-480p-124_merged/step-2600.safetensors")
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))



# One control video per branch. Point these at your own inputs -- e.g. a lineart
# clip for the union ControlNet and the blurred clip for a checkpoint trained on
# the super-resolution task.
# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA warm, sunlit interior scene in a traditional Japanese home, featuring soft natural lighting and a peaceful, nostalgic atmosphere. [Shot 1] A young girl with short brown hair sits on a tatami floor, facing away from the camera. She wears an orange sleeveless top over a white shirt and a green skirt. Her legs are extended, and she appears to be resting or looking out a sliding glass door. Beside her, a small orange cat sleeps on a folded cloth. A wicker basket, a standing fan, and a wooden cabinet filled with books and trinkets occupy the background. The room is tidy and bathed in gentle daylight."

# The leading fields mirror the schema the ControlNet checkpoint was trained with
# (subject_definitions / summary / retention_analysis). That run was the
# super-resolution task, so the field *contents* here are rewritten for 3D-render-
# to-2D while the field *names* stay put. Both control videos are named explicitly
# so each branch's role is stated rather than implied: branch 0 is the public union
# ControlNet fed <Video Line-art>, branch 1 the self-trained checkpoint fed
# <Video Blur>. Everything from `detailed_description` onward is the per-clip
# caption and is left exactly as it was.
# SUBJECT_DEFINITIONS = (
#     "<Video Line-art> is the reference video providing the line-art contours, "
#     "character and object outlines, structural edges\n"
#     "<Video Blur> is the reference video providing the color palette, regional "
#     "color assignment, lighting direction, character motion, and temporal structure."
# )

# SUMMARY = (
#     "[3D-render-to-2D re-rendering] Please generate the target video, which is a high-quality Japanese "
#     "anime re-rendering of a live-action source, generated with reference to both "
#     "<Video Line-art> and <Video Blur>, featuring crisp clean linework, flat "
#     "cel-shaded color regions with sharp shadow boundaries, vivid saturated colors, "
# )

# RETENTION_ANALYSIS = (
#     "Preserve the outlines and edge placement from <Video Line-art>, and redraw them "
#     "as clean hand-drawn anime linework.\n"
#     "Preserve the color palette, lighting, camera motion, character movements and "
#     "pacing from <Video Blur>, and replace photographic texture and gradients with "
#     "flat cel-shaded regions."
# )

DETAILED_DESCRIPTION = (
    "A cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man "
    "in a vibrant pink suit standing in a lush green field with sheep.\n"
    "[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, "
    "stands in a grassy field holding a small black lamb. Several white lambs graze "
    "nearby on the rolling green hills. The man looks upward with a contemplative "
    "expression, gently cradling the black lamb against his chest. The lighting is "
    "soft and directional, casting long shadows and highlighting the texture of the "
    "grass and the man's clothing."
)
# DETAILED_DESCRIPTION = (
#     "A bright, sunlit high-rise interior with floor-to-ceiling windows overlooking a city, creating a cheerful and airy atmosphere.\n[Shot 1] A young woman with long black hair dances energetically in a white sailor-style dress with blue trim. She performs various moves, including arm waves, leg kicks, and jumps, while smiling at the camera. The background remains a consistent view of a city skyline through large glass windows, with sunlight reflecting off the polished floor."
# )

# DETAILED_DESCRIPTION = (
#     "A close-up indoor scene with soft, diffused lighting focuses on a young woman's hands holding a smartphone, creating an intimate and casual atmosphere.\n[Shot 1] A young woman with long blonde hair, wearing a black short-sleeved top, is seen from the chest up. The background is a blurred, neutral-colored wall. The person holds a white smartphone with both hands, the back of the device facing the camera. Their fingernails are painted a bright red. The hands remain relatively steady, with subtle movements of the fingers suggesting interaction with the screen, such as typing or scrolling."
# )
prefix = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video and the colors, lighting and motion of the blurred video."

prompt = (
    f"{prefix}\n"
    f"detailed_description:\n{DETAILED_DESCRIPTION}"
)
# prompt = (
#     f"subject_definitions:\n{SUBJECT_DEFINITIONS}\n\n"
#     f"summary:\n{SUMMARY}\n\n"
#     f"retention_analysis:\n{RETENTION_ANALYSIS}\n\n"
#     f"detailed_description:\n{DETAILED_DESCRIPTION}"
# )
print(prompt)

# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs."
# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA girl is dancing."

video, audio = pipe(
    prompt=prompt,
    height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=4, seed=43,
    control_video=[control_video_union, control_video_trained],
    control_scale=[0.8, 0.6],
)
write_video_audio(
    video=video, audio=audio, output_path="mandogblur20_86_1600.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)