import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from modelscope import dataset_snapshot_download

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
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
pipe.load_lora(pipe.dit, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split/step-3000.safetensors")

pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

# dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Retake/*")

# `input_video` re-noises the source clip up to `denoising_strength` and denoises from there, so the
# result keeps the source layout and motion while the prompt steers appearance and sound. Lower
# values stay closer to the source; 1.0 discards it entirely and is plain text-to-video.
#
# MIND THE SCALE. `denoising_strength` sets the position on the schedule *before* `flow_shift`, and
# at the default shift of 12 that curve is steep, so useful values are far smaller than the 0.5-0.8
# usual elsewhere. How much noise actually lands on the video latents:
#
#   denoising_strength | 1.0  0.5  0.3  0.2  0.15  0.1  0.05
#   video noise level  | 1.00 0.92 0.84 0.75 0.68  0.57 0.39
#
# Start around 0.1-0.2. Above ~0.3 the source is almost gone; the step count is unchanged either
# way, so a low strength costs the same as a full run.
#
# Frames are resized onto the height/width canvas and trimmed to `num_frames`; a short clip is
# padded with its last frame. `num_frames` snaps up to the nearest 17n+5.

# Video -> Video + Audio
# The audio branch has nothing to start from here, so it runs from pure noise on a schedule that
# has already been shortened for the video. The lower `denoising_strength` goes the less room the
# audio has to converge, so pass `input_audio` (below) whenever the source has a soundtrack.
# prompt = "A cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man in a vibrant pink suit standing in a lush green field with sheep.\n[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, stands in a grassy field holding a small black lamb. Several white lambs graze nearby on the rolling green hills. The man looks upward with a contemplative expression, gently cradling the black lamb against his chest. The lighting is soft and directional, casting long shadows and highlighting the texture of the grass and the man's clothing."
# source_video, _, _ = read_video_audio("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video.mp4", height=704, width=1344, num_frames=124, fps=24)
# video, audio = pipe(
#     prompt=prompt,
#     height=704, width=1344, num_frames=124, num_inference_steps=4, seed=0,
#     input_video=source_video, denoising_strength=0.1,
# )
# write_video_audio(
#     video=video, audio=audio,
#     output_path="v2va-1.mp4", fps=24, audio_sample_rate=32000,
# )

# Video + Audio -> Video + Audio
# Both branches start from the source, re-noised at the same `denoising_strength`. The waveform is
# [C, L] at `input_audio_sample_rate` and is trimmed or padded to the clip length on its own.
prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description:\nA cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a young man in a vibrant pink suit standing in a lush green field with sheep.\n[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, stands in a grassy field holding a small black lamb. Several white lambs graze nearby on the rolling green hills. The man looks upward with a contemplative expression, gently cradling the black lamb against his chest. The lighting is soft and directional, casting long shadows and highlighting the texture of the grass and the man's clothing."
# prompt = "A cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man in a vibrant pink suit standing in a lush green field with sheep.\n[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, stands in a grassy field holding a small black lamb. Several white lambs graze nearby on the rolling green hills. The man looks upward with a contemplative expression, gently cradling the black lamb against his chest. The lighting is soft and directional, casting long shadows and highlighting the texture of the grass and the man's clothing."
source_video, source_audio, sample_rate = read_video_audio("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video.mp4", height=704, width=1344, num_frames=124, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
video, audio = pipe(
    prompt=prompt,
    height=704, width=1344, num_frames=124, num_inference_steps=3, seed=1, flow_shift=6.0,
    input_video=source_video,
    input_audio=source_audio, input_audio_sample_rate=sample_rate,
    denoising_strength=0.8,
)
write_video_audio(
    video=video, audio=audio,
    output_path="v2va_with_audio-step3-08-compose.mp4", fps=24, audio_sample_rate=32000,
)
