import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
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

dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Retake/*")

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
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a handheld medium shot follows two swordsmen exchanging blows on a stone courtyard at dusk. Blades meet edge-on, grind along each other and separate; the fighters break apart with boots scraping over grit, then close again and lock their swords overhead while their arms shake against the pressure.\n\noverall_soundscape: Steel rings sharply on steel at each contact, with a long scraping rasp where the edges grind together. Boots scrape over grit, leather creaks, and both fighters breathe hard with short grunts of effort. A low evening wind moves through the courtyard.\n\nnon_diegetic_music: A distorted electric guitar riff at a fast tempo over a double-kick drum pattern and a sustained synth bass drone."
source_video, _, _ = read_video_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_video.mp4", height=480, width=832, num_frames=124, fps=24)
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=0,
    input_video=source_video, denoising_strength=0.2,
)
write_video_audio(
    video=video, audio=audio,
    output_path="v2va.mp4", fps=24, audio_sample_rate=32000,
)

# Video + Audio -> Video + Audio
# Both branches start from the source, re-noised at the same `denoising_strength`. The waveform is
# [C, L] at `input_audio_sample_rate` and is trimmed or padded to the clip length on its own.
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a locked-off wide shot looks across an arid desert valley from a high vantage point, a long rocky ridge running low across the frame under a cloudless deep blue sky. Harsh midday sun bleaches the pale rock. Heat shimmer rises off the sunlit stone, thin veils of dust stream off the crest, and dry scrub thrashes in the wind. A single grey fighter jet crosses the sky from the left edge, passes above the ridge in level flight at very high speed, and exits past the right edge leaving no contrail.\n\noverall_soundscape: A strong dry wind drives steadily across the valley for the whole take, with a deep roar rolling along the ridge and a dry clatter from the thrashing scrub. A distant jet whine builds into a hard roar, drops sharply in pitch the instant it passes overhead, then stretches out into a long descending rumble that leaves only the wind.\n\nnon_diegetic_music: A quiet sustained synthesizer pad holds one low chord at a slow, even level from start to finish, far beneath the wind."
source_video, source_audio, sample_rate = read_video_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_video1.mp4", height=480, width=832, num_frames=175, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=175, num_inference_steps=50, seed=1,
    input_video=source_video,
    input_audio=source_audio, input_audio_sample_rate=sample_rate,
    denoising_strength=0.15,
)
write_video_audio(
    video=video, audio=audio,
    output_path="v2va_with_audio.mp4", fps=24, audio_sample_rate=32000,
)
