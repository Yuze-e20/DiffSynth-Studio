import torch
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.data import VideoData
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
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

dataset_base_path = "data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA"
height, width, num_frames = 704, 1344, 124

# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description:\nA warm, sunlit interior scene in a traditional Japanese home, featuring soft natural lighting and a peaceful, nostalgic atmosphere. [Shot 1] A young girl with short brown hair sits on a tatami floor, facing away from the camera. She wears an orange sleeveless top over a white shirt and a green skirt. Her legs are extended, and she appears to be resting or looking out a sliding glass door. Beside her, a small orange cat sleeps on a folded cloth. A wicker basket, a standing fan, and a wooden cabinet filled with books and trinkets occupy the background. The room is tidy and bathed in gentle daylight."
prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description:\nA real giant T-Rex riding a bicycle by itself, using its own legs to pedal furiously, tiny arms gripping handlebars, massive volcano erupting behind with lava and ash explosion"
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))
pipe.load_lora(pipe.dit, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split/step-3000.safetensors")
video, audio = pipe(
    prompt=prompt,
    height=height, width=width, num_frames=num_frames, flow_shift=6.0,
    num_inference_steps=4, seed=0,
)
write_video_audio(
    video=video, audio=audio, output_path="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/minimax_h3_t2va_lora-3000-720p-T-Rex.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va_lora.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
