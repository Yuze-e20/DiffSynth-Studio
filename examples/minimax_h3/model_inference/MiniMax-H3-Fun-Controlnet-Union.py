import torch
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio
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
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*")

# Control video (canny / depth / hed / mlsd / pose) -> Video + Audio.
control_video = VideoData("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4", height=480, width=832)
control_video = [control_video[i] for i in range(124)]
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
prefix = "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, following the outlines of the line-art video"

prompt = (
    f"{prefix}\n"
    f"detailed_description:\n{DETAILED_DESCRIPTION}"
)
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,
    num_inference_steps=40, seed=43,
    control_video=control_video, control_scale=1.0,
)
write_video_audio(
    video=video, audio=audio, output_path="minimax_h3_fun_controlnet_union_control_lineart.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
