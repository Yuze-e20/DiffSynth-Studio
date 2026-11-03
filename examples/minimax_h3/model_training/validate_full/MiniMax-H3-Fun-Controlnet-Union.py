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
        # Load the trained ControlNet here rather than patching it in afterwards:
        # `from_pretrained` fills the state dict before `enable_vram_management` wraps
        # every RMSNorm in an `AutoWrappedModule`, which renames the norm parameters to
        # `blocks.N.norm1.module.weight`. A manual `load_state_dict` after
        # `from_pretrained` sees the wrapped names and fails on every norm key.
        ModelConfig(path="/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/lineart-720p-39/step-1900.safetensors", **vram_config),
        # ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),

    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
# dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*")
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))

# Control video (canny / depth / hed / mlsd / pose) -> Video + Audio.
control_video = VideoData("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4", height=768, width=1344)
control_video = [control_video[i] for i in range(39)]
# prompt = "subject_definitions:\n<Video Line-art> is the source line-art video that defines every visual element of the output: composition, character layout, poses, object placement, motion paths, and camera movement.\n\nsummary:\n[reference generation] The target video is a fully colored anime rendering of <Video Line-art>. This is a rendering task: the line art already specifies the complete content, and the output only adds color, shading, lighting.\n\nretention_analysis:\n<Video Line-art> (composition, motion, camera, and all visual content): fully_preserved — every character position, pose, object, spatial layout, motion trajectory, and camera movement in the line art is retained exactly in the output.\n\ndetailed_description:\nA warm, sunlit interior scene in a traditional Japanese home, featuring soft natural lighting and a peaceful, nostalgic atmosphere. [Shot 1] A young girl with short brown hair sits on a tatami floor, facing away from the camera. She wears an orange sleeveless top over a white shirt and a green skirt. Her legs are extended, and she appears to be resting or looking out a sliding glass door. Beside her, a small orange cat sleeps on a folded cloth. A wicker basket, a standing fan, and a wooden cabinet filled with books and trinkets occupy the background. The room is tidy and bathed in gentle daylight."
prompt = "subject_definitions:\n<Video Line-art> is the source line-art video that defines every visual element of the output: composition, character layout, poses, object placement, motion paths, and camera movement.\n\nsummary:\n[reference generation] The target video is a fully colored anime rendering of <Video Line-art>. This is a rendering task: the line art already specifies the complete content, and the output only adds color, shading, lighting.\n\nretention_analysis:\n<Video Line-art> (composition, motion, camera, and all visual content): fully_preserved — every character position, pose, object, spatial layout, motion trajectory, and camera movement in the line art is retained exactly in the output.\n\ndetailed_description:\nA cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man in a vibrant pink suit standing in a lush green field with sheep.\n[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, stands in a grassy field holding a small black lamb. Several white lambs graze nearby on the rolling green hills. The man looks upward with a contemplative expression, gently cradling the black lamb against his chest. The lighting is soft and directional, casting long shadows and highlighting the texture of the grass and the man's clothing."

video, audio = pipe(
    prompt=prompt,
    height=768, width=1344, num_frames=39,
    num_inference_steps=4, seed=43,
    control_video=control_video, control_scale=1.0,
)
write_video_audio(
    video=video, audio=audio, output_path="lineart-mandog-1900.mp4",
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)