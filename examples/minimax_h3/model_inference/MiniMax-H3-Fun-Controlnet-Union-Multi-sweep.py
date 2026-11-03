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
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio_video import write_video_audio

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
# TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma30/step-1800.safetensors"
TRAINED_CONTROLNET = os.path.join("/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20",
    "step-" + os.environ["SWEEP_CN_STEP"] + ".safetensors")
assert os.path.exists(TRAINED_CONTROLNET), TRAINED_CONTROLNET

# TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma15/step-3000.safetensors"

# HEIGHT, WIDTH, NUM_FRAMES = 1568, 1568, 124
HEIGHT, WIDTH, NUM_FRAMES = 768, 1344, 124
def load_control_video(path):
    video = VideoData(path, height=HEIGHT, width=WIDTH)
    print(video.height, video.width, len(video))
    return [video[i] for i in range(NUM_FRAMES)]

control_video_union = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4")
control_video_trained = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_gaussian20.mp4")
# control_video_union = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4")
# control_video_trained = load_control_video("/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_gaussian30.mp4")

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        # Branch 0: the public union ControlNet.
        ModelConfig(model_id="PAI/MiniMax-H3-Fun-Controlnet-Union", origin_file_pattern="MiniMax-H3-Fun-Controlnet-Union.safetensors", **vram_config),
        # Branch 1: your own checkpoint. Branch order here is the order used by
        # `control_video` and `control_scale` below.
        ModelConfig(path=TRAINED_CONTROLNET, **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
print(f"ControlNet branches loaded: {len(pipe.controlnet)}")


_toon_lora = os.path.join("/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split-480p-124_merged",
    "step-" + os.environ["SWEEP_LORA_STEP"] + ".safetensors")
assert os.path.exists(_toon_lora), _toon_lora
print("SWEEP controlnet:", TRAINED_CONTROLNET, flush=True)
print("SWEEP toon lora:", _toon_lora, flush=True)
print("SWEEP control_scale:", os.environ["SWEEP_SCALE_UNION"], os.environ["SWEEP_SCALE_TRAINED"], flush=True)
pipe.load_lora(pipe.dit, _toon_lora)
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))



# One control video per branch. Point these at your own inputs -- e.g. a lineart
# clip for the union ControlNet and the blurred clip for a checkpoint trained on
# the super-resolution task.
# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA warm, sunlit interior scene in a traditional Japanese home, featuring soft natural lighting and a peaceful, nostalgic atmosphere. [Shot 1] A young girl with short brown hair sits on a tatami floor, facing away from the camera. She wears an orange sleeveless top over a white shirt and a green skirt. Her legs are extended, and she appears to be resting or looking out a sliding glass door. Beside her, a small orange cat sleeps on a folded cloth. A wicker basket, a standing fan, and a wooden cabinet filled with books and trinkets occupy the background. The room is tidy and bathed in gentle daylight."

prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA cinematic outdoor scene bathed in warm, golden hour sunlight, featuring a man in a vibrant pink suit standing in a lush green field with sheep.\n[Shot 1] A man with wavy blonde hair, wearing a bright pink suit and white shirt, stands in a grassy field holding a small black lamb. Several white lambs graze nearby on the rolling green hills. The man looks upward with a contemplative expression, gently cradling the black lamb against his chest. The lighting is soft and directional, casting long shadows and highlighting the texture of the grass and the man's clothing."
# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs."
# prompt = "Fully colored Japanese anime-style animation, hand-drawn cel-shaded artwork with clean line work, vibrant colors, and expressive character designs.\ndetailed_description\nA girl is dancing."

video, audio = pipe(
    prompt=prompt,
    height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=4, seed=43,
    control_video=[control_video_union, control_video_trained],
    control_scale=[float(os.environ["SWEEP_SCALE_UNION"]),
                   float(os.environ["SWEEP_SCALE_TRAINED"])],
)
write_video_audio(
    video=video, audio=audio, output_path=os.environ["SWEEP_OUT"],
    fps=24, audio_sample_rate=pipe.audio_vae.sample_rate,
)
