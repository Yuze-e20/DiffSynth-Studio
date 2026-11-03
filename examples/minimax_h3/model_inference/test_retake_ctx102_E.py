"""E only: same span [22, 146) and same seed as D, retake disabled.

D and A are already on disk from the previous run; this fills in the baseline the
22 new frames must be judged against. Config is copied verbatim from
test_retake_ctx102.py so the comparison stays valid.
"""
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

HEIGHT, WIDTH = 1568, 1568
NUM_FRAMES, CONTEXT_FRAMES = 124, 102
FPS, STEPS, SEED = 24, 4, 43
G1 = NUM_FRAMES - CONTEXT_FRAMES
NEED = G1 + NUM_FRAMES

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


def load_control(path):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=NEED, fps=FPS)
    assert len(frames) == NEED, f"{path} gave {len(frames)}, need {NEED}"
    return frames


control_a, control_b = load_control(CONTROL_A), load_control(CONTROL_B)
video_e, audio_e = pipe(
    prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
    num_inference_steps=STEPS, seed=SEED + 1, control_scale=[0.2, 0.8],
    control_video=[control_a[G1:G1 + NUM_FRAMES], control_b[G1:G1 + NUM_FRAMES]],
)
write_video_audio(video=video_e, audio=audio_e, output_path="test_E_fresh.mp4",
                  fps=FPS, audio_sample_rate=pipe.audio_vae.sample_rate)
print("done")
