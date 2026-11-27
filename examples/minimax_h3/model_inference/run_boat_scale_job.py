"""Boat scale sweep job — Multi.py setup with boat lineart/blur20 + qwen caption."""
import argparse, os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig, MiniMaxH3MultiControlNet
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet

# Match Multi.py (current)
Structure_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-39-sigma20-10block1-lineart/step-1800.safetensors"
TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-39-sigma20-10block1-blur/step-1800.safetensors"
CONTROL_LAYERS = [
    "0,5,10,15,20,25,30,35,40",
    "0,5,10,15,20,25,30,35,40",
]
LINEART = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS-MiraData_nf328_ctx1g_s20_u05_t05/DAVIS2017/_work/boat_auto/lineart.mp4"
BLUR = "/mnt/nas3/sunyuzework/Diffutoon-2/data/DAVIS-MiraData_nf328_ctx1g_s20_u05_t05/DAVIS2017/_work/boat_auto/blur20.mp4"
# boat has 75 frames @24fps; 17n+5 max = 73
HEIGHT, WIDTH, NUM_FRAMES, FPS = 768, 1344, 73, 24
DETAILED_DESCRIPTION = (
    "A white fishing boat with blue trim and a name in Greek lettering on its bow travels across calm turquoise water close to a dry, low-lying coastline. "
    "The vessel has an enclosed cabin, a deck canopy, coiled ropes, and a white float, with a life ring on the cabin wall. "
    "As it moves, a churning white wake trails from its stern. Over the sequence the boat drifts gradually toward the left of the frame while keeping the same heading, "
    "the frothy wake shifting with its motion. The background of arid brownish hills, scattered white houses, and a clear pale sky remains consistent throughout."
)
PREFIX = (
    "Generate a Japanese anime-style cartoon video with clean line work and flat cel-shaded colors, "
    "following the outlines of the line-art video and the colors, lighting and motion of the blurred video."
)
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

def load_control_video(path):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, fps=FPS)
    print(f"{os.path.basename(path)}: {len(frames)} frames @ {FPS}fps, {frames[0].size}", flush=True)
    assert len(frames) == NUM_FRAMES
    return frames

def apply_control_layers(controlnet, layers_per):
    branches = list(controlnet) if isinstance(controlnet, MiniMaxH3MultiControlNet) else [controlnet]
    for i, (branch, layers) in enumerate(zip(branches, layers_per)):
        before = branch.control_layers
        branch.set_control_layers(layers)
        print(f"branch {i}: {list(before)} -> {list(branch.control_layers)} ({len(branch.blocks)} blocks)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scale = [float(a.scale[0]), float(a.scale[1])]
    for p in (Structure_CONTROLNET, TRAINED_CONTROLNET, LINEART, BLUR):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"scale={scale}\nout={a.out}\nstruct={Structure_CONTROLNET}\ntrained={TRAINED_CONTROLNET}", flush=True)
    cv0, cv1 = load_control_video(LINEART), load_control_video(BLUR)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
            ModelConfig(path=Structure_CONTROLNET, **vram_config),
            ModelConfig(path=TRAINED_CONTROLNET, **vram_config),
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
    )
    apply_control_layers(pipe.controlnet, CONTROL_LAYERS)
    pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo",
                    origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))
    prompt = f"{PREFIX}\ndetailed_description:\n{DETAILED_DESCRIPTION}"
    video, audio = pipe(
        prompt=prompt, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=4, seed=43,
        control_video=[cv0, cv1], control_scale=scale,
    )
    write_video_audio(video=video, audio=audio, output_path=a.out, fps=24,
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    print(f"wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
