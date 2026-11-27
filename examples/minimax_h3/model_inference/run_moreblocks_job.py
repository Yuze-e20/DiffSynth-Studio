"""Multi.py-aligned job: fixed structure/trained ckpts+layers, tunable --scale."""
import argparse
import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig, MiniMaxH3MultiControlNet
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio
from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet

# Mirrors MiniMax-H3-Fun-Controlnet-Union-Multi.py (user-updated)
Structure_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-39-sigma20-9block1-lineart-2/step-1800.safetensors"
TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/blur-720p-39-sigma20-10block1-blur/step-2000.safetensors"
CONTROL_LAYERS = [
    "0,8,13,18,23,28,33,38,43",  # structure 9-block
    "0,5,10,15,20,25,30,35,40",  # trained
]
LINEART = "/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_lineart.mp4"
BLUR = "/mnt/nas3/sunyuzework/myown/DiffSynth-Studio/data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/video_gaussian20.mp4"
HEIGHT, WIDTH, NUM_FRAMES, FPS = 768, 1344, 141, 24
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
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
PREFIX = (
    "Generate a Japanese anime-style cartoon video with clean line work and flat "
    "cel-shaded colors, following the outlines of the line-art video and the colors, "
    "lighting and motion of the blurred video."
)

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
    ap.add_argument("--scale", type=float, nargs=2, required=True, metavar=("STRUCT", "TRAINED"))
    ap.add_argument("--out", required=True)
    # kept for old callers; ignored (ckpts fixed to Multi.py)
    ap.add_argument("--branch", default="both")
    ap.add_argument("--step", type=int, default=0)
    a = ap.parse_args()
    scale = [float(a.scale[0]), float(a.scale[1])]
    for p in (Structure_CONTROLNET, TRAINED_CONTROLNET):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"scale={scale}\nstructure={Structure_CONTROLNET}\ntrained={TRAINED_CONTROLNET}\nlayers={CONTROL_LAYERS}\nout={a.out}", flush=True)
    cv0 = load_control_video(LINEART)
    cv1 = load_control_video(BLUR)
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
    print(f"ControlNet branches: {len(pipe.controlnet)}", flush=True)
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
