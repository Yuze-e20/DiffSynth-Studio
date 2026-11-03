import os
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/root/models"
import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
import diffsynth.pipelines.minimax_h3_audio_video as mm
from diffsynth.models.minimax_h3_controlnet import MiniMaxH3ControlNet
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio

# ---------------------------------------------------------------- configuration
HEIGHT, WIDTH = 1344, 1344
NUM_FRAMES    = 124         # window size, must be 17n+5
CONTEXT_FRAMES = 90          # pinned tail from the previous segment；
TOTAL_FRAMES  = 360         # how long the finished video should be
FPS           = 24
STEPS         = 4
SEED          = 43

# Also hand every continuation segment its first frame through FL2VA's native
# first-frame conditioning, on top of the retake context region. The two act on
# different channels: retake pins the first PIN *latents* of the noisy video
# stream, while a keyframe is encoded into its own condition rows that the
# FL2VA DiT was trained to continue from. Set to False to fall back to
# retake-only, which is what this script did before.
USE_FIRST_FRAME = True

# Cross-fade the overlap region instead of cutting at it. Segment k regenerates
# the CONTEXT_FRAMES it was pinned on, and those re-renders are currently thrown
# away (`video[new_from:]`) -- the assembled video jumps from segment k-1's decode
# straight to segment k's decode at one frame. Both versions depict the same
# content but differ by a VAE round-trip, so that jump is visible. Blending the
# two across the overlap spreads the difference over CONTEXT_FRAMES instead of
# concentrating it at a single cut. Set to False to restore the hard cut.
BLEND_CONTEXT = True

VAE_CLIP = 17
assert NUM_FRAMES % VAE_CLIP == 5, "num_frames must be 17n+5"
def latents_to_pin(frames, clip=VAE_CLIP, per_clip=5):
    """
    统计context对应多少个latent
    """
    n = 0
    for L in range(10 ** 6):
        clip_idx, within = divmod(L, per_clip)
        end = clip_idx * clip + (0 if within == 0 else 4 * within)   # last frame latent L sees
        if end < frames:
            n = L + 1
        else:
            break
    return n


PIN = latents_to_pin(CONTEXT_FRAMES)
STRIDE = NUM_FRAMES - CONTEXT_FRAMES
_orig_retake = mm.MiniMaxH3Unit_VideoRetakeEmbedder.process


def _patched_retake(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw):
    """The stock mask starts at `floor(start / 17) * 5`, which only recognises
    multiples of 17 and floors anything shorter to 0 -- a 5-frame context silently
    pins nothing and retake becomes a no-op. Recompute the start index from the
    real frame -> latent table instead."""
    out = _orig_retake(self, pipe, retake_video, frame_regions_to_retake, video_latents, *a, **kw)
    if not out:
        return out
    latents = out["input_latents_video"]
    mask = torch.zeros(latents.shape[2], dtype=pipe.torch_dtype, device=pipe.device)
    mask[PIN:] = 1                                   # 1 = regenerate
    return {"input_latents_video": latents,
            "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *latents.shape[-2:])}


mm.MiniMaxH3Unit_VideoRetakeEmbedder.process = _patched_retake

# ------------------------------------------- ControlNet x first-frame condition
# Passing `keyframes` switches the packed layout to [text | cond | audio | video],
# so `img_pos` grows by one frame of condition rows at the front. The ControlNet
# encoder still produces exactly one control row per *video* latent row, so the
# stock forward rejects the pair outright:
#     control_rows hold N rows but the packed sequence has N+frame_rows video rows
# Front-pad the control stream to cover the condition rows and zero the hint
# there, so the control signal keeps landing on the video rows one-to-one and the
# keyframe anchor is passed through untouched. This also matches how the
# ControlNets were trained -- `--extra_inputs "input_audio,control_video"`, never
# with `input_image`, so they have never seen a condition row carry control.
_orig_cn_forward = MiniMaxH3ControlNet.forward


def _patched_cn_forward(self, embeddings, control_rows, img_pos, audio_pos, **kw):
    n_cond = img_pos.shape[0] - control_rows.shape[0]
    if n_cond <= 0:
        return _orig_cn_forward(self, embeddings, control_rows, img_pos, audio_pos, **kw)
    pad = control_rows.new_zeros((n_cond, control_rows.shape[-1]))
    hints = _orig_cn_forward(self, embeddings, torch.cat([pad, control_rows], dim=0),
                             img_pos, audio_pos, **kw)
    keep = torch.ones(embeddings.shape[0], dtype=hints[0].dtype, device=hints[0].device)
    keep[img_pos[:n_cond]] = 0
    return tuple(hint * keep.unsqueeze(-1) for hint in hints)


MiniMaxH3ControlNet.forward = _patched_cn_forward

TRAINED_CONTROLNET = "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39-sigma20/step-1600.safetensors"
CONTROL_A = "/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_lineart.mp4"
CONTROL_B = "/mnt/nas3/sunyuzework/Diffutoon-2/data/examples/data/examples/diffutoon/input_video_blur15.mp4"
OUTPUT    = "dancing_long_video_controlnet_multi_regions_371_blur15_framectx90.mp4"

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
print(f"ControlNet branches: {len(pipe.controlnet)}")
# pipe.load_lora(pipe.dit, "/mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/MiniMax-H3-ToonLoRA-split-480p-124_merged/step-2600.safetensors")
pipe.load_lora(pipe.dit, ModelConfig(model_id="lightx2v/Minimax-h3-Turbo", origin_file_pattern="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"))
# --------------------------------------------------------------- control tracks
def load_control(path, num_frames, max_padding):
    frames, _, _ = read_video_audio(path, height=HEIGHT, width=WIDTH,
                                    num_frames=num_frames, fps=FPS)
    short = num_frames - len(frames)
    print(f"{path.split('/')[-1]}: {len(frames)} frames @ {FPS}fps ({len(frames) / FPS:.2f}s)"
          + (f" · padding {short} discarded frames" if short else ""))
    assert short <= max_padding, (
        f"{path.split('/')[-1]} is {short - max_padding} frames too short: it gives "
        f"{len(frames)} frames at {FPS}fps but {num_frames - max_padding} are needed for "
        f"{TOTAL_FRAMES} real output frames. Set TOTAL_FRAMES to at most {len(frames)} "
        f"and ideally to a value whose schedule fits, e.g. {NUM_FRAMES + ((len(frames) - NUM_FRAMES) // STRIDE) * STRIDE}.")
    return frames + [frames[-1]] * short

# --------------------------------------------------------------------- schedule
segments = [] # [0, 119, 238]
g = 0
while g < TOTAL_FRAMES:
    segments.append(g)
    if g + NUM_FRAMES >= TOTAL_FRAMES:
        break
    g += STRIDE
print(f"{len(segments)} segments · window {NUM_FRAMES} · context {CONTEXT_FRAMES} "
      f"({PIN} latents) · stride {STRIDE}")

# The last window may reach past TOTAL_FRAMES; the control tracks must cover it,
# since every window is generated in full before being truncated at the end.
CONTROL_FRAMES = segments[-1] + NUM_FRAMES
DISCARDED = CONTROL_FRAMES - TOTAL_FRAMES
print(f"assembling {CONTROL_FRAMES} -> truncating to {TOTAL_FRAMES}"
      + (f" (discarding {DISCARDED})" if DISCARDED else ""))

# Read both tracks up front, resampled onto FPS. The assert inside fires now rather
# than after several segments of generation if a track is too short.
control_a = load_control(CONTROL_A, CONTROL_FRAMES, DISCARDED)
control_b = load_control(CONTROL_B, CONTROL_FRAMES, DISCARDED)

# ------------------------------------------------------------------------- loop
all_frames = []          # the assembled long video
audio_chunks = []        # per-segment newly generated audio
prev_audio = None
audio_sr = pipe.audio_vae.sample_rate
ctx_seconds = CONTEXT_FRAMES / FPS

# Regenerate everything after the pinned context. The region's `start` only has to
# land inside the first latent that should be regenerated -- the patched mask reads
# the boundary from PIN, not from this number.
video_regions = [(CONTEXT_FRAMES, NUM_FRAMES)]
audio_regions = [(CONTEXT_FRAMES / FPS, NUM_FRAMES / FPS)]
print(f"retake regions · frames {video_regions[0]} · seconds "
      f"({audio_regions[0][0]:.4f}, {audio_regions[0][1]:.4f}) · pinning {PIN} latents")
print(f"first-frame conditioning: {'on' if USE_FIRST_FRAME else 'off'}")
if BLEND_CONTEXT and CONTEXT_FRAMES > 1:
    _w = [i / (CONTEXT_FRAMES - 1) for i in range(CONTEXT_FRAMES)]
    _p = ", ".join(f"{1-w:.2f}" for w in _w[:3]) + " ... " + f"{1-_w[-1]:.2f}"
    _n = ", ".join(f"{w:.2f}" for w in _w[:3]) + " ... " + f"{_w[-1]:.2f}"
    print(f"context blend · prev [{_p}] · new [{_n}]")
else:
    print("context blend: off (hard cut)")

for k, g0 in enumerate(segments):
    kwargs = dict(
        prompt=PROMPT,
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        seed=SEED + k,                       # vary per segment or every one repeats
        control_video=[control_a[g0:g0 + NUM_FRAMES],
                       control_b[g0:g0 + NUM_FRAMES]],
        control_scale=[0.3, 0.7],
    )

    if k > 0:
        context = all_frames[-CONTEXT_FRAMES:]
        kwargs["retake_video"] = context + control_b[g0 + CONTEXT_FRAMES:g0 + NUM_FRAMES]
        kwargs["frame_regions_to_retake"] = video_regions
        assert len(kwargs["retake_video"]) == NUM_FRAMES, len(kwargs["retake_video"])
        if USE_FIRST_FRAME:
            # all_frames currently ends at global frame g0 + CONTEXT_FRAMES, so
            # context[0] is global frame g0 -- window-local frame 0, exactly what
            # keyframe_indices=[0] conditions on.
            kwargs["keyframes"] = [context[0]]
            kwargs["keyframe_indices"] = [0]
        if prev_audio is not None:
            tail = int(ctx_seconds * audio_sr)
            kwargs["retake_audio"] = prev_audio[:, -tail:]
            kwargs["retake_audio_sample_rate"] = audio_sr
            kwargs["seconds_regions_to_retake"] = audio_regions

    video, audio = pipe(**kwargs)

    new_from = 0 if k == 0 else CONTEXT_FRAMES
    if k > 0 and BLEND_CONTEXT and CONTEXT_FRAMES > 1:
        # Linear cross-fade over the overlap. The two weight ramps must be
        # complementary (w_new = 1 - w_prev), not mirrored -- a mirrored pair like
        # (1.0 .. 0.2) against (0.2 .. 1.0) sums to 1.2 at every position and
        # brightens the whole overlap by 20%. i/(n-1) reaches 1.0 on the last
        # overlap frame, so it meets segment k's own output with no step left over.
        n = CONTEXT_FRAMES
        base = len(all_frames) - n          # global index of overlap frame 0
        for i in range(n):
            w_new = i / (n - 1)
            all_frames[base + i] = Image.blend(
                all_frames[base + i].convert("RGB"),
                video[i].convert("RGB"),
                w_new,
            )
    all_frames.extend(video[new_from:])
    if audio is not None:
        cut = int((new_from / FPS) * audio_sr)
        audio_chunks.append(audio[:, cut:])
        prev_audio = audio

    print(f"  seg {k+1}/{len(segments)} · global [{g0},{g0+NUM_FRAMES}) "
          f"· +{len(video) - new_from} frames · total {len(all_frames)}")

all_frames = all_frames[:TOTAL_FRAMES]
full_audio = torch.cat(audio_chunks, dim=-1) if audio_chunks else None
if full_audio is not None:
    full_audio = full_audio[:, :int(len(all_frames) / FPS * audio_sr)]

write_video_audio(video=all_frames, audio=full_audio, output_path=OUTPUT,
                  fps=FPS, audio_sample_rate=audio_sr)
print(f"wrote {OUTPUT} · {len(all_frames)} frames · {len(all_frames)/FPS:.1f}s")