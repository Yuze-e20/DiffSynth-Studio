# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-FL2VA/*" --local_dir ./data/diffsynth_example_dataset
export DIFFSYNTH_MODEL_BASE_PATH="/root/models"
# Optional
# 1. Fuse the DeCFG training adapter into the DiT while training, for a better optimization landscape on this CFG-distilled base. Training only -- do not load it at inference.
# modelscope download --model DiffSynth-Studio/MiniMax-H3-TrainingAdapter --local_dir ./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter
#   --preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model.safetensors" \
#   --preset_lora_model "dit"
# 2. Add `--training_cfg_scale 4` to both stages below to enable CFG-aware training. Both stages must use the same value because the unconditional embeddings are cached in stage 1.

# T2VA - stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/data \
  --dataset_metadata_path /mnt/nas3/sunyuzework/Diffutoon-2/data/xinhaicheng_124_t2va_allshot_audio.jsonl \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --max_pixels 399360 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-ToonLoRA-split-cache-480p-124" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 64 \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --audio_loss_weight 0.0 \

# T2VA - stage 2 (train)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-ToonLoRA-split-cache-480p-124 \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --max_pixels 399360 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-ToonLoRA-split-480p-124" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --preset_lora_path /root/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model.safetensors \
  --preset_lora_model dit \
  --lora_rank 64 \
  --use_gradient_checkpointing \
  --audio_loss_weight 0.0 \
  --find_unused_parameters \
  --task "sft:train"