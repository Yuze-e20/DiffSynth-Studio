# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset
export DIFFSYNTH_MODEL_BASE_PATH="/root/models"

# DiT block ids that receive ControlNet hints. Must start with 0.
# Default (omit --control_layers or set to the value below) matches Fun-Controlnet-Union:
#   CONTROL_LAYERS="0,10,20,30,40"
# Example denser layout (6 blocks): CONTROL_LAYERS="0,8,16,24,32,40"


# stage 1 (data process) -- ControlNet is not loaded here; --control_layers is ignored.
# accelerate launch examples/minimax_h3/model_training/train.py \
#   --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/data \
#   --dataset_metadata_path /mnt/nas3/sunyuzework/Diffutoon-2/data/merged_sigma20_newprompt.jsonl \
#   --data_file_keys "video,input_audio,control_video" \
#   --extra_inputs "input_audio,control_video" \
#   --max_pixels 1032192 \
#   --num_frames 39 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.controlnet." \
#   --output_path "./models/train/blur-split-cache-720p-39-sigma20-newprompt" \
#   --trainable_models "controlnet" \
#   --use_gradient_checkpointing \
#   --task "sft:data_process" \
#   --audio_loss_weight 0.0

# stage 2 (train)
accelerate launch --config_file /mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/examples/minimax_h3/model_training/full/accelerate_config_zero2.yaml examples/minimax_h3/model_training/train.py \
  --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/models/train/lineart-split-cache-720p-39 \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --max_pixels 1032192 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --save_steps 100 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/blur-720p-39-sigma20-9block1-lineart-2" \
  --preset_lora_path /root/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model.safetensors \
  --preset_lora_model dit \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train" \
  --control_layers "0,8,13,18,23,28,33,38,43" \
  --audio_loss_weight 0.0
