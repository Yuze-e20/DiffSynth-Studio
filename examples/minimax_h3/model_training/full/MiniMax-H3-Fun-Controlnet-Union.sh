# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset
export DIFFSYNTH_MODEL_BASE_PATH="/root/models"

# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/data \
  --dataset_metadata_path /mnt/nas3/sunyuzework/Diffutoon-2/data/xinhaicheng_39_sr_allshot.jsonl \
  --data_file_keys "video,input_audio,control_video" \
  --extra_inputs "input_audio,control_video" \
  --max_pixels 1032192 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-split-cache-720p-39" \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --audio_loss_weight 0.0 \

# stage 2 (train)
accelerate launch --config_file /mnt/nas3/sunyuzework/Diffutoon-2/DiffSynth-Studio/examples/minimax_h3/model_training/full/accelerate_config_zero2.yaml examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Fun-Controlnet-Union-split-cache-720p-39 \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --max_pixels 1032192 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union_full-720p-39" \
  --preset_lora_path /root/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model.safetensors \
  --preset_lora_model dit \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train" \
  --audio_loss_weight 0.0 \
