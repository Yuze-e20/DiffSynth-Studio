export DIFFSYNTH_MODEL_BASE_PATH="/root/models"
# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/data \
  --dataset_metadata_path /mnt/nas3/sunyuzework/Diffutoon-2/data/xinhaicheng_39_oneshot_nocaption_render.jsonl \
  --data_file_keys "video,input_audio,vace_video" \
  --extra_inputs "input_audio,vace_video" \
  --max_pixels 399360 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/MiniMax-H3-VACE-480p-nocaption-cache-Ref2VA-final" \
  --use_gradient_checkpointing \
  --trainable_models "vace" \
  --task "sft:data_process"

# stage 2 (train)
accelerate launch --config_file /mnt/nas3/sunyuzework/myown/DiffSynth-Studio/examples/minimax_h3/model_training/full/accelerate_config_zero2.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path /mnt/nas3/sunyuzework/myown/DiffSynth-Studio/models/train/MiniMax-H3-VACE-720p-nocaption-cache \
  --data_file_keys "video,input_audio,vace_video" \
  --extra_inputs "input_audio,vace_video" \
  --max_pixels 399360 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --save_steps 200 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/MiniMax-H3-VACE-480p-nocaption-Ref2VA-final" \
  --trainable_models "vace" \
  --vace_layers "0,1,2,3,4,5,6,7,8,9" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train" \
  --enable_tensorboard_log \
  --vace_log_variance