#!/usr/bin/env bash
# 批量为 species_csv 下每个 CSV 训练一个独立 LoRA。
# 不修改 DiffSynth 代码，只是循环调用 train.py。
# 用法:
#   bash train_inaturalist_all.sh [START] [END] [GPU]
#   START/END: 处理 CSV 的序号区间 [START, END)，缺省=全部。可用于多卡分片。
#   GPU:       CUDA_VISIBLE_DEVICES，缺省 0。
DIFFSYNTH_MODEL_BASE_PATH="/root/models"
set -euo pipefail

# ---- 路径配置（按需修改）----
DIFFSYNTH_ROOT="/mnt/nas3/sunyuzework/ImageGenRAG/DiffSynth-Studio"
CSV_DIR="/mnt/nas3/sunyuzework/ImageGenRAG/data/inaturalist/species_csv"
BASE_PATH="/mnt/nas3/sunyuzework/ImageGenRAG/data/inaturalist"
OUT_ROOT="/mnt/nas3/sunyuzework/ImageGenRAG/models/inaturalist_lora"

# ---- 训练超参（测试时建议调小）----
NUM_EPOCHS=5
DATASET_REPEAT=1
LEARNING_RATE=1e-4
LORA_RANK=32
MAX_PIXELS=1048576

# ---- 分片参数 ----
START="${1:-0}"
END="${2:-999999}"
export CUDA_VISIBLE_DEVICES="${3:-0}"

cd "$DIFFSYNTH_ROOT"
mkdir -p "$OUT_ROOT"

# 末个 epoch 的 checkpoint 文件名，用于判断是否已训练完成
LAST_CKPT="epoch-$((NUM_EPOCHS - 1)).safetensors"

LORA_TARGET_MODULES="to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out"

idx=0
for CSV in "$CSV_DIR"/*.csv; do
  # 区间分片：[START, END)
  if [ "$idx" -lt "$START" ] || [ "$idx" -ge "$END" ]; then
    idx=$((idx + 1)); continue
  fi

  name="$(basename "$CSV" .csv)"          # 例如 00000
  out_dir="$OUT_ROOT/$name"

  # 断点续跑：最终 epoch 已存在则跳过
  if [ -f "$out_dir/$LAST_CKPT" ]; then
    echo "[SKIP] $name 已完成 ($out_dir/$LAST_CKPT)"
    idx=$((idx + 1)); continue
  fi

  echo "[TRAIN] ($idx) $name -> $out_dir  (GPU=$CUDA_VISIBLE_DEVICES)"
  mkdir -p "$out_dir"

  accelerate launch examples/flux2/model_training/train.py \
    --dataset_base_path "$BASE_PATH" \
    --dataset_metadata_path "$CSV" \
    --max_pixels "$MAX_PIXELS" \
    --dataset_repeat "$DATASET_REPEAT" \
    --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
    --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$out_dir" \
    --lora_base_model "dit" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_rank "$LORA_RANK" \

  idx=$((idx + 1))
done

echo "全部完成。范围 [$START, $END)，GPU=$CUDA_VISIBLE_DEVICES"
