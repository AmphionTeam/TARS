#!/bin/bash

# ============================================
# ================= CONFIG ===================
# ============================================

eval "$(conda shell.bash hook)"
conda activate trl

# update dataset_path via env var or modify below
dataset_path=${DATASET_PATH:-"yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1"}
ds_short=ftcotv1
batch_size_per_gpu=1

prompt_modality=a
prompt_type=systhink

chosen_type=am2
rejected_type=am2

num_train_epochs=1

# training config
WORLD_BATCH_SIZE=64 # FIXME: set batch size to same with grpo

# version config
version="dpo"

# ============================================
# ================ CODE ENV ==================
# ============================================

code_dir=$(dirname $0)
code_dir=$(realpath $code_dir)
parent_dir=$(dirname $code_dir)

current_date=$(date +'%Y%m%d_%H%M') # 20250611_2004
version=$version"_"$current_date
# FIXME: where to save the ckpts
output_dir=output/phi4-dpo/$version

mkdir -p $output_dir

echo "================================================"
echo "Start training!"
echo "WORLD_BATCH_SIZE: $WORLD_BATCH_SIZE"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"
echo "================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
accelerate launch -m \
--config_file config/deepspeed_zero2.yaml \
--num_processes=$NUM_PROCESSES \
--main_process_port 23456 trainer.dpo \
--use_flash_attention \
--batch_size_per_gpu $batch_size_per_gpu \
--batch_size $WORLD_BATCH_SIZE \
--output_dir $output_dir \
--num_train_epochs $num_train_epochs \
--learning_rate 2e-5 \
--beta 0.1 \
--loss_type sigmoid \
--prompt_modality $prompt_modality \
--prompt_type $prompt_type \
--chosen_type $chosen_type \
--rejected_type $rejected_type \
--dataset_path $dataset_path

# FIXME: loss_type choose sft / sigmoid (dpo)
# FIXME: learning_rate & num_train_epochs