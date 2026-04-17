#!/bin/bash
set -e

export NCCL_DEBUG=WARN

eval "$(conda shell.bash hook)"
conda activate trl

model_path=$1

# ==== MMLU on Speech Text ====
if [ ! -d "$model_path/speech-lora" ]; then
    echo "model_path: $model_path does not contain speech-lora, extract it..."
    python scripts/separate_lora_adapator/extract_speech_lora.py  --model_path $model_path --adapter_name speech
fi

# ==== Dataset Path ====
dataset_path=${DATASET_PATH:-"yuantuo666/LIBRISPEECH-test_hf_format.v1"}

# ==== Run ASR ====
asr_dataset_path=$dataset_path
accelerate launch --multi_gpu --main_process_port 69503 -m eval.asr.evaluate_vllm --batch_size_per_gpu 16 --model_path $model_path --dataset_path $asr_dataset_path