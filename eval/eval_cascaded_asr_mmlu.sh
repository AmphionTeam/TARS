#!/bin/bash
set -e

# Running following command to evaluate cascaded asr mmlu. Where $1 refers to path to checkpoint-xxx
# export dataset_path=${DATASET_PATH:-"/path/to/your/dataset/VB_MMSU/full_3074_hf_format.v0"}
# bash eval/eval_cascaded_asr_mmlu.sh $1
# export dataset_path=${DATASET_PATH:-"/path/to/your/dataset/VB_OBQA/full_455_hf_format.v0"}
# bash eval/eval_cascaded_asr_mmlu.sh $1
# export dataset_path=${DATASET_PATH:-"/path/to/your/dataset/MMLU_FULL/full_5k_hf_format.v1"}
# bash eval/eval_cascaded_asr_mmlu.sh $1

export NCCL_DEBUG=WARN

eval "$(conda shell.bash hook)"
conda activate trl

model_path=$1

if [ ! -d "$model_path/speech-lora" ]; then
    echo "model_path: $model_path does not contain speech-lora, extract it..."
    python scripts/separate_lora_adapator/extract_speech_lora.py  --model_path $model_path --adapter_name speech
fi

# ==== Run ASR ====
accelerate launch --multi_gpu --main_process_port 69503 -m eval.asr.evaluate_vllm --batch_size_per_gpu 16 --model_path $model_path --dataset_path $dataset_path


# ==== Run in asr mode ====
echo "model_path: $model_path"

# prompt=cot
prompt=systhink
# prompt=none

accelerate launch --multi_gpu --main_process_port 35534 -m eval.qa_xfinder.evaluate_vllm --model_path $model_path --dataset_path $dataset_path --mode asr_mode2 --prompt $prompt

echo "DONE!!! SAVED TO $model_path"