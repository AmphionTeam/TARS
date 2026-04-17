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

# ==== Run in 2 modes ====
echo "model_path: $model_path"

# prompt=cot
prompt=systhink
# prompt=none

# FIXME: save dir
save_dir=output

accelerate launch --multi_gpu --main_process_port 65532 -m eval.qa_xfinder.evaluate_vllm --model_path $model_path --dataset_path $DATASET_PATH --save_dir $save_dir --mode audio_mode2 --prompt $prompt
accelerate launch --multi_gpu --main_process_port 65531 -m eval.qa_xfinder.evaluate_vllm --model_path $model_path --dataset_path $DATASET_PATH --save_dir $save_dir --mode text_mode2 --prompt $prompt

echo "DONE!!! SAVED TO $model_path"