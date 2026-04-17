#!/bin/bash

# pip install matplotlib seaborn

# MAX_SAMPLES=100 # for debug use, set to a large number for full run
MAX_SAMPLES=10000

MODEL_PATH=${MODEL_PATH:-"/path/to/your/checkpoint"}

# ===== MMLU =====
RESULT_JSON="${MODEL_PATH}/MMLU_FULL--full_5k_hf_format.v1/results_eval_text_mode2_systhink_vllm_xfinder.json"
DATASET_PATH=${DATASET_PATH:-"/path/to/your/dataset/MMLU_FULL/full_5k_hf_format.v1"}
python analysis_and_draw.py \
--text_result_json "$RESULT_JSON" \
--dataset_path "$DATASET_PATH" \
--model_path "$MODEL_PATH" \
--max_samples $MAX_SAMPLES

# ===== MMSU =====
RESULT_JSON="${MODEL_PATH}/VB_MMSU--full_3074_hf_format.v0/results_eval_text_mode2_systhink_vllm_xfinder.json"
DATASET_PATH=${DATASET_PATH:-"/path/to/your/dataset/VB_MMSU/full_3074_hf_format.v0"}
python analysis_and_draw.py \
--text_result_json "$RESULT_JSON" \
--dataset_path "$DATASET_PATH" \
--model_path "$MODEL_PATH" \
--max_samples $MAX_SAMPLES

# ===== OBQA =====
RESULT_JSON="${MODEL_PATH}/VB_OBQA--full_455_hf_format.v0/results_eval_text_mode2_systhink_vllm_xfinder.json"
DATASET_PATH=${DATASET_PATH:-"/path/to/your/dataset/VB_OBQA/full_455_hf_format.v0"}
python analysis_and_draw.py \
--text_result_json "$RESULT_JSON" \
--dataset_path "$DATASET_PATH" \
--model_path "$MODEL_PATH" \
--max_samples $MAX_SAMPLES


python aggregate_plots.py