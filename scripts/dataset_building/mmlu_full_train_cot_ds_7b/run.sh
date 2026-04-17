# audio_mode2 generation (as student)
# generate cot response, saved in outputs folder
accelerate launch --main_process_port 36990 -m scripts.dataset_building.mmlu_full_train_cot_ds_7b.gen_phi_mmlu_dpo_cot \
    --model_path microsoft/Phi-4-multimodal-instruct \
    --mode audio_mode2 \
    --dataset_path yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1

# aggregate outputs from different ranks
cat output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_rank*.jsonl > output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_aggregated.jsonl

# check results
head output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_aggregated.jsonl

# Run all above again, with different mode text_mode2 (as teacher)
accelerate ...
cat ...
head ...

# merge cot response back to dataset
# Here, using text as chosen (teacher) and audio as rejected (student).
python -m scripts.dataset_building.mmlu_full_train_cot_ds_7b.build_ds \
    --dataset_path yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1 \
    --chosen_aggregated_jsonl_path output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_text_mode2_cot_outputs_aggregated.jsonl \
    --rejected_aggregated_jsonl_path output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_aggregated.jsonl \
    --save_dataset_path output/MMLU_FULL/full_train_dpo_7b_hf_format.merged.v1