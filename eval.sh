# eval qa task

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_path>"
    echo "Note: Before running evaluation with vLLM, you MUST merge the speech LoRA first."
    echo "      Run: python scripts/merge_speech_lora.py --output_path <your_merged_model_path>"
    echo "      Then pass <your_merged_model_path> as the <model_path> here."
    exit 1
fi

export DATASET_PATH=yuantuo666/VB_MMSU-full_3074_hf_format.v0
bash eval/eval_qa.sh $1
sleep 30

export DATASET_PATH=yuantuo666/VB_OBQA-full_455_hf_format.v0
bash eval/eval_qa.sh $1
sleep 30

export DATASET_PATH=yuantuo666/MMSU-full_5k_hf_format.v0
bash eval/eval_qa.sh $1

# eval asr
export DATASET_PATH=yuantuo666/LIBRISPEECH-test_hf_format.v1
bash eval/eval_asr.sh $1