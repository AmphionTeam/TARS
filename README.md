# [ACL 2026] TARS: Closing the Modality Reasoning Gap for Speech Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2601.05543-b31b1b.svg)](https://arxiv.org/abs/2601.05543)

This repository contains the official implementation of **TARS** (Trajectory Alignment for Reasoning in Speech), as presented in the paper *"Closing the Modality Reasoning Gap for Speech Large Language Models"*.

## 📖 Project Introduction

> **Note:** The implementation for the Qwen-based models is currently being organized and will be updated in a separate branch of this repository soon. Stay tuned!

Although Speech Large Language Models (Speech LLMs) have achieved notable progress, a substantial **modality reasoning gap** remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap is primarily driven by representational drift across Transformer layers and behavior deviations in long-chain reasoning.

To address this issue, we introduce **TARS**, a reinforcement-learning (RL) framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. By leveraging the text modality as a stable reference, our method effectively steers the speech modality toward the text reasoning trajectory without degrading text-native capabilities.

## ✨ Core Features

- **Asymmetric Reward Design**: Optimizes reasoning in speech using text as a moving reference rather than a static teacher, allowing the speech modality to co-evolve with improving text reasoning.
- **Representation Alignment**: Mitigates representational drift by measuring layer-wise hidden-state similarity between speech- and text-conditioned trajectories.
- **Behavior Alignment**: Evaluates semantic consistency between generated outputs and reference text completions.
- **On-Policy RL (GRPO)**: Integrates Group Relative Policy Optimization to learn from sparse rewards and self-exploration, achieving State-of-the-Art (SoTA) performance on challenging reasoning benchmarks (e.g., MMSU, OBQA) among 7B-scale models.

## 📦 Model Zoo

| Model Name | HuggingFace Link |
| :--- | :--- |
| TARS-Qwen2.5-Omni-7B | [🤗 HuggingFace](https://huggingface.co/yuantuo666/TARS-Qwen2.5-Omni-7B) |
| TARS-Phi-4-MM-7B | *Not provided (Base model is currently proprietary)* |

## 🗺️ Released Code Reference

To facilitate reproducibility, we have organized the codebase as follows.&#x20;

| Component Mentioned in Paper / Review     | Code Location                                                                | Description                                                                                                                                                       |
| :---------------------------------------- | :--------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RL Training Framework (GRPO)**          | [`train_grpo.sh`](./train_grpo.sh)                                           | Core implementation of the GRPO trainer adapted for speech-text multimodal inputs, handling padded masking and advantage calculations.                            |
| **Asymmetric Alignment Rewards**          | [`trainer/grpo_reward_funcs.py`](./trainer/grpo_reward_funcs.py)             | Contains the reward functions for representation alignment (layer-wise cosine similarity) and behavior alignment (semantic consistency via Qwen3-Embedding-0.6B). |
| **Layer Sensitivity Analysis (Fig 2, 3)** | [`scripts/layer_similarity_analysis/`](./scripts/layer_similarity_analysis/) | Scripts to reproduce the layer-wise representation alignment analysis and visualize where representational drift occurs.                                          |
| **Reasoning Benchmarks (MMSU, OBQA)**     | [`scripts/dataset_building/`](./scripts/dataset_building/)                   | Jupyter notebooks used to build and format the MMSU and OBQA datasets for training and evaluation.                                                                |
| **Evaluation & xFinder Integration**      | [`eval/`](./eval/)                                                           | Scripts for evaluating the models on MMSU/OBQA, including `evaluate_vllm.py` and xFinder-based answer extraction (`finder.py`).                                   |
| **Inference Scripts**                     | [`infer/`](./infer/)                                                         | Inference script for Phi-4-MM with `<\|audio_1\|>` template support.                                                                                              |

## ⚙️ Installation

**Requirements:**
- Python $\ge$ 3.12
- PyTorch $\ge$ 2.6.0 (Recommended)
- At least 1$\times$ A100 GPU (80GB) for inference; 8$\times$ A100 GPUs for distributed GRPO/DPO training.

1. Clone the repository:
   ```bash
   git clone https://github.com/amphionteam/TARS.git
   cd TARS
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   source env.sh
   ```

## 📂 Data Preparation & Paths

The foundation MMLU training dataset with synthesized audio has been open-sourced on Hugging Face: [yuantuo666/MMLU\_FULL-full\_train\_hf\_format.merged.v1](https://huggingface.co/datasets/yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1). You may also build this dataset on your own using scripts: [`scripts/dataset_building/mmlu_full_trainingset`](scripts/dataset_building/mmlu_full_trainingset).

### Generate SFT / DPO Datasets

To train with DPO or SFT, you need to generate model responses (e.g., student audio responses and teacher text responses) and construct the preference pairs. We provide a pipeline for this in [`scripts/dataset_building/mmlu_full_train_cot_ds_7b/run.sh`](./scripts/dataset_building/mmlu_full_train_cot_ds_7b/run.sh).

**Step 1: Generate audio-conditioned responses (Student/Rejected)**

```bash
accelerate launch -m scripts.dataset_building.mmlu_full_train_cot_ds_7b.gen_phi_mmlu_dpo_cot \
    --model_path microsoft/Phi-4-multimodal-instruct \
    --mode audio_mode2 \
    --dataset_path yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1

cat output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_rank*.jsonl > output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_aggregated.jsonl
```

**Step 2: Generate text-conditioned responses (Teacher/Chosen)**
Repeat the above process but set `--mode text_mode2`.

**Step 3: Build the final DPO/SFT dataset**

```bash
python -m scripts.dataset_building.mmlu_full_train_cot_ds_7b.build_ds \
    --dataset_path yuantuo666/MMLU_FULL-full_train_hf_format.merged.v1 \
    --chosen_aggregated_jsonl_path output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_text_mode2_cot_outputs_aggregated.jsonl \
    --rejected_aggregated_jsonl_path output/microsoft_Phi-4-multimodal-instruct/phi4_mmlu_audio_mode2_cot_outputs_aggregated.jsonl \
    --save_dataset_path output/MMLU_FULL/full_train_dpo_7b_hf_format.merged.v1
```

## 🚀 Usage

### 1. Training

We provide scripts to run DPO and GRPO training on clusters using DeepSpeed. You can initiate the training using the provided bash scripts:

```bash
# For GRPO Training with TARS
bash train_grpo.sh

# For standard DPO Training
bash train_dpo.sh

# For SFT Training
bash train_sft.sh
```

### 2. Evaluation

**⚠️ Important Pre-requisite for vLLM Evaluation:**
Before you can run inference or evaluation using vLLM on Phi-4-Multimodal, you **must** merge the `speech` LoRA adapter into the base model weights. vLLM does not support stacking multiple LoRAs out-of-the-box in this specific architecture without merging.

Run the merge script first (and only once):

```bash
export SPEECH_LORA_MERGED_MODEL_PATH=output/Phi-4-MM-speech
python scripts/merge_speech_lora.py \
    --model_id microsoft/Phi-4-multimodal-instruct \
    --output_path $SPEECH_LORA_MERGED_MODEL_PATH
```

To evaluate the trained checkpoints on speech reasoning benchmarks (like MMSU and OBQA):

```bash
# Run evaluation on saved ckpts
bash eval.sh /your/local/path/to/save/ckpts

# Run evaluation specifically for ASR/MMLU cascaded baseline
export dataset_path=...
bash eval/eval_cascaded_asr_mmlu.sh

# Run evaluation specifically for ASR/MMLU text only baseline
export dataset_path=...
bash eval/eval_qa_mode0_baseline.sh
```

### 3. Inference

For direct inference or interactive testing, you can refer to the provided Jupyter notebooks or Python scripts:

- [`infer/phi4_audio.py`](./infer/phi4_audio.py): Standalone inference script for the Phi-4-Multimodal based model.

## 🙏 Acknowledgements

Our work is built upon and inspired by several excellent open-source projects:
- **[ms-swift](https://github.com/modelscope/ms-swift)**: Providing the core lightweight infrastructure for fine-tuning.
- **[vLLM](https://github.com/vllm-project/vllm)**: For high-throughput and memory-efficient inference.
- **[Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)**: The robust multimodal foundation model from Microsoft.

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{wang2026closing,
  title={Closing the Modality Reasoning Gap for Speech Large Language Models},
  author={Wang, Chaoren and Lu, Heng and Zhang, Xueyao and Liu, Shujie and Lu, Yan and Li, Jinyu and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2601.05543},
  year={2026}
}
```

