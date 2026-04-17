#!/usr/bin/env python

# python scripts/merge_small_rank/merge_lora_adapter_hf.py --lora_path /scratch/phi4-dpo/OUTPUT_GRPO/v12_cotrwAccEmbed_dapo_mixm2_e1_dsftcotv1_bs32_20251111_1857/checkpoint-2438

# Phi-4-MM-speech (base_layer, speech, vision).
# - mode0 = Phi-4-Mini + speech-lora
# - mode2 = redudent; vision = not used

# This generates: *-merged-hf (base_layer, speech, vision).
# - mode0 = Phi-4-Mini + speech-lora
# - mode2 = (base: mode0) + adapter; vision = not used
# hf ready w/o modification. compatitive w/ Phi-4-MM official code.

# We also need a scripts to generate *-merged-vllm (base_layer). This will be used when evaluate adaters saved by hf when training based on *-merged-hf.
# - mode0 = (Phi-4-Mini + speech-lora) + adapter.

import argparse
import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model (PEFT) and save merged model."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/mnt/blob/v-chaorwang/Phi-4-MM-speech",
        help="Path to base model (already speech-merged for Phi-4-MM).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (PEFT format).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="dtype used when loading model for merge.",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def build_merged_path(lora_path: str) -> str:
    lora_path = lora_path.rstrip("/")

    parent = os.path.dirname(lora_path)
    name = os.path.basename(lora_path)

    merged_name = name + "-merged-hf"
    merged_path = os.path.join(parent, merged_name)
    return merged_path


def main():
    args = parse_args()

    base_model_path = args.base_model_path
    lora_path = args.lora_path
    dtype = get_dtype(args.dtype)

    if not os.path.isdir(base_model_path):
        raise ValueError(f"Base model path not found: {base_model_path}")

    if not os.path.isdir(lora_path):
        raise ValueError(f"LoRA path not found: {lora_path}")

    save_path = build_merged_path(lora_path)

    print(f"[INFO] Base model: {base_model_path}")
    print(f"[INFO] LoRA adapter: {lora_path}")
    print(f"[INFO] Merged model will be saved to: {save_path}")
    print(f"[INFO] Using dtype: {args.dtype}")

    # 1. Load base model
    print("[STEP 1] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    print("[STEP 1] Base model loaded.")

    print("[STEP 2] Loading LoRA adapter weights into 'speech' adapter...")

    # If an adapter named "speech" already exists, delete it first
    base_model.peft_model.delete_adapter("speech")
    # base_model.peft_model.delete_adapter("vision")

    del base_model.peft_model # remove pointer to avoid duplicate

    # Load the weights from lora_path as an adapter named "speech"
    base_model.load_adapter(
        lora_path,
        adapter_name="speech",
        is_trainable=False,
    )
    # note, from this on, base_model becomes PEFT class

    print("[STEP 2] LoRA adapter loaded into 'speech'.")

    print("[STEP 3] Cast to BF16.")
    base_model = base_model.to(torch.bfloat16)
    base_model.config.speech_lora["dp"] = 0.05
    base_model.config.speech_lora["r"] = 8
    base_model.config.speech_lora["lora_alpha"] = 32

    # 4. Save merged model
    print("[STEP 4] Saving merged model...")
    os.makedirs(save_path, exist_ok=True)
    del base_model._hf_peft_config_loaded # to save full ckpts
    base_model.save_pretrained(
        save_path,
        safe_serialization=True
    )
    print(f"[STEP 4] Merged model saved to: {save_path}")

    # 5. (optional) print size
    print("[STEP 5] Computing parameter statistics...")
    state = base_model.state_dict()
    total_bytes = sum(v.numel() * v.element_size() for v in state.values())
    total_mb = total_bytes / 1024**2
    print(f"[INFO] Total parameters: {sum(v.numel() for v in state.values()):,}")
    print(f"[INFO] Total model size: {total_mb:.2f} MB")

    print("[DONE] All finished.")


if __name__ == "__main__":
    main()
