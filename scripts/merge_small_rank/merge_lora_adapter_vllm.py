#!/usr/bin/env python

# python scripts/merge_small_rank/merge_lora_adapter_vllm.py --lora_path /scratch/phi4-dpo/OUTPUT_GRPO/v12_cotrwAccEmbed_dapo_mixm2_e1_dsftcotv1_bs32_20251111_1857/checkpoint-2438

# Phi-4-MM-speech (base_layer, speech, vision).
# - mode0 = Phi-4-Mini + speech-lora
# - mode2 = redudent; vision = not used

# *-merged-hf (base_layer, speech, vision).
# - mode0 = Phi-4-Mini + speech-lora
# - mode2 = (base: mode0) + adapter; vision = not used
# hf ready w/o modification. compatitive w/ Phi-4-MM official code.

# This generates *-merged (base_layer). This will be used when evaluate adaters saved by hf when training based on *-merged-hf.
# - mode0 = (Phi-4-Mini + speech-lora) + adapter.
# vllm ready w/o lora rank.

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

    merged_name = name + "-merged"
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
    if getattr(base_model, "peft_model"):
        print("Detected: peft_model pointer, deleting...")
        del base_model.peft_model
    print("[STEP 1] Base model loaded.")

    # 2. Load LoRA with PEFT
    print("[STEP 2] Loading LoRA adapter with PEFT...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
    )
    print("[STEP 2] LoRA adapter loaded.")

    # 3. Merge & unload PEFT wrappers
    print("[STEP 3] Merging LoRA into base model (merge_and_unload)...")
    # merge_and_unload will write the LoRA weights back into the original weights and return a pure base model.
    merged_model = model.merge_and_unload()
    print("[STEP 3] Merge finished.")

    # 4. Save merged model
    print("[STEP 4] Saving merged model...")
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,
    )
    print(f"[STEP 4] Merged model saved to: {save_path}")

    # 5. (optional) print size
    print("[STEP 5] Computing parameter statistics...")
    state = merged_model.state_dict()
    total_bytes = sum(v.numel() * v.element_size() for v in state.values())
    total_mb = total_bytes / 1024**2
    print(f"[INFO] Total parameters: {sum(v.numel() for v in state.values()):,}")
    print(f"[INFO] Total model size: {total_mb:.2f} MB")

    print("[DONE] All finished. NOTE: The output model can be used in vllm directly (w/o lora loading). But for transformer implementation, further updata is required.")


if __name__ == "__main__":
    main()
