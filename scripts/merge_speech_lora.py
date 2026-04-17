import argparse
import torch
from transformers import AutoModelForCausalLM
from peft.tuners.lora.layer import LoraLayer

def main():
    parser = argparse.ArgumentParser(
        description="Merge the built-in speech LoRA of Phi-4-multimodal into the base model weights. "
                    "This is required before using vLLM for inference and evaluation."
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="microsoft/Phi-4-multimodal-instruct", 
        help="Path or HF repo ID of the base model"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the merged model"
    )
    args = parser.parse_args()

    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)

    print("Merging 'speech' adapter into base weights...")
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge(adapter_names=["speech"])
            module.merged_adapters = []

    print("Converting model to bfloat16...")
    model = model.to(torch.bfloat16)

    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path, safe_serialization=True)
    print("Done! The model can now be loaded with vLLM for speech evaluation.")

if __name__ == "__main__":
    main()
