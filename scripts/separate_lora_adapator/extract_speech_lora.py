"""
python scripts/separate_lora_adapator/extract_speech_lora.py --adapter_name speech --model_path /scratch/phi4-dpo/OUTPUT/v6_dpoeq_caudio_mode2_raudio_mode2_beta01_lr1e5_20250701_1459/checkpoint-2550
python scripts/separate_lora_adapator/extract_speech_lora.py --adapter_name llmbackbone --model_path 
"""

import os
import shutil
import argparse

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

# https://huggingface.co/docs/peft/developer_guides/checkpoint
def normalize_lora_key(k, adapter_name):
    return "base_model.model." + k.replace(f".{adapter_name}.weight", ".weight").replace(f".{adapter_name}.weight", ".weight")

def extract_speech_lora(model_path, adapter_name):
    all_lora_data = {}

    # check if lora already
    if os.path.exists(model_path + "/adapter_model.safetensors"):
        os.symlink(".", model_path + "/speech-lora") # create a symlink to the current directory
        return model_path + "/speech-lora"

    for i in range(1, 4):
        data = load_file(model_path + f"/model-0000{i}-of-00003.safetensors")

        # filter out the speech lora weights
        lora_data = {normalize_lora_key(k, adapter_name): v for k, v in data.items() if "lora" in k and adapter_name in k}
        all_lora_data.update(lora_data)

    lora_path = os.path.join(model_path, f"{adapter_name}-lora")

    os.makedirs(lora_path, exist_ok=True)
    save_file(all_lora_data, os.path.join(lora_path, "adapter_model.safetensors"))

    # copy the adapter_config.json to the model_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(script_dir, f"{adapter_name}_adapter_config.json"), os.path.join(lora_path, "adapter_config.json"))
    
    return lora_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_name", type=str, required=True, choices=["speech", "llmbackbone"])
    args = parser.parse_args()

    model_path = args.model_path

    if model_path == "microsoft/Phi-4-multimodal-instruct":
        print("No need to extract lora weights.")
        exit(0)

    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    lora_path = extract_speech_lora(model_path, args.adapter_name)
    print(f"Extracted Lora path: {lora_path}")
    

   