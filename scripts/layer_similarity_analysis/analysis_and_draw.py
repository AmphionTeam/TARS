import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phi-4 Layer-wise Teacher-Forcing Alignment Analysis"
    )

    parser.add_argument(
        "--text_result_json",
        type=str,
        required=True,
        help="JSON file containing Text CoT results (Teacher Trajectory)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="HuggingFace dataset path"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Phi-4-Multimodal model path"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./alignment_analysis",
        help="Root directory for saving results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference Batch Size (suggested to keep small to prevent OOM)",
    )

    return parser.parse_args()


class HiddenStateExtractor:
    def __init__(self, model_path, device):
        print(f"Loading model from {model_path}...")
        self.device = device
        # trust_remote_code=True for Phi-4
        # FIXME: update processor path
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation=(
                "flash_attention_2" if torch.cuda.is_available() else "eager"
            ),
        ).to(device)
        self.model.eval()

        self.assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|assistant|>"
        )
        print(f"Assistant Token ID found: {self.assistant_token_id}")

    @torch.no_grad()
    def get_aligned_hidden_states(self, batch_items):
        """
        Input: batch_items containing 'text_full_str', 'audio_full_str', 'audio_array'
        """

        text_full_strs = [x["text_full_str"] for x in batch_items]

        inputs_text = self.processor(
            text=text_full_strs, return_tensors="pt", padding=True
        ).to(self.device)

        audio_full_strs = [x["audio_full_str"] for x in batch_items]
        audios = [x["audio"] for x in batch_items]

        inputs_audio = self.processor(
            text=audio_full_strs, audios=audios, return_tensors="pt", padding=True
        ).to(self.device)

        inputs_text["input_mode"] = 2
        inputs_audio["input_mode"] = 2

        # --- C. Forward Pass ---

        # 1. Text Forward
        out_text = self.model(**inputs_text, output_hidden_states=True)

        # 2. Audio Forward
        out_audio = self.model(**inputs_audio, output_hidden_states=True)

        aligned_layers_audio = []
        aligned_layers_text = []

        num_layers = len(out_text.hidden_states)
        batch_size = len(batch_items)

        assert batch_size == 1, "only support batch_size == 1 currently."

        slices = []
        for i in range(batch_size):
            # Text Slicing
            t_ids = inputs_text.input_ids[i]
            t_len_total = inputs_text.attention_mask[i].sum().item()

            t_locs = (t_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]
            assert len(t_locs) > 0, "Assistant token not found in Text input!"
            t_start = t_locs[-1].item() + 1

            # Audio Slicing
            a_ids = inputs_audio.input_ids[i]
            a_len_total = inputs_audio.attention_mask[i].sum().item()

            a_locs = (a_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]
            assert len(a_locs) > 0, "Assistant token not found in Audio input!"
            a_start = a_locs[-1].item() + 1

            t_resp_len = t_len_total - t_start
            a_resp_len = a_len_total - a_start
            if t_resp_len != a_resp_len or t_resp_len <= 0 or a_resp_len <= 0:
                slices.append(None)
                print(
                    f"Warning: Misaligned response lengths for sample {i}: Text={t_resp_len}, Audio={a_resp_len}. Skipping. {t_ids.tolist()} vs {a_ids.tolist()}"
                )
                continue  # Skip misaligned samples

            common_len = min(t_resp_len, a_resp_len)
            slices.append(
                {
                    "t_start": t_start,
                    "t_end": t_start + common_len,
                    "a_start": a_start,
                    "a_end": a_start + common_len,
                }
            )

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                continue  # Skip embedding

            l_a_states = []
            l_t_states = []

            t_hidden = out_text.hidden_states[layer_idx]  # (B, S, D)
            a_hidden = out_audio.hidden_states[layer_idx]

            for i in range(batch_size):
                sl = slices[i]
                if sl is None:
                    continue

                t_h = t_hidden[i, sl["t_start"] : sl["t_end"], :].float().cpu()
                a_h = a_hidden[i, sl["a_start"] : sl["a_end"], :].float().cpu()

                if layer_idx == 1 and i == 0:
                    if torch.equal(t_h, a_h):
                        print(
                            "CRITICAL WARNING: Text and Audio hidden states are IDENTICAL. Check inputs!"
                        )

                l_t_states.append(t_h)
                l_a_states.append(a_h)

            aligned_layers_text.append(l_t_states)
            aligned_layers_audio.append(l_a_states)

        return aligned_layers_audio, aligned_layers_text


def prepare_data(args):
    print(f"Loading Text Results (Teacher): {args.text_result_json}")
    with open(args.text_result_json, "r") as f:
        res_text = json.load(f)["results"]

    dict_text = {x["key"]: x for x in res_text}

    print(f"Loading Dataset: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)

    sys_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>[THINKING PROCESS]</think><answer>The answer is [CHOICE].</answer>"
    sys_prompt_fmt = f"<|system|>{sys_prompt}<|end|>"

    aligned_data = []

    valid_keys = sorted(list(dict_text.keys()))
    if args.max_samples:
        valid_keys = valid_keys[: args.max_samples]

    ds_map = {}
    if "key" in ds.column_names:
        for item in ds:
            ds_map[item["key"]] = item
    else:
        print("Warning: 'key' column not found in dataset. Using index matching.")
        ds_map = {k: ds[i] for i, k in enumerate(valid_keys) if i < len(ds)}

    print("Constructing Inputs...")
    for k in valid_keys:
        if k not in ds_map:
            continue

        row_ds = ds_map[k]
        res_t = dict_text[k]

        teacher_output = res_t["llm_output"]

        # 1. Text Full Input (Teacher)
        prompt_text = (
            f"{sys_prompt_fmt}<|user|>{row_ds['prompt_for_tts']}<|end|><|assistant|>"
        )
        text_full_str = f"{prompt_text}{teacher_output}"

        # 2. Audio Full Input (Student)
        prompt_audio = f"{sys_prompt_fmt}<|user|><|audio_1|><|end|><|assistant|>"
        audio_full_str = f"{prompt_audio}{teacher_output}"

        def map_audio(hf_audio):
            return (hf_audio["array"], hf_audio["sampling_rate"])

        aligned_data.append(
            {
                "key": k,
                "text_full_str": text_full_str,
                "audio_full_str": audio_full_str,
                "audio": map_audio(row_ds["audio"]),
            }
        )

    print(f"Prepared {len(aligned_data)} samples.")
    return aligned_data


def analyze_and_plot(layer_stats, output_dir, model_name):
    layers = sorted(layer_stats.keys())
    means = []
    cis = []

    print("Calculating statistics...")
    for l in layers:
        sims = np.array(layer_stats[l])
        sims = sims[~np.isnan(sims)]

        if len(sims) == 0:
            means.append(0)
            cis.append((0, 0))
            continue

        mu = np.mean(sims)
        means.append(mu)

        if len(sims) > 1:
            sem = stats.sem(sims)
            ci = stats.t.interval(0.95, len(sims) - 1, loc=mu, scale=sem)
            cis.append(ci)
        else:
            cis.append((mu, mu))

    cis = np.array(cis).T

    # Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(layers) + 1)

    plt.plot(
        x,
        means,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label=f"{model_name} (Audio-Text Sim)",
    )
    plt.fill_between(x, cis[0], cis[1], color="#1f77b4", alpha=0.2)

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title(f"Teacher-Forcing Layer-wise Alignment\n{model_name}", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0.95, 1.00)
    plt.legend()

    out_png = os.path.join(output_dir, "layer_similarity.png")
    plt.savefig(out_png, dpi=300)
    print(f"Plot saved to {out_png}")

    # Save JSON
    out_json = os.path.join(output_dir, "layer_stats.json")
    with open(out_json, "w") as f:
        # Save explicit data for reproduction
        json.dump(
            {
                "layers": layers,
                "means": [float(x) for x in means],
                "ci_lower": [float(x) for x in cis[0]],
                "ci_upper": [float(x) for x in cis[1]],
            },
            f,
            indent=4,
        )


# --- 5. Main ---


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "--".join(args.model_path.rstrip("/").split("/")[-2:])
    dataset_name = "--".join(args.dataset_path.rstrip("/").split("/")[-2:])

    output_dir = os.path.join(args.save_dir, dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 1. Prepare Data
    data = prepare_data(args)
    if not data:
        print("No data found.")
        return

    # 2. Init Model
    extractor = HiddenStateExtractor(args.model_path, device)

    # 3. Inference Loop
    global_layer_sims = {}

    print("Running Inference...")
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]

        # Returns: [Layer][Batch] -> Tensor
        a_layers_batch, t_layers_batch = extractor.get_aligned_hidden_states(batch)

        num_layers = len(a_layers_batch)

        for l in range(num_layers):  # layer
            if l not in global_layer_sims:
                global_layer_sims[l] = []

            a_list = a_layers_batch[l]
            t_list = t_layers_batch[l]

            for j in range(len(a_list)):  # batch item
                # a_list[j], t_list[j]: (S, D)

                seq_sim = []

                for s in range(
                    len(a_list[j])
                ):  # iterate through sequence dimension and average
                    v_a = a_list[j][s]
                    v_t = t_list[j][s]
                    sim = F.cosine_similarity(v_a, v_t, dim=0).item()
                    seq_sim.append(sim)

                sim = sum(seq_sim) / len(seq_sim)
                global_layer_sims[l].append(sim)

    # 4. Analyze
    analyze_and_plot(global_layer_sims, output_dir, model_name)


if __name__ == "__main__":
    main()
