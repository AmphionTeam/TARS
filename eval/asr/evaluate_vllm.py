import sys
import argparse
import os
from accelerate.utils import gather_object
import torch
from tqdm import tqdm
import jiwer
from utils.tool import gather_list
import json
from accelerate import Accelerator
from datasets import load_from_disk
import json
import os
from infer.phi4_audio import Phi4Audio


def get_label(data):
    if "prompt_for_tts" in data:  # for mmlu
        return data["prompt_for_tts"]
    if "text" in data:  # for librispeech
        return data["text"]


def get_key(data):
    if "key" in data:
        return data["key"]
    if "id" in data:
        return data["id"]


@torch.no_grad()
def evaluate(model, ds, save_path=None):
    all_generated_texts = []
    effective_batch_size = 1

    if accelerator.num_processes > 1:
        dataset_size = len(ds)
        indices_per_rank = dataset_size // accelerator.num_processes
        start_idx = accelerator.process_index * indices_per_rank
        end_idx = (
            start_idx + indices_per_rank
            if accelerator.process_index < accelerator.num_processes - 1
            else dataset_size
        )
        rank_dataset = ds.select(range(start_idx, end_idx))
    else:
        rank_dataset = ds

    for i in tqdm(
        range(0, len(rank_dataset), effective_batch_size),
        desc=f"Running eval on rank {rank}",
        position=rank,
    ):
        batch_data = rank_dataset[i : i + effective_batch_size]
        answer_prompt = "<|audio_1|>Transcribe the audio clip into text."

        inputs = [answer_prompt] * len(batch_data["audio"])
        audios = [
            (audio["array"], audio["sampling_rate"]) for audio in batch_data["audio"]
        ]
        generated_text = model(
            inputs,
            audios=audios,
            temperature=0.0,
            no_repeat_ngram_size=10,  # adding no_repeat_ngram_size to avoid repeating the same word
        )

        all_generated_texts.extend(generated_text)
        print(
            "=" * 40 + f"rank={rank}" + "=" * 40 + "\n",
            generated_text[0],
            "\n" + "=" * 80,
        )

    print("Start gather_object...")
    all_generated_texts = gather_list(all_generated_texts, rank, world_size)

    if rank == 0:
        return calculate_scores(all_generated_texts, ds, save_path=save_path)
    return None


def calculate_scores(all_generated_texts, ds, save_path=None):
    results = []
    print("start calculating wer")
    assert len(all_generated_texts) == len(
        ds
    ), f"len(all_generated_texts): {len(all_generated_texts)}, len(ds): {len(ds)}"

    # Calculate WER
    wers = []
    for generated_text, data in zip(all_generated_texts, ds):
        try:
            label = get_label(data)
            # Skip empty or invalid pairs
            if not label or not generated_text:
                print(
                    f"Warning: Skipping empty text pair - Generated: '{generated_text}', Label: '{label}'"
                )
                continue

            transformation = jiwer.Compose(
                [
                    jiwer.ToLowerCase(),
                    jiwer.RemovePunctuation(),
                    jiwer.ExpandCommonEnglishContractions(),
                    jiwer.RemoveWhiteSpace(replace_by_space=True),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.Strip(),
                ]
            )

            processed_reference = transformation(label)
            processed_hypothesis = transformation(generated_text)

            wer = jiwer.wer(processed_reference, processed_hypothesis)
            wer = min(
                wer, 1.0
            )  # max set to 1.0, avoid repeatition affect the performance
            wers.append(wer)

            results.append(
                {
                    "key": get_key(data),
                    "wer": wer,
                    "generated_text": generated_text,
                    "label": label,
                }
            )
        except Exception as e:
            print(f"Warning: Error calculating WER: {e}")
            print(f"Generated text: '{generated_text}'")
            print(f"Label text: '{label}'")
            continue

    if not wers:
        print("Warning: No valid WER scores calculated")
        wer_score = 1.0  # Worst possible WER
    else:
        wer_score = sum(wers) / len(wers)
        print(f"Average WER: {wer_score:.2%}")

    if save_path:
        with open(save_path, "w") as f:
            save_dict = {"score": wer_score, "results": results}
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved to {save_path}")

    return wer_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument(
        "--model_path", type=str, default="microsoft/Phi-4-multimodal-instruct"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "DATASET_PATH", "/path/to/your/dataset/MMLU_FISH/testset_hf_format"
        ),
    )
    args = parser.parse_args()

    # ==================== prepare arguments ====================
    accelerator = Accelerator()
    rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes

    print(f"rank: {rank}, local_rank: {local_rank}")

    if os.path.exists(args.model_path):
        output_dir = args.model_path
    else:
        output_dir = os.path.join(
            os.environ.get("OUTPUT_BASE_DIR", "/path/to/your/output/phi4-dpo"),
            args.model_path,
        )

    flatten_dataset_path = args.dataset_path.replace(
        os.environ.get("BASE_PATH_PREFIX", "/path/to/your/dataset/"), ""
    ).replace("/", "--")
    output_dir = os.path.join(output_dir, flatten_dataset_path)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # ==================== load dataset ====================
    ds = load_from_disk(args.dataset_path)
    ds = ds.shuffle(seed=42)  # Shuffle the dataset for faster evaluation

    if rank == 0:
        print(f"len(ds): {len(ds)}")
        print(f"ds: {ds}")

    # ==================== skip if generated results ====================
    save_path = os.path.join(output_dir, "vllm_eval_wer_fixnewlinerepeat.json")
    is_ok = False
    if rank == 0 and os.path.exists(save_path):
        # trying to read the results.
        try:
            print(f"Checking: {save_path}")
            with open(save_path, "r") as f:
                data = json.load(f)
                wer_score = data["score"]
                results = data["results"]
                generated_texts = [r["generated_text"] for r in results]

                score = calculate_scores(generated_texts, ds, save_path)
                is_ok = True
        except:
            pass

    accelerator.wait_for_everyone()
    is_ok = gather_object([is_ok])
    if any(is_ok):
        print("Detected ok. Exiting.")
        sys.exit(0)

    # ==================== load model ====================
    model = Phi4Audio(
        model_path=args.model_path,
        device=f"cuda:{local_rank}",
        use_vllm=True,
        load_lora=True,
        system_prompt="You are a speech recognition model.",
        transcribe=True,
    )

    # ==================== doing evaluation ====================
    score = evaluate(
        model,
        ds,
        save_path=save_path,
    )
