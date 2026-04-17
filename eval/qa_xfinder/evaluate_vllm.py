# python -m eval.qa_xfinder.evaluate_vllm --model_path microsoft/Phi-4-multimodal-instruct --mode audio_mode2
# accelerate launch --multi_gpu --num_processes 4 --main_process_port 29502 -m eval.qa_xfinder.evaluate_vllm --model_path microsoft/Phi-4-multimodal-instruct --mode audio_mode2

import os
import json
import argparse
import subprocess
import tempfile
import sys
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import DatasetDict, load_from_disk, load_dataset
from infer.phi4_audio import Phi4Audio
from utils.constant import ANSWER_SUFFIX
from tqdm import tqdm
from utils.tool import gather_list, test_gather_list


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.setup_accelerator()
        self.setup_dataset()

    def setup_accelerator(self):
        print("Setting up Accelerator")
        kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=1))]
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.local_rank = self.accelerator.local_process_index
        test_gather_list(self.rank, self.world_size)

    def get_cuda_index(self):
        return f"cuda:{self.local_rank}"

    def setup_model(self):
        # Use Phi4Audio with VLLM for improved inference speed
        self.model = Phi4Audio(
            model_path=self.args.model_path,
            device=self.get_cuda_index(),
            use_vllm=True,
            load_lora=(False if "mode0" in self.args.mode else True),
            system_prompt="You are a helpful assistant.",  # works for omni only.
        )

    def setup_dataset(self):
        if os.path.exists(self.args.dataset_path):
            self.ds = load_from_disk(self.args.dataset_path)
        else:
            self.ds = load_dataset(self.args.dataset_path)["train"]
        if isinstance(self.ds, DatasetDict):
            self.ds = self.ds["test"]
            if self.rank == 0:
                print("Selecting test split in dataset")

        self.ds = self.ds.shuffle(
            42
        )  # shuffle to make the dataset evenly distribute on each card

        if self.rank == 0:
            print("ds", self.ds, self.ds[0])

    def get_result_file_prefix(self):
        return f"results_eval_{self.args.mode}_{self.args.prompt}_vllm_xfinder"

    def get_prompt(self):
        prompt_lib = {
            "none": "",
            "systhink": "",
            "simple": "\nAnswer the multiple choice question.",
            "cot": "\nSolve the multiple-choice question step-by-step.",
            "boxed": "\nSolve the multiple-choice question, write the final selected option letter wrapped in \\boxed{}",  # most of times works.
            "cotboxed": "\nSolve the multiple-choice question step-by-step. At the end, write the final selected option letter wrapped in \\boxed{}",  # will produce CoT.
            "selfasrcot": "\nTranscribe the audio to text, then solve the multiple-choice question step-by-step, finally write the answer like: The answer is Option X. Use <sep> as a separator.",  # will produce CoT rarely...
        }
        return prompt_lib[self.args.prompt]

    def infer(self, output_dir):
        # Process in smaller batches for VLLM
        effective_batch_size = min(
            2, self.args.batch_size
        )  # VLLM works better with smaller batches

        # Process the dataset without using dataloader for VLLM
        all_generated_texts = []

        asr_result = {}
        if self.args.mode == "asr_mode2":
            with open(
                os.path.join(output_dir, "vllm_eval_wer_fixnewlinerepeat.json"), "r"
            ) as f:
                asr_results = json.load(f)["results"]
                for item in asr_results:
                    asr_result[item["key"]] = item
        if self.args.mode == "whisper_mode2":
            with open(
                os.environ.get(
                    "WHISPER_RESULTS_PATH", "/path/to/your/whisper_results.json"
                ),
                "r",
            ) as f:
                asr_results = json.load(f)["results"]
                for item in asr_results:
                    asr_result[item["key"]] = item
        if self.args.mode == "whisper_mode0":
            if "VB_OBQA" in output_dir:
                whisper_path = "workspace/1230_whisper_llm_baselines/VB_OBQA--full_455_hf_format.v0.json"
            elif "VB_MMSU" in output_dir:
                whisper_path = "workspace/1230_whisper_llm_baselines/VB_MMSU--full_3074_hf_format.v0.json"
            elif "MMLU_FULL" in output_dir:
                whisper_path = "workspace/1230_whisper_llm_baselines/MMLU_FULL--full_5k_hf_format.v1.json"
            else:
                print(
                    "cannot load corresponding whisper transcrpts, please run workspace/1230_whisper_llm_baselines/large_transcripts.py to produce the transcripts json file"
                )
                raise
            with open(whisper_path, "r") as f:
                asr_results = json.load(f)["results"]
                for item in asr_results:
                    asr_result[item["key"]] = item

        # Use torch.distributed for data splitting instead of accelerator
        if self.world_size > 1:
            dataset_size = len(self.ds)
            indices_per_rank = dataset_size // self.world_size
            start_idx = self.rank * indices_per_rank
            end_idx = (
                start_idx + indices_per_rank
                if self.rank < self.world_size - 1
                else dataset_size
            )
            rank_dataset = self.ds.select(range(start_idx, end_idx))
        else:
            rank_dataset = self.ds

        answer_prompt = self.get_prompt()

        # same as qwen think.
        sys_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>[THINKING PROCESS]</think><answer>The answer is [CHOICE].</answer>"
            if self.args.prompt == "systhink"
            else ""
        )

        for i in tqdm(
            range(0, len(rank_dataset), effective_batch_size),
            desc=f"Running eval on rank {self.rank} {self.get_cuda_index()}",
            position=self.rank,
        ):
            batch_data = rank_dataset[i : i + effective_batch_size]

            if "MMSU-full_5k" in output_dir:
                # only support audio_mode2, text mode is not support.
                assert self.args.mode == "audio_mode2"

            if self.args.mode == "audio_mode2":
                # Process audio inputs
                inputs = []
                for i in range(len(batch_data["audio"])):
                    if "MMSU-full_5k" in output_dir:
                        # manually adding the prompt and question
                        inputs.append(
                            "<|audio_1|>"
                            + batch_data["prompt_for_tts"][i]
                            + answer_prompt
                        )
                    else:
                        inputs.append("<|audio_1|>" + answer_prompt)
                audios = [
                    (audio["array"], audio["sampling_rate"])
                    for audio in batch_data["audio"]
                ]
                generated_text = self.model(
                    inputs, audios=audios, temperature=0.0, sys_prompt=sys_prompt
                )
            else:
                # Process text inputs
                if self.args.mode in ["asr_mode2"]:
                    # Read the prompt from asr result
                    inputs = [
                        asr_result[key]["generated_text"] + answer_prompt
                        for key in batch_data["key"]
                    ]
                elif self.args.mode in ["whisper_mode2", "whisper_mode0"]:
                    # Read the prompt from asr result
                    inputs = [
                        asr_result[key]["whisper_result"] + answer_prompt
                        for key in batch_data["key"]
                    ]
                elif self.args.mode == "texttn_mode2":
                    inputs = [
                        question + answer_prompt
                        for question in batch_data[
                            "prompt_for_tts_tn"
                        ]  # using TN text normalized version
                    ]
                elif self.args.mode == "selfasritn_mode2":
                    inputs = [
                        question + answer_prompt
                        for question in batch_data["selfasr_result_itn"]
                    ]
                else:
                    if "prompt" in batch_data:
                        inputs = batch_data["prompt"]
                    else:
                        inputs = [
                            question + answer_prompt
                            for question in batch_data["prompt_for_tts"]
                        ]
                generated_text = self.model(
                    inputs, temperature=0.0, sys_prompt=sys_prompt
                )

            # Clean up generated texts
            generated_text = [
                text.replace(ANSWER_SUFFIX, "") for text in generated_text
            ]

            for idx, text in enumerate(generated_text):
                all_generated_texts.append(
                    {"key": batch_data["key"][idx], "text": text}
                )

            if self.rank == 0:
                print(
                    f"===================\n{batch_data['key'][idx]}: {text}\n==================="
                )

        # Gather results from all processes
        all_generated_texts = gather_list(
            all_generated_texts, self.rank, self.world_size
        )

        if self.accelerator.is_main_process:
            all_generated_texts = sorted(all_generated_texts, key=lambda x: x["key"])
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(
                output_dir, f"{self.get_result_file_prefix()}.tmp.json"
            )
            with open(result_file, "w") as f:
                json.dump(all_generated_texts, f, indent=2, ensure_ascii=False)

    def call_xfinder(self, to_be_processed):
        # Write to_be_processed to a temporary JSON file
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tmp_input:
            json.dump(to_be_processed, tmp_input, ensure_ascii=False, indent=2)
            tmp_input_path = tmp_input.name

        # Prepare a temporary file for xfinder.py to write results
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tmp_output:
            tmp_output_path = tmp_output.name

        # Call xfinder.py via command line, passing input and output file paths
        # Assumes xfinder.py is in the same directory as this script
        xfinder_script = os.path.join(os.path.dirname(__file__), "finder.py")
        cmd = [
            "python",
            xfinder_script,
            "--input",
            tmp_input_path,
            "--output",
            tmp_output_path,
        ]

        env = os.environ.copy()
        env["WORLD_SIZE"] = str(0)
        env["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)

        subprocess.run(cmd, check=True, stderr=sys.stderr, stdout=sys.stdout, env=env)

        # Read results from the output file
        with open(tmp_output_path, "r") as f:
            results = json.load(f)

        # Delete the temporary input and output files after reading results
        try:
            os.remove(tmp_input_path)
        except Exception as e:
            print(f"Warning: Failed to delete tmp_input_path {tmp_input_path}: {e}")
        try:
            os.remove(tmp_output_path)
        except Exception as e:
            print(f"Warning: Failed to delete tmp_output_path {tmp_output_path}: {e}")

        return results

    def report(self, output_dir):
        self.accelerator.wait_for_everyone()

        result_file = os.path.join(
            output_dir, f"{self.get_result_file_prefix()}.tmp.json"
        )
        with open(result_file, "r") as f:
            texts = json.load(f)

        visited_keys = set()
        all_generated_texts = []
        for generated_text in texts:
            if generated_text["key"] in visited_keys:
                continue
            visited_keys.add(generated_text["key"])
            all_generated_texts.append(generated_text)

        assert len(all_generated_texts) == len(
            self.ds
        ), f"len(all_generated_texts)={len(all_generated_texts)}, len(self.ds)={len(self.ds)}"

        all_generated_texts = sorted(all_generated_texts, key=lambda x: x["key"])
        with self.accelerator.main_process_first():
            ds = self.ds.sort("key")

        def calculate_accuracy(rs):
            acc = [r["correct"] for r in rs]
            acc = sum(acc) / len(acc)
            return acc

        all_generated_texts = all_generated_texts[self.rank :: self.world_size]
        ds = ds.select(range(self.rank, len(ds), self.world_size))

        to_be_processed = []

        for generated_text, key, question, options, answer_index in zip(
            all_generated_texts,
            ds["key"],
            ds["question"],
            ds["options"],
            ds["answer_index"],
        ):
            assert generated_text["key"] == key

            to_be_processed.append(
                {
                    "key": key,
                    "question": question,
                    "options": options,
                    "answer": "ABCD"[answer_index],
                    "llm_output": generated_text["text"],
                }
            )

        results = self.call_xfinder(to_be_processed)

        # Gather results from all processes
        results = gather_list(results, self.rank, self.world_size)

        if self.accelerator.is_main_process:
            accuracy = calculate_accuracy(results)

            print(f"Overall Accuracy: {accuracy:.2%}")
            outputs = {"accuracy": accuracy, "results": results}

            final_result_file = os.path.join(
                output_dir, f"{self.get_result_file_prefix()}.json"
            )
            with open(final_result_file, "w") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

            print(f"Results saved to {final_result_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "DATASET_PATH", "/path/to/your/dataset/MMLU/hf_format_en_14k_0823"
        ),
        help="Path to the evaluation dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("OUTPUT_BASE_DIR", "/path/to/your/output/phi4-dpo"),
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "text_mode0",
            "whisper_mode0",
            "text_mode2",
            "audio_mode2",
            "asr_mode2",
            "whisper_mode2",
            "texttn_mode2",
            "selfasritn_mode2",
        ],
        default="audio_mode2",
        help="Evaluation mode: text_mode0, text_mode2, or audio_mode2",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=[
            "none",
            "simple",
            "cot",
            "boxed",
            "cotboxed",
            "selfasrcot",
            "systhink",
        ],
        default="cotboxed",
    )
    args = parser.parse_args()

    print(f"args: {args}")

    if os.path.exists(args.model_path):
        output_dir = args.model_path
    else:
        output_dir = os.path.join(args.save_dir, args.model_path)

    flatten_dataset_path = args.dataset_path.replace(
        os.environ.get("BASE_PATH_PREFIX", "/path/to/your/dataset/"), ""
    ).replace("/", "--")
    output_dir = os.path.join(output_dir, flatten_dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    evaluator = Evaluator(args)

    result_file = os.path.join(
        output_dir, f"{evaluator.get_result_file_prefix()}.tmp.json"
    )

    if evaluator.rank == 0:
        print(f"Checking: {result_file}")

    if not os.path.exists(result_file):
        evaluator.setup_model()
        evaluator.infer(output_dir)

    if evaluator.rank == 0:
        print(f"Results saved/found: {result_file}")

    evaluator.report(output_dir)
