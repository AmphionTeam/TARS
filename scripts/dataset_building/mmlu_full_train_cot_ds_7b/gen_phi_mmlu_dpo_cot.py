# accelerate launch --main_process_port 36990 -m scripts.dataset_building.mmlu_full_train_cot_ds_7b.gen_phi_mmlu_dpo_cot --model_path microsoft/Phi-4-multimodal-instruct --mode audio_mode2

import accelerate
import os
import json
import argparse
import random
import sys
from tqdm import tqdm
from infer.phi4_audio import Phi4Audio
from datasets import load_from_disk
from huggingface_hub import snapshot_download

accelerator = accelerate.Accelerator()


def save_outputs(output, file_path):
    os.makedirs("outputs", exist_ok=True)
    with open(file_path, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False))
        f.write("\n")


evaluator = None


def xfinder_check(question, options, answer, llm_output):
    # Define input for a single evaluation
    standard_answer_range = []
    for a, o in zip("ABCD", options):
        standard_answer_range.append([a, str(o)])
    standard_answer_range = json.dumps(standard_answer_range)

    # Perform single example evaluation
    result = evaluator.evaluate_single_item(
        question, llm_output, standard_answer_range, "alphabet_option", answer
    )
    return result[-1]


def correct(completion, data):
    from xfinder.eval import Evaluator

    # Initialize the evaluator
    global evaluator
    if not isinstance(evaluator, Evaluator):
        print("initializing xfinder evaluator...")
        evaluator = Evaluator(
            model_name="xFinder-qwen1505",  # Model name
            inference_mode="local",  # Inference mode, 'local' or 'api'
            model_path_or_url=snapshot_download(
                "IAAR-Shanghai/xFinder-qwen1505"
            ),  # Anonymized model path or URL
            device=accelerator.device,  # Device to run the model on
        )

    return xfinder_check(
        data["question"], data["options"], "ABCD"[data["answer_index"]], completion
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument(
        "--mode",
        choices=[
            "text_mode2",
            "text_mode0",
            "audio_mode2",
            "asr_mode2",
            # "texttn_mode2",
        ],
        required=True,
    )
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)
    ds = ds.shuffle(42)

    # for test largest audio
    # ds = ds.map(lambda x: {"prompt_len": len(x)}, input_columns=["prompt_for_tts"], writer_batch_size=200, num_proc=8)
    # ds = ds.sort("prompt_len", reverse=True)

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if rank == 0:
        print(f"args: {args}")
        print("dataset", ds)

    print(f"rank={rank}, world_size={world_size}")

    ds = ds.shard(world_size, rank)

    model_path = args.model_path
    model = Phi4Audio(
        model_path=model_path,
        use_vllm=True,
        load_lora=(False if args.mode == "text_mode0" else True),
        device=accelerator.device,
    )

    flatten_model_path = "_".join(model_path.split("/")[-2:])

    output_file = f"output/{flatten_model_path}/phi4_mmlu_{args.mode}_cot_outputs_rank{rank}.jsonl"
    print(f"output_file: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # check resume.
    resume_from_id_string = None
    if os.path.exists(output_file):
        import subprocess

        print(f"resume from {output_file}")
        # read the last two line of the jsonl file
        output = subprocess.check_output(f"tail -n 1 {output_file}", shell=True).decode(
            "utf-8"
        )
        try:
            output = json.loads(output)
            resume_from_id_string = output["id_string"]
        except:
            pass
        if resume_from_id_string:
            print(f"resume from {resume_from_id_string}")
        else:
            print(
                f"resume from nothing. you need to manually fix the output file. (make sure last line is complete json)"
            )
            sys.exit(1)

    print("resume_from_id_string", resume_from_id_string)

    for data in tqdm(ds, position=rank + 1, total=len(ds)):
        id_string = f"mmlu@{data['question_id']}"

        if resume_from_id_string:
            if id_string != resume_from_id_string:
                continue
            else:
                resume_from_id_string = None
                continue  # clear the resume_from_id_string and continue

        question_text = data["question"]

        q = data["prompt_for_tts"]
        # qtn = data["prompt_for_tts_tn"]
        a = data["options"][data["answer_index"]]
        q_audio = data["audio"]
        q_audio = (q_audio["array"], q_audio["sampling_rate"])

        if not a:
            print("No answer.", data)
            continue

        output = {
            "id_string": id_string,
            "gen_times": 0,
            "question_text": question_text,
            "answer": a,
        }

        chosens = []
        rejecteds = []

        gen_flag = 0

        sys_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>[THINKING PROCESS]</think><answer>The answer is [CHOICE].</answer>"
        # answer_prompt = "\nSolve the multiple-choice question step-by-step."  # cot
        answer_prompt = ""

        while True:
            if len(chosens) + len(rejecteds) >= 100:  # max, 100 times
                break

            # Random temperature between 0.9 and 1.1
            temp = random.uniform(0.9, 1.1)

            if args.mode == "audio_mode2":
                completion = model(
                    ["<|audio_1|>" + answer_prompt],
                    temperature=temp,
                    audios=[q_audio],
                    sys_prompt=sys_prompt,
                )[0]
            elif args.mode == "text_mode2" or args.mode == "text_mode0":
                completion = model(
                    [q + answer_prompt], temperature=temp, sys_prompt=sys_prompt
                )[0]
            # elif args.mode == "texttn_mode2":
            #     completion = model([qtn + answer_prompt], temperature=temp)[0]
            else:
                raise ValueError(f"Invalid mode: {args.mode}")

            if correct(completion, data):
                chosens.append(completion)
            else:
                rejecteds.append(completion)

            if gen_flag == 0:
                if len(chosens) > len(rejecteds):
                    # chosen first, its a easy question
                    gen_flag = -1
                if len(chosens) < len(rejecteds):
                    # rejected first, its a hard question
                    gen_flag = 1

            if len(chosens) * len(rejecteds) > 0:
                break

        output["chosen"] = chosens[0] if len(chosens) > 0 else None
        output["rejected"] = rejecteds[0] if len(rejecteds) > 0 else None
        output["gen_times"] = (len(chosens) + len(rejecteds)) * gen_flag

        save_outputs(output, output_file)
