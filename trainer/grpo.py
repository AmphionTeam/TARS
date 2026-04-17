"""
GRPO on Phi-4-multimodal-instruct

Building env:
. env.sh
huggingface-cli login

Test env commands:
WANDB_MODE=offline RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=$(hostname) MASTER_PORT=50080 python -m trainer.grpo --debug --num_train_epochs 1 --learning_rate 2e-5 --beta 0.0 --warmup_ratio 0.01 --num_generations 2 --dataset_path yuantuo666/MMLU_FULL-full_5k_hf_format.v1 --prompt_modality mixed --prompt_type cot --align_audio_to_text
"""

import os
import torch
import argparse
from pathlib import Path

import wandb
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from peft import LoraConfig, TaskType, get_peft_model
from trainer.grpo_reward_funcs import (
    reward_format,
    reward_len,
    reward_xfinder,
    reward_align_audio_to_text,
    HiddenStateSimRewardORM,
)


from trainer.vllm_patch import *

from trl import GRPOConfig
from trainer.audio_grpo_trainer import AudioGRPOTrainer

from datasets import load_dataset, load_from_disk
from utils.constant import _EVAL_SIZE, _TRAIN_SIZE
from utils.tool import print_trainable_parameters

accelerator = Accelerator()


def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda")

    # Invalid this function to make sure training works, for text mode 0 grpo
    # model.set_lora_adapter("speech")
    # def unset_lora_adapter():
    #     pass
    # model.unset_lora_adapter = unset_lora_adapter

    # FIXME: merge speech lora into base layer.
    from peft.tuners.lora.layer import LoraLayer

    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge(adapter_names=["speech"])
            module.merged_adapters = []  # reset to allow further lora training

    if accelerator.is_main_process:
        print_trainable_parameters(model)

    # FIXME: lora config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,  # 4 times as rank
        lora_dropout=0.05,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ],  # same as Phi-4-MM 's speech lora
    )
    model.add_adapter(
        peft_config, adapter_name="speech"
    )  # using same name to support future calling
    model.set_lora_adapter("speech")

    if accelerator.is_main_process:
        print_trainable_parameters(model)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-multimodal-instruct",
        help="Model name or path to load from",
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="Use Flash Attention"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_dpo/", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=1,
        help="Batch size per GPU (adjust this to fit in GPU memory)",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4.0e-5, help="Learning rate"
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta", type=float, default=0.1, help="beta")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup_ratio")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument(
        "--num_generations", type=int, default=2, help="Number of generation"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get(
            "DATASET_PATH", "output/MMLU_FULL/full_train_dpo_7b_hf_format.merged.v1"
        ),
        help="Path to the dataset",
    )
    parser.add_argument(
        "--prompt_modality",
        type=str,
        default="a",
        choices=["a", "t", "tn", "mixed", "mixed_tn"],
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="none",
        choices=["none", "simple", "cot", "systhink"],
    )
    parser.add_argument(
        "--align_audio_to_text", action="store_true", help="align audio to text"
    )
    parser.add_argument(
        "--reward_funcs",
        nargs="+",
        type=str,
        default=["reward_format", "reward_xfinder", "hsorm.hidden_state_sim"],
        help="List of reward functions to use",
    )
    args = parser.parse_args()

    def slash_path(path):
        to_be_replaced = ["/", ":", "#", "?", "\\", "%"]
        for char in to_be_replaced:
            path = path.replace(char, "_")
        return path

    resume_from_checkpoint = None
    if args.resume:
        # get the latest checkpoint
        try:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            resume_from_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
        except:
            pass

    if accelerator.is_main_process and not args.debug:
        wandb_id = slash_path(args.output_dir)[-40:]
        run = wandb.init(
            project="phi4-grpo",
            name=os.path.basename(args.output_dir),
            id=wandb_id,
        )

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )

        # FIXME: change model name or path
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )
        model.set_lora_adapter("speech")

        # hook stuff like pad_token etc...
        def get_config(self, name):
            try:
                return getattr(self, name)
            except:
                return getattr(self.tokenizer, name)

        processor.__class__.__getattr__ = get_config

    with accelerator.local_main_process_first():
        if os.path.exists(args.dataset_path):
            print(f"Loading dataset from local {args.dataset_path}")
            ds = load_from_disk(args.dataset_path)
        else:
            print(f"Loading dataset from huggingface {args.dataset_path}")
            ds = load_dataset(args.dataset_path)
            if "train" in ds.keys():
                ds = ds["train"]
            elif "test" in ds.keys():
                print(
                    "Warning: test split is used for training. You should use train split instead."
                )
                ds = ds["test"]
        ds = ds.shuffle(42)

        answer_prompt = {
            "none": "",
            "systhink": "",
            "simple": "\nAnswer the multiple choice question.",
            "cot": "\nSolve the multiple-choice question step-by-step.",
        }[args.prompt_type]

        # same as qwen think.
        sys_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>[THINKING PROCESS]</think><answer>The answer is [CHOICE].</answer>"
            if args.prompt_type == "systhink"
            else ""
        )
        if sys_prompt:
            sys_prompt = f"<|system|>{sys_prompt}<|end|>"
        else:
            sys_prompt = ""

        # train: pa
        if args.prompt_modality == "a":
            ds = ds.map(
                lambda x: {
                    "prompt": f"{sys_prompt}<|user|><|audio_1|>{answer_prompt}<|end|><|assistant|>"
                },
                input_columns=["key"],
                writer_batch_size=200,
                num_proc=8,
            )

        # train: pt
        elif args.prompt_modality == "t":
            ds = ds.map(
                lambda x: {
                    "prompt": f"{sys_prompt}<|user|>{x}{answer_prompt}<|end|><|assistant|>"
                },
                input_columns=["prompt_for_tts"],
                writer_batch_size=200,
                num_proc=8,
            )

        # train: ptn
        elif args.prompt_modality == "tn":
            ds = ds.map(
                lambda x: {
                    "prompt": f"{sys_prompt}<|user|>{x}{answer_prompt}<|end|><|assistant|>"
                },
                input_columns=["prompt_for_tts_tn"],
                writer_batch_size=200,
                num_proc=8,
            )

        # === alignment between text and audio ===
        # create interleaved ds
        elif args.prompt_modality == "mixed":
            assert (
                args.align_audio_to_text
            ), "must set --align_audio_to_text for mixed mode"
            ds = ds.map(
                lambda x: {
                    "prompt": f"{sys_prompt}<|user|><|audio_1|>{answer_prompt}<|end|><|assistant|>",
                    "prompt_text": f"{sys_prompt}<|user|>{x}{answer_prompt}<|end|><|assistant|>",
                },
                input_columns=["prompt_for_tts"],
                writer_batch_size=200,
                num_proc=8,
            )

        elif args.prompt_modality == "mixed_tn":
            assert (
                args.align_audio_to_text
            ), "must set --align_audio_to_text for mixed mode"
            ds = ds.map(
                lambda x, y: {
                    "prompt": f"{sys_prompt}<|user|>{x}{answer_prompt}<|end|><|assistant|>",
                    "prompt_text": f"{sys_prompt}<|user|>{y}{answer_prompt}<|end|><|assistant|>",
                },
                input_columns=["prompt_for_tts_tn", "prompt_for_tts"],
                writer_batch_size=200,
                num_proc=8,
            )

        if args.prompt_modality != "a" and args.prompt_modality != "mixed":
            ds = ds.remove_columns(["audio"])

        print("ds", ds)

        # === debug start ===
        eval_size = _EVAL_SIZE
        if args.debug:
            ds = ds.map(
                lambda x, z: {"prompt_len": len(x) + len(z)},
                input_columns=["prompt", "chosen"],
                writer_batch_size=200,
            )
            ds = ds.sort("prompt_len")
            ds = ds.select(
                list(range(len(ds) - 1, len(ds) - 3, -1))
                + list(range(len(ds) - 1, len(ds) - 301, -1))
            )  # from the end to the 300th from the end
            eval_size = 2
        # === debug end ===

        if len(ds) < eval_size + 1:
            eval_size = len(ds) // 10
            print(f"reduce eval size to {eval_size}")

        eval_dataset = ds.select(range(eval_size))
        if _TRAIN_SIZE == 0:
            train_dataset = ds.select(range(eval_size, len(ds)))
        else:
            train_dataset = ds.select(range(eval_size, _TRAIN_SIZE + eval_size))

    num_gpus = accelerator.num_processes
    print(f"training on {num_gpus} GPUs")
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), "Batch size must be divisible by the number of GPUs"
    gradient_accumulation_steps = args.batch_size // (
        num_gpus * args.batch_size_per_gpu
    )

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        bf16=bf16,
        fp16=fp16,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        temperature=1.0,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,  # KL coefficent, default 0.04
        num_generations=args.num_generations,  # default 8
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=4,  # default 8
        bf16_full_eval=True,  # faster the eval...
        num_iterations=1,  # default 1
        log_completions=True,
        report_to="all",
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=50,
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=True,
        loss_type="dapo",
        dataloader_num_workers=4,
        # gradient_checkpointing=True, # use if OOM
        ddp_find_unused_parameters=False,  # for unused SigLIP layers
        max_prompt_length=8192,
        max_completion_length=512,
        max_grad_norm=1.0,  # clip if too large
        num_completions_to_print=4,
        epsilon_high=0.28,  # follow DAPO
        mask_truncated_completions=True,  # follow DAPO
        scale_rewards=False,  # follow Dr. GRPO
    )

    # eval before dpo
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hsorm = HiddenStateSimRewardORM()

    reward_func_map = {
        "reward_format": reward_format,
        "reward_len": reward_len,
        "reward_xfinder": reward_xfinder,
        "reward_align_audio_to_text": reward_align_audio_to_text,
        "hsorm.hidden_state_sim": hsorm.hidden_state_sim,
    }

    selected_reward_funcs = []
    for func_name in args.reward_funcs:
        if func_name in reward_func_map:
            selected_reward_funcs.append(reward_func_map[func_name])
        else:
            raise ValueError(f"Unknown reward function: {func_name}")

    trainer = AudioGRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=selected_reward_funcs,
        align_audio_to_text=args.align_audio_to_text,
        reward_calculation="sum",  # sum or prod
        vllm_importance_sampling_correction=True,  # follow TIS
        separate_advantage_calculation=True,  # True: odd and even's reward will be sparately calculated.
    )

    print(f"resume_from_checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if accelerator.is_main_process:
        trainer.save_model()
        print(f"Saved model to {training_args.output_dir}")
        processor.save_pretrained(training_args.output_dir)
        print(f"Saved processor to {training_args.output_dir}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run.finish()


if __name__ == "__main__":
    main()
