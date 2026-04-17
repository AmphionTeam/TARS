import os
import shutil
from trl.trainer.dpo_trainer import DPOTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict, List, Optional
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import pad
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
import time
import functools
from contextlib import contextmanager, nullcontext
import torch.amp as amp
from trl.trainer.dpo_config import FDivergenceConstants, FDivergenceType


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function `{func.__name__}` execution time: {elapsed:.2f} seconds")
        return result

    return wrapper


@dataclass
class DataCollatorForAudioPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForPreference
    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]}
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],xr
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def __init__(
        self,
        pad_token_id,
        processing_class,
        max_prompt_length,
        max_completion_length,
        *args,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.processing_class = processing_class
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def torch_call(self, raw_examples):
        # Step 1: Apply tokenize_row to each example
        examples = []
        for example in raw_examples:
            processed = self.tokenize_row(
                features=example,
                processing_class=self.processing_class,
                max_prompt_length=self.max_prompt_length,
                max_completion_length=self.max_completion_length,
                add_special_tokens=False,
            )
            examples.append(processed)

        # Convert to tensor
        prompt_input_ids = [(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        chosen_input_ids = [(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [
            torch.ones_like(input_ids) for input_ids in chosen_input_ids
        ]
        rejected_input_ids = [(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [
            torch.ones_like(input_ids) for input_ids in rejected_input_ids
        ]
        if "input_audio_embeds" in examples[0]:
            input_audio_embeds = [
                (example["input_audio_embeds"]) for example in examples
            ]
        if "audio_attention_mask" in examples[0]:
            audio_attention_mask = [
                torch.tensor(example["audio_attention_mask"]) for example in examples
            ]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor(
                [example["ref_chosen_logps"] for example in examples]
            )
            ref_rejected_logps = torch.tensor(
                [example["ref_rejected_logps"] for example in examples]
            )

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(
            chosen_input_ids, padding_value=self.pad_token_id
        )
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(
            rejected_input_ids, padding_value=self.pad_token_id
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0
        )
        if "input_audio_embeds" in examples[0]:
            output["input_audio_embeds"] = pad(input_audio_embeds, padding_value=0.0)
        if "audio_embed_sizes" in examples[0]:
            output["audio_embed_sizes"] = torch.tensor(
                [example["audio_embed_sizes"] for example in examples]
            )
        if "audio_attention_mask" in examples[0]:
            output["audio_attention_mask"] = pad(audio_attention_mask, padding_value=0)
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ):
        """
        Same as `tokenize_row` but for audio models. This method processes audio inputs instead of images.

        Args:
            features (`dict`): Dictionary containing:
                - "audio": Audio inputs
                - "prompt": Text prompt
                - "chosen": Chosen completion
                - "rejected": Rejected completion
            processing_class: The processor class that handles both audio and text
            max_prompt_length (`int` or `None`): Maximum length for prompt sequence
            max_completion_length (`int` or `None`): Maximum length for completion sequences
            add_special_tokens (`bool`): Whether to add special tokens

        Returns:
            `dict`: Processed features including audio values and tokenized text
        """
        processor, tokenizer = (
            processing_class,
            processing_class.tokenizer,
        )  # the processing class is a processor
        processed_features = processor(
            text=features["prompt"],
            audios=(
                [(features["audio"]["array"], features["audio"]["sampling_rate"])]
                if "audio" in features
                else None
            ),
        )

        prompt_input_ids = processed_features["input_ids"][0]
        if "audio" in features:
            input_audio_embeds = processed_features["input_audio_embeds"][0]
            audio_embed_sizes = processed_features["audio_embed_sizes"][0]

            assert len(torch.nonzero(prompt_input_ids == 200011).tolist()) == int(
                audio_embed_sizes
            ), f"audio_embed_sizes: {audio_embed_sizes}, prompt_input_ids: {prompt_input_ids}. not match!!! features: {features}"

        else:
            input_audio_embeds = None
            audio_embed_sizes = None

        chosen_input_ids = tokenizer(features["chosen"], return_tensors="pt").input_ids[
            0
        ]
        rejected_input_ids = tokenizer(
            features["rejected"], return_tensors="pt"
        ).input_ids[0]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = torch.cat(
                    [torch.tensor([tokenizer.bos_token_id]), prompt_input_ids], dim=0
                )
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = torch.cat(
                    [prompt_input_ids, torch.tensor([tokenizer.eos_token_id])], dim=0
                )
            chosen_input_ids = torch.cat(
                [chosen_input_ids, torch.tensor([tokenizer.eos_token_id])], dim=0
            )
            rejected_input_ids = torch.cat(
                [rejected_input_ids, torch.tensor([tokenizer.eos_token_id])], dim=0
            )

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]

            if len(prompt_input_ids) > max_prompt_length:
                print(
                    f"WARN: prompt length exceeds max_prompt_length, len(prompt)={len(prompt_input_ids)}, max_prompt_length={max_prompt_length}"
                )

        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

            if len(chosen_input_ids) > max_completion_length:
                print(
                    f"WARN: chosen length exceeds max_completion_length, len(chosen)={len(chosen_input_ids)}, max_completion_length={max_completion_length}"
                )
            if len(rejected_input_ids) > max_completion_length:
                print(
                    f"WARN: rejected length exceeds max_completion_length, len(rejected)={len(rejected_input_ids)}, max_completion_length={max_completion_length}"
                )

        output = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if input_audio_embeds is not None and audio_embed_sizes is not None:
            output["input_audio_embeds"] = input_audio_embeds
            output["audio_embed_sizes"] = audio_embed_sizes

        if (
            "audio_attention_mask" in processed_features
            and processed_features["audio_attention_mask"] is not None
        ):
            output["audio_attention_mask"] = processed_features["audio_attention_mask"][
                0
            ]

        return output


class AudioDPOTrainer(DPOTrainer):
    @timeit
    def __init__(self, *args, **kwargs):
        """Initialize the AudioDPOTrainer.

        Args:
            *args: Positional arguments to pass to the parent class.
            **kwargs: Keyword arguments to pass to the parent class.
                Must include 'processing_class' for handling audio and text inputs.
        """
        if "prepare_dataset" in kwargs:
            self.prepare_dataset = kwargs["prepare_dataset"]
            del kwargs["prepare_dataset"]
        else:
            self.prepare_dataset = False

        super().__init__(*args, **kwargs)
        self.data_collator = DataCollatorForAudioPreference(
            processing_class=kwargs["processing_class"],
            max_prompt_length=self.max_prompt_length,
            max_completion_length=self.max_completion_length,
            pad_token_id=self.padding_value,
        )

    @timeit
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name: str,
    ):
        # Build the kwargs for the `map` function
        map_kwargs = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )

            # Skip tokenizing here

            # filter the long questions prompts
            # around 10s audio questions, question length <= 160
            # VA(Train): 469848 => 448028
            # print(f"filter the long questions prompts, VA(Train): {len(dataset)}")
            # dataset = dataset.filter(lambda x: len(x) <= 160, input_columns=["question"])
            # print(f"filter the long questions prompts, VA(Train) filtered: {len(dataset)}")

        if self.prepare_dataset:
            print(f"prepare_dataset done: size={len(dataset)}")
            exit(0)

        return dataset

    @staticmethod
    def process_row(
        features: Dict,
        processing_class,
        max_prompt_length: Optional[int],
        max_completion_length: Optional[int],
        add_special_tokens: bool,
    ) -> Dict:
        raise NotImplementedError("process_row is not implemented for audio models")

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int, **kwargs
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences. Modified to handle audio inputs instead of image inputs.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:
                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"input_audio_embeds"`: Tensor for audio values.
                - `"audio_embed_sizes"`: Tensor for audio attention masks.
                - `"audio_attention_mask"`: Tensor for audio lengths.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences.

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:
                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"input_audio_embeds"`: Concatenated audio values.
                - `"audio_attention_mask"`: Concatenated audio attention masks.
                - `"audio_embed_sizes"`: Concatenated audio lengths.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        )
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "input_audio_embeds" in batch:
            output["input_audio_embeds"] = torch.cat(
                [batch["input_audio_embeds"], batch["input_audio_embeds"]], dim=0
            )
        if "audio_attention_mask" in batch:
            output["audio_attention_mask"] = torch.cat(
                [batch["audio_attention_mask"], batch["audio_attention_mask"]], dim=0
            )
        if "audio_embed_sizes" in batch:
            output["audio_embed_sizes"] = torch.cat(
                [batch["audio_embed_sizes"], batch["audio_embed_sizes"]], dim=0
            )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(
            batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
        )
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
                pad_to_length(
                    batch["rejected_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_attention_mask"], max_completion_length, pad_value=0
                ),
                pad_to_length(
                    batch["rejected_attention_mask"], max_completion_length, pad_value=0
                ),
            ),
        )

        return output

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        model_output=None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (
            not self.reference_free
        ) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (
            not self.reference_free
        ) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if (
                self.f_divergence_params
                and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY
                in self.f_divergence_params
            ):
                alpha_coef = float(
                    self.f_divergence_params[
                        FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY
                    ]
                )
            logits = (
                cap_exp(rejected_logratios * -alpha_coef)
                - cap_exp(chosen_logratios * -alpha_coef)
            ) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor(
                    [0], dtype=logratios.dtype, device=logratios.device
                )
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)

        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (
                F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing)
            )

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid(
                (self.beta * chosen_logratios) - delta
            ) - F.logsigmoid(-(self.beta * rejected_logratios - delta))

        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(
                self.beta * chosen_logratios
            )  # Increase chosen likelihood
            losses_rejected = F.sigmoid(
                self.beta * rejected_logratios
            )  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(
                self.beta * (chosen_logratios - rejected_logratios)
            )
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = (
                logistic_component * (1 - log_ratio_modulation)
                + exp_component * log_ratio_modulation
            )

        elif self.loss_type == "sft":
            # SFT loss is the negative log likelihood loss on chosen responses
            # This acts as the generation loss component in MPO
            sft_loss = model_output["nll_loss"]
            # Create losses tensor with same shape as other losses (per-sample)
            batch_size = chosen_logps.shape[0]
            losses = sft_loss.expand(batch_size)
            # For SFT, we don't have preference rewards, so use zeros
            chosen_rewards = torch.zeros_like(chosen_logps)
            rejected_rewards = torch.zeros_like(rejected_logps)

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = (
            self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        )
        rejected_rewards = (
            self.beta
            * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        **kwargs,
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        Modified to handle audio inputs instead of image inputs.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.padding_value
        )

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the audio values and attention masks for audio models
        if "input_audio_embeds" in concatenated_batch:
            model_kwargs["input_audio_embeds"] = concatenated_batch[
                "input_audio_embeds"
            ]
        if "audio_attention_mask" in concatenated_batch:
            model_kwargs["audio_attention_mask"] = concatenated_batch[
                "audio_attention_mask"
            ]
        if "audio_embed_sizes" in concatenated_batch:
            model_kwargs["audio_embed_sizes"] = concatenated_batch["audio_embed_sizes"]
        model_kwargs["input_mode"] = 2  # speech mode

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:  # False
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(
                attention_mask, input_ids, loss_mask
            )

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (
                    loss_mask.shape[1] - first_compute_index
                ).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = (
                    attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                )
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            if "input_audio_embeds" not in model_kwargs:
                model_kwargs["input_audio_embeds"] = [1] * num_examples
                model_kwargs["input_audio_embeds"] = torch.tensor(
                    model_kwargs["input_audio_embeds"]
                )

            try:
                outputs = model(input_ids, **model_kwargs)
                logits = outputs.logits
            except Exception as e:
                print(f"Error in concatenated_forward: {e}")
                print(f"input_ids: {input_ids.shape} {input_ids}")
                print(f"attention_mask: {attention_mask.shape} {attention_mask}")
                print(f"loss_mask: {loss_mask.shape} {loss_mask}")
                if "input_audio_embeds" in model_kwargs:
                    print(
                        f"input_audio_embeds: {model_kwargs['input_audio_embeds'].shape} {model_kwargs['input_audio_embeds']}"
                    )
                if "audio_embed_sizes" in model_kwargs:
                    print(
                        f"audio_embed_sizes: {model_kwargs['audio_embed_sizes'].shape} {model_kwargs['audio_embed_sizes']}"
                    )
                if "audio_attention_mask" in model_kwargs:
                    print(
                        f"audio_attention_mask: {model_kwargs['audio_attention_mask'].shape} {model_kwargs['audio_attention_mask']}"
                    )
                raise e

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = (
            0  # dummy token; we'll ignore the losses on these tokens later
        )
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size,
                seq_len,
                device=outputs.logits.device,
                dtype=outputs.logits.dtype,
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        # Calculate response lengths for chosen and rejected responses
        chosen_lengths = loss_mask[:num_examples].sum(dim=1).float()
        rejected_lengths = loss_mask[num_examples:].sum(dim=1).float()

        output["chosen_lengths"] = chosen_lengths
        output["rejected_lengths"] = rejected_lengths

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(
                    2 * logprobs, dim=-1
                )  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(
                    -1
                ) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )

        if self.args.rpo_alpha is not None or self.loss_type == "sft":
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][
                loss_mask[0, split_idx:]
            ].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][
                loss_mask[num_examples:]
            ].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval="train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            model_output,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = (
                losses + self.args.rpo_alpha * model_output["nll_loss"]
            )  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}lengths/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_lengths"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}lengths/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_lengths"])
            .detach()
            .mean()
            .item()
        )
        if self.args.rpo_alpha is not None or self.loss_type == "sft":
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"])
                .detach()
                .mean()
                .item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )

        return losses.mean(), metrics

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime, output_dir=output_dir
        )

        # Make a backup of the current checkpoint if needed.
        # Parse checkpoint name to get step number
        if len(checkpoints_sorted) > 0:
            checkpoint_filename = os.path.basename(checkpoints_sorted[-1])
            checkpoint_step = int(checkpoint_filename.split("-")[-1])

            # Save backup if step is divisible by 5000
            if checkpoint_step % 5000 == 0:
                try:
                    backup_dir = os.path.join(output_dir, "backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, checkpoint_filename)
                    shutil.copytree(checkpoints_sorted[-1], backup_path)
                    print(f"Saving backup state to {backup_path}...")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Failed to save backup state: {e}")

        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - save_total_limit
        )
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            print(
                f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
            )
            shutil.rmtree(checkpoint, ignore_errors=True)

    def generate_from_model_and_ref(
        self, model, batch: dict[str, torch.LongTensor]
    ) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        device_type = "cuda"
        generate_context_manager = (
            amp.autocast(device_type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

        # breakpoint()

        # print(f"batch: {batch}")
        model_kwargs = {}

        # Add the audio values and attention masks for audio models
        if "input_audio_embeds" in batch:
            model_kwargs["input_audio_embeds"] = batch["input_audio_embeds"]
        if "audio_attention_mask" in batch:
            model_kwargs["audio_attention_mask"] = batch["audio_attention_mask"]
        if "audio_embed_sizes" in batch:
            model_kwargs["audio_embed_sizes"] = batch["audio_embed_sizes"]

        with generate_context_manager:

            print("Generating Policy Output...")
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,  # 4096
                do_sample=True,
                pad_token_id=self.padding_value,  # 199999
                input_mode=2,  # speech mode
                **model_kwargs,
            )  # B x T

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.padding_value,
                        )
                else:
                    print("Generating Ref Output...")
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.padding_value,
                        input_mode=2,  # speech mode
                        **model_kwargs,
                    )  # B x T

        policy_output = pad_to_length(
            policy_output, self.max_length, self.padding_value
        )
        policy_output_decoded = self.processing_class.batch_decode(
            policy_output, skip_special_tokens=True
        )

        ref_output = pad_to_length(ref_output, self.max_length, self.padding_value)
        ref_output_decoded = self.processing_class.batch_decode(
            ref_output, skip_special_tokens=True
        )

        return policy_output_decoded, ref_output_decoded
