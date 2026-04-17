import warnings
from trl import GRPOTrainer
from contextlib import nullcontext
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from accelerate.utils import gather, gather_object
from trl.trainer.utils import selective_log_softmax, pad
from trl.extras.profiling import profiling_context, profiling_decorator
from torch import nn, mean
from trl.trainer.grpo_trainer import nanstd, nanmin, nanmax
import os
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature


# replace with new implementation supporting DAPO: https://github.com/huggingface/trl/commit/2324245cad81047ca6dec2caf26fade9e9ec2e27
def split_tensor_dict(tensor_dict, num_chunks):
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for i in range(num_chunks):
        chunk_dict = {}
        for key, tensor in tensor_dict.items():
            if tensor is not None and tensor.ndim > 0:
                chunk_dict[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            elif tensor is not None and tensor.ndim == 0:
                chunk_dict[key] = tensor
            else:
                chunk_dict[key] = None
        chunks.append(chunk_dict)
    return chunks


def shuffle_sequence_dict(seq_dict):
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return v
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


class AudioGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        args,
        align_audio_to_text=False,
        reward_calculation="sum",  # "sum" or "prod"
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_cap=2.0,
        separate_advantage_calculation=True,
        **kwargs,
    ):
        args.use_vllm = False
        super().__init__(model=model, args=args, **kwargs)

        if self.use_liger_loss:
            raise Exception("Liger loss not supported in AudioGRPOTrainer")

        # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
        # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
        # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
        # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
        # that behavior without rewriting `training_step`.
        self.compute_loss_func = "non-None value to disable scaling"
        self.use_vllm = True  # manually init the vllm

        # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
        # the same number of ranks
        if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
            raise ValueError(
                f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                f"({self.accelerator.num_processes}) evenly."
            )

        if self.vllm_tensor_parallel_size > 1:
            # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
            # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
            self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [
                    list(
                        range(
                            i * self.vllm_tensor_parallel_size,
                            (i + 1) * self.vllm_tensor_parallel_size,
                        )
                    )
                    for i in range(
                        self.accelerator.num_processes // self.vllm_tensor_parallel_size
                    )
                ]
            )

        self.llm = LLM(
            model=model.name_or_path,
            tokenizer=model.name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            max_num_seqs=self.args.per_device_train_batch_size
            * self.vllm_tensor_parallel_size
            * self.args.gradient_accumulation_steps,
            max_model_len=self.max_prompt_length + self.max_completion_length,
            distributed_executor_backend="external_launcher",
            # Feed identical seed for tp groups to ensure sampling results are the same across workers
            seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
            enable_lora=True,
            max_lora_rank=320,
            limit_mm_per_prompt={"audio": 1},
        )

        # setting up lora request
        model_path = model.name_or_path
        if not os.path.exists(model_path):
            model_path = snapshot_download(model_path)

        if "-merged-hf" in model_path:
            # BUG: This used to solve mis-configuration of adapter_config.json
            lora_path = model_path.split("-merged-hf")[
                0
            ]  # using unmerged lora adapters (vllm will load only the base_layer part in checkpoint-*-merged-hf, loading the lora on-the-fly)
        else:
            lora_path = os.path.join(
                model_path, "speech-lora"
            )  # HF standard format should have speech-lora subfolder

        assert os.path.exists(
            os.path.join(lora_path, "adapter_model.safetensors")
        ), f"Lora path {lora_path} does not exist"
        self.lora = [LoRARequest("speech", 1, lora_path)]

        # vLLM specific sampling arguments
        self.guided_decoding_regex = args.vllm_guided_decoding_regex

        self._last_loaded_step = (
            -1
        )  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Feature dimension of audio embeddings, set according to your model
        self.audio_dim_in = 80
        # Placeholder length for samples without audio
        self.placeholder_audio_len = 500

        self.align_audio_to_text = align_audio_to_text
        self.reward_calculation = reward_calculation
        self.vllm_importance_sampling_correction = vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = vllm_importance_sampling_cap
        self.separate_advantage_calculation = separate_advantage_calculation

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
        # merging adapters in a sharded manner is not supported.
        # TODO: does this work with FSDP?
        with gather_if_zero3(list(self.model.parameters())):
            from peft.tuners.lora.layer import LoraLayer

            for module in self.model.modules():
                if isinstance(module, LoraLayer):
                    if not module.merged:
                        module.merge(adapter_names=["speech"])
                        # module.merge(adapter_names=["llmbackbone"])
            # self.model.merge_adapter()

            # Update vLLM weights while parameters are gathered
            if (
                self.is_fsdp_enabled
            ):  # note if using FSDP, gather_if_zero3 is nullcontext
                # Update vLLM weights while parameters are gathered
                # For PEFT with FSDP we need to use the memory efficient post-order traversal
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # DeepSpeed ZeRO-3 with PEFT
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.replace(
                        "base_layer.", "base_layer.base_layer."
                    )  # duplicate .base_layer as this is in vllm too. and will be removed once by converter...
                    # if self.model.prefix in name:
                    #     continue
                    if "image_embed" in name:
                        continue
                    if "audio_embed" in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights([(name, param.data)])
            # Unmerge adapters while parameters are still gathered
            # self.model.unmerge_adapter()

            for module in self.model.modules():
                if isinstance(module, LoraLayer):
                    if module.merged:
                        module.unmerge()
            # Parameters will automatically be repartitioned when exiting the context

        # Reset cache on vLLM
        self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(self, generation_batch):
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = shuffle_sequence_dict(generation_batch)
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        input_audio_embeds,
        audio_embed_sizes,
        batch_size=None,
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            input_audio_embeds_batch = input_audio_embeds[i : i + batch_size]
            audio_embed_sizes_batch = audio_embed_sizes[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                num_logits_to_keep=logits_to_keep + 1,
                input_audio_embeds=input_audio_embeds_batch,
                audio_embed_sizes=audio_embed_sizes_batch,
                input_mode=2,  # audio
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_audio_embeds, audio_embed_sizes = (
            inputs["input_audio_embeds"],
            inputs["audio_embed_sizes"],
        )

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # TODO: add entropies to support logging & potential GSPO use
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            input_audio_embeds,
            audio_embed_sizes,
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        input_audio_embeds,
                        audio_embed_sizes,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            input_audio_embeds,
                            audio_embed_sizes,
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using vLLM, we always compute old_per_token_logps for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )

        self._metrics[mode]["audio_input/audio_embed_sizes_mean"].append(
            mean(audio_embed_sizes.float()).item()
        )
        return loss

    def collate_fn(self, processed_items):
        """
        Manually pads a list of processed items to form a batch.
        This function correctly handles batches with a mix of audio and non-audio samples.
        """
        # 1. Text padding (same as before)
        input_ids = [item["input_ids"] for item in processed_items]
        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processing_class.tokenizer.pad_token_id,
        )

        attention_mask = [item["attention_mask"] for item in processed_items]
        attention_mask_padded = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
        }

        # Initialize lists to collect audio-related data
        input_audio_embeds_list = []
        audio_embed_sizes_list = []
        audio_lengths = []  # Used to generate attention mask

        for item in processed_items:
            # Check if current sample has audio
            if "input_audio_embeds" in item and item["input_audio_embeds"] is not None:
                # --- Case 1: Has audio ---
                audio_embed = torch.tensor(item["input_audio_embeds"])
                input_audio_embeds_list.append(audio_embed)

                # Record original length for attention mask
                audio_lengths.append(audio_embed.shape[0])

                # Record audio_embed_sizes
                # Assume item['audio_embed_sizes'] is a number or single-element list
                size_tensor = torch.tensor(item["audio_embed_sizes"]).long()
                audio_embed_sizes_list.append(size_tensor)

            else:
                # --- Case 2: No audio, create placeholder ---
                placeholder_embed = torch.zeros(
                    self.placeholder_audio_len, self.audio_dim_in
                )
                input_audio_embeds_list.append(placeholder_embed)

                # Key: For placeholder, effective length is 0, so attention mask will be all False
                audio_lengths.append(0)

                # Corresponding audio_embed_sizes should also be 0
                audio_embed_sizes_list.append(torch.tensor(0).long())

        # 3. Pad and stack audio data
        # Pad audio embeddings to have consistent length within batch
        padded_audio_embeds = pad_sequence(
            input_audio_embeds_list, batch_first=True, padding_value=0.0
        )
        batch["input_audio_embeds"] = padded_audio_embeds

        # Stack audio_embed_sizes
        stacked_audio_embed_sizes = torch.stack(audio_embed_sizes_list, dim=0)
        batch["audio_embed_sizes"] = stacked_audio_embed_sizes

        # 4. Create audio attention mask
        # max_len is the maximum padded length of all audios (including placeholders) in the batch
        max_len = padded_audio_embeds.shape[1]

        # Convert audio_lengths to tensor
        audio_lengths_tensor = torch.tensor(audio_lengths, dtype=torch.long)

        # Create a range tensor [0, 1, ..., max_len-1]
        arange_tensor = torch.arange(max_len).unsqueeze(0)

        # Generate attention mask via broadcast comparison
        # For real audio, generates [True, True, ..., False, False]
        # For placeholder (length 0), generates [False, False, ...]
        audio_attention_mask = arange_tensor < audio_lengths_tensor.unsqueeze(1)
        batch["audio_attention_mask"] = audio_attention_mask.long()

        return BatchFeature(batch)

    def _get_hidden_states_for_reward(
        self,
        model: nn.Module,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_audio_embeds: torch.Tensor,
        audio_embed_sizes: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ):
        """
        Performs a no-grad forward pass in chunks to get hidden states for the reward function.
        This version is compatible with multi-modal inputs (e.g., audio) and reuses pre-computed tensors
        to avoid re-tokenization.

        Note: The hidden states include the hidden state of the last prompt token to help compute reward
        on the first completion token. The shape will be List[sample_idx] -> List[layer_idx] -> Tensor[completion_length + 1, hidden_size].
        """
        mode = "train" if model.training else "eval"
        model.eval()

        all_samples_all_layers_states = []
        # Determine chunk size based on training arguments to control memory usage
        chunk_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # Iterate through the batch in chunks to prevent OOM, handling all tensors
            for i in range(0, prompt_completion_ids.size(0), chunk_size):
                # Create chunks from all the provided tensors
                chunk_input_ids = prompt_completion_ids[i : i + chunk_size]
                chunk_attention_mask = attention_mask[i : i + chunk_size]
                chunk_input_audio_embeds = input_audio_embeds[i : i + chunk_size]
                chunk_audio_embed_sizes = audio_embed_sizes[i : i + chunk_size]
                chunk_prompt_lengths = prompt_lengths[i : i + chunk_size]

                # Perform forward pass on the chunk with all required multi-modal arguments
                outputs = model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    input_audio_embeds=chunk_input_audio_embeds,
                    audio_embed_sizes=chunk_audio_embed_sizes,
                    input_mode=2,  # Assuming input_mode=2 is standard for audio, as per reference
                    output_hidden_states=True,
                )

                # outputs.hidden_states is a tuple of tensors, one for each layer.
                # Shape of each element: (chunk_batch_size, sequence_length, hidden_size)
                all_layers_hidden_states = outputs.hidden_states

                # Prepare a list to hold the structured hidden states for this chunk
                chunk_num_samples = chunk_input_ids.size(0)
                chunk_all_samples_states = [[] for _ in range(chunk_num_samples)]

                # Iterate through hidden states of each layer
                for layer_states in all_layers_hidden_states:
                    # Extract relevant hidden states for each sample in the chunk
                    for j in range(chunk_num_samples):
                        # Get true prompt length of current sample
                        current_prompt_len = chunk_prompt_lengths[j].item()

                        # Start index: position of last prompt token (used to predict first completion token)
                        start_index = current_prompt_len - 1

                        # End index: total actual length of prompt + completion
                        end_index = chunk_attention_mask[j].sum()
                        # chunk_input_ids[0][end_index] = <|eos|> / padding token
                        # relevant_states[-1] => <|end|> (NOTE: should use -2 to represent entire sentence)

                        # Extract all hidden states from last prompt token to last completion token
                        relevant_states = layer_states[j, start_index:end_index, :]

                        chunk_all_samples_states[j].append(relevant_states)

                # Extend the main list with the results from this chunk
                all_samples_all_layers_states.extend(chunk_all_samples_states)

        if mode == "train":
            model.train()  # Switch model back to training mode

        return all_samples_all_layers_states

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self.align_audio_to_text:
            # here we replace the odd prompts with text-only prompts (prompt_text)
            prompts = [
                x["prompt"] if i % 2 == 0 else x["prompt_text"]
                for i, x in enumerate(inputs)
            ]
            prompts_text = prompts
            prompts_audio = [
                (
                    (x["audio"]["array"], x["audio"]["sampling_rate"])
                    if i % 2 == 0 and "audio" in x
                    else None
                )
                for i, x in enumerate(inputs)
            ]
        else:
            prompts = [x["prompt"] for x in inputs]
            prompts_text = prompts  # done in ds part.
            prompts_audio = [
                (
                    (x["audio"]["array"], x["audio"]["sampling_rate"])
                    if "audio" in x and x["audio"]
                    else None
                )
                for x in inputs
            ]

        processed_items = []
        # Iterate through each input sample
        for item in inputs:
            prompt_text = item["prompt"]
            prompt_audio = (
                (item["audio"]["array"], item["audio"]["sampling_rate"])
                if "audio" in item and item["audio"]
                else None
            )

            # Process single sample, turn off auto-padding
            processed_item = self.processing_class(
                text=[prompt_text],
                audios=[prompt_audio] if prompt_audio else None,
                return_tensors="np",  # Use numpy for subsequent processing
                padding=False,
            )

            # Extract features of single sample
            single_item_features = {
                key: value[0] if value is not None and len(value) > 0 else None
                for key, value in processed_item.items()
            }
            processed_items.append(single_item_features)

        # Use modified collate_fn for batching and padding
        prompt_inputs = self.collate_fn(processed_items)

        prompt_inputs = super()._prepare_input(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        # adding audio support
        input_audio_embeds, audio_embed_sizes = (
            prompt_inputs["input_audio_embeds"],
            prompt_inputs["audio_embed_sizes"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # ======= Generation Start =======
        torch.cuda.empty_cache()  # required to avoid OOM in some cases

        # First, update the vLLM weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        if self.guided_decoding_regex:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=self.guided_decoding_regex
            )
        else:
            guided_decoding = None

        sampling_params = SamplingParams(
            n=1,  # vLLM on each GPU generates only 1 in colocate mode
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            max_tokens=self.max_completion_length,
            guided_decoding=guided_decoding,
            logprobs=0,  # only return the logprob of the generated token
        )

        if self.vllm_tensor_parallel_size > 1:
            # Gather prompts from all ranks in the TP group and flatten.
            # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
            orig_size = len(prompts_text)
            gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(
                gathered_prompts, prompts_text, group=self.tp_group
            )
            all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
        else:
            all_prompts_text = prompts_text

        inputs_vllm = []
        for i in range(len(all_prompts_text)):
            inputs_vllm.append(
                {
                    "prompt": all_prompts_text[i],
                    "multi_modal_data": (
                        {"audio": prompts_audio[i]}
                        if i < len(prompts_audio) and prompts_audio[i]
                        else None
                    ),
                }
            )

        with profiling_context(self, "vLLM.generate"):
            all_outputs = self.llm.generate(
                inputs_vllm,
                sampling_params=sampling_params,
                # lora_request=lora, # the weight is merged and updated by trl
                use_tqdm=False,
            )

        # Replace the <|end|> token to eos token
        completion_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        all_logprobs = [
            [next(iter(lp.values())).logprob for lp in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]

        if self.vllm_tensor_parallel_size > 1:
            # Slice completions for this rank within its TP group.
            # Each rank generates all outputs — we keep only our share.
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(
                local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size
            )
            completion_ids = completion_ids[tp_slice]
            all_logprobs = all_logprobs[tp_slice]

        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        sampling_per_token_logps = [
            torch.tensor(logprobs, device=device, dtype=torch.float32)
            for logprobs in all_logprobs
        ]
        sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)

        # ======= Generation Done =======

        # Mask everything after the first EOS token
        end_token_id = self.processing_class.encode("<|end|>")[0]
        is_eos = completion_ids == end_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = (
            agg_completion_lengths.sum()
        )  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            old_per_token_logps = self._get_per_token_logps(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                input_audio_embeds,
                audio_embed_sizes,
                batch_size,
            )

        # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
        if self.use_vllm and self.vllm_importance_sampling_correction:
            importance_sampling_ratio = torch.exp(
                old_per_token_logps - sampling_per_token_logps
            )
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=self.vllm_importance_sampling_cap
            )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completions = completions_text

        # ----------------- Final modified code block start -----------------
        # Pre-compute hidden states from the CURRENT model using existing tensors.
        # This call is now multi-modal aware and highly efficient.

        # Use prompt_mask to calculate true length of prompt in each sample (handle padding)
        # This returns a tensor of shape (batch_size,)
        prompt_lengths = prompt_mask.sum(dim=1)

        # Call helper function to efficiently get multimodal hidden states for reward model
        hidden_states_for_reward = self._get_hidden_states_for_reward(
            model=self.model,
            prompt_completion_ids=prompt_completion_ids,
            attention_mask=attention_mask,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            prompt_lengths=prompt_lengths,  # Pass tensor containing true lengths
        )
        assert len(inputs) == len(
            hidden_states_for_reward
        ), "Mismatch between inputs and computed hidden states"
        for i, hs in enumerate(hidden_states_for_reward):
            inputs[i]["hidden_states"] = hs
        # ----------------- Final modified code block end -----------------

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                reward_kwargs["__" + reward_func_name + "__"] = (
                    output_reward_func  # if further funcs need the rewards...
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan
                    for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        if self.reward_calculation == "sum":
            # Apply weights to each reward function's output and sum
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
        else:  # self.reward_calculation == "prod"
            # Use multiply instead of nansum to aggregate
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).prod(dim=1)

        # --- Handle NaN values in reward ---
        # 1. Record sample positions where reward is NaN
        is_nan_reward = torch.isnan(rewards)

        # Divide rewards into groups
        grouped_rewards = rewards.view(-1, self.num_generations)

        if self.separate_advantage_calculation:
            # When odd and even position advantages need to be calculated separately

            # Initialize tensor to store separate calculation results
            mean_grouped_rewards = torch.zeros_like(grouped_rewards)
            std_grouped_rewards = torch.zeros_like(grouped_rewards)

            # --- Process samples at odd positions ---
            odd_indices = torch.arange(
                0, self.num_generations, 2, device=rewards.device
            )
            odd_rewards = grouped_rewards[:, odd_indices]

            # Create NaN mask for odd samples
            nan_mask_odd = torch.isnan(odd_rewards)
            # Calculate number of non-NaN elements
            non_nan_count_odd = (~nan_mask_odd).sum(dim=1)
            non_nan_count_odd[non_nan_count_odd == 0] = 1  # Avoid division by zero

            # Replace NaN with 0 for calculation
            rewards_for_calc_odd = torch.nan_to_num(odd_rewards, nan=0.0)

            # Calculate mean of odd samples
            sum_of_rewards_odd = rewards_for_calc_odd.sum(dim=1)
            mean_odd = sum_of_rewards_odd / non_nan_count_odd

            # Calculate standard deviation of odd samples
            sum_of_squares_odd = (rewards_for_calc_odd**2).sum(dim=1)
            mean_of_squares_odd = sum_of_squares_odd / non_nan_count_odd
            var_odd = torch.clamp(mean_of_squares_odd - mean_odd.pow(2), min=0.0)
            std_odd = torch.sqrt(var_odd)

            # Fill calculated odd mean and std into result tensor
            mean_grouped_rewards[:, odd_indices] = mean_odd.unsqueeze(1)
            std_grouped_rewards[:, odd_indices] = std_odd.unsqueeze(1)

            # --- Process samples at even positions ---
            even_indices = torch.arange(
                1, self.num_generations, 2, device=rewards.device
            )
            even_rewards = grouped_rewards[:, even_indices]

            # Create NaN mask for even samples
            nan_mask_even = torch.isnan(even_rewards)
            # Calculate number of non-NaN elements
            non_nan_count_even = (~nan_mask_even).sum(dim=1)
            non_nan_count_even[non_nan_count_even == 0] = 1  # Avoid division by zero

            # Replace NaN with 0 for calculation
            rewards_for_calc_even = torch.nan_to_num(even_rewards, nan=0.0)

            # Calculate mean of even samples
            sum_of_rewards_even = rewards_for_calc_even.sum(dim=1)
            mean_even = sum_of_rewards_even / non_nan_count_even

            # Calculate standard deviation of even samples
            sum_of_squares_even = (rewards_for_calc_even**2).sum(dim=1)
            mean_of_squares_even = sum_of_squares_even / non_nan_count_even
            var_even = torch.clamp(mean_of_squares_even - mean_even.pow(2), min=0.0)
            std_even = torch.sqrt(var_even)

            # Fill calculated even mean and std into result tensor
            mean_grouped_rewards[:, even_indices] = mean_even.unsqueeze(1)
            std_grouped_rewards[:, even_indices] = std_even.unsqueeze(1)

            # Flatten padded mean and std tensors to match shape of rewards
            mean_grouped_rewards = mean_grouped_rewards.flatten()
            std_grouped_rewards = std_grouped_rewards.flatten()

        else:
            # 2. Ignore NaN values when calculating mean and std
            #    We achieve this by replacing NaN with 0 and dividing by count of non-NaN elements

            # Create boolean mask to identify NaN positions
            nan_mask = torch.isnan(grouped_rewards)

            # Calculate number of non-NaN elements in each group
            non_nan_count = (~nan_mask).sum(dim=1)
            # To avoid division by zero (if a group is all NaNs), replace count of 0 with 1
            non_nan_count[non_nan_count == 0] = 1

            # Replace NaN with 0 for sum calculation
            rewards_for_calc = torch.nan_to_num(grouped_rewards, nan=0.0)

            # Calculate mean ignoring NaNs
            sum_of_rewards = rewards_for_calc.sum(dim=1)
            mean_grouped_rewards = sum_of_rewards / non_nan_count

            # Calculate standard deviation ignoring NaNs (using Var(X) = E[X^2] - (E[X])^2)
            sum_of_squares = (rewards_for_calc**2).sum(dim=1)
            mean_of_squares = sum_of_squares / non_nan_count
            var_grouped_rewards = mean_of_squares - mean_grouped_rewards.pow(2)
            # Ensure variance is non-negative to avoid float precision issues
            std_grouped_rewards = torch.sqrt(torch.clamp(var_grouped_rewards, min=0.0))

            # 3. Expand mean and std for advantage calculation
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )

        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # Calculate advantage normally, samples with NaN reward will have NaN advantage
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # 4. Zero out advantage for samples with original NaN reward
        advantages[is_nan_reward] = 0.0

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # --- Unified Metrics logging logic ---
        # 1. Decide which data groups to process based on align_audio_to_text flag
        if self.align_audio_to_text:
            # Define groups:
            # case_name used for slicing, metric_suffix used for log key suffix
            cases_to_process = [
                ("odd", "_text"),  # Odd items (text only)
                ("even", ""),  # Even items (text+audio)
            ]
        else:
            # If not distinguishing, only process one group 'all', log key uses original
            cases_to_process = [("all", "")]

        # 2. Loop through each defined group
        for case_name, metric_suffix in cases_to_process:

            # 3. Get corresponding tensor slice index based on group name
            if case_name == "odd":
                indices = slice(1, None, 2)  # Odd indices: 1, 3, 5, ...
            elif case_name == "even":
                indices = slice(0, None, 2)  # Even indices: 0, 2, 4, ...
            else:  # case_name == "all"
                indices = slice(None)  # All indices (equivalent to [:])

            # 4. Slice current group data from aggregated tensor
            completion_lengths = agg_completion_lengths[indices]

            # If current slice is empty (e.g., batch size 1, even slice is empty), skip logging for this group
            if completion_lengths.numel() == 0:
                continue

            # Apply slice to all tensors to be logged
            terminated_with_eos = self.accelerator.gather(is_eos[indices].any(dim=1))
            rewards_subset = rewards_per_func[indices]
            mean_rewards_subset = mean_grouped_rewards[indices]
            std_rewards_subset = std_grouped_rewards[indices]
            is_std_zero_subset = is_std_zero[indices]

            # --- Start logging metrics for current group ---

            # Record completion length
            self._metrics[mode][f"completions{metric_suffix}/mean_length"].append(
                completion_lengths.float().mean().item()
            )
            self._metrics[mode][f"completions{metric_suffix}/min_length"].append(
                completion_lengths.float().min().item()
            )
            self._metrics[mode][f"completions{metric_suffix}/max_length"].append(
                completion_lengths.float().max().item()
            )

            # Record info related to sequences ending with EOS
            term_completion_lengths = completion_lengths[terminated_with_eos]
            clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
                completion_lengths
            )
            self._metrics[mode][f"completions{metric_suffix}/clipped_ratio"].append(
                clipped_completions_ratio
            )

            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)

            self._metrics[mode][
                f"completions{metric_suffix}/mean_terminated_length"
            ].append(term_completion_lengths.float().mean().item())
            self._metrics[mode][
                f"completions{metric_suffix}/min_terminated_length"
            ].append(term_completion_lengths.float().min().item())
            self._metrics[mode][
                f"completions{metric_suffix}/max_terminated_length"
            ].append(term_completion_lengths.float().max().item())

            # Record metrics related to reward functions
            for i, reward_func_name in enumerate(self.reward_func_names):
                mean_rewards = torch.nanmean(rewards_subset[:, i]).item()
                self._metrics[mode][
                    f"rewards{metric_suffix}/{reward_func_name}/mean"
                ].append(mean_rewards)
                std_rewards = nanstd(rewards_subset[:, i]).item()
                self._metrics[mode][
                    f"rewards{metric_suffix}/{reward_func_name}/std"
                ].append(std_rewards)

            # Record metrics related to total reward
            self._metrics[mode][f"reward{metric_suffix}"].append(
                mean_rewards_subset.mean().item()
            )
            self._metrics[mode][f"reward_std{metric_suffix}"].append(
                std_rewards_subset.mean().item()
            )
            self._metrics[mode][f"frac_reward_zero_std{metric_suffix}"].append(
                is_std_zero_subset.float().mean().item()
            )

            # --- [Added Code] Record vLLM importance sampling related metrics ---
            if self.use_vllm and self.vllm_importance_sampling_correction:
                # Apply slice to tensors required for new metrics
                old_logps_subset = old_per_token_logps[indices]
                sampling_logps_subset = sampling_per_token_logps[indices]
                mask_subset = completion_mask[indices]
                is_ratio_subset = importance_sampling_ratio[indices]

                # Calculate logp difference
                delta = torch.abs(old_logps_subset - sampling_logps_subset)
                delta = delta[mask_subset.bool()]  # Apply mask
                mean_delta = (
                    torch.mean(delta)
                    if delta.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                max_delta = (
                    torch.max(delta)
                    if delta.numel() > 0
                    else torch.tensor(0.0, device=device)
                )

                # Record mean and max of logp difference
                self._metrics[mode][
                    f"sampling{metric_suffix}/sampling_logp_difference/mean"
                ].append(self.accelerator.gather(mean_delta).mean().item())
                self._metrics[mode][
                    f"sampling{metric_suffix}/sampling_logp_difference/max"
                ].append(self.accelerator.gather(max_delta).max().item())

                # Calculate importance sampling ratio
                flat_is_ratio = is_ratio_subset[mask_subset.bool()]  # Apply mask
                min_importance_sampling_ratio = (
                    torch.min(flat_is_ratio)
                    if flat_is_ratio.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                mean_importance_sampling_ratio = (
                    torch.mean(flat_is_ratio)
                    if flat_is_ratio.numel() > 0
                    else torch.tensor(0.0, device=device)
                )
                max_importance_sampling_ratio = (
                    torch.max(flat_is_ratio)
                    if flat_is_ratio.numel() > 0
                    else torch.tensor(0.0, device=device)
                )

                # Record min, mean, max of importance sampling ratio
                self._metrics[mode][
                    f"sampling{metric_suffix}/importance_sampling_ratio/min"
                ].append(
                    nanmin(
                        self.accelerator.gather(min_importance_sampling_ratio)
                    ).item()
                )
                self._metrics[mode][
                    f"sampling{metric_suffix}/importance_sampling_ratio/mean"
                ].append(
                    self.accelerator.gather(mean_importance_sampling_ratio)
                    .nanmean()
                    .item()
                )
                self._metrics[mode][
                    f"sampling{metric_suffix}/importance_sampling_ratio/max"
                ].append(
                    nanmax(
                        self.accelerator.gather(max_importance_sampling_ratio)
                    ).item()
                )

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes": audio_embed_sizes,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            "importance_sampling_ratio": (
                importance_sampling_ratio
                if self.use_vllm and self.vllm_importance_sampling_correction
                else None
            ),
        }
