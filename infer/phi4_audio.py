import os
import torch
import soundfile
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download
from trainer.vllm_patch import *
from vllm.sampling_params import SamplingParams
from vllm import LLM
from vllm.lora.request import LoRARequest
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(
        self,
        prompt_tokens_ids: tuple,
        past_tokens_ids: tuple,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        :ref: https://github.com/vllm-project/vllm/blob/911c8eb0000b1f9d1fef99ac9e209f83d801bd0a/vllm/model_executor/layers/logits_processor.py#L186
        """
        # score: [B, vocab_size]
        # input_ids: [B, cur_len]
        input_ids = prompt_tokens_ids + past_tokens_ids
        if len(input_ids) < self.ngram_size:
            return scores

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]
        input_ids = torch.LongTensor(input_ids).reshape(num_batch_hypotheses, -1)
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size, input_ids, num_batch_hypotheses, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class Phi4Audio:
    def __init__(
        self,
        model_path="microsoft/Phi-4-multimodal-instruct",
        device="cuda:0",
        use_vllm=False,
        load_lora=True,
        adapter_name=["speech"],
        system_prompt=None,
        **kwargs,
    ):
        """
        Initialize the Phi4Audio model.

        Args:
            model_path (str): Path to the model checkpoint or model name
            device (str): Device to run the model on ('cuda:0' or 'cpu')
        """
        self.device = device
        self.model_path = model_path

        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
        )

        self.use_vllm = use_vllm

        # Initialize model
        if self.use_vllm:
            # Since the vision-lora and speech-lora co-exist with the base model,
            # we have to manually specify the path of the lora weights.
            if load_lora:
                if "microsoft/Phi-4-multimodal-instruct" == model_path:
                    model_path = snapshot_download(model_path)

                self.lora = []
                for name in adapter_name:
                    lora_path = os.path.join(model_path, f"{name}-lora")
                    assert os.path.exists(
                        lora_path
                    ), f"Lora path {lora_path} does not exist"
                    self.lora.append(LoRARequest(name, 1 + len(self.lora), lora_path))
            else:
                self.lora = None
            
            if os.path.exists(model_path + "/adapter_model.safetensors"):
                with open(f"{model_path}/adapter_config.json") as f:
                    config = json.load(f)
                    model_path = config["base_model_name_or_path"]
                    # since the vllm will not automatically merge the lora adapters
                    # we need to update the model_path pointing to a merged weights.
                    
                    if model_path == "microsoft/Phi-4-multimodal-instruct":
                        # FIXME: replace with the merged speech path. Run jupter notebook to merge.
                        model_path = os.environ.get("SPEECH_LORA_MERGED_MODEL_PATH", "output/Phi-4-MM-speech") # (base: Phi-4-Mini + speech-lora, adapters: speech-lora)
                        if not os.path.exists(model_path):
                            raise Exception(f"Model path {model_path} does not exist, check the README file and set env var SPEECH_LORA_MERGED_MODEL_PATH")
                    
                    # BUG: This used to solve mis-configuration of adapter_config.json
                    if "-merged-hf" in model_path:
                        model_path = model_path.replace("-merged-hf", "-merged")
                print(f"Detected LoRA adapters. Using base model from its config: {model_path}")

            self.llm = LLM(
                model=model_path,
                tokenizer="microsoft/Phi-4-multimodal-instruct",
                trust_remote_code=True,
                max_model_len=6400,
                max_num_seqs=2,
                enable_lora=load_lora,
                max_lora_rank=320,
                limit_mm_per_prompt={"audio": 1},
                gpu_memory_utilization=0.45,
                device=device,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="flash_attention_2",
            ).to(device)

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(
                model_path, "generation_config.json"
            )

        # Set up prompts
        self.user_prompt = "<|user|>"
        self.assistant_prompt = "<|assistant|>"
        self.prompt_suffix = "<|end|>"

    def __call__(
        self,
        inputs,
        audios=None,
        max_new_tokens=1000,
        temperature=1.0,
        no_repeat_ngram_size=None,
        **kwargs,
    ):
        """
        Basic inference function.

        Args:
            inputs (str): Input prompt text. DO NOT INCLUDE <|user|> or <|assistant|>.
            audios (list): List of audio data
            max_new_tokens (int): Maximum number of tokens to generate

        Returns:
            str: Model's response
        """

        sys_prompt = kwargs.pop("sys_prompt", None)
        if sys_prompt:
            sys_prompt = f"<|system|>{sys_prompt}<|end|>"
        else:
            sys_prompt = ""

        texts = []
        inputs_vllm = []
        for i, input in enumerate(inputs):
            texts.append(
                f"{sys_prompt}{self.user_prompt}{input}{self.prompt_suffix}{self.assistant_prompt}"
            )
            inputs_vllm.append(
                {
                    "prompt": texts[i],
                    "multi_modal_data": (
                        {"audio": audios[i]} if audios is not None else None
                    ),
                }
            )

        if not self.use_vllm:
            # Process inputs
            inputs = self.processor(text=texts, audios=audios, return_tensors="pt").to(
                self.device
            )

            inputs = {
                **inputs,
                **kwargs
            }

            # Generate response
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.generation_config,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_logits_to_keep=0
                # use_tqdm=False
            )

            # Process output
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return response

        # use vllm
        # We set temperature to 0.2 so that outputs can be different
        # even when all prompts are identical when running batch inference.
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            logits_processors=(
                [NoRepeatNGramLogitsProcessor(ngram_size=no_repeat_ngram_size)]
                if no_repeat_ngram_size is not None
                else None
            ),
        )

        if self.lora is not None:
            lora = self.lora * len(inputs_vllm)
        else:
            lora = None

        try:
            outputs = self.llm.generate(
                inputs_vllm,
                sampling_params=sampling_params,
                lora_request=lora,
            )
        except Exception as e:
            print(f"Error during generation: {e}")
            return [""] * len(inputs_vllm)
        # RequestOutput(request_id=0, prompt='<|user|><|endoftext11|>...<|endoftext11|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.<|end|><|assistant|>', prompt_token_ids=[200021, ..., 200011, 28326, 402, 290, 18618, 11065, 11, 10419, 261, 16796, 2201, 66912, 328, 290, 36116, 3100, 13, 200020, 200019], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='We do not break. We never give in. We never back down.', token_ids=[2167, 621, 625, 2338, 13, 1416, 3779, 3644, 306, 13, 1416, 3779, 1602, 1917, 13, 200020], cumulative_logprob=None, logprobs=None, finish_reason=stop, stop_reason=200020)], finished=True, metrics=None, lora_request=None, num_cached_tokens=None, multi_modal_placeholders={})

        response = [output.outputs[0].text for output in outputs]
        return response

    def transcribe_audio(self, audio_path, prompt=None):
        """
        Transcribe audio file to text.

        Args:
            audio_path (str): Path to the audio file
            prompt (str, optional): Custom prompt for transcription.
                                  If None, uses default prompt.

        Returns:
            str: Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Read audio file
        audio = soundfile.read(audio_path)

        # Prepare prompt
        if prompt is None:
            prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."

        full_prompt = f"<|audio_1|>{prompt}"

        return self.__call__(full_prompt, [audio])


if __name__ == "__main__":
    phi4 = Phi4Audio(use_vllm=True)
    audio = soundfile.read("../wav/prompt.wav")
    print(
        phi4(
            [
                "<|audio_1|>Based on the attached audio, generate a comprehensive text transcription of the spoken content."
            ],
            [audio],
        )
    )
