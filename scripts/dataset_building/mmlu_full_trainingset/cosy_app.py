# accelerate launch --num_processes=2 cosy_app.py

import sys

sys.path.append("third_party/Matcha-TTS")

import gradio as gr
from huggingface_hub import snapshot_download
import torch
import numpy as np
import threading
from accelerate import Accelerator

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from vllm.engine.arg_utils import EngineArgs

accelerator = Accelerator()
device = accelerator.device
local_rank = accelerator.local_process_index
print(f"local_rank: {local_rank}, device: {device}")

_raw_func = EngineArgs.create_engine_config


def create_engine_config(self, usage_context):
    print(f"**hooked** create_engine_config, {device=}")
    config = _raw_func(self, usage_context)
    config.device_config.device = device
    return config


EngineArgs.create_engine_config = create_engine_config


def __init__(
    self,
    llm: torch.nn.Module,
    flow: torch.nn.Module,
    hift: torch.nn.Module,
    fp16: bool = False,
):
    print(f"**hooked** __init__, {device=}")
    self.device = device
    self.llm = llm
    self.flow = flow
    self.hift = hift
    self.fp16 = fp16
    if self.fp16 is True:
        self.llm.half()
        self.flow.half()
    # NOTE must matching training static_chunk_size
    self.token_hop_len = 25
    # hift cache
    self.mel_cache_len = 8
    self.source_cache_len = int(self.mel_cache_len * 480)
    # speech fade in out
    self.speech_window = np.hamming(2 * self.source_cache_len)
    # rtf and decoding related
    self.llm_context = torch.cuda.Stream(self.device)
    self.lock = threading.Lock()
    # dict used to store session related variable
    self.tts_speech_token_dict = {}
    self.llm_end_dict = {}
    self.hift_cache_dict = {}


CosyVoice2Model.__init__ = __init__

with accelerator.local_main_process_first():
    # Download model
    snapshot_download(
        "FunAudioLLM/CosyVoice2-0.5B", local_dir="pretrained_models/CosyVoice2-0.5B"
    )

from vllm import ModelRegistry

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# Initialize CosyVoice2 model
cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B",
    load_jit=False,
    load_trt=False,
    load_vllm=True,
    fp16=True,
)


# Gradio interface function
def generate_audio(text, reference_text, reference_audio_path):
    # Set random seed for reproducibility
    set_all_random_seed(0)

    # Load prompt speech from the uploaded audio file
    if reference_audio_path:
        prompt_speech_16k = load_wav(reference_audio_path, 16000)
    else:
        return None, "Please upload a reference audio file."

    # Perform inference
    try:
        # CosyVoice's inference function returns a generator
        # We need to iterate through it to get the final audio
        generated_audio_list = []
        for j in cosyvoice.inference_zero_shot(
            text, reference_text, prompt_speech_16k, stream=False
        ):
            generated_audio_list.append(j["tts_speech"])

        # Concatenate all generated audio tensors
        generated_audio = torch.cat(generated_audio_list, dim=-1)

        # Convert the generated audio tensor to a numpy array
        # and return it in a format that Gradio's Audio component can handle
        return (cosyvoice.sample_rate, generated_audio.squeeze().cpu().numpy()), None

    except Exception as e:
        return None, f"An error occurred: {str(e)}"


# Build the Gradio app
with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("# CosyVoice2 Zero-Shot TTS")

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Input Text",
                placeholder="Enter the text to be synthesized.",
                lines=5,
            )
            reference_text = gr.Textbox(
                label="Reference Text",
                placeholder="Enter the text that corresponds to the reference audio.",
                lines=1,
            )
            reference_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
            )

        with gr.Column(scale=3):
            error = gr.HTML(
                label="Error Message",
                visible=True,
            )
            audio = gr.Audio(
                label="Generated Audio",
                type="numpy",
                interactive=False,
                visible=True,
            )

            generate_button = gr.Button(
                value="Generate Audio",
                variant="primary",
            )

    # Submit action
    generate_button.click(
        fn=generate_audio,
        inputs=[text, reference_text, reference_audio],
        outputs=[audio, error],
        concurrency_limit=1,
    )

# Launch the app
if __name__ == "__main__":
    app.launch(
        show_error=True, show_api=True, server_port=local_rank + 7860, share=True
    )
