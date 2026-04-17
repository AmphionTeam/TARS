# FISH ENV

# conda create -n fish python=3.10
# conda activate fish

# cd scripts/dataset_building/mmlu_full_trainingset

# git clone https://huggingface.co/spaces/fishaudio/openaudio-s1-mini

# cd openaudio-s1-mini
# pip install uv
# uv pip install -r requirements.txt
# uv pip install accelerate
# cp ../app.py .

# huggingface-cli login --token $(cat /path/to/your/huggingface_token)

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# COSYVOICE ENV
sudo DEBIAN_FRONTEND=noninteractive apt install -y git
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git submodule update --init --recursive

sudo DEBIAN_FRONTEND=noninteractive apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -y build-essential

conda create -y -n cosy python=3.10
conda activate cosy
pip install uv
uv pip install numpy nvidia_stub
uv pip install -r cosy_requirements.txt --index-strategy unsafe-best-match --no-build-isolation
uv pip install pyworld==0.3.4 accelerate vllm==v0.9.0
uv pip install gradio==5.44.1 gradio-client==1.12.1
conda install -y -c nvidia cuda-toolkit

# WHISPER ENV

conda create -y -n whisper python=3.10
conda activate whisper

conda install -y nvidia::libcublas cuda-version=12
conda install -y conda-forge::cudnn=8
pip install uv
# uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
uv pip install faster-whisper datasets==3.6.0 librosa soundfile jiwer gradio_client tqdm jsonlines accelerate
uv pip install --force-reinstall ctranslate2==4.4.0