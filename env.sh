set -x
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "Run this script using: source ${BASH_SOURCE[0]} or . ${BASH_SOURCE[0]}"
    exit 1
fi

conda create -y -n trl python=3.12
conda activate trl

# if CUDA_HOME is not set:
# conda install -c nvidia cuda-toolkit

pip install uv

uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install trl==0.18.1 trl[vllm] vllm==0.8.3 scipy==1.15.1 peft==0.13.2 backoff==2.2.1 transformers==4.51.0 accelerate==1.3.0 datasets==3.6.0 wandb==0.21.0 --no-build-isolation
uv pip install jiwer librosa soundfile num2words word2number
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

uv pip install sentence_transformers deepspeed
bash scripts/install_xfinder/setup.sh

wandb login
# If you want offline training, set WANDB_MODE=offline
# export WANDB_MODE=offline
