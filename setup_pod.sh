#!/usr/bin/env bash
# setup_pod.sh -- one-time setup of a FRESH RunPod pod (template runpod/pytorch 2.8.0, Ubuntu 24.04, Python 3.12) for the stage-3 runs.
# Run it from /workspace/p2 after unpacking tools_sync.zip:   bash setup_pod.sh
# Needs the environment variable HF_TOKEN (a Hugging Face READ token whose account accepted the Llama-3.1 licence): set it in the pod's environment variables, never in a file.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
mkdir -p "$HF_HOME" /workspace/p2/run_logs /workspace/p2/stage3/acts
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Add it as an environment variable of the pod (value {{ RUNPOD_SECRET_hf_token }} if you stored it as a RunPod secret), then restart the pod." >&2
  exit 2
fi
# a virtual environment on the persistent volume that keeps the template's torch
if [ ! -d /workspace/venv_p2 ]; then python3 -m venv --system-site-packages /workspace/venv_p2; fi
source /workspace/venv_p2/bin/activate
pip install -q -U pip wheel
# versions verified on the university VM: transformers 4.46.1 with a transformer_lens 2.x
pip install -q "transformers==4.46.1" "transformer_lens<3" datasets sentencepiece protobuf accelerate "huggingface_hub" numpy scipy pandas requests python-dotenv pyyaml tenacity openai pydantic tqdm einops
python - << 'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU", "| transformers", transformers.__version__)
assert torch.cuda.is_available(), "no GPU visible"
import transformer_lens
print("transformer_lens ok")
PY
# the two models (safetensors only; about 30 GB in all)
python - << 'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for repo in ("mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-3.1-8B-Instruct"):
    p = snapshot_download(repo, token=tok, allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "tokenizer*"])
    print("cached:", repo, "->", p)
PY
# a small file that activates everything in later sessions
cat > /workspace/p2/env_pod.sh << 'ENV'
export HF_HOME=/workspace/.cache/huggingface
export PYTHONUNBUFFERED=1
source /workspace/venv_p2/bin/activate
cd /workspace/p2
ENV
echo "setup done. In every new terminal: source /workspace/p2/env_pod.sh"
df -h /workspace | tail -1