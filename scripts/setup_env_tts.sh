#!/bin/sh
# Phase 5 - create the isolated data-generation environment (env_tts).
#
# Kept separate from env_asr on purpose: qwen-asr pins transformers==4.57.6
# while qwen-tts pins transformers==4.57.3, so the two cannot share an
# interpreter. This also keeps generation and training separated, which is the
# standing rule for this machine.
#
# gradio is pinned for the same reason it is pinned in env_asr: qwen-tts leaves
# it unbounded, gradio 6.x requires huggingface_hub>=1.0, and transformers 4.57
# forbids that, so an unpinned resolve backtracks through dozens of 30 MB
# wheels. Pinning costs nothing here - gradio is only used by the demo UI.
#
# Run from the project root on the H20:
#     sh scripts/setup_env_tts.sh
set -e

PROJECT="/data/shenxin/qwen3_asr_hospital"
ENV_DIR="$PROJECT/env_tts"
INDEX="https://mirrors.aliyun.com/pypi/simple/"
export PIP_CACHE_DIR=/tmp/pipcache

cd "$PROJECT"

if [ -d "$ENV_DIR" ]; then
    echo "env_tts already exists at $ENV_DIR - nothing to create."
else
    echo "creating $ENV_DIR"
    python3 -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/pip" install -q --index-url "$INDEX" --upgrade pip

# Heavy shared dependencies first. If the wheels are already in the pip cache
# from the env_asr install they come from disk; the mirror does not always send
# cacheable responses, so this may re-download.
echo "installing torch (this is the slow part)"
"$ENV_DIR/bin/pip" install --index-url "$INDEX" torch==2.9.1 torchaudio==2.9.1

echo "installing qwen-tts and its pinned dependencies"
"$ENV_DIR/bin/pip" install --index-url "$INDEX" \
    "gradio==5.50.0" \
    "transformers==4.57.3" \
    "accelerate==1.12.0" \
    librosa soundfile sox onnxruntime einops

# qwen-tts itself is installed from the local checkout of the official repo.
# If a PyPI release is available, replace this with: pip install qwen-tts
if [ -d "$PROJECT/Qwen3-TTS-main" ]; then
    echo "installing qwen-tts from the local checkout"
    "$ENV_DIR/bin/pip" install --no-deps -e "$PROJECT/Qwen3-TTS-main"
else
    echo "installing qwen-tts from the index"
    "$ENV_DIR/bin/pip" install --index-url "$INDEX" --no-deps qwen-tts
fi

echo ""
echo "verifying"
"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers
print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("cuda         :", torch.cuda.is_available())
try:
    import qwen_tts
    print("qwen_tts     : importable")
except Exception as exc:
    print("qwen_tts     : NOT importable ->", exc)
PY

echo ""
echo "env_tts ready. Record it with:"
echo "  $ENV_DIR/bin/python scripts/audit_environment.py --outdir logs/audit_env_tts"
