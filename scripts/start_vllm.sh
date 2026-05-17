#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible server for MINJA reproduction.
#
# Hardware target: 4 x NVIDIA RTX 4090 (24 GB each, 96 GB total).
# Default model: Qwen2.5-72B-Instruct-AWQ (4-bit, ~40 GB weights, fits with TP=4).
#
# Usage:
#   1. conda activate vllm  (env with `pip install vllm==0.7.3`)
#   2. Download weights once (huggingface_hub >= 0.30 uses `hf`, not `huggingface-cli`):
#      hf download Qwen/Qwen2.5-72B-Instruct-AWQ \
#          --local-dir /data/models/Qwen2.5-72B-AWQ
#      # Optional acceleration: prepend `HF_XET_HIGH_PERFORMANCE=1`
#      # Fallback if the CLI misbehaves:
#      #   python -c "from huggingface_hub import snapshot_download; \
#      #     snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', \
#      #     local_dir='/data/models/Qwen2.5-72B-AWQ')"
#   3. bash scripts/start_vllm.sh
#      # Or skip step 2 and let vLLM auto-download on first start:
#      #   MODEL_PATH=Qwen/Qwen2.5-72B-Instruct-AWQ bash scripts/start_vllm.sh
#
# Override defaults with env vars, e.g.:
#   MODEL_PATH=/data/models/Qwen2.5-72B-AWQ TP=4 PORT=8000 bash scripts/start_vllm.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-72B-AWQ}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-72b}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_LEN="${MAX_LEN:-16384}"
MAX_SEQS="${MAX_SEQS:-16}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"   # hermes for Qwen2.5; llama3_json for Llama-3.x

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Preflight: if MODEL_PATH looks like a local path, fail fast with a clear
# message when the directory is missing. Otherwise (e.g. "Qwen/Qwen2.5-72B-
# Instruct-AWQ") let vLLM resolve it as a HuggingFace repo id and download.
case "$MODEL_PATH" in
    /*|~/*|./*|../*)
        if [ ! -d "$MODEL_PATH" ]; then
            echo "[vllm] ERROR: MODEL_PATH does not exist locally: $MODEL_PATH" >&2
            echo "[vllm] hint: override it, e.g." >&2
            echo "[vllm]   MODEL_PATH=\$HOME/data/models/Qwen2.5-72B-AWQ bash scripts/start_vllm.sh" >&2
            echo "[vllm] hint: or pass a HuggingFace repo id to auto-download, e.g." >&2
            echo "[vllm]   MODEL_PATH=Qwen/Qwen2.5-72B-Instruct-AWQ bash scripts/start_vllm.sh" >&2
            exit 1
        fi
        ;;
esac

# Make torch's bundled CUDA libs win over any system CUDA in LD_LIBRARY_PATH
# (e.g. /usr/local/cuda-12.2/lib64). Without this, torch 2.5 + the CUDA 12.4
# nvidia-*-cu12 wheels crash on import with
#   undefined symbol: __nvJitLinkComplete_12_4
# because the dynamic linker picks the older libnvJitLink.so.12 from the
# system CUDA dir before torch's bundled one.
_PIP_NVIDIA_LIB_DIRS=$(python - <<'PY' 2>/dev/null || true
import os, pkgutil, importlib.util
try:
    import nvidia
except ImportError:
    raise SystemExit
dirs = []
for _, name, ispkg in pkgutil.iter_modules(nvidia.__path__):
    if not ispkg:
        continue
    spec = importlib.util.find_spec(f"nvidia.{name}")
    if not spec or not spec.submodule_search_locations:
        continue
    for base in spec.submodule_search_locations:
        lib = os.path.join(base, "lib")
        if os.path.isdir(lib):
            dirs.append(lib)
print(":".join(dirs))
PY
)
if [ -n "${_PIP_NVIDIA_LIB_DIRS:-}" ]; then
    export LD_LIBRARY_PATH="${_PIP_NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[vllm] prepended pip nvidia libs to LD_LIBRARY_PATH"
fi

echo "[vllm] model=$MODEL_PATH served-as=$SERVED_NAME tp=$TP port=$PORT"
echo "[vllm] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --quantization awq_marlin \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --port "$PORT" \
    --host "$HOST"
