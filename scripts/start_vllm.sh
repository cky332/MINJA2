#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible server for MINJA reproduction.
#
# Hardware target: 4 x NVIDIA RTX 4090 (24 GB each, 96 GB total).
# Default model: Qwen2.5-72B-Instruct-AWQ (4-bit, ~40 GB weights, fits with TP=4).
#
# Usage:
#   1. conda activate vllm  (env with `pip install vllm==0.7.3`)
#   2. Download weights once:
#      HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
#          Qwen/Qwen2.5-72B-Instruct-AWQ \
#          --local-dir /data/models/Qwen2.5-72B-AWQ
#   3. bash scripts/start_vllm.sh
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
