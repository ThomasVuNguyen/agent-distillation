#!/bin/bash

# ===================== Configuration ===================== #
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
PORT=8000
RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ========================================================= #

# Cleanup handler
cleanup() {
  echo ""
  echo "🧹 Cleaning up retriever and vLLM..."
  ps aux | grep '/home/jovyan/conda/agents/bin/python3.12' | grep 'vllm serve' | awk '{print $2}' | xargs kill
  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ Cleanup done."
}

# Trap Ctrl+C
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM

echo "🔍 Launching retriever in background..."
# Conda shell hook (MUST be before activate)
source "$(conda info --base)/etc/profile.d/conda.sh"
(
  conda activate "$RETRIEVER_CONDA_ENV"
  CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
    python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
  RETRIEVER_PID=$!
  echo "🛰️  Retriever server started (PID: $RETRIEVER_PID, GPUs: $RETRIEVER_GPU_DEVICES)"
  conda deactivate
) &

# Wait briefly to ensure retriever has started
sleep 10

echo "🚀 Launching vLLM model in foreground on all GPUs..."
CMD="python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --tensor-parallel-size 4 \
  --port $PORT"

eval $CMD

cleanup
exit 0