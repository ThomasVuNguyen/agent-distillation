#!/bin/bash

# ===================== user setting ===================== #
BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
LORA_PATH=""
EXP_TYPE="agent"
PORT_BASE=8000
MAX_LORA_RANK=64

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"
  ["math"]="data_processor/math_dataset/train/math_1000_20250414.json"
  ["math2"]="data_processor/math_dataset/train/math_medium_1000_20250430.json"
)

# Update paths as needed
declare -A PREFIXS=(
  ["hotpotqa"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/hotpotqa_1000_20250402_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning.json"
  ["math"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_1000_20250414_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning.json"
  ["math2"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_medium_1000_20250430_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.5_seed=42_type=reasoning.json"
)

MAX_TOKENS=1024
RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

SKIP_SERVING=false
USE_WEB_SEARCH=false
USE_PREFIX=false

# Parse command line args
for arg in "$@"; do
  case $arg in
    --skip-serving)
      SKIP_SERVING=true
      ;;
    --use-web-search)
      USE_WEB_SEARCH=true
      ;;
    --use-prefix)
      USE_PREFIX=true
      ;;
  esac
done

PIDS=()

cleanup() {
  echo ""
  echo "🧹 Cleaning up vLLM servers..."
  kill ${PIDS[*]} 2>/dev/null
  ps aux | grep '/home/jovyan/conda/agents/bin/python3.12' | grep 'vllm serve' | awk '{print $2}' | xargs kill
  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ All vLLM servers stopped."
}

trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM

# Conda shell hook
source "$(conda info --base)/etc/profile.d/conda.sh"

# ===================================================== #
# Start Retriever and vLLM Servers
# ===================================================== #
if [ "$SKIP_SERVING" = false ]; then
  echo "🔍 Launching retriever server in Conda env \"$RETRIEVER_CONDA_ENV\" …"
  (
    conda activate "$RETRIEVER_CONDA_ENV"
    CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
      python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
    RETRIEVER_PID=$!
    echo "🛰️  Retriever server started (PID: $RETRIEVER_PID, GPUs: $RETRIEVER_GPU_DEVICES)"
    conda deactivate
  )&
  PIDS+=($RETRIEVER_PID)

  i=0
  LOG_FILE="vllm.log"
  CMD="python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --tensor-parallel-size 4 \
    --port $((PORT_BASE + i))"

  eval $CMD > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
  echo "📺 Started final vLLM on all GPUs (port $((PORT_BASE + i))), watching for startup completion..."

  ( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
    echo "$line"
    if [[ "$line" == *"Application startup complete."* ]]; then
      echo "✅ vLLM fully started."
      break
    fi
  done
fi

# ===================================================== #
# Run Experiments
# ===================================================== #
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Running reasoning on $dataset..."
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose \
    --do_filtering"

  if [ "$USE_WEB_SEARCH" = true ]; then
    AGENT_CMD="$AGENT_CMD --search_engine_type duckduckgo"
  else
    AGENT_CMD="$AGENT_CMD --multithreading --use_process_pool --use_single_endpoint"
  fi

  if [ "$USE_PREFIX" = true ]; then
    AGENT_CMD="$AGENT_CMD --prefix_memory \"${PREFIXS[$dataset]}\""
  fi

  eval $AGENT_CMD
done

RUN_EXIT_CODE=$?
cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Agent script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ Script completed successfully"
  exit 0
fi
