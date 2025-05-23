#!/bin/bash

# ===================== User Setting ===================== #
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH="training_outputs/qwen-1.5B-instruct/cot_baseline_qwen2.5_32B_teacher" # set lora path here
EXP_TYPE="reasoning"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["math"]="data_processor/math_dataset/test/math_500_20250414.json"
  ["aime"]="data_processor/math_dataset/test/aime_90_20250504.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["bamboogle"]="data_processor/qa_dataset/test/bamboogle_125_20250507.json"
  ["gsm"]="data_processor/math_dataset/test/gsm_hard_500_20250507.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
  ["olymath"]="data_processor/math_dataset/test/olymath_200_20250511.json"
)
MAX_TOKENS=4096
# ===================================================== #

PIDS=()

# set end handler
cleanup() {
  echo ""
  echo "🧹 Cleaning up vLLM servers..."
  kill ${PIDS[*]} 2>/dev/null
  # If the process is not cleaned well
  ps aux | grep '/home/jovyan/conda/agents/bin/python3.12' | grep 'vllm serve' | awk '{print $2}' | xargs kill
  wait
  echo "✅ All vLLM servers stopped."
}

# Ctrl-C
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# 0. run retriever as background if

# 1. GPU 0~2 background
for i in 0 1 2; do
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --port $((PORT_BASE + i)) \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
  fi

  eval $CMD > vllm_gpu${i}.log 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $i (port $((PORT_BASE + i)))"
done

# 2. GPU 3 execute + log monitoring
i=3
LOG_FILE="vllm_gpu${i}.log"
CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --port $((PORT_BASE + i)) \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

if [ -n "$LORA_PATH" ]; then
  CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD > "$LOG_FILE" 2>&1 &
PIDS+=($!)
echo "📺 Started final vLLM on GPU $i (port $((PORT_BASE + i))), watching for startup completion..."

# 3. wait until "Application startup complete." detected
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started, launching reasoning agent!"
    break
  fi
done

for dataset in "${!DATASETS[@]}"; do
  # 4. run experiment
  echo "🧠 Running reasoning..."
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose"

  if [ -n "$LORA_PATH" ]; then
    AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""
  fi

  eval $AGENT_CMD
done

RUN_EXIT_CODE=$?

# 5. clean up server
cleanup

# 6. check exit code
if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Agent script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ Agent script completed successfully"
  exit 0
fi