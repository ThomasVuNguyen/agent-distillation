#!/bin/bash

# Activate virtual environment
source myenv/bin/activate

# ===================== user setting ===================== #
MODEL_ID="@cf/meta/llama-3.1-70b-instruct"
EXP_TYPE="agent"  # Choose "agent" or "reasoning"
TEMPERATURE=0.0
PARALLEL_WORKERS=4
USE_PREFIX=false
LIMIT_SAMPLES=10  # Limit to a small number of samples for testing

# ===================== argument parsing ===================== #
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model=*)
      MODEL_ID="${1#*=}"
      shift
      ;;
    --temperature=*)
      TEMPERATURE="${1#*=}"
      shift
      ;;
    --workers=*)
      PARALLEL_WORKERS="${1#*=}"
      shift
      ;;
    --use-prefix)
      USE_PREFIX=true
      shift
      ;;
    --limit=*)
      LIMIT_SAMPLES="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Get model name for output directory
MODEL_NAME=$(echo $MODEL_ID | sed 's/@cf\/meta\///g' | sed 's/@cf\/mistral\///g' | sed 's/@cf\/\|\//-/g')
MODEL_NAME=$(echo $MODEL_NAME | sed 's/\//-/g')

# Get today's date
DATE=$(date +%Y%m%d)

# Print configuration
echo "🚀 Starting CloudflareWorkersAI inference with model: $MODEL_ID"
echo "⚙️  Experiment type: $EXP_TYPE"
echo "🌡️  Temperature: $TEMPERATURE"
echo "👥 Parallel workers: $PARALLEL_WORKERS"
echo "🔢 Sample limit: $LIMIT_SAMPLES"

# Process math dataset
process_dataset() {
  local dataset_name=$1
  local dataset_path=$2
  
  echo ""
  echo "📊 Processing dataset: $dataset_name"
  echo "📂 Dataset path: $dataset_path"
  
  # Create output directory
  local output_dir="logs/qa_results/cloudflare/$MODEL_NAME/${dataset_name}_${DATE}"
  echo "💾 Output directory: $output_dir"
  mkdir -p "$output_dir"
  
  # Check if we should use first-thought prefix
  local prefix_args=""
  if [ "$USE_PREFIX" = true ]; then
    local prefix_path="exps_research/prompts/first_thought_prefix_memory.json"
    if [ -f "$prefix_path" ]; then
      prefix_args="--prefix_memory_path=$prefix_path"
      echo "🧠 Using first-thought prefix memory: $prefix_path"
    else
      echo "⚠️  Warning: Prefix file not found: $prefix_path"
    fi
  fi

  # Run the experiment using CloudflareWorkersAIModel
  python3 -m exps_research.unified_framework.run_experiment \
    --data_path "$dataset_path" \
    --model_type cloudflare \
    --model_id "$MODEL_ID" \
    --log_folder "$output_dir" \
    --parallel_workers "$PARALLEL_WORKERS" \
    --temperature "$TEMPERATURE" \
    --experiment_type "$EXP_TYPE" \
    --debug \
    $prefix_args
  
  if [ $? -eq 0 ]; then
    echo "✅ Successfully processed $dataset_name"
  else
    echo "❌ Failed to process $dataset_name"
  fi
}

# ===================== main ===================== #
# Process datasets
if [ -f "data_processor/math_dataset/train/math_1000_20250414.json" ]; then
  process_dataset "math" "data_processor/math_dataset/train/math_1000_20250414.json"
fi

if [ -f "data_processor/math_dataset/train/math_medium_1000_20250430.json" ]; then
  process_dataset "math2" "data_processor/math_dataset/train/math_medium_1000_20250430.json"
fi

if [ -f "data_processor/qa_dataset/train/hotpotqa_1000_20250402.json" ]; then
  process_dataset "hotpotqa" "data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"
fi

echo ""
echo "🎉 All experiments completed!"
echo "📊 Results saved to: logs/qa_results/cloudflare/$MODEL_NAME"
