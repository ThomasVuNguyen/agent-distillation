#!/bin/bash

# Activate virtual environment
source myenv/bin/activate

# ===================== user setting ===================== #
MODEL_ID="@cf/meta/llama-3.1-70b-instruct"  # Use the corrected model ID format
EXP_TYPE="agent"  # Choose "agent" or "reasoning"

# Datasets configuration
declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"
  ["math"]="data_processor/math_dataset/train/math_1000_20250414.json"
  ["math2"]="data_processor/math_dataset/train/math_medium_1000_20250430.json"
)

# Settings for the API call
MAX_TOKENS=1024
TEMPERATURE=0.0
PARALLEL_WORKERS=4

# Optional: Use prefix memory (first-thought prefix)
USE_PREFIX=false

# Update paths as needed if using prefix memory
declare -A PREFIXS=(
  ["hotpotqa"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/hotpotqa_1000_20250402_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning.json"
  ["math"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_1000_20250414_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning.json"
  ["math2"]="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_medium_1000_20250430_train/prefix_memory/Qwen2.5-32B-Instruct_temp=0.5_seed=42_type=reasoning.json"
)
# ===================================================== #

# Parse command line args
for arg in "$@"; do
  case $arg in
    --use-prefix)
      USE_PREFIX=true
      ;;
    --model=*)
      MODEL_ID="${arg#*=}"
      ;;
    --temperature=*)
      TEMPERATURE="${arg#*=}"
      ;;
    --workers=*)
      PARALLEL_WORKERS="${arg#*=}"
      ;;
  esac
done

# Create output directory
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g' | sed 's/@cf_//')
OUTPUT_BASE="logs/qa_results/cloudflare/$MODEL_NAME"
mkdir -p "$OUTPUT_BASE"

echo "🚀 Starting CloudflareWorkersAI inference with model: $MODEL_ID"
echo "⚙️  Experiment type: $EXP_TYPE"
echo "🌡️  Temperature: $TEMPERATURE"
echo "👥 Parallel workers: $PARALLEL_WORKERS"

# Process each dataset
for dataset_name in "${!DATASETS[@]}"; do
  dataset_path="${DATASETS[$dataset_name]}"
  output_dir="$OUTPUT_BASE/${dataset_name}_$(date +%Y%m%d)"
  mkdir -p "$output_dir"

  echo ""
  echo "📊 Processing dataset: $dataset_name"
  echo "📂 Dataset path: $dataset_path"
  echo "💾 Output directory: $output_dir"
  
  prefix_args=""
  if [ "$USE_PREFIX" = true ] && [ -n "${PREFIXS[$dataset_name]}" ]; then
    prefix_path="${PREFIXS[$dataset_name]}"
    if [ -f "$prefix_path" ]; then
      prefix_args="--prefix_file $prefix_path"
      echo "🔖 Using prefix memory from: $prefix_path"
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
    --max_tokens "$MAX_TOKENS" \
    --experiment_type "$EXP_TYPE" \
    --verbose \
    $prefix_args

  # Check if the experiment was successful
  if [ $? -eq 0 ]; then
    echo "✅ Successfully processed $dataset_name"
  else
    echo "❌ Failed to process $dataset_name"
  fi
done

echo ""
echo "🎉 All experiments completed!"
echo "📊 Results saved to: $OUTPUT_BASE"
