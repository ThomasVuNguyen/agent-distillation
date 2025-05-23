set -e
set -x

MODEL=$1
DATASETS=(
    "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/hotpotqa_1000_20250402_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning_filtered.jsonl"
    "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_1000_20250414_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning_filtered.jsonl"
    "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_medium_1000_20250430_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.5_seed=42_type=reasoning_filtered.jsonl"
)
SAVENAME="qwen2.5_32B_teacher"
JOINED_DATASETS=$(IFS=" "; echo "${DATASETS[*]}")

sh exps_research/scripts_train/finetune_sft_cot.sh \
    $MODEL \
    "$JOINED_DATASETS" \
    $SAVENAME