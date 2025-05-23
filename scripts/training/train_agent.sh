set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
EPOCH=${2:-2}
DATASETS=(
  "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/hotpotqa_1000_20250402_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=agent_steps=5_filtered.jsonl"
  "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_1000_20250414_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=agent_steps=5_filtered.jsonl"
  "logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_medium_1000_20250430_train/filtered_data/Qwen2.5-32B-Instruct_temp=0.1_seed=42_type=agent_steps=5_filtered.jsonl"
)
SAVENAME="qwen2.5_32B_teacher"
JOINED_DATASETS=$(IFS=" "; echo "${DATASETS[*]}")

sh exps_research/scripts_train/finetune_sft_agent.sh \
    $MODEL \
    "$JOINED_DATASETS" \
    $SAVENAME \
    $EPOCH