#!/bin/bash

set -e
set -x

model=$1
datapath=$2
postfix=${3:-"simple"}
epoch=${4:-"1"}

torchrun --nproc_per_node=4 exps_research/finetune_sft.py \
    --model_name $model \
    --num_epochs $epoch \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr 2e-4 \
    --train_filepath $datapath \
    --postfix $postfix \
    --solution_type agent \
    --fsdp exps_research/mp_configs/fsdp.json \
    --max_length 10240