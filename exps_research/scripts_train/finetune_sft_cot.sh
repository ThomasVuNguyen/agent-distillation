#!/bin/bash

set -e
set -x

model=$1
datapath=$2
postfix=${3:-"simple"}

torchrun --nproc_per_node=4 exps_research/finetune_sft.py \
    --model_name $model \
    --num_epochs 2 \
    --batch_size 2 \
    --max_length 4096 \
    --lr 2e-4 \
    --train_filepath $datapath \
    --postfix $postfix \
    --solution_type cot \
    --fsdp exps_research/mp_configs/fsdp.json