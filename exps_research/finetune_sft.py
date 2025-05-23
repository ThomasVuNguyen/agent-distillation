import os
import sys
sys.path.append(".")
import json
import torch
import random
from datetime import datetime

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AutoPeftModelForCausalLM
)

import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    DataCollatorForCompletionOnlyLM
)

from train_utils.preprocess import (
    preprocess_sft_dataset,
)
from train_utils.utils import DataCollatorForCompletionOnlyLMMultiTurn

MODEL_IDENTIFIERS = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama-1B-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3B-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-8B-instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-0.5B-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-1.5B-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-3B-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7B-instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "qwen-coder-1.5B-instruct",
}

def setup_savedir(args):
    # Step 3-1: Setup save dir
    if "training_outputs" in args.model_name:
        # Extract model_identifier from the path
        path_parts = args.model_name.split('/')

        # Find the part that might be a model identifier
        for part in path_parts:
            # Check if this part is a value in MODEL_IDENTIFIERS
            for model_name, identifier in MODEL_IDENTIFIERS.items():
                if part == identifier:
                    model_identifier = part
                    break

            # If we found a match, break out of the outer loop
            if 'model_identifier' in locals():
                break

        # If no match was found in the path, use a default
        if 'model_identifier' not in locals():
            # Try to infer from the directory structure
            if len(path_parts) >= 3 and path_parts[-3] == "training_outputs":
                model_identifier = path_parts[-2]
            else:
                model_identifier = "qwen-7B-instruct"  # Default fallback
    else:
        model_identifier = MODEL_IDENTIFIERS.get(args.model_name)
        if model_identifier is None:
            raise NotImplementedError

    print(f"Model: {args.model_name}")

    if args.exp_id:
        exp_id = f"{args.solution_type}_{args.exp_id}"
    else:
        exp_id = f"{args.solution_type}_baseline"
        if args.num_epochs > 1:
            exp_id += f"_{args.num_epochs}epochs"
        if args.full_finetuning:
            exp_id += "_full"
        if len(args.postfix) > 0:
            if args.postfix.startswith("_"):
                exp_id += args.postfix
            else:
                exp_id += "_" + args.postfix

    output_dir = f"./training_outputs/{model_identifier}/{exp_id}"
    print("Output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata = vars(args)
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return output_dir
def main(args):
    # Set Seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if "SmolLM" in args.model_name:
        layer_name = "LlamaDecoderLayer"
    elif "Llama" in args.model_name:
        layer_name = "LlamaDecoderLayer"
    elif "Qwen2" in args.model_name:
        layer_name = "Qwen2DecoderLayer"
    else:
        layer_name = "Qwen2DecoderLayer"

    if args.model_name not in MODEL_IDENTIFIERS.keys() and "training_outputs" not in args.model_name:
        import pdb; pdb.set_trace()
        args.model_name = "meta-llama/Llama-3.2-1B-Instruct"

    if not args.full_finetuning:
        if args.peft_name:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
            model = PeftModel.from_pretrained(
                model,
                args.peft_name,
                is_trainable=True
            )
        else:
            model = None

        lora_r = 64
        lora_alpha = lora_r * 2
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        peft_config = None

    print("peft config", peft_config)

    ########## Setup model done ###############

    # Step 2: Setup dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, pad_token='<|endoftext|>', padding_side='left', add_eos_token=True)

    # Load dataset
    train_dataset = None
    for _train_filepath in args.train_filepath:
        _train_dataset = preprocess_sft_dataset(args.solution_type, _train_filepath)
        if train_dataset:
            train_dataset = concatenate_datasets([train_dataset, _train_dataset])
        else:
            train_dataset = _train_dataset
    if args.cot_filepath:
        _train_dataset = preprocess_sft_dataset("cot", args.cot_filepath)
        train_dataset = concatenate_datasets([train_dataset, _train_dataset]) # Add this

    if args.valid_filepath is not None:
        eval_dataset = preprocess_sft_dataset(args.solution_type, args.valid_filepath)
    else:
        eval_dataset = None

    if args.dataset_size > 0:
        train_dataset = train_dataset[:args.dataset_size]

    data_module = {
        "train_dataset": train_dataset
    }

    print("# Train Dataset: ", len(data_module["train_dataset"]))
    if "eval_dataset" in data_module.keys():
        print("# Valid Dataset: ", len(data_module["eval_dataset"]))
        eval_strategy = "epoch"
        save_strategy = "epoch"
        load_best_model_at_end = True
    else:
        eval_strategy = "no"
        save_strategy = "no"
        load_best_model_at_end = False

    output_dir = setup_savedir(args)
    ########## Setup dataset done ###############

    batch_size = args.batch_size
    # Step 3: Train
    train_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp is not None,
        fsdp_config=args.fsdp,
        # Strategy
        logging_steps=10,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        output_dir=output_dir,
        load_best_model_at_end=load_best_model_at_end,
        gradient_checkpointing=args.gradient_checkpointing,
        save_safetensors=False,
        model_init_kwargs={"attn_implementation": "flash_attention_2"}
    )

    # only for Qwen
    response_template = "<|im_start|>assistant"
    instruction_template = "<|im_start|>user"

    if args.solution_type == "agent":
        collator = DataCollatorForCompletionOnlyLMMultiTurn(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer
        )

    trainer = SFTTrainer(
        args.model_name if not model else model,
        args=train_args,
        peft_config=peft_config,
        data_collator=collator,
        **data_module
    )
    trainer.train()
    ########## Train done ###############

    # Step 4: Save best model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
        default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--peft_name", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument("--solution_type", type=str, default="agent", choices=["cot", "reasoning", "agent"])

    parser.add_argument(
        "--train_filepath",
        type=str,
        default="logs/qa_results/openai/gpt-4o/hotpotqa_1000_20250402_20250402.jsonl",
        nargs='+'
    )
    parser.add_argument(
        "--cot_filepath",
        type=str,
        help="Additional CoT dataset in agent training"
    )
    parser.add_argument("--valid_filepath", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)

    # Deepspeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)

    args = parser.parse_args()

    main(args)