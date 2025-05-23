'''
This code builds prefix memory based on the CoT completions - which will be used in agent.
'''

import argparse
import os
import json
from tqdm import tqdm

def open_jsonl(datapath):
    dataset = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def main(args):
    cot_logs = open_jsonl(args.cot_log_path)
    
    dirname = os.path.dirname(args.cot_log_path)
    basename = os.path.basename(args.cot_log_path)
    savedir = os.path.join(dirname, "prefix_memory")
    os.makedirs(savedir, exist_ok=True)
    save_basename = basename.replace(".jsonl", ".json")

    q_to_prefix = dict()
    for cot_log in tqdm(cot_logs):
        question = cot_log["question"]
        first_prefix = cot_log["response"].split("\n\n")[0] + '\n\n'
        q_to_prefix[question] = "Thought: " + first_prefix
    
    savepath = os.path.join(savedir, save_basename)

    with open(savepath, 'w') as f:
        json.dump(q_to_prefix, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cot_log_path", 
        type=str,
        default="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/math_1000_20250414_train/Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning.jsonl"
    )
    args = parser.parse_args()

    main(args)