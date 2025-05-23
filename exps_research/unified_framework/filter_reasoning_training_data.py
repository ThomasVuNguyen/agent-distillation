import argparse
import os
import json
from tqdm import tqdm

def main(args):
    output_path = args.output_path if args.output_path else args.result_path.replace("_scored.jsonl", "_filtered.jsonl")
    output_path = output_path.replace("/evaluations", "/filtered_data")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_path = args.result_path
    results = []
    with open(result_path) as f:
        for line in f:
            results.append(json.loads(line))
            # results = json.load(f)

    correct_questions = set()
    for result in results:
        if result["score"] == 1:
            correct_questions.add(result["question"])
    
    valid_results = []
    # Initialize counters
    incorrect_answer_count = 0
    error_parsing_count = 0 
    max_steps_count = 0
    valid_count = 0
    for result in results:
        error_flag = False
        
        if result["question"] not in correct_questions:
            incorrect_answer_count += 1
            continue
        valid_results.append(result)
        valid_count += 1
    
    print("Original log size:", len(results))
    print("Filtered log size:", len(valid_results))

    print(f"Incorrect answer count: {incorrect_answer_count}")
    print(f"Error parsing count: {error_parsing_count}")
    print(f"Max steps count: {max_steps_count}")
    print(f"Valid count: {valid_count}")

    print(output_path)
    with open(output_path, 'w', encoding="utf-8") as f:
        for entry in valid_results:
            f.write(json.dumps(entry) + '\n')
    print("Done")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to output filtered jsonl file. If not provided, will append '_filtered' to input filename")
    parser.add_argument("--result_path", type=str, 
        default="logs/qa_results/vllm/Qwen_Qwen2.5-32B-Instruct/evaluations/math_1000_20250414_train_Qwen2.5-32B-Instruct_temp=0.0_seed=42_type=reasoning_scored.jsonl",
        help="Path to evaluation results json file"
    )
    args = parser.parse_args()

    main(args)