import argparse
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

ERR_MSG1 = "Error in code parsing"
ERR_MSG2 = "Reached max steps" # Maybe this is not an error
ERR_MSG3 = "Code execution failed"

def filter_agent_trajectories(result_path, do_save=True):
    output_path = result_path.replace("_scored.jsonl", "_filtered.jsonl")
    output_path = output_path.replace("/evaluations", "/filtered_data")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("Save filtered data to", output_path)
    results = []
    with open(result_path) as f:
        for line in f:
            results.append(json.loads(line))
            # results = json.load(f)

    valid_results = []

    # Initialize counters with more detailed categorization
    stats = {
        "total": len(results),
        "correct_answers": 0,
        "incorrect_answers": 0,
        "error_parsing": 0,
        "error_parsing_correct": 0,
        "error_execution": 0,
        "error_execution_correct": 0,
        "max_steps": 0,
        "max_steps_correct": 0,
        "empty_log": 0,
        "valid_entries": 0
    }

    for result in results:
        if not result["log_data"]:
            stats["empty_log"] += 1
            continue

        is_correct = result["score"] == 1
        if is_correct:
            stats["correct_answers"] += 1
        else:
            stats["incorrect_answers"] += 1

        messages = result["log_data"]["messages"]
        error_type = None

        # Check for various error types
        for message in messages:
            if message["role"] == "tool-response":
                if ERR_MSG1 in message["content"][0]["text"]:
                    error_type = "parsing"
                    stats["error_parsing"] += 1
                    if is_correct:
                        stats["error_parsing_correct"] += 1
                    break

                if ERR_MSG2 in message["content"][0]["text"]:
                    error_type = "max_steps"
                    stats["max_steps"] += 1
                    if is_correct:
                        stats["max_steps_correct"] += 1
                    break

                if ERR_MSG3 in message["content"][0]["text"]:
                    # error_type = "execution"
                    stats["error_execution"] += 1
                    if is_correct:
                        stats["error_execution_correct"] += 1
                    # break
                    # Keep the code error for reflection and further revise

        # Keep result if it's correct and has no errors
        if is_correct and not error_type:
            valid_results.append(result)
            stats["valid_entries"] += 1

    print(f"Original log size: {stats['total']}")
    print(f"Filtered log size: {stats['valid_entries']}")
    print(f"Output path: {output_path}")
    print("\nDetailed statistics:")
    print(f"Correct answers: {stats['correct_answers']}")
    print(f"Incorrect answers: {stats['incorrect_answers']}")
    print(f"Empty logs: {stats['empty_log']}")
    print("\nError categories:")
    print(f"Error parsing: {stats['error_parsing']} (with correct answers: {stats['error_parsing_correct']})")
    print(f"Error execution: {stats['error_execution']} (with correct answers: {stats['error_execution_correct']})")
    print(f"Max steps: {stats['max_steps']} (with correct answers: {stats['max_steps_correct']})")
    print(f"Valid entries (correct with no errors): {stats['valid_entries']}")

    if do_save:
        with open(output_path, 'w', encoding="utf-8") as f:
            for entry in valid_results:
                f.write(json.dumps(entry) + '\n')


def main(args):
    filter_agent_trajectories(args.result_path, args.do_save)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to output filtered jsonl file. If not provided, will append '_filtered' to input filename")
    parser.add_argument("--result_path", type=str, default="logs/qa_results/openai/gpt-4o/evaluations/evaluation_summary_20250402_152239.json",
                      help="Path to evaluation results json file")
    parser.add_argument("--do_save", action='store_true')
    parser.add_argument("--visualize", action='store_true',
                      help="Generate visualization figures of the filtering results")
    args = parser.parse_args()

    main(args)