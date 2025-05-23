"""
Utility functions for unified experiments
"""

import os
import json
import threading
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from fractions import Fraction  # This is where Rational comes from
import numpy as np  # To handle ndarray and numpy scalars
import sympy

# Global lock for thread-safe file writing
APPEND_ANSWER_LOCK = threading.Lock()


def calculate_cost(
    input_tokens: int, 
    output_tokens: int, 
    model_id: str = "gpt-4o"
) -> float:
    """
    Calculate cost based on model pricing
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model ID for pricing information
    
    Returns:
        Estimated cost in USD
    """
    # Model-specific pricing (as of 2024)
    pricing = {
        "gpt-4o-mini": {
            "input_cost_per_1m": 0.15,  # $0.15 per 1M input tokens
            "output_cost_per_1m": 0.6,  # $0.6 per 1M output tokens
        },
        "gpt-4o": {
            "input_cost_per_1m": 2.5,   # $2.5 per 1M input tokens
            "output_cost_per_1m": 10,   # $10 per 1M output tokens
        },
        # Add more models as needed
    }
    
    # Default to gpt-4o pricing if model not found
    model_pricing = pricing.get(model_id, pricing["gpt-4o"])
    
    input_cost = (input_tokens / 1000000) * model_pricing["input_cost_per_1m"]
    output_cost = (output_tokens / 1000000) * model_pricing["output_cost_per_1m"]
    
    return input_cost + output_cost


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from file
    
    Args:
        file_path: Path to dataset file
    
    Returns:
        List of examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)["examples"]


def get_answered_questions(file_path: str) -> Set[str]:
    """
    Get set of questions that have already been answered
    
    Args:
        file_path: Path to results file
    
    Returns:
        Set of already answered questions
    """
    answered_questions = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if type(entry) == str:
                        continue
                    answered_questions.add(entry["question"])
                except json.JSONDecodeError:
                    continue
    return answered_questions


def robust_serialize(obj):
    """
    Attempt to serialize an object to JSON. If it fails, convert to string.
    """
    try:
        return json.dumps(obj)
    except TypeError:
        try:
            return json.dumps(make_serializable(obj))
        except Exception:
            return json.dumps(str(obj))  # Absolute last resort

def make_serializable(obj):
    """
    Convert non-serializable types to serializable formats.
    """
    if isinstance(obj, set):
        return sorted(make_serializable(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, Fraction):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, sympy.Basic):
        return str(obj)
    else:
        return str(obj)

def append_result(result: Dict, output_file: str) -> None:
    """
    Append result to output file in thread-safe manner
    
    Args:
        result: Result dictionary to append
        output_file: Path to output file
    """
    safe_result = robust_serialize(result)
    with APPEND_ANSWER_LOCK, open(output_file, "a", encoding="utf-8") as fp:
        fp.write(safe_result + "\n")


def prepare_output_path(
    dataset_file: str, 
    model_id: str, 
    log_folder: str = None,
    lora_folder: str = None,
    temperature: float = 0.0,
    n: int = 1,
    seed: int = 42,
    max_steps: int = None,
    experiment_type: str = None,
    additional_postfix: list = [],
) -> Dict[str, Any]:
    """
    Prepare output path for experiment results
    
    Args:
        dataset_file: Path to dataset file
        model_id: Model ID
        log_folder: Base folder for logs
        lora_folder: Folder for trained lora (optional)
        temperature: Sampling temperature
        seed: Random seed
        max_steps: Maximum steps for agent (optional)
        experiment_type: Type of experiment (e.g., "reasoning", "agent")
    
    Returns:
        Dictionary with output paths and metadata
    """
    # Extract dataset name from path
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    fold = dataset_file.strip("/").split("/")[-2]
    dataset_name = f"{dataset_name}_{fold}"
    model_id_base = model_id.split("/")[-1]
    
    # Handle folder hierarchy
    if lora_folder:
        log_folder = os.path.join(lora_folder, "qa_results")
        output_dir = os.path.join(log_folder, dataset_name)
        # Create filename based on parameters
        filename_parts = [
            f"{model_id_base}",
            f"temp={temperature}",
            f"n={n}",
            f"seed={seed}"
        ]
    else:
        # If no lora folder is provided, use the default log folder inside the dataset folder
        output_dir = log_folder or os.path.join(os.path.dirname(dataset_file), "generated_answers")
        output_dir = os.path.join(output_dir, dataset_name)
        # Create filename based on parameters
        filename_parts = [
            f"{model_id_base}",
            f"temp={temperature}",
            f"seed={seed}"
        ]

    os.makedirs(output_dir, exist_ok=True)

    if experiment_type:
        filename_parts.append(f"type={experiment_type}")
    
    if max_steps:
        filename_parts.append(f"steps={max_steps}")

    if additional_postfix:
        filename_parts.extend(additional_postfix)
    
    filename = "_".join(filename_parts) + ".jsonl"
    output_file = os.path.join(output_dir, filename)
    print(" #### Output file:", output_file)
    
    return {
        "output_dir": output_dir,
        "output_file": output_file,
        "dataset_name": dataset_name,
        "model_id_base": model_id_base,
    } 