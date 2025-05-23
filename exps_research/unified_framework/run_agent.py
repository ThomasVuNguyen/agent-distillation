"""
Run agent experiments using the unified framework
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import the unified experiment runner
from exps_research.unified_framework.run_experiment import run_experiment


def run_agent_experiment():
    """Compatibility wrapper for agent experiments"""
    # Inject experiment_type argument
    if "--experiment_type" not in sys.argv:
        sys.argv.extend(["--experiment_type", "agent"])
    
    # Default data_path if not provided
    if "--data_path" not in sys.argv:
        sys.argv.extend(["--data_path", "data_processor/qa_dataset/train/hotpotqa_1000_20250402.json"])
    
    # Call the unified experiment runner
    run_experiment()


if __name__ == "__main__":
    run_agent_experiment() 