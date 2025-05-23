"""
Run reasoning experiments using the unified framework
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import the unified experiment runner
from exps_research.unified_framework.run_experiment import run_experiment


def run_reasoning_experiment():
    """Compatibility wrapper for reasoning experiments"""
    # Inject experiment_type argument
    if "--experiment_type" not in sys.argv:
        sys.argv.extend(["--experiment_type", "reasoning"])
    
    # Require data_path since it's needed for reasoning tasks
    if "--data_path" not in sys.argv:
        # If data_path is not provided, let the unified script handle the required argument
        pass
    
    # Call the unified experiment runner
    run_experiment()


if __name__ == "__main__":
    run_reasoning_experiment() 