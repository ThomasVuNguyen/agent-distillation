#!/usr/bin/env python3
import argparse
import os
from typing import Optional

def main():
    parser = argparse.ArgumentParser(description="Serve a VLLM model with OpenAI-compatible API")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to serve (e.g., 'NousResearch/Meta-Llama-3-8B-Instruct')"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to serve the model on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve the model on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights (e.g., 'auto', 'float16', 'bfloat16')"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc",
        help="API key for authentication"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models"
    )
    parser.add_argument(
        "--generation-config",
        type=str,
        default="vllm",
        help="Generation config to use ('vllm' or 'hf')"
    )
    parser.add_argument(
        "--disable-log-requests",
        action="store_true",
        help="Disable request logging"
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable stats logging"
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        help="lora modeul in format {name}={path}"
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        help="maximum lora rank"
    )

    args = parser.parse_args()

    # Build the command
    cmd = ["vllm", "serve"]
    cmd.extend([args.model])
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])
    cmd.extend(["--dtype", args.dtype])
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.tensor_parallel_size:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.generation_config:
        cmd.extend(["--generation-config", args.generation_config])
    if args.disable_log_requests:
        cmd.append("--disable-log-requests")
    if args.disable_log_stats:
        cmd.append("--disable-log-stats")
    if args.lora_modules:
        cmd.extend(["--enable-lora", "--lora-modules", str(args.lora_modules)])
    if args.max_lora_rank:
        cmd.extend(["--max-lora-rank", str(args.max_lora_rank)])

    # Print the command that will be executed
    print("Executing command:", " ".join(cmd))
    
    # Execute the command
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main() 