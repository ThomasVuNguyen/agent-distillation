"""
Unified script for running both agent and reasoning experiments
"""

import argparse
import json
import os
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Change from absolute imports to relative imports to avoid circular reference
from . import (
    setup_model,
    prepare_output_path,
    process_qa_experiment,
    score_qa_results,
    filter_agent_trajectories
)


def run_experiment():
    """Run an experiment with the unified framework - supports both agent and reasoning modes"""
    # Set up rich console if available
    if RICH_AVAILABLE:
        console = Console()

    parser = argparse.ArgumentParser(description="Run experiments on QA datasets using agent or reasoning approaches")

    # Common arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--model_type", type=str, default="openai", choices=["openai", "vllm"],
                      help="Type of model to use for generation")
    parser.add_argument("--model_id", type=str, help="Specific model ID to use (e.g., gpt-4o-mini, gpt-4o)")
    parser.add_argument("--parallel_workers", type=int, default=4, help="Maximum number of concurrent threads to use")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode without multi-threading")
    parser.add_argument("--track_cost", action="store_true", help="Track and display cost information for OpenAI models")
    parser.add_argument("--cost_threshold", type=float, help="Stop execution when total cost exceeds this amount (in USD)")
    parser.add_argument("--log_folder", type=str, help="Folder to save the results")
    parser.add_argument("--multithreading", action="store_true", help="Run in multithreading mode")
    parser.add_argument("--use_process_pool", action="store_true", help="Use ProcessPoolExecutor instead of ThreadPoolExecutor for more reliable timeouts")
    parser.add_argument("--use_single_endpoint", action="store_true", help="Use a single API endpoint (port 8000) for all workers instead of one per worker")
    parser.add_argument('--api_base', type=str, help='API base URL for the model')
    parser.add_argument('--api_key', type=str, help='API key for the model')
    parser.add_argument("--fine_tuned", action="store_true", help="whether using fine-tuned lora or not")
    parser.add_argument("--lora_folder", type=str, help="The folder for trained lora -- for saving corresponding logs")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during processing")
    parser.add_argument("--do_filtering", action="store_true", help="Save filtered trajectories also (for training dataset)")

    # Model args
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--max_output_tokens", type=int)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--use_local_model", action='store_true', help="Use local model for reasoning experiments")

    # Experiment type selection
    parser.add_argument("--experiment_type", type=str, choices=["agent", "reasoning"], required=True,
                      help="Type of experiment to run")
    parser.add_argument("--suffix", type=str, help="suffix for saved filename")

    # Agent-specific arguments
    parser.add_argument("--search_engine_type", type=str, default="wikipedia", help="Search engine for agent tool")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum number of steps for agent")
    parser.add_argument("--use_planning", action="store_true", help="Enable planning in agent")
    parser.add_argument("--prefix_memory", type=str, help="Path for prefix memory")
    parser.add_argument("--cot_memory", type=str, help="Path for CoT memory")

    # Reasoning-specific arguments
    parser.add_argument('--task_type', type=str, choices=["fact", "math"],
                      help="Task type for reasoning experiments")
    parser.add_argument("--add_think_token", action="store_true", help="Add think token for reasoning experiments")
    parser.add_argument("--use_rag", action="store_true", help="Use retrievd document using question as a query")

    args = parser.parse_args()

    # Display experiment setup with rich if available
    if RICH_AVAILABLE and args.verbose:
        console.rule("[bold blue]Experiment Setup")

        # Create configuration table
        config_table = Table(title="Experiment Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Experiment Type", args.experiment_type)
        config_table.add_row("Model Type", args.model_type)
        config_table.add_row("Model ID", args.model_id or "default")
        config_table.add_row("Dataset", os.path.basename(args.data_path))
        config_table.add_row("Debug Mode", "Yes" if args.debug else "No")
        config_table.add_row("Track Cost", "Yes" if args.track_cost else "No")
        config_table.add_row("Multithreading", "Yes" if args.multithreading else "No")
        config_table.add_row("Workers", str(args.parallel_workers if args.multithreading else 1))

        if args.experiment_type == "reasoning":
            config_table.add_row("Task Type", args.task_type)
            config_table.add_row("Add Think Token", "Yes" if args.add_think_token else "No")
        elif args.experiment_type == "agent":
            config_table.add_row("Search Engine", args.search_engine_type)
            config_table.add_row("Max Steps", str(args.max_steps))

        console.print(config_table)

    # Validate required arguments based on experiment type
    if args.fine_tuned and not args.lora_folder:
        parser.error("--lora_folder is required when --fine_tuned is set")

    if not args.task_type:
        # Auto-set task type for reasoning experiments if not provided
        if "math" in args.data_path:
            args.task_type = "math"
        elif "qa" in args.data_path:
            args.task_type = "fact"
        else:
            parser.error("--task_type is required for experiments when it cannot be inferred from data_path")

    # Set up log folder
    model_name = args.model_id.replace("/", "_") if args.model_id else "default"
    if args.lora_folder:
        args.log_folder = os.path.join(args.lora_folder, "qa_results")
    elif not args.log_folder:
        args.log_folder = f"logs/qa_results/{args.model_type}/{model_name}"

    # Create the log directory if it doesn't exist
    os.makedirs(args.log_folder, exist_ok=True)

    # Set up model parameters
    model_kwargs = {
        "model_type": args.model_type,
        "model_id": args.model_id,
        "fine_tuned": args.fine_tuned,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    if args.api_base:
        model_kwargs['api_base'] = args.api_base
    if args.api_key:
        model_kwargs['api_key'] = args.api_key
    if args.max_tokens:
        model_kwargs['max_tokens'] = args.max_tokens
    if args.max_output_tokens:
        model_kwargs['max_output_tokens'] = args.max_output_tokens
    if args.n:
        model_kwargs['n'] = args.n
    if args.top_p:
        model_kwargs['top_p'] = args.top_p
    if args.top_k:
        model_kwargs['top_k'] = args.top_k
    if args.lora_folder and args.use_local_model:
        model_kwargs['lora_folder'] = args.lora_folder

    # Additional experiment-specific args
    extra_kwargs = {}
    additional_postfix = []

    if args.experiment_type == "agent":
        extra_kwargs["search_engine_type"] = args.search_engine_type
        if args.search_engine_type != "wikipedia":
            additional_postfix.append("duckduckgo")
        extra_kwargs["max_steps"] = args.max_steps
        extra_kwargs["use_planning"] = args.use_planning
        if args.use_planning:
            additional_postfix.append("planning")

    elif args.experiment_type == "reasoning":
        extra_kwargs["add_think_token"] = args.add_think_token
        extra_kwargs["use_local_model"] = args.use_local_model
        extra_kwargs["task_type"] = args.task_type
        if args.use_rag:
            basepath = os.path.basename(args.data_path)
            rag_dirname = "data_processor/retrieved_documents"
            rag_path = os.path.join(rag_dirname, basepath)
            with open(rag_path, 'r') as f:
                retrieved_documents = json.load(f)
            extra_kwargs["retrieved_documents"] = retrieved_documents
            additional_postfix.append("rag")

    if args.prefix_memory and len(args.prefix_memory) > 0:
        with open(args.prefix_memory, 'r') as f:
            prefix_memory = json.load(f)
        extra_kwargs["prefix_memory"] = prefix_memory
        additional_postfix.append("prefix")

    if args.cot_memory:
        with open(args.cot_memory, 'r') as f:
            cot_memory = json.load(f)
        extra_kwargs["cot_memory"] = cot_memory
        additional_postfix.append("cot_memory")

    if args.suffix:
        additional_postfix.append(args.suffix)

    # Prepare output path
    paths = prepare_output_path(
        dataset_file=args.data_path,
        model_id=args.model_id or "default",
        log_folder=args.log_folder,
        lora_folder=args.lora_folder,
        temperature=args.temperature,
        n=args.n,
        seed=args.seed,
        max_steps=args.max_steps if args.experiment_type == "agent" else None,
        experiment_type=args.experiment_type,
        additional_postfix=additional_postfix
    )

    if RICH_AVAILABLE and args.verbose:
        console.print(f"[bold cyan]Output file:[/bold cyan] {paths['output_file']}")
    else:
        print(f"Output file: {paths['output_file']}")

    # Run experiment
    stats = process_qa_experiment(
        dataset_file=args.data_path,
        model_kwargs=model_kwargs,
        experiment_type=args.experiment_type,
        output_file=paths["output_file"],
        max_workers=args.parallel_workers if args.multithreading else 1,
        debug=args.debug,
        track_cost=args.track_cost,
        cost_threshold=args.cost_threshold,
        fine_tuned=args.fine_tuned,
        verbose=args.verbose,
        use_process_pool=args.use_process_pool,
        use_single_endpoint=args.use_single_endpoint,
        **extra_kwargs
    )

    # Print experiment summary
    if RICH_AVAILABLE:
        console.rule("[bold blue]Experiment Summary")

        # Create summary table
        summary_table = Table(title=f"{args.experiment_type.capitalize()} Experiment Results")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Data", str(args.data_path))
        summary_table.add_row("Total Questions", str(stats['total_questions']))
        summary_table.add_row("Processed Questions", str(stats['processed_questions']))
        summary_table.add_row("Model Type", args.model_type)
        summary_table.add_row("Model ID", args.model_id or "default")
        summary_table.add_row("Total Cost", f"${stats['costs']['total_cost']:.4f}")
        summary_table.add_row("Average Cost per Question", f"${stats['costs']['average_cost_per_question']:.4f}")

        console.print(summary_table)
    else:
        print(f"\nExperiment Summary:")
        print(f"Experiment Type: {args.experiment_type}")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Processed Questions: {stats['processed_questions']}")
        print(f"Model Type: {args.model_type}")
        print(f"Model ID: {args.model_id or 'default'}")
        print(f"Debug Mode: {'Yes' if args.debug else 'No'}")
        print(f"\nCost Summary:")
        print(f"Total Cost: ${stats['costs']['total_cost']:.4f}")
        print(f"Average Cost per Question: ${stats['costs']['average_cost_per_question']:.4f}")

    # Score results for reasoning experiments
    single_thread = args.task_type in ["math", "mmlu"]
    if args.experiment_type == "reasoning":
        do_extract_answer = args.task_type == "math"
    else:
        do_extract_answer = False

    output_file, score_stats = score_qa_results(
        paths["output_file"],
        max_workers=4,
        task_type=args.task_type,
        single_thread=single_thread,
        do_extract_answer=do_extract_answer
    )

    if RICH_AVAILABLE:
        accuracy_panel = Panel(
            f"[bold green]{score_stats['accuracy']:.2%}[/bold green]\n"
            f"[bold]{score_stats['correct_answers']}/{score_stats['total_questions']}[/bold] correct answers",
            title="Accuracy Results",
            border_style="green"
        )
        console.print(accuracy_panel)
    else:
        print(f"Accuracy: {score_stats['accuracy']:.2%}")
        print(f"Correct: {score_stats['correct_answers']}/{score_stats['total_questions']}")

    # Apply filtering if applicable
    if args.do_filtering:
        if args.experiment_type == "agent":
            filter_agent_trajectories(output_file)

if __name__ == "__main__":
    run_experiment()