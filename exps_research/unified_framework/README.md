# Unified Experiment Framework

This module provides a unified framework for running both reasoning-based and agent-based experiments on question-answering datasets.

## Structure

- `models.py`: Common model setup code for different model types (OpenAI, VLLM)
- `utils.py`: Shared utility functions for cost calculation, file handling, etc.
- `experiment.py`: Core experiment execution logic with support for both reasoning and agent-based experiments
- `run_reasoning.py`: Main script for running reasoning experiments
- `run_agent.py`: Main script for running agent experiments

## Usage

### Reasoning Experiments

```bash
python -m exps_research.unified_framework.run_reasoning \
  --data_path data_processor/qa_dataset/train/hotpotqa_1000.json \
  --model_type openai \
  --model_id gpt-4o-mini \
  --parallel_workers 4 \
  --multithreading \
  --temperature 0.0 \
  --seed 42 \
  --track_cost
```

### Agent Experiments

```bash
python -m exps_research.unified_framework.run_agent \
  --data_path data_processor/qa_dataset/train/hotpotqa_1000.json \
  --model_type openai \
  --model_id gpt-4o-mini \
  --parallel_workers 8 \
  --multithreading \
  --search_engine_type wikipedia \
  --max_steps 5 \
  --temperature 0.0 \
  --seed 42 \
  --track_cost
```

## Common Parameters

Both experiment types support the following common parameters:

- `--data_path`: Path to the dataset file
- `--model_type`: Type of model to use (`openai` or `vllm`)
- `--model_id`: Model ID to use (e.g., `gpt-4o-mini`, `gpt-4o`)
- `--parallel_workers`: Maximum number of concurrent threads to use
- `--multithreading`: Run in multithreading mode
- `--temperature`: Sampling temperature (default: 0.0)
- `--seed`: Random seed (default: 42)
- `--debug`: Run in debug mode with limited questions
- `--track_cost`: Track and display cost information
- `--cost_threshold`: Stop execution when total cost exceeds this amount (in USD)
- `--fine_tuned`: Whether using a fine-tuned model
- `--lora_folder`: The folder for trained LoRA weights and logs
- `--log_folder`: Base folder for storing results

## Experiment-specific Parameters

### Agent Experiments
- `--search_engine_type`: Type of search engine tool to use (`wikipedia` or `duckduckgo`)
- `--max_steps`: Maximum number of steps for the agent (default: 5)

## Extending the Framework

To add support for new experiment types or models:

1. Add any new model types to `models.py`
2. Create new processing functions in `experiment.py` 
3. Update the common utilities in `utils.py` as needed
4. Create a new run script for your experiment type

## Benefits of the Unified Framework

- **Code Reuse**: Common functionality shared between experiment types
- **Consistent Interface**: Similar parameters and output formats
- **Maintainability**: Easier to maintain and extend with modular design
- **Parallel Execution**: Built-in support for multithreading
- **Cost Tracking**: Unified cost calculation and tracking 