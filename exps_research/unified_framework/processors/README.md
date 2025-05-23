# Experiment Processors

This folder contains modular experiment processors that handle different types of experiments in the unified framework.

## Overview

- **Base Processor (`base.py`)**: Abstract class that defines the interface and common functionality
- **Cost Tracker (`cost_tracker.py`)**: Singleton class for tracking experiment costs across threads
- **Reasoning Processor (`reasoning.py`)**: Handles direct QA/reasoning experiments
- **Agent Processor (`agent.py`)**: Handles agent-based experiments with tools
- **Factory (`factory.py`)**: Creates the appropriate processor based on experiment type

## How It Works

Each processor:
1. Handles a specific experiment type
2. Creates and manages its own models
3. Processes entries individually or in parallel
4. Tracks costs and results
5. Returns structured output

## Adding a New Processor

1. Create a new file like `my_experiment.py` that inherits from `ExperimentProcessor`:

```python
from .base import ExperimentProcessor

class MyExperimentProcessor(ExperimentProcessor):
    def process_entry(self, entry, model, **kwargs):
        # Implement your experiment-specific logic here
        # Process the entry with the model
        # Return a result dictionary
        ...
```

2. Register your processor in `factory.py`:

```python
def get_experiment_processor(experiment_type):
    processors = {
        "reasoning": ReasoningExperimentProcessor,
        "agent": AgentExperimentProcessor,
        "my_experiment": MyExperimentProcessor,  # Add your processor here
    }
    ...
```

3. Use your processor in the main experiment runner by specifying `experiment_type="my_experiment"`

## Common Customizations

- Override `create_model()` to change how models are instantiated
- Add experiment-specific parameters via `**kwargs` in `process_entry()`
- Extend `process_dataset()` for custom parallel processing

## Architecture Benefits

- Each experiment type is isolated in its own file
- Common functionality is shared through inheritance
- New experiment types don't require changes to existing code
- Support for both sequential and parallel processing