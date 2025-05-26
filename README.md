# Agent Distillation

<p align="center">
  <img src="images/agent_distillation_entry.png" alt="Alt text" width="350"/>
</p>

`agent-distillation` is a library that supports **distillation** of large language agents into small langauge models, with just a few scripts!

This library accompanies our academic paper, [**Distilling LLM Agents into Small Models with Retrieval and Code Tools**](https://arxiv.org/abs/2505.17612), where we demonstrate how small language models can learn to act like powerful LLM agents by mimicking their agentic behaviors, augmented with retrieval and code execution capabilities.

Built on top of [`smolagents` v1.13.0.dev0](https://github.com/huggingface/smolagents), this library supercharges the agent training pipeline with essential utilities for logging, training, and benchmarking, all optimized for simplicity and reproducibility.

## 🔧 What This Library Offers

In addition to the powerful capabilities of `smolagents`, this library introduces:

1. 📜 **Logging**: Seamlessly save agent run logs to create training-ready trajectories.
2. 🎓 **Training**: Use [TRL](https://github.com/huggingface/trl)'s SFT trainer to train small agents that remain compatible with `smolagents`.
3. 📊 **Benchmarking**: Evaluate your distilled agents on factual and mathematical reasoning benchmarks using a single script.

## Recent Updates
- [2025.05] We open-source the Agent Distillation codebase.

## 📦 Contents

1. [Installation](#installation)
2. [Quickstart: How to Distill Agents](#quickstart-how-to-distill-agents)
3. [Acknowledgements](#acknowledgements)


## 🛠 Installation

To install with the required libraries:

```bash
conda create -n agents python=3.12
conda activate agents
pip install -e .[distill]
```

> Note: If you want to run benchmarking, place your OpenAI API key in a file at `keys/openai-key/key.env`. This is required for LLM-as-a-judge evaluation on factual reasoning benchmarks.

### ➕ Optional: Retriever Environment (used in our paper)

Want to reproduce or extend our retriever-enhanced setup? We follow the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) environment.

Expand the section below for setup instructions.
<details>
<summary>Open for the detailed setup guideline.</summary>

1. Make a conda environment for the retriever.

```bash
conda create -n retriever python=3.10
conda activate retriever
```

2. Install related libraries.

```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

3. Save the index and corpus from the repo.

```bash
save_path=./search/database/wikipedia
mkdir -p $save_path
python scripts/download.sh --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

</details>

## ⚗️ Quickstart: How to Distill Agents

All scripts assume access to 4 GPUs.

1. 🧪 Generate Trajectories from Teacher Agent

```bash
bash scripts/inference/run_agent_teacher_train.sh
```

2. 🎓 Train the Student Agent

```bash
bash scripts/training/train_agent.sh Qwen/Qwen2.5-1.5B-Instruct
```

3. ✅ Evaluate the Trained Agent on Benchmarks

Runs with self-consistent action generation enabled by default:

```bash
bash scripts/inference/run_agent_student.sh Qwen/Qwen2.5-1.5B-Instruct training_outputs/qwen-1.5B-instruct/agent_baseline_qwen2.5_32B_teacher
```

Or test manually:

```bash
bash scripts/inference/serve_slm.sh
# In a separate terminal:
python examples/test_small_agent.py
```

### More on `smolagents`

Curious about more capabilities? Check out the [original smolagents repository](https://github.com/huggingface/smolagents) for advanced usage and custom environments.

## 🚧 Future Plan

- [ ] Release teacher trajectories and distilled small LMs as baselines.
- [ ] Add detailed instructions for first-thought prefix.
- [ ] Provide utilities for small LMs to use tools via MCP.

## 🙏 Acknowledgements

This project is made possible by the foundational work of the following open-source libraries:

- [**smolagents**](https://github.com/huggingface/smolagents): Provides the core framework for building and running lightweight language agents, which we extend for distillation.

- [**Search-R1**](https://github.com/PeterGriffinJin/Search-R1): Supplies a dense retrieval environment used in our retriever-based experiments.

- [**TRL**](https://github.com/huggingface/trl): Offers the supervised fine-tuning framework we use to train distilled agents effectively.

We sincerely thank the developers and maintainers of these projects.

## ⚠️ Disclaimer
This is not an official product of KRAFTON Inc. or DeepAuto.ai. It is released solely for research purposes.
