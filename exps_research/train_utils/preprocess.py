import json
import yaml
import importlib
import re
from datasets import Dataset
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from smolagents import WikipediaRetrieverTool, FinalAnswerTool
from smolagents.models import remove_tool_call_from_messages, get_clean_message_list
from smolagents.agents import populate_template

PROMPT_TEMPLATES = yaml.safe_load(
    importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
)

ROLE_CONVERSION_DICT = {
    "MessageRole.SYSTEM": "system",
    "MessageRole.USER": "user",
    "MessageRole.ASSISTANT": "assistant",
    "MessageRole.TOOL_CALL": "tool-call",
    "MessageRole.TOOL_RESPONSE": "tool-response",
}


def print_pretty_messages(messages):
    console = Console()
    role_styles = {
        "system": ("System", "cyan"),
        "user": ("User", "green"),
        "assistant": ("Assistant", "magenta"),
    }

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        role_label, color = role_styles.get(role, ("Unknown", "red"))

        title_text = Text(f"{role_label}", style=f"bold {color}")
        content_text = Text(content, style="white")

        panel = Panel(content_text, title=title_text, border_style=color)
        console.print(panel)

# Only used for CoT models
def load_prompt() -> str:
    """Load the system prompt from YAML file"""
    prompt_file = Path(__file__).parent.parent.parent / "src" / "smolagents" / "prompts" / "teacher_model.yaml"
    with open(prompt_file, 'r') as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['system_prompt']

def clean_roles(messages):
    for message in messages:
        if message["role"] in ROLE_CONVERSION_DICT.keys():
            message["role"] = ROLE_CONVERSION_DICT[message["role"]]
    return messages

def remove_reference_tags(text):
    # remove every contents in <reference> ... </reference>
    return re.sub(r'<reference>.*?</reference>', '', text, flags=re.DOTALL)

def clean_user_message(messages):
    user_message = messages[1]["content"]
    messages[1]["content"] = remove_reference_tags(user_message)
    return messages

def check_two_system_messages(messages):
    n_systems = 0
    for message in messages:
        role = message["role"]
        assert role in ['user', 'assistant', 'system', 'tool-call', 'tool-response']
        if role == "system":
            n_systems += 1

    if n_systems > 1:
        return True
    else:
        return False

def preprocess_sft_dataset(solution_type, datapath):
    if solution_type in ["cot", "reasoning"]:
        dataset = preprocess_cot_dataset(datapath)
    else:
        dataset = preprocess_logs(datapath)
    return dataset

def preprocess_cot_dataset(datapath):
    system_prompt = load_prompt()
    dataset = []
    with open(datapath) as f:
        for line in f:
            dataset.append(json.loads(line))

    processed_dataset = []
    for data in dataset:
        response = data["response"]
        messages = data['messages']
        messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        processed_dataset.append({"messages": messages})
    processed_dataset = Dataset.from_list(processed_dataset)
    return processed_dataset

def remove_last_user_message(
    messages
):
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    return messages

# Preprocess logs to messages only
def preprocess_logs(log_path):
    prompt_template = PROMPT_TEMPLATES["system_prompt_short"]

    tools = [WikipediaRetrieverTool()]
    tools = {tool.name: tool for tool in tools}
    tools.setdefault("final_answer", FinalAnswerTool())

    system_prompt = populate_template(
        prompt_template,
        variables={
            "tools": tools
        }
    )

    logs = []
    with open(log_path) as f:
        for line in f:
            logs.append(json.loads(line))

    dataset = []
    n_planning = 0
    for i, log in enumerate(logs):
        if not log["log_data"]:
            continue
        messages = log["log_data"]["messages"]
        messages = clean_roles(messages)
        is_two_system = check_two_system_messages(messages)
        if is_two_system:
            print("Two system message in messages detected!")
            continue
        messages = remove_tool_call_from_messages(messages)
        messages = get_clean_message_list(
            messages,
            role_conversions={
                "tool-response": "user",
                "tool-call": "assistant"
            },
            flatten_messages_as_text=True,
        )
        messages = clean_user_message(messages)
        messages = remove_last_user_message(messages)
        messages[0]["content"] = system_prompt

        dataset.append({"messages": messages})
        if i == 0:
            print_pretty_messages(messages)

        # Append additional messages
        if len(log["log_data"]["original_memory"]["steps"]) > 1:
            steps = log["log_data"]["original_memory"]["steps"]
            for j in range(1, len(steps)):
                step = steps[j]
                input_messages = step["model_input_messages"]
                output_message = step["model_output_message"]
                additional_messages = input_messages + [output_message]
                dataset.append({"messages": additional_messages})
                n_planning += 1

    print("##### Planning data", n_planning)
    dataset = Dataset.from_list(dataset)
    return dataset
