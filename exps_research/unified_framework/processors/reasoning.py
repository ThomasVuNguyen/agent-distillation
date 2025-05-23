"""
Reasoning experiment processor for direct QA experiments
"""

import re
import time
from typing import Dict, Any
from copy import deepcopy
from collections import Counter

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .base import ExperimentProcessor
from .qwen_math_parser import extract_answer
from exps_research.unified_framework.utils import calculate_cost


def find_majority_and_first_index(lst):
    # count frequency of elements
    count = Counter(lst)
    # find the most frequent value
    max_count = max(count.values())
    majority_candidates = [key for key, val in count.items() if val == max_count]

    for idx, val in enumerate(lst):
        if val in majority_candidates:
            return val, idx

class ReasoningExperimentProcessor(ExperimentProcessor):
    """
    Processor for reasoning experiments

    This processor handles direct reasoning-based QA experiments,
    where the model is given a question and expected to respond
    with an answer directly.
    """

    def __init__(self, model_kwargs: Dict[str, Any], **kwargs):
        """Initialize the reasoning experiment processor with a rich console if available"""
        super().__init__(model_kwargs, **kwargs)
        if RICH_AVAILABLE:
            self.console = Console()

    def process_entry(self, entry: Dict, model, **kwargs) -> Dict:
        """
        Process a reasoning experiment entry

        Args:
            entry: Dictionary containing a question
            model: Model instance
            **kwargs: Additional parameters including:
                - system_prompt: System prompt to provide context to the model
                - extract_answer_tags: Whether to extract answers from <answer> tags
                - add_think_token: Whether to add <think> token to prompts
                - verbose: Whether to print verbose output
                - verbose_worker: Whether this worker should display verbose output

        Returns:
            Processed result dictionary
        """
        if 'error' in entry:
            return None

        # Get experiment parameters
        system_prompt = kwargs.get('system_prompt')
        extract_answer_with_tags = kwargs.get('extract_answer_tags', True)
        add_think_token = kwargs.get('add_think_token', False)
        prefix_memory = kwargs.get('prefix_memory', None)
        retrieved_documents = kwargs.get('retrieved_documents', None)
        task_type = kwargs.get('task_type', None)

        # Determine if this worker should show verbose output
        verbose = self.verbose
        verbose_worker = kwargs.get('verbose_worker', True)
        should_show_output = verbose and verbose_worker

        if should_show_output:
            if RICH_AVAILABLE:
                self.console.rule(f"[bold blue]Processing Question")
                self.console.print(Panel(entry['question'], title="Question", border_style="green"))
            else:
                print(f"\n\n--- Processing question: {entry['question']} ---")

        if add_think_token:
            user_prompt = entry['question'] + '\n<think>'
        else:
            user_prompt = entry['question']

        if retrieved_documents:
            question = entry['question']
            _retrieved_document = retrieved_documents[question]
            user_prompt = _retrieved_document + "\n" + question

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        if should_show_output:
            if RICH_AVAILABLE:
                self.console.print("[bold cyan]System prompt:[/bold cyan]", f"{system_prompt[:100]}...")
                self.console.print("[bold cyan]User prompt:[/bold cyan]", f"{user_prompt}")
            else:
                print(f"System prompt: {system_prompt[:100]}...")
                print(f"User prompt: {user_prompt}")

        # Try up to 5 times to get properly formatted response
        for attempt in range(5):
            try:
                if should_show_output:
                    if RICH_AVAILABLE:
                        self.console.print(f"[yellow]Attempt {attempt+1} to get response...[/yellow]")
                    else:
                        print(f"Attempt {attempt+1} to get response...")

                if prefix_memory:
                    prefix = prefix_memory.get(entry["question"]) # It must be a dictionary
                    prefix = prefix.replace("Thought: ", "")
                    response = model(messages=messages, prefix=prefix)
                else:
                    response = model(messages=messages)

                if type(response) == list:
                    content = [_r.content for _r in response]
                else:
                    content = response.content

                if should_show_output:
                    if type(content) == list:
                        _content = content[0]
                    else:
                        _content = content
                    if RICH_AVAILABLE:
                        self.console.print(f"[green]Got response of length {len(_content)}[/green]")
                        syntax = Syntax(_content[:300] + "...", "markdown", theme="monokai", line_numbers=False)
                        self.console.print(Panel(syntax, title="Response Preview", border_style="blue"))
                    else:
                        print(f"Got response of length {len(_content)}")
                        print(f"Response preview: {_content[:2000]}...")

            except Exception as e:
                if RICH_AVAILABLE and should_show_output:
                    self.console.print(f"[bold red]Error in attempt {attempt+1}:[/bold red] {str(e)}")
                else:
                    print(f"Error in attempt {attempt+1}: {str(e)}")
                time.sleep(1)  # Brief pause before retrying
                continue

            if extract_answer_with_tags:
                if type(content) == str:
                    content = [content]

                all_answers = []
                all_explanations = []
                for _content in content:
                    # Extract answer between tags
                    answer_match = re.search(r'<answer>(.*?)</answer>', _content, re.DOTALL)

                    if answer_match:
                        answer = answer_match.group(1).strip()
                        explanation = re.sub(r'<answer>.*?</answer>', '', _content, flags=re.DOTALL).strip()

                        if should_show_output:
                            if RICH_AVAILABLE:
                                self.console.print(Panel(answer, title="[bold green]Extracted Answer[/bold green]", border_style="green"))
                                if explanation:
                                    self.console.print(Panel(explanation[:150] + "...", title="[bold]Explanation[/bold]", border_style="blue"))
                            else:
                                print(f"Extracted answer: {answer}")
                                print(f"Extracted explanation: {explanation[:150]}...")
                    else:
                        answer = _content
                        explanation = ""

                    all_answers.append(answer)
                    all_explanations.append(answer)
            else:
                all_answers = []
                all_explanations = []
                for _content in content:
                    answer = content
                    explanation = ""
                    all_answers.append(answer)
                    all_explanations.append(answer)

                if should_show_output:
                    if RICH_AVAILABLE:
                        self.console.print("[yellow]Using full response as answer (not extracting tags)[/yellow]")
                    else:
                        print("Using full response as answer (not extracting tags)")

            # Do majority voting here
            if task_type == "math":
                # In case of math, extract answer again
                parsed_answers = []
                for _answer in all_answers:
                    if "\boxed" not in _answer and len(_answer.split("\n\n")) == 1:
                        _answer = "\boxed{" + _answer + "}"
                    _answer = extract_answer(_answer)
                    parsed_answers.append(_answer)
                all_answers = parsed_answers

            if type(response) != list:
                final_answer = all_answers[0]
                final_explanation = all_explanations[0]
                final_response = response.content
            else:
                final_answer, final_index = find_majority_and_first_index(all_answers)
                final_explanation = all_explanations[final_index]
                final_response = response[final_index].content

            # Get token counts and calculate cost
            token_counts = model.get_token_counts()
            cost = calculate_cost(
                token_counts["input_token_count"],
                token_counts["output_token_count"],
                model.model_id
            )

            if should_show_output:
                if RICH_AVAILABLE:
                    self.console.print("[bold]Token Usage:[/bold]")
                    self.console.print(f"  [blue]Input tokens:[/blue] {token_counts['input_token_count']}")
                    self.console.print(f"  [blue]Output tokens:[/blue] {token_counts['output_token_count']}")
                    self.console.print(f"  [green]Cost:[/green] ${cost:.4f}")
                else:
                    print(f"Input tokens: {token_counts['input_token_count']}")
                    print(f"Output tokens: {token_counts['output_token_count']}")
                    print(f"Cost: ${cost:.4f}")

            result = deepcopy(entry)
            result.update({
                "generated_answer": final_answer,
                "explanation": final_explanation,
                "response": final_response,
                "messages": messages,
                "input_tokens": token_counts["input_token_count"],
                "output_tokens": token_counts["output_token_count"],
                "cost": cost
            })
            return result

        # Return error result if all attempts fail
        token_counts = model.get_token_counts() if 'response' in locals() else {"input_token_count": 0, "output_token_count": 0}
        cost = calculate_cost(token_counts["input_token_count"], token_counts["output_token_count"], model.model_id)

        if should_show_output:
            if RICH_AVAILABLE:
                self.console.print("[bold red]Failed to get properly formatted response after 5 attempts[/bold red]")
            else:
                print("Failed to get properly formatted response after 5 attempts")

        result = deepcopy(entry)
        result.update({
            "generated_answer": None,
            "explanation": "Failed to get properly formatted response after 5 attempts",
            "response": response.content if 'response' in locals() else "",
            "messages": messages,
            "input_tokens": token_counts["input_token_count"],
            "output_tokens": token_counts["output_token_count"],
            "cost": cost,
            "error": "Failed to get properly formatted response after 5 attempts"
        })
        return result