"""
Agent experiment processor for tool-based experiments
"""

from typing import Dict, Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .base import ExperimentProcessor
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    WikipediaRetrieverTool
)


class AgentExperimentProcessor(ExperimentProcessor):
    """
    Processor for agent experiments

    This processor handles agent-based experiments where the model
    interacts with tools to solve problems.
    """

    def __init__(self, model_kwargs: Dict[str, Any], **kwargs):
        """Initialize the agent experiment processor with a rich console if available"""
        super().__init__(model_kwargs, **kwargs)
        if RICH_AVAILABLE:
            self.console = Console()

    def process_entry(self, entry: Dict, model, **kwargs) -> Dict:
        """
        Process an agent experiment entry

        Args:
            entry: Dictionary containing a question
            model: Model instance
            **kwargs: Additional parameters including:
                - search_engine_type: "wikipedia" or "duckduckgo"
                - max_steps: Maximum number of steps for the agent
                - fine_tuned: Whether using a fine-tuned model
                - set_timeout: Whether to set timeouts for code execution
                - verbose_worker: Whether this worker should display verbose output

        Returns:
            Processed result dictionary
        """
        if self.cost_tracker.stop_requested:
            return None

        # Get experiment parameters
        search_engine_type = kwargs.get('search_engine_type', 'wikipedia')
        max_steps = kwargs.get('max_steps', 5)
        fine_tuned = kwargs.get('fine_tuned', False)
        use_planning = kwargs.get('use_planning', False)
        prefix_memory = kwargs.get('prefix_memory', None)
        cot_memory = kwargs.get('cot_memory', None)
        set_timeout = True

        # Determine if this worker should show verbose output
        verbose_worker = kwargs.get('verbose_worker', True)
        should_show_output = self.verbose and verbose_worker

        # Set verbosity level for the agent based on whether this is the designated verbose worker
        verbosity_level = 2 if should_show_output else 0

        if should_show_output and RICH_AVAILABLE:
            self.console.rule(f"[bold blue]Processing Agent Question")
            self.console.print(Panel(entry['question'], title="Question", border_style="green"))

        # Configure tools based on search engine type
        if search_engine_type == "duckduckgo":
            tools = [DuckDuckGoSearchTool()]
        else:  # Default to Wikipedia
            tools = [WikipediaRetrieverTool()]

        # Create agent with specified configuration
        agent_kwargs = {
            "max_steps": max_steps
        }
        if use_planning:
            agent_kwargs = {
                "planning_interval": 10,
                "max_steps": max_steps + 1
            }
        agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=["numpy", "sympy", "numpy.linalg"],
            verbosity_level=verbosity_level,  # Set based on should_show_output
            **agent_kwargs
        )

        question = entry["question"]
        instruction = "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail. For math problems that are not multiple-choice, always output the final answer using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), not a decimal approximation (e.g., \\boxed{0.642857})."

        try:
            # Run agent with appropriate prompting
            _question = question + instruction

            if should_show_output and RICH_AVAILABLE:
                self.console.print("[bold cyan]Running agent with max_steps:[/bold cyan]", str(max_steps))
                self.console.print("[bold cyan]Search engine:[/bold cyan]", search_engine_type)

            if prefix_memory:
                prefix_list = []
                prefix = prefix_memory.get(question) # It must be a dictionary
                prefix_list.append(prefix)
                agent.register_prefix(prefix_list)

            if cot_memory:
                existing_cot = cot_memory.get(question, None)
                if existing_cot:
                    cot_guide = "<reference>\nUse this REFERENCE solution for solving problem. DO NOT directly mention the reference solution in your solution:\n\n" + existing_cot + "</reference>"
                    _question = question + cot_guide + instruction

            result, log_data = agent.run(
                _question,
                return_log_data=True,
                use_short_system_message=fine_tuned,
            )

            # Track cost if requested
            if self.track_cost:
                cost_info = log_data["metadata"]["cost"]["total_cost"]
                current_total = self.cost_tracker.update_cost(cost_info)

                # Only print cost information for the verbose worker
                if should_show_output:
                    if RICH_AVAILABLE:
                        self.console.print(f"[green]Current total cost: ${current_total:.4f}[/green]")
                    else:
                        print(f"Current total cost: ${current_total:.4f}")

            # Clean up memory to make logs more compact
            for step in agent.memory.steps:
                if hasattr(step, 'agent_memory'):
                    step.agent_memory = None

            # Display final answer for verbose worker
            if should_show_output and RICH_AVAILABLE:
                try:
                    self.console.print(Panel(result, title="[bold green]Agent Final Answer[/bold green]", border_style="green"))
                    self.console.print(f"[bold]Total steps:[/bold] {len(agent.memory.steps)}")
                    self.console.print(f"[bold]Total cost:[/bold] ${log_data['metadata']['cost']['total_cost']:.4f}")
                except:
                    print(result)

            # Create result dictionary
            annotated_result = {
                "model_id": model.model_id,
                "question": question,
                "generated_answer": result,
                "true_answer": entry.get("answer", None),
                "log_data": log_data,
                "cost": log_data["metadata"]["cost"]["total_cost"] if "metadata" in log_data else 0.0
            }
        except Exception as e:
            # Only print errors for the verbose worker
            if should_show_output:
                if RICH_AVAILABLE:
                    self.console.print(f"[bold red]Error processing question:[/bold red] {question}")
                    self.console.print(f"[red]{str(e)}[/red]")
                else:
                    print(f"Error processing question: {question}", str(e))

            annotated_result = {
                "model_id": model.model_id,
                "question": question,
                "error": str(e),
                "log_data": None,
                "cost": 0.0
            }

        return annotated_result