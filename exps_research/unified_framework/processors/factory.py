"""
Factory module for creating experiment processors
"""

from typing import Type, Dict

from .base import ExperimentProcessor
from .reasoning import ReasoningExperimentProcessor
from .agent import AgentExperimentProcessor


def get_experiment_processor(experiment_type: str) -> Type[ExperimentProcessor]:
    """
    Factory function to get the appropriate experiment processor class
    
    Args:
        experiment_type: Type of experiment ("reasoning" or "agent")
        
    Returns:
        Experiment processor class
        
    Raises:
        ValueError: If experiment_type is not recognized
    """
    processors: Dict[str, Type[ExperimentProcessor]] = {
        "reasoning": ReasoningExperimentProcessor,
        "agent": AgentExperimentProcessor
    }
    
    if experiment_type not in processors:
        raise ValueError(f"Unknown experiment type: {experiment_type}. Available types: {', '.join(processors.keys())}")
    
    return processors[experiment_type] 