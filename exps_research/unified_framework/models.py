"""
Unified model setup for experiments
"""

import os
from typing import Dict, Any, Optional, Union

from smolagents import OpenAIServerModel, VLLMServerModel, VLLMModel


def load_api_key(key_path: str = "keys/openai-key/key.env") -> str:
    """Load API key from file"""
    with open(key_path) as f:
        return f.read().strip()


def setup_model(
    model_type: str = "openai", 
    model_id: str = None, 
    fine_tuned: bool = False,
    local_device_id: int = -1,
    lora_path: str = None,
    **kwargs
) -> Union[OpenAIServerModel, VLLMServerModel, VLLMModel]:
    """
    Initialize a model for experiments
    
    Args:
        model_type: Type of model to use ("openai" or "vllm")
        model_id: Model ID to use (e.g., gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct)
        fine_tuned: Whether to use a fine-tuned model
        **kwargs: Additional keyword arguments for model initialization
    
    Returns:
        Initialized model
    """
    default_models = {
        "openai": "gpt-4o-mini",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }
    model_id = model_id or default_models.get(model_type)    
    if model_type == "openai":
        # It is possible that api_base and api_key are provided in kwargs
        # In this case, we need to remove them from kwargs  
        _api_base = kwargs.pop("api_base", None)
        _api_key = kwargs.pop("api_key", None)
        api_key = load_api_key()
        return OpenAIServerModel(
            model_id=model_id,
            api_key=api_key,
            **kwargs
        )
    elif model_type == "vllm":
        if fine_tuned:
            if int(local_device_id) >= 0:
                return VLLMModel(
                    model_id=model_id,
                    lora_path=lora_path,
                    local_device_id=local_device_id,
                    **kwargs
                )
            else:
                return VLLMServerModel(
                    model_id=model_id,
                    # api_base="http://0.0.0.0:8000/v1",
                    # api_key="token-abc",
                    lora_name="finetune",
                    **kwargs
                )
        else:
            if int(local_device_id) >= 0:
                return VLLMModel(
                    model_id=model_id,
                    local_device_id=local_device_id,
                    **kwargs
                )
            else:
                return VLLMServerModel(
                    model_id=model_id,
                    # api_base="http://0.0.0.0:8000/v1",
                    # api_key="token-abc",
                    **kwargs
                )
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 