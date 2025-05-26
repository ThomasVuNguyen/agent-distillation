"""
Unified model setup for experiments
"""

import importlib.util
import json
import os
import requests
from typing import Dict, Any, List, Optional, Union

from smolagents import OpenAIServerModel, VLLMServerModel, VLLMModel, ApiModel, ChatMessage, Tool, MessageRole


def load_api_key(key_path: str = "keys/openai-key/key.env") -> str:
    """Load API key from file"""
    with open(key_path) as f:
        return f.read().strip()


class CloudflareWorkersAIModel(ApiModel):
    """This model connects to Cloudflare Workers AI API.

    Parameters:
        model_id (`str`):
            The model identifier to use on Cloudflare Workers AI (e.g. "@cf/meta/llama-3-70b-instruct").
        api_base (`str`, *optional*):
            The base URL of the Cloudflare Workers AI API.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = "https://api.cloudflare.com/client/v4/accounts",
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        super().__init__(flatten_messages_as_text=flatten_messages_as_text, **kwargs)
        self.model_id = model_id
        self.api_base = api_base
        self.account_id = account_id
        self.api_key = api_key
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        # Prepare completion kwargs similar to OpenAI format
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        
        # Extract messages from completion kwargs
        prepared_messages = completion_kwargs.get("messages", [])
        
        # Cloudflare API URL format
        if not self.account_id:
            raise ValueError("account_id is required for Cloudflare Workers AI")
            
        # Format for Cloudflare Workers AI API endpoint
        # Based on testing, the following models work:
        # - @cf/meta/llama-3.1-70b-instruct
        # - @cf/meta/llama-3-8b-instruct
        
        # Convert Llama 3 to Llama 3.1 if needed
        model_id = self.model_id
        if "llama-3-" in model_id and ".1" not in model_id:
            # Replace llama-3- with llama-3.1- to match available models
            model_id = model_id.replace("llama-3-", "llama-3.1-")
        
        api_url = f"{self.api_base}/{self.account_id}/ai/run/{model_id}"
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare request payload according to Cloudflare documentation
        # Convert the chat messages format to a single prompt string for Cloudflare API
        prompt = self._convert_messages_to_prompt(prepared_messages)
        payload = {
            "prompt": prompt,
            "stream": False
        }
        
        # Add optional parameters if provided
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
            
        # Add temperature and other parameters if available in kwargs
        for param in ["temperature", "max_tokens", "top_p", "top_k"]:
            if param in completion_kwargs:
                payload[param] = completion_kwargs[param]
        
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Cloudflare Workers AI API error: {response.status_code} - {response.text}")
        
        # Parse the response according to the provided format
        # {
        #   "result": {
        #     "response": "Hello, World first appeared..."
        #   },
        #   "success": true,
        #   "errors": [],
        #   "messages": []
        # }
        response_data = response.json()
        
        # Check if the request was successful
        if not response_data.get("success", False):
            errors = response_data.get("errors", [])
            error_msg = '; '.join([err.get("message", "Unknown error") for err in errors]) if errors else "Unknown error"
            raise Exception(f"Cloudflare Workers AI API error: {error_msg}")
            
        # Extract content from the response
        result = response_data.get("result", {})
        content = result.get("response", "")
        
        # Create ChatMessage object for returning the response
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content
        )
        
        return message
        
    def _convert_messages_to_prompt(self, messages):
        """Convert a list of messages to a single prompt string for Cloudflare API."""
        prompt = ""
        for message in messages:
            role = message.get("role", "").upper()
            content = message.get("content", "")
            if role == "SYSTEM":
                prompt += f"SYSTEM: {content}\n\n"
            elif role == "USER":
                prompt += f"USER: {content}\n\n"
            elif role == "ASSISTANT":
                prompt += f"ASSISTANT: {content}\n\n"
        return prompt.strip()


def load_cloudflare_api_key(key_path: str = "keys/cloudflare-key/key.env") -> str:
    """Load Cloudflare API key from file"""
    with open(key_path) as f:
        return f.read().strip()


def load_cloudflare_account_id(id_path: str = "keys/cloudflare-key/account_id.env") -> str:
    """Load Cloudflare account ID from file"""
    with open(id_path) as f:
        return f.read().strip()


def setup_model(
    model_type: str = "openai", 
    model_id: str = None, 
    fine_tuned: bool = False,
    local_device_id: int = -1,
    lora_path: str = None,
    **kwargs
) -> Union[OpenAIServerModel, VLLMServerModel, VLLMModel, CloudflareWorkersAIModel]:
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
        "cloudflare": "@cf/meta/llama-3-70b-instruct",
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
    elif model_type == "cloudflare":
        # Handle Cloudflare Workers AI model
        _api_base = kwargs.pop("api_base", None)
        _api_key = kwargs.pop("api_key", None)
        _account_id = kwargs.pop("account_id", None)
        
        # Load API key and account ID from files if not provided
        api_key = _api_key or load_cloudflare_api_key()
        account_id = _account_id or load_cloudflare_account_id()
        
        return CloudflareWorkersAIModel(
            model_id=model_id,
            api_key=api_key,
            account_id=account_id,
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
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'openai', 'vllm', or 'cloudflare'") 