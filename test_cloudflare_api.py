#!/usr/bin/env python3
"""
Test script for Cloudflare Workers AI API
"""
import os
import sys
import requests
import json

def load_api_key(key_path="keys/cloudflare-key/key.env"):
    """Load API key from file"""
    if not os.path.exists(key_path):
        print(f"Error: API key file not found at {key_path}")
        return None
    
    with open(key_path, "r") as f:
        return f.read().strip()

def load_account_id(id_path="keys/cloudflare-key/account_id.env"):
    """Load account ID from file"""
    if not os.path.exists(id_path):
        print(f"Error: Account ID file not found at {id_path}")
        return None
    
    with open(id_path, "r") as f:
        return f.read().strip()

def test_cloudflare_api():
    """Test Cloudflare Workers AI API with simple prompt"""
    # Load credentials
    api_key = load_api_key()
    account_id = load_account_id()
    
    if not api_key or not account_id:
        sys.exit(1)
    
    # Try different model ID formats
    model_formats = [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        # With @cf/ prefix as in the docs
        "@cf/meta/llama-3-70b-instruct",
        # Without @cf/ prefix
        "meta/llama-3-70b-instruct",
        # Use llama-3.1 instead of llama-3 as shown in docs
        "@cf/meta/llama-3.1-70b-instruct",
        # Simple model name only
        "llama-3-70b-instruct",
        # Try a different model size
        "@cf/meta/llama-3-8b-instruct",
        # With dots instead of hyphens
        "@cf/meta/llama.3.70b.instruct"
    ]
    
    for model_id in model_formats:
        print(f"\n\n===== Testing with model: {model_id} =====")
        api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"
    
        # Set up headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare request payload
        payload = {
            "prompt": "What is the capital of France?",
            "stream": False
        }
        
        print(f"Making API request to: {api_url}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        # Make API request
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success", False):
                    result = response_data.get("result", {})
                    content = result.get("response", "")
                    print(f"\nSuccessful response content: {content}")
                else:
                    errors = response_data.get("errors", [])
                    error_msg = '; '.join([err.get("message", "Unknown error") for err in errors]) if errors else "Unknown error"
                    print(f"\nAPI Error: {error_msg}")
            else:
                print(f"\nHTTP Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_cloudflare_api()
