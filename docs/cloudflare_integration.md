# Cloudflare Workers AI Integration

This document explains how to use Cloudflare Workers AI with agent-distillation to run teacher model inference in the cloud.

## Overview

The Cloudflare Workers AI integration allows you to generate trajectories using large language models hosted on Cloudflare's infrastructure, eliminating the need for local GPU resources. This is especially useful if you:

- Don't have access to sufficient local GPU resources
- Want to experiment with different large language models
- Need to scale your experiments more easily

## Setup

### 1. Prerequisites

You'll need:
- A Cloudflare account
- API access to Cloudflare Workers AI
- Your Cloudflare API key and account ID

### 2. Configure API Credentials

Save your credentials in the appropriate files:

```bash
# Create directories if they don't exist
mkdir -p keys/cloudflare-key

# Save API key
echo "your_cloudflare_api_key" > keys/cloudflare-key/key.env

# Save account ID
echo "your_cloudflare_account_id" > keys/cloudflare-key/account_id.env
```

## Usage

The `run_cloudflare_inference.sh` script allows you to generate trajectories using Cloudflare Workers AI:

```bash
# Basic usage
bash scripts/inference/run_cloudflare_inference.sh

# With custom options
bash scripts/inference/run_cloudflare_inference.sh --model="@cf/meta/llama-3-70b-instruct" --temperature=0.2 --workers=4
```

### Available Options

- `--model=MODEL_ID`: Specify the Cloudflare Workers AI model ID (default: "@cf/meta/llama-3-70b-instruct")
- `--temperature=TEMP`: Set the temperature for generation (default: 0.0)
- `--workers=NUM`: Set the number of parallel workers (default: 4)
- `--use-prefix`: Enable first-thought prefix memory (if available)

## Available Models

Cloudflare Workers AI supports various models, including:

- `@cf/meta/llama-3.1-70b-instruct`: Meta's Llama 3.1 70B instructed model (**note the `.1` in the model name**)
- `@cf/meta/llama-3-8b-instruct`: Meta's Llama 3 8B instructed model
- `@cf/mistral/mistral-7b-instruct-v0.2`: Mistral 7B instructed model

> **Important Note**: The 70B model uses the format `@cf/meta/llama-3.1-70b-instruct` (with `.1`), while the 8B model uses `@cf/meta/llama-3-8b-instruct` (without `.1`). Using the wrong format will result in API errors.

For a complete list of available models, refer to the [Cloudflare Workers AI documentation](https://developers.cloudflare.com/workers-ai/models/).

## Implementation Details

The integration is implemented in the following files:

- `exps_research/unified_framework/models.py`: Contains the `CloudflareWorkersAIModel` class that interfaces with the Cloudflare Workers AI API
- `scripts/inference/run_cloudflare_inference.sh`: Script to run inference using Cloudflare Workers AI

### Technical Implementation Notes

1. **API Endpoint Format**: The correct format for the Cloudflare Workers AI API endpoint is:
   ```
   https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}
   ```

2. **Model ID Handling**: Our implementation automatically converts `llama-3-70b-instruct` to `llama-3.1-70b-instruct` for compatibility with Cloudflare's naming conventions.

3. **Message Format**: Cloudflare Workers AI expects a single `prompt` field rather than a list of messages. The `CloudflareWorkersAIModel` class handles converting the chat messages format to a simple prompt string.

4. **Response Format**: The API returns responses in the following format:
   ```json
   {
     "result": {
       "response": "The model's response text..."
     },
     "success": true,
     "errors": [],
     "messages": []
   }
   ```

The integration maintains the same output format and compatibility with the rest of the agent-distillation pipeline, so you can seamlessly transition to training and evaluation after generating trajectories.

## Troubleshooting

If you encounter issues:

### "No route for that URI" Error

This error typically indicates a problem with the model ID format or API endpoint structure.

1. **Check model ID format**: Make sure you're using the correct model format:
   - For the 70B model: `@cf/meta/llama-3.1-70b-instruct` (with `.1`)
   - For the 8B model: `@cf/meta/llama-3-8b-instruct` (without `.1`)

2. **Verify API endpoint**: The correct endpoint format is:
   ```
   https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}
   ```

### Authentication Issues

1. **Verify API credentials**: Check that your API key in `keys/cloudflare-key/key.env` is correct
2. **Check account ID**: Ensure your account ID in `keys/cloudflare-key/account_id.env` is valid
3. **Confirm API access**: Make sure your Cloudflare account has access to the Workers AI service

### Response Format Issues

If you encounter errors related to parsing the response:

1. **Check response structure**: The API should return a JSON object with `result.response` containing the model output
2. **Review error details**: Look for specific error messages in the `errors` array of the response

### General Troubleshooting

1. Use the included `test_cloudflare_api.py` script to test connectivity and API functionality
2. Try with a smaller batch size using the `--debug` flag to identify issues
3. Verify your network connection can reach the Cloudflare API
4. Check the Cloudflare API status dashboard for any service outages

For persistent issues, consider contacting Cloudflare support or opening an issue in the agent-distillation repository.
