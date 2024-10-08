#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-13 19:18:20 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py

import pandas as pd

# https://openai.com/api/pricing/

MODELS = pd.DataFrame(
    [
        {
            "name": "o1-mini",
            "input_cost": 3.00,
            "output_cost": 12.00,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "o1-preview",
            "input_cost": 15.00,
            "output_cost": 60.00,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-4",
            "input_cost": 30.00,
            "output_cost": 60.00,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-4o",
            "input_cost": 5.00,
            "output_cost": 15.00,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-4o-mini",
            "input_cost": 0.150,
            "output_cost": 0.600,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-4-turbo",
            "input_cost": 10.00,
            "output_cost": 30.00,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-3.5-turbo",
            "input_cost": 0.50,
            "output_cost": 1.50,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "claude-3-5-sonnet-20240620",
            "input_cost": 3.00,
            "output_cost": 15.00,
            "api_key_env": "CLAUDE_API_KEY",
            "provider": "Claude",
        },
        {
            "name": "claude-3-opus-20240229",
            "input_cost": 15.00,
            "output_cost": 75.00,
            "api_key_env": "CLAUDE_API_KEY",
            "provider": "Claude",
        },
        {
            "name": "claude-3-haiku-20240307",
            "input_cost": 0.25,
            "output_cost": 1.25,
            "api_key_env": "CLAUDE_API_KEY",
            "provider": "Claude",
        },
        {
            "name": "gemini-1.5-pro-latest",
            "input_cost": 3.50,
            "output_cost": 10.50,
            "api_key_env": "GOOGLE_API_KEY",
            "provider": "Gemini",
        },
        {
            "name": "gemini-1.5-flash-latest",
            "input_cost": 0.15,
            "output_cost": 0.0375,
            "api_key_env": "GOOGLE_API_KEY",
            "provider": "Gemini",
        },
        {
            "name": "llama-3.1-sonar-small-128k-online",
            "input_cost": 0.20,
            "output_cost": 0.20,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3.1-sonar-large-128k-online",
            "input_cost": 1.00,
            "output_cost": 1.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3.1-sonar-huge-128k-online",
            "input_cost": 5.00,
            "output_cost": 5.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3.1-sonar-small-128k-chat",
            "input_cost": 0.20,
            "output_cost": 0.20,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3.1-sonar-large-128k-chat",
            "input_cost": 1.00,
            "output_cost": 1.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-sonar-small-32k-chat",
            "input_cost": 0.20,
            "output_cost": 0.20,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-sonar-small-32k-online",
            "input_cost": 0.20,
            "output_cost": 0.20,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-sonar-large-32k-chat",
            "input_cost": 1.00,
            "output_cost": 1.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-sonar-large-32k-online",
            "input_cost": 1.00,
            "output_cost": 1.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-8b-instruct",
            "input_cost": 0.20,
            "output_cost": 0.20,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-70b-instruct",
            "input_cost": 1.00,
            "output_cost": 1.00,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "mixtral-8x7b-instruct",
            "input_cost": 0.60,
            "output_cost": 0.60,
            "api_key_env": "PERPLEXITY_API_KEY",
            "provider": "Perplexity",
        },
        {
            "name": "llama-3-70b",
            "input_cost": 0.00,
            "output_cost": 0.00,
            "api_key_env": "LLAMA_API_KEY",
            "provider": "Llama",
        },
        {
            "name": "llama-3-70-instruct",
            "input_cost": 0.00,
            "output_cost": 0.00,
            "api_key_env": "LLAMA_API_KEY",
            "provider": "Llama",
        },
        {
            "name": "llama-3-8b",
            "input_cost": 0.00,
            "output_cost": 0.00,
            "api_key_env": "LLAMA_API_KEY",
            "provider": "Llama",
        },
    ]
)
