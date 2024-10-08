#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-15 10:29:17 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py

import pandas as pd

MODELS = pd.DataFrame(
    [
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
            "output_cost": 2.50,
            "api_key_env": "OPENAI_API_KEY",
            "provider": "OpenAI",
        },
        {
            "name": "gpt-4o-mini",
            "input_cost": 0.150,
            "output_cost": 0.075,
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


# MODELS = mngs.io.load("./models.yaml")

# MODELS = pd.DataFrame({
#   - name: gpt-4
#     input_cost: 30.00
#     output_cost: 60.00
#     api_key_env: OPENAI_API_KEY
#     class: ChatGPT
#   - name: gpt-4o
#     input_cost: 5.00
#     output_cost: 2.50
#     api_key_env: OPENAI_API_KEY
#     class: ChatGPT
#   - name: gpt-4o-mini
#     input_cost: 0.150
#     output_cost: 0.075
#     api_key_env: OPENAI_API_KEY
#     class: ChatGPT
#   - name: gpt-4-turbo
#     input_cost: 10.00
#     output_cost: 30.00
#     api_key_env: OPENAI_API_KEY
#     class: ChatGPT
#   # - name: gpt-3.5-turbo
#   #   input_cost: 0.50
#   #   output_cost: 1.50
#   #   api_key_env: OPENAI_API_KEY
#   #   class: ChatGPT
#   - name: claude-3-5-sonnet-20240620
#     input_cost: 3.00
#     output_cost: 15.00
#     api_key_env: CLAUDE_API_KEY
#     class: Claude
#   - name: claude-3-opus-20240229
#     input_cost: 15.00
#     output_cost: 75.00
#     api_key_env: CLAUDE_API_KEY
#     class: Claude
#   # - name: claude-3-sonnet-20240229
#   #   input_cost: 3.00
#   #   output_cost: 15.00
#   #   api_key_env: CLAUDE_API_KEY
#   #   class: Claude
#   - name: claude-3-haiku-20240307
#     input_cost: 0.25
#     output_cost: 1.25
#     api_key_env: CLAUDE_API_KEY
#     class: Claude
#   - name: gemini-1.5-pro-latest
#     input_cost: 3.50
#     output_cost: 10.50
#     api_key_env: GOOGLE_API_KEY
#     class: Gemini
#   - name: gemini-1.5-flash-latest
#     input_cost: 0.35
#     output_cost: 1.05
#     api_key_env: GOOGLE_API_KEY
#     class: Gemini
#   # - name: gemini-1.5-pro
#   #   input_cost: 3.50
#   #   output_cost: 10.50
#   #   api_key_env: GOOGLE_API_KEY
#   #   class: Gemini
#   # - name: gemini-pro
#   #   input_cost: 0.00
#   #   output_cost: 0.00
#   #   api_key_env: GOOGLE_API_KEY
#   #   class: Gemini
#   - name: llama-3-sonar-small-32k-chat
#     input_cost: 0.20
#     output_cost: 0.20
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-sonar-small-32k-online
#     input_cost: 0.20
#     output_cost: 0.20
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-sonar-large-32k-chat
#     input_cost: 1.00
#     output_cost: 1.00
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-sonar-large-32k-online
#     input_cost: 1.00
#     output_cost: 1.00
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-8b-instruct
#     input_cost: 0.20
#     output_cost: 0.20
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-70b-instruct
#     input_cost: 1.00
#     output_cost: 1.00
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: mixtral-8x7b-instruct
#     input_cost: 0.60
#     output_cost: 0.60
#     api_key_env: PERPLEXITY_API_KEY
#     class: Perplexity
#   - name: llama-3-70b
#     input_cost: 0.00
#     output_cost: 0.00
#     api_key_env: LLAMA_API_KEY
#     class: Llama
#   - name: llama-3-70-instruct
#     input_cost: 0.00
#     output_cost: 0.00
#     api_key_env: LLAMA_API_KEY
#     class: Llama
#   - name: llama-3-8b
#     input_cost: 0.00
#     output_cost: 0.00
#     api_key_env: LLAMA_API_KEY
#     class: Llama
#   - name: llama-3-8b-instruct
#     input_cost: 0.00
#     output_cost: 0.00
#     api_key_env: LLAMA_API_KEY
#     class: Llama

# })

# MODEL_CONFIG = {
#     "ChatGPT": {
#         "models": [
#             "gpt-4o-mini",
#             "gpt-4o",
#             "gpt-4-turbo",
#             "gpt-4",
#             "gpt-3.5-turbo",
#         ],
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     "Claude": {
#         "models": [
#             "claude-3-5-sonnet-20240620",
#             "claude-3-opus-20240229",
#             "claude-3-sonnet-20240229",
#             "claude-3-haiku-20240307",
#         ],
#         "api_key_env": "CLAUDE_API_KEY",
#         "provider": "Claude",
#     },
#     "Gemini": {
#         "models": [
#             "gemini-1.5-pro-latest",
#             "gemini-1.5-flash-latest",
#             "gemini-1.5-pro",
#             "gemini-pro",
#         ],
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Gemini",
#     },
#     "Perplexity": {
#         "models": [
#             "llama-3-sonar-small-32k-chat",
#             "llama-3-sonar-small-32k-online",
#             "llama-3-sonar-large-32k-chat",
#             "llama-3-sonar-large-32k-online",
#             "llama-3-8b-instruct",
#             "llama-3-70b-instruct",
#             "mixtral-8x7b-instruct",
#         ],
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     "Llama": {
#         "models": [
#             "Llama-3-70B",
#             "Llama-3-70-Instruct",
#             "Llama-3-8B",
#             "Llama-3-8B-Instruct",
#         ],
#         "api_key_env": "LLAMA_API_KEY",  # Fake
#         "provider": "Llama",
#     },
# }


# PRICING_USD_PER_MILLIION = {
#     # GPT
#     "gpt-4": (30.00, 60.00),
#     "gpt-4-turbo": (10.00, 30.00),
#     "gpt-4o": (5.00, 2.50),
#     "gpt-3.5-turbo-0125": (0.50, 1.50),
#     "gpt-4o-mini": (0.150, 0.075),
#     # Claude
#     "claude-3-5-sonnet-20240620": (3.00, 15.00),
#     "claude-3-opus-20240229": (15.00, 75.00),
#     "claude-3-haiku-20240307": (0.25, 1.25),
#     # Gemini
#     "gemini-1.5-pro-latest": (3.50, 10.50),
#     "gemini-1.5-pro": (3.50, 10.50),
#     "gemini-1.5-flash": (0.35, 1.05),
#     # Llama
#     "llama-3-sonar-small-32k-chat": (0.20, 0.20),
#     "llama-3-sonar-small-32k-online": (0.20, 0.20),
#     "llama-3-sonar-large-32k-chat": (1.00, 1.00),
#     "llama-3-sonar-large-32k-online": (1.00, 1.00),
#     "llama-3-8b-instruct": (0.20, 0.20),
#     "llama-3-70b-instruct": (1.00, 1.00),
#     "mixtral-8x7b-instruct": (0.60, 0.60),
# }

# PARAMS = {
#     "PRICING_USD_PER_MILLIION": PRICING_USD_PER_MILLIION,
# }
