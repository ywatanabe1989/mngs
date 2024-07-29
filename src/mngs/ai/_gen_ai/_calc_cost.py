#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-29 13:33:06 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_PRICING.py


PRICING_USD_PER_MILLIION = {
    # GPT
    "gpt-4": (30.00, 60.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4o": (5.00, 2.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-4o-mini": (0.150, 0.075),
    # Claude
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Gemini
    "gemini-1.5-pro-latest": (3.50, 10.50),
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.35, 1.05),
    # Llama
    "llama-3-sonar-small-32k-chat": (0.20, 0.20),
    "llama-3-sonar-small-32k-online": (0.20, 0.20),
    "llama-3-sonar-large-32k-chat": (1.00, 1.00),
    "llama-3-sonar-large-32k-online": (1.00, 1.00),
    "llama-3-8b-instruct": (0.20, 0.20),
    "llama-3-70b-instruct": (1.00, 1.00),
    "mixtral-8x7b-instruct": (0.60, 0.60),
}


def calc_cost(model, input_tokens, output_tokens):
    input_rate, output_rate = PRICING_USD_PER_MILLIION[model]
    cost = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    return cost
