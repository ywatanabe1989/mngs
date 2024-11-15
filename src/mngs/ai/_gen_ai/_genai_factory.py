#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-08 20:31:33 (ywatanabe)"
# File: ./mngs_repo/src/mngs/ai/_gen_ai/_genai_factory.py

import os
import random

from ._Anthropic import Anthropic
from ._Google import Google
from ._Llama import Llama
from ._OpenAI import OpenAI
from ._Perplexity import Perplexity
from ._DeepSeek import DeepSeek
from .PARAMS import MODELS

"""Imports"""
"""Functions & Classes""""""Parameters"""

def genai_factory(
    model="gpt-3.5-turbo",
    stream=False,
    api_key=None,
    seed=None,
    temperature=1.0,
    n_keep=1,
    chat_history=None,
    max_tokens=4096,
):
    """Factory function to create an instance of an AI model handler."""
    AVAILABLE_MODELS = MODELS.name.tolist()

    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f'Model "{model}" is not available. Please choose from:{MODELS.name.tolist()}'
        )

    provider = MODELS[MODELS.name == model].provider.iloc[0]

    # model_class = globals()[provider]
    model_class = {
        "OpenAI": OpenAI,
        "Anthropic": Anthropic,
        "Google": Google,
        "Llama": Llama,
        "Perplexity": Perplexity,
        "DeepSeek": DeepSeek,
    }[provider]

    # Select a random API key from the list
    if isinstance(api_key, (list, tuple)):
        api_key = random.choice(api_key)

    return model_class(
        model=model,
        stream=stream,
        api_key=api_key,
        seed=seed,
        temperature=temperature,
        n_keep=n_keep,
        chat_history=chat_history,
        max_tokens=max_tokens,
    )

def main(
    model="gpt-3.5-turbo",
    stream=False,
    prompt="Hi, please tell me about the hippocampus",
    seed=None,
    temperature=1.0,
):
    m = genai_factory(model, stream=stream, seed=seed, temperature=temperature)
    out = m(prompt)
    return out


def main(
    model="gemini-1.5-pro-latest",
    stream=False,
    prompt="Hi, please tell me about the hippocampus",
    seed=None,
    temperature=1.0,
):

    m = mngs.ai.GenAI(
        model=model,
        api_key=os.getenv("GOOGLE_API_KEY"),
        stream=stream,
        seed=seed,
        temperature=temperature,
    )
    out = m(prompt)

    return out

def main(
    model="deepseek-coder",
    stream=False,
    prompt="Hi, please tell me about the hippocampus",
    seed=None,
    temperature=1.0,
):

    m = mngs.ai.GenAI(
        model=model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        stream=stream,
        seed=seed,
        temperature=temperature,
    )
    out = m(prompt)

    return out

if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    test_all()

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
