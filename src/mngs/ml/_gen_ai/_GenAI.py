#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 22:52:03 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/chat.py


"""
This script does XYZ.
"""


"""
Imports
"""
# mngs.gen.reload(mngs)
import io
import json
import os
import re
import sys
import warnings
from abc import ABC, abstractmethod
from glob import glob
from pprint import pprint

import anthropic
import google.generativeai as genai
import markdown2
import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic

# from mngs.gen import notify
from natsort import natsorted
from openai import OpenAI
from tqdm import tqdm

from ._ChatGPT import ChatGPT
from ._Claude import Claude
from ._Gemini import Gemini
from ._Perplexity import Perplexity

# from sciwriter_app._email import notify

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


MODEL_CONFIG = {
    "ChatGPT": {
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "api_key_env": "OPENAI_API_KEY",
        "class": "ChatGPT",
    },
    "Claude": {
        "models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "api_key_env": "CLAUDE_API_KEY",
        "class": "Claude",
    },
    "Gemini": {
        "models": [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-pro",
        ],
        "api_key_env": "GOOGLE_API_KEY",
        "class": "Gemini",
    },
    "Perplexity": {
        "models": [
            "llama-3-sonar-small-32k-chat",
            "llama-3-sonar-small-32k-online",
            "llama-3-sonar-large-32k-chat",
            "llama-3-sonar-large-32k-online",
            "llama-3-8b-instruct",
            "llama-3-70b-instruct",
            "mixtral-8x7b-instruct",
        ],
        "api_key_env": "PERPLEXITY_API_KEY",
        "class": "Perplexity",
    },
}


def GenAI(
    model="gpt-3.5-turbo",
    stream=False,
    api_key=None,
    seed=None,
    temperature=1.0,
):
    """Factory function to create an instance of an AI model handler."""
    if api_key is None:
        api_key = model2apikey(model)

    for config in MODEL_CONFIG.values():
        if model in config["models"]:
            model_class = globals()[config["class"]]
            return model_class(
                model=model,
                stream=stream,
                api_key=api_key,
                seed=seed,
                temperature=temperature,
            )

    raise ValueError(f"No handler available for model {model}.")


################################################################################
# Helper functions
################################################################################
def model2apikey(model):
    """Retrieve the API key for a given model from environment variables."""
    for config in MODEL_CONFIG.values():
        if model in config["models"]:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                raise EnvironmentError(
                    f"API key for {model} not found in environment."
                )
            return api_key
    raise ValueError(f"Model {model} is not supported.")


def test_all(seed=None, temperature=1.0):
    model_names = [
        "gpt-4",
        "claude-3-opus-20240229",
        "gemini-pro",
        "llama-3-sonar-large-32k-online",
    ]

    for model_name in model_names:
        for stream in [False, True]:
            model = GenAI(
                model_name, stream=stream, seed=seed, temperature=temperature
            )
            prompt = "Hi. Tell me your name just within a line."

            print(
                f"\n{'-'*40}\n{model.model}\nStream: {stream}\nSeed: {seed}\nTemperature: {temperature}\n{'-'*40}"
            )
            print(model(prompt))
            print(model.get_available_models())


def main(
    model="gpt-3.5-turbo",
    stream=False,
    prompt="Hi, please tell me about the hippocampus",
    seed=None,
    temperature=1.0,
):

    m = GenAI(model, stream=stream, seed=seed, temperature=temperature)
    out = m(prompt)

    return out


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    test_all()
    # out_static = main(stream=False, seed=42)
    # out_static = main(stream=False, seed=42)

    # out_stream = main(stream=True)

    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
