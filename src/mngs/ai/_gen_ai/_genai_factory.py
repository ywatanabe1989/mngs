#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-15 11:24:10 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/chat.py


"""
This script does XYZ.
"""


"""
Imports
"""
# mngs.gen.reload(mngs)
import os

from ._OpenAI import OpenAI
from ._Claude import Claude
from ._Gemini import Gemini
from ._Llama import Llama
from ._Perplexity import Perplexity
from .PARAMS import MODELS
import random

# # from mngs.gen import notify
# from natsort import natsorted
# from openai import OpenAI
# from tqdm import tqdm

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


def genai_factory(
    model="gpt-3.5-turbo",
    stream=False,
    api_key=None,
    seed=None,
    temperature=1.0,
    n_keep=1,
    chat_history=None,
):
    """Factory function to create an instance of an AI model handler."""
    AVAILABLE_MODELS = MODELS.name.tolist()

    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f'Model "{model}" is not available. Please choose from:{MODELS.name.tolist()}'
        )

    provider = MODELS[MODELS.name == model].provider.iloc[0]
    model_class = globals()[provider]

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
    )


def test_all(seed=None, temperature=1.0):
    model_names = [
        "claude-3-5-sonnet-20240620",
        "gpt-4",
        "gemini-pro",
        "llama-3-sonar-large-32k-online",
        "llama-7b",
    ]

    for model_name in model_names:
        for stream in [False, True]:
            model = GenAI(
                model_name, stream=stream, seed=seed, temperature=temperature
            )
            prompt = "Hi. Tell me about the hippocampus."

            print(
                f"\n{'-'*40}\n{model.model}\nStream: {stream}\nSeed: {seed}\nTemperature: {temperature}\n{'-'*40}"
            )
            print(model(prompt))
            print(model.available_models)


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


################################################################################
# Helper functions
################################################################################
# def model2apikey(model):
#     """Retrieve the API key for a given model from environment variables."""
#     for config in MODEL_CONFIG.values():
#         if model in config["models"]:
#             api_key = os.getenv(config["api_key_env"])
#             if not api_key:
#                 raise EnvironmentError(
#                     f"API key for {model} not found in environment."
#                 )
#             return api_key
#     raise ValueError(f"Model {model} is not supported.")


def test_all(seed=None, temperature=1.0):
    model_names = [
        "claude-3-5-sonnet-20240620",
        # "gpt-4",
        # "claude-3-opus-20240229",
        # "gemini-pro",
        # "llama-3-sonar-large-32k-online",
    ]

    for model_name in model_names:
        for stream in [False, True]:
            model = GenAI(
                model_name, stream=stream, seed=seed, temperature=temperature
            )
            # prompt = "Hi. Tell me your name just within a line."
            prompt = "Hi. Tell me about the hippocampus."

            print(
                f"\n{'-'*40}\n{model.model}\nStream: {stream}\nSeed: {seed}\nTemperature: {temperature}\n{'-'*40}"
            )
            print(model(prompt))
            print(model.available_models)


# def main(
#     model="gpt-3.5-turbo",
#     stream=False,
#     prompt="Hi, please tell me about the hippocampus",
#     seed=None,
#     temperature=1.0,
# ):

#     m = GenAI(model, stream=stream, seed=seed, temperature=temperature)
#     out = m(prompt)

#     return out


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

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
