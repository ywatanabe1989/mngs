# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:01:42 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_genai_factory.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-27 13:38:02 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/ml/chat.py
# 
# 
# """Imports"""
# import os
# import random
# 
# from ._Anthropic import Anthropic
# from ._Google import Google
# from ._Llama import Llama
# from ._OpenAI import OpenAI
# from ._Perplexity import Perplexity
# from .PARAMS import MODELS
# import pandas as pd
# 
# """Functions & Classes""""""Parameters"""
# 
# def genai_factory(
#     model="gpt-3.5-turbo",
#     stream=False,
#     api_key=None,
#     seed=None,
#     temperature=1.0,
#     n_keep=1,
#     chat_history=None,
#     max_tokens=4096,
# ):
#     """Factory function to create an instance of an AI model handler."""
#     AVAILABLE_MODELS = MODELS.name.tolist()
# 
#     if model not in AVAILABLE_MODELS:
#         raise ValueError(
#             f'Model "{model}" is not available. Please choose from:{MODELS.name.tolist()}'
#         )
# 
#     provider = MODELS[MODELS.name == model].provider.iloc[0]
# 
#     # model_class = globals()[provider]
#     model_class = {
#         "OpenAI": OpenAI,
#         "Anthropic": Anthropic,
#         "Google": Google,
#         "Llama": Llama,
#         "Perplexity": Perplexity,
#     }[provider]
# 
#     # Select a random API key from the list
#     if isinstance(api_key, (list, tuple)):
#         api_key = random.choice(api_key)
# 
#     return model_class(
#         model=model,
#         stream=stream,
#         api_key=api_key,
#         seed=seed,
#         temperature=temperature,
#         n_keep=n_keep,
#         chat_history=chat_history,
#         max_tokens=max_tokens,
#     )
# 
# 
# def main(
#     model="gpt-3.5-turbo",
#     stream=False,
#     prompt="Hi, please tell me about the hippocampus",
#     seed=None,
#     temperature=1.0,
# ):
#     m = genai_factory(model, stream=stream, seed=seed, temperature=temperature)
#     out = m(prompt)
#     return out
# 
# 
# ################################################################################
# # Helper functions
# ################################################################################
# # def model2apikey(model):
# #     """Retrieve the API key for a given model from environment variables."""
# #     for config in MODEL_CONFIG.values():
# #         if model in config["models"]:
# #             api_key = os.getenv(config["api_key_env"])
# #             if not api_key:
# #                 raise EnvironmentError(
# #                     f"API key for {model} not found in environment."
# #                 )
# #             return api_key
# #     raise ValueError(f"Model {model} is not supported.")
# 
# 
# def test_all(seed=None, temperature=1.0):
#     model_names = [
#         "claude-3-5-sonnet-20240620",
#         # "gpt-4",
#         # "claude-3-opus-20240229",
#         # "gemini-pro",
#         # "llama-3-sonar-large-32k-online",
#     ]
# 
#     for model_name in model_names:
#         for stream in [False, True]:
#             model = GenAI(
#                 model_name, stream=stream, seed=seed, temperature=temperature
#             )
#             # prompt = "Hi. Tell me your name just within a line."
#             prompt = "Hi. Tell me about the hippocampus."
# 
#             print(
#                 f"\n{'-'*40}\n{model.model}\nStream: {stream}\nSeed: {seed}\nTemperature: {temperature}\n{'-'*40}"
#             )
#             print(model(prompt))
#             print(model.available_models)
# 
# 
# def main(
#     model="gemini-1.5-pro-latest",
#     stream=False,
#     prompt="Hi, please tell me about the hippocampus",
#     seed=None,
#     temperature=1.0,
# ):
# 
#     m = mngs.ai.GenAI(
#         model=model,
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         stream=stream,
#         seed=seed,
#         temperature=temperature,
#     )
#     out = m(prompt)
# 
#     return out
# 
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import mngs
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     test_all()
# 
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.ai._gen_ai._genai_factory import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
