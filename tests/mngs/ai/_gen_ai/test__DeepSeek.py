# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/_gen_ai/_DeepSeek.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-08 20:33:49 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_DeepSeek.py
# 
# """
# 1. Functionality:
#    - Implements DeepSeek Code LLM API interface
# 2. Input:
#    - Text prompts for code generation
# 3. Output:
#    - Generated code responses (streaming or static)
# 4. Prerequisites:
#    - DEEPSEEK_API_KEY environment variable
#    - requests library
# """
# 
# """Imports"""
# import json
# import os
# import sys
# from typing import Dict, Generator, List, Optional
# 
# import mngs
# import requests
# 
# from ._BaseGenAI import BaseGenAI
# 
# """Warnings"""
# # mngs.pd.ignore_SettingWithCopyWarning()
# # warnings.simplefilter("ignore", UserWarning)
# # with warnings.catch_warnings():
# #     warnings.simplefilter("ignore", UserWarning)
# 
# """Parameters"""
# # from mngs.io import load_configs
# # CONFIG = load_configs()
# 
# """Functions & Classes"""
# """Imports"""
# from ._BaseGenAI import BaseGenAI
# from openai import OpenAI as _OpenAI
# 
# """Functions & Classes"""
# 
# class DeepSeek(BaseGenAI):
#     def __init__(
#         self,
#         system_setting="",
#         model="deepseek-chat",
#         api_key="",
#         stream=False,
#         seed=None,
#         n_keep=1,
#         temperature=1.0,
#         chat_history=None,
#         max_tokens=4096,
#     ):
#         super().__init__(
#             system_setting=system_setting,
#             model=model,
#             api_key=api_key,
#             stream=stream,
#             n_keep=n_keep,
#             temperature=temperature,
#             provider="DeepSeek",
#             chat_history=chat_history,
#             max_tokens=max_tokens,
#         )
# 
#     def _init_client(self):
#         # client = _OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
#         client = _OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/beta")
# 
#         return client
# 
#     def _api_call_static(self):
#         kwargs = dict(
#             model=self.model,
#             messages=self.history,
#             seed=self.seed,
#             stream=False,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#         )
# 
#         output = self.client.chat.completions.create(**kwargs)
#         self.input_tokens += output.usage.prompt_tokens
#         self.output_tokens += output.usage.completion_tokens
# 
#         out_text = output.choices[0].message.content
# 
#         return out_text
# 
#     def _api_call_stream(self):
#         kwargs = dict(
#             model=self.model,
#             messages=self.history,
#             max_tokens=self.max_tokens,
#             n=1,
#             stream=self.stream,
#             seed=self.seed,
#             temperature=self.temperature,
#         )
# 
#         stream = self.client.chat.completions.create(**kwargs)
#         buffer = ""
# 
#         for chunk in stream:
#             if chunk:
#                 try:
#                     self.input_tokens += chunk.usage.prompt_tokens
#                 except:
#                     pass
#                 try:
#                     self.output_tokens += chunk.usage.completion_tokens
#                 except:
#                     pass
# 
#                 try:
#                     current_text = chunk.choices[0].delta.content
#                     if current_text:
#                         buffer += current_text
#                         if any(char in '.!?\n ' for char in current_text):
#                             yield buffer
#                             buffer = ""
#                 except Exception as e:
#                     pass
# 
#         if buffer:
#             yield buffer
# 
# if __name__ == '__main__':
#     # -----------------------------------
#     # Initiatialization of mngs format
#     # -----------------------------------
#     import sys
# 
#     import matplotlib.pyplot as plt
# 
#     # Configurations
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys,
#         plt,
#         verbose=False,
#         agg=True,
#         # sdir_suffix="",
#     )
# 
#     # # Argument parser
#     # script_mode = mngs.gen.is_script()
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, choices=None, default=1, help='(default: %%(default)s)')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='(default: %%(default)s)')
#     # args = parser.parse_args()
#     # mngs.gen.print_block(args, c='yellow')
# 
#     # -----------------------------------
#     # Main
#     # -----------------------------------
#     exit_status = main()
# 
#     # -----------------------------------
#     # Cleanup mngs format
#     # -----------------------------------
#     mngs.gen.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         message="",
#         exit_status=exit_status,
#     )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/_gen_ai/_DeepSeek.py
# --------------------------------------------------------------------------------
