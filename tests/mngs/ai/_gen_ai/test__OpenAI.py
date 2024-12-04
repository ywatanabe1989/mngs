# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-01 06:26:23 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_OpenAI.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_OpenAI.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-01 06:25:27 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_OpenAI.py
# 
# import os
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_OpenAI.py"
# 
# """Imports"""
# from openai import OpenAI as _OpenAI
# 
# from ._BaseGenAI import BaseGenAI
# 
# """Functions & Classes"""
# 
# 
# class OpenAI(BaseGenAI):
#     def __init__(
#         self,
#         system_setting="",
#         model="",
#         api_key=os.getenv("OPENAI_API_KEY"),
#         stream=False,
#         seed=None,
#         n_keep=1,
#         temperature=1.0,
#         chat_history=None,
#         max_tokens=None,
#     ):
#         # Set max_tokens based on model
#         if max_tokens is None:
#             if "gpt-4-turbo" in model:
#                 max_tokens = 128_000
#             elif "gpt-4" in model:
#                 max_tokens = 8_192
#             elif "gpt-3.5-turbo-16k" in model:
#                 max_tokens = 16_384
#             elif "gpt-3.5" in model:
#                 max_tokens = 4_096
#             else:
#                 max_tokens = 4_096
# 
#         super().__init__(
#             system_setting=system_setting,
#             model=model,
#             api_key=api_key,
#             stream=stream,
#             n_keep=n_keep,
#             temperature=temperature,
#             provider="OpenAI",
#             chat_history=chat_history,
#             max_tokens=max_tokens,
#         )
# 
#     def _init_client(
#         self,
#     ):
#         client = _OpenAI(api_key=self.api_key)
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
#         if kwargs.get("model") in ["o1-mini", "o1-preview"]:
#             kwargs.pop("max_tokens")
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
#             stream_options={"include_usage": True},
#         )
# 
#         if kwargs.get("model") in ["o1-mini", "o1-preview"]:
#             full_response = self._api_call_static()
#             for char in full_response:
#                 yield char
#             return
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
#                         # Yield complete sentences or words
#                         if any(char in ".!?\n " for char in current_text):
#                             yield buffer
#                             buffer = ""
#                 except Exception as e:
#                     pass
# 
#         # Yield any remaining text
#         if buffer:
#             yield buffer
# 
#     def _api_format_history(self, history):
#         formatted_history = []
#         for msg in history:
#             if isinstance(msg["content"], list):
#                 content = []
#                 for item in msg["content"]:
#                     if item["type"] == "text":
#                         content.append({"type": "text", "text": item["text"]})
#                     elif item["type"] == "_image":
#                         content.append(
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/jpeg;base64,{item['_image']}"
#                                 },
#                             }
#                         )
#                 formatted_msg = {"role": msg["role"], "content": content}
#             else:
#                 formatted_msg = {
#                     "role": msg["role"],
#                     "content": msg["content"],
#                 }
#             formatted_history.append(formatted_msg)
#         return formatted_history
# 
# 
# def main() -> None:
#     import mngs
# 
#     ai = mngs.ai.GenAI(
#         model="gpt-4o",
#         api_key=os.getenv("OPENAI_API_KEY"),
#     )
#     print(
#         ai(
#             "hi, could you tell me what is in the pic?",
#             images=[
#                 "/home/ywatanabe/Downloads/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#             ],
#         )
#     )
#     pass
# 
# 
# # def main():
# #     model = "o1-mini"
# #     # model = "o1-preview"
# #     # model = "gpt-4o"
# #     stream = True
# #     max_tokens = 4906
# #     m = mngs.ai.GenAI(model, stream=stream, max_tokens=max_tokens)
# #     m("hi")
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
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF
# """
# python -m mngs.ai._gen_ai._OpenAI
# """
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

from mngs..ai._gen_ai._OpenAI import *

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