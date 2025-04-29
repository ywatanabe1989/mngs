# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/_gen_ai/_Llama.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 21:11:08 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_Llama.py
# 
# """Imports"""
# import os
# import sys
# from typing import List, Optional
# 
# import matplotlib.pyplot as plt
# import mngs
# 
# try:
#     from llama import Dialog
#     from llama import Llama as _Llama
# except:
#     pass
# 
# from ._BaseGenAI import BaseGenAI
# 
# """Functions & Classes"""
# def print_envs():
#     settings = {
#         "MASTER_ADDR": os.getenv("MASTER_ADDR", "localhost"),
#         "MASTER_PORT": os.getenv("MASTER_PORT", "12355"),
#         "WORLD_SIZE": os.getenv("WORLD_SIZE", "1"),
#         "RANK": os.getenv("RANK", "0"),
#     }
# 
#     print("Environment Variable Settings:")
#     for key, value in settings.items():
#         print(f"{key}: {value}")
#     print()
# 
# 
# class Llama(BaseGenAI):
#     def __init__(
#         self,
#         ckpt_dir: str = "",
#         tokenizer_path: str = "",
#         system_setting: str = "",
#         model: str = "Meta-Llama-3-8B",
#         max_seq_len: int = 32_768,
#         max_batch_size: int = 4,
#         max_gen_len: Optional[int] = None,
#         stream: bool = False,
#         seed: Optional[int] = None,
#         n_keep: int = 1,
#         temperature: float = 1.0,
#         provider="Llama",
#         chat_history=None,
#         **kwargs,
#     ):
# 
#         # Configure environment variables
#         os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
#         os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
#         os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
#         os.environ["RANK"] = os.getenv("RANK", "0")
#         print_envs()
# 
#         self.ckpt_dir = (
#             ckpt_dir if ckpt_dir else f"Meta-{model}/"
#         )
#         self.tokenizer_path = (
#             tokenizer_path
#             if tokenizer_path
#             else f"./Meta-{model}/tokenizer.model"
#         )
#         self.max_seq_len = max_seq_len
#         self.max_batch_size = max_batch_size
#         self.max_gen_len = max_gen_len
# 
#         super().__init__(
#             system_setting=system_setting,
#             model=model,
#             api_key="",
#             stream=stream,
#             seed=seed,
#             n_keep=n_keep,
#             temperature=temperature,
#             chat_history=chat_history,
#         )
# 
#     def __str__(self):
#         return "Llama"
# 
#     def _init_client(self):
#         generator = _Llama.build(
#             ckpt_dir=self.ckpt_dir,
#             tokenizer_path=self.tokenizer_path,
#             max_seq_len=self.max_seq_len,
#             max_batch_size=self.max_batch_size,
#         )
#         return generator
# 
#     def _api_call_static(self):
#         dialogs: List[Dialog] = [self.history]
#         results = self.client.chat_completion(
#             dialogs,
#             max_gen_len=self.max_gen_len,
#             temperature=self.temperature,
#             top_p=0.9,
#         )
#         out_text = results[0]["generation"]["content"]
#         return out_text
# 
#     def _api_call_stream(self):
#         # Llama3 doesn't have built-in streaming, so we'll simulate it
#         full_response = self._api_call_static()
#         for char in full_response:
#             yield char
# 
#     # def _get_available_models(self):
#     #     # Llama3 doesn't have a list of available models, so we'll return a placeholder
#     #     return ["llama3"]
# 
#     def verify_model(self):
#         # Llama3 doesn't require model verification, so we'll skip it
#         pass
# 
# 
# def main():
#     m = Llama(
#         ckpt_dir="/path/to/checkpoint",
#         tokenizer_path="/path/to/tokenizer",
#         system_setting="You are a helpful assistant.",
#         max_seq_len=512,
#         max_batch_size=4,
#         stream=True,
#         temperature=0.7,
#     )
#     m("Hi")
#     pass
# 
# 
# if __name__ == "__main__":
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/_gen_ai/_Llama.py
# --------------------------------------------------------------------------------
