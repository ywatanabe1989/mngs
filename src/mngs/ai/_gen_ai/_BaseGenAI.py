#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 21:08:29 (ywatanabe)"
# File: ./mngs_repo/src/mngs/ai/_gen_ai/_BaseGenAI.py

"""
Functionality:
    - Provides base class for generative AI model implementations
    - Handles chat history, error handling, and token tracking
    - Manages API calls in both streaming and static modes
Input:
    - Model configurations (API key, model name, system settings)
    - User prompts and chat history
Output:
    - Generated text responses (streaming or static)
    - Cost calculations and token usage statistics
Prerequisites:
    - API keys for respective AI providers
    - Model-specific implementation in child classes
"""

"""Imports"""
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import matplotlib.pyplot as plt

from ._calc_cost import calc_cost
from ._format_output_func import format_output_func
from .PARAMS import MODELS

"""Functions & Classes"""
def to_stream(string: Union[str, List[str]]) -> Generator[str, None, None]:
    """Converts string or list of strings to generator for streaming."""
    chunks = string if isinstance(string, list) else [string]
    for chunk in chunks:
        if chunk:
            yield chunk

class BaseGenAI(ABC):
    def __init__(
        self,
        system_setting: str = "",
        model: str = "",
        api_key: str = "",
        stream: bool = False,
        seed: Optional[int] = None,
        n_keep: int = 1,
        temperature: float = 1.0,
        provider: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 4_096,
    ) -> None:
        self.provider = provider
        self.system_setting = system_setting
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.seed = seed
        self.n_keep = n_keep
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self._error_messages: List[str] = []

        self.reset(system_setting)
        self.history = chat_history if chat_history else []

        try:
            self.verify_model()
            self.client = self._init_client()
        except Exception as error:
            print(error)
            self._error_messages.append(f"\nError:\n{str(error)}")

    def gen_error(self, return_stream: bool) -> tuple[bool, Optional[Union[str, Generator]]]:
        error_exists = bool(self._error_messages)
        if not error_exists:
            return False, None

        error_msgs = self._error_messages
        self._error_messages = []

        if not self.stream:
            return True, "".join(error_msgs)

        stream_obj = to_stream(error_msgs)
        return True, self._yield_stream(stream_obj) if not return_stream else stream_obj

    # def __call_chunks__(self, prompt: Optional[str], format_output: bool = False, return_stream: bool = False):
    #     if len(prompt) > self.chunk_size:
    #         chunks = [prompt[i:i+self.chunk_size]
    #                  for i in range(0, len(prompt), self.chunk_size)]

    #         # Send chunks for understanding
    #         for i, chunk in enumerate(chunks[:-1]):
    #             self._process_chunk(chunk, f"Part {i+1}/{len(chunks)}")

    #         # Final chunk with request for response
    #         final_prompt = (
    #             f"{chunks[-1]}\n\n"
    #             "Above are all the changes. Based on ALL the changes shown above, "
    #             "generate a single, comprehensive commit message."
    #         )
    #         return super().__call_simple__(final_prompt, format_output, return_stream)

    #     return super().__call_simple__(prompt, format_output, return_stream)

    # def _process_chunk(self, chunk: str, part_label: str):
    #     prompt = f"{part_label}:\n{chunk}\n\nPlease read and understand this part. No response needed yet."
    #     self.update_history("user", prompt)
    #     self.update_history("assistant", "Understood.")

    def __call__(
        self,
        prompt: Optional[str],
        format_output: bool = False,
        return_stream: bool = False
    ) -> Union[str, Generator]:
        self.update_history("user", prompt or "")

        error_flag, error_obj = self.gen_error(return_stream)
        if error_flag:
            return error_obj

        try:
            if not self.stream:
                return self._call_static(format_output)

            if return_stream:
                self.stream, orig_stream = return_stream, self.stream
                stream_obj = self._call_stream(format_output)
                self.stream = orig_stream
                return stream_obj

            return self._yield_stream(self._call_stream(format_output))

        except Exception as error:
            self._error_messages.append(f"\nError:\n{str(error)}")
            error_flag, error_obj = self.gen_error(return_stream)
            if error_flag:
                return error_obj

    def _yield_stream(self, stream_obj: Generator) -> str:
        accumulated = []
        for chunk in stream_obj:
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                accumulated.append(chunk)
        result = "".join(accumulated)
        self.update_history("assistant", result)
        return result

    def _call_static(self, format_output: bool = True) -> str:
        out_text = self._api_call_static()
        out_text = format_output_func(out_text) if format_output else out_text
        self.update_history("assistant", out_text)
        return out_text

    def _call_stream(self, format_output: Optional[bool] = None) -> Generator:
        return self._api_call_stream()

    @abstractmethod
    def _init_client(self) -> Any:
        """Returns client"""
        pass

    @abstractmethod
    def _api_call_static(self) -> str:
        """Returns out_text"""
        pass

    @abstractmethod
    def _api_call_stream(self) -> Generator:
        """Returns stream"""
        pass

    def _get_available_models(self) -> List[str]:
        indi = [
            self.provider.lower() in api_key_env.lower()
            for api_key_env in MODELS["api_key_env"]
        ]
        return MODELS[indi].name.tolist()

    @property
    def available_models(self) -> List[str]:
        return self._get_available_models()

    def reset(self, system_setting: str = "") -> None:
        self.history = []
        if system_setting:
            self.history.append({"role": "system", "content": system_setting})

    def _ensure_alternative_history(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if len(history) < 2:
            return history

        if history[-1]["role"] == history[-2]["role"]:
            last_content = history.pop()["content"]
            history[-1]["content"] += f"\n\n{last_content}"
            return self._ensure_alternative_history(history)

        return history

    @staticmethod
    def _ensure_start_from_user(
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if history and history[0]["role"] != "user":
            history.pop(0)
        return history

    def update_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

        if len(self.history) > self.n_keep:
            self.history = self.history[-self.n_keep:]

        self.history = self._ensure_alternative_history(self.history)
        self.history = self._ensure_start_from_user(self.history)

    def verify_model(self) -> None:
        if self.model not in self.available_models:
            message = (
                f"Specified model {self.model} is not supported for the API Key ({self.masked_api_key}). "
                f"Available models for {str(self)} are as follows:\n{self.available_models}"
            )
            raise ValueError(message)

    @property
    def masked_api_key(self) -> str:
        return f"{self.api_key[:4]}****{self.api_key[-4:]}"

    def _add_masked_api_key(self, text: str) -> str:
        return text + f"\n(API Key: {self.masked_api_key}"

    @property
    def cost(self) -> float:
        return calc_cost(self.model, self.input_tokens, self.output_tokens)

def main() -> None:
    pass

if __name__ == "__main__":
    import mngs
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 01:27:50 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/_BaseGenAI.py

# """
# Functionality:
#     - Provides base class for generative AI model implementations
#     - Handles chat history, error handling, and token tracking
#     - Manages API calls in both streaming and static modes
# Input:
#     - Model configurations (API key, model name, system settings)
#     - User prompts and chat history
# Output:
#     - Generated text responses (streaming or static)
#     - Cost calculations and token usage statistics
# Prerequisites:
#     - API keys for respective AI providers
#     - Model-specific implementation in child classes
# """


# """Imports"""
# import sys
# from abc import ABC, abstractmethod

# import matplotlib.pyplot as plt

# from ._calc_cost import calc_cost
# from ._format_output_func import format_output_func
# from .PARAMS import MODELS

# """Functions & Classes"""
# class BaseGenAI(ABC):
#     def __init__(
#         self,
#         system_setting="",
#         model="",
#         api_key="",
#         stream=False,
#         seed=None,
#         n_keep=1,
#         temperature=1.0,
#         provider="",
#         chat_history=None,
#         max_tokens=4096,
#     ):
#         # Attributes
#         self.provider = provider
#         self.system_setting = system_setting
#         self.model = model
#         self.api_key = api_key
#         self.stream = stream
#         self.seed = seed
#         self.n_keep = n_keep
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.input_tokens = 0
#         self.output_tokens = 0

#         # Initialization
#         self.reset(system_setting)
#         self.history = chat_history if chat_history else []
#         # Errror handling
#         # Store Error Messages until the main function call
#         # to send the error message as output
#         self._error_messages = []

#         try:
#             self.verify_model()  # fixme for Gemini
#             self.client = self._init_client()
#         except Exception as e:
#             print(e)
#             message = f"\nError:\n{str(e)}"
#             self._error_messages.append(message)

#     def gen_error(self, return_stream):
#         """Return error messages in the same format of expected call function"""

#         error_exists = False
#         return_obj = None

#         if self._error_messages:
#             error_exists = True

#             # Reset the stored error messages
#             error_messages = self._error_messages
#             self._error_messages = []

#             # Static
#             if self.stream is False:
#                 return_obj = "".join(error_messages)

#             # Streaming
#             else:
#                 stream_obj_error = to_stream(error_messages)
#                 if not return_stream:
#                     return_obj = self._yield_stream(stream_obj_error)

#                 elif return_stream:
#                     return_obj = stream_obj_error

#         return error_exists, return_obj

#     def __call__(self, prompt, format_output=False, return_stream=False):
#         self.update_history("user", prompt)
#         if prompt is None:
#             prompt = ""

#         error_flag, error_obj = self.gen_error(return_stream)

#         # Static
#         try:
#             if self.stream is False:
#                 if error_flag:
#                     return error_obj
#                 else:
#                     return self._call_static(format_output)

#             else:
#                 if not return_stream:
#                     if error_flag:
#                         return error_obj
#                     else:
#                         return self._yield_stream(
#                             self._call_stream(format_output)
#                         )

#                 elif return_stream:
#                     if error_flag:
#                         return error_obj
#                     else:
#                         self.stream, _orig = return_stream, self.stream
#                         stream_obj = self._call_stream(format_output)
#                         self.stream = _orig
#                         return stream_obj

#         except Exception as e:
#             message = f"\nError:\n{str(e)}"
#             self._error_messages.append(message)
#             error_flag, error_obj = self.gen_error(return_stream)
#             if error_flag:
#                 return error_obj

#     def _yield_stream(self, stream_obj):
#         accumulated = []
#         for chunk in stream_obj:
#             if chunk:
#                 sys.stdout.write(chunk)
#                 sys.stdout.flush()
#                 accumulated.append(chunk)
#         accumulated = "".join(accumulated)
#         self.update_history("assistant", accumulated)
#         return accumulated

#     def _call_static(self, format_output=True):
#         out_text = self._api_call_static()
#         out_text = format_output_func(out_text) if format_output else out_text
#         self.update_history("assistant", out_text)
#         return out_text

#     def _call_stream(self, format_output=None):
#         text_generator = self._api_call_stream()
#         return text_generator

#     @abstractmethod
#     def _init_client(self):
#         """Returns client"""
#         pass

#     @abstractmethod
#     def _api_call_static(self):
#         """Returns out_text"""
#         pass

#     @abstractmethod
#     def _api_call_stream(self):
#         """Returns stream"""
#         pass

#     def _get_available_models(self):
#         indi = [
#             self.provider.lower() in api_key_env.lower()
#             for api_key_env in MODELS["api_key_env"]
#         ]
#         return MODELS[indi].name.tolist()

#     @property
#     def available_models(self):
#         return self._get_available_models()

#     def reset(self, system_setting=""):
#         self.history = []
#         if system_setting != "":
#             self.history.append(
#                 {
#                     "role": "system",
#                     "content": system_setting,
#                 }
#             )

#     def _ensure_alternative_history(self, history):
#         if len(history) < 2:
#             return history

#         if history[-1]["role"] == history[-2]["role"]:
#             last_content = history.pop()["content"]
#             history[-1]["content"] += f"\n\n{last_content}"
#             return self._ensure_alternative_history(history)

#         return history

#     @staticmethod
#     def _ensure_start_from_user(history):
#         if history[0]["role"] != "user":
#             history.pop(0)
#         return history

#     def update_history(self, role, content):
#         self.history.append({"role": role, "content": content})

#         # Trim the history to keep only the last 'n_keep' entries
#         if len(self.history) > self.n_keep:
#             self.history = self.history[-self.n_keep :]

#         self.history = self._ensure_alternative_history(self.history)
#         self.history = self._ensure_start_from_user(self.history)

#     def verify_model(
#         self,
#     ):

#         if self.model not in self.available_models:
#             message = (
#                 f"Specified model {self.model} is not supported for the API Key ({self.masked_api_key}). "
#                 f"Available models for {str(self)} are as follows:\n{self.available_models}"
#             )
#             raise ValueError(message)

#     @property
#     def masked_api_key(
#         self,
#     ):
#         return f"{self.api_key[:4]}****{self.api_key[-4:]}"

#     def _add_masked_api_key(self, text):
#         return text + f"\n(API Key: {self.masked_api_key}"

#     @property
#     def cost(
#         self,
#     ):
#         return calc_cost(self.model, self.input_tokens, self.output_tokens)


# def to_stream(string):
#     chunks = string
#     for chunk in chunks:
#         if chunk:
#             yield chunk

# def main():
#     pass


# if __name__ == "__main__":
#     import mngs
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

#

# EOF
