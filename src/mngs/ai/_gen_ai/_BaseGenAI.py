#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-27 13:01:38 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_BaseAI.py


"""This script does XYZ."""


"""Imports"""

import re
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mngs
from ansi_escapes import ansiEscapes

from ._calc_cost import calc_cost
from ._format_output_func import format_output_func
from .PARAMS import MODELS

"""Functions & Classes"""


def to_stream(string):
    chunks = string
    for chunk in chunks:
        if chunk:
            yield chunk


def clean_text(chunk):
    """
    Removes ANSI escape sequences from a given text chunk.

    Parameters:
    - chunk (str): The text chunk to be cleaned.

    Returns:
    - str: The cleaned text chunk.
    """
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", chunk)


class BaseGenAI(ABC):
    def __init__(
        self,
        system_setting="",
        model="",
        api_key="",
        stream=False,
        seed=None,
        n_keep=1,
        temperature=1.0,
        provider="",
        chat_history=None,
        max_tokens=4096,
    ):
        # Attributes
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

        # Initialization
        self.reset(system_setting)
        self.history = chat_history if chat_history else []
        # Errror handling
        # Store Error Messages until the main function call
        # to send the error message as output
        self._error_messages = []

        try:
            self.verify_model()  # fixme for Gemini
            self.client = self._init_client()
        except Exception as e:
            print(e)
            message = f"\nError:\n{str(e)}"
            self._error_messages.append(message)

    def gen_error(self, return_stream):
        """Return error messages in the same format of expected call function"""

        error_exists = False
        return_obj = None

        if self._error_messages:
            error_exists = True

            # Reset the stored error messages
            error_messages = self._error_messages
            self._error_messages = []

            # Static
            if self.stream is False:
                return_obj = "".join(error_messages)

            # Streaming
            else:
                stream_obj_error = to_stream(error_messages)
                if not return_stream:
                    return_obj = self._yield_stream(stream_obj_error)

                elif return_stream:
                    return_obj = stream_obj_error

        return error_exists, return_obj

    def __call__(self, prompt, format_output=False, return_stream=False):
        self.update_history("user", prompt)
        if prompt is None:
            prompt = ""

        error_flag, error_obj = self.gen_error(return_stream)

        # Static
        try:
            if self.stream is False:
                if error_flag:
                    return error_obj
                else:
                    return self._call_static(format_output)

            else:
                if not return_stream:
                    if error_flag:
                        return error_obj
                    else:
                        return self._yield_stream(
                            self._call_stream(format_output)
                        )

                elif return_stream:
                    if error_flag:
                        return error_obj
                    else:
                        self.stream, _orig = return_stream, self.stream
                        stream_obj = self._call_stream(format_output)
                        self.stream = _orig
                        return stream_obj

        except Exception as e:
            message = f"\nError:\n{str(e)}"
            self._error_messages.append(message)
            error_flag, error_obj = self.gen_error(return_stream)
            if error_flag:
                return error_obj

    def _yield_stream(self, stream_obj):
        accumulated = []
        for chunk in stream_obj:
            if chunk:
                clean_chunk = chunk
                sys.stdout.write(clean_chunk)
                sys.stdout.flush()
                accumulated.append(chunk)
        accumulated = "".join(accumulated)
        self.update_history("assistant", accumulated)
        return accumulated

    def _call_static(self, format_output=True):
        out_text = self._api_call_static()
        out_text = format_output_func(out_text) if format_output else out_text
        self.update_history("assistant", out_text)
        return out_text

    def _call_stream(self, format_output=None):
        text_generator = self._api_call_stream()
        return text_generator

    @abstractmethod
    def _init_client(self):
        """Returns client"""
        pass

    @abstractmethod
    def _api_call_static(self):
        """Returns out_text"""
        pass

    @abstractmethod
    def _api_call_stream(self):
        """Returns stream"""
        pass

    def _get_available_models(self):
        indi = [
            self.provider.lower() in api_key_env.lower()
            for api_key_env in MODELS["api_key_env"]
        ]
        return MODELS[indi].name.tolist()

    @property
    def available_models(self):
        return self._get_available_models()

    def reset(self, system_setting=""):
        self.history = []
        if system_setting != "":
            self.history.append(
                {
                    "role": "system",
                    "content": system_setting,
                }
            )

    def _ensure_alternative_history(self, history):
        if len(history) < 2:
            return history

        if history[-1]["role"] == history[-2]["role"]:
            last_content = history.pop()["content"]
            history[-1]["content"] += f"\n\n{last_content}"
            return self._ensure_alternative_history(history)

        return history

    @staticmethod
    def _ensure_start_from_user(history):
        if history[0]["role"] != "user":
            history.pop(0)
        return history

    def update_history(self, role, content):
        self.history.append({"role": role, "content": content})

        # Trim the history to keep only the last 'n_keep' entries
        if len(self.history) > self.n_keep:
            self.history = self.history[-self.n_keep :]

        self.history = self._ensure_alternative_history(self.history)
        self.history = self._ensure_start_from_user(self.history)

    def verify_model(
        self,
    ):

        if self.model not in self.available_models:
            message = (
                f"Specified model {self.model} is not supported for the API Key ({self.masked_api_key}). "
                f"Available models for {str(self)} are as follows:\n{self.available_models}"
            )
            raise ValueError(message)

    @property
    def masked_api_key(
        self,
    ):
        return f"{self.api_key[:4]}****{self.api_key[-4:]}"

    def _add_masked_api_key(self, text):
        return text + f"\n(API Key: {self.masked_api_key}"

    @property
    def cost(
        self,
    ):
        return calc_cost(self.model, self.input_tokens, self.output_tokens)


def main():
    pass


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
