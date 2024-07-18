#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-19 00:07:34 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_BaseAI.py


"""
This script does XYZ.
"""


"""
Imports
"""

import re
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mngs
from ansi_escapes import ansiEscapes

from ._format_output_func import format_output_func

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
    ):
        self.system_setting = system_setting
        self.model = model
        self.api_key = api_key
        self.stream = stream
        self.seed = seed
        self.n_keep = n_keep
        self.temperature = temperature

        self.reset(system_setting)
        self.verify_model()
        self.client = self._init_client()

    def __call__(self, prompt, format_output=True, return_stream=False):
        self.update_history("user", prompt)
        if prompt is None:
            prompt = ""

        if self.stream is False:
            return self._call_static(format_output)

        elif self.stream and (not return_stream):
            return self._yield_stream(self._call_stream(format_output))

        elif self.stream and return_stream:
            self.stream, _orig = return_stream, self.stream
            stream_obj = self._call_stream(format_output)
            self.stream = _orig
            return stream_obj

    def _yield_stream(self, stream_obj):
        accumulated = []
        for chunk in stream_obj:
            if chunk:
                # clean_chunk = clean_text(chunk)
                clean_chunk = chunk
                sys.stdout.write(clean_chunk)
                sys.stdout.flush()
                accumulated.append(chunk)
        accumulated = "".join(accumulated)
        self.update_history("assistant", accumulated)
        return accumulated

    def _call_static(self, format_output=True):
        try:
            out_text = self._api_call_static()

        except Exception as e:
            out_text = self._add_masked_api_key(f"Response timed out: {e}")
            mngs.gen.notify(message=out_text)

        out_text = format_output_func(out_text) if format_output else out_text
        self.update_history("assistant", out_text)
        return out_text

    def _call_stream(self, format_output=None):
        try:
            text_generator = self._api_call_stream()
            return text_generator

        except Exception as e:
            out_text = self._add_masked_api_key(f"Response timed out: {e}")
            mngs.gen.notify(message=out_text)
            return iter([out_text])

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

    @abstractmethod
    def _get_available_models(self):
        """Returns available models"""
        pass

    @property
    def available_models(self):
        return self._get_available_models()

    # history
    def reset(self, system_setting=""):
        self.history = []
        if system_setting != "":
            self.history.append(
                {
                    "role": "system",
                    "content": system_setting,
                }
            )

    def update_history(self, role, content):
        if self.history and self.history[-1]["role"] == role:
            # If the last entry in the history is from the same role, concatenate the content
            last_content = self.history[-1]["content"]
            concatenated_content = f"{last_content}\n{content}"
            self.history[-1]["content"] = concatenated_content
        else:
            # Otherwise, just append a new entry to the history
            self.history.append({"role": role, "content": content})

        # Trim the history to keep only the last 'n_keep' entries
        if len(self.history) > self.n_keep:
            self.history = self.history[-self.n_keep :]

    # def update_history(self, role="", content=""):
    #     self.history.append({"role": role, "content": content})
    #     if len(self.history) > self.n_keep:
    #         self.history = self.history[-self.n_keep :]

    def verify_model(
        self,
    ):
        if self.model not in self.available_models:
            message = (
                f"Specified model {self.model} is not supported. "
                f"Currently, available models are as follows:\n{self.available_models}"
            )
            message = self._add_masked_api_key(message)

            mngs.gen.notify(message=message)

            return message

    @property
    def masked_api_key(
        self,
    ):
        return f"{self.api_key[:4]}****{self.api_key[-4:]}"

    def _add_masked_api_key(self, text):
        return text + f"\n(API Key: {self.masked_api_key}"


def main():
    pass


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
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
