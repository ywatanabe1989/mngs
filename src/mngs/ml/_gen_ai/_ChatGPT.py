#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 22:58:01 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_ChatGPT.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
from openai import OpenAI

from ._BaseGenAI import BaseGenAI

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


class ChatGPT(BaseGenAI):
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
        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=api_key,
            stream=stream,
            n_keep=n_keep,
            temperature=temperature,
        )

    def _init_client(
        self,
    ):
        client = OpenAI()
        return client

    def _api_call_static(self):
        out_text = (
            self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                seed=self.seed,
                stream=False,
                temperature=self.temperature,
            )
            .choices[0]
            .message.content
        )
        return out_text

    def _api_call_stream(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            max_tokens=4096,  # You can adjust this as needed
            n=1,
            stream=self.stream,
            seed=self.seed,
            temperature=self.temperature,
        )

        for chunk in stream:
            if chunk.choices:
                current_text = chunk.choices[0].delta.content
                if current_text:
                    yield current_text

    def _get_available_models(self):
        return [OpenAI(api_key=self.api_key).models.list()]
        # return [m.id for m in self.client.models.list()]


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
