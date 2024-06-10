#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 22:58:12 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_ChatGPT.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import sys

import anthropic
import matplotlib.pyplot as plt
import mngs

from ._BaseGenAI import BaseGenAI

# from mngs.gen import notify


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


class Claude(BaseGenAI):
    def __init__(
        self,
        system_setting="",
        api_key=os.getenv("Claude_API_KEY"),
        model="claude-3-opus-20240229",
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
        client = anthropic.Anthropic(
            api_key=self.api_key,
        )
        return client

    def _api_call_static(
        self,
    ):
        out_text = (
            self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=self.chat_history,
                temperature=self.temperature,
                # seed=self.seed, # fixme
            )
            .content[0]
            .text
        )
        return out_text

    def _api_call_stream(self):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            messages=self.chat_history,
            temperature=self.temperature,
            # seed=self.seed, # fixme
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _get_available_models(self):
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]


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
