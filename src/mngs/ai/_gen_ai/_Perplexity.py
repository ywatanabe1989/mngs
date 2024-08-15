#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-15 10:29:46 (ywatanabe)"
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

# from mngs.gen import notify
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


class Perplexity(BaseGenAI):
    def __init__(
        self,
        system_setting="",
        model="",
        api_key="",
        stream=False,
        seed=None,
        n_keep=1,
        temperature=1.0,
        chat_history=None,
    ):
        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=api_key,
            stream=stream,
            n_keep=n_keep,
            temperature=temperature,
            provider="Perplexity",
            chat_history=chat_history,
        )

    def _init_client(
        self,
    ):
        client = OpenAI(
            api_key=self.api_key, base_url="https://api.perplexity.ai"
        )
        return client

    def _api_call_static(self):
        output = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=4096,
            stream=False,
            temperature=self.temperature,
            # return_citations=True,
        )

        out_text = output.choices[0].message.content
        self.input_tokens += output.usage.prompt_tokens
        self.output_tokens += output.usage.completion_tokens

        return out_text

    def _api_call_stream(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=4096,
            n=1,
            stream=self.stream,
            temperature=self.temperature,
            # return_citations=True,
        )

        for chunk in stream:
            if chunk:
                if chunk.choices[0].finish_reason == "stop":
                    try:
                        self.input_tokens += chunk.usage.prompt_tokens
                    except:
                        pass
                    try:
                        self.output_tokens += chunk.usage.completion_tokens
                    except:
                        pass

            if chunk.choices:
                current_text = chunk.choices[0].delta.content
                if current_text:
                    yield current_text

    def _get_available_models(self):
        return [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3-sonar-small-32k-chat",
            "llama-3-sonar-small-32k-online",
            "llama-3-sonar-large-32k-chat",
            "llama-3-sonar-large-32k-online",
            "llama-3-8b-instruct",
            "llama-3-70b-instruct",
            "mixtral-8x7b-instruct",
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
