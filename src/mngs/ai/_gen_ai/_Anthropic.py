#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-13 20:10:40 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_ChatGPT.py


"""Imports"""
import os
import sys

import anthropic
import matplotlib.pyplot as plt
import mngs

from ._BaseGenAI import BaseGenAI

"""Functions & Classes"""


class Anthropic(BaseGenAI):
    def __init__(
        self,
        system_setting="",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-opus-20240229",
        stream=False,
        seed=None,
        n_keep=1,
        temperature=1.0,
        chat_history=None,
        max_tokens=4096,
    ):
        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=api_key,
            stream=stream,
            n_keep=n_keep,
            temperature=temperature,
            provider="Anthropic",
            chat_history=chat_history,
            max_tokens=max_tokens,
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
        output = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.history,
            temperature=self.temperature,
        )
        out_text = output.content[0].text

        self.input_tokens += output.usage.input_tokens
        self.output_tokens += output.usage.output_tokens

        return out_text

    def _api_call_stream(self):

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self.history,
            temperature=self.temperature,
        ) as stream:
            for chunk in stream:

                try:
                    self.input_tokens += chunk.message.usage.input_tokens
                    self.output_tokens += chunk.message.usage.output_tokens
                except:
                    pass
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text

    # def _get_available_models(self):
    #     return [
    #         "claude-3-5-sonnet-20240620",
    #         "claude-3-opus-20240229",
    #         "claude-3-sonnet-20240229",
    #         "claude-3-haiku-20240307",
    #     ]


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
