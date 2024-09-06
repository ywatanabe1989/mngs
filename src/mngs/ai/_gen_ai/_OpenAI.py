#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-29 15:09:38 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_OpenAI.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs
from openai import OpenAI as _OpenAI

from mngs.ai._gen_ai._BaseGenAI import BaseGenAI

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


class OpenAI(BaseGenAI):
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
            provider="OpenAI",
            chat_history=chat_history,
        )

    def _init_client(
        self,
    ):
        client = _OpenAI(api_key=self.api_key)
        return client

    def _api_call_static(self):
        output = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            seed=self.seed,
            stream=False,
            temperature=self.temperature,
            max_tokens=4096,
        )
        self.input_tokens += output.usage.prompt_tokens
        self.output_tokens += output.usage.completion_tokens

        out_text = output.choices[0].message.content

        return out_text

    def _api_call_stream(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=4096,
            n=1,
            stream=self.stream,
            seed=self.seed,
            temperature=self.temperature,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if chunk:
                try:
                    self.input_tokens += chunk.usage.prompt_tokens
                except:
                    pass
                try:
                    self.output_tokens += chunk.usage.completion_tokens
                except:
                    pass

                try:
                    current_text = chunk.choices[0].delta.content
                    if current_text:
                        yield f"{current_text}"
                except Exception as e:
                    # print(e)
                    pass

    # def _get_available_models(self):
    #     return [m.id for m in OpenAI(api_key=self.api_key).models.list()]


def main():
    m = mngs.ai.GenAI("gpt-4o", stream=True)
    m("hi")
    # m("Hi")


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