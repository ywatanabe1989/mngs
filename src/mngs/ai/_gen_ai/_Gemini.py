#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-29 15:12:11 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_ChatGPT.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import sys

import google.generativeai as genai
import matplotlib.pyplot as plt
import mngs

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


class Gemini(BaseGenAI):
    def __init__(
        self,
        system_setting="",
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-pro-latest",
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
            seed=seed,
            n_keep=n_keep,
            temperature=temperature,
            provider="Gemini",
            chat_history=chat_history,
        )
        genai.configure(api_key=self.api_key)

    def _init_client(
        self,
    ):
        genai.configure(api_key=self.api_key)
        generation_config = genai.GenerationConfig(
            temperature=self.temperature
        )
        client = genai.GenerativeModel(
            self.model, generation_config=generation_config
        )
        chat_client = client.start_chat(history=self.history)
        return chat_client

    def _api_call_static(
        self,
    ):
        prompt = self.history[-1]["content"]
        response = self.client.send_message(prompt)
        self.input_tokens += response.usage_metadata.prompt_token_count
        self.output_tokens += response.usage_metadata.candidates_token_count
        out_text = response.text
        return out_text

    def _api_call_stream(
        self,
    ):
        prompt = self.history[-1]["content"]
        responses = self.client.send_message(prompt, stream=True)
        for chunk in responses:
            if chunk:
                try:
                    self.input_tokens += (
                        chunk.usage_metadata.prompt_token_count
                    )
                except:
                    pass
                try:
                    self.output_tokens += (
                        chunk.usage_metadata.candidates_token_count
                    )
                except:
                    pass

                yield chunk.text


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
