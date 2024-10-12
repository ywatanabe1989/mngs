#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-13 20:39:12 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_AI/_ChatGPT.py


"""Imports"""
import os
import sys

import google.generativeai as genai
import matplotlib.pyplot as plt
import mngs
from mngs.ai._gen_ai._BaseGenAI import BaseGenAI

"""Functions & Classes"""


class Google(BaseGenAI):
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
        max_tokens=4096,
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
            max_tokens=max_tokens,
        )

        genai.configure(api_key=self.api_key)
        # genai.configure(api_key=self.api_key)

    def _init_client(
        self,
    ):
        genai.configure(api_key=self.api_key)
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
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

    # def _api_call_static(self):
    #     prompt = self.history[-1]["content"]
    #     response = self.client.send_message(prompt)
    #     if response.usage_metadata:
    #         self.input_tokens += response.usage_metadata.prompt_token_count
    #         self.output_tokens += (
    #             response.usage_metadata.candidates_token_count
    #         )
    #     out_text = response.text
    #     return out_text

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

    # def _api_call_stream(self):
    #     prompt = self.history[-1]["content"]
    #     responses = self.client.send_message(prompt, stream=True)
    #     for chunk in responses:
    #         if chunk:
    #             if chunk.usage_metadata:
    #                 self.input_tokens += (
    #                     chunk.usage_metadata.prompt_token_count
    #                 )
    #                 self.output_tokens += (
    #                     chunk.usage_metadata.candidates_token_count
    #                 )
    #             yield chunk.text


def main():
    # ai = mngs.ai.GenAI("gpt-4o")
    # ai = mngs.ai.GenAI("gemini-1.5-pro-latest", stream=False)
    ai = mngs.ai.GenAI("gemini-1.5-flash-latest", stream=False)
    # ai = mngs.ai.GenAI("gpt-4o", stream=False)
    print(ai("hi"))


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
