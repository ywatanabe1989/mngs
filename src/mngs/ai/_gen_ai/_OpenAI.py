#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 21:15:56 (ywatanabe)"
# File: ./mngs_repo/src/mngs/ai/_gen_ai/_OpenAI.py

"""Imports"""
from ._BaseGenAI import BaseGenAI
from openai import OpenAI as _OpenAI

"""Functions & Classes"""
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
        max_tokens=None,
    ):
        # Set max_tokens based on model
        if max_tokens is None:
            if "gpt-4-turbo" in model:
                max_tokens = 128_000
            elif "gpt-4" in model:
                max_tokens = 8_192
            elif "gpt-3.5-turbo-16k" in model:
                max_tokens = 16_384
            elif "gpt-3.5" in model:
                max_tokens = 4_096
            else:
                max_tokens = 4_096

        super().__init__(
            system_setting=system_setting,
            model=model,
            api_key=api_key,
            stream=stream,
            n_keep=n_keep,
            temperature=temperature,
            provider="OpenAI",
            chat_history=chat_history,
            max_tokens=max_tokens,
        )

    def _init_client(
        self,
    ):
        client = _OpenAI(api_key=self.api_key)
        return client

    def _api_call_static(self):
        kwargs = dict(
            model=self.model,
            messages=self.history,
            seed=self.seed,
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if kwargs.get("model") in ["o1-mini", "o1-preview"]:
            kwargs.pop("max_tokens")

        output = self.client.chat.completions.create(**kwargs)
        self.input_tokens += output.usage.prompt_tokens
        self.output_tokens += output.usage.completion_tokens

        out_text = output.choices[0].message.content

        return out_text

    # def _api_call_stream(self):
    #     kwargs = dict(
    #         model=self.model,
    #         messages=self.history,
    #         max_tokens=self.max_tokens,
    #         n=1,
    #         stream=self.stream,
    #         seed=self.seed,
    #         temperature=self.temperature,
    #         stream_options={"include_usage": True},
    #     )

    #     if kwargs.get("model") in ["o1-mini", "o1-preview"]:
    #         full_response = self._api_call_static()
    #         for char in full_response:
    #             yield char
    #         return

    #     stream = self.client.chat.completions.create(**kwargs)

    #     for chunk in stream:
    #         if chunk:
    #             try:
    #                 self.input_tokens += chunk.usage.prompt_tokens
    #             except:
    #                 pass
    #             try:
    #                 self.output_tokens += chunk.usage.completion_tokens
    #             except:
    #                 pass

    #             try:
    #                 current_text = chunk.choices[0].delta.content
    #                 if current_text:
    #                     yield f"{current_text}"
    #             except Exception as e:
    #                 pass

    def _api_call_stream(self):
        kwargs = dict(
            model=self.model,
            messages=self.history,
            max_tokens=self.max_tokens,
            n=1,
            stream=self.stream,
            seed=self.seed,
            temperature=self.temperature,
            stream_options={"include_usage": True},
        )

        if kwargs.get("model") in ["o1-mini", "o1-preview"]:
            full_response = self._api_call_static()
            for char in full_response:
                yield char
            return

        stream = self.client.chat.completions.create(**kwargs)
        buffer = ""

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
                        buffer += current_text
                        # Yield complete sentences or words
                        if any(char in '.!?\n ' for char in current_text):
                            yield buffer
                            buffer = ""
                except Exception as e:
                    pass

        # Yield any remaining text
        if buffer:
            yield buffer


def main():
    model = "o1-mini"
    # model = "o1-preview"
    # model = "gpt-4o"
    stream = True
    max_tokens = 4906
    m = mngs.ai.GenAI(model, stream=stream, max_tokens=max_tokens)
    m("hi")


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
