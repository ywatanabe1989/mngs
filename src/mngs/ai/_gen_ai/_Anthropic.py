#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-08 20:07:44 (ywatanabe)"
# File: ./mngs_repo/src/mngs/ai/_gen_ai/_Anthropic.py

"""
Functionality:
    - Implements Anthropic AI (Claude) interface
    - Handles both streaming and static text generation
Input:
    - User prompts and chat history
    - Model configurations and API credentials
Output:
    - Generated text responses from Claude models
    - Token usage statistics
Prerequisites:
    - Anthropic API key (ANTHROPIC_API_KEY environment variable)
    - anthropic package
"""

"""Imports"""
import os
import sys
from typing import Any, Dict, Generator, List, Optional, Union

import anthropic
import matplotlib.pyplot as plt

from ._BaseGenAI import BaseGenAI
import re

"""Functions & Classes"""
class Anthropic(BaseGenAI):
    def __init__(
        self,
        system_setting: str = "",
        api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY"),
        model: str = "claude-3-opus-20240229",
        stream: bool = False,
        seed: Optional[int] = None,
        n_keep: int = 1,
        temperature: float = 1.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 100_000,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

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

    def _init_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=self.api_key)

    def _api_call_static(self) -> str:
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

    def _api_call_stream(self) -> Generator[str, None, None]:
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
                except AttributeError:
                    pass

                if chunk.type == "content_block_delta":
                    yield chunk.delta.text


def main() -> None:
    pass

if __name__ == "__main__":
    import mngs
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)


# EOF
