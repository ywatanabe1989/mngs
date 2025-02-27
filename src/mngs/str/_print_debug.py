#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 17:17:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/str/_print_debug.py

__file__ = "./src/mngs/str/_print_debug.py"

from ._printc import printc


def print_debug():
    printc(
        (
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*24} DEBUG MODE {'!'*24}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}\n"
            f"{'!'*60}"
        ),
        c="yellow",
        char="!",
        n=60,
    )


# EOF
