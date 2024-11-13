#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:27 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_json.py

import json


def _load_json(lpath, **kwargs):
    """Load JSON file."""
    if not lpath.endswith(".json"):
        raise ValueError("File must have .json extension")
    with open(lpath, "r") as f:
        return json.load(f)


# EOF
