#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:26 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_joblib.py

import joblib


def _load_joblib(lpath, **kwargs):
    """Load joblib file."""
    if not lpath.endswith(".joblib"):
        raise ValueError("File must have .joblib extension")
    with open(lpath, "rb") as f:
        return joblib.load(f, **kwargs)


# EOF
