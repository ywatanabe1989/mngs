#!/usr/bin/env python3
"""Sklearn wrappers and utilities."""

import warnings

try:
    from .clf import *
except ImportError as e:
    warnings.warn(
        f"Could not import clf from mngs.ai.sklearn: {str(e)}. "
        f"Some functionality may be unavailable. "
        f"Consider installing missing dependencies if you need this module.",
        ImportWarning,
        stacklevel=2
    )

try:
    from .to_sktime import *
except ImportError as e:
    warnings.warn(
        f"Could not import to_sktime from mngs.ai.sklearn: {str(e)}. "
        f"Some functionality may be unavailable. "
        f"Consider installing missing dependencies if you need this module.",
        ImportWarning,
        stacklevel=2
    )
