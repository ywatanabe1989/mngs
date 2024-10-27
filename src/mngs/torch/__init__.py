#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 19:05:01 (ywatanabe)"

try:
    from ._apply_to import apply_to
except ImportError as e:
    warnings.warn(f"Warning: Failed to import apply_to from ._apply_to.")

# from ._apply_to import apply_to
