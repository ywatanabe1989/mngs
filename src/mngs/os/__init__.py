#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 19:04:55 (ywatanabe)"

try:
    from ._mv import mv
except ImportError as e:
    pass # print(f"Warning: Failed to import mv from ._mv.")
