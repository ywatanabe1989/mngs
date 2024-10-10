#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-11 08:35:47 (ywatanabe)"

try:
    from ._mv import mv
except ImportError as e:
    print(f"Warning: Failed to import mv from ._mv.")
