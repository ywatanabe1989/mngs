#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 19:04:32 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/web/__init__.py

try:
    from ._summarize_url import summarize_url
except ImportError as e:
    pass
    # print(f"Warning: Failed to import summarize_url from ._summarize_url.")

# from ._summarize_url import summarize_url
