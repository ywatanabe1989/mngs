#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 20:48:53 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/gen/_mask_api_key.py


def mask_api(api_key, n=4):
    return f"{api_key[:n]}****{api_key[-n:]}"
