#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 14:27:20 (ywatanabe)"
# File: ./mngs_repo/src/mngs/utils/_gen_timestamp.py

def gen_timestamp():
    from datetime import datetime

    now = datetime.now()
    now_str = now.strftime("%Y-%m%d-%H%M")
    return now_str


# EOF
