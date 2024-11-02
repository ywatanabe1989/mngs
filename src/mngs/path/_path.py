#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 20:46:35 (ywatanabe)"
# File: ./mngs_repo/src/mngs/path/_path.py

import inspect


def this_path(when_ipython="/tmp/fake.py"):
    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:
        __file__ = when_ipython
    return __file__

get_this_path = this_path



# EOF
