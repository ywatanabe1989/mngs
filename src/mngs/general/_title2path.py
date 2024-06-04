#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: 2024-05-12 21:02:21 (7)
# /sshx:ywatanabe@444:/home/ywatanabe/proj/mngs/src/mngs/general/_title2spath.py


def title2path(title):
    path = title

    # Comma patterns
    patterns = [":", ";", "=", "[", "]"]
    for pp in patterns:
        path = path.replace(pp, "")

    # Exceptions
    path = path.replace("_-_", "-")
    path = path.replace(" ", "_")

    # Consective under scores
    for _ in range(10):
        path = path.replace("__", "_")

    return path.lower()
