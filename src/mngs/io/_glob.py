#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-16 12:49:08 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_glob.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_glob.py"

import re as _re
from glob import glob as _glob
from ..str._parse import parse as _parse
from natsort import natsorted as _natsorted


def glob(expression, parse=False, ensure_one=False):
    """
    Perform a glob operation with natural sorting and extended pattern support.

    This function extends the standard glob functionality by adding natural sorting
    and support for curly brace expansion in the glob pattern.

    Parameters:
    -----------
    expression : str
        The glob pattern to match against file paths. Supports standard glob syntax
        and curly brace expansion (e.g., 'dir/{a,b}/*.txt').

    Returns:
    --------
    list
        A naturally sorted list of file paths that match the given expression.

    Examples:
    ---------
    >>> glob('data/*.txt')
    ['data/file1.txt', 'data/file2.txt', 'data/file10.txt']

    >>> glob('data/{a,b}/*.txt')
    ['data/a/file1.txt', 'data/a/file2.txt', 'data/b/file1.txt']
    """
    glob_pattern = _re.sub(r"{[^}]*}", "*", expression)
    try:
        found_paths = _natsorted(_glob(eval(glob_pattern)))
    except:
        found_paths = _natsorted(_glob(glob_pattern))

    if ensure_one:
        assert len(found_paths) == 1

    if parse:
        parsed = [_parse(found_path, expression) for found_path in found_paths]
        return found_paths, parsed

    else:
        return found_paths


# EOF
