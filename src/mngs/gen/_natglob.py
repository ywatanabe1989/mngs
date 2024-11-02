#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 04:03:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/gen/_natglob.py

import re
from glob import glob
from ._deprecated import deprecated

@deprecated("Use mngs.io.glob instead.")
def natglob(expression):
    """
    Perform a natural-sorted glob operation on the given expression.

    This function is deprecated. Use mngs.io.glob instead.

    Parameters
    ----------
    expression : str
        The glob expression to evaluate. Can include wildcards and curly brace expansions.

    Returns
    -------
    list
        A naturally sorted list of file paths matching the glob expression.

    Example
    -------
    >>> natglob("*.txt")
    ['1.txt', '2.txt', '10.txt']
    >>> natglob("file_{1..3}.txt")
    ['file_1.txt', 'file_2.txt', 'file_3.txt']

    Notes
    -----
    This function first attempts to evaluate the expression as a Python expression.
    If that fails, it treats the expression as a literal glob pattern.
    """
    glob_pattern = re.sub(r"{[^}]*}", "*", expression)
    try:
        return natsorted(glob(eval(glob_pattern)))
    except:
        return natsorted(glob(glob_pattern))


# EOF
