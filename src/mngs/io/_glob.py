#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-22 23:19:07 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/io/_glob.py

import re
from natsort import natsorted
from glob import glob as _glob


def glob(expression, ensure_one=False):
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
    glob_pattern = re.sub(r"{[^}]*}", "*", expression)
    try:
        found_paths = natsorted(_glob(eval(glob_pattern)))
    except:
        found_paths = natsorted(_glob(glob_pattern))

    if ensure_one:
        assert len(found_paths) == 1

    return found_paths

# import subprocess
# import re
# from natsort import natsorted

# def glob(expression, ensure_one=False):
#     """
#     Perform a file search operation using 'find' command with natural sorting and extended pattern support.

#     Parameters:
#     -----------
#     expression : str
#         The search pattern to match against file paths. Supports 'find' command syntax.

#     Returns:
#     --------
#     list
#         A naturally sorted list of file paths that match the given expression.

#     Examples:
#     ---------
#     >>> glob('data/*.txt')
#     ['data/file1.txt', 'data/file2.txt', 'data/file10.txt']

#     >>> glob('data/{a,b}/*.txt')
#     ['data/a/file1.txt', 'data/a/file2.txt', 'data/b/file1.txt']
#     """
#     glob_pattern = re.sub(r"{[^}]*}", "*", expression)
#     glob_pattern = eval(glob_pattern)
#     find_pattern = glob_pattern

#     cmd = f"find -wholename '*{find_pattern}' -type f -print0 | sort -z | xargs -0 ls -1d 2>/dev/null"
#     try:
#         result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
#         found_paths = result.stdout.strip().split('\n')
#         found_paths = [path[2:] for path in found_paths if path]  # Remove './' prefix
#     except subprocess.CalledProcessError:
#         found_paths = []

#     found_paths = natsorted(found_paths)

#     if ensure_one:
#         assert len(found_paths) == 1

#     return found_paths
