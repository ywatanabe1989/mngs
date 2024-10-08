#!/usr/bin/env python3

import collections
import contextlib
import math
import os
import re
import shutil
import threading
import time
import warnings
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps
from glob import glob

import mngs
import numpy as np
import pandas as pd
import readchar
import torch
import xarray as xr
# from mngs.gen import deprecated
from natsort import natsorted

from .decorators._deprecated import deprecated
from collections import abc
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

################################################################################
## strings
################################################################################
def decapitalize(s):
    """Converts the first character of a string to lowercase.

    Parameters
    ----------
    s : str
        The input string to be decapitalized.

    Returns
    -------
    str
        The input string with its first character converted to lowercase.

    Example
    -------
    >>> decapitalize("Hello")
    'hello'
    >>> decapitalize("WORLD")
    'wORLD'
    """
    if not s:
        return s
    return s[0].lower() + s[1:]


def connect_strs(strs, filler="_"):
    """Connects a list of strings using a specified filler.

    Parameters
    ----------
    strs : list or tuple of str
        The list of strings to be connected.
    filler : str, optional
        The string used to connect the input strings (default is "_").

    Returns
    -------
    str
        A single string with all input strings connected by the filler.

    Example
    -------
    >>> connect_strs(['a', 'b', 'c'], filler='_')
    'a_b_c'
    >>> connect_strs(['hello', 'world'], filler='-')
    'hello-world'
    """
    if isinstance(strs, (list, tuple)):
        connected = ""
        for s in strs:
            connected += filler + s
        return connected[len(filler) :]
    else:
        return strs


def connect_nums(nums, filler="_"):
    """Connects a list of numbers using a specified filler.

    Parameters
    ----------
    nums : list or tuple of int or float
        The list of numbers to be connected.
    filler : str, optional
        The string used to connect the input numbers (default is "_").

    Returns
    -------
    str
        A single string with all input numbers connected by the filler.

    Example
    -------
    >>> connect_nums([1, 2, 3], filler='_')
    '1_2_3'
    >>> connect_nums([3.14, 2.718, 1.414], filler='-')
    '3.14-2.718-1.414'
    """
    if isinstance(nums, (list, tuple)):
        connected = ""
        for n in nums:
            connected += filler + str(n)
        return connected[len(filler) :]
    else:
        return nums


def squeeze_spaces(string, pattern=" +", repl=" "):
    """Replace multiple occurrences of a pattern in a string with a single replacement.

    Parameters
    ----------
    string : str
        The input string to be processed.
    pattern : str, optional
        The regular expression pattern to match (default is " +", which matches one or more spaces).
    repl : str or callable, optional
        The replacement string or function (default is " ", a single space).

    Returns
    -------
    str
        The processed string with pattern occurrences replaced.

    Example
    -------
    >>> squeeze_spaces("Hello   world")
    'Hello world'
    >>> squeeze_spaces("a---b--c-d", pattern="-+", repl="-")
    'a-b-c-d'
    """
    return re.sub(pattern, repl, string)


import re

import numpy as np


# def search(patterns, strings, only_perfect_match=False, as_bool=False, ensure_one=False):
#     """Search for patterns in strings using regular expressions.

#     Parameters
#     ----------
#     patterns : str or list of str
#         The pattern(s) to search for. Can be a single string or a list of strings.
#     strings : str or list of str
#         The string(s) to search in. Can be a single string or a list of strings.
#     only_perfect_match : bool, optional
#         If True, only exact matches are considered (default is False).
#     as_bool : bool, optional
#         If True, return a boolean array instead of indices (default is False).

#     Returns
#     -------
#     tuple
#         A tuple containing two elements:
#         - If as_bool is False: (list of int, list of str)
#           The first element is a list of indices where matches were found.
#           The second element is a list of matched strings.
#         - If as_bool is True: (numpy.ndarray of bool, list of str)
#           The first element is a boolean array indicating matches.
#           The second element is a list of matched strings.

#     Example
#     -------
#     >>> patterns = ['orange', 'banana']
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 4, 5], ['orange', 'banana', 'orange_juice'])

#     >>> patterns = 'orange'
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 5], ['orange', 'orange_juice'])
#     """

#     def to_list(s_or_p):
#         if isinstance(s_or_p, (np.ndarray, pd.Series, xr.DataArray)):
#             return s_or_p.tolist()
#         elif isinstance(s_or_p, collections.abc.KeysView):
#             return list(s_or_p)
#         elif not isinstance(s_or_p, (list, tuple, pd.Index)):
#             return [s_or_p]
#         return s_or_p

#     patterns = to_list(patterns)
#     strings = to_list(strings)

#     if not only_perfect_match:
#         indi_matched = []
#         for pattern in patterns:
#             for i_str, string in enumerate(strings):
#                 m = re.search(pattern, string)
#                 if m is not None:
#                     indi_matched.append(i_str)
#     else:
#         indi_matched = []
#         for pattern in patterns:
#             for i_str, string in enumerate(strings):
#                 if pattern == string:
#                     indi_matched.append(i_str)

#     indi_matched = natsorted(indi_matched)
#     keys_matched = list(np.array(strings)[indi_matched])

#     if ensure_one:
#         asssert len(indi_matched) == 1

#     if as_bool:
#         bool_matched = np.zeros(len(strings), dtype=bool)
#         if np.unique(indi_matched).size != 0:
#             bool_matched[np.unique(indi_matched)] = True
#         return bool_matched, keys_matched
#     else:
#         return indi_matched, keys_matched

def search(patterns, strings, only_perfect_match=False, as_bool=False, ensure_one=False):
    """Search for patterns in strings using regular expressions.

    Parameters
    ----------
    patterns : str or list of str
        The pattern(s) to search for. Can be a single string or a list of strings.
    strings : str or list of str
        The string(s) to search in. Can be a single string or a list of strings.
    only_perfect_match : bool, optional
        If True, only exact matches are considered (default is False).
    as_bool : bool, optional
        If True, return a boolean array instead of indices (default is False).
    ensure_one : bool, optional
        If True, ensures only one match is found (default is False).

    Returns
    -------
    tuple
        A tuple containing two elements:
        - If as_bool is False: (list of int, list of str)
          The first element is a list of indices where matches were found.
          The second element is a list of matched strings.
        - If as_bool is True: (numpy.ndarray of bool, list of str)
          The first element is a boolean array indicating matches.
          The second element is a list of matched strings.

    Example
    -------
    >>> patterns = ['orange', 'banana']
    >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
    >>> search(patterns, strings)
    ([1, 4, 5], ['orange', 'banana', 'orange_juice'])

    >>> patterns = 'orange'
    >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
    >>> search(patterns, strings)
    ([1, 5], ['orange', 'orange_juice'])
    """

    def to_list(string_or_pattern):
        if isinstance(string_or_pattern, (np.ndarray, pd.Series, xr.DataArray)):
            return string_or_pattern.tolist()
        elif isinstance(string_or_pattern, abc.KeysView):
            return list(string_or_pattern)
        elif not isinstance(string_or_pattern, (list, tuple, pd.Index)):
            return [string_or_pattern]
        return string_or_pattern

    patterns = to_list(patterns)
    strings = to_list(strings)

    indices_matched = []
    for pattern in patterns:
        for index_str, string in enumerate(strings):
            if only_perfect_match:
                if pattern == string:
                    indices_matched.append(index_str)
            else:
                if re.search(pattern, string):
                    indices_matched.append(index_str)

    indices_matched = natsorted(indices_matched)
    keys_matched = list(np.array(strings)[indices_matched])

    if ensure_one:
        assert len(indices_matched) == 1, "Expected exactly one match, but found {}".format(len(indices_matched))

    if as_bool:
        bool_matched = np.zeros(len(strings), dtype=bool)
        bool_matched[np.unique(indices_matched)] = True
        return bool_matched, keys_matched
    else:
        return indices_matched, keys_matched


def grep(str_list, search_key):
    """Search for a key in a list of strings and return matching items.

    Parameters
    ----------
    str_list : list of str
        The list of strings to search through.
    search_key : str
        The key to search for in the strings.

    Returns
    -------
    list
        A list of strings from str_list that contain the search_key.

    Example
    -------
    >>> grep(['apple', 'banana', 'cherry'], 'a')
    ['apple', 'banana']
    >>> grep(['cat', 'dog', 'elephant'], 'e')
    ['elephant']
    """
    """
    Example:
        str_list = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        search_key = 'orange'
        print(grep(str_list, search_key))
        # ([1, 5], ['orange', 'orange_juice'])
    """
    matched_keys = []
    indi = []
    for ii, string in enumerate(str_list):
        m = re.search(search_key, string)
        if m is not None:
            matched_keys.append(string)
            indi.append(ii)
    return indi, matched_keys


def pop_keys(keys_list, keys_to_pop):
    """Remove specified keys from a list of keys.

    Parameters
    ----------
    keys_list : list
        The original list of keys.
    keys_to_pop : list
        The list of keys to remove from keys_list.

    Returns
    -------
    list
        A new list with the specified keys removed.

    Example
    -------
    >>> keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
    >>> keys_to_pop = ['b', 'd']
    >>> pop_keys(keys_list, keys_to_pop)
    ['a', 'c', 'e', 'bde']
    """
    indi_to_remain = [k not in keys_to_pop for k in keys_list]
    keys_remainded_list = list(np.array(keys_list)[list(indi_to_remain)])
    return keys_remainded_list


def readable_bytes(num, suffix="B"):
    """Convert a number of bytes to a human-readable format.

    Parameters
    ----------
    num : int
        The number of bytes to convert.
    suffix : str, optional
        The suffix to append to the unit (default is "B" for bytes).

    Returns
    -------
    str
        A human-readable string representation of the byte size.

    Example
    -------
    >>> readable_bytes(1024)
    '1.0 KiB'
    >>> readable_bytes(1048576)
    '1.0 MiB'
    >>> readable_bytes(1073741824)
    '1.0 GiB'
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


################################################################################
## list
################################################################################
def is_listed_X(obj, types):
    """
    Example:
        obj = [3, 2, 1, 5]
        _is_listed_X(obj,
    """
    import numpy as np

    try:
        conditions = []
        condition_list = isinstance(obj, list)

        if not (isinstance(types, list) or isinstance(types, tuple)):
            types = [types]

        _conditions_susp = []
        for typ in types:
            _conditions_susp.append(
                (np.array([isinstance(o, typ) for o in obj]) == True).all()
            )

        condition_susp = np.any(_conditions_susp)

        _is_listed_X = np.all([condition_list, condition_susp])
        return _is_listed_X

    except:
        return False


def find_closest(list_obj, num_insert):
    """Find the closest value in a sorted list to a given number.

    Parameters
    ----------
    list_obj : list
        A sorted list of numbers.
    num_insert : float or int
        The number to find the closest value to.

    Returns
    -------
    tuple
        A tuple containing (closest_value, index_of_closest_value).

    Example
    -------
    >>> find_closest([1, 3, 5, 7, 9], 6)
    (5, 2)
    >>> find_closest([1, 3, 5, 7, 9], 8)
    (7, 3)
    """
    """
    Assumes list_obj is sorted. Returns the closest value to num.
    If the same number is included in list_obj, the smaller number is returned.

    Example:
        list_obj = np.array([0, 1, 1, 2, 3, 3])
        num = 1.2
        closest_num, closest_pos = take_the_closest(list_obj, num)
        print(closest_num, closest_pos)
        # 1 2

        list_obj = np.array([0, 1, 1, 2, 3, 3])
        num = 1
        closest_num, closest_pos = take_the_closest(list_obj, num)
        print(closest_num, closest_pos)
        # 1 1
    """
    if math.isnan(num_insert):
        closest_num = np.nan
        closest_pos = np.nan

    pos_num_insert = bisect_left(list_obj, num_insert)

    if pos_num_insert == 0:
        closest_num = list_obj[0]
        closest_pos = pos_num_insert

    if pos_num_insert == len(list_obj):
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert

    else:
        pos_before = pos_num_insert - 1

        before_num = list_obj[pos_before]
        after_num = list_obj[pos_num_insert]

        delta_after = abs(after_num - num_insert)
        delta_before = abs(before_num - num_insert)

        if np.abs(delta_after) < np.abs(delta_before):
            closest_num = after_num
            closest_pos = pos_num_insert

        else:
            closest_num = before_num
            closest_pos = pos_before

    return closest_num, closest_pos


################################################################################
## mutable
################################################################################
def isclose(mutable_a, mutable_b):
    """Check if two mutable objects are close to each other.

    This function compares two mutable objects (e.g., lists, numpy arrays) element-wise
    to determine if they are close to each other.

    Parameters
    ----------
    mutable_a : list or numpy.ndarray
        The first mutable object to compare.
    mutable_b : list or numpy.ndarray
        The second mutable object to compare.

    Returns
    -------
    bool
        True if the objects are close to each other, False otherwise.

    Example
    -------
    >>> isclose([1.0, 2.0, 3.0], [1.0, 2.0001, 3.0])
    True
    >>> isclose([1.0, 2.0, 3.0], [1.0, 2.1, 3.0])
    False
    """
    return [math.isclose(a, b) for a, b in zip(mutable_a, mutable_b)]


################################################################################
## dictionary
################################################################################
def merge_dicts_wo_overlaps(*dicts):
    """Merge multiple dictionaries without overlapping keys.

    This function merges multiple dictionaries into a single dictionary,
    ensuring that there are no overlapping keys between the input dictionaries.

    Parameters
    ----------
    *dicts : dict
        Variable number of dictionaries to merge.

    Returns
    -------
    dict
        A new dictionary containing all key-value pairs from the input dictionaries.

    Raises
    ------
    AssertionError
        If there are overlapping keys between the input dictionaries.

    Example
    -------
    >>> d1 = {'a': 1, 'b': 2}
    >>> d2 = {'c': 3, 'd': 4}
    >>> merge_dicts_wo_overlaps(d1, d2)
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    >>> d3 = {'b': 5}  # This would raise an AssertionError due to overlapping key 'b'
    >>> merge_dicts_wo_overlaps(d1, d3)
    AssertionError
    """
    merged_dict = {}
    for dict in dicts:
        assert mngs.general.search(
            merged_dict.keys(), dict.keys(), only_perfect_match=True
        ) == ([], [])
        merged_dict.update(dict)
    return merged_dict


def listed_dict(keys=None):
    """
    Example 1:
        import random
        random.seed(42)
        d = listed_dict()
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 1, 0, 4, 3, 3, 2, 1, 10, 8]})

    Example 2:
        import random
        random.seed(42)
        keys = ['a', 'b', 'c']
        d = listed_dict(keys)
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
            d['b'].append(random.randint(0, 10))
            d['c'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 4, 2, 8, 6, 1, 8, 8, 8, 7],
        #                              'b': [1, 3, 1, 1, 0, 3, 9, 3, 6, 9],
        #                              'c': [0, 3, 10, 9, 0, 3, 0, 10, 3, 4]})
    """
    dict_list = defaultdict(list)
    # initialize with keys if possible
    if keys is not None:
        for k in keys:
            dict_list[k] = []
    return dict_list


################################################################################
## variables
################################################################################
def is_defined_global(x_str):
    """
    Example:
        print(is_defined('a'))
        # False

        a = 5
        print(is_defined('a'))
        # True
    """
    return x_str in globals()


def is_defined_local(x_str):
    """
    Example:
        print(is_defined('a'))
        # False

        a = 5
        print(is_defined('a'))
        # True
    """
    return x_str in locals()


# def does_exist(suspicious_var_str):
#     return suspicious_var_str in globals()


################################################################################
## versioning
################################################################################
def is_later_or_equal(package, tgt_version, format="MAJOR.MINOR.PATCH"):
    """Check if the installed version of a package is later than or equal to a target version.

    Parameters
    ----------
    package : str
        The name of the package to check.
    tgt_version : str
        The target version to compare against.
    format : str, optional
        The version format (default is "MAJOR.MINOR.PATCH").

    Returns
    -------
    bool
        True if the installed version is later than or equal to the target version, False otherwise.

    Example
    -------
    >>> is_later_or_equal('numpy', '1.18.0')
    True
    >>> is_later_or_equal('pandas', '2.0.0')
    False
    """
    import mngs
    import numpy as np

    indi, matched = mngs.general.search(
        ["MAJOR", "MINOR", "PATCH"], format.split(".")
    )
    imp_major, imp_minor, imp_patch = [
        int(v) for v in np.array(package.__version__.split("."))[indi]
    ]
    tgt_major, tgt_minor, tgt_patch = [
        int(v) for v in np.array(tgt_version.split("."))[indi]
    ]

    print(
        f"\npackage: {package.__name__}\n"
        f"target_version: {tgt_version}\n"
        f"imported_version: {imp_major}.{imp_minor}.{imp_patch}\n"
    )

    ## Mjorr
    if imp_major > tgt_major:
        return True

    if imp_major < tgt_major:
        return False

    if imp_major == tgt_major:

        ## Minor
        if imp_minor > tgt_minor:
            return True

        if imp_minor < tgt_minor:
            return False

        if imp_minor == tgt_minor:

            ## Patch
            if imp_patch > tgt_patch:
                return True
            if imp_patch < tgt_patch:
                return False
            if imp_patch == tgt_patch:
                return True


################################################################################
## File
################################################################################
def _copy_a_file(src, dst, allow_overwrite=False):
    """Copy a single file from source to destination.

    Parameters
    ----------
    src : str
        The path to the source file.
    dst : str
        The path to the destination file.
    allow_overwrite : bool, optional
        If True, allows overwriting existing files (default is False).

    Raises
    ------
    FileExistsError
        If the destination file already exists and allow_overwrite is False.

    Example
    -------
    >>> _copy_a_file('/path/to/source.txt', '/path/to/destination.txt')
    >>> _copy_a_file('/path/to/source.txt', '/path/to/existing.txt', allow_overwrite=True)
    """
    if src == "/dev/null":
        print(f"\n/dev/null was not copied.\n")

    else:

        if dst.endswith("/"):
            _, src_fname, src_ext = mngs.path.split(src)
            # src_fname = src + src_ext
            dst = dst + src_fname + src_ext

        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(f'\nCopied "{src}" to "{dst}".\n')

        else:
            if allow_overwrite:
                shutil.copyfile(src, dst)
                print(f'\nCopied "{src}" to "{dst}" (overwritten).\n')

            if not allow_overwrite:
                print(
                    f'\n"{dst}" exists and copying from "{src}" was aborted.\n'
                )


def copy_files(src_files, dists, allow_overwrite=False):
    """Copy multiple files from source(s) to destination(s).

    Parameters
    ----------
    src_files : str or list of str
        The path(s) to the source file(s).
    dists : str or list of str
        The path(s) to the destination file(s) or directory(ies).
    allow_overwrite : bool, optional
        If True, allows overwriting existing files (default is False).

    Example
    -------
    >>> copy_files('/path/to/source.txt', '/path/to/destination/')
    >>> copy_files(['/path/to/file1.txt', '/path/to/file2.txt'], ['/path/to/dest1/', '/path/to/dest2/'])
    >>> copy_files('/path/to/source.txt', '/path/to/existing.txt', allow_overwrite=True)
    """
    if isinstance(src_files, str):
        src_files = [src_files]

    if isinstance(dists, str):
        dists = [dists]

    for sf in src_files:
        for dst in dists:
            _copy_a_file(sf, dst, allow_overwrite=allow_overwrite)


def copy_the_file(sdir):
    """Copy the current script file to a specified directory.

    This function copies the script file that called it to a specified directory.
    It uses the calling script's filename and copies it to the given directory.

    Parameters
    ----------
    sdir : str
        The destination directory where the file should be copied.

    Note
    ----
    This function will not copy the file if it's run in an IPython environment.

    Example
    -------
    >>> copy_the_file('/path/to/destination/')
    """
    __file__ = inspect.stack()[1].filename
    _, fname, ext = mngs.path.split(__file__)

    #     dst = sdir + fname + ext

    if "ipython" not in __file__:
        _copy_a_file(__file__, dst)


def is_nan(X):
    """Check if the input contains any NaN values and raise an error if found.

    This function checks for NaN values in various data types including pandas DataFrames,
    numpy arrays, PyTorch tensors, and scalar values.

    Parameters
    ----------
    X : pandas.DataFrame, numpy.ndarray, torch.Tensor, float, or int
        The input data to check for NaN values.

    Raises
    ------
    ValueError
        If any NaN value is found in the input.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> is_nan(pd.DataFrame({'a': [1, 2, np.nan]}))
    ValueError: NaN was found in X
    >>> is_nan(np.array([1, 2, 3]))
    # No error raised
    >>> is_nan(torch.tensor([1.0, float('nan'), 3.0]))
    ValueError: NaN was found in X
    >>> is_nan(float('nan'))
    ValueError: X was NaN
    """
    if isinstance(X, pd.DataFrame):
        if X.isna().any().any():
            raise ValueError("NaN was found in X")
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():
            raise ValueError("NaN was found in X")
    elif torch.is_tensor(X):
        if X.isnan().any():
            raise ValueError("NaN was found in X")
    elif isinstance(X, (float, int)):
        if math.isnan(X):
            raise ValueError("X was NaN")


def partial_at(func, index, value):
    """Create a partial function with a fixed argument at a specific position.

    This function creates a new function that calls the original function with a
    fixed argument inserted at the specified index position.

    Parameters
    ----------
    func : callable
        The original function to be partially applied.
    index : int
        The position at which to insert the fixed argument.
    value : any
        The fixed argument value to be inserted.

    Returns
    -------
    callable
        A new function that calls the original function with the fixed argument.

    Example
    -------
    >>> def greet(greeting, name):
    ...     return f"{greeting}, {name}!"
    >>> hello = partial_at(greet, 0, "Hello")
    >>> hello("Alice")
    'Hello, Alice!'
    >>> hello("Bob")
    'Hello, Bob!'
    """

    @wraps(func)
    def result(*rest, **kwargs):
        args = []
        args.extend(rest[:index])
        args.append(value)
        args.extend(rest[index:])
        return func(*args, **kwargs)

    return result


# def describe(df, method="mean", round_factor=1, axis=0):
# assert method in ["mean_std", "mean_ci", "median_iqr"]
#     df = pd.DataFrame(df)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         if method == "mean":
#             return round(np.nanmean(df, axis=axis), 3), round(
#                 np.nanstd(df, axis=axis) / round_factor, 3
#             )
#         if method == "median":
#             med = df.median(axis=axis)
#             IQR = df.quantile(0.75, axis=axis) - df.quantile(0.25, axis=axis)
#             return round(med, 3), round(IQR / round_factor, 3)


def describe(df, method="mean_std", round_factor=3, axis=0):
    """
    Compute descriptive statistics for a DataFrame.

    Example
    -------
    import pandas as pd
    import numpy as np
    data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    result = describe(data, method='mean_std')
    print(f"n={result['n']}, mean={result['mean']}, std={result['std']}")

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        Input data.
    method : str, optional
        Statistical method to use. Options are 'mean_std', 'mean_ci', 'median_iqr'.
        Default is 'mean_std'.
    round_factor : int, optional
        Factor to divide the spread statistic by. Default is 3.
    axis : int, optional
        Axis along which to compute statistics. Default is 0.

    Returns
    -------
    dict
        Dictionary containing statistics based on the method chosen.
    """
    assert method in ["mean_std", "mean_ci", "median_iqr"]
    df = pd.DataFrame(df)
    nn = df.notna().sum(axis=axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method in ["mean_std", "mean_ci"]:
            mm = np.nanmean(df, axis=axis)
            if method == "mean_std":
                ss = np.nanstd(df, axis=axis)
                key = "std"
            else:  # mean_ci
                ss = 1.96 * np.nanstd(df, axis=axis) / np.sqrt(nn)
                key = "ci"
            return {
                "n": np.round(nn, 3),
                "mean": np.round(mm, 3),
                key: np.round(ss, 3),
            }
        else:  # median_iqr
            med = df.median(axis=axis)
            iqr = df.quantile(0.75, axis=axis) - df.quantile(0.25, axis=axis)
            return {
                "n": np.round(nn, round_factor),
                "median": np.round(med, round_factor),
                "iqr": np.round(iqr, round_factor),
            }

def _return_counting_process():
    import multiprocessing

    def _count():
        counter = 0
        while True:
            print(counter)
            time.sleep(1)
            counter += 1

    p1 = multiprocessing.Process(target=_count)
    p1.start()
    return p1


def wait_key(process, tgt_key="q"):
    """Wait for a specific key press while a process is running.

    This function waits for a specific key to be pressed while a given process
    is running. It's typically used to provide a way to interrupt or terminate
    a long-running process.

    Parameters
    ----------
    process : multiprocessing.Process
        The process to monitor while waiting for the key press.
    tgt_key : str, optional
        The target key to wait for (default is "q" for quit).

    Returns
    -------
    None

    Note
    ----
    This function will block until either the target key is pressed or the
    monitored process terminates.

    Example
    -------
    >>> import multiprocessing
    >>> def long_running_task():
    ...     while True:
    ...         pass
    >>> p = multiprocessing.Process(target=long_running_task)
    >>> p.start()
    >>> wait_key(p)  # This will wait until 'q' is pressed or the process ends
    """
    """
    Example:
        import mngs
        p1 = mngs.general._return_counting_process()
        mngs.gen.wait_key(p1)
        # press q
    """
    pressed_key = None
    while pressed_key != tgt_key:
        pressed_key = readchar.readchar()
        print(pressed_key)
    process.terminate()


class ThreadWithReturnValue(threading.Thread):
    """
    Example:
        t = ThreadWithReturnValue(
            target=func, args=(,), kwargs={key: val}
        )
        t.start()
        out = t.join()

    """

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs={},
        Verbose=None,
    ):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        ### fixme
        Thread.join(self, *args)
        return self._return


@contextmanager
def suppress_output():
    """
    A context manager that suppresses stdout and stderr.

    Example:
        with suppress_output():
            print("This will not be printed to the console.")
    """
    # Open a file descriptor that points to os.devnull (a black hole for data)
    with open(os.devnull, "w") as fnull:
        # Temporarily redirect stdout and stderr to the file descriptor fnull
        with redirect_stdout(fnull), redirect_stderr(fnull):
            # Yield control back to the context block
            yield


quiet = suppress_output


def unique(data, axis=None):
    """
    Identifies unique elements in the data along the specified axis and their counts, returning a DataFrame.

    Parameters:
    - data (array-like): The input data to analyze for unique elements.
    - axis (int, optional): The axis along which to find the unique elements. Defaults to None.

    Returns:
    - df (pandas.DataFrame): DataFrame with unique elements and their counts.
    """
    if axis is None:
        uqs, counts = np.unique(data, return_counts=True)
    else:
        uqs, counts = np.unique(data, axis=axis, return_counts=True)

    if axis is None:
        df = pd.DataFrame({"uq": uqs, "n": counts})
    else:
        df = pd.DataFrame(
            uqs, columns=[f"axis_{i}" for i in range(uqs.shape[1])]
        )
        df["n"] = counts

    df["n"] = df["n"].apply(lambda x: f"{int(x):,}")

    return df


def unique(data, axis=None):
    """
    Identifies unique elements in the data along the specified axis and their counts, returning a DataFrame.

    Parameters:
    - data (array-like): The input data to analyze for unique elements.
    - axis (int, optional): The axis along which to find the unique elements. Defaults to None.

    Returns:
    - df (pandas.DataFrame): DataFrame with unique elements and their counts.
    """
    # Find unique elements and their counts
    if axis is None:
        uqs, counts = np.unique(data, return_counts=True)
        df = pd.DataFrame({"Unique Elements": uqs, "Counts": counts})
    else:
        uqs, counts = np.unique(data, axis=axis, return_counts=True)
        # Create a DataFrame with unique elements
        df = pd.DataFrame(
            uqs,
            columns=[f"Unique Elements Axis {i}" for i in range(uqs.shape[1])],
        )
        # Add a column for counts
        df["Counts"] = counts

    # Format the 'Counts' column with commas for thousands
    df["Counts"] = df["Counts"].apply(lambda x: f"{x:,}")

    return df


def uq(*args, **kwargs):
    """Alias for the unique function.

    This function is a wrapper around the unique function, providing the same
    functionality with a shorter name.

    Parameters
    ----------
    *args : positional arguments
        Positional arguments to be passed to the unique function.
    **kwargs : keyword arguments
        Keyword arguments to be passed to the unique function.

    Returns
    -------
    array_like
        The result of calling the unique function with the given arguments.

    See Also
    --------
    unique : The main function for finding unique elements.

    Example
    -------
    >>> uq([1, 2, 2, 3, 3, 3])
    array([1, 2, 3])
    """
    return unique(*args, **kwargs)


def print_block(message, char="-", n=40, c=None):
    """Print a message surrounded by a character border.

    This function prints a given message surrounded by a border made of
    a specified character. The border can be colored if desired.

    Parameters
    ----------
    message : str
        The message to be printed inside the border.
    char : str, optional
        The character used to create the border (default is "-").
    n : int, optional
        The width of the border (default is 40).
    c : str, optional
        The color of the border. Can be 'red', 'green', 'yellow', 'blue',
        'magenta', 'cyan', 'white', or 'grey' (default is None, which means no color).

    Returns
    -------
    None

    Example
    -------
    >>> print_block("Hello, World!", char="*", n=20, c="blue")
    ********************
    * Hello, World!    *
    ********************

    Note: The actual output will be in green color.
    """
    border = char * n
    text = f"\n{border}\n{message}\n{border}\n"
    if c is not None:
        text = color_text(text, c)
    print(text)

print_ = print_block

def color_text(text, c="green"):
    """Apply ANSI color codes to text.

    Parameters
    ----------
    text : str
        The text to be colored.
    c : str, optional
        The color to apply. Available colors are 'red', 'green', 'yellow',
        'blue', 'magenta', 'cyan', 'white', and 'grey' (default is "green").

    Returns
    -------
    str
        The input text with ANSI color codes applied.

    Example
    -------
    >>> print(color_text("Hello, World!", "blue"))
    # This will print "Hello, World!" in blue text
    """
    ANSI_COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "grey": "\033[90m",
        "gray": "\033[90m",
        "reset": "\033[0m",
    }
    ANSI_COLORS["tra"] = ANSI_COLORS["white"]
    ANSI_COLORS["val"] = ANSI_COLORS["green"]
    ANSI_COLORS["tes"] = ANSI_COLORS["red"]

    start_code = ANSI_COLORS.get(c, ANSI_COLORS["reset"])
    end_code = ANSI_COLORS["reset"]
    return f"{start_code}{text}{end_code}"


ct = color_text


# def mv_col(dataframe, column_name, position):
#     temp_col = dataframe[column_name]
#     dataframe.drop(labels=[column_name], axis=1, inplace=True)
#     dataframe.insert(loc=position, column=column_name, value=temp_col)
#     return dataframe


def symlink(tgt, src, force=False):
    """Create a symbolic link.

    This function creates a symbolic link from the target to the source.
    If the force parameter is True, it will remove any existing file at
    the source path before creating the symlink.

    Parameters
    ----------
    tgt : str
        The target path (the file or directory to be linked to).
    src : str
        The source path (where the symbolic link will be created).
    force : bool, optional
        If True, remove the existing file at the src path before creating
        the symlink (default is False).

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the symlink creation fails.

    Example
    -------
    >>> symlink('/path/to/target', '/path/to/link')
    >>> symlink('/path/to/target', '/path/to/existing_file', force=True)
    """
    if force:
        try:
            os.remove(src)
        except FileNotFoundError:
            pass

    # Calculate the relative path from src to tgt
    src_dir = os.path.dirname(src)
    relative_tgt = os.path.relpath(tgt, src_dir)

    os.symlink(relative_tgt, src)
    print(
        mngs.gen.ct(
            f"\nSymlink was created: {src} -> {relative_tgt}\n", c="yellow"
        )
    )


#     os.symlink(tgt, src)
#     print(mngs.gen.ct(f"\nSymlink was created: {src} -> {tgt}\n", c="yellow"))
# Symlink was created: ./scripts/ml/clf/sct_optuna/optuna_studies/optuna_study_stent_3_classes/best_trial -> /home/ywatanabe/proj/ecog_stent_sheep_visual/scripts/ml/clf/sct_optuna/RUNNING/2024Y-03M-29D-21h55m09s_IBSy/objective/Trial#00068/


def to_even(n):
    """Convert a number to the nearest even number less than or equal to itself.

    Parameters
    ----------
    n : int or float
        The input number to be converted.

    Returns
    -------
    int
        The nearest even number less than or equal to the input.

    Example
    -------
    >>> to_even(5)
    4
    >>> to_even(6)
    6
    >>> to_even(3.7)
    2
    """
    return int(n) - (int(n) % 2)


def to_odd(n):
    """Convert a number to the nearest odd number less than or equal to itself.

    Parameters
    ----------
    n : int or float
        The input number to be converted.

    Returns
    -------
    int
        The nearest odd number less than or equal to the input.

    Example
    -------
    >>> to_odd(6)
    5
    >>> to_odd(7)
    7
    >>> to_odd(5.8)
    5
    """
    return int(n) - ((int(n) + 1) % 2)


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


# def natglob(expression):
#     glob_pattern = re.sub(r"{[^}]*}", "*", expression)
#     if isinsntace(eval(glob_pattern), str):
#         abort
#     else:
#         glob_pattern = eval(glob_pattern)
#         return natsorted(glob(glob_pattern))
#     # return natsorted(glob(expression))


def float_linspace(start, stop, num_points):
    """Generate evenly spaced floating-point numbers over a specified interval.

    This function is similar to numpy's linspace, but ensures that the output
    consists of floating-point numbers with a specified number of decimal places.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The end value of the sequence.
    num_points : int
        Number of points to generate.

    Returns
    -------
    numpy.ndarray
        Array of evenly spaced floating-point values.

    Example
    -------
    >>> float_linspace(0, 1, 5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> float_linspace(1, 2, 3)
    array([1. , 1.5, 2. ])
    """
    num_points = int(num_points)  # Ensure num_points is an integer

    if num_points < 2:
        return (
            np.array([start, stop]) if num_points == 2 else np.array([start])
        )

    step = (stop - start) / (num_points - 1)
    values = [start + i * step for i in range(num_points)]

    return np.array(values)


def replace(string, replacements=None):
    """Replace placeholders in the string with corresponding values from replacements.

    This function replaces placeholders in the format {key} within the input string
    with corresponding values from the replacements dictionary. If replacements is
    a string, it replaces the entire input string.

    Parameters
    ----------
    string : str
        The string containing placeholders in the format {key}.
    replacements : dict or str, optional
        A dictionary containing key-value pairs for replacing placeholders in the string,
        or a single string to replace the entire string.

    Returns
    -------
    str
        The input string with placeholders replaced by their corresponding values.

    Examples
    --------
    >>> replace("Hello, {name}!", {"name": "World"})
    'Hello, World!'
    >>> replace("Original string", "New string")
    'New string'
    >>> replace("Value: {x}", {"x": 42})
    'Value: 42'
    >>> template = "Hello, {name}! You are {age} years old."
    >>> replacements = {"name": "Alice", "age": "30"}
    >>> replace(template, replacements)
    'Hello, Alice! You are 30 years old.'
    """
    if isinstance(replacements, str):
        return replacements

    if replacements is None:
        replacements = {}

    for k, v in replacements.items():
        if v is not None:
            string = string.replace("{" + k + "}", v)
    return string
