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

import mngs
import numpy as np
import pandas as pd
import readchar
import torch
from natsort import natsorted


################################################################################
## strings
################################################################################
def decapitalize(s):
    if not s:  # check that s is not empty string
        return s
    return s[0].lower() + s[1:]


def connect_strs(strs, filler="_"):  # connect_nums also works as connect_strs
    """
    Example:
        print(connect_strs(['a', 'b', 'c'], filler='_'))
        # 'a_b_c'
    """
    if isinstance(strs, list) or isinstance(strs, tuple):
        connected = ""
        for s in strs:
            connected += filler + s
        return connected[len(filler) :]
    else:
        return strs


def connect_nums(nums, filler="_"):
    """
    Example:
        print(connect_nums([1, 2, 3], filler='_'))
        # '1_2_3'
    """
    if isinstance(nums, list) or isinstance(nums, tuple):
        connected = ""
        for n in nums:
            connected += filler + str(n)
        return connected[len(filler) :]
    else:
        return nums


def squeeze_spaces(string, pattern=" +", repl=" "):
    """Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the Match object and must return
    a replacement string to be used.
    """
    # return re.sub(" +", " ", string)
    return re.sub(pattern, repl, string)


def search(patterns, strings, only_perfect_match=False, as_bool=False):
    """
    regular expression is acceptable for patterns.

    Example:
        patterns = ['orange', 'banana']
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        print(search(patterns, strings))
        # ([1, 4, 5], ['orange', 'banana', 'orange_juice'])

        patterns = 'orange'
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        print(search(patterns, strings))
        # ([1, 5], ['orange', 'orange_juice'])
    """

    ## For single string objects
    def to_str_list(data):
        data_arr = np.array(data).astype(str)
        if data_arr.ndim == 0:
            data_arr = data_arr[np.newaxis]
        return list(data_arr)

    # def to_list(s_or_p):
    #     if isinstance(s_or_p, collections.abc.KeysView):
    #         s_or_p = list(s_or_p)

    #     elif not isinstance(
    #         s_or_p,
    #         (list, tuple, pd.core.indexes.base.Index, pd.core.series.Series),
    #     ):
    #         s_or_p = [s_or_p]

    #     return s_or_p

    # patterns = to_list(patterns)
    # strings = to_list(strings)
    patterns = to_str_list(patterns)
    strings = to_str_list(strings)

    ## Main
    if not only_perfect_match:
        indi_matched = []
        for pattern in patterns:
            for i_str, string in enumerate(strings):
                m = re.search(pattern, string)
                if m is not None:
                    indi_matched.append(i_str)
    else:
        indi_matched = []
        for pattern in patterns:
            for i_str, string in enumerate(strings):
                if pattern == string:
                    indi_matched.append(i_str)

    ## Sorts the indices according to the original strings
    indi_matched = natsorted(indi_matched)
    keys_matched = list(np.array(strings)[indi_matched])

    if as_bool:
        bool_matched = np.zeros(len(strings), dtype=bool)
        if np.unique(indi_matched).size != 0:
            bool_matched[np.unique(indi_matched)] = True
        return bool_matched, keys_matched

    else:
        return indi_matched, keys_matched


def grep(str_list, search_key):
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
    """
    keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
    keys_to_pop = ['b', 'd']
    pop_keys(keys_list, keys_to_pop)
    """
    indi_to_remain = [k not in keys_to_pop for k in keys_list]
    keys_remainded_list = list(np.array(keys_list)[list(indi_to_remain)])
    return keys_remainded_list


def fmt_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


################################################################################
## list
################################################################################
def take_the_closest(list_obj, num_insert):
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

    if pos_num_insert == 0:  # When the insertion is at the first position
        closest_num = list_obj[0]
        closest_pos = pos_num_insert  # 0

    if pos_num_insert == len(
        list_obj
    ):  # When the insertion is at the last position
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert  # len(list_obj)

    else:  # When the insertion is anywhere between the first and the last positions
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
    return [math.isclose(a, b) for a, b in zip(mutable_a, mutable_b)]


################################################################################
## dictionary
################################################################################
def merge_dicts_wo_overlaps(*dicts):
    merged_dict = {}  # init
    for dict in dicts:
        assert mngs.general.search(
            merged_dict.keys(), dict.keys(), only_perfect_match=True
        ) == ([], [])
        merged_dict.update(dict)
    return merged_dict


def listed_dict(keys=None):  # Is there a better name?
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
    if src == "/dev/null":
        print(f"\n/dev/null was not copied.\n")

    else:

        if dst.endswith("/"):
            _, src_fname, src_ext = mngs.general.split_fpath(src)
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
    if isinstance(src_files, str):
        src_files = [src_files]

    if isinstance(dists, str):
        dists = [dists]

    for sf in src_files:
        for dst in dists:
            _copy_a_file(sf, dst, allow_overwrite=allow_overwrite)


def copy_the_file(sdir):  # dst
    __file__ = inspect.stack()[1].filename
    _, fname, ext = mngs.general.split_fpath(__file__)

    dst = sdir + fname + ext

    if "ipython" not in __file__:  # for ipython
        # shutil.copyfile(__file__, dst)
        # print(f"Saved to: {dst}")
        _copy_a_file(__file__, dst)


def is_nan(X):
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
    @wraps(func)
    def result(*rest, **kwargs):
        args = []
        args.extend(rest[:index])
        args.append(value)
        args.extend(rest[index:])
        return func(*args, **kwargs)

    return result


def describe(df, method="mean", factor=1):
    df = pd.DataFrame(df)
    # df = df[~df[0].isna()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "mean":
            return round(np.nanmean(df), 3), round(np.nanstd(df) / factor, 3)
        if method == "median":
            med = df.describe().T["50%"].iloc[0]
            IQR = (
                df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
            )
            return round(med, 3), round(IQR / factor, 3)


# def describe(arr, method="mean", factor=1):
#     arr = pd.DataFrame(arr)
#     arr = np.hstack(arr[~np.isnan(arr)])

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         if method == "mean":
#             return np.nanmean(df), np.nanstd(df) / factor
#         if method == "median":
#             med = df.describe().T["50%"].iloc[0]
#             IQR = df.describe().T["75%"].iloc[0] - df.describe().T["25%"].iloc[0]
#             return med, IQR / factor

# def count():
#     counter = 0
#     while True:
#         print(counter)
#         time.sleep(1)
#         counter += 1
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
    """
    Example:
        import mngs
        p1 = mngs.general._return_counting_process()
        mngs.gen.wait_key(p1)
        # press q
    """
    pressed_key = None
    while pressed_key != tgt_key:  # "q"
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


# @contextmanager
# def suppress_output():
#     """A context manager that suppresses stdout and stderr."""
#     with open(os.devnull, "w") as fnull:
#         with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(
#             fnull
#         ):
#             yield
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
        # Temporarily redirect stdout to the file descriptor fnull
        with contextlib.redirect_stdout(fnull):
            # Temporarily redirect stderr to the file descriptor fnull
            with contextlib.redirect_stderr(fnull):
                # Yield control back to the context block
                yield


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
    return unique(*args, **kwargs)


# def uq(data, axis=None):
#     def _uq(data):
#         uqs, counts = np.unique(data, return_counts=True)
#         df = pd.DataFrame({"uq": uqs, "n": counts})
#         # Format the 'Counts' column with commas for thousands
#         df["n"] = df["n"].apply(lambda x: f"{x:,}")
#         return df

#     data = pd.DataFrame(data)

#     if axis == 1:
#         dfs = {}
#         for col in data.columns:
#             df = _uq(data[col])
#             dfs[col] = df
#         return dfs

#     if axis == 0:
#         dfs = {}
#         for col in data.T.columns:
#             df = _uq(data.T[col])
#             dfs[col] = df
#         return dfs

#     if axis is None:
#         return _uq(data)


# def unique(data, axis=None):
#     """
#     Identifies unique elements in the data and their counts, returning a DataFrame.

#     Parameters:
#     - data (array-like): The input data to analyze for unique elements.
#     - show (bool, optional): If True, prints the DataFrame. Defaults to True.

#     Returns:
#     - df (pandas.DataFrame): DataFrame with unique elements and their counts.
#     """
#     uqs, counts = np.unique(data, return_counts=True)  # [REVISED]
#     df = pd.DataFrame(
#         np.vstack([uqs, counts]).T, columns=["uq", "n"]  # [REVISED]
#     ).set_index(
#         "uq"
#     )  # [REVISED]

#     df_show = df.copy()
#     df_show["n"] = df_show["n"].apply(lambda x: f"{int(x):,}")  # [REVISED]

#     return df_show
def print_block(message, char="-", n=40, c=None):
    border = char * n
    text = f"\n{border}\n{message}\n{border}\n"
    if c is not None:
        text = color_text(text, c)
    print(text)


def color_text(text, c="green"):
    ANSI_COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "grey": "\033[90m",  # Added grey color
        "reset": "\033[0m",
    }
    ANSI_COLORS["tra"] = ANSI_COLORS["white"]
    ANSI_COLORS["val"] = ANSI_COLORS["green"]
    ANSI_COLORS["tes"] = ANSI_COLORS["red"]

    start_code = ANSI_COLORS.get(c, ANSI_COLORS["reset"])
    end_code = ANSI_COLORS["reset"]
    return f"{start_code}{text}{end_code}"


ct = color_text


def mv_col(dataframe, column_name, position):
    temp_col = dataframe[column_name]
    dataframe.drop(labels=[column_name], axis=1, inplace=True)
    dataframe.insert(loc=position, column=column_name, value=temp_col)
    return dataframe


def symlink(tgt, src, force=False):
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
    if n % 2 == 0:
        return n
    else:
        return n - 1


def to_odd(n):
    if n % 2 == 0:
        return n - 1
    else:
        return n
