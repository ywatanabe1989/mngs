#!/usr/bin/env python3

import math
import re
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import torch


################################################################################
## strings
################################################################################
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

    if pos_num_insert == len(list_obj):  # When the insertion is at the last position
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
## dictionary
################################################################################
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
def does_exist(suspicious_var_str):
    """
    Example:
        print(does_exist('a'))
        # False

        a = 5
        print(does_exist('a'))
        # True
    """
    return suspicious_var_str in globals()
