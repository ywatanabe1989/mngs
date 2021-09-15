#!/usr/bin/env python3

import numpy as np

import pandas as pd


################################################################################
## Pandas
################################################################################
def col_to_last(df, col):
    df_orig = df.copy()
    cols = list(df_orig.columns)
    cols_remain = pop_keys_from_keys_list(cols, col)
    out = pd.concat((df_orig[cols_remain], df_orig[col]), axis=1)
    return out


def col_to_top(df, col):
    df_orig = df.copy()
    cols = list(df_orig.columns)
    cols_remain = pop_keys_from_keys_list(cols, col)
    out = pd.concat((df_orig[col], df_orig[cols_remain]), axis=1)
    return out


class IDAllocator(object):
    """
    Example1:
        patterns = np.array([3, 2, 1, 2, 50, 20])
        alc = IDAllocator(patterns)
        input = np.array([2, 20, 3, 1])
        IDs = alc(input)
        print(IDs) # [1, 3, 2, 0]

    Example2:
        patterns = np.array(['a', 'b', 'c', 'zzz'])
        alc = IDAllocator(patterns)
        input = np.array(['c', 'a', 'zzz', 'b'])
        IDs = alc(input)
        print(IDs) # [2, 0, 3, 1]
    """

    def __init__(self, patterns):
        patterns_uq = np.unique(patterns)  # natural sorting is executed.
        new_IDs = np.arange(len(patterns_uq))
        self.correspondence_table = pd.DataFrame(
            {
                "Original": patterns_uq,
                "new_ID": new_IDs,
            }
        ).set_index("Original")

    def __call__(self, x):
        allocated = np.array(self.correspondence_table.loc[x]).squeeze()
        return allocated
