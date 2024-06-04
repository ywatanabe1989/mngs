#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 19:11:48 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/pd/_misc.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

# sys.path = ["."] + sys.path
# from scripts import utils, load


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def find_indi(df, conditions):
    # Generate a DataFrame where each column is a boolean Series representing one condition
    condition_df = pd.concat(
        [
            (df[k] == v) if not isinstance(v, list) else df[k].isin(v)
            for k, v in conditions.items()
        ],
        axis=1,
    )
    # Return a boolean Series where each value is True only if all conditions are met for that row
    return condition_df.all(axis=1)


def slice(df, conditions):
    _df = df.copy()
    # Use the find_indi function to get the indices of rows that meet all conditions
    indi = find_indi(df, conditions)
    # Return the filtered DataFrame
    return _df[indi]


def main():
    pass


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
