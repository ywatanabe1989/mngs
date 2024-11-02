#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-07 12:03:29 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/pd/_merge_cols.py


"""
This script does XYZ.
"""

#!/usr/bin/env python3

import mngs
import numpy as np
import pandas as pd


################################################################################
## Pandas
################################################################################
def merge_columns(df, *args, sep1="_", sep2="-", name="merged"):
    """
    Join specified columns with their labels.

    Example:
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(
            data=np.arange(25).reshape(5, 5),
            columns=["A", "B", "C", "D", "E"],
        )

        df1 = merge_columns(df, "A", "B", sep1="_", sep2="-")
        df2 = merge_columns(df, ["A", "B"], sep1="_", sep2="-")
        assert (df1 == df2).all().all() # True

        #     A   B   C   D   E        A_B
        # 0   0   1   2   3   4    A-0_B-1
        # 1   5   6   7   8   9    A-5_B-6
        # 2  10  11  12  13  14  A-10_B-11
        # 3  15  16  17  18  19  A-15_B-16
        # 4  20  21  22  23  24  A-20_B-21


    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    *args : str or list
        Column names to join, either as separate arguments or a single list
    sep1 : str, optional
        Separator for joining column names, default "_"
    sep2 : str, optional
        Separator between column name and value, default "-"

    Returns
    -------
    pandas.DataFrame
        DataFrame with added merged column
    """
    _df = df.copy()
    columns = (
        args[0]
        if len(args) == 1 and isinstance(args[0], (list, tuple))
        else args
    )
    merged_col = _df[list(columns)].apply(
        lambda row: sep1.join(f"{col}{sep2}{val}" for col, val in row.items()),
        axis=1,
    )

    new_col_name = sep1.join(columns) if not name else str(name)
    _df[new_col_name] = merged_col
    return _df


merge_cols = merge_columns

# def merge_columns(_df, *columns):
#     import numpy as np
#     import pandas as pd

#     df = pd.DataFrame(
#         data=np.arange(25).reshape(5, 5),
#         columns=["A", "B", "C", "D", "E"],
#     )

#     columns = ["A", "B"]

#     df[columns].astype(str).apply("_".join, axis=1)
#     # 0      0_1
#     # 1      5_6
#     # 2    10_11
#     # 3    15_16
#     # 4    20_21

#     # How can I join like this?
#     # # 0      A_0-B_1
#     # # 1      A_5-B_6
#     # ...


# def merge_columns(_df, *columns):
#     """
#     Add merged columns in string.

#     Example:
#         import pandas as pd
#         import numpy as np

#         df = pd.DataFrame(data=np.arange(25).reshape(5,5),
#                           columns=["A", "B", "C", "D", "E"],
#         )

#         print(df)

#         # A   B   C   D   E
#         # 0   0   1   2   3   4
#         # 1   5   6   7   8   9
#         # 2  10  11  12  13  14
#         # 3  15  16  17  18  19
#         # 4  20  21  22  23  24

#         print(merge_columns(df, "A", "B", "C"))

#         #     A   B   C   D   E     A_B_C
#         # 0   0   1   2   3   4     0_1_2
#         # 1   5   6   7   8   9     5_6_7
#         # 2  10  11  12  13  14  10_11_12
#         # 3  15  16  17  18  19  15_16_17
#         # 4  20  21  22  23  24  20_21_22
#     """
#     from copy import deepcopy

#     df = deepcopy(_df)
#     merged = deepcopy(df[columns[0]])  # initialization
#     for c in columns[1:]:
#         merged = mngs.ml.utils.merge_labels(list(merged), deepcopy(df[c]))
#     df.loc[:, mngs.gen.connect_strs(columns)] = merged
#     return df


# """
# Imports
# """
# import os
# import re
# import sys

# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import importlib

# import mngs

# importlib.reload(mngs)

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from icecream import ic
# from natsort import natsorted
# from glob import glob
# from pprint import pprint
# import warnings
# import logging
# from tqdm import tqdm
# import xarray as xr

# # sys.path = ["."] + sys.path
# # from scripts import utils, load

# """
# Warnings
# """
# # warnings.simplefilter("ignore", UserWarning)


# """
# Config
# """
# # CONFIG = mngs.gen.load_configs()


# """
# Functions & Classes
# """


# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()

#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

# # EOF
