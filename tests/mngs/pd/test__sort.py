# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-25 09:35:39 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/pd/_sort.py
# 
# import pandas as pd
# 
# 
# def sort(
#     dataframe,
#     by=None,
#     ascending=True,
#     inplace=False,
#     kind="quicksort",
#     na_position="last",
#     ignore_index=False,
#     key=None,
#     orders=None,
# ):
#     """
#     Sort DataFrame by specified column(s) with optional custom ordering and column reordering.
# 
#     Example
#     -------
#     import pandas as pd
#     df = pd.DataFrame({'A': ['foo', 'bar', 'baz'], 'B': [3, 2, 1]})
#     custom_order = {'A': ['bar', 'baz', 'foo']}
#     sorted_df = sort(df, by=None, orders=custom_order)
#     print(sorted_df)
# 
#     Parameters
#     ----------
#     dataframe : pandas.DataFrame
#         The DataFrame to sort.
#     by : str or list of str, optional
#         Name(s) of column(s) to sort by.
#     ascending : bool or list of bool, default True
#         Sort ascending vs. descending.
#     inplace : bool, default False
#         If True, perform operation in-place.
#     kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
#         Choice of sorting algorithm.
#     na_position : {'first', 'last'}, default 'last'
#         Puts NaNs at the beginning if 'first'; 'last' puts NaNs at the end.
#     ignore_index : bool, default False
#         If True, the resulting axis will be labeled 0, 1, …, n - 1.
#     key : callable, optional
#         Apply the key function to the values before sorting.
#     orders : dict, optional
#         Dictionary of column names and their custom sort orders.
# 
#     Returns
#     -------
#     pandas.DataFrame
#         Sorted DataFrame with reordered columns.
#     """
#     if orders:
#         by = (
#             [by]
#             if isinstance(by, str)
#             else list(orders.keys()) if by is None else by
#         )
# 
#         def apply_custom_order(column):
#             return (
#                 pd.Categorical(
#                     column, categories=orders[column.name], ordered=True
#                 )
#                 if column.name in orders
#                 else column
#             )
# 
#         key = apply_custom_order
#     elif isinstance(by, str):
#         by = [by]
# 
#     sorted_df = dataframe.sort_values(
#         by=by,
#         ascending=ascending,
#         inplace=False,
#         kind=kind,
#         na_position=na_position,
#         ignore_index=ignore_index,
#         key=key,
#     )
# 
#     # Reorder columns
#     if by:
#         other_columns = [col for col in sorted_df.columns if col not in by]
#         sorted_df = sorted_df[by + other_columns]
# 
#     if inplace:
#         dataframe.update(sorted_df)
#         dataframe.reindex(columns=sorted_df.columns)
#         return dataframe
#     else:
#         return sorted_df

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..pd._sort import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
