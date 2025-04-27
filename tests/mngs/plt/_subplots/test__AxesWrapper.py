#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:39:19 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__AxesWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__AxesWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_subplots/_AxesWrapper.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-02 09:36:05 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/AxesWrapper.py
#
# from functools import wraps
#
# import numpy as np
# import pandas as pd
#
#
# class AxesWrapper:
#     """
#     A wrapper class for multiple axes objects that can convert each to a pandas DataFrame
#     and concatenate them into a single DataFrame.
#     """
#
#     def __init__(self, fig, axes):
#         """
#         Initializes the AxesWrapper with a list of axes.
#
#         Parameters:
#             axes (iterable): An iterable of objects that have a to_sigma() method returning a DataFrame.
#         """
#
#         self.fig = fig
#         self.axes = axes
#
#     def get_figure(
#         self,
#     ):
#         return self.fig
#
#     def __getitem__(self, index):
#         """
#         Allows the AxesWrapper to be indexed.
#
#         Parameters:
#             index (int or slice): The index or slice of the axes to retrieve.
#
#         Returns:
#             The axis at the specified index or a new AxesWrapper with the sliced axes.
#         """
#         if isinstance(index, slice):
#             # Return a new AxesWrapper object containing the sliced part
#             return AxesWrapper(self.axes[index])
#         else:
#             # Return the specific axis at the index
#             return self.axes[index]
#
#     def __iter__(self):
#         return iter(self.axes)
#
#     def __len__(self):
#         return len(self.axes)
#
#     def legend(self, loc="upper left"):
#         return [ax.legend(loc=loc) for ax in self.axes]
#
#     @property
#     def history(self):
#         if isinstance(self.axes, np.ndarray):
#             return [ax.history for ax in self.axes.flat]
#         else:
#             return self.axes.history
#
#     @property
#     def shape(self):
#         return self.axes.shape
#
#     def to_sigma(self):
#         """
#         Converts each axis to a DataFrame using their to_sigma method and concatenates them along columns.
#
#         Returns:
#             DataFrame: A concatenated DataFrame of all axes.
#         """
#         dfs = []
#         for i_ax, ax in enumerate(self.axes.ravel()):
#             df = ax.to_sigma()
#             df.columns = [f"ax_{i_ax:02d}_{col}" for col in df.columns]
#             dfs.append(df)
#
#             # # Add a spacer column after each dataframe except the last one
#             # if i_ax < len(self.axes) - 1:
#             #     spacer = pd.DataFrame({"Spacer": [np.nan] * len(df)})
#             #     dfs.append(spacer)
#             # __import__("ipdb").set_trace()
#
#         return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
#
#     def ravel(self):
#         """
#         Flattens the AxesWrapper into a 1D array-like structure of axes.
#
#         Returns:
#             list: A list containing all axes objects.
#         """
#         self.axes = self.axes.ravel()  # [REVISED]
#         return self.axes
#
#     def flatten(self):
#         """
#         Flattens the AxesWrapper into a 1D array-like structure of axes.
#
#         Returns:
#             list: A list containing all axes objects.
#         """
#         self.axes = self.axes.flatten()
#         return self.axes
#
#     @property
#     def flat(self):
#         """
#         Flattens the AxesWrapper into a 1D array-like structure of axes.
#
#         Returns:
#             list: A list containing all axes objects.
#         """
#         if isinstance(self.axes, list):
#             return self.axes
#         else:
#             return self.axes.flat

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

# tests/mngs/plt/_subplots/test__AxesWrapper.py

project_root = Path(__file__).resolve().parents[3]
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mngs.plt._subplots._AxesWrapper import AxesWrapper


class Dummy:
    def __init__(self, name):
        self.name = name

    def to_sigma(self):
        return pd.DataFrame({self.name: [5, 6]})


def test_axeswrapper_len_getitem_and_to_sigma():
    arr = np.array([Dummy("x"), Dummy("y")])
    wrapper = AxesWrapper("fig", arr)

    assert len(wrapper) == 2
    assert wrapper[0].name == "x"

    out = wrapper.to_sigma()
    assert list(out.columns) == ["ax_00_x", "ax_01_y"]
    assert out["ax_00_x"].tolist() == [5, 6]
    assert out["ax_01_y"].tolist() == [5, 6]

# EOF