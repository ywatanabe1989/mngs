#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:35:16 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__AxesWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__AxesWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest

# class TestAxesWrapper:
#     def setup_method(self):
#         self.fig_mock = MagicMock()
#         self.ax1 = MagicMock()
#         self.ax2 = MagicMock()
#         self.axes_array = np.array([[self.ax1, self.ax2]])
#         self.wrapper = AxesWrapper(self.fig_mock, self.axes_array)

#     def test_init(self):
#         assert self.wrapper.fig is self.fig_mock
#         assert np.array_equal(self.wrapper.axes, self.axes_array)

#     def test_get_figure(self):
#         assert self.wrapper.get_figure() is self.fig_mock

#     def test_getattr_existing_method(self):
#         # Set up a method on both axes
#         self.ax1.set_title = MagicMock()
#         self.ax2.set_title = MagicMock()

#         # Call the method on the wrapper
#         self.wrapper.set_title("Test Title")

#         # Check that it was called on both axes
#         self.ax1.set_title.assert_called_once_with("Test Title")
#         self.ax2.set_title.assert_called_once_with("Test Title")

#     def test_getattr_property(self):
#         # Set up a property on both axes
#         type(self.ax1).figbox = PropertyMock(return_value="figbox1")
#         type(self.ax2).figbox = PropertyMock(return_value="figbox2")

#         # Get the property from the wrapper
#         result = self.wrapper.figbox

#         # Should return a list of the property values
#         assert result == ["figbox1", "figbox2"]

#     def test_getattr_warning(self):
#         # Test attempting to access a non-existent attribute
#         with pytest.warns(UserWarning, match="not implemented, ignored"):
#             result = self.wrapper.nonexistent_method()
#             assert result is None

#     def test_getitem(self):
#         # Test accessing by index
#         result = self.wrapper[0, 0]
#         assert result is self.ax1

#         # Test slice returning AxesWrapper
#         result = self.wrapper[0, :]
#         assert isinstance(result, AxesWrapper)
#         assert np.array_equal(result.axes, np.array([self.ax1, self.ax2]))

#     def test_iteration(self):
#         # Test iteration through axes
#         axes = list(self.wrapper)
#         assert axes == [self.ax1, self.ax2]

#     def test_len(self):
#         # Test len() returns number of axes
#         assert len(self.wrapper) == 2

#     def test_legend(self):
#         # Test legend method
#         self.wrapper.legend(loc="upper left")

#         # Should call legend on both axes
#         self.ax1.legend.assert_called_once_with(loc="upper left")
#         self.ax2.legend.assert_called_once_with(loc="upper left")

#     def test_history_property(self):
#         # Set up history on both axes
#         self.ax1.history = {"plot1": "data1"}
#         self.ax2.history = {"plot2": "data2"}

#         # Get history from wrapper
#         result = self.wrapper.history

#         # Should return list of histories
#         assert result == [{"plot1": "data1"}, {"plot2": "data2"}]

#     def test_shape_property(self):
#         # Test shape property
#         assert self.wrapper.shape == (1, 2)

#     def test_export_as_csv(self):
#         # Set up export_as_csv on both axes
#         self.ax1.export_as_csv = MagicMock(
#             return_value=pd.DataFrame({"x1": [1, 2, 3], "y1": [4, 5, 6]})
#         )
#         self.ax2.export_as_csv = MagicMock(
#             return_value=pd.DataFrame({"x2": [7, 8, 9], "y2": [10, 11, 12]})
#         )

#         # Call export_as_csv on wrapper
#         result = self.wrapper.export_as_csv()

#         # Check the result
#         assert isinstance(result, pd.DataFrame)
#         assert list(result.columns) == [
#             "ax_00_x1",
#             "ax_00_y1",
#             "ax_01_x2",
#             "ax_01_y2",
#         ]

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 23:06:28 (ywatanabe)"
# # File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_AxesWrapper.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from functools import wraps
# 
# import pandas as pd
# 
# 
# class AxesWrapper:
#     def __init__(self, fig_mngs, axes_mngs):
#         self._fig_mngs = fig_mngs
#         self._axes_mngs = axes_mngs
# 
#     def get_figure(self):
#         return self._fig_mngs
# 
#     def __getattr__(self, attr):
#         # print(f"Attribute of FigWrapper: {attr}")
#         attr_mpl = getattr(self._axes_mngs, attr)
# 
#         if callable(attr_mpl):
# 
#             @wraps(attr_mpl)
#             def wrapper(*args, track=None, id=None, **kwargs):
#                 results = attr_mpl(*args, **kwargs)
#                 # self._track(track, id, attr, args, kwargs)
#                 return results
# 
#             return wrapper
# 
#         else:
#             return attr_mpl
# 
#     # def __getattr__(self, name):
#     #     print(f"Attribute of AxesWrapper: {name}")
#     #     methods = []
#     #     try:
#     #         for axis in self._axes_mngs.flat:
#     #             methods.append(getattr(axis, name))
#     #     except Exception:
#     #         methods = []
# 
#     #     if methods and all(callable(m) for m in methods):
# 
#     #         @wraps(methods[0])
#     #         def wrapper(*args, **kwargs):
#     #             return [
#     #                 getattr(ax, name)(*args, **kwargs)
#     #                 for ax in self._axes_mngs.flat
#     #             ]
# 
#     #         return wrapper
# 
#     #     if methods and not callable(methods[0]):
#     #         return methods
# 
#     #     warnings.warn(
#     #         f"MNGS AxesWrapper: '{name}' not implemented, ignored.",
#     #         UserWarning,
#     #     )
# 
#     #     def dummy(*args, **kwargs):
#     #         return None
# 
#     #     return dummy
# 
#     def __getitem__(self, index):
#         subset = self._axes_mngs[index]
#         if isinstance(index, slice):
#             return AxesWrapper(self._fig_mngs, subset)
#         return subset
# 
#     def __iter__(self):
#         return iter(self._axes_mngs.flat)
# 
#     def __len__(self):
#         return self._axes_mngs.size
# 
#     def legend(self, loc="upper left"):
#         return [ax.legend(loc=loc) for ax in self._axes_mngs.flat]
# 
#     @property
#     def history(self):
#         return [ax.history for ax in self._axes_mngs.flat]
# 
#     @property
#     def shape(self):
#         return self._axes_mngs.shape
# 
#     def export_as_csv(self):
#         dfs = []
#         for ii, ax in enumerate(self._axes_mngs.flat):
#             df = ax.export_as_csv()
#             df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
#             dfs.append(df)
#         return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# --------------------------------------------------------------------------------
