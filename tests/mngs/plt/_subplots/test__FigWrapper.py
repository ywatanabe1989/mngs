#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:49:04 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__FigWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__FigWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from mngs.plt._subplots._FigWrapper import FigWrapper


class TestFigWrapper:
    def setup_method(self):
        self.fig = plt.figure()
        self.wrapper = FigWrapper(self.fig)

    def test_init(self):
        assert self.wrapper.fig is self.fig
        assert hasattr(self.wrapper, "axes")
        assert self.wrapper.axes == []

    def test_getattr_existing_attribute(self):
        # Test accessing an existing attribute on the figure
        assert hasattr(self.wrapper, "figsize")

    def test_getattr_existing_method(self):
        # Test accessing an existing method on the figure
        assert callable(self.wrapper.add_subplot)

    def test_getattr_warning(self):
        # Test attempting to access a non-existent attribute
        with pytest.warns(UserWarning, match="not implemented, ignored"):
            result = self.wrapper.nonexistent_method()
            assert result is None

    def test_legend(self):
        # Create mock axes
        ax1 = MagicMock()
        ax2 = MagicMock()
        self.wrapper.axes = MagicMock()
        self.wrapper.axes.__iter__ = lambda _: iter([ax1, ax2])

        # Call legend
        self.wrapper.legend(loc="upper right")

        # Check that legend was called on each axis
        ax1.legend.assert_called_once_with(loc="upper right")
        ax2.legend.assert_called_once_with(loc="upper right")

    def test_to_sigma_with_empty_axes(self):
        # Test with no axes
        self.wrapper.axes = MagicMock()
        self.wrapper.axes.flat = []

        result = self.wrapper.to_sigma()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_to_sigma_with_data(self):
        # Create mock axes with sigma data
        ax1 = MagicMock()
        ax1.to_sigma.return_value = pd.DataFrame(
            {"x": [1, 2, 3], "y": [4, 5, 6]}
        )

        self.wrapper.axes = MagicMock()
        self.wrapper.axes.flat = [ax1]

        result = self.wrapper.to_sigma()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "ax_00_x" in result.columns
        assert "ax_00_y" in result.columns

    def test_supxyt(self):
        # Test supxyt method
        self.wrapper.fig = MagicMock()

        # Call with x and y labels
        self.wrapper.supxyt(x="X Label", y="Y Label")

        # Check that appropriate methods were called
        self.wrapper.fig.supxlabel.assert_called_once_with("X Label")
        self.wrapper.fig.supylabel.assert_called_once_with("Y Label")
        self.wrapper.fig.suptitle.assert_not_called()

        # Reset and test with title
        self.wrapper.fig.reset_mock()
        self.wrapper.supxyt(t="Title")

        self.wrapper.fig.supxlabel.assert_not_called()
        self.wrapper.fig.supylabel.assert_not_called()
        self.wrapper.fig.suptitle.assert_called_once_with("Title")

    def test_tight_layout(self):
        # Test tight_layout method
        self.wrapper.fig = MagicMock()

        # Call with default rect
        self.wrapper.tight_layout()

        # Check that tight_layout was called with the correct rect
        self.wrapper.fig.tight_layout.assert_called_once_with(
            rect=[0, 0.03, 1, 0.95]
        )

        # Reset and test with custom rect
        self.wrapper.fig.reset_mock()
        custom_rect = [0.1, 0.1, 0.9, 0.9]
        self.wrapper.tight_layout(rect=custom_rect)

        self.wrapper.fig.tight_layout.assert_called_once_with(rect=custom_rect)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_FigWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 12:19:10 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_FigWrapper_v03.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_FigWrapper_v03.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import warnings
# from functools import wraps
# 
# import pandas as pd
# 
# 
# class FigWrapper:
#     def __init__(self, fig):
#         self.fig = fig
#         self.axes = []
# 
#     def __getattr__(self, name):
#         if hasattr(self.fig, name):
#             orig = getattr(self.fig, name)
#             if callable(orig):
# 
#                 @wraps(orig)
#                 def wrapper(*args, **kwargs):
#                     return orig(*args, **kwargs)
# 
#                 return wrapper
#             return orig
# 
#         warnings.warn(
#             f"MNGS FigWrapper: '{name}' not implemented, ignored.",
#             UserWarning,
#         )
# 
#         def dummy(*args, **kwargs):
#             return None
# 
#         return dummy
# 
#     def legend(self, loc="upper left"):
#         for ax in self.axes:
#             try:
#                 ax.legend(loc=loc)
#             except:
#                 pass
# 
#     def to_sigma(self):
#         dfs = []
#         for ii, ax in enumerate(self.axes.flat):
#             if hasattr(ax, "to_sigma"):
#                 df = ax.to_sigma()
#                 if not df.empty:
#                     df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
#                     dfs.append(df)
#         return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
# 
#     def supxyt(self, x=False, y=False, t=False):
#         if x is not False:
#             self.fig.supxlabel(x)
#         if y is not False:
#             self.fig.supylabel(y)
#         if t is not False:
#             self.fig.suptitle(t)
#         return self.fig
# 
#     def tight_layout(self, rect=[0, 0.03, 1, 0.95]):
#         self.fig.tight_layout(rect=rect)
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_FigWrapper.py
# --------------------------------------------------------------------------------
