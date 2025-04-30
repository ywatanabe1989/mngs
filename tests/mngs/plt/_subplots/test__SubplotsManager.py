#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 11:18:58 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__SubplotsManager.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__SubplotsManager.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import mngs


class TestSubplotsManager:
    def setup_method(self):
        self.manager = mngs.plt.subplots()

    def test_init(self):
        assert hasattr(self.manager, "_subplots_manager_history")
        assert self.manager.fig_wrapper is None

    def test_single_axis_creation(self):
        fig, ax = self.manager(figsize=(5, 4))
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 1
        assert ax.track is True

    def test_multi_axes_creation(self):
        fig, axes = self.manager(2, 2)
        assert hasattr(fig, "axes")
        assert axes.shape == (2, 2)
        assert all(ax.track is True for ax in axes.flat)

    def test_tracking_disabled(self):
        fig, ax = self.manager(track=False)
        assert ax.track is False

    def test_method_delegation(self):
        fig, ax = self.manager()
        # Test delegation to figure
        assert hasattr(self.manager, "savefig")

    def test_warning_for_unknown_method(self):
        with pytest.warns(UserWarning, match="not found, ignored"):
            result = self.manager.nonexistent_method()
            assert result is None


class TestSubplotsIntegration:
    def test_subplots_single_axis(self):
        fig, ax = mngs.plt.subplots()
        assert hasattr(ax, "plot")
        assert hasattr(ax, "scatter")
        assert hasattr(ax, "export_as_csv")

    def test_plot_with_tracking(self):
        fig, ax = mngs.plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6], id="test_plot")
        df = ax.export_as_csv()
        assert not df.empty
        assert "test_plot_plot_x" in df.columns
        assert "test_plot_plot_y" in df.columns

    def test_tracking_disabled_globally(self):
        fig, ax = subplots(track=False)
        ax.plot([1, 2, 3], [4, 5, 6], id="test_plot")
        df = ax.export_as_csv()
        assert df.empty

    def test_multiple_plot_types(self):
        fig, ax = mngs.plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
        ax.scatter([1, 2, 3], [6, 5, 4], id="scatter1")
        ax.boxplot([1, 2, 3], id="box1")

        df = ax.export_as_csv()
        assert not df.empty
        assert "plot1_plot_x" in df.columns
        assert "scatter1_scatter_x" in df.columns
        assert "box1_boxplot_x" in df.columns


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_SubplotsManager.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 19:47:47 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_SubplotsManager.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_SubplotsManager.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import warnings
# from collections import OrderedDict
# from functools import wraps
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# from ._AxesWrapper import AxesWrapper
# from ._AxisWrapper import AxisWrapper
# from ._FigWrapper import FigWrapper
#
#
# class SubplotsManager:
#     """
#     A manager class monitors data plotted using the ax methods from matplotlib.pyplot.
#     This data can be converted into a CSV file formatted for SigmaPlot compatibility.
#     Additionally, refer to the SigmaPlot macros available at mngs.gists (https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/gists).
#
#     Currently, available methods are as follows:
#         ax.plot, ax.scatter, ax.boxplot, and ax.bar
#
#     Example:
#         import matplotlib
#         import mngs
#
#         matplotlib.use("Agg")  # "TkAgg"
#
#         fig, ax = mngs.plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#         ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#         mngs.io.save(fig, "/tmp/subplots_demo/plots.png")
#
#         # Behaves like native matplotlib.pyplot.subplots without tracking
#         fig, ax = subplots(track=False)
#         ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
#         ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
#         mngs.io.save(fig, "/tmp/subplots_demo/plots.png")
#
#         fig, ax = mngs.plt.subplots()
#         ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
#         ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
#         mngs.io.save(fig, "/tmp/subplots_demo/scatters.png")
#
#         fig, ax = mngs.plt.subplots()
#         ax.boxplot([1, 2, 3], id="boxplot1")
#         mngs.io.save(fig, "/tmp/subplots_demo/boxplot1.png")
#
#         fig, ax = mngs.plt.subplots()
#         ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
#         mngs.io.save(fig, "/tmp/subplots_demo/bar1.png")
#
#         print(ax.export_as_csv())
#         #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
#         # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
#         # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
#         # 2           3.0           6.0           6.0  ...                 3.0           C         6.0
#
#         print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
#         # [3 rows x 11 columns]
#         # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
#         #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
#         #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
#         #       dtype='object')
#
#         # If a path is passed, the SigmaPlot-friendly dataframe is saved as a csv file.
#         ax.export_as_csv("./tmp/subplots_demo/for_sigmaplot.csv")
#         # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv
#     """
#
#     def __init__(self):
#         self._subplots_manager_history = OrderedDict()
#         self.fig_wrapper = None
#
#     def __call__(self, *args, track=True, sharex=True, sharey=True, **kwargs):
#         self.fig_mpl, self.axes_mpl = plt.subplots(
#             *args, sharex=sharex, sharey=sharey, **kwargs
#         )
#         self.fig_wrapper = FigWrapper(self.fig_mpl)
#
#         axes_array = np.atleast_1d(self.axes_mpl)
#         axes_shape = axes_array.shape
#
#         if axes_shape == (1,):
#             self.ax_wrapper = AxisWrapper(
#                 self.fig_wrapper, axes_array[0], track
#             )
#             self.fig_wrapper.axes = [self.ax_wrapper]
#             return self.fig_wrapper, self.ax_wrapper
#
#         axes_flat = axes_array.ravel()
#         wrapped = [
#             AxisWrapper(self.fig_wrapper, ax_, track) for ax_ in axes_flat
#         ]
#         # reshaped = np.array(wrapped).reshape(axes_shape)
#         reshaped = (
#             wrapped
#             if len(axes_shape) == 1
#             else np.reshape(wrapped, axes_shape)
#         )
#         self.axes_wrapper = AxesWrapper(self.fig_wrapper, reshaped)
#         self.fig_wrapper.axes = self.axes_wrapper
#         return self.fig_wrapper, self.axes_wrapper
#
#     def __getattr__(self, name):
#         if hasattr(self.fig_wrapper, name):
#             return getattr(self.fig_wrapper, name)
#         if hasattr(self.fig_mpl, name):
#             orig = getattr(self.fig_mpl, name)
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
#             f"MNGS SubplotsManager: '{name}' not found, ignored.",
#             UserWarning,
#         )
#
#         def dummy(*args, **kwargs):
#             return None
#
#         return dummy
#
#
# subplots = SubplotsManager()
#
# if __name__ == "__main__":
#     import matplotlib
#     import mngs
#
#     matplotlib.use("TkAgg")  # "TkAgg"
#
#     fig, ax = mngs.plt.subplots()
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#     mngs.io.save(fig, "/tmp/subplots_demo/plots.png")
#
#     # Behaves like native matplotlib.pyplot.subplots without tracking
#     fig, ax = subplots(track=False)
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
#     mngs.io.save(fig, "/tmp/subplots_demo/plots.png")
#
#     fig, ax = mngs.plt.subplots()
#     ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
#     ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
#     mngs.io.save(fig, "/tmp/subplots_demo/scatters.png")
#
#     fig, ax = mngs.plt.subplots()
#     ax.boxplot([1, 2, 3], id="boxplot1")
#     mngs.io.save(fig, "/tmp/subplots_demo/boxplot1.png")
#
#     fig, ax = mngs.plt.subplots()
#     ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
#     mngs.io.save(fig, "/tmp/subplots_demo/bar1.png")
#
#     print(ax.export_as_csv())
#     #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
#     # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
#     # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
#     # 2           3.0           6.0           6.0  ...                 3.0           C         6.0
#
#     print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
#     # [3 rows x 11 columns]
#     # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
#     #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
#     #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
#     #       dtype='object')
#
#     # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
#     ax.export_as_csv("./tmp/subplots_demo/for_sigmaplot.csv")
#     # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_SubplotsManager.py
# --------------------------------------------------------------------------------

# EOF