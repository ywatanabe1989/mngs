#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:47:47 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_SubplotsManager.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_SubplotsManager.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from collections import OrderedDict
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

from ._AxesWrapper import AxesWrapper
from ._AxisWrapper import AxisWrapper
from ._FigWrapper import FigWrapper


class SubplotsManager:
    """
    A manager class monitors data plotted using the ax methods from matplotlib.pyplot.
    This data can be converted into a CSV file formatted for SigmaPlot compatibility.
    Additionally, refer to the SigmaPlot macros available at mngs.gists (https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/gists).

    Currently, available methods are as follows:
        ax.plot, ax.scatter, ax.boxplot, and ax.bar

    Example:
        import matplotlib
        import mngs

        matplotlib.use("Agg")  # "TkAgg"

        fig, ax = subplots()
        ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
        ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
        mngs.io.save(fig, "/tmp/subplots_demo/plots.png")

        # Behaves like native matplotlib.pyplot.subplots without tracking
        fig, ax = subplots(track=False)
        ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
        ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
        mngs.io.save(fig, "/tmp/subplots_demo/plots.png")

        fig, ax = subplots()
        ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
        ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
        mngs.io.save(fig, "/tmp/subplots_demo/scatters.png")

        fig, ax = subplots()
        ax.boxplot([1, 2, 3], id="boxplot1")
        mngs.io.save(fig, "/tmp/subplots_demo/boxplot1.png")

        fig, ax = subplots()
        ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
        mngs.io.save(fig, "/tmp/subplots_demo/bar1.png")

        print(ax.to_sigma())
        #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
        # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
        # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
        # 2           3.0           6.0           6.0  ...                 3.0           C         6.0

        print(ax.to_sigma().keys())  # plot3 and plot 4 are not tracked
        # [3 rows x 11 columns]
        # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
        #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
        #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
        #       dtype='object')

        # If a path is passed, the SigmaPlot-friendly dataframe is saved as a csv file.
        ax.to_sigma("./tmp/subplots_demo/for_sigmaplot.csv")
        # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv
    """

    def __init__(self):
        self._subplots_manager_history = OrderedDict()
        self.fig_wrapper = None

    def __call__(self, *args, track=True, sharex=True, sharey=True, **kwargs):
        self.fig_mpl, self.axes_mpl = plt.subplots(
            *args, sharex=sharex, sharey=sharey, **kwargs
        )
        self.fig_wrapper = FigWrapper(self.fig_mpl)

        axes_array = np.atleast_1d(self.axes_mpl)
        axes_shape = axes_array.shape

        if axes_shape == (1,):
            self.ax_wrapper = AxisWrapper(
                self.fig_wrapper, axes_array[0], track
            )
            self.fig_wrapper.axes = [self.ax_wrapper]
            return self.fig_wrapper, self.ax_wrapper

        axes_flat = axes_array.ravel()
        wrapped = [
            AxisWrapper(self.fig_wrapper, ax_, track) for ax_ in axes_flat
        ]
        # reshaped = np.array(wrapped).reshape(axes_shape)
        reshaped = (
            wrapped
            if len(axes_shape) == 1
            else np.reshape(wrapped, axes_shape)
        )
        self.axes_wrapper = AxesWrapper(self.fig_wrapper, reshaped)
        self.fig_wrapper.axes = self.axes_wrapper
        return self.fig_wrapper, self.axes_wrapper

    def __getattr__(self, name):
        if hasattr(self.fig_wrapper, name):
            return getattr(self.fig_wrapper, name)
        if hasattr(self.fig_mpl, name):
            orig = getattr(self.fig_mpl, name)
            if callable(orig):

                @wraps(orig)
                def wrapper(*args, **kwargs):
                    return orig(*args, **kwargs)

                return wrapper
            return orig

        warnings.warn(
            f"MNGS SubplotsManager: '{name}' not found, ignored.",
            UserWarning,
        )

        def dummy(*args, **kwargs):
            return None

        return dummy


subplots = SubplotsManager()

if __name__ == "__main__":
    import matplotlib
    import mngs

    matplotlib.use("TkAgg")  # "TkAgg"

    fig, ax = subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
    mngs.io.save(fig, "/tmp/subplots_demo/plots.png")

    # Behaves like native matplotlib.pyplot.subplots without tracking
    fig, ax = subplots(track=False)
    ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
    mngs.io.save(fig, "/tmp/subplots_demo/plots.png")

    fig, ax = subplots()
    ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
    ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
    mngs.io.save(fig, "/tmp/subplots_demo/scatters.png")

    fig, ax = subplots()
    ax.boxplot([1, 2, 3], id="boxplot1")
    mngs.io.save(fig, "/tmp/subplots_demo/boxplot1.png")

    fig, ax = subplots()
    ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
    mngs.io.save(fig, "/tmp/subplots_demo/bar1.png")

    print(ax.to_sigma())
    #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
    # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
    # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
    # 2           3.0           6.0           6.0  ...                 3.0           C         6.0

    print(ax.to_sigma().keys())  # plot3 and plot 4 are not tracked
    # [3 rows x 11 columns]
    # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
    #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
    #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
    #       dtype='object')

    # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
    ax.to_sigma("./tmp/subplots_demo/for_sigmaplot.csv")
    # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv

# EOF