#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-23 20:52:51 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/SubplotsManager.py

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._AxesWrapper import AxesWrapper
from ._AxisWrapper import AxisWrapper
from ._FigWrapper import FigWrapper
from ._to_sigma import to_sigma as _to_sigma


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

    def __init__(
        self,
    ):
        """Initialize the SubplotsManager with an empty plot history."""
        self._subplots_manager_history = OrderedDict()

    def __call__(self, *args, track=True, sharex=True, sharey=True, **kwargs):
        """
        Create subplots and wrap the axes with AxisWrapper.

        Returns:
            tuple: A tuple containing the figure and wrapped axes
            in the same manner with matplotlib.pyplot.subplots.
        """

        fig, axes = plt.subplots(*args, sharex=sharex, sharey=sharey, **kwargs)

        # Fig
        fig = FigWrapper(fig)

        # Axes
        axes = np.atleast_1d(axes)
        axes_orig_shape = axes.shape

        if axes_orig_shape == (1,):
            ax_wrapped = AxisWrapper(fig, axes[0], track)
            # fig.axes = [ax_wrapped]
            fig.axes = np.array([ax_wrapped])
            return fig, ax_wrapped

        else:
            axes = axes.ravel()
            axes_wrapped = [AxisWrapper(fig, ax, track) for ax in axes]
            axes = (
                np.array(axes_wrapped).reshape(axes_orig_shape)
                if axes_orig_shape
                else axes_wrapped[0]
            )
            axes = AxesWrapper(fig, axes)
            fig.axes = axes
            return fig, axes


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
