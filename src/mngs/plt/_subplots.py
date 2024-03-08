#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-04 13:05:40 (ywatanabe)"

from collections import OrderedDict

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd


def set_n_ticks(axis, n_xticks=4, n_yticks=4):
    axis.xaxis.set_major_locator(plt.MaxNLocator(n_xticks))
    axis.yaxis.set_major_locator(plt.MaxNLocator(n_yticks))
    return axis


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
        # self._plot_history = {}

        self._plot_history = OrderedDict()

    def __call__(self, *args, track=True, **kwargs):
        """
        Create subplots and wrap the axes with AxisDataCollector.

        Returns:
            tuple: A tuple containing the figure and wrapped axes
            in the same manner with matplotlib.pyplot.subplots.
        """
        fig, ax = plt.subplots(*args, **kwargs)
        ax = np.atleast_1d(ax)
        ax_wrapped = [
            AxisDataCollector(a, self._plot_history, track) for a in ax
        ]
        return fig, ax_wrapped if len(ax_wrapped) > 1 else ax_wrapped[0]

    @property
    def history(self):
        """
        Get the sorted plot history.

        Returns:
            dict: The sorted plot history.
        """
        return {k: self._plot_history[k] for k in self._plot_history}

    def reset_history(self):
        """Reset the plot history to an empty state."""
        self._plot_history = OrderedDict()

    @property
    def to_sigma(self):
        """
        Convert the plot history data to a SigmaPlot-compatible format DataFrame.

        Returns:
            DataFrame: The plot history in SigmaPlot format.
        """
        data_frames = [to_sigmaplot_format(v) for v in self.history.values()]
        combined_data = pd.concat(data_frames)
        return combined_data.apply(
            lambda col: col.dropna().reset_index(drop=True)
        )


class AxisDataCollector:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.

    Attributes:
        axis (matplotlib.axes.Axes): The Matplotlib axis being wrapped.
        _history (dict): A reference to the history dictionary from SubplotsManager.
    """

    def __init__(self, axis, history, track):
        """
        Initialize the AxisDataCollector with a given axis and history reference.

        Arguments:
            axis (matplotlib.axes.Axes): The Matplotlib axis to wrap.
            history (dict): A reference to the SubplotsManager's history dictionary.
        """
        self.axis = axis
        self._history = history
        self.track = track

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls.

        Returns:
            function: A wrapper function that calls the axis method and stores the history.
        """

        def wrapper(
            *args, id=None, track=True, n_xticks=4, n_yticks=4, **kwargs
        ):
            method = getattr(self.axis, attr)
            result = method(*args, **kwargs)

            # Apply set_n_ticks after the plotting method is called
            if method in ["plot", "scatter"]:
                self.axis = set_n_ticks(
                    self.axis,
                    n_xticks=n_xticks,
                    n_yticks=n_yticks,
                )

            # Only store the history if tracking is enabled and an ID is provided
            if self.track and (id is not None):
                self._history[id] = (id, attr, args, kwargs)
            return result

        return wrapper

    @property
    def history(self):
        """
        Get the sorted history for this axis.

        Returns:
            dict: The sorted history for this axis.
        """
        return {k: self._history[k] for k in self._history}

    def reset_history(self):
        """Reset the history for this axis."""
        self._history = {}

    def to_sigma(self, lpath=None):
        """
        Convert the axis history to a sigma format DataFrame.

        Returns:
            DataFrame: The axis history in sigma format.
        """
        try:
            data_frames = [
                to_sigmaplot_format(v) for v in self.history.values()
            ]
            combined_data = pd.concat(data_frames)
            combined_data = combined_data.apply(
                lambda col: col.dropna().reset_index(drop=True)
            )

        except Exception as e:
            print(e)
            combined_data = pd.DataFrame()

        if lpath is not None:
            mngs.io.save(combined_data, lpath)

        return combined_data


def to_sigmaplot_format(record):
    """
    Convert a single plot record to a sigma format DataFrame.

    Arguments:
        record (tuple): A tuple containing the plot id, method, arguments, and keyword arguments.

    Returns:
        DataFrame: The plot data in sigma format.
    """

    id, method, args, kwargs = record
    if method in ["plot", "scatter"]:
        x, y = args
        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
    elif method == "bar":
        x, y = args
        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
    elif method == "boxplot":
        x = args[0]
        df = pd.DataFrame({f"{id}_{method}_x": x})
    return df


subplots = SubplotsManager()


if __name__ == "__main__":
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

    # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
    ax.to_sigma("./tmp/subplots_demo/for_sigmaplot.csv")
    # Saved to: ./tmp/subplots_demo/for_sigmaplot.csv
