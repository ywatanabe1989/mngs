#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 08:07:11 (ywatanabe)"

from collections import OrderedDict

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from mngs.general import deprecated


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
        self._plot_history = OrderedDict()

    def __call__(self, *args, track=True, **kwargs):
        """
        Create subplots and wrap the axes with AxisWrapper.

        Returns:
            tuple: A tuple containing the figure and wrapped axes
            in the same manner with matplotlib.pyplot.subplots.
        """

        fig, axes = plt.subplots(*args, **kwargs)

        fig = FigWrapper(fig)
        axes = np.atleast_1d(axes)
        axes_orig_shape = axes.shape

        if axes_orig_shape == (1,):
            ax_wrapped = AxisWrapper(axes[0], self._plot_history, track)
            return fig, ax_wrapped

        else:

            axes = axes.ravel()
            axes_wrapped = [
                AxisWrapper(ax, self._plot_history, track) for ax in axes
            ]

            axes = (
                np.array(axes_wrapped).reshape(axes_orig_shape)
                if axes_orig_shape
                else axes_wrapped[0]
            )
            axes = AxesWrapper(axes)
            return fig, axes

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
        combined_data = pd.concat(data_frames, ignore_index=True, join="inner")
        return combined_data.apply(
            lambda col: col.dropna().reset_index(drop=True)
        )


class FigWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, fig):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.fig = fig

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls or return the attribute directly.
        """
        original_attr = getattr(self.fig, attr)

        if callable(original_attr):

            def wrapper(
                *args, id=None, track=True, n_xticks=4, n_yticks=4, **kwargs
            ):
                result = original_attr(*args, **kwargs)

                return result

            return wrapper
        else:

            return original_attr

    ################################################################################
    # Original methods
    ################################################################################
    @deprecated("Use supxyt() instead.")
    def set_supxyt(self, *args, **kwargs):
        return self.supxyt(*args, **kwargs)

    def supxyt(self, xlabel=None, ylabel=None, title=None):
        """Sets xlabel, ylabel and title"""
        if xlabel is not None:
            self.fig.supxlabel(xlabel)
        if ylabel is not None:
            self.fig.supylabel(ylabel)
        if title is not None:
            self.fig.suptitle(title)
        return self.fig

    def tight_layout(self, rect=[0, 0.03, 1, 0.95]):
        self.fig.tight_layout(rect=rect)


class AxesWrapper:
    """
    A wrapper class for multiple axes objects that can convert each to a pandas DataFrame
    and concatenate them into a single DataFrame.
    """

    def __init__(self, axes):
        """
        Initializes the AxesWrapper with a list of axes.

        Parameters:
            axes (iterable): An iterable of objects that have a to_sigma() method returning a DataFrame.
        """

        self.axes = axes
        # self.shape = (len(axes),)  # [REVISED]

    @property
    def shape(self):
        return self.axes.shape

    def to_sigma(self):
        """
        Converts each axis to a DataFrame using their to_sigma method and concatenates them along columns.

        Returns:
            DataFrame: A concatenated DataFrame of all axes.
        """
        dfs = []
        for i_ax, ax in enumerate(self.axes.ravel()):
            df = ax.to_sigma()
            df.columns = [f"{i_ax}_{col}" for col in df.columns]
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    def ravel(self):
        """
        Flattens the AxesWrapper into a 1D array-like structure of axes.

        Returns:
            list: A list containing all axes objects.
        """
        self.axes = self.axes.ravel()  # [REVISED]
        return self.axes

    def flatten(self):
        """
        Flattens the AxesWrapper into a 1D array-like structure of axes.

        Returns:
            list: A list containing all axes objects.
        """
        self.axes = self.axes.flatten()
        return self.axes

    def __iter__(self):
        return iter(self.axes)


class AxisWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, axis, history, track):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.axis = axis
        self._history = history
        self.track = track
        self.id = 0

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls or return the attribute directly.
        """
        original_attr = getattr(self.axis, attr)

        if callable(original_attr):

            def wrapper(
                *args, id=None, track=True, n_xticks=4, n_yticks=4, **kwargs
            ):
                result = original_attr(*args, **kwargs)

                # Apply set_n_ticks after the plotting method is called
                if attr in ["plot", "scatter", "bar", "plot_with_ci"]:
                    self.axis = mngs.plt.ax.set_n_ticks(
                        self.axis, n_xticks=n_xticks, n_yticks=n_yticks
                    )

                # Only store the history if tracking is enabled and an ID is provided
                # if self.track and (id is not None):
                if self.track:
                    if id is None:
                        id = self.id
                        self.id += 1
                    self._history[id] = (id, attr, args, kwargs)
                return result

            return wrapper
        else:
            # Return the attribute directly if it's not callable
            return original_attr

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

    def to_sigma(self):
        """
        Convert the axis history to a sigma format DataFrame.

        Returns:
            DataFrame: The axis history in sigma format.
        """
        try:
            data_frames = [
                to_sigmaplot_format(v) for v in self.history.values()
            ]
            combined_data = pd.concat(data_frames, axis=1)
            # combined_data = combined_data.apply(
            #     lambda col: col.dropna().reset_index(drop=True)
            # )

        except Exception as e:
            print(e)
            combined_data = pd.DataFrame()

        return combined_data

    ################################################################################
    # Original methods
    ################################################################################
    def raster(self, positions, time=None, track=True, id=None, **kwargs):
        self.axis, df = mngs.plt.ax.raster(
            self.axis, positions, time=time, **kwargs
        )

        if track:
            if id is None:
                id = mngs.gen.gen_ID()[-8:]
            self._history[id] = (id, "raster", df, None)

    def set_xyt(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
    ):
        self.axis = mngs.plt.ax.set_xyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    def set_supxyt(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
    ):
        self.axis = mngs.plt.ax.set_supxyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    def set_ticks(
        self,
        xvals=None,
        xticks=None,
        yvals=None,
        yticks=None,
        **kwargs,
    ):

        self.axis = mngs.plt.ax.set_ticks(
            self.axis,
            xvals=xvals,
            xticks=xticks,
            yvals=yvals,
            yticks=yticks,
        )

    def set_n_ticks(self, n_xticks=4, n_yticks=4):
        self.axis = mngs.plt.ax.set_n_ticks(
            self.axis, n_xticks=n_xticks, n_yticks=n_yticks
        )

    def hide_spines(
        self,
        top=True,
        bottom=True,
        left=True,
        right=True,
        ticks=True,
        labels=True,
    ):
        self.axis = mngs.plt.ax.hide_spines(
            self.axis,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            ticks=ticks,
            labels=labels,
        )

    def plot_with_ci(
        self, xx, mean, std, label=None, alpha=0.5, track=True, id=None
    ):
        self.axis = mngs.plt.ax.plot_with_ci(
            self.axis, xx, mean, std, label=label, alpha=alpha
        )
        if track:
            if id is None:
                id = mngs.gen.gen_ID()[-8:]
            self._history[id] = (id, "fill_between", (xx, mean, std), None)

    def extend(self, x_ratio=1.0, y_ratio=1.0):
        self.axis = mngs.plt.ax.extend(
            self.axis, x_ratio=x_ratio, y_ratio=y_ratio
        )

    def rectangle(self, xx, yy, ww, hh, **kwargs):
        self.axis = mngs.plt.ax.rectangle(self.axis, xx, yy, ww, hh, **kwargs)

    def shift(self, dx=0, dy=0):
        self.axis = mngs.plt.ax.shift(self.axis, dx=dx, dy=dy)

    def imshow2d(
        self,
        arr_2d,
        cbar=True,
        cbar_label=None,
        cbar_shrink=1.0,
        cbar_fraction=0.046,
        cbar_pad=0.04,
        cmap="viridis",
        aspect="auto",
        vmin=None,
        vmax=None,
        track=True,
        id=None,
        **kwargs,
    ):
        self.axis = mngs.plt.ax.imshow2d(
            self.axis,
            arr_2d,
            cbar=cbar,
            cbar_label=cbar_label,
            cbar_shrink=cbar_shrink,
            cbar_fraction=cbar_fraction,
            cbar_pad=cbar_pad,
            cmap=cmap,
            aspect=aspect,
            vmin=vmin,
            vmax=vmax,
        )

        if track:
            if id is None:
                id = mngs.gen.gen_ID()[-8:]
            self._history[id] = (id, "imshow2d", arr_2d, None)


def to_sigmaplot_format(record):
    """
    Convert a single plot record to a sigma format DataFrame.

    Arguments:
        record (tuple): A tuple containing the plot id, method, arguments, and keyword arguments.

    Returns:
        DataFrame: The plot data in sigma format.
    """

    id, method, args, kwargs = record

    try:
        if method in ["plot", "scatter", "bar"]:
            x, y = args
            df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
            df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            return df
        elif method == "plot_with_ci":
            xx, mm, ss = args
            df = pd.DataFrame(
                {
                    f"{id}_{method}_x": xx,
                    f"{id}_{method}_under": mm - ss,
                    f"{id}_{method}_mean": mm,
                    f"{id}_{method}_upper": mm + ss,
                }
            )
            return df

        elif method == "boxplot":
            x = args[0]
            df = mngs.gen.force_dataframe(
                {i_x: _x for i_x, _x in enumerate(x)}
            )
            df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
            # df = pd.DataFrame({f"{id}_{method}_x": x})
            df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            return df
        elif method == "raster":
            df = args[0]  # record[2]
            return df

    except IndexError:
        raise ValueError(
            f"Arguments for the method '{method}' are missing or in the wrong format."
        )


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

"""
/home/ywatanabe/proj/entrance/mngs/plt/_subplots.py
"""
