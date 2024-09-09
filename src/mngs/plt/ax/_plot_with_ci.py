#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 10:07:22 (ywatanabe)"

from mngs.gen import deprecated


@deprecated("Use plot_with_ci(0 instaed.")
def fill_between(axis, xx_values, mean_values, std_values, n=None, **kwargs):
    """
    Deprecated function to plot mean with confidence interval.

    Example
    -------
    import matplotlib.pyplot as plt
    import numpy as np

    xx = np.linspace(0, 10, 100)
    mean = np.sin(xx)
    std = 0.1 * np.ones_like(xx)

    fig, axis = plt.subplots()
    fill_between(axis, xx, mean, std, n=50, label='Sine wave')
    plt.legend()
    plt.show()

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes to plot on
    xx_values : array-like
        x-axis values
    mean_values : array-like
        Mean values to plot
    std_values : array-like
        Standard deviation values
    n : int, optional
        Sample size to display in label
    **kwargs : dict
        Additional keyword arguments to pass to plot and fill_between

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot
    """
    axis.plot(xx_values, mean_values, **kwargs)
    if n is not None:
        label = kwargs.get("label", "")
        kwargs["label"] = f"{label} (n={n})"

    axis.fill_between(
        xx_values, mean_values - std_values, mean_values + std_values, **kwargs
    )

    return axis


def plot_with_ci(axis, xx_values, mean_values, std_values, n=None, **kwargs):
    """
    Plot mean with confidence interval.

    Example
    -------
    import matplotlib.pyplot as plt
    import numpy as np

    xx = np.linspace(0, 10, 100)
    mean = np.sin(xx)
    std = 0.1 * np.ones_like(xx)

    fig, axis = plt.subplots()
    plot_with_ci(axis, xx, mean, std, n=50, label='Sine wave')
    plt.legend()
    plt.show()

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes to plot on
    xx_values : array-like
        x-axis values
    mean_values : array-like
        Mean values to plot
    std_values : array-like
        Standard deviation values
    n : int, optional
        Sample size to display in label
    **kwargs : dict
        Additional keyword arguments to pass to plot and fill_between

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot
    """
    return fill_between(axis, xx_values, mean_values, std_values, n, **kwargs)


# def fill_between(ax, xx, mean, std, n=None, **kwargs):
#     ax.plot(xx, mean, **kwargs)
#     if n is not None:
#         label = kwargs.get("label", "")
#         label += f" (n={n})"

#     ax.fill_between(xx, mean - std, mean + std, **kwargs)

#     return ax


# def plot_with_ci(ax, xx, mean, std, n=None, **kwargs):
#     return fill_between(ax, xx, mean, std, **kwargs)
