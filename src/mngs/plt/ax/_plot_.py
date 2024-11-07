#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-06 00:03:06 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/ax/_plot_.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-29 13:59:59 (ywatanabe)"

import numpy as np
import pandas as pd
from ...decorators import deprecated


@deprecated("Use plot_() instead.")
def fill_between(
    ax,
    xx=None,
    mean=None,
    median=None,
    std=None,
    ci=None,
    iqr=None,
    n=None,
    alpha=0.3,
    **kwargs,
):
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
    return plot_with_ci(
        ax,
        xx=xx,
        mean=mean,
        median=median,
        std=std,
        ci=ci,
        iqr=iqr,
        n=n,
        alpha=alpha,
        **kwargs,
    )


def plot_with_ci(
    axis,
    xx=None,
    mean=None,
    median=None,
    std=None,
    ci=None,
    iqr=None,
    n=None,
    alpha=0.3,
    **kwargs,
):
    """
    Plot mean/median with confidence interval or interquartile range.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes to plot on
    xx : array-like
        x-axis values
    mean : array-like, optional
        Mean values to plot
    median : array-like, optional
        Median values to plot
    std : array-like, optional
        Standard deviation values
    ci : array-like, optional
        Confidence interval values
    iqr : array-like, optional
        Interquartile range values
    n : int or array-like, optional
        Sample size to display in label
    **kwargs : dict
        Additional keyword arguments to pass to plot and fill_between

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot
    """
    label = kwargs.pop("label", "")

    if median is not None:
        central = median
        label += " (median"
        if iqr is not None:
            label += " ± IQR)"
            lower, upper = median - iqr / 2, median + iqr / 2
        else:
            raise ValueError(
                "If median is provided, iqr must also be provided"
            )
    elif mean is not None:
        central = mean
        label += " (mean"
        if std is not None:
            label += " ± std)"
            lower, upper = mean - std, mean + std
        elif ci is not None:
            label += " with 95% CI)"
            lower, upper = mean - ci / 2, mean + ci / 2
        else:
            raise ValueError(
                "If mean is provided, either std or ci must also be provided"
            )
    else:
        raise ValueError("Either mean or median must be provided")

    n_label = ""
    _n = None
    if n is not None:
        if isinstance(n, int):
            n_label = f" (n={n:,})"
            _n = n * np.ones_like(xx)
        elif isinstance(n, float):
            n_label = f" (n={n:.1f})"
            _n = n * np.ones_like(xx)
        elif len(n) == len(xx):
            if min(n) == max(n):
                n_label = f" (ns={min(n):,})"
            else:
                n_label = f" (ns={min(n):,}–{max(n):,})"
            _n = n
        else:
            n_label = f" (n (mean) ={np.mean(n):.1f})"
            _n = np.mean(n) * np.ones_like(xx)
        label += n_label

    if xx is None:
        xx = np.arange(len(central))

    axis.plot(xx, central, label=label, **kwargs)
    axis.fill_between(xx, lower, upper, alpha=alpha, **kwargs)

    return axis, pd.DataFrame(
        {
            "label": [label for _ in range(len(xx))],
            "xx": xx,
            "lower": lower,
            "central": central,
            "upper": upper,
            "n": _n,
        }
    )


def plot_(
    axis,
    xx=None,
    yy=None,
    mean=None,
    median=None,
    std=None,
    ci=None,
    iqr=None,
    n=None,
    alpha=0.3,
    **kwargs,
):
    """
    Automatically choose between ordinary plot and plot with confidence interval.

    Parameters
    ----------
    (same as before)

    Returns
    -------
    (same as before)
    """
    if yy is not None:
        # Ordinary plot
        if xx is None:
            xx = np.arange(len(yy))
        axis.plot(xx, yy, **kwargs)
        return axis, pd.DataFrame({"xx": xx, "yy": yy})
    elif mean is not None or median is not None:
        # Plot with confidence interval
        return plot_with_ci(
            axis,
            xx=xx,
            mean=mean,
            median=median,
            std=std,
            ci=ci,
            iqr=iqr,
            n=n,
            alpha=alpha,
            **kwargs,
        )
    else:
        raise ValueError("Either yy, mean, or median must be provided")


# EOF
