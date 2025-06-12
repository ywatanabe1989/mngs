#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 09:00:52 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_style/_force_aspect.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/ax/_style/_force_aspect.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib


def force_aspect(axis, aspect=1):
    """Force a specific aspect ratio on axes containing images.

    Adjusts the aspect ratio of axes to ensure images are displayed with
    the desired proportions, preventing distortion. Particularly useful
    for heatmaps, spectrograms, or other image-based visualizations.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes object containing images to adjust.
    aspect : float, optional
        Desired aspect ratio (width/height). Default is 1 (square).

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object with forced aspect ratio.

    Raises
    ------
    AssertionError
        If axis is not a matplotlib axes object.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> data = np.random.rand(10, 20)
    >>> ax.imshow(data)
    >>> ax = force_aspect(ax, aspect=0.5)  # Make twice as tall as wide

    >>> # For square heatmaps
    >>> ax.imshow(correlation_matrix)
    >>> ax = force_aspect(ax, aspect=1)

    Notes
    -----
    This function requires that at least one image has been added to the
    axes using imshow() or similar before calling.
    """
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axes"

    im = axis.get_images()

    extent = im[0].get_extent()

    axis.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
    return axis


# EOF
