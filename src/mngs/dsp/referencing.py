#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 12:04:12 (ywatanabe)"


def common_average(sig_2D):
    """
    Normalize a 2D signal by subtracting the mean and dividing by the standard deviation.

    This function computes the common average referencing of a 2D signal array, where
    the mean is subtracted from each element and the result is divided by the standard
    deviation of the entire array. This is a common preprocessing step in signal processing
    to standardize the signal.

    Parameters
    ----------
    sig_2D : ndarray
        A 2D NumPy array of signal values. The array should not be empty.

    Returns
    -------
    ndarray
        The normalized 2D signal array, with the same shape as the input `sig_2D`.

    Notes
    -----
    The function does not handle cases where the standard deviation is zero (i.e., all
    elements in `sig_2D` are the same). In such cases, the function will raise a
    `RuntimeWarning` due to division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> sig_2D = np.array([[1, 2, 3], [4, 5, 6]])
    >>> common_average(sig_2D)
    array([[-1.46385011, -0.87831007, -0.29277002],
           [ 0.29277002,  0.87831007,  1.46385011]])

    References
    ----------
    For more information on common average referencing and its applications in signal
    processing, see:
    Nunez, P. L., & Srinivasan, R. (2006). Electric fields of the brain: The
    neurophysics of EEG. Oxford University Press, USA.

    """
    return (sig_2D - sig_2D.mean()) / sig_2D.std()
