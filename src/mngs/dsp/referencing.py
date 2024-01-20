#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 23:36:39 (ywatanabe)"

import numpy as np
import pandas as pd
import torch

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


def subtract_from_random_column(data, random_state=42, rename=False):
    np.random.seed(random_state)

    if torch.is_tensor(data):
        data = data.numpy()
    
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    cols_orig = data.columns.tolist()
    cols_shuffled = np.random.permutation(cols_orig)
    cols_diff = [f"{co} - {cs}" for co, cs in zip(cols_orig, cols_shuffled)]

    diff_df = data[cols_orig].copy()
    diff_df -= data[cols_shuffled].values

    if rename:
        diff_df.columns = cols_diff
    else:
        diff_df.columns = cols_orig

    return diff_df

    
    # np.random.seed(random_state)
    # data_values = data.values
    # shuffled_indices = np.random.permutation(data_values.shape[1])
    # subtracted_data = data_values - data_values[:, shuffled_indices]

    # # Create new column names based on the subtraction
    # new_column_names = [
    #     f"{data.columns[i]} - {data.columns[shuffled_indices[i]]}"
    #     for i in range(len(data.columns))
    # ]

    # # Create a new DataFrame with the subtracted data and new column names
    # subtracted_df = pd.DataFrame(
    #     subtracted_data, index=data.index, columns=new_column_names
    # )
    # return subtracted_df
