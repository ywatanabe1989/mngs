#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-01 13:23:42 (ywatanabe)"


def gaussian_filter1d(xx, radius):
    """
    Apply a one-dimensional Gaussian filter to an input array, tensor, or DataFrame.

    Arguments:
        xx (numpy.ndarray, torch.Tensor, or pandas.DataFrame): The input data to filter. It can be a 1D or 2D array/tensor/DataFrame.
        radius (int): The radius of the Gaussian kernel. The standard deviation of the Gaussian kernel is implicitly set to 1.

    Returns:
        numpy.ndarray, torch.Tensor, or pandas.DataFrame: The filtered data, with the same type as the input.

    Data Types:
        Input can be either numpy.ndarray, torch.Tensor, or pandas.DataFrame. Output will match the input data type.

    Data Shapes:
        - Input xx: If 1D, shape (n,), if 2D, shape (m, n)
        - Output: Same shape as input xx

    References:
        - SciPy documentation for gaussian_filter1d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
    """
    sigma = 1
    truncate = radius / sigma

    # Convert input to NumPy array if it is a pandas DataFrame or PyTorch tensor
    if isinstance(xx, pd.DataFrame):
        values = xx.values
        result = scipy.ndimage.gaussian_filter1d(
            values, sigma, truncate=truncate
        )
        return pd.DataFrame(result, index=xx.index, columns=xx.columns)
    elif isinstance(xx, torch.Tensor):
        values = xx.numpy()
        result = scipy.ndimage.gaussian_filter1d(
            values, sigma, truncate=truncate
        )
        return torch.from_numpy(result)
    else:
        # Assume input is a NumPy array
        return scipy.ndimage.gaussian_filter1d(xx, sigma, truncate=truncate)


def bandpass(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(
        _bandpass_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    # out = _apply_to_the_time_dim(fn, x, time_dim)
    out = mngs.torch.apply_to(fn, x, time_dim)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def _bandpass_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (freq < low_hz) + (high_hz < freq)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)


def bandstop(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(
        _bandstop_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    # out = _apply_to_the_time_dim(fn, x, time_dim)
    out = mngs.torch.apply_to(fn, x, time_dim)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def _bandstop_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (low_hz < freq) & (freq < high_hz)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)
