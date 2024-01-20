#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import fftpack

def calc_psd(signals_2d, samp_rate, normalize=True):
    return calc_fft_amps(signals_2d, samp_rate, normalize=normalize)

def calc_fft_powers(signals_2d, samp_rate, normalize=True):
    return calc_fft_amps(signals_2d, samp_rate, normalize=normalize)

def calc_fft_amps(signals_2d, samp_rate, normalize=True):
    """
    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_df = calc_fft_amps(signals_2d, samp_rate)
    """
    fft_amps = np.abs(fftpack.fft(signals_2d))
    fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
    mask = fft_freqs >= 0
    fft_amps, fft_freqs = fft_amps[:, mask], np.round(fft_freqs[mask], 1)

    if normalize == True:
        fft_amps = fft_amps / np.sum(fft_amps, axis=1, keepdims=True)
        # fft_outs[0] / np.sum(np.array(fft_outs[0]), axis=1, keepdims=True)
        
    fft_df = pd.DataFrame(data=fft_amps, columns=fft_freqs.astype(str))
    return fft_df
