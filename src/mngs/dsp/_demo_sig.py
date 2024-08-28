#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-13 01:58:46 (ywatanabe)"


"""
This script provides functions for generating demo signals for digital signal processing tasks.
"""

# Imports
import random
import sys
import warnings

import matplotlib.pyplot as plt
import mne
import mngs
import numpy as np
from mne.datasets import sample
from mne.simulation import (
    add_ecg,
    add_eog,
    add_noise,
    simulate_raw,
    simulate_sparse_stc,
)
from ripple_detection.simulate import simulate_LFP, simulate_time
from scipy.signal import chirp
from tensorpac.signals import pac_signals_wavelet

# Config
CONFIG = mngs.gen.load_configs(verbose=False)

# Functions
def demo_sig(
    sig_type="periodic",
    batch_size=8,
    n_chs=19,
    n_segments=20,
    t_sec=4,
    fs=512,
    freqs_hz=None,
    verbose=False,
):
    """
    Generate demo signals for various signal types.

    Parameters:
    -----------
    sig_type : str, optional
        Type of signal to generate. Options are "uniform", "gauss", "periodic", "chirp", "ripple", "meg", "tensorpac", "pac".
        Default is "periodic".
    batch_size : int, optional
        Number of batches to generate. Default is 8.
    n_chs : int, optional
        Number of channels. Default is 19.
    n_segments : int, optional
        Number of segments for tensorpac and pac signals. Default is 20.
    t_sec : float, optional
        Duration of the signal in seconds. Default is 4.
    fs : int, optional
        Sampling frequency in Hz. Default is 512.
    freqs_hz : list or None, optional
        List of frequencies in Hz for periodic signals. If None, random frequencies will be used.
    verbose : bool, optional
        If True, print additional information. Default is False.

    Returns:
    --------
    tuple
        A tuple containing:
        - np.ndarray: Generated signal(s) with shape (batch_size, n_chs, time_samples) or (batch_size, n_chs, n_segments, time_samples) for tensorpac and pac signals.
        - np.ndarray: Time array.
        - int: Sampling frequency.
    """
    assert sig_type in [
        "uniform",
        "gauss",
        "periodic",
        "chirp",
        "ripple",
        "meg",
        "tensorpac",
        "pac",
    ]
    tt = np.linspace(0, t_sec, int(t_sec * fs), endpoint=False)

    if sig_type == "uniform":
        return (
            np.random.uniform(
                low=-0.5, high=0.5, size=(batch_size, n_chs, len(tt))
            ),
            tt,
            fs,
        )

    elif sig_type == "gauss":
        return np.random.randn(batch_size, n_chs, len(tt)), tt, fs

    elif sig_type == "meg":
        return (
            _demo_sig_meg(
                batch_size=batch_size,
                n_chs=n_chs,
                t_sec=t_sec,
                fs=fs,
                verbose=verbose,
            ).astype(np.float32)[..., : len(tt)],
            tt,
            fs,
        )

    elif sig_type == "tensorpac":
        xx, tt = _demo_sig_tensorpac(
            batch_size=batch_size,
            n_chs=n_chs,
            n_segments=n_segments,
            t_sec=t_sec,
            fs=fs,
        )
        return xx.astype(np.float32)[..., : len(tt)], tt, fs

    elif sig_type == "pac":
        xx = _demo_sig_pac(
            batch_size=batch_size,
            n_chs=n_chs,
            n_segments=n_segments,
            t_sec=t_sec,
            fs=fs,
        )
        return xx.astype(np.float32)[..., : len(tt)], tt, fs

    else:
        fn_1d = {
            "periodic": _demo_sig_periodic_1d,
            "chirp": _demo_sig_chirp_1d,
            "ripple": _demo_sig_ripple_1d,
        }.get(sig_type)

        return (
            (
                np.array(
                    [
                        fn_1d(
                            t_sec=t_sec,
                            fs=fs,
                            freqs_hz=freqs_hz,
                            verbose=verbose,
                        )
                        for _ in range(int(batch_size * n_chs))
                    ]
                )
                .reshape(batch_size, n_chs, -1)
                .astype(np.float32)[..., : len(tt)]
            ),
            tt,
            fs,
        )

# ... (rest of the file remains unchanged)

