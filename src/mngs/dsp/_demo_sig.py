#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-05 16:01:37 (ywatanabe)"

import random
import warnings

import mne
import numpy as np
from mne import Epochs, compute_covariance, find_events, make_ad_hoc_cov
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


def demo_sig(
    batch_size=8,
    n_chs=19,
    t_sec=4,
    fs=512,
    freqs_hz=None,
    sig_type="periodic",
    verbose=False,
):

    assert sig_type in [
        "uniform",
        "gauss",
        "periodic",
        "chirp",
        "ripple",
        "meg",
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

    if sig_type == "gauss":
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


def _demo_sig_meg(
    batch_size=8, n_chs=19, t_sec=10, fs=512, verbose=False, **kwargs
):
    data_path = sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    raw_fname = meg_path / "sample_audvis_raw.fif"
    fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    # Load real data as the template
    raw = mne.io.read_raw_fif(raw_fname, verbose=verbose)
    raw = raw.crop(tmax=t_sec, verbose=verbose)
    raw = raw.resample(fs, verbose=verbose)
    raw.set_eeg_reference(projection=True, verbose=verbose)

    return raw.get_data(
        picks=raw.ch_names[: batch_size * n_chs], verbose=verbose
    ).reshape(batch_size, n_chs, -1)


def _demo_sig_periodic_1d(
    t_sec=10, fs=512, freqs_hz=None, verbose=False, **kwargs
):
    """Returns a demo signal with the shape (t_sec*fs,)."""

    if freqs_hz is None:
        n_freqs = random.randint(1, 5)
        freqs_hz = np.random.permutation(np.arange(fs))[:n_freqs]
        if verbose:
            print(f"freqs_hz was randomly determined as {freqs_hz}")

    n = int(t_sec * fs)
    t = np.linspace(0, t_sec, n, endpoint=False)

    summed = np.array(
        [
            np.random.rand()
            * np.sin((f_hz * t + np.random.rand()) * (2 * np.pi))
            for f_hz in freqs_hz
        ]
    ).sum(axis=0)
    return summed


def _demo_sig_chirp_1d(
    t_sec=10, fs=512, low_hz=None, high_hz=None, verbose=False, **kwargs
):
    if low_hz is None:
        low_hz = random.randint(1, 20)
        if verbose:
            warnings.warn(f"low_hz was randomly determined as {low_hz}.")

    if high_hz is None:
        high_hz = random.randint(100, 1000)
        if verbose:
            warnings.warn(f"high_hz was randomly determined as {high_hz}.")

    n = int(t_sec * fs)
    t = np.linspace(0, t_sec, n, endpoint=False)
    x = chirp(t, low_hz, t[-1], high_hz)
    x *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    return x


def _demo_sig_ripple_1d(t_sec=10, fs=512, **kwargs):
    n_samples = t_sec * fs
    t = simulate_time(n_samples, fs)
    n_ripples = random.randint(1, 5)
    mid_time = np.random.permutation(t)[:n_ripples]
    return simulate_LFP(t, mid_time, noise_amplitude=1.2, ripple_amplitude=5)


if __name__ == "__main__":
    uu, tt, fs = demo_sig(sig_type="uniform")
    gg, tt, fs = demo_sig(sig_type="gauss")
    mm, tt, fs = demo_sig(sig_type="meg")
    pp, tt, fs = demo_sig(sig_type="periodic")
    cc, tt, fs = demo_sig(sig_type="chirp")
    rr, tt, fs = demo_sig(sig_type="ripple")

    fig, axes = mngs.plt.subplots(nrows=6)
    axes[0].plot(uu[0, 0], label="uniform")
    axes[1].plot(gg[0, 0], label="gauss")
    axes[2].plot(mm[0, 0], label="meg")
    axes[3].plot(pp[0, 0], label="periodic")
    axes[4].plot(cc[0, 0], label="chirp")
    axes[5].plot(rr[0, 0], label="ripple")
    for ax in axes:
        ax.legend(loc="upper right")

    plt.show()
