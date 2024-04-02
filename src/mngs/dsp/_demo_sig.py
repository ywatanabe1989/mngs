#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-02 21:25:50 (ywatanabe)"

import random

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
    t_sec=3,
    fs=512,
    freqs_hz="random",
    type="periodic",
):
    if type == "meg":
        return _demo_sig_meg(
            batch_size=batch_size, n_chs=n_chs, t_sec=t_sec, fs=fs
        ).astype(np.float32)

    else:
        fn_1d = {
            "periodic": _demo_sig_periodic_1d,
            "chirp": _demo_sig_chirp_1d,
            "ripple": _demo_sig_ripple_1d,
        }.get(type)

        return (
            np.array(
                [
                    fn_1d(
                        t_sec=t_sec,
                        fs=fs,
                        freqs_hz=freqs_hz,
                    )
                    for _ in range(int(batch_size * n_chs))
                ]
            )
            .reshape(batch_size, n_chs, -1)
            .astype(np.float32)
        )


def _demo_sig_meg(batch_size=8, n_chs=19, t_sec=10, fs=512, **kwargs):
    data_path = sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    raw_fname = meg_path / "sample_audvis_raw.fif"
    fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    # Load real data as the template
    raw = mne.io.read_raw_fif(raw_fname)
    raw.set_eeg_reference(projection=True)

    raw = raw.resample(fs)
    raw = raw.crop(tmax=t_sec)

    return raw.get_data(picks=raw.ch_names[: batch_size * n_chs]).reshape(
        batch_size, n_chs, -1
    )


def _demo_sig_periodic_1d(t_sec=10, fs=512, freqs_hz="random", **kwargs):
    """Returns a demo signal with the shape (t_sec*fs,)."""

    if freqs_hz == "random":
        n_freqs = random.randint(1, 5)
        freqs_hz = np.random.permutation(np.arange(fs))[:n_freqs]

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
    t_sec=10, fs=512, low_hz="random", high_hz="random", **kwargs
):
    if low_hz == "random":
        low_hz = random.randint(1, 20)

    if high_hz == "random":
        high_hz = random.randint(100, 1000)

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
    mm = demo_sig(type="meg")
    pp = demo_sig(type="periodic")
    cc = demo_sig(type="chirp")
    rr = demo_sig(type="ripple")

    fig, axes = mngs.plt.subplots(nrows=4)
    axes[0].plot(mm[0, 0], label="meg")
    axes[1].plot(pp[0, 0], label="periodic")
    axes[2].plot(cc[0, 0], label="chirp")
    axes[3].plot(rr[0, 0], label="ripple")
    for ax in axes:
        ax.legend(loc="upper right")

    plt.show()
