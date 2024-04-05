#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 08:02:10 (ywatanabe)"

import mngs
from mngs.general import torch_fn
from mngs.nn import Hilbert

from .filt import bandpass


@torch_fn
def hilbert(
    x,
    dim=-1,
):
    y = Hilbert(dim=dim)(x)
    return y[..., 0], y[..., 1]


def _get_scipy_x(t_sec, fs):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import chirp, hilbert

    # duration = 1.0
    # fs = 400.0

    duration = t_sec
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)

    x = signal

    x = mngs.dsp.ensure_3d(x)

    t = torch.linspace(0, T_SEC, x.shape[-1])

    return x, t, fs


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import mngs
    from scipy.signal import chirp

    T_SEC = 1.0  # 0
    FS = 400  # 128

    # x, t, f = _get_scipy_x(T_SEC, FS)
    x, t, f = mngs.dsp.demo_sig(t_sec=T_SEC, fs=FS, type="chirp")

    pha, amp = hilbert(
        x,
        dim=-1,
    )  # (32, 19, 1280, 2)

    fig, axes = mngs.plt.subplots(nrows=2)
    axes[0].plot(x[0, 0], label="orig")
    axes[0].plot(amp[0, 0], label="amp")
    axes[1].plot(pha[0, 0], label="phase")
    axes[0].legend()
    axes[1].legend()
    plt.show()
