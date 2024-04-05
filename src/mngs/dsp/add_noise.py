#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-05 14:19:18 (ywatanabe)"

import mngs

# import numpy as np
import torch
from mngs.general import torch_fn


def _uniform(shape, amp=1.0):
    a, b = -amp, amp
    return -amp + (2 * amp) * torch.rand(shape)


@torch_fn
def gauss(x, amp=1.0):
    noise = amp * torch.randn(x.shape)
    return x + noise.to(x.device)


@torch_fn
def white(x, amp=1.0):
    return x + _uniform(x.shape, amp=amp).to(x.device)


@torch_fn
def pink(x, amp=1.0, dim=-1):
    """
    Adds pink noise to a given tensor along a specified dimension.

    Parameters:
    - x (torch.Tensor): The input tensor to which pink noise will be added.
    - amp (float, optional): The amplitude of the pink noise. Defaults to 1.0.
    - dim (int, optional): The dimension along which to add pink noise. Defaults to -1.

    Returns:
    - torch.Tensor: The input tensor with added pink noise.
    """
    cols = x.size(dim)
    noise = torch.randn(cols, dtype=x.dtype, device=x.device)
    noise = torch.fft.rfft(noise)
    indices = torch.arange(1, noise.size(0), dtype=x.dtype, device=x.device)
    noise[1:] /= torch.sqrt(indices)
    noise = torch.fft.irfft(noise, n=cols)
    noise = noise - noise.mean()  # [REVISED]
    noise_amp = torch.sqrt(torch.mean(noise**2))
    noise = noise * (amp / noise_amp)  # [REVISED]
    return x + noise.to(x.device)


@torch_fn
def brown(x, amp=1.0, dim=-1):
    noise = _uniform(x.shape, amp=amp)
    noise = torch.cumsum(noise, dim=dim)
    noise = mngs.dsp.norm.minmax(noise, amp=amp, dim=dim)
    return x + noise.to(x.device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt, CC = mngs.plt.configure_mpl(plt)

    t_sec = 1
    fs = 128
    xx, tt, fs = mngs.dsp.demo_sig(t_sec=t_sec, fs=fs)

    # t = np.linspace(0, t_sec, x.shape[-1])

    funcs = {
        "orig": lambda x: x,
        "gauss": gauss,
        "white": white,
        "pink": pink,
        "brown": brown,
    }

    fig, axes = plt.subplots(
        nrows=len(funcs), ncols=2, sharex=True, sharey=True
    )
    count = 0
    for (k, fn), axes_row in zip(funcs.items(), axes):
        for ax in axes_row:
            if count % 2 == 0:
                ax.plot(tt, fn(xx)[0, 0], label=k, c="blue")
            else:
                ax.plot(tt, (fn(xx) - xx)[0, 0], label=f"{k} - orig", c="red")
            count += 1
            ax.legend(loc="upper right")
    plt.show()

    # fig, axes = plt.subplots(nrows=len(funcs), sharex=True, sharey=True)
    # for (k, fn), ax in zip(funcs.items(), axes):
    #     ax.plot(t, fn(x)[0, 0], label=k, c="blue")
    #     ax.plot(t, (fn(x) - x)[0, 0], label=f"{k} - orig", c="red")
    #     ax.legend(loc="upper right")
    # plt.show()
