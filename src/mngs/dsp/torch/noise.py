#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 11:37:05 (ywatanabe)"

import torch


def unbias_1d(x):
    """
    Removes the mean from a 1D tensor to center it around zero.

    Parameters:
    - x (torch.Tensor): Input 1D tensor.

    Returns:
    - torch.Tensor: Unbiased 1D tensor.
    """
    assert x.dim() == 1
    return x - x.mean()


def normalize_1d(x, amp=1.0):
    """
    Normalizes a 1D tensor to have a specified amplitude.

    Parameters:
    - x (torch.Tensor): Input 1D tensor.
    - amp (float): Target amplitude.

    Returns:
    - torch.Tensor: Normalized 1D tensor.
    """
    assert x.dim() == 1
    return amp * x / max(abs(x.max()), abs(x.min()))


def white_1d(x, amp=1.0):
    """
    Adds white noise to the input signal.

    Parameters:
    - x (torch.Tensor): Input signal.
    - amp (float): Amplitude of the white noise.

    Returns:
    - torch.Tensor: Signal with added white noise.
    """
    assert x.dim() == 1
    return x + torch.rand(x.size()).to(x.device) * 2 * amp - amp


def gauss_1d(x, amp=1.0):
    """
    Adds Gaussian noise to the input signal.

    Parameters:
    - x (torch.Tensor): Input signal.
    - amp (float): Amplitude of the Gaussian noise.

    Returns:
    - torch.Tensor: Signal with added Gaussian noise.
    """
    assert x.dim() == 1
    noise = torch.randn(x.size()).to(x.device)
    noise = unbias_1d(noise)
    return x + normalize_1d(noise, amp)


def brown_1d(x, amp=1.0):
    """
    Adds Brownian (Brown) noise to the input signal.

    Parameters:
    - x (torch.Tensor): Input signal.
    - amp (float): Amplitude of the Brown noise.

    Returns:
    - torch.Tensor: Signal with added Brown noise.
    """
    assert x.dim() == 1
    noise = torch.cumsum(torch.randn(x.size()).to(x.device), dim=0)
    return x + normalize_1d(unbias_1d(noise), amp)


def pink_1d(x, amp=1.0):
    """
    Adds Pink noise to the input signal.

    Parameters:
    - x (torch.Tensor): Input signal.
    - amp (float): Amplitude of the Pink noise.

    Returns:
    - torch.Tensor: Signal with added Pink noise.
    """
    assert x.dim() == 1
    cols = x.size(0)
    noise = torch.randn(cols).to(x.device)
    noise_fft = torch.fft.rfft(noise)
    indices = torch.arange(1.0, noise_fft.size(0)).to(x.device)
    noise_fft[1:] /= torch.sqrt(indices)
    noise = torch.fft.irfft(noise_fft, n=cols)
    return x + normalize_1d(unbias_1d(noise), amp)


if __name__ == "__main__":
    import mngs

    x = torch.tensor(0 * mngs.dsp.np.demo_sig_1d()).cuda()
    amp = 1.0

    fig, axes = plt.subplots(nrows=5, sharex=True, sharey=True)
    axes[0].plot(x.cpu(), label="orig")
    axes[1].plot(white_1d(x, amp=amp).cpu(), label="white")
    axes[2].plot(gauss_1d(x, amp=amp).cpu(), label="gauss")
    axes[3].plot(brown_1d(x, amp=amp).cpu(), label="brown")
    axes[4].plot(pink_1d(x, amp=amp).cpu(), label="pink")
    for ax in axes:
        ax.legend(loc="upper right")
    plt.show()
