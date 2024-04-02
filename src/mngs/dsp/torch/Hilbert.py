#!/usr/bin/env python

import numpy as np
import torch  # 1.7.1
import torch.nn as nn
from torch.fft import fft, ifft


class BaseHilbert(nn.Module):
    def __init__(self, axis=-1, n=None):
        super().__init__()
        self.axis = axis
        self.n = n

    def transform(self, x):
        n = x.shape[self.axis] if self.n is None else self.n

        # Create frequency axis
        f = torch.cat(
            [
                torch.arange(0, (n - 1) // 2 + 1, device=x.device) / float(n),
                torch.arange(-(n // 2), 0, device=x.device) / float(n),
            ]
        )

        xf = fft(x, n=n, dim=self.axis)

        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = u.to(dtype=x.dtype, device=x.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u = u.unsqueeze(0)
        for _ in range(new_dims_after):
            u = u.unsqueeze(-1)

        transformed = ifft(xf * 2 * u, dim=self.axis)

        return transformed


class Hilbert(BaseHilbert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hilbert_transform = self.transform
        self.register_buffer("pi", torch.tensor(np.pi))

    def forward(self, sig):
        sig_comp = self.hilbert_transform(sig)

        pha = self._calc_Arg(sig_comp)
        amp = sig_comp.abs()

        out = torch.cat(
            [
                pha.unsqueeze(-1),
                amp.unsqueeze(-1),
            ],
            dim=-1,
        )

        return out

    def _calc_Arg(self, comp):
        x, y = comp.real, comp.imag
        Arg = torch.zeros_like(x).type_as(x)

        self.pi = self.pi.to(x.dtype)

        # Conditions and corresponding phase calculations
        c1 = x > 0
        Arg[c1] = torch.atan(y / x)[c1]

        c2 = (x < 0) & (y >= 0)
        Arg[c2] = torch.atan(y / x)[c2] + self.pi

        c3 = (x < 0) & (y < 0)
        Arg[c3] = torch.atan(y / x)[c3] - self.pi

        c4 = (x == 0) & (y > 0)
        Arg[c4] = self.pi / 2.0

        c5 = (x == 0) & (y < 0)
        Arg[c5] = -self.pi / 2.0

        c6 = (x == 0) & (y == 0)
        Arg[c6] = 0.0

        return Arg


# def _unwrap(x):
#     pi = torch.tensor(np.pi)
#     y = x % (2 * pi)
#     return torch.where(y > pi, y - 2 * pi, y)


def hilbert(x, axis=-1):
    x, x_type = mngs.gen.my2tensor(x)

    if axis == -1:
        axis = x.ndim - 1

    out = Hilbert(axis=axis).to(x.device)(x.double())

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


if __name__ == "__main__":
    # import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import mngs
    from scipy.signal import chirp

    # x = mngs.dsp.np.demo_sig_hip()[:, 0]
    t = np.linspace(0, 10, 1500)
    x = chirp(t, f0=6, f1=1, t1=10, method="linear")

    duration = 1.0
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    x = signal
    # x = mngs.dsp.np.demo_sig_1d(freqs_hz=[8, 16, 32, 64])
    # x = mngs.dsp.np.demo_sig_batch()  # (32, 19, 1280)

    y = hilbert(torch.tensor(x).cuda(), axis=0)  # (32, 19, 1280, 2)

    pha, amp = y[..., 0], y[..., 1]

    fig, axes = mngs.plt.subplots(nrows=2)
    axes[0].plot(x, label="orig")
    axes[0].plot(amp.cpu().numpy(), label="amp")
    axes[1].plot(pha.cpu().numpy(), label="phase")
    axes[0].legend()
    axes[1].legend()
    plt.show()

    # hilbert(torch.tensor(x))
