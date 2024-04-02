#!/usr/bin/env python

import torch  # 1.7.1
import torch.nn as nn
from torch.fft import fft, ifft


class Hilbert(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        # self.register_buffer("pi", torch.tensor(torch.pi))

        # self.hilbert_transform = self.transform

    def hilbert_transform(self, x):
        n = x.shape[self.dim]

        # Create frequency dim
        f = torch.cat(
            [
                torch.arange(0, (n - 1) // 2 + 1, device=x.device) / float(n),
                torch.arange(-(n // 2), 0, device=x.device) / float(n),
            ]
        )

        xf = fft(x, n=n, dim=self.dim)

        # Create step function
        # u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        # u = u.to(dtype=x.dtype, device=x.device)

        # Differentiable approximation of the step function using sigmoid
        # Adjust the 'steepness' parameter to control the transition sharpness
        steepness = 50  # This value can be adjusted
        u = torch.sigmoid(steepness * f)

        # new_dims_before = self.dim
        # new_dims_after = len(xf.shape) - self.dim - 1
        # for _ in range(new_dims_before):
        #     u = u.unsqueeze(0)
        # for _ in range(new_dims_after):
        #     u = u.unsqueeze(-1)

        transformed = ifft(xf * 2 * u, dim=self.dim)

        return transformed

    def forward(self, x):
        x_comp = self.hilbert_transform(x)

        # pha = self._calc_Arg(x_comp)
        pha = torch.atan2(x_comp.imag, x_comp.real)
        amp = x_comp.abs()

        assert x.shape == pha.shape == amp.shape

        out = torch.cat(
            [
                pha.unsqueeze(-1),
                amp.unsqueeze(-1),
            ],
            dim=-1,
        )

        return out

    # def _calc_Arg(self, comp):
    #     x, y = comp.real, comp.imag
    #     Arg = torch.zeros_like(x).type_as(x)

    #     self.pi = self.pi.to(x.dtype)

    #     # Conditions and corresponding phase calculations
    #     c1 = x > 0
    #     Arg[c1] = torch.atan(y / x)[c1]

    #     c2 = (x < 0) & (y >= 0)
    #     Arg[c2] = torch.atan(y / x)[c2] + self.pi

    #     c3 = (x < 0) & (y < 0)
    #     Arg[c3] = torch.atan(y / x)[c3] - self.pi

    #     c4 = (x == 0) & (y > 0)
    #     Arg[c4] = self.pi / 2.0

    #     c5 = (x == 0) & (y < 0)
    #     Arg[c5] = -self.pi / 2.0

    #     c6 = (x == 0) & (y == 0)
    #     Arg[c6] = 0.0

    #     return Arg


# def _unwrap(x):
#     pi = torch.tensor(np.pi)
#     y = x % (2 * pi)
#     return torch.where(y > pi, y - 2 * pi, y)
