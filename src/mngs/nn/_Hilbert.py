#!/usr/bin/env python

import torch  # 1.7.1
import torch.nn as nn
from torch.fft import fft, ifft


class Hilbert(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        # self.fp16 = fp16

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
        steepness = 50  # This value can be adjusted
        u = torch.sigmoid(
            steepness * f
        )  # by using soft step function like this, this module is fully differential

        transformed = ifft(xf * 2 * u, dim=self.dim)

        return transformed

    def forward(self, x):
        x = x.float()

        x_comp = self.hilbert_transform(x)

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
