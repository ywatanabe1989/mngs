#!/usr/bin/env python

import numpy as np
import torch  # 1.7.1
import torch.nn as nn
from torch.fft import fft, ifft


class BaseHilbertTransformerTorch(nn.Module):
    """
    Base class for Hilbert transformation on a given axis of the input tensor.

    Attributes:
        axis (int): The axis along which to compute the Hilbert transform.
        n (int or None): The number of points for the FFT computation. If None, the number of points is taken from the input tensor's size along the axis.

    References:
        - MONAI implementation of Hilbert transform: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/simplelayers.py
    """

    def __init__(self, axis=2, n=None):
        """
        Initialize the BaseHilbertTransformerTorch.

        Args:
            axis (int): The axis along which to compute the Hilbert transform.
            n (int or None): The number of points for the FFT computation. If None, the number of points is taken from the input tensor's size along the axis.
        """
        super().__init__()
        self.axis = axis
        self.n = n

    def transform(self, x):
        """
        Apply the Hilbert transform to the input tensor along the specified axis.

        Args:
            x (torch.Tensor): The input tensor to transform. Must be real and in shape [Batch, chns, spatial1, spatial2, ...].

        Returns:
            torch.Tensor: The analytical signal of `x`, transformed along the axis specified in `self.axis` using FFT of size `self.n`. The absolute value of the result relates to the envelope of `x` along `self.axis`.
        """
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


# class BaseHilbertTransformerTorch(nn.Module):
#     # https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/simplelayers.py
#     def __init__(self, axis=2, n=None):
#         super().__init__()
#         self.axis = axis
#         self.n = n

#     def transform(self, x):
#         """
#         Args:
#             x: Tensor or array-like to transform.
#                Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
#         Returns:
#             torch.Tensor: Analytical signal of ``x``,
#                           transformed along axis specified in ``self.axis`` using
#                           FFT of size ``self.N``.
#                           The absolute value of ``x_ht`` relates to the envelope of ``x``
#                           along axis ``self.axis``.
#         """

#         n = x.shape[self.axis] if self.n is None else self.n

#         # Create frequency axis
#         f = torch.cat(
#             [
#                 torch.true_divide(
#                     torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)
#                 ),
#                 torch.true_divide(
#                     torch.arange(-(n // 2), 0, device=x.device), float(n)
#                 ),
#             ]
#         )

#         xf = fft(x, n=n, dim=self.axis)

#         # Create step functionb
#         u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
#         u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
#         new_dims_before = self.axis
#         new_dims_after = len(xf.shape) - self.axis - 1
#         for _ in range(new_dims_before):
#             u.unsqueeze_(0)
#         for _ in range(new_dims_after):
#             u.unsqueeze_(-1)

#         transformed = ifft(xf * 2 * u, dim=self.axis)

#         return transformed


class HilbertTransformerTorch(BaseHilbertTransformerTorch):
    """
    Hilbert transformer module that computes the analytical signal of the input tensor and extracts its amplitude and phase components.

    Inherits from BaseHilbertTransformerTorchTorch.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the HilbertTransformerTorch.

        Args:
            *args: Variable length argument list to pass to the BaseHilbertTransformerTorchTorch.
            **kwargs: Arbitrary keyword arguments to pass to the BaseHilbertTransformerTorchTorch.
        """
        super().__init__(*args, **kwargs)
        self.hilbert_transform = self.transform
        self.register_buffer("pi", torch.tensor(np.pi))

    def forward(self, sig):
        """
        Apply the Hilbert transform to the input signal and extract its amplitude and phase components.

        Args:
            sig (torch.Tensor): The input signal tensor.

        Returns:
            torch.Tensor: A tensor containing the phase and amplitude components of the input signal's analytical signal.
        """
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
        """
        Calculates the argument (phase) of complex numbers in the range (-pi, pi].

        Args:
            comp (torch.Tensor): The complex tensor for which to calculate the argument.

        Returns:
            torch.Tensor: The argument (phase) of the complex tensor.
        """
        x, y = comp.real, comp.imag
        Arg = torch.zeros_like(x).type_as(x)

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


# class HilbertTransformerTorch(BaseHilbertTransformerTorchTorch):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.hilbert_transform = self.transform
#         self.register_buffer("pi", torch.tensor(np.pi))

#     def forward(self, sig):
#         sig_comp = self.hilbert_transform(sig)

#         pha = self._calc_Arg(sig_comp)
#         amp = sig_comp.abs()

#         out = torch.cat(
#             [
#                 pha.unsqueeze(-1),
#                 amp.unsqueeze(-1),
#             ],
#             dim=-1,
#         )

#         return out

#     def _calc_Arg(self, comp):
#         """Calculates argument of complex numbers in (-pi, pi] space.
#         Although torch.angle() does not have been implemented the derivative function,
#         this function seems to work.
#         """
#         x, y = comp.real, comp.imag
#         Arg = torch.zeros_like(x).type_as(x)

#         c1 = x > 0  # condition #1
#         Arg[c1] += torch.atan(y / x)[c1]

#         c2 = (x < 0) * (y >= 0)
#         Arg[c2] += torch.atan(y / x)[c2] + self.pi

#         c3 = (x < 0) * (y < 0)
#         Arg[c3] += torch.atan(y / x)[c3] - self.pi

#         c4 = (x == 0) * (y > 0)
#         Arg[c4] += self.pi / 2.0

#         c5 = (x == 0) * (y < 0)
#         Arg[c5] += -self.pi / 2.0

#         c6 = (x == 0) * (y == 0)
#         Arg[c6] += 0.0

#         return Arg


def _unwrap(x):
    """
    Unwrap a phase tensor by changing absolute jumps greater than `pi` to their 2*pi complement.

    Args:
        x (torch.Tensor): The input phase tensor.

    Returns:
        torch.Tensor: The unwrapped phase tensor.
    """
    pi = torch.tensor(np.pi)
    y = x % (2 * pi)
    return torch.where(y > pi, y - 2 * pi, y)


# def _unwrap(x):
#     pi = torch.tensor(np.pi)
#     y = x % (2 * pi)
#     return torch.where(y > pi, 2 * pi - y, y)


def test_hilbert_torch(sig, axis=1, test_layer=True):
    """
    Test function for the HilbertTransformerTorch.

    Args:
        sig (torch.Tensor): The input signal tensor.
        axis (int): The axis along which to apply the Hilbert transform.
        test_layer (bool): If True, use the HilbertTransformerTorch as a layer; otherwise, use the transform method directly.

    Returns:
        tuple: A tuple containing the phase and amplitude components of the input signal's analytical signal.
    """
    if test_layer:
        # Hilbert Layer
        hilbert_layer = HilbertTransformerTorch(axis=axis).cuda()
        sig_comp = hilbert_layer(sig)
    else:
        # Hilbert Transformation
        hilbert = HilbertTransformerTorch(axis=axis)
        sig_comp = hilbert.transform(sig)

    print(sig.shape)
    print(sig_comp.shape)

    print(sig.dtype)
    print(sig_comp.dtype)

    print(sig_comp)

    pha, amp = sig_comp[..., 0], sig_comp[..., 1]
    return pha, amp


# def test_hilbert_torch(sig, axis=1, test_layer=True):

#     if test_layer:
#         ## Hilbert Layer
#         hilbert_layer = HilbertTransformerTorch(axis=axis).cuda()
#         sig_comp = hilbert_layer(sig)

#     else:
#         ## Hilbert Transformation
#         hilbert = HilbertTransformerTorch(axis=1)
#         sig_comp = hilbert.transform(sig)

#     print(sig.shape)
#     print(sig_comp.shape)

#     print(sig.dtype)
#     print(sig_comp.dtype)

#     print(sig_comp)

#     # ## Extract Amplitude and Phase signals
#     # amp = sig_comp.abs()
#     # pha = _unwrap(sig_comp.angle())

#     # instantaneous_freq = (np.diff(pha.cpu()) / (2.0*np.pi) * fs)

#     # ## Plot
#     # fig, ax = plt.subplots(3, 1)
#     # ax[0].set_title('GPU Hilbert Transformation')

#     # ax[0].plot(t, sig[0].cpu(), label='original signal')
#     # ax[0].plot(t, amp[0].cpu(), label='envelope')
#     # ax[0].set_xlabel("time in seconds")
#     # ax[0].legend()

#     # ax[1].plot(t, pha[0].cpu(), label='phase')
#     # ax[1].set_xlabel("time in seconds")
#     # ax[1].legend()

#     # ax[2].plot(t[1:], instantaneous_freq[0], label='instantenous frequency')
#     # ax[2].set_xlabel("time in seconds")
#     # ax[2].set_ylim(0.0, 120.0)
#     # ax[2].legend()

#     # fig.show()

#     pha, amp = sig_comp[..., 0], sig_comp[..., 1]
#     return pha, amp


def mk_sig():
    """
    Create a batch of demo signals using a chirp function.

    Returns:
        torch.Tensor: A batch of demo signals as a CUDA tensor.
    """
    from scipy.signal import chirp

    def _mk_sig():
        sig = chirp(t, 20.0, t[-1], 100.0)
        sig *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
        return sig

    sig = np.array([_mk_sig() for _ in range(batch_size)]).astype(np.float32)
    sig = torch.tensor(sig).cuda()
    return sig


# def mk_sig():
#     from scipy.signal import chirp

#     def _mk_sig():
#         sig = chirp(t, 20.0, t[-1], 100.0)
#         sig *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
#         return sig

#     sig = np.array([_mk_sig() for _ in range(batch_size)]).astype(np.float32)
#     sig = torch.tensor(sig).cuda()
#     return sig


if __name__ == "__main__":
    # import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.signal import chirp

    ## Parameters
    duration = 1.0
    fs = 400.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    batch_size = 64

    ## Create demo signal
    sig = mk_sig()

    ## Test Code
    pha, amp = test_hilbert_torch(sig, axis=1, test_layer=True)
    # ## EOF
