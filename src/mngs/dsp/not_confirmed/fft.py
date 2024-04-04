#!/usr/bin/env python3
# Time-stamp: "2024-04-04 13:44:18 (ywatanabe)"

from functools import partial

import mngs
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from mngs.dsp.PARAMS import BANDS_LIM_HZ_DICT
from mngs.general import numpy_fn, torch_fn

from .filt import bandpass


@torch_fn
def fft(x, fs, dim=-1):
    if x.is_complex():
        fft_coef = torch.fft.fft(x, dim=dim)
        # Use fftfreq for complex inputs to get the full frequency range
        freqs = torch.fft.fftfreq(n=x.shape[dim], d=1.0 / fs)
    else:
        fft_coef = torch.fft.rfft(x, dim=dim)
        # Use rfftfreq for real inputs to get the non-negative frequency range
        freqs = torch.fft.rfftfreq(n=x.shape[dim], d=1.0 / fs)
    return fft_coef, freqs


def fft_powers(signals_2d, samp_rate, normalize=True):
    """
    Calculate the power spectrum of the FFT (Fast Fourier Transform) for each signal in a 2D array or tensor.

    Arguments:
        signals_2d (numpy.ndarray or torch.Tensor): A 2D array or tensor containing multiple signals,
                                                     where each row represents a signal.
        samp_rate (float): The sampling rate of the signals.
        normalize (bool, optional): If True, normalize the FFT powers by the sum of powers for each signal.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the FFT powers for each signal. The columns represent the
                          frequencies, and the rows represent the individual signals.

    Data Types:
        Input can be either numpy.ndarray or torch.Tensor. Output is always a pandas.DataFrame.

    Data Shapes:
        - signals_2d: (n_signals, signal_length)
        - Output DataFrame: (n_signals, n_frequencies)

    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_powers_df = calc_fft_powers(signals_2d, samp_rate)

    References:
        - NumPy documentation for FFT: https://numpy.org/doc/stable/reference/routines.fft.html
        - PyTorch documentation for FFT: https://pytorch.org/docs/stable/fft.html
        - pandas.DataFrame documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """

    return calc_fft_amps_2d(signals_2d, samp_rate, normalize=normalize)


# def fft_amps(signals_2d, samp_rate, normalize=True):
#     """
#     Example:
#         sig_len = 1024
#         n_sigs = 32
#         signals_2d = np.random.rand(n_sigs, sig_len)
#         samp_rate = 256
#         fft_df = calc_fft_amps(signals_2d, samp_rate)
#     """
#     fft_amps = np.abs(fftpack.fft(signals_2d))
#     fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
#     mask = fft_freqs >= 0
#     fft_amps, fft_freqs = fft_amps[:, mask], np.round(fft_freqs[mask], 1)

#     if normalize == True:
#         fft_amps = fft_amps / np.sum(fft_amps, axis=1, keepdims=True)
#         # fft_outs[0] / np.sum(np.array(fft_outs[0]), axis=1, keepdims=True)

#     fft_df = pd.DataFrame(data=fft_amps, columns=fft_freqs.astype(str))
#     return fft_df


def fft_amps(signals_2d, samp_rate, normalize=True):
    """
    Calculate the amplitude of the FFT (Fast Fourier Transform) for each signal in a 2D array or tensor.

    Arguments:
        signals_2d (numpy.ndarray or torch.Tensor): A 2D array or tensor containing multiple signals,
                                                     where each row represents a signal.
        samp_rate (float): The sampling rate of the signals.
        normalize (bool, optional): If True, normalize the FFT amplitudes by the sum of amplitudes for each signal.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the FFT amplitudes for each signal. The columns represent the
                          frequencies, and the rows represent the individual signals.

    Data Types:
        Input can be either numpy.ndarray or torch.Tensor. Output is always a pandas.DataFrame.

    Data Shapes:
        - signals_2d: (n_signals, signal_length)
        - Output DataFrame: (n_signals, n_frequencies)

    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_df = calc_fft_amps(signals_2d, samp_rate)

    """
    if isinstance(signals_2d, torch.Tensor):
        signals_2d = (
            signals_2d.detach().cpu().numpy()
        )  # Convert to NumPy array if input is a PyTorch tensor

    fft_amps = np.abs(fftpack.fft(signals_2d))
    fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
    mask = fft_freqs >= 0
    fft_amps, fft_freqs = fft_amps[:, mask], np.round(fft_freqs[mask], 1)

    if normalize:
        fft_amps = fft_amps / np.sum(fft_amps, axis=1, keepdims=True)

    fft_df = pd.DataFrame(data=fft_amps, columns=fft_freqs.astype(str))
    return fft_df


def rfft_bands(
    x,
    samp_rate,
    bands_str=["delta", "theta", "lalpha", "halpha", "beta", "gamma"],
    normalize=False,
):
    """
    Returns mean absolute rfft coefficients of bands.
    Bands' definitions are as follows.

    BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }

    rfft_bands_p = partial(mngs.dsp.rfft_bands, samp_rate=samp_rate)
    """
    x, x_type = mngs.gen.my2tensor(x)

    coef, freq = rfft(x, samp_rate)
    amp = coef.abs()

    if normalize:
        amp /= amp.sum(axis=-1, keepdims=True)

    amp_bands_abs_mean = []
    for band_str in bands_str:
        low, high = BANDS_LIM_HZ_DICT[band_str]
        indi_band = (low <= freq) & (freq <= high)
        amp_band_abs_mean = amp[..., indi_band].mean(dim=-1)
        amp_bands_abs_mean.append(amp_band_abs_mean)

    out = torch.stack(amp_bands_abs_mean, dim=-1)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def spectrogram(x, fft_size, device="cuda"):
    """
    Short-time FFT for signals.

    Input: [BS, N_CHS, SEQ_LEN]
    Output: [BS, N_CHS, fft_size//2 + 1, ?]

    spect = spectrogram(x, 50)
    print(spect.shape)
    """
    x, x_type = mngs.gen.my2tensor(x)

    transform = torchaudio.transforms.Spectrogram(n_fft=fft_size).to(device)
    out = transform(x)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def rms(x):
    x, x_type = mngs.gen.my2tensor(x)

    out = torch.square(x).sqrt().mean(dim=-1, keepdims=True)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def beyond_r_sigma_ratio(x, r=2.0):
    x, x_type = mngs.gen.my2tensor(x)

    sigma = std(x)
    out = (sigma < x).float().mean(dim=-1, keepdims=True)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


class FeatureExtractorTorch(nn.Module):
    def __init__(
        self,
        samp_rate,
        features_list=[
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "median",
            "q25",
            "q75",
            "rms",
            "rfft_bands",
            "beyond_r_sigma_ratio",
        ],
        batch_size=8,
    ):
        super().__init__()

        self.func_dict = dict(
            mean=mngs.dsp.mean,
            std=mngs.dsp.std,
            skewness=mngs.dsp.skewness,
            kurtosis=mngs.dsp.kurtosis,
            median=mngs.dsp.median,
            q25=mngs.dsp.q25,
            q75=mngs.dsp.q75,
            rms=mngs.dsp.rms,
            rfft_bands=partial(mngs.dsp.rfft_bands, samp_rate=samp_rate),
            beyond_r_sigma_ratio=mngs.dsp.beyond_r_sigma_ratio,
        )

        self.features_list = features_list

        self.batch_size = batch_size

    def forward(self, x):
        if self.batch_size is None:
            conc = torch.cat(
                [self.func_dict[f_str](x) for f_str in self.features_list],
                dim=-1,
            )

        else:
            conc = []
            n_batches = len(x) // self.batch_size + 1
            for i_batch in range(n_batches):
                try:
                    start = i_batch * self.batch_size
                    end = (i_batch + 1) * self.batch_size
                    conc.append(
                        torch.cat(
                            [
                                self.func_dict[f_str](x[start:end])
                                for f_str in self.features_list
                            ],
                            dim=-1,
                        )
                    )
                except Exception as e:
                    print(e)
            return torch.cat(conc)


def main():
    BS = 32
    N_CHS = 19
    SEQ_LEN = 2000
    SAMP_RATE = 1000
    # NYQ = int(SAMP_RATE / 2)

    x = torch.randn(BS, N_CHS, SEQ_LEN)

    m = FeatureExtractorTorch(SAMP_RATE, batch_size=8)

    out = m(x)  # 15 features


if __name__ == "__main__":
    import mngs

    SAMP_RATE = 1000
    x = mngs.dsp.demo_sig_torch(
        freqs_hz=[2, 3, 5, 10], samp_rate=SAMP_RATE, len_sec=2
    )
    # x = torch.tensor(chirp(time, 3, 500, 100))
    coef, freq = rfft(x, SAMP_RATE)
    amp = coef.abs()
