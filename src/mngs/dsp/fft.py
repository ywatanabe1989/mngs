#!/usr/bin/env python3
# Time-stamp: "2024-04-03 00:50:51 (ywatanabe)"

from functools import partial

import mngs
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from mngs.dsp.PARAMS import BANDS_LIM_HZ_DICT
from mngs.general import numpy_fn, torch_fn


@torch_fn
def phase(x, dim=-1):
    analytical_x = hilbert(x, dim=dim)
    out = analytical_x[..., 0]
    return out


def phase_band(x, samp_rate, band_str="delta", l=None, h=None):
    """
    Example:
        x = mngs.dsp.demo_sig_torch(x)
        mngs.dsp.phase_band_torch(x, samp_rate=1000, band_str="delta")

        x = mngs.dsp.demo_sig_torch(x)
        mngs.dsp.phase_band_torch(x, samp_rate=1000, band_str=None, l=0.5, h=4)

    Bands definitions:
        BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }
    """
    x, x_type = mngs.gen.my2tensor(x)

    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass(x, samp_rate, low_hz=l, high_hz=h)
    out = phase(x)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def amp(x, axis=-1):
    x, x_type = mngs.gen.my2tensor(x)

    analytical_x = hilbert(x, axis=axis)
    out = analytical_x[..., 1]

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def amp_band(x, samp_rate, band_str="delta", l=None, h=None):
    """
    Example:
        x = mngs.dsp.demo_sig_torch(x)
        mngs.dsp.amp_band_torch(x, samp_rate=1000, band_str="delta")

        x = mngs.dsp.demo_sig_torch(x)
        mngs.dsp.amp_band_torch(x, samp_rate=1000, band_str=None, l=0.5, h=4)

    Bands definitions:
        BANDS_LIM_HZ_DICT = {
        "delta": [0.5, 4],
        "theta": [4, 8],
        "lalpha": [8, 10],
        "halpha": [10, 13],
        "beta": [13, 32],
        "gamma": [32, 75],
    }
    """
    x, x_type = mngs.gen.my2tensor(x)

    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass(x, samp_rate, low_hz=l, high_hz=h)

    out = amp(x)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def fft(x, samp_rate, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(_fft_1d, samp_rate=samp_rate, return_freq=False)
    fft_coef = mngs.torch.apply_to(fn, x, time_dim)
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)

    if x_type == "numpy":
        return mngs.gen.my2array(fft_coef)[0], mngs.gen.my2array(freq)[0]
    else:
        return fft_coef, freq


def _fft_1d(x, samp_rate, return_freq=True):
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.fft(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


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


def rfft(x, samp_rate, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(_rfft_1d, samp_rate=samp_rate, return_freq=False)
    fft_coef = mngs.torch.apply_to(fn, x, time_dim)
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)

    if x_type == "numpy":
        return mngs.gen.my2array(fft_coef)[0], mngs.gen.my2array(freq)[0]
    else:
        return fft_coef, freq


def _rfft_1d(x, samp_rate, return_freq=True):
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.rfft(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


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
