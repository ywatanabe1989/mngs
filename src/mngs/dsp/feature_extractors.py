#!/usr/bin/env python3
# Time-stamp: "2024-01-21 19:17:13 (ywatanabe)"

from functools import partial

import mngs
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from mngs.dsp.HilbertTransformationTorch import HilbertTransformerTorch

# Global definitions
BANDS_LIM_HZ_DICT = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "lalpha": [8, 10],
    "halpha": [10, 13],
    "beta": [13, 32],
    "gamma": [32, 75],
}


def phase(x, axis=-1):
    x, x_type = mngs.gen.my2tensor(x)

    analytical_x = hilbert(x, axis=axis)
    out = analytical_x[..., 0]

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
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


def hilbert(x, axis=-1):
    x, x_type = mngs.gen.my2tensor(x)

    if axis == -1:
        axis = x.ndim - 1

    out = HilbertTransformerTorch(axis=axis).to(x.device)(x)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def fft(x, samp_rate, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(_fft_1d, samp_rate=samp_rate, return_freq=False)
    fft_coef = _apply_to_the_time_dim(fn, x, time_dim)
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
    fft_coef = _apply_to_the_time_dim(fn, x, time_dim)
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


def bandstop(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(
        _bandstop_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    out = _apply_to_the_time_dim(fn, x, time_dim)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def _bandstop_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (low_hz < freq) & (freq < high_hz)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)


def bandpass(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    x, x_type = mngs.gen.my2tensor(x)

    fn = partial(
        _bandpass_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    out = _apply_to_the_time_dim(fn, x, time_dim)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def _bandpass_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d(x, samp_rate)
    indi_to_cut = (freq < low_hz) + (high_hz < freq)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft(fft_coef)


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


def mean(x):
    return x.mean(-1, keepdims=True)


def std(x):
    return x.std(-1, keepdims=True)


def zscore(x):
    x, x_type = mngs.gen.my2tensor(x)

    _mean = mean(x)
    diffs = x - _mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=-1, keepdims=True)
    std = torch.pow(var, 0.5)
    out = diffs / std

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def kurtosis(x):
    x, x_type = mngs.gen.my2tensor(x)

    zscores = zscore(x)
    out = torch.mean(torch.pow(zscores, 4.0), dim=-1, keepdims=True) - 3.0

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def skewness(x):
    x, x_type = mngs.gen.my2tensor(x)

    zscores = zscore(x)
    out = torch.mean(torch.pow(zscores, 3.0), dim=-1, keepdims=True)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def median(x):
    x, x_type = mngs.gen.my2tensor(x)

    out = torch.median(x, dim=-1, keepdims=True)[0]

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def q25(x, q=0.25):
    x, x_type = mngs.gen.my2tensor(x)

    out = torch.quantile(x, q, dim=-1, keepdims=True)

    if x_type == "numpy":
        return mngs.gen.my2array(out)[0]
    else:
        return out


def q75(x, q=0.75):
    x, x_type = mngs.gen.my2tensor(x)

    out = torch.quantile(x, q, dim=-1, keepdims=True)

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


def _apply_to_the_time_dim(fn, x, time_dim):
    # Permute the tensor to bring the time dimension to the last position if it's not already
    if time_dim != -1:
        dims = list(range(x.dim()))
        dims[-1], dims[time_dim] = dims[time_dim], dims[-1]
        x = x.permute(*dims)

    # Flatten the tensor along the time dimension
    shape = x.shape
    x = x.reshape(-1, shape[-1])

    # Apply the function to each slice along the time dimension
    applied = torch.stack([fn(x_i) for x_i in torch.unbind(x, dim=0)], dim=0)

    # Reshape the tensor to its original shape (with the time dimension at the end)
    applied = applied.reshape(*shape[:-1], -1)

    # Permute back to the original dimension order if necessary
    if time_dim != -1:
        applied = applied.permute(*dims)

    return applied


# def _apply_to_the_time_dim(fn, x):
#     """
#     x: [BS, N_CHS, SEQ_LEN]
#     When fn(x[0,0]) works, _apply_to_the_time_dim(fn, x) works.
#     """
#     shape = x.shape
#     x = x.reshape(-1, shape[-1])
#     dim = 0
#     applied = torch.stack(
#         [fn(x_i) for x_i in torch.unbind(x, dim=dim)], dim=dim
#     )
#     return applied.reshape(shape[0], shape[1], -1)


# def _test_notch_filter_1d():
#     time = torch.linspace(0, 1, 999)
#     sig = (
#         torch.cos(60 * 2 * torch.pi * time)
#         + torch.cos(200 * 2 * torch.pi * time)
#         + torch.cos(300 * 2 * torch.pi * time)
#     )

#     sig_filted = notch_filter_1d(sig, SAMP_RATE, cutoff_hz=60, width_hz=5)
#     fig, axes = plt.subplots(4, 1)
#     axes[0].plot(sig, label="sig")
#     fft_coef_sig, freq_sig = _rfft_1d(sig, SAMP_RATE)
#     axes[1].plot(freq_sig, fft_coef_sig.abs(), label="fft_coef_sig")
#     axes[2].plot(sig_filted, label="sig_filted")

#     fft_coef_sig_filted, freq_sig_filted = _rfft_1d(sig_filted, SAMP_RATE)
#     axes[3].plot(
#         freq_sig_filted, fft_coef_sig_filted.abs(), label="fft_coef_sig_filted"
#     )
#     for ax in axes:
#         ax.legend()
#     fig.show()


# def test_phase_amp_bandpass():
#     samp_rate = 1000
#     len_seq = 10
#     time = torch.arange(0, len_seq, 1 / samp_rate)
#     freqs_hz = [2, 5, 10]
#     sig = torch.vstack(
#         [torch.sin(f * 2 * torch.pi * time) for f in freqs_hz]
#     ).sum(dim=0)
#     # sig = _bandstop_1d(sig, samp_rate, low_hz=4, high_hz=12)
#     sig = sig.unsqueeze(0).unsqueeze(0).repeat(BS, N_CHS, 1)
#     sig = bandpass(sig, samp_rate, low_hz=0, high_hz=3)

#     phase = _phase_1d(sig)
#     amp = _amp_1d(sig)

#     fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
#     axes[0].plot(sig[0, 0], label="sig")
#     axes[0].legend()
#     axes[1].plot(phase[0, 0], label="phase")
#     axes[1].legend()
#     axes[2].plot(amp[0, 0], label="amp")
#     axes[2].legend()
#     fig.show()


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
