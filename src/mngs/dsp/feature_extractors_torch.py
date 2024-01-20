#!/usr/bin/env python3
# Time-stamp: "2024-01-20 13:57:31 (ywatanabe)"

from functools import partial

import torch
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


def phase_torch(x, axis=-1):
    analytical_x = hilbert_torch(x, axis=-1)
    return analytical_x[..., 0]


def phase_band_torch(x, samp_rate, band_str="delta", l=None, h=None):
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
    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass_torch(x, samp_rate, low_hz=l, high_hz=h)
    return phase_torch(x)


def amp_torch(x, axs=-1):
    analytical_x = hilbert_torch(x, axis=-1)
    return analytical_x[..., 1]


def amp_band_torch(x, samp_rate, band_str="delta", l=None, h=None):
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
    is_band_passed = (band_str is not None) & (l is None) & (h is None)
    is_lh_passed = (band_str is None) & (l is not None) & (h is not None)
    assert is_band_passed or is_lh_passed

    if is_band_passed:
        l, h = BANDS_LIM_HZ_DICT[band_str]

    x = bandpass_torch(x, samp_rate, low_hz=l, high_hz=h)
    return amp_torch(x)


def hilbert_torch(x, axis=-1):
    if axis == -1:
        axis = x.ndim - 1
    return HilbertTransformerTorch(axis=axis).to(x.device)(x)


def fft_torch(x, samp_rate, time_dim=-1):
    fn = partial(_fft_1d_torch, samp_rate=samp_rate, return_freq=False)
    fft_coef = _apply_to_the_time_dim(fn, x, time_dim)
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)
    return fft_coef, freq


def _fft_1d_torch(x, samp_rate, return_freq=True):
    freq = torch.fft.fftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.fft_torch(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


def rfft_bands_torch(
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
    coef, freq = rfft_torch(x, samp_rate)
    amp = coef.abs()

    if normalize:
        amp /= amp.sum(axis=-1, keepdims=True)

    amp_bands_abs_mean = []
    for band_str in bands_str:
        low, high = BANDS_LIM_HZ_DICT[band_str]
        indi_band = (low <= freq) & (freq <= high)
        amp_band_abs_mean = amp[..., indi_band].mean(dim=-1)
        amp_bands_abs_mean.append(amp_band_abs_mean)

    return torch.stack(amp_bands_abs_mean, dim=-1)


def rfft_torch(x, samp_rate, time_dim=-1):
    fn = partial(_rfft_1d_torch, samp_rate=samp_rate, return_freq=False)
    fft_coef = _apply_to_the_time_dim(fn, x, time_dim)
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)
    return fft_coef, freq


def _rfft_1d_torch(x, samp_rate, return_freq=True):
    freq = torch.fft.rfftfreq(x.shape[-1], d=1 / samp_rate)
    fft_coef = torch.fft.rfft_torch(x)  # [:, :, :nyq]

    if return_freq:
        return fft_coef, freq
    else:
        return fft_coef


def bandstop_torch(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    fn = partial(
        _bandstop_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    return _apply_to_the_time_dim(fn, x, time_dim)


def _bandstop_1d(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d_torch(x, samp_rate)
    indi_to_cut = (low_hz < freq) & (freq < high_hz)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft_torch(fft_coef)


def bandpass_torch(x, samp_rate, low_hz=55, high_hz=65, time_dim=-1):
    fn = partial(
        _bandpass_1d, samp_rate=samp_rate, low_hz=low_hz, high_hz=high_hz
    )
    return _apply_to_the_time_dim(fn, x, time_dim)


def _bandpass_1d_torch(x, samp_rate, low_hz=55, high_hz=65):
    fft_coef, freq = _rfft_1d_torch(x, samp_rate)
    indi_to_cut = (freq < low_hz) + (high_hz < freq)
    fft_coef[indi_to_cut] = 0
    return torch.fft.irfft_torch(fft_coef)


def spectrogram_torch(x, fft_size, device="cuda"):
    """
    Short-time FFT for signals.

    Input: [BS, N_CHS, SEQ_LEN]
    Output: [BS, N_CHS, fft_size//2 + 1, ?]

    spect = spectrogram_torch(x, 50)
    print(spect.shape)
    """

    transform = torchaudio.transforms.Spectrogram(n_fft=fft_size).to(device)
    spectrogram = transform(x)
    return spectrogram


def mean_np_torch(x):
    return x.mean(-1, keepdims=True)


def std_np_torch(x):
    return x.std(-1, keepdims=True)


def zscore_torch(x):
    _mean = mean_np_torch(x)
    diffs = x - _mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=-1, keepdims=True)
    std = torch.pow(var, 0.5)
    return diffs / std


def kurtosis_torch(x):
    zscores = zscore_torch(x)
    kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=-1, keepdims=True) - 3.0
    return kurtoses


def skewness_torch(x):
    zscores = zscore_torch(x)
    return torch.mean(torch.pow(zscores, 3.0), dim=-1, keepdims=True)


def median_torch(x):
    return torch.median(x, dim=-1, keepdims=True)[0]


def q25_torch(x, q=0.25):
    return torch.quantile(x, q, dim=-1, keepdims=True)


def q75_torch(x, q=0.75):
    return torch.quantile(x, q, dim=-1, keepdims=True)


def rms_torch(x):
    return torch.square(x).sqrt().mean(dim=-1, keepdims=True)


def beyond_r_sigma_ratio_torch(x, r=2.0):
    sigma = std_np_torch(x)
    return (sigma < x).float().mean(dim=-1, keepdims=True)


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


# def _test_notch_filter_1d_torch():
#     time = torch.linspace(0, 1, 999)
#     sig = (
#         torch.cos(60 * 2 * torch.pi * time)
#         + torch.cos(200 * 2 * torch.pi * time)
#         + torch.cos(300 * 2 * torch.pi * time)
#     )

#     sig_filted = notch_filter_1d(sig, SAMP_RATE, cutoff_hz=60, width_hz=5)
#     fig, axes = plt.subplots(4, 1)
#     axes[0].plot(sig, label="sig")
#     fft_coef_sig, freq_sig = _rfft_1d_torch(sig, SAMP_RATE)
#     axes[1].plot(freq_sig, fft_coef_sig.abs(), label="fft_coef_sig")
#     axes[2].plot(sig_filted, label="sig_filted")

#     fft_coef_sig_filted, freq_sig_filted = _rfft_1d_torch(sig_filted, SAMP_RATE)
#     axes[3].plot(
#         freq_sig_filted, fft_coef_sig_filted.abs(), label="fft_coef_sig_filted"
#     )
#     for ax in axes:
#         ax.legend()
#     fig.show()


# def test_phase_amp_bandpass_torch():
#     samp_rate = 1000
#     len_seq = 10
#     time = torch.arange(0, len_seq, 1 / samp_rate)
#     freqs_hz = [2, 5, 10]
#     sig = torch.vstack(
#         [torch.sin(f * 2 * torch.pi * time) for f in freqs_hz]
#     ).sum(dim=0)
#     # sig = _bandstop_1d(sig, samp_rate, low_hz=4, high_hz=12)
#     sig = sig.unsqueeze(0).unsqueeze(0).repeat(BS, N_CHS, 1)
#     sig = bandpass_torch(sig, samp_rate, low_hz=0, high_hz=3)

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


if __name__ == "__main__":
    import mngs

    SAMP_RATE = 1000
    x = mngs.dsp.demo_sig_torch(
        freqs_hz=[2, 3, 5, 10], samp_rate=SAMP_RATE, len_sec=2
    )
    # x = torch.tensor(chirp(time, 3, 500, 100))
    coef, freq = rfft_torch(x, SAMP_RATE)
    amp = coef.abs()
