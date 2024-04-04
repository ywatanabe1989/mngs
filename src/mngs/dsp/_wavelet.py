#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 07:26:57 (ywatanabe)"

import mngs
from mngs.general import torch_fn
from mngs.nn import Wavelet


@torch_fn
def wavelet(
    x,
    fs,
    freq_scale="linear",
    out_scale="log",
):
    m = Wavelet(fs, freq_scale=freq_scale, out_scale=out_scale)
    return m(x), m.freqs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 1024  # Sampling rate in Hz
    freqs_hz = [30, 100, 250]
    x = mngs.dsp.np.demo_sig(fs=fs, freqs_hz=freqs_hz, type="ripple")
    y, freqs = wavelet(
        x,
        fs,
    )

    plt, CC = mngs.plt.configure_mpl(plt)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(x[0, 0])
    axes[1].imshow(y[0, 0])
    # axes[1].imshow(np.log(y[0, 0] + 1e-5))
    y, freqs = wavelet(
        x,
        fs,
    )
    axes[1].invert_yaxis()
    axes[1] = mngs.plt.ax.set_n_ticks(axes[1], n_xticks=4, n_yticks=4)

    plt.tight_layout()
    plt.show()

    ########################################
    # # working
    # from wavelets_pytorch.transform import WaveletTransformTorch
    # dt = 1 / fs
    # dj = 0.125
    # wa_torch = WaveletTransformTorch(dt, dj, )
    # cwt_torch = wa_torch.cwt(x[0, 0])
    # sns.heatmap(np.abs(cwt_torch).astype(np.float32))
    # plt.show()
    ########################################

    # # Example usage
    # m = WaveletFilter(fs, scale="linear").cuda()

    # out = []
    # for kk in m.kernel:
    #     out.append(np.convolve(x[0, 0], kk.cpu().numpy()))
    # out = np.array(out)

    # plt.plot(np.convolve(x[0, 0], m.kernel[0]))
    # y = m(torch.tensor(x).float().cuda())

    # y = y.detach().cpu().numpy()
    # y = np.abs(y)

    # axes[1].set_yticklabels(freqs)
    # sns.heatmap(
    #     np.log(y[0, 0] + 1e-5),
    #     ax=axes[1],
    #     yticklabels=freqs,
    #     xticklabels=np.arange(y.shape[-1]),
    #     cbar=False,
    #     # cbar_kws={"cbar", "aspect": 5},
    # )


# print(morlets.shape)
# print(freqs)
# # plt.plot(morlets.T)
# # plt.show()

# # 2D heatmap
# import seaborn as sns

# sns.heatmap(morlets.real)
# plt.show()


# # 3D heatmap
# wavelet_bank = morlets
# # Creating the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Create a meshgrid for the X and Y axes
# x = np.linspace(0, wavelet_bank.shape[1], wavelet_bank.shape[1])
# y = np.linspace(0, wavelet_bank.shape[0], wavelet_bank.shape[0])
# X, Y = np.meshgrid(x, y)

# # Z axis will be the magnitude of the wavelet coefficients
# Z = wavelet_bank

# # Create a surface plot
# surf = ax.plot_surface(X, Y, Z, cmap="viridis")

# # Add a color bar to indicate the magnitude
# fig.colorbar(surf)

# # Labels
# ax.set_xlabel("Time or Spatial Dimension")
# ax.set_ylabel("Wavelet Filter Index")
# ax.set_zlabel("Magnitude")

# plt.show()
