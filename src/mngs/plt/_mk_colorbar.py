import mngs
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def mk_colorbar(start="white", end="blue"):
    xx = np.linspace(0, 1, 256)

    start = np.array(mngs.plt.colors.RGB_d[start])
    end = np.array(mngs.plt.colors.RGB_d[end])
    colors = (end-start)[:, np.newaxis]*xx

    colors -= colors.min()
    colors /= colors.max()

    fig, ax = plt.subplots()
    [ax.axvline(_xx, color=colors[:,i_xx]) for i_xx, _xx in enumerate(xx)]
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_aspect(0.2)
    return fig


