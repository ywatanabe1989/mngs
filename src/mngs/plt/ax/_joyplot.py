#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-26 08:38:55 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_joyplot.py

import joypy
import mngs

# def plot_joy(df, cols_NT):
#     df_plot = df[cols_NT]
#     fig, axes = joypy.joyplot(
#         data=df_plot,
#         colormap=plt.cm.viridis,
#         title="Distribution of Ranked Data",
#         labels=cols_NT,
#         overlap=0.5,
#         orientation="vertical",
#     )
#     plt.xlabel("Variables")
#     plt.ylabel("Rank")
#     return fig


def joyplot(ax, data, **kwargs):
    fig, axes = joypy.joyplot(
        data=data,
        **kwargs,
    )

    if kwargs.get("orientation") == "vertical":
        xlabel = None
        ylabel = "Density"
    else:
        xlabel = "Density"
        ylabel = None

    ax = mngs.plt.ax.set_xyt(ax, xlabel, ylabel, "Joyplot")

    return ax
