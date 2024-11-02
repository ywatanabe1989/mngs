#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-02 23:55:19)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_to_sigma.py
#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-27 00:34:10 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_to_sigma.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""
import mngs
import pandas as pd


def to_sigma(history):
    if len(history) > 0:
        values_list = list(history.values())
        data_frames = [format_plotting_args(record) for record in values_list]
        try:
            combined_data = pd.concat(data_frames, axis=1)
            return combined_data
        except Exception as e:
            print(e)
            return pd.DataFrame()
    else:
        return


def format_plotting_args(record):
    id, method, args, kwargs = record

    # Matplotlib
    if method in ["plot"]:
        if len(args) == 1:
            args = args[0]
            if args.ndim == 2:
                x, y = args[:, 0], args[:, 1]
                df = pd.DataFrame({f"{id}_{method}_x": x})
                return df

        elif len(args) == 2:
            x, y = args

            if isinstance(y, (np.ndarray, xr.DataArray)):
                if y.ndim == 2:
                    from collections import OrderedDict

                    out = OrderedDict()

                    for ii in range(y.shape[1]):
                        out[f"{id}_{method}_x{ii:02d}"] = x
                        out[f"{id}_{method}_y{ii:02d}"] = y[:, ii]
                    df = pd.DataFrame(out)

                    return df

            if isinstance(y, pd.DataFrame):
                df = pd.DataFrame(
                    {
                        f"{id}_{method}_x": x,
                        **{
                            f"{id}_{method}_y{ii:02d}": np.array(y[col])
                            for ii, col in enumerate(y.columns)
                        },
                    }
                )
                return df

            else:
                if isinstance(y, (np.ndarray, xr.DataArray, list)):
                    df = pd.DataFrame(
                        {f"{id}_{method}_x": x, f"{id}_{method}_y": y}
                    )
                    return df

    elif method == "scatter":
        x, y = args
        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
        return df

    elif method == "bar":
        x, y = args
        yerr = kwargs.get("yerr")

        if isinstance(x, (int, float)):
            x = pd.Series(x, name="x")
        if isinstance(y, (int, float)):
            y = pd.Series(y, name="y")

        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})

        if yerr is not None:
            if isinstance(yerr, (int, float)):
                yerr = pd.Series(yerr, name="yerr")
            df[f"{id}_{method}_yerr"] = yerr
        return df

    elif method == "hist":
        x = args
        df = pd.DataFrame({f"{id}_{method}_x": x})
        return df

    elif method == "boxplot":
        x = args[0]

        # One box plot
        if isinstance(x, np.ndarray) or mngs.gen.is_listed_X(x, [float, int]):
            df = pd.DataFrame(x)

        else:
            # Multiple boxes
            df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
        df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
        df = df.apply(lambda col: col.dropna().reset_index(drop=True))
        return df

    # elif method == "boxplot_":
    #     __import__("ipdb").set_trace()
    #     x = args[0]
    #     df =
    #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]

    #     return df

    elif method == "plot_":
        df = args
        df.columns = [f"{id}_{method}_{col}" for col in df.columns]
        return df

    elif method == "fillv":
        starts, ends = args
        df = pd.DataFrame(
            {
                f"{id}_{method}_starts": starts,
                f"{id}_{method}_ends": ends,
            }
        )
        return df

    elif method == "raster":
        df = args
        return df

    elif method == "ecdf":
        df = args
        return df

    elif method == "kde":
        df = args
        if id is not None:
            df.columns = [f"{id}_{method}_{col}" for col in df.columns]
        return df

    elif method == "imshow2d":
        df = args
        return df

    elif method == "sns_barplot":
        df = args

        # When xyhue, without errorbar
        df = pd.DataFrame(pd.Series(np.array(df).diagonal(), index=df.columns)).T
        return df

    elif method == "sns_boxplot":
        df = args
        if id is not None:
            df.columns = [f"{id}_{method}_{col}" for col in df.columns]
        return df

    elif method == "sns_heatmap":
        df = args
        return df

    elif method == "sns_histplot":
        df = args
        return df

    elif method == "sns_kdeplot":
        pass
        # df = args
        # __import__("ipdb").set_trace()
        # return df

    elif method == "sns_lineplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_pairplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_scatterplot":
        return df

    elif method == "sns_violinplot":
        df = args
        return df

    elif method == "sns_jointplot":
        __import__("ipdb").set_trace()
        return df

    else:
        pass
        # if not method.startswith("set_"):
        #     logging.warn(
        #         f"{method} is not implemented in _to_sigma method of the mngs.plt module."
        #     )


def main():
    # Line
    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
    mngs.io.save(fig, "./plots.png")
    mngs.io.save(ax.to_sigma(), "./plots.csv")

    # No tracking
    fig, ax = mngs.plt.subplots(track=False)
    ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
    mngs.io.save(fig, "./plots_wo_tracking.png")
    mngs.io.save(ax.to_sigma(), "./plots_wo_tracking.csv")

    # Scatter
    fig, ax = mngs.plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
    ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
    mngs.io.save(fig, "./scatters.png")
    mngs.io.save(ax.to_sigma(), "./scatters.csv")

    # Box
    fig, ax = mngs.plt.subplots()
    ax.boxplot([1, 2, 3], id="boxplot1")
    mngs.io.save(fig, "./boxplot1.png")
    mngs.io.save(ax.to_sigma(), "./boxplot1.csv")

    # Bar
    fig, ax = mngs.plt.subplots()
    ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
    mngs.io.save(fig, "./bar1.png")
    mngs.io.save(ax.to_sigma(), "./bar1.csv")

    # # Bar
    # # fig, ax = mngs.plt.subplots()
    # fig, ax = plt.subplots()
    # ax.bar(["A", "B", "C"], [4, 5, 6], id="bar2")
    # mngs.io.save(fig, "./bar2.png")
    # mngs.io.save(ax.to_sigma(), "./bar2.csv")

    # print(ax.to_sigma())
    # #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
    # # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
    # # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
    # # 2           3.0           6.0           6.0  ...                 3.0           C         6.0

    # print(ax.to_sigma().keys())  # plot3 and plot 4 are not tracked
    # # [3 rows x 11 columns]
    # # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
    # #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
    # #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
    # #       dtype='object')

    # # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
    # ax.to_sigma("../for_sigmaplot.csv")
    # # Saved to: ../for_sigmaplot.csv


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
