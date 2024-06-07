#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-07 21:27:06 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_to_sigmaplot_format.py

import mngs
import pandas as pd


def to_sigmaplot_format(id, method, args, kwargs):
    """
    Convert a single plot record (one of the history) to a sigma format DataFrame.

    Arguments:
        record (tuple): A tuple containing the plot id, method, arguments, and keyword arguments.

    Returns:
        DataFrame: The plot data in sigma format.
    """

    # id, method, args, kwargs = record
    try:
        if method in ["plot", "scatter"]:
            if len(args) == 2:  # matplotlib
                x, y = args
            else:  # seaborn
                __import__("ipdb").set_trace()
                # fixus
                x = kwargs["x"]
                y = kwargs["height"]

            df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
            df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            return df

        if method == "bar":
            print(kwargs)
            if len(args) == 2:  # matplotlib
                x, y = args
            else:  # seaborn
                x, y = kwargs["x"], kwargs["height"]
            df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
            df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            return df

        elif method == "plot_with_ci":
            xx, mm, ss = args
            df = pd.DataFrame(
                {
                    f"{id}_{method}_x": xx,
                    f"{id}_{method}_under": mm - ss,
                    f"{id}_{method}_mean": mm,
                    f"{id}_{method}_upper": mm + ss,
                }
            )
            return df

        elif method == "boxplot":
            x = args[0]
            df = mngs.gen.force_dataframe(
                {i_x: _x for i_x, _x in enumerate(x)}
            )
            df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
            # df = pd.DataFrame({f"{id}_{method}_x": x})
            df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            return df
        elif method == "raster":
            df = args[0]  # record[2]
            return df

    except IndexError:
        raise ValueError(
            f"Arguments for the method '{method}' are missing or in the wrong format."
        )
