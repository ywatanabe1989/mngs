#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-07 20:42:49 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_format_sns_args.py

import pandas as pd


def format_sns_args_for_sigmaplot(method, *args, **kwargs):
    """
    Formats arguments for seaborn plotting functions.

    Parameters:
    - method (str): The name of the seaborn method to validate.
    - args (tuple): Positional arguments for the seaborn plotting function.
    - kwargs (dict): Keyword arguments for the seaborn plotting function.

    Returns:
    - pd.DataFrame: A DataFrame object that contains formatted data for saving as a CSV file.
    """

    df = pd.DataFrame()

    if method == "barplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "boxplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "heatmap":
        __import__("ipdb").set_trace()
        return df

    elif method == "histplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "kdeplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "lineplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "pairplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "scatterplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "violinplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "jointplot":
        __import__("ipdb").set_trace()
        return df

    else:
        raise NotImplementedError(f"{method} is not implemented.")
