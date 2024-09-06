#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-29 23:08:35 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/pd/_replace.py


def replace(dataframe, replace_dict, regex=False, cols=None):
    """
    Replace values in specified columns of a DataFrame using a dictionary.

    Example
    -------
    import pandas as pd
    df = pd.DataFrame({'A': ['abc-123', 'def-456'], 'B': ['ghi-789', 'jkl-012']})
    replace_dict = {'-': '_', '1': 'one'}
    df_replaced = replace(df, replace_dict, cols=['A'])
    print(df_replaced)

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input DataFrame to modify.
    replace_dict : dict
        Dictionary of old values (keys) and new values (values) to replace.
    regex : bool, optional
        If True, treat `replace_dict` keys as regular expressions. Default is False.
    cols : list of str, optional
        List of column names to apply replacements. If None, apply to all string columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified replacements applied.
    """

    dataframe = dataframe.copy()

    if cols is None:
        cols = dataframe.select_dtypes(include=["object"]).columns

    for column in cols:
        if dataframe[column].dtype == "object":
            for old_value, new_value in replace_dict.items():
                dataframe[column] = dataframe[column].str.replace(
                    old_value, new_value, regex=regex
                )

    return dataframe
