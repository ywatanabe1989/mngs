#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-07 19:14:40 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/AxesWrapper.py

import pandas as pd


class AxesWrapper:
    """
    A wrapper class for multiple axes objects that can convert each to a pandas DataFrame
    and concatenate them into a single DataFrame.
    """

    def __init__(self, axes):
        """
        Initializes the AxesWrapper with a list of axes.

        Parameters:
            axes (iterable): An iterable of objects that have a to_sigma() method returning a DataFrame.
        """

        self.axes = axes
        # self.shape = (len(axes),)  # [REVISED]

    def __getitem__(self, index):
        """
        Allows the AxesWrapper to be indexed.

        Parameters:
            index (int or slice): The index or slice of the axes to retrieve.

        Returns:
            The axis at the specified index or a new AxesWrapper with the sliced axes.
        """
        if isinstance(index, slice):
            # Return a new AxesWrapper object containing the sliced part
            return AxesWrapper(self.axes[index])
        else:
            # Return the specific axis at the index
            return self.axes[index]

    def __iter__(self):
        return iter(self.axes)

    def __len__(self):
        return len(self.axes)

    @property
    def shape(self):
        return self.axes.shape

    def to_sigma(self):
        """
        Converts each axis to a DataFrame using their to_sigma method and concatenates them along columns.

        Returns:
            DataFrame: A concatenated DataFrame of all axes.
        """
        dfs = []
        for i_ax, ax in enumerate(self.axes.ravel()):
            df = ax.to_sigma()
            df.columns = [f"{i_ax}_{col}" for col in df.columns]
            dfs.append(df)
        return pd.concat(dfs, axis=1)

    def ravel(self):
        """
        Flattens the AxesWrapper into a 1D array-like structure of axes.

        Returns:
            list: A list containing all axes objects.
        """
        self.axes = self.axes.ravel()  # [REVISED]
        return self.axes

    def flatten(self):
        """
        Flattens the AxesWrapper into a 1D array-like structure of axes.

        Returns:
            list: A list containing all axes objects.
        """
        self.axes = self.axes.flatten()
        return self.axes
