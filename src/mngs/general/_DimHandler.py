#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-19 01:40:36"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script demonstrates DimHandler, which:
1) Keeps designated dimensions,
2) Permutes the kept dimensions to the last while maintaining their relative order,
3) Reshapes the remaining dimensions to the first, batch dimension,
4) (Performs calculations),
5) Restores the summarized dimensions to their original shapes.
"""

# Imports
import sys

import matplotlib.pyplot as plt
import mngs
import torch

# Config
CONFIG = mngs.gen.load_configs()

# Functions
class DimHandler:
    """
    A utility class for handling dimension manipulations on tensors or arrays, including reshaping and permuting dimensions.

    Attributes:
        orig_shape (tuple): The original shape of the input tensor or array before any manipulation.
        keepdims (list): The list of dimensions to be kept and moved to the end.
        n_non_keepdims (list): The sizes of the dimensions not kept, used for reshaping back to the original shape.
        n_keepdims (list): The sizes of the kept dimensions, used for reshaping.

    Example1:
        import torch

        dh = DimHandler()
        x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
        print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
        x = dh.fit(x, keepdims=[0, 2, 5])
        print(x.shape)  # torch.Size([40, 1, 3, 6])
        x = dh.unfit(x)
        print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])

    Example 2:
        import torch

        dh = DimHandler()
        x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
        print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
        x = dh.fit(x, keepdims=[0, 2, 5])
        print(x.shape)  # torch.Size([40, 1, 3, 6])
        y = x.mean(axis=-2) # calculation on the kept dims
        print(y.shape) # torch.Size([40, 1, 6])
        y = dh.unfit(y)
        print(y.shape) # torch.Size([2, 4, 5, 1, 6])
    """

    def __init__(self):
        self.orig_shape = None
        self.keepdims = None

    def fit(self, x, keepdims=[]):
        """
        Reshapes the input tensor or array by flattening the dimensions not in `keepdims` and moving the kept dimensions to the end.

        Arguments:
            x (torch.Tensor or numpy.array): The input tensor or array to be reshaped.
            keepdims (list of int): The indices of the dimensions to keep.

        Returns:
            x_flattend (torch.Tensor or numpy.array): The reshaped tensor or array with kept dimensions moved to the end.

        Note:
            This method modifies the `orig_shape`, `keepdims`, `n_non_keepdims`, and `n_keepdims` attributes based on the input.
        """
        assert len(keepdims) <= len(x.shape)

        keepdims = sorted(keepdims)

        orig_shape = x.shape
        keepdims = keepdims
        non_keepdims = [
            int(ii)
            for ii in torch.arange(len(orig_shape))
            if ii not in keepdims
        ]

        self.n_non_keepdims = [orig_shape[nkd] for nkd in non_keepdims]
        self.n_keepdims = [orig_shape[kd] for kd in keepdims]

        x_permuted = x.permute(*non_keepdims, *keepdims)
        x_flattend = x_permuted.reshape(-1, *self.n_keepdims)

        return x_flattend

    def unfit(self, y):
        """
        Restores the reshaped tensor or array back to its original shape before the `fit` operation.

        Arguments:
            y (torch.Tensor or numpy.array): The tensor or array to be restored to its original shape.

        Returns:
            y_restored (torch.Tensor or numpy.array): The tensor or array restored to its original shape.
        """
        orig_shape = y.shape
        return y.reshape(
            *self.n_non_keepdims,
            *orig_shape[len(self.n_non_keepdims) - len(self.n_keepdims) + 1 :]
        )


if __name__ == "__main__":
    import torch

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Example1:
    mngs.gen.print_block("Example 1")
    dh = DimHandler()
    x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
    print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
    x = dh.fit(x, keepdims=[0, 2, 5])
    print(x.shape)  # torch.Size([40, 1, 3, 6])
    x = dh.unfit(x)
    print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])

    # Example 2:
    mngs.gen.print_block("Example 2")
    dh = DimHandler()
    x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
    print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
    x = dh.fit(x, keepdims=[0, 2, 5])
    print(x.shape)  # torch.Size([40, 1, 3, 6])
    y = x.mean(axis=-2)  # calculation on the kept dims
    print(y.shape)  # torch.Size([40, 1, 6])
    y = dh.unfit(y)
    print(y.shape)  # torch.Size([2, 4, 5, 1, 6])

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/general/_DimHandler.py
"""
