#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-19 00:17:06"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

import os
import sys

import matplotlib.pyplot as plt

# Imports
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# Config
CONFIG = mngs.gen.load_configs()

# Functions
# For example, I have these variables.
# pha: (batch_size, n_channels, n_freqs_pha, n_segments, sequence_length)
# amp: (batch_size, n_channels, n_freqs_amp, n_segments, sequence_length)

# Then, I want to calculate pac values like this
# pac = modulation_index_in_place(pha_reshaped, amp_reshaped).cpu()
# pha should be (-1, n_freqs_pha, sequence_length)
# amp should be (-1, n_freqs_amp, sequence_length)

# However, for me, reshape and permute is laborious. So, I want you to write a decorator to handle organize dimensions probably using reshape and permute.
# Also, sometimes, due to RAM or VRAM limitations, we have to use for loops although we prefer matrix calculations.

# Overall, I want you to write a decorator named "batch_fn", with arguments "non_batch_dims" and "forloop_dims"


def batch_fn(non_batch_dims_pha, non_batch_dims_amp, forloop_dims=False):
    """
    A decorator to adapt functions to work with additional batch dimensions, dynamically reshaping and permuting
    tensors based on specified non-batch dimensions. Optionally applies the function in a loop to manage memory usage.
    Includes dimension checks and error handling.

    Parameters:
    - non_batch_dims_pha: Tuple indicating the non-batch dimensions for the pha tensor.
    - non_batch_dims_amp: Tuple indicating the non-batch dimensions for the amp tensor.
    - forloop_dims: Boolean indicating whether to apply the function in a loop to manage memory usage.

    Returns:
    - A decorator that modifies the function to work with tensors that have additional batch dimensions.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(pha, amp):
            # Check if tensors have enough dimensions
            if pha.dim() < len(non_batch_dims_pha) or amp.dim() < len(
                non_batch_dims_amp
            ):
                raise ValueError(
                    "Input tensors do not have enough dimensions based on the specified non-batch dimensions."
                )

            # Check if the non-batch dimensions match the specified shapes
            if (
                pha.shape[-len(non_batch_dims_pha) :] != non_batch_dims_pha
                or amp.shape[-len(non_batch_dims_amp) :] != non_batch_dims_amp
            ):
                raise ValueError(
                    "The shape of the non-batch dimensions does not match the specified shapes."
                )

            # Calculate the product of non-batch dimensions for reshaping
            non_batch_prod_pha = torch.prod(
                torch.tensor(pha.shape[-len(non_batch_dims_pha) :])
            )
            non_batch_prod_amp = torch.prod(
                torch.tensor(amp.shape[-len(non_batch_dims_amp) :])
            )

            # Reshape tensors to (-1, *non_batch_dims)
            pha_reshaped = pha.reshape(-1, *non_batch_dims_pha)
            amp_reshaped = amp.reshape(-1, *non_batch_dims_amp)

            if forloop_dims:
                # Apply the function in a loop to manage memory usage
                results = []
                for i in range(pha_reshaped.shape[0]):
                    result = func(
                        pha_reshaped[i : i + 1], amp_reshaped[i : i + 1]
                    ).cpu()
                    results.append(result)
                result_tensor = torch.cat(results, dim=0)
            else:
                # Apply the function directly
                result_tensor = func(pha_reshaped, amp_reshaped).cpu()

            # Assuming the function does not change the number of non-batch dimensions,
            # reshape the result tensor back to the original batch shape with non-batch dimensions combined
            final_shape = (
                pha.shape[: -len(non_batch_dims_pha)] + result_tensor.shape[1:]
            )
            result_tensor = result_tensor.reshape(final_shape)

            return result_tensor

        return wrapper

    return decorator


non_batch_dims, forloop_dims


def batch_fn(*non_batch_dims):
    def decorator(func):
        def wrapper(x):
            # Save the original shape
            orig_shape = x.shape
            # Compute the shape for batch dimensions and non-batch dimensions
            batch_shape = orig_shape[: len(orig_shape) + sum(non_batch_dims)]
            non_batch_shape = orig_shape[
                len(orig_shape) + sum(non_batch_dims) :
            ]

            # Reshape x to merge batch dimensions
            x_reshaped = x.reshape(-1, *non_batch_shape)

            # Apply the function
            y = func(x_reshaped)

            # Compute the new shape for the output
            new_shape = batch_shape + y.shape[1:]
            y_reshaped = y.reshape(new_shape)

            return y_reshaped

        return wrapper

    return decorator


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    def process_dimensions(x):
        return x.mean(dim=(-2, -1), keepdims=True)

    @batch_fn(-2, -1)
    def batch_process_dimensions(x):
        return process_dimensions(x)

    # Example usage
    x = torch.rand(1, 2, 3, 4, 5, 6)
    y = batch_process_dimensions(x)
    print(y.shape)

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/general/_batch_fn.py
"""
