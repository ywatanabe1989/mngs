#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 16:53:02 (ywatanabe)"

import numpy as np
import torch


def wrap_to_numpy(func):
    def wrapper(*args, **kwargs):
        # Convert all tensor arguments to numpy arrays
        new_args = [
            arg.numpy() if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        new_kwargs = {
            k: v.numpy() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        result = func(*new_args, **new_kwargs)

        return result

    return wrapper


@wrap_to_numpy
def process_with_numpy(x):
    # Example function that expects numpy arrays
    print("Processing with NumPy:", type(x))
    return np.mean(x)


if __name__ == "__main__":
    x_torch = torch.rand(10)
    process_with_numpy(
        x_torch
    )  # This will now accept a torch tensor and convert it to numpy
