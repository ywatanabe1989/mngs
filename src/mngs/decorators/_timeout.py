#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-23 19:11:33"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""
import time
from multiprocessing import Process, Queue


def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def queue_wrapper(queue, args, kwargs):
                result = func(*args, **kwargs)
                queue.put(result)

            queue = Queue()
            args_for_process = (queue, args, kwargs)
            process = Process(target=queue_wrapper, args=args_for_process)
            process.start()
            process.join(timeout=seconds)

            if process.is_alive():
                process.terminate()
                raise TimeoutError(error_message)
            else:
                return queue.get()

        return wrapper

    return decorator


def main():
    # Example usage
    @timeout(seconds=3, error_message="Function call timed out")
    def long_running_function(x):
        time.sleep(4)  # Simulate a long-running operation
        return x

    try:
        result = long_running_function(10)
        print(f"Result: {result}")
    except TimeoutError as e:
        print(e)


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
