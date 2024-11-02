"""
This script does XYZ.
"""

import os
import sys

import matplotlib.pyplot as plt


# Imports
import numpy as np
import pandas as pd
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
CONFIG = mngs.gen.load_configs()

# Functions
def listen(x, fs, chs=(0, 1)):
    """
    Play selected channels of a multichannel signal array as audio.

    Parameters:
    - x (numpy.ndarray): Signal array of shape (batch_size, n_chs, seq_len).
    - fs (int): Sampling frequency of the signal.
    - chs (tuple of int): Tuple of channel indices to listen to.

    Returns:
    - None

    Memo:
        The PortAudio package is required.
        sudo yum update && sudo yum install -y portaudio portaudio-devel
    """
    COMMON_FS = 44100

    # Ensure the input channels are within the range of available channels
    if max(chs) >= x.shape[1]:
        raise ValueError("Channel index out of range")

    # Extract the desired channels (average if more than one)
    selected_channels = x[:, chs, :].mean(axis=1)

    # Flatten the batch dimension by averaging (or any other method as necessary)
    signal_to_play = selected_channels.mean(axis=0)

    # Resample the signal to the common sample rate if necessary
    if fs != COMMON_FS:
        from scipy.signal import resample

        num_samples = int(round(len(signal_to_play) * common_fs / fs))
        signal_to_play = resample(signal_to_play, num_samples)

    # Play the sound
    sd.play(signal_to_play, fs)
    sd.wait()  # Wait until the sound is finished playing


def print_device_info():
    device_info = sd.query_devices(kind="output")
    print(f"Default Output Device Info: \n{device_info}")


def list_and_select_device():
    print("Available audio devices:")
    print(sd.query_devices())
    device_id = int(input("Enter the ID of the device you want to use: "))
    return device_id


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    xx, tt, fs = mngs.dsp.demo_sig("chirp")

    device_id = list_and_select_device()
    sd.default.device = device_id

    listen(xx, common_fs)

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/dsp/_listen.py
"""
