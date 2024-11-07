#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 18:58:37 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dsp/_listen.py

import sys
from typing import Tuple

import matplotlib.pyplot as plt
import mngs
import numpy as np
import sounddevice as sd
from scipy.signal import resample

"""
Functionality:
    - Provides audio playback functionality for multichannel signal arrays
    - Includes device selection and audio information display utilities
Input:
    - Multichannel signal arrays (numpy.ndarray)
    - Sampling frequency and channel selection
Output:
    - Audio playback through specified output device
Prerequisites:
    - PortAudio library (install with: sudo apt-get install portaudio19-dev)
    - sounddevice package
"""

"""Imports"""
"""Config"""
CONFIG = mngs.gen.load_configs()

"""Functions"""
def listen(
    signal_array: np.ndarray,
    sampling_freq: int,
    channels: Tuple[int, ...] = (0, 1),
    target_fs: int = 44_100,
) -> None:
    """
    Play selected channels of a multichannel signal array as audio.

    Example
    -------
    >>> signal = np.random.randn(1, 2, 1000)  # Random stereo signal
    >>> listen(signal, 16000, channels=(0, 1))

    Parameters
    ----------
    signal_array : np.ndarray
        Signal array of shape (batch_size, n_channels, sequence_length)
    sampling_freq : int
        Original sampling frequency of the signal
    channels : Tuple[int, ...]
        Tuple of channel indices to listen to
    target_fs : int
        Target sampling frequency for playback

    Returns
    -------
    None
    """
    if not isinstance(signal_array, np.ndarray):
        signal_array = np.array(signal_array)

    if len(signal_array.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {signal_array.shape}")

    if max(channels) >= signal_array.shape[1]:
        raise ValueError(f"Channel index {max(channels)} out of range (max: {signal_array.shape[1]-1})")

    selected_channels = signal_array[:, channels, :].mean(axis=1)
    audio_signal = selected_channels.mean(axis=0)

    if sampling_freq != target_fs:
        num_samples = int(round(len(audio_signal) * target_fs / sampling_freq))
        audio_signal = resample(audio_signal, num_samples)

    sd.play(audio_signal, target_fs)
    sd.wait()

def print_device_info() -> None:
    """
    Display information about the default audio output device.

    Example
    -------
    >>> print_device_info()
    Default Output Device Info:
    <device info details>
    """
    try:
        device_info = sd.query_devices(kind="output")
        print(f"Default Output Device Info: \n{device_info}")
    except sd.PortAudioError as err:
        print(f"Error querying audio devices: {err}")

def list_and_select_device() -> int:
    """
    List available audio devices and prompt user to select one.

    Example
    -------
    >>> device_id = list_and_select_device()
    Available audio devices:
    ...
    Enter the ID of the device you want to use:

    Returns
    -------
    int
        Selected device ID
    """
    try:
        print("Available audio devices:")
        devices = sd.query_devices()
        print(devices)
        device_id = int(input("Enter the ID of the device you want to use: "))
        if device_id not in range(len(devices)):
            raise ValueError(f"Invalid device ID: {device_id}")
        return device_id
    except (ValueError, sd.PortAudioError) as err:
        print(f"Error during device selection: {err}")
        return 0

if __name__ == "__main__":
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    signal, time_points, sampling_freq = mngs.dsp.demo_sig("chirp")

    device_id = list_and_select_device()
    sd.default.device = device_id

    listen(signal, sampling_freq)

    mngs.gen.close(CONFIG)

# EOF
