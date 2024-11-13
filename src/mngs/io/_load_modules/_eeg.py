#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:23 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_eeg.py

import os
import warnings

import mne

# # EEG data
# elif extension in [
#     ".vhdr",
#     ".vmrk",
#     ".edf",
#     ".bdf",
#     ".gdf",
#     ".cnt",
#     ".egi",
#     ".eeg",
#     ".set",
# ]:
#     obj = _load_eeg_data(lpath, **kwargs)


def _load_eeg_data(filename, **kwargs):
    """
    Load EEG data based on file extension and associated files using MNE-Python.

    This function supports various EEG file formats including BrainVision, EDF, BDF, GDF, CNT, EGI, and SET.
    It also handles special cases for .eeg files (BrainVision and Nihon Koden).

    Parameters:
    -----------
    filename : str
        The path to the EEG file to be loaded.
    **kwargs : dict
        Additional keyword arguments to be passed to the specific MNE loading function.

    Returns:
    --------
    raw : mne.io.Raw
        The loaded raw EEG data.

    Raises:
    -------
    ValueError
        If the file extension is not supported.

    Notes:
    ------
    This function uses MNE-Python to load the EEG data. It automatically detects the file format
    based on the file extension and uses the appropriate MNE function to load the data.
    """
    # Get the file extension
    extension = filename.split(".")[-1]

    allowed_extensions = [
        ".vhdr",
        ".vmrk",
        ".edf",
        ".bdf",
        ".gdf",
        ".cnt",
        ".egi",
        ".eeg",
        ".set",
    ]

    if extension not in allowed_extensions:
        raise ValueError(
            f"File must have one of these extensions: {', '.join(allowed_extensions)}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Load the data based on the file extension
        if extension in ["vhdr", "vmrk"]:
            # Load BrainVision data
            raw = mne.io.read_raw_brainvision(filename, preload=True, **kwargs)
        elif extension == "edf":
            # Load European data format
            raw = mne.io.read_raw_edf(filename, preload=True, **kwargs)
        elif extension == "bdf":
            # Load BioSemi data format
            raw = mne.io.read_raw_bdf(filename, preload=True, **kwargs)
        elif extension == "gdf":
            # Load Gen data format
            raw = mne.io.read_raw_gdf(filename, preload=True, **kwargs)
        elif extension == "cnt":
            # Load Neuroscan CNT data
            raw = mne.io.read_raw_cnt(filename, preload=True, **kwargs)
        elif extension == "egi":
            # Load EGI simple binary data
            raw = mne.io.read_raw_egi(filename, preload=True, **kwargs)
        elif extension == "set":
            # ???
            raw = mne.io.read_raw(filename, preload=True, **kwargs)
        elif extension == "eeg":
            is_BrainVision = any(
                os.path.isfile(filename.replace(".eeg", ext))
                for ext in [".vhdr", ".vmrk"]
            )
            is_NihonKoden = any(
                os.path.isfile(filename.replace(".eeg", ext))
                for ext in [".21e", ".pnt", ".log"]
            )

            # Brain Vision
            if is_BrainVision:
                filename_v = filename.replace(".eeg", ".vhdr")
                raw = mne.io.read_raw_brainvision(
                    filename_v, preload=True, **kwargs
                )
            # Nihon Koden
            if is_NihonKoden:
                # raw = mne.io.read_raw_nihon(filename, preload=True, **kwargs)
                raw = mne.io.read_raw(filename, preload=True, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        return raw


# EOF
