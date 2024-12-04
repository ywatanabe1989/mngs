# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:56:27 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_eeg.py
# 
# import os
# import warnings
# from typing import Any
# 
# import mne
# 
# 
# def _load_eeg_data(path: str, **kwargs) -> Any:
#     """
#     Load EEG data based on file extension and associated files using MNE-Python.
# 
#     This function supports various EEG file formats including BrainVision, EDF, BDF, GDF, CNT, EGI, and SET.
#     It also handles special cases for .eeg files (BrainVision and Nihon Koden).
# 
#     Parameters:
#     -----------
#     lpath : str
#         The path to the EEG file to be loaded.
#     **kwargs : dict
#         Additional keyword arguments to be passed to the specific MNE loading function.
# 
#     Returns:
#     --------
#     raw : mne.io.Raw
#         The loaded raw EEG data.
# 
#     Raises:
#     -------
#     ValueError
#         If the file extension is not supported.
# 
#     Notes:
#     ------
#     This function uses MNE-Python to load the EEG data. It automatically detects the file format
#     based on the file extension and uses the appropriate MNE function to load the data.
#     """
#     # Get the file extension
#     extension = lpath.split(".")[-1]
# 
#     allowed_extensions = [
#         ".vhdr",
#         ".vmrk",
#         ".edf",
#         ".bdf",
#         ".gdf",
#         ".cnt",
#         ".egi",
#         ".eeg",
#         ".set",
#     ]
# 
#     if extension not in allowed_extensions:
#         raise ValueError(
#             f"File must have one of these extensions: {', '.join(allowed_extensions)}"
#         )
# 
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
# 
#         # Load the data based on the file extension
#         if extension in ["vhdr", "vmrk"]:
#             # Load BrainVision data
#             raw = mne.io.read_raw_brainvision(lpath, preload=True, **kwargs)
#         elif extension == "edf":
#             # Load European data format
#             raw = mne.io.read_raw_edf(lpath, preload=True, **kwargs)
#         elif extension == "bdf":
#             # Load BioSemi data format
#             raw = mne.io.read_raw_bdf(lpath, preload=True, **kwargs)
#         elif extension == "gdf":
#             # Load Gen data format
#             raw = mne.io.read_raw_gdf(lpath, preload=True, **kwargs)
#         elif extension == "cnt":
#             # Load Neuroscan CNT data
#             raw = mne.io.read_raw_cnt(lpath, preload=True, **kwargs)
#         elif extension == "egi":
#             # Load EGI simple binary data
#             raw = mne.io.read_raw_egi(lpath, preload=True, **kwargs)
#         elif extension == "set":
#             # ???
#             raw = mne.io.read_raw(lpath, preload=True, **kwargs)
#         elif extension == "eeg":
#             is_BrainVision = any(
#                 os.path.isfile(lpath.replace(".eeg", ext))
#                 for ext in [".vhdr", ".vmrk"]
#             )
#             is_NihonKoden = any(
#                 os.path.isfile(lpath.replace(".eeg", ext))
#                 for ext in [".21e", ".pnt", ".log"]
#             )
# 
#             # Brain Vision
#             if is_BrainVision:
#                 lpath_v = lpath.replace(".eeg", ".vhdr")
#                 raw = mne.io.read_raw_brainvision(
#                     lpath_v, preload=True, **kwargs
#                 )
#             # Nihon Koden
#             if is_NihonKoden:
#                 # raw = mne.io.read_raw_nihon(lpath, preload=True, **kwargs)
#                 raw = mne.io.read_raw(lpath, preload=True, **kwargs)
#         else:
#             raise ValueError(f"Unsupported file extension: {extension}")
# 
#         return raw
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..io._load_modules._eeg import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
