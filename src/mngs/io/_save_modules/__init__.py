#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:05:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Save modules for mngs.io.save functionality

This package contains format-specific save handlers for various file types.
Each module provides a save_<format> function that handles saving objects
to that specific format.
"""

# Import save functions from individual modules
from ._csv import save_csv
from ._excel import save_excel
from ._numpy import save_npy, save_npz
from ._pickle import save_pickle, save_pickle_compressed
from ._joblib import save_joblib
from ._torch import save_torch
from ._json import save_json
from ._yaml import save_yaml
from ._hdf5 import save_hdf5
from ._matlab import save_matlab
from ._catboost import save_catboost
from ._text import save_text
from ._html import save_html
from ._image import save_image
from ._mp4 import save_mp4

# Import additional save utilities
from ._save_listed_dfs_as_csv import save_listed_dfs_as_csv
from ._save_listed_scalars_as_csv import save_listed_scalars_as_csv
from ._save_optuna_study_as_csv_and_pngs import save_optuna_study_as_csv_and_pngs

# Define what gets imported with "from mngs.io._save_modules import *"
__all__ = [
    "save_csv",
    "save_excel",
    "save_npy",
    "save_npz",
    "save_pickle",
    "save_pickle_compressed",
    "save_joblib",
    "save_torch",
    "save_json",
    "save_yaml",
    "save_hdf5",
    "save_matlab",
    "save_catboost",
    "save_text",
    "save_html",
    "save_image",
    "save_mp4",
    "save_listed_dfs_as_csv",
    "save_listed_scalars_as_csv",
    "save_optuna_study_as_csv_and_pngs",
]

# EOF