#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 16:49:44 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_save.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/io/_save.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_save.py"

"""
1. Functionality:
   - Provides utilities for saving various data types to different file formats.
2. Input:
   - Objects to be saved (e.g., NumPy arrays, PyTorch tensors, Pandas DataFrames, etc.)
   - File path or name where the object should be saved
3. Output:
   - Saved files in various formats (e.g., CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, CBM)
4. Prerequisites:
   - Python 3.x
   - Required libraries: numpy, pandas, torch, matplotlib, plotly, h5py, joblib, PIL, ruamel.yaml
"""

"""Imports"""
import inspect
import logging
import os as _os
from typing import Any

from .._sh import sh
from ..path._clean import clean
from ..path._getsize import getsize
from ..str._clean_path import clean_path
from ..str._color_text import color_text
from ..str._readable_bytes import readable_bytes
from ._save_modules._catboost import _save_catboost
# Import individual save modules
from ._save_modules._csv import _save_csv
from ._save_modules._hdf5 import _save_hdf5
from ._save_modules._image import _save_image
from ._save_modules._joblib import _save_joblib
from ._save_modules._json import _save_json
from ._save_modules._matlab import _save_matlab
from ._save_modules._numpy import _save_npy, _save_npz
from ._save_modules._pickle import _save_pickle, _save_pickle_gz
from ._save_modules._plotly import _save_plotly_html
from ._save_modules._text import _save_text
from ._save_modules._torch import _save_torch
from ._save_modules._yaml import _save_yaml


def save(
    obj: Any,
    specified_path: str,
    makedirs: bool = True,
    verbose: bool = True,
    symlink_from_cwd: bool = False,
    dry_run: bool = False,
    no_csv: bool = False,
    **kwargs,
) -> None:
    """
    Save an object to a file with the specified format.

    Parameters
    ----------
    obj : Any
        The object to be saved. Can be a NumPy array, PyTorch tensor, Pandas DataFrame, or any serializable object.
    specified_path : str
        The file name or path where the object should be saved. The file extension determines the format.
    makedirs : bool, optional
        If True, create the directory path if it does not exist. Default is True.
    verbose : bool, optional
        If True, print a message upon successful saving. Default is True.
    symlink_from_cwd : bool, optional
        If True, create a _symlink from the current working directory. Default is False.
    dry_run : bool, optional
        If True, simulate the saving process without actually writing files. Default is False.
    **kwargs
        Additional keyword arguments to pass to the underlying save function of the specific format.

    Returns
    -------
    None

    Notes
    -----
    Supported formats include CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, and CBM.
    The function dynamically selects the appropriate saving mechanism based on the file extension.

    Examples
    --------
    >>> import mngs
    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> import matplotlib.pyplot as plt

    >>> # Save NumPy array
    >>> arr = np.array([1, 2, 3])
    >>> mngs.io.save(arr, "data.npy")

    >>> # Save Pandas DataFrame
    >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    >>> mngs.io.save(df, "data.csv")

    >>> # Save PyTorch tensor
    >>> tensor = torch.tensor([1, 2, 3])
    >>> mngs.io.save(tensor, "model.pth")

    >>> # Save dictionary
    >>> data_dict = {"a": 1, "b": 2, "c": [3, 4, 5]}
    >>> mngs.io.save(data_dict, "data.pkl")

    >>> # Save matplotlib figure
    >>> plt.figure()
    >>> plt.plot(np.array([1, 2, 3]))
    >>> mngs.io.save(plt, "plot.png")

    >>> # Save as YAML
    >>> mngs.io.save(data_dict, "config.yaml")

    >>> # Save as JSON
    >>> mngs.io.save(data_dict, "data.json")
    """
    try:
        ########################################
        # DO NOT MODIFY THIS SECTION
        ########################################
        #
        # Determine saving directory from the script.
        #
        # When called in /path/to/script.py,
        # data will be saved under `/path/to/script.py_out/`
        #
        # On the other hand, when called in ipython environment,
        # data will be saved under `/tmp/{_os.getenv("USER")/`
        #
        ########################################
        spath, sfname = None, None

        # f-expression handling
        if specified_path.startswith('f"'):
            specified_path = eval(specified_path)

        # When full path
        if specified_path.startswith("/"):
            spath = specified_path

        # When relative path
        else:
            script_path = inspect.stack()[1].filename

            # Fake path if in ipython
            if ("ipython" in script_path) or ("<stdin>" in script_path):
                script_path = f'/tmp/{_os.getenv("USER")}'

            sdir = clean_path(_os.path.splitext(script_path)[0] + "_out")
            spath = _os.path.join(sdir, specified_path)

        # Sanitization
        spath_final = clean(spath)
        ########################################

        # Potential path to _symlink
        spath_cwd = _os.getcwd() + "/" + specified_path
        spath_cwd = clean(spath_cwd)

        # Removes spath and spath_cwd to prevent potential circular links
        for path in [spath_final, spath_cwd]:
            sh(f"rm -f {path}", verbose=False)

        if dry_run:
            print(
                color_text(f"\n(dry run) Saved to: {spath_final}", c="yellow")
            )
            return

        # Ensure directory exists
        if makedirs:
            _os.makedirs(_os.path.dirname(spath_final), exist_ok=True)

        # Main
        _save(
            obj,
            spath_final,
            verbose=verbose,
            symlink_from_cwd=symlink_from_cwd,
            dry_run=dry_run,
            no_csv=no_csv,
            **kwargs,
        )

        # Symbolic link
        _symlink(spath, spath_cwd, symlink_from_cwd, verbose)

    except Exception as e:
        logging.error(
            f"Error occurred while saving: {str(e)}"
            f"Debug: Initial script_path = {inspect.stack()[1].filename}"
            f"Debug: Final spath = {spath}"
        )


def _symlink(spath, spath_cwd, symlink_from_cwd, verbose):
    """Create a symbolic link from the current working directory."""
    if symlink_from_cwd and (spath != spath_cwd):
        _os.makedirs(_os.path.dirname(spath_cwd), exist_ok=True)
        sh(f"rm -f {spath_cwd}", verbose=False)
        sh(f"ln -sfr {spath} {spath_cwd}", verbose=False)
        if verbose:
            print(color_text(f"\n(Symlinked to: {spath_cwd})", "yellow"))


def _save(
    obj,
    spath,
    verbose=True,
    symlink_from_cwd=False,
    dry_run=False,
    no_csv=False,
    **kwargs,
):
    """
    Save an object based on the file extension.

    This function dispatches to the appropriate specialized save function
    based on the file extension of the provided path.
    """
    # Dispatch based on file extension
    if spath.endswith(".csv"):
        _save_csv(obj, spath, **kwargs)

    # numpy
    elif spath.endswith(".npy"):
        _save_npy(obj, spath)

    # numpy npz
    elif spath.endswith(".npz"):
        _save_npz(obj, spath)

    # pkl
    elif spath.endswith(".pkl"):
        _save_pickle(obj, spath)

    # pkl.gz
    elif spath.endswith(".pkl.gz"):
        _save_pickle_gz(obj, spath)

    # joblib
    elif spath.endswith(".joblib"):
        _save_joblib(obj, spath)

    # html
    elif spath.endswith(".html"):
        # plotly
        import plotly

        if isinstance(obj, plotly.graph_objs.Figure):
            _save_plotly_html(obj, spath)

    # image ----------------------------------------
    elif any(
        [
            spath.endswith(image_ext)
            for image_ext in [
                ".png",
                ".tiff",
                ".tif",
                ".jpeg",
                ".jpg",
                ".svg",
            ]
        ]
    ):
        _save_image(obj, spath, **kwargs)

        # Handles CSV exporting
        try:
            if not no_csv:
                _, im_ext = _os.path.splitext(spath)
                im_ext_wo_dot = im_ext.replace(".", "")
                spath_csv = spath.replace(im_ext_wo_dot, "csv")
                save(
                    obj.export_as_csv(),
                    spath_csv,
                    symlink_from_cwd=symlink_from_cwd,
                    dry_run=dry_run,
                    **kwargs,
                )
        except Exception as e:
            warnings.warn(f"CSV note saved to: {spath_csv}\n{str(e)}")
            # __import__("ipdb").set_trace()

    # mp4
    elif spath.endswith(".mp4"):
        obj.save(spath, writer="ffmpeg", **kwargs)
        del obj

    # yaml
    elif spath.endswith(".yaml"):
        _save_yaml(obj, spath)

    # json
    elif spath.endswith(".json"):
        _save_json(obj, spath)

    # hdf5
    elif spath.endswith(".hdf5"):
        _save_hdf5(obj, spath)

    # pth
    elif spath.endswith(".pth") or spath.endswith(".pt"):
        _save_torch(obj, spath, **kwargs)

    # mat
    elif spath.endswith(".mat"):
        _save_matlab(obj, spath)

    # catboost model
    elif spath.endswith(".cbm"):
        _save_catboost(obj, spath)

    # Text
    elif any(
        spath.endswith(ext)
        for ext in [".txt", ".md", ".py", ".html", ".css", ".js"]
    ):
        _save_text(obj, spath)

    else:
        warnings.warn(f"Unsupported file format. {spath} was not saved.")

    if verbose:
        if _os.path.exists(spath):
            file_size = getsize(spath)
            file_size = readable_bytes(file_size)
            print(color_text(f"\nSaved to: {spath} ({file_size})", c="yellow"))

# EOF
