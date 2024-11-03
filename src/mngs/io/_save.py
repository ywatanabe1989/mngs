#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 06:27:54 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_save.py

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
import json
import logging
import os
import pickle
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import plotly
import scipy
import torch
from ruamel.yaml import YAML

from .._sh import sh
from ..path._clean import clean
from ..path._getsize import getsize
from ..path._split import split
from ..str._color_text import color_text
from ..types._is_listed_X import is_listed_X
from ._save_text import _save_text
from ._save_listed_scalars_as_csv import _save_listed_scalars_as_csv
from ._save_listed_dfs_as_csv import _save_listed_dfs_as_csv
from ._save_image import _save_image

def save(
    obj: Any,
    sfname_or_spath: str,
    makedirs: bool = True,
    verbose: bool = True,
    from_cwd: bool = False,
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
    sfname_or_spath : str
        The file name or path where the object should be saved. The file extension determines the format.
    makedirs : bool, optional
        If True, create the directory path if it does not exist. Default is True.
    verbose : bool, optional
        If True, print a message upon successful saving. Default is True.
    from_cwd : bool, optional
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
        # Determines the save directory from the script.
        # When save() is called in a /path/to/script.py, data will be saved under /path/to/ directoy.
        # On the other hand, when save() is called in an ipython environment, data will be saved in /tmp/USERNAME/
        # This process should be in this function for the intended behavior of inspect.
        ########################################
        spath, sfname = None, None

        if sfname_or_spath.startswith('f"'):
            sfname_or_spath = eval(sfname_or_spath)

        if sfname_or_spath.startswith("/"):
            spath = sfname_or_spath

        else:
            fpath = inspect.stack()[1].filename

            if ("ipython" in fpath) or ("<stdin>" in fpath):
                fpath = f'/tmp/{os.getenv("USER")}.py'

            fdir, fname, _ = split(fpath)
            spath = fdir + fname + "/" + sfname_or_spath

        # Corrects the spath
        spath = clean(spath)
        ########################################

        # Potential path to _symlink
        spath_cwd = os.getcwd() + "/" + sfname_or_spath
        spath_cwd = clean(spath_cwd)

        # Removes spath and spath_cwd to prevent potential circular links
        for path in [spath, spath_cwd]:
            sh(f"rm -f {path}", verbose=False)

        if dry_run:
            print(color_text(f"\n(dry run) Saved to: {spath}", c="yellow"))
            return

        # Makes directory
        if makedirs:
            os.makedirs(os.path.dirname(spath), exist_ok=True)

        _save(
            obj,
            spath,
            verbose=verbose,
            from_cwd=from_cwd,
            dry_run=dry_run,
            no_csv=no_csv,
            **kwargs,
        )
        _symlink(spath, spath_cwd, from_cwd, verbose)

    except Exception as e:
        logging.error(
            f"Error occurred while saving: {str(e)}"
            f"Debug: Initial fpath = {inspect.stack()[1].filename}"
            # f"Debug: Final fpath = {fpath}"
            # f"Debug: fdir = {fdir}, fname = {fname}"
            f"Debug: Final spath = {spath}"
        )


def _symlink(spath, spath_cwd, from_cwd, verbose):
    if from_cwd and (spath != spath_cwd):
        os.makedirs(os.path.dirname(spath_cwd), exist_ok=True)
        sh(f"rm -f {spath_cwd}", verbose=False)
        sh(f"ln -sfr {spath} {spath_cwd}", verbose=False)
        if verbose:
            print(
                color_text(f"\n(_Symlinked to: {spath_cwd})", "yellow")
            )


def _save(
    obj,
    spath,
    verbose=True,
    from_cwd=False,
    dry_run=False,
    no_csv=False,
    **kwargs,
):
    # csv
    if spath.endswith(".csv"):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            obj.to_csv(spath, **kwargs)

        if is_listed_X(obj, [int, float]):
            _save_listed_scalars_as_csv(
                obj,
                spath,
                **kwargs,
            )
        if is_listed_X(obj, pd.DataFrame):
            _save_listed_dfs_as_csv(obj, spath, **kwargs)

    # numpy
    elif spath.endswith(".npy"):
        np.save(spath, obj)

    # numpy npz
    elif spath.endswith(".npz"):
        if isinstance(obj, dict):
            np.savez_compressed(spath, **obj)
        elif isinstance(obj, (list, tuple)) and all(
            isinstance(x, np.ndarray) for x in obj
        ):
            obj = {str(ii): obj[ii] for ii in range(len(obj))}
            np.savez_compressed(spath, **obj)
        else:
            raise ValueError(
                "For .npz files, obj must be a dict of arrays or a list/tuple of arrays."
            )
    # pkl
    elif spath.endswith(".pkl"):
        with open(spath, "wb") as s:
            pickle.dump(obj, s)

    # joblib
    elif spath.endswith(".joblib"):
        with open(spath, "wb") as s:
            joblib.dump(obj, s, compress=3)

    # html
    elif spath.endswith(".html"):
        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_html(file=spath)

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
                ".svc",
            ]
        ]
    ):
        _save_image(obj, spath, **kwargs)
        ext = os.path.splitext(spath)[1].lower()
        try:
            if not no_csv:
                ext_wo_dot = ext.replace(".", "")
                save(
                    obj.to_sigma(),
                    spath.replace(ext_wo_dot, "csv"),
                    from_cwd=from_cwd,
                    dry_run=dry_run,
                    **kwargs,
                )
        except Exception as e:
            print(e)

    # mp4
    elif spath.endswith(".mp4"):
        obj.save(
            spath, writer="ffmpeg", **kwargs
        )
        del obj
        # _mk_mp4(obj, spath)  # obj is matplotlib.pyplot.figure object
        # del obj

    # yaml
    elif spath.endswith(".yaml"):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(
            mapping=4, sequence=4, offset=4
        )

        with open(spath, "w") as f:
            yaml.dump(obj, f)

    # json
    elif spath.endswith(".json"):
        with open(spath, "w") as f:
            json.dump(obj, f, indent=4)

    # hdf5
    elif spath.endswith(".hdf5"):
        name_list, obj_list = []
        for k, v in obj.items():
            name_list.append(k)
            obj_list.append(v)
        with h5py.File(spath, "w") as hf:
            for name, obj in zip(name_list, obj_list):
                hf.create_dataset(name, data=obj)
    # pth
    elif spath.endswith(".pth"):
        torch.save(obj, spath)

    # mat
    elif spath.endswith(".mat"):
        scipy.io.savemat(spath, obj)

    # catboost model
    elif spath.endswith(".cbm"):
        obj.save_model(spath)

    # Text
    elif any(
        spath.endswith(ext)
        for ext in [".txt", ".md", ".py", ".html", ".css", ".js"]
    ):
        _save_text(obj, spath)

    else:
        raise ValueError(f"Unsupported file format. {spath} was not saved.")

    if verbose:
        if os.path.exists(spath):
            file_size = getsize(spath)
            print(
                color_text(f"\nSaved to: {spath} ({file_size})", c="yellow")
            )


# EOF


# EOF
