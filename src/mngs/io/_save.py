#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 14:38:43 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/io/_save.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/io/_save.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

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
import gzip
import inspect
import json
import logging
import os as _os
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
from ..str._clean_path import clean_path
from ..str._color_text import color_text
from ..str._readable_bytes import readable_bytes
from ._save_image import _save_image
from ._save_text import _save_text


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
            # f"Debug: Final script_path = {script_path}"
            # f"Debug: fdir = {fdir}, fname = {fname}"
            f"Debug: Final spath = {spath}"
        )


def _symlink(spath, spath_cwd, symlink_from_cwd, verbose):
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

    # csv
    if spath.endswith(".csv"):
        _save_csv(obj, spath, **kwargs)

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

    # pkl.gz
    elif spath.endswith(".pkl.gz"):
        with gzip.open(spath, "wb") as f:
            pickle.dump(obj, f)

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
        ext = _os.path.splitext(spath)[1].lower()
        try:
            if not no_csv:
                ext_wo_dot = ext.replace(".", "")
                save(
                    obj.export_as_csv(),
                    spath.replace(ext_wo_dot, "csv"),
                    symlink_from_cwd=symlink_from_cwd,
                    dry_run=dry_run,
                    **kwargs,
                )
        except Exception as e:
            pass
            # print(e)

    # mp4
    elif spath.endswith(".mp4"):
        obj.save(spath, writer="ffmpeg", **kwargs)
        del obj
        # _mk_mp4(obj, spath)  # obj is matplotlib.pyplot.figure object
        # del obj

    # yaml
    elif spath.endswith(".yaml"):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=4, sequence=4, offset=4)

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
        if _os.path.exists(spath):
            file_size = getsize(spath)
            file_size = readable_bytes(file_size)
            print(color_text(f"\nSaved to: {spath} ({file_size})", c="yellow"))


# def _save_csv(obj, spath: str, **kwargs) -> None:
#     """Handle various input types for CSV saving."""
#     if isinstance(obj, (pd.Series, pd.DataFrame)):
#         obj.to_csv(spath, **kwargs)
#     elif isinstance(obj, np.ndarray):
#         pd.DataFrame(obj).to_csv(spath, **kwargs)
#     elif isinstance(obj, (int, float)):
#         pd.DataFrame([obj]).to_csv(spath, index=False, **kwargs)
#     elif isinstance(obj, (list, tuple)):
#         if all(isinstance(x, (int, float)) for x in obj):
#             pd.DataFrame(obj).to_csv(spath, index=False, **kwargs)
#         elif all(isinstance(x, pd.DataFrame) for x in obj):
#             pd.concat(obj).to_csv(spath, **kwargs)
#         else:
#             pd.DataFrame({"data": obj}).to_csv(spath, index=False, **kwargs)
#     elif isinstance(obj, dict):
#         pd.DataFrame.from_dict(obj).to_csv(spath, **kwargs)
#     else:
#         try:
#             pd.DataFrame({"data": [obj]}).to_csv(spath, index=False, **kwargs)
#         except:
#             raise ValueError(f"Unable to save type {type(obj)} as CSV")


def _save_csv(obj, spath: str, **kwargs) -> None:
    """Handle various input types for CSV saving."""
    # Check if path already exists
    if os.path.exists(spath):
        # Calculate hash of new data
        data_hash = None

        # Process based on type
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            data_hash = hash(obj.to_string())
        elif isinstance(obj, np.ndarray):
            data_hash = hash(pd.DataFrame(obj).to_string())
        else:
            # For other types, create a string representation and hash it
            try:
                data_str = str(obj)
                data_hash = hash(data_str)
            except:
                # If we can't hash it, proceed with saving
                pass

        # Compare with existing file if hash calculation was successful
        if data_hash is not None:
            try:
                existing_df = pd.read_csv(spath)
                existing_hash = hash(existing_df.to_string())

                # Skip if hashes match
                if existing_hash == data_hash:
                    return
            except:
                # If reading fails, proceed with saving
                pass

    # Save the file based on type
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj.to_csv(spath, **kwargs)
    elif isinstance(obj, np.ndarray):
        pd.DataFrame(obj).to_csv(spath, **kwargs)
    elif isinstance(obj, (int, float)):
        pd.DataFrame([obj]).to_csv(spath, index=False, **kwargs)
    elif isinstance(obj, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in obj):
            pd.DataFrame(obj).to_csv(spath, index=False, **kwargs)
        elif all(isinstance(x, pd.DataFrame) for x in obj):
            pd.concat(obj).to_csv(spath, **kwargs)
        else:
            pd.DataFrame({"data": obj}).to_csv(spath, index=False, **kwargs)
    elif isinstance(obj, dict):
        pd.DataFrame.from_dict(obj).to_csv(spath, **kwargs)
    else:
        try:
            pd.DataFrame({"data": [obj]}).to_csv(spath, index=False, **kwargs)
        except:
            raise ValueError(f"Unable to save type {type(obj)} as CSV")

# EOF