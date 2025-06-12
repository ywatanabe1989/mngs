#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 16:49:44 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_save.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save.py"
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

# Import save functions from the new modular structure
from ._save_modules import (
    save_csv,
    save_excel,
    save_npy,
    save_npz,
    save_pickle,
    save_pickle_compressed,
    save_joblib,
    save_torch,
    save_json,
    save_yaml,
    save_hdf5,
    save_matlab,
    save_catboost,
    save_text,
    save_html,
    save_image,
    save_mp4,
    save_listed_dfs_as_csv,
    save_listed_scalars_as_csv,
    save_optuna_study_as_csv_and_pngs,
)


def _get_figure_with_data(obj):
    """
    Extract figure or axes object that may contain plotting data for CSV export.
    
    Parameters
    ----------
    obj : various matplotlib objects
        Could be Figure, Axes, FigWrapper, AxisWrapper, or other matplotlib objects
        
    Returns
    -------
    object or None
        Figure or axes object that has export_as_csv methods, or None if not found
    """
    import matplotlib.pyplot as plt
    import matplotlib.figure
    import matplotlib.axes
    
    # Check if object already has export methods (MNGS wrapped objects)
    if hasattr(obj, 'export_as_csv'):
        return obj
    
    # Handle matplotlib Figure objects
    if isinstance(obj, matplotlib.figure.Figure):
        # Get the current axes that might be wrapped with MNGS functionality
        current_ax = plt.gca()
        if hasattr(current_ax, 'export_as_csv'):
            return current_ax
        
        # Check all axes in the figure
        for ax in obj.axes:
            if hasattr(ax, 'export_as_csv'):
                return ax
                
        return None
    
    # Handle matplotlib Axes objects  
    if isinstance(obj, matplotlib.axes.Axes):
        if hasattr(obj, 'export_as_csv'):
            return obj
        return None
    
    # Handle FigWrapper or similar MNGS objects
    if hasattr(obj, 'figure') and hasattr(obj.figure, 'axes'):
        # Check if the wrapper itself has export methods
        if hasattr(obj, 'export_as_csv'):
            return obj
            
        # Check the underlying figure's axes
        for ax in obj.figure.axes:
            if hasattr(ax, 'export_as_csv'):
                return ax
                
        return None
    
    # Handle AxisWrapper or similar MNGS objects
    if hasattr(obj, '_axis_mpl') or hasattr(obj, '_ax'):
        if hasattr(obj, 'export_as_csv'):
            return obj
        return None
    
    # Try to get the current figure and its axes as fallback
    try:
        current_fig = plt.gcf()
        current_ax = plt.gca() 
        
        if hasattr(current_ax, 'export_as_csv'):
            return current_ax
        elif hasattr(current_fig, 'export_as_csv'):
            return current_fig
            
        # Check all axes in current figure
        for ax in current_fig.axes:
            if hasattr(ax, 'export_as_csv'):
                return ax
                
    except:
        pass
    
    return None


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

        # f-expression handling - safely parse f-strings
        if specified_path.startswith('f"') or specified_path.startswith("f'"):
            # Remove the f prefix and quotes
            path_content = specified_path[2:-1]
            
            # Get the caller's frame to access their local variables
            frame = inspect.currentframe().f_back
            try:
                # Use string formatting with the caller's locals and globals
                # This is much safer than eval() as it only does string substitution
                import re
                # Find all {variable} patterns
                variables = re.findall(r'\{([^}]+)\}', path_content)
                format_dict = {}
                for var in variables:
                    # Only allow simple variable names, not arbitrary expressions
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                        if var in frame.f_locals:
                            format_dict[var] = frame.f_locals[var]
                        elif var in frame.f_globals:
                            format_dict[var] = frame.f_globals[var]
                    else:
                        raise ValueError(f"Invalid variable name in f-string: {var}")
                
                # Use str.format() which is safe
                specified_path = path_content.format(**format_dict)
            finally:
                del frame  # Avoid reference cycles

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
        # Skip deletion for CSV files to allow caching to work
        for path in [spath_final, spath_cwd]:
            if not path.endswith('.csv'):
                sh(f"rm -f {path}", verbose=False)

        if dry_run:
            print(color_text(f"\n(dry run) Saved to: {spath_final}", c="yellow"))
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
    # Get file extension
    ext = _os.path.splitext(spath)[1].lower()
    
    # Try dispatch dictionary first for O(1) lookup
    if ext in _FILE_HANDLERS:
        # Check if handler needs special parameters
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.svc']:
            _FILE_HANDLERS[ext](obj, spath, no_csv=no_csv, symlink_from_cwd=symlink_from_cwd, dry_run=dry_run, **kwargs)
        else:
            _FILE_HANDLERS[ext](obj, spath, **kwargs)
    # csv - special case as it doesn't have a dot prefix in dispatch
    elif spath.endswith(".csv"):
        save_csv(obj, spath, **kwargs)
    # Check for special extension cases not in dispatch
    elif spath.endswith(".pkl.gz"):
        save_pickle_compressed(obj, spath, **kwargs)
    else:
        warnings.warn(f"Unsupported file format. {spath} was not saved.")

    if verbose:
        if _os.path.exists(spath):
            file_size = getsize(spath)
            file_size = readable_bytes(file_size)
            print(color_text(f"\nSaved to: {spath} ({file_size})", c="yellow"))


def _handle_image_with_csv(obj, spath, no_csv=False, symlink_from_cwd=False, dry_run=False, **kwargs):
    """Handle image file saving with optional CSV export."""
    save_image(obj, spath, **kwargs)
    
    if not no_csv:
        ext = _os.path.splitext(spath)[1].lower()
        ext_wo_dot = ext.replace(".", "")
        
        try:
            # Get the figure object that may contain plot data
            fig_obj = _get_figure_with_data(obj)
            
            if fig_obj is not None:
                # Save regular CSV if export method exists
                if hasattr(fig_obj, 'export_as_csv'):
                    csv_data = fig_obj.export_as_csv()
                    if csv_data is not None and not csv_data.empty:
                        save(
                            csv_data,
                            spath.replace(ext_wo_dot, "csv"),
                            symlink_from_cwd=symlink_from_cwd,
                            dry_run=dry_run,
                            no_csv=True,
                            **kwargs,
                        )
                
                # Save SigmaPlot CSV if method exists
                if hasattr(fig_obj, 'export_as_csv_for_sigmaplot'):
                    sigmaplot_data = fig_obj.export_as_csv_for_sigmaplot()
                    if sigmaplot_data is not None and not sigmaplot_data.empty:
                        save(
                            sigmaplot_data,
                            spath.replace(ext_wo_dot, "csv").replace(".csv", "_for_sigmaplot.csv"),
                            symlink_from_cwd=symlink_from_cwd,
                            dry_run=dry_run,
                            no_csv=True,
                            **kwargs,
                        )
        except Exception:
            pass


# Dispatch dictionary for O(1) file format lookup
_FILE_HANDLERS = {
    # Excel formats
    '.xlsx': save_excel,
    '.xls': save_excel,
    
    # NumPy formats
    '.npy': save_npy,
    '.npz': save_npz,
    
    # Pickle formats
    '.pkl': save_pickle,
    '.pickle': save_pickle,
    '.pkl.gz': save_pickle_compressed,
    
    # Other binary formats
    '.joblib': save_joblib,
    '.pth': save_torch,
    '.pt': save_torch,
    '.mat': save_matlab,
    '.cbm': save_catboost,
    
    # Text formats
    '.json': save_json,
    '.yaml': save_yaml,
    '.yml': save_yaml,
    '.txt': save_text,
    '.md': save_text,
    '.py': save_text,
    '.css': save_text,
    '.js': save_text,
    
    # Data formats
    '.html': save_html,
    '.hdf5': save_hdf5,
    '.h5': save_hdf5,
    
    # Media formats
    '.mp4': save_mp4,
    '.png': _handle_image_with_csv,
    '.jpg': _handle_image_with_csv,
    '.jpeg': _handle_image_with_csv,
    '.gif': _handle_image_with_csv,
    '.tiff': _handle_image_with_csv,
    '.tif': _handle_image_with_csv,
    '.svc': _handle_image_with_csv,
}


# EOF
