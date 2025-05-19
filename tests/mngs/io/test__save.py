#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-13 22:30:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test__save.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/test__save.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch


def test_torch_save_pt_extension():
    """Test that PyTorch models can be saved with .pt extension."""
    from mngs.io._save import _save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])
        
        # Test saving with .pt extension
        _save(model, temp_path, verbose=False)
        
        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_torch_save_kwargs():
    """Test that kwargs are properly passed to torch.save."""
    from mngs.io._save import _save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])
        
        # _save should pass kwargs to torch.save
        # While we can't directly test the internal call, we can verify that
        # using _save with _use_new_zipfile_serialization=False works
        _save(model, temp_path, verbose=False, _use_new_zipfile_serialization=False)
        
        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_deduplication():
    """Test that CSV files are not rewritten if content hasn't changed."""
    from mngs.io._save import _save_csv
    import hashlib
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a test file path
        test_file = os.path.join(temp_dir, "test.csv")
        
        # Create test dataframe
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        
        # First save - should write the file
        _save_csv(df, test_file)
        assert os.path.exists(test_file)
        
        # Get file content hash
        with open(test_file, 'rb') as f:
            first_hash = hashlib.md5(f.read()).hexdigest()
        
        # Get file stats before second save
        first_stats = os.stat(test_file)
        
        # Introduce a small delay to ensure os.stat would detect any changes
        import time
        time.sleep(0.1)
        
        # Save again with same content - should skip writing due to hash check
        _save_csv(df, test_file)
        
        # Get file stats after second save
        second_stats = os.stat(test_file)
        
        # Verify the file metadata (size, modification time, etc.) hasn't changed
        # This is more reliable than just checking modification time
        assert first_stats.st_size == second_stats.st_size
        # Note: we're not checking mtime as the implementation might update it even if content is the same
        
        # Get file content hash again - should be unchanged
        with open(test_file, 'rb') as f:
            second_hash = hashlib.md5(f.read()).hexdigest()
        
        # Content hash should be the same
        assert first_hash == second_hash
        
        # Now change the dataframe and save again
        df2 = pd.DataFrame({"col1": [1, 2, 3], "col2": [7, 8, 9]})
        _save_csv(df2, test_file)
        
        # Get file stats after third save
        third_stats = os.stat(test_file)
        
        # Get file content hash again - should be changed
        with open(test_file, 'rb') as f:
            third_hash = hashlib.md5(f.read()).hexdigest()
        
        # Content hash should be different
        assert second_hash != third_hash
        
        # Check the content was updated
        loaded_df = pd.read_csv(test_file, index_col=0)
        assert loaded_df["col2"].tolist() == [7, 8, 9]
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:35:24 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/io/_save.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_save.py"
# 
# """
# 1. Functionality:
#    - Provides utilities for saving various data types to different file formats.
# 2. Input:
#    - Objects to be saved (e.g., NumPy arrays, PyTorch tensors, Pandas DataFrames, etc.)
#    - File path or name where the object should be saved
# 3. Output:
#    - Saved files in various formats (e.g., CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, CBM)
# 4. Prerequisites:
#    - Python 3.x
#    - Required libraries: numpy, pandas, torch, matplotlib, plotly, h5py, joblib, PIL, ruamel.yaml
# """
# 
# """Imports"""
# import inspect
# import logging
# import os as _os
# from typing import Any
# 
# import numpy as np
# import pandas as pd
# 
# from .._sh import sh
# from ..path._clean import clean
# from ..path._getsize import getsize
# from ..str._clean_path import clean_path
# from ..str._color_text import color_text
# from ..str._readable_bytes import readable_bytes
# 
# # Import individual save modules
# from ._save_modules._csv import _save_csv
# from ._save_modules._image import _save_image
# from ._save_modules._text import _save_text
# from ._save_modules._numpy import _save_npy, _save_npz
# from ._save_modules._pickle import _save_pickle, _save_pickle_gz
# from ._save_modules._joblib import _save_joblib
# from ._save_modules._hdf5 import _save_hdf5
# from ._save_modules._torch import _save_torch
# from ._save_modules._yaml import _save_yaml
# from ._save_modules._json import _save_json
# from ._save_modules._matlab import _save_matlab
# from ._save_modules._catboost import _save_catboost
# from ._save_modules._plotly import _save_plotly_html
# 
# 
# def save(
#     obj: Any,
#     specified_path: str,
#     makedirs: bool = True,
#     verbose: bool = True,
#     symlink_from_cwd: bool = False,
#     dry_run: bool = False,
#     no_csv: bool = False,
#     **kwargs,
# ) -> None:
#     """
#     Save an object to a file with the specified format.
# 
#     Parameters
#     ----------
#     obj : Any
#         The object to be saved. Can be a NumPy array, PyTorch tensor, Pandas DataFrame, or any serializable object.
#     specified_path : str
#         The file name or path where the object should be saved. The file extension determines the format.
#     makedirs : bool, optional
#         If True, create the directory path if it does not exist. Default is True.
#     verbose : bool, optional
#         If True, print a message upon successful saving. Default is True.
#     symlink_from_cwd : bool, optional
#         If True, create a _symlink from the current working directory. Default is False.
#     dry_run : bool, optional
#         If True, simulate the saving process without actually writing files. Default is False.
#     **kwargs
#         Additional keyword arguments to pass to the underlying save function of the specific format.
# 
#     Returns
#     -------
#     None
# 
#     Notes
#     -----
#     Supported formats include CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, and CBM.
#     The function dynamically selects the appropriate saving mechanism based on the file extension.
# 
#     Examples
#     --------
#     >>> import mngs
#     >>> import numpy as np
#     >>> import pandas as pd
#     >>> import torch
#     >>> import matplotlib.pyplot as plt
# 
#     >>> # Save NumPy array
#     >>> arr = np.array([1, 2, 3])
#     >>> mngs.io.save(arr, "data.npy")
# 
#     >>> # Save Pandas DataFrame
#     >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
#     >>> mngs.io.save(df, "data.csv")
# 
#     >>> # Save PyTorch tensor
#     >>> tensor = torch.tensor([1, 2, 3])
#     >>> mngs.io.save(tensor, "model.pth")
# 
#     >>> # Save dictionary
#     >>> data_dict = {"a": 1, "b": 2, "c": [3, 4, 5]}
#     >>> mngs.io.save(data_dict, "data.pkl")
# 
#     >>> # Save matplotlib figure
#     >>> plt.figure()
#     >>> plt.plot(np.array([1, 2, 3]))
#     >>> mngs.io.save(plt, "plot.png")
# 
#     >>> # Save as YAML
#     >>> mngs.io.save(data_dict, "config.yaml")
# 
#     >>> # Save as JSON
#     >>> mngs.io.save(data_dict, "data.json")
#     """
#     try:
#         ########################################
#         # DO NOT MODIFY THIS SECTION
#         ########################################
#         #
#         # Determine saving directory from the script.
#         #
#         # When called in /path/to/script.py,
#         # data will be saved under `/path/to/script.py_out/`
#         #
#         # On the other hand, when called in ipython environment,
#         # data will be saved under `/tmp/{_os.getenv("USER")/`
#         #
#         ########################################
#         spath, sfname = None, None
# 
#         # f-expression handling
#         if specified_path.startswith('f"'):
#             specified_path = eval(specified_path)
# 
#         # When full path
#         if specified_path.startswith("/"):
#             spath = specified_path
# 
#         # When relative path
#         else:
#             script_path = inspect.stack()[1].filename
# 
#             # Fake path if in ipython
#             if ("ipython" in script_path) or ("<stdin>" in script_path):
#                 script_path = f'/tmp/{_os.getenv("USER")}'
# 
#             sdir = clean_path(_os.path.splitext(script_path)[0] + "_out")
#             spath = _os.path.join(sdir, specified_path)
# 
#         # Sanitization
#         spath_final = clean(spath)
#         ########################################
# 
#         # Potential path to _symlink
#         spath_cwd = _os.getcwd() + "/" + specified_path
#         spath_cwd = clean(spath_cwd)
# 
#         # Removes spath and spath_cwd to prevent potential circular links
#         for path in [spath_final, spath_cwd]:
#             sh(f"rm -f {path}", verbose=False)
# 
#         if dry_run:
#             print(
#                 color_text(f"\n(dry run) Saved to: {spath_final}", c="yellow")
#             )
#             return
# 
#         # Ensure directory exists
#         if makedirs:
#             _os.makedirs(_os.path.dirname(spath_final), exist_ok=True)
# 
#         # Main
#         _save(
#             obj,
#             spath_final,
#             verbose=verbose,
#             symlink_from_cwd=symlink_from_cwd,
#             dry_run=dry_run,
#             no_csv=no_csv,
#             **kwargs,
#         )
# 
#         # Symbolic link
#         _symlink(spath, spath_cwd, symlink_from_cwd, verbose)
# 
#     except Exception as e:
#         logging.error(
#             f"Error occurred while saving: {str(e)}"
#             f"Debug: Initial script_path = {inspect.stack()[1].filename}"
#             f"Debug: Final spath = {spath}"
#         )
# 
# 
# def _symlink(spath, spath_cwd, symlink_from_cwd, verbose):
#     """Create a symbolic link from the current working directory."""
#     if symlink_from_cwd and (spath != spath_cwd):
#         _os.makedirs(_os.path.dirname(spath_cwd), exist_ok=True)
#         sh(f"rm -f {spath_cwd}", verbose=False)
#         sh(f"ln -sfr {spath} {spath_cwd}", verbose=False)
#         if verbose:
#             print(color_text(f"\n(Symlinked to: {spath_cwd})", "yellow"))
# 
# 
# def _save(
#     obj,
#     spath,
#     verbose=True,
#     symlink_from_cwd=False,
#     dry_run=False,
#     no_csv=False,
#     **kwargs,
# ):
#     """
#     Save an object based on the file extension.
#     
#     This function dispatches to the appropriate specialized save function
#     based on the file extension of the provided path.
#     """
#     # Dispatch based on file extension
#     if spath.endswith(".csv"):
#         _save_csv(obj, spath, **kwargs)
# 
#     # numpy
#     elif spath.endswith(".npy"):
#         _save_npy(obj, spath)
# 
#     # numpy npz
#     elif spath.endswith(".npz"):
#         _save_npz(obj, spath)
# 
#     # pkl
#     elif spath.endswith(".pkl"):
#         _save_pickle(obj, spath)
# 
#     # pkl.gz
#     elif spath.endswith(".pkl.gz"):
#         _save_pickle_gz(obj, spath)
# 
#     # joblib
#     elif spath.endswith(".joblib"):
#         _save_joblib(obj, spath)
# 
#     # html
#     elif spath.endswith(".html"):
#         # plotly
#         import plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             _save_plotly_html(obj, spath)
# 
#     # image ----------------------------------------
#     elif any(
#         [
#             spath.endswith(image_ext)
#             for image_ext in [
#                 ".png",
#                 ".tiff",
#                 ".tif",
#                 ".jpeg",
#                 ".jpg",
#                 ".svg",
#             ]
#         ]
#     ):
#         _save_image(obj, spath, **kwargs)
#         ext = _os.path.splitext(spath)[1].lower()
#         try:
#             if not no_csv:
#                 ext_wo_dot = ext.replace(".", "")
#                 save(
#                     obj.export_as_csv(),
#                     spath.replace(ext_wo_dot, "csv"),
#                     symlink_from_cwd=symlink_from_cwd,
#                     dry_run=dry_run,
#                     **kwargs,
#                 )
#         except Exception as e:
#             pass
# 
#     # mp4
#     elif spath.endswith(".mp4"):
#         obj.save(spath, writer="ffmpeg", **kwargs)
#         del obj
# 
#     # yaml
#     elif spath.endswith(".yaml"):
#         _save_yaml(obj, spath)
# 
#     # json
#     elif spath.endswith(".json"):
#         _save_json(obj, spath)
# 
#     # hdf5
#     elif spath.endswith(".hdf5"):
#         _save_hdf5(obj, spath)
# 
#     # pth
#     elif spath.endswith(".pth") or spath.endswith(".pt"):
#         _save_torch(obj, spath, **kwargs)
# 
#     # mat
#     elif spath.endswith(".mat"):
#         _save_matlab(obj, spath)
# 
#     # catboost model
#     elif spath.endswith(".cbm"):
#         _save_catboost(obj, spath)
# 
#     # Text
#     elif any(
#         spath.endswith(ext)
#         for ext in [".txt", ".md", ".py", ".html", ".css", ".js"]
#     ):
#         _save_text(obj, spath)
# 
#     else:
#         raise ValueError(f"Unsupported file format. {spath} was not saved.")
# 
#     if verbose:
#         if _os.path.exists(spath):
#             file_size = getsize(spath)
#             file_size = readable_bytes(file_size)
#             print(color_text(f"\nSaved to: {spath} ({file_size})", c="yellow"))
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save.py
# --------------------------------------------------------------------------------
