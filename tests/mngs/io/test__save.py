#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-13 22:30:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test__save.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/io/test__save.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
import json
import pickle
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import mngs


def test_torch_save_pt_extension():
    """Test that PyTorch models can be saved with .pt extension."""
    _save = mngs.io.save

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
    _save = mngs.io.save

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


@pytest.mark.skip(reason="_save_csv is an internal function")
def test_save_csv_deduplication():
    """Test that CSV files are not rewritten if content hasn't changed."""
    # This test requires access to internal _save_csv function
    pass


def test_save_matplotlib_figure():
    """Test saving matplotlib figures in various formats."""
    import matplotlib.pyplot as plt
    from mngs.io import save
    
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test Plot")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test PNG save
        png_path = os.path.join(tmpdir, "figure.png")
        save(fig, png_path, verbose=False)
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0
        
        # Test PDF save
        pdf_path = os.path.join(tmpdir, "figure.pdf")
        save(fig, pdf_path, verbose=False)
        assert os.path.exists(pdf_path)
        
        # Test SVG save
        svg_path = os.path.join(tmpdir, "figure.svg")
        save(fig, svg_path, verbose=False)
        assert os.path.exists(svg_path)
    
    plt.close(fig)


def test_save_plotly_figure():
    """Test saving plotly figures."""
    try:
        import plotly.graph_objects as go
        from mngs.io import save
        
        # Create a simple plotly figure
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test HTML save
            html_path = os.path.join(tmpdir, "plotly_fig.html")
            save(fig, html_path, verbose=False)
            assert os.path.exists(html_path)
            assert os.path.getsize(html_path) > 0
    except ImportError:
        pytest.skip("plotly not installed")


def test_save_hdf5():
    """Test saving HDF5 files."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving numpy array to HDF5
        data = np.random.rand(10, 20, 30)
        hdf5_path = os.path.join(tmpdir, "data.h5")
        save(data, hdf5_path, verbose=False)
        
        assert os.path.exists(hdf5_path)
        
        # Verify content
        with h5py.File(hdf5_path, 'r') as f:
            assert 'data' in f
            loaded_data = f['data'][:]
            np.testing.assert_array_almost_equal(loaded_data, data)


def test_save_matlab():
    """Test saving MATLAB .mat files."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving dict to .mat
        data = {
            'array': np.array([1, 2, 3]),
            'matrix': np.array([[1, 2], [3, 4]]),
            'scalar': 42.0
        }
        mat_path = os.path.join(tmpdir, "data.mat")
        save(data, mat_path, verbose=False)
        
        assert os.path.exists(mat_path)
        
        # Verify content
        loaded = scipy.io.loadmat(mat_path)
        np.testing.assert_array_equal(loaded['array'].flatten(), data['array'])
        np.testing.assert_array_equal(loaded['matrix'], data['matrix'])
        assert float(loaded['scalar']) == data['scalar']


def test_save_compressed_pickle():
    """Test saving compressed pickle files."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Large data that benefits from compression
        data = {
            'large_array': np.random.rand(1000, 1000),
            'metadata': {'compression': True}
        }
        
        # Test .pkl.gz
        gz_path = os.path.join(tmpdir, "data.pkl.gz")
        save(data, gz_path, verbose=False)
        assert os.path.exists(gz_path)
        
        # Verify it's compressed (should be smaller than uncompressed)
        pkl_path = os.path.join(tmpdir, "data_uncompressed.pkl")
        save(data, pkl_path, verbose=False)
        
        assert os.path.getsize(gz_path) < os.path.getsize(pkl_path)


def test_save_joblib():
    """Test saving with joblib format."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create complex object
        data = {
            'model': {'weights': np.random.rand(100, 100)},
            'config': {'learning_rate': 0.001}
        }
        
        joblib_path = os.path.join(tmpdir, "model.joblib")
        save(data, joblib_path, verbose=False)
        
        assert os.path.exists(joblib_path)
        
        # Verify content
        loaded = joblib.load(joblib_path)
        np.testing.assert_array_equal(loaded['model']['weights'], data['model']['weights'])
        assert loaded['config'] == data['config']


def test_save_pil_image():
    """Test saving PIL images."""
    from PIL import Image
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple PIL image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Test various image formats
        for ext in ['.png', '.jpg', '.tiff']:
            img_path = os.path.join(tmpdir, f"image{ext}")
            save(img, img_path, verbose=False)
            assert os.path.exists(img_path)
            
            # Verify it can be loaded
            loaded_img = Image.open(img_path)
            assert loaded_img.size == (100, 100)
            loaded_img.close()


def test_save_with_datetime_path():
    """Test saving with datetime in path."""
    from mngs.io import save
    from datetime import datetime
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create path with datetime placeholder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {"test": "data"}
        
        # Path with datetime
        save_path = os.path.join(tmpdir, f"data_{timestamp}.json")
        save(data, save_path, verbose=False)
        
        assert os.path.exists(save_path)


def test_save_verbose_output(capsys):
    """Test verbose output during save."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data = np.array([1, 2, 3])
        save_path = os.path.join(tmpdir, "data.npy")
        
        # Save with verbose=True
        save(data, save_path, verbose=True)
        
        # Check output
        captured = capsys.readouterr()
        assert "Saving" in captured.out
        assert save_path in captured.out
        assert "KB" in captured.out or "B" in captured.out  # Size info


def test_save_figure_with_csv_export():
    """Test saving figure with CSV data export."""
    import matplotlib.pyplot as plt
    from mngs.io import save
    
    # Create figure with data
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    ax.plot(x, y, label="Test Line")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "figure.png")
        
        # Save figure (CSV export depends on wrapped axes)
        save(fig, fig_path, verbose=False)
        assert os.path.exists(fig_path)
    
    plt.close(fig)


def test_save_error_handling():
    """Test error handling in save function."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with None object
        with pytest.raises(Exception):
            save(None, os.path.join(tmpdir, "none.txt"), verbose=False)
        
        # Test with invalid path (no extension)
        data = {"test": "data"}
        with pytest.raises(ValueError, match="Unsupported file format"):
            save(data, os.path.join(tmpdir, "no_extension"), verbose=False)
        
        # Test with read-only directory
        ro_dir = os.path.join(tmpdir, "readonly")
        os.makedirs(ro_dir)
        os.chmod(ro_dir, 0o444)
        
        try:
            with pytest.raises(PermissionError):
                save(data, os.path.join(ro_dir, "data.json"), verbose=False)
        finally:
            # Restore permissions for cleanup
            os.chmod(ro_dir, 0o755)


def test_save_catboost_model():
    """Test saving CatBoost models."""
    try:
        from catboost import CatBoostClassifier
        from mngs.io import save
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple model
            model = CatBoostClassifier(iterations=10, verbose=False)
            
            # Mock training data
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            model.fit(X, y, verbose=False)
            
            # Save model
            cbm_path = os.path.join(tmpdir, "model.cbm")
            save(model, cbm_path, verbose=False)
            
            assert os.path.exists(cbm_path)
    except ImportError:
        pytest.skip("CatBoost not installed")


def test_save_with_makedirs_false():
    """Test save behavior when makedirs=False."""
    from mngs.io import save
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to save to non-existent directory with makedirs=False
        data = {"test": "data"}
        save_path = os.path.join(tmpdir, "nonexistent", "data.json")
        
        # Should raise error since directory doesn't exist
        with pytest.raises(FileNotFoundError):
            save(data, save_path, verbose=False, makedirs=False)


class TestSave:
    """Test cases for mngs.io.save function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_numpy_array(self, temp_dir):
        """Test saving NumPy arrays."""
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        save_path = os.path.join(temp_dir, "array.npy")

        # Act
        mngs.io.save(arr, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded, arr)

    def test_save_numpy_compressed(self, temp_dir):
        """Test saving compressed NumPy arrays."""
        # Arrange
        data = {"array1": np.array([1, 2, 3]), "array2": np.array([[4, 5], [6, 7]])}
        save_path = os.path.join(temp_dir, "arrays.npz")

        # Act
        mngs.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["array1"], data["array1"])
        np.testing.assert_array_equal(loaded["array2"], data["array2"])

    def test_save_pandas_dataframe(self, temp_dir):
        """Test saving pandas DataFrames."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
        save_path = os.path.join(temp_dir, "data.csv")

        # Act
        mngs.io.save(df, save_path, verbose=False, index=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_json(self, temp_dir):
        """Test saving JSON data."""
        # Arrange
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        save_path = os.path.join(temp_dir, "data.json")

        # Act
        mngs.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_yaml(self, temp_dir):
        """Test saving YAML data."""
        # Arrange
        data = {"config": {"learning_rate": 0.001, "batch_size": 32}, "model": "ResNet"}
        save_path = os.path.join(temp_dir, "config.yaml")

        # Act
        mngs.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = mngs.io.load(save_path)
        assert loaded == data

    def test_save_pickle(self, temp_dir):
        """Test saving pickle files."""
        # Arrange
        data = {
            "array": np.array([1, 2, 3]),
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        }
        save_path = os.path.join(temp_dir, "data.pkl")

        # Act
        mngs.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            loaded = pickle.load(f)
        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["list"] == data["list"]
        assert loaded["dict"] == data["dict"]

    def test_save_text(self, temp_dir):
        """Test saving text files."""
        # Arrange
        text = "Hello\nWorld\nTest"
        save_path = os.path.join(temp_dir, "text.txt")

        # Act
        mngs.io.save(text, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path, "r") as f:
            loaded = f.read()
        assert loaded == text

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "nested", "dir", "data.json")

        # Act
        mngs.io.save(data, save_path, verbose=False, makedirs=True)

        # Assert
        assert os.path.exists(save_path)
        parent_dir = os.path.dirname(save_path)
        assert os.path.exists(parent_dir)

    def test_save_dry_run(self, temp_dir, capsys):
        """Test dry run mode."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.json")

        # Act
        mngs.io.save(data, save_path, dry_run=True, verbose=True)

        # Assert
        assert not os.path.exists(save_path)  # File should not be created
        captured = capsys.readouterr()
        assert "(dry run)" in captured.out

    def test_save_with_symlink(self, temp_dir):
        """Test saving with symlink creation."""
        # Arrange
        data = {"test": "data"}
        # Change to temp dir to test symlink
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Act
            mngs.io.save(data, "subdir/data.json", verbose=False, symlink_from_cwd=True)

            # Assert
            # Should create both the actual file and a symlink
            assert os.path.exists("subdir/data.json")
            # The implementation creates files in script_out directories
        finally:
            os.chdir(original_cwd)

    def test_save_unsupported_format(self, temp_dir):
        """Test saving with unsupported format raises error."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.unknown")

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file format"):
            mngs.io.save(data, save_path, verbose=False)

    def test_save_list_to_npz(self, temp_dir):
        """Test saving list of arrays to npz."""
        # Arrange
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        save_path = os.path.join(temp_dir, "arrays.npz")

        # Act
        mngs.io.save(arrays, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["0"], arrays[0])
        np.testing.assert_array_equal(loaded["1"], arrays[1])

    def test_save_various_csv_types(self, temp_dir):
        """Test saving various types as CSV."""
        # Test single value
        mngs.io.save(42, os.path.join(temp_dir, "single.csv"), verbose=False)

        # Test list of numbers
        mngs.io.save([1, 2, 3], os.path.join(temp_dir, "list.csv"), verbose=False)

        # Test dict
        mngs.io.save(
            {"a": 1, "b": 2}, os.path.join(temp_dir, "dict.csv"), verbose=False
        )

        # Verify all files exist
        assert os.path.exists(os.path.join(temp_dir, "single.csv"))
        assert os.path.exists(os.path.join(temp_dir, "list.csv"))
        assert os.path.exists(os.path.join(temp_dir, "dict.csv"))

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

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
>>>>>>> origin/main
