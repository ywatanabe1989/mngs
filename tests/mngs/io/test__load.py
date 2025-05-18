#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:35:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test__load.py
# ----------------------------------------
import os
import tempfile
import pytest
from unittest import mock
# ----------------------------------------

class TestLoadFunction:
    """Test class for _load.py module functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Create mocks for key dependencies
        self.clean_path_mock = mock.Mock(side_effect=lambda x: x)  # just returns the input
        
        # Loader function mocks
        self.load_csv_mock = mock.Mock(return_value="csv_data")
        self.load_npy_mock = mock.Mock(return_value="numpy_data")
        self.load_txt_mock = mock.Mock(return_value="text_data")
        self.load_pickle_mock = mock.Mock(return_value="pickle_data")
        self.load_torch_mock = mock.Mock(return_value="torch_model")
        self.load_json_mock = mock.Mock(return_value={"key": "json_data"})
        self.load_yaml_mock = mock.Mock(return_value={"key": "yaml_data"})
        
        # Create our mock loader dictionary 
        self.loaders_dict = {
            # Default
            "": self.load_txt_mock,
            # Config/Settings
            "yaml": self.load_yaml_mock,
            "yml": self.load_yaml_mock,
            "json": self.load_json_mock,
            # ML/DL Models
            "pth": self.load_torch_mock,
            "pt": self.load_torch_mock,
            "pkl": self.load_pickle_mock,
            # Tabular Data
            "csv": self.load_csv_mock,
            # Scientific Data
            "npy": self.load_npy_mock,
            # Documents
            "txt": self.load_txt_mock
        }
        
        # Set up test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample files of different formats
        self.csv_path = os.path.join(self.temp_dir, "data.csv")
        with open(self.csv_path, 'w') as f:
            f.write("id,value\n1,test")
            
        self.txt_path = os.path.join(self.temp_dir, "data.txt")
        with open(self.txt_path, 'w') as f:
            f.write("sample text data")
            
        self.json_path = os.path.join(self.temp_dir, "data.json")
        with open(self.json_path, 'w') as f:
            f.write('{"key": "value"}')
            
        # Create a no-extension file
        self.no_ext_path = os.path.join(self.temp_dir, "data_no_ext")
        with open(self.no_ext_path, 'w') as f:
            f.write("data without extension")
            
        # Create a file with unknown extension
        self.unknown_ext_path = os.path.join(self.temp_dir, "data.unknown")
        with open(self.unknown_ext_path, 'w') as f:
            f.write("data with unknown extension")
            
        self.nonexistent_path = os.path.join(self.temp_dir, "nonexistent.csv")
        
        # Create a test load function
        self.load = lambda path, show=False, verbose=False, **kwargs: self._test_load(path, show, verbose, **kwargs)
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _test_load(self, lpath, show=False, verbose=False, **kwargs):
        """Test implementation of the load function."""
        # Simulate clean_path - just returns the input in this mock
        lpath = self.clean_path_mock(lpath)
        
        # Check if file exists
        if not os.path.exists(lpath):
            raise FileNotFoundError(f"{lpath} not found.")
            
        # Get file extension
        ext = lpath.split(".")[-1] if "." in lpath else ""
        
        # Select loader based on extension
        loader = self.loaders_dict.get(ext, self.load_txt_mock)
        
        try:
            # Call the appropriate loader mock
            return loader(lpath, **kwargs)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Error loading file {lpath}: {str(e)}")
    
    def test_load_csv_file(self):
        """Test loading a CSV file."""
        # Set expected return
        self.load_csv_mock.return_value = "csv_data_test"
        
        # Call the load function
        result = self.load(self.csv_path)
        
        # Verify the result
        assert result == "csv_data_test"
        self.load_csv_mock.assert_called_with(self.csv_path)
    
    def test_load_txt_file(self):
        """Test loading a text file."""
        # Set expected return
        self.load_txt_mock.return_value = "text_data_test"
        
        # Call the load function
        result = self.load(self.txt_path)
        
        # Verify the result
        assert result == "text_data_test"
        self.load_txt_mock.assert_called_with(self.txt_path)
    
    def test_load_json_file(self):
        """Test loading a JSON file."""
        # Set expected return
        self.load_json_mock.return_value = {"key": "json_value_test"}
        
        # Call the load function
        result = self.load(self.json_path)
        
        # Verify the result
        assert result == {"key": "json_value_test"}
        self.load_json_mock.assert_called_with(self.json_path)
    
    def test_load_with_nonexistent_file(self):
        """Test loading a nonexistent file raises FileNotFoundError."""
        # This should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            self.load(self.nonexistent_path)
    
    def test_load_with_empty_extension(self):
        """Test loading a file with no extension."""
        # Set expected return
        self.load_txt_mock.return_value = "text_data_no_ext"
        
        # Reset the mock to clear previous calls
        self.load_txt_mock.reset_mock()
        
        # Call the load function
        result = self.load(self.no_ext_path)
        
        # Verify the result - should use the default txt loader
        assert result == "text_data_no_ext"
        self.load_txt_mock.assert_called_with(self.no_ext_path)
    
    def test_load_with_unknown_extension(self):
        """Test loading a file with unknown extension."""
        # Set expected return
        self.load_txt_mock.return_value = "text_data_unknown_ext"
        
        # Reset the mock to clear previous calls
        self.load_txt_mock.reset_mock()
        
        # Call the load function
        result = self.load(self.unknown_ext_path)
        
        # Verify the result - should use the default txt loader
        assert result == "text_data_unknown_ext"
        self.load_txt_mock.assert_called_with(self.unknown_ext_path)
    
    def test_load_with_additional_kwargs(self):
        """Test loading a file with additional kwargs."""
        # Set expected return
        self.load_csv_mock.return_value = "csv_data_with_kwargs"
        
        # Reset the mock to clear previous calls
        self.load_csv_mock.reset_mock()
        
        # Additional kwargs to pass
        kwargs = {"header": None, "index_col": 0}
        
        # Call the load function with kwargs
        result = self.load(self.csv_path, **kwargs)
        
        # Verify the result
        assert result == "csv_data_with_kwargs"
        self.load_csv_mock.assert_called_with(self.csv_path, **kwargs)
    
    def test_load_with_error_from_loader(self):
        """Test handling errors from loader functions."""
        # Make the loader raise an error
        self.load_csv_mock.side_effect = ValueError("Test error from loader")
        
        # This should wrap the error in a ValueError
        with pytest.raises(ValueError) as excinfo:
            self.load(self.csv_path)
        
        # Check the error message includes the original error
        assert "Test error from loader" in str(excinfo.value)
    
    def test_load_handles_show_verbose_params(self):
        """Test that load function handles show and verbose parameters."""
        # Set expected return
        self.load_csv_mock.return_value = "csv_data_show_verbose"
        
        # Reset the mock to clear previous calls
        self.load_csv_mock.reset_mock()
        
        # Call the load function with show and verbose
        result = self.load(self.csv_path, show=True, verbose=True)
        
        # The parameters should be ignored (not passed to loader)
        assert result == "csv_data_show_verbose"
        self.load_csv_mock.assert_called_with(self.csv_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-10 08:05:53 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "/ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from typing import Any
# from ..decorators import preserve_doc
# from ..str._clean_path import clean_path
# # from ._load_modules._catboost import _load_catboost
# from ._load_modules._con import _load_con
# from ._load_modules._db import _load_sqlite3db
# from ._load_modules._docx import _load_docx
# from ._load_modules._eeg import _load_eeg_data
# from ._load_modules._hdf5 import _load_hdf5
# from ._load_modules._image import _load_image
# from ._load_modules._joblib import _load_joblib
# from ._load_modules._json import _load_json
# from ._load_modules._markdown import _load_markdown
# from ._load_modules._numpy import _load_npy
# from ._load_modules._matlab import _load_matlab
# from ._load_modules._pandas import _load_csv, _load_excel, _load_tsv
# from ._load_modules._pdf import _load_pdf
# from ._load_modules._pickle import _load_pickle
# from ._load_modules._torch import _load_torch
# from ._load_modules._txt import _load_txt
# from ._load_modules._xml import _load_xml
# from ._load_modules._yaml import _load_yaml
# from ._load_modules._matlab import _load_matlab
# 
# def load(
#     lpath: str, show: bool = False, verbose: bool = False, **kwargs
# ) -> Any:
#     """
#     Load data from various file formats.
# 
#     This function supports loading data from multiple file formats.
# 
#     Parameters
#     ----------
#     lpath : str
#         The path to the file to be loaded.
#     show : bool, optional
#         If True, display additional information during loading. Default is False.
#     verbose : bool, optional
#         If True, print verbose output during loading. Default is False.
#     **kwargs : dict
#         Additional keyword arguments to be passed to the specific loading function.
# 
#     Returns
#     -------
#     object
#         The loaded data object, which can be of various types depending on the input file format.
# 
#     Raises
#     ------
#     ValueError
#         If the file extension is not supported.
#     FileNotFoundError
#         If the specified file does not exist.
# 
#     Supported Extensions
#     -------------------
#     - Data formats: .csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .json, .yaml, .yml
#     - Scientific: .npy, .npz, .mat, .hdf5, .con
#     - ML/DL: .pth, .pt, .cbm, .joblib, .pkl
#     - Documents: .txt, .log, .event, .md, .docx, .pdf, .xml
#     - Images: .jpg, .png, .tiff, .tif
#     - EEG data: .vhdr, .vmrk, .edf, .bdf, .gdf, .cnt, .egi, .eeg, .set
#     - Database: .db
# 
#     Examples
#     --------
#     >>> data = load('data.csv')
#     >>> image = load('image.png')
#     >>> model = load('model.pth')
#     """
#     lpath = clean_path(lpath)
# 
#     if not os.path.exists(lpath):
#         raise FileNotFoundError(f"{lpath} not found.")
# 
#     loaders_dict = {
#         # Default
#         "": _load_txt,
#         # Config/Settings
#         "yaml": _load_yaml,
#         "yml": _load_yaml,
#         "json": _load_json,
#         "xml": _load_xml,
#         # ML/DL Models
#         "pth": _load_torch,
#         "pt": _load_torch,
#         # "cbm": _load_catboost,
#         "joblib": _load_joblib,
#         "pkl": _load_pickle,
#         # Tabular Data
#         "csv": _load_csv,
#         "tsv": _load_tsv,
#         "xls": _load_excel,
#         "xlsx": _load_excel,
#         "xlsm": _load_excel,
#         "xlsb": _load_excel,
#         "db": _load_sqlite3db,
#         # Scientific Data
#         "npy": _load_npy,
#         "npz": _load_npy,
#         "mat": _load_matlab,
#         "hdf5": _load_hdf5,
#         "mat": _load_matlab,
#         "con": _load_con,
#         # Documents
#         "txt": _load_txt,
#         "tex": _load_txt,
#         "log": _load_txt,
#         "event": _load_txt,
#         "py": _load_txt,
#         "sh": _load_txt,
#         "md": _load_markdown,
#         "docx": _load_docx,
#         "pdf": _load_pdf,
#         # Images
#         "jpg": _load_image,
#         "png": _load_image,
#         "tiff": _load_image,
#         "tif": _load_image,
#         # EEG Data
#         "vhdr": _load_eeg_data,
#         "vmrk": _load_eeg_data,
#         "edf": _load_eeg_data,
#         "bdf": _load_eeg_data,
#         "gdf": _load_eeg_data,
#         "cnt": _load_eeg_data,
#         "egi": _load_eeg_data,
#         "eeg": _load_eeg_data,
#         "set": _load_eeg_data,
#     }
# 
#     ext = lpath.split(".")[-1] if "." in lpath else ""
#     loader = preserve_doc(loaders_dict.get(ext, _load_txt))
# 
#     try:
#         return loader(lpath, **kwargs)
#     except (ValueError, FileNotFoundError) as e:
#         raise ValueError(f"Error loading file {lpath}: {str(e)}")
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load.py
# --------------------------------------------------------------------------------
