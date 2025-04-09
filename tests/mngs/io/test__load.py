#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-02-28 00:32:18 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_dev/tests/mngs/io/test__load.py

__file__ = "./tests/mngs/io/test__load.py"
# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/io/_load.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-12 06:50:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load.py
#
# __file__ = "./src/mngs/io/_load.py"
#
# import os
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
# from ._load_modules._pandas import _load_csv, _load_excel, _load_tsv
# from ._load_modules._pdf import _load_pdf
# from ._load_modules._pickle import _load_pickle
# from ._load_modules._torch import _load_torch
# from ._load_modules._txt import _load_txt
# from ._load_modules._xml import _load_xml
# from ._load_modules._yaml import _load_yaml
#
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
#     - Scientific: .npy, .npz, .hdf5, .con
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
#         "hdf5": _load_hdf5,
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
#
# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np
import pytest

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.io._load import *

# class TestLoadFunction(unittest.TestCase):
#     """
#     Comprehensive test suite for the load function.
#     Uses mocks to test without requiring actual files.
#     """

#     def setUp(self):
#         """Set up test data and mocks"""
#         # Sample data for different formats
#         self.csv_data = "id,name,value\n1,Alice,42\n2,Bob,73\n3,Charlie,91"
#         self.tsv_data = "id\tname\tvalue\n1\tAlice\t42\n2\tBob\t73\n3\tCharlie\t91"
#         self.json_data = json.dumps({
#             "users": [
#                 {"id": 1, "name": "Alice", "value": 42},
#                 {"id": 2, "name": "Bob", "value": 73},
#                 {"id": 3, "name": "Charlie", "value": 91}
#             ]
#         })
#         self.yaml_data = """
#         users:
#           - id: 1
#             name: Alice
#             value: 42
#           - id: 2
#             name: Bob
#             value: 73
#           - id: 3
#             name: Charlie
#             value: 91
#         """
#         # Create a sample numpy array and pickle it for testing
#         self.numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
#         self.pickle_buffer = io.BytesIO()
#         pickle.dump({"array": self.numpy_array, "name": "test_array"}, self.pickle_buffer)
#         self.pickle_data = self.pickle_buffer.getvalue()

#         # Create a sample image for testing
#         self.image = Image.new('RGB', (10, 10), color='red')
#         self.image_buffer = io.BytesIO()
#         self.image.save(self.image_buffer, format='PNG')
#         self.image_data = self.image_buffer.getvalue()

#         # Test file paths
#         self.csv_path = "/path/to/test.csv"
#         self.tsv_path = "/path/to/test.tsv"
#         self.json_path = "/path/to/test.json"
#         self.yaml_path = "/path/to/test.yaml"
#         self.pickle_path = "/path/to/test.pkl"
#         self.numpy_path = "/path/to/test.npy"
#         self.image_path = "/path/to/test.png"
#         self.unsupported_path = "/path/to/test.xyz"

#     def test_csv_loading(self):
#         """Test loading CSV files with mocked file open"""
#         with patch('builtins.open', mock_open(read_data=self.csv_data)):
#             # Mock os.path.exists to return True
#             with patch('os.path.exists', return_value=True):
#                 # Mock pandas.read_csv to return a properly structured DataFrame
#                 mock_df = pd.DataFrame({
#                     'name': ['Alice', 'Bob', 'Charlie'],
#                     'value': [42, 73, 91],
#                     'id': [1, 2, 3]
#                 })
#                 with patch('pandas.read_csv', return_value=mock_df):
#                     # Test basic CSV loading
#                     result = load(self.csv_path)

#                     # Verify the result is a DataFrame with expected structure
#                     self.assertIsInstance(result, pd.DataFrame)
#                     self.assertEqual(result.shape, (3, 3))
#                     self.assertEqual(result.loc[0, 'name'], 'Alice')
#                     self.assertEqual(result.loc[1, 'value'], 73)

#                     # Test with custom parameters
#                     result_no_index = load(self.csv_path, index_col=None)
#                     self.assertEqual(result_no_index.shape[1], 3)

#     def test_tsv_loading(self):
#         """Test loading TSV files with mocked file open"""
#         with patch('builtins.open', mock_open(read_data=self.tsv_data)):
#             with patch('os.path.exists', return_value=True):
#                 result = load(self.tsv_path)

#                 self.assertIsInstance(result, pd.DataFrame)
#                 self.assertEqual(result.shape, (3, 3))
#                 self.assertEqual(result.loc[2, 'name'], 'Charlie')

#     def test_json_loading(self):
#         """Test loading JSON files with mocked file open"""
#         with patch('builtins.open', mock_open(read_data=self.json_data)):
#             with patch('os.path.exists', return_value=True):
#                 result = load(self.json_path)

#                 self.assertIsInstance(result, dict)
#                 self.assertIn('users', result)
#                 self.assertEqual(len(result['users']), 3)
#                 self.assertEqual(result['users'][0]['name'], 'Alice')
#                 self.assertEqual(result['users'][2]['value'], 91)

#     def test_yaml_loading(self):
#         """Test loading YAML files with mocked file open"""
#         with patch('builtins.open', mock_open(read_data=self.yaml_data)):
#             with patch('os.path.exists', return_value=True):
#                 result = load(self.yaml_path)

#                 self.assertIsInstance(result, dict)
#                 self.assertIn('users', result)
#                 self.assertEqual(len(result['users']), 3)
#                 self.assertEqual(result['users'][1]['name'], 'Bob')

#     @patch('pickle.load')
#     def test_pickle_loading(self, mock_pickle_load):
#         """Test loading Pickle files with mocked pickle.load"""
#         # Setup the mock to return our test data
#         mock_pickle_load.return_value = {"array": self.numpy_array, "name": "test_array"}

#         with patch('builtins.open', mock_open(read_data=self.pickle_data)):
#             with patch('os.path.exists', return_value=True):
#                 result = load(self.pickle_path)

#                 self.assertIsInstance(result, dict)
#                 self.assertIn('array', result)
#                 self.assertIn('name', result)
#                 self.assertEqual(result['name'], 'test_array')
#                 # Verify mock was called
#                 mock_pickle_load.assert_called_once()

#     @patch('numpy.load')
#     def test_numpy_loading(self, mock_numpy_load):
#         """Test loading NumPy files with mocked numpy.load"""
#         # Setup the mock to return our test array
#         mock_numpy_load.return_value = self.numpy_array

#         with patch('os.path.exists', return_value=True):
#             result = load(self.numpy_path)

#             # Verify numpy.load was called with correct arguments
#             mock_numpy_load.assert_called_once_with(self.numpy_path, allow_pickle=True)
#             self.assertTrue(np.array_equal(result, self.numpy_array))

#     @patch('PIL.Image.open')
#     def test_image_loading(self, mock_image_open):
#         """Test loading image files with mocked Image.open"""
#         # Setup the mock to return our test image
#         mock_image_open.return_value = self.image

#         with patch('os.path.exists', return_value=True):
#             result = load(self.image_path)

#             # Verify Image.open was called with correct path
#             mock_image_open.assert_called_once_with(self.image_path)
#             self.assertEqual(result.size, (10, 10))

#     def test_file_not_found(self):
#         """Test handling of non-existent files"""
#         with patch('os.path.exists', return_value=False):
#             with self.assertRaises(FileNotFoundError):
#                 load("/path/to/nonexistent.csv")

#     def test_unsupported_extension(self):
#         """Test handling of unsupported file extensions"""
#         with patch('os.path.exists', return_value=True):
#             with self.assertRaises(ValueError) as context:
#                 load(self.unsupported_path)

#             # Verify the error message contains the extension
#             self.assertIn(".xyz", str(context.Exception))
#             self.assertIn("not supported", str(context.Exception))

#     def test_malformed_json(self):
#         """Test handling of malformed JSON files"""
#         malformed_json = '{"key": "value", broken json'

#         with patch('builtins.open', mock_open(read_data=malformed_json)):
#             with patch('os.path.exists', return_value=True):
#                 with self.assertRaises(json.JSONDecodeError):
#                     load(self.json_path)

#     def test_empty_file(self):
#         """Test handling of empty files"""
#         with patch('builtins.open', mock_open(read_data="")):
#             with patch('os.path.exists', return_value=True):
#                 # Empty CSV should raise an error with pandas
#                 with self.assertRaises(Exception):
#                     load(self.csv_path)

#     @patch('pandas.read_csv')
#     def test_csv_with_custom_options(self, mock_read_csv):
#         """Test CSV loading with various custom options"""
#         # Setup mock to return a DataFrame
#         mock_df = pd.DataFrame({
#             'name': ['Alice', 'Bob', 'Charlie'],
#             'value': [42, 73, 91]
#         })
#         mock_read_csv.return_value = mock_df

#         with patch('os.path.exists', return_value=True):
#             # Test with custom separator
#             load(self.csv_path, sep=';')
#             # Check that the first positional argument is the path and the keyword arguments include sep
#             args, kwargs = mock_read_csv.call_args
#             self.assertEqual(args[0], self.csv_path)
#             self.assertEqual(kwargs['sep'], ';')

#             # Test with no header
#             load(self.csv_path, header=None)
#             args, kwargs = mock_read_csv.call_args
#             self.assertEqual(args[0], self.csv_path)
#             self.assertEqual(kwargs['header'], None)

#             # Test with custom encoding
#             load(self.csv_path, encoding='utf-16')
#             args, kwargs = mock_read_csv.call_args
#             self.assertEqual(args[0], self.csv_path)
#             self.assertEqual(kwargs['encoding'], 'utf-16')

#     def test_path_normalization(self):
#         """Test path normalization in the load function"""
#         # Path with double slashes
#         double_slash_path = "/path//to/test.csv"

#         with patch('builtins.open', mock_open(read_data=self.csv_data)):
#             with patch('os.path.exists', return_value=True):
#                 # Mock pandas.read_csv to return a DataFrame and track calls
#                 mock_df = pd.DataFrame({
#                     'name': ['Alice', 'Bob', 'Charlie'],
#                     'value': [42, 73, 91]
#                 })
#                 with patch('pandas.read_csv', return_value=mock_df) as mock_read_csv:
#                     load(double_slash_path)

#                     # The normalized path should be used
#                     args, _ = mock_read_csv.call_args
#                     self.assertEqual(args[0], double_slash_path)  # Our simplified load doesn't normalize paths

#     def test_f_string_path(self):
#         """Test handling of f-string paths"""
#         f_string_path = 'f"/path/to/test.csv"'

#         with patch('builtins.open', mock_open(read_data=self.csv_data)):
#             with patch('os.path.exists', return_value=True):
#                 # Mock pandas.read_csv to return a DataFrame and track calls
#                 mock_df = pd.DataFrame({
#                     'name': ['Alice', 'Bob', 'Charlie'],
#                     'value': [42, 73, 91]
#                 })
#                 with patch('pandas.read_csv', return_value=mock_df) as mock_read_csv:
#                     load(f_string_path)

#                     # The f-string should be processed
#                     args, _ = mock_read_csv.call_args
#                     # In our simplified load function, f-strings are processed
#                     self.assertEqual(args[0], "/path/to/test.csv")

# # if __name__ == "__main__":
# #     unittest.main()

# # class TestMainFunctionality:
# #     def setup_method(self):
# #         # Setup test fixtures
# #         pass

# #     def teardown_method(self):
# #         # Clean up after tests
# #         pass

# #     def test_basic_functionality(self):
# #         # Basic test case
# #         raise NotImplementedError("Test not yet implemented")

# #     def test_edge_cases(self):
# #         # Edge case testing
# #         raise NotImplementedError("Test not yet implemented")

# #     def test_error_handling(self):
# #         # Error handling testing
# #         raise NotImplementedError("Test not yet implemented")

def test_load_text_file():
    test_file_path = 'test_sample.txt'
    with open(test_file_path, 'w') as f:
        f.write('Sample text content')
    result = load(test_file_path)
    assert result == ['Sample text content']
    os.remove(test_file_path)

def test_load_csv_file():
    import pandas as pd
    test_file_path = 'test_sample.csv'
    df_original = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_original.to_csv(test_file_path, index=True)
    result = load(test_file_path)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df_original)
    os.remove(test_file_path)

def test_load_nonexistent_file():
    test_file_path = 'nonexistent.file'
    with pytest.raises(FileNotFoundError):
        load(test_file_path)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF