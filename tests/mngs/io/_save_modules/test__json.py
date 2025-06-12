#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:50:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/tests/mngs/io/_save_modules/test__json.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/io/_save_modules/test__json.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for JSON saving functionality
"""

=======
# Timestamp: "2025-05-16 13:35:40 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__json.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__json.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

>>>>>>> origin/main
import os
import tempfile
import pytest
import json
<<<<<<< HEAD
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date

from mngs.io._save_modules._json import save_json


class TestSaveJSON:
    """Test suite for save_json function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_simple_dict(self):
        """Test saving simple dictionary"""
        data = {"a": 1, "b": 2, "c": "hello"}
        save_json(data, self.test_file)
        
        assert os.path.exists(self.test_file)
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_nested_dict(self):
        """Test saving nested dictionary"""
        data = {
            "level1": {
                "level2": {
                    "level3": {"value": 42}
                }
            },
            "list": [1, 2, 3],
            "mixed": [{"a": 1}, {"b": 2}]
        }
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_list(self):
        """Test saving list"""
        data = [1, 2, 3, "four", 5.5, True, None]
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_numpy_array(self):
        """Test saving numpy array with custom encoder"""
        arr = np.array([1, 2, 3, 4, 5])
        save_json(arr, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == arr.tolist()

    def test_save_numpy_multidimensional(self):
        """Test saving multi-dimensional numpy array"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        save_json(arr, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == arr.tolist()

    def test_save_numpy_types(self):
        """Test saving various numpy scalar types"""
        data = {
            "int32": np.int32(42),
            "int64": np.int64(42),
            "float32": np.float32(3.14),
            "float64": np.float64(3.14),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3])
        }
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["int32"] == 42
        assert loaded["int64"] == 42
        assert loaded["float32"] == pytest.approx(3.14, rel=1e-6)
        assert loaded["float64"] == pytest.approx(3.14)
        assert loaded["bool"] is True
        assert loaded["array"] == [1, 2, 3]

    def test_save_pandas_series(self):
        """Test saving pandas Series"""
        series = pd.Series([1, 2, 3, 4], name="test_series")
        save_json(series.to_dict(), self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)

    def test_save_with_indent(self):
        """Test saving with pretty printing"""
        data = {"a": 1, "b": {"c": 2, "d": 3}}
        save_json(data, self.test_file, indent=2)
        
        with open(self.test_file, "r") as f:
            content = f.read()
        
        # Check that content is indented
        assert "  " in content  # Has indentation
        
        # Still loads correctly
        loaded = json.loads(content)
        assert loaded == data

    def test_save_with_sort_keys(self):
        """Test saving with sorted keys"""
        data = {"z": 1, "a": 2, "m": 3}
        save_json(data, self.test_file, sort_keys=True)
        
        with open(self.test_file, "r") as f:
            content = f.read()
        
        # Check key order in file
        a_pos = content.index('"a"')
        m_pos = content.index('"m"')
        z_pos = content.index('"z"')
        assert a_pos < m_pos < z_pos

    def test_save_special_values(self):
        """Test saving special float values"""
        data = {
            "infinity": float('inf'),
            "neg_infinity": float('-inf'),
            "nan": float('nan')
        }
        save_json(data, self.test_file, allow_nan=True)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["infinity"] == float('inf')
        assert loaded["neg_infinity"] == float('-inf')
        assert np.isnan(loaded["nan"])

    def test_save_unicode(self):
        """Test saving Unicode characters"""
        data = {
            "english": "Hello",
            "japanese": "こんにちは",
            "emoji": "😊🎉",
            "special": "café",
            "mixed": "Hello世界"
        }
        save_json(data, self.test_file, ensure_ascii=False)
        
        with open(self.test_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_large_data(self):
        """Test saving large data structure"""
        data = {
            f"key_{i}": list(range(100)) 
            for i in range(100)
        }
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert len(loaded) == 100
        assert loaded["key_0"] == list(range(100))

    def test_save_empty_containers(self):
        """Test saving empty containers"""
        data = {
            "empty_dict": {},
            "empty_list": [],
            "empty_string": "",
            "null": None
        }
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_mixed_numpy_native(self):
        """Test saving mix of numpy and native Python types"""
        data = {
            "numpy_int": np.int64(42),
            "python_int": 42,
            "numpy_array": np.array([1, 2, 3]),
            "python_list": [1, 2, 3],
            "numpy_float": np.float64(3.14),
            "python_float": 3.14
        }
        save_json(data, self.test_file)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        
        # All should be converted to native Python types
        assert loaded["numpy_int"] == 42
        assert loaded["python_int"] == 42
        assert loaded["numpy_array"] == [1, 2, 3]
        assert loaded["python_list"] == [1, 2, 3]

    def test_error_circular_reference(self):
        """Test handling circular references"""
        # JSON doesn't support circular references
        data = {"a": {}}
        data["a"]["circular"] = data["a"]
        
        with pytest.raises(ValueError):
            save_json(data, self.test_file)

    def test_save_scientific_data(self):
        """Test saving scientific/numerical data"""
        data = {
            "experiment": "test_001",
            "parameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            },
            "results": {
                "accuracy": 0.95,
                "loss": 0.0234,
                "confusion_matrix": [[10, 2], [1, 15]]
            },
            "metadata": {
                "timestamp": "2023-01-01T00:00:00",
                "version": "1.0.0"
            }
        }
        save_json(data, self.test_file, indent=2)
        
        with open(self.test_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data


# EOF
=======


def test_save_json_dict():
    """Test saving a dictionary to JSON format."""
    from mngs.io._save_modules._json import _save_json
    
    # Create test dictionary
    test_dict = {
        'string': 'value',
        'number': 42,
        'float': 3.14,
        'boolean': True,
        'null': None,
        'array': [1, 2, 3],
        'nested': {
            'a': 1,
            'b': 2
        }
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the dictionary
        _save_json(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with open(temp_path, 'r') as f:
            loaded_dict = json.load(f)
        
        # Check the loaded data matches the original
        assert loaded_dict == test_dict
        assert loaded_dict['string'] == 'value'
        assert loaded_dict['number'] == 42
        assert loaded_dict['float'] == 3.14
        assert loaded_dict['boolean'] is True
        assert loaded_dict['null'] is None
        assert loaded_dict['array'] == [1, 2, 3]
        assert loaded_dict['nested']['a'] == 1
        assert loaded_dict['nested']['b'] == 2
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_json_list():
    """Test saving a list to JSON format."""
    from mngs.io._save_modules._json import _save_json
    
    # Create test list
    test_list = [
        'string',
        42,
        3.14,
        True,
        None,
        [1, 2, 3],
        {'a': 1, 'b': 2}
    ]
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the list
        _save_json(test_list, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with open(temp_path, 'r') as f:
            loaded_list = json.load(f)
        
        # Check the loaded data matches the original
        assert loaded_list == test_list
        assert loaded_list[0] == 'string'
        assert loaded_list[1] == 42
        assert loaded_list[2] == 3.14
        assert loaded_list[3] is True
        assert loaded_list[4] is None
        assert loaded_list[5] == [1, 2, 3]
        assert loaded_list[6]['a'] == 1
        assert loaded_list[6]['b'] == 2
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_json_formatting():
    """Test that the JSON is formatted with indentation."""
    from mngs.io._save_modules._json import _save_json
    
    # Create test data
    test_data = {'a': 1, 'b': 2, 'c': 3}
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_json(test_data, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Read the raw file content
        with open(temp_path, 'r') as f:
            content = f.read()
        
        # Check that the content has indentation (formatted JSON)
        assert '    "a": 1' in content
        assert '    "b": 2' in content
        assert '    "c": 3' in content
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_json.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:27:18 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_json.py
# 
# import json
# 
# 
# def _save_json(obj, spath):
#     """
#     Save a Python object as a JSON file.
#     
#     Parameters
#     ----------
#     obj : dict or list
#         The object to serialize to JSON.
#     spath : str
#         Path where the JSON file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     with open(spath, "w") as f:
#         json.dump(obj, f, indent=4)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_json.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
