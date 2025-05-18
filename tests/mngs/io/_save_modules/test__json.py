#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:35:40 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__json.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__json.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import json


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
