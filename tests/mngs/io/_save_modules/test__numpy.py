#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:15:20 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__numpy.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__numpy.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import numpy as np


def test_save_npy():
    """Test saving a numpy array to .npy format."""
    from mngs.io._save_modules._numpy import _save_npy
    
    # Create test array
    test_array = np.array([1, 2, 3, 4, 5])
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the array
        _save_npy(test_array, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_array = np.load(temp_path)
        assert np.array_equal(test_array, loaded_array)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_npz_with_dict():
    """Test saving a dictionary of arrays to .npz format."""
    from mngs.io._save_modules._numpy import _save_npz
    
    # Create test dictionary of arrays
    test_dict = {
        'arr1': np.array([1, 2, 3]),
        'arr2': np.array([[4, 5], [6, 7]])
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the dict of arrays
        _save_npz(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_data = np.load(temp_path)
        
        # Check each array in the dictionary
        for key, expected_array in test_dict.items():
            assert key in loaded_data.files
            assert np.array_equal(expected_array, loaded_data[key])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_npz_with_list():
    """Test saving a list of arrays to .npz format."""
    from mngs.io._save_modules._numpy import _save_npz
    
    # Create test list of arrays
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([[4, 5], [6, 7]])
    test_list = [arr1, arr2]
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the list of arrays
        _save_npz(test_list, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_data = np.load(temp_path)
        
        # Check keys were auto-generated (0, 1)
        assert '0' in loaded_data.files
        assert '1' in loaded_data.files
        
        # Check contents
        assert np.array_equal(arr1, loaded_data['0'])
        assert np.array_equal(arr2, loaded_data['1'])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_npz_with_invalid_input():
    """Test that saving an invalid object to .npz raises ValueError."""
    from mngs.io._save_modules._numpy import _save_npz
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # Try to save a string (invalid input)
        with pytest.raises(ValueError):
            _save_npz("invalid input", temp_path)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_numpy.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:19:07 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_numpy.py
# 
# import numpy as np
# 
# 
# def _save_npy(obj, spath):
#     """
#     Save a numpy array to .npy format.
#     
#     Parameters
#     ----------
#     obj : numpy.ndarray
#         The numpy array to save.
#     spath : str
#         Path where the .npy file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     np.save(spath, obj)
# 
# 
# def _save_npz(obj, spath):
#     """
#     Save numpy arrays to .npz format.
#     
#     Parameters
#     ----------
#     obj : dict or list/tuple of numpy.ndarray
#         Either a dictionary of arrays or a list/tuple of arrays.
#     spath : str
#         Path where the .npz file will be saved.
#         
#     Returns
#     -------
#     None
#     
#     Raises
#     ------
#     ValueError
#         If obj is not a dict of arrays or a list/tuple of arrays.
#     """
#     if isinstance(obj, dict):
#         np.savez_compressed(spath, **obj)
#     elif isinstance(obj, (list, tuple)) and all(
#         isinstance(x, np.ndarray) for x in obj
#     ):
#         obj = {str(ii): obj[ii] for ii in range(len(obj))}
#         np.savez_compressed(spath, **obj)
#     else:
#         raise ValueError(
#             "For .npz files, obj must be a dict of arrays or a list/tuple of arrays."
#         )
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_numpy.py
# --------------------------------------------------------------------------------
