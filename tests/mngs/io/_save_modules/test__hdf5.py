#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:45:20 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__hdf5.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__hdf5.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import numpy as np
import h5py


def test_save_hdf5_with_numpy_arrays():
    """Test saving a dictionary of NumPy arrays to HDF5."""
    from mngs.io._save_modules._hdf5 import _save_hdf5
    
    # Create test dictionary with numpy arrays
    test_dict = {
        'array1': np.array([1, 2, 3, 4, 5]),
        'array2': np.array([[1, 2], [3, 4], [5, 6]]),
        'array3': np.zeros((3, 4, 5))  # 3D array
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_hdf5(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        with h5py.File(temp_path, 'r') as hf:
            # Check all keys exist
            assert set(hf.keys()) == set(test_dict.keys())
            
            # Check array contents
            for key, expected_array in test_dict.items():
                loaded_array = hf[key][()]
                assert np.array_equal(loaded_array, expected_array)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_hdf5_with_different_dtypes():
    """Test saving arrays with different data types to HDF5."""
    from mngs.io._save_modules._hdf5 import _save_hdf5
    
    # Create test arrays with different dtypes
    test_dict = {
        'int_array': np.array([1, 2, 3], dtype=np.int32),
        'float_array': np.array([1.1, 2.2, 3.3], dtype=np.float64),
        'bool_array': np.array([True, False, True], dtype=np.bool_)
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_hdf5(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents and dtypes
        with h5py.File(temp_path, 'r') as hf:
            # Check int array
            loaded_int_array = hf['int_array'][()]
            assert np.array_equal(loaded_int_array, test_dict['int_array'])
            assert loaded_int_array.dtype == test_dict['int_array'].dtype
            
            # Check float array
            loaded_float_array = hf['float_array'][()]
            assert np.array_equal(loaded_float_array, test_dict['float_array'])
            assert loaded_float_array.dtype == test_dict['float_array'].dtype
            
            # Check bool array (note: may be stored as uint8 in HDF5)
            loaded_bool_array = hf['bool_array'][()]
            assert np.array_equal(loaded_bool_array.astype(np.bool_), 
                                test_dict['bool_array'])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_hdf5_empty_dict():
    """Test saving an empty dictionary to HDF5."""
    from mngs.io._save_modules._hdf5 import _save_hdf5
    
    # Create an empty dictionary
    test_dict = {}
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the empty dictionary
        _save_hdf5(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify it's empty
        with h5py.File(temp_path, 'r') as hf:
            assert len(hf.keys()) == 0
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_hdf5.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:24:04 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_hdf5.py
# 
# import h5py
# 
# 
# def _save_hdf5(obj, spath):
#     """
#     Save a dictionary of arrays to an HDF5 file.
#     
#     Parameters
#     ----------
#     obj : dict
#         Dictionary of arrays to save. Keys will be dataset names.
#     spath : str
#         Path where the HDF5 file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     name_list, obj_list = [], []
#     for k, v in obj.items():
#         name_list.append(k)
#         obj_list.append(v)
#     with h5py.File(spath, "w") as hf:
#         for name, obj in zip(name_list, obj_list):
#             hf.create_dataset(name, data=obj)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_hdf5.py
# --------------------------------------------------------------------------------
