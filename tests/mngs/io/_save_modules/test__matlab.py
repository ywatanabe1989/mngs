#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 14:10:15 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__matlab.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__matlab.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import numpy as np
import scipy.io


def test_save_matlab_simple_arrays():
    """Test saving a dictionary of simple arrays to a MATLAB .mat file."""
    from mngs.io._save_modules._matlab import _save_matlab
    
    # Create test dictionary with simple arrays
    test_dict = {
        'array1d': np.array([1, 2, 3, 4, 5]),
        'array2d': np.array([[1, 2, 3], [4, 5, 6]]),
        'scalar': np.array(42)
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_matlab(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_dict = scipy.io.loadmat(temp_path)
        
        # MATLAB files always include some metadata fields like '__header__'
        # So we need to check our specific keys
        assert 'array1d' in loaded_dict
        assert 'array2d' in loaded_dict
        assert 'scalar' in loaded_dict
        
        # Check array shapes and content
        # MATLAB arrays are always at least 2D, so shapes will be adjusted
        assert np.array_equal(loaded_dict['array1d'], test_dict['array1d'].reshape(1, -1))
        assert np.array_equal(loaded_dict['array2d'], test_dict['array2d'])
        # Scalar values get converted to 1x1 arrays
        assert loaded_dict['scalar'][0, 0] == test_dict['scalar']
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_matlab_complex_arrays():
    """Test saving complex arrays to a MATLAB .mat file."""
    from mngs.io._save_modules._matlab import _save_matlab
    
    # Create test dictionary with complex arrays
    test_dict = {
        'complex_array': np.array([1+2j, 3+4j, 5+6j]),
        'mixed_array': np.array([[1, 2j], [3+4j, 5]])
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_matlab(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_dict = scipy.io.loadmat(temp_path, squeeze_me=True)
        
        # Check complex array
        # Note: MATLAB reshape may affect dimensions
        loaded_complex = loaded_dict['complex_array'].flatten() if loaded_dict['complex_array'].ndim > 1 else loaded_dict['complex_array']
        test_complex = test_dict['complex_array'].flatten() if test_dict['complex_array'].ndim > 1 else test_dict['complex_array']
        assert np.allclose(loaded_complex, test_complex)
        
        # Check mixed array
        assert np.allclose(loaded_dict['mixed_array'], test_dict['mixed_array'])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_matlab_with_different_dtypes():
    """Test saving arrays with different data types to a MATLAB .mat file."""
    from mngs.io._save_modules._matlab import _save_matlab
    
    # Create test dictionary with different dtypes
    test_dict = {
        'int_array': np.array([1, 2, 3], dtype=np.int32),
        'float_array': np.array([1.1, 2.2, 3.3], dtype=np.float64),
        'bool_array': np.array([True, False, True], dtype=np.bool_)
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the data
        _save_matlab(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_dict = scipy.io.loadmat(temp_path)
        
        # Check array contents (note: dtypes might be converted)
        # loadmat returns rows as the first dimension for 1D arrays
        assert np.array_equal(loaded_dict['int_array'].ravel(), test_dict['int_array'])
        assert np.allclose(loaded_dict['float_array'].ravel(), test_dict['float_array'])
        # Boolean arrays get converted to uint8 in MATLAB
        assert np.array_equal(loaded_dict['bool_array'].astype(bool).ravel(), test_dict['bool_array'])
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_matlab_empty_dict():
    """Test saving an empty dictionary to a MATLAB .mat file."""
    from mngs.io._save_modules._matlab import _save_matlab
    
    # Create an empty dictionary
    test_dict = {}
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the empty dictionary
        _save_matlab(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify it only contains metadata fields
        loaded_dict = scipy.io.loadmat(temp_path)
        
        # MATLAB files always include metadata fields
        assert '__header__' in loaded_dict
        assert '__version__' in loaded_dict
        assert '__globals__' in loaded_dict
        
        # No other fields should be present
        user_keys = [k for k in loaded_dict.keys() if not k.startswith('__')]
        assert len(user_keys) == 0
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_matlab.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:28:15 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_matlab.py
# 
# import scipy.io
# 
# 
# def _save_matlab(obj, spath):
#     """
#     Save a Python dictionary to a MATLAB .mat file.
#     
#     Parameters
#     ----------
#     obj : dict
#         Dictionary of arrays to save in MATLAB format.
#     spath : str
#         Path where the MATLAB file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     scipy.io.savemat(spath, obj)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_matlab.py
# --------------------------------------------------------------------------------
