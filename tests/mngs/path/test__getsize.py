#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-15 02:45:22 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/path/test__getsize.py
# ----------------------------------------
import os
import tempfile
import importlib.util
# ----------------------------------------

import pytest
import numpy as np

# Direct import from file path
getsize_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/path/_getsize.py"))
spec = importlib.util.spec_from_file_location("_getsize", getsize_module_path)
getsize_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(getsize_module)
getsize = getsize_module.getsize


def test_getsize_with_existing_file():
    """Test getsize function with an existing file."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b'test content')
        tmp_path = tmp.name
    
    try:
        # Get file size using the function
        file_size = getsize(tmp_path)
        
        # Get file size directly for comparison
        expected_size = os.path.getsize(tmp_path)
        
        # Assert sizes match
        assert file_size == expected_size
        assert file_size == 12  # 'test content' is 12 bytes
    finally:
        # Clean up: remove the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_getsize_with_empty_file():
    """Test getsize function with an empty file."""
    # Create a temporary empty file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Get file size using the function
        file_size = getsize(tmp_path)
        
        # Assert size is 0
        assert file_size == 0
    finally:
        # Clean up: remove the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_getsize_with_nonexistent_file():
    """Test getsize function with a nonexistent file."""
    # Create a path to a file that doesn't exist
    nonexistent_path = '/path/to/nonexistent/file.txt'
    
    # Ensure the file really doesn't exist
    if os.path.exists(nonexistent_path):
        pytest.skip("Test file unexpectedly exists")
    
    # Get file size using the function
    file_size = getsize(nonexistent_path)
    
    # Assert size is NaN
    assert np.isnan(file_size)
    

def test_getsize_with_directory():
    """Test getsize function with a directory."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get directory size
        dir_size = getsize(temp_dir)
        
        # The size of a directory is platform-dependent
        # We just check that it's a number and not NaN
        assert not np.isnan(dir_size)
        assert isinstance(dir_size, (int, float))
    finally:
        # Clean up: remove the temporary directory
        os.rmdir(temp_dir)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 19:54:02 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_getsize.py
# 
# import os
# 
# import numpy as np
# 
# 
# def getsize(path):
#     if os.path.exists(path):
#         return os.path.getsize(path)
#     else:
#         return np.nan
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------
