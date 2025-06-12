#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Time-stamp: "2024-11-02 19:54:02 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/path/test__getsize.py

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


def test_getsize_existing_file():
    """Test getsize with existing file."""
    from mngs.path._getsize import getsize
    
    # Create temporary file with known content
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        content = "Hello, World!"
        f.write(content)
        temp_path = f.name
    
    try:
        size = getsize(temp_path)
        assert size == len(content.encode())
        assert isinstance(size, int)
    finally:
        os.unlink(temp_path)


def test_getsize_empty_file():
    """Test getsize with empty file."""
    from mngs.path._getsize import getsize
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    try:
        size = getsize(temp_path)
        assert size == 0
    finally:
        os.unlink(temp_path)


def test_getsize_large_file():
    """Test getsize with larger file."""
    from mngs.path._getsize import getsize
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        # Write 1MB of data
        data = b'x' * (1024 * 1024)
        f.write(data)
        temp_path = f.name
    
    try:
        size = getsize(temp_path)
        assert size == 1024 * 1024
    finally:
        os.unlink(temp_path)


def test_getsize_nonexistent_file():
    """Test getsize with non-existent file."""
    from mngs.path._getsize import getsize
    
    nonexistent_path = "/path/that/does/not/exist/file.txt"
    size = getsize(nonexistent_path)
    assert np.isnan(size)


def test_getsize_directory():
    """Test getsize with directory."""
    from mngs.path._getsize import getsize
    
    with tempfile.TemporaryDirectory() as temp_dir:
        size = getsize(temp_dir)
        # Directory size varies by filesystem
        assert isinstance(size, int)
        assert size >= 0


def test_getsize_symlink():
    """Test getsize with symlink."""
    from mngs.path._getsize import getsize
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Symlink target content")
        target_path = f.name
    
    with tempfile.TemporaryDirectory() as temp_dir:
        symlink_path = os.path.join(temp_dir, "symlink")
        
        try:
            os.symlink(target_path, symlink_path)
            
            # getsize should return size of symlink itself, not target
            size = getsize(symlink_path)
            assert isinstance(size, int)
            assert size > 0
        finally:
            os.unlink(target_path)


def test_getsize_pathlib_path():
    """Test getsize with pathlib.Path object."""
    from mngs.path._getsize import getsize
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Pathlib test")
        temp_path = Path(f.name)
    
    try:
        size = getsize(temp_path)
        assert size == len("Pathlib test".encode())
    finally:
        os.unlink(str(temp_path))


def test_getsize_binary_file():
    """Test getsize with binary file."""
    from mngs.path._getsize import getsize
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        binary_data = bytes(range(256))
        f.write(binary_data)
        temp_path = f.name
    
    try:
        size = getsize(temp_path)
        assert size == 256
    finally:
        os.unlink(temp_path)


def test_getsize_unicode_filename():
    """Test getsize with unicode filename."""
    from mngs.path._getsize import getsize
    
    with tempfile.TemporaryDirectory() as temp_dir:
        unicode_path = os.path.join(temp_dir, "文件名.txt")
        
        with open(unicode_path, 'w', encoding='utf-8') as f:
            f.write("Unicode filename test")
        
        size = getsize(unicode_path)
        assert size == len("Unicode filename test".encode('utf-8'))


def test_getsize_permission_error():
    """Test getsize with permission error."""
    from mngs.path._getsize import getsize
    
    with patch('os.path.exists', return_value=True):
        with patch('os.path.getsize', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                getsize("/restricted/file")


def test_getsize_special_files():
    """Test getsize with special files like /dev/null."""
    from mngs.path._getsize import getsize
    
    if os.path.exists("/dev/null"):
        size = getsize("/dev/null")
        assert size == 0


def test_getsize_relative_path():
    """Test getsize with relative path."""
    from mngs.path._getsize import getsize
    
    current_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        try:
            # Create file with relative path
            with open("relative_test.txt", 'w') as f:
                f.write("Relative path test")
            
            size = getsize("relative_test.txt")
            assert size == len("Relative path test".encode())
        finally:
            os.chdir(current_dir)


def test_getsize_spaces_in_path():
    """Test getsize with spaces in path."""
    from mngs.path._getsize import getsize
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path_with_spaces = os.path.join(temp_dir, "file with spaces.txt")
        
        with open(path_with_spaces, 'w') as f:
            f.write("Spaces in filename")
        
        size = getsize(path_with_spaces)
        assert size == len("Spaces in filename".encode())


def test_getsize_empty_string():
    """Test getsize with empty string path."""
    from mngs.path._getsize import getsize
    
    size = getsize("")
    assert np.isnan(size)


def test_getsize_none_path():
    """Test getsize with None path."""
    from mngs.path._getsize import getsize
    
    with pytest.raises(TypeError):
        getsize(None)

=======
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
>>>>>>> origin/main

if __name__ == "__main__":
    pytest.main([__file__])

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
<<<<<<< HEAD
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------
=======
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
