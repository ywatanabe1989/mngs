#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-02 12:50:00 (ywatanabe)"
# File: ./tests/mngs/path/test__increment_version.py

import pytest
import os
import tempfile
import shutil
from pathlib import Path
=======
# Timestamp: "2025-05-15 03:00:18 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/path/test__increment_version.py
# ----------------------------------------
import os
import tempfile
import importlib.util
import shutil
# ----------------------------------------

import pytest

# Direct import from file path
inc_ver_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/path/_increment_version.py"))
spec = importlib.util.spec_from_file_location("_increment_version", inc_ver_module_path)
inc_ver_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inc_ver_module)
increment_version = inc_ver_module.increment_version
>>>>>>> origin/main


def test_increment_version_no_existing_files():
    """Test increment_version when no versioned files exist."""
<<<<<<< HEAD
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        expected = os.path.join(tmpdir, 'test_file_v001.txt')
        assert result == expected


def test_increment_version_single_existing_file():
    """Test increment_version with one existing versioned file."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an existing versioned file
        existing = os.path.join(tmpdir, 'test_file_v001.txt')
        Path(existing).touch()
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        expected = os.path.join(tmpdir, 'test_file_v002.txt')
        assert result == expected


def test_increment_version_multiple_existing_files():
    """Test increment_version with multiple existing versioned files."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple existing versioned files
        for i in [1, 2, 3, 5, 7]:  # Note: gap in sequence
            Path(os.path.join(tmpdir, f'test_file_v{i:03d}.txt')).touch()
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        expected = os.path.join(tmpdir, 'test_file_v008.txt')
        assert result == expected


def test_increment_version_custom_prefix():
    """Test increment_version with custom version prefix."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with custom prefix
        Path(os.path.join(tmpdir, 'test_file-ver001.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file-ver002.txt')).touch()
        
        result = increment_version(tmpdir, 'test_file', '.txt', version_prefix='-ver')
        
        expected = os.path.join(tmpdir, 'test_file-ver003.txt')
        assert result == expected


def test_increment_version_different_extensions():
    """Test increment_version correctly handles different extensions."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with different extensions
        Path(os.path.join(tmpdir, 'test_file_v001.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file_v001.csv')).touch()
        Path(os.path.join(tmpdir, 'test_file_v002.csv')).touch()
        
        # Should only consider .txt files
        result_txt = increment_version(tmpdir, 'test_file', '.txt')
        expected_txt = os.path.join(tmpdir, 'test_file_v002.txt')
        assert result_txt == expected_txt
        
        # Should only consider .csv files
        result_csv = increment_version(tmpdir, 'test_file', '.csv')
        expected_csv = os.path.join(tmpdir, 'test_file_v003.csv')
        assert result_csv == expected_csv


def test_increment_version_special_characters_in_filename():
    """Test increment_version with special characters in filename.
    
    The implementation uses re.escape() to handle special regex characters.
    However, glob patterns might not find files with certain special characters.
    """
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with filename containing dots (common case)
        fname = "test.file.name"
        existing_file = os.path.join(tmpdir, f'{fname}_v001.txt')
        Path(existing_file).touch()
        
        result = increment_version(tmpdir, fname, '.txt')
        expected = os.path.join(tmpdir, f'{fname}_v002.txt')
        assert result == expected
        
        # Test with parentheses
        fname2 = "test_file(1)"
        existing_file2 = os.path.join(tmpdir, f'{fname2}_v001.txt')
        Path(existing_file2).touch()
        
        result2 = increment_version(tmpdir, fname2, '.txt')
        expected2 = os.path.join(tmpdir, f'{fname2}_v002.txt')
        assert result2 == expected2


def test_increment_version_large_version_numbers():
    """Test increment_version with large version numbers."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create file with large version number
        Path(os.path.join(tmpdir, 'test_file_v999.txt')).touch()
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        expected = os.path.join(tmpdir, 'test_file_v1000.txt')
        assert result == expected


def test_increment_version_mixed_valid_invalid_files():
    """Test increment_version with mix of valid and invalid filenames."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mix of files
        Path(os.path.join(tmpdir, 'test_file_v001.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file_v002.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file_vABC.txt')).touch()  # Invalid version
        Path(os.path.join(tmpdir, 'test_file.txt')).touch()  # No version
        Path(os.path.join(tmpdir, 'other_file_v001.txt')).touch()  # Different base
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        expected = os.path.join(tmpdir, 'test_file_v003.txt')
        assert result == expected


def test_increment_version_empty_directory_path():
    """Test increment_version with empty string as directory."""
    from mngs.path._increment_version import increment_version
    
    # Should handle empty dirname gracefully
    result = increment_version('', 'test_file', '.txt')
    assert result == 'test_file_v001.txt'


def test_increment_version_nested_directory():
    """Test increment_version in nested directory structure."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directory
        nested_dir = os.path.join(tmpdir, 'sub1', 'sub2')
        os.makedirs(nested_dir)
        
        # Create versioned file in nested directory
        Path(os.path.join(nested_dir, 'test_file_v001.txt')).touch()
        
        result = increment_version(nested_dir, 'test_file', '.txt')
        
        expected = os.path.join(nested_dir, 'test_file_v002.txt')
        assert result == expected


def test_increment_version_compound_extension():
    """Test increment_version with compound extensions."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create file with compound extension
        Path(os.path.join(tmpdir, 'test_file_v001.tar.gz')).touch()
        
        result = increment_version(tmpdir, 'test_file', '.tar.gz')
        
        expected = os.path.join(tmpdir, 'test_file_v002.tar.gz')
        assert result == expected


def test_increment_version_similar_filenames():
    """Test increment_version doesn't confuse similar filenames."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with similar names
        Path(os.path.join(tmpdir, 'test_file_v001.txt')).touch()
        Path(os.path.join(tmpdir, 'test_file2_v005.txt')).touch()
        Path(os.path.join(tmpdir, 'prefix_test_file_v010.txt')).touch()
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        # Should only increment based on exact match
        expected = os.path.join(tmpdir, 'test_file_v002.txt')
        assert result == expected


def test_increment_version_zero_padded_versions():
    """Test increment_version maintains zero padding."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with different padding
        Path(os.path.join(tmpdir, 'test_file_v01.txt')).touch()  # 2 digits
        Path(os.path.join(tmpdir, 'test_file_v002.txt')).touch()  # 3 digits
        
        result = increment_version(tmpdir, 'test_file', '.txt')
        
        # Should use 3-digit padding as minimum
        expected = os.path.join(tmpdir, 'test_file_v003.txt')
        assert result == expected


def test_increment_version_with_dots_in_filename():
    """Test increment_version with dots in the filename."""
    from mngs.path._increment_version import increment_version
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Filename with dots
        fname = "test.file.name"
        Path(os.path.join(tmpdir, f'{fname}_v001.txt')).touch()
        
        result = increment_version(tmpdir, fname, '.txt')
        
        expected = os.path.join(tmpdir, f'{fname}_v002.txt')
        assert result == expected

=======
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Call increment_version with no existing files
        result = increment_version(temp_dir, "file", ".txt")
        
        # Check result
        expected = os.path.join(temp_dir, "file_v001.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_increment_version_with_existing_files():
    """Test increment_version with existing versioned files."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create existing versioned files
        for version in [1, 2, 5]:
            file_path = os.path.join(temp_dir, f"file_v{version:03d}.txt")
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Call increment_version
        result = increment_version(temp_dir, "file", ".txt")
        
        # Check result (should be version 006)
        expected = os.path.join(temp_dir, "file_v006.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_increment_version_with_custom_prefix():
    """Test increment_version with a custom version prefix."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create existing versioned files with custom prefix
        for version in [1, 3]:
            file_path = os.path.join(temp_dir, f"file-ver{version:03d}.txt")
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Call increment_version with custom prefix
        result = increment_version(temp_dir, "file", ".txt", version_prefix="-ver")
        
        # Check result (should be version 004)
        expected = os.path.join(temp_dir, "file-ver004.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_increment_version_ignores_unrelated_files():
    """Test that increment_version ignores unrelated files."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create some versioned files
        for version in [1, 2]:
            file_path = os.path.join(temp_dir, f"file_v{version:03d}.txt")
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Create some unrelated files
        for name in ["other_v001.txt", "file_other.txt", "file_vXXX.txt"]:
            file_path = os.path.join(temp_dir, name)
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Call increment_version
        result = increment_version(temp_dir, "file", ".txt")
        
        # Check result (should only count the properly versioned files)
        expected = os.path.join(temp_dir, "file_v003.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_increment_version_with_different_extensions():
    """Test increment_version handles file extensions correctly."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create versioned files with different extensions
        # These should be ignored
        for version in [1, 2, 5]:
            file_path = os.path.join(temp_dir, f"file_v{version:03d}.csv")
            with open(file_path, "w") as f:
                f.write("test content")
        
        # Create one versioned file with the correct extension
        file_path = os.path.join(temp_dir, "file_v003.txt")
        with open(file_path, "w") as f:
            f.write("test content")
        
        # Call increment_version for .txt files
        result = increment_version(temp_dir, "file", ".txt")
        
        # Check result (should only count the .txt files)
        expected = os.path.join(temp_dir, "file_v004.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_increment_version_with_nonexistent_directory():
    """Test increment_version with a directory that doesn't exist yet."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    nonexistent_dir = os.path.join(temp_dir, "subdir")
    
    try:
        # Call increment_version with a nonexistent directory
        # This should work because the glob pattern will just not match anything
        result = increment_version(nonexistent_dir, "file", ".txt")
        
        # Check result (should be version 001)
        expected = os.path.join(nonexistent_dir, "file_v001.txt")
        assert result == expected
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_increment_version.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 19:45:32 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_increment_version.py
#
# import os
# import re
# from glob import glob
#
#
# def increment_version(dirname, fname, ext, version_prefix="_v"):
#     """
#     Generate the next version of a filename based on existing versioned files.
#
#     This function searches for files in the given directory that match the pattern:
#     {fname}{version_prefix}{number}{ext} and returns the path for the next version.
#
#     Parameters:
#     -----------
#     dirname : str
#         The directory to search in and where the new file will be created.
#     fname : str
#         The base filename without version number or extension.
#     ext : str
#         The file extension, including the dot (e.g., '.txt').
#     version_prefix : str, optional
#         The prefix used before the version number. Default is '_v'.
#
#     Returns:
#     --------
#     str
#         The full path for the next version of the file.
#
#     Example:
#     --------
#     >>> increment_version('/path/to/dir', 'myfile', '.txt')
#     '/path/to/dir/myfile_v004.txt'
#
#     Notes:
#     ------
#     - If no existing versioned files are found, it starts with version 001.
#     - The version number is always formatted with at least 3 digits.
#     """
#     # Create a regex pattern to match the version number in the filename
#     version_pattern = re.compile(
#         rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
#     )
#
#     # Construct the glob pattern to find all files that match the pattern
#     glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")
#
#     # Use glob to find all files that match the pattern
#     files = glob(glob_pattern)
#
#     # Initialize the highest version number
#     highest_version = 0
#     base, suffix = None, None
#
#     # Loop through the files to find the highest version number
#     for file in files:
#         filename = os.path.basename(file)
#         match = version_pattern.search(filename)
#         if match:
#             base, version_str, suffix = match.groups()
#             version_num = int(version_str)
#             if version_num > highest_version:
#                 highest_version = version_num
#
#     # If no versioned files were found, use the provided filename and extension
#     if base is None or suffix is None:
#         base = f"{fname}{version_prefix}"
#         suffix = ext
#         highest_version = 0  # No previous versions
#
#     # Increment the highest version number
#     next_version_number = highest_version + 1
#
#     # Format the next version number with the same number of digits as the original
#     next_version_str = f"{base}{next_version_number:03d}{suffix}"
#
#     # Combine the directory and new filename to create the full path
#     next_filepath = os.path.join(dirname, next_version_str)
#
#     return next_filepath
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_increment_version.py
# --------------------------------------------------------------------------------
