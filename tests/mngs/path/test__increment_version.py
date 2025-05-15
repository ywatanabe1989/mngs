#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def test_increment_version_no_existing_files():
    """Test increment_version when no versioned files exist."""
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


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Source Code Reference (for maintenance):
# --------------------------------------------------------------------------------
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