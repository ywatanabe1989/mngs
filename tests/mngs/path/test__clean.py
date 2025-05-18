#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-15 00:55:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/path/test__clean.py
# ----------------------------------------
import os
import sys
import importlib.util
# ----------------------------------------

import pytest

# Direct import from file path
clean_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/path/_clean.py"))
spec = importlib.util.spec_from_file_location("_clean", clean_module_path)
clean_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_module)
clean = clean_module.clean


def test_clean_basic_functionality():
    """Test basic functionality of clean function."""
    # Test no changes needed
    assert clean("normal/path/no/changes") == "normal/path/no/changes"
    
    # Test empty input
    assert clean("") == ""


def test_clean_removes_redundant_path_separators():
    """Test that clean removes redundant path separators."""
    # Test double slashes
    assert clean("path//with//double//slashes") == "path/with/double/slashes"
    
    # Test multiple slashes
    assert clean("path///with/////many//////slashes") == "path/with/many/slashes"


def test_clean_normalizes_current_directory_references():
    """Test that clean normalizes references to current directory."""
    # Test current directory references
    assert clean("path/./with/./dot/./references") == "path/with/dot/references"
    
    # Test mixed current directory and double slashes
    assert clean("path/.//with//./mixed/.//patterns") == "path/with/mixed/patterns"


def test_clean_replaces_spaces_with_underscores():
    """Test that clean replaces spaces with underscores."""
    # Test spaces
    assert clean("path with spaces") == "path_with_spaces"
    
    # Test spaces with other transformations
    # Note: The function first replaces spaces, then normalizes paths,
    # so the ./ is transformed into _ before normalization
    assert clean("path/./ with //spaces") == "path/_with_/spaces"


def test_clean_handles_edge_cases():
    """Test that clean handles various edge cases correctly."""
    # Test leading and trailing slashes - should preserve them
    assert clean("/absolute/./path//") == "/absolute/path/"
    
    # Test only spaces
    assert clean("   ") == "___"
    
    # Test only patterns to clean
    assert clean("/.//.//.//") == "/"


def test_clean_multiple_applications():
    """Test that applying clean multiple times is idempotent."""
    # Clean messy path
    messy_path = "//path/.//with///.///lots/.//of///mess//"
    cleaned_once = clean(messy_path)
    cleaned_twice = clean(cleaned_once)
    
    # Should be the same after first cleaning
    assert cleaned_once == "/path/with/lots/of/mess/"
    
    # Further applications should not change result
    assert cleaned_twice == cleaned_once

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_clean.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-05-15 00:55:30 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_clean.py
# 
# import os
# 
# def clean(path_string):
#     """Cleans and normalizes a file system path string.
# 
#     Example
#     -------
#     >>> clean('/home/user/./folder/../file.txt')
#     '/home/user/file.txt'
#     >>> clean('path/./to//file.txt')
#     'path/to/file.txt'
#     >>> clean('path with spaces')
#     'path_with_spaces'
# 
#     Parameters
#     ----------
#     path_string : str
#         File path to clean
# 
#     Returns
#     -------
#     str
#         Normalized path string
#     """
#     if not path_string:
#         return ""
#     
#     # Remember if path ends with a slash (indicating a directory)
#     is_directory = path_string.endswith("/")
#     
#     # Replace spaces with underscores
#     path_string = path_string.replace(" ", "_")
#     
#     # Use normpath to handle ../ and ./ references
#     cleaned_path = os.path.normpath(path_string)
#     
#     # Replace multiple slashes with single slash
#     while "//" in cleaned_path:
#         cleaned_path = cleaned_path.replace("//", "/")
#         
#     # Restore trailing slash if it was a directory
#     if is_directory and not cleaned_path.endswith("/"):
#         cleaned_path += "/"
#         
#     return cleaned_path
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_clean.py
# --------------------------------------------------------------------------------
