#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Time-stamp: "2024-11-08 05:54:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/path/test__clean.py

"""
Tests for path cleaning functionality.
"""

import pytest
from mngs.path._clean import clean


class TestClean:
    """Test clean function."""
    
    def test_clean_single_dot_slash(self):
        """Test removal of /./ sequences."""
        assert clean("/home/./user/./file.txt") == "/home/user/file.txt"
        assert clean("./file.txt") == "./file.txt"  # Leading ./ is preserved
        assert clean("/././file.txt") == "/file.txt"
    
    def test_clean_double_slash(self):
        """Test removal of // sequences."""
        assert clean("/home//user//file.txt") == "/home/user/file.txt"
        assert clean("//network/share") == "/network/share"
        assert clean("file://path") == "file:/path"  # Be careful with protocols
    
    def test_clean_spaces_to_underscores(self):
        """Test replacement of spaces with underscores."""
        assert clean("/home/user/my file.txt") == "/home/user/my_file.txt"
        assert clean("file with spaces.txt") == "file_with_spaces.txt"
        assert clean("multiple   spaces") == "multiple___spaces"
    
    def test_clean_combined_issues(self):
        """Test cleaning paths with multiple issues."""
        assert clean("/home/./user//my file.txt") == "/home/user/my_file.txt"
        assert clean("./path//to/./some file") == "./path/to/some_file"
        assert clean("//server/./share//my folder/./file") == "/server/share/my_folder/file"
    
    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        assert clean("") == ""
    
    def test_clean_already_clean(self):
        """Test that clean paths remain unchanged."""
        assert clean("/home/user/file.txt") == "/home/user/file.txt"
        assert clean("relative/path/file.txt") == "relative/path/file.txt"
    
    def test_clean_windows_style_paths(self):
        """Test cleaning Windows-style paths."""
        # Note: This function doesn't handle backslashes, but tests behavior
        assert clean("C://Users//John Doe//Documents") == "C:/Users/John_Doe/Documents"
        assert clean("D:/./Projects//my project") == "D:/Projects/my_project"
    
    def test_clean_special_cases(self):
        """Test special edge cases."""
        # Multiple consecutive replacements
        assert clean("/.//./") == "/"
        assert clean("// // //") == "/_/_/"  # Spaces become underscores first
        
        # Trailing slashes
        assert clean("/home/user//") == "/home/user/"
        assert clean("/home/user/./") == "/home/user/"
    
    def test_clean_unicode_paths(self):
        """Test cleaning paths with unicode characters."""
        assert clean("/home/user/ñoño file.txt") == "/home/user/ñoño_file.txt"
        assert clean("/путь//к/./файлу") == "/путь/к/файлу"
        assert clean("/文件夹/./子 文件夹//文件.txt") == "/文件夹/子_文件夹/文件.txt"
    
    def test_clean_order_of_operations(self):
        """Test that replacements happen in the correct order."""
        # The function replaces in order: /./ -> /, // -> /, space -> _
        # This tests that order doesn't cause issues
        assert clean("/ ./ /") == "/_/"  # Space becomes _ after /./ is cleaned
        assert clean("//  //") == "/__/"  # Spaces replaced after // cleaned
    
    def test_clean_repeated_application(self):
        """Test that applying clean multiple times is idempotent."""
        path = "/home/./user//my file.txt"
        cleaned_once = clean(path)
        cleaned_twice = clean(cleaned_once)
        assert cleaned_once == cleaned_twice == "/home/user/my_file.txt"
    
    def test_clean_preserves_important_sequences(self):
        """Test that important sequences are preserved."""
        # Single dots and slashes that shouldn't be removed
        assert clean("../file.txt") == "../file.txt"
        assert clean("./script.py") == "./script.py"
        assert clean("/") == "/"
        assert clean(".") == "."
    
    def test_clean_network_paths(self):
        """Test cleaning network paths."""
        assert clean("\\\\server\\share\\file.txt") == "\\\\server\\share\\file.txt"  # UNC paths unchanged
        assert clean("//server/./share//my file") == "/server/share/my_file"
    
    def test_clean_url_like_paths(self):
        """Test paths that look like URLs."""
        # Note: This function may not handle URLs correctly
        assert clean("http://example.com/./path") == "http:/example.com/path"
        assert clean("file:///home//user") == "file:/home/user"
    
    def test_clean_performance_with_long_paths(self):
        """Test performance with very long paths."""
        long_path = "/home" + "/./user" * 100 + "//file.txt"
        result = clean(long_path)
        assert "/./" not in result
        assert "//" not in result
        assert result.startswith("/home")
        assert result.endswith("file.txt")

=======
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
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__)])
=======
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
>>>>>>> origin/main
