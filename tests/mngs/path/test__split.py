#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-02 13:10:00 (ywatanabe)"
# File: ./tests/mngs/path/test__split.py

import pytest
import os
from pathlib import Path


def test_split_basic():
    """Test split with basic file path."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/file.txt')
    
    assert dirname == '/path/to/'
    assert fname == 'file'
    assert ext == '.txt'


def test_split_relative_path():
    """Test split with relative path."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
    
=======
# Timestamp: "2025-05-15 02:15:34 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/path/test__split.py
# ----------------------------------------
import os
import sys
import importlib.util
# ----------------------------------------

import pytest

# Direct import from file path
split_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/path/_split.py"))
spec = importlib.util.spec_from_file_location("_split", split_module_path)
split_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_module)
split = split_module.split


def test_split_basic_functionality():
    """Test basic functionality of the split function."""
    # Test case from the docstring
    dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
>>>>>>> origin/main
    assert dirname == '../data/01/day1/split_octave/2kHz_mat/'
    assert fname == 'tt8-2'
    assert ext == '.mat'


<<<<<<< HEAD
def test_split_no_extension():
    """Test split with file without extension."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/README')
    
    assert dirname == '/path/to/'
=======
def test_split_absolute_path():
    """Test splitting of absolute paths."""
    # Test absolute path
    dirname, fname, ext = split('/home/user/documents/file.txt')
    assert dirname == '/home/user/documents/'
    assert fname == 'file'
    assert ext == '.txt'


def test_split_relative_path():
    """Test splitting of relative paths."""
    # Test relative path
    dirname, fname, ext = split('docs/report.pdf')
    assert dirname == 'docs/'
    assert fname == 'report'
    assert ext == '.pdf'


def test_split_no_extension():
    """Test splitting of paths with no file extension."""
    # Test file with no extension
    dirname, fname, ext = split('/home/user/notes/README')
    assert dirname == '/home/user/notes/'
>>>>>>> origin/main
    assert fname == 'README'
    assert ext == ''


<<<<<<< HEAD
def test_split_multiple_dots():
    """Test split with filename containing multiple dots."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/file.backup.tar.gz')
    
    assert dirname == '/path/to/'
    assert fname == 'file.backup.tar'
    assert ext == '.gz'


def test_split_hidden_file():
    """Test split with hidden file (starting with dot)."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/home/user/.bashrc')
    
    assert dirname == '/home/user/'
    assert fname == '.bashrc'
    assert ext == ''


def test_split_hidden_file_with_extension():
    """Test split with hidden file that has extension."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/home/user/.config.yaml')
    
    assert dirname == '/home/user/'
    assert fname == '.config'
    assert ext == '.yaml'


def test_split_root_directory():
    """Test split with file in root directory."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/file.txt')
    
    assert dirname == '/'
    assert fname == 'file'
    assert ext == '.txt'


def test_split_current_directory():
    """Test split with file in current directory."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('file.txt')
    
    assert dirname == '/'  # When no directory, returns '/'
    assert fname == 'file'
    assert ext == '.txt'


def test_split_trailing_slash():
    """Test split with path ending in slash (directory)."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/directory/')
    
    assert dirname == '/path/to/directory/'
    assert fname == ''
    assert ext == ''


def test_split_windows_path():
    """Test split with Windows-style path."""
    from mngs.path._split import split
    
    # Note: os.path handles this based on the OS
    if os.name == 'nt':
        dirname, fname, ext = split('C:\\Users\\user\\file.txt')
        assert dirname == 'C:\\Users\\user\\'
        assert fname == 'file'
        assert ext == '.txt'
    else:
        # On Unix, backslashes are part of filename
        dirname, fname, ext = split('C:\\Users\\user\\file.txt')
        # Behavior depends on OS


def test_split_empty_path():
    """Test split with empty path."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('')
    
    assert dirname == '/'
    assert fname == ''
    assert ext == ''


def test_split_special_characters():
    """Test split with special characters in filename."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/file[with]special(chars).txt')
    
    assert dirname == '/path/to/'
    assert fname == 'file[with]special(chars)'
    assert ext == '.txt'


def test_split_unicode_characters():
    """Test split with unicode characters."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path/to/ファイル.txt')
    
    assert dirname == '/path/to/'
    assert fname == 'ファイル'
    assert ext == '.txt'


def test_split_spaces_in_path():
    """Test split with spaces in path and filename."""
    from mngs.path._split import split
    
    dirname, fname, ext = split('/path with spaces/file name.txt')
    
    assert dirname == '/path with spaces/'
    assert fname == 'file name'
    assert ext == '.txt'


def test_split_double_extension():
    """Test split behavior with double extensions."""
    from mngs.path._split import split
    
    # Only the last extension is considered
    dirname, fname, ext = split('/path/to/archive.tar.gz')
    
    assert dirname == '/path/to/'
    assert fname == 'archive.tar'
    assert ext == '.gz'

=======
def test_split_filename_only():
    """Test splitting of filenames without directories."""
    # Test filename only
    dirname, fname, ext = split('config.json')
    assert dirname == '/'  # Current directory is represented as '/'
    assert fname == 'config'
    assert ext == '.json'


def test_split_hidden_file():
    """Test splitting of hidden files."""
    # Test hidden file
    dirname, fname, ext = split('/home/user/.bashrc')
    assert dirname == '/home/user/'
    assert fname == '.bashrc'
    assert ext == ''
    
    # Test hidden file with extension
    dirname, fname, ext = split('/home/user/.config.json')
    assert dirname == '/home/user/'
    assert fname == '.config'
    assert ext == '.json'


def test_split_multiple_extensions():
    """Test splitting of files with multiple extensions."""
    # Test file with multiple extensions (only the last one should be returned)
    dirname, fname, ext = split('/data/archive.tar.gz')
    assert dirname == '/data/'
    assert fname == 'archive.tar'
    assert ext == '.gz'


def test_split_with_empty_string():
    """Test splitting an empty string."""
    # Test empty string
    dirname, fname, ext = split('')
    assert dirname == '/'  # Function returns '/' for empty string dirname
    assert fname == ''
    assert ext == ''
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_split.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:18:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_split.py
#
# import os
#
# def split(fpath):
#     """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
#     Example:
#         dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
#         print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
#         print(fname) # 'tt8-2'
#         print(ext) # '.mat'
#     """
#     dirname = os.path.dirname(fpath) + "/"
#     base = os.path.basename(fpath)
#     fname, ext = os.path.splitext(base)
#     return dirname, fname, ext
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_split.py
# --------------------------------------------------------------------------------
