#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    assert dirname == '../data/01/day1/split_octave/2kHz_mat/'
    assert fname == 'tt8-2'
    assert ext == '.mat'


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
    assert fname == 'README'
    assert ext == ''


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


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Source Code Reference (for maintenance):
# --------------------------------------------------------------------------------
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