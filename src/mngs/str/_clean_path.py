#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 11:24:24 (ywatanabe)"
# File: ./mngs_repo/src/mngs/str/_clean_path.py

"""
Functionality:
    - Cleans and normalizes file system paths
Input:
    - File path string containing redundant separators or current directory references
Output:
    - Cleaned path string with normalized separators
Prerequisites:
    - Python's os.path module
"""

"""Imports"""
import os

"""Functions & Classes"""
def clean(path_string: str) -> str:
    """Cleans and normalizes a file system path string.

    Example
    -------
    >>> clean('/home/user/./folder/../file.txt')
    '/home/user/file.txt'
    >>> clean('path/./to//file.txt')
    'path/to/file.txt'

    Parameters
    ----------
    path_string : str
        File path to clean

    Returns
    -------
    str
        Normalized path string
    """
    try:
        if not isinstance(path_string, str):
            raise TypeError("Input must be a string")

        # Normalize path separators
        cleaned_path = os.path.normpath(path_string)

        # Remove redundant separators
        cleaned_path = os.path.normpath(cleaned_path)

        return cleaned_path

    except Exception as error:
        raise ValueError(f"Path cleaning failed: {str(error)}")

# EOF
