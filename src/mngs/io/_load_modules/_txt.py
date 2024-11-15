#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:35 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_txt.py

import warnings


def _load_txt(lpath, **kwargs):
    """Load text file and return non-empty lines."""
    try:
        if not lpath.endswith((".txt", ".log", ".event", ".py", ".sh", "")):
            warnings.warn("File must have .txt, .log or .event extension")

        encoding = _check_encoding(lpath)
        with open(lpath, "r", encoding=encoding) as f:
            return [
                line.strip() for line in f.read().splitlines() if line.strip()
            ]
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(f"Error loading file {lpath}: {str(e)}")


def _check_encoding(file_path):
    """
    Check the encoding of a given file.

    This function attempts to read the file with different encodings
    to determine the correct one.

    Parameters:
    -----------
    file_path : str
        The path to the file to check.

    Returns:
    --------
    str
        The detected encoding of the file.

    Raises:
    -------
    IOError
        If the file cannot be read or the encoding cannot be determined.
    """
    import chardet

    with open(file_path, "rb") as file:
        raw_data = file.read()

    result = chardet.detect(raw_data)
    return result["encoding"]


# def _load_txt(lpath, **kwargs):
#     """Load text file and return non-empty lines."""
#     if not lpath.endswith(('.txt', '.log', '.event')):
#         raise ValueError("File must have .txt, .log or .event extension")
#     with open(lpath, "r") as f:
#         return [line.strip() for line in f.read().splitlines() if line.strip()]

# EOF

#         # Text
#         elif extension in [".txt", ".log", ".event"]:
#             with open(lpath, "r") as f:
#                 obj = [line.strip() for line in f.read().splitlines() if line.strip()]
#                 # obj = tqdm(f.read().splitlines(), desc=f"Reading {lpath}")


# # EOF


# def _load_text(lpath):
#     """
#     Load text from a file.

#     Parameters:
#     -----------
#     lpath : str
#         The path to the text file to be loaded.

#     Returns:
#     --------
#     str
#         The content of the text file as a string.

#     Raises:
#     -------
#     FileNotFoundError
#         If the specified file does not exist.
#     IOError
#         If there's an error reading the file.
#     """
#     with open(lpath, "r") as f:
#         return f.read()


#

# EOF
