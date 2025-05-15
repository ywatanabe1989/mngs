#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-15 01:30:45 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/str/test__readable_bytes.py
# ----------------------------------------
import os
import sys
import importlib.util
# ----------------------------------------

import pytest

# Direct import from file path
readable_bytes_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/str/_readable_bytes.py"))
spec = importlib.util.spec_from_file_location("_readable_bytes", readable_bytes_module_path)
readable_bytes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readable_bytes_module)
readable_bytes = readable_bytes_module.readable_bytes


def test_readable_bytes_basic_functionality():
    """Test basic functionality of readable_bytes function."""
    # Test zero bytes
    assert readable_bytes(0) == "0.0 B"
    
    # Test basic conversions
    assert readable_bytes(1) == "1.0 B"
    assert readable_bytes(1023) == "1023.0 B"


def test_readable_bytes_kilobytes():
    """Test kilobyte conversions."""
    # Test kilobyte boundary
    assert readable_bytes(1024) == "1.0 KiB"
    assert readable_bytes(1536) == "1.5 KiB"
    assert readable_bytes(2048) == "2.0 KiB"
    
    # Test values close to the next boundary
    assert readable_bytes(1023 * 1024) == "1023.0 KiB"


def test_readable_bytes_megabytes_and_higher():
    """Test megabyte and higher conversions."""
    # Test megabyte conversions
    assert readable_bytes(1024 * 1024) == "1.0 MiB"
    assert readable_bytes(1.5 * 1024 * 1024) == "1.5 MiB"
    
    # Test gigabyte conversions
    assert readable_bytes(1024 * 1024 * 1024) == "1.0 GiB"
    
    # Test terabyte conversions
    assert readable_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TiB"


def test_readable_bytes_with_custom_suffix():
    """Test readable_bytes with a custom suffix."""
    # Test custom suffixes
    assert readable_bytes(1024, suffix="iB") == "1.0 KiiB"
    assert readable_bytes(1024 * 1024, suffix="b") == "1.0 Mib"
    assert readable_bytes(1024, suffix="") == "1.0 Ki"


def test_readable_bytes_with_negative_values():
    """Test readable_bytes with negative values."""
    # Test negative values
    # The function uses abs() for comparison but preserves sign in output
    assert readable_bytes(-1024) == "-1.0 KiB"
    assert readable_bytes(-2048) == "-2.0 KiB"


def test_readable_bytes_with_very_large_values():
    """Test readable_bytes with very large values that exceed standard prefixes."""
    # Test extremely large values
    yotta_bytes = 1024 ** 8  # 1 YiB (yottabyte)
    assert readable_bytes(yotta_bytes) == "1.0 YiB"
    
    # Test value larger than yottabyte
    beyond_yotta = 1024 ** 9  # 1024 YiB
    assert readable_bytes(beyond_yotta) == "1024.0 YiB"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Source Code Reference (for maintenance):
# --------------------------------------------------------------------------------
# def readable_bytes(num, suffix="B"):
#     """Convert a number of bytes to a human-readable format.
#
#     Parameters
#     ----------
#     num : int
#         The number of bytes to convert.
#     suffix : str, optional
#         The suffix to append to the unit (default is "B" for bytes).
#
#     Returns
#     -------
#     str
#         A human-readable string representation of the byte size.
#
#     Example
#     -------
#     >>> readable_bytes(1024)
#     '1.0 KiB'
#     >>> readable_bytes(1048576)
#     '1.0 MiB'
#     >>> readable_bytes(1073741824)
#     '1.0 GiB'
#     """
#     for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, "Yi", suffix)