#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Time-stamp: "2025-06-11 03:00:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/str/test__readable_bytes.py

"""Comprehensive tests for readable bytes formatting functionality.

This module tests the readable_bytes function which converts byte counts
to human-readable formats using binary prefixes (KiB, MiB, GiB, etc.).
"""

import pytest
import os
import sys
from typing import List, Tuple
import math
import decimal
from decimal import Decimal


class TestReadableBytesBasic:
    """Basic functionality tests for readable_bytes."""
    
    def test_zero_bytes(self):
        """Test formatting of zero bytes."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(0) == "0.0 B"
        assert readable_bytes(0.0) == "0.0 B"
        assert readable_bytes(-0) == "0.0 B"
    
    def test_single_bytes(self):
        """Test formatting of single byte values."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(1) == "1.0 B"
        assert readable_bytes(2) == "2.0 B"
        assert readable_bytes(10) == "10.0 B"
        assert readable_bytes(100) == "100.0 B"
        assert readable_bytes(999) == "999.0 B"
    
    def test_boundary_values(self):
        """Test values at unit boundaries."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Just below 1 KiB
        assert readable_bytes(1023) == "1023.0 B"
        
        # Exactly 1 KiB
        assert readable_bytes(1024) == "1.0 KiB"
        
        # Just above 1 KiB
        assert readable_bytes(1025) == "1.0 KiB"
    
    def test_negative_bytes(self):
        """Test formatting of negative byte values."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(-1) == "-1.0 B"
        assert readable_bytes(-1024) == "-1.0 KiB"
        assert readable_bytes(-1048576) == "-1.0 MiB"
        assert readable_bytes(-5242880) == "-5.0 MiB"


class TestReadableBytesUnits:
    """Test all binary unit conversions."""
    
    def test_kibibytes(self):
        """Test KiB (kibibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        kb = 1024
        assert readable_bytes(1 * kb) == "1.0 KiB"
        assert readable_bytes(1.5 * kb) == "1.5 KiB"
        assert readable_bytes(2 * kb) == "2.0 KiB"
        assert readable_bytes(10 * kb) == "10.0 KiB"
        assert readable_bytes(100 * kb) == "100.0 KiB"
        assert readable_bytes(1023 * kb) == "1023.0 KiB"
    
    def test_mebibytes(self):
        """Test MiB (mebibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        mb = 1024 ** 2
        assert readable_bytes(1 * mb) == "1.0 MiB"
        assert readable_bytes(1.5 * mb) == "1.5 MiB"
        assert readable_bytes(5.5 * mb) == "5.5 MiB"
        assert readable_bytes(10 * mb) == "10.0 MiB"
        assert readable_bytes(100 * mb) == "100.0 MiB"
        assert readable_bytes(1023 * mb) == "1023.0 MiB"
    
    def test_gibibytes(self):
        """Test GiB (gibibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        gb = 1024 ** 3
        assert readable_bytes(1 * gb) == "1.0 GiB"
        assert readable_bytes(2.3 * gb) == "2.3 GiB"
        assert readable_bytes(10 * gb) == "10.0 GiB"
        assert readable_bytes(100 * gb) == "100.0 GiB"
        assert readable_bytes(512 * gb) == "512.0 GiB"
    
    def test_tebibytes(self):
        """Test TiB (tebibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        tb = 1024 ** 4
        assert readable_bytes(1 * tb) == "1.0 TiB"
        assert readable_bytes(1.5 * tb) == "1.5 TiB"
        assert readable_bytes(10 * tb) == "10.0 TiB"
        assert readable_bytes(100 * tb) == "100.0 TiB"
    
    def test_pebibytes(self):
        """Test PiB (pebibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        pb = 1024 ** 5
        assert readable_bytes(1 * pb) == "1.0 PiB"
        assert readable_bytes(2.5 * pb) == "2.5 PiB"
        assert readable_bytes(10 * pb) == "10.0 PiB"
    
    def test_exbibytes(self):
        """Test EiB (exbibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        eb = 1024 ** 6
        assert readable_bytes(1 * eb) == "1.0 EiB"
        assert readable_bytes(3.7 * eb) == "3.7 EiB"
    
    def test_zebibytes(self):
        """Test ZiB (zebibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        zb = 1024 ** 7
        assert readable_bytes(1 * zb) == "1.0 ZiB"
        assert readable_bytes(5.2 * zb) == "5.2 ZiB"
    
    def test_yobibytes(self):
        """Test YiB (yobibyte) formatting."""
        from mngs.str._readable_bytes import readable_bytes
        
        yb = 1024 ** 8
        assert readable_bytes(1 * yb) == "1.0 YiB"
        assert readable_bytes(10 * yb) == "10.0 YiB"
        assert readable_bytes(1000 * yb) == "1000.0 YiB"


class TestReadableBytesPrecision:
    """Test formatting precision and rounding."""
    
    def test_decimal_precision(self):
        """Test that values are formatted to 1 decimal place."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Test various values that should round
        test_cases = [
            (1536, "1.5 KiB"),      # 1536/1024 = 1.5
            (1434, "1.4 KiB"),      # 1434/1024 ≈ 1.4
            (1638, "1.6 KiB"),      # 1638/1024 ≈ 1.6
            (1126, "1.1 KiB"),      # 1126/1024 ≈ 1.1
            (2867, "2.8 KiB"),      # 2867/1024 ≈ 2.8
        ]
        
        for byte_count, expected in test_cases:
            assert readable_bytes(byte_count) == expected
    
    def test_rounding_behavior(self):
        """Test rounding behavior for edge cases."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Test rounding near .5
        kb = 1024
        
        # 1.449 KiB should round to 1.4
        assert readable_bytes(int(1.449 * kb)) == "1.4 KiB"
        
        # 1.451 KiB should round to 1.5
        assert readable_bytes(int(1.451 * kb)) == "1.5 KiB"
        
        # 1.95 KiB should round to 2.0 (or 1.9)
        result = readable_bytes(int(1.95 * kb))
        assert result in ["1.9 KiB", "2.0 KiB"]
    
    def test_formatting_width(self):
        """Test that formatting maintains consistent width."""
        from mngs.str._readable_bytes import readable_bytes
        
        # The format string uses %3.1f, so numbers should be padded
        assert readable_bytes(1) == "1.0 B"
        assert readable_bytes(10) == "10.0 B"
        assert readable_bytes(100) == "100.0 B"
        
        # Check spacing is consistent
        assert len(readable_bytes(1).split()[0]) <= 5  # "1.0" or padded
        assert len(readable_bytes(999).split()[0]) <= 5  # "999.0"


class TestReadableBytesCustomSuffix:
    """Test custom suffix functionality."""
    
    def test_custom_suffix_basic(self):
        """Test basic custom suffix usage."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(1024, suffix="bytes") == "1.0 Kibytes"
        assert readable_bytes(1048576, suffix="B") == "1.0 MiB"
        assert readable_bytes(1073741824, suffix="ytes") == "1.0 GiBytes"
    
    def test_empty_suffix(self):
        """Test with empty suffix."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(0, suffix="") == "0.0 "
        assert readable_bytes(1024, suffix="") == "1.0 Ki"
        assert readable_bytes(1048576, suffix="") == "1.0 Mi"
    
    def test_special_character_suffix(self):
        """Test with special characters in suffix."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(1024, suffix="字节") == "1.0 Ki字节"
        assert readable_bytes(1024, suffix="バイト") == "1.0 Kiバイト"
        assert readable_bytes(1024, suffix="📦") == "1.0 Ki📦"
    
    def test_long_suffix(self):
        """Test with long suffix strings."""
        from mngs.str._readable_bytes import readable_bytes
        
        long_suffix = "bytes_of_data"
        result = readable_bytes(2048, suffix=long_suffix)
        assert result == "2.0 Kibytes_of_data"


class TestReadableBytesEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_float_inputs(self):
        """Test with float inputs."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(1024.0) == "1.0 KiB"
        assert readable_bytes(1536.5) == "1.5 KiB"
        assert readable_bytes(1024.9) == "1.0 KiB"
        
        # Very small floats
        assert readable_bytes(0.1) == "0.1 B"
        assert readable_bytes(0.5) == "0.5 B"
    
    def test_very_large_numbers(self):
        """Test with extremely large byte counts."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Beyond YiB
        huge = 1024 ** 9  # 1024 YiB
        result = readable_bytes(huge)
        assert "YiB" in result
        assert result == "1024.0 YiB"
        
        # Even larger
        result2 = readable_bytes(huge * 1000)
        assert "YiB" in result2
    
    def test_scientific_notation_inputs(self):
        """Test with numbers in scientific notation."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert readable_bytes(1e3) == "1000.0 B"
        assert readable_bytes(1e6) == "976.6 KiB"  # 1,000,000 / 1024 ≈ 976.6
        assert readable_bytes(1e9) == "953.7 MiB"  # 1 billion bytes
    
    def test_boundary_precision(self):
        """Test precision at unit boundaries."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Just below boundaries
        assert readable_bytes(1023.9) == "1023.9 B"
        assert readable_bytes(1024 * 1023.9) == "1023.9 KiB"
        
        # Just above boundaries  
        assert readable_bytes(1024.1) == "1.0 KiB"
        assert readable_bytes(1024 * 1024.1) == "1.0 MiB"


class TestReadableBytesSpecialValues:
    """Test handling of special numeric values."""
    
    def test_infinity(self):
        """Test with infinity values."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Positive infinity
        result = readable_bytes(float('inf'))
        # Should handle gracefully - might be "inf YiB" or similar
        assert 'inf' in result.lower() or 'YiB' in result
        
        # Negative infinity
        result = readable_bytes(float('-inf'))
        assert '-inf' in result.lower() or '-' in result
    
    def test_nan(self):
        """Test with NaN values."""
        from mngs.str._readable_bytes import readable_bytes
        
        result = readable_bytes(float('nan'))
        # Should handle gracefully
        assert 'nan' in result.lower() or 'B' in result
    
    def test_type_conversion(self):
        """Test implicit type conversion."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Test with various numeric types
        import numpy as np
        
        # NumPy integers
        assert readable_bytes(np.int32(1024)) == "1.0 KiB"
        assert readable_bytes(np.int64(1048576)) == "1.0 MiB"
        
        # NumPy floats
        assert readable_bytes(np.float32(2048)) == "2.0 KiB"
        assert readable_bytes(np.float64(3072)) == "3.0 KiB"


class TestReadableBytesComparison:
    """Test by comparing different byte values."""
    
    def test_unit_progression(self):
        """Test that units progress correctly."""
        from mngs.str._readable_bytes import readable_bytes
        
        values_and_units = [
            (1, "B"),
            (1024, "KiB"),
            (1024**2, "MiB"),
            (1024**3, "GiB"),
            (1024**4, "TiB"),
            (1024**5, "PiB"),
            (1024**6, "EiB"),
            (1024**7, "ZiB"),
            (1024**8, "YiB"),
        ]
        
        for value, expected_unit in values_and_units:
            result = readable_bytes(value)
            assert expected_unit in result
            assert result.startswith("1.0")
    
    def test_common_file_sizes(self):
        """Test with common real-world file sizes."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Common file sizes
        test_cases = [
            (1440 * 1024, "1.4 MiB"),  # 1.44MB floppy
            (700 * 1024 * 1024, "700.0 MiB"),  # CD-ROM
            (4.7 * 1024 * 1024 * 1024, "4.7 GiB"),  # Single-layer DVD
            (25 * 1024 * 1024 * 1024, "25.0 GiB"),  # Blu-ray
        ]
        
        for size, expected in test_cases:
            assert readable_bytes(size) == expected


class TestReadableBytesFunctionality:
    """Test overall function behavior and contracts."""
    
    def test_return_type(self):
        """Test that function always returns a string."""
        from mngs.str._readable_bytes import readable_bytes
        
        test_values = [0, 1, 1024, -1, 1.5, float('inf'), 1024**9]
        
        for value in test_values:
            result = readable_bytes(value)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_format_consistency(self):
        """Test that format is consistent across calls."""
        from mngs.str._readable_bytes import readable_bytes
        
        # All results should follow pattern: "number unit"
        test_values = [1, 1024, 1048576, 1073741824]
        
        for value in test_values:
            result = readable_bytes(value)
            parts = result.split()
            assert len(parts) == 2  # Number and unit
            
            # First part should be a number
            try:
                float(parts[0])
            except ValueError:
                pytest.fail(f"First part '{parts[0]}' is not a number")
            
            # Second part should be the unit
            assert parts[1].endswith('B') or parts[1].endswith('bytes')
    
    def test_docstring_examples(self):
        """Test examples from the function docstring."""
        from mngs.str._readable_bytes import readable_bytes
        
        # Examples from docstring
        assert readable_bytes(1024) == '1.0 KiB'
        assert readable_bytes(1048576) == '1.0 MiB' 
        assert readable_bytes(1073741824) == '1.0 GiB'


class TestReadableBytesPerformance:
    """Test performance characteristics."""
    
    def test_large_value_performance(self):
        """Test that large values are handled efficiently."""
        from mngs.str._readable_bytes import readable_bytes
        import time
        
        # Very large value
        huge = 1024 ** 10
        
        start = time.time()
        result = readable_bytes(huge)
        duration = time.time() - start
        
        # Should complete quickly
        assert duration < 0.001  # Less than 1ms
        assert "YiB" in result
    
    def test_many_calls_performance(self):
        """Test performance with many function calls."""
        from mngs.str._readable_bytes import readable_bytes
        import time
        
        values = list(range(0, 1024 * 1024, 1024))  # 1024 different values
        
        start = time.time()
        results = [readable_bytes(v) for v in values]
        duration = time.time() - start
        
        # Should handle many calls efficiently
        assert duration < 0.1  # Less than 100ms for 1024 calls
        assert len(results) == len(values)


class TestReadableBytesDocumentation:
    """Test function documentation and interface."""
    
    def test_function_signature(self):
        """Test function has expected signature."""
        from mngs.str._readable_bytes import readable_bytes
        import inspect
        
        sig = inspect.signature(readable_bytes)
        params = list(sig.parameters.keys())
        
        assert len(params) == 2
        assert params[0] == 'num'
        assert params[1] == 'suffix'
        
        # Check default value for suffix
        assert sig.parameters['suffix'].default == 'B'
    
    def test_function_attributes(self):
        """Test function has proper attributes."""
        from mngs.str._readable_bytes import readable_bytes
        
        assert hasattr(readable_bytes, '__doc__')
        assert readable_bytes.__doc__ is not None
        assert "Convert a number of bytes to a human-readable format" in readable_bytes.__doc__
        assert hasattr(readable_bytes, '__name__')
        assert readable_bytes.__name__ == 'readable_bytes'


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v", "-s"])
=======
# Timestamp: "2025-05-18 04:31:10 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/str/test__readable_bytes.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/str/test__readable_bytes.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-15 01:30:45 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/str/test__readable_bytes.py
# # ----------------------------------------
# import os
# import sys
# import importlib.util
# # ----------------------------------------

# import pytest

# # Direct import from file path
# readable_bytes_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/str/_readable_bytes.py"))
# spec = importlib.util.spec_from_file_location("_readable_bytes", readable_bytes_module_path)
# readable_bytes_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(readable_bytes_module)
# readable_bytes = readable_bytes_module.readable_bytes


# def test_readable_bytes_basic_functionality():
#     """Test basic functionality of readable_bytes function."""
#     # Test zero bytes
#     assert readable_bytes(0) == "0.0 B"

#     # Test basic conversions
#     assert readable_bytes(1) == "1.0 B"
#     assert readable_bytes(1023) == "1023.0 B"


# def test_readable_bytes_kilobytes():
#     """Test kilobyte conversions."""
#     # Test kilobyte boundary
#     assert readable_bytes(1024) == "1.0 KiB"
#     assert readable_bytes(1536) == "1.5 KiB"
#     assert readable_bytes(2048) == "2.0 KiB"

#     # Test values close to the next boundary
#     assert readable_bytes(1023 * 1024) == "1023.0 KiB"


# def test_readable_bytes_megabytes_and_higher():
#     """Test megabyte and higher conversions."""
#     # Test megabyte conversions
#     assert readable_bytes(1024 * 1024) == "1.0 MiB"
#     assert readable_bytes(1.5 * 1024 * 1024) == "1.5 MiB"

#     # Test gigabyte conversions
#     assert readable_bytes(1024 * 1024 * 1024) == "1.0 GiB"

#     # Test terabyte conversions
#     assert readable_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TiB"


# def test_readable_bytes_with_custom_suffix():
#     """Test readable_bytes with a custom suffix."""
#     # Test custom suffixes
#     assert readable_bytes(1024, suffix="iB") == "1.0 KiiB"
#     assert readable_bytes(1024 * 1024, suffix="b") == "1.0 Mib"
#     assert readable_bytes(1024, suffix="") == "1.0 Ki"


# def test_readable_bytes_with_negative_values():
#     """Test readable_bytes with negative values."""
#     # Test negative values
#     # The function uses abs() for comparison but preserves sign in output
#     assert readable_bytes(-1024) == "-1.0 KiB"
#     assert readable_bytes(-2048) == "-2.0 KiB"


# def test_readable_bytes_with_very_large_values():
#     """Test readable_bytes with very large values that exceed standard prefixes."""
#     # Test extremely large values
#     yotta_bytes = 1024 ** 8  # 1 YiB (yottabyte)
#     assert readable_bytes(yotta_bytes) == "1.0 YiB"

#     # Test value larger than yottabyte
#     beyond_yotta = 1024 ** 9  # 1024 YiB
#     assert readable_bytes(beyond_yotta) == "1024.0 YiB"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_readable_bytes.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 04:06:54 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_readable_bytes.py
# 
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
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_readable_bytes.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
