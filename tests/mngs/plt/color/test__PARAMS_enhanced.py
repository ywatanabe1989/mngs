#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-10 16:52:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/mngs_repo/tests/mngs/plt/color/test__PARAMS_enhanced.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/plt/color/test__PARAMS_enhanced.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
import re


def test_params_rgb_keys():
    """Test that RGB dictionary has expected keys."""
    from mngs.plt.color._PARAMS import PARAMS

    RGB = PARAMS["RGB"]

    expected_keys = [
        "white",
        "black",
        "blue",
        "red",
        "pink",
        "green",
        "yellow",
        "gray",
        "grey",
        "purple",
        "light_blue",
        "brown",
        "navy",
        "orange",
    ]

    assert set(RGB.keys()) == set(expected_keys)


def test_params_rgb_values():
    """Test that RGB values are valid."""
    from mngs.plt.color._PARAMS import PARAMS

    RGB = PARAMS["RGB"]

    for color, values in RGB.items():
        assert len(values) == 3, f"RGB color {color} should have 3 values"
        for value in values:
            assert (
                0 <= value <= 255
            ), f"RGB value for {color} should be between 0 and 255"


def test_params_rgba_norm():
    """Test that RGBA_NORM values are normalized correctly."""
    from mngs.plt.color._PARAMS import PARAMS

    RGB = PARAMS["RGB"]
    RGBA_NORM = PARAMS["RGBA_NORM"]

    for color in RGB:
        rgb_values = RGB[color]
        rgba_norm_values = RGBA_NORM[color]

        assert len(rgba_norm_values) == 4, f"RGBA_NORM for {color} should have 4 values"

        for idx in range(3):
            expected = round(rgb_values[idx] / 255, 2)
            assert (
                rgba_norm_values[idx] == expected
            ), f"RGBA_NORM for {color} not correctly normalized"


def test_def_alpha_constant():
    """Test DEF_ALPHA constant value."""
    from mngs.plt.color._PARAMS import DEF_ALPHA
    
    assert isinstance(DEF_ALPHA, (int, float))
    assert 0 <= DEF_ALPHA <= 1
    assert DEF_ALPHA == 0.9


def test_rgba_dictionary():
    """Test RGBA dictionary structure and values."""
    from mngs.plt.color._PARAMS import PARAMS, DEF_ALPHA
    
    RGB = PARAMS["RGB"]
    RGBA = PARAMS["RGBA"]
    
    # Check all RGB colors have RGBA equivalents
    assert set(RGBA.keys()) == set(RGB.keys())
    
    # Check RGBA values
    for color in RGB:
        rgb_values = RGB[color]
        rgba_values = RGBA[color]
        
        assert len(rgba_values) == 4, f"RGBA for {color} should have 4 values"
        
        # Check RGB components match
        for i in range(3):
            assert rgba_values[i] == rgb_values[i], f"RGBA RGB components for {color} don't match RGB"
        
        # Check alpha value
        assert rgba_values[3] == DEF_ALPHA, f"RGBA alpha for {color} should be {DEF_ALPHA}"


def test_rgb_norm_dictionary():
    """Test RGB_NORM dictionary values are properly normalized."""
    from mngs.plt.color._PARAMS import PARAMS
    
    RGB = PARAMS["RGB"]
    RGB_NORM = PARAMS.get("RGB_NORM")
    
    # Check RGB_NORM exists
    assert RGB_NORM is not None, "RGB_NORM should be in PARAMS"
    
    # Check all colors are present
    assert set(RGB_NORM.keys()) == set(RGB.keys())
    
    # Check normalization
    for color in RGB:
        rgb_values = RGB[color]
        rgb_norm_values = RGB_NORM[color]
        
        assert len(rgb_norm_values) == 3, f"RGB_NORM for {color} should have 3 values"
        
        for i in range(3):
            expected = round(rgb_values[i] / 255, 2)
            assert rgb_norm_values[i] == expected, f"RGB_NORM value for {color} incorrect"
            assert 0 <= rgb_norm_values[i] <= 1, f"RGB_NORM value for {color} out of range"


def test_rgba_norm_for_cycle():
    """Test RGBA_NORM_FOR_CYCLE excludes certain colors."""
    from mngs.plt.color._PARAMS import PARAMS
    
    RGBA_NORM = PARAMS["RGBA_NORM"]
    RGBA_NORM_FOR_CYCLE = PARAMS["RGBA_NORM_FOR_CYCLE"]
    
    # Check excluded colors
    excluded_colors = {"white", "grey", "black"}
    
    for color in excluded_colors:
        assert color not in RGBA_NORM_FOR_CYCLE, f"{color} should not be in RGBA_NORM_FOR_CYCLE"
    
    # Check included colors
    for color in RGBA_NORM:
        if color not in excluded_colors:
            assert color in RGBA_NORM_FOR_CYCLE, f"{color} should be in RGBA_NORM_FOR_CYCLE"
            assert RGBA_NORM_FOR_CYCLE[color] == RGBA_NORM[color], f"Values for {color} should match"


def test_hex_dictionary():
    """Test HEX color dictionary."""
    from mngs.plt.color._PARAMS import PARAMS
    
    RGB = PARAMS["RGB"]
    HEX = PARAMS["HEX"]
    
    # Check HEX has most colors (white and black might not have hex)
    assert len(HEX) >= len(RGB) - 2, "HEX should have most RGB colors"
    
    # Check HEX format
    hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
    
    for color, hex_value in HEX.items():
        assert isinstance(hex_value, str), f"HEX value for {color} should be string"
        assert hex_pattern.match(hex_value), f"HEX value {hex_value} for {color} has invalid format"


def test_hex_rgb_correspondence():
    """Test that HEX values correspond to RGB values."""
    from mngs.plt.color._PARAMS import PARAMS
    
    RGB = PARAMS["RGB"]
    HEX = PARAMS["HEX"]
    
    # Define tolerance for color conversion
    tolerance = 5  # Allow small differences due to rounding
    
    for color in HEX:
        if color in RGB:
            hex_value = HEX[color]
            rgb_values = RGB[color]
            
            # Convert hex to RGB
            hex_r = int(hex_value[1:3], 16)
            hex_g = int(hex_value[3:5], 16)
            hex_b = int(hex_value[5:7], 16)
            
            # Check with tolerance
            assert abs(hex_r - rgb_values[0]) <= tolerance, f"Red component mismatch for {color}"
            assert abs(hex_g - rgb_values[1]) <= tolerance, f"Green component mismatch for {color}"
            assert abs(hex_b - rgb_values[2]) <= tolerance, f"Blue component mismatch for {color}"


def test_params_dictionary_structure():
    """Test overall PARAMS dictionary structure."""
    from mngs.plt.color._PARAMS import PARAMS
    
    expected_keys = {"RGB", "RGBA", "RGBA_NORM", "RGBA_NORM_FOR_CYCLE", "HEX"}
    
    assert set(PARAMS.keys()) == expected_keys, "PARAMS should have all expected keys"
    
    # Check all values are dictionaries
    for key, value in PARAMS.items():
        assert isinstance(value, dict), f"PARAMS['{key}'] should be a dictionary"


def test_color_consistency_across_formats():
    """Test that same colors exist across different formats."""
    from mngs.plt.color._PARAMS import PARAMS
    
    RGB = PARAMS["RGB"]
    RGBA = PARAMS["RGBA"]
    RGBA_NORM = PARAMS["RGBA_NORM"]
    
    # All formats should have same color keys
    assert set(RGB.keys()) == set(RGBA.keys()) == set(RGBA_NORM.keys())


def test_grey_gray_equivalence():
    """Test that 'grey' and 'gray' have identical values."""
    from mngs.plt.color._PARAMS import PARAMS
    
    for format_name in ["RGB", "RGBA", "RGBA_NORM", "HEX"]:
        format_dict = PARAMS.get(format_name, {})
        if "grey" in format_dict and "gray" in format_dict:
            assert format_dict["grey"] == format_dict["gray"], f"grey and gray should be identical in {format_name}"


def test_alpha_values_in_rgba():
    """Test all alpha values in RGBA dictionaries."""
    from mngs.plt.color._PARAMS import PARAMS, DEF_ALPHA
    
    RGBA = PARAMS["RGBA"]
    RGBA_NORM = PARAMS["RGBA_NORM"]
    
    # Check all alpha values equal DEF_ALPHA
    for color in RGBA:
        assert RGBA[color][3] == DEF_ALPHA
        assert RGBA_NORM[color][3] == DEF_ALPHA


def test_color_value_ranges():
    """Test that all color values are in valid ranges."""
    from mngs.plt.color._PARAMS import PARAMS
    
    # Test RGB values
    RGB = PARAMS["RGB"]
    for color, values in RGB.items():
        for i, val in enumerate(values):
            assert isinstance(val, int), f"RGB value should be int for {color}[{i}]"
            assert 0 <= val <= 255, f"RGB value out of range for {color}[{i}]"
    
    # Test normalized values
    for format_name in ["RGBA_NORM", "RGBA_NORM_FOR_CYCLE"]:
        format_dict = PARAMS.get(format_name, {})
        for color, values in format_dict.items():
            for i, val in enumerate(values):
                assert isinstance(val, (int, float)), f"Normalized value should be numeric for {color}[{i}]"
                assert 0 <= val <= 1, f"Normalized value out of range for {color}[{i}] in {format_name}"


def test_params_immutability():
    """Test that PARAMS values are not accidentally modified."""
    from mngs.plt.color._PARAMS import PARAMS
    
    # Try to modify a value (shouldn't affect original)
    rgb_blue_original = PARAMS["RGB"]["blue"].copy()
    PARAMS["RGB"]["blue"][0] = 999
    
    # Re-import and check value is unchanged
    from mngs.plt.color._PARAMS import PARAMS as PARAMS_NEW
    
    assert PARAMS_NEW["RGB"]["blue"] == rgb_blue_original, "PARAMS should not be mutable"


if __name__ == "__main__":
    import os
    
    import pytest
    
    pytest.main([os.path.abspath(__file__)])

# EOF