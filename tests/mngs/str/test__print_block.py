#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:00:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/str/test__print_block.py

"""Tests for block printing functionality."""

import os
import pytest
from unittest.mock import patch


class TestPrintcBasic:
    """Test basic printc functionality."""
    
    def test_printc_basic_message(self, capsys):
        """Test basic message printing with default parameters."""
        from mngs.str._print_block import printc
        
        printc("Test Message")
        captured = capsys.readouterr()
        
        assert "Test Message" in captured.out
        assert "-" in captured.out  # Default border character
        assert captured.out.count("-") >= 40  # Default width
    
    def test_printc_custom_character(self, capsys):
        """Test printing with custom border character."""
        from mngs.str._print_block import printc
        
        printc("Custom Border", char="*")
        captured = capsys.readouterr()
        
        assert "Custom Border" in captured.out
        assert "*" in captured.out
        assert "-" not in captured.out  # Should not use default char
    
    def test_printc_custom_width(self, capsys):
        """Test printing with custom border width."""
        from mngs.str._print_block import printc
        
        printc("Width Test", char="#", n=20)
        captured = capsys.readouterr()
        
        assert "Width Test" in captured.out
        assert "#" * 20 in captured.out  # Should have exact width
    
    def test_printc_no_color(self, capsys):
        """Test printing without color."""
        from mngs.str._print_block import printc
        
        printc("No Color", c=None)
        captured = capsys.readouterr()
        
        assert "No Color" in captured.out
        assert "-" in captured.out


class TestPrintcColors:
    """Test printc color functionality."""
    
    @pytest.mark.parametrize("color", [
        "red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey"
    ])
    def test_printc_valid_colors(self, capsys, color):
        """Test printing with valid color options."""
        from mngs.str._print_block import printc
        
        printc("Color Test", c=color)
        captured = capsys.readouterr()
        
        assert "Color Test" in captured.out
        # Color codes should be present (ANSI escape sequences)
        assert "\x1b[" in captured.out or "Color Test" in captured.out
    
    def test_printc_default_color(self, capsys):
        """Test printing with default color (cyan)."""
        from mngs.str._print_block import printc
        
        printc("Default Color")
        captured = capsys.readouterr()
        
        assert "Default Color" in captured.out
        # Should have color codes for cyan
        assert "\x1b[" in captured.out or "Default Color" in captured.out
    
    def test_printc_invalid_color_handling(self, capsys):
        """Test behavior with invalid color (should still work)."""
        from mngs.str._print_block import printc
        
        # This should not crash, even with invalid color
        printc("Invalid Color", c="invalidcolor")
        captured = capsys.readouterr()
        
        assert "Invalid Color" in captured.out


class TestPrintcEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_printc_empty_message(self, capsys):
        """Test printing empty message."""
        from mngs.str._print_block import printc
        
        printc("")
        captured = capsys.readouterr()
        
        assert "-" in captured.out  # Border should still appear
        lines = captured.out.strip().split('\n')
        assert len(lines) >= 3  # Should have top border, content, bottom border
    
    def test_printc_multiline_message(self, capsys):
        """Test printing multiline message."""
        from mngs.str._print_block import printc
        
        multiline_msg = "Line 1\nLine 2\nLine 3"
        printc(multiline_msg)
        captured = capsys.readouterr()
        
        assert "Line 1" in captured.out
        assert "Line 2" in captured.out
        assert "Line 3" in captured.out
        assert "-" in captured.out
    
    def test_printc_unicode_message(self, capsys):
        """Test printing unicode characters."""
        from mngs.str._print_block import printc
        
        unicode_msg = "Unicode: 測試 🚀 ñáöü"
        printc(unicode_msg)
        captured = capsys.readouterr()
        
        assert "Unicode:" in captured.out
        assert "測試" in captured.out
        assert "🚀" in captured.out
    
    def test_printc_long_message(self, capsys):
        """Test printing very long message."""
        from mngs.str._print_block import printc
        
        long_msg = "A" * 100  # Message longer than default border
        printc(long_msg, n=20)  # Short border
        captured = capsys.readouterr()
        
        assert long_msg in captured.out
        assert "-" * 20 in captured.out
    
    def test_printc_special_characters(self, capsys):
        """Test printing message with special characters."""
        from mngs.str._print_block import printc
        
        special_msg = "Special: @#$%^&*()_+{}|:<>?[]\\;'\",./"
        printc(special_msg)
        captured = capsys.readouterr()
        
        assert "Special:" in captured.out
        assert "@#$%^&*" in captured.out


class TestPrintcParameters:
    """Test parameter combinations and validation."""
    
    def test_printc_zero_width(self, capsys):
        """Test behavior with zero width border."""
        from mngs.str._print_block import printc
        
        printc("Zero Width", n=0)
        captured = capsys.readouterr()
        
        assert "Zero Width" in captured.out
        # Should handle gracefully, even if border is empty
    
    def test_printc_negative_width(self, capsys):
        """Test behavior with negative width."""
        from mngs.str._print_block import printc
        
        printc("Negative Width", n=-5)
        captured = capsys.readouterr()
        
        assert "Negative Width" in captured.out
        # Should handle gracefully
    
    def test_printc_large_width(self, capsys):
        """Test with very large width."""
        from mngs.str._print_block import printc
        
        printc("Large Width", n=200)
        captured = capsys.readouterr()
        
        assert "Large Width" in captured.out
        assert "-" * 200 in captured.out
    
    def test_printc_multi_character_border(self, capsys):
        """Test with multi-character border (should use first char)."""
        from mngs.str._print_block import printc
        
        printc("Multi Char", char="ABC")
        captured = capsys.readouterr()
        
        assert "Multi Char" in captured.out
        # Should repeat the string as-is
        assert "ABC" in captured.out


class TestPrintcFormatting:
    """Test output formatting and structure."""
    
    def test_printc_output_structure(self, capsys):
        """Test that output has correct structure (border-message-border)."""
        from mngs.str._print_block import printc
        
        printc("Structure Test", char="=", n=30)
        captured = capsys.readouterr()
        
        lines = captured.out.strip().split('\n')
        # Should have: empty line, border, message, border, empty line
        assert len(lines) >= 4
        
        # Find border lines
        border_lines = [line for line in lines if "=" in line and len(line.strip()) > 0]
        assert len(border_lines) >= 2  # Top and bottom borders
        
        # Check border content
        expected_border = "=" * 30
        assert any(expected_border in line for line in border_lines)
    
    def test_printc_newlines_in_output(self, capsys):
        """Test that output includes proper newlines."""
        from mngs.str._print_block import printc
        
        printc("Newline Test")
        captured = capsys.readouterr()
        
        # Should start and end with newlines
        assert captured.out.startswith('\n')
        assert captured.out.endswith('\n')


class TestPrintcIntegration:
    """Test integration with color_text function."""
    
    @patch('mngs.str._print_block.color_text')
    def test_printc_color_text_called(self, mock_color_text, capsys):
        """Test that color_text is called when color is specified."""
        from mngs.str._print_block import printc
        
        mock_color_text.return_value = "colored_text"
        
        printc("Color Integration", c="blue")
        
        # color_text should be called with the formatted text and color
        mock_color_text.assert_called_once()
        args, kwargs = mock_color_text.call_args
        assert "Color Integration" in args[0]
        assert args[1] == "blue"
    
    @patch('mngs.str._print_block.color_text')
    def test_printc_no_color_text_call(self, mock_color_text, capsys):
        """Test that color_text is not called when color is None."""
        from mngs.str._print_block import printc
        
        printc("No Color", c=None)
        
        # color_text should not be called
        mock_color_text.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])