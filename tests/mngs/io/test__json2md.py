#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-05-31"
# File: test__json2md.py

"""Tests for mngs.io._json2md module."""

import json
import os
import tempfile
from pathlib import Path

import pytest


class TestJson2MdBasic:
    """Test basic JSON to Markdown conversion functionality."""

    def test_simple_dict_conversion(self):
        """Test converting simple dictionary to markdown."""
        from mngs.io._json2md import json2md

        data = {"title": "Test Document", "author": "John Doe"}
        result = json2md(data)

        expected = "# title\nTest Document\n\n# author\nJohn Doe\n"
        assert result == expected

    def test_nested_dict_conversion(self):
        """Test converting nested dictionary to markdown."""
        from mngs.io._json2md import json2md

        data = {
            "chapter": "Introduction",
            "sections": {
                "overview": "This is an overview",
                "details": "These are the details",
            },
        }
        result = json2md(data)

        assert "# chapter" in result
        assert "## overview" in result
        assert "## details" in result
        assert "Introduction" in result
        assert "This is an overview" in result

    def test_simple_list_conversion(self):
        """Test converting simple list to markdown."""
        from mngs.io._json2md import json2md

        data = ["item1", "item2", "item3"]
        result = json2md(data)

        expected = "* item1\n* item2\n* item3"
        assert result == expected

    def test_mixed_types_conversion(self):
        """Test converting mixed types to markdown."""
        from mngs.io._json2md import json2md

        data = {
            "string": "text value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
        }
        result = json2md(data)

        assert "text value" in result
        assert "42" in result
        assert "3.14" in result
        assert "True" in result
        assert "None" in result


class TestJson2MdNested:
    """Test nested structure conversions."""

    def test_dict_with_list(self):
        """Test dictionary containing lists."""
        from mngs.io._json2md import json2md

        data = {"title": "Shopping List", "items": ["apples", "bananas", "oranges"]}
        result = json2md(data)

        assert "# title" in result
        assert "# items" in result
        assert "* apples" in result
        assert "* bananas" in result
        assert "* oranges" in result

    def test_list_of_dicts(self):
        """Test list containing dictionaries."""
        from mngs.io._json2md import json2md

        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = json2md(data)

        assert "# name" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "# age" in result
        assert "30" in result
        assert "25" in result

    def test_deeply_nested_structure(self):
        """Test deeply nested structure conversion."""
        from mngs.io._json2md import json2md

        data = {"level1": {"level2": {"level3": {"value": "deeply nested"}}}}
        result = json2md(data)

        assert "# level1" in result
        assert "## level2" in result
        assert "### level3" in result
        assert "#### value" in result
        assert "deeply nested" in result

    def test_complex_nested_structure(self):
        """Test complex nested structure with mixed types."""
        from mngs.io._json2md import json2md

        data = {
            "project": {
                "name": "Test Project",
                "team": ["Alice", "Bob", "Charlie"],
                "metadata": {"version": "1.0", "features": ["feature1", "feature2"]},
            }
        }
        result = json2md(data)

        assert "# project" in result
        assert "## name" in result
        assert "## team" in result
        assert "* Alice" in result
        assert "## metadata" in result
        assert "### version" in result
        assert "### features" in result


class TestJson2MdFormatting:
    """Test markdown formatting aspects."""

    def test_header_levels(self):
        """Test correct header level generation."""
        from mngs.io._json2md import json2md

        # Test up to 6 levels (markdown limit)
        data = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "value"}}}}}}
        result = json2md(data)

        assert "# l1" in result
        assert "## l2" in result
        assert "### l3" in result
        assert "#### l4" in result
        assert "##### l5" in result
        assert "###### l6" in result

    def test_spacing_between_sections(self):
        """Test proper spacing between sections."""
        from mngs.io._json2md import json2md

        data = {"section1": "content1", "section2": "content2", "section3": "content3"}
        result = json2md(data)

        # Should have empty lines between sections
        lines = result.split("\n")
        # Check that there are empty lines (after filtering)
        assert "" in lines or result.count("\n\n") >= 2

    def test_list_formatting(self):
        """Test proper list formatting."""
        from mngs.io._json2md import json2md

        data = {
            "groceries": ["milk", "bread", "eggs"],
            "tasks": ["code", "test", "deploy"],
        }
        result = json2md(data)

        # Lists should use asterisk markers
        assert result.count("* ") == 6
        assert "* milk" in result
        assert "* code" in result


class TestJson2MdEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_dict(self):
        """Test converting empty dictionary."""
        from mngs.io._json2md import json2md

        data = {}
        result = json2md(data)

        assert result == ""

    def test_empty_list(self):
        """Test converting empty list."""
        from mngs.io._json2md import json2md

        data = []
        result = json2md(data)

        assert result == ""

    def test_single_value_types(self):
        """Test converting single value types."""
        from mngs.io._json2md import json2md

        # These are edge cases - normally json2md expects dict or list
        assert json2md("string") == ""  # Non-dict/list returns empty
        assert json2md(123) == ""
        assert json2md(True) == ""
        assert json2md(None) == ""

    def test_special_characters_in_keys(self):
        """Test keys with special characters."""
        from mngs.io._json2md import json2md

        data = {
            "key with spaces": "value1",
            "key-with-dashes": "value2",
            "key_with_underscores": "value3",
            "key.with.dots": "value4",
        }
        result = json2md(data)

        assert "# key with spaces" in result
        assert "# key-with-dashes" in result
        assert "# key_with_underscores" in result
        assert "# key.with.dots" in result

    def test_unicode_content(self):
        """Test Unicode content handling."""
        from mngs.io._json2md import json2md

        data = {
            "greeting": "你好",
            "emoji": "🚀 Launch",
            "languages": ["English", "中文", "日本語", "한국어"],
        }
        result = json2md(data)

        assert "你好" in result
        assert "🚀 Launch" in result
        assert "* 中文" in result
        assert "* 日本語" in result


class TestJson2MdMain:
    """Test the main function and CLI interface."""

    def test_main_with_valid_json_file(self, tmp_path):
        """Test main function with valid JSON file."""
        from mngs.io._json2md import main
        import sys

        # Create test JSON file
        test_data = {"title": "Test", "items": ["a", "b", "c"]}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(test_data))

        # Mock command line arguments
        original_argv = sys.argv
        try:
            sys.argv = ["json2md", str(json_file)]

            # Capture stdout
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                main()

            output = f.getvalue()
            assert "# title" in output
            assert "* a" in output

        finally:
            sys.argv = original_argv

    def test_main_with_output_file(self, tmp_path):
        """Test main function with output file option."""
        from mngs.io._json2md import main
        import sys

        # Create test JSON file
        test_data = {"section": "content"}
        json_file = tmp_path / "input.json"
        json_file.write_text(json.dumps(test_data))
        output_file = tmp_path / "output.md"

        # Mock command line arguments
        original_argv = sys.argv
        try:
            sys.argv = ["json2md", str(json_file), "-o", str(output_file)]
            main()

            # Check output file created and contains expected content
            assert output_file.exists()
            content = output_file.read_text()
            assert "# section" in content
            assert "content" in content

        finally:
            sys.argv = original_argv

    def test_main_with_nonexistent_file(self, tmp_path):
        """Test main function with nonexistent file."""
        from mngs.io._json2md import main
        import sys

        nonexistent = tmp_path / "does_not_exist.json"

        original_argv = sys.argv
        try:
            sys.argv = ["json2md", str(nonexistent)]

            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

        finally:
            sys.argv = original_argv


class TestJson2MdIntegration:
    """Test integration scenarios."""

    def test_round_trip_json_to_md(self):
        """Test converting various JSON structures to markdown."""
        from mngs.io._json2md import json2md

        test_cases = [
            # Simple dict
            {"key": "value"},
            # Nested dict
            {"outer": {"inner": "value"}},
            # Dict with list
            {"items": [1, 2, 3]},
            # List of dicts
            [{"a": 1}, {"b": 2}],
            # Complex structure
            {
                "metadata": {"title": "Report", "date": "2024-01-01"},
                "sections": [
                    {"name": "Introduction", "content": "..."},
                    {"name": "Conclusion", "content": "..."},
                ],
            },
        ]

        for data in test_cases:
            result = json2md(data)
            assert isinstance(result, str)
            # Verify some conversion happened (not empty for non-empty inputs)
            if data:
                assert len(result) > 0

    def test_large_json_conversion(self):
        """Test converting large JSON structure."""
        from mngs.io._json2md import json2md

        # Create a large nested structure
        large_data = {
            f"section_{i}": {
                f"subsection_{j}": [f"item_{k}" for k in range(10)] for j in range(5)
            }
            for i in range(10)
        }

        result = json2md(large_data)

        # Should handle large structures without error
        assert isinstance(result, str)
        assert "section_0" in result
        assert "subsection_0" in result
        assert "* item_0" in result


# --------------------------------------------------------------------------------
=======
# Timestamp: "2025-05-15 03:40:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test__json2md.py
# ----------------------------------------
import os
import importlib.util
# ----------------------------------------

import pytest

# Direct import from file path
json2md_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/mngs/io/_json2md.py"))
spec = importlib.util.spec_from_file_location("_json2md", json2md_module_path)
json2md_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(json2md_module)
json2md = json2md_module.json2md


def test_json2md_basic_dict():
    """Test json2md with a basic dictionary."""
    # Test with a simple dictionary
    data = {"title": "Test Document", "description": "This is a test"}
    result = json2md(data)
    
    expected = "# title\nTest Document\n\n# description\nThis is a test\n"
    assert result == expected


def test_json2md_nested_dict():
    """Test json2md with a nested dictionary."""
    # Test with a nested dictionary
    data = {
        "project": "Test Project",
        "details": {
            "version": "1.0",
            "status": "active"
        }
    }
    result = json2md(data)
    
    expected = "# project\nTest Project\n\n# details\n## version\n1.0\n\n## status\nactive\n"
    assert result == expected


def test_json2md_list():
    """Test json2md with a list."""
    # Test with a list
    data = ["apple", "banana", "cherry"]
    result = json2md(data)
    
    expected = "* apple\n* banana\n* cherry"
    assert result == expected


def test_json2md_list_of_dicts():
    """Test json2md with a list of dictionaries."""
    # Test with a list of dictionaries
    data = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25}
    ]
    result = json2md(data)
    
    # The actual output includes a newline at the end for each scalar value
    expected = "# name\nJohn\n\n# age\n30\n\n# name\nJane\n\n# age\n25\n"
    assert result == expected


def test_json2md_complex_structure():
    """Test json2md with a complex nested structure."""
    # Test with a complex structure
    data = {
        "person": {
            "name": "John Doe",
            "contact": {
                "email": "john@example.com",
                "phone": "555-1234"
            },
            "hobbies": ["reading", "hiking", "coding"]
        },
        "notes": "Additional information"
    }
    result = json2md(data)
    
    # Instead of comparing exact strings with potentially different newline patterns,
    # we'll normalize and compare the structure by splitting lines and checking content
    result_lines = [line for line in result.split('\n') if line]
    expected_lines = [
        "# person",
        "## name",
        "John Doe",
        "## contact",
        "### email",
        "john@example.com",
        "### phone",
        "555-1234",
        "## hobbies",
        "* reading",
        "* hiking",
        "* coding",
        "# notes",
        "Additional information"
    ]
    
    # Remove empty lines from both for comparison
    assert len(result_lines) == len(expected_lines)
    for i, (result_line, expected_line) in enumerate(zip(result_lines, expected_lines)):
        assert result_line == expected_line, f"Line {i+1} mismatch: '{result_line}' != '{expected_line}'"


def test_json2md_empty_structures():
    """Test json2md with empty structures."""
    # Test with empty structures
    assert json2md({}) == ""
    assert json2md([]) == ""
    assert json2md({"empty": {}}) == "# empty"
    assert json2md({"empty": []}) == "# empty"


def test_json2md_non_string_values():
    """Test json2md with non-string values."""
    # Test with non-string values
    data = {
        "number": 42,
        "boolean": True,
        "null": None,
        "float": 3.14
    }
    result = json2md(data)
    
    # Similar to complex test, use line-by-line comparison for consistency
    result_lines = [line for line in result.split('\n') if line]
    expected_lines = [
        "# number",
        "42",
        "# boolean",
        "True",
        "# null",
        "None",
        "# float",
        "3.14"
    ]
    
    # Remove empty lines for comparison
    assert len(result_lines) == len(expected_lines)
    for i, (result_line, expected_line) in enumerate(zip(result_lines, expected_lines)):
        assert result_line == expected_line, f"Line {i+1} mismatch: '{result_line}' != '{expected_line}'"
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_json2md.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-19 15:37:46 (ywatanabe)"
# # File: ./Ninja/workspace/formats/json2md.py
# 
# THIS_FILE = "/home/ywatanabe/.emacs.d/lisp/Ninja/workspace/formats/json2md.py"
# 
# import json
# import sys
# import argparse
# 
# def json2md(obj, level=1):
#     output = []
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             if output:  # Add extra newline between sections
#                 output.append("")
#             output.append("#" * level + " " + str(key))
#             if isinstance(value, (dict, list)):
#                 output.append(json2md(value, level + 1))
#             else:
#                 output.append(str(value) + "\n")
#     elif isinstance(obj, list):
#         for item in obj:
#             if isinstance(item, (dict, list)):
#                 output.append(json2md(item, level))
#             else:
#                 output.append("* " + str(item))
#     return "\n".join(filter(None, output))
# 
# def main():
#     parser = argparse.ArgumentParser(description='Convert JSON to Markdown')
#     parser.add_argument('input', help='Input JSON file')
#     parser.add_argument('-o', '--output', help='Output file (default: stdout)')
#     args = parser.parse_args()
# 
#     try:
#         with open(args.input, 'r') as f:
#             data = json.load(f)
# 
#         result = json2md(data)
# 
#         if args.output:
#             with open(args.output, 'w') as f:
#                 f.write(result)
#         else:
#             print(result)
# 
#     except FileNotFoundError:
#         print(f"Error: File {args.input} not found", file=sys.stderr)
#         sys.exit(1)
# 
# if __name__ == "__main__":
#     main()
# 
# """
# python ./Ninja/workspace/formats/json2md.py
# python -m workspace.formats.json2md
# """
# # EOF
# 
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # # Time-stamp: "2024-12-19 15:29:28 (ywatanabe)"
# # # File: ./Ninja/workspace/formats/json2md.py
# 
# # THIS_FILE = "/home/ywatanabe/.emacs.d/lisp/Ninja/workspace/formats/json2md.py"
# 
# # import json
# # import sys
# 
# # def json2md(obj, level=1):
# #     output = []
# #     if isinstance(obj, dict):
# #         for key, value in obj.items():
# #             if output:  # Add extra newline between sections
# #                 output.append("")
# #             output.append("#" * level + " " + str(key))
# #             if isinstance(value, (dict, list)):
# #                 output.append(json2md(value, level + 1))
# #             else:
# #                 output.append(str(value) + "\n")
# #     elif isinstance(obj, list):
# #         for item in obj:
# #             if isinstance(item, (dict, list)):
# #                 output.append(json2md(item, level))
# #             else:
# #                 output.append("* " + str(item))
# #     return "\n".join(filter(None, output))
# 
# # def main():
# #     if len(sys.argv) != 2:
# #         print("Usage: json2md.py <input.json>")
# #         sys.exit(1)
# 
# #     lpath = sys.argv[1].replace("/./", "/")
# #     with open(lpath, "r") as f:
# #         data = json.load(f)
# 
# 
# # if __name__ == "__main__":
# #     main()
# 
# 
# # """
# # python ./Ninja/workspace/formats/json2md.py
# # python -m workspace.formats.json2md
# # """
# 
# # # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_json2md.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
