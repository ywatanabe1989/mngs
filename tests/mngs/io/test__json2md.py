#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

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
