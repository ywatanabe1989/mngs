#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 04:07:27 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/io/test__glob.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/test__glob.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 04:04:51 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/io/test__glob.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./tests/mngs/io/test__glob.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# import importlib.util
# import sys
# import tempfile
# from unittest import mock

# import pytest

# # Import the glob module directly from the source file
# glob_module_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../../../src/mngs/io/_glob.py")
# )
# spec = importlib.util.spec_from_file_location("_glob", glob_module_path)
# glob_module = importlib.util.module_from_spec(spec)
# sys.modules["_glob"] = glob_module
# spec.loader.exec_module(glob_module)
# glob = glob_module.glob
# parse_glob = glob_module.parse_glob

# # Import DotDict for testing the parsing functionality
# dotdict_module_path = os.path.abspath(
#     os.path.join(
#         os.path.dirname(__file__), "../../../src/mngs/dict/_DotDict.py"
#     )
# )
# spec_dotdict = importlib.util.spec_from_file_location(
#     "_DotDict", dotdict_module_path
# )
# dotdict_module = importlib.util.module_from_spec(spec_dotdict)
# sys.modules["_DotDict"] = dotdict_module
# spec_dotdict.loader.exec_module(dotdict_module)


# class TestGlobFunctions:
#     """Test class for _glob.py module functions using actual file operations."""

#     @pytest.fixture(autouse=True)
#     def setup(self):
#         """Setup test environment with actual test files."""
#         # Create a temporary directory for tests
#         self.temp_dir = tempfile.mkdtemp()

#         # Create a test directory structure
#         self.data_dir = os.path.join(self.temp_dir, "data")
#         os.makedirs(self.data_dir, exist_ok=True)

#         # Create subdirectories for testing curly brace expansion
#         os.makedirs(os.path.join(self.data_dir, "a"), exist_ok=True)
#         os.makedirs(os.path.join(self.data_dir, "b"), exist_ok=True)

#         # Create test files in the main data directory
#         self.create_test_file(
#             os.path.join(self.data_dir, "file1.txt"), "Test file 1"
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "file2.txt"), "Test file 2"
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "file10.txt"), "Test file 10"
#         )

#         # Create test files in the subdirectories
#         self.create_test_file(
#             os.path.join(self.data_dir, "a", "file1.txt"), "A-Test file 1"
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "a", "file2.txt"), "A-Test file 2"
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "b", "file1.txt"), "B-Test file 1"
#         )

#         # Create test files for pattern parsing
#         os.makedirs(os.path.join(self.data_dir, "subj_001"), exist_ok=True)
#         os.makedirs(os.path.join(self.data_dir, "subj_002"), exist_ok=True)

#         self.create_test_file(
#             os.path.join(self.data_dir, "subj_001", "run_01.txt"),
#             "Subject 1, Run 1",
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "subj_001", "run_02.txt"),
#             "Subject 1, Run 2",
#         )
#         self.create_test_file(
#             os.path.join(self.data_dir, "subj_002", "run_01.txt"),
#             "Subject 2, Run 1",
#         )

#     def teardown_method(self):
#         """Clean up after tests."""
#         import shutil

#         if os.path.exists(self.temp_dir):
#             shutil.rmtree(self.temp_dir)

#     def create_test_file(self, path, content):
#         """Helper method to create a test file with given content."""
#         with open(path, "w") as f:
#             f.write(content)
#         return path

#     def test_glob_basic_functionality(self):
#         """Test basic functionality of glob function."""
#         # Test globbing with a simple pattern
#         pattern = os.path.join(self.data_dir, "*.txt")
#         results = glob(pattern)

#         # Verify we get all 3 files in the main directory
#         assert len(results) == 3
#         assert os.path.join(self.data_dir, "file1.txt") in results
#         assert os.path.join(self.data_dir, "file2.txt") in results
#         assert os.path.join(self.data_dir, "file10.txt") in results

#         # Verify natural sorting (file10.txt should come after file2.txt)
#         assert results.index(
#             os.path.join(self.data_dir, "file10.txt")
#         ) > results.index(os.path.join(self.data_dir, "file2.txt"))

#     def test_glob_with_complex_patterns(self):
#         """Test glob function with various complex patterns."""
#         # Test with wildcard in directory name
#         pattern = os.path.join(self.temp_dir, "*", "*.txt")
#         results = glob(pattern)

#         # Verify we get all 6 files from all directories
#         assert len(results) == 6

#         # Test with specific prefix
#         pattern = os.path.join(self.data_dir, "file1.txt")
#         results = glob(pattern)

#         # Verify we get exactly the file requested
#         assert len(results) == 1
#         assert results[0] == os.path.join(self.data_dir, "file1.txt")

#         # Test with pattern that doesn't match
#         pattern = os.path.join(self.data_dir, "nonexistent*.txt")
#         results = glob(pattern)

#         # Verify we get empty list for non-matching pattern
#         assert len(results) == 0

#     def test_glob_with_curly_braces(self):
#         """Test glob function with curly brace expansion."""
#         # Pattern with curly braces - should expand to match both a/ and b/ directories
#         pattern = os.path.join(self.data_dir, "{a,b}", "*.txt")

#         # Use glob with the original _re.sub and _glob functionality
#         with mock.patch.object(glob_module, "_re") as re_mock:
#             with mock.patch.object(glob_module, "_glob") as glob_mock:
#                 # Mock re.sub to return custom pattern for this test
#                 re_mock.sub.return_value = os.path.join(
#                     self.data_dir, "*", "*.txt"
#                 )

#                 # Mock _glob to return expected paths
#                 expected_paths = [
#                     os.path.join(self.data_dir, "a", "file1.txt"),
#                     os.path.join(self.data_dir, "a", "file2.txt"),
#                     os.path.join(self.data_dir, "b", "file1.txt"),
#                 ]
#                 glob_mock.return_value = expected_paths

#                 # Mock _natsorted to return the same list
#                 with mock.patch.object(
#                     glob_module, "_natsorted", side_effect=lambda x: x
#                 ):
#                     results = glob(pattern)

#                     # Verify re.sub was called with the correct pattern
#                     re_mock.sub.assert_called_with(r"{[^}]*}", "*", pattern)

#                     # Verify _glob was called with the modified pattern
#                     expected_pattern = os.path.join(
#                         self.data_dir, "*", "*.txt"
#                     )
#                     glob_mock.assert_called_with(expected_pattern)

#                     # Verify results
#                     assert results == expected_paths

#     def test_glob_with_ensure_one(self):
#         """Test glob function with ensure_one=True."""
#         # Test with pattern that matches exactly one file
#         pattern = os.path.join(self.data_dir, "file1.txt")
#         result = glob(pattern, ensure_one=True)

#         # Verify we get the expected file
#         assert len(result) == 1
#         assert result[0] == os.path.join(self.data_dir, "file1.txt")

#         # Test with pattern that matches multiple files
#         pattern = os.path.join(self.data_dir, "*.txt")

#         # This should raise an assertion error
#         with pytest.raises(AssertionError):
#             glob(pattern, ensure_one=True)

#         # Test with pattern that matches no files
#         pattern = os.path.join(self.data_dir, "nonexistent.txt")

#         # This should also raise an assertion error
#         with pytest.raises(AssertionError):
#             glob(pattern, ensure_one=True)

#     def test_glob_with_parse(self):
#         """Test glob function with parse=True."""
#         # Define a pattern with placeholders
#         pattern = os.path.join(self.data_dir, "subj_{id}", "run_{run}.txt")

#         # Get results with parsing
#         paths, parsed = glob(pattern, parse=True)

#         # Verify we get the expected files
#         assert len(paths) == 3
#         assert os.path.join(self.data_dir, "subj_001", "run_01.txt") in paths
#         assert os.path.join(self.data_dir, "subj_001", "run_02.txt") in paths
#         assert os.path.join(self.data_dir, "subj_002", "run_01.txt") in paths

#         # Verify parsing results
#         assert len(parsed) == 3

#         # Create a lookup table for easier verification
#         parsed_lookup = {p: v for p, v in zip(paths, parsed)}

#         # Check parsed values
#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_001", "run_01.txt")
#             ]["id"]
#             == "001"
#         )
#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_001", "run_01.txt")
#             ]["run"]
#             == "01"
#         )

#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_001", "run_02.txt")
#             ]["id"]
#             == "001"
#         )
#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_001", "run_02.txt")
#             ]["run"]
#             == "02"
#         )

#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_002", "run_01.txt")
#             ]["id"]
#             == "002"
#         )
#         assert (
#             parsed_lookup[
#                 os.path.join(self.data_dir, "subj_002", "run_01.txt")
#             ]["run"]
#             == "01"
#         )

#     def test_parse_glob_convenience_function(self):
#         """Test the parse_glob convenience function."""
#         # Define a pattern with placeholders
#         pattern = os.path.join(self.data_dir, "subj_{id}", "run_{run}.txt")

#         # Call parse_glob
#         paths, parsed = parse_glob(pattern)

#         # Verify we get the expected files
#         assert len(paths) == 3
#         assert len(parsed) == 3

#         # Create a lookup table for easier verification
#         parsed_lookup = {p: v for p, v in zip(paths, parsed)}

#         # Ensure all paths are parsed correctly
#         for path in paths:
#             parts = path.split(os.sep)
#             subj_part = parts[-2]  # Get the subject part of the path
#             run_part = parts[-1].split(".")[
#                 0
#             ]  # Get the run part without extension

#             # Extract the values using string manipulation for verification
#             id_val = subj_part.replace("subj_", "")
#             run_val = run_part.replace("run_", "")

#             # Verify parsed values match expected values
#             assert parsed_lookup[path]["id"] == id_val
#             assert parsed_lookup[path]["run"] == run_val

#     def test_parse_glob_with_ensure_one(self):
#         """Test parse_glob with ensure_one=True."""
#         # Test with pattern that matches exactly one file
#         pattern = os.path.join(self.data_dir, "subj_002", "run_{run}.txt")
#         paths, parsed = parse_glob(pattern, ensure_one=True)

#         # Verify we get one file and its parsed data
#         assert len(paths) == 1
#         assert len(parsed) == 1
#         assert parsed[0]["id"] == "002"
#         assert parsed[0]["run"] == "01"

#         # Test with pattern that matches multiple files
#         pattern = os.path.join(self.data_dir, "subj_{id}", "run_{run}.txt")

#         # This should raise an assertion error
#         with pytest.raises(AssertionError):
#             parse_glob(pattern, ensure_one=True)

#     def test_glob_with_directory_ending_slash(self):
#         """Test glob on directory paths ending with a slash."""
#         # Create a test directory for this specific test
#         test_dir = os.path.join(self.temp_dir, "test_dir_slash")
#         os.makedirs(test_dir, exist_ok=True)

#         # Test pattern with trailing slash
#         pattern = os.path.join(self.temp_dir, "*/")
#         results = glob(pattern)

#         # Verify that directories are found
#         assert len(results) > 0
#         # Due to natural sorting, we can't predict the exact order
#         # But the data directory should be in results
#         assert any(os.path.normpath(r).endswith("data") for r in results)

#     def test_glob_eval_security(self):
#         """Test the safety of eval used in glob function."""
#         # This test verifies the exception handling for the eval statement

#         # Create a pattern with a malformed string that would cause eval to fail
#         malformed_pattern = (
#             "f'{self.data_dir}/*.txt'"  # This would fail in eval
#         )

#         # The function should catch the exception and fall back to using the pattern directly
#         results = glob(malformed_pattern)

#         # Since this pattern likely won't match any files, results should be empty
#         # But the important thing is that it doesn't raise an exception
#         assert isinstance(results, list)

#     def test_glob_with_invalid_patterns(self):
#         """Test glob with invalid or unusual patterns."""
#         # Test with empty pattern
#         results = glob("")
#         assert isinstance(results, list)

#         # Test with None pattern - this should raise a TypeError
#         with pytest.raises(TypeError):
#             glob(None)

#     def test_parse_glob_with_complex_expressions(self):
#         """Test parse_glob with more complex parsing expressions."""
#         # Create a complex test directory structure
#         complex_dir = os.path.join(self.temp_dir, "complex")
#         os.makedirs(complex_dir, exist_ok=True)

#         # Create files with year_month_day_hour format
#         self.create_test_file(
#             os.path.join(complex_dir, "data_2023_01_15_08.csv"),
#             "Data for 2023-01-15 08:00",
#         )
#         self.create_test_file(
#             os.path.join(complex_dir, "data_2023_01_15_14.csv"),
#             "Data for 2023-01-15 14:00",
#         )
#         self.create_test_file(
#             os.path.join(complex_dir, "data_2023_01_16_09.csv"),
#             "Data for 2023-01-16 09:00",
#         )

#         # Define pattern with year, month, day, hour fields
#         pattern = os.path.join(
#             complex_dir, "data_{year}_{month}_{day}_{hour}.csv"
#         )

#         # Call parse_glob
#         paths, parsed = parse_glob(pattern)

#         # Verify we get the expected files
#         assert len(paths) == 3
#         assert len(parsed) == 3

#         # Create a lookup table for easier verification
#         parsed_lookup = {p: v for p, v in zip(paths, parsed)}

#         # Check first file
#         file1 = os.path.join(complex_dir, "data_2023_01_15_08.csv")
#         assert parsed_lookup[file1]["year"] == "2023"
#         assert parsed_lookup[file1]["month"] == "01"
#         assert parsed_lookup[file1]["day"] == "15"
#         assert parsed_lookup[file1]["hour"] == "08"

#         # Check second file
#         file2 = os.path.join(complex_dir, "data_2023_01_15_14.csv")
#         assert parsed_lookup[file2]["year"] == "2023"
#         assert parsed_lookup[file2]["month"] == "01"
#         assert parsed_lookup[file2]["day"] == "15"
#         assert parsed_lookup[file2]["hour"] == "14"

#         # Check third file
#         file3 = os.path.join(complex_dir, "data_2023_01_16_09.csv")
#         assert parsed_lookup[file3]["year"] == "2023"
#         assert parsed_lookup[file3]["month"] == "01"
#         assert parsed_lookup[file3]["day"] == "16"
#         assert parsed_lookup[file3]["hour"] == "09"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_glob.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 00:31:08 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_glob.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_glob.py"
# 
# import re as _re
# from glob import glob as _glob
# from ..str._parse import parse as _parse
# from natsort import natsorted as _natsorted
# 
# 
# def glob(expression, parse=False, ensure_one=False):
#     """
#     Perform a glob operation with natural sorting and extended pattern support.
# 
#     This function extends the standard glob functionality by adding natural sorting
#     and support for curly brace expansion in the glob pattern.
# 
#     Parameters:
#     -----------
#     expression : str
#         The glob pattern to match against file paths. Supports standard glob syntax
#         and curly brace expansion (e.g., 'dir/{a,b}/*.txt').
#     parse : bool, optional
#         Whether to parse the matched paths. Default is False.
#     ensure_one : bool, optional
#         Ensure exactly one match is found. Default is False.
# 
#     Returns:
#     --------
#     Union[List[str], Tuple[List[str], List[dict]]]
#         If parse=False: A naturally sorted list of file paths
#         If parse=True: Tuple of (paths, parsed results)
# 
#     Examples:
#     ---------
#     >>> glob('data/*.txt')
#     ['data/file1.txt', 'data/file2.txt', 'data/file10.txt']
# 
#     >>> glob('data/{a,b}/*.txt')
#     ['data/a/file1.txt', 'data/a/file2.txt', 'data/b/file1.txt']
# 
#     >>> paths, parsed = glob('data/subj_{id}/run_{run}.txt', parse=True)
#     >>> paths
#     ['data/subj_001/run_01.txt', 'data/subj_001/run_02.txt']
#     >>> parsed
#     [{'id': '001', 'run': '01'}, {'id': '001', 'run': '02'}]
# 
#     >>> paths, parsed = glob('data/subj_{id}/run_{run}.txt', parse=True, ensure_one=True)
#     AssertionError  # if more than one file matches
#     """
#     glob_pattern = _re.sub(r"{[^}]*}", "*", expression)
#     try:
#         found_paths = _natsorted(_glob(eval(glob_pattern)))
#     except:
#         found_paths = _natsorted(_glob(glob_pattern))
# 
#     if ensure_one:
#         assert len(found_paths) == 1
# 
#     if parse:
#         parsed = [_parse(found_path, expression) for found_path in found_paths]
#         return found_paths, parsed
# 
#     else:
#         return found_paths
# 
# def parse_glob(expression, ensure_one=False):
#     """
#     Convenience function for glob with parsing enabled.
# 
#     Parameters:
#     -----------
#     expression : str
#         The glob pattern to match against file paths.
#     ensure_one : bool, optional
#         Ensure exactly one match is found. Default is False.
# 
#     Returns:
#     --------
#     Tuple[List[str], List[dict]]
#         Matched paths and parsed results.
# 
#     Examples:
#     ---------
#     >>> paths, parsed = pglob('data/subj_{id}/run_{run}.txt')
#     >>> paths
#     ['data/subj_001/run_01.txt', 'data/subj_001/run_02.txt']
#     >>> parsed
#     [{'id': '001', 'run': '01'}, {'id': '001', 'run': '02'}]
# 
#     >>> paths, parsed = pglob('data/subj_{id}/run_{run}.txt', ensure_one=True)
#     AssertionError  # if more than one file matches
#     """
#     return glob(expression, parse=True, ensure_one=ensure_one)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_glob.py
# --------------------------------------------------------------------------------
