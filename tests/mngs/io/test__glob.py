#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-05-31"
# File: test__glob.py

"""Tests for the glob and parse_glob functions in mngs.io module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestGlobBasic:
    """Test basic glob functionality."""

    def test_glob_simple_pattern(self, tmp_path):
        """Test glob with simple wildcard pattern."""
        from mngs.io import glob

        # Create test files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file10.txt").touch()
        (tmp_path / "other.csv").touch()

        # Test glob
        pattern = str(tmp_path / "*.txt")
        result = glob(pattern)

        # Should return naturally sorted list
        assert len(result) == 3
        assert result[0].endswith("file1.txt")
        assert result[1].endswith("file2.txt")
        assert result[2].endswith("file10.txt")  # Natural sort: 10 after 2

    def test_glob_natural_sorting(self, tmp_path):
        """Test that glob returns naturally sorted results."""
        from mngs.io import glob

        # Create files with numbers
        for i in [1, 2, 10, 20, 100]:
            (tmp_path / f"data{i}.txt").touch()

        pattern = str(tmp_path / "data*.txt")
        result = glob(pattern)

        # Check natural sorting order
        basenames = [os.path.basename(p) for p in result]
        assert basenames == [
            "data1.txt",
            "data2.txt",
            "data10.txt",
            "data20.txt",
            "data100.txt",
        ]

    def test_glob_no_matches(self, tmp_path):
        """Test glob with pattern that matches nothing."""
        from mngs.io import glob

        pattern = str(tmp_path / "nonexistent*.txt")
        result = glob(pattern)

        assert result == []

    def test_glob_recursive_pattern(self, tmp_path):
        """Test glob with recursive ** pattern."""
        from mngs.io import glob

        # Create nested structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "file1.txt").touch()
        (tmp_path / "dir2").mkdir()
        (tmp_path / "dir2" / "subdir").mkdir()
        (tmp_path / "dir2" / "subdir" / "file2.txt").touch()

        # Test recursive glob
        pattern = str(tmp_path / "**" / "*.txt")
        result = glob(pattern)

        assert len(result) == 2
        assert any("file1.txt" in p for p in result)
        assert any("file2.txt" in p for p in result)


class TestGlobParsing:
    """Test glob with parsing functionality."""

    def test_glob_with_parse(self, tmp_path):
        """Test glob with parse=True."""
        from mngs.io import glob

        # Create test files with pattern
        (tmp_path / "subj_001").mkdir()
        (tmp_path / "subj_001" / "run_01.txt").touch()
        (tmp_path / "subj_001" / "run_02.txt").touch()
        (tmp_path / "subj_002").mkdir()
        (tmp_path / "subj_002" / "run_01.txt").touch()

        # Test with parsing
        pattern = str(tmp_path / "subj_{id}" / "run_{run}.txt")
        paths, parsed = glob(pattern, parse=True)

        assert len(paths) == 3
        assert len(parsed) == 3

        # Check parsed results
        assert parsed[0]["id"] == "001"
        assert parsed[0]["run"] == "01"
        assert parsed[1]["id"] == "001"
        assert parsed[1]["run"] == "02"
        assert parsed[2]["id"] == "002"
        assert parsed[2]["run"] == "01"

    def test_parse_glob_function(self, tmp_path):
        """Test the parse_glob convenience function."""
        from mngs.io import parse_glob

        # Create test files
        (tmp_path / "exp_01_trial_001.dat").touch()
        (tmp_path / "exp_01_trial_002.dat").touch()
        (tmp_path / "exp_02_trial_001.dat").touch()

        # Test parse_glob
        pattern = str(tmp_path / "exp_{exp}_trial_{trial}.dat")
        paths, parsed = parse_glob(pattern)

        assert len(paths) == 3
        assert len(parsed) == 3

        # Verify parsing
        assert all("exp" in p and "trial" in p for p in parsed)

    def test_glob_parse_complex_pattern(self, tmp_path):
        """Test parsing with complex patterns."""
        from mngs.io import glob

        # Create complex structure
        base = tmp_path / "data" / "2024"
        base.mkdir(parents=True)
        (base / "patient_A01_session_pre_scan_001.nii").touch()
        (base / "patient_A01_session_post_scan_001.nii").touch()
        (base / "patient_B02_session_pre_scan_001.nii").touch()

        # Test complex parsing
        pattern = str(
            tmp_path / "data" / "*" / "patient_{pid}_session_{session}_scan_{scan}.nii"
        )
        paths, parsed = glob(pattern, parse=True)

        assert len(parsed) == 3
        assert parsed[0]["pid"] == "A01"
        assert parsed[0]["session"] == "pre"
        assert parsed[0]["scan"] == "001"


class TestGlobEnsureOne:
    """Test glob with ensure_one parameter."""

    def test_glob_ensure_one_success(self, tmp_path):
        """Test glob with ensure_one when exactly one match exists."""
        from mngs.io import glob

        # Create exactly one matching file
        (tmp_path / "unique.txt").touch()

        pattern = str(tmp_path / "unique.txt")
        result = glob(pattern, ensure_one=True)

        assert len(result) == 1
        assert result[0].endswith("unique.txt")

    def test_glob_ensure_one_failure(self, tmp_path):
        """Test glob with ensure_one when multiple matches exist."""
        from mngs.io import glob

        # Create multiple matching files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()

        pattern = str(tmp_path / "*.txt")

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            glob(pattern, ensure_one=True)

    def test_glob_ensure_one_no_match(self, tmp_path):
        """Test glob with ensure_one when no matches exist."""
        from mngs.io import glob

        pattern = str(tmp_path / "nonexistent.txt")

        # Should raise AssertionError
        with pytest.raises(AssertionError):
            glob(pattern, ensure_one=True)

    def test_parse_glob_ensure_one(self, tmp_path):
        """Test parse_glob with ensure_one parameter."""
        from mngs.io import parse_glob

        # Create one file
        (tmp_path / "data_001.txt").touch()

        pattern = str(tmp_path / "data_{id}.txt")
        paths, parsed = parse_glob(pattern, ensure_one=True)

        assert len(paths) == 1
        assert parsed[0]["id"] == "001"


class TestGlobAdvanced:
    """Test advanced glob scenarios."""

    def test_glob_curly_brace_pattern(self, tmp_path):
        """Test glob with curly brace expansion pattern."""
        from mngs.io import glob

        # Create files in different directories
        for subdir in ["a", "b", "c"]:
            (tmp_path / subdir).mkdir()
            (tmp_path / subdir / "data.txt").touch()

        # Pattern with braces should be converted to wildcards
        pattern = str(tmp_path / "{a,b}" / "*.txt")
        result = glob(pattern)

        # Should match files in all directories (braces become *)
        assert len(result) >= 2

    def test_glob_eval_safety(self, tmp_path):
        """Test that glob handles eval safely."""
        from mngs.io import glob

        # Create a file
        (tmp_path / "test.txt").touch()

        # Pattern that might cause eval issues
        pattern = str(tmp_path / "test.txt'; import os; os.system('echo hacked')")

        # Should handle safely (fall back to regular glob)
        result = glob(pattern)
        assert result == []  # No match for malicious pattern

    def test_glob_special_characters(self, tmp_path):
        """Test glob with special characters in filenames."""
        from mngs.io import glob

        # Create files with special characters
        special_files = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
        ]

        for fname in special_files:
            (tmp_path / fname).touch()

        pattern = str(tmp_path / "*.txt")
        result = glob(pattern)

        assert len(result) == len(special_files)

    def test_glob_hidden_files(self, tmp_path):
        """Test glob with hidden files."""
        from mngs.io import glob

        # Create hidden and regular files
        (tmp_path / ".hidden.txt").touch()
        (tmp_path / "visible.txt").touch()

        # Test with pattern that should match hidden files
        pattern = str(tmp_path / ".*")
        result = glob(pattern)

        assert any(".hidden.txt" in p for p in result)


class TestGlobIntegration:
    """Test glob integration with other features."""

    def test_glob_with_pathlib(self, tmp_path):
        """Test glob works with pathlib paths."""
        from mngs.io import glob

        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.csv").touch()
        (data_dir / "file2.csv").touch()

        # Use pathlib Path for pattern
        pattern = data_dir / "*.csv"
        result = glob(str(pattern))

        assert len(result) == 2

    def test_glob_parse_integration(self, tmp_path):
        """Test glob parsing integrates with mngs.str.parse."""
        from mngs.io import glob

        # Create structured data
        for year in [2022, 2023]:
            for month in [1, 2, 12]:
                dir_path = tmp_path / f"data_{year}_{month:02d}"
                dir_path.mkdir()
                (dir_path / "report.txt").touch()

        # Parse structured pattern
        pattern = str(tmp_path / "data_{year}_{month}" / "report.txt")
        paths, parsed = glob(pattern, parse=True)

        assert len(parsed) == 6
        # Check parsing worked correctly
        years = [p["year"] for p in parsed]
        assert "2022" in years and "2023" in years

    def test_glob_empty_directory(self, tmp_path):
        """Test glob on empty directory."""
        from mngs.io import glob

        # Empty directory
        pattern = str(tmp_path / "*")
        result = glob(pattern)

        assert result == []


class TestGlobEdgeCases:
    """Test edge cases for glob function."""

    def test_glob_root_pattern(self):
        """Test glob with root directory pattern."""
        from mngs.io import glob

        # Pattern at root - should work but return limited results
        result = glob("/*.txt")

        # Should return list (possibly empty)
        assert isinstance(result, list)

    def test_glob_invalid_pattern(self, tmp_path):
        """Test glob with invalid pattern."""
        from mngs.io import glob

        # Pattern with invalid syntax
        pattern = str(tmp_path / "[")

        # Should handle gracefully
        result = glob(pattern)
        assert isinstance(result, list)

    def test_glob_unicode_filenames(self, tmp_path):
        """Test glob with unicode filenames."""
        from mngs.io import glob

        # Create files with unicode names
        unicode_files = [
            "файл.txt",  # Russian
            "文件.txt",  # Chinese
            "ファイル.txt",  # Japanese
            "café.txt",  # French
        ]

        for fname in unicode_files:
            try:
                (tmp_path / fname).touch()
            except:
                pass  # Skip if filesystem doesn't support

        pattern = str(tmp_path / "*.txt")
        result = glob(pattern)

        # Should handle unicode gracefully
        assert isinstance(result, list)


# --------------------------------------------------------------------------------
=======
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
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

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
>>>>>>> origin/main
