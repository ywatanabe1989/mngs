#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:00:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/str/test__search.py

"""Tests for string search functionality."""

import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch


class TestSearchBasic:
    """Test basic search functionality."""
    
    def test_search_single_pattern_single_string(self):
        """Test searching single pattern in single string."""
        from mngs.str._search import search
        
        indices, matches = search("quick", "The quick brown fox")
        assert indices == [0]
        assert matches == ["The quick brown fox"]
    
    def test_search_single_pattern_multiple_strings(self):
        """Test searching single pattern in multiple strings."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana", "orange_juice"]
        indices, matches = search("orange", strings)
        assert indices == [1, 3]
        assert matches == ["orange", "orange_juice"]
    
    def test_search_multiple_patterns_single_string(self):
        """Test searching multiple patterns in single string."""
        from mngs.str._search import search
        
        patterns = ["quick", "fox"]
        indices, matches = search(patterns, "The quick brown fox")
        assert indices == [0, 0]  # Same string matched twice
        assert matches == ["The quick brown fox", "The quick brown fox"]
    
    def test_search_multiple_patterns_multiple_strings(self):
        """Test searching multiple patterns in multiple strings."""
        from mngs.str._search import search
        
        patterns = ["orange", "banana"]
        strings = ["apple", "orange", "apple", "apple_juice", "banana", "orange_juice"]
        indices, matches = search(patterns, strings)
        assert indices == [1, 4, 5]
        assert matches == ["orange", "banana", "orange_juice"]
    
    def test_search_no_matches(self):
        """Test search with no matches."""
        from mngs.str._search import search
        
        indices, matches = search("xyz", ["apple", "orange", "banana"])
        assert indices == []
        assert matches == []


class TestSearchPerfectMatch:
    """Test perfect match functionality."""
    
    def test_search_perfect_match_true(self):
        """Test with only_perfect_match=True."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "orange_juice"]
        indices, matches = search("orange", strings, only_perfect_match=True)
        assert indices == [1]
        assert matches == ["orange"]
    
    def test_search_perfect_match_false(self):
        """Test with only_perfect_match=False (default)."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "orange_juice"]
        indices, matches = search("orange", strings, only_perfect_match=False)
        assert indices == [1, 2]
        assert matches == ["orange", "orange_juice"]
    
    def test_search_perfect_match_no_results(self):
        """Test perfect match when no exact matches exist."""
        from mngs.str._search import search
        
        strings = ["orange_juice", "apple_orange"]
        indices, matches = search("orange", strings, only_perfect_match=True)
        assert indices == []
        assert matches == []


class TestSearchBooleanOutput:
    """Test boolean output functionality."""
    
    def test_search_as_bool_true(self):
        """Test with as_bool=True."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana", "orange_juice"]
        bool_array, matches = search("orange", strings, as_bool=True)
        
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(bool_array, expected)
        assert matches == ["orange", "orange_juice"]
    
    def test_search_as_bool_false(self):
        """Test with as_bool=False (default)."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana", "orange_juice"]
        indices, matches = search("orange", strings, as_bool=False)
        
        assert indices == [1, 3]
        assert matches == ["orange", "orange_juice"]
    
    def test_search_as_bool_no_matches(self):
        """Test boolean output with no matches."""
        from mngs.str._search import search
        
        strings = ["apple", "banana", "grape"]
        bool_array, matches = search("orange", strings, as_bool=True)
        
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(bool_array, expected)
        assert matches == []


class TestSearchEnsureOne:
    """Test ensure_one functionality."""
    
    def test_search_ensure_one_success(self):
        """Test ensure_one with exactly one match."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana"]
        indices, matches = search("orange", strings, ensure_one=True)
        assert indices == [1]
        assert matches == ["orange"]
    
    def test_search_ensure_one_failure_no_match(self):
        """Test ensure_one with no matches."""
        from mngs.str._search import search
        
        strings = ["apple", "banana", "grape"]
        with pytest.raises(AssertionError, match="Expected exactly one match, but found 0"):
            search("orange", strings, ensure_one=True)
    
    def test_search_ensure_one_failure_multiple_matches(self):
        """Test ensure_one with multiple matches."""
        from mngs.str._search import search
        
        strings = ["orange", "orange_juice", "banana"]
        with pytest.raises(AssertionError, match="Expected exactly one match, but found 2"):
            search("orange", strings, ensure_one=True)


class TestSearchInputTypes:
    """Test different input data types."""
    
    def test_search_numpy_array_strings(self):
        """Test with numpy array as strings input."""
        from mngs.str._search import search
        
        strings = np.array(["apple", "orange", "banana"])
        indices, matches = search("orange", strings)
        assert indices == [1]
        assert matches == ["orange"]
    
    def test_search_pandas_series_strings(self):
        """Test with pandas Series as strings input."""
        from mngs.str._search import search
        
        strings = pd.Series(["apple", "orange", "banana"])
        indices, matches = search("orange", strings)
        assert indices == [1]
        assert matches == ["orange"]
    
    def test_search_xarray_dataarray_strings(self):
        """Test with xarray DataArray as strings input."""
        from mngs.str._search import search
        
        strings = xr.DataArray(["apple", "orange", "banana"])
        indices, matches = search("orange", strings)
        assert indices == [1]
        assert matches == ["orange"]
    
    def test_search_dict_keys_patterns(self):
        """Test with dict keys as patterns input."""
        from mngs.str._search import search
        
        patterns = {"orange": 1, "banana": 2}.keys()
        strings = ["apple", "orange", "banana", "grape"]
        indices, matches = search(patterns, strings)
        assert sorted(indices) == [1, 2]
        assert sorted(matches) == ["banana", "orange"]
    
    def test_search_tuple_input(self):
        """Test with tuple as input."""
        from mngs.str._search import search
        
        patterns = ("orange", "banana")
        strings = ("apple", "orange", "banana", "grape")
        indices, matches = search(patterns, strings)
        assert sorted(indices) == [1, 2]
        assert sorted(matches) == ["banana", "orange"]
    
    def test_search_pandas_index_strings(self):
        """Test with pandas Index as strings input."""
        from mngs.str._search import search
        
        strings = pd.Index(["apple", "orange", "banana"])
        indices, matches = search("orange", strings)
        assert indices == [1]
        assert matches == ["orange"]


class TestSearchRegexPatterns:
    """Test regex pattern functionality."""
    
    def test_search_simple_regex(self):
        """Test with simple regex patterns."""
        from mngs.str._search import search
        
        strings = ["test123", "hello", "test456", "world"]
        indices, matches = search(r"test\d+", strings)
        assert indices == [0, 2]
        assert matches == ["test123", "test456"]
    
    def test_search_case_insensitive_regex(self):
        """Test case-insensitive regex."""
        from mngs.str._search import search
        
        strings = ["Hello", "WORLD", "hello", "world"]
        indices, matches = search(r"(?i)hello", strings)
        assert indices == [0, 2]
        assert matches == ["Hello", "hello"]
    
    def test_search_complex_regex(self):
        """Test complex regex patterns."""
        from mngs.str._search import search
        
        strings = ["user@example.com", "invalid-email", "test@domain.org", "no-at-sign"]
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        indices, matches = search(pattern, strings)
        assert indices == [0, 2]
        assert matches == ["user@example.com", "test@domain.org"]
    
    def test_search_anchor_patterns(self):
        """Test regex with anchors (^, $)."""
        from mngs.str._search import search
        
        strings = ["prefix_test", "test_suffix", "test", "not_test_not"]
        indices, matches = search("^test", strings)
        assert indices == [1, 2]
        assert matches == ["test_suffix", "test"]


class TestSearchEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_search_empty_pattern(self):
        """Test with empty pattern."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana"]
        indices, matches = search("", strings)
        # Empty pattern should match all strings
        assert indices == [0, 1, 2]
        assert matches == ["apple", "orange", "banana"]
    
    def test_search_empty_strings(self):
        """Test with empty strings list."""
        from mngs.str._search import search
        
        indices, matches = search("pattern", [])
        assert indices == []
        assert matches == []
    
    def test_search_empty_string_in_list(self):
        """Test with empty string in strings list."""
        from mngs.str._search import search
        
        strings = ["apple", "", "banana"]
        indices, matches = search("", strings)
        # Empty pattern matches all, including empty string
        assert indices == [0, 1, 2]
        assert matches == ["apple", "", "banana"]
    
    def test_search_special_regex_characters(self):
        """Test with special regex characters in patterns."""
        from mngs.str._search import search
        
        strings = ["test.", "test*", "test+", "test[1]"]
        # Need to escape special characters for literal match
        indices, matches = search(r"test\.", strings)
        assert indices == [0]
        assert matches == ["test."]
    
    def test_search_unicode_strings(self):
        """Test with unicode characters."""
        from mngs.str._search import search
        
        strings = ["apple", "りんご", "banana", "バナナ"]
        indices, matches = search("りんご", strings)
        assert indices == [1]
        assert matches == ["りんご"]
    
    def test_search_very_long_strings(self):
        """Test with very long strings."""
        from mngs.str._search import search
        
        long_string = "a" * 10000 + "needle" + "b" * 10000
        strings = ["haystack", long_string, "another"]
        indices, matches = search("needle", strings)
        assert indices == [1]
        assert matches == [long_string]


class TestSearchDocstrings:
    """Test examples from docstrings work correctly."""
    
    def test_docstring_example_1(self):
        """Test first docstring example."""
        from mngs.str._search import search
        
        patterns = ['orange', 'banana']
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        indices, matches = search(patterns, strings)
        assert indices == [1, 4, 5]
        assert matches == ['orange', 'banana', 'orange_juice']
    
    def test_docstring_example_2(self):
        """Test second docstring example."""
        from mngs.str._search import search
        
        patterns = 'orange'
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        indices, matches = search(patterns, strings)
        assert indices == [1, 5]
        assert matches == ['orange', 'orange_juice']


class TestSearchInternalFunctions:
    """Test internal helper functions."""
    
    def test_to_list_function(self):
        """Test the internal to_list function behavior."""
        from mngs.str._search import search
        
        # Test that different input types are handled correctly
        # This is tested indirectly through the main search function
        
        # String input
        indices, matches = search("test", "test_string")
        assert len(indices) == 1
        
        # List input
        indices, matches = search(["test"], ["test_string"])
        assert len(indices) == 1
        
        # Numpy array input
        strings = np.array(["test_string", "other"])
        indices, matches = search("test", strings)
        assert indices == [0]


class TestSearchCombinedOptions:
    """Test combinations of different options."""
    
    def test_search_perfect_match_and_bool(self):
        """Test combining perfect_match and as_bool options."""
        from mngs.str._search import search
        
        strings = ["orange", "orange_juice", "banana"]
        bool_array, matches = search("orange", strings, 
                                    only_perfect_match=True, as_bool=True)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(bool_array, expected)
        assert matches == ["orange"]
    
    def test_search_bool_and_ensure_one(self):
        """Test combining as_bool and ensure_one options."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana"]
        bool_array, matches = search("orange", strings, 
                                    as_bool=True, ensure_one=True)
        
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(bool_array, expected)
        assert matches == ["orange"]
    
    def test_search_all_options_combined(self):
        """Test all options combined."""
        from mngs.str._search import search
        
        strings = ["apple", "orange", "banana"]
        bool_array, matches = search("orange", strings,
                                    only_perfect_match=True,
                                    as_bool=True,
                                    ensure_one=True)
        
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(bool_array, expected)
        assert matches == ["orange"]


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_search.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-13 14:25:59 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_search.py
# 
# import re
# from collections import abc
# 
# import numpy as np
# import pandas as pd
# import xarray as xr
# from natsort import natsorted
# 
# 
# def search(
#     patterns,
#     strings,
#     only_perfect_match=False,
#     as_bool=False,
#     ensure_one=False,
# ):
#     """Search for patterns in strings using regular expressions.
# 
#     Parameters
#     ----------
#     patterns : str or list of str
#         The pattern(s) to search for. Can be a single string or a list of strings.
#     strings : str or list of str
#         The string(s) to search in. Can be a single string or a list of strings.
#     only_perfect_match : bool, optional
#         If True, only exact matches are considered (default is False).
#     as_bool : bool, optional
#         If True, return a boolean array instead of indices (default is False).
#     ensure_one : bool, optional
#         If True, ensures only one match is found (default is False).
# 
#     Returns
#     -------
#     tuple
#         A tuple containing two elements:
#         - If as_bool is False: (list of int, list of str)
#           The first element is a list of indices where matches were found.
#           The second element is a list of matched strings.
#         - If as_bool is True: (numpy.ndarray of bool, list of str)
#           The first element is a boolean array indicating matches.
#           The second element is a list of matched strings.
# 
#     Example
#     -------
#     >>> patterns = ['orange', 'banana']
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 4, 5], ['orange', 'banana', 'orange_juice'])
# 
#     >>> patterns = 'orange'
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 5], ['orange', 'orange_juice'])
#     """
# 
#     def to_list(string_or_pattern):
#         if isinstance(
#             string_or_pattern, (np.ndarray, pd.Series, xr.DataArray)
#         ):
#             return string_or_pattern.tolist()
#         elif isinstance(string_or_pattern, abc.KeysView):
#             return list(string_or_pattern)
#         elif not isinstance(string_or_pattern, (list, tuple, pd.Index)):
#             return [string_or_pattern]
#         return string_or_pattern
# 
#     patterns = to_list(patterns)
#     strings = to_list(strings)
# 
#     indices_matched = []
#     for pattern in patterns:
#         for index_str, string in enumerate(strings):
#             if only_perfect_match:
#                 if pattern == string:
#                     indices_matched.append(index_str)
#             else:
#                 if re.search(pattern, string):
#                     indices_matched.append(index_str)
# 
#     indices_matched = natsorted(indices_matched)
#     keys_matched = list(np.array(strings)[indices_matched])
# 
#     if ensure_one:
#         assert (
#             len(indices_matched) == 1
#         ), "Expected exactly one match, but found {}".format(
#             len(indices_matched)
#         )
# 
#     if as_bool:
#         bool_matched = np.zeros(len(strings), dtype=bool)
#         bool_matched[np.unique(indices_matched)] = True
#         return bool_matched, keys_matched
#     else:
#         return indices_matched, keys_matched
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_search.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
