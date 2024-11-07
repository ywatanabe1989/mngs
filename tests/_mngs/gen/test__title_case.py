# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-24 15:05:34"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# 
# """
# This script does XYZ.
# """
# 
# 
# """
# Imports
# """
# import sys
# 
# import matplotlib.pyplot as plt
# 
# 
# """
# Config
# """
# # CONFIG = mngs.gen.load_configs()
# 
# 
# """
# Functions & Classes
# """
# 
# 
# def title_case(text):
#     """
#     Converts a string to title case while keeping certain prepositions, conjunctions, and articles in lowercase,
#     and ensuring words detected as potential acronyms (all uppercase) are fully capitalized.
# 
#     Parameters:
#     - text (str): The text to convert to title case.
# 
#     Returns:
#     - str: The converted text in title case with certain words in lowercase and potential acronyms fully capitalized.
# 
#     Examples:
#     --------
#         print(title_case("welcome to the world of ai and using CPUs for gaming"))  # Welcome to the World of AI and Using CPUs for Gaming
#     """
#     # List of words to keep in lowercase
#     lowercase_words = [
#         "a",
#         "an",
#         "the",
#         "and",
#         "but",
#         "or",
#         "nor",
#         "at",
#         "by",
#         "to",
#         "in",
#         "with",
#         "of",
#         "on",
#     ]
# 
#     words = text.split()
#     final_words = []
#     for word in words:
#         # Check if the word is fully in uppercase and more than one character, suggesting an acronym
#         if word.isupper() and len(word) > 1:
#             final_words.append(word)
#         elif word.lower() in lowercase_words:
#             final_words.append(word.lower())
#         else:
#             final_words.append(word.capitalize())
#     return " ".join(final_words)
# 
# 
# def main():
#     # Example usage:
#     text = "welcome to the world of ai and using CPUs for gaming"
#     print(title_case(text))
# 
# 
# if __name__ == "__main__":
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.gen._title_case import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
