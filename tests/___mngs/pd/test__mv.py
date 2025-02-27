# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/pd/_mv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 07:39:12 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/pd/_mv.py
# 
# def mv(df, key, position, axis=1):
#     """
#     Move a row or column to a specified position in a DataFrame.
# 
#     Args:
#     df (pandas.DataFrame): The input DataFrame.
#     key (str): The label of the row or column to move.
#     position (int): The position to move the row or column to.
#     axis (int, optional): 0 for rows, 1 for columns. Defaults to 1.
# 
#     Returns:
#     pandas.DataFrame: A new DataFrame with the row or column moved.
#     """
#     if axis == 0:
#         items = df.index.tolist()
#     else:
#         items = df.columns.tolist()
#     items.remove(key)
# 
#     if position < 0:
#         position += len(items) + 1
# 
#     items.insert(position, key)
#     return df.reindex(items, axis=axis)
# 
# 
# # def mv_to_first(df, key, axis=0):
# #     """
# #     Move a row or column to the first position in a DataFrame.
# 
# #     Args:
# #     df (pandas.DataFrame): The input DataFrame.
# #     key (str): The label of the row or column to move.
# #     axis (int, optional): 0 for rows, 1 for columns. Defaults to 0.
# 
# #     Returns:
# #     pandas.DataFrame: A new DataFrame with the row or column moved to the first position.
# #     """
# #     return mv(df, key, 0, axis)
# 
# 
# # def mv_to_last(df, key, axis=0):
# #     """
# #     Move a row or column to the last position in a DataFrame.
# 
# #     Args:
# #     df (pandas.DataFrame): The input DataFrame.
# #     key (str): The label of the row or column to move.
# #     axis (int, optional): 0 for rows, 1 for columns. Defaults to 0.
# 
# #     Returns:
# #     pandas.DataFrame: A new DataFrame with the row or column moved to the last position.
# #     """
# #     return mv(df, key, -1, axis)
# 
# 
# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.pd._mv import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
