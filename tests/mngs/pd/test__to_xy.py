# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/pd/_to_xy.py
# --------------------------------------------------------------------------------
# #!/./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-03 07:01:31 (ywatanabe)"
# # ./src/mngs/pd/_to_xy.py
# 
# import mngs
# import numpy as np
# import pandas as pd
# 
# 
# def to_xy(data_frame):
#     """
#     Convert a heatmap DataFrame into x, y, z format.
# 
#     Ensure the index and columns are the same, and if either exists, replace with that.
# 
#     Example
#     -------
#     data_frame = pd.DataFrame(...)  # Your DataFrame here
#     out = to_xy(data_frame)
#     print(out)
# 
#     Parameters
#     ----------
#     data_frame : pandas.DataFrame
#         The input DataFrame to be converted.
# 
#     Returns
#     -------
#     pandas.DataFrame
#         A DataFrame formatted with columns ['x', 'y', 'z']
#     """
#     assert data_frame.shape[0] == data_frame.shape[1]
# 
#     if not data_frame.index.equals(data_frame.columns):
# 
#         if (data_frame.index == np.array(range(len(data_frame.index)))).all():
#             data_frame.columns = data_frame.index
#         elif (
#             data_frame.columns == np.array(range(len(data_frame.columns)))
#         ).all():
#             data_frame.index = data_frame.columns
#         else:
#             ValueError
#         # else:
#         #     ValueError "Either of index or columns has to be passed"
# 
#     formatted_data_frames = []
# 
#     for column in data_frame.columns:
#         column_data_frame = data_frame[column]
#         y_label = column_data_frame.name
#         column_data_frame = pd.DataFrame(column_data_frame)
#         column_data_frame["x"] = column_data_frame.index
#         column_data_frame["y"] = y_label
#         column_data_frame = column_data_frame.reset_index().drop(
#             columns=["index"]
#         )
#         column_data_frame = column_data_frame.rename(columns={y_label: "z"})
#         column_data_frame = mngs.pd.mv(column_data_frame, "z", -1)
#         formatted_data_frames.append(column_data_frame)
# 
#     return pd.concat(formatted_data_frames, ignore_index=True)

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.pd._to_xy import *

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
