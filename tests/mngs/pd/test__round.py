# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-06 11:13:00 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/pd/_round.py
# 
# import numpy as np
# import pandas as pd
# 
# def round(df: pd.DataFrame, factor: int = 3) -> pd.DataFrame:
#     """
#     Round numeric values in a DataFrame to a specified number of significant digits.
# 
#     Example
#     -------
#     >>> df = pd.DataFrame({'A': [1.23456, 2.34567], 'B': ['abc', 'def'], 'C': [3, 4]})
#     >>> round(df, 3)
#        A    B  C
#     0  1.23  abc  3
#     1  2.35  def  4
# 
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame
#     factor : int, optional
#         Number of significant digits to round to (default is 3)
# 
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with rounded numeric values
#     """
#     def custom_round(column):
#         try:
#             numeric_column = pd.to_numeric(column, errors='coerce')
#             if np.issubdtype(numeric_column.dtype, np.integer):
#                 return numeric_column.astype(int)
#             rounded = numeric_column.apply(lambda x: float(f'{x:.{factor}g}'))
# 
#             # Try converting to int if possible
#             if (rounded == rounded.astype(int)).all():
#                 return rounded.astype(int)
# 
#             return rounded
#         except (ValueError, TypeError):
#             return column
# 
#     return df.apply(custom_round)
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-05 20:40:32 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/pd/_round.py
# 
# # import numpy as np
# 
# # def round(df, factor=3):
# #     return df.apply(lambda x: x.round(factor) if np.issubdtype(x.dtype, np.number) else x)
# 
# 
# # def round(df, factor=3):
# #     def custom_round(x):
# #         try:
# #             numeric_x = pd.to_numeric(x, errors='raise')
# #             if np.issubdtype(numeric_x.dtype, np.integer):
# #                 return numeric_x
# #             else:
# #                 return numeric_x.apply(lambda y: float(f'{y:.{factor}g}'))
# #         except (ValueError, TypeError):
# #             return x
# 
# #     return df.apply(custom_round)

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
    sys.path.insert(0, project_root)

from src.mngs.pd._round import *

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