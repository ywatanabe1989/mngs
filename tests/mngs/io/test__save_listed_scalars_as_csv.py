# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:26:48 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_listed_scalars_as_csv.py
# 
# import numpy as np
# import pandas as pd
# from ._mv_to_tmp import _mv_to_tmp
# 
# def _save_listed_scalars_as_csv(
#     listed_scalars,
#     spath_csv,
#     column_name="_",
#     indi_suffix=None,
#     round=3,
#     overwrite=False,
#     verbose=False,
# ):
#     """Puts to df and save it as csv"""
# 
#     if overwrite == True:
#         _mv_to_tmp(spath_csv, L=2)
#     indi_suffix = (
#         np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
#     )
#     df = pd.DataFrame(
#         {"{}".format(column_name): listed_scalars}, index=indi_suffix
#     ).round(round)
#     df.to_csv(spath_csv)
#     if verbose:
#         print("\nSaved to: {}\n".format(spath_csv))
# 
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

from ...src.mngs..io._save_listed_scalars_as_csv import *

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
