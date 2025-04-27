# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/stats/tests/_smirnov_grubbs.py
# --------------------------------------------------------------------------------
# import numpy as np
# from scipy import stats
# 
# 
# def smirnov_grubbs(data_arr, alpha=0.05):
#     """
#     Find outliers based on Smirnov-Grubbs test.
# 
#     Arguments:
# 
#     Returns | indices of outliers
#     """
#     data_1D_sorted = sorted(np.array(data_arr).reshape(-1))
#     in_data, out_data = list(data_1D_sorted), []
# 
#     # while True:
#     n = len(in_data)
#     for i_data in range(n):
#         n = len(in_data)
#         t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
#         tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
#         i_min, i_max = np.argmin(in_data), np.argmax(in_data)
#         mu, std = np.mean(in_data), np.std(in_data, ddof=1)
# 
#         i_far = (
#             i_max
#             if np.abs(in_data[i_max] - mu) > np.abs(in_data[i_min] - mu)
#             else i_min
#         )
# 
#         tau_far = np.abs((in_data[i_far] - mu) / std)
# 
#         if tau_far < tau:
#             break
# 
#         out_data.append(in_data.pop(i_far))
# 
#     if len(out_data) == 0:
#         return None
# 
#     else:
#         out_data_uq = np.unique(out_data)
#         indi_outliers = np.hstack(
#             [
#                 np.vstack(np.where(data_arr == out_data_uq[i_out])).T
#                 for i_out in range(len(out_data_uq))
#             ]
#         ).squeeze()
# 
#         if indi_outliers.ndim == 0:
#             indi_outliers = indi_outliers[np.newaxis]
#         return indi_outliers
# 
# 
# # def smirnov_grubbs(data_arr, alpha=0.05):
# #     """
# #     Find outliers based on Smirnov-Grubbs test.
# 
# #     Arguments:
# 
# #     Returns | indices of outliers
# #     """
# #     data_1D_sorted = sorted(np.array(data_arr).reshape(-1))
# #     in_data, out_data = list(data_1D_sorted), []
# 
# #     while True:
# #         n = len(in_data)
# #         t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
# #         tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
# #         i_min, i_max = np.argmin(in_data), np.argmax(in_data)
# #         mu, std = np.mean(in_data), np.std(in_data, ddof=1)
# 
# #         i_far = (
# #             i_max
# #             if np.abs(in_data[i_max] - mu) > np.abs(in_data[i_min] - mu)
# #             else i_min
# #         )
# 
# #         tau_far = np.abs((in_data[i_far] - mu) / std)
# 
# #         if tau_far < tau:
# #             break
# 
# #         out_data.append(in_data.pop(i_far))
# 
# #     if len(out_data) == 0:
# #         return None
# 
# #     else:
# #         out_data_uq = np.unique(out_data)
# #         indi_outliers = np.vstack(
# #             [
# #                 np.vstack(np.where(data_arr == out_data_uq[i_out])).T
# #                 for i_out in range(len(out_data_uq))
# #             ]
# #         )
# 
# #         return np.array(indi_outliers).squeeze()

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

from mngs.stats.tests._smirnov_grubbs import *

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
