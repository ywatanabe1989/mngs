# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/stats/multiple/_multicompair.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import numpy as np
# import scipy.stats as stats
# from statsmodels.stats.multicomp import MultiComparison
# 
# 
# def multicompair(data, labels, testfunc=None):
#     # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
#     _labels = labels.copy()
#     # Set up the data for comparison (creates a specialised object)
#     for i_labels in range(len(_labels)):
#         _labels[i_labels] = [_labels[i_labels] for i_data in range(len(data[i_labels]))]
# 
#     data, _labels = np.hstack(data), np.hstack(_labels)
#     MultiComp = MultiComparison(data, _labels)
# 
#     if testfunc is not None:
#         # print(MultiComp.allpairtest(testfunc, mehotd='bonf', pvalidx=1))
#         return MultiComp.allpairtest(testfunc, method="bonf", pvalidx=1)
#     else:
#         # print(MultiComp.tukeyhsd().summary())
#         return MultiComp.tukeyhsd().summary()
# 
# 
# # t_statistic, p_value = scipy.stats.ttest_ind(data1, data2, equal_var=False) # Welch's t test
# # W_statistic, p_value = scipy.stats.brunnermunzel(data1, data2)
# # H_statistic, p_value = scipy.stats.kruskal(*data) # one-way ANOVA on RANKs

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

from mngs.stats.multiple._multicompair import *

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
