# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_force_aspect.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: 2024-05-14 00:03:52 (2)
# # /ssh:ywatanabe@444:/home/ywatanabe/proj/mngs/src/mngs/plt/ax/_force_aspect.py
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# 
# def force_aspect(ax, aspect=1):
#     im = ax.get_images()
# 
#     extent = im[0].get_extent()
# 
#     ax.set_aspect(
#         abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
#     )
#     return ax
# 
# 
# # Traceback (most recent call last):
# #   File "/home/ywatanabe/proj/entrance/neurovista/./scripts/ml/clustering_vit.py", line 199, in <module>
# #     main(args.model, args.clf_model)
# #   File "/home/ywatanabe/proj/entrance/neurovista/./scripts/ml/clustering_vit.py", line 152, in main
# #     fig, _legend_figs, _model = clustering_fn(
# #   File "/home/ywatanabe/proj/mngs/src/mngs/ml/clustering/_pca.py", line 64, in pca
# #     ax = mngs.plt.ax.force_aspect(ax)
# #   File "/home/ywatanabe/proj/mngs/src/mngs/plt/ax/_force_aspect.py", line 13, in force_aspect
# #     extent = im[0].get_extent()
# # IndexError: list index out of range

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

from mngs.plt.ax._force_aspect import *

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
