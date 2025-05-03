# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_get_RGBA_from_colormap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# 
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# 
# 
# class ColorGetter:
#     # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
#     def __init__(self, cmap_name, start_val, stop_val):
#         self.cmap_name = cmap_name
#         self.cmap = plt.get_cmap(cmap_name)
#         self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
#         self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
# 
#     def get_rgb(self, val):
#         return self.scalarMap.to_rgba(val)
# 
# 
# def get_RGBA_from_colormap(val, cmap="Blues", cmap_start_val=0, cmap_stop_val=1):
#     ColGetter = ColorGetter(cmap, cmap_start_val, cmap_stop_val)
#     return ColGetter.get_rgb(val)

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

from mngs.plt._get_RGBA_from_colormap import *

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
