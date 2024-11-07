# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# 
# 
# def annotated_heatmap(
#     cm,
#     labels=None,
#     title=None,
#     cmap="Blues",
#     norm=None,
#     xlabel=None,
#     ylabel=None,
#     annot=True,
# ):
#     df = pd.DataFrame(data=cm)
# 
#     if labels is not None:
#         df.index = labels
#         df.columns = labels
# 
#     fig, ax = plt.subplots()
#     res = sns.heatmap(
#         df,
#         annot=annot,
#         fmt=".3f",
#         cmap=cmap,
#         norm=norm,
#     )  # cbar_kws={"shrink": 0.82}
#     res.invert_yaxis()
# 
#     # make the frame invisible
#     for _, spine in res.spines.items():
#         spine.set_visible(False)
# 
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
# 
#     return fig
# 
# 
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib import colors
# 
#     labels = ["T2T", "F2T", "T2F", "F2F"]
#     arr = np.random.randint(0, 10, len(labels) ** 2).reshape(len(labels), len(labels))
# 
#     ## quantized, arbitrary range colormap you want
#     cmap = colors.ListedColormap(
#         ["navy", "royalblue", "lightsteelblue", "beige"],
#     )
#     norm = colors.BoundaryNorm([2, 4, 6, 8], cmap.N - 1)
# 
#     fig = annotated_heatmap(arr, cmap=cmap, norm=norm, labels=labels)
#     fig.show()

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

from mngs.plt._annotated_heatmap import *

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
