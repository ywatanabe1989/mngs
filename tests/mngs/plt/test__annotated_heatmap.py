#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:20:25 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__annotated_heatmap.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__annotated_heatmap.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np

from mngs.plt._annotated_heatmap import annotated_heatmap


def test_annotated_heatmap_basic():
    data = np.array([[1, 2], [3, 4]])
    fig = annotated_heatmap(
        cm=data,
        labels=["a", "b"],
        title="title",
        cmap="Blues",
        xlabel="x",
        ylabel="y",
    )
    ax = fig.axes[0]
    assert ax.get_title() == "title"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_annotated_heatmap.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_annotated_heatmap.py
# --------------------------------------------------------------------------------

# EOF