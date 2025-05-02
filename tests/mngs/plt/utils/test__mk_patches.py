#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:27:14 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__mk_patches.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__mk_patches.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.patches as mpatches
from mngs.plt.utils._mk_patches import mk_patches


def test_mk_patches_basic():
    colors = ["#f00", "#0f0"]
    labels = ["a", "b"]
    patches = mk_patches(colors, labels)
    assert isinstance(patches, list)
    assert isinstance(patches[0], mpatches.Patch)
    assert patches[0].get_label() == "a"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_mk_patches.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 21:18:45 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_mk_patches.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_mk_patches.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib.patches as mpatches
# 
# 
# def mk_patches(colors, labels):
#     """
#     colors = ["red", "blue"]
#     labels = ["label_1", "label_2"]
#     ax.legend(handles=mngs.plt.mk_patches(colors, labels))
#     """
# 
#     patches = [
#         mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)
#     ]
#     return patches
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_mk_patches.py
# --------------------------------------------------------------------------------
