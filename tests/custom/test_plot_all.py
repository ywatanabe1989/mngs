#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 21:12:26 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/custom/test_plot_all.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/custom/test_plot_all.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib

matplotlib.use("TkAgg")  # Use non-interactive Agg backend


# if __name__ == "__main__":
#     test_sns_barplot()
if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# pytest ./tests/custom/test_plot_all.py

# EOF