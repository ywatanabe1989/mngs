# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:07:36 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/_mne.py
# 
# import mne
# import pandas as pd
# from .PARAMS import EEG_MONTAGE_1020
# 
# 
# def get_eeg_pos(channel_names=EEG_MONTAGE_1020):
#     # Load the standard 10-20 montage
#     standard_montage = mne.channels.make_standard_montage("standard_1020")
#     standard_montage.ch_names = [
#         ch_name.upper() for ch_name in standard_montage.ch_names
#     ]
# 
#     # Get the positions of the electrodes in the standard montage
#     positions = standard_montage.get_positions()
# 
#     df = pd.DataFrame(positions["ch_pos"])[channel_names]
# 
#     df.set_index(pd.Series(["x", "y", "z"]))
# 
#     return df
# 
# 
# if __name__ == "__main__":
#     print(get_eeg_pos())
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

from mngs..dsp._mne import *

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
