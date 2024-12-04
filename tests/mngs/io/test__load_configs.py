# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 08:06:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_configs.py
# 
# import os
# from ..dict import DotDict
# from ..io._load import load
# from ._glob import glob
# 
# def load_configs(IS_DEBUG=None, show=False, verbose=False):
#     """Load YAML configuration files from ./config directory.
# 
#     Parameters
#     ----------
#     IS_DEBUG : bool, optional
#         Debug mode flag. If None, reads from IS_DEBUG.yaml
#     show : bool
#         Show configuration changes
#     verbose : bool
#         Print detailed information
# 
#     Returns
#     -------
#     DotDict
#         Merged configuration dictionary
#     """
#     def apply_debug_values(config, IS_DEBUG):
#         """Apply debug values if IS_DEBUG is True."""
#         if not IS_DEBUG or not isinstance(config, (dict, DotDict)):
#             return config
# 
#         for key, value in list(config.items()):
#             if key.startswith(("DEBUG_", "debug_")):
#                 dk_wo_debug_prefix = key.split("_", 1)[1]
#                 config[dk_wo_debug_prefix] = value
#                 if show or verbose:
#                     print(f"{key} -> {dk_wo_debug_prefix}")
#             elif isinstance(value, (dict, DotDict)):
#                 config[key] = apply_debug_values(value, IS_DEBUG)
#         return config
# 
#     try:
#         # Set debug mode
#         IS_DEBUG = IS_DEBUG or os.getenv("CI") == "True" or (
#             os.path.exists("./config/IS_DEBUG.yaml") and
#             load("./config/IS_DEBUG.yaml").get("IS_DEBUG")
#         )
# 
#         # Load and merge configs
#         CONFIGS = {}
#         for lpath in glob("./config/*.yaml"):
#             if config := load(lpath):
#                 CONFIGS.update(apply_debug_values(config, IS_DEBUG))
# 
#         return DotDict(CONFIGS)
# 
#     except Exception as e:
#         print(f"Error loading configs: {e}")
#         return DotDict({})
# # def load_configs(IS_DEBUG=None, show=False, verbose=False):
# #     """
# #     Load configuration files from the ./config directory.
# 
# #     Parameters:
# #     -----------
# #     IS_DEBUG : bool, optional
# #         If True, use debug configurations. If None, check ./config/IS_DEBUG.yaml.
# #     show : bool, optional
# #         If True, display additional information during loading.
# #     verbose : bool, optional
# #         If True, print verbose output during loading.
# 
# #     Returns:
# #     --------
# #     DotDict
# #         A dictionary-like object containing the loaded configurations.
# #     """
# 
# #     def apply_debug_values(config, IS_DEBUG):
# #         if IS_DEBUG:
# #             if isinstance(config, (dict, DotDict)):
# #                 for key, value in list(config.items()):
# #                     try:
# #                         if key.startswith(("DEBUG_", "debug_")):
# #                             dk_wo_debug_prefix = key.split("_", 1)[1]
# #                             config[dk_wo_debug_prefix] = value
# #                             if show or verbose:
# #                                 print(f"\n{key} -> {dk_wo_debug_prefix}\n")
# #                         elif isinstance(value, (dict, DotDict)):
# #                             config[key] = apply_debug_values(value, IS_DEBUG)
# #                     except Exception as e:
# #                         print(e)
# #         return config
# 
# #     if os.getenv("CI") == "True":
# #         IS_DEBUG = True
# 
# #     try:
# #         # Check ./config/IS_DEBUG.yaml file if IS_DEBUG argument is not passed
# #         if IS_DEBUG is None:
# #             IS_DEBUG_PATH = "./config/IS_DEBUG.yaml"
# #             if os.path.exists(IS_DEBUG_PATH):
# #                 IS_DEBUG = load("./config/IS_DEBUG.yaml").get("IS_DEBUG")
# #             else:
# #                 IS_DEBUG = False
# 
# #         # Main
# #         CONFIGS = {}
# #         for lpath in glob("./config/*.yaml"):
# #             config = load(lpath)
# #             if config:
# #                 CONFIG = apply_debug_values(config, IS_DEBUG)
# #                 CONFIGS.update(CONFIG)
# 
# #         CONFIGS = DotDict(CONFIGS)
# 
# #     except Exception as e:
# #         print(e)
# #         CONFIGS = DotDict({})
# 
# #     return CONFIGS
# 
# 
# # 
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

from mngs..io._load_configs import *

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