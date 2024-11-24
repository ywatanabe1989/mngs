# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python
# 
# import os
# from setuptools import find_packages, setup
# 
# 
# def read(fname):
#     with open(os.path.join(os.path.dirname(__file__), fname)) as f:
#         return f.read()
# 
# 
# setup(
#     name='ranger',
#     version='0.1.dev0',
#     packages=find_packages(
#         exclude=['tests', '*.tests', '*.tests.*', 'tests.*']
#     ),
#     package_dir={'ranger': os.path.join('.', 'ranger')},
#     description='Ranger - a synergistic optimizer using RAdam '
#                 '(Rectified Adam) and LookAhead in one codebase ',
#     long_description=read('README.md'),
#     long_description_content_type='text/markdown',
#     author='Less Wright',
#     license='Apache',
#     install_requires=['torch']
# )

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

from mngs..ai.optim.Ranger_Deep_Learning_Optimizer.setup import *

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
