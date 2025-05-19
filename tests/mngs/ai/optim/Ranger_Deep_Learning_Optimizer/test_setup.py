#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:02:01 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/test_setup.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/test_setup.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/setup.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/optim/Ranger_Deep_Learning_Optimizer/setup.py
# --------------------------------------------------------------------------------
