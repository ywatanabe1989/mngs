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

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
