#!/bin/bash
# Time-stamp: "2024-11-03 02:42:16 (ywatanabe)"
# File: ./mngs_repo/update_package.sh


rm -rf build dist/* src/mngs.egg-info
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*


# EOF
