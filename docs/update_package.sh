#!/bin/bash
# Time-stamp: "2024-11-07 18:18:23 (ywatanabe)"
# File: ./mngs_repo/docs/update_package.sh


rm -rf build dist/* src/mngs.egg-info
python3 setup.py sdist bdist_wheel
twine upload -r pypi dist/*


# EOF
