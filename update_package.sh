#!/bin/bash

rm -rf build dist/* mngs.egg-info
python3 setup.py sdist bdist_wheel
twine upload -r testpypi dist/*
pip uninstall ripple_detector_CNN -y
pip install --no-cache-dir --upgrade ./dist/mngs-*-py3-none-any.whl --force-reinstall

## EOF
